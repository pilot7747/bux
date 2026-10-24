"""remat=auto (bux.train.fitting): a training step that keeps the first of remat's
candidates, the fastest first, whose compiled step fits in the device's memory."""

import contextlib
import threading
from collections.abc import Generator

import jax
import numpy as np
from flax import nnx

from bux import parallel
from bux.architecture import Model
from bux.architecture.repeats import remat
from bux.lang import compile_runs
from bux.train import fitting
from bux.train.data import plain
from bux.train.objective import objective
from bux.train.optim import lr_multiplier, optimizer
from bux.train.trainer import Trainer

LLAMA = """
d_model = 64
vocab_size = 128
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=256, activation=swish)
block = [norm attn] [norm ffn]
model = embed block^4 norm head
"""
TRAIN = (
    "data = synthetic\noptimizer = adamw(lr=1e-2)\nseq_len = 64\nbatch = 8\nsteps = 2"
    "\nprecision = precision(compute=float32)\nmethods = methods(remat=auto)"
)
TOKENS = np.random.default_rng(0).integers(0, 128, (8, 64))
COMPILE = "/jax/core/compile/backend_compile_duration"  # jax.monitoring's event
STEP = "jit(_train_step_on_shards)"


@contextlib.contextmanager
def _step_compiles() -> Generator[list[float]]:
    """The seconds of each compile of a training step inside the block."""
    seconds: list[float] = []

    def listen(event: str, duration_secs: float, **kwargs: str | int) -> None:
        if event == COMPILE and kwargs.get("fun_name") == STEP:
            seconds.append(duration_secs)

    jax.monitoring.register_event_duration_secs_listener(listen)
    try:
        yield seconds
    finally:
        jax.monitoring.unregister_event_duration_listener(listen)


def _stepped() -> fitting.Fitting:
    """The training step of LLAMA under remat=auto, after a step on TOKENS."""
    [run] = compile_runs(f"{LLAMA}\n{TRAIN}")
    t = run.train
    with parallel.use(t.mesh, t.layout, t.methods):
        model = Model(run.arch, rngs=nnx.Rngs(0))
        tx = optimizer(t.optimizer, lr_multiplier(t.schedule, t.steps), model, t.clip)
        trainer = Trainer(model, tx, objective(t.loss, run.arch))
        trainer.train_on_batch(plain(TOKENS))
    assert isinstance(trainer._train, fitting.Fitting)
    return trainer._train


def test_auto_keeps_everything_where_the_device_does_not_say_its_memory(losses):
    with _step_compiles() as compiles:
        auto = losses(LLAMA, methods="methods(remat=auto)")
    assert len(compiles) == 1  # the step runs as it compiled to be tried
    none = losses(LLAMA, methods='methods(remat="none")')
    assert auto == none
    step = _stepped()  # on a CPU, whose memory is the host's
    assert (step.policy, [t.policy for t in step.tried]) == ("none", ["none"])


def test_auto_takes_the_first_candidate_that_fits_or_else_the_last(monkeypatch):
    monkeypatch.setattr(fitting, "capacity", lambda: 1)  # nothing fits
    step = _stepped()
    assert step.policy == remat.CANDIDATES[-1]
    assert [t.policy for t in step.tried] == list(remat.CANDIDATES)
    none, room = step.tried[0].bytes, step.tried[1].bytes
    assert none is not None and room is not None
    assert room < none  # dots_attention keeps less
    monkeypatch.setattr(fitting, "capacity", lambda: int(room / fitting.MARGIN) + 1)
    step = _stepped()
    assert step.policy == "dots_attention"
    assert [t.policy for t in step.tried] == list(remat.CANDIDATES)


def test_the_candidates_after_the_first_compile_side_by_side(monkeypatch):
    monkeypatch.setattr(fitting, "capacity", lambda: 1)
    rest = threading.Barrier(len(remat.CANDIDATES) - 1, timeout=60)
    held, calls = fitting.held, []

    def held_once_all_compile(lowered: object) -> int | None:
        calls.append(lowered)
        if len(calls) > 1:
            rest.wait()  # breaks, and fails the test, unless the rest are all here
        return held(lowered)

    monkeypatch.setattr(fitting, "held", held_once_all_compile)
    step = _stepped()
    assert [t.policy for t in step.tried] == list(remat.CANDIDATES)


def test_auto_trains_as_the_policy_it_takes(losses, monkeypatch):
    monkeypatch.setattr(fitting, "capacity", lambda: 1)
    with _step_compiles() as compiles:
        auto = losses(LLAMA, methods="methods(remat=auto)")
    assert len(compiles) == len(remat.CANDIDATES)  # the last runs as it compiled
    assert auto == losses(LLAMA, methods="methods(remat=full)")
