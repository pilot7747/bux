import itertools
import json
import math
import struct
import sys
import types
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from conftest import RUN
from flax import nnx

from bux import parallel
from bux.architecture import Model, observe
from bux.lang import compile_runs
from bux.train import observe as observed
from bux.train.config import WandbConfig
from bux.train.data import plain
from bux.train.objective import objective
from bux.train.sinks import JSONL, Console, Histogram, Sink, summary
from bux.train.trackers import TensorBoard, Wandb
from bux.train.trainer import Trainer, train

DENSE = """
d_model = 32
vocab_size = 64
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=64, activation=swish)
block = [norm attn] [norm ffn]
model = embed block^4 norm head
"""

MOE = """
d_model = 32
vocab_size = 64
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
moe = moe(num_experts=4, top_k=2, expert=gated_mlp(intermediate_size=16, activation=swish), aux_loss_coef=0.01)
block = [norm attn] [norm moe]
model = embed block^2 @h norm head $h nextn(norm=norm) [norm attn] norm head
"""


class Steps(Sink):
    """Every write's scalars, by step."""

    def __init__(self):
        self.steps: dict[int, dict[str, float]] = {}
        self.folder = Path()  # the run's directory, once trained

    def write(self, step, scalars):
        self.steps.setdefault(step, {}).update(scalars)


def source(arch: str, **settings: str) -> str:
    """`arch` with the settings of `RUN`, `settings` over them."""
    given = RUN | {"steps": "1"} | settings
    return arch + "".join(f"\n{k} = {v}" for k, v in given.items())


@pytest.fixture
def steps(tmp_path):
    """Train the model of `source` with `settings` and `RUN` where they are silent,
    and return what every step logged, and the run's directory."""
    folders = itertools.count()

    def trained(arch: str, n: int = 2, **settings: str) -> Steps:
        [run] = compile_runs(source(arch, steps=str(n), **settings))
        sink = Steps()
        out = tmp_path / str(next(folders))
        train(run, out, echo=lambda _: None, sinks=[sink])
        sink.folder = next(out.iterdir())
        return sink

    return trained


def losses(sink: Steps) -> list[float]:
    return [sink.steps[s]["loss"] for s in sorted(sink.steps)]


def layered(scalars: dict[str, float], stat: str) -> dict[str, float]:
    """The metric `stat` of each layer in `scalars`, by layer."""
    return {
        k.rsplit("/", 1)[1]: v for k, v in scalars.items() if k.startswith(stat + "/")
    }


def by_layer(scalars: dict[str, float]) -> list[str]:
    """The metrics of a layer in `scalars`: ``act/rms/attn.00``."""
    return [k for k in scalars if k.count("/") >= 2]


SCAN = "methods(repeat=scan)"


@pytest.mark.parametrize(
    "how",
    [
        {},
        {"methods": SCAN},
        {"methods": "methods(microbatches=2)", "batch": "16"},
        {"mesh": "mesh(fsdp=4, tensor=2)"},
        {"methods": "methods(microbatches=2, interleave=true)", "batch": "16"},
        {
            "mesh": "mesh(stage=2)",
            "methods": 'methods(repeat=scan, microbatches=2, pipeline="gpipe")',
        },
        {
            "mesh": "mesh(stage=2)",
            "methods": 'methods(repeat=scan, microbatches=2, pipeline="gpipe", '
            "stages=[3, 1])",
        },
    ],
    ids=["plain", "scan", "microbatches", "fsdp_tp", "interleave", "gpipe", "uneven"],
)
def test_a_step_that_observes_updates_as_the_others_and_measures_every_layer(
    steps, how
):
    plain_run = steps(DENSE, 3, **how)
    seen = steps(DENSE, 3, observe="observe(every=2)", **how)
    np.testing.assert_allclose(losses(seen), losses(plain_run), rtol=1e-5)
    at = seen.steps[2]
    assert not by_layer(plain_run.steps[2])  # the others measure no layer
    assert layered(at, "act/rms").keys() == {
        f"{a}.{i:02d}" for a in ("attn", "ffn") for i in range(4)
    }
    assert layered(at, "attn/entropy").keys() == {f"attn.{i:02d}" for i in range(4)}
    names = {"embed.00", "norm.00", "head.00", *layered(at, "act/rms")}
    assert (
        layered(at, "grad/norm").keys() == layered(at, "update/ratio").keys() == names
    )
    # The layers' gradients make up the whole one.
    whole = math.sqrt(sum(v**2 for v in layered(at, "grad/norm").values()))
    assert whole == pytest.approx(at["grad_norm"], rel=1e-4)
    assert 0 < min(layered(at, "attn/entropy").values()) <= 1
    assert not by_layer(seen.steps[3])  # nor do the steps between


@pytest.mark.parametrize("pipeline", ["1f1b", "zbh1"])
def test_a_scheduled_pipeline_observes_the_layers_after_its_cut(steps, pipeline):
    how = {
        "batch": "16",
        "mesh": "mesh(stage=2)",
        "methods": f'methods(repeat=scan, microbatches=4, pipeline="{pipeline}")',
    }
    gpipe = {**how, "methods": how["methods"].replace(pipeline, "gpipe")}
    plain_run = steps(MOE, 2, **how)
    seen = steps(MOE, 2, observe="observe(every=1)", **how)
    whole = steps(MOE, 2, observe="observe(every=1)", **gpipe)
    np.testing.assert_allclose(losses(seen), losses(plain_run), rtol=1e-5)
    at, all_of = seen.steps[2], whole.steps[2]
    # Numbered as gpipe numbers them, and the repeat's alike, a microbatch at a
    # time in both; the MTP's after it, a microbatch at a time here alone.
    for stat in ("act/rms", "act/max_abs", "router/max_load", "grad/norm"):
        assert layered(at, stat).keys() == layered(all_of, stat).keys()
        for layer, value in layered(at, stat).items():
            if layer != "attn.02":
                assert value == pytest.approx(layered(all_of, stat)[layer], rel=1e-4)
    assert layered(at, "router/max_load").keys() == {"moe.00", "moe.01"}
    assert "act/rms/attn.02" in at


def test_a_step_that_does_not_observe_compiles_as_without_the_setting():
    [run] = compile_runs(source(MOE))
    batch = plain(np.zeros((8, 16), np.int32))
    texts = []
    for every in (None, 1):
        with parallel.use(run.train.mesh):
            model = Model(run.arch, rngs=nnx.Rngs(0))
            loss = objective(run.train.loss, run.arch)
            trainer = Trainer(model, optax.adamw(1e-3), loss, observe_every=every)
            texts.append(trainer.lower(batch).as_text())
    assert texts[0] == texts[1]


def test_moe_observes_its_routing_and_every_head_its_loss(steps):
    seen = steps(MOE, 2, observe="observe(every=1)", methods=SCAN)
    at = seen.steps[2]
    loads = layered(at, "router/max_load")
    assert loads.keys() == {"moe.00", "moe.01"}
    assert all(v >= 1 for v in loads.values())
    assert all(0 <= v <= 1 for v in layered(at, "router/entropy").values())
    assert set(layered(at, "act/rms")) == {
        "attn.00",
        "attn.01",
        "attn.02",
        "moe.00",
        "moe.01",
    }
    # Every head's loss, the first's as `loss`, each step; their z-losses observed.
    assert at["lm_loss/t+1"] == at["loss"]
    assert at["mtp"] == pytest.approx(at["lm_loss/t+2"])
    assert "lm_loss/t+2" in seen.steps[1] and "z_loss/t+2" in at


def test_layers_are_named_alike_unrolled_and_scanned():
    keys = []
    for repeat in ("unroll", "scan"):
        [run] = compile_runs(source(MOE, methods=f"methods(repeat={repeat})"))
        with parallel.use(run.train.mesh, run.train.layout, run.train.methods):
            model = Model(run.arch, rngs=nnx.Rngs(0))
            where = observed.layers(model)
            keys.append(observed.norms(where, nnx.state(model, nnx.Param)))
    assert keys[0].keys() == keys[1].keys()
    for k in keys[0]:
        assert float(keys[0][k]) == pytest.approx(float(keys[1][k]), rel=1e-6)


# A MoE outside any branch: a layer of its own, by its name.
BARE_MOE = """
d_model = 32
vocab_size = 64
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
moe = moe(num_experts=4, top_k=2, expert=gated_mlp(intermediate_size=16, activation=swish))
model = embed [norm attn] moe^2 norm head
"""


@pytest.mark.parametrize("repeat", ["unroll", "scan"])
def test_a_layer_outside_any_branch_observes_by_its_name_as_its_parameters_are(
    repeat,
):
    [run] = compile_runs(source(BARE_MOE, methods=f"methods(repeat={repeat})"))
    with parallel.use(run.train.mesh, run.train.layout, run.train.methods):
        model = Model(run.arch, rngs=nnx.Rngs(0))
        with observe.use():
            _, reported = model.training_heads(jnp.zeros((2, 8), jnp.int32))
        where = observed.layers(model)
    seen = observe.named(r for r in reported if observe.is_observed(r.name))
    loads = {k for k in seen if k.startswith("router/max_load/")}
    keys = {key for layer in where.values() for key in layer.keys}
    assert loads == {"router/max_load/moe.00", "router/max_load/moe.01"}
    assert {"moe.00", "moe.01"} <= keys


def test_histograms_share_out_every_entry_of_a_layer(steps):
    seen = steps(DENSE, 2, observe="observe(every=2, histograms=true)")
    lines = [json.loads(line) for line in (seen.folder / "histograms.jsonl").open()]
    [line] = lines
    assert line["step"] == 2 and len(line["edges"]) == observe.HIGH - observe.LOW + 2
    assert {"act/hist/attn.00", "grad/hist/embed.00"} <= line.keys()
    for name, shares in line.items():
        if "hist" in name:
            assert sum(shares) == pytest.approx(1, abs=1e-5), name
    assert not any("hist" in k for k in seen.steps[2])  # in histograms.jsonl alone


def test_the_log_says_where_the_host_spends_a_step(steps):
    seen = steps(DENSE, 2, log="log(every=1)")
    first, second = seen.steps[1], seen.steps[2]
    assert first["time/compile_s"] > 0 and "time/compile_s" not in second
    for part in ("step", "data", "launch", "wait", "hooks"):
        assert second[f"time/{part}_ms"] >= 0


def test_the_console_writes_lines_a_container_log_reads():
    lines = []
    console = Console("run", 1000, echo=lines.append)
    console.write(20, {"loss": 2.5, "grad_norm": float("nan"), "time/step_ms": 50.0})
    console.write(
        20,
        {
            "act/rms/attn.00": 0.9,
            "act/rms/attn.01": 41.0,
            "attn/entropy/attn.00": 0.2,
            "attn/entropy/attn.01": 0.9,
        },
    )
    assert lines[0].startswith("run step   20/1000  loss 2.5000  grad nan")
    assert lines[0].endswith("0:00:49  !! not finite: grad_norm")
    assert (
        lines[1]
        == "run step   20/1000  observed  rms 0.9-41 (attn.01)  attn entropy 0.20-0.90 (attn.00)"
    )
    assert not any(c in line for line in lines for c in "\r\x1b")
    assert summary({"loss": 1.0}) is None


def test_histograms_go_to_a_file_of_their_own_which_a_resume_cuts(tmp_path):
    h = Histogram(np.arange(3.0), np.array([0.25, 0.75]))
    sink = JSONL(tmp_path / "m.jsonl", histograms=tmp_path / "h.jsonl")
    for step in (10, 20):
        sink.write(step, {"loss": 1.0})
        sink.histograms(step, {"act/hist/attn.00": h})
    sink.close()
    JSONL(tmp_path / "m.jsonl", start=10, histograms=tmp_path / "h.jsonl").histograms(
        15, {"act/hist/attn.00": h}
    )
    lines = [json.loads(line) for line in (tmp_path / "h.jsonl").open()]
    assert [line["step"] for line in lines] == [10, 15]
    assert lines[0]["act/hist/attn.00"] == [0.25, 0.75]


def fake_wandb(monkeypatch) -> types.SimpleNamespace:
    """A wandb module that keeps what it was given."""
    calls = types.SimpleNamespace(init=[], logged=[], finished=0)

    class Run:
        def log(self, data, step):
            calls.logged.append((step, data))

        def finish(self):
            calls.finished += 1

    def init(**kwargs):
        calls.init.append(kwargs)
        return Run()

    class Wandb(types.ModuleType):
        @staticmethod
        def init(**kwargs):
            return init(**kwargs)

        @staticmethod
        def Histogram(np_histogram):
            return "histogram", np_histogram

    module = Wandb("wandb")
    monkeypatch.setitem(sys.modules, "wandb", module)
    return calls


def test_wandb_logs_a_run_s_metrics_and_resumes_the_w_b_run_it_started(
    tmp_path, monkeypatch
):
    calls = fake_wandb(monkeypatch)
    path = "examples/experiments/smoke.bux"
    sweep = ["optimizer.lr = 1e-3 | 3e-3"]
    run = compile_runs(Path(path).read_text(), filename=path, overrides=sweep)[0]
    sink = Wandb(WandbConfig(project="p", tags=("a",)), run, tmp_path, start=0)
    sink.write(10, {"loss": 2.0})
    sink.histograms(10, {"act/hist/attn.00": Histogram(np.arange(3.0), np.ones(2) / 2)})
    sink.close()
    [made] = calls.init
    assert (
        made["project"] == "p" and made["name"] == run.name and made["group"] == "smoke"
    )
    assert made["config"]["id"] == run.id and made["config"]["choices"] == list(
        run.choices
    )
    assert calls.logged[0] == (10, {"loss": 2.0}) and calls.finished == 1
    assert calls.logged[1][1]["act/hist/attn.00"][0] == "histogram"
    Wandb(WandbConfig(), run, tmp_path, start=10)  # resumes the W&B run it started
    Wandb(WandbConfig(), run, tmp_path, start=0)  # a fresh run starts its own
    ids = [c["id"] for c in calls.init]
    assert ids[0] == ids[1] != ids[2] and all(len(i) == 8 for i in ids)


def events(folder) -> list:
    """The events of the TensorBoard files in `folder`, in order: records of a
    length, its checksum, an Event and its checksum."""
    from tensorboardX.proto import event_pb2

    found = []
    for path in sorted(folder.glob("events.out.tfevents.*")):
        data, i = path.read_bytes(), 0
        while i < len(data):
            (n,) = struct.unpack("<Q", data[i : i + 8])
            event = event_pb2.Event()  # ty: ignore[unresolved-attribute]
            event.ParseFromString(data[i + 12 : i + 12 + n])
            found.append(event)
            i += 12 + n + 4
    return found


def test_tensorboard_writes_scalars_and_histograms_and_drops_what_a_resume_redoes(
    tmp_path,
):
    sink = TensorBoard(tmp_path, start=0)
    sink.write(1, {"loss": 2.0})
    sink.histograms(1, {"act/hist/attn.00": Histogram(np.arange(3.0), np.ones(2) / 2)})
    sink.close()
    values = [(e.step, v.tag) for e in events(tmp_path) for v in e.summary.value]
    assert values == [(1, "loss"), (1, "act/hist/attn.00")]
    TensorBoard(tmp_path, start=1).close()
    restart = [e for e in events(tmp_path) if e.HasField("session_log")]
    assert [(e.step, e.session_log.status) for e in restart] == [(2, 1)]  # START


def test_observing_and_tracking_keep_the_run_s_id():
    [plain_run] = compile_runs(source(DENSE))
    [watched] = compile_runs(
        source(
            DENSE,
            observe="observe(every=5, histograms=true)",
            log='log(every=1, trackers=[wandb(project="x"), tensorboard])',
        )
    )
    assert watched.id == plain_run.id
    assert watched.train.log.trackers[1].type == "tensorboard"
