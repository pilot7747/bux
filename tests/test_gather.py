"""A step's parameters gathered once and its gradients reduced once (bux.parallel.gather):
the collectives of a compiled step on the 8 CPU devices the tests have, each counted as
many times as the loops around it run, and the gradients against one device's."""

import re
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import PartitionSpec as P
from jaxtyping import Float, UInt32

from bux import parallel
from bux.architecture import Model
from bux.lang import compile_runs
from bux.parallel.gather import own_streams, spans_processes
from bux.train import config as tcfg
from bux.train.data import plain
from bux.train.objective import objective
from bux.train.optim import lr_multiplier, optimizer
from bux.train.trainer import Trainer, _accumulated, _reduce_dtype

PRELUDE = """
d_model = 64
vocab_size = 128
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=256, activation=swish)
block = [norm attn] [norm ffn]
"""
LLAMA = PRELUDE + "model = embed block^4 norm head"
MOE = (
    PRELUDE
    + "experts = moe(num_experts=4, top_k=2, "
    + "expert=gated_mlp(intermediate_size=64, activation=swish))\n"
    + "model = embed [norm attn] [norm experts] norm head"
)
TRAIN = (
    "data = synthetic\noptimizer = adamw(lr=1e-2)\nseq_len = 16\nbatch = 64\nsteps = 3"
)
TOKENS = np.random.default_rng(0).integers(0, 128, (64, 16))
SCAN = "methods(repeat=scan, microbatches={n})"
UNREDUCED = "methods(repeat=scan, microbatches={n}, accumulate=unreduced)"
BEFORE = "methods(repeat=scan, microbatches={n}, gather=layer, accumulate=reduced)"
GRADIENTS = ("reduce-scatter", "all-reduce")


# --- The collectives of a step ----------------------------------------------------------


def test_a_step_of_8_microbatches_moves_the_parameters_and_gradients_once():
    """As one of 1 does, in a scan of layers that gathers each one's as it runs, forward
    and back; where before the parameters moved 8 times as much, and the gradients.
    On one process, auto gathers once and reduces each microbatch."""
    one = _moved(LLAMA, "mesh(fsdp=8)", SCAN.format(n=1))
    eight = _moved(LLAMA, "mesh(fsdp=8)", UNREDUCED.format(n=8))
    auto = _moved(LLAMA, "mesh(fsdp=8)", SCAN.format(n=8))
    before = _moved(LLAMA, "mesh(fsdp=8)", BEFORE.format(n=8))
    assert eight["all-gather"] <= one["all-gather"]
    assert _sum(eight, GRADIENTS) <= _sum(one, GRADIENTS)
    assert auto["all-gather"] == eight["all-gather"]
    assert _sum(auto, GRADIENTS) >= 7 * _sum(one, GRADIENTS)
    assert before["all-gather"] >= 8 * eight["all-gather"]
    assert _sum(before, GRADIENTS) >= 7 * _sum(one, GRADIENTS)


def test_gathered_once_a_step_of_an_moe_reduces_its_gradients_each_microbatch():
    """Its router learns from the whole batch, so the devices cannot sum their own."""
    eight = _moved(MOE, "mesh(fsdp=8)", "methods(microbatches=8)")
    one = _moved(MOE, "mesh(fsdp=8)", "methods(microbatches=1)")
    assert eight["all-gather"] <= one["all-gather"]
    assert _sum(eight, GRADIENTS) >= 7 * _sum(one, GRADIENTS)


def test_a_pipeline_gathers_its_stages_parameters_and_reduces_them_once_a_step():
    """Its 4 microbatches go through 2 stages in 5 ticks of gpipe, gathering and
    reducing as they run before; and reduced each tick, only gathered once."""
    how = "methods(repeat=scan, microbatches=4, pipeline=gpipe{})"
    after = _moved(LLAMA, "mesh(stage=2, fsdp=4)", how.format(", accumulate=unreduced"))
    gathered = _moved(LLAMA, "mesh(stage=2, fsdp=4)", how.format(""))
    before = _moved(LLAMA, "mesh(stage=2, fsdp=4)", how.format(", gather=layer"))
    assert before["all-gather"] >= 4 * after["all-gather"] > 0
    assert gathered["all-gather"] == after["all-gather"]
    assert _sum(gathered, GRADIENTS) >= 3 * _sum(after, GRADIENTS)


# Each device's process on a mesh's grid, the rows' axes, and whether they span more.
PROCESSES = [
    (np.zeros((2, 4), int), (1,), False),  # one host
    (np.array([[0, 0, 1, 1]]), (1,), True),  # the rows split over two hosts
    (np.array([[0, 0], [1, 1]]), (1,), False),  # the hosts on another axis: stage
    (np.array([[0, 0], [1, 1]]), (0,), True),
    (np.array([[0, 0], [1, 1]]), (0, 1), True),  # data across hosts, fsdp within
    (np.array([[0, 1], [0, 1]]), (), False),  # no axis splits the rows
]


@pytest.mark.parametrize(("processes", "axes", "spans"), PROCESSES)
def test_a_reduction_over_axes_goes_between_hosts_where_their_devices_do(
    processes, axes, spans
):
    assert spans_processes(processes, axes) == spans


def test_the_tests_8_devices_are_one_process_whose_rows_span_none():
    with parallel.use(tcfg.MeshConfig(data=2, fsdp=4)):
        assert parallel.gather.rows() == ("data", "fsdp")
        assert not parallel.gather.rows_span_processes()


# --- What it computes -------------------------------------------------------------------

MESHES = [
    ("mesh(fsdp=8)", "layout()"),
    ("mesh(data=2, fsdp=4)", "layout()"),
    ("mesh(fsdp=4, tensor=2)", "layout(zero=1)"),
    ("mesh(fsdp=2, expert=2, tensor=2)", "layout()"),
]


@pytest.mark.parametrize(("mesh", "layout"), MESHES)
@pytest.mark.parametrize("accumulate", ["unreduced", "reduced", "auto"])
def test_microbatches_added_up_either_way_give_the_whole_batch_s_gradient(
    mesh, layout, accumulate
):
    whole, _ = _gradients("mesh(data=1)", "layout()", "methods(repeat=scan)")
    how = f"methods(repeat=scan, microbatches=2, accumulate={accumulate})"
    split, trainer = _gradients(mesh, layout, how)
    assert trainer.reduces_once == (accumulate == "unreduced")
    for a, b in zip(jax.tree.leaves(whole), jax.tree.leaves(split), strict=True):
        assert np.allclose(a, b, atol=2e-6)


FAMILIES = {
    "hc": PRELUDE
    + "residual = hc(expansion=2, dynamic=true)\n"
    + "model = embed [norm attn] [norm ffn] merge norm head",
    "attnres": PRELUDE
    + "residual = attnres(block_size=2)\n"
    + "model = embed [norm attn] [norm ffn] [norm attn] merge norm head",
    "nextn": PRELUDE
    + "mtp = $h! nextn [norm ffn] norm @h head\n"
    + "model = embed [norm attn] [norm ffn] norm @h head mtp^2",
    "gdn": PRELUDE
    + "delta = gdn(num_heads=2, head_dim=32)\n"
    + "model = embed [norm delta] [norm ffn] [norm attn] [norm ffn] norm head",
}


@pytest.mark.parametrize("family", FAMILIES)
def test_a_run_reduced_once_trains_as_the_whole_batch(losses, family):
    one = losses(FAMILIES[family], mesh="mesh(data=1)", batch="16")
    split = losses(
        FAMILIES[family],
        mesh="mesh(data=2, fsdp=4)",
        batch="16",
        methods="methods(microbatches=2, accumulate=unreduced)",
    )
    assert np.allclose(split, one, atol=1e-5)


@pytest.mark.parametrize("attention", ["ring", "ulysses"])
def test_context_parallel_attention_inside_a_step_reduced_once_trains_the_same(
    losses, attention
):
    """Its shard_map, manual over `context`, nests in the step's over the rows."""
    one = losses(LLAMA, mesh="mesh(data=1)", batch="16")
    how = f"methods(attention={attention}, microbatches=2, accumulate=unreduced)"
    split = losses(LLAMA, mesh="mesh(fsdp=2, context=4)", batch="16", methods=how)
    assert np.allclose(split, one, atol=1e-5)


def test_gradients_reduced_in_float32_are_not_rounded_to_bfloat16():
    """XLA:CPU sums either in float32; in compute, the sum is rounded to it. Both are
    a bfloat16 step's gradient: within 5% of float32's, as one device's is."""
    exact, _ = _gradients("mesh(data=1)", "layout()", "methods()")
    bf16 = "precision(compute=bfloat16{})"
    rounded, _ = _gradients("mesh(fsdp=8)", "layout()", "methods()", bf16.format(""))
    summed, trainer = _gradients(
        "mesh(fsdp=8)", "layout()", "methods()", bf16.format(", reduce=float32")
    )
    assert trainer.reduces_once
    for g, a, b in zip(exact, rounded, summed, strict=True):
        assert a.dtype == b.dtype == jnp.float32
        assert np.linalg.norm(a - g) < 0.05 * np.linalg.norm(g)
        assert np.linalg.norm(b - g) < 0.05 * np.linalg.norm(g)
        assert np.array_equal(_in_bfloat16(a), a)
    assert not all(np.array_equal(_in_bfloat16(b), b) for b in summed)


def test_each_device_draws_its_own_rows_masks():
    with parallel.use(tcfg.MeshConfig(fsdp=8)):
        rngs = nnx.Rngs(dropout=0)
        keys = nnx.state(rngs, nnx.RngKey)

        def drawn(keys: nnx.State) -> UInt32[jax.Array, "1 k"]:
            own = own_streams(keys, ("fsdp",))
            key = jax.tree.leaves(own)[0]
            return jax.random.key_data(key)[None]

        per_device = jax.shard_map(
            drawn, in_specs=P(), out_specs=P("fsdp"), axis_names={"fsdp"}
        )(keys)
    assert len({tuple(k) for k in np.asarray(per_device)}) == 8


# --- Settings -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "mesh", "methods", "precision", "error"),
    [
        (MOE, "mesh(fsdp=8)", "methods(accumulate=unreduced)", "", "MoE's router"),
        (MOE, "mesh(fsdp=8)", "methods()", "precision(reduce=float32)", "MoE's router"),
        (
            LLAMA,
            "mesh(fsdp=8)",
            "methods(accumulate=reduced)",
            "precision(reduce=float32)",
            "accumulate=reduced",
        ),
        (
            LLAMA,
            "mesh(fsdp=8)",
            "methods(gather=layer, accumulate=unreduced)",
            "",
            "gather=step",
        ),
        (
            LLAMA,
            "mesh(stage=2, fsdp=4)",
            "methods(repeat=scan, pipeline=gpipe)",
            "precision(reduce=float32)",
            "embedding and head",
        ),
    ],
)
def test_adding_up_unreduced_where_it_would_not_compute_the_same_is_an_error(
    source, mesh, methods, precision, error
):
    with _use(mesh, "layout()", methods), pytest.raises(ValueError, match=error):
        _trainer(source, mesh, "layout()", methods, precision or "precision()")


# --- Helpers ------------------------------------------------------------------------------


def _trainer(
    source: str, mesh: str, layout: str, methods: str, precision: str
) -> Trainer:
    """A trainer of `source` with the settings given, made inside `parallel.use`."""
    settings = f"\nmesh = {mesh}\nlayout = {layout}\nmethods = {methods}"
    [run] = compile_runs(f"{source}\n{TRAIN}\nprecision = {precision}{settings}")
    t = run.train
    model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(0)))
    tx = optimizer(t.optimizer, lr_multiplier(t.schedule, t.steps), model, t.clip)
    return Trainer(
        model,
        tx,
        objective(t.loss, run.arch),
        compute=t.precision.compute,
        microbatches=_accumulated(t.methods.microbatches),
        gather=t.methods.gather,
        accumulate=t.methods.accumulate,
        reduce=_reduce_dtype(t.precision),
    )


def _use(mesh: str, layout: str, methods: str):
    [run] = compile_runs(
        f"{LLAMA}\n{TRAIN}\nmesh = {mesh}\nlayout = {layout}\nmethods = {methods}"
    )
    return parallel.use(run.train.mesh, run.train.layout, run.train.methods)


def _gradients(
    mesh: str, layout: str, methods: str, precision: str = "precision(compute=float32)"
) -> tuple[list[Float[np.ndarray, "..."]], Trainer]:
    """The gradients of LLAMA's loss on a batch, with the settings given."""
    with _use(mesh, layout, methods):
        trainer = _trainer(LLAMA, mesh, layout, methods, precision)
        batch = parallel.shard_batch(plain(TOKENS))
        grads, _ = nnx.jit(trainer.gradients)(trainer.model, batch)
    return [np.asarray(g) for g in jax.tree.leaves(grads)], trainer


def _moved(
    source: str, mesh: str, methods: str, precision: str = "precision(compute=float32)"
) -> dict[str, int]:
    """The bytes each kind of collective moves in a compiled training step."""
    with _use(mesh, "layout()", methods):
        trainer = _trainer(source, mesh, "layout()", methods, precision)
        batch = parallel.shard_batch(plain(TOKENS))
        step = trainer._train.lower(trainer.model, trainer.optimizer, batch).compile()
        return _bytes_moved(step.as_text() or "")


def _sum(moved: dict[str, int], ops: tuple[str, ...]) -> int:
    return sum(moved[op] for op in ops)


def _in_bfloat16(a: Float[np.ndarray, "*s"]) -> Float[np.ndarray, "*s"]:
    return np.asarray(jnp.asarray(a).astype(jnp.bfloat16).astype(jnp.float32))


DTYPES = {"f32": 4, "bf16": 2, "s32": 4, "u32": 4, "pred": 1}
CALLS = r"(body|condition|calls|to_apply)=%?([\w.\-]+)"
COLLECTIVE = r"= (.+?) (all-gather|reduce-scatter|all-reduce|collective-permute)"


def _bytes_moved(hlo: str) -> dict[str, int]:
    """Return the bytes of the whole arrays each kind of collective gathers or sums,
    summed over the times it runs: every loop's trip count times its caller's."""
    computations = _computations(hlo)
    runs = _runs(computations)
    moved: dict[str, int] = defaultdict(int)
    for name, lines in computations.items():
        for line in lines:
            if m := re.search(COLLECTIVE + r"(?:-start)?\(", line):
                whole = _group(line) if m.group(2) == "reduce-scatter" else 1
                moved[m.group(2)] += runs[name] * whole * _shape_bytes(m.group(1))
    return moved


def _group(line: str) -> int:
    """Return how many devices each replica group of a collective has."""
    if m := re.search(r"replica_groups=mesh\[([^\]]*)\] \{([^}]*)\}", line):
        sizes = {a: int(n) for a, n in re.findall(r"'(\w+)'=(\d+)", m.group(1))}
        return int(np.prod([sizes[a] for a in re.findall(r"'(\w+)'", m.group(2))]))
    if m := re.search(r"replica_groups=\{\{([\d,]*)\}", line):
        return len(m.group(1).split(","))
    return 1


def _computations(hlo: str) -> dict[str, list[str]]:
    """Return each computation's instructions, the entry's under "ENTRY"."""
    found: dict[str, list[str]] = {}
    lines: list[str] = []
    for line in hlo.splitlines():
        if m := re.match(r"^(ENTRY )?%?([\w.\-]+) .*\{$", line):
            lines = found.setdefault("ENTRY" if m.group(1) else m.group(2), [])
        elif line.startswith(" "):
            lines.append(line)
    return found


def _runs(computations: dict[str, list[str]]) -> dict[str, int]:
    """Return how many times each computation runs in a step."""
    callers = defaultdict(list)
    for name, lines in computations.items():
        for line in lines:
            trip = re.search(r'"known_trip_count":\{"n":"(\d+)"', line)
            for kind, callee in re.findall(CALLS, line):
                looped = kind in ("body", "condition") and trip
                callers[callee].append((name, int(trip.group(1)) if looped else 1))
    runs: dict[str, int] = {"ENTRY": 1}

    def count(name: str) -> int:
        if name not in runs:
            runs[name] = 0  # a computation no one calls, or a cycle
            runs[name] = sum(count(c) * n for c, n in callers[name])
        return runs[name]

    return {name: count(name) for name in computations}


def _shape_bytes(shape: str) -> int:
    total = 0
    for dtype, dims in re.findall(r"(\w+)\[([\d,]*)\]", shape):
        total += DTYPES.get(dtype, 4) * int(
            np.prod([int(d) for d in dims.split(",") if d])
        )
    return total
