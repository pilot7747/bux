"""Context parallelism for recurrent mixers (bux.parallel.recurrence): `pass_state`
against `scan` on one device, with the positions split over `context` on the 8 CPU
devices the tests have, and documents that cross the devices' chunks."""

import re
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jaxtyping import Array, Float, Int

from bux import parallel
from bux.architecture import Model, Segments
from bux.architecture.linear_attention import chunk as linear_chunk
from bux.architecture.linear_attention import recurrence
from bux.architecture.rwkv7 import RWKV7
from bux.architecture.ssm import chunk as ssm_chunk
from bux.architecture.ssm import selective_scan
from bux.lang import compile
from bux.parallel.recurrence import (
    Chunk,
    Chunked,
    Positions,
    Recurrence,
    Shared,
    then,
)
from bux.train import config as tcfg
from bux.train.data import Batch
from bux.train.objective import objective
from bux.train.trainer import Trainer

# Each context size with another axis: heads over tensor, or rows over fsdp.
MESHES = {
    2: tcfg.MeshConfig(context=2, tensor=2),
    4: tcfg.MeshConfig(context=4, fsdp=2),
}
PASS = tcfg.MethodsConfig(recurrence="pass_state")
B, T = 4, 16
# Documents that cross the chunks of 2 and of 4 devices, start where a chunk does,
# and start at a chunk's last position.
POSITIONS = np.asarray(
    [
        [*range(3, 8), *range(7), *range(4)],
        [*range(T)],
        [*range(4), *range(4), *range(8)],
        [*range(7), *range(9)],
    ],
    np.int32,
)
DOCS = Segments.of(jnp.asarray(POSITIONS))


# --- The recurrences -------------------------------------------------------------------


def _normal(seed: int, *shape: int) -> Float[Array, "..."]:
    return jax.random.normal(jax.random.key(seed), shape)


def _decay(seed: int, *shape: int) -> Float[Array, "..."]:
    return -jax.random.uniform(jax.random.key(seed), shape, maxval=0.7)


H, K, V = 2, 4, 3
Q, KEYS, VALUES = _normal(0, B, T, H, K), _normal(1, B, T, H, K), _normal(2, B, T, H, V)
A_CHANNEL, A_HEAD = _decay(3, B, T, H, K), _decay(3, B, T, H, 1)  # log decays
RANK_ONE = _normal(4, B, T, H, K) / 3, _normal(5, B, T, H, K) / 3  # a and b
X, DT = _normal(6, B, T, 4, 2), jax.nn.softplus(_normal(7, B, T, 4))
STATES = _normal(8, B, T, 2, 3), _normal(9, B, T, 2, 3)  # B and C, of 2 groups
SSM_CHANNEL, SSM_HEAD = -jnp.tile(jnp.arange(1.0, 4.0), (4, 1)) / 4, -jnp.ones((4, 1))
CONTINUES = (DOCS.within > 0).astype(jnp.float32)
# Each form of a recurrence, as a function of its inputs, and the inputs.
FORMS: dict[str, tuple[Callable[..., Float[Array, "b t h v"]], tuple]] = {
    "decay_a_channel": (
        lambda q, k, v, g: recurrence(q, k, v, g, DOCS),
        (Q, KEYS, VALUES, A_CHANNEL),
    ),
    "decay_a_head": (
        lambda q, k, v, g: recurrence(q, k, v, g, DOCS),
        (Q, KEYS, VALUES, A_HEAD),
    ),
    "rank_one": (
        lambda q, k, v, g, a, b: recurrence(q, k, v, g, DOCS, a=a, b=b),
        (Q, KEYS, VALUES, A_CHANNEL, *RANK_ONE),
    ),
    "selective_scan_a_channel": (
        lambda x, dt, A, B, C: selective_scan(x, dt, A, B, C, DOCS),
        (X, DT, SSM_CHANNEL, *STATES),
    ),
    "selective_scan_a_head": (
        lambda x, dt, A, B, C: selective_scan(x, dt, A, B, C, DOCS),
        (X, DT, SSM_HEAD, *STATES),
    ),
}


def _grads(
    f: Callable[..., Float[Array, "b t h v"]], args: tuple
) -> tuple[Float[Array, "b t h v"], tuple]:
    """f's outputs, and the gradients of a weighted sum of them for every input."""

    def loss(*args):
        out = f(*args)
        return (out * jnp.cos(jnp.arange(out.size)).reshape(out.shape)).sum(), out

    argnums = tuple(range(len(args)))
    grads, out = jax.jit(jax.grad(loss, argnums=argnums, has_aux=True))(*args)
    return out, grads


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("context", MESHES)
def test_pass_state_runs_each_recurrence_as_scan_does(context, form):
    f, args = FORMS[form]
    out, grads = _grads(f, args)
    with parallel.use(MESHES[context], methods=PASS):
        split_out, split_grads = _grads(f, args)
    np.testing.assert_allclose(split_out, out, atol=1e-5)
    for split, whole in zip(split_grads, grads, strict=True):
        np.testing.assert_allclose(split, whole, atol=1e-5)


# Each form's chunk, and what it takes of positions and whole.
CHUNKS: dict[str, tuple[Chunked, Positions, Shared]] = {
    "decay_a_channel": (linear_chunk, (Q, KEYS, VALUES, A_CHANNEL, CONTINUES), ()),
    "decay_a_head": (linear_chunk, (Q, KEYS, VALUES, A_HEAD, CONTINUES), ()),
    "rank_one": (
        linear_chunk,
        (Q, KEYS, VALUES, A_CHANNEL, CONTINUES, *RANK_ONE),
        (),
    ),
    "selective_scan_a_channel": (
        ssm_chunk,
        (X, DT, *STATES, CONTINUES),
        (SSM_CHANNEL,),
    ),
    "selective_scan_a_head": (ssm_chunk, (X, DT, *STATES, CONTINUES), (SSM_HEAD,)),
}


@pytest.mark.parametrize("form", CHUNKS)
def test_a_chunk_s_transition_maps_the_state_it_starts_from_to_where_it_ends(form):
    """The second half of the positions, from the state the first ends with, is the
    rest of all of them."""
    chunk, xs, shared = CHUNKS[form]
    whole = chunk(xs, shared, True)
    first, second = (
        chunk(tuple(x[:, half] for x in xs), shared, True)
        for half in (slice(T // 2), slice(T // 2, None))
    )
    assert whole.state is not None and first.state is not None
    assert second.state is not None and second.transition is not None
    assert second.reads is not None
    read = jnp.einsum("bnhk,bhkv->bnhv", second.reads, first.state)
    np.testing.assert_allclose(second.out + read, whole.out[:, T // 2 :], atol=1e-5)
    after = then(second.transition, first.state) + second.state
    np.testing.assert_allclose(after, whole.state, atol=1e-5)
    plain = chunk(xs, shared, False)
    assert plain.transition is None and plain.reads is None


def test_pass_state_runs_the_recurrence_whole_where_nothing_splits_the_positions():
    f, args = FORMS["rank_one"]
    out = f(*args)
    for mesh in (tcfg.MeshConfig(fsdp=8), tcfg.MeshConfig(tensor=2)):
        with parallel.use(mesh, methods=PASS):
            jaxpr = str(jax.make_jaxpr(f)(*args))
            np.testing.assert_allclose(jax.jit(f)(*args), out, atol=1e-6)
        assert "shard_map" not in jaxpr


def test_a_family_without_a_transition_is_an_error_under_pass_state():
    def chunk(xs: Positions, shared: Shared, transition: bool) -> Chunk:
        return Chunk(xs[0], None, None, None)

    mine = Recurrence("my recurrence", chunk, (Q,))
    match = "pass_state: my recurrence gives no transition"
    with (
        parallel.use(MESHES[2], methods=PASS),
        pytest.raises(NotImplementedError, match=match),
    ):
        jax.jit(lambda: parallel.method("recurrence")(mine))()


# --- In a model --------------------------------------------------------------------------

PRELUDE = """
d_model = 32
vocab_size = 64
norm = rmsnorm
ffn = gated_mlp(intermediate_size=48, activation=swish)
"""
FAMILIES = {
    "deltanet": "deltanet(num_heads=2, head_dim=8)",
    "gdn": "gdn(num_heads=2, num_v_heads=4, head_dim=8, v_head_dim=4)",
    "kda": "kda(num_heads=2, head_dim=8, gate_rank=4, output_gate_rank=4)",
    "gla": "gla(num_heads=2, head_dim=8, gate_rank=4)",
    "lightning": "lightning(num_heads=2, head_dim=8, v_head_dim=4)",
    "rwkv7": (
        "rwkv7(num_heads=2, head_dim=8, decay_rank=4, a_rank=4, v_rank=4, gate_rank=4)"
    ),
    "mamba": "mamba(state_size=4)",
    "mamba2": "mamba2(state_size=4, head_dim=8, num_groups=2)",
}


def source(family: str) -> str:
    """Two blocks of the mixer; rwkv7's second mixes in the first's values."""
    first, rest = ("mix(v=@v0)", "mix(v=$v0)") if family == "rwkv7" else ("mix",) * 2
    body = f"model = embed [norm {first}] [norm ffn] [norm {rest}] [norm ffn] norm head"
    return PRELUDE + f"mix = {FAMILIES[family]}\n" + body


def model(family: str) -> Model:
    """The model of `source`, born sharded; rwkv7's output projections, which start
    at zero, drawn at random."""
    m = parallel.build(lambda: Model(compile(source(family)), rngs=nnx.Rngs(0)))
    for i, (_, node) in enumerate(nnx.iter_graph(m)):
        if isinstance(node, RWKV7):
            kernel = node.o.kernel
            kernel[...] = jax.random.normal(jax.random.key(i), kernel.shape) / 4
    return m


TOKENS = jax.random.randint(jax.random.key(0), (B, T), 0, 64)


def logits_and_grads(m: Model) -> tuple[Float[Array, "b t vocab"], list]:
    """The logits of packed rows, and the gradients of a weighted sum of them."""

    def loss(m: Model, tokens: Int[Array, "b t"], positions: Int[Array, "b t"]):
        [logits] = m.outputs(tokens, positions)
        weights = jnp.sin(jnp.arange(logits.size)).reshape(logits.shape)
        return (logits * weights).sum(), logits

    grad = nnx.jit(nnx.grad(loss, has_aux=True))
    grads, logits = grad(m, TOKENS, jnp.asarray(POSITIONS))
    return logits, jax.tree.leaves(grads)


def assert_rounds_alike(split: Float[Array, "..."], whole: Float[Array, "..."]) -> None:
    """`split` is `whole` to 5e-5 of its largest entry: as float32 sums of many
    terms round, a gradient's taken over every position, and the chunks'
    (``chunkwise``) in other orders on each device's positions than on all."""
    scale = float(jnp.abs(whole).max())
    np.testing.assert_allclose(split, whole, atol=5e-5 * scale)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("context", MESHES)
def test_every_family_computes_and_differentiates_packed_rows_as_on_one_device(
    context, family
):
    logits, grads = logits_and_grads(model(family))
    with parallel.use(MESHES[context], methods=PASS):
        split_logits, split_grads = logits_and_grads(model(family))
    assert_rounds_alike(split_logits, logits)
    for split, whole in zip(split_grads, grads, strict=True):
        assert_rounds_alike(split, whole)


# At 1e-2, these small runs are chaotic enough that rounding grows past 1e-5 (KDA).
STABLE = "adamw(lr=1e-3)"


@pytest.mark.parametrize("family", FAMILIES)
def test_a_run_trains_the_same_by_pass_state_and_rematerializes_nothing(
    losses, involuntary, family
):
    one = losses(source(family), mesh="mesh(data=1)", optimizer=STABLE)
    for mesh in ("mesh(context=2, tensor=2)", "mesh(context=4, fsdp=2)"):
        involuntary()
        split = losses(
            source(family),
            mesh=mesh,
            optimizer=STABLE,
            methods="methods(recurrence=pass_state)",
        )
        assert np.allclose(split, one, atol=1e-5), mesh
        assert involuntary() == 0, mesh


LONG = 24  # positions that no parameter has as many of a dimension


def packed_rows() -> Batch:
    """8 rows of LONG tokens, in documents of 7 that cross the chunks of 4 devices."""
    ids = jax.random.randint(jax.random.key(1), (8, LONG), 0, 64)
    positions = np.arange(8 * LONG, dtype=np.int32).reshape(8, LONG) % 7
    return Batch(np.asarray(ids), positions, np.ones((8, LONG), bool))


def collectives(family: str, method: str) -> tuple[int, int]:
    """The all-gathers of a compiled train step on context=4 × fsdp=2, and how many
    of them gather float32 arrays of every position, (b, t, ...) or (t, b, ...)."""
    arch = compile(source(family))
    with parallel.use(MESHES[4], methods=tcfg.MethodsConfig(recurrence=method)):
        loss = objective(tcfg.LossConfig(), arch)
        trainer = Trainer(model(family), optax.adamw(1e-3), loss)
        batch = parallel.shard_batch(packed_rows())
        step = trainer._train.lower(trainer.model, trainer.optimizer, batch)
        text = step.compile().as_text() or ""
    gathers = re.findall(r"= (\S+) all-gather(?:-start)?\(", text)
    whole = [g for g in gathers if re.match(rf"f32\[(\d+,)?{LONG}[,\]]", g)]
    return len(gathers), len(whole)


@pytest.mark.parametrize("family", ["gdn", "mamba2"])
def test_pass_state_gathers_transitions_where_scan_gathers_every_position(
    involuntary, family
):
    """A matrix a head, and a scalar."""
    involuntary()
    scan, scan_whole = collectives(family, "scan")
    passed, passed_whole = collectives(family, "pass_state")
    assert scan_whole > 0 and passed_whole == 0
    assert passed < scan
    assert involuntary() == 0
