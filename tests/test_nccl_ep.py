"""methods(moe=nccl_ep): what runs without TransformerEngine's JAX bindings and GPUs,
which NCCL EP needs. scripts/gpu_smoke/experts.py runs the rest."""

import functools
import re
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
import test_experts

from bux import parallel, runtime
from bux.lang import compile_runs
from bux.parallel import experts, methods, nccl_ep
from bux.train import config as tcfg

SOURCE = """
d_model = 64
vocab_size = 128
norm = rmsnorm(eps=1e-5)
experts = moe(num_experts=8, top_k=2, expert=gated_mlp(intermediate_size=32, activation=swish))
model = embed [norm experts] norm head
data = synthetic
optimizer = adamw(lr=3e-4)
seq_len = 16
batch = {batch}
steps = 2
precision = precision(compute=bfloat16{dispatch})
methods = methods(microbatches={microbatches})
"""


def _run(dispatch: str = "compute", microbatches: int = 1):
    more = "" if dispatch == "compute" else f", dispatch={dispatch}"
    batch = 8 * microbatches
    source = SOURCE.format(dispatch=more, batch=batch, microbatches=microbatches)
    return compile_runs(source)[0]


def _use(sizes: dict[str, int], chunks: int = 1, stream: bool = False):
    mesh = tcfg.MeshConfig.model_validate(sizes)
    how = tcfg.MethodsConfig(
        moe="nccl_ep", capacity=1.25, moe_chunks=chunks, moe_stream=stream
    )
    return parallel.use(mesh, methods=how)


def test_a_run_by_it_starts_it_before_its_step_traces():
    assert methods.PREPARATIONS["moe", "nccl_ep"] is nccl_ep.start


def test_without_transformer_engine_it_says_how_to_get_it():
    with (
        _use({"data": 2, "expert": 4}),
        pytest.raises(RuntimeError, match="Docker image has"),
    ):
        nccl_ep.start(_run())


def test_it_sends_no_mxfp8():
    with _use({"data": 2, "expert": 4}), pytest.raises(ValueError, match="bfloat16"):
        nccl_ep.start(_run("mxfp8"))


def test_where_no_axis_splits_the_experts_it_starts_nothing():
    with _use({"fsdp": 8}):
        nccl_ep.start(_run("mxfp8"))  # it runs as sorted, which takes MXFP8
    assert nccl_ep.started is None


def test_a_device_sends_its_share_of_a_steps_tokens():
    with _use({"data": 2, "expert": 4}):
        assert nccl_ep._tokens_a_device(_run(), ("expert",)) == 8 * 16 // 8


def test_in_microbatches_a_device_sends_its_share_of_a_microbatchs_tokens():
    with _use({"data": 2, "expert": 4}):
        tokens = nccl_ep._tokens_a_device(_run(microbatches=2), ("expert",))
        assert tokens == 16 * 16 // 8 // 2


def test_in_chunks_a_device_sends_a_chunk_of_its_share_at_once():
    with _use({"data": 2, "expert": 4}, chunks=4):
        assert nccl_ep._tokens_a_device(_run(), ("expert",)) == 8 * 16 // 8 // 4
    with _use({"data": 2, "expert": 4}, chunks=3):  # 3 cuts 16 tokens in 1
        assert nccl_ep._tokens_a_device(_run(), ("expert",)) == 8 * 16 // 8


def _primitive(bind):
    return SimpleNamespace(inner_primitive=SimpleNamespace(bind=bind))


@pytest.mark.parametrize("stream", [False, True])
def test_under_moe_stream_tes_primitives_run_on_nccl_eps_stream(stream):
    bound = jax.jit(lambda x: nccl_ep._bind(_primitive(lambda x: 2 * x), x))
    with _use({"data": 2, "expert": 4}, stream=stream):
        lowered = bound.lower(jnp.ones(3)).as_text()
        assert (bound(jnp.ones(3)) == 2).all()
    assert ('_xla_stream_annotation = "1"' in lowered) == stream


def test_a_dispatch_prepares_and_sends_in_one_call_on_nccl_eps_stream(monkeypatch):
    def prepared(idx, **_):
        sizes, total = jnp.ones((1, 2), jnp.int32), jnp.ones((1, 1), jnp.int32)
        return sizes, total, jnp.ones((1, 4), jnp.uint8)

    def sent(handle, idx, x, weights, *, recv_capacity_per_rank, **_):
        rows = recv_capacity_per_rank
        return jnp.zeros((1, rows, x.shape[-1])), jnp.zeros((1, rows))

    te = SimpleNamespace(
        EpPreparePrimitive=_primitive(prepared), EpDispatchPrimitive=_primitive(sent)
    )
    monkeypatch.setattr(nccl_ep, "transformer_engine", lambda: te)
    dispatched = jax.jit(lambda *a: nccl_ep._dispatched(16, 4, 2, *a))
    x, idx, w = jnp.ones((4, 8)), jnp.zeros((4, 2), jnp.int32), jnp.ones((4, 2))
    with _use({"data": 2, "expert": 4}, stream=True):
        lowered = dispatched.lower(x, idx, w).as_text()
        assert dispatched(x, idx, w).rows.shape == (16, 8)
    assert len(re.findall(r"call @\S+ .*_xla_stream_annotation", lowered)) == 1


def test_the_rows_that_hold_picks_are_each_experts_first():
    sizes, picks = jnp.array([128, 256]), jnp.array([100, 130])
    held = nccl_ep._picks_in(sizes, picks, 512)
    at = jnp.arange(512)
    assert (held == ((at < 100) | ((at >= 128) & (at < 258)))).all()


def test_nccl_eps_settings_default_and_give_way():
    env = {"NCCL_GIN_TYPE": "3"}
    runtime.configure(env)
    assert env["NCCL_GIN_TYPE"] == "3"
    assert env["NCCL_EP_TOKENS_PER_CHUNK"] == "256"


# --- A token's path, by an emulation of TE's primitives that XLA:CPU runs ----------

ALIGNED = 4  # TE's 128, for a few tokens
ROWS = 128  # each dispatch's, as NCCL EP was started: more than a device below takes
RAGGED = "ragged_by_all_to_all"  # ragged_all_to_all, through an exchange XLA:CPU runs
EP = ("expert",)  # the experts' axis on the meshes below


def _emulated_dispatch(rows, tokens, k, x, idx, weights):
    """NCCL EP's dispatch, as its rows arrive: each expert's picks from each device in
    turn, a block a local expert padded to ALIGNED with zeros, their weights garbage
    (NaN), and the rows past the last block garbage too. It takes no more tokens than
    NCCL EP was started for."""
    assert nccl_ep.started is not None and tokens <= nccl_ep.started.tokens
    axes = EP
    peers, device = jax.lax.axis_size(axes), jax.lax.axis_index(axes)
    gathered = functools.partial(jax.lax.all_gather, axis_name=axes, tiled=True)
    group_x, picks, chosen = gathered(x), gathered(idx).reshape(-1), gathered(weights)
    local = 8 // peers
    expert = picks - device * local
    expert = jnp.where((expert >= 0) & (expert < local), expert, local)
    order = jnp.argsort(expert, stable=True)  # by expert, then device and token
    counts = jnp.bincount(expert, length=local + 1)[:local]
    sizes = -(-counts // ALIGNED) * ALIGNED
    starts = jnp.cumsum(sizes) - sizes
    sorted_expert = expert[order]
    first = jnp.cumsum(counts) - counts
    at = jnp.arange(picks.size) - first[jnp.minimum(sorted_expert, local - 1)]
    slot = jnp.where(
        sorted_expert < local, starts[jnp.minimum(sorted_expert, local - 1)] + at, rows
    )
    blocks = jnp.arange(rows)[:, None] < sizes.sum()
    arrived = jnp.where(blocks, jnp.zeros((rows, x.shape[-1]), x.dtype), jnp.nan)
    arrived = arrived.at[slot].set(group_x[order // k], mode="drop")
    weight = (
        jnp.full((rows,), jnp.nan).at[slot].set(chosen.reshape(-1)[order], mode="drop")
    )
    handle = jnp.stack([slot, order])
    return nccl_ep.Dispatched(arrived, weight, handle, sizes)


def _emulated_combine(rows, tokens, k, handle, made):
    """NCCL EP's combine: each token's sum of what its picks' rows made."""
    axes = EP
    slot, order = handle
    per_pick = jnp.take(made, slot, axis=0, mode="fill", fill_value=0)
    group = jnp.zeros((jax.lax.axis_size(axes) * tokens, made.shape[-1]), made.dtype)
    summed = jax.lax.psum(group.at[order // k].add(per_pick), axes)
    return jax.lax.dynamic_slice_in_dim(
        summed, jax.lax.axis_index(axes) * tokens, tokens
    )


@pytest.fixture
def emulated(monkeypatch):
    monkeypatch.setattr(nccl_ep, "_dispatched", _emulated_dispatch)
    monkeypatch.setattr(nccl_ep, "_combined", _emulated_combine)
    monkeypatch.setattr(nccl_ep, "ALIGNED", ALIGNED)
    started = nccl_ep.Started(8, 2, 8, test_experts.D, ROWS)
    monkeypatch.setattr(nccl_ep, "started", started)
    ragged = functools.partial(
        experts.ragged_all_to_all, exchange=test_experts.emulated
    )
    monkeypatch.setitem(methods.moe.builders, RAGGED, experts.expert_parallel(ragged))


@pytest.mark.parametrize("option", test_experts.OPTIONS)
@pytest.mark.parametrize("mesh", ["data=2, expert=4", "fsdp=2, expert=4"])
def test_a_tokens_path_computes_what_sorted_does_with_the_same_gradients(
    emulated, mesh, option
):
    one = test_experts._outputs("sorted", "one device", option)
    test_experts.assert_close(test_experts._outputs("nccl_ep", mesh, option), one)


def test_in_chunks_a_tokens_path_computes_what_sorted_does(emulated):
    one = test_experts._outputs("sorted", "one device", "latent, expert norm")
    mesh = "data=2, expert=4"
    chunked = test_experts._outputs("nccl_ep", mesh, "latent, expert norm", chunks=2)
    test_experts.assert_close(chunked, one)


def test_more_tokens_than_it_started_for_go_in_pieces(emulated):
    rows = 32  # 16 tokens a device, which NCCL EP started for 8 of (``emulated``)
    one = test_experts._outputs("sorted", "one device", "gated", rows=rows)
    mesh = "data=2, expert=4"
    pieces = test_experts._outputs("nccl_ep", mesh, "gated", rows=rows)
    test_experts.assert_close(pieces, one)


@pytest.mark.parametrize("capacity", [0.5, 1.0])
def test_under_a_capacity_it_drops_what_ragged_all_to_all_drops(emulated, capacity):
    mesh = "data=2, expert=4"
    ragged = test_experts._with_capacity(RAGGED, mesh, capacity)
    test_experts.assert_close(
        test_experts._with_capacity("nccl_ep", mesh, capacity), ragged
    )
