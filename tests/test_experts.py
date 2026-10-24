"""Expert parallelism: the methods of methods.moe that split the experts over devices,
against `sorted` on one device."""

import functools
import importlib.util
import itertools
import math
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jaxtyping import Array, Bool, Float, Int, Shaped

from bux import config as cfg
from bux import parallel
from bux.architecture import Model, Taps
from bux.architecture.experts.moe import MoE, routed
from bux.architecture.precision import blockscaled, matmuls
from bux.lang import compile
from bux.parallel import experts, methods
from bux.train import config as tcfg
from bux.train.data import plain
from bux.train.objective import objective
from bux.train.trainer import Trainer

D = 16
GATED = cfg.MLPConfig(type="gated_mlp", intermediate_size=8, activation="swish")
RMS = cfg.NormalizationConfig(type="rmsnorm")
OPTIONS = {
    "gated": {},
    "mlp": {"expert": GATED.model_copy(update={"type": "mlp", "activation": "relu2"})},
    "latent, expert norm": {
        "latent_size": 8,
        "latent_input_norm": RMS,
        "latent_output_norm": RMS,
        # With a bias, a row of zeros comes out not 0: the buffers' padding shows.
        "expert_norm": cfg.NormalizationConfig(type="layernorm", bias=True),
    },
}
# Mesh sizes and layout: expert=2 and 4, with data and fsdp; and the experts over
# `tensor`, over which `batch` leaves the tokens whole.
MESHES = {
    "data=2, fsdp=2, expert=2": ({"data": 2, "fsdp": 2, "expert": 2}, {}),
    "fsdp=2, expert=4": ({"fsdp": 2, "expert": 4}, {}),
    "data=2, expert=4": ({"data": 2, "expert": 4}, {}),
    "experts over tensor=4": ({"data": 2, "tensor": 4}, {"experts": "tensor"}),
}
ONE = {"one device": ({"data": 1}, {})}
RAGGED = "ragged_all_to_all"
EMULATED = "ragged_by_all_to_all"  # RAGGED, through an exchange XLA:CPU runs
TOKENS = "ragged_tokens_by_all_to_all"  # ragged_tokens, through it too
CPU = ["all_gather", "all_to_all", EMULATED, TOKENS]


def _varying(x: Shaped[Array, "*s"]) -> frozenset[str]:
    return jax.typeof(x).manual_axis_type.varying


def emulated(
    operand: Shaped[Array, "n d"],
    output: Shaped[Array, "m d"],
    input_offsets: Int[Array, " p"],
    send_sizes: Int[Array, " p"],
    output_offsets: Int[Array, " p"],
    recv_sizes: Int[Array, " p"],
    *,
    axis_name: tuple[str, ...],
) -> Shaped[Array, "m d"]:
    """jax.lax.ragged_all_to_all, by an all_to_all of n rows a slice."""
    # The real one types its result as `output`, so that must vary as `operand` does.
    assert _varying(output) >= _varying(operand)
    n = operand.shape[0]
    slot = jnp.arange(n)
    swap = functools.partial(
        jax.lax.all_to_all, axis_name=axis_name, split_axis=0, concat_axis=0, tiled=True
    )
    chunks = swap(operand[jnp.minimum(input_offsets[:, None] + slot, n - 1)])
    at = swap(output_offsets)[:, None] + slot
    at = jnp.where(slot < recv_sizes[:, None], at, output.shape[0])  # dropped
    return output.at[at].set(chunks, mode="drop")


@pytest.fixture(autouse=True)
def ragged_on_cpu(monkeypatch):
    """The ragged methods under the names EMULATED and TOKENS, with the exchange
    above."""
    for name, body in [
        (EMULATED, experts.ragged_all_to_all),
        (TOKENS, experts.ragged_tokens),
    ]:
        on_cpu = functools.partial(body, exchange=emulated)
        monkeypatch.setitem(methods.moe.builders, name, experts.expert_parallel(on_cpu))


def layer(**settings) -> MoE:
    given = {"num_experts": 8, "top_k": 2, "expert": GATED} | settings
    return MoE(cfg.MoEConfig.model_validate(given), D, nnx.Rngs(0))


def split(method: str, mesh: str, capacity: float | None = None):
    """`parallel.use` of the mesh and layout MESHES or ONE say, `method` and
    `capacity`."""
    sizes, layout = (MESHES | ONE)[mesh]
    return parallel.use(
        tcfg.MeshConfig.model_validate(sizes),
        tcfg.LayoutConfig.model_validate(layout),
        tcfg.MethodsConfig(moe=method, capacity=capacity),
    )


def _inputs(rows: int = 8) -> tuple[Float[Array, "b t d"], Float[Array, "b t d"]]:
    """The layer's input, and what its output is multiplied by for a loss."""
    x = jax.random.normal(jax.random.key(1), (rows, 4, D))
    return x, jax.random.normal(jax.random.key(2), x.shape)


def _loss(e: MoE, x: Float[Array, "b t d"], probe: Float[Array, "b t d"]):
    return (e(x, Taps()) * probe).sum()


@functools.cache
def _outputs(method: str, mesh: str, option: str, rows: int = 8):
    """The layer's output, and the gradients of its parameters and of its input, by
    `method` on `mesh`, on the host."""
    x, probe = _inputs(rows)
    with split(method, mesh):
        e = parallel.build(lambda: layer(**OPTIONS[option]))
        y = nnx.jit(lambda e, x: e(x, Taps()))(e, x)
        grads = nnx.jit(nnx.grad(_loss, argnums=(0, 1)))(e, x, probe)
        return jax.device_get((y, grads))


def assert_close(split, one):
    close = jax.tree.map(lambda a, b: np.allclose(a, b, atol=2e-5), split, one)
    assert jax.tree.all(close)


@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("method", CPU)
def test_each_method_computes_what_sorted_does_with_the_same_gradients(
    method, mesh, option
):
    one = _outputs("sorted", "one device", option)
    assert_close(_outputs(method, mesh, option), one)


@pytest.mark.skipif(jax.default_backend() == "cpu", reason="XLA:CPU lacks it")
@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("method", [RAGGED, "ragged_tokens"])
def test_the_ragged_methods_compute_what_sorted_does(method, mesh):
    for option in OPTIONS:
        one = _outputs("sorted", "one device", option)
        assert_close(_outputs(method, mesh, option), one)


@pytest.mark.parametrize(
    ("method", "exchanges"),
    # Forward: the rows there and back, and ragged_tokens' picks and weights there;
    # backward, the transposes of all but the picks.
    [(RAGGED, 4), ("ragged_tokens", 7)],
)
def test_the_ragged_methods_lower_on_cpu_forward_and_backward(method, exchanges):
    """XLA:CPU cannot compile them, but they trace."""
    x, probe = _inputs()
    with split(method, "fsdp=2, expert=4"):
        e = parallel.build(lambda: layer(**OPTIONS["latent, expert norm"]))
        lowered = nnx.jit(nnx.grad(_loss, argnums=(0, 1))).lower(e, x, probe)
    assert lowered.as_text().count("ragged_all_to_all") == exchanges


def test_a_slice_in_pieces_is_sent_in_a_row_and_received_as_it_is_sent():
    whole = experts.Plan(*map(jnp.array, ([0, 5], [5, 2], [3, 0], [4, 1])))
    cut = experts._in_pieces(whole, 3)
    assert cut.input_offsets.tolist() == [0, 2, 4, 5, 6, 7]
    assert cut.send_sizes.tolist() == [2, 2, 1, 1, 1, 0]
    assert cut.output_offsets.tolist() == [3, 5, 7, 0, 1, 2]
    assert cut.recv_sizes.tolist() == [2, 2, 0, 1, 0, 0]  # as the senders cut theirs


def test_where_the_mesh_does_not_split_the_experts_a_method_sorts():
    e = layer(num_experts=6)  # expert=4 does not divide them
    x, _ = _inputs()
    with split("all_to_all", "fsdp=2, expert=4"):
        hlo = nnx.jit(lambda e, x: e(x, Taps())).lower(e, x).compile().as_text() or ""
    assert "all-to-all" not in hlo


def test_tokens_that_do_not_split_over_the_experts_axes_are_an_error():
    with pytest.raises(
        ValueError, match=r"2 rows do not split over \('data', 'tensor'"
    ):
        _outputs("all_to_all", "experts over tensor=4", "gated", rows=2)


# --- A capacity ------------------------------------------------------------------------

GROUPED = "data=2, expert=4"  # two groups of 4 devices, a row of 4 tokens each
RAGGEDS = [EMULATED, TOKENS]
BOUNDED = ["all_gather", *RAGGEDS]  # the methods a capacity bounds


def fits(
    idx: Int[np.ndarray, "b t k"], capacity: float, once: bool
) -> Bool[np.ndarray, "b t k"]:
    """Which picks fit on GROUPED with `capacity`, as the methods it bounds lay them
    out: on device j of a group, expert 0's picks from each of the group's devices in
    turn, then expert 1's, and so on, to `capacity` times the picks a device has.
    `once`, as ragged_tokens: of the tokens that visit it from each device in turn,
    first, that many of them, or as many as could come."""
    groups, peers, local = 2, 4, 2
    picks = idx.reshape(groups, peers, -1, idx.shape[-1])  # [g, i, t, s]
    tokens, k = picks.shape[2:]
    room = math.ceil(capacity * tokens * k)
    kept = np.zeros(picks.shape, bool)
    for g, j in itertools.product(range(groups), range(peers)):
        visits = [
            (i, t)
            for i, t in itertools.product(range(peers), range(tokens))
            if (picks[g, i, t] // local == j).any()
        ]
        visits = visits[: min(room, peers * tokens)] if once else visits
        taken = 0
        for e in range(j * local, (j + 1) * local):
            for i, t in visits:
                for s in np.flatnonzero(picks[g, i, t] == e):
                    kept[g, i, t, s], taken = taken < room, taken + 1
    return kept.reshape(idx.shape)


def _training(e: MoE, x: Float[Array, "b t d"]):
    """The layer's output in a training forward, and the metrics it reports."""
    taps = Taps(training=True)
    return e(x, taps), {name: value for name, _, value in taps.losses}


def _grads(e: MoE, x: Float[Array, "b t d"], probe: Float[Array, "b t d"]):
    return nnx.grad(_loss, argnums=(0, 1))(e, x, probe)


def _with_capacity(method: str, mesh: str, capacity: float | None):
    """As `_outputs`, for the option `gated`, and the metrics of a training forward."""
    x, probe = _inputs()
    with split(method, mesh, capacity):
        e = parallel.build(lambda: layer(**OPTIONS["gated"]))
        (y, reported), grads = nnx.jit(_training)(e, x), nnx.jit(_grads)(e, x, probe)
        return jax.device_get((y, grads, reported))


@pytest.fixture
def masked(monkeypatch):
    """Register "masked": `sorted` with the weights of the picks `fits` drops at 0;
    return their share."""

    def using(capacity: float, once: bool) -> float:
        x, _ = _inputs()
        e = layer(**OPTIONS["gated"])
        mask = jnp.asarray(fits(np.asarray(e._route(x)[0]), capacity, once))

        def run(moe, z, idx, weights, taps=None):
            kept = jnp.where(mask, weights, 0)
            kernels, ffn = moe.kernels(), moe.config.expert
            return routed(z, idx, kept, *kernels, ffn, moe.expert_norm)

        monkeypatch.setitem(methods.moe.builders, "masked", run)
        return float(1 - mask.mean())

    return using


@pytest.mark.parametrize("capacity", [0.5, 1.0])
@pytest.mark.parametrize("method", BOUNDED)
def test_a_capacity_drops_the_picks_past_it_and_reports_their_share(
    masked, method, capacity
):
    share = masked(capacity, once=method == TOKENS)
    assert share > 0  # else the test tests nothing
    y, grads, reported = _with_capacity(method, GROUPED, capacity)
    want, want_grads, _ = _with_capacity("masked", "one device", None)
    assert_close((y, grads), (want, want_grads))
    assert reported["dropped"] == pytest.approx(share)


@pytest.mark.parametrize("method", BOUNDED)
def test_a_capacity_that_every_pick_fits_in_drops_none(method):
    y, grads, reported = _with_capacity(method, GROUPED, 4.0)  # all 4 peers' picks
    want, want_grads, nothing = _with_capacity("sorted", "one device", None)
    assert_close((y, grads), (want, want_grads))
    assert reported["dropped"] == 0 and "dropped" not in nothing


def _garbage(rows: int, like: Shaped[Array, "*s d"]) -> Shaped[Array, "rows d"]:
    """What jax.lax.empty may hold: NaN, or of integers, one far past any expert."""
    fill = jnp.nan if jnp.issubdtype(like.dtype, jnp.floating) else 2**30
    return jnp.full((rows, like.shape[-1]), fill, like.dtype)


@pytest.mark.parametrize("capacity", [None, 1.0])
@pytest.mark.parametrize("method", RAGGEDS)
def test_what_the_buffers_held_before_the_exchange_never_shows(
    monkeypatch, masked, method, capacity
):
    """The buffers are jax.lax.empty, garbage on a GPU and zeros on a CPU."""
    monkeypatch.setattr(experts, "_buffer", _garbage)
    y, grads, _ = _with_capacity(method, GROUPED, capacity)
    if capacity is None:
        want, want_grads, _ = _with_capacity("sorted", "one device", None)
    else:
        masked(capacity, once=method == TOKENS)
        want, want_grads, _ = _with_capacity("masked", "one device", None)
    assert_close((y, grads), (want, want_grads))


# --- FP8 dispatch ----------------------------------------------------------------------

WIDE = 64  # whole blocks of 32, which MXFP8 takes


def _quantized(z: Float[Array, "*s d"]) -> Float[Array, "*s d"]:
    """z in MXFP8 and back, its gradient as z's."""
    back = blockscaled.dequantized(blockscaled.mx_quantize(z.reshape(-1, WIDE)))
    return z + jax.lax.stop_gradient(back.reshape(z.shape) - z)


@functools.cache
def _dispatched(method: str, mesh: str, dispatch: Literal["compute", "mxfp8"]):
    """The layer's output and gradients, as `_outputs`, at WIDE, its tokens sent in
    `dispatch`."""
    x = jax.random.normal(jax.random.key(1), (8, 4, WIDE))
    probe = jax.random.normal(jax.random.key(2), x.shape)
    config = cfg.MoEConfig(num_experts=8, top_k=2, expert=GATED)
    with split(method, mesh), matmuls.use(tcfg.PrecisionConfig(dispatch=dispatch)):
        e = parallel.build(lambda: MoE(config, WIDE, nnx.Rngs(0)))
        y = nnx.jit(lambda e, x: e(x, Taps()))(e, x)
        grads = nnx.jit(nnx.grad(_loss, argnums=(0, 1)))(e, x, probe)
        return jax.device_get((y, grads))


@pytest.mark.parametrize("mesh", ["data=2, fsdp=2, expert=2", GROUPED])
@pytest.mark.parametrize("method", RAGGEDS)
def test_a_dispatch_in_mxfp8_computes_what_sorted_does_of_its_tokens_quantized(
    monkeypatch, method, mesh
):
    """Its gradient is the exchange's, as if the tokens went unquantized."""

    def run(moe, z, idx, weights, taps=None):
        kernels, ffn = moe.kernels(), moe.config.expert
        return routed(_quantized(z), idx, weights, *kernels, ffn, moe.expert_norm)

    monkeypatch.setitem(methods.moe.builders, "quantized", run)
    want = _dispatched("quantized", "one device", "compute")
    got = _dispatched(method, mesh, "mxfp8")
    assert_close(got, want)
    assert not np.allclose(got[0], _dispatched(method, mesh, "compute")[0])


def test_a_dispatch_in_mxfp8_sends_a_byte_a_value_and_one_a_scale_of_32():
    x = jax.random.normal(jax.random.key(1), (8, 4, WIDE))
    config = cfg.MoEConfig(num_experts=8, top_k=2, expert=GATED)
    precision = tcfg.PrecisionConfig(dispatch="mxfp8")
    with split(RAGGED, "fsdp=2, expert=4"), matmuls.use(precision):
        e = parallel.build(lambda: MoE(config, WIDE, nnx.Rngs(0)))
        loss = nnx.grad(lambda e, x: e(x, Taps()).sum(), argnums=(0, 1))
        lowered = nnx.jit(loss).lower(e, x).as_text()
    exchanges = [line for line in lowered.splitlines() if "@ragged_all_to_all" in line]
    sent = [line.split(" : (")[1].split(",")[0] for line in exchanges]
    # Forward, there in bytes and back; backward, both ways in float32.
    assert sorted(sent) == sorted(
        [f"tensor<8x{WIDE + WIDE // 32}xui8>"]
        + ["tensor<32x64xf32>", "tensor<8x64xf32>", "tensor<32x64xf32>"]
    )


# --- Training --------------------------------------------------------------------------

ARCH = """
d_model = 64
vocab_size = 128
norm = rmsnorm
rescale = rmsnorm(eps=1e-8, zero_centered=true, scale_input=true)
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
experts = moe(num_experts=8, top_k=2, num_shared_experts=1, shared_expert_gate=true,
              router=sigmoid, load_balancing=aux_free, latent_size=32,
              latent_input_norm=norm, expert_norm=rescale,
              expert=gated_mlp(intermediate_size=64, activation=swish))
model = embed [norm attn] [norm experts] [norm attn] [norm experts] norm head
"""


# Of each of ARCH's 2 MoEs: forward there and back, backward the transposes; and the
# forward's again, under a policy that keeps none of what they bring.
@pytest.mark.parametrize(
    ("remat", "exchanges"), [("none", 4), ("dots", 4), ("full", 6)]
)
def test_dots_keeps_what_the_ragged_exchanges_bring_and_runs_none_again(
    remat, exchanges
):
    how = tcfg.MethodsConfig(moe=RAGGED, remat=remat)
    with parallel.use(tcfg.MeshConfig(fsdp=2, expert=4), methods=how):
        model = parallel.build(lambda: Model(compile(ARCH), rngs=nnx.Rngs(0)))
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)
        tokens = jnp.zeros((8, 16), jnp.int32)

        def grads(params: nnx.State, rest: nnx.State) -> nnx.State:
            m = nnx.merge(graphdef, params, rest)
            return nnx.grad(lambda m: m(tokens).mean())(m)

        text = jax.jit(grads).lower(params, rest).as_text()
    assert text.count("@ragged_all_to_all") == 2 * exchanges


TRAINING = [
    "mesh(data=2, fsdp=2, expert=2)",
    "mesh(fsdp=2, expert=4)",
    "mesh(fsdp=2, expert=2, tensor=2)",
]


@pytest.fixture(scope="module")
def on_one_device() -> dict[str, list[float]]:
    """The losses of ARCH on one device, once trained."""
    return {}


@pytest.mark.parametrize("mesh", TRAINING)
@pytest.mark.parametrize("method", CPU)
def test_a_run_trains_the_same_with_its_experts_split_and_rematerializes_nothing(
    losses, involuntary, on_one_device, method, mesh
):
    if not on_one_device:
        on_one_device["losses"] = losses(ARCH, mesh="mesh(data=1)")
    involuntary()
    split = losses(ARCH, mesh=mesh, methods=f"methods(moe={method})")
    assert involuntary() == 0
    assert np.allclose(split, on_one_device["losses"], atol=1e-5)


@functools.cache
def _biases(method: str, mesh: str) -> list[Float[np.ndarray, " e"]]:
    """Every MoE's router bias after two steps of training by `method` on `mesh`."""
    arch = compile(ARCH)
    batch = plain(np.random.default_rng(0).integers(0, 128, (8, 16)))
    with split(method, mesh):
        model = parallel.build(lambda: Model(arch, rngs=nnx.Rngs(0)))
        trainer = Trainer(model, optax.adamw(1e-2), objective(tcfg.LossConfig(), arch))
        trainer.train_on_batch(batch)
        trainer.train_on_batch(batch)
    moes = [m for _, m in nnx.iter_modules(model) if isinstance(m, MoE)]
    return [np.asarray(m.router_bias[...]) for m in moes if m.router_bias is not None]


@pytest.mark.parametrize("method", CPU)
def test_the_router_bias_moves_as_it_does_on_one_device(method):
    one = _biases("sorted", "one device")
    assert len(one) == 2 and all((b != 0).any() for b in one)
    split = _biases(method, "data=2, fsdp=2, expert=2")
    assert all(np.array_equal(a, b) for a, b in zip(split, one, strict=True))


def test_the_gpu_smoke_script_compares_a_layer_on_cpu():
    path = Path(__file__).parent.parent / "scripts" / "gpu_smoke" / "experts.py"
    spec = importlib.util.spec_from_file_location("gpu_smoke_experts", path)
    assert spec is not None and spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    one = smoke.layer(smoke.ONE, "sorted")
    split = smoke.layer({"fsdp": 2, "expert": 4}, "all_to_all")
    assert smoke.differs(split, one) < 1e-4
