"""Expert parallelism: the methods of methods.moe that split the experts over devices,
against `sorted` on one device."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jaxtyping import Array, Float, Int, Shaped

from bux import config as cfg
from bux import parallel
from bux.architecture import Model
from bux.architecture.moe import MoE
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
CPU = ["all_gather", "all_to_all", EMULATED]


def _varying(x: Shaped[Array, "*s"]) -> frozenset[str]:
    return jax.typeof(x).manual_axis_type.varying


def emulated(
    operand: Float[Array, "n d"],
    output: Float[Array, "m d"],
    input_offsets: Int[Array, " p"],
    send_sizes: Int[Array, " p"],
    output_offsets: Int[Array, " p"],
    recv_sizes: Int[Array, " p"],
    *,
    axis_name: tuple[str, ...],
) -> Float[Array, "m d"]:
    """jax.lax.ragged_all_to_all, a slice a peer, by an all_to_all of n rows a
    peer."""
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
    """RAGGED's method under the name EMULATED, with the exchange above."""
    ragged = functools.partial(experts.ragged, exchange=emulated)
    monkeypatch.setitem(methods.moe.builders, EMULATED, experts.method(ragged))


def layer(**settings) -> MoE:
    given = {"num_experts": 8, "top_k": 2, "expert": GATED} | settings
    return MoE(cfg.MoEConfig.model_validate(given), D, nnx.Rngs(0))


def split(method: str, mesh: str):
    """`parallel.use` of the mesh and layout MESHES or ONE say, and `method`."""
    sizes, layout = (MESHES | ONE)[mesh]
    return parallel.use(
        tcfg.MeshConfig.model_validate(sizes),
        tcfg.LayoutConfig.model_validate(layout),
        tcfg.MethodsConfig(moe=method),
    )


def _inputs(rows: int = 8) -> tuple[Float[Array, "b t d"], Float[Array, "b t d"]]:
    """The layer's input, and what its output is multiplied by for a loss."""
    x = jax.random.normal(jax.random.key(1), (rows, 4, D))
    return x, jax.random.normal(jax.random.key(2), x.shape)


def _loss(e: MoE, x: Float[Array, "b t d"], probe: Float[Array, "b t d"]):
    return (e(x, {}) * probe).sum()


@functools.cache
def _outputs(method: str, mesh: str, option: str, rows: int = 8):
    """The layer's output, and the gradients of its parameters and of its input, by
    `method` on `mesh`, on the host."""
    x, probe = _inputs(rows)
    with split(method, mesh):
        e = parallel.build(lambda: layer(**OPTIONS[option]))
        y = nnx.jit(lambda e, x: e(x, {}))(e, x)
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
def test_ragged_all_to_all_computes_what_sorted_does(mesh):
    for option in OPTIONS:
        one = _outputs("sorted", "one device", option)
        assert_close(_outputs(RAGGED, mesh, option), one)


def test_ragged_all_to_all_lowers_on_cpu_forward_and_backward():
    """XLA:CPU cannot compile it, but it traces: two exchanges forward, and their
    transposes."""
    x, probe = _inputs()
    with split(RAGGED, "fsdp=2, expert=4"):
        e = parallel.build(lambda: layer(**OPTIONS["latent, expert norm"]))
        lowered = nnx.jit(nnx.grad(_loss, argnums=(0, 1))).lower(e, x, probe)
    assert lowered.as_text().count("ragged_all_to_all") == 4


def test_where_the_mesh_does_not_split_the_experts_a_method_sorts():
    e = layer(num_experts=6)  # expert=4 does not divide them
    x, _ = _inputs()
    with split("all_to_all", "fsdp=2, expert=4"):
        hlo = nnx.jit(lambda e, x: e(x, {})).lower(e, x).compile().as_text() or ""
    assert "all-to-all" not in hlo


def test_tokens_that_do_not_split_over_the_experts_axes_are_an_error():
    with pytest.raises(
        ValueError, match=r"2 rows do not split over \('data', 'tensor'"
    ):
        _outputs("all_to_all", "experts over tensor=4", "gated", rows=2)


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
