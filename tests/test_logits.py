"""The heads' losses by each method of `methods.loss` (bux.architecture.logits), the
logits whole and by chunks of positions, against the reference cross-entropy: values
and gradients, on one device and split over the 8 CPU devices the tests have, how
often each multiplies by the vocabulary, and the memory chunks leave."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.extend.core import ClosedJaxpr, Jaxpr
from jaxtyping import Array, Float, Int

from bux import parallel
from bux.architecture import Model, logits, observe
from bux.lang import compile, compile_runs, ir
from bux.train import config as tcfg
from bux.train.data import Batch
from bux.train.objective import (
    Heads,
    Term,
    cross_entropy,
    objective,
    prediction_offsets,
    target_mask,
    z_loss,
)

PRELUDE = """
d_model = 32
vocab_size = 256
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=64, activation=swish)
"""
MTP = """
mtp = $h! nextn [norm ffn] norm @h out
model = embed [norm attn] [norm ffn] norm @h out mtp^2
"""
TRAIN = (
    "data = synthetic\noptimizer = adamw(lr=1e-3)\nseq_len = 16\nbatch = 8\nsteps = 1\n"
)
HEADS = {
    "untied": "head",
    "bias_softcap": "head(bias=true, softcap=5)",
    "tied": "head(tied=true)",
    "tied_bias_softcap": "head(tied=true, bias=true, softcap=5)",
}
LOSS = tcfg.LossConfig(z=1e-2, mtp=0.5)
CHUNK = 4 * 256 * 8  # bytes of float32 logits: 8 positions of 256 a chunk
MESHES = ["mesh(tensor=2, fsdp=4)", "mesh(data=2, context=2, tensor=2)"]
LOSSES = ["logits", "chunks"]
ONE = "mesh(data=1)"
Loss = Callable[[Model, Batch], Float[Array, ""]]


@pytest.fixture
def chunked(monkeypatch):
    """Chunks of 8 positions, where these models' logits would make one."""
    monkeypatch.setattr(logits, "CHUNK_BYTES", CHUNK)


def _arch(head: str) -> ir.Architecture:
    return compile(PRELUDE + MTP + f"out = {head}\n")


def _batch(rows: int = 8, t: int = 16) -> Batch:
    """Rows of two documents each, and a loss mask that leaves out some targets."""
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 256, (rows, t)).astype(np.int32)
    first = t * 5 // 8
    positions = np.concatenate([np.arange(first), np.arange(t - first)])
    positions = np.broadcast_to(positions, (rows, t)).astype(np.int32)
    return Batch(tokens, positions, rng.random((rows, t)) > 0.2)


def _inputs(batch: Batch) -> tuple[Int[Array, "b t"], Int[Array, "b t"]]:
    return jnp.asarray(batch.input_ids), jnp.asarray(batch.position_ids)


def _reference(arch: ir.Architecture) -> Loss:
    """The default objective's loss, from the heads' whole logits."""
    k = prediction_offsets(arch)

    def loss(model: Model, batch: Batch) -> Float[Array, ""]:
        out = model.outputs(*_inputs(batch))
        z = z_loss(out[0], target_mask(batch, k[0]))
        loss = cross_entropy(out[0], batch, k[0]) + LOSS.z * z
        mtp = [cross_entropy(o, batch, offset) for o, offset in zip(out[1:], k[1:])]
        return loss + LOSS.mtp * jnp.mean(jnp.stack(mtp)) if mtp else loss

    return loss


def _fused(arch: ir.Architecture) -> Loss:
    """The default objective's loss, a chunk at a time."""
    obj = objective(LOSS, arch)
    return lambda model, batch: obj(model.heads(*_inputs(batch)), batch)[0].loss


def _value_and_grads(loss: Loss, model: Model, batch: Batch):
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def of(params: nnx.State) -> Float[Array, ""]:
        return loss(nnx.merge(graphdef, params, rest), batch)

    return jax.jit(jax.value_and_grad(of))(params)


def _assert_close(a, b) -> None:
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-4, atol=1e-6)


def _use(mesh: str, loss: str):
    """`parallel.use`, on `mesh`, the loss by the method `loss`."""
    settings = f"mesh = {mesh}\nmethods = methods(loss={loss})\n"
    [run] = compile_runs(PRELUDE + MTP + "out = head\n" + TRAIN + settings)
    t = run.train
    return parallel.use(t.mesh, t.layout, t.methods)


# --- Against the reference --------------------------------------------------------------


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("head", HEADS)
def test_the_loss_and_its_gradients_are_the_reference_cross_entropy(
    chunked, head, loss
):
    arch, batch = _arch(HEADS[head]), _batch()
    assert logits.Chunks.of(8, 16, 256).count == 16
    model = Model(arch, rngs=nnx.Rngs(0))
    with _use(ONE, loss):
        fused = _value_and_grads(_fused(arch), model, batch)
    _assert_close(fused, _value_and_grads(_reference(arch), model, batch))


@pytest.mark.parametrize("loss", LOSSES)
def test_the_loss_and_its_gradient_hold_where_float32_rounds_the_targets_p_to_1(loss):
    # Barron et al.'s example: logits (9, -9, -9) and target 0, a margin of 18, where
    # float32's 1 + 2e^-18 is 1. log Z - 9 is then 0, and so is p - 1.
    e = np.exp(-18.0)
    x = jnp.array([[[9.0, -9.0, -9.0]]])
    head = logits.HeadOutput(x, logits.Projection(jnp.eye(3), None, None))
    targets, weights = jnp.zeros((1, 1), jnp.int32), jnp.ones((1, 1))

    def ce(x: Float[Array, "1 1 3"]) -> Float[Array, ""]:
        return head._replace(hidden=x).sums(targets, weights, False).ce

    with _use(ONE, loss):
        value, grad = jax.jit(jax.value_and_grad(ce))(x)
    np.testing.assert_allclose(value, 2 * e, rtol=1e-6)
    np.testing.assert_allclose(grad[0, 0], [-2 * e, e, e], rtol=1e-6)


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("head", ["bias_softcap", "tied_bias_softcap"])
def test_the_loss_split_over_devices_is_the_one_device_loss(
    chunked, involuntary, head, mesh, loss
):
    arch, batch = _arch(HEADS[head]), _batch()
    one = _value_and_grads(_fused(arch), Model(arch, rngs=nnx.Rngs(0)), batch)
    involuntary()
    with _use(mesh, loss):
        model = parallel.build(lambda: Model(arch, rngs=nnx.Rngs(0)))
        split = _value_and_grads(_fused(arch), model, parallel.shard_batch(batch))
    _assert_close(split, one)
    assert involuntary() == 0


def test_a_term_that_reads_the_logits_makes_them_once():
    arch, batch = _arch(HEADS["untied"]), _batch()
    model = Model(arch, rngs=nnx.Rngs(0))
    heads = Heads(model.heads(*_inputs(batch)), batch)
    assert heads[0] is heads[0] and len(heads[1:]) == 2
    assert jnp.array_equal(heads[2], model.outputs(*_inputs(batch))[2])
    obj = objective(LOSS, arch)
    obj.terms["logit_l2"] = Term(1e-4, lambda logits, _: jnp.mean(logits[0] ** 2))
    (_, metrics), _ = obj(model.heads(*_inputs(batch)), batch)
    assert jnp.allclose(metrics["logit_l2"], jnp.mean(heads[0] ** 2))


def _vocab_products(jaxpr: Jaxpr) -> int:
    """How many products of matrices in `jaxpr` take or make a vocabulary's 256."""
    n = 0
    for eqn in jaxpr.eqns:
        shapes = [getattr(v.aval, "shape", ()) for v in (*eqn.invars, *eqn.outvars)]
        n += eqn.primitive.name == "dot_general" and any(256 in s for s in shapes)
        for param in eqn.params.values():
            for sub in param if isinstance(param, tuple | list) else (param,):
                if isinstance(sub, ClosedJaxpr):
                    sub = sub.jaxpr
                if isinstance(sub, Jaxpr):
                    n += _vocab_products(sub)
    return n


@pytest.mark.parametrize("loss", LOSSES)
def test_a_head_multiplies_by_the_vocabulary_three_times_by_either_method(
    chunked, loss
):
    # Its logits, and their gradients by W and by x: by chunks, all in the forward, in
    # a scan whose body the jaxpr has once.
    arch, batch = _arch(HEADS["untied"]), _batch()
    graphdef, params, rest = nnx.split(Model(arch, rngs=nnx.Rngs(0)), nnx.Param, ...)
    obj = objective(tcfg.LossConfig(mtp=0.5), arch)

    def of(params: nnx.State) -> Float[Array, ""]:
        model = nnx.merge(graphdef, params, rest)
        return obj(model.heads(*_inputs(batch)), batch)[0].loss

    with _use(ONE, loss):
        jaxpr = jax.make_jaxpr(jax.grad(of))(params)
    assert _vocab_products(jaxpr.jaxpr) == 3 * 3  # 3 heads


@pytest.mark.parametrize("loss", LOSSES)
def test_a_step_that_observes_has_each_head_s_z_loss_with_no_more_products(
    chunked, loss
):
    # Its z-loss's gradient of the head, by chunks a second one, only where the loss
    # weighs it: observed, it is a sum of the log Z the cross-entropy takes anyway.
    arch, batch = _arch(HEADS["untied"]), _batch()
    graphdef, params, rest = nnx.split(Model(arch, rngs=nnx.Rngs(0)), nnx.Param, ...)
    obj = objective(tcfg.LossConfig(mtp=0.5), arch)
    weighed = objective(tcfg.LossConfig(mtp=0.5, z=1.0), arch)

    def of(params: nnx.State) -> tuple[Float[Array, ""], dict]:
        model = nnx.merge(graphdef, params, rest)
        per_token, _ = obj(model.heads(*_inputs(batch)), batch)
        return per_token.loss, per_token.metrics

    with _use(ONE, loss), observe.use():
        jaxpr = jax.make_jaxpr(jax.grad(of, has_aux=True))(params)
        _, metrics = of(params)
        model = nnx.merge(graphdef, params, rest)
        z = weighed(model.heads(*_inputs(batch)), batch)[0].metrics["z_loss"]
    assert _vocab_products(jaxpr.jaxpr) == 3 * 3  # 3 heads, as without observing
    assert float(metrics["z_loss/t+1"]) == pytest.approx(float(z), rel=1e-6)


def test_chunks_give_the_loss_without_its_gradient_as_logits_do(chunked):
    arch, batch = _arch(HEADS["tied_bias_softcap"]), _batch()
    model = Model(arch, rngs=nnx.Rngs(0))

    def loss(method: str) -> Float[Array, ""]:
        with _use(ONE, method):
            return jax.jit(_fused(arch))(model, batch)

    np.testing.assert_allclose(loss("chunks"), loss("logits"), rtol=1e-5)


# --- Memory -----------------------------------------------------------------------------


def test_chunks_leave_a_fraction_of_the_logits_memory(monkeypatch):
    """At 8 rows of 128 positions and a vocabulary of 8192, the logits take 32 MiB."""
    monkeypatch.setattr(logits, "CHUNK_BYTES", 2**21)  # 16 chunks
    source = PRELUDE.replace("256", "8192")
    arch = compile(source + "model = embed [norm attn] [norm ffn] norm head\n")
    model = Model(arch, rngs=nnx.Rngs(0))
    batch = _batch(8, 128)

    def temp(method: str) -> int:
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)

        def of(params: nnx.State) -> Float[Array, ""]:
            return _fused(arch)(nnx.merge(graphdef, params, rest), batch)

        with _use(ONE, method):
            compiled = jax.jit(jax.grad(of)).lower(params).compile()
        memory = compiled.memory_analysis()
        assert memory is not None
        return memory.temp_size_in_bytes

    assert temp("chunks") < 0.25 * temp("logits")
