"""MoE: the methods that run the experts, and load balancing: the balance loss, the
router bias, and the losses layers report."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux import parallel
from bux.architecture import Model, Taps
from bux.architecture.moe import MoE, balance_loss, experts, routed, run_sorted
from bux.architecture.registry import Loss
from bux.lang import compile
from bux.train import config as tcfg
from bux.train.data import plain
from bux.train.hooks import Checkpoint
from bux.train.objective import cross_entropy, objective
from bux.train.trainer import Trainer

D = 16
ARCH = """
d_model = 16
vocab_size = 32
norm = rmsnorm
attn = gqa(num_heads=2, num_kv_heads=2)
ffn = gated_mlp(intermediate_size=32, activation=swish)
"""


GATED = cfg.MLPConfig(type="gated_mlp", intermediate_size=8, activation="swish")
RMS = cfg.NormalizationConfig(type="rmsnorm")


def moe_layer(**settings) -> MoE:
    given = {"num_experts": 4, "top_k": 2, "expert": GATED} | settings
    c = cfg.MoEConfig.model_validate(given)
    return MoE(c, D, nnx.Rngs(0))


def first_moe(model: Model) -> MoE:
    return next(m for _, m in nnx.iter_modules(model) if isinstance(m, MoE))


def bias(model: Model) -> Float[Array, " e"]:
    b = first_moe(model).router_bias
    assert b is not None
    return b[...]


def tokens(b: int = 2, t: int = 8) -> Int[np.ndarray, "b t"]:
    return np.asarray(jax.random.randint(jax.random.key(0), (b, t), 0, 32))


def test_the_balance_loss_is_its_closed_form_and_one_when_balanced():
    # Two sequences of two tokens, the top 2 of 4 experts: the first sequence picks
    # experts 0 and 1, scoring them 3 and the others 1; the second, 2 and 3.
    first = jnp.array([3.0, 3, 1, 1])
    scores = jnp.stack([jnp.stack([first] * 2), jnp.stack([first[::-1]] * 2)])
    picks = (scores > 2).astype(jnp.float32)
    # Over the batch, each expert has a quarter of the picks and a mean score of 1/4.
    assert jnp.isclose(balance_loss(scores, picks, 2, sequence=False), 1)
    # In a sequence, f = [1/2, 1/2, 0, 0] and P = [3/8, 3/8, 1/8, 1/8].
    assert jnp.isclose(balance_loss(scores, picks, 2, sequence=True), 4 * 3 / 8)
    # One sequence: f = [1/2, 1/4, 1/4, 0] and P = [1/2, 1/4, 1/8, 1/8].
    scores = jnp.array([[[4.0, 2, 1, 1], [2.0, 1, 0.5, 0.5]]])
    picks = jnp.array([[[1.0, 1, 0, 0], [1.0, 0, 1, 0]]])
    expected = 4 * (1 / 2 * 1 / 2 + 1 / 4 * 1 / 4 + 1 / 4 * 1 / 8)
    for sequence in (False, True):
        assert jnp.isclose(balance_loss(scores, picks, 2, sequence), expected)


def test_an_moe_reports_its_balance_loss_in_training_only():
    e = moe_layer(router="sigmoid", aux_loss_coef=0.5, aux_loss_scope="sequence")
    x = jax.random.normal(jax.random.key(1), (2, 5, D))
    taps = Taps(training=True)
    e(x, taps)
    [loss] = taps.losses
    idx, _, _ = e._route(x)
    picks = jax.nn.one_hot(idx, 4).sum(-2)
    scores = jax.nn.sigmoid(e.router(x))
    expected = balance_loss(scores, picks, 2, sequence=True)
    assert loss.name == "balance_loss" and loss.weight == 0.5
    assert jnp.isclose(loss.value, expected)
    # The weight defaults to 1e-2, and to 0 with the bias, so nothing is reported.
    assert moe_layer().config.aux_loss_weight == 1e-2
    free = moe_layer(load_balancing="aux_free")
    assert free.config.aux_loss_weight == 0
    taps = Taps(training=True)
    free(x, taps)
    assert taps.losses == []


def test_training_moves_the_router_bias_toward_the_experts_picked_less():
    e = moe_layer(router="sigmoid", load_balancing="aux_free", bias_update_rate=0.01)
    assert e.router_bias is not None
    x = jax.random.normal(jax.random.key(1), (4, 8, D))
    idx, _, _ = e._route(x)
    load = jax.nn.one_hot(idx, 4).sum((0, 1, 2))
    e(x, Taps())  # not a training forward
    assert (e.router_bias[...] == 0).all()
    e(x, Taps(training=True))
    assert jnp.array_equal(e.router_bias[...], 0.01 * jnp.sign(load.mean() - load))
    assert (e.router_bias[...] != 0).any()


def test_a_train_step_moves_the_bias_eval_does_not_and_checkpoints_keep_it(tmp_path):
    arch = compile(
        ARCH
        + "experts = moe(num_experts=4, top_k=2, router=sigmoid,"
        + " load_balancing=aux_free, expert=ffn)\n"
        + "model = embed [norm attn] [norm experts] norm head\n"
    )
    model = Model(arch, rngs=nnx.Rngs(0))
    trainer = Trainer(model, optax.adamw(1e-3), objective(tcfg.LossConfig(), arch))
    batch = plain(tokens())
    eager = nnx.clone(model)
    eager.training_outputs(jnp.asarray(batch.input_ids))
    trainer.train_on_batch(batch)
    first = bias(model)
    assert jnp.array_equal(first, bias(eager)) and (first != 0).any()
    trainer.train_on_batch(batch)  # from where the first step left it
    assert jnp.allclose(jnp.abs(bias(model) - first).max(), 1e-3)
    moved = bias(model)
    trainer.evaluate([batch])
    assert jnp.array_equal(bias(model), moved)

    save = Checkpoint(tmp_path, every=1, keep=1)
    save(trainer, {})
    save.close()
    fresh = Model(arch, rngs=nnx.Rngs(1))
    again = Trainer(fresh, optax.adamw(1e-3), objective(tcfg.LossConfig(), arch))
    load = Checkpoint(tmp_path, every=1, keep=1)
    assert load.restore(again)
    load.close()
    assert jnp.array_equal(bias(fresh), moved)


def test_reported_losses_are_terms_of_the_training_loss_weighted_by_each_layer():
    arch = compile(
        ARCH
        + "experts = moe(num_experts=4, top_k=2, aux_loss_coef=0.1, expert=ffn)\n"
        + "other = experts(aux_loss_coef=0.3, aux_loss_scope=sequence)\n"
        + "model = embed [norm experts] [norm other] norm head\n"
    )
    model = Model(arch, rngs=nnx.Rngs(0))
    batch = plain(tokens())
    logits, reported = model.training_outputs(jnp.asarray(batch.input_ids))
    assert [(r.name, r.weight) for r in reported] == [
        ("balance_loss", 0.1),
        ("balance_loss", 0.3),
    ]
    a, b = (r.value for r in reported)
    obj = objective(tcfg.LossConfig(), arch)
    per_token, per_row = obj(logits, batch, reported)
    total, metrics = per_token.loss + per_row.loss, per_token.metrics | per_row.metrics
    assert set(metrics) == {"loss", "balance_loss"}
    assert jnp.isclose(metrics["balance_loss"], a + b)
    assert jnp.isclose(total, metrics["loss"] + 0.1 * a + 0.3 * b)
    with pytest.raises(ValueError, match="a layer reports 'loss'"):
        obj(logits, batch, [Loss("loss", 1.0, a)])

    # A train step logs the term; eval is the cross-entropy alone.
    trainer = Trainer(model, optax.adamw(1e-3), obj)
    assert "balance_loss" in trainer.train_on_batch(batch)
    ce = cross_entropy(model.outputs(jnp.asarray(batch.input_ids))[0], batch)
    assert np.isclose(trainer.evaluate([batch]), float(ce), rtol=1e-5)


def test_a_model_without_reporting_layers_trains_as_before():
    arch = compile(ARCH + "model = embed [norm attn] [norm ffn] norm head\n")
    model = Model(arch, rngs=nnx.Rngs(0))
    batch = plain(tokens())
    logits, reported = model.training_outputs(jnp.asarray(batch.input_ids))
    assert reported == []
    assert jnp.array_equal(logits[0], model.outputs(jnp.asarray(batch.input_ids))[0])
    trainer = Trainer(model, optax.adamw(1e-3), objective(tcfg.LossConfig(), arch))
    assert set(trainer.train_on_batch(batch)) == {"loss", "grad_norm"}


# --- The methods that run the experts --------------------------------------------------

# Every MoE option the example architectures use.
OPTIONS = {
    "gated": {},
    "mlp": {"expert": GATED.model_copy(update={"type": "mlp", "activation": "relu2"})},
    "latent": {"latent_size": 8, "latent_input_norm": RMS, "latent_output_norm": RMS},
    "expert norm": {
        "expert_norm": RMS.model_copy(
            update={"zero_centered": True, "scale_input": True}
        )
    },
    "limit": {"expert": GATED.model_copy(update={"limit": 0.3})},
    "softcap": {
        "expert": GATED.model_copy(update={"activation": "situ", "up_softcap": 0.3})
    },
    "shared": {"num_shared_experts": 2, "shared_expert_gate": True},
    "sinks": {"num_shared_experts": 2, "shared_expert_sink": True, "router": "sigmoid"},
    "groups": {"num_experts": 8, "top_k": 3, "num_groups": 4, "top_k_groups": 2},
    "scaled": {
        "router": "sqrtsoftplus",
        "router_logit_scale": 2.0,
        "routed_scaling_factor": 2.5,
        "normalize_top_k": False,
    },
}


def run(method: str, e: MoE, x: Float[Array, "b t d"]):
    """The output, and the gradients of the parameters and of `x`, by `method`."""
    probe = jax.random.normal(jax.random.key(2), x.shape)
    with parallel.use(tcfg.MeshConfig(data=1), methods=tcfg.MethodsConfig(moe=method)):
        y = e(x, Taps())
        grads = nnx.grad(lambda e, x: (e(x, Taps()) * probe).sum(), argnums=(0, 1))(
            e, x
        )
    return y, grads


@pytest.mark.parametrize("option", OPTIONS)
def test_sorted_computes_what_dense_does_with_the_same_gradients(option):
    e = moe_layer(**OPTIONS[option])
    x = jax.random.normal(jax.random.key(1), (2, 5, D))
    (dense, dense_grads), (fast, fast_grads) = run("dense", e, x), run("sorted", e, x)
    assert jnp.allclose(fast, dense, atol=1e-5)
    close = jax.tree.map(
        lambda a, b: jnp.allclose(a, b, atol=1e-5), fast_grads, dense_grads
    )
    assert jax.tree.all(close)


def test_sorted_drops_no_token_when_every_token_picks_the_same_experts():
    e = moe_layer()
    e.router.kernel[...] = jnp.zeros_like(e.router.kernel[...])  # every score ties
    x = jax.random.normal(jax.random.key(1), (2, 5, D))
    idx, _, _ = e._route(x)
    assert (idx == jnp.array([0, 1])).all()
    assert jnp.allclose(run("sorted", e, x)[0], run("dense", e, x)[0], atol=1e-5)


def test_experts_run_each_row_through_its_own_and_rows_past_them_are_zero():
    keys = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(keys[0], (9, D))
    up, gate = (jax.random.normal(k, (3, D, 8)) for k in keys[1:3])
    down = jax.random.normal(keys[3], (3, 8, D))
    out = experts(x, jnp.array([3, 0, 4]), up, gate, down, GATED)
    for row, expert in enumerate([0, 0, 0, 2, 2, 2, 2]):
        h = jax.nn.swish(x[row] @ gate[expert]) * (x[row] @ up[expert])
        assert jnp.allclose(out[row], h @ down[expert], rtol=1e-5, atol=1e-4)
    assert (out[7:] == 0).all()


def test_some_of_the_experts_add_their_own_picks_and_nothing_for_the_others():
    """What a device that holds some of the experts computes: the picks numbered from
    its first expert, the others' outside its range."""
    e = moe_layer(expert_norm=cfg.NormalizationConfig(type="layernorm", bias=True))
    assert e.expert_norm is not None and e.expert_norm.bias is not None
    e.expert_norm.bias[...] = jnp.ones(D)  # so that a row of zeros comes out not 0
    x = jax.random.normal(jax.random.key(1), (2, 5, D))
    z, (idx, weights, _) = e._latent(x), e._route(x)
    up, gate, down = e.kernels()
    assert gate is not None

    def some(first: int, n: int) -> Float[Array, "b t d"]:
        mine = slice(first, first + n)
        kernels = (up[mine], gate[mine], down[mine])
        return routed(z, idx - first, weights, *kernels, e.config.expert, e.expert_norm)

    assert jnp.allclose(some(0, 1) + some(1, 3), some(0, 4), atol=1e-5)
    assert jnp.allclose(some(0, 4), run("dense", e, x)[0], atol=1e-5)


def test_sorted_is_the_default_method():
    assert tcfg.MethodsConfig().moe == "sorted"
    assert parallel.method("moe") is run_sorted


# --- Split over devices ----------------------------------------------------------------

SPLIT = """
d_model = 64
vocab_size = 128
norm = rmsnorm
rescale = rmsnorm(eps=1e-8, zero_centered=true, scale_input=true)
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
experts = moe(num_experts=8, top_k=2, num_shared_experts=1, shared_expert_gate=true,
              router=sigmoid, load_balancing=aux_free, latent_size=32,
              latent_input_norm=norm, expert_norm=rescale,
              expert=gated_mlp(intermediate_size=64, activation=swish))
model = embed [norm attn] [norm experts] norm head
"""


@pytest.mark.parametrize("mesh", ["mesh(fsdp=8)", "mesh(fsdp=2, expert=2, tensor=2)"])
@pytest.mark.parametrize("method", ["dense", "sorted"])
def test_an_moe_trains_the_same_split_over_devices_and_rematerializes_nothing(
    losses, involuntary, method, mesh
):
    methods = f"methods(moe={method})"
    one = losses(SPLIT, mesh="mesh(data=1)", methods=methods)
    involuntary()
    split = losses(SPLIT, mesh=mesh, methods=methods)
    assert involuntary() == 0
    assert np.allclose(split, one, atol=1e-5)


def test_an_moe_names_its_experts_and_its_projections():
    expert = GATED.model_copy(update={"intermediate_size": 64})
    c = cfg.MoEConfig(num_experts=8, top_k=2, latent_size=32, expert=expert)
    with parallel.use(tcfg.MeshConfig(fsdp=2, expert=2, tensor=2)):
        state = nnx.state(parallel.build(lambda: MoE(c, 64, nnx.Rngs(0))))
        found = parallel.shardings(state)
    # The experts over `expert` and their hidden units over `tensor`; FSDP splits the
    # largest dimension left whole.
    assert found["up"].spec == found["gate"].spec == P("expert", "fsdp", "tensor")
    assert found["down"].spec == P("expert", "tensor", "fsdp")
    assert found["router"]["kernel"].spec == P("fsdp", None)
    assert found["to_latent"]["kernel"].spec == P("fsdp", None)
