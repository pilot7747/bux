"""MoE: the methods that run the experts, and load balancing: the balance loss, the
router bias, and the losses layers report."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from conftest import random_biases
from flax import nnx
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int
from optax.contrib import MuonDimensionNumbers

from bux import config as cfg
from bux import parallel
from bux.architecture import Model, Taps
from bux.architecture.experts.moe import (
    Kernels,
    MoE,
    balance_loss,
    budget_loss,
    budget_move,
    experts,
    random_logits,
    routed,
    run_sorted,
    smebu_move,
    zero_experts,
)
from bux.architecture.layers.mlp import activate
from bux.architecture.layers.norm import Norm
from bux.architecture.registry import Loss
from bux.architecture.repeats.remat import rematted
from bux.lang import compile, compile_runs
from bux.train import config as tcfg
from bux.train.data import plain
from bux.train.hooks import Checkpoint
from bux.train.objective import cross_entropy, objective
from bux.train.optim import muon_matrices, newton_schulz, orthogonalized
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
    return random_biases(MoE(c, D, nnx.Rngs(0)))


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


def test_a_centered_aux_free_bias_keeps_a_mean_of_zero():
    e = moe_layer(router="sigmoid", load_balancing="aux_free", center_bias=True)
    assert e.router_bias is not None
    x = jax.random.normal(jax.random.key(1), (4, 8, D))
    idx, _, _ = e._route(x)
    load = jax.nn.one_hot(idx, 4).sum((0, 1, 2))
    e(x, Taps(training=True))
    step = 1e-3 * jnp.sign(load.mean() - load)
    assert jnp.allclose(e.router_bias[...], step - step.mean())
    assert jnp.isclose(e.router_bias[...].mean(), 0, atol=1e-9)


def test_smebu_moves_the_bias_by_its_centered_soft_clamped_momentum():
    """Trinity's eqs. 23-28, a training step at a time."""
    e = moe_layer(
        router="sigmoid",
        load_balancing="smebu",
        bias_update_rate=0.01,
        bias_softclamp=3,
        bias_momentum=0.6,
    )
    assert e.config.aux_loss_weight == 0
    assert e.router_bias is not None and e.router_momentum is not None
    x = jax.random.normal(jax.random.key(1), (4, 8, D))
    b = m = jnp.zeros(4)
    for _ in range(3):
        idx, _, _ = e._route(x)  # by the bias so far
        n = jax.nn.one_hot(idx, 4).sum((0, 1, 2))
        move = 0.01 * jnp.tanh(3 * (n.mean() - n) / n.mean())
        m = 0.6 * m + 0.4 * (move - move.mean())
        b = b + m
        e(x, Taps(training=True))
        assert jnp.allclose(e.router_momentum[...], m, atol=1e-8)
        assert jnp.allclose(e.router_bias[...], b, atol=1e-8)
    assert (b != 0).any() and jnp.isclose(b.mean(), 0, atol=1e-8)


def test_smebu_leaves_an_expert_at_the_mean_load_and_each_layer_on_its_own():
    c = cfg.MoEConfig(num_experts=4, top_k=2, expert=GATED, load_balancing="smebu")
    load = jnp.array([4.0, 4.0, 2.0, 6.0])  # the mean is 4
    expected = 1e-3 * jnp.tanh(2 * jnp.array([0.0, 0.0, 0.5, -0.5]))
    assert jnp.allclose(smebu_move(load, c), expected)
    # A scanned repeat's layers, stacked: each centered on its own.
    stacked = smebu_move(jnp.stack([load, jnp.array([8.0, 0.0, 0.0, 0.0])]), c)
    assert jnp.allclose(stacked[0], expected)
    assert jnp.allclose(stacked[1].mean(), 0, atol=1e-9)
    # No picks moves nothing, rather than by 0 / 0.
    assert (smebu_move(jnp.zeros(4), c) == 0).all()


def test_random_logits_are_drawn_and_their_gradient_is_the_router_s():
    logits = 100 * jax.random.normal(jax.random.key(0), (2, 3, 8))
    key, probe = jax.random.key(1), jax.random.normal(jax.random.key(2), logits.shape)
    drawn = jax.random.normal(key, logits.shape)
    assert jnp.array_equal(random_logits(logits, key), drawn)
    grad = jax.grad(lambda x: (random_logits(x, key) * probe).sum())(logits)
    assert jnp.array_equal(grad, probe)


def test_random_logits_route_a_training_forward_evenly_drawn_anew_each_time():
    """A router that picks experts 0 and 1 for every token: a training forward picks
    each of the 8 as often as any, and learns; any other forward routes by it."""
    e = moe_layer(num_experts=8, random_logits=True)
    e.router.kernel[...] = jnp.tile(jnp.where(jnp.arange(8) < 2, 1.0, -1.0), (D, 1))
    x = jnp.abs(jax.random.normal(jax.random.key(1), (32, 128, D)))
    assert set(np.unique(e._route(x)[0]).tolist()) == {0, 1}
    first, second = (e._route(x, Taps(training=True))[0] for _ in range(2))
    picks = np.bincount(np.asarray(first).ravel(), minlength=8)
    assert picks.min() > 0.9 * picks.mean()  # 1024 a expert, give or take 3%
    assert (first != second).any()
    taps = Taps(training=True)
    e(x, taps)
    [balance] = taps.losses
    assert balance.value == pytest.approx(1, abs=0.02)  # its value when even
    grads = nnx.grad(lambda e: e(x, Taps(training=True)).sum())(e)
    assert (grads.router.kernel[...] != 0).all()


def test_random_logits_are_in_the_run_s_id_and_train_on_a_mesh(losses):
    source = (
        ARCH
        + "experts = moe(num_experts=4, top_k=2, expert=ffn)\n"
        + "model = embed ([norm attn] [norm experts])^2 norm head\n"
    )
    random = source + "experts.random_logits = true\n"
    train = "data = synthetic\noptimizer = adamw(lr=1e-3)\nseq_len = 16\nbatch = 8\n"
    [learned], [drawn] = (
        compile_runs(s + train + "steps = 1") for s in (source, random)
    )
    assert learned.id != drawn.id
    by_router = losses(source, mesh="mesh(fsdp=4)")
    by_random = losses(random, mesh="mesh(fsdp=4)")
    assert by_random[0] != by_router[0] and np.isfinite(by_random).all()


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
    eager.training_heads(jnp.asarray(batch.input_ids))
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
    heads, reported = model.training_heads(jnp.asarray(batch.input_ids))
    assert [(r.name, r.weight) for r in reported] == [
        ("balance_loss", 0.1),
        ("balance_loss", 0.3),
    ]
    a, b = (r.value for r in reported)
    obj = objective(tcfg.LossConfig(), arch)
    per_token, per_row = obj(heads, batch, reported)
    total, metrics = per_token.loss + per_row.loss, per_token.metrics | per_row.metrics
    assert set(metrics) == {"loss", "balance_loss"}
    assert jnp.isclose(metrics["balance_loss"], a + b)
    assert jnp.isclose(total, metrics["loss"] + 0.1 * a + 0.3 * b)
    with pytest.raises(ValueError, match="a layer reports 'loss'"):
        obj(heads, batch, [Loss("loss", 1.0, a)])

    # A train step logs the term; eval is the cross-entropy alone.
    trainer = Trainer(model, optax.adamw(1e-3), obj)
    assert "balance_loss" in trainer.train_on_batch(batch)
    ce = cross_entropy(model.outputs(jnp.asarray(batch.input_ids))[0], batch)
    assert np.isclose(trainer.evaluate([batch])["loss"], float(ce), rtol=1e-5)


def test_a_model_without_reporting_layers_trains_as_before():
    arch = compile(ARCH + "model = embed [norm attn] [norm ffn] norm head\n")
    model = Model(arch, rngs=nnx.Rngs(0))
    batch = plain(tokens())
    heads, reported = model.training_heads(jnp.asarray(batch.input_ids))
    assert reported == []
    logits = model.outputs(jnp.asarray(batch.input_ids))[0]
    assert jnp.array_equal(heads[0].logits(), logits)
    trainer = Trainer(model, optax.adamw(1e-3), objective(tcfg.LossConfig(), arch))
    assert set(trainer.train_on_batch(batch)) == {"loss", "grad_norm"}


# --- Zero experts (LongCat-Flash) -------------------------------------------------------


def zero_layer(**settings) -> MoE:
    """4 FFN experts and 2 zero experts, 3 picks a token, sigmoid and aux-free."""
    given = {
        "num_zero_experts": 2,
        "top_k": 3,
        "router": "sigmoid",
        "load_balancing": "aux_free",
    }
    return moe_layer(**given | settings)


def test_zero_experts_add_the_token_times_their_weights():
    """A router that scores the zero experts alone high picks them, and each adds
    the token times its weight: half of 2.5 each, normalized over two picks."""
    e = zero_layer(top_k=2, router_bias=True, routed_scaling_factor=2.5)
    assert e.router.kernel[...].shape == (D, 6) and e.router.bias is not None
    e.router.kernel[...] = jnp.zeros_like(e.router.kernel[...])
    e.router.bias[...] = jnp.array([-9.0, -9, -9, -9, 9, 9])
    x = jax.random.normal(jax.random.key(1), (2, 5, D))
    idx, weights, _ = e._route(x)
    assert (idx >= 4).all()
    for method in ("dense", "sorted"):
        assert jnp.allclose(run(method, e, x)[0], 2.5 * x, atol=1e-6)
    # Picks of routed experts add nothing of the token.
    assert (zero_experts(x, idx - 4, weights, 4) == 0).all()


def test_zero_experts_run_only_by_dense_and_sorted():
    e = zero_layer()
    x = jnp.ones((1, 4, D))
    for method in ("all_to_all", "ragged_tokens"):
        with (
            parallel.use(
                tcfg.MeshConfig(data=1), methods=tcfg.MethodsConfig(moe=method)
            ),
            pytest.raises(NotImplementedError, match="zero experts"),
        ):
            e(x, Taps())


def test_the_budget_moves_ffn_experts_by_longcat_s_rule_and_zero_experts_not():
    """LongCat-Flash's eq. 2, mu (K_e / (K N) - T_i / (K T)), with mu = rate K N /
    K_e: 2 tokens of 3 picks, K_e 2 of them FFN experts', 1 a target of each of 4."""
    c = cfg.MoEConfig(
        num_experts=4,
        top_k=3,
        num_zero_experts=2,
        ffn_experts_per_token=2,
        expert=GATED,
        load_balancing="aux_free",
        bias_update_rate=0.1,
    )
    load = jnp.array([2.0, 1, 0, 1, 1, 1])  # T = 2 tokens, K T = 6 picks
    mu, k, n, k_e, t = 0.1 * 3 * 4 / 2, 3, 4, 2, 2
    paper = mu * (k_e / (k * n) - load[:4] / (k * t))
    assert jnp.allclose(budget_move(load, c), jnp.concatenate([paper, jnp.zeros(2)]))
    assert jnp.allclose(budget_move(load, c)[:4], 0.1 * jnp.array([-1.0, 0, 1, 0]))
    # A scanned repeat's layers, each its own; no picks move nothing.
    stacked = budget_move(jnp.stack([load, jnp.zeros(6)]), c)
    assert jnp.allclose(stacked[0], budget_move(load, c)) and (stacked[1] == 0).all()


def test_training_moves_the_ffn_experts_bias_by_the_budget():
    e = zero_layer(bias_update_rate=0.01)
    assert e.router_bias is not None and e.router_bias[...].shape == (6,)
    x = jax.random.normal(jax.random.key(1), (4, 8, D))
    idx, _, _ = e._route(x)
    load = jax.nn.one_hot(idx, 6).sum((0, 1, 2))
    e(x, Taps(training=True))
    assert jnp.allclose(e.router_bias[...], budget_move(load, e.config))
    assert (e.router_bias[...][:4] != 0).any() and (e.router_bias[...][4:] == 0).all()


def test_the_budget_loss_is_longcat_s_and_one_at_the_budget():
    """Eqs. 3-5, a group an FFN expert: one sequence of two tokens of 3 picks over 2
    FFN experts and 2 zero experts, K_e = 2."""
    c = cfg.MoEConfig(
        num_experts=2,
        top_k=3,
        num_zero_experts=2,
        ffn_experts_per_token=2,
        expert=GATED,
        aux_loss_coef=1.0,
    )
    scores = jnp.array([[[2.0, 1, 1, 0], [1.0, 2, 0, 1]]])
    picks = jnp.array(
        [[[1.0, 1, 1, 0], [1.0, 1, 0, 1]]]
    )  # each FFN expert once a token
    p = (scores / 4).mean(1)[0]  # [3/8, 3/8, 1/8, 1/8]
    f_ffn = 2 / (2 * 2) * picks[0, :, :2].sum(0)  # N / (K_e T) times the count: 1
    f_zero = picks[0, :, 2:].sum() / ((3 - 2) * 2)  # 1
    expected = (f_ffn * p[:2]).sum() + f_zero * p[2:].sum()
    for sequence in (False, True):
        loss = budget_loss(scores, picks, c, sequence)
        assert jnp.isclose(loss, expected) and jnp.isclose(loss, 1)


def test_zero_experts_settings_are_checked():
    zero = {"num_experts": 4, "top_k": 2, "num_zero_experts": 2, "expert": GATED}
    assert cfg.MoEConfig.model_validate(zero).ffn_budget == 2 * 4 / 6
    assert cfg.MoEConfig.model_validate(zero | {"top_k": 6}).top_k == 6
    for wrong, message in [
        ({"ffn_experts_per_token": 2.0}, "between 0 and top_k"),
        ({"load_balancing": "smebu"}, "load_balancing=smebu"),
        ({"center_bias": True}, "center_bias"),
        ({"num_groups": 2}, "num_groups"),
        ({"num_zero_experts": 0, "ffn_experts_per_token": 1.0}, "needs num_zero"),
    ]:
        with pytest.raises(ValueError, match=message):
            cfg.MoEConfig.model_validate(zero | wrong)


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
    # gpt-oss's: biases on the router and the experts, whose up branch adds 1.
    "biased": {
        "router_bias": True,
        "expert": GATED.model_copy(
            update={
                "bias": True,
                "activation": "quick_gelu",
                "limit": 0.3,
                "up_offset": 1.0,
            }
        ),
    },
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
    "zero experts": {
        "num_zero_experts": 2,
        "top_k": 3,
        "num_shared_experts": 1,
        "router": "sigmoid",
        "routed_scaling_factor": 2.5,
        "load_balancing": "aux_free",
        "latent_size": 8,
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


@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("norm", [None, RMS])
def test_experts_run_each_row_through_its_own_times_its_weight(norm, biased):
    """Each row's output times its weight, after the norm, which a weight before it
    would not change, and after its expert's biases; and the rows past the experts'
    are 0, biases and all."""
    keys = jax.random.split(jax.random.key(0), 6)
    x = jax.random.normal(keys[0], (9, D))
    up = jax.random.normal(keys[1], (3, D, 8, 2))  # each hidden unit's up and gate
    down = jax.random.normal(keys[2], (3, 8, D))
    weights = jax.random.uniform(keys[3], (9,))
    up_bias = jax.random.normal(keys[4], (3, 8, 2)) if biased else jnp.zeros((3, 8, 2))
    down_bias = jax.random.normal(keys[5], (3, D)) if biased else jnp.zeros((3, D))
    rescale = None if norm is None else Norm(norm, D)
    kernels = Kernels(up, down, *((up_bias, down_bias) if biased else ()))
    out = experts(x, jnp.array([3, 0, 4]), weights, kernels, GATED, rescale)
    for row, expert in enumerate([0, 0, 0, 2, 2, 2, 2]):
        u, g = (x[row] @ up[expert, ..., i] + up_bias[expert, :, i] for i in (0, 1))
        y = jax.nn.swish(g) * u @ down[expert] + down_bias[expert]
        y = y if rescale is None else rescale(y)
        assert jnp.allclose(out[row], weights[row] * y, rtol=1e-5, atol=1e-4)
    assert (out[7:] == 0).all()


def test_gpt_oss_gated_units_clamp_the_gate_and_add_1_to_up():
    """gpt-oss's experts: (clip(up, -7, 7) + 1) * g * sigmoid(1.702 g), g = min(gate,
    7), as transformers' GptOssExperts computes them."""
    ffn = GATED.model_copy(
        update={"activation": "quick_gelu", "limit": 7.0, "up_offset": 1.0}
    )
    up, gate = 10 * jax.random.normal(jax.random.key(0), (2, 64))
    g = jnp.minimum(gate, 7.0)
    want = (jnp.clip(up, -7.0, 7.0) + 1) * g * jax.nn.sigmoid(1.702 * g)
    assert jnp.allclose(activate(ffn, up, gate), want, rtol=1e-6)


def test_some_of_the_experts_add_their_own_picks_and_nothing_for_the_others():
    """What a device that holds some of the experts computes: the picks numbered from
    its first expert, the others' outside its range."""
    e = moe_layer(expert_norm=cfg.NormalizationConfig(type="layernorm", bias=True))
    assert e.expert_norm is not None and e.expert_norm.bias is not None
    e.expert_norm.bias[...] = jnp.ones(D)  # so that a row of zeros comes out not 0
    x = jax.random.normal(jax.random.key(1), (2, 5, D))
    z, (idx, weights, _) = e._latent(x), e._route(x)
    up, down, _, _ = e.kernels()

    def some(first: int, n: int) -> Float[Array, "b t d"]:
        mine = slice(first, first + n)
        kernels = Kernels(up[mine], down[mine])
        return routed(z, idx - first, weights, kernels, e.config.expert, e.expert_norm)

    assert jnp.allclose(some(0, 1) + some(1, 3), some(0, 4), atol=1e-5)
    assert jnp.allclose(some(0, 4), run("dense", e, x)[0], atol=1e-5)


@pytest.mark.parametrize(
    ("remat", "products"),
    [("none", 6), ("full", 7), ("dots", 6), ("dots_no_batch", 6)],
)
def test_the_dots_policies_keep_the_experts_products(remat, products):
    """Two grouped products forward, up and gate as one, and two for each backward,
    for the rows and for the kernels: what the policy does not keep, the backward
    computes again. Nothing in the backward takes down's output, so `full` computes
    up and gate's alone again."""
    e, x = moe_layer(), jax.random.normal(jax.random.key(1), (2, 5, D))
    graphdef, state = nnx.split(e)

    def loss(state: nnx.State, x: Float[Array, "b t d"]) -> Float[Array, ""]:
        return rematted(nnx.merge(graphdef, state), x, Taps()).sum()

    how = tcfg.MethodsConfig(remat=remat)
    with parallel.use(tcfg.MeshConfig(data=1), methods=how):
        jaxpr = jax.make_jaxpr(jax.grad(loss, argnums=(0, 1)))(state, x)
    assert str(jaxpr).count("ragged_dot_general") == products


def test_sorted_is_the_default_method():
    assert tcfg.MethodsConfig().moe == "sorted"
    assert parallel.method("moe") is run_sorted


# --- Split over devices ----------------------------------------------------------------

# Weights large next to adamw's first steps at 1e-2, of about 0.01 each: from 0.02,
# two experts' scores come within rounding of each other by the third step, and a
# split breaks the tie the other way.
SPLIT = """
d_model = 64
vocab_size = 128
init = normal(std=0.1)
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
    # The experts over `expert` and their hidden units over `tensor`, up's and gate's
    # alike; FSDP splits the largest dimension left whole, over `expert` too where no
    # dimension takes it.
    assert found["up"].spec == P("expert", "fsdp", "tensor", None)
    assert found["down"].spec == P("expert", "tensor", "fsdp")
    assert found["router"]["kernel"].spec == P(("fsdp", "expert"), None)
    assert found["to_latent"]["kernel"].spec == P(("fsdp", "expert"), None)


@pytest.mark.parametrize("method", ["gspmd", "gather", "owner"])
def test_muon_orthogonalizes_up_and_gate_as_two_matrices(method):
    """As it did when they were two parameters, on one device, whichever method
    moves them on a mesh."""
    expert = GATED.model_copy(update={"intermediate_size": 64})
    c = cfg.MoEConfig(num_experts=8, top_k=2, expert=expert)
    how = tcfg.MethodsConfig(orthogonalize=method)
    with parallel.use(tcfg.MeshConfig(fsdp=2, expert=2, tensor=2), methods=how):
        e = parallel.build(lambda: MoE(c, 32, nnx.Rngs(0)))
        grads = {"up": jnp.sin(e.up[...])}
        tx = orthogonalized(newton_schulz(5), muon_matrices(e), e)
        together = jax.jit(tx.update)(grads, tx.init(grads))[0]["up"]
    ns, matrices = newton_schulz(5), MuonDimensionNumbers(1, 2)
    apart = [ns(grads["up"][..., i], matrices) for i in (0, 1)]
    assert np.allclose(together, jnp.stack(apart, -1), atol=1e-6)
