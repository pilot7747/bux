"""MoE load balancing: the balance loss, the router bias, and the losses layers report."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux.architecture import Model
from bux.architecture.moe import MoE, balance_loss
from bux.architecture.registry import LOSSES, Loss
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


def moe_layer(**settings) -> MoE:
    expert = cfg.MLPConfig(type="gated_mlp", intermediate_size=8, activation="swish")
    c = cfg.MoEConfig(num_experts=4, top_k=2, expert=expert, **settings)
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
    taps: dict[str, list] = {LOSSES: []}
    e(x, taps)
    [loss] = taps[LOSSES]
    dense, _ = e._route(x)
    picks = (dense > 0).astype(jnp.float32)
    scores = jax.nn.sigmoid(e.router(x))
    expected = balance_loss(scores, picks, 2, sequence=True)
    assert loss.name == "balance_loss" and loss.weight == 0.5
    assert jnp.isclose(loss.value, expected)
    # The weight defaults to 1e-2, and to 0 with the bias, so nothing is reported.
    assert moe_layer().aux_loss_weight == 1e-2
    free = moe_layer(load_balancing="aux_free")
    assert free.aux_loss_weight == 0
    taps = {LOSSES: []}
    free(x, taps)
    assert taps[LOSSES] == []


def test_training_moves_the_router_bias_toward_the_experts_picked_less():
    e = moe_layer(router="sigmoid", load_balancing="aux_free", bias_update_rate=0.01)
    assert e.router_bias is not None
    x = jax.random.normal(jax.random.key(1), (4, 8, D))
    dense, _ = e._route(x)
    load = (dense > 0).sum((0, 1))
    e(x, {})  # not a training forward
    assert (e.router_bias[...] == 0).all()
    e(x, {LOSSES: []})
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
    trainer.update(batch)
    first = bias(model)
    assert jnp.array_equal(first, bias(eager)) and (first != 0).any()
    trainer.update(batch)  # from where the first step left it
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
    total, metrics = obj(logits, batch, reported)
    assert set(metrics) == {"loss", "balance_loss"}
    assert jnp.isclose(metrics["balance_loss"], a + b)
    assert jnp.isclose(total, metrics["loss"] + 0.1 * a + 0.3 * b)
    with pytest.raises(ValueError, match="a layer reports 'loss'"):
        obj(logits, batch, [Loss("loss", 1.0, a)])

    # A train step logs the term; eval is the cross-entropy alone.
    trainer = Trainer(model, optax.adamw(1e-3), obj)
    assert "balance_loss" in trainer.update(batch)
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
    assert set(trainer.update(batch)) == {"loss", "grad_norm"}
