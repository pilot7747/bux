"""Training options beside the update rule: cautious weight decay, NorMuon's
normalization by neuron, and the tail EMA a run ends on."""

import shutil
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jaxtyping import Float
from optax.contrib import MuonDimensionNumbers

from bux.architecture import Model
from bux.export.runs import restore
from bux.lang import compile_runs
from bux.train import config as tcfg
from bux.train import ema
from bux.train.optim import (
    decayed_weights,
    lr_multiplier,
    muon_matrices,
    neuron_normalized,
    optimizer,
)
from bux.train.trainer import directory, train

SMOKE = Path(__file__).parent.parent / "examples" / "experiments" / "smoke.bux"


def smoke(*overrides: str):
    """The smoke run, shortened: its warmup would not fit in a few steps."""
    settings = ("schedule.warmup = 1", *overrides)
    [run] = compile_runs(SMOKE.read_text(), filename=str(SMOKE), overrides=settings)
    return run


def constant(steps: int) -> tcfg.TrainConfig:
    return tcfg.TrainConfig(
        data=tcfg.SyntheticConfig(),
        optimizer=tcfg.AdamWConfig(lr=1e-3),
        seq_len=8,
        batch=2,
        steps=steps,
    )


def at(tree: object, key: str) -> Any:
    """`tree`'s entry `key`: optax's trees are typed as any tree of arrays."""
    return cast(Any, tree)[key]


class Matrix(nnx.Module):
    def __init__(self):
        self.w = nnx.Param(jnp.array([[1.0, -2.0, 0.5], [3.0, -4.0, -0.5]]))


# --- Cautious weight decay --------------------------------------------------------


def test_cautious_decay_decays_where_the_update_has_the_weight_s_sign():
    """Chen et al.'s rule, u + lambda I(u x >= 0) x, of the entries the mask keeps."""
    x = {"w": jnp.array([1.0, -2.0, 3.0, 0.0]), "b": jnp.array([1.0, 1.0])}
    u = {"w": jnp.array([0.5, 0.5, -0.5, 0.5]), "b": jnp.array([1.0, -1.0])}
    mask = lambda params: {"w": True, "b": False}
    tx = decayed_weights(0.1, mask, cautious=True)
    got, _ = tx.update(u, tx.init(x), x)
    np.testing.assert_allclose(at(got, "w"), [0.5 + 0.1, 0.5, -0.5, 0.5])
    np.testing.assert_array_equal(at(got, "b"), u["b"])
    plain = decayed_weights(0.1, mask, cautious=False)
    np.testing.assert_allclose(
        at(plain.update(u, plain.init(x), x)[0], "w"), u["w"] + 0.1 * x["w"]
    )


def test_adamw_s_first_step_decays_cautiously_and_as_optax_without_it():
    m = Matrix()
    params = nnx.state(m, nnx.Param)
    grads = jax.tree.map(
        lambda w: jnp.array([[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]]), params
    )
    lr = lr_multiplier(constant(1))

    def step(**settings) -> Float[np.ndarray, "2 3"]:
        c = tcfg.AdamWConfig(lr=0.1, weight_decay=0.5, **settings)
        tx = optimizer(c, lr, m)
        updates, _ = tx.update(grads, tx.init(params), params)
        return np.asarray(at(updates, "w")[...])

    x, g = np.asarray(params["w"][...]), np.asarray(grads["w"][...])
    u = g / (np.abs(g) + 1e-8)  # Adam's first step: m and v bias-corrected
    want = -0.1 * (u + 0.5 * np.where(u * x >= 0, x, 0))
    np.testing.assert_allclose(step(cautious_weight_decay=True), want, rtol=1e-6)
    theirs = optax.adamw(0.1, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.5)
    expected = theirs.update(grads, theirs.init(params), params)[0]
    np.testing.assert_array_equal(step(), np.asarray(at(expected, "w")[...]))


# --- NorMuon ---------------------------------------------------------------------------


def official(
    o: Float[np.ndarray, "out n"], v: Float[np.ndarray, "out 1"], b2: float
) -> tuple[Float[np.ndarray, "out n"], Float[np.ndarray, "out 1"]]:
    """NorMuon's official step on a torch weight (out, in): each row's mean square."""
    v = b2 * v + (1 - b2) * np.mean(o * o, -1, keepdims=True)
    y = o / (np.sqrt(v) + 1e-10)
    return y * np.linalg.norm(o) / (np.linalg.norm(y) + 1e-10), v


def test_neurons_are_normalized_as_normuon_s_code_does():
    """A kernel (in, out) is a torch weight transposed: its neurons are its columns."""
    rng = np.random.default_rng(0)
    kernels = [rng.normal(size=(6, 4)).astype(np.float32) for _ in range(2)]
    tx = neuron_normalized({("w",): MuonDimensionNumbers(0, 1)}, b2=0.9, eps=1e-10)
    state = tx.init({"w": jnp.zeros((6, 4))})
    v = np.zeros((4, 1), np.float32)
    for k in kernels:
        got, state = tx.update({"w": jnp.asarray(k)}, state)
        want, v = official(k.T, v, 0.9)
        np.testing.assert_allclose(np.asarray(at(got, "w")), want.T, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(at(state, "w")), v.T, rtol=1e-6)


def test_a_stack_of_experts_normalizes_each_expert_s_neurons_on_its_own():
    """MoE's up, (E, d, f, p): each expert's up and gate are matrices of their own."""
    rng = np.random.default_rng(1)
    up = rng.normal(size=(2, 6, 4, 2)).astype(np.float32)
    tx = neuron_normalized({("up",): MuonDimensionNumbers(1, 2)}, b2=0.95, eps=1e-10)
    got, _ = tx.update({"up": jnp.asarray(up)}, tx.init({"up": jnp.zeros(up.shape)}))
    for e in range(2):
        for p in range(2):
            want, _ = official(up[e, :, :, p].T, np.zeros((4, 1)), 0.95)
            np.testing.assert_allclose(
                np.asarray(at(got, "up"))[e, :, :, p], want.T, rtol=1e-5
            )


def test_normuon_trains_and_muon_keeps_its_chain():
    model = Model(
        compile_runs(SMOKE.read_text(), filename=str(SMOKE))[0].arch, rngs=nnx.Rngs(0)
    )
    lr = lr_multiplier(constant(1))
    plain = optimizer(tcfg.MuonConfig(lr=0.02), lr, model)
    normed = optimizer(tcfg.NorMuonConfig(lr=0.02), lr, model)
    params = nnx.state(model, nnx.Param)
    grads = jax.tree.map(jnp.sin, params)
    muon_state, normuon_state = plain.init(params), normed.init(params)
    # NorMuon's state is Muon's and the averages, one a neuron.
    extra = len(jax.tree.leaves(normuon_state)) - len(jax.tree.leaves(muon_state))
    assert extra == len(muon_matrices(model))
    updates = normed.update(grads, normuon_state, params)[0]
    assert all(jnp.isfinite(u).all() for u in jax.tree.leaves(updates))
    a, b = (
        o.update(grads, s, params)[0]
        for o, s in ((plain, muon_state), (normed, normuon_state))
    )
    assert not jax.tree.all(jax.tree.map(jnp.array_equal, a, b))


# --- Run ids ---------------------------------------------------------------------------


def test_the_options_off_keep_run_ids_and_on_make_their_own():
    base = smoke()
    assert smoke("optimizer.cautious_weight_decay = false").id == base.id
    assert smoke("ema = none").id == base.id
    assert smoke("optimizer.cautious_weight_decay = true").id != base.id
    assert smoke("ema = ema()").id != base.id
    muon = smoke("optimizer = muon(lr=0.02, adam=adamw(lr=3e-3))")
    off = "optimizer = muon(lr=0.02, adam=adamw(lr=3e-3, cautious_weight_decay=false))"
    assert smoke(off).id == muon.id
    assert smoke("optimizer = normuon(lr=0.02, adam=adamw(lr=3e-3))").id != muon.id


# --- The tail EMA ---------------------------------------------------------------------


def test_the_average_seeds_at_its_window_s_first_step_and_moves_by_its_rate():
    """A window of 3 of 5 steps, a rate of 2 / 4: seeded by step 3's weights, then
    halfway toward steps 4's and 5's."""
    tx = ema.tracked(tcfg.EMAConfig(window=3), 5)
    w = {"w": jnp.zeros(2)}
    state = tx.init(w)
    seen = []
    for i in range(1, 6):
        u = {"w": jnp.full(2, float(i))}
        out, state = tx.update(u, state, w)
        assert out is u
        w = optax.apply_updates(w, u)
        seen.append(float(cast(ema.EMAState, state).average["w"][0]))
    after = [1.0, 3.0, 6.0, 10.0, 15.0]  # the weights after each step
    seed = after[2]
    fourth = seed + 0.5 * (after[3] - seed)
    assert seen == [0.0, 0.0, seed, fourth, fourth + 0.5 * (after[4] - fourth)]
    assert tcfg.EMAConfig(window=0.25).steps(10) == 2
    assert tcfg.EMAConfig(window=0.01).steps(10) == 1


def test_a_run_resumes_inside_its_window_and_ends_where_it_did(tmp_path):
    """The average is in the checkpoint: a run that resumes at step 4, the window's
    first, ships the weights it shipped, which an export reads."""
    run = smoke(
        "steps = 6",
        "checkpoint = checkpoint(every=2, keep=4)",
        "ema = ema(window=3, blend=0.5)",
        "mesh = mesh(data=1)",
    )
    whole = train(run, tmp_path)
    folder = directory(run, tmp_path)
    shipped = nnx.state(restore(folder, 6).model, nnx.Param)
    shutil.rmtree(folder / "checkpoints" / "6")
    resumed = train(run, tmp_path)
    assert resumed["loss"] == whole["loss"]
    again = nnx.state(restore(folder, 6).model, nnx.Param)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, again, shipped))


def test_the_last_step_s_weights_are_the_blend_of_the_average_and_its_own(tmp_path):
    def ended(blend: float | None) -> nnx.State:
        given = ["steps = 4", "checkpoint = checkpoint(every=4)", "mesh = mesh(data=1)"]
        given.append("schedule = constant(warmup=1)")  # wsd's last steps barely move
        if blend is not None:
            given.append(f"ema = ema(window=2, blend={blend})")
        run = smoke(*given)
        train(run, tmp_path)
        return nnx.state(restore(directory(run, tmp_path), 4).model, nnx.Param)

    last, none_of_it = ended(None), ended(0.0)  # the average changes no step
    close = jax.tree.map(lambda a, b: jnp.allclose(a, b, atol=1e-7), last, none_of_it)
    assert jax.tree.all(close)
    whole, half = ended(1.0), ended(0.5)
    blend = jax.tree.map(lambda w, a: w + 0.5 * (a - w), last, whole)
    near = jax.tree.map(lambda a, b: jnp.allclose(a, b, atol=1e-6), half, blend)
    assert jax.tree.all(near)
    assert not jax.tree.all(jax.tree.map(jnp.array_equal, last, whole))


def test_a_window_past_the_run_s_steps_averages_every_step():
    assert ema.first(tcfg.EMAConfig(window=50), 10) == 1


@pytest.mark.parametrize("window", [0, 1.5])
def test_a_window_is_some_steps_or_a_share_of_them(window):
    with pytest.raises(ValueError):
        tcfg.EMAConfig(window=window)
