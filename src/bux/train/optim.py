"""Learning-rate schedules and optimizers, built from their configs with optax.

Each is found by its config's type: a schedule in ``schedules``, an optimizer in
``optimizers``. A schedule's builder gets the config and the run's steps, and returns
its curve after the warmup; an optimizer's gets the config, the schedule's multiplier
and the model, and returns an optax transformation.
"""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxtyping import Array, ArrayLike, Float, PyTree, Real
from optax.contrib import MuonDimensionNumbers, scale_by_muon
from optax.contrib._muon import scale_by_shape

from bux.architecture import DyT, Embedding, Engram, Head, MoE, Norm
from bux.train import config as tcfg
from bux.train.registry import Registry

Schedule = Callable[[Real[ArrayLike, ""]], Float[Array, ""]]  # of the step

schedules = Registry[Schedule]("schedule", "bux.train.optim.schedules")
optimizers = Registry[optax.GradientTransformation](
    "optimizer", "bux.train.optim.optimizers"
)


# --- Schedules --------------------------------------------------------------------------


def schedule(c: tcfg.ScheduleConfig, steps: int) -> Schedule:
    """The multiplier of the peak learning rate at each step of a run of `steps`: a
    linear warmup, times the schedule's own curve."""
    curve = schedules.build(c, steps)

    def multiplier(step: Real[ArrayLike, ""]) -> Float[Array, ""]:
        step = jnp.asarray(step, jnp.float32)
        warm = jnp.minimum((step + 1) / c.warmup, 1.0) if c.warmup else 1.0
        return jnp.asarray(warm * curve(step))

    return multiplier


def peak(lr: float, multiplier: Schedule) -> Schedule:
    """The learning rate at each step: the peak `lr`, scaled by `multiplier`."""
    return lambda step: lr * multiplier(step)


@schedules.register("constant")
def _(c: tcfg.ConstantSchedule, steps: int) -> Schedule:
    return lambda step: jnp.asarray(1.0)


@schedules.register("cosine")
def _(c: tcfg.CosineSchedule, steps: int) -> Schedule:
    def curve(step: Real[ArrayLike, ""]) -> Float[Array, ""]:
        t = jnp.clip((step + 1 - c.warmup) / max(steps - c.warmup, 1), 0, 1)
        return c.final + (1 - c.final) * 0.5 * (1 + jnp.cos(math.pi * t))

    return curve


COOLDOWNS: dict[str, Callable[[Float[Array, ""]], Float[Array, ""]]] = {
    "linear": lambda t: 1 - t,
    "sqrt": lambda t: 1 - jnp.sqrt(t),
    "cosine": lambda t: 0.5 * (1 + jnp.cos(math.pi * t)),
}


@schedules.register("wsd")
def _(c: tcfg.WSDSchedule, steps: int) -> Schedule:
    cooldown = c.cooldown(steps)

    def curve(step: Real[ArrayLike, ""]) -> Float[Array, ""]:
        t = jnp.clip((step + 1 - (steps - cooldown)) / max(cooldown, 1), 0, 1)
        return c.final + (1 - c.final) * COOLDOWNS[c.shape](t)

    return curve


# --- Optimizers -------------------------------------------------------------------------


def optimizer(
    c: tcfg.OptimizerConfig,
    multiplier: Schedule,
    model: nnx.Module,
    clip: float | None = None,
) -> optax.GradientTransformation:
    """The update rule of `c`, its learning rate scaled by `multiplier`, after clipping
    gradients to the global norm `clip`."""
    tx = optimizers.build(c, multiplier, model)
    if clip is None:
        return tx
    return optax.chain(optax.clip_by_global_norm(clip), tx)


def _keys(path: tuple) -> tuple:
    """A pytree path as plain keys, without the Variable's trailing `value`."""
    keys = tuple(
        next(getattr(k, a) for a in ("key", "idx", "name") if hasattr(k, a))
        for k in path
    )
    return keys[:-1] if keys and keys[-1] == "value" else keys


def undecayed(model: nnx.Module) -> list[tuple]:
    """The paths that weight decay leaves alone, whatever their shape: norms' and
    Engram's n-gram tables."""
    found: list[tuple] = []
    for path, node in nnx.iter_graph(model):
        if isinstance(node, Norm | DyT):
            found.append(path)
        elif isinstance(node, Engram):
            found.append((*path, "table"))
    return found


def decayed(model: nnx.Module) -> Callable[[PyTree], PyTree[bool]]:
    """The weight decay mask: matrices and stacks of them, outside `undecayed(model)`."""
    skip = undecayed(model)

    def decays(path: tuple, x: Float[Array, "*_"]) -> bool:
        keys = _keys(path)
        return x.ndim >= 2 and not any(keys[: len(s)] == s for s in skip)

    return lambda params: jax.tree_util.tree_map_with_path(decays, params)


@optimizers.register("adamw")
def adamw(
    c: tcfg.AdamWConfig, multiplier: Schedule, model: nnx.Module
) -> optax.GradientTransformation:
    return optax.adamw(
        peak(c.lr, multiplier),
        b1=c.b1,
        b2=c.b2,
        eps=c.eps,
        weight_decay=c.weight_decay,
        mask=decayed(model),
    )


def matrices(model: nnx.Module) -> dict[tuple, MuonDimensionNumbers]:
    """The parameters Muon trains, by path: the kernels of linear layers in the blocks,
    and stacked expert weights, one expert at a time. The embedding and the LM head
    are left out."""
    outside = [
        path
        for path, node in nnx.iter_graph(model)
        if isinstance(node, Embedding | Head)
    ]
    found: dict[tuple, MuonDimensionNumbers] = {}
    for path, node in nnx.iter_graph(model):
        if any(path[: len(o)] == o for o in outside):
            continue
        if isinstance(node, nnx.Linear):
            found[(*path, "kernel")] = MuonDimensionNumbers(0, 1)
        elif isinstance(node, MoE):
            for name in ("up", "gate", "down"):
                if getattr(node, name, None) is not None:
                    found[(*path, name)] = MuonDimensionNumbers(1, 2)
    return found


@optimizers.register("muon")
def muon(
    c: tcfg.MuonConfig, multiplier: Schedule, model: nnx.Module
) -> optax.GradientTransformation:
    """Muon for `matrices(model)`, and AdamW for the rest."""
    dims = matrices(model)

    def labels(params: PyTree[Float[Array, "*_"]]) -> PyTree[str]:
        return jax.tree_util.tree_map_with_path(
            lambda p, _: "muon" if _keys(p) in dims else "adam", params
        )

    # Muon sees only its own parameters; the others are masked out of its tree.
    def shapes(updates: PyTree[Float[Array, "*_"]]) -> PyTree[MuonDimensionNumbers]:
        return jax.tree_util.tree_map_with_path(lambda p, _: dims[_keys(p)], updates)

    orthogonalized = optax.chain(
        scale_by_muon(
            ns_steps=c.ns_steps,
            beta=c.momentum,
            nesterov=c.nesterov,
            weight_dimension_numbers=shapes,
        ),
        scale_by_shape(
            weight_dimension_numbers=shapes, consistent_rms=c.consistent_rms
        ),
        optax.add_decayed_weights(c.weight_decay, mask=decayed(model)),
        optax.scale_by_learning_rate(peak(c.lr, multiplier)),
    )
    adam = c.adam or tcfg.AdamWConfig(lr=c.lr)
    return optax.partition(
        {"muon": orthogonalized, "adam": adamw(adam, multiplier, model)}, labels
    )
