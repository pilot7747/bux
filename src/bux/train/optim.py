"""Learning-rate schedules and optimizers, built from their configs with optax."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.typing import ArrayLike
from optax.contrib import MuonDimensionNumbers, scale_by_muon
from optax.contrib._muon import scale_by_shape

from bux.architecture import Embedding, Head, MoE
from bux.train import config as tcfg

Schedule = Callable[[ArrayLike], jax.Array]


def schedule(c: tcfg.ScheduleConfig, steps: int) -> Schedule:
    """The multiplier of the peak learning rate at each step of a run of `steps`."""

    def multiplier(step: ArrayLike) -> jax.Array:
        step = jnp.asarray(step, jnp.float32)
        warm = jnp.minimum((step + 1) / c.warmup, 1.0) if c.warmup else 1.0
        match c:
            case tcfg.CosineSchedule():
                t = jnp.clip((step + 1 - c.warmup) / max(steps - c.warmup, 1), 0, 1)
                rest = c.final + (1 - c.final) * 0.5 * (1 + jnp.cos(math.pi * t))
            case tcfg.WSDSchedule():
                cooldown = c.cooldown(steps)
                t = jnp.clip((step + 1 - (steps - cooldown)) / max(cooldown, 1), 0, 1)
                curve = {
                    "linear": 1 - t,
                    "sqrt": 1 - jnp.sqrt(t),
                    "cosine": 0.5 * (1 + jnp.cos(math.pi * t)),
                }[c.shape]
                rest = c.final + (1 - c.final) * curve
            case _:
                rest = 1.0
        return jnp.asarray(warm * rest)

    return multiplier


def _keys(path: tuple) -> tuple:
    """A pytree path as plain keys, without the Variable's trailing `value`."""
    keys = tuple(
        next(getattr(k, a) for a in ("key", "idx", "name") if hasattr(k, a))
        for k in path
    )
    return keys[:-1] if keys and keys[-1] == "value" else keys


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


def _adamw(c: tcfg.AdamWConfig, lr: Schedule) -> optax.GradientTransformation:
    return optax.adamw(
        lr,
        b1=c.b1,
        b2=c.b2,
        eps=c.eps,
        weight_decay=c.weight_decay,
        mask=lambda params: jax.tree.map(lambda x: x.ndim >= 2, params),
    )


def optimizer(
    c: tcfg.OptimizerConfig,
    multiplier: Schedule,
    model: nnx.Module,
    clip: float | None = None,
) -> optax.GradientTransformation:
    """The update rule of `c`, its learning rate scaled by `multiplier`, after clipping
    gradients to the global norm `clip`."""

    def lr(peak: float) -> Schedule:
        return lambda step: peak * multiplier(step)

    if isinstance(c, tcfg.AdamWConfig):
        tx = _adamw(c, lr(c.lr))
    else:
        dims = matrices(model)

        def labels(params):
            return jax.tree_util.tree_map_with_path(
                lambda p, _: "muon" if _keys(p) in dims else "adam", params
            )

        # Muon sees only its own parameters; the others are masked out of its tree.
        def shapes(updates):
            return jax.tree_util.tree_map_with_path(
                lambda p, _: dims[_keys(p)], updates
            )

        muon = optax.chain(
            scale_by_muon(
                ns_steps=c.ns_steps,
                beta=c.momentum,
                nesterov=c.nesterov,
                weight_dimension_numbers=shapes,
            ),
            scale_by_shape(
                weight_dimension_numbers=shapes, consistent_rms=c.consistent_rms
            ),
            optax.add_decayed_weights(c.weight_decay),
            optax.scale_by_learning_rate(lr(c.lr)),
        )
        adam = c.adam or tcfg.AdamWConfig(lr=c.lr)
        tx = optax.partition({"muon": muon, "adam": _adamw(adam, lr(adam.lr))}, labels)
    if clip is None:
        return tx
    return optax.chain(optax.clip_by_global_norm(clip), tx)
