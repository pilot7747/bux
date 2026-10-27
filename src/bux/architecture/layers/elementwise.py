"""Elementwise and plumbing layers: scale, linear, act and dropout."""

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux.architecture.common.functional import activation
from bux.architecture.common.sharding import projection
from bux.architecture.registry import Context, Taps, register
from bux.lang import ir


class Scale(nnx.Module):
    """A learned multiplier, ``scale``: x times a scalar, or times one per channel.

    ``scale(init=0)`` at the end of a branch is ReZero, ``scale(per_channel=true)``
    LayerScale, and ``[$x0 scale]`` adds a learned multiple of a tap to the stream.

    Args:
        c: Its settings (`bux.config.ScaleConfig`): where the multiplier starts, and
            whether it is one per channel.
        width: The width of the stream.
    """

    def __init__(self, c: cfg.ScaleConfig, width: int):
        shape = width if c.per_channel else ()
        self.value = nnx.Param(jnp.full(shape, c.init, jnp.float32))

    def __call__(self, x: Float[Array, "*shape"], taps: Taps) -> Float[Array, "*shape"]:
        """Return x times the multiplier."""
        return x * self.value


@register("scale")
def _(
    c: cfg.ScaleConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Scale:
    return Scale(c, width)


class Linear(nnx.Module):
    """A dense projection to `features` channels, ``linear``: ``x W + b``, in the
    format of the run's ``precision.matmuls``.

    Args:
        c: Its settings (`bux.config.LinearConfig`): the output channels and whether
            it has a bias.
        width: The width of its input.
        rngs: The random streams W starts from.
    """

    def __init__(self, c: cfg.LinearConfig, width: int, rngs: nnx.Rngs):
        self.proj = projection(width, c.features, use_bias=c.bias, rngs=rngs)

    def __call__(
        self, x: Float[Array, "*lead d"], taps: Taps
    ) -> Float[Array, "*lead f"]:
        """Return x projected to f, the config's `features`, channels."""
        return self.proj(x)


@register("linear")
def _(
    c: cfg.LinearConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Linear:
    return Linear(c, width, ctx.rngs)


class Activation(nnx.Module):
    """An activation applied elementwise, ``act``, or one written bare by its name:
    ``linear(features=h) swish``.

    Args:
        fn: The activation's name, one of `bux.config.Activation`.
    """

    def __init__(self, fn: str):
        self.fn = fn

    def __call__(self, x: Float[Array, "*shape"], taps: Taps) -> Float[Array, "*shape"]:
        """Return the activation of every entry of x."""
        return activation(self.fn, x)


@register("act")
def _(
    c: cfg.ActivationConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Activation:
    return Activation(c.fn)


class Dropout(nnx.Module):
    """Dropout of the stream, ``dropout``: in training, each entry zeroed at `rate`
    and the rest scaled by ``1 / (1 - rate)``; outside it (``model.eval()``), x as it
    is.

    Args:
        rate: The fraction of entries zeroed.
        rngs: The random streams the masks are drawn from.
    """

    def __init__(self, rate: float, rngs: nnx.Rngs):
        self.dropout = nnx.Dropout(rate, rngs=rngs)

    def __call__(self, x: Float[Array, "*shape"], taps: Taps) -> Float[Array, "*shape"]:
        """Return x, dropped out in training."""
        return self.dropout(x)


@register("dropout")
def _(
    c: cfg.DropoutConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Dropout:
    return Dropout(c.rate, ctx.rngs)
