"""Softmax and log-softmax whose gradients hold up where a float rounds the largest
probability to 1 (Barron et al., 2026, "A Fix for Softmax and Cross-Entropy").

The logits' margin m is the largest one minus the logsumexp of the rest. The largest
probability is ``1 / (1 + e^-m)``, and past m of about 16.6 in float32 (5.5 in
bfloat16), ``1 + e^-m`` rounds to 1. Its gradient takes ``1 - p``, which is then 0,
while the others' do not, so the gradient no longer sums to 0, and the log-softmax
there is 0 where it is ``-e^-m``. Here neither backward takes ``1 - p`` from p, and the
log-softmax is ``-log1p`` of the others' mass. A probability over 3/4 is the only one
of its row, so the largest one is found entry by entry. The softmax's forward is
jax's; each backward takes a pass more than jax's.

The cross-entropy of hard targets does the same with the target in place of the
largest logit (`bux.architecture.logits.Scores`).
"""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

#: The probability over which an entry is the largest of its row, and the only one.
TOP = 0.75


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def softmax(x: Float[Array, "*s"], axis: int = -1) -> Float[Array, "*s"]:
    """Return `jax.nn.softmax` of `x` over `axis`, whose gradient sums to 0 at any
    margin.

    Args:
        x: The logits, in the dtype to compute in.
        axis: The axis to normalize over.

    Returns:
        The probabilities, as jax gives them.
    """
    return jax.nn.softmax(x, axis)


def _softmax_fwd(x: Float[Array, "*s"], axis: int) -> tuple[Float[Array, "*s"], tuple]:
    p = jax.nn.softmax(x, axis)
    return p, (p,)


def _softmax_bwd(
    axis: int, saved: tuple, g: Float[Array, "*s"]
) -> tuple[Float[Array, "*s"]]:
    """Return jax's ``p * (g - sum(p * g))`` of g less the largest entry's g, which
    changes nothing but that entry's term of the sum, now 0: so its ``g * (1 - p)``
    does not round off."""
    (p,) = saved
    g = g - _sum(g, p > TOP, axis)
    return (p * (g - jnp.sum(p * g, axis, keepdims=True)),)


softmax.defvjp(_softmax_fwd, _softmax_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def log_softmax(x: Float[Array, "*s"], axis: int = -1) -> Float[Array, "*s"]:
    """Return the log-softmax of `x` over `axis`, as `jax.nn.log_softmax`, but at the
    largest entry with no tie, ``-log1p`` of the others' mass rather than 0, and with
    gradients that sum to 0 at any margin.

    Args:
        x: The logits, in the dtype to compute in.
        axis: The axis to normalize over.

    Returns:
        ``x - logsumexp(x)``.
    """
    return _log_softmax_fwd(x, axis)[0]


def _log_softmax_fwd(
    x: Float[Array, "*s"], axis: int
) -> tuple[Float[Array, "*s"], tuple]:
    """Return ``x - max - log(total)``, or ``-log1p(rest)`` where the largest logit's
    1 is most of the total: under 2, it has no tie."""
    shifted = x - jnp.max(x, axis, keepdims=True)  # 0 exactly where x is largest
    e = jnp.exp(shifted)
    total = jnp.sum(e, axis, keepdims=True)
    rest = _sum(e, shifted < 0, axis)
    out = shifted - jnp.where(total < 2, jnp.log1p(rest), jnp.log(total))
    return out, (out, rest)


def _log_softmax_bwd(
    axis: int, saved: tuple, g: Float[Array, "*s"]
) -> tuple[Float[Array, "*s"]]:
    """Return ``g - p * sum(g)``, and at the largest entry ``g * (1 - p) - p * (the
    others' sum of g)``, 1 - p from the others' mass."""
    out, rest = saved
    p = jnp.exp(out)
    top = p > TOP  # so rest is the others' mass, and 1 - p is rest * p
    g_top, g_rest = _sum(g, top, axis), _sum(g, ~top, axis)
    exact = p * (g_top * rest - g_rest)
    return (jnp.where(top, exact, g - p * (g_top + g_rest)),)


log_softmax.defvjp(_log_softmax_fwd, _log_softmax_bwd)


def _sum(
    x: Float[Array, "*s"], where: Bool[Array, "*s"], axis: int
) -> Float[Array, "*s"]:
    """Return the sum over `axis` of `x` where `where`."""
    return jnp.sum(jnp.where(where, x, 0), axis, keepdims=True)
