"""Rotary position embeddings, with position interpolation and YaRN scaling."""

import math

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from bux import config as cfg


def _attention_factor(y: cfg.YaRNConfig) -> float:
    if y.attention_factor is not None:
        return y.attention_factor
    return 0.1 * math.log(y.factor) + 1 if y.factor > 1 else 1.0


def frequencies(
    rope: cfg.RoPEConfig, dims: int
) -> tuple[Float[Array, " pairs"], float]:
    """Inverse frequencies of the `dims` rotated dims, and the factor on cos and sin."""
    inv_freq = 1.0 / rope.base ** (jnp.arange(0, dims, 2) / dims)
    match rope.scaling:
        case None:
            return inv_freq, 1.0
        case cfg.PositionInterpolationConfig(factor=factor):
            return inv_freq / factor, 1.0
        case cfg.YaRNConfig() as y:
            # The pair that turns `turns` times over the original context, as
            # transformers computes it (with truncation to whole pairs).
            def pair(turns: float) -> float:
                return (
                    dims
                    * math.log(y.original_max_len / (turns * 2 * math.pi))
                    / (2 * math.log(rope.base))
                )

            low = max(math.floor(pair(y.beta_fast)), 0)
            high = min(math.ceil(pair(y.beta_slow)), dims - 1)
            ramp = jnp.clip(
                (jnp.arange(dims // 2) - low) / ((high - low) or 1e-3), 0, 1
            )
            inv_freq = inv_freq / y.factor * ramp + inv_freq * (1 - ramp)
            on_rotary = y.attention_factor_on == "rotary"
            return inv_freq, _attention_factor(y) if on_rotary else 1.0
        case other:
            raise NotImplementedError(f"{other.type} RoPE scaling")


def logit_gain(rope: cfg.RoPEConfig | None) -> float:
    """YaRN's attention_factor ** 2 if it goes on the softmax scale, else 1."""
    y = rope.scaling if rope is not None else None
    if isinstance(y, cfg.YaRNConfig) and y.attention_factor_on == "softmax":
        return _attention_factor(y) ** 2
    return 1.0


def rotate(
    x: Float[Array, "b t h k"],
    rope: cfg.RoPEConfig,
    positions: Int[Array, "b t"] | None = None,
) -> Float[Array, "b t h k"]:
    """Rotate the first `fraction` of each head's dims, paired by `layout`, by the
    positions of the tokens, or their indices."""
    dims = int(x.shape[-1] * rope.fraction) // 2 * 2
    if dims == 0:
        return x
    inv_freq, gain = frequencies(rope, dims)
    if positions is None:
        positions = jnp.arange(x.shape[1])
    angles = positions[..., None] * inv_freq
    sin, cos = (
        gain * jnp.sin(angles)[..., None, :],
        gain * jnp.cos(angles)[..., None, :],
    )
    x, rest = x[..., :dims], x[..., dims:]
    if rope.layout == "interleaved":
        x1, x2 = x[..., 0::2], x[..., 1::2]
        y = jnp.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).reshape(x.shape)
    else:
        x1, x2 = x[..., : dims // 2], x[..., dims // 2 :]
        y = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
    return jnp.concatenate([y, rest], -1)
