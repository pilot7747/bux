"""Rotary position embeddings, with position interpolation, YaRN and Llama 3 scaling."""

import math

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from bux import config as cfg


def _attention_factor(yarn: cfg.YaRNConfig) -> float:
    if yarn.attention_factor is not None:
        return yarn.attention_factor
    return 0.1 * math.log(yarn.factor) + 1 if yarn.factor > 1 else 1.0


def rotated_dims(head_dim: int, rope: cfg.RoPEConfig | None) -> int:
    """Return how many of a head's dims RoPE rotates, in whole pairs."""
    return 0 if rope is None else int(head_dim * rope.fraction) // 2 * 2


def frequencies(
    rope: cfg.RoPEConfig, dims: int
) -> tuple[Float[Array, " pairs"], float]:
    """Return the inverse frequencies of the `dims` rotated dims, and the factor on
    cos and sin."""
    inv_freq = 1.0 / rope.base ** (jnp.arange(0, dims, 2) / dims)
    match rope.scaling:
        case None:
            return inv_freq, 1.0
        case cfg.PositionInterpolationConfig(factor=factor):
            return inv_freq / factor, 1.0
        case cfg.YaRNConfig() as yarn:
            # The pair that turns `turns` times over the original context, as
            # transformers computes it (with truncation to whole pairs).
            def pair(turns: float) -> float:
                return (
                    dims
                    * math.log(yarn.original_max_len / (turns * 2 * math.pi))
                    / (2 * math.log(rope.base))
                )

            low = max(math.floor(pair(yarn.beta_fast)), 0)
            high = min(math.ceil(pair(yarn.beta_slow)), dims - 1)
            ramp = jnp.clip(
                (jnp.arange(dims // 2) - low) / ((high - low) or 1e-3), 0, 1
            )
            inv_freq = inv_freq / yarn.factor * ramp + inv_freq * (1 - ramp)
            on_rotary = yarn.attention_factor_on == "rotary"
            return inv_freq, _attention_factor(yarn) if on_rotary else 1.0
        case cfg.Llama3RoPEScalingConfig() as llama3:
            # Pairs that turn fewer than low_freq_factor times over the original
            # context are divided by factor, those that turn more than
            # high_freq_factor times keep their frequency, with a linear blend in
            # between, as transformers computes it.
            turns = llama3.original_max_len * inv_freq / (2 * math.pi)
            span = llama3.high_freq_factor - llama3.low_freq_factor
            ramp = jnp.clip((turns - llama3.low_freq_factor) / span, 0, 1)
            return inv_freq / llama3.factor * (1 - ramp) + inv_freq * ramp, 1.0


def logit_gain(rope: cfg.RoPEConfig | None) -> float:
    """Return YaRN's attention_factor ** 2 if it goes on the softmax scale, else 1."""
    scaling = rope.scaling if rope is not None else None
    if isinstance(scaling, cfg.YaRNConfig) and scaling.attention_factor_on == "softmax":
        return _attention_factor(scaling) ** 2
    return 1.0


def rotate(
    x: Float[Array, "b t h k"],
    rope: cfg.RoPEConfig,
    positions: Int[Array, "b t"] | None = None,
) -> Float[Array, "b t h k"]:
    """Rotate the first `fraction` of each head's dims, paired as `layout` says, by the
    tokens' positions, or their indices: in float32, the result in x's dtype."""
    dims = rotated_dims(x.shape[-1], rope)
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
    return jnp.concatenate([y.astype(x.dtype), rest], -1)
