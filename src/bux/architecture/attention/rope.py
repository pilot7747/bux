"""Rotary position embeddings (RoPE), with position interpolation, YaRN and Llama 3
scaling.

`rotate` rotates the first ``rope.fraction`` of each head's dims by the tokens'
positions, in pairs as ``rope.layout`` says, at the inverse frequencies `frequencies`
gives under ``rope.scaling``. The softmax attention layers rotate by it, and so does
sparse attention's indexer.

A position may have several axes, ``b t k``: an image's patch has a row and a column,
and Qwen's M-RoPE gives every token a time, a height and a width. ``rope.sections``
deals the pairs out to the axes (M-RoPE), and ``rope.axial`` makes each axis a RoPE of
its own over a run of the pairs (a ViT's). A position of one axis, ``b t``, is the
same on every axis (`angles`).
"""

import math

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from bux import config as cfg


def _attention_factor(yarn: cfg.YaRNConfig) -> float:
    """Return YaRN's attention factor: as given, else 0.1 ln(factor) + 1."""
    if yarn.attention_factor is not None:
        return yarn.attention_factor
    return 0.1 * math.log(yarn.factor) + 1 if yarn.factor > 1 else 1.0


def rotated_dims(head_dim: int, rope: cfg.RoPEConfig | None) -> int:
    """Return how many of a head's `head_dim` dims `rope` rotates, in whole pairs: 0
    for no RoPE."""
    return 0 if rope is None else int(head_dim * rope.fraction) // 2 * 2


def frequencies(
    rope: cfg.RoPEConfig, dims: int
) -> tuple[Float[Array, " pairs"], float]:
    """Return the inverse frequencies of the rotated dims' pairs, and the factor on
    cos and sin.

    Args:
        rope: the RoPE settings. ``rope.scaling`` lowers the frequencies as
            transformers computes them: position interpolation divides them all by
            its factor; YaRN and Llama 3 divide the low ones, keep the high ones, and
            blend between them.
        dims: how many dims rotate, an even number.

    Returns:
        The ``dims / 2`` inverse frequencies, and YaRN's attention factor where it
        goes on cos and sin (``attention_factor_on=rotary``), else 1.
    """
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
    """Return YaRN's ``attention_factor ** 2`` if it goes on the softmax scale
    (``attention_factor_on=softmax``), else 1."""
    scaling = rope.scaling if rope is not None else None
    if isinstance(scaling, cfg.YaRNConfig) and scaling.attention_factor_on == "softmax":
        return _attention_factor(scaling) ** 2
    return 1.0


def angles(
    rope: cfg.RoPEConfig, positions: Int[Array, "*lead a"], dims: int
) -> tuple[Float[Array, "*lead pairs"], float]:
    """Return the angle each of the ``dims / 2`` rotated pairs turns by at each
    position, and the factor on cos and sin.

    Args:
        rope: the RoPE settings.
        positions: each token's position on its a axes, the last dimension: as many
            as ``rope.sections`` or ``rope.axial`` has, or one, the same on all. A
            RoPE of one axis turns by the first.
        dims: how many dims rotate, an even number.

    Raises:
        ValueError: The positions have another number of axes than the rope;
            ``rope.sections`` do not sum to the pairs, or ``rope.axial`` does not
            divide them.
    """
    pairs, axes = dims // 2, len(rope.sections or ()) or rope.axial or 1
    if axes == 1:
        positions = positions[..., :1]
    elif positions.shape[-1] == 1:
        positions = jnp.repeat(positions, axes, -1)  # one axis: all alike
    elif positions.shape[-1] != axes:
        raise ValueError(
            f"positions of {positions.shape[-1]} axes, for a RoPE of {axes}"
        )
    if rope.axial is not None:
        if pairs % rope.axial:
            raise ValueError(f"rope(axial={rope.axial}) splits {pairs} pairs unevenly")
        inv_freq, gain = frequencies(rope, dims // rope.axial)
        turns = positions[..., :, None] * inv_freq  # each axis its own RoPE
        return turns.reshape(*positions.shape[:-1], pairs), gain
    inv_freq, gain = frequencies(rope, dims)
    axis = jnp.asarray(_axis_of_pairs(rope, pairs))
    return jnp.take(positions, axis, -1) * inv_freq, gain


def _axis_of_pairs(rope: cfg.RoPEConfig, pairs: int) -> list[int]:
    """Return the axis each pair turns by under M-RoPE: 0 for every pair without."""
    if rope.sections is None:
        return [0] * pairs
    if sum(rope.sections) != pairs:
        raise ValueError(
            f"rope(sections={rope.sections}) deals out {sum(rope.sections)} pairs, "
            f"but {pairs} rotate"
        )
    k = len(rope.sections)
    if rope.sections_layout == "blocks":
        return [a for a, n in enumerate(rope.sections) for _ in range(n)]
    # Pair i goes to axis i mod k while that axis has pairs left; the rest to axis 0.
    return [
        a if a and i < rope.sections[a] * k else 0
        for i in range(pairs)
        for a in [i % k]
    ]


def rotate(
    x: Float[Array, "b t h k"],
    rope: cfg.RoPEConfig,
    positions: Int[Array, "b t"] | Int[Array, "b t a"] | None = None,
) -> Float[Array, "b t h k"]:
    """Rotate the first ``rope.fraction`` of each head's dims by the tokens' positions.

    ``rope.layout`` pairs the dims: ``half`` rotates dim i with i + dims / 2 (GPT-NeoX,
    Llama), ``interleaved`` dim 2i with 2i + 1 (GPT-J). The rotation computes in
    float32.

    Args:
        x: queries or keys: b rows of t tokens, h heads of k dims.
        rope: the RoPE settings.
        positions: each token's position in its document, of one axis or of a, as
            the rope's sections or axial say (`angles`); None for its index in the
            row.

    Returns:
        `x` with its first dims rotated and the rest as they are, in x's dtype.
    """
    dims = rotated_dims(x.shape[-1], rope)
    if dims == 0:
        return x
    if positions is None:
        positions = jnp.arange(x.shape[1])
    if positions.ndim < 3:  # of one axis
        positions = positions[..., None]
    turns, gain = angles(rope, positions, dims)
    sin, cos = (
        gain * jnp.sin(turns)[..., None, :],
        gain * jnp.cos(turns)[..., None, :],
    )
    x, rest = x[..., :dims], x[..., dims:]
    if rope.layout == "interleaved":
        x1, x2 = x[..., 0::2], x[..., 1::2]
        y = jnp.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).reshape(x.shape)
    else:
        x1, x2 = x[..., : dims // 2], x[..., dims // 2 :]
        y = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
    return jnp.concatenate([y.astype(x.dtype), rest], -1)
