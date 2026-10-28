"""QK-norm then RoPE: each head's RMSNorm, then its rotation, in one pass over the
queries or keys forward and one backward.

XLA takes two passes forward (the norm's statistics, then the rest) and three back
(the rotation's transpose written by halves, the norm's statistics, then the rest).
On a GPU, Pallas Triton kernels make each one (`norm_and_rotate`): in megatron.bux's
step on a B300, Qwen3-30B-A3B's queries, 8 rows of 4096 of 32 heads of 128, take
0.12 ms a layer forward and 0.24 back, where XLA's fusions took 0.17 and 0.46. They
take an RMSNorm with a scale and no bias, and RoPE by halves of every dim; elsewhere
the norm and `rotate` run as they are.

A program takes a few tokens, every head of each, (tokens, heads, d), and loads them
by halves, so that dim i rotates with i + d / 2 without slicing a value. The
backward computes in float32 throughout, where XLA rounds the rotation's gradient
and the norm's to bfloat16 between them.
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux import kernels, parallel
from bux.architecture.attention.rope import frequencies, rotate, rotated_dims
from bux.architecture.layers.norm import Norm
from bux.parallel.mesh import manual_axes

F32 = jnp.float32
Ref = Any  # a Pallas ref, which indexes and loads as an array does not
# About how many rows (a token's head each) a program takes, forward and backward: on a
# B300 5.5 TB/s forward and 4.1 back at 32 heads; at 4 heads 5.9 and 3.9.
FORWARD_ROWS, BACKWARD_ROWS = 32, 8


def norm_and_rotate(
    x: Float[Array, "b t h d"],
    norm: Norm | None,
    rope: cfg.RoPEConfig | None,
    positions: Int[Array, "b t"] | Int[Array, "b t a"] | None,
    heads: parallel.Name,
) -> Float[Array, "b t h d"]:
    """Return x's heads normalized by `norm`, then rotated by `rope`, each if given.

    The kernels make both in one pass where they take the call (`takes`); elsewhere
    `norm` and `~bux.architecture.attention.rope.rotate` run one after the other. On
    a mesh each device runs the kernels on its own rows, tokens and heads.

    Args:
        x: queries or keys: b rows of t tokens, h heads of d dims.
        norm: QK-norm, a norm of each head's d dims; None for none.
        rope: the RoPE settings; None for no rotation.
        positions: each token's position in its document, of one axis or several
            (`~bux.architecture.attention.rope.angles`); None for its index in the
            row.
        heads: the name of x's third dimension, ``heads`` or ``kv_heads``, as the
            layout splits it (`bux.parallel`).

    Returns:
        `x` normalized and rotated, in its dtype.
    """
    if norm is None or rope is None or not takes(x, norm, rope, heads):
        x = x if norm is None else norm(x)
        return x if rope is None else rotate(x, rope, positions)
    if positions is None:
        positions = jnp.broadcast_to(jnp.arange(x.shape[1]), x.shape[:2])
    if positions.ndim == 3:  # a RoPE of one axis turns by the first
        positions = positions[..., 0]
    run = partial(_fused, rope=rope, eps=norm.eps)
    if parallel.active() is None:
        return run(x, _gain(norm), positions)
    rows = parallel.spec(("batch", "seq", heads, None), x.shape)
    specs = (rows, P(), P(*rows[:2]))
    return kernels.per_device(run, specs, rows)(x, _gain(norm), positions)


def takes(
    x: Float[Array, "b t h d"], norm: Norm, rope: cfg.RoPEConfig, heads: parallel.Name
) -> bool:
    """Return whether the kernels take a call of `norm_and_rotate`.

    They take one on a GPU outside any shard_map, of 16-bit x whose head dim (32 to
    256) and heads on each device are powers of two, an RMSNorm with no bias and a
    scale after it (not ``scale_input``), and RoPE by halves of every dim, of
    positions of one axis.
    """
    _, _, h, d = x.shape
    local = h // parallel.splits(("batch", "seq", heads, None), x.shape)[2]
    rms = norm.kind == "rmsnorm" and norm.scale is not None and norm.bias is None
    halves = rope.layout == "half" and rotated_dims(d, rope) == d
    one_axis = rope.sections is None and rope.axial is None
    return (
        kernels.gpu()
        and not manual_axes()
        and x.dtype in (jnp.bfloat16, jnp.float16)
        and _power_of_two(local)
        and _power_of_two(d)
        and 32 <= d <= 256
        and rms
        and not norm.scale_input
        and halves
        and one_axis
    )


def _power_of_two(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _gain(norm: Norm) -> Float[Array, " d"]:
    """Return what the norm multiplies by: its scale, or 1 + it if zero-centered."""
    assert norm.scale is not None
    scale = norm.scale[...]
    return 1 + scale if norm.zero_centered else scale


def _angles(
    rope: cfg.RoPEConfig, positions: Int[Array, "b t"], d: int
) -> tuple[Float[Array, "bt half"], Float[Array, "bt half"]]:
    """Return the cos and sin of each token's angles, as `rotate` computes them, a
    token a row: the forward and backward compute them, so the backward keeps only
    the positions."""
    inv_freq, gain = frequencies(rope, d)
    angles = positions.reshape(-1)[:, None] * inv_freq
    return gain * jnp.cos(angles), gain * jnp.sin(angles)


# --- The kernels: one device's tokens ---------------------------------------------------


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _fused(
    x: Float[Array, "b t h d"],
    gain: Float[Array, " d"],
    positions: Int[Array, "b t"],
    rope: cfg.RoPEConfig,
    eps: float,
) -> Float[Array, "b t h d"]:
    """Return x normalized with `gain` and rotated, by the kernels, forward and back."""
    return _forward(x, gain, positions, rope, eps)[0]


def _forward(
    x: Float[Array, "b t h d"],
    gain: Float[Array, " d"],
    positions: Int[Array, "b t"],
    rope: cfg.RoPEConfig,
    eps: float,
) -> tuple[Float[Array, "b t h d"], tuple]:
    """Return `_fused`'s output by the forward kernel, and what the backward keeps:
    x, the gain and the positions."""
    b, t, h, d = x.shape
    tokens = _tokens(b * t, h, FORWARD_ROWS)
    rows, gains, angles = _blocks(h, d, tokens)
    y = _call(
        partial(_forward_kernel, eps=eps),
        grid=(b * t // tokens,),
        in_specs=[rows, gains, angles, angles],
        out_specs=rows,
        out_shape=jax.ShapeDtypeStruct((b * t, h, d), x.dtype),
        warps=4,
        name="qk_norm_rope",
    )(_by_token(x), gain.reshape(1, d), *_angles(rope, positions, d))
    return y.reshape(x.shape), (x, gain, positions)


def _backward(
    rope: cfg.RoPEConfig, eps: float, saved: tuple, dy: Float[Array, "b t h d"]
) -> tuple[Float[Array, "b t h d"], Float[Array, " d"], None]:
    """Return the gradients of x and the gain: the gain's summed over each program's
    rows by the kernel, and over the programs by XLA."""
    x, gain, positions = saved
    b, t, h, d = x.shape
    tokens = _tokens(b * t, h, BACKWARD_ROWS)
    programs = b * t // tokens
    rows, gains, angles = _blocks(h, d, tokens)
    dx, d_gain = _call(
        partial(_backward_kernel, eps=eps),
        grid=(programs,),
        in_specs=[rows, rows, gains, angles, angles],
        out_specs=[rows, pl.BlockSpec((1, d), lambda i: (i, 0))],
        out_shape=[
            jax.ShapeDtypeStruct((b * t, h, d), x.dtype),
            jax.ShapeDtypeStruct((programs, d), F32),
        ],
        warps=4 if tokens * h >= 32 else 1,
        name="qk_norm_rope_bwd",
    )(_by_token(x), _by_token(dy), gain.reshape(1, d), *_angles(rope, positions, d))
    return dx.reshape(x.shape), d_gain.sum(0).astype(gain.dtype), None


_fused.defvjp(_forward, _backward)


def _tokens(n: int, h: int, rows: int) -> int:
    """Return the tokens a program takes, about `rows` rows: a power of two that
    divides the n tokens."""
    tokens = max(1, rows // h)
    while n % tokens:
        tokens //= 2
    return tokens


def _blocks(h: int, d: int, tokens: int) -> tuple[pl.BlockSpec, ...]:
    """Return the blocks of a program: its tokens' rows, the gain, and their angles."""
    rows = pl.BlockSpec((tokens, h, d), lambda i: (i, 0, 0))
    gains = pl.BlockSpec((1, d), lambda i: (0, 0))
    angles = pl.BlockSpec((tokens, d // 2), lambda i: (i, 0))
    return rows, gains, angles


def _by_token(a: Float[Array, "b t ..."]) -> Float[Array, "tokens ..."]:
    """Return `a` with its rows' positions flattened into tokens."""
    b, t, *e = a.shape
    return a.reshape(b * t, *e)


def _call(kernel: Any, warps: int, name: str, **specs: Any) -> Any:
    """Return the Pallas call of `kernel`, by Triton on a GPU."""
    from jax.experimental.pallas import triton as plgpu

    params = plgpu.CompilerParams(num_warps=warps, num_stages=1)
    return pl.pallas_call(
        kernel,
        compiler_params=params,
        interpret=kernels.interpreted(),
        name=name,
        **specs,
    )


def _halves(ref: Ref) -> tuple[Float[Array, "n h half"], Float[Array, "n h half"]]:
    """Load a block of rows by halves, in float32."""
    half = ref.shape[-1] // 2
    return ref[:, :, :half].astype(F32), ref[:, :, half:].astype(F32)


def _rsqrt(
    x1: Float[Array, "n h half"], x2: Float[Array, "n h half"], eps: float
) -> Float[Array, "n h 1"]:
    """Return each row's 1 / RMS, the RMS of both its halves."""
    squares = jnp.sum(x1 * x1, -1) + jnp.sum(x2 * x2, -1)
    return jax.lax.rsqrt(squares / (2 * x1.shape[-1]) + eps)[..., None]


def _rounded(x: Float[Array, "*s"], dtype: Any) -> Float[Array, "*s"]:
    """Return x rounded to `dtype`, in float32: as XLA rounds between ops."""
    return x.astype(dtype).astype(F32)


def _forward_kernel(
    x_ref: Ref, gain_ref: Ref, cos_ref: Ref, sin_ref: Ref, y_ref: Ref, eps: float
) -> None:
    """Normalize and rotate a block of rows, rounding as `Norm` and `rotate` do."""
    dtype, half = x_ref.dtype, x_ref.shape[-1] // 2
    x1, x2 = _halves(x_ref)
    r = _rsqrt(x1, x2, eps)
    g1, g2 = gain_ref[:, :half].astype(F32), gain_ref[:, half:].astype(F32)
    m1 = _rounded(_rounded(x1 * r, dtype) * g1[None], dtype)
    m2 = _rounded(_rounded(x2 * r, dtype) * g2[None], dtype)
    c, s = cos_ref[...][:, None, :], sin_ref[...][:, None, :]
    y_ref[:, :, :half] = (m1 * c - m2 * s).astype(y_ref.dtype)
    y_ref[:, :, half:] = (m2 * c + m1 * s).astype(y_ref.dtype)


def _backward_kernel(
    x_ref: Ref,
    dy_ref: Ref,
    gain_ref: Ref,
    cos_ref: Ref,
    sin_ref: Ref,
    dx_ref: Ref,
    d_gain_ref: Ref,
    eps: float,
) -> None:
    """The gradients of a block of rows: dx, and the gain's summed over the rows."""
    half = x_ref.shape[-1] // 2
    x1, x2 = _halves(x_ref)
    r = _rsqrt(x1, x2, eps)
    c, s = cos_ref[...][:, None, :], sin_ref[...][:, None, :]
    dy1, dy2 = _halves(dy_ref)
    dm1, dm2 = dy1 * c + dy2 * s, dy2 * c - dy1 * s  # the rotation's transpose
    xh1, xh2 = x1 * r, x2 * r
    dtype = x_ref.dtype
    d_gain_ref[0, :half] = jnp.sum(jnp.sum(dm1 * _rounded(xh1, dtype), 0), 0)
    d_gain_ref[0, half:] = jnp.sum(jnp.sum(dm2 * _rounded(xh2, dtype), 0), 0)
    dn1 = dm1 * gain_ref[:, :half].astype(F32)[None]
    dn2 = dm2 * gain_ref[:, half:].astype(F32)[None]
    mean = (jnp.sum(dn1 * xh1, -1) + jnp.sum(dn2 * xh2, -1))[..., None] / (2 * half)
    dx_ref[:, :, :half] = (r * (dn1 - xh1 * mean)).astype(dx_ref.dtype)
    dx_ref[:, :, half:] = (r * (dn2 - xh2 * mean)).astype(dx_ref.dtype)
