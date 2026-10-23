"""Grouped block-scaled GEMMs: an MoE's experts, y = x W[g] for the rows of each group
g as `grouped.ragged_dot` runs them, and its gradients, each operand of each of the
three GEMMs quantized along what that GEMM contracts as a `blockscaled.Recipe` says.
On a GPU Triton's `tl.dot_scaled` multiplies them, Blackwell's block-scaled MMA
(``grouped_triton``); elsewhere XLA dequantizes and multiplies, which computes the
same values.

The recipes whose values are E4M3 with a power-of-two scale per 32 run here
(RECIPES). Each group's W quantizes on its own. x^T and dy^T, which dW contracts along
the rows, quantize in blocks of 32 rows whatever the groups: a block two groups share
has one scale, and each group counts the other's rows as 0. Rows past the groups come
out 0, and add nothing to dW. Only the GEMMs in `passes` multiply in the format, the
others by ``grouped``'s kernels, as they are.
"""

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int, UInt32

from bux import kernels
from bux.architecture import blockscaled, grouped
from bux.architecture.blockscaled import PASSES, Blocks, Pass, Twice

try:  # Triton's kernels, where the cuda extra is installed (Linux)
    from bux.architecture import grouped_triton
except ImportError:
    grouped_triton = None

BLOCK = blockscaled.MX_BLOCK
TRITON_QUANTIZES = True  # MXFP8's quantizer on a GPU: Triton's kernel, or blockscaled's
# The formats of ``matmuls`` that run here, by their recipes: train.config.GROUPED.
RECIPES = {"mxfp8": blockscaled.MXFP8, "mxfp8_2d": blockscaled.MXFP8_2D}


def fits(x: Float[Array, "m k"], w: Float[Array, "g k n"]) -> bool:
    """Whether what x W contracts, and its columns, are whole blocks."""
    return x.shape[-1] % BLOCK == 0 and w.shape[-1] % BLOCK == 0


def ragged_dot(
    x: Float[Array, "m k"],
    w: Float[Array, "g k n"],
    sizes: Int[Array, " g"],
    recipe: blockscaled.Recipe,
    seed: UInt32[Array, ""],
    passes: tuple[Pass, ...] = PASSES,
) -> Float[Array, "m n"]:
    """Return what `grouped.ragged_dot` does, its GEMMs in `passes` as `recipe`
    quantizes them. Rows go in blocks of 32 too, so x gains rows of 0 up to them."""
    m = x.shape[0]
    whole = -(-m // BLOCK) * BLOCK
    x = jnp.pad(x, ((0, whole - m), (0, 0))) if whole > m else x
    dtypes = (x.dtype, w.dtype, jnp.result_type(x, w))
    return _ragged(x, w, sizes, seed, recipe, dtypes, passes)[:m]


Dtypes = tuple[np.dtype, np.dtype, np.dtype]  # x's, W's and y's
# An operand of a GEMM: in the format, or as it is where its pass multiplies as it is.
Rows = Blocks | Float[Array, "m k"]
Matrices = Blocks | Float[Array, "g k n"]
Saved = tuple[Rows, Matrices, Int[Array, " g"], UInt32[Array, ""]]
# How each group's (n, k) matrix stacks in blocks: by rows, (g n, k); by columns, (n, g k).
Stacked = Literal["rows", "columns"]


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _ragged(
    x: Float[Array, "m k"],
    w: Float[Array, "g k n"],
    sizes: Int[Array, " g"],
    seed: UInt32[Array, ""],
    recipe: blockscaled.Recipe,
    dtypes: Dtypes,
    passes: tuple[Pass, ...],
) -> Float[Array, "m n"]:
    return _forward(x, w, sizes, seed, recipe, dtypes, passes)[0]


def _forward(
    x: Float[Array, "m k"],
    w: Float[Array, "g k n"],
    sizes: Int[Array, " g"],
    seed: UInt32[Array, ""],
    recipe: blockscaled.Recipe,
    dtypes: Dtypes,
    passes: tuple[Pass, ...],
) -> tuple[Float[Array, "m n"], Saved]:
    fprop, dgrad, wgrad = (p in passes for p in PASSES)
    rows = sizes.sum()
    x_k, x_m = _twice(recipe.x, x, seed, rows=fprop, columns=wgrad, kept=rows)
    w_n, w_k = _each_twice(recipe.w, w, seed, rows=dgrad, columns=fprop)
    y = _rows(x_k, w_k, sizes, dtypes[2], stacked="columns")
    return y, (x_m, w_n, sizes, seed)


def _backward(
    recipe: blockscaled.Recipe,
    dtypes: Dtypes,
    passes: tuple[Pass, ...],
    saved: Saved,
    dy: Float[Array, "m n"],
) -> tuple[Float[Array, "m k"], Float[Array, "g k n"], None, None]:
    x_m, w_n, sizes, seed = saved
    _, dgrad, wgrad = (p in passes for p in PASSES)
    rows = sizes.sum()
    dy_n, dy_m = _twice(recipe.dy, dy, seed, rows=dgrad, columns=wgrad, kept=rows)
    w_t = w_n if isinstance(w_n, Blocks) else jnp.swapaxes(w_n, 1, 2)
    dx = _rows(dy_n, w_t, sizes, dtypes[0], stacked="rows")
    return dx, _contract(x_m, dy_m, sizes, dtypes[1]), None, None


_ragged.defvjp(_forward, _backward)


def _twice(
    quantize: Twice,
    a: Float[Array, "r c"],
    seed: UInt32[Array, ""],
    rows: bool,
    columns: bool,
    kept: Int[Array, ""] | None = None,
) -> tuple[Rows, Rows]:
    """Return a in blocks along its rows where `rows`, and a^T in blocks along its
    columns where `columns`; else a, as it is. Its rows from `kept` on quantize as 0:
    an exchange's buffer leaves them as they were, and a block of a^T would scale
    them with the groups' last rows."""
    if not rows and not columns:
        return a, a
    kept = jnp.asarray(a.shape[0]) if kept is None else kept
    along_rows, along_columns = _quantized(quantize, a, seed, kept)
    return along_rows if rows else a, along_columns if columns else a


def _quantized(
    quantize: Twice,
    a: Float[Array, "r c"],
    seed: UInt32[Array, ""],
    kept: Int[Array, ""],
) -> tuple[Blocks, Blocks]:
    """Return quantize(a), a's rows from `kept` on as 0: MXFP8 by Triton's kernel on
    a GPU, in one pass with the zeros."""
    if TRITON_QUANTIZES and quantize is blockscaled.mx_twice and _triton():
        assert grouped_triton is not None
        return grouped_triton.quantized_twice(a, kept)
    return quantize(jnp.where((jnp.arange(a.shape[0]) < kept)[:, None], a, 0), seed)


def _triton() -> bool:
    """Whether Triton's kernels run here."""
    return grouped_triton is not None and kernels.gpu()


def _each_twice(
    quantize: Twice,
    w: Float[Array, "g k n"],
    seed: UInt32[Array, ""],
    rows: bool,
    columns: bool,
) -> tuple[Matrices, Matrices]:
    """Return each W[g] in blocks along n where `rows`, stacked by rows, (g k, n); and
    each W[g]^T in blocks along k where `columns`, stacked by columns, (n, g k); else
    W, as it is. A block along k stays inside its group's, k being whole blocks."""
    g, k, n = w.shape
    along_rows, along_columns = _twice(
        quantize, w.reshape(g * k, n), seed, rows, columns
    )
    return (
        along_rows if isinstance(along_rows, Blocks) else w,
        along_columns if isinstance(along_columns, Blocks) else w,
    )


def _rows(
    a: Rows,
    b: Matrices,
    sizes: Int[Array, " g"],
    out: DTypeLike,
    stacked: Stacked,
) -> Float[Array, "m n"]:
    """Return each group's rows of a times its matrix: in the format where both are,
    b's (n, k) matrices `stacked` so, by Triton on a GPU; else b[g] (k, n), by
    ``grouped``'s kernels."""
    if not isinstance(a, Blocks):
        assert not isinstance(b, Blocks)
        return grouped.dot(a, b, sizes).astype(out)
    assert isinstance(b, Blocks)
    if _triton():
        assert grouped_triton is not None
        at = tiles_of(sizes, a.values.shape[0], grouped_triton.ROWS.m)
        y = grouped_triton.rows(a, b, at, sizes.shape[0], stacked, out)
        return jnp.where((jnp.arange(y.shape[0]) < sizes.sum())[:, None], y, 0)
    matrices = _stacked(dequantized(b), sizes.shape[0], stacked)
    return jax.lax.ragged_dot(dequantized(a), matrices, sizes).astype(out)


def _contract(
    a: Rows,
    b: Blocks | Float[Array, "m n"],
    sizes: Int[Array, " g"],
    out: DTypeLike,
) -> Float[Array, "g k n"]:
    """Return a^T b over each group's rows: in the format where both are, given as
    a^T and b^T in blocks along the rows, by Triton on a GPU; else by ``grouped``."""
    if not isinstance(a, Blocks):
        assert not isinstance(b, Blocks)
        return grouped.weight_grad(a, b, sizes).astype(out)
    assert isinstance(b, Blocks)
    if _triton():
        assert grouped_triton is not None
        return grouped_triton.contract(a, b, sizes, out)
    rows = (dequantized(a).T, dequantized(b).T)
    dw = jax.lax.ragged_dot_general(*rows, sizes, grouped.RAGGED_CONTRACTING)
    return dw.astype(out)


def dequantized(blocks: Blocks) -> Float[Array, "r c"]:
    """Return the values `blocks` hold, in float32: each times its block's scale."""
    assert blocks.scale is None  # E4M3 with power-of-two scales: RECIPES
    scales = jnp.repeat(blocks.scales.astype(jnp.float32), BLOCK, axis=-1)
    return blocks.values.astype(jnp.float32) * scales


def _stacked(
    values: Float[Array, "r c"], groups: int, stacked: Stacked
) -> Float[Array, "g k n"]:
    """Return each group's (n, k) matrix in `values`, stacked so, as `ragged_dot`
    takes it: (g, k, n)."""
    if stacked == "rows":
        return jnp.swapaxes(values.reshape(groups, -1, values.shape[-1]), 1, 2)
    return values.reshape(values.shape[0], groups, -1).transpose(1, 2, 0)


def tiles_of(sizes: Int[Array, " g"], m: int, tile: int) -> Int[Array, "t 3"]:
    """Return, for each tile of `tile` rows of m that a group has rows in, the group
    and where its rows there start and end; then (0, 0, 0), as many as make the most
    there can be: one a tile of m, and one more a group but the first."""
    g = sizes.shape[0]
    ends = jnp.cumsum(sizes)
    starts = ends - sizes
    first = starts // tile
    count = jnp.where(sizes > 0, -(-ends // tile) - first, 0)
    before = jnp.cumsum(count) - count
    at = jnp.arange(-(-m // tile) + g - 1)
    group = jnp.minimum(jnp.searchsorted(before + count, at, side="right"), g - 1)
    index = first[group] + at - before[group]
    lo = jnp.maximum(starts[group], index * tile)
    hi = jnp.minimum(ends[group], (index + 1) * tile)
    real = at < count.sum()
    spans = [group, jnp.where(real, lo, 0), jnp.where(real, hi, 0)]
    return jnp.stack(spans, -1).astype(jnp.int32)
