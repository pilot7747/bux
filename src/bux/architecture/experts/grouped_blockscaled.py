"""Grouped block-scaled GEMMs: an MoE's experts, ``y = x W[g]`` for the rows of each
group g as `bux.architecture.experts.grouped.ragged_dot` runs them, and its gradients,
each operand quantized along what its GEMM contracts, as a
`~bux.architecture.precision.blockscaled.Recipe` says::

    fprop   y  = x W[g]      x along k, each W[g] along k
    dgrad   dx = dy W[g]^T   dy along n, each W[g] along n
    wgrad   dW = x^T dy      x and dy along the rows, in blocks of 32 whatever the groups

A block of rows that two groups share has one scale, and each group counts the other's
rows as 0; rows past the groups come out 0, and add nothing to dW. Only the GEMMs in
`passes` multiply in the format, the others by `~bux.architecture.experts.grouped`'s
kernels, as they are. In the format, ``_times`` and ``_contract`` multiply by
tcgen05's block-scaled MMA on a Blackwell GPU
(`~bux.architecture.experts.grouped_mxfp8`), MXFP8's operands quantized straight into
its layout by Triton's kernel (`~bux.architecture.experts.grouped_triton`); and
elsewhere by XLA, which dequantizes and computes the same values. Each W[g] quantizes
on its own. The recipes of E4M3 values with a power-of-two scale per 32 run here
(``RECIPES``).

An MoE's rows are its tokens' rows, each as many times as it has picks: given where
they come from (`Picked`), Triton's quantizer reads them there, so that the rows
themselves are never written; given the tokens they sum into (`Summed`), y is the
tokens' sums, and the backward reads each row's dy from its token's.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int, Shaped, UInt32

from bux.architecture.experts import grouped, grouped_mxfp8
from bux.architecture.precision import blockscaled
from bux.architecture.precision.blockscaled import PASSES, Blocks, Pass, Twice

try:  # Triton's kernels, where the cuda extra is installed (Linux)
    from bux.architecture.experts import grouped_triton
except ImportError:
    grouped_triton = None

BLOCK = blockscaled.MX_BLOCK
# The formats of ``matmuls`` that run here, by their recipes: train.config.GROUPED.
RECIPES = {"mxfp8": blockscaled.MXFP8, "mxfp8_2d": blockscaled.MXFP8_2D}
# An operand of a GEMM, a matrix or each group's: in the format, as the kernels take
# it or as Blocks, or as it is where its pass multiplies as it is.
Formatted = grouped_mxfp8.Tiled | Blocks
Operand = Formatted | Float[Array, "..."]


class Picked(NamedTuple):
    """Where a GEMM's rows x come from: ``x = tokens[index]``, as
    `bux.architecture.experts.moe.token_rows` gathers an MoE's picks. Where Triton's
    quantizer runs, it reads them from `tokens`, and XLA drops the gather that made x,
    which nothing else reads."""

    #: The tokens the rows are gathered from.
    tokens: Float[Array, "n k"]
    #: The token of each row.
    index: Int[Array, " m"]


class Summed(NamedTuple):
    """Where a GEMM's rows y go: into their tokens' sums, as
    `bux.architecture.experts.moe.token_sums` sums an MoE's picks. Its transpose
    gathers each row's dy from its token's, which Triton's quantizer reads in place,
    as `Picked` rows."""

    #: The pick of each row, row i's ``order[i]``, k picks a token.
    order: Int[Array, " m"]
    #: Where each token's k picks lie among the rows.
    inverse: Int[Array, "t k"]

    @property
    def row_tokens(self) -> Int[Array, " m"]:
        """Each row's token."""
        return self.order // self.inverse.shape[1]


# What the backward takes of the forward: x and W, quantized as their gradients'
# GEMMs contract them, and where y's rows were summed.
Saved = tuple[Operand, Operand, Int[Array, " g"], UInt32[Array, ""], Summed | None]


def fits(x: Float[Array, "m k"], w: Float[Array, "g k n"]) -> bool:
    """Return whether what ``x W`` contracts, and its columns, are whole blocks."""
    return x.shape[-1] % BLOCK == 0 and w.shape[-1] % BLOCK == 0


def ragged_dot(
    x: Float[Array, "m k"],
    w: Float[Array, "g k n"],
    sizes: Int[Array, " g"],
    recipe: blockscaled.Recipe,
    seed: UInt32[Array, ""],
    passes: tuple[Pass, ...] = PASSES,
    picked: Picked | None = None,
    summed: Summed | None = None,
) -> Float[Array, "r n"]:
    """Return what `bux.architecture.experts.grouped.ragged_dot` does, its GEMMs in
    `passes` as `recipe` quantizes them.

    Rows go in the kernels' tiles of 128 too, so x gains rows of 0 up to them.

    Args:
        x: the rows, m of k, sorted by group; whole blocks of k (`fits`).
        w: the groups' matrices, (g, k, n).
        sizes: how many rows each group has.
        recipe: how the format quantizes x, W and dy.
        seed: what a format that rounds stochastically draws its noise from.
        passes: the GEMMs that multiply in the format; the others multiply as they
            are.
        picked: where the rows were gathered from, for Triton's quantizer to read
            them there; None to read `x`.
        summed: the tokens the rows sum into; None to return the rows.

    Returns:
        The products, (m, n), or with `summed` the tokens' sums, (t, n).
    """
    m, tile = x.shape[0], grouped_mxfp8.LANES
    whole = -(-m // tile) * tile
    x = _padded(x, whole)
    if picked is not None:
        picked = Picked(picked.tokens, _padded(picked.index, whole))
    if summed is not None:
        summed = Summed(_padded(summed.order, whole), summed.inverse)
    dtypes = blockscaled.Dtypes(x.dtype, w.dtype, jnp.result_type(x, w))
    y = _ragged(x, w, sizes, seed, picked, summed, recipe, dtypes, passes)
    return y if summed is not None else y[:m]


def _padded(a: Shaped[Array, "m ..."], rows: int) -> Shaped[Array, "r ..."]:
    """Return `a` with rows of 0 after its own, up to `rows`."""
    if a.shape[0] == rows:
        return a
    return jnp.pad(a, [(0, rows - a.shape[0])] + [(0, 0)] * (a.ndim - 1))


def _token_sums(
    y: Float[Array, "m n"], inverse: Int[Array, "t k"]
) -> Float[Array, "t n"]:
    """Return each token's sum of its k rows of y, at `inverse`, in float32: a row
    past y's adds 0."""
    rows = jnp.take(y, inverse.reshape(-1), axis=0, mode="fill", fill_value=0)
    summed = jnp.sum(rows.reshape(*inverse.shape, -1), 1, dtype=jnp.float32)
    return summed.astype(y.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
def _ragged(
    x: Float[Array, "m k"],
    w: Float[Array, "g k n"],
    sizes: Int[Array, " g"],
    seed: UInt32[Array, ""],
    picked: Picked | None,
    summed: Summed | None,
    recipe: blockscaled.Recipe,
    dtypes: blockscaled.Dtypes,
    passes: tuple[Pass, ...],
) -> Float[Array, "r n"]:
    """Return `ragged_dot` of rows already in whole tiles, with its own gradients."""
    return _forward(x, w, sizes, seed, picked, summed, recipe, dtypes, passes)[0]


def _forward(
    x: Float[Array, "m k"],
    w: Float[Array, "g k n"],
    sizes: Int[Array, " g"],
    seed: UInt32[Array, ""],
    picked: Picked | None,
    summed: Summed | None,
    recipe: blockscaled.Recipe,
    dtypes: blockscaled.Dtypes,
    passes: tuple[Pass, ...],
) -> tuple[Float[Array, "r n"], Saved]:
    """Return y by fprop, and x and W as the backward's GEMMs take them."""
    fprop, dgrad, wgrad = (p in passes for p in PASSES)
    filled = sizes.sum()
    x_k, x_m = _twice(
        recipe.x, x, seed, rows=fprop, columns=wgrad, filled=filled, picked=picked
    )
    w_n, w_k = _twice(recipe.w, w, seed, rows=dgrad, columns=fprop)
    y = _times(x_k, w_k, sizes, dtypes.y)
    if summed is not None:
        y = _token_sums(y, summed.inverse)
    return y, (x_m, w_n, sizes, seed, summed)


def _backward(
    recipe: blockscaled.Recipe,
    dtypes: blockscaled.Dtypes,
    passes: tuple[Pass, ...],
    saved: Saved,
    dy: Float[Array, "r n"],
) -> tuple[Float[Array, "m k"], Float[Array, "g k n"], None, None, None, None]:
    """Return dx by dgrad and dW by wgrad."""
    x_m, w_n, sizes, seed, summed = saved
    _, dgrad, wgrad = (p in passes for p in PASSES)
    filled = sizes.sum()
    picked = None
    if summed is not None:  # each row's dy is its token's, which Triton reads in place
        picked = Picked(dy, summed.row_tokens)
        dy = dy[summed.row_tokens]
    dy_n, dy_m = _twice(
        recipe.dy, dy, seed, rows=dgrad, columns=wgrad, filled=filled, picked=picked
    )
    w_t = w_n if isinstance(w_n, Formatted) else jnp.swapaxes(w_n, 1, 2)
    dx = _times(dy_n, w_t, sizes, dtypes.x)
    return dx, _contract(x_m, dy_m, sizes, dtypes.w), None, None, None, None


_ragged.defvjp(_forward, _backward)


# --- The GEMMs ---------------------------------------------------------------------


def _times(
    a: Operand, b: Operand, sizes: Int[Array, " g"], out: DTypeLike
) -> Float[Array, "m n"]:
    """Return each group's rows of a times its matrix: in the format where both are,
    each b[g] (n, k), by ``grouped_mxfp8`` on a Blackwell GPU, else by XLA; as they
    are, b[g] (k, n), by ``grouped``'s kernels."""
    if not isinstance(a, Formatted):
        assert not isinstance(b, Formatted)
        return grouped.dot(a, b, sizes).astype(out)
    assert isinstance(b, Formatted)
    if grouped_mxfp8.runs():
        return grouped_mxfp8.rows(_tiled(a), _tiled(b), sizes, out)
    each = jnp.swapaxes(_dequantized(b), 1, 2)  # (g, k, n), as ragged_dot takes it
    return jax.lax.ragged_dot(_dequantized(a), each, sizes).astype(out)


def _contract(
    a: Operand, b: Operand, sizes: Int[Array, " g"], out: DTypeLike
) -> Float[Array, "g k n"]:
    """Return a^T b over each group's rows: in the format where both are, given as a^T
    and b^T in blocks along the rows, by ``grouped_mxfp8`` on a Blackwell GPU, else by
    XLA; as they are, by ``grouped``'s kernels."""
    if not isinstance(a, Formatted):
        assert not isinstance(b, Formatted)
        return grouped.weight_grad(a, b, sizes).astype(out)
    assert isinstance(b, Formatted)
    if grouped_mxfp8.runs():
        return grouped_mxfp8.contract(_tiled(a), _tiled(b), sizes, out)
    rows = (_dequantized(a).T, _dequantized(b).T)
    dw = jax.lax.ragged_dot_general(*rows, sizes, grouped.RAGGED_CONTRACTING)
    return dw.astype(out)


def _tiled(a: Formatted) -> grouped_mxfp8.Tiled:
    """Return `a` in the format as the kernels take it."""
    if isinstance(a, grouped_mxfp8.Tiled):
        return a
    assert isinstance(a, Blocks)
    return grouped_mxfp8.tiled(a)


def _dequantized(a: Formatted) -> Float[Array, "*g r c"]:
    """Return the values that `a`, Blocks off a Blackwell GPU, holds: of each matrix
    of a stack of them."""
    assert isinstance(a, Blocks)
    if a.values.ndim == 3:
        return jax.vmap(blockscaled.dequantized)(a)
    return blockscaled.dequantized(a)


# --- Quantizing --------------------------------------------------------------------


def _twice(
    quantize: Twice,
    a: Float[Array, "*g r c"],
    seed: UInt32[Array, ""],
    rows: bool,
    columns: bool,
    filled: Int[Array, ""] | None = None,
    picked: Picked | None = None,
) -> tuple[Operand, Operand]:
    """Return a, a matrix or each of a stack of them, in blocks along its rows if
    `rows`, and a^T in blocks along its columns if `columns`; else a, as it is. Its
    rows from `filled` on quantize as 0: an exchange's buffer leaves them as they
    were, and a block of a^T would scale them with the groups' last rows."""
    if not rows and not columns:
        return a, a
    filled = jnp.asarray(a.shape[-2]) if filled is None else filled
    along_rows, along_columns = _quantized(quantize, a, seed, filled, columns, picked)
    if not columns:
        return along_rows, a
    assert along_columns is not None
    return along_rows if rows else a, along_columns


def _quantized(
    quantize: Twice,
    a: Float[Array, "*g r c"],
    seed: UInt32[Array, ""],
    filled: Int[Array, ""],
    columns: bool,
    picked: Picked | None = None,
) -> tuple[Operand, Operand | None]:
    """Return quantize(a), a's rows from `filled` on as 0: in MXFP8 on a Blackwell GPU
    by Triton's kernel, in one pass with the zeros and straight into the GEMMs'
    layout, a^T only if `columns`, a's rows read from where they were `picked`; else
    `quantize`'s Blocks, of each matrix of a stack on its own."""
    if quantize is blockscaled.mx_twice and _triton():
        assert grouped_triton is not None
        if picked is None:
            return grouped_triton.quantized_twice(a, filled, columns)
        tokens, index = picked
        return grouped_triton.quantized_twice(tokens, filled, columns, index)
    kept = jnp.where((jnp.arange(a.shape[-2]) < filled)[:, None], a, 0)
    if a.ndim == 2:
        return quantize(kept, seed)
    return jax.vmap(lambda each: quantize(each, seed))(kept)


def _triton() -> bool:
    """Whether Triton's quantizer runs here: where the kernels whose layout it
    writes do, on a Blackwell GPU."""
    return grouped_triton is not None and grouped_mxfp8.runs()
