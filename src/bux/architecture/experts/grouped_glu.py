"""The experts' gated MLP in MXFP8 by cuDNN's fused kernels (``grouped_cudnn``) on a
Blackwell GPU: SwiGLU, each row's weight and both quantizations of the hidden units in
the epilogue of up's GEMM, and their gradient in that of down's dgrad; x, dy and the
weights quantized by Triton (``grouped_triton``). A row of ``moe.experts`` goes

    forward   x, W_up          swiglu    h, and the product x W_up, kept
              h, W_down        rows      y, summed into its token
    backward  dy, W_down       dswiglu   the product's gradient dp, and the weight's
              dp, W_up         rows      dx, summed into its token
              h, dy and x, dp  contract  dW_down and dW_up

The kernels take each expert's rows from a multiple of 256 (`Padded`): the quantizer
reads each row from its token where it lies, and the sums read the rows back. Their
GLU takes each gate 32 columns before its up (``grouped_cudnn.halves_of``), where W_up
holds them side by side: the quantizer reads W_up's pairs so, and the transpose of dp
for W_up's wgrad turns them back into pairs.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Shaped

from bux import config as cfg
from bux.architecture.experts import grouped_blockscaled, grouped_cudnn
from bux.architecture.experts.grouped_cudnn import PAD
from bux.architecture.experts.grouped_mxfp8 import Tiled
from bux.architecture.precision import blockscaled
from bux.architecture.precision.blockscaled import PASSES, Pass

try:  # Triton's kernels, where the cuda extra is installed (Linux)
    from bux.architecture.experts import grouped_triton
except ImportError:
    grouped_triton = None


class Padded(NamedTuple):
    """Rows by expert, each expert's from a multiple of 256: the row of the tokens each
    reads (-1: a row of 0), where each token's k rows lie among them (past them: none),
    and where each expert's end."""

    source: Int[Array, " p"]
    picks: Int[Array, "t k"]
    ends: Int[Array, " E"]


def runs(
    recipe: blockscaled.Recipe | None,
    passes: tuple[Pass, ...],
    ffn: cfg.MLPConfig,
    up: Float[Array, "E d f p"],
) -> bool:
    """Whether the fused kernels run an MoE's experts here: on a Blackwell GPU, a SwiGLU
    FFN of shapes they take, in MXFP8 in every pass."""
    swiglu = ffn.type == "gated_mlp" and ffn.activation == "swish"
    plain = ffn.limit is None and ffn.up_softcap is None
    in_format = recipe is blockscaled.MXFP8 and set(passes) == set(PASSES)
    groups, d, f, _ = up.shape
    here = grouped_triton is not None and grouped_cudnn.runs()
    return here and swiglu and plain and in_format and grouped_cudnn.fits(d, f, groups)


def mlp(
    x: Float[Array, "m d"],
    sizes: Int[Array, " E"],
    weights: Float[Array, " m"],
    up: Float[Array, "E d f 2"],
    down: Float[Array, "E f d"],
    picked: grouped_blockscaled.Picked | None = None,
    summed: grouped_blockscaled.Summed | None = None,
) -> Float[Array, "r d"]:
    """Return what ``moe.experts`` does of a SwiGLU FFN with no norm: each row of x
    through its expert's FFN times its weight, x read from the tokens where it was
    `picked`, or the sums of the tokens it is `summed` into."""
    m = x.shape[0]
    tokens = x if picked is None else picked.tokens
    source = jnp.arange(m) if picked is None else picked.index
    picks = jnp.arange(m)[:, None] if summed is None else summed.inverse
    layout, at = padded(sizes, source, picks)
    row_weights = jnp.zeros(layout.source.shape[0], jnp.float32)
    row_weights = row_weights.at[at].set(weights.astype(jnp.float32), mode="drop")
    # The kernels take, and give, bfloat16.
    tokens, up, down = (a.astype(jnp.bfloat16) for a in (tokens, up, down))
    return _mlp(tokens, row_weights, up, down, layout).astype(x.dtype)


def padded(
    sizes: Int[Array, " E"], source: Int[Array, " m"], picks: Int[Array, "t k"]
) -> tuple[Padded, Int[Array, " m"]]:
    """Return the layout of m rows sorted by expert, the first sizes[0] expert 0's and
    so on, each row reading its token from `source`, each token's at `picks`; and where
    each row lies in it. Rows past the experts' lie nowhere."""
    m, groups = source.shape[0], sizes.shape[0]
    rows = (m + groups * (PAD - 1)) // PAD * PAD  # the most the padding can take
    whole = -(-sizes // PAD) * PAD
    ends = jnp.cumsum(whole)
    later = (
        ends - whole - (jnp.cumsum(sizes) - sizes)
    )  # how far each expert's rows move
    r = jnp.arange(m)
    moved = r + jnp.repeat(later, sizes, total_repeat_length=m)
    at = jnp.where(r < sizes.sum(), moved, rows)
    reads = jnp.full(rows, -1, jnp.int32).at[at].set(source, mode="drop")
    lying = jnp.take(at, picks, mode="fill", fill_value=rows)
    return Padded(reads, lying.astype(jnp.int32), ends.astype(jnp.int32)), at


def _sums(y: Float[Array, "p d"], picks: Int[Array, "t k"]) -> Float[Array, "t d"]:
    """Return each token's sum of its rows of y at `picks`, in float32: a pick past y
    adds 0."""
    rows = jnp.take(y, picks.reshape(-1), axis=0, mode="fill", fill_value=0)
    summed = jnp.sum(rows.reshape(*picks.shape, -1), 1, dtype=jnp.float32)
    return summed.astype(y.dtype)


# What the backward takes of the forward: x^T and the weights as their gradients'
# GEMMs contract them, the product, h down its columns, the rows' weights and where
# the rows lie.
Saved = tuple[
    Tiled,
    Tiled,
    Tiled,
    Float[Array, "p n"],
    grouped_cudnn.Columns,
    Float[Array, " p"],
    Padded,
]


@jax.custom_vjp
def _mlp(
    tokens: Float[Array, "t d"],
    weights: Float[Array, " p"],
    up: Float[Array, "E d f 2"],
    down: Float[Array, "E f d"],
    layout: Padded,
) -> Float[Array, "t d"]:
    return _forward(tokens, weights, up, down, layout)[0]


def _forward(
    tokens: Float[Array, "t d"],
    weights: Float[Array, " p"],
    up: Float[Array, "E d f 2"],
    down: Float[Array, "E f d"],
    layout: Padded,
) -> tuple[Float[Array, "t d"], Saved]:
    assert grouped_triton is not None
    groups, d, f, _ = up.shape
    rows = jnp.int32(layout.source.shape[0])
    x, x_t = grouped_triton.quantized_twice(tokens, rows, index=layout.source)
    in_pairs = up.reshape(groups, d, 2 * f)
    up_dgrad, up_fprop = grouped_triton.quantized_twice(
        in_pairs, jnp.int32(d), halves=True
    )
    down_dgrad, down_fprop = grouped_triton.quantized_twice(down, jnp.int32(f))
    assert x_t is not None and up_fprop is not None and down_fprop is not None
    glu = grouped_cudnn.swiglu(x, up_fprop, layout.ends, weights)
    y = grouped_cudnn.rows(glu.h, down_fprop, layout.ends)
    saved = (x_t, up_dgrad, down_dgrad, glu.product, glu.h_columns, weights, layout)
    return _sums(y, layout.picks), saved


def _backward(
    saved: Saved, dy: Float[Array, "t d"]
) -> tuple[
    Float[Array, "t d"],
    Float[Array, " p"],
    Float[Array, "E d f 2"],
    Float[Array, "E f d"],
    None,
]:
    assert grouped_triton is not None
    x_t, up_dgrad, down_dgrad, product, h_columns, weights, layout = saved
    rows = jnp.int32(layout.source.shape[0])
    dy_rows, dy_t = grouped_triton.quantized_twice(dy, rows, index=layout.source)
    assert dy_t is not None
    grad = grouped_cudnn.dswiglu(dy_rows, down_dgrad, product, layout.ends, weights)
    dx = grouped_cudnn.rows(grad.product, up_dgrad, layout.ends)
    d_down = grouped_cudnn.contract(h_columns, dy_t, layout.ends)
    d_up = grouped_cudnn.contract(x_t, _in_pairs(grad.product_columns), layout.ends)
    groups, d, f2 = d_up.shape
    d_up = d_up.reshape(groups, d, f2 // 2, 2)
    return _sums(dx, layout.picks), grad.weights, d_up, d_down, None


_mlp.defvjp(_forward, _backward)


def _in_pairs(product: grouped_cudnn.Columns) -> Tiled:
    """Return the product's gradient down its columns, (p, 2f) in halves, as its
    transpose (2f, p) in pairs, for W_up's wgrad: its bytes by Triton, its scales'
    rows by XLA."""
    assert grouped_triton is not None
    values = grouped_triton.transposed(product.values, pairs=True)
    return Tiled(values, _scales_in_pairs(product.scales))


def _scales_in_pairs(
    scales: Shaped[Array, "p q 32 16"],
) -> Shaped[Array, "p q 32 16"]:
    """Return tiles of scales of rows in halves with their rows in pairs: each tile of
    128 rows by 4 scales, row i's at bytes (i % 32) * 16 + (i // 32) * 4."""
    p, q = scales.shape[:2]
    by_row = (
        scales.reshape(p, q, 32, 4, 4).transpose(0, 1, 3, 2, 4).reshape(p, q, 128, 4)
    )
    paired = by_row[:, :, grouped_cudnn.halves_of(jnp.arange(128))]
    return paired.reshape(p, q, 4, 32, 4).transpose(0, 1, 3, 2, 4).reshape(p, q, 32, 16)
