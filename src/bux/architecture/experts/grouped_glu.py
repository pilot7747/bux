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
reads each row from its token where it lies, and the sums read the rows back. Rows
that sum into no token, as an exchange's, come out as they lay, each expert's after
the last's, by bux's own GEMM (``grouped_mxfp8.rows``), which packs them so. Their
GLU takes each gate 32 columns before its up (``grouped_cudnn.halves_of``), where W_up
holds them side by side: the quantizer reads W_up's pairs so, and the transpose of dp
for W_up's wgrad turns them back into pairs.

Where FSDP splits the experts' kernels by expert, `gathered` quantizes each device's
shard of them before it gathers them, in MXFP8 (`Weights`): a share of the work, and
the bytes of bfloat16's.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Shaped

from bux import config as cfg
from bux.architecture.experts import grouped_blockscaled, grouped_cudnn, grouped_mxfp8
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
    reads (-1: a row of 0), where each token's k rows lie among them (past them: none;
    None: the rows are the output), where each expert's end, and how many it has."""

    source: Int[Array, " p"]
    picks: Int[Array, "t k"] | None
    ends: Int[Array, " E"]
    sizes: Int[Array, " E"]

    @property
    def starts(self) -> Int[Array, " E"]:
        """Where each expert's rows start."""
        return jnp.concatenate([jnp.zeros(1, self.ends.dtype), self.ends[:-1]])


class Weights(NamedTuple):
    """The experts' kernels in MXFP8 as the GEMMs take them: up's, its pairs as halves,
    along what its dgrad and its forward contract, and down's."""

    up_dgrad: Tiled
    up_fprop: Tiled
    down_dgrad: Tiled
    down_fprop: Tiled


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
    quantized: Weights | None = None,
) -> Float[Array, "r d"]:
    """Return what ``moe.experts`` does of a SwiGLU FFN with no norm: each row of x
    through its expert's FFN times its weight, x read from the tokens where it was
    `picked`, or the sums of the tokens it is `summed` into; the kernels as they are
    `quantized`, if they are, up and down then taking only their gradients."""
    m = x.shape[0]
    tokens = x if picked is None else picked.tokens
    source = jnp.arange(m) if picked is None else picked.index
    picks = None if summed is None else summed.inverse
    layout, at = padded(sizes, source, picks)
    row_weights = jnp.zeros(layout.source.shape[0], jnp.float32)
    row_weights = row_weights.at[at].set(weights.astype(jnp.float32), mode="drop")
    # The kernels take, and give, bfloat16.
    tokens, up, down = (a.astype(jnp.bfloat16) for a in (tokens, up, down))
    return _mlp(tokens, row_weights, up, down, layout, quantized).astype(x.dtype)


def in_mxfp8(up: Float[Array, "E d f 2"], down: Float[Array, "E f d"]) -> Weights:
    """Return the experts' kernels in MXFP8 as the GEMMs take them, by Triton."""
    assert grouped_triton is not None
    groups, d, f, _ = up.shape
    in_pairs = up.astype(jnp.bfloat16).reshape(groups, d, 2 * f)
    up_dgrad, up_fprop = grouped_triton.quantized_twice(
        in_pairs, jnp.int32(d), halves=True
    )
    down_dgrad, down_fprop = grouped_triton.quantized_twice(
        down.astype(jnp.bfloat16), jnp.int32(f)
    )
    assert up_fprop is not None and down_fprop is not None
    return Weights(up_dgrad, up_fprop, down_dgrad, down_fprop)


def gathered(
    up: Float[Array, "e d f 2"], down: Float[Array, "e f d"], axes: tuple[str, ...]
) -> tuple[Float[Array, "E d f 2"], Float[Array, "E f d"], Weights]:
    """Return a device's shards of the experts' kernels, in a shard_map manual over
    `axes`, gathered whole over them; and quantized, each device its own shard, then
    gathered in MXFP8. The kernels gathered in bfloat16 carry only their gradients,
    which reduce-scatter back into shards: XLA drops their forward."""
    whole = (jax.lax.all_gather(k, axes, axis=0, tiled=True) for k in (up, down))
    shards = in_mxfp8(jax.lax.stop_gradient(up), jax.lax.stop_gradient(down))
    return *whole, jax.tree.map(partial(_bytes_gathered, axes=axes), shards)


def _bytes_gathered(
    a: Shaped[Array, "e ..."], axes: tuple[str, ...]
) -> Shaped[Array, "E ..."]:
    """Return a stack of E4M3 values or E8M0 scales gathered whole over `axes`, as
    bytes: the collectives take no 8-bit float."""
    raw = jax.lax.bitcast_convert_type(a, jnp.uint8)
    whole = jax.lax.all_gather(raw, axes, axis=0, tiled=True)
    return jax.lax.bitcast_convert_type(whole, a.dtype)


def padded(
    sizes: Int[Array, " E"],
    source: Int[Array, " m"],
    picks: Int[Array, "t k"] | None = None,
) -> tuple[Padded, Int[Array, " m"]]:
    """Return the layout of m rows sorted by expert, the first sizes[0] expert 0's and
    so on, each row reading its token from `source`, each token's at `picks`, if they
    sum into tokens; and where each row lies in it. Rows past the experts' lie
    nowhere."""
    m, groups = source.shape[0], sizes.shape[0]
    rows = (m + groups * (PAD - 1)) // PAD * PAD  # the most the padding can take
    whole = -(-sizes // PAD) * PAD
    ends = jnp.cumsum(whole)
    # How far each expert's rows move, from following each other to their padding.
    later = ends - whole - (jnp.cumsum(sizes) - sizes)
    r = jnp.arange(m)
    moved = r + jnp.repeat(later, sizes, total_repeat_length=m)
    at = jnp.where(r < sizes.sum(), moved, rows)
    reads = jnp.full(rows, -1, jnp.int32).at[at].set(source, mode="drop")
    lying = None
    if picks is not None:
        lying = jnp.take(at, picks, mode="fill", fill_value=rows).astype(jnp.int32)
    int32 = jnp.int32
    return Padded(reads, lying, ends.astype(int32), sizes.astype(int32)), at


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
    given: Weights | None,
) -> Float[Array, "t d"]:
    return _forward(tokens, weights, up, down, layout, given)[0]


def _forward(
    tokens: Float[Array, "t d"],
    weights: Float[Array, " p"],
    up: Float[Array, "E d f 2"],
    down: Float[Array, "E f d"],
    layout: Padded,
    given: Weights | None,
) -> tuple[Float[Array, "t d"], Saved]:
    assert grouped_triton is not None
    rows = jnp.int32(layout.source.shape[0])
    x, x_t = grouped_triton.quantized_twice(tokens, rows, index=layout.source)
    assert x_t is not None
    w = in_mxfp8(up, down) if given is None else given
    glu = grouped_cudnn.swiglu(x, w.up_fprop, layout.ends, weights)
    y = _outputs(glu.h, w.down_fprop, layout, tokens.shape[0])
    saved = (x_t, w.up_dgrad, w.down_dgrad, glu.product, glu.h_columns, weights, layout)
    return y, saved


def _backward(
    saved: Saved, dy: Float[Array, "t d"]
) -> tuple[
    Float[Array, "t d"],
    Float[Array, " p"],
    Float[Array, "E d f 2"],
    Float[Array, "E f d"],
    None,
    None,
]:
    assert grouped_triton is not None
    x_t, up_dgrad, down_dgrad, product, h_columns, weights, layout = saved
    rows = jnp.int32(layout.source.shape[0])
    dy_rows, dy_t = grouped_triton.quantized_twice(dy, rows, index=layout.source)
    assert dy_t is not None
    grad = grouped_cudnn.dswiglu(dy_rows, down_dgrad, product, layout.ends, weights)
    dx = _outputs(grad.product, up_dgrad, layout, dy.shape[0])
    d_down = grouped_cudnn.contract(h_columns, dy_t, layout.ends)
    d_up = grouped_cudnn.contract(x_t, _in_pairs(grad.product_columns), layout.ends)
    groups, d, f2 = d_up.shape
    d_up = d_up.reshape(groups, d, f2 // 2, 2)
    return dx, grad.weights, d_up, d_down, None, None


_mlp.defvjp(_forward, _backward)


def _outputs(a: Tiled, b: Tiled, layout: Padded, height: int) -> Float[Array, "t n"]:
    """Return each row's a b[g]^T, summed into its token, or as the rows lay: the
    `height` of them, each expert's after the last's."""
    if layout.picks is None:
        starts, sizes = layout.starts, layout.sizes
        return grouped_mxfp8.rows(
            a, b, sizes, jnp.bfloat16, starts=starts, height=height
        )
    return _sums(grouped_cudnn.rows(a, b, layout.ends), layout.picks)


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
