"""The experts' MLP in MXFP8 by cuDNN's fused kernels
(`~bux.architecture.experts.grouped_cudnn`) on a Blackwell GPU: a ReLU^2 MLP, or a
gated one of SwiGLU, a clamped or offset GeGLU, or SiTU-GLU (`epilogue`).

The activation, each row's weight and both quantizations of the hidden units run in
the epilogue of up's GEMM, and their gradient in that of down's dgrad; Triton
quantizes x, dy and the weights (`~bux.architecture.experts.grouped_triton`). A row of
`bux.architecture.experts.moe.experts` goes through these kernels::

    forward   x, W_up          activated        h, and the product x W_up, kept
              h, W_down        rows             y, summed into its token
    backward  dy, W_down       activation_grad  the product's gradient dp, and the
                                                weight's
              dp, W_up         rows             dx, summed into its token
              h, dy and x, dp  contract         dW_down and dW_up

The kernels take each expert's rows from a multiple of 256 (`Padded`): the quantizer
reads each row from its token where it lies, and the sums read the rows back. Rows
that sum into no token, as an exchange's, come out as they lay, each expert's after
the last's, by bux's own GEMM (`bux.architecture.experts.grouped_mxfp8.rows`), which
packs them so. A GLU takes each gate 32 columns before its up
(`~bux.architecture.experts.grouped_cudnn.halves_of`), where W_up holds them side by
side: the quantizer reads W_up's pairs so, and the transpose of dp for W_up's wgrad
turns them back into pairs. The kernels take the hidden units in whole tiles of 128,
as the quantizer pads what down contracts: it pads the weights with units of 0
(Nemotron 3 Nano's 1856 run as 1920), and the wgrads write the real units' gradients
alone.

Where FSDP splits the experts' kernels by expert, `gathered` quantizes each device's
shard of them before it gathers them, in MXFP8 (`Weights`): each device does its share
of the work, and the gathers move about half of bfloat16's bytes.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Shaped

from bux import config as cfg
from bux.architecture.experts import grouped_blockscaled, grouped_cudnn, grouped_mxfp8
from bux.architecture.experts.grouped_cudnn import PAD, SRELU, SWIGLU, Epilogue
from bux.architecture.experts.grouped_mxfp8 import Tiled
from bux.architecture.precision import blockscaled
from bux.architecture.precision.blockscaled import PASSES, Pass

try:  # Triton's kernels, where the cuda extra is installed (Linux)
    from bux.architecture.experts import grouped_triton
except ImportError:
    grouped_triton = None

# A gated FFN's activation of its gate, as GeGLU's form scales it inside its sigmoid.
GATE_SCALES = {"swish": 1.0, "quick_gelu": 1.702}


class Padded(NamedTuple):
    """Rows by expert, each expert's from a multiple of 256, as the kernels take
    them."""

    #: The row of the tokens each row reads; -1 for a row of 0.
    source: Int[Array, " p"]
    #: Where each token's k rows lie among them, past them for none; None when the
    #: rows are the output.
    picks: Int[Array, "t k"] | None
    #: Where each expert's rows end.
    ends: Int[Array, " E"]
    #: How many rows each expert has.
    sizes: Int[Array, " E"]

    @property
    def starts(self) -> Int[Array, " E"]:
        """Where each expert's rows start."""
        return jnp.concatenate([jnp.zeros(1, self.ends.dtype), self.ends[:-1]])


class Weights(NamedTuple):
    """The experts' kernels in MXFP8 as the GEMMs take them, each in blocks along what
    its GEMM contracts, their hidden units padded (`in_mxfp8`); a GLU's pairs of
    up's columns as its halves."""

    #: W_up, in blocks along what its dgrad contracts.
    up_dgrad: Tiled
    #: W_up, in blocks along what its forward contracts.
    up_fprop: Tiled
    #: W_down, in blocks along what its dgrad contracts.
    down_dgrad: Tiled
    #: W_down, in blocks along what its forward contracts.
    down_fprop: Tiled


def runs(
    recipe: blockscaled.Recipe | None,
    passes: tuple[Pass, ...],
    ffn: cfg.MLPConfig,
    up: Float[Array, "E d f q"],
) -> bool:
    """Return whether the fused kernels run an MoE's experts here.

    They do on a Blackwell GPU with cudnn-frontend and Triton, for an FFN whose
    activation they fuse (`epilogue`) and which has no bias, its width whole blocks
    of 64 and its hidden units of 32, of at most 1024 experts, in MXFP8 in every
    pass.

    Args:
        recipe: the experts' block-scaled format; None for compute.
        passes: the GEMMs that multiply in the format.
        ffn: the experts' FFN settings.
        up: the experts' up kernels, (E, d, f, q).
    """
    fused = epilogue(ffn) is not None and not ffn.bias
    in_format = recipe is blockscaled.MXFP8 and set(passes) == set(PASSES)
    groups, d, f, _ = up.shape
    fits = grouped_cudnn.fits(d, f, groups)
    here = grouped_triton is not None and grouped_cudnn.runs()
    return here and fused and in_format and fits


def epilogue(ffn: cfg.MLPConfig) -> Epilogue | None:
    """Return the activation of an FFN as cuDNN's kernels fuse it: relu2 of an
    ``mlp``; SwiGLU, GeGLU's form of a gate by swish or quick_gelu with a `limit` or
    an `up_offset`, or SiTU-GLU's of situ with an `up_softcap`, of a ``gated_mlp``.
    None for any other."""
    if ffn.type == "mlp":
        plain = ffn.limit is None and ffn.up_softcap is None
        return SRELU if ffn.activation == "relu2" and plain else None
    if ffn.activation == "situ":
        capped = ffn.up_softcap is not None and ffn.limit is None and not ffn.up_offset
        return Epilogue("situglu", cap=ffn.up_softcap) if capped else None
    alpha = GATE_SCALES.get(ffn.activation)
    if alpha is None or ffn.up_softcap is not None:
        return None
    if alpha == 1.0 and ffn.limit is None and not ffn.up_offset:
        return SWIGLU
    limit = grouped_cudnn.UNCLAMPED if ffn.limit is None else ffn.limit
    return Epilogue("geglu", alpha=alpha, limit=limit, offset=ffn.up_offset)


def mlp(
    x: Float[Array, "m d"],
    sizes: Int[Array, " E"],
    weights: Float[Array, " m"],
    up: Float[Array, "E d f q"],
    down: Float[Array, "E f d"],
    ffn: cfg.MLPConfig,
    picked: grouped_blockscaled.Picked | None = None,
    summed: grouped_blockscaled.Summed | None = None,
    quantized: Weights | None = None,
) -> Float[Array, "r d"]:
    """Return what `bux.architecture.experts.moe.experts` does of an FFN with no norm
    whose activation the kernels fuse: each row of x through its expert's FFN, times
    its weight.

    Args:
        x: the rows, m of d, sorted by expert.
        sizes: how many rows each expert has.
        weights: each row's weight.
        up: the experts' up kernels; a GLU's each hidden unit's up and gate side by
            side, q = 2.
        down: their down kernels.
        ffn: the experts' FFN settings (`epilogue`).
        picked: where the rows were gathered from, for the quantizer to read them
            there; None to read `x`.
        summed: the tokens the rows sum into; None to return the rows.
        quantized: the kernels in MXFP8, for the GEMMs to take as they are, `up`
            and `down` then taking only their gradients; None to quantize them here.

    Returns:
        The rows' outputs, (m, d), or with `summed` the tokens' sums, (t, d).
    """
    activation = epilogue(ffn)
    assert activation is not None
    m = x.shape[0]
    tokens = x if picked is None else picked.tokens
    source = jnp.arange(m) if picked is None else picked.index
    picks = None if summed is None else summed.inverse
    layout, at = padded(sizes, source, picks)
    row_weights = jnp.zeros(layout.source.shape[0], jnp.float32)
    row_weights = row_weights.at[at].set(weights.astype(jnp.float32), mode="drop")
    # The kernels take, and give, bfloat16.
    tokens, up, down = (a.astype(jnp.bfloat16) for a in (tokens, up, down))
    units = up.shape[2]
    y = _mlp(tokens, row_weights, up, down, layout, quantized, activation, units)
    return y.astype(x.dtype)


def in_mxfp8(up: Float[Array, "E d f q"], down: Float[Array, "E f d"]) -> Weights:
    """Return the experts' kernels in MXFP8 as the GEMMs take them, by Triton: their
    hidden units padded by units of 0 to whole tiles of 128
    (`~bux.architecture.experts.grouped_cudnn.hidden_tiles`), whose activation is 0
    and which add nothing to down's product."""
    assert grouped_triton is not None
    groups, d, f, q = up.shape
    n = grouped_cudnn.hidden_tiles(f)
    in_pairs = up.astype(jnp.bfloat16).reshape(groups, d, f * q)
    up_dgrad, up_fprop = grouped_triton.quantized_twice(
        in_pairs, jnp.int32(d), halves=q == 2, padded=(d, n * q)
    )
    down_dgrad, down_fprop = grouped_triton.quantized_twice(
        down.astype(jnp.bfloat16), jnp.int32(f), padded=(n, d)
    )
    assert up_fprop is not None and down_fprop is not None
    return Weights(up_dgrad, up_fprop, down_dgrad, down_fprop)


def gathered(
    up: Float[Array, "e d f q"], down: Float[Array, "e f d"], axes: tuple[str, ...]
) -> tuple[Float[Array, "E d f q"], Float[Array, "E f d"], Weights]:
    """Return the experts' kernels gathered whole from each device's shard of them,
    both as they are and in MXFP8, each device quantizing its own shard.

    The kernels gathered as they are carry only their gradients, which
    reduce-scatter back into shards: XLA drops their forward.

    Args:
        up: a device's shard of the up kernels, its experts', in a shard_map manual
            over `axes`.
        down: its shard of the down kernels.
        axes: the mesh axes FSDP splits the experts over.

    Returns:
        The whole up and down kernels, and them in MXFP8 (`Weights`).
    """
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
    """Return the padded layout of m rows sorted by expert, and where each row lies
    in it.

    Args:
        sizes: how many rows each expert has: the first ``sizes[0]`` are expert 0's,
            and so on.
        source: the token each row reads.
        picks: where each token's k rows are among the m, if they sum into tokens;
            None if not.

    Returns:
        The layout, and where each row lies in it; rows past the experts' lie
        nowhere.
    """
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


@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _mlp(
    tokens: Float[Array, "t d"],
    weights: Float[Array, " p"],
    up: Float[Array, "E d f q"],
    down: Float[Array, "E f d"],
    layout: Padded,
    given: Weights | None,
    activation: Epilogue,
    units: int,
) -> Float[Array, "t d"]:
    """Return `mlp`'s output in bfloat16 by the kernels, forward and backward: of
    experts of `units` hidden units."""
    return _forward(tokens, weights, up, down, layout, given, activation, units)[0]


def _forward(
    tokens: Float[Array, "t d"],
    weights: Float[Array, " p"],
    up: Float[Array, "E d f q"],
    down: Float[Array, "E f d"],
    layout: Padded,
    given: Weights | None,
    activation: Epilogue,
    units: int,
) -> tuple[Float[Array, "t d"], Saved]:
    """Return the forward's output: the activation, then down's rows; and what the
    backward keeps."""
    assert grouped_triton is not None
    rows = jnp.int32(layout.source.shape[0])
    x, x_t = grouped_triton.quantized_twice(tokens, rows, index=layout.source)
    assert x_t is not None
    w = in_mxfp8(up, down) if given is None else given
    ends = layout.ends
    hidden = grouped_cudnn.activated(x, w.up_fprop, ends, weights, activation)
    y = _outputs(hidden.h, w.down_fprop, layout, tokens.shape[0])
    product, h_columns = hidden.product, hidden.h_columns
    saved = (x_t, w.up_dgrad, w.down_dgrad, product, h_columns, weights, layout)
    return y, saved


def _backward(
    activation: Epilogue, units: int, saved: Saved, dy: Float[Array, "t d"]
) -> tuple[
    Float[Array, "t d"],
    Float[Array, " p"],
    Float[Array, "E d f q"],
    Float[Array, "E f d"],
    None,
    None,
]:
    """Return the gradients of the tokens, the rows' weights and both kernels: by
    the activation's gradient, up's rows, and two contractions, which write the
    kernels' real units alone."""
    assert grouped_triton is not None
    x_t, up_dgrad, down_dgrad, product, h_columns, weights, layout = saved
    rows = jnp.int32(layout.source.shape[0])
    dy_rows, dy_t = grouped_triton.quantized_twice(dy, rows, index=layout.source)
    assert dy_t is not None
    ends = layout.ends
    grad = grouped_cudnn.activation_grad(
        dy_rows, down_dgrad, product, ends, weights, activation
    )
    dx = _outputs(grad.product, up_dgrad, layout, dy.shape[0])
    d = dy.shape[1]
    d_down = grouped_cudnn.contract(h_columns, dy_t, ends, shape=(units, d))
    q = 2 if activation.gated else 1
    by_unit = _transposed(grad.product_columns, activation.gated)
    d_up = grouped_cudnn.contract(x_t, by_unit, ends, shape=(d, units * q))
    d_up = d_up.reshape(*d_up.shape[:2], units, q)
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


def _transposed(product: grouped_cudnn.Columns, gated: bool) -> Tiled:
    """Return the product's gradient down its columns, (p, n), as its transpose
    (n, p), for W_up's wgrad: a GLU's halves in pairs again. Its bytes by Triton, its
    scales' rows by XLA."""
    assert grouped_triton is not None
    values = grouped_triton.transposed(product.values, pairs=gated)
    scales = _scales_in_pairs(product.scales) if gated else product.scales
    return Tiled(values, scales)


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
