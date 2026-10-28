"""The layers of a vision encoder that reads images as patches: ``patches``, which
embeds them, and ``merge_patches``, which joins each block of them into one.

An encoder is the body of ``image(...)`` (`bux.architecture.image`), which runs it on
the forward pass's images (`bux.architecture.registry.Images`). Its stream is the
patches of the images in a row, one after another, each image's patches in the order
`bux.export.hf.pixels.patches` cuts them: a block's patches one after another, so that
``merge_patches`` joins neighbours.
"""

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int, UInt8

from bux import config as cfg
from bux.architecture.common.sharding import projection
from bux.architecture.registry import Context, Taps, register
from bux.lang import ir


class Patches(nnx.Module):
    """A patch embedding, ``patches``: each patch's pixels, normalized per channel,
    projected to `features` channels, plus a learned position embedding of its row
    and column (Qwen2-VL, Qwen3.5).

    The learned positions are a ``grid`` x ``grid`` table, bilinearly resized to each
    image's grid of patches, corners on corners.

    Args:
        c: Its settings (`bux.config.PatchesConfig`).
        rngs: The random streams its weights start from.
    """

    def __init__(self, c: cfg.PatchesConfig, rngs: nnx.Rngs):
        n = 3 * c.size * c.size
        self.proj = projection(n, c.features, use_bias=True, rngs=rngs)
        self.positions = (
            nnx.Embed(c.grid * c.grid, c.features, rngs=rngs) if c.grid else None
        )
        self.grid = c.grid
        # Per input channel, repeated over each patch's size * size pixels.
        self.mean = tuple(m for m in c.mean for _ in range(c.size * c.size))
        self.std = tuple(s for s in c.std for _ in range(c.size * c.size))

    def __call__(self, x: UInt8[Array, "b n p"], taps: Taps) -> Float[Array, "b n d"]:
        """Return the embeddings of the patches x, each 3 * size * size bytes."""
        mean, std = jnp.asarray(self.mean), jnp.asarray(self.std)
        # Note: in the weights' dtype, the run's compute, so that the encoder after
        # computes in it too, and its attention runs by the flash kernel.
        normed = (x.astype(jnp.float32) / 255 - mean) / std
        y = self.proj(normed.astype(self.proj.kernel[...].dtype))
        if self.positions is None or self.grid is None:
            return y
        images = taps.images
        assert images is not None, "patches reads the images of image(...)"
        table = self.positions.embedding[...]
        at = resized_positions(table, self.grid, images.grid, images.sizes)
        return y + at.astype(y.dtype)


def resized_positions(
    table: Float[Array, "gg d"],
    g: int,
    grid: Int[Array, "b n 2"],
    sizes: Int[Array, "b n 2"],
) -> Float[Array, "b n d"]:
    """Return each patch's learned position: the g x g `table` resized bilinearly to
    its image's grid of `sizes`, corners on corners, as ``F.interpolate`` with
    ``align_corners=True`` resizes it."""
    src = grid * (g - 1) / jnp.maximum(sizes - 1, 1)
    lo = jnp.floor(src).astype(jnp.int32)
    hi = jnp.minimum(lo + 1, g - 1)
    w = src - lo

    def at(r: Int[Array, "b n"], c: Int[Array, "b n"]) -> Float[Array, "b n d"]:
        return table[r * g + c]

    (r0, c0), (r1, c1), (wr, wc) = ((a[..., 0], a[..., 1]) for a in (lo, hi, w))
    return (
        at(r0, c0) * ((1 - wr) * (1 - wc))[..., None]
        + at(r0, c1) * ((1 - wr) * wc)[..., None]
        + at(r1, c0) * (wr * (1 - wc))[..., None]
        + at(r1, c1) * (wr * wc)[..., None]
    )


@register("patches")
def _(
    c: cfg.PatchesConfig, width: str, ports: tuple[ir.Port, ...], ctx: Context
) -> Patches:
    return Patches(c, ctx.rngs)


class MergePatches(nnx.Module):
    """``merge_patches``: each run of ``size * size`` patches, a block of the image,
    joined into one: their channels side by side (Qwen2-VL's merger), or their mean.

    Args:
        c: Its settings (`bux.config.MergePatchesConfig`).
    """

    def __init__(self, c: cfg.MergePatchesConfig):
        self.n, self.pool = c.size * c.size, c.pool

    def __call__(self, x: Float[Array, "b n d"], taps: Taps) -> Float[Array, "b m e"]:
        """Return the n / size² blocks of the patches x: each size² * d wide, or the
        mean of its patches, d wide."""
        b, n, d = x.shape
        if self.pool == "mean":
            return x.reshape(b, n // self.n, self.n, d).mean(2)
        return x.reshape(b, n // self.n, self.n * d)


@register("merge_patches")
def _(
    c: cfg.MergePatchesConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> MergePatches:
    return MergePatches(c)
