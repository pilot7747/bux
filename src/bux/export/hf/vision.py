"""Images, exported: bux's vision encoder layers (``patches``, ``merge_patches``),
``image(...)``'s splice of the encoder's outputs into the stream, and the
preprocessing that makes the images a model reads.

A model with ``image(...)`` takes `Images` beside its token ids, in bux's layout
(`bux.architecture.registry.Images`). `inputs` makes them from a prompt and its
images: each image resized and cut into patches as bux's data does, its one image
token repeated once for each block of patches, and the M-RoPE positions of the
tokens. The encoder runs on the first pass of a generation, where the images are;
later passes go on from the M-RoPE positions it reached (`rope_positions`).

The preprocessing is `pixels`', numpy and PIL alone, which bux's data reads images
with too (`bux.data.images`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import nn

from .common import Ctx, Site
from .pixels import mrope, patches, resize, resized


class Images(NamedTuple):
    """The images of a batch, as an exported model takes them: bux's layout
    (`bux.architecture.registry.Images`), each row's patches one image after
    another, then padding to n."""

    pixels: torch.Tensor  #: b n p: each patch's bytes.
    positions: torch.Tensor  #: b n: each patch's index in its image.
    grid: torch.Tensor  #: b n 2: each patch's row and column.
    sizes: torch.Tensor  #: b n 2: the rows and columns of its image.
    rope: torch.Tensor  #: b t 3: each token's M-RoPE position.


def inputs(
    ids: Sequence[int],
    pictures: Sequence[np.ndarray],
    token: int,
    cut: dict,
) -> tuple[torch.Tensor, Images]:
    """Return one prompt's token ids and images, as an exported model takes them.

    Args:
        ids: The prompt's token ids, with one image token where each image goes.
        pictures: The images, h x w x 3 bytes each, in order.
        token: The image token: the model's ``image(token=...)``.
        cut: How the images are cut: the model config's ``bux_images``, as
            ``images(...)`` of the data it trained on says: ``size``, ``merge``,
            ``min_pixels``, ``max_pixels``.

    Returns:
        The token ids, each image's token repeated once for each of its blocks, a
        row of them; and the images.
    """
    size, merge = cut["size"], cut["merge"]
    pixels, index, grid, sizes, blocks = [], [], [], [], []
    for image in pictures:
        h, w = resized(
            *image.shape[:2],
            size=size,
            merge=merge,
            min_pixels=cut["min_pixels"],
            max_pixels=cut["max_pixels"],
        )
        x, at = patches(resize(image, h, w), size=size, merge=merge)
        pixels.append(x)
        index.append(np.arange(len(x)))
        grid.append(at)
        sizes.append(np.tile([h // size, w // size], (len(x), 1)))
        blocks.append((h // size // merge, w // size // merge))
    found = [i for i, t in enumerate(ids) if t == token]
    if len(found) != len(pictures):
        raise ValueError(f"{len(pictures)} images, but {len(found)} image tokens")
    counts = np.ones(len(ids), np.int64)
    counts[found] = [r * c for r, c in blocks]
    row = np.repeat(np.asarray(ids), counts)
    rope = mrope(row == token, blocks)
    longs = (np.concatenate(a) for a in (index, grid, sizes))
    images = Images(
        torch.from_numpy(np.concatenate(pixels))[None],
        *(torch.from_numpy(np.asarray(a, np.int64))[None] for a in (*longs, rope)),
    )
    return torch.from_numpy(row)[None], images


# --- The layers -----------------------------------------------------------------------


class Patches(nn.Module):
    """bux's ``patches``: each patch's pixels, normalized per channel, projected to
    `features` channels, plus a learned position of its row and column, the table
    resized bilinearly to its image's grid, corners on corners."""

    bux_types = ("patches",)
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, c: dict, width: Any, site: Site | None = None):
        super().__init__()
        n = 3 * c["size"] * c["size"]
        self.proj = nn.Linear(n, c["features"])
        grid = c["grid"]
        self.positions = nn.Embedding(grid * grid, c["features"]) if grid else None
        self.grid = grid
        per = c["size"] * c["size"]
        mean = torch.tensor(c["mean"]).repeat_interleave(per)
        std = torch.tensor(c["std"]).repeat_interleave(per)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        self.config = c

    def reset(self) -> None:
        """Recompute the normalization, which transformers' meta-device init leaves
        unset."""
        per = self.config["size"] * self.config["size"]
        with torch.no_grad():
            self.mean.copy_(torch.tensor(self.config["mean"]).repeat_interleave(per))
            self.std.copy_(torch.tensor(self.config["std"]).repeat_interleave(per))

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        dtype = self.proj.weight.dtype
        y = self.proj(((x.float() / 255 - self.mean) / self.std).to(dtype))
        if self.positions is None:
            return y
        return y + self._positions(ctx.images.grid, ctx.images.sizes).to(y.dtype)

    def _positions(self, grid: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
        """Return each patch's learned position, b n d."""
        assert self.positions is not None
        g, table = self.grid, self.positions.weight
        src = grid.float() * (g - 1) / (sizes - 1).clamp(min=1)
        lo = src.floor().long()
        hi = (lo + 1).clamp(max=g - 1)
        w = src - lo
        (r0, c0), (r1, c1), (wr, wc) = ((a[..., 0], a[..., 1]) for a in (lo, hi, w))
        return (
            table[r0 * g + c0] * ((1 - wr) * (1 - wc))[..., None]
            + table[r0 * g + c1] * ((1 - wr) * wc)[..., None]
            + table[r1 * g + c0] * (wr * (1 - wc))[..., None]
            + table[r1 * g + c1] * (wr * wc)[..., None]
        )


class MergePatches(nn.Module):
    """bux's ``merge_patches``: each run of ``size * size`` patches joined into one,
    their channels side by side, or their mean."""

    bux_types = ("merge_patches",)

    def __init__(self, c: dict, width: Any, site: Site | None = None):
        super().__init__()
        self.n, self.pool = c["size"] * c["size"], c["pool"]

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        b, n, d = x.shape
        if self.pool == "mean":
            return x.reshape(b, n // self.n, self.n, d).mean(2)
        return x.reshape(b, n // self.n, self.n * d)


# --- image(...) -------------------------------------------------------------------------


def splice(x: torch.Tensor, encoder: nn.Module, ctx: Ctx, token: int) -> torch.Tensor:
    """bux's ``image(encoder, token=...)``: x with each token `token` replaced by the
    next output of `encoder` on the pass's images; x as it is without images, as on
    the passes of a generation after the first."""
    images = ctx.images
    if images is None:
        return x
    inner = Ctx(images.pixels, images.positions, None, None, ctx.shared, _image_masks)
    inner.dtype, inner.config, inner.images = ctx.dtype, ctx.config, images
    inner.rotary = images.grid
    y = encoder(images.pixels, inner)
    is_image = ctx.tokens == token
    k = (is_image.long().cumsum(1) - 1).clamp(min=0)
    out = y.gather(1, k[..., None].expand(-1, -1, y.shape[-1])).to(x.dtype)
    return torch.where(is_image[..., None], out, x)


def _image_masks(ctx: Ctx, kind: str, arg: Any) -> torch.Tensor:
    """The encoder's one mask, ``bidirectional``: which patches each patch sees,
    b 1 n n, those of its own image, where the patch positions restart."""
    assert kind == "bidirectional", kind
    positions = ctx.positions
    starts = torch.ones_like(positions, dtype=torch.bool)
    starts[:, 1:] = positions[:, 1:] <= positions[:, :-1]
    ids = starts.long().cumsum(1)
    return (ids[:, :, None] == ids[:, None, :])[:, None]


def rope_positions(ctx: Ctx, images: Images | None, seen: int) -> torch.Tensor | None:
    """Return the pass's M-RoPE positions, b t 3, of its tokens from the `seen`-th:
    the prompt's, given with its images, and one past the prompt's last on every
    axis for each token after it, kept on the cache between passes. None for a model
    that generates without images.

    Each token after the prompt is at its index plus the prompt's offset, the most
    its positions reach, plus 1, less its length, as Qwen2-VL's ``rope_deltas``."""
    t = ctx.tokens.shape[1]
    if images is not None:
        rope, n = images.rope, images.rope.shape[1]
        offset = rope.amax((1, 2)) + 1 - n
        if ctx.cache is not None:
            ctx.cache.bux_rope_offset = offset
        after = torch.arange(n, max(n, seen + t), device=rope.device)
        later = (after[None] + offset[:, None])[..., None].expand(-1, -1, 3)
        return torch.cat([rope, later], 1)[:, seen : seen + t]
    offset = getattr(ctx.cache, "bux_rope_offset", None)
    if offset is None:
        return None
    return (ctx.positions + offset[:, None])[..., None].expand(-1, -1, 3)
