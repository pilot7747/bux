"""Images as a vision encoder reads them, numpy and PIL alone: resized to whole
blocks of patches (`resized`, `resize`), cut into patches in the order the encoder
merges them (`patches`), and the M-RoPE positions of a document's tokens (`mrope`).

This is Qwen2-VL's preprocessing, as transformers' ``Qwen2VLImageProcessor`` does
it, but for one frame. bux's data (`bux.data.images`) and an exported model
(`vision`) both use it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def resized(
    h: int, w: int, *, size: int, merge: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Return the height and width an h x w image is resized to: the nearest
    multiples of ``size * merge``, scaled to between `min_pixels` and `max_pixels`
    pixels, keeping the aspect ratio as near as they allow.

    Raises:
        ValueError: The image is more than 200 times as long as it is wide.
    """
    factor = size * merge
    if max(h, w) / min(h, w) > 200:
        raise ValueError(f"an image of {h} x {w} pixels is too long: at most 200:1")
    hh, ww = round(h / factor) * factor, round(w / factor) * factor
    if hh * ww > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hh = max(factor, math.floor(h / beta / factor) * factor)
        ww = max(factor, math.floor(w / beta / factor) * factor)
    elif hh * ww < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hh, ww = (math.ceil(x * beta / factor) * factor for x in (h, w))
    return hh, ww


def resize(image: np.ndarray, h: int, w: int) -> np.ndarray:
    """Return `image`, h x w x 3 bytes, resized to h x w pixels, bicubic."""
    from PIL import Image

    if image.shape[:2] == (h, w):
        return image
    out = Image.fromarray(image).resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(out)


def patches(
    image: np.ndarray, *, size: int, merge: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the patches of `image`, whose sides are multiples of ``size * merge``,
    in the order the encoder merges them, and each one's row and column.

    A block's ``merge * merge`` patches come one after another, blocks row by row.
    Within a block, they come in Z-order (`within`): each 2 x 2 of them in a row,
    then the next, so that ``merge_patches(size=2)`` twice merges a block of 4 x 4
    as ``merge_patches(size=4)`` once does. For a block of 2 x 2, that is row by row.

    Returns:
        The patches, n of ``3 * size * size`` bytes as ``(channel, row, column)``;
        and each one's ``(row, column)`` in the grid of patches, n x 2.
    """
    h, w, c = image.shape
    gh, gw = h // size, w // size
    assert gh * size == h and gw * size == w and gh % merge == 0 and gw % merge == 0
    x = image.reshape(gh // merge, merge, size, gw // merge, merge, size, c)
    x = x.transpose(0, 3, 1, 4, 6, 2, 5).reshape(-1, merge * merge, c * size * size)
    rows, cols = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    grid = np.stack([rows, cols], -1).reshape(gh // merge, merge, gw // merge, merge, 2)
    grid = grid.transpose(0, 2, 1, 3, 4).reshape(-1, merge * merge, 2)
    order = within(merge)
    x, grid = x[:, order].reshape(gh * gw, -1), grid[:, order].reshape(gh * gw, 2)
    return np.ascontiguousarray(x), grid.astype(np.int32)


def within(merge: int) -> list[int]:
    """Return the order of a block's ``merge * merge`` patches, by their index row by
    row: Z-order (Morton's, a row's bit above a column's at each level) where
    `merge` is a power of 2, else row by row."""
    if merge & (merge - 1):
        return list(range(merge * merge))

    def z(i: int) -> int:
        r, c = divmod(i, merge)
        return sum(
            ((r >> b & 1) << (2 * b + 1)) | ((c >> b & 1) << (2 * b))
            for b in range(merge.bit_length())
        )

    return sorted(range(merge * merge), key=z)


def mrope(is_image: np.ndarray, grids: Sequence[tuple[int, int]]) -> np.ndarray:
    """Return a document's M-RoPE positions, n x 3 (time, height, width), as Qwen2-VL
    gives them: a token of text is one past the last position on all three; an
    image's tokens, a block of patches each, are at the image's start in time and at
    their row and column past it, and the next token is past its longer side.

    Args:
        is_image: Which of the document's n tokens are image tokens.
        grids: Each image's rows and columns of tokens (of merged blocks), in order;
            its tokens are a run of ``rows * cols`` image tokens.
    """
    out = np.zeros((len(is_image), 3), np.int32)
    at, i, images = 0, 0, iter(grids)
    while i < len(is_image):
        if not is_image[i]:
            out[i], at, i = at, at + 1, i + 1
            continue
        rows, cols = next(images)
        r, c = np.divmod(np.arange(rows * cols), cols)
        out[i : i + rows * cols] = np.stack([np.full_like(r, at), at + r, at + c], -1)
        at, i = at + max(rows, cols), i + rows * cols
    return out
