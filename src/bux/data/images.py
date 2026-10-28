"""The images of a prepared dataset, and the patches the loader cuts them into.

The bytes of each image are kept beside the dataset's rows (`Store`, written by
`write_part` and `write_store`), and a row says which image each token stands for.
`row` reads a row's images back, resized and cut into patches by
`bux.export.hf.pixels`, the same preprocessing an exported model takes its images
with, and gives each token its M-RoPE position. It is Qwen2-VL's preprocessing,
for one frame: a video's two frames of a still image are the same, so the encoder
sums its weights over them instead (`bux.pretrained.qwen3_5`).
"""

from __future__ import annotations

import io
import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from jaxtyping import Bool, Int32, Int64, Shaped, UInt8

from bux.export.hf.pixels import mrope, patches, resize


class Rows(NamedTuple):
    """The images of rows on the host, as `row` makes a row's: what
    `bux.architecture.registry.Images` holds on a device."""

    pixels: UInt8[np.ndarray, "b n p"]  #: Each patch's pixels.
    positions: Int32[np.ndarray, "b n"]  #: Each patch's index in its image.
    grid: Int32[np.ndarray, "b n 2"]  #: Each patch's row and column.
    sizes: Int32[np.ndarray, "b n 2"]  #: The rows and columns of its image.
    rope: Int32[np.ndarray, "b t 3"]  #: Each token's M-RoPE position.


class Store:
    """The images of a prepared dataset, by key: their bytes as given, in files
    ``images-*.bin`` beside its rows, and ``images.npz``, which says where each one
    is and its grid of patches.

    A row holds an image's key at each of its tokens (`bux.data.packing`), and the
    loader reads its bytes here (`read`).
    """

    def __init__(self, folder: Path):
        index = np.load(folder / "images.npz")
        self.keys, self.files = index["keys"], index["files"]
        self.offsets, self.sizes = index["offsets"], index["sizes"]
        self.grids = index["grids"]
        self.blobs = [_mapped(f) for f in sorted(folder.glob("images-*.bin"))]

    def at(self, key: int) -> int:
        """Return the index of the image of `key`."""
        i = int(np.searchsorted(self.keys, key))
        if i == len(self.keys) or self.keys[i] != key:
            raise KeyError(f"no image of key {key} in the dataset")
        return i

    def read(self, key: int) -> tuple[bytes, tuple[int, int]]:
        """Return the bytes of the image of `key`, and its rows and columns of
        patches."""
        i = self.at(key)
        blob = self.blobs[int(self.files[i])]
        a = int(self.offsets[i])
        return bytes(blob[a : a + int(self.sizes[i])]), tuple(self.grids[i].tolist())


def _mapped(path: Path) -> UInt8[np.ndarray, " n"]:
    """Return a file's bytes, memory-mapped; an empty file's, as an empty array."""
    if not path.stat().st_size:  # numpy cannot map an empty file
        return np.zeros(0, np.uint8)
    return np.memmap(path, np.uint8, "r")


def write_store(folder: Path, parts: Sequence[Path]) -> None:
    """Write the image store of a dataset into `folder` from the map's parts, each a
    folder with an ``images.bin`` and its ``images.npz``: the bins move, and one
    index, by key, says where each image is. An image in several parts is kept once.
    """
    keys, files, offsets, sizes, grids = [], [], [], [], []
    for f, part in enumerate(parts):
        index = np.load(part / "images.npz")
        (part / "images.bin").rename(folder / f"images-{f:05d}.bin")
        keys.append(index["keys"])
        files.append(np.full(len(index["keys"]), f, np.int32))
        offsets.append(index["offsets"])
        sizes.append(index["sizes"])
        grids.append(index["grids"])
    k = np.concatenate([np.zeros(0, np.int64), *keys])
    _, first = np.unique(k, return_index=True)  # sorted by key, each once

    def joined(
        xs: list[Shaped[np.ndarray, "_ *k"]], empty: Shaped[np.ndarray, "_ *k"]
    ) -> Shaped[np.ndarray, "_ *k"]:
        return np.concatenate([empty, *xs])[first]

    np.savez(
        folder / "images.npz",
        keys=k[first],
        files=joined(files, np.zeros(0, np.int32)),
        offsets=joined(offsets, np.zeros(0, np.int64)),
        sizes=joined(sizes, np.zeros(0, np.int64)),
        grids=joined(grids, np.zeros((0, 2), np.int32)),
    )


def write_part(
    folder: Path, images: Sequence[tuple[int, bytes, tuple[int, int]]]
) -> None:
    """Write one map task's images, each its key, bytes and grid of patches, into
    `folder` as ``images.bin`` and ``images.npz``, for `write_store`."""
    sizes = np.array([len(b) for _, b, _ in images], np.int64)
    with open(folder / "images.bin", "wb") as out:
        out.writelines(b for _, b, _ in images)
    np.savez(
        folder / "images.npz",
        keys=np.array([k for k, _, _ in images], np.int64),
        offsets=np.cumsum(sizes) - sizes,
        sizes=sizes,
        grids=np.array([g for _, _, g in images], np.int32).reshape(-1, 2),
    )


def decoded(data: bytes) -> UInt8[np.ndarray, "h w 3"]:
    """Return an image's pixels, RGB, from its encoded bytes (PNG, JPEG, ...)."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"))


def size(data: bytes) -> tuple[int, int]:
    """Return an image's height and width, from its encoded bytes, without decoding
    its pixels."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        w, h = image.size
    return h, w


def row(
    keys: Int64[np.ndarray, " t"],
    positions: Int32[np.ndarray, " t"],
    store: Store,
    *,
    size: int,
    merge: int,
    patches_per_row: int,
) -> tuple[
    UInt8[np.ndarray, "n p"],
    Int32[np.ndarray, " n"],
    Int32[np.ndarray, "n 2"],
    Int32[np.ndarray, "n 2"],
    Int32[np.ndarray, "t 3"],
]:
    """Return the images of a row, as `bux.architecture.registry.Images` holds a
    row's: their patches, padded to `patches_per_row`, each patch's index in its
    image, its row and column, its image's grid, and each token's M-RoPE position.

    Args:
        keys: Each token's image's key, 0 for a token of text; an image is a run of
            ``rows * cols / merge**2`` tokens of its key.
        positions: Each token's position in its document, which starts where they
            stop increasing.
        store: The dataset's images.
        size: Pixels on a side of a patch.
        merge: Patches on a side of a block, which an image token stands for.
        patches_per_row: The patches a row has room for.

    Raises:
        ValueError: The row's images have more patches than it has room for.
    """
    pixels, index, grid, sizes, blocks = [], [], [], [], []
    i = 0
    while i < len(keys):
        if not keys[i]:
            i += 1
            continue
        data, (gh, gw) = store.read(int(keys[i]))
        x, at = patches(
            resize(decoded(data), gh * size, gw * size), size=size, merge=merge
        )
        pixels.append(x)
        index.append(np.arange(len(x), dtype=np.int32))
        grid.append(at)
        sizes.append(np.tile(np.array([gh, gw], np.int32), (len(x), 1)))
        blocks.append((gh // merge, gw // merge))
        i += gh * gw // (merge * merge)
    n = sum(len(x) for x in pixels)
    if n > patches_per_row:
        raise ValueError(
            f"a row's images have {n} patches, but it has room for {patches_per_row}"
        )
    pad = patches_per_row - n
    pixels.append(np.zeros((pad, 3 * size * size), np.uint8))
    index.append(np.arange(pad, dtype=np.int32))
    grid.append(np.zeros((pad, 2), np.int32))
    sizes.append(np.ones((pad, 2), np.int32))
    return (
        np.concatenate(pixels),
        np.concatenate(index),
        np.concatenate(grid),
        np.concatenate(sizes),
        _row_mrope(keys != 0, positions, blocks),
    )


def _row_mrope(
    is_image: Bool[np.ndarray, " t"],
    positions: Int32[np.ndarray, " t"],
    blocks: Sequence[tuple[int, int]],
) -> Int32[np.ndarray, "t 3"]:
    """Return the M-RoPE positions of a row's documents (`mrope`), each from 0, its
    images' grids of blocks taken in order."""
    starts = np.flatnonzero(np.r_[True, positions[1:] <= positions[:-1]])
    out, grids = [], iter(blocks)
    for a, b in itertools.pairwise([*starts.tolist(), len(positions)]):
        doc, mine = is_image[a:b], []
        need = int(doc.sum())  # the document's image tokens, of its images in order
        while need > 0:
            mine.append(next(grids))
            need -= mine[-1][0] * mine[-1][1]
        out.append(mrope(doc, mine))
    return np.concatenate(out) if out else np.zeros((0, 3), np.int32)
