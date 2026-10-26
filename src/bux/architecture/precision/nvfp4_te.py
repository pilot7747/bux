"""NVFP4's quantizers by Transformer Engine's kernels, on a B200 or B300 where TE's JAX
bindings are installed (bux's Docker image): fewer and faster passes over each operand
than `~bux.architecture.precision.nvfp4`'s Pallas kernels make.

- x and dy: TE's amax kernel finds the largest magnitude of the operand and of its
  transpose through the Hadamard transform, in one pass; then one call of its NVFP4
  cast writes the operand's copy, and its Hadamard GEMM on the tensor cores, fused
  with the cast, the transpose's.
- W: its largest magnitude by XLA, as for the Pallas kernels; then TE's cast of
  16 x 16 tiles writes both copies, in one pass.

They take bfloat16 operands whose part on each device is whole tiles of 64 rows by
128 columns (`takes`), and quantize as the Pallas kernels do but for their rounding:
TE divides exactly, rounds the transform to bfloat16 before it quantizes it, and draws
dy's stochastic rounding from Philox, of the seed and each device's part, where the
Pallas kernels hash each element's place in the whole.
"""

import functools
import math
from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Shaped, UInt32

from bux import kernels, parallel
from bux.architecture.precision import blockscaled, nvfp4
from bux.architecture.precision.blockscaled import Blocks, Shardings
from bux.architecture.precision.nvfp4 import BLOCK, Kind
from bux.parallel.layout import as_split, axes_of

SM = ("10.0", "10.3")  # the GPUs TE's NVFP4 kernels run on: B200, B300
TILE = (64, 128)  # what each device's part of an operand must be whole tiles of
#: The fewest rows of x or dy a device that TE's kernels take, as a step of fewer
#: tokens waits on their extra launches more than their faster passes save (2% slower
#: at a row of 4096 on one B300, 23% on 8 under FSDP; faster from two rows).
ROWS = 8192
#: `nvfp4.HADAMARD`'s signs as TE's amax kernel takes them: bit i set where row i's
#: sign is negative. TE's transform is then bux's.
SIGN_MASK = sum(1 << i for i, sign in enumerate(nvfp4.SIGNS) if sign < 0)

# x's E2M1 values and E4M3 scales, then x^T's.
TeArrays = tuple[
    Shaped[Array, "r c"],
    Shaped[Array, "r n"],
    Shaped[Array, "c r"],
    Shaped[Array, "c q"],
]


def takes(x: Float[Array, "r c"], tiles: bool = False) -> bool:
    """Return whether TE's kernels quantize x: bfloat16, whole tiles of 64 x 128, on a
    B200 or B300 with TE's JAX bindings; unless W (`tiles`), `ROWS` or more of its
    rows for each device of the mesh. On a mesh, a dimension whose parts are not whole
    tiles stays whole on each device."""
    r, c = x.shape
    whole = r % TILE[0] == 0 and c % TILE[1] == 0
    on = parallel.active()
    enough = tiles or r >= ROWS * (1 if on is None else on.mesh.size)
    here = kernels.gpu() and _te() is not None
    return x.dtype == jnp.bfloat16 and whole and enough and here


class Te(NamedTuple):
    """The TE primitives bux binds, each on one device, and the enums they take."""

    #: The largest magnitudes of x and of x^T transformed (``te_rht_amax_ffi``).
    amax: Any
    #: The NVFP4 cast (``te_dbias_quantize_ffi``).
    cast: Any
    #: The amax kernel's scope, the device's part alone.
    scope: Any
    #: Which copies a cast writes (``QuantizeLayout``).
    layout: Any
    #: A scale per 16 of a row, or per 16 x 16 tile (``ScalingMode``).
    mode: Any


@functools.cache
def _te() -> Te | None:
    """Return TE's primitives; None where TE is not installed or the GPUs are not
    ones its NVFP4 kernels run on. TE imports once jax's GPU backend has loaded NCCL,
    which TE's core library loads too."""
    device = jax.local_devices()[0]
    if getattr(device, "compute_capability", "") not in SM:
        return None
    try:
        from transformer_engine.jax.cpp_extensions import amax, quantization
        from transformer_engine.jax.quantize import QuantizeLayout, ScalingMode
    except ImportError:
        return None
    return Te(
        amax.RHTAmaxCalculationPrimitive.inner_primitive,
        quantization.QuantizePrimitive.inner_primitive,
        amax.AmaxScope.LOCAL,
        QuantizeLayout,
        ScalingMode,
    )


def twice(
    x: Float[Array, "r c"],
    seed: UInt32[Array, ""] | None = None,
    tiles: bool = False,
    stochastic: bool = False,
) -> tuple[Blocks, Blocks]:
    """Return x and x^T in NVFP4 as `nvfp4.kernel_twice` does, up to rounding, by TE's
    kernels, each device on its part of x.

    Args:
        x: the operand, (r, c), which `takes`.
        seed: what stochastic rounding draws its noise from; None for 0.
        tiles: whether to scale per 16 x 16 tile, the same for x and x^T, as W is.
        stochastic: whether to round stochastically, as dy is.

    Returns:
        x and x^T, each in blocks of 16 along its rows. x^T goes through the random
        Hadamard transform first, unless `tiles`.
    """
    amaxes = _amaxes(x, tiles)
    # An operand, not a constant, which custom partitioning does not take.
    hadamard = jnp.asarray(nvfp4.HADAMARD, jnp.bfloat16)  # 1/4s, exact
    salt = nvfp4.salt_of(seed).reshape(1)
    quantize = blockscaled.partitioned_unless_manual(_cast, _cast_partitioned)
    rows, row_scales, cols, col_scales = quantize(
        x, hadamard, amaxes, salt, Kind(tiles, stochastic)
    )
    decode = 1 / nvfp4.encode_scale(amaxes)
    return Blocks(rows, row_scales, decode[0]), Blocks(cols, col_scales, decode[1])


def _amaxes(x: Float[Array, "r c"], tiles: bool) -> Float[Array, "2"]:
    """Return the largest magnitude of x and of x^T transformed, by TE's kernel in one
    pass, each device on its part of x; W's, which is not transformed, by XLA, as the
    Pallas kernels' does, which gathers it early."""
    if tiles:
        return jnp.abs(x).max().astype(jnp.float32).repeat(2)
    amaxes = blockscaled.partitioned_unless_manual(_te_amaxes, _amaxes_partitioned)
    return amaxes(x)


def _te_amaxes(x: Float[Array, "r c"]) -> Float[Array, "2"]:
    """Return the largest magnitude of x and of x^T transformed, by TE's kernel."""
    te = _te()
    assert te is not None
    amax, transformed = te.amax.bind(
        x,
        amax_scope=te.scope,
        transpose_batch_sequence=False,
        rht_matrix_random_sign_mask_t=SIGN_MASK,
        produce_regular_amax=True,
        flatten_axis=1,
    )
    return jnp.concatenate([amax, transformed])


def _cast(
    x: Float[Array, "r c"],
    hadamard: Float[Array, "16 16"],
    amaxes: Float[Array, "2"],
    salt: UInt32[Array, "1"],
    kind: Kind,
    start: UInt32[Array, "2"] | None = None,
) -> TeArrays:
    """Return x's copy and x^T's by TE's NVFP4 cast, in one call, by the tensor scales
    of `amaxes`: x^T through the `hadamard` transform by TE's Hadamard GEMM fused with
    the cast, or for W both in 16 x 16 tiles. x is the part at `start` of the whole,
    whose noise is its own."""
    te = _te()
    assert te is not None
    start = jnp.zeros(2, jnp.uint32) if start is None else start
    noise = _noise(salt, start) if kind.stochastic else jnp.zeros(0, jnp.uint32)
    mode = te.mode.NVFP4_2D_SCALING if kind.tiles else te.mode.NVFP4_1D_SCALING
    rows, cols, row_scales, col_scales, *_ = te.cast.bind(
        x,
        jnp.ones(1, jnp.float32),  # a tensor scale for FP8, which NVFP4 ignores
        amaxes[:1],
        noise,
        amaxes[1:],
        jnp.zeros((1, 1), jnp.bfloat16) if kind.tiles else hadamard,
        out_dtype=jnp.float4_e2m1fn,
        scaling_mode=mode.value,
        q_layout=te.layout.ROWWISE_COLWISE,
        flatten_axis=-1,
        scale_dtype=jnp.float8_e4m3fn,
        is_dbias=False,
        is_outer=False,
        stochastic_rounding=kind.stochastic,
        use_rht=not kind.tiles,
    )
    (r, c), b = x.shape, BLOCK
    # TE pads the scales to whole tiles of 128 x 4 for cuBLAS; the GEMM takes them
    # as they are.
    return rows, row_scales[:r, : c // b], cols, col_scales[:c, : r // b]


def _noise(salt: UInt32[Array, "1"], start: UInt32[Array, "2"]) -> UInt32[Array, "4"]:
    """Return the Philox state that TE's stochastic rounding draws its noise from on
    this device, four words: a key of the seed and the first row of the device's part,
    then an offset of its first column, far past any part's draws."""
    return jnp.stack([salt[0], start[0], jnp.uint32(0), start[1]])


# --- On a mesh: each device on its part of x, in whole tiles ---------------------------


@custom_partitioning
def _amaxes_partitioned(x: Float[Array, "r c"]) -> Float[Array, "2"]:
    """Return `_te_amaxes`, each device running it on its part of x."""
    return _te_amaxes(x)


def _amaxes_infer(mesh: Mesh, shapes: tuple[Any, ...], out: Any) -> NamedSharding:
    """Return how the largest magnitudes split: every device has them."""
    return NamedSharding(mesh, P())


def _amaxes_partition(
    mesh: Mesh, shapes: tuple[Any, ...], out: Any
) -> tuple[Mesh, Callable[..., Float[Array, "2"]], NamedSharding, Shardings]:
    """Each device runs the kernel on its part of x, and the largest of theirs is
    every device's."""
    spec = _in_tiles(shapes[0].sharding.spec, shapes[0].shape, mesh)
    split = axes_of(spec)

    def amaxes(x: Float[Array, "r c"]) -> Float[Array, "2"]:
        mine = _te_amaxes(x)
        return jax.lax.pmax(mine, split) if split else mine

    return mesh, amaxes, NamedSharding(mesh, P()), (NamedSharding(mesh, spec),)


_amaxes_partitioned.def_partition(
    infer_sharding_from_operands=_amaxes_infer,
    partition=_amaxes_partition,
    sharding_rule="r c -> a",
)


@partial(custom_partitioning, static_argnums=(4,))
def _cast_partitioned(
    x: Float[Array, "r c"],
    hadamard: Float[Array, "16 16"],
    amaxes: Float[Array, "2"],
    salt: UInt32[Array, "1"],
    kind: Kind,
) -> TeArrays:
    """Return `_cast`'s copies, each device running it on its part of x."""
    return _cast(x, hadamard, amaxes, salt, kind)


def _cast_infer(kind: Kind, mesh: Mesh, shapes: tuple[Any, ...], out: Any) -> Shardings:
    """Return how the copies split, of x as `_cast_partition` splits it."""
    spec = _in_tiles(shapes[0].sharding.spec, shapes[0].shape, mesh)
    return blockscaled.copies_shardings(mesh, spec)


def _cast_partition(
    kind: Kind, mesh: Mesh, shapes: tuple[Any, ...], out: Any
) -> tuple[Mesh, Callable[..., TeArrays], Shardings, Shardings]:
    """Each device runs the cast on its part of x, in whole tiles, the blocks whole on
    every device (``matmuls.blocks_whole``), its noise that part's own."""
    spec = _in_tiles(shapes[0].sharding.spec, shapes[0].shape, mesh)
    r, c = blockscaled.split_of(spec)

    def cast(
        x: Float[Array, "r c"],
        hadamard: Float[Array, "16 16"],
        amaxes: Float[Array, "2"],
        salt: UInt32[Array, "1"],
    ) -> TeArrays:
        start = jnp.stack(
            [nvfp4.part_start(r, x.shape[0]), nvfp4.part_start(c, x.shape[1])]
        )
        return _cast(x, hadamard, amaxes, salt, kind, start)

    everywhere = NamedSharding(mesh, P())
    args = (NamedSharding(mesh, spec), *[everywhere] * 3)
    return mesh, cast, blockscaled.copies_shardings(mesh, spec), args


_cast_partitioned.def_partition(
    infer_sharding_from_operands=_cast_infer,
    partition=_cast_partition,
    sharding_rule="r c, h k, a, s -> r c, r c, c r, c r",
)


def _in_tiles(spec: P, shape: tuple[int, int], mesh: Mesh) -> P:
    """Return how an (r, c) array split by `spec` splits for the kernels: as it is,
    but whole along a dimension whose parts would not be whole tiles."""
    split = blockscaled.split_of(spec)
    fits = [_fits(a, n, t, mesh) for a, n, t in zip(split, shape, TILE, strict=True)]
    return P(*(a if fit else None for a, fit in zip(split, fits, strict=True)))


def _fits(axes: Any, size: int, tile: int, mesh: Mesh) -> bool:
    """Return whether a dimension of `size` splits over the mesh's `axes` into whole
    tiles."""
    parts = math.prod(mesh.shape[a] for a in as_split(axes))
    return size % (parts * tile) == 0
