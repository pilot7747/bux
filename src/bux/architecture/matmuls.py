"""The format the projections inside blocks multiply in, which ``precision.matmuls``
names:

    precision = precision(compute=bfloat16, matmuls=mxfp8)

`compute`, the default, multiplies in the dtype the run computes in. `mxfp8` and
`nvfp4` quantize both operands of the forward's GEMM and of the backward's two, in
blocks along what each contracts with a scale per block (``blockscaled``): cuDNN's
block-scaled GEMM on a Blackwell GPU; elsewhere XLA dequantizes and multiplies, which
computes the same values. They multiply x @ W, as projections do, whose dimensions are
whole blocks; anything else multiplies in `compute`.

A format is a dot_general by name, so a new recipe is a function:

    from bux.architecture import matmuls

    @matmuls.formats.register("mine")
    def _(lhs, rhs, dimension_numbers, **kwargs): ...  # jax.lax.dot_general's

Every projection inside a block multiplies by `dot_general`, which runs the format in
force (`use`, which the trainer enters as it traces its steps). The embedding, the LM
head, softmax attention's own matmuls, the router, norms and recurrent states stay in
`compute`; docs/parallelism.md lists them.
"""

import contextlib
import math
from collections.abc import Callable, Generator
from contextvars import ContextVar
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int, UInt32

from bux import parallel
from bux.architecture import blockscaled
from bux.train.registry import Registry

Dimensions = jax.lax.DotDimensionNumbers
Gemm = Callable[[Float[Array, "..."], Float[Array, "..."]], Float[Array, "..."]]

formats = Registry[Any]("matmul format", "bux.architecture.matmuls.formats")
FORMAT = ContextVar("matmuls", default="compute")
SEED = ContextVar[UInt32[Array, ""] | None]("matmuls seed", default=None)


@contextlib.contextmanager
def use(name: str, tokens: Int[Array, "*s"] | None = None) -> Generator[None]:
    """Multiply block projections in the format `name` in the forwards traced
    inside, what a format rounds stochastically seeded by a hash of `tokens`: so
    anew each batch, and the same on any mesh."""
    formats.find(name)
    seed = None if tokens is None else _seed(tokens)
    format_token, seed_token = FORMAT.set(name), SEED.set(seed)
    try:
        yield
    finally:
        FORMAT.reset(format_token)
        SEED.reset(seed_token)


def seed() -> UInt32[Array, ""]:
    """Return the seed in force, or 0."""
    return jnp.uint32(0) if (s := SEED.get()) is None else s


def _seed(tokens: Int[Array, "*s"]) -> UInt32[Array, ""]:
    positions = jnp.arange(tokens.size, dtype=jnp.uint32).reshape(tokens.shape)
    return blockscaled.mix(tokens.astype(jnp.uint32) * (2 * positions + 1)).sum()


def dot_general(
    lhs: Float[Array, "..."],
    rhs: Float[Array, "..."],
    dimension_numbers: Dimensions,
    precision: Any = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding: Any = None,
) -> Float[Array, "..."]:
    """`jax.lax.dot_general` in the format in force: what block projections take."""
    return formats.find(FORMAT.get())(
        lhs,
        rhs,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        out_sharding=out_sharding,
    )


@formats.register("compute")
def compute(
    lhs: Float[Array, "..."],
    rhs: Float[Array, "..."],
    dimension_numbers: Dimensions,
    precision: Any = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding: Any = None,
) -> Float[Array, "..."]:
    """XLA's dot_general, in the operands' dtype."""
    return jax.lax.dot_general(
        lhs,
        rhs,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        out_sharding=out_sharding,
    )


def block_scaled(recipe: blockscaled.Recipe) -> Callable[..., Float[Array, "..."]]:
    """Return the format that multiplies x @ W by `recipe`, where its dimensions are
    whole blocks, and anything else in `compute`."""

    def multiply(
        lhs: Float[Array, "..."],
        rhs: Float[Array, "..."],
        dimension_numbers: Dimensions,
        precision: Any = None,
        preferred_element_type: DTypeLike | None = None,
        out_sharding: Any = None,
    ) -> Float[Array, "..."]:
        if not projects(lhs, rhs, dimension_numbers) or not in_blocks(
            lhs, rhs, dimension_numbers, recipe.block
        ):
            return compute(
                lhs,
                rhs,
                dimension_numbers,
                precision,
                preferred_element_type,
                out_sharding,
            )
        out = preferred_element_type or jnp.result_type(lhs, rhs)

        def gemm(
            x: Float[Array, "*lead k"], w: Float[Array, "k n"]
        ) -> Float[Array, "*lead n"]:
            return blockscaled.project(x, w, recipe, out, seed())

        return blocks_whole(gemm, lhs, rhs, dimension_numbers, recipe.block)

    return multiply


def projects(
    lhs: Float[Array, "..."], rhs: Float[Array, "..."], dimension_numbers: Dimensions
) -> bool:
    """Whether `dimension_numbers` multiply x @ W: x's last dimension by a matrix's
    first."""
    (lc, rc), (lb, rb) = dimension_numbers
    contracts = tuple(lc) == (lhs.ndim - 1,) and tuple(rc) == (0,)
    return rhs.ndim == 2 and contracts and not lb and not rb


def gemm_sizes(
    lhs: Float[Array, "..."], rhs: Float[Array, "..."], dimension_numbers: Dimensions
) -> tuple[int, int, int]:
    """Return the m, n and k of the GEMM `dimension_numbers` makes, its batch aside:
    what it keeps of each operand, and what it contracts."""
    (lc, rc), (lb, rb) = dimension_numbers
    m = math.prod(d for i, d in enumerate(lhs.shape) if i not in (*lc, *lb))
    n = math.prod(d for i, d in enumerate(rhs.shape) if i not in (*rc, *rb))
    return m, n, math.prod(lhs.shape[i] for i in lc)


def in_blocks(
    lhs: Float[Array, "..."],
    rhs: Float[Array, "..."],
    dimension_numbers: Dimensions,
    block: int,
) -> bool:
    """Whether m, n and k are whole blocks: what each of the three GEMMs, forward
    and backward, contracts."""
    return all(s % block == 0 for s in gemm_sizes(lhs, rhs, dimension_numbers))


def blocks_whole(
    gemm: Gemm,
    lhs: Float[Array, "..."],
    rhs: Float[Array, "..."],
    dimension_numbers: Dimensions,
    block: int,
) -> Float[Array, "..."]:
    """Return gemm(lhs, rhs), where each of m, n and k that does not split into whole
    blocks on every device stays whole on each: a device's part of a block would be
    scaled on its own. So the blocks, and what a run computes, are the same on any
    mesh."""
    on = parallel.active()
    devices = 1 if on is None else on.mesh.size
    (lc, rc), (lb, rb) = dimension_numbers
    lhs_kept = [i for i in range(lhs.ndim) if i not in (*lc, *lb)]
    rhs_kept = [i for i in range(rhs.ndim) if i not in (*rc, *rb)]
    sizes = gemm_sizes(lhs, rhs, dimension_numbers)
    m, n, k = (s % (block * devices) != 0 for s in sizes)
    out_m = range(len(lb), len(lb) + len(lhs_kept))
    out_n = range(out_m.stop, out_m.stop + len(rhs_kept))
    lhs = _whole(lhs, [*(lc if k else ()), *(lhs_kept if m else ())])
    rhs = _whole(rhs, [*(rc if k else ()), *(rhs_kept if n else ())])
    return _whole(gemm(lhs, rhs), [*(out_m if m else ()), *(out_n if n else ())])


def _whole(x: Float[Array, "..."], dims: list[int]) -> Float[Array, "..."]:
    """Constrain `x`'s `dims` to be whole on every device, and leave the rest to the
    compiler; and its gradient likewise."""
    if not dims or parallel.active() is None:
        return x
    spec = [None if i in dims else P.UNCONSTRAINED for i in range(x.ndim)]
    return jax.lax.with_sharding_constraint(x, P(*spec))


# MXFP8 (OCP): E4M3 values, a power-of-two scale per 32, as Transformer Engine's
# MXFP8 recipe quantizes all three GEMMs.
formats.register("mxfp8")(block_scaled(blockscaled.MXFP8))
# NVFP4: E2M1 values, an E4M3 scale per 16 and a float32 scale per tensor, W in 16 x 16
# tiles, dy rounded stochastically, and x and dy transformed for dW (NVIDIA's
# pretraining recipe).
formats.register("nvfp4")(block_scaled(blockscaled.NVFP4))
