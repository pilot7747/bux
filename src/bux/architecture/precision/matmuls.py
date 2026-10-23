"""The format each projection inside a block multiplies in, which ``precision`` says:

    precision = precision(compute=bfloat16, matmuls=mxfp8, roles=roles(kv=compute),
                          first=2, last=2, passes=[fprop, dgrad])

`compute`, the default, multiplies in the dtype the run computes in. `mxfp8`,
`mxfp8_2d` and `nvfp4` quantize the operands of the forward's GEMM and of the
backward's two, in blocks along what each contracts with a scale per block
(``blockscaled``): cuDNN's block-scaled GEMM on a Blackwell GPU; elsewhere XLA
dequantizes and multiplies, which computes the same values. They multiply x @ W, as
projections do, whose dimensions are whole blocks; anything else multiplies in
`compute`. Of the three GEMMs, only those in `passes` multiply in the format.

A projection has a role, which its layer gives it (``role``), and `roles` can give a
role a format of its own. The residual branches outside any other are numbered as
they trace (``tracing``), so that the `first` and `last` of the model, before its
first head, multiply in compute, and those after it as `roles.mtp` says.

A format is a dot_general by name, so a new recipe is a function:

    from bux.architecture.precision import matmuls

    @matmuls.formats.register("mine")
    def _(lhs, rhs, dimension_numbers, **kwargs): ...  # jax.lax.dot_general's

Every projection inside a block multiplies by `dot_general`, which runs the format in
force (`use`, which the trainer enters as it traces its steps). The embedding, the LM
head, softmax attention's own matmuls, the router, norms and recurrent states stay in
`compute`; docs/parallelism.md lists them.
"""

import contextlib
import functools
import math
from collections.abc import Callable, Generator
from contextvars import ContextVar
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int, UInt32

from bux import parallel
from bux.architecture.precision import blockscaled, nvfp4
from bux.architecture.repeats import tracing
from bux.train.config import PrecisionConfig, RolesConfig
from bux.train.registry import Registry

Dimensions = jax.lax.DotDimensionNumbers
Gemm = Callable[[Float[Array, "..."], Float[Array, "..."]], Float[Array, "..."]]

formats = Registry[Any]("matmul format", "bux.architecture.precision.matmuls.formats")
PRECISION = ContextVar[PrecisionConfig | None]("precision", default=None)
_COMPUTE = PrecisionConfig()  # in force outside `use`
SEED = ContextVar[UInt32[Array, ""] | None]("matmuls seed", default=None)


@contextlib.contextmanager
def use(
    precision: PrecisionConfig | str, tokens: Int[Array, "*s"] | None = None
) -> Generator[None]:
    """Multiply block projections as `precision` says, or in the format it names, in
    the forwards traced inside, what a format rounds stochastically seeded by a hash
    of `tokens`: so anew each batch, and the same on any mesh."""
    if isinstance(precision, str):
        precision = PrecisionConfig(matmuls=precision)
    for name in _formats(precision):
        formats.find(name)
    seed = None if tokens is None else _seed(tokens)
    precision_token, seed_token = PRECISION.set(precision), SEED.set(seed)
    try:
        yield
    finally:
        PRECISION.reset(precision_token)
        SEED.reset(seed_token)


def _formats(precision: PrecisionConfig) -> set[str]:
    """The formats `precision` names: its `matmuls`, and its roles'."""
    roles = precision.roles.model_dump(exclude={"type"}).values()
    return {precision.matmuls, *(fmt for fmt in roles if fmt)}


def in_force() -> PrecisionConfig:
    """Return the precision in force: all in compute outside `use`."""
    return PRECISION.get() or _COMPUTE


def seed() -> UInt32[Array, ""]:
    """Return the seed in force, or 0."""
    return jnp.uint32(0) if (s := SEED.get()) is None else s


def _seed(tokens: Int[Array, "*s"]) -> UInt32[Array, ""]:
    positions = jnp.arange(tokens.size, dtype=jnp.uint32).reshape(tokens.shape)
    return blockscaled.mix(tokens.astype(jnp.uint32) * (2 * positions + 1)).sum()


def late(precision: PrecisionConfig, steps: int) -> tuple[int, PrecisionConfig] | None:
    """Return the step of a run of `steps` at which `precision` switches, and what it
    switches to; None if it does not."""
    if (switch := precision.switch) is None:
        return None
    after = precision.model_copy(update={"passes": switch.passes, "switch": None})
    return round(switch.at * steps), after


def evaluating(precision: PrecisionConfig | str) -> PrecisionConfig | str:
    """Return the precision evaluation multiplies in: training's, or compute."""
    if isinstance(precision, PrecisionConfig) and precision.eval == "compute":
        return PrecisionConfig(compute=precision.compute)
    return precision


# --- Roles ----------------------------------------------------------------------------


@functools.cache
def role(name: str | None) -> Callable[..., Float[Array, "..."]]:
    """Return the dot_general of the projections of role `name` (``RolesConfig``):
    one function for all of them, so that a scanned repeat's passes stay alike."""
    if name is None:
        return dot_general
    if name not in RolesConfig.model_fields or name == "type":
        raise ValueError(f"{name!r} is not a role of bux.train.config.RolesConfig")
    return functools.partial(dot_general, role=name)


def format_of(role: str | None) -> str:
    """Return the format a projection of `role` multiplies in, in the branch that is
    tracing: compute in the first and last, `roles.mtp`'s after the first head, else
    its role's or `matmuls`."""
    precision = in_force()
    kind = _kind(tracing.current_branch(), precision)
    if kind == "compute":
        return "compute"
    name = "mtp" if kind == "mtp" else role
    given = getattr(precision.roles, name) if name else None
    return given or precision.matmuls


# --- Residual branches ------------------------------------------------------------------


def _kind(branch: tracing.Branch | None, precision: PrecisionConfig) -> str:
    """Return how `branch` multiplies: in `compute` among the model's first and last,
    as `mtp` after its first head where `roles.mtp` says, else as `roles`, as
    outside any branch."""
    if branch is None:
        return "roles"
    n, main = branch
    first, last = _number(precision.first, main), _number(precision.last, main)
    if n < first or main - last <= n < main:
        return "compute"
    return "mtp" if n >= main and precision.roles.mtp else "roles"


def _number(given: float, of: int) -> int:
    """Return `given` branches, or that share of `of`, rounded up."""
    return given if isinstance(given, int) else math.ceil(given * of)


def cuts(times: int, each: int) -> list[int]:
    """Return the passes of the next `times` of a repeat, of `each` branches outside
    any other, whose branches multiply unlike the pass's before: where the precision's
    first or last start or stop (``tracing.runs``)."""
    precision = in_force()
    kinds = [_kind(b, precision) for b in tracing.next_branches(times * each)]
    passes = [kinds[p * each : (p + 1) * each] for p in range(times)]
    return [p for p in range(1, times) if passes[p] != passes[p - 1]]


# --- Formats ----------------------------------------------------------------------------


def dot_general(
    lhs: Float[Array, "..."],
    rhs: Float[Array, "..."],
    dimension_numbers: Dimensions,
    precision: Any = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding: Any = None,
    role: str | None = None,
) -> Float[Array, "..."]:
    """`jax.lax.dot_general` in the format in force for a projection of `role`: what
    block projections take (``role``)."""
    return formats.find(format_of(role))(
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
    whole blocks, its GEMMs in the precision's `passes`; anything else in `compute`."""

    def multiply(
        lhs: Float[Array, "..."],
        rhs: Float[Array, "..."],
        dimension_numbers: Dimensions,
        precision: Any = None,
        preferred_element_type: DTypeLike | None = None,
        out_sharding: Any = None,
    ) -> Float[Array, "..."]:
        passes = in_force().passes
        if (
            not passes
            or not projects(lhs, rhs, dimension_numbers)
            or not in_blocks(lhs, rhs, dimension_numbers, recipe.block)
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
            return blockscaled.project(x, w, recipe, out, seed(), passes)

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
    compiler; and its gradient likewise. In a shard_map manual over every axis, each
    device holds its own part already."""
    auto = AxisType.Auto in jax.sharding.get_abstract_mesh().axis_types
    if not dims or parallel.active() is None or not auto:
        return x
    spec = [None if i in dims else P.UNCONSTRAINED for i in range(x.ndim)]
    return jax.lax.with_sharding_constraint(x, P(*spec))


# MXFP8 (OCP): E4M3 values, a power-of-two scale per 32, as Transformer Engine's
# MXFP8 recipe quantizes all three GEMMs.
formats.register("mxfp8")(block_scaled(blockscaled.MXFP8))
# MXFP8 with W's scales per 32 x 32 tile, the same for y and dx (TorchTitan's).
formats.register("mxfp8_2d")(block_scaled(blockscaled.MXFP8_2D))
# NVFP4 (``nvfp4``): NVIDIA's pretraining recipe.
formats.register("nvfp4")(block_scaled(nvfp4.NVFP4))
