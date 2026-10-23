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
they run (``branch``), so that the `first` and `last` of the model, before its first
head, multiply in compute, and those after it as `roles.mtp` says.

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
import dataclasses
import functools
import itertools
import math
from collections.abc import Callable, Generator, Iterator
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
from bux.lang import ir
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
    precision, branch, count = in_force(), _BRANCH.get(), _COUNT.get()
    kind = None if branch is None or count is None else count.kind(branch, precision)
    if kind == "compute":
        return "compute"
    name = "mtp" if kind == "mtp" else role
    given = getattr(precision.roles, name) if name else None
    return given or precision.matmuls


# --- Residual branches ------------------------------------------------------------------


@dataclasses.dataclass
class Count:
    """The residual branches outside any other in a forward: `main` of them before its
    first head, and the number of the `next` to run. Each of `scans` is a scan that
    traces one pass for all of them: its first branch, and the branches of a pass."""

    main: int
    next: int = 0
    scans: list[tuple[int, int]] = dataclasses.field(default_factory=list)

    def take(self) -> int:
        """Return the number of the next branch, and count it."""
        n = self.peek()
        self.next += 1
        return n

    def peek(self) -> int:
        """Return the number of the next branch: in a scan, its first pass's."""
        n = self.next
        for first, each in reversed(self.scans):
            n = first + (n - first) % each
        return n

    def kind(self, n: int, precision: PrecisionConfig) -> str:
        """Return how branch `n` multiplies: in `compute`, as `mtp` after the first
        head where `roles.mtp` says, else as `roles`."""
        first = _number(precision.first, self.main)
        last = _number(precision.last, self.main)
        if n < first or self.main - last <= n < self.main:
            return "compute"
        return "mtp" if n >= self.main and precision.roles.mtp else "roles"


def _number(given: float, of: int) -> int:
    """Return `given` branches, or that share of `of`, rounded up."""
    return given if isinstance(given, int) else math.ceil(given * of)


_COUNT = ContextVar[Count | None]("branches", default=None)
_BRANCH = ContextVar[int | None]("branch", default=None)


@contextlib.contextmanager
def forward(main: int) -> Generator[None]:
    """Number the residual branches of a forward of a model with `main` of them before
    its first head (``main_branches``)."""
    token = _COUNT.set(Count(main))
    try:
        yield
    finally:
        _COUNT.reset(token)


@contextlib.contextmanager
def branch() -> Generator[None]:
    """Number the residual branch outside any other that traces inside, which decides
    what its projections multiply in (``format_of``)."""
    count = _COUNT.get()
    token = _BRANCH.set(None if count is None else count.take())
    try:
        yield
    finally:
        _BRANCH.reset(token)


@contextlib.contextmanager
def scanned(times: int, each: int) -> Generator[None]:
    """Number the branches of `times` passes of `each` branches that one trace runs, a
    scan's: each as its first pass's, however often it traces, and the branches after
    them as if every pass had run."""
    count = _COUNT.get()
    if count is None or each == 0:
        yield
        return
    first = count.next
    count.scans.append((first, each))
    try:
        yield
    finally:
        count.scans.pop()
        count.next = first + times * each


def spans(times: int, each: int) -> list[range]:
    """Split the next `times` passes of `each` branches into runs of passes whose
    branches multiply alike, which a scan can trace once (`scanned`)."""
    count = _COUNT.get()
    if count is None or each == 0:
        return [range(times)]
    start, precision = count.peek(), in_force()
    kinds = [
        tuple(count.kind(start + p * each + j, precision) for j in range(each))
        for p in range(times)
    ]
    cuts = [p for p in range(1, times) if kinds[p] != kinds[p - 1]]
    ends = [0, *cuts, times]
    return [range(a, b) for a, b in itertools.pairwise(ends)]


def main_branches(graph: ir.Node) -> int:
    """Return the residual branches outside any other that run before the first head:
    those of the model, which `first` and `last` count, and not its MTP modules'."""
    count = 0
    for event in _events(graph):
        if event == "head":
            break
        count += 1
    return count


def branches_of(node: ir.Node) -> int:
    """Return the residual branches outside any other that one run of `node` takes,
    as an item of a sequence or the body of a repeat outside any branch."""
    return sum(event == "branch" for event in _item(node))


def _events(node: ir.Node) -> Iterator[str]:
    """Yield "branch" for each residual branch outside any other, and "head" for each
    head, in the order they run: as ``graph.branches`` finds the branches."""
    match node:
        case ir.Layer(config=config) if config.type == "head":
            yield "head"
        case ir.Seq(items=items):
            for item in items:
                yield from _item(item)
        case ir.Repeat(body=body, times=times):
            for _ in range(times):
                yield from _item(body)
        case ir.Residual():
            return  # what it holds is inside a branch
        case _:
            for child in ir.children(node):
                yield from _events(child)


def _item(node: ir.Node) -> Iterator[str]:
    """The events of an item of a sequence or a repeat's body: a branch, if it is one."""
    if isinstance(node, ir.Residual):
        yield "branch"
    else:
        yield from _events(node)


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
