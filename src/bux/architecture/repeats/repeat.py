"""Repeats: the methods of `methods.repeat`, and the scanned module they can build.

A method builds `times` passes of a repeat's body, and the module that runs them.
``unroll`` runs one module per pass, in a row. ``scan`` stacks the passes into one
``Scanned`` module where it can, each variable gaining a leading `layers` dimension,
and runs them by `jax.lax.scan` (``stacked.run``), a scan for each run of passes that
trace alike (``tracing.runs``). With more than one `stage`, the longest scanned repeat
outside any other is pipelined over them (``pipe_longest``).
"""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from bux.architecture import residual
from bux.architecture.graph import Sequence, branches, build_node, run_branch
from bux.architecture.precision import matmuls
from bux.architecture.registry import Context, Stream, Taps
from bux.architecture.repeats import pipelined, remat, stacked, tracing
from bux.lang import ir
from bux.parallel import methods, pipeline
from bux.parallel.layout import LAYERS, NAMES


@methods.repeat.register("unroll")
def unroll(body: ir.Node, times: int, ctx: Context) -> nnx.Module:
    """Build one module per pass, run in a row."""
    return Passes(_passes(body, times, ctx), branches((body,) * times, ctx))


@methods.repeat.register("scan")
def scan(body: ir.Node, times: int, ctx: Context) -> nnx.Module:
    """Build the passes as ``unroll`` does, and stack them into one ``Scanned`` module
    if they can be (``scannable``); otherwise run them in a row."""
    passes = _passes(body, times, ctx)
    if scannable(passes):
        each = tracing.branches_of(body) if ctx.depth == 0 else 0
        [is_branch] = branches((body,), ctx)
        return Scanned(_stack(passes), len(passes), each, is_branch)
    return Passes(passes, branches((body,) * times, ctx))


def _passes(body: ir.Node, times: int, ctx: Context) -> list[nnx.Module]:
    return [build_node(body, ctx) for _ in range(times)]


class Passes(Sequence):
    """The passes of a repeat, run in a row, each traced as its pass
    (``tracing.passes``), which the run's `remat(first=n)` counts. A pass that is a
    branch outside any other runs as one (``graph.run_branch``)."""

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        for i, (item, branch) in enumerate(zip(self.items, self.branches, strict=True)):
            with tracing.passes(range(i, i + 1)):
                x = run_branch(item, x, taps) if branch else item(x, taps)
        return x


def scannable(passes: list[nnx.Module]) -> bool:
    """Return whether `passes` stack into one: they have one graphdef and the same
    variable shapes, share no module or variable (a `shared` definition, nextn's
    embedding), and leave the stream as wide as they found it, which attnres's grow."""
    if len(passes) < 2:
        return False
    first = passes[0]
    kind = nnx.graphdef(first), _shapes(first)
    same = all((nnx.graphdef(p), _shapes(p)) == kind for p in passes[1:])
    grows = any(isinstance(n, residual.AttentionResidual) for n in _nodes(first))
    return same and not grows and not _shares(passes)


def _nodes(module: nnx.Module) -> list[Any]:
    return [node for _, node in nnx.iter_graph(module)]


def _variables(module: nnx.Module) -> list[nnx.Variable]:
    return [n for n in _nodes(module) if isinstance(n, nnx.Variable)]


def _shapes(module: nnx.Module) -> list[tuple]:
    return [(v.shape, v.get_value().dtype) for v in _variables(module)]


def _shares(passes: list[nnx.Module]) -> bool:
    """Return whether a module or variable is in more than one of `passes`."""
    seen: set[int] = set()
    for p in passes:
        ids = {id(n) for n in _nodes(p) if isinstance(n, nnx.Variable | nnx.Pytree)}
        if ids & seen:
            return True
        seen |= ids
    return False


def _stack(passes: list[nnx.Module]) -> nnx.Module:
    """Return the first of `passes`, each of its variables replaced by the stack of
    every pass's, along a leading dimension named `layers`."""
    for vs in zip(*map(_variables, passes), strict=True):
        names = vs[0].get_metadata().get(NAMES) or (None,) * len(vs[0].shape)
        vs[0].set_value(jnp.stack([v.get_value() for v in vs]))
        vs[0].set_metadata(names=(LAYERS, *names))
    return passes[0]


class Scanned(nnx.Module):
    """A repeat of `times` passes run as one module. `layers` is one pass whose
    variables hold every pass's, stacked (``_stack``). A call scans them
    (``stacked.run``), or runs them by the run's pipeline method once ``pipe`` has
    split them over the stages. Where `compute` is set, a training step's cast of the
    model left them as they are (``scanned_variables``), and each pass casts its own
    to it.

    A pass holds `each` residual branches outside any other, or is one (`is_branch`):
    a scan numbers them as its first pass's (``tracing.passes``)."""

    def __init__(
        self, layers: nnx.Module, times: int, each: int = 0, is_branch: bool = False
    ):
        self.times, self.layers = times, layers
        self.each, self.is_branch = each, is_branch
        self.rounds = 0  # the rounds a pipeline takes them in; 0 if none does
        self.compute: DTypeLike | None = None

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        if self.rounds and pipeline.pipelined():
            return self._pipelined(x, taps)
        graphdef, state = nnx.split(self.layers)
        rests = []
        for run in self._runs():
            part = stacked.passes(state, run)
            with tracing.passes(run, self.each, self.is_branch):
                x, rest = stacked.run(graphdef, part, len(run), x, taps, self.compute)
            rests.append(rest)
        nnx.update(self.layers, stacked.joined(rests))
        return x

    def _runs(self) -> list[range]:
        """Split the passes into runs that trace alike: whose branches multiply alike
        (``matmuls.cuts``), and that rematerialize alike (``remat.cuts``)."""
        cuts = [*matmuls.cuts(self.times, self.each), *remat.cuts()]
        return tracing.runs(self.times, cuts)

    def _pipelined(self, x: Stream, taps: Taps) -> Stream:
        """Run the passes by the run's pipeline method, their branches numbered here,
        and they must all multiply alike. Each stage counts its own passes, for remat
        (``pipelined.Stage``)."""
        if matmuls.cuts(self.times, self.each):
            raise ValueError(
                "precision's first and last branches fall inside a pipelined repeat, "
                "whose passes multiply alike: keep them in the repeats before or after"
            )
        layers, passes = self._round()
        with tracing.branches(self.times, self.each, self.is_branch):
            return pipelined.run(layers, passes, self.rounds, x, taps, self.compute)

    def pipe(self, rounds: int) -> None:
        """Split the stacked layers over `stage`, by a dimension named `stages`.

        With one round, each stage holds a contiguous share of the passes: where
        `methods.stages` gives each stage its share, as many as the most any takes
        (``pipeline.spread``), and `times` still counts the passes that run. With more,
        the passes split into `rounds` rounds, each a scanned repeat of its own, and
        each stage holds a share of every round."""
        if rounds == 1:
            self._by_stage()
        else:
            self._by_round(rounds)
        self.rounds = rounds

    def _by_stage(self) -> None:
        """Lay the stacked passes out by stage, as many to a stage as the most any runs
        (``pipeline.spread``)."""
        shares = pipeline.shares(self.times)
        for v in _variables(self.layers):
            names = v.get_metadata()[NAMES][1:]
            v.set_value(pipeline.spread(v.get_value(), shares))
            v.set_metadata(names=(pipeline.STAGES, *names))

    def _by_round(self, rounds: int) -> None:
        """Stack the passes by round, each round a scanned repeat of its own."""
        n = self.times // rounds
        for v in _variables(self.layers):
            names = v.get_metadata()[NAMES][1:]
            v.set_value(v.get_value().reshape(rounds, n, *v.shape[1:]))
            v.set_metadata(names=(LAYERS, pipeline.STAGES, *names))
        self.layers = Scanned(self.layers, n, self.each, self.is_branch)
        self.times, self.each, self.is_branch = rounds, n * self.each, False

    def _round(self) -> tuple[nnx.Module, int]:
        """Return one pass of a pipelined repeat, its variables stacked by round, and
        the number of passes per round."""
        if isinstance(self.layers, Scanned):
            return self.layers.layers, self.layers.times
        return self.layers, self.times


def scanned_variables(root: nnx.Module) -> Callable[[tuple, object], bool]:
    """Return a filter of the variables of `root`'s scanned repeats."""
    found = [p for p, m in nnx.iter_graph(root) if isinstance(m, Scanned)]
    return lambda path, _: any(path[: len(p)] == p for p in found)


def pipe_longest(root: nnx.Module) -> nnx.Module:
    """Pipeline `root`'s longest scanned repeat outside any other (the first of several
    as long) if the mesh in force has stages, and return `root`."""
    if (n := pipeline.stages()) == 1:
        return root
    if not (outer := _outermost(root)):
        raise ValueError(
            f"mesh.stage = {n} pipelines the longest scanned repeat, and the model has "
            "none: repeat its blocks (block^8), and scan them, methods(repeat=scan)"
        )
    longest = max(outer, key=lambda m: m.times)
    longest.pipe(pipeline.rounds(longest.times))
    return root


def _outermost(root: nnx.Module) -> list[Scanned]:
    """Return `root`'s scanned repeats outside any other."""
    found = [(p, m) for p, m in nnx.iter_graph(root) if isinstance(m, Scanned)]
    inside = {p for p, _ in found for q, _ in found if p != q and p[: len(q)] == q}
    return [m for p, m in found if p not in inside]
