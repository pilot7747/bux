"""Pipeline parallelism over `stage`: the methods of `methods.pipeline`.

With ``mesh.stage`` above 1, the longest scanned repeat outside any other is
pipelined (bux.architecture.graph): its stacked layers split over `stage`, a share a
stage, and the rest of the model stays whole on every stage. A method runs them on
the ``methods.microbatches`` microbatches of the batch's rows, in a shard_map manual
over `stage` alone, each stage on its own devices:

    a lax.scan over R·M + S − 1 ticks: M microbatches through S stages, R rounds;
    each tick, a stage's share of a round on a microbatch,
    and each stage's output to the next by a ppermute, the last's to the first.

The first stage takes each microbatch from a buffer of its own, where the last
stage's outputs come back: there a circular round starts again, and there the stream
leaves, summed over the stages to all of them. ``gpipe`` sends every microbatch
through the stages once; ``circular`` round them ``methods.rounds`` times, each stage
holding a share of every round's layers, so that a stage idles (S − 1) / (R·M + S − 1)
of a step (``bubble``).

A method is ``(stage, params, rest, x, rows) -> (y, rest, losses)``. ``stage`` runs a
stage's share of a round on a microbatch; ``params`` and ``rest`` are the layers'
parameters and other state, each leaf (rounds, layers a round, ...); ``x`` is the
stream, and ``rows`` the taps of its rows. It returns the stream, the state as the
layers leave it, and the losses each layer reports, the mean over microbatches,
(rounds, layers a round, reports).
"""

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int, PyTree, Shaped

from bux.parallel import methods
from bux.parallel.mesh import Active, active

AXIS = "stage"
STAGES = "stages"  # the stacked dimension of a pipelined repeat, split over stage
# A stage's share of a round on a microbatch: (params, rest, x, rows) -> (y, rest,
# losses), its losses (layers, reports).
Stage = Callable[..., tuple[Shaped[Array, "b t *r"], PyTree, Float[Array, "l n"]]]
Out = tuple[Shaped[Array, "b t *r"], PyTree, Float[Array, "rounds l n"]]


def stages() -> int:
    """The stages of the mesh in force: 1 outside ``use``."""
    on = active()
    return 1 if on is None else on.size(AXIS)


def pipelined() -> bool:
    """Whether the run in force pipelines its longest scanned repeat: then the
    pipeline takes ``methods.microbatches``, and a step does not accumulate gradients
    over them."""
    return stages() > 1


def bubble(stages: int, microbatches: int, rounds: int = 1) -> float:
    """The fraction of a step each stage idles: (S − 1) / (R·M + S − 1)."""
    return (stages - 1) / (rounds * microbatches + stages - 1)


def _on() -> Active:
    on = active()
    if on is None:
        raise RuntimeError("a pipeline runs on the mesh `bux.parallel.use` sets")
    return on


def rounds(passes: int) -> int:
    """The rounds the run in force pipelines a repeat of `passes` passes in. Raise
    unless they split evenly over the stages of every round, and circular rounds have
    a microbatch a stage to keep the stages busy."""
    on = _on()
    s, c = on.size(AXIS), on.methods
    if passes % (s * c.rounds):
        raise ValueError(
            f"mesh.stage = {s} pipelines the longest scanned repeat, of {passes} "
            f"passes, in methods.rounds = {c.rounds}: make its passes a multiple "
            f"of {s * c.rounds}"
        )
    if c.rounds > 1 and c.microbatches < s:
        raise ValueError(
            f"a circular pipeline takes a microbatch a stage at least: "
            f"methods.microbatches = {c.microbatches} < mesh.stage = {s}"
        )
    return c.rounds


@methods.pipeline.register("gpipe")
def gpipe(
    stage: Stage,
    params: PyTree,
    rest: PyTree,
    x: Shaped[Array, "b t *r"],
    rows: PyTree,
) -> Out:
    """Each microbatch through the stages once: they idle (S − 1) / (M + S − 1)."""
    if (n := _on().methods.rounds) > 1:
        raise ValueError(
            f"gpipe goes round the stages once, not methods.rounds = {n} times: "
            "use methods(pipeline=circular)"
        )
    return circular(stage, params, rest, x, rows)


@methods.pipeline.register("circular")
def circular(
    stage: Stage,
    params: PyTree,
    rest: PyTree,
    x: Shaped[Array, "b t *r"],
    rows: PyTree,
) -> Out:
    """Each microbatch round the stages `methods.rounds` times, a round's layers split
    over them: they idle (S − 1) / (R·M + S − 1)."""
    on = _on()
    clock = Clock(on.size(AXIS), on.methods.microbatches, on.methods.rounds)
    return clock.run(stage, params, rest, x, rows)


class Clock(NamedTuple):
    """M microbatches through S stages, R rounds: k ticks in, a stage takes the k-th
    of the R·M runs, microbatch k mod M of round k // M. Stage s is s ticks in at
    tick s."""

    stages: int
    microbatches: int
    rounds: int

    @property
    def ticks(self) -> int:
        return self.rounds * self.microbatches + self.stages - 1

    def at(
        self, k: Int[Array, ""]
    ) -> tuple[Bool[Array, ""], Int[Array, ""], Int[Array, ""]]:
        """Of a stage `k` ticks in: whether it works, on which microbatch, of which
        round."""
        runs = self.rounds * self.microbatches
        busy = (k >= 0) & (k < runs)
        k = jnp.clip(k, 0, runs - 1)
        return busy, k % self.microbatches, k // self.microbatches

    def run(
        self,
        stage: Stage,
        params: PyTree,
        rest: PyTree,
        x: Shaped[Array, "b t *r"],
        rows: PyTree,
    ) -> Out:
        """The pipeline, each stage on its own devices (``_device``); then the first
        stage's buffer, where the stream ends, to every stage."""
        s, m = self.stages, self.microbatches
        if x.shape[0] % m:
            raise ValueError(
                f"{x.shape[0]} rows do not split into methods.microbatches = {m}"
            )
        rows = jax.tree.map(lambda a: _micro(a, m), rows)
        own = P(AXIS)  # a stage's, on its devices
        io, rest, losses = jax.shard_map(
            functools.partial(_device, self, stage),
            in_specs=(own, own, P(), P()),
            out_specs=own,
            axis_names={AXIS},
            check_vma=False,  # else nested shard_maps' gradients fail to lower
        )(_by_stage(params, s), _by_stage(rest, s), _micro(x, m), rows)
        y = jnp.sum(jnp.where(_first(io), io, 0), 0)  # the first stage's buffer
        return _whole(y), _by_round(rest), _by_round(losses)

    def mean(
        self, losses: Float[Array, "t l n"], rounds: Int[Array, " t"]
    ) -> Float[Array, "r l n"]:
        """A stage's losses of each round, the mean over microbatches."""
        hot = jax.nn.one_hot(rounds, self.rounds, dtype=losses.dtype)
        return jnp.einsum("tr,tln->rln", hot, losses) / self.microbatches


def _device(
    clock: Clock,
    stage: Stage,
    params: PyTree,
    rest: PyTree,
    xs: Shaped[Array, "m n t *r"],
    rows: PyTree,
) -> tuple[Shaped[Array, "1 m n t *r"], PyTree, Float[Array, "1 rounds l k"]]:
    """A stage's part, on its devices: every tick, its share of a round on a
    microbatch (``_tick``). The first stage's buffer ends with the stream."""
    me = jax.lax.axis_index(AXIS)
    params, rest = _at(params, 0), _at(rest, 0)  # this stage's
    tick = functools.partial(_tick, clock, stage, params, rows, me)
    carry = (jnp.zeros_like(xs[0]), xs, rest)
    (_, io, rest), (losses, of) = jax.lax.scan(tick, carry, jnp.arange(clock.ticks))
    return io[None], jax.tree.map(lambda a: a[None], rest), clock.mean(losses, of)[None]


def _tick(
    clock: Clock,
    stage: Stage,
    params: PyTree,
    rows: PyTree,
    me: Int[Array, ""],
    carry: tuple[Shaped[Array, "n t *r"], Shaped[Array, "m n t *r"], PyTree],
    t: Int[Array, ""],
) -> tuple[
    tuple[Shaped[Array, "n t *r"], Shaped[Array, "m n t *r"], PyTree],
    tuple[Float[Array, "l k"], Int[Array, ""]],
]:
    """Stage `me`'s run at tick `t`: the first takes its microbatch from its buffer,
    the others what the stage before gave them; its state as it was where it is idle.
    Then its output on to the next stage, the last's into the first's buffer."""
    shift, io, rest = carry
    busy, micro, of = clock.at(t - me)
    x = jnp.where(me == 0, _at(io, t % clock.microbatches), shift)
    own = _at(rest, of)
    y, new, losses = stage(_at(params, of), own, x, _at(rows, micro))
    new = jax.tree.map(lambda a, b: jnp.where(busy, a, b), new, own)
    rest = _put(rest, of, new)
    shift = _rotate(y)
    back, done = (t - clock.stages + 1) % clock.microbatches, t >= clock.stages - 1
    io = _put(io, back, jnp.where(done, shift, _at(io, back)))
    return (shift, io, rest), (jnp.where(busy, losses, 0), of)


def _rotate(y: Shaped[Array, "n t *r"]) -> Shaped[Array, "n t *r"]:
    """A stage's output to the next stage, the last's to the first."""
    n = jax.lax.axis_size(AXIS)
    return jax.lax.ppermute(y, AXIS, [(i, (i + 1) % n) for i in range(n)])


def _first(a: Shaped[Array, "s ..."]) -> Bool[Array, "s ..."]:
    """Whether each slice of `a` is the first stage's, broadcastable to `a`."""
    return (jnp.arange(a.shape[0]) == 0).reshape(-1, *(1,) * (a.ndim - 1))


def _micro(a: Shaped[Array, "b t *r"], m: int) -> Shaped[Array, "m n t *r"]:
    """Rows i·m + j of `a` as microbatch j, so that each splits as the rows do."""
    return a.reshape(a.shape[0] // m, m, *a.shape[1:]).swapaxes(0, 1)


def _whole(a: Shaped[Array, "m n t *r"]) -> Shaped[Array, "b t *r"]:
    """The rows of microbatches, as `_micro` took them."""
    return a.swapaxes(0, 1).reshape(a.shape[0] * a.shape[1], *a.shape[2:])


# By dynamic slices, not gathers and scatters: XLA fails to split a scatter over Auto
# axes inside the shard_map, as a gather's gradient is.
def _at(tree: PyTree, i: Int[Array, ""] | int) -> PyTree:
    """Slice `i` of every leaf of `tree`."""
    return jax.tree.map(lambda a: jax.lax.dynamic_index_in_dim(a, i, 0, False), tree)


def _put(tree: PyTree, i: Int[Array, ""], new: PyTree) -> PyTree:
    """`tree`, slice `i` of every leaf `new`'s."""
    return jax.tree.map(
        lambda a, v: jax.lax.dynamic_update_index_in_dim(a, v, i, 0), tree, new
    )


def _by_stage(tree: PyTree, s: int) -> PyTree:
    """Each leaf (rounds, layers, ...) as (s, rounds, layers / s, ...): a stage's
    share of every round."""

    def split(a: Shaped[Array, "r n *_"]) -> Shaped[Array, "s r l *_"]:
        r, n = a.shape[:2]
        return a.reshape(r, s, n // s, *a.shape[2:]).swapaxes(0, 1)

    return jax.tree.map(split, tree)


def _by_round(tree: PyTree) -> PyTree:
    """Each leaf (s, rounds, layers / s, ...) as (rounds, layers, ...)."""

    def join(a: Shaped[Array, "s r l *_"]) -> Shaped[Array, "r n *_"]:
        s, r, n = a.shape[:3]
        return a.swapaxes(0, 1).reshape(r, s * n, *a.shape[3:])

    return jax.tree.map(join, tree)
