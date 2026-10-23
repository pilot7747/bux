"""Pipeline parallelism over `stage`: the methods of `methods.pipeline`.

With ``mesh.stage`` above 1, the longest scanned repeat outside any other is
pipelined (bux.architecture.pipelined): its stacked layers split over `stage`, one
share per stage, and the rest of the model stays whole on every stage. A method runs
them on the ``methods.microbatches`` microbatches of the batch's rows, in a shard_map
manual over `stage` alone, each stage on its own devices:

    a lax.scan over R·M + S − 1 ticks: M microbatches through S stages, R rounds;
    at each tick, every stage runs its share of a round on one microbatch,
    and a ppermute sends each stage's output to the next, the last's to the first.

The first stage takes each microbatch from a buffer of its own, where the last
stage's outputs come back: there a circular round starts again, and there the stream
leaves, summed over the stages to all of them. ``gpipe`` sends every microbatch
through the stages once; ``circular`` sends them round ``methods.rounds`` times, each
stage holding a share of every round's layers, so that a stage idles
(S − 1) / (R·M + S − 1) of a step (``bubble``).

A method is ``(stage, params, rest, x, rows) -> (y, rest, losses)``. ``stage`` runs a
stage's share of a round on one microbatch; ``params`` and ``rest`` are the layers'
parameters and other state, each leaf (rounds, layers per round, ...); ``x`` is the
stream, and ``rows`` the tokens and documents of its rows. It returns the stream, the
state as the layers leave it, and the losses each layer reports, averaged over
microbatches, (rounds, layers per round, reports).
"""

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int, PyTree, Shaped

from bux.parallel import gather, methods
from bux.parallel.mesh import Active, active

AXIS = "stage"
STAGES = "stages"  # the stacked dimension of a pipelined repeat, split over stage
# A stage's share of a round, run on one microbatch: (params, rest, x, rows) -> (y,
# rest, losses), with losses (layers, reports).
Stage = Callable[..., tuple[Shaped[Array, "b t *r"], PyTree, Float[Array, "l n"]]]
Out = tuple[Shaped[Array, "b t *r"], PyTree, Float[Array, "rounds l n"]]


def stages() -> int:
    """Return the number of stages of the mesh in force: 1 outside ``use``."""
    on = active()
    return 1 if on is None else on.size(AXIS)


def pipelined() -> bool:
    """Return whether the run in force pipelines its longest scanned repeat. Then the
    pipeline takes the ``methods.microbatches``, and a step does not accumulate
    gradients over them."""
    return stages() > 1


def bubble(stages: int, microbatches: int, rounds: int = 1) -> float:
    """Return the fraction of a step each stage idles: (S − 1) / (R·M + S − 1)."""
    return (stages - 1) / (rounds * microbatches + stages - 1)


def _on() -> Active:
    on = active()
    if on is None:
        raise RuntimeError("a pipeline runs on the mesh `bux.parallel.use` sets")
    return on


def rounds(passes: int) -> int:
    """Return how many rounds the run in force pipelines a repeat of `passes` passes
    in. Raise unless the passes split evenly over the stages of every round, and a
    circular pipeline has at least one microbatch per stage to keep the stages busy."""
    on = _on()
    s, settings = on.size(AXIS), on.methods
    if passes % (s * settings.rounds):
        raise ValueError(
            f"mesh.stage = {s} pipelines the longest scanned repeat, of {passes} "
            f"passes, in methods.rounds = {settings.rounds}: make its passes a "
            f"multiple of {s * settings.rounds}"
        )
    if settings.rounds > 1 and settings.microbatches < s:
        raise ValueError(
            f"a circular pipeline takes a microbatch a stage at least: "
            f"methods.microbatches = {settings.microbatches} < mesh.stage = {s}"
        )
    return settings.rounds


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
    """Send each microbatch round the stages `methods.rounds` times, each round's
    layers split over them: they idle (S − 1) / (R·M + S − 1)."""
    on = _on()
    clock = Clock(on.size(AXIS), on.methods.microbatches, on.methods.rounds)
    return clock.run(stage, params, rest, x, rows)


class Clock(NamedTuple):
    """The schedule of M microbatches through S stages in R rounds. Stage s starts at
    tick s, and k ticks after it starts, it runs the k-th of the R·M runs: microbatch
    k mod M of round k // M."""

    stages: int
    microbatches: int
    rounds: int

    @property
    def ticks(self) -> int:
        return self.rounds * self.microbatches + self.stages - 1

    def at(
        self, k: Int[Array, ""]
    ) -> tuple[Bool[Array, ""], Int[Array, ""], Int[Array, ""]]:
        """Return, for a stage `k` ticks after it starts, whether it is busy, and
        which microbatch and round it runs."""
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
        """Run the pipeline, each stage on its own devices (``_device``), and return
        the first stage's buffer, where the stream ends up, on every stage."""
        s, m = self.stages, self.microbatches
        if x.shape[0] % m:
            raise ValueError(
                f"{x.shape[0]} rows do not split into methods.microbatches = {m}"
            )
        rows = jax.tree.map(lambda a: _micro(a, m), rows)
        apart = gather.rows_apart()  # the axes that split the rows, if manual here too
        by_stage = P(AXIS)  # each stage's own, on its devices
        by_row = P(None, apart or None)  # each device's rows of every microbatch
        io, rest, losses = jax.shard_map(
            functools.partial(_device, self, stage, apart),
            in_specs=(by_stage, by_stage, by_row, by_row),
            out_specs=(P(AXIS, None, apart or None), by_stage, by_stage),
            axis_names={AXIS, *apart},
            check_vma=False,  # else nested shard_maps' gradients fail to lower
        )(_by_stage(params, s), _by_stage(rest, s), _micro(x, m), rows)
        y = jnp.sum(jnp.where(_first(io), io, 0), 0)  # the first stage's buffer
        return _whole(y), _by_round(rest), _by_round(losses)

    def mean(
        self, losses: Float[Array, "t l n"], rounds: Int[Array, " t"]
    ) -> Float[Array, "r l n"]:
        """Return a stage's losses in each round, averaged over microbatches."""
        hot = jax.nn.one_hot(rounds, self.rounds, dtype=losses.dtype)
        return jnp.einsum("tr,tln->rln", hot, losses) / self.microbatches


def _device(
    clock: Clock,
    stage: Stage,
    apart: tuple[str, ...],
    params: PyTree,
    rest: PyTree,
    xs: Shaped[Array, "m n t *r"],
    rows: PyTree,
) -> tuple[Shaped[Array, "1 m n t *r"], PyTree, Float[Array, "1 rounds l k"]]:
    """Run one stage on its devices: at every tick, its share of a round on one
    microbatch (``_tick``). The first stage's buffer ends up holding the stream.

    Manual over the axes `apart` too, it runs this device's rows alone: its gradients
    are its own, which the shard_map's transpose sums over `apart` as they leave."""
    index = jax.lax.axis_index(AXIS)
    params, before = _at(params, 0), _at(rest, 0)  # this stage's
    tick = functools.partial(_tick, clock, stage, params, rows, index)
    carry = (jnp.zeros_like(xs[0]), xs, gather.own_streams(before, apart))
    (_, io, rest), (losses, r) = jax.lax.scan(tick, carry, jnp.arange(clock.ticks))
    rest = gather.keys_of(before, rest)
    losses = clock.mean(losses, r)
    losses = jax.lax.pmean(losses, apart) if apart else losses
    return io[None], jax.tree.map(lambda a: a[None], rest), losses[None]


def _tick(
    clock: Clock,
    stage: Stage,
    params: PyTree,
    rows: PyTree,
    index: Int[Array, ""],
    carry: tuple[Shaped[Array, "n t *r"], Shaped[Array, "m n t *r"], PyTree],
    t: Int[Array, ""],
) -> tuple[
    tuple[Shaped[Array, "n t *r"], Shaped[Array, "m n t *r"], PyTree],
    tuple[Float[Array, "l k"], Int[Array, ""]],
]:
    """Run stage `index` at tick `t`. The first stage takes its microbatch from its
    buffer, the others what the stage before sent them; an idle stage keeps its state.
    Then send its output on to the next stage, the last's into the first's buffer."""
    shift, io, rest = carry
    busy, micro, r = clock.at(t - index)
    x = jnp.where(index == 0, _at(io, t % clock.microbatches), shift)
    before = _at(rest, r)
    y, new, losses = stage(_at(params, r), before, x, _at(rows, micro))
    new = jax.tree.map(lambda a, b: jnp.where(busy, a, b), new, before)
    rest = _put(rest, r, new)
    shift = _rotate(y)
    back, done = (t - clock.stages + 1) % clock.microbatches, t >= clock.stages - 1
    io = _put(io, back, jnp.where(done, shift, _at(io, back)))
    return (shift, io, rest), (jnp.where(busy, losses, 0), r)


def _rotate(y: Shaped[Array, "n t *r"]) -> Shaped[Array, "n t *r"]:
    """Send each stage's output to the next stage, the last's to the first."""
    n = jax.lax.axis_size(AXIS)
    return jax.lax.ppermute(y, AXIS, [(i, (i + 1) % n) for i in range(n)])


def _first(a: Shaped[Array, "s ..."]) -> Bool[Array, "s ..."]:
    """Return whether each slice of `a` is the first stage's, broadcastable to `a`."""
    return (jnp.arange(a.shape[0]) == 0).reshape(-1, *(1,) * (a.ndim - 1))


def _micro(a: Shaped[Array, "b t *r"], m: int) -> Shaped[Array, "m n t *r"]:
    """Split `a` into m microbatches, row i·m + j into microbatch j, so that each
    splits over the devices as the rows do."""
    return a.reshape(a.shape[0] // m, m, *a.shape[1:]).swapaxes(0, 1)


def _whole(a: Shaped[Array, "m n t *r"]) -> Shaped[Array, "b t *r"]:
    """Join microbatches back into rows: the inverse of `_micro`."""
    return a.swapaxes(0, 1).reshape(a.shape[0] * a.shape[1], *a.shape[2:])


# By dynamic slices, not gathers and scatters: XLA fails to split a scatter over Auto
# axes inside the shard_map, as a gather's gradient is.
def _at(tree: PyTree, i: Int[Array, ""] | int) -> PyTree:
    """Return slice `i` of every leaf of `tree`."""
    return jax.tree.map(lambda a: jax.lax.dynamic_index_in_dim(a, i, 0, False), tree)


def _put(tree: PyTree, i: Int[Array, ""], new: PyTree) -> PyTree:
    """Return `tree` with slice `i` of every leaf replaced by `new`'s."""
    return jax.tree.map(
        lambda a, v: jax.lax.dynamic_update_index_in_dim(a, v, i, 0), tree, new
    )


def _by_stage(tree: PyTree, s: int) -> PyTree:
    """Reshape each leaf from (rounds, layers, ...) to (s, rounds, layers / s, ...):
    for each stage, its share of every round."""

    def split(a: Shaped[Array, "r n *_"]) -> Shaped[Array, "s r l *_"]:
        r, n = a.shape[:2]
        return a.reshape(r, s, n // s, *a.shape[2:]).swapaxes(0, 1)

    return jax.tree.map(split, tree)


def _by_round(tree: PyTree) -> PyTree:
    """Reshape each leaf from (s, rounds, layers / s, ...) back to (rounds, layers,
    ...)."""

    def join(a: Shaped[Array, "s r l *_"]) -> Shaped[Array, "r n *_"]:
        s, r, n = a.shape[:3]
        return a.swapaxes(0, 1).reshape(r, s * n, *a.shape[3:])

    return jax.tree.map(join, tree)
