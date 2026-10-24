"""Pipeline parallelism over `stage`: the methods of `methods.pipeline`.

With ``mesh.stage`` above 1, the longest scanned repeat outside any other is
pipelined (bux.architecture.repeats.pipelined): its stacked layers split over `stage`,
one share per stage (``shares``), and the rest of the model stays whole on every
stage. A method runs them on the ``methods.microbatches`` microbatches of the batch's
rows, in a shard_map manual over `stage` alone, each stage on its own devices:

    a lax.scan over R·M + S − 1 ticks: M microbatches through S stages, R rounds;
    at each tick, every stage runs its share of a round on one microbatch,
    and a ppermute sends each stage's output to the next, the last's to the first.

The first stage takes each microbatch from a buffer of its own, where the last
stage's outputs come back: there a circular round starts again, and there the stream
leaves, summed over the stages to all of them. ``gpipe`` sends every microbatch
through the stages once; ``circular`` sends them round ``methods.rounds`` times, each
stage holding a share of every round's layers, so that a stage idles
(S − 1) / (R·M + S − 1) of a step (``bubble``).

Where the step computes its rows apart (``gather.apart``), the shard_map is manual
over the axes that split them too: each device runs its own rows, and a stage's
gradients reduce once, as they leave it, not once a tick.

A method is ``(stage, params, rest, x, rows) -> (y, rest, losses)``. ``stage`` runs a
stage's share of a round on one microbatch; ``params`` and ``rest`` are the layers'
parameters and other state, each leaf (rounds, layers per round, ...); ``x`` is the
stream, and ``rows`` the tokens and documents of its rows. It returns the stream, the
state as the layers leave it, and the losses each layer reports, averaged over
microbatches, (rounds, layers per round, reports).
"""

import functools
from collections.abc import Callable, Iterator
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int, PyTree, Shaped

from bux.parallel import gather, methods, schedule
from bux.parallel.mesh import Active, active
from bux.train.config import MethodsConfig

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


def bubble(
    stages: int, microbatches: int, rounds: int = 1, method: str = "gpipe"
) -> float:
    """Return the fraction of a step each stage idles: (S − 1) / (R·M + S − 1), or
    for a method run by a table, as its table says (``schedule.Table.bubble``)."""
    if method in schedule.TABLES and stages > 1:
        return schedule.TABLES[method](stages, microbatches).bubble()
    return (stages - 1) / (rounds * microbatches + stages - 1)


def _on() -> Active:
    on = active()
    if on is None:
        raise RuntimeError("a pipeline runs on the mesh `bux.parallel.use` sets")
    return on


def shares(passes: int) -> tuple[int, ...]:
    """Return how many of a round's `passes` each stage runs: as `methods.stages` gives
    them, or as evenly as they split (``even``). A run's training fills in the split
    that balances a schedule's stages (``balanced``, bux.train.stages)."""
    on, n = active(), stages()
    given = None if on is None else on.methods.stages
    return even(passes, n) if given is None else tuple(given)


def even(passes: int, stages: int) -> tuple[int, ...]:
    """Return `passes` split over `stages` as evenly as they go, the first stages
    taking one more where they do not divide."""
    each, more = divmod(passes, stages)
    return tuple(each + (s < more) for s in range(stages))


class Work(NamedTuple):
    """What a scheduled pipeline's stages compute for a microbatch, in FLOPs: every
    stage `each` a pass, the first also the model before the pipelined repeat
    (`before`: the embedding), and the last the model after it, to the loss
    (`after`: the head)."""

    before: float
    each: float
    after: float

    def of(self, split: tuple[int, ...]) -> list[float]:
        """Return each stage's work, of `split` passes each."""
        work = [n * self.each for n in split]
        work[0] += self.before
        work[-1] += self.after
        return work


def balanced(passes: int, stages: int, work: Work) -> tuple[int, ...]:
    """Return the split of `passes` over `stages`, a pass a stage at least, whose
    slowest stage does the least `work`; of those as slow, the most even, and then
    the one heaviest at the front."""

    def slowest_then_spread(split: tuple[int, ...]) -> tuple[float, float]:
        each = work.of(split)
        return max(each), sum(w * w for w in each)

    return min(_splits(passes, stages), key=slowest_then_spread)


def _splits(passes: int, stages: int) -> Iterator[tuple[int, ...]]:
    """Yield the splits of `passes` over `stages` that give each a pass, the most
    passes on the first stage first. The stages between the first and the last work
    alike, so they only split what those two leave, as evenly as it goes."""
    if stages == 1:
        yield (passes,)
        return
    between = stages - 2
    for first in reversed(range(1, passes)):
        for last in range(1, passes - first + 1):
            rest = passes - first - last
            if between == 0 and rest == 0:
                yield first, last
            elif between > 0 and rest >= between:
                yield first, *even(rest, between), last


def rounds(passes: int) -> int:
    """Return how many rounds the run in force pipelines a repeat of `passes` passes
    in. Raise unless each stage runs a pass at least, in one round as `shares` gives
    them, or in a circular pipeline's rounds evenly, with a microbatch per stage at
    least to keep the stages busy."""
    on = _on()
    s, settings = on.size(AXIS), on.methods
    if passes < s:
        raise ValueError(
            f"mesh.stage = {s} pipelines the longest scanned repeat, of {passes} "
            f"passes: give it a pass a stage at least"
        )
    if settings.rounds == 1:
        _check_shares(shares(passes), s, passes)
    else:
        _check_circular(passes, s, settings)
    return settings.rounds


def _check_shares(given: tuple[int, ...], stages: int, passes: int) -> None:
    """Raise unless `given` gives each of the `stages` its passes, `passes` in all."""
    if len(given) != stages or sum(given) != passes:
        raise ValueError(
            f"methods.stages = {list(given)} splits {sum(given)} passes over "
            f"{len(given)} stages, and the pipelined repeat has {passes} passes over "
            f"mesh.stage = {stages}"
        )


def _check_circular(passes: int, stages: int, settings: MethodsConfig) -> None:
    """Raise unless a circular pipeline can send `passes` round `stages` in
    `settings.rounds` rounds."""
    rounds, total = settings.rounds, stages * settings.rounds
    if settings.stages is not None:
        raise ValueError("methods.stages splits one round, not methods.rounds")
    if passes % total:
        raise ValueError(
            f"mesh.stage = {stages} pipelines the longest scanned repeat, of {passes} "
            f"passes, in methods.rounds = {rounds}: make its passes a multiple of "
            f"{total}"
        )
    if settings.pipeline in schedule.TABLES:
        raise ValueError(
            f"methods(pipeline={settings.pipeline}) goes round the stages once, not "
            f"methods.rounds = {rounds} times: use methods(pipeline=circular)"
        )
    if settings.microbatches < stages:
        raise ValueError(
            f"a circular pipeline takes a microbatch a stage at least: "
            f"methods.microbatches = {settings.microbatches} < mesh.stage = {stages}"
        )


def spread(
    stacked: Shaped[Array, "l ..."], shares: tuple[int, ...]
) -> Shaped[Array, "m ..."]:
    """Return a stacked variable of `shares` of its passes for each stage, laid out
    as many to a stage as the most any runs: each stage's own, then copies of its
    first, which it skips (``layers_run``)."""
    most = max(shares)
    if min(shares) == most:
        return stacked
    start, parts = 0, []
    for n in shares:
        own = stacked[start : start + n]
        parts += [own, jnp.broadcast_to(own[:1], (most - n, *own.shape[1:]))]
        start += n
    return jnp.concatenate(parts)


def layers_run(shares: tuple[int, ...]) -> Bool[np.ndarray, "s l"]:
    """Return which of the layers it holds each stage runs, of `shares` of the
    passes (``spread``)."""
    return np.arange(max(shares))[None, :] < np.array(shares)[:, None]


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
        )(split_by_stage(params, s), split_by_stage(rest, s), _micro(x, m), rows)
        y = jnp.sum(jnp.where(_first(io), io, 0), 0)  # the first stage's buffer
        return _whole(y), join_stages(rest), join_stages(losses)

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
    params, before = at(params, 0), at(rest, 0)  # this stage's
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
    x = jnp.where(index == 0, at(io, t % clock.microbatches), shift)
    before = at(rest, r)
    y, new, losses = stage(at(params, r), before, x, at(rows, micro))
    new = jax.tree.map(lambda a, b: jnp.where(busy, a, b), new, before)
    rest = put(rest, r, new)
    shift = _rotate(y)
    back, done = (t - clock.stages + 1) % clock.microbatches, t >= clock.stages - 1
    io = put(io, back, jnp.where(done, shift, at(io, back)))
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
def at(tree: PyTree, i: Int[Array, ""] | int) -> PyTree:
    """Return slice `i` of every leaf of `tree`."""
    return jax.tree.map(lambda a: jax.lax.dynamic_index_in_dim(a, i, 0, False), tree)


def put(tree: PyTree, i: Int[Array, ""], new: PyTree) -> PyTree:
    """Return `tree` with slice `i` of every leaf replaced by `new`'s."""
    return jax.tree.map(
        lambda a, v: jax.lax.dynamic_update_index_in_dim(a, v, i, 0), tree, new
    )


def split_by_stage(tree: PyTree, s: int) -> PyTree:
    """Reshape each leaf from (rounds, layers, ...) to (s, rounds, layers / s, ...):
    for each stage, its share of every round."""

    def split(a: Shaped[Array, "r n *_"]) -> Shaped[Array, "s r l *_"]:
        r, n = a.shape[:2]
        return a.reshape(r, s, n // s, *a.shape[2:]).swapaxes(0, 1)

    return jax.tree.map(split, tree)


def join_stages(tree: PyTree) -> PyTree:
    """Reshape each leaf from (s, rounds, layers / s, ...) back to (rounds, layers,
    ...)."""

    def join(a: Shaped[Array, "s r l *_"]) -> Shaped[Array, "r n *_"]:
        s, r, n = a.shape[:3]
        return a.swapaxes(0, 1).reshape(r, s * n, *a.shape[3:])

    return jax.tree.map(join, tree)
