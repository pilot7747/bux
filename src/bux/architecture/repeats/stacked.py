"""Run layers whose variables are stacked into one state, as a scanned repeat's are.

``run`` scans them by `jax.lax.scan` where a pass leaves the stream and the taps in
the shapes it found them, and otherwise calls them one at a time. Each pass is
rematerialized as a whole, as the run's `remat` says, and casts its own parameters to
the dtype a step computes in (``cast``). With the run's `remat(first=n)`, the passes
after the first n run in a scan of their own, which rematerializes none.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from bux.architecture.registry import Stream, Taps
from bux.architecture.repeats import pure, remat


def run(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    times: int,
    x: Stream,
    taps: Taps,
    compute: DTypeLike | None = None,
    start: int = 0,
) -> tuple[Stream, nnx.State]:
    """Run the `times` layers that `graphdef` and the stacked `state` make on `x`, each
    with its parameters cast to `compute`, if given (``cast``): passes `start` on of
    their repeat, as the run's `remat(first=n)` counts them.

    Return the stream, and the layers' state other than parameters as they leave it.
    The scan starts at the first layer whose pass is steady (``_steady``): the first,
    or the second where the first starts a tap, as single-pass mhc's first branch
    starts its read. If neither is, as when every pass pushes to a tap, the layers
    run one at a time.
    """
    done: list[nnx.State] = []
    for i in range(times):
        layer = cast(_at(state, i), compute)
        if i < 2 and (carry := _steady(graphdef, layer, x, taps)) is not None:
            passes = (start + i, start + times)
            x, scanned = _scans(graphdef, state, passes, x, taps, carry, compute, start)
            return x, _join([*done, *scanned])
        with remat.at_pass(start + i):
            x, rest = _layer(graphdef, layer, x, taps)
        done.append(jax.tree.map(lambda a: a[None], rest))
    return x, _join(done)


def _scans(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    passes: tuple[int, int],
    x: Stream,
    taps: Taps,
    carry: pure.Changes,
    compute: DTypeLike | None,
    start: int = 0,
) -> tuple[Stream, list[nnx.State]]:
    """Scan passes `passes[0]` to `passes[1]` of their repeat, a scan for each run of
    them that rematerializes alike (``remat.spans``), starting from `carry`; `state`
    holds their repeat's passes from `start` on."""
    rests = []
    for first, stop in remat.spans(*passes):
        if first > passes[0]:  # from the taps the last scan left
            layer = cast(_at(state, first - start), compute)
            again = _steady(graphdef, layer, x, taps)
            assert again is not None, "a pass that was steady is still"
            carry = again
        with remat.at_pass(first):
            layers = _between(state, first - start, stop - start)
            x, rest = _scan(graphdef, layers, stop - first, x, taps, carry, compute)
        rests.append(rest)
    return x, rests


def cast(state: nnx.State, dtype: DTypeLike | None) -> nnx.State:
    """Return `state` with its floating-point parameters in `dtype`, if given. A scan
    casts each layer's as it runs it: a copy of all the stacked ones, cast before it,
    would stay through a training step, and so would one of their gradients."""
    if dtype is None:
        return state

    def one(v: nnx.Variable) -> nnx.Variable:
        x = v.get_value()
        if isinstance(v, nnx.Param) and jnp.issubdtype(x.dtype, jnp.floating):
            return v.replace(x.astype(dtype))
        return v

    return jax.tree.map(one, state, is_leaf=lambda v: isinstance(v, nnx.Variable))


def _at(state: nnx.State, i: int) -> nnx.State:
    return jax.tree.map(lambda a: a[i], state)


def _between(state: nnx.State, start: int, stop: int) -> nnx.State:
    return jax.tree.map(lambda a: a[start:stop], state)


def _join(parts: list[nnx.State]) -> nnx.State:
    """Concatenate stacked states along their stacked dimension."""
    if len(parts) == 1:
        return parts[0]
    return jax.tree.map(lambda *a: jnp.concatenate(a), *parts)


def _steady(
    graphdef: nnx.GraphDef, layer: nnx.State, x: Stream, taps: Taps
) -> pure.Changes | None:
    """Return what a scan of passes like `layer` carries: the values in `taps` of what
    one pass changes. None if the pass changes the shapes of those or of `x`, or runs
    a nextn, which looks one token further each time."""
    call = pure.Pure(graphdef, taps)
    with remat.off():
        y, _, changed, _ = jax.eval_shape(call, layer, x, pure.NOTHING)
    if (
        pure.signature(y) != pure.signature(x)
        or not changed.stacks.keys() <= taps.keys()
        or call.nextn_runs != taps.nextn_runs
    ):
        return None
    carry = pure.Changes(
        None if changed.heads is None else taps.heads,
        None if changed.pending_read is None else taps.pending_read,
        {k: taps[k] for k in changed.stacks},
    )
    return carry if pure.signature(changed) == pure.signature(carry) else None


def _layer(
    graphdef: nnx.GraphDef, layer: nnx.State, x: Stream, taps: Taps
) -> tuple[Stream, nnx.State]:
    """Run one layer, rematerialized as a whole as the run's `remat` says."""
    call = pure.Pure(graphdef, taps)
    y, rest, changes, losses = remat.checkpoint(call, prevent_cse=True)(
        layer, x, pure.NOTHING
    )
    pure.apply(taps, changes, call, [losses])
    return y, rest


def _scan(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    times: int,
    x: Stream,
    taps: Taps,
    carry: pure.Changes,
    compute: DTypeLike | None,
) -> tuple[Stream, nnx.State]:
    """Run the layers by `jax.lax.scan`, carrying `x` and `carry`, each cast to
    `compute` and rematerialized as a whole as the run's `remat` says."""
    call = pure.Pure(graphdef, taps)

    def cast_call(
        s: nnx.State, x: Stream, carry: pure.Changes
    ) -> tuple[Stream, nnx.State, pure.Changes, list[Float[Array, ""]]]:
        return call(cast(s, compute), x, carry)

    # A scan keeps XLA from merging them.
    layer = remat.checkpoint(cast_call, prevent_cse=False, scanned=True)

    def step(
        c: tuple[Stream, pure.Changes], s: nnx.State
    ) -> tuple[tuple[Stream, pure.Changes], tuple[nnx.State, list[Float[Array, ""]]]]:
        y, rest, changes, losses = layer(s, *c)
        return (y, changes), (rest, losses)

    (x, changes), (rest, losses) = jax.lax.scan(step, (x, carry), state, times)
    pure.apply(taps, changes, call, [[v[i] for v in losses] for i in range(times)])
    return x, rest
