"""Run layers whose variables are stacked into one state, as a scanned repeat's are.

``run`` scans them by `jax.lax.scan` where a pass leaves the stream and the taps in
the shapes it found them, and otherwise calls them one at a time. Each pass is
rematerialized as a whole, as the run's `remat` says.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux.architecture import pure, remat
from bux.architecture.registry import Stream, Taps


def run(
    graphdef: nnx.GraphDef, state: nnx.State, times: int, x: Stream, taps: Taps
) -> tuple[Stream, nnx.State]:
    """Run the `times` layers that `graphdef` and the stacked `state` make on `x`.

    Return the stream, and the layers' state other than parameters as they leave it.
    The scan starts at the first layer whose pass is steady (``_steady``): the first,
    or the second where the first starts a tap, as single-pass mhc's first branch
    starts its read. If neither is, as when every pass pushes to a tap, the layers
    run one at a time.
    """
    done: list[nnx.State] = []
    for i in range(times):
        if i < 2 and (carry := _steady(graphdef, _at(state, i), x, taps)) is not None:
            x, rest = _scan(graphdef, _since(state, i), times - i, x, taps, carry)
            return x, _join([*done, rest])
        x, rest = _layer(graphdef, _at(state, i), x, taps)
        done.append(jax.tree.map(lambda a: a[None], rest))
    return x, _join(done)


def _at(state: nnx.State, i: int) -> nnx.State:
    return jax.tree.map(lambda a: a[i], state)


def _since(state: nnx.State, i: int) -> nnx.State:
    return jax.tree.map(lambda a: a[i:], state)


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
        None if changed.logits is None else taps.logits,
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
) -> tuple[Stream, nnx.State]:
    """Run the layers by `jax.lax.scan`, carrying `x` and `carry`, each rematerialized
    as a whole as the run's `remat` says."""
    call = pure.Pure(graphdef, taps)
    layer = remat.checkpoint(
        call, prevent_cse=False, scanned=True
    )  # a scan keeps XLA from merging them

    def step(
        c: tuple[Stream, pure.Changes], s: nnx.State
    ) -> tuple[tuple[Stream, pure.Changes], tuple[nnx.State, list[Float[Array, ""]]]]:
        y, rest, changes, losses = layer(s, *c)
        return (y, changes), (rest, losses)

    (x, changes), (rest, losses) = jax.lax.scan(step, (x, carry), state, times)
    pure.apply(taps, changes, call, [[v[i] for v in losses] for i in range(times)])
    return x, rest
