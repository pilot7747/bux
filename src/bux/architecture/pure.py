"""Calls of a module as functions of arrays alone, for jax.checkpoint and jax.lax.scan.

A module changes the taps it is called with, but jax's transforms take functions of
arrays. ``Pure`` calls the module on a copy of the taps instead and returns what it
changed (``Changes``), and ``apply`` writes that back into the taps.
"""

from typing import Any, NamedTuple

import jax
from flax import nnx
from jaxtyping import Array, Float

from bux.architecture.logits import HeadOutput
from bux.architecture.registry import Loss, Stream, Taps


class Changes(NamedTuple):
    """What a call changed of the taps: every head's output, once one ran; the
    single-pass read, once a branch set it; and the stacks it pushed to or popped."""

    heads: list[HeadOutput] | None
    pending_read: Float[Array, "b t n"] | None
    stacks: dict[str, list]


NOTHING = Changes(None, None, {})


class Pure:
    """The call of the module that `graphdef` and a state make, as a function of arrays
    alone: ``(state, x, carry) -> (y, rest, changes, losses)``.

    It reads `taps`, with what `carry` holds in place of theirs, and returns the stream,
    the module's state other than parameters (an MoE's router bias), the ``Changes`` it
    made and those of `carry`, and the values of the losses it reports. As it traces,
    `reports` keeps the losses' names and weights, and `nextn_runs` the nextn count it
    leaves. `taps` itself does not change.
    """

    def __init__(self, graphdef: nnx.GraphDef, taps: Taps):
        self.graphdef, self.taps = graphdef, taps
        self.reports: list[tuple[str, float]] = []
        self.nextn_runs = taps.nextn_runs

    def __call__(
        self, state: nnx.State, x: Stream, carry: Changes
    ) -> tuple[Stream, nnx.State, Changes, list[Float[Array, ""]]]:
        local = _copy(self.taps, carry)
        module = nnx.merge(self.graphdef, state)
        y = module(x, local)
        losses = local.losses
        self.reports = [(loss.name, loss.weight) for loss in losses]
        self.nextn_runs = local.nextn_runs
        rest = nnx.state(module, nnx.Not(nnx.Param))
        changes = _changes(self.taps, local, carry)
        return y, rest, changes, [loss.value for loss in losses]


def _copy(taps: Taps, carry: Changes) -> Taps:
    """Copy `taps`, with `carry`'s values in place of its own, and no losses yet."""
    stacks = {k: list(v) for k, v in (dict(taps) | carry.stacks).items()}
    local = Taps(
        stacks, tokens=taps.tokens, segments=taps.segments, training=taps.training
    )
    local.heads = list(taps.heads if carry.heads is None else carry.heads)
    local.pending_read = (
        taps.pending_read if carry.pending_read is None else carry.pending_read
    )
    local.nextn_runs = taps.nextn_runs
    return local


def _changes(taps: Taps, local: Taps, carry: Changes) -> Changes:
    """Return what a call on `local`, a copy of `taps`, changed, and what it carried."""
    heads = carry.heads is not None or not same(taps.heads, local.heads)
    read = carry.pending_read is not None or local.pending_read is not taps.pending_read
    return Changes(
        local.heads if heads else None,
        local.pending_read if read else None,
        {
            k: v
            for k, v in local.items()
            if k in carry.stacks or not same(taps.get(k), v)
        },
    )


def same(before: list | None, after: list) -> bool:
    """Return whether `after` holds the very items `before` does."""
    return (
        before is not None
        and len(before) == len(after)
        and all(a is b for a, b in zip(before, after))
    )


def apply(
    taps: Taps, changes: Changes, call: Pure, losses: list[list[Float[Array, ""]]]
) -> None:
    """Write what pure calls of `call` changed back into `taps`: the ``Changes``, the
    nextn count, and the losses each call reported, call by call."""
    taps.update(changes.stacks)
    if changes.heads is not None:
        taps.heads = changes.heads
    if changes.pending_read is not None:
        taps.pending_read = changes.pending_read
    taps.nextn_runs = call.nextn_runs
    if not call.reports:
        return
    for values in losses:
        taps.losses.extend(
            Loss(name, weight, v) for (name, weight), v in zip(call.reports, values)
        )


def signature(tree: Any) -> tuple:
    """Return the structure of `tree`, and the shape and dtype of every array in it."""
    leaves, structure = jax.tree.flatten(jax.eval_shape(lambda t: t, tree))
    return structure, [(a.shape, a.dtype, a.weak_type) for a in leaves]
