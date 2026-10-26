"""The steps a run observes, every ``observe.every``: the training step with what it
measures of the model by layer, on top of its update, which other steps leave out.

On those steps the trainer runs a step of its own (``Trainer.observes``), compiled
once more, which makes the same update and also measures:

- what the forward observes, by branch (`bux.architecture.observe`): the stream's
  ``act/*``, each attention's ``attn/*`` and each MoE's ``router/*``;
- each head's ``z_loss/t+k`` (`bux.train.objective.Objective.observed`);
- by layer, the norms of the gradient, ``grad/norm``, before clipping; of the
  parameters before the update, ``param/norm``; and of the update, ``update/norm``,
  with its ratio to the parameters', ``update/ratio`` (`params`);
- with histograms, those of ``log2 |x|`` of the stream after each branch, ``act/hist``,
  and of each layer's gradient, ``grad/hist`` (`Histogram`).

A layer is a branch outside any other or an item of the model's graph that is not a
branch, as the embedding and the head; its key is its name and its count among
those of its name, in the order they run: ``attn.03`` is the fourth branch named
``attn`` (`bux.architecture.observe.item_name`), and its parameters are those of the
branch, its norm's too. A scanned repeat's passes count one by one, as they run.

A step of microbatches observes each and takes their mean. A scheduled pipeline's
forward stops at its pipelined repeat for the layers before it, which it numbers but
does not observe (``Trainer``'s ``_Cut``).

The other steps compile and run as they would without the setting.
"""

import contextlib
from collections.abc import Generator, Iterator
from contextvars import ContextVar
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jaxtyping import Array, Float, Shaped

from bux.architecture import observe
from bux.architecture.graph import Sequence
from bux.architecture.registry import Loss
from bux.architecture.repeats.repeat import Scanned
from bux.parallel import pipeline
from bux.train.objective import Metrics
from bux.train.sinks import Histogram, Scalars


class Layer(NamedTuple):
    """Where a layer's arrays are in a model: under how many stacked dimensions, one
    a scanned repeat around it, and its key in each of the slots they make, in the
    order they flatten; None for a copy that a stage of an uneven pipeline holds and
    skips (``pipeline.spread``)."""

    stacked: int  #: The stacked dimensions leading its arrays.
    keys: tuple[str | None, ...]  #: Its key in each slot: ``attn.03``.


#: Each layer of a model, by its module's path.
Layers = dict[tuple, Layer]


EDGES = np.arange(observe.LOW, observe.HIGH + 2, dtype=np.float64)


# --- Which layer each array is of ---------------------------------------------------


def layers(model: nnx.Module) -> Layers:
    """Return each layer of `model`, by its module's path, with its keys: ``attn.03``.
    Call it under the run's mesh, which says how a pipeline lays out its stages'
    passes."""
    found = list(_layers(model, (), 0, 0))
    counts: dict[str, int] = {}
    for *_, name in found:
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    seen: dict[str, int] = {}
    slots: dict[tuple, dict[int, str | None]] = {}
    stacked: dict[tuple, int] = {}
    for path, depth, slot, name in found:
        key = None
        if name is not None:
            i = seen[name] = seen.get(name, -1) + 1
            key = observe.layer_key(name, i, counts[name])
        slots.setdefault(path, {})[slot] = key
        stacked[path] = depth
    return {
        path: Layer(stacked[path], tuple(keys[i] for i in range(len(keys))))
        for path, keys in slots.items()
    }


def _layers(
    node: Any, path: tuple, depth: int, slot: int, runs: bool = True
) -> Iterator[tuple[tuple, int, int, str | None]]:
    """Yield each layer under `node` as it runs: its path, the scanned repeats around
    it, its slot of the dimensions they stack, and its name; then those of the slots
    a stage skips, of no name, as of a `node` that `runs` not."""
    if isinstance(node, Scanned):
        width, inner = _width(node), (*path, "layers")
        ran = _slots(node)
        for at in [*ran, *sorted(set(range(width)) - set(ran))]:
            each = runs and at in ran
            if node.name is not None:
                yield inner, depth + 1, slot * width + at, node.name if each else None
            else:
                yield from _layers(
                    node.layers, inner, depth + 1, slot * width + at, each
                )
    elif isinstance(node, Sequence):
        for i, (item, name) in enumerate(zip(node.items, node.names, strict=True)):
            at = (*path, "items", i)
            if name is not None:
                yield at, depth, slot, name if runs else None
            else:
                yield from _layers(item, at, depth, slot, runs)
    elif isinstance(node, nnx.Module | nnx.List):
        for key, child in nnx.iter_children(node):
            yield from _layers(child, (*path, key), depth, slot, runs)


def _width(node: Scanned) -> int:
    """Return the slots of a scanned repeat's stacked dimension: its passes, or as
    many to a stage as the most any runs, for an uneven pipeline (``spread``)."""
    shares = _shares(node)
    return node.times if shares is None else len(shares) * max(shares)


def _slots(node: Scanned) -> list[int]:
    """Return the slot of each of a scanned repeat's passes, in the order they run."""
    shares = _shares(node)
    if shares is None:
        return list(range(node.times))
    most = max(shares)
    return [s * most + j for s, n in enumerate(shares) for j in range(n)]


def _shares(node: Scanned) -> tuple[int, ...] | None:
    """Return the passes each stage of an uneven pipeline holds of `node`, laid out
    by ``spread``; None where they are stacked as they run."""
    if node.rounds != 1 or not pipeline.pipelined():
        return None
    shares = pipeline.shares(node.times)
    return None if min(shares) == max(shares) else shares


# --- Norms by layer -----------------------------------------------------------------


def _flat(tree: Any) -> dict[tuple, Shaped[Array, "..."]]:
    """Return the arrays of `tree`, a state or the pure tree of one, by their paths."""

    def is_leaf(x: object) -> bool:
        return isinstance(x, nnx.Variable)

    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree, is_leaf=is_leaf):
        value = leaf.get_value() if isinstance(leaf, nnx.Variable) else leaf
        if isinstance(value, jax.Array):
            out[tuple(_key(k) for k in path)] = value
    return out


def _key(entry: Any) -> Any:
    for attr in ("key", "name", "idx"):
        if hasattr(entry, attr):
            return getattr(entry, attr)
    return entry


def _owner(where: Layers, path: tuple) -> Layer:
    """Return the layer that holds the array at `path`: ``model`` for one of none."""
    for n in range(len(path), -1, -1):
        if path[:n] in where:
            return where[path[:n]]
    return Layer(0, ("model",))


def by_layer(
    where: Layers, tree: Any, per_entry: Any = jnp.square, sample: int | None = None
) -> dict[str, Float[Array, "*n"]]:
    """Return the sum of `per_entry` over each layer's entries of `tree`, by its key:
    a scalar where `per_entry` gives one an entry, else what it gives, summed. With
    `sample`, of an even sample of each array's entries of a layer, about that many
    of them, along its longest dimension: a routed expert's last is its 2
    projections."""
    sums: dict[str, Float[Array, "*n"]] = {}
    for path, x in _flat(tree).items():
        layer = _owner(where, path)
        if sample is not None and x.ndim > layer.stacked:
            own = x.shape[layer.stacked :]
            axis = layer.stacked + own.index(max(own))
            x = observe.sampled(x, axis, sample * max(1, len(layer.keys)))
        values = per_entry(x.astype(jnp.float32))
        summed = jnp.sum(values, axis=tuple(range(layer.stacked, x.ndim)))
        each = summed.reshape(-1, *summed.shape[layer.stacked :])
        for key, value in zip(layer.keys, each, strict=True):
            if key is not None:
                sums[key] = sums.get(key, 0) + value
    return sums


def norms(where: Layers, tree: Any) -> dict[str, Float[Array, ""]]:
    """Return each layer's norm of `tree`'s arrays, by key."""
    return {key: jnp.sqrt(s) for key, s in by_layer(where, tree).items()}


#: The most entries of each of a layer's arrays a histogram takes: an even sample.
#: Each entry is compared with every bin, and all of a 1.3B's took 90 ms a B300 step.
SAMPLE = 2**20


def histograms(where: Layers, tree: Any) -> dict[str, Float[Array, " n"]]:
    """Return the share of each layer's entries of `tree` in each bin of ``log2 |x|``
    (`~bux.architecture.observe.bins`), of a sample of each array (`SAMPLE`), by
    key."""
    counts = by_layer(where, tree, observe.bins, SAMPLE)
    return {k: c / jnp.maximum(c.sum(), 1) for k, c in counts.items()}


def params(
    where: Layers,
    before: nnx.State | dict,
    grads: nnx.State | None,
    updates: Any,
    histogram: bool = False,
) -> dict[str, Float[Array, ""]]:
    """Return what an observed step measures of the parameters, by layer: the norms
    of the gradient, of the parameters `before` the update and of the update, and
    the update's over the parameters'; with `histogram`, the gradient's histograms.

    Args:
        where: Which layer each array is of (`layers`).
        before: The parameters as the step found them.
        grads: Their gradient, before clipping, if the step recorded one.
        updates: What the optimizer added to them, if it recorded it.
        histogram: Whether to take the gradient's histograms too.

    Returns:
        The metrics, ``grad/norm/attn.03``, ``param/norm/attn.03``, ...
    """
    out, size = {}, norms(where, before)
    out |= {f"param/norm/{k}": v for k, v in size.items()}
    if grads is not None:
        out |= {f"grad/norm/{k}": v for k, v in norms(where, grads).items()}
    if updates is not None:
        for k, v in norms(where, updates).items():
            out[f"update/norm/{k}"] = v
            out[f"update/ratio/{k}"] = v / jnp.maximum(size[k], 1e-30)
    if histogram and grads is not None:
        for k, shares in histograms(where, grads).items():
            for e, share in zip(range(observe.LOW, observe.HIGH + 1), shares):
                out[f"grad/hist[{e}]/{k}"] = share
    return out


def numbered(parts: dict[str, list[str]], part: str, losses: list[Loss]) -> Metrics:
    """Return what the layers of one part of a model cut in parts observed,
    `losses`, named as a forward of the whole model numbers them
    (`bux.architecture.observe.named`): after what the parts before it observed.

    Args:
        parts: The names of what each part's layers observe, in the order they run.
        part: The part that observed `losses`.
        losses: What it observed, in the order it ran.
    """
    order = list(parts)
    at = order.index(part)
    head = [n for p in order[:at] for n in parts[p]]
    tail = [n for p in order[at + 1 :] for n in parts[p]]
    zero = jnp.zeros((), jnp.float32)
    stand_ins = ([Loss(n, 0.0, zero) for n in names] for names in (head, tail))
    before, after = stand_ins
    named = observe.named([*before, *losses, *after])
    return dict(list(named.items())[len(head) : len(head) + len(losses)])


# --- What a step records as it traces -----------------------------------------------


class Recorded:
    """What an observed step's trace hands its trainer: the gradient its step took
    (``Trainer.gradients``), and the update its optimizer made (`recording`)."""

    def __init__(self) -> None:
        self.grads: nnx.State | None = None
        self.updates: Any = None

    def wrap(self, tx: optax.GradientTransformation) -> optax.GradientTransformation:
        """Return `tx`, which records the updates it makes here."""

        def update(
            updates: Any, state: Any, params: Any = None, **extra: Any
        ) -> tuple[Any, Any]:
            updates, state = tx.update(updates, state, params, **extra)
            self.updates = updates
            return updates, state

        return optax.GradientTransformationExtraArgs(tx.init, update)


_RECORDED = ContextVar[Recorded | None]("recorded", default=None)


@contextlib.contextmanager
def observing(histograms: bool = False) -> Generator[Recorded]:
    """Observe the step traced inside the block: its forward
    (`bux.architecture.observe.use`), and what it takes and makes (`Recorded`)."""
    recorded = Recorded()
    token = _RECORDED.set(recorded)
    try:
        with observe.use(histograms):
            yield recorded
    finally:
        _RECORDED.reset(token)


def record_grads(grads: nnx.State) -> None:
    """Record the gradient a step takes, if it is observed."""
    if (recorded := _RECORDED.get()) is not None:
        recorded.grads = grads


# --- On the host ---------------------------------------------------------------------


def histograms_of(scalars: Scalars) -> tuple[dict[str, float], dict[str, Histogram]]:
    """Split `scalars` into the rest and the histograms whose bins they hold
    (``act/hist[-3]/attn.03``), each named without its bin: ``act/hist/attn.03``."""
    rest: dict[str, float] = {}
    bins: dict[str, dict[int, float]] = {}
    for name, value in scalars.items():
        if (m := observe.BIN.match(name)) is None:
            rest[name] = value
        else:
            key = f"{m['name']}/{m['layer']}"
            bins.setdefault(key, {})[int(m["bin"])] = value
    found = {}
    for key, shares in bins.items():
        each = [shares.get(e, 0.0) for e in range(observe.LOW, observe.HIGH + 1)]
        found[key] = Histogram(EDGES, np.asarray(each))
    return rest, found
