"""What a training step observes of the model's forward, on the steps a run's
``observe`` setting picks (`bux.train.observe`): the residual stream after each
branch outside any other (`stream`), each MoE's routing, and a sample of each
attention's logits. A layer observes a value in one line::

    observe.stat(taps, "router/max_load", load.max() / load.mean())

Only a training forward traced inside `use` collects it, and the trainer traces one
step so; elsewhere it does nothing, so that the step a run takes on other steps holds
none of it. A value leaves the forward as a loss the layer reports, of weight 0 and
no gradient (`~bux.architecture.registry.report`), so it leaves scans, remat and
pipelines as those do. Its name says the branch it was observed in (`branch`), which
the objective numbers among those of its name, in the order they run:
``router/max_load`` of the fourth branch named ``moe`` is ``router/max_load/moe.03``
(`named`).
"""

import contextlib
import re
from collections.abc import Callable, Generator, Iterable
from contextvars import ContextVar
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Real

from bux.architecture.registry import Loss, Stream, Taps, report, training
from bux.lang import ir

#: What joins a value's name to its branch's in a reported loss: ``act/rms@attn``.
MARK = "@"
#: The bins of a histogram (`histogram`): ``log2 |x|`` from LOW (and below, zeros too)
#: to HIGH (and above), a bin each.
LOW, HIGH = -24, 15
#: The most entries a histogram of the stream takes: an even sample of its positions.
#: Each entry is compared with every bin, and all of a 1.3B's took 40 ms a B300 step.
SAMPLE = 2**22
#: The layer types a branch is not named for: a branch ``[norm attn]`` is ``attn``.
NORMS = frozenset({"rmsnorm", "layernorm", "dyt", "scale", "act", "dropout"})


class Settings(NamedTuple):
    """What a forward traced inside `use` observes."""

    histograms: bool  #: Histograms of the stream after each branch, as well.


_SETTINGS = ContextVar[Settings | None]("observe", default=None)
_BRANCH = ContextVar[str | None]("observed branch", default=None)


@contextlib.contextmanager
def use(histograms: bool = False) -> Generator[None]:
    """Observe the training forwards traced inside the block.

    Args:
        histograms: Whether to take histograms of the stream too, which cost more.
    """
    token = _SETTINGS.set(Settings(histograms))
    try:
        yield
    finally:
        _SETTINGS.reset(token)


def active() -> Settings | None:
    """Return what the forward being traced observes; None if it observes nothing."""
    return _SETTINGS.get()


def observing(taps: Taps | None) -> bool:
    """Return whether the forward of `taps` observes: a training forward inside
    `use`."""
    return active() is not None and training(taps)


@contextlib.contextmanager
def branch(name: str | None) -> Generator[None]:
    """Name what the block observes for the branch `name`; None keeps the name of the
    branch outside it, or ``model`` outside any."""
    token = _BRANCH.set(name or _BRANCH.get())
    try:
        yield
    finally:
        _BRANCH.reset(token)


def stat(taps: Taps | None, name: str, value: Real[Array, ""]) -> None:
    """Observe `value` as the metric `name` of the branch tracing, in float32 and
    with no gradient; nothing unless the forward observes (`observing`).

    Args:
        taps: The forward pass's taps.
        name: The metric, ``group/what``: ``act/rms``, ``router/max_load``.
        value: A scalar of this forward's batch.
    """
    if observing(taps):
        value = jax.lax.stop_gradient(jnp.asarray(value, jnp.float32))
        report(taps, f"{name}{MARK}{_BRANCH.get() or 'model'}", 0.0, value)


def stream(taps: Taps, x: Stream, y: Stream) -> None:
    """Observe the stream after a branch, `y`, and what the branch added to `x`:

    - ``act/rms``: y's root mean square;
    - ``act/update_rms``: that of y - x, the branch's update;
    - ``act/max_abs``: y's largest magnitude, which outliers raise;
    - ``act/kurtosis``: the kurtosis of y's features' mean squares, 1 when every
      feature is alike and large when a few carry the stream (outlier features);
    - ``act/hist``: with histograms, the share of y's entries in each bin of
      ``log2 |y|`` (`histogram`), of an even sample of its positions (`SAMPLE`).
    """
    if not observing(taps):
        return
    y32 = y.astype(jnp.float32)
    stat(taps, "act/rms", _rms(y32))
    if x.shape == y.shape:  # attnres's stream grows by a branch's output instead
        stat(taps, "act/update_rms", _rms(y32 - x.astype(jnp.float32)))
    stat(taps, "act/max_abs", jnp.max(jnp.abs(y32)))
    stat(taps, "act/kurtosis", feature_kurtosis(y32))
    settings = active()
    if settings is not None and settings.histograms:
        histogram(taps, "act/hist", sampled(y, 1, SAMPLE))


def _rms(x: Float[Array, "*s"]) -> Float[Array, ""]:
    return jnp.sqrt(jnp.mean(jnp.square(x)))


def feature_kurtosis(x: Float[Array, "*s d"]) -> Float[Array, ""]:
    """Return the kurtosis of `x`'s features' mean squares, ``mean(m²) / mean(m)²``
    with ``m`` each feature's mean square over the rest: 1 when they are all alike."""
    m = jnp.mean(jnp.square(x), axis=tuple(range(x.ndim - 1)))
    return jnp.mean(jnp.square(m)) / jnp.maximum(jnp.square(jnp.mean(m)), 1e-30)


def histogram(taps: Taps, name: str, x: Float[Array, "*s"]) -> None:
    """Observe the share of `x`'s entries in each bin of ``log2 |x|`` (`bins`). Each bin
    is a value of its own, ``name[e]``, which `bux.train.observe` puts together
    again."""
    shares = jnp.mean(bins(x), axis=tuple(range(x.ndim)))
    for e, share in zip(range(LOW, HIGH + 1), shares, strict=True):
        stat(taps, f"{name}[{e}]", share)


def bins(x: Float[Array, "*s"]) -> Float[Array, "*s n"]:
    """Return which bin of ``log2 |x|`` each entry of `x` falls in, one-hot, from
    `LOW` to `HIGH`: bin e holds ``|x|`` in ``[2^e, 2^(e+1))``, the first everything
    below, zeros too, and the last everything above."""
    exponent = jnp.clip(jnp.floor(jnp.log2(jnp.abs(x.astype(jnp.float32)))), LOW, HIGH)
    edges = jnp.arange(LOW, HIGH + 1, dtype=jnp.float32)
    return (exponent[..., None] == edges).astype(jnp.float32)


def sampled(x: Float[Array, "*s"], axis: int, most: int) -> Float[Array, "*t"]:
    """Return an even sample of `x` along `axis`, every so many of its entries there,
    of no more than about `most` entries in all."""
    if x.ndim <= axis:
        return x
    every = min(-(-x.size // most), x.shape[axis])
    return jax.lax.slice_in_dim(x, 0, None, every, axis) if every > 1 else x


_PASSES = ContextVar[str | None]("observed passes", default=None)


@contextlib.contextmanager
def passes(name: str | None) -> Generator[None]:
    """Observe each pass of a repeat run inside the block as the branch `name`, which
    each is (`run_pass`); None for passes that are not branches."""
    token = _PASSES.set(name)
    try:
        yield
    finally:
        _PASSES.reset(token)


def run_pass(module: Callable[[Stream, Taps], Stream], x: Stream, taps: Taps) -> Stream:
    """Run a pass of a repeat on `x`: as a branch, named and its stream observed
    (`stream`), if `passes` says its repeat's passes are branches."""
    name = _PASSES.get()
    if name is None:
        return module(x, taps)
    with passes(None), branch(name):  # not the passes of a repeat inside it
        y = module(x, taps)
        stream(taps, x, y)
    return y


# --- Names --------------------------------------------------------------------------


def branch_name(node: ir.Residual) -> str:
    """Return the name a branch is observed by: its first layer that is not a norm
    (`NORMS`), by its definition's name, else its type; else the branch's own
    definition, else ``branch``."""
    layers = [n for n in ir.walk(node.body) if isinstance(n, ir.Layer)]
    main = next((n for n in layers if n.config.type not in NORMS), None)
    main = main or (layers[0] if layers else None)
    if main is not None:
        return main.label or main.config.type
    return node.label or "branch"


def item_name(node: ir.Node, branch: bool) -> str | None:
    """Return the name of an item of a sequence: a branch's (`branch_name`), or a
    layer's, by its definition, else its type; None for the rest, whose items are
    named themselves."""
    if branch and isinstance(node, ir.Residual):
        return branch_name(node)
    if isinstance(node, ir.Layer):
        return node.label or node.config.type
    return None


def layer_key(name: str, index: int, count: int) -> str:
    """Return the key of the `index`-th of `count` layers of a name: ``attn.03``."""
    return f"{name}.{index:0{max(2, len(str(count - 1)))}d}"


def is_observed(name: str) -> bool:
    """Return whether a reported loss of `name` is an observed value (`stat`)."""
    return MARK in name


def named(observed: Iterable[Loss]) -> dict[str, Float[Array, ""]]:
    """Return the values `stat` observed, in the order they ran, by their metrics'
    names: each value's, then its branch's name and count among those of its name
    and value, ``act/rms/attn.03``."""
    observed = list(observed)
    counts: dict[str, int] = {}
    for loss in observed:
        counts[loss.name] = counts.get(loss.name, 0) + 1
    seen: dict[str, int] = {}
    out = {}
    for loss in observed:
        i = seen[loss.name] = seen.get(loss.name, -1) + 1
        what, where = loss.name.split(MARK, 1)
        out[f"{what}/{layer_key(where, i, counts[loss.name])}"] = loss.value
    return out


#: A histogram bin's metric: its histogram's name, its bin, and its layer.
BIN = re.compile(r"^(?P<name>.+)\[(?P<bin>-?\d+)\]/(?P<layer>[^/]+)$")
