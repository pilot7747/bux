"""FLOPs, counted from a program, and the peak of the devices that run it, for MFU.

``count`` traces a function on shapes alone, so nothing runs, and adds up its matrix
products and convolutions: 2·m·n·k each, times the length of every scan around them,
the larger branch of a cond, a while loop's body once. So a new layer counts with no
formula of its own. A kernel (a pallas_call, a cuDNN call) hides its FLOPs: trace
inside ``bux.kernels.off``, as ``trainer.step_flops`` does. A product counts once:
what a backward pass computes again, jax.checkpoint's or in ``recomputed``, counts
none.

Attention counts only the keys each query sees, as Megatron counts half of a causal
mask's logits and a flash kernel computes no more. Inside ``attention(window, top_k)``
its products over every key of a query's row (``over_keys``), or over each query's
top_k picks (``over_picks``), count apart, by their `Mask`, as if every query saw
every one; ``Work.of`` then counts them for a batch at the keys its queries see:
those of their own document up to themselves, within the window, and no more than
top_k, a sparse attention's picks. So MFU on packed documents and sparse attention
counts what their masks keep. Other products inside count whole: the projections of
an indexer's queries, say, which it makes as it ranks the keys.
"""

import math
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, NamedTuple

import jax
import numpy as np
from jaxtyping import Array, Int

from bux import jax_internals
from bux.jax_internals import core

_ATTENTION = re.compile(r"attention_mask=(\w+),(\w+)")  # its window and top_k
_KEYS, _PICKS = "over_keys", "over_picks"
_AGAIN = "rematted_computation"  # jax.checkpoint's scope of what it computes again

#: Dense bf16 tensor FLOPs a second of one device, by a name in its device_kind, from
#: NVIDIA's datasheets of the SXM parts. The log's MFU is over these.
PEAK = {
    "B300": 2.25e15,
    "B200": 2.25e15,
    "H200": 989e12,
    "H100": 989e12,
    "A100": 312e12,
}


class Mask(NamedTuple):
    """Which keys of its row a query sees in an attention's products: those of its
    document up to itself, fewer than `window` behind it, and of those `top_k` at
    most. Products over each query's top_k picks alone, not its row's keys, are
    `picks`."""

    window: int | None = None  #: It sees no key this far behind it, or None.
    top_k: int | None = None  #: It sees this many keys at most, or None.
    picks: bool = False  #: Whether the products are over the picks alone.

    def share(self, within: Int[np.ndarray, "b t"]) -> float:
        """Return the share of a product's keys a query sees, on average over the
        queries, `within` the tokens of its document before each: of its row's t
        keys, or of its top_k picks."""
        seen = within + 1
        for most in (self.window, self.top_k):
            if most is not None:
                seen = np.minimum(seen, most)
        keys = self.top_k if self.picks and self.top_k else within.shape[1]
        return float(seen.mean()) / keys


class Work(NamedTuple):
    """A training step's FLOPs: `fixed`, those of every batch, and attention's by
    their Mask, counted as if every query saw every key."""

    fixed: float  #: The FLOPs every batch's step takes.
    #: Attention's FLOPs by their mask, as if every query saw every key.
    attention: dict[Mask, float]

    def of(self, positions: Int[np.ndarray | Array, "b t"]) -> float:
        """Return the FLOPs of a step on rows at `positions`, a document starting
        wherever they stop increasing: attention's at the keys its queries see."""
        within = _within(np.asarray(positions))
        seen = (f * mask.share(within) for mask, f in self.attention.items())
        return self.fixed + sum(seen)

    def one_document(self, t: int) -> float:
        """Return the FLOPs of a step on rows of t tokens of one document each."""
        return self.of(np.arange(t)[None])


def _within(positions: Int[np.ndarray, "b t"]) -> Int[np.ndarray, "b t"]:
    """Return how many tokens of its document come before each token, as
    ``Segments.of`` does on a device."""
    index = np.arange(positions.shape[1])
    starts = np.ones(positions.shape, bool)
    starts[:, 1:] = positions[:, 1:] <= positions[:, :-1]
    first = np.maximum.accumulate(np.where(starts, index, 0), axis=1)
    return index - first


def attention(window: int | None = None, top_k: int | None = None) -> Any:
    """A scope of an attention whose queries see `Mask(window, top_k)` of their keys:
    its products `over_keys` and `over_picks` count at the share of them they see."""
    return jax.named_scope(f"attention_mask={window},{top_k}")


def over_keys() -> Any:
    """A scope of products over every key of each query's row."""
    return jax.named_scope(_KEYS)


def over_picks() -> Any:
    """A scope of products over each query's top_k picks, a sparse attention's."""
    return jax.named_scope(_PICKS)


def recomputed() -> Any:
    """A scope of products a backward pass computes again, as jax.checkpoint names
    them: they count none."""
    return jax.named_scope(_AGAIN)


def count(f: Callable[..., Any], *args: Any) -> Work:
    """Return the FLOPs of `f(*args)`, `args` arrays or their shapes."""
    tally = _tally(jax.make_jaxpr(f)(*args).jaxpr, Scope())
    fixed = tally.pop(None, 0.0)
    return Work(fixed, {mask: f for mask, f in tally.items() if mask is not None})


class Scope(NamedTuple):
    """The scopes an equation is in: the innermost `attention`'s mask, what its
    products run over, `over_keys` or `over_picks`, if it is in either, and whether
    it is `recomputed`."""

    mask: Mask | None = None
    over: str | None = None
    again: bool = False

    def inner(self, e: core.JaxprEqn) -> "Scope":
        """Return the scopes of `e`, in these and its own."""
        mask, over, again = self
        for name in str(e.source_info.name_stack).split("/"):
            if found := _ATTENTION.search(name):
                mask = Mask(_number(found[1]), _number(found[2]))
            elif _KEYS in name or _PICKS in name:
                over = _PICKS if _PICKS in name else _KEYS
            again |= _AGAIN in name
        return Scope(mask, over, again)

    def counted(self) -> Mask | None:
        """Return the mask the products of these scopes count at: None for whole."""
        if self.mask is None or self.over is None:
            return None
        return self.mask._replace(picks=self.over == _PICKS)


def _number(text: str) -> int | None:
    return None if text == "None" else int(text)


Tally = dict[Mask | None, float]  # FLOPs by the mask they count at; None for whole


def _tally(jaxpr: core.Jaxpr, outer: Scope) -> Tally:
    """Return the FLOPs of the matrix products and convolutions in `jaxpr`, by the
    mask they count at, in the scopes `outer` and their own."""
    return _added(_equation(e, outer.inner(e)) for e in jaxpr.eqns)


def _added(tallies: Iterable[Tally]) -> Tally:
    """Return the sum of `tallies`, mask by mask."""
    total: Tally = {}
    for tally in tallies:
        for mask, f in tally.items():
            total[mask] = total.get(mask, 0.0) + f
    return total


def _equation(e: core.JaxprEqn, scope: Scope) -> Tally:
    if scope.again:
        return {}
    match e.primitive.name:
        case "dot_general":
            (_, contracting), (_, batch) = e.params["dimension_numbers"]
            return {scope.counted(): _product(e, (*contracting, *batch))}
        case "ragged_dot_general":
            numbers = e.params["ragged_dot_dimension_numbers"]
            (_, contracting), (_, batch) = numbers.dot_dimension_numbers
            shared = (*contracting, *batch, *numbers.rhs_group_dimensions)
            return {scope.counted(): _product(e, shared)}
        case "conv_general_dilated":
            rhs = _shape(e.invars[1])  # out features, in features / groups, window
            each = math.prod(rhs) / rhs[e.params["dimension_numbers"].rhs_spec[0]]
            return {scope.counted(): 2 * math.prod(_shape(e.outvars[0])) * each}
        case "scan":
            length = jax_internals.scan_options(e)["length"]
            body = _tally(jax_internals.scan_body(e).jaxpr, scope)
            return {mask: length * f for mask, f in body.items()}
        case "cond":
            branches = [_tally(b.jaxpr, scope) for b in jax_internals.cond_branches(e)]
            return max(branches, key=lambda tally: sum(tally.values()))
    # jit, shard_map, custom_vjp, ...
    return _added(_tally(j, scope) for j in _inner(e.params))


def _inner(params: dict[str, Any]) -> Iterator[core.Jaxpr]:
    """The jaxprs an equation calls."""
    for value in params.values():
        for v in value if isinstance(value, tuple | list) else (value,):
            if isinstance(v, core.ClosedJaxpr):
                yield v.jaxpr
            elif isinstance(v, core.Jaxpr):
                yield v


def _product(e: core.JaxprEqn, rhs_shared: Sequence[int]) -> float:
    """2 for every pair of the lhs's elements and the rhs's free ones: those not in
    `rhs_shared`, which the lhs's elements meet one at a time."""
    rhs = _shape(e.invars[1])
    free = math.prod(n for d, n in enumerate(rhs) if d not in rhs_shared)
    return 2 * math.prod(_shape(e.invars[0])) * free


def _shape(v: Any) -> tuple[int, ...]:
    """The shape of an equation's input or output."""
    return v.aval.shape


def peak(kind: str) -> float | None:
    """Return the dense bf16 FLOPs a second of a device of `kind` (its device_kind),
    or None if it is not in PEAK."""
    return next((p for name, p in PEAK.items() if name in kind), None)
