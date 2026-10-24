"""Microbatches that overlap: each one's forward runs beside the backward of the one
before it, a piece of the model at a time, as Megatron's combined 1F1B runs them
without a pipeline (`methods(interleave=true)`).

One after another (``Trainer._accumulate``), a microbatch's collectives each wait on
the compute just before them, and what a layer's data flow leaves nothing beside stays
exposed. Here the model is cut into pieces (``pieces``): the items of its graph, each
pass of an unrolled repeat a piece of its own, the first taking the tokens and the
last, the head, ending in the loss. A piece's forward is a ``jax.vjp``, whose
residuals wait in a slot (``Slots``) until its backward:

    forward of microbatch 0
    a lax.scan, for i in 0 .. n-2:
        tick t, for t in 0 .. r-1:  piece t's forward of i+1, piece r-1-t's backward of i
        the head's loss of i+1, and its gradient
    backward of microbatch n-1

Each tick ends in an optimization barrier on the stream and its gradient, so XLA's
scheduler runs one half's collectives beside the other's compute, and no tick before
its turn: the backward frees the top pieces' residuals as the forward fills the
bottom ones, and a step holds about one microbatch's. The barrier also takes the
gradients of the tick before, whose reduction then runs beside this tick, and which
are the last to read the slot this tick's forward writes. Its program is about twice
that of a step in a row: a forward and a backward on each side of the loop.
"""

import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, Float, Int, Shaped

from bux import jax_internals
from bux.architecture import Model
from bux.architecture.common.segments import Segments
from bux.architecture.graph import Sequence, run_branch
from bux.architecture.logits import HeadOutput
from bux.architecture.precision import matmuls
from bux.architecture.registry import Loss, Stream, Taps
from bux.architecture.repeats import remat, tracing
from bux.architecture.repeats.repeat import Passes
from bux.train.config import PrecisionConfig
from bux.train.data import Batch

Leaf = Shaped[Array, "..."]
Flat = dict[tuple, Leaf]  # a state's arrays, by their paths in the model's
Metrics = dict[str, Float[Array, ""]]
Buffers = list[list[Leaf]]  # for each kind of residuals, an array of slots of each
Signature = tuple[tuple[tuple[int, ...], Any], ...]  # residuals' shapes and dtypes
Vjp = Callable[..., Any]  # a jax.vjp function: a pytree of its residuals
Call = Callable[[nnx.Module, Stream, Taps], Stream]


class Hooks(NamedTuple):
    """What the schedule takes of the trainer: a piece's module for a training
    forward, made from its states (cast to the step's dtype), and a microbatch's
    share of the step's loss and metrics, from its heads and the losses its layers
    report, given its share of the batch's targets and of its rows."""

    module: Callable[[nnx.GraphDef, nnx.State, nnx.State], nnx.Module]
    loss: Callable[
        [list[HeadOutput], Batch, list[Loss], Float[Array, ""], float],
        tuple[Float[Array, ""], Metrics],
    ]


# --- The model in pieces --------------------------------------------------------------


class Piece:
    """A piece of a model: its module's graphdef and states, whose arrays a step
    holds by their paths in the whole model's state, and how its module runs (`call`:
    as a residual branch, or as a pass of a repeat). Once traced, `reports` holds the
    names and weights of the losses it reports."""

    def __init__(self, module: nnx.Module, paths: dict[int, tuple], call: Call):
        self.graphdef, self.params, self.rest = nnx.split(module, nnx.Param, ...)
        self.paths = {p: paths[id(v)] for p, v in _variables(module)}
        self.call = call
        self.reports: list[tuple[str, float]] = []

    def state(self, template: nnx.State, values: Flat) -> nnx.State:
        """Return `template`, one of the piece's states, holding the step's `values`."""
        flat = nnx.to_flat_state(template)
        return nnx.from_flat_state(
            [(p, v.replace(values[self.paths[p]])) for p, v in flat]
        )

    def flat(self, state: nnx.State) -> Flat:
        """Return the arrays of `state`, one of the piece's, by their model paths."""
        return {self.paths[p]: v.get_value() for p, v in nnx.to_flat_state(state)}

    def losses(self, values: list[Float[Array, ""]]) -> list[Loss]:
        """Return the losses the piece reported, of `values`."""
        return [Loss(n, w, v) for (n, w), v in zip(self.reports, values, strict=True)]


def pieces(model: Model) -> list[Piece]:
    """Cut `model` into its pieces: the items of its graph in a row, those of an
    unrolled repeat each a pass. A variable two pieces share, as a tied head's, is
    held once, at its path in the model."""
    graph = model.graph
    if not isinstance(graph, Sequence):
        raise TypeError(
            "methods(interleave=true) cuts a model into the items of its graph, "
            f"which is a {type(graph).__name__}, not a sequence"
        )
    paths = {id(v): p for p, v in _variables(model)}
    found: list[Piece] = []
    for item, branch in zip(graph.items, graph.branches, strict=True):
        if isinstance(item, Passes):
            passes = enumerate(zip(item.items, item.branches, strict=True))
            found += [Piece(m, paths, _pass(i, b)) for i, (m, b) in passes]
        else:
            found.append(Piece(item, paths, _item(branch)))
    return found


def _variables(module: nnx.Module) -> list[tuple[tuple, nnx.Variable]]:
    return [(p, v) for p, v in nnx.iter_graph(module) if isinstance(v, nnx.Variable)]


def _item(branch: bool) -> Call:
    """Return how an item of a sequence runs: a branch outside any other as one."""

    def call(module: nnx.Module, x: Stream, taps: Taps) -> Stream:
        return run_branch(module, x, taps) if branch else module(x, taps)

    return call


def _pass(i: int, branch: bool) -> Call:
    """Return how pass `i` of an unrolled repeat runs, as ``repeat.Passes`` runs it."""
    item = _item(branch)

    def call(module: nnx.Module, x: Stream, taps: Taps) -> Stream:
        with tracing.passes(range(i, i + 1)):
            return item(module, x, taps)

    return call


def _check(taps: Taps, head: bool) -> None:
    """Raise if a piece left `taps` changed for the next: the pieces pass each other
    the stream alone, and the head ends the model."""
    crossing = [
        name
        for name, crosses in [
            ("a tap", any(taps.values())),
            ("a nextn", taps.nextn_runs > 0),
            ("the single-pass read", taps.pending_read is not None),
            ("a head's output", bool(taps.heads) != head),
        ]
        if crosses
    ]
    if crossing:
        raise ValueError(
            "methods(interleave=true) runs a model's items as pieces that pass each "
            f"other the stream alone, and one leaves {', '.join(crossing)} for the "
            "next, or the model does not end in its one head"
        )


# --- Where residuals wait --------------------------------------------------------------


class Slots:
    """Where each piece's residuals wait for its backward: a buffer for each kind of
    residuals (those of pieces whose residuals have the same shapes and dtypes), of
    as many slots as a step holds of them at once (`sizes`). A forward takes the slot
    its kind freed last, before its tick, and the backward beside it frees its own: in
    a repeat of passes alike, each forward's residuals go where the tick before read
    the other microbatch's (`table`, microbatch m's piece p's slot), and a step holds
    one microbatch's."""

    def __init__(self, kinds: list[Signature], microbatches: int):
        self.shapes = list(dict.fromkeys(kinds))
        self.kind = [self.shapes.index(k) for k in kinds]
        self.table, self.sizes = _assign(self.kind, len(self.shapes), microbatches)

    def unwritten(self) -> Buffers:
        """Return the buffers before the first microbatch: none yet."""
        return [[] for _ in self.sizes]

    def first(self, buffers: Buffers, p: int, leaves: list[Leaf]) -> Buffers:
        """Return `buffers` with the first microbatch's piece `p`'s residuals,
        `leaves`, in its slot. The first piece of a kind starts its kind's buffer, as
        its residuals in every slot, so that XLA splits it over the mesh as it splits
        them: an empty one it may keep whole on every device, as it did those of the
        experts' kernels."""
        k = self.kind[p]
        if buffers[k] or not leaves:
            return self.write(buffers, p, int(self.table[0, p]), leaves)
        started = [jnp.broadcast_to(a, (self.sizes[k], *a.shape)) for a in leaves]
        return [started if j == k else b for j, b in enumerate(buffers)]

    def read(self, buffers: Buffers, p: int, slot: int | Int[Array, ""]) -> list[Leaf]:
        """Return piece `p`'s residuals from `slot` of its kind's buffer."""
        at = jax.lax.dynamic_index_in_dim
        return [at(b, slot, keepdims=False) for b in buffers[self.kind[p]]]

    def write(
        self, buffers: Buffers, p: int, slot: int | Int[Array, ""], leaves: list[Leaf]
    ) -> Buffers:
        """Return `buffers` with piece `p`'s residuals, `leaves`, in `slot`."""
        put, k = jax.lax.dynamic_update_index_in_dim, self.kind[p]
        written = [put(b, a, slot, 0) for b, a in zip(buffers[k], leaves, strict=True)]
        return [written if j == k else b for j, b in enumerate(buffers)]


def _assign(
    kind: list[int], kinds: int, microbatches: int
) -> tuple[Int[np.ndarray, "n p"], list[int]]:
    """Return the slot of each microbatch's pieces, and how many each kind takes, as
    the ticks take and free them: at each, the forward takes the last its kind freed
    before (or a new one), and then the backward beside it frees its own."""
    free: list[list[int]] = [[] for _ in range(kinds)]
    sizes = [0] * kinds

    def take(k: int) -> int:
        if free[k]:
            return free[k].pop()
        sizes[k] += 1
        return sizes[k] - 1

    p = len(kind)
    table = np.zeros((microbatches, p), np.int32)
    table[0] = [take(k) for k in kind]
    for m in range(1, microbatches):
        for t in range(p):
            table[m, t] = take(kind[t])
            free[kind[p - 1 - t]].append(int(table[m - 1, p - 1 - t]))
    return table, sizes


def _reads_and_writes(slots: Slots) -> tuple[Int[Array, "m p"], Int[Array, "m p"]]:
    """Return the slots each step of the loop reads, the backward's microbatch's, and
    writes, the forward's: microbatches 0 to n-2, and 1 to n-1."""
    return jnp.asarray(slots.table[:-1]), jnp.asarray(slots.table[1:])


def _leaves(vjp: Vjp) -> list[Leaf]:
    return jax.tree.leaves(vjp)


# --- The step --------------------------------------------------------------------------


class Rows(NamedTuple):
    """A microbatch, its documents, and its share of the batch's target tokens."""

    batch: Batch
    segments: Segments
    share: Float[Array, ""]

    @classmethod
    def of(cls, batch: Batch, share: Float[Array, ""]) -> "Rows":
        return cls(batch, Segments.of(jnp.asarray(batch.position_ids)), share)


class Forward(NamedTuple):
    """What a piece's forward leaves: the stream on, the losses it reports, its vjp,
    and the model's other state after it."""

    y: Stream
    losses: list[Loss]
    vjp: Vjp
    rest: Flat


class Held(NamedTuple):
    """What a step carries from microbatch to microbatch: the gradients summed so far,
    the model's other state as the forwards leave it, the residuals' buffers, and the
    loss's gradient with respect to the stream into the head, of the microbatch whose
    backward runs next."""

    grads: Flat
    rest: Flat
    buffers: Buffers
    dy: Stream


class Step:
    """A step's microbatches, each one's forward beside the backward of the one
    before, on the pieces of `model` (``pieces``) as the trainer runs them (`hooks`),
    their matmuls in the format `formats` (``bux.architecture.precision.matmuls``).
    Each microbatch's losses that layers report weigh `row_share` of the step's."""

    def __init__(
        self,
        model: Model,
        hooks: Hooks,
        formats: PrecisionConfig | str,
        row_share: float,
    ):
        *self.pieces, self.head = pieces(model)
        self.hooks, self.formats, self.row_share = hooks, formats, row_share
        self.main = model.main_branches  # the branches tracing.forward numbers
        self.params: Flat = {}
        self.likes: list[Vjp] = []  # each piece's vjp, for its structure

    def gradients(
        self,
        params: Flat,
        rest: Flat,
        zeros: Flat,
        micro: Batch,
        shares: Float[Array, " n"],
    ) -> tuple[Flat, Metrics, Flat]:
        """Return the gradients of the loss on the microbatches of `micro`, each leaf
        (microbatches, ...), summed into `zeros`; their metrics, summed; and the
        model's other state after the last forward. Microbatch i's terms of the
        objective weigh `shares[i]`."""
        self.params, n = params, shares.shape[0]
        first = Rows.of(_at(micro, 0), shares[0])
        slots = Slots(self._signatures(rest, first), n)
        held, metrics = self._first(zeros, rest, first, slots)
        if n > 1:
            xs = (_after_first(micro), shares[1:], *_reads_and_writes(slots))
            held, later = jax.lax.scan(lambda h, x: self._ticks(h, slots, *x), held, xs)
            metrics = jax.tree.map(
                lambda a, b: jnp.concatenate([a[None], b]), metrics, later
            )
        held = self._last(held, slots, slots.table[n - 1])
        return held.grads, {k: v.sum() for k, v in metrics.items()}, held.rest

    def _signatures(self, rest: Flat, rows: Rows) -> list[Signature]:
        """Return the shapes of each piece's residuals on the first microbatch, traced
        on their own (``jax.eval_shape``), for the slots before the first forward."""

        def residuals() -> list[list[Leaf]]:
            x, leaves = jnp.asarray(rows.batch.input_ids), []
            with tracing.forward(self.main):
                for p in range(len(self.pieces)):
                    f = self._forward(p, rest, x, rows)
                    x, leaves = f.y, [*leaves, _leaves(f.vjp)]
            return leaves

        shapes = jax.eval_shape(residuals)
        return [tuple((a.shape, a.dtype) for a in piece) for piece in shapes]

    def _first(
        self, zeros: Flat, rest: Flat, rows: Rows, slots: Slots
    ) -> tuple[Held, Metrics]:
        """Run the first microbatch forward to its loss, each piece's residuals into
        their slots before the next piece runs: else XLA may hold them all twice, as
        made and in their slots."""
        x, losses = jnp.asarray(rows.batch.input_ids), []
        buffers, self.likes = slots.unwritten(), []
        with tracing.forward(self.main):
            for p in range(len(self.pieces)):
                f = self._forward(p, rest, x, rows)
                buffers = slots.first(buffers, p, _leaves(f.vjp))
                x, buffers = jax.lax.optimization_barrier((f.y, buffers))
                rest, losses = f.rest, losses + f.losses
                self.likes.append(f.vjp)
            metrics, d_head, dy, rest = self._loss(rest, x, rows, losses)
        return Held(_added(zeros, d_head), rest, buffers, dy), metrics

    def _ticks(
        self,
        held: Held,
        slots: Slots,
        batch: Batch,
        share: Float[Array, ""],
        read: Int[Array, " p"],
        write: Int[Array, " p"],
    ) -> tuple[Held, Metrics]:
        """Run the next microbatch, `batch`, forward to its loss, each piece's beside
        the backward of a piece of the microbatch before, from the top: their
        residuals in the slots `read`, and the new ones into those of `write`."""
        rows, (grads, rest, buffers, dy) = Rows.of(batch, share), held
        x, losses, r = jnp.asarray(rows.batch.input_ids), [], len(self.pieces)
        before: Flat = {}  # the tick before's gradients
        with tracing.forward(self.main):
            for t in range(r):
                q = r - 1 - t
                f = self._forward(t, rest, x, rows)
                d_params, dx = self._backward(q, slots.read(buffers, q, read[q]), dy)
                # The tick's end: the next starts once both halves are done, and the
                # forward writes its slot once what read it last, the gradients of
                # the tick before, is reduced too.
                done = (f.y, dx, before, buffers)
                x, dy, before, buffers = jax.lax.optimization_barrier(done)
                buffers = slots.write(buffers, t, write[t], _leaves(f.vjp))
                grads, before = _added(grads, before), d_params
                rest, losses = f.rest, losses + f.losses
            metrics, d_head, dy, rest = self._loss(rest, x, rows, losses)
        grads = _added(_added(grads, before), d_head)
        return Held(grads, rest, buffers, dy), metrics

    def _last(self, held: Held, slots: Slots, table: Int[np.ndarray, " p"]) -> Held:
        """Take the last microbatch's gradient back through every piece but the head,
        from its residuals in the slots of `table`, each piece's read once the piece
        above is done: else XLA may copy them all out of their slots at once."""
        grads, buffers, dy = held.grads, held.buffers, held.dy
        for q in reversed(range(len(self.pieces))):
            d_params, dy = self._backward(q, slots.read(buffers, q, int(table[q])), dy)
            grads = _added(grads, d_params)
            dy, buffers = jax.lax.optimization_barrier((dy, buffers))
        return held._replace(grads=grads)

    # --- A piece's forward and backward, and the head's loss ---------------------------

    def _forward(self, p: int, rest: Flat, x: Stream, rows: Rows) -> Forward:
        """Run piece `p` forward on `x`, as the vjp of its parameters, and of `x` but
        for the first piece's, the tokens."""
        piece = self.pieces[p]
        state = piece.state(piece.rest, rest)

        def run(params: nnx.State, x: Stream) -> tuple[tuple, nnx.State]:
            y, after, losses = self._run(piece, params, state, x, rows)
            return (y, losses), after

        params, f = piece.state(piece.params, self.params), _unfused(run)
        if p == 0:  # x is the tokens, which take no gradient
            f, primals, saveable = functools.partial(f, x=x), (params,), (False,)
        else:
            primals, saveable = (params, x), (False, True)
        (y, losses), vjp, after = jax.vjp(
            f, *primals, has_aux=True, saveable_args=saveable
        )
        return Forward(y, piece.losses(losses), vjp, rest | piece.flat(after))

    def _run(
        self, piece: Piece, params: nnx.State, rest: nnx.State, x: Stream, rows: Rows
    ) -> tuple[Stream, nnx.State, list[Float[Array, ""]]]:
        """Run `piece`'s module on `x` in a training forward of `rows`: the stream it
        leaves, its other state after, and the values of the losses it reports."""
        module = self.hooks.module(piece.graphdef, params, rest)
        taps = _taps(rows)
        with matmuls.use(self.formats, jnp.asarray(rows.batch.input_ids)):
            y = piece.call(module, x, taps)
        _check(taps, head=False)
        piece.reports = [(a.name, a.weight) for a in taps.losses]
        return y, nnx.state(module, nnx.Not(nnx.Param)), [a.value for a in taps.losses]

    def _backward(
        self, p: int, residuals: list[Leaf], dy: Stream | None
    ) -> tuple[Flat, Stream | None]:
        """Take `dy`, the gradient with respect to piece `p`'s output, back through it
        from its `residuals`: the gradients of its parameters, and of its input but
        for the first piece's (None)."""
        piece = self.pieces[p]
        params = piece.state(piece.params, self.params)
        vjp = jax_internals.rebuild_vjp(self.likes[p], residuals, params)
        cotangents = vjp((dy, self._losses_cotangent(piece)))
        return piece.flat(cotangents[0]), cotangents[1] if p > 0 else None

    def _losses_cotangent(self, piece: Piece) -> list[Float[Array, ""]]:
        """Return the loss's gradient with respect to the losses `piece` reports: each
        one's weight times a microbatch's share of rows, in float32, as the
        objective's sum takes them."""
        share = np.float32(self.row_share)
        return [jnp.asarray(np.float32(w) * share) for _, w in piece.reports]

    def _loss(
        self, rest: Flat, x: Stream, rows: Rows, losses: list[Loss]
    ) -> tuple[Metrics, Flat, Stream, Flat]:
        """Run the head on `x` to the microbatch's loss, with the `losses` the other
        pieces reported: its metrics, the loss's gradients with respect to the head's
        parameters and to `x`, and the model's other state after."""
        piece = self.head
        state = piece.state(piece.rest, rest)

        def loss(params: nnx.State, x: Stream) -> tuple[Float[Array, ""], tuple]:
            module = self.hooks.module(piece.graphdef, params, state)
            taps = _taps(rows)
            with matmuls.use(self.formats, jnp.asarray(rows.batch.input_ids)):
                piece.call(module, x, taps)
            _check(taps, head=True)
            every = [*losses, *taps.losses]
            value, metrics = self.hooks.loss(
                taps.heads, rows.batch, every, rows.share, self.row_share
            )
            return value, (metrics, nnx.state(module, nnx.Not(nnx.Param)))

        params = piece.state(piece.params, self.params)
        grad = jax.grad(loss, argnums=(0, 1), has_aux=True)
        (d_params, dx), (metrics, after) = grad(params, x)
        return metrics, piece.flat(d_params), dx, rest | piece.flat(after)


def _unfused[**A, R](f: Callable[A, R]) -> Callable[A, R]:
    """Return `f`, keeping all but what elementwise ops make for its backward where
    the run's remat keeps everything (``remat.unfused``), as a scanned pass does: a
    slot holds whatever a piece's vjp keeps, where in a step in a row XLA fuses those
    ops into the backward's that read them."""
    if remat.policy() is not None:
        return f  # each branch inside is rematerialized by the run's policy
    return jax.checkpoint(f, policy=remat.unfused, prevent_cse=False)


def _taps(rows: Rows) -> Taps:
    """Return the taps a piece's module starts from in a training forward of `rows`."""
    tokens = jnp.asarray(rows.batch.input_ids)
    return Taps(tokens=tokens, segments=rows.segments, training=True)


def _added(total: Flat, part: Flat) -> Flat:
    """Return `total` with `part` added where it has arrays, in `total`'s dtypes, as
    ``gather.added`` adds a microbatch's gradients."""
    return total | {k: total[k] + v.astype(total[k].dtype) for k, v in part.items()}


def _at(micro: Batch, i: int) -> Batch:
    return jax.tree.map(lambda a: a[i], micro)


def _after_first(micro: Batch) -> Batch:
    return jax.tree.map(lambda a: a[1:], micro)
