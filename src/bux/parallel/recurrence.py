"""Context parallelism for recurrent mixers: what the methods of `methods.recurrence`
take, and `pass_state`.

Every recurrent mixer keeps a state S of k by v a head, which each position decays
and adds to, and reads its outputs from, linearly. So a chunk of positions maps the
state it starts from to the one it ends with by an affine map, its transition

    S -> A S + B,

with A a k by k matrix a head (the delta rule, RWKV-7), or its diagonal (GLA, Mamba),
or a scalar (Lightning, Mamba-2), and zero across a document start; and each output
of the chunk gains reads_t . S from the state S it starts from.

A method is ``(recurrence) -> out``: a `Recurrence` names the family and holds a
function that runs it over a chunk of positions from a zero state (`Chunk`), with
its inputs; `out` is the outputs (b, t, h, v) in float32. `scan`, the default, runs
every position as one chunk, and is in bux.architecture.linear_attention. A method
that cannot run a family raises NotImplementedError naming it.
"""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Shaped

from bux.parallel import methods
from bux.parallel.layout import spec
from bux.parallel.mesh import active

Positions = tuple[Shaped[Array, "b t ..."], ...]  # what a chunk takes a position of
Shared = tuple[Float[Array, "..."], ...]  # what it takes whole: parameters
# A matrix a head, or its diagonal: of k entries, or of 1.
Transition = Float[Array, "b h k k"] | Float[Array, "b h _"]


class Chunk(NamedTuple):
    """What a recurrence gives over a chunk of positions from a zero state; with its
    transition, also what a state it started from would add."""

    out: Float[Array, "b t h v"]  # the outputs, in float32
    state: Float[Array, "b h k v"] | None  # the state it ends with: B
    transition: Transition | None  # A
    reads: Float[Array, "b t h k"] | None  # out_t gains reads_t . S from a start S


# (xs, shared, transition) -> Chunk: the transition and the reads only if asked for.
Chunked = Callable[[Positions, Shared, bool], Chunk]


class Recurrence(NamedTuple):
    """What a method of `methods.recurrence` runs: `chunk` over (b, t, ...) arrays
    `xs` of positions, with `shared` whole along them."""

    family: str  # what errors call it: "the selective scan"
    chunk: Chunked
    xs: Positions
    shared: Shared = ()


def then(
    transition: Transition, state: Float[Array, "b h k v"]
) -> Float[Array, "b h k v"]:
    """A S: a chunk's transition, applied to the state it starts from."""
    if transition.ndim == state.ndim:
        return jnp.einsum("bhij,bhjv->bhiv", transition, state)
    return transition[..., None] * state


# --- pass_state -----------------------------------------------------------------------


@methods.recurrence.register("pass_state")
def pass_state(r: Recurrence) -> Float[Array, "b t h v"]:
    """Each device of the axes that split positions runs its own chunk from a zero
    state, keeping its transition; the transitions fold, in turn, into the state each
    chunk starts from (LASP-2: XLA all-gathers them); and each device corrects its
    outputs by what they read of it.

    Both shard_maps are manual over every axis that splits the tokens and hold no
    collective, so that the axes they leave auto (`tensor`) keep the heads split: the
    fold between them is in the global view."""
    tokens = _tokens(r.xs[0])
    if tokens is None:
        return r.chunk(r.xs, r.shared, False).out
    chunks = P(tokens[1], tokens[0])  # (c, b, ...): one of c chunks a device
    manual = set(_axes(tokens))
    out, states, transitions, reads = jax.shard_map(
        partial(_run, r.chunk, r.family),
        in_specs=(tuple(tokens for _ in r.xs), tuple(P() for _ in r.shared)),
        out_specs=(tokens, chunks, chunks, tokens),
        axis_names=manual,
    )(r.xs, r.shared)
    return jax.shard_map(
        _correct, in_specs=(tokens, tokens, chunks), out_specs=tokens, axis_names=manual
    )(out, reads, _starts(states, transitions))


def _tokens(x: Shaped[Array, "b t ..."]) -> P | None:
    """How a (b, t, ...) array splits, rows over `batch` and positions over `seq`;
    None if the positions do not."""
    if active() is None:
        return None
    rows, positions = spec(("batch", "seq"), x.shape[:2])
    return None if positions is None else P(rows, positions)


def _axes(s: P) -> set[str]:
    """The mesh axes `s` splits over."""
    return {a for d in s if d for a in ((d,) if isinstance(d, str) else d)}


def _run(
    chunk: Chunked, family: str, xs: Positions, shared: Shared
) -> tuple[
    Float[Array, "b n h v"],
    Float[Array, "1 b h k v"],
    Float[Array, "1 b h ..."],
    Float[Array, "b n h k"],
]:
    """A device's chunk: its outputs, its state and transition (one of a stack of
    chunks), and its reads."""
    c = chunk(xs, shared, True)
    if c.state is None or c.transition is None or c.reads is None:
        raise NotImplementedError(
            f"methods.recurrence = pass_state: {family} gives no transition"
        )
    return c.out, c.state[None], c.transition[None], c.reads


def _starts(
    states: Float[Array, "c b h k v"], transitions: Float[Array, "c b h ..."]
) -> Float[Array, "c b h k v"]:
    """The state each of c chunks starts from: every chunk's map before it, in turn,
    from a zero state."""

    def step(
        s: Float[Array, "b h k v"], chunk: tuple[Float[Array, "b h k v"], Transition]
    ) -> tuple[Float[Array, "b h k v"], Float[Array, "b h k v"]]:
        ends, transition = chunk
        return then(transition, s) + ends, s

    return jax.lax.scan(step, jnp.zeros_like(states[0]), (states, transitions))[1]


def _correct(
    out: Float[Array, "b n h v"],
    reads: Float[Array, "b n h k"],
    start: Float[Array, "1 b h k v"],
) -> Float[Array, "b n h v"]:
    """A device's outputs, with what they read of the state its chunk starts from."""
    return out + jnp.einsum("bnhk,bhkv->bnhv", reads, start[0])
