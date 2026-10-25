"""The gated delta rule with a decay per key channel (KDA) by bux's own Pallas kernels:
exact at any gate, forward and backward.

The positions go by chunks of 64 (`SIZE`), in blocks of 16 (`BLOCK`). Within a
chunk, with G_t the sum of its log decays up to t, the decay from s to t is
``exp(G_t - G_s)`` per channel. Each pair of positions takes it through a position r
between them, as a matmul of the two factors::

    exp(G_t - G_s) = exp(G_t - G_r) exp(G_r - G_s),

both at most 1: for blocks i > j, r is the end of block j (``_across``); within a
block, the end of the first half of the 8, 4, 2 or 1 positions it is in, a level at a
time (``_in_block``). So no exponent is positive, whatever the gate. From the state S
a chunk starts from, ``chunkwise``'s WY form gives its outputs and the state it ends
with::

    U = Y - W S,    o = Q S + M U,    S' = D S + K^T U,

where A holds the pairs' ``beta_t k_t . k_s`` decayed (s < t), KK the same without
beta, M the pairs' ``q_t . k_s`` decayed (s <= t), ``T = (I + A)^-1``, and::

    W = T (beta k exp(G)),   Y = T (beta v),   Q = q exp(G),
    K = k exp(G_end - G),    D = exp(G_end):

the chunk's parts. A document starts the state again: no pair spans its start, and
nothing before it reaches it. The kernels, a program each of::

    pairs        a pair of blocks: its M and KK, and a block's inverse of I + A_ii
    solves       a chunk: T from its blocks' inverses by halves, then W, Y, Q, K, D
    states       a row, head and block of the state's columns: the state from
                 chunk to chunk, the outputs, and the state before each chunk
    back         the same, the states' gradients, from the last chunk back
    back_states  a chunk: its parts' gradients, from its states' and outputs'
    back_keys    a chunk and a slab of columns: the gradients of q, k and G by W,
    back_values  Q, K and D, and of v by Y; through T, by T^T, a column at a time
    back_steps   a chunk: KK's gradient by A, -T^T (dX X^T) of all the columns
    back_pairs   a block: the gradients of q, k and G by the pairs

The parts and states are kept in the values' dtype. With ``methods.states = keep``
the backward pass keeps them from the forward pass, else it runs ``pairs``,
``solves`` and ``states`` again. On a mesh, a `jax.shard_map` gives the kernels each
device's rows and heads.
"""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int, Shaped

from bux import kernels, parallel
from bux.architecture.common.segments import Segments

SIZE = 64  # positions per chunk
BLOCK = 16  # positions per block of a chunk
BLOCKS = SIZE // BLOCK
HALVES = (8, 4, 2, 1)  # within a block, the pairs' levels: halves of twice as many
COLUMNS = 64  # columns of the state a program of `states` and `back` carries
SLAB = 32  # columns of a block that a program of the solves takes at a time
DIMS = (64, 128)  # the head dims the kernels take

Tile = Float[Array, "c e"]  # a block's positions, of one head
Pairs = Float[Array, "c c"]  # of each position of a block with each of another's
# Every chunk's parts: W, Y, Q and K (b, t, h, e), M (b, t, h, c) and D (b, n, h, dk).
Every = tuple[Float[Array, "b _ h ..."], ...]
Rule = Callable[..., Float[Array, "b t h dv"]]  # linear_attention.delta_rule
Ref = Any  # a Pallas ref, which indexes and loads as an array does not


def runs(dk: int, dv: int, dtype: Any, decays: int) -> bool:
    """Return whether the kernels run a layer of these head dims, its q, k and v of
    `dtype` and `decays` log decays a head, by the run's methods: on a GPU, a decay
    per channel and 16-bit values."""
    on = parallel.active()
    method = on.methods.recurrence if on else "scan"
    shapes = dk in DIMS and dv in DIMS and dtype in (jnp.bfloat16, jnp.float16)
    return kernels.gpu() and method == "scan" and shapes and decays == dk


def delta_rule(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    log_decay: Float[Array, "b t h #dk"],
    beta: Float[Array, "b t h"],
    docs: Segments | None,
    otherwise: Rule,
) -> Float[Array, "b t h dv"]:
    """Run `linear_attention.delta_rule` by the kernels, in v's dtype; by `otherwise`
    where a mesh splits the positions."""
    xs = (q, k, v, log_decay, beta, _within(docs, q.shape[:2]))
    if parallel.active() is None:
        return _rows(*xs)
    rows = parallel.spec(("batch", "seq", "heads", None), q.shape)
    if rows[1] is not None:  # positions split over devices, which XLA gathers
        return otherwise(q, k, v, log_decay, beta, docs)
    specs = (rows,) * 4 + (P(*rows[:3]), P(*rows[:2]))
    return kernels.per_device(_rows, specs, rows)(*xs)


def _within(docs: Segments | None, shape: tuple[int, ...]) -> Int[Array, "b t"]:
    """Return how many tokens of its document come before each token: one document
    a row without `docs`."""
    if docs is None:
        return jnp.broadcast_to(jnp.arange(shape[1]), shape)
    return docs.within


def _rows(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    g: Float[Array, "b t h #dk"],
    beta: Float[Array, "b t h"],
    within: Int[Array, "b t"],
) -> Float[Array, "b t h dv"]:
    """Run one device's rows by chunks, padded to whole ones."""
    t = q.shape[1]
    q, k, v, g, beta, within = (_padded(x) for x in (q, k, v, g, beta, within))
    G = _summed(jnp.broadcast_to(g, k.shape).astype(jnp.float32))
    out = _chunks(_keeps(), q, k, v, G, beta.astype(jnp.float32), _documents(within))
    return out[:, :t]


def _keeps() -> bool:
    """Return whether the backward pass keeps the chunks' parts and states from the
    forward pass, by the run's `methods.states`: keep, outside ``parallel.use``."""
    on = parallel.active()
    return on is None or on.methods.states == "keep"


def _padded(x: Shaped[Array, "b t *e"]) -> Shaped[Array, "b _ *e"]:
    """Pad x with zeros after its positions to whole chunks."""
    pad = -x.shape[1] % SIZE
    if pad == 0:
        return x
    return jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))


def _by_chunk(x: Shaped[Array, "b t *e"]) -> Shaped[Array, "b n c *e"]:
    """Reshape (b, t, ...) into chunks, (b, n, SIZE, ...)."""
    b, t = x.shape[:2]
    return x.reshape(b, t // SIZE, SIZE, *x.shape[2:])


def _summed(g: Float[Array, "b t h dk"]) -> Float[Array, "b t h dk"]:
    """Return G, the log decays summed within each chunk up to each position."""
    return jnp.cumsum(_by_chunk(g), axis=2).reshape(g.shape)


def _documents(within: Int[Array, "b t"]) -> Int[Array, "b t"]:
    """Return how many documents have started within each position's chunk, up to
    it: 0 where the state the chunk starts from reaches."""
    starts = _by_chunk(within == 0)
    return jnp.cumsum(starts, axis=2, dtype=jnp.int32).reshape(within.shape)


# --- the forward and backward passes ------------------------------------------------

Inputs = tuple[Shaped[Array, "b t ..."], ...]  # q, k, v, G, beta and the documents
# What the forward pass keeps for the backward beside its inputs: with `keep`, the
# chunks' parts, KK, T and the state before each chunk; else nothing.
Kept = tuple[
    Every, Float[Array, "b t h c"], Float[Array, "b t h c"], Float[Array, "b n h dk dv"]
]


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _chunks(
    keep: bool,
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    G: Float[Array, "b t h dk"],
    beta: Float[Array, "b t h"],
    documents: Int[Array, "b t"],
) -> Float[Array, "b t h dv"]:
    """Return the delta rule's outputs, in v's dtype; `keep` the chunks' parts and
    states for the backward pass, or compute them again there."""
    return _states(_parts(q, k, v, G, beta, documents)[0], states=False)[0]


def _chunks_forward(
    keep: bool, *xs: Shaped[Array, "b t ..."]
) -> tuple[Float[Array, "..."], tuple[Inputs, Kept | None]]:
    parts, KK, T = _parts(*xs)
    if not keep:
        return _states(parts, states=False)[0], (xs, None)
    out, states = _states(parts, states=True)
    return out, (xs, (parts, KK, T, states))


def _chunks_backward(
    keep: bool, saved: tuple[Inputs, Kept | None], d_out: Float[Array, "b t h dv"]
) -> tuple[Float[Array, "..."] | None, ...]:
    """Carry the states' gradients back, take each chunk's parts' gradients from
    them and the states, and the inputs' from those: through the solves and the
    rest by chunk, through the pairs by block. Unless kept, run the chunks again
    for their parts and the state before each."""
    xs, kept = saved
    if kept is None:
        parts, KK, T = _parts(*xs)
        kept = (parts, KK, T, _states(parts, outputs=False, states=True)[0])
    parts, KK, T, states = kept
    d_states = _back(d_out, parts)
    dW, dY, dQ, dK, dM, dD = _back_states(parts, states, d_states, d_out)
    d_parts = (dW, dY, dQ, dK, dD)
    (dq, dk, dv, dG, dbeta), dKK = _back_solves(xs, parts, KK, T, d_parts)
    dq_pairs, dk_pairs, dG_pairs = _back_pairs(xs, dM, dKK)
    grads = (dq + dq_pairs, dk + dk_pairs, dv, dG + dG_pairs, dbeta)
    return (*(g.astype(x.dtype) for g, x in zip(grads, xs, strict=False)), None)


_chunks.defvjp(_chunks_forward, _chunks_backward)


# --- a chunk's pairs, in float32, by blocks of 16 positions -------------------------


def _across(
    q: Tile, k: Tile, G: Tile, k_before: Tile, G_before: Tile
) -> tuple[Pairs, Pairs]:
    """Return M and KK of a block with an earlier block, through the decay at that
    block's end: exp(G_t - G_end) exp(G_end - G_s), both at most 1."""
    end = _row(G_before, BLOCK - 1)
    left = jnp.exp(G - end)
    R = (k_before * jnp.exp(end - G_before)).T
    return _dot(q * left, R), _dot(k * left, R)


def _across_back(
    q: Tile, k: Tile, G: Tile, k_before: Tile, G_before: Tile, dM: Pairs, dKK: Pairs
) -> tuple[Tile, Tile, Tile]:
    """Return the gradients of q, k and G by `_across`, of the later block. Those of
    G_end are 0: a pair's decay does not depend on the position it goes through."""
    end = _row(G_before, BLOCK - 1)
    left = jnp.exp(G - end)
    R = k_before * jnp.exp(end - G_before)
    dLq, dLk = _dot(dM, R), _dot(dKK, R)
    return left * dLq, left * dLk, left * (q * dLq + k * dLk)


def _before_back(
    q: Tile, k: Tile, G: Tile, k_before: Tile, G_before: Tile, dM: Pairs, dKK: Pairs
) -> tuple[Tile, Tile]:
    """Return the gradients of k_before and G_before by `_across`, of the earlier
    block."""
    end = _row(G_before, BLOCK - 1)
    left, right = jnp.exp(G - end), jnp.exp(end - G_before)
    dR = right * (_dot(dM.T, q * left) + _dot(dKK.T, k * left))
    return dR, -k_before * dR


def _in_block(q: Tile, k: Tile, G: Tile, ends: tuple[Tile, ...]) -> tuple[Pairs, Pairs]:
    """Return M and KK within a block: on the diagonal by a sum, below it by halves,
    a position in each half of a block of 2 `half` through the decay at the end of
    its first half, G there given by `ends` (`_halves`)."""
    M = jnp.where(_index(0) == _index(1), jnp.sum(q * k, 1, keepdims=True), 0.0)
    KK = jnp.zeros_like(M)
    for half, end in zip(HALVES, ends, strict=True):
        left, right, across = _halves(G, end, half)
        R = (k * right).T
        M += jnp.where(across, _dot(q * left, R), 0.0)
        KK += jnp.where(across, _dot(k * left, R), 0.0)
    return M, KK


def _in_block_back(
    q: Tile, k: Tile, G: Tile, ends: tuple[Tile, ...], dM: Pairs, dKK: Pairs
) -> tuple[Tile, Tile, Tile]:
    """Return the gradients of q, k and G by `_in_block`."""
    diagonal = jnp.sum(jnp.where(_index(0) == _index(1), dM, 0.0), 1, keepdims=True)
    dq, dk, dG = diagonal * k, diagonal * q, jnp.zeros_like(G)
    for half, end in zip(HALVES, ends, strict=True):
        left, right, across = _halves(G, end, half)
        dM_h, dKK_h = jnp.where(across, dM, 0.0), jnp.where(across, dKK, 0.0)
        R = k * right
        dLq, dLk = _dot(dM_h, R), _dot(dKK_h, R)
        dR = _dot(dM_h.T, q * left) + _dot(dKK_h.T, k * left)
        dq += left * dLq
        dk += left * dLk + right * dR
        dG += left * (q * dLq + k * dLk) - right * k * dR  # by G_t, and by G_s
    return dq, dk, dG


def _halves(G: Tile, end: Tile, half: int) -> tuple[Tile, Tile, Bool[Array, "c c"]]:
    """Return, for blocks of 2 `half` positions, G at the end of whose first half is
    `end`, the decay from there to each position of its second half (`left`, else 0),
    from each position of its first half to there (`right`, else 0), and the pairs
    of a position in each half of a block (`across`): every factor at most 1."""
    second = (_positions() % (2 * half)) >= half
    left = jnp.exp(jnp.where(second, G - end, -jnp.inf))
    right = jnp.exp(jnp.where(second, -jnp.inf, end - G))
    t, s = _index(0), _index(1)
    same = t // (2 * half) == s // (2 * half)
    return left, right, same & (t % (2 * half) >= half) & (s % (2 * half) < half)


def _ends(G: Ref, first: Any) -> tuple[Tile, ...]:
    """Return, for each level of `HALVES`, G at the end of the first half of each
    position's block of 2 `half`, of the block of G's positions in `G` from
    `first`: rows the load gathers."""
    rows = jax.lax.broadcasted_iota(jnp.int32, (BLOCK,), 0)
    at = (first + rows // (2 * half) * 2 * half + half - 1 for half in HALVES)
    return tuple(G[i, :].astype(jnp.float32) for i in at)


# --- a block's inverse, and positions -------------------------------------------


def _inverse(A: Pairs) -> Pairs:
    """Return T = (I + A)^-1 of a strictly lower triangular A, a row at a time: row i
    is e_i - A_i T, of the rows before it."""
    eye = (_index(0) == _index(1)).astype(jnp.float32)
    T = eye
    for i in range(1, BLOCK):
        row = _row(eye, i) - jnp.sum(_row(A, i)[:, None] * T, 0)
        T = jnp.where(_index(0) == i, row[None, :], T)
    return T


def _index(axis: int) -> Int[Array, "c c"]:
    """Return each pair's position in its block along `axis`: 0 for t, 1 for s."""
    return jax.lax.broadcasted_iota(jnp.int32, (BLOCK, BLOCK), axis)


def _positions() -> Int[Array, "c 1"]:
    """Return each position of a block, as a column."""
    return jax.lax.broadcasted_iota(jnp.int32, (BLOCK, 1), 0)


def _row(x: Shaped[Array, "c ..."], i: int) -> Shaped[Array, "..."]:
    """Return x[i], by a sum: a kernel cannot index its values."""
    at = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0) == i
    return jnp.sum(jnp.where(at, x, 0), 0)


def _dot(x: Float[Array, "m k"], y: Float[Array, "k n"]) -> Float[Array, "m n"]:
    """Return x @ y in float32, y in x's dtype: by TF32 on a GPU, where x and y are
    float32."""
    return jax.lax.dot(x, y.astype(x.dtype), preferred_element_type=jnp.float32)


# --- the kernels --------------------------------------------------------------------


def _parts(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    G: Float[Array, "b t h dk"],
    beta: Float[Array, "b t h"],
    documents: Int[Array, "b t"],
) -> tuple[Every, Float[Array, "b t h c"], Float[Array, "b t h c"]]:
    """Return every chunk's parts, in v's dtype but D, and, for the backward pass,
    its pairs' KK and T = (I + A)^-1, in float32: W, Y, Q and K by position
    (b, t, h, ...), M, KK and T by position and position of its chunk (b, t, h,
    SIZE), and D by chunk (b, n, h, dk)."""
    M, KK, inverses = _pairs(q, k, G, beta, documents, v.dtype)
    W, Y, Q, K, D, T = _solves(q, k, v, G, beta, documents, KK, inverses)
    return (W, Y, Q, K, M, D), KK, T


def _pairs(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    G: Float[Array, "b t h dk"],
    beta: Float[Array, "b t h"],
    documents: Int[Array, "b t"],
    dtype: Any,
) -> tuple[Float[Array, "b t h c"], ...]:
    """Return every chunk's M, in `dtype`, and KK, in float32, of the pairs of each
    document, and the inverses of I + A_ii of its blocks i, by position and position
    of the chunk, 0 off the blocks: a program a pair of blocks, 0 where its rows'
    block is before its columns'."""
    b, t, h, dk = k.shape
    pairs = BLOCKS * BLOCKS

    def kernel(*refs: Ref) -> None:
        p = pl.program_id(1) % pairs
        i, j = p // BLOCKS, p % BLOCKS
        q, k, G, k_before, G_before = (ref[...].astype(jnp.float32) for ref in refs[:5])
        later, earlier = refs[5][...], refs[6][...]
        zeros = jnp.zeros((BLOCK, BLOCK), jnp.float32)
        M, KK = jax.lax.cond(
            i > j,
            lambda: _across(q, k, G, k_before, G_before),
            lambda: jax.lax.cond(
                i == j,
                lambda: _in_block(q, k, G, _ends(refs[2], 0)),
                lambda: (zeros, zeros),
            ),
        )
        same = later[:, None] == earlier[None, :]
        before = (i > j) | (_index(0) > _index(1))
        KK = jnp.where(same & before, KK, 0.0)
        A = refs[7][...].astype(jnp.float32)[:, None] * KK
        inverse = jax.lax.cond(i == j, lambda: _inverse(A), lambda: zeros)
        refs[8][...] = jnp.where(same, M, 0.0).astype(refs[8].dtype)
        refs[9][...], refs[10][...] = KK, inverse

    def rows(bb: Any, p: Any, hh: Any) -> tuple[Any, ...]:
        return bb, p // pairs * BLOCKS + p % pairs // BLOCKS, hh

    def columns(bb: Any, p: Any, hh: Any) -> tuple[Any, ...]:
        return bb, p // pairs * BLOCKS + p % BLOCKS, hh

    def tile(at: Callable[..., tuple[Any, ...]]) -> pl.BlockSpec:
        return pl.BlockSpec((None, BLOCK, None, dk), lambda *ids: (*at(*ids), 0))

    def docs(at: Callable[..., tuple[Any, ...]]) -> pl.BlockSpec:
        return pl.BlockSpec((None, BLOCK), lambda *ids: at(*ids)[:2])

    steps = pl.BlockSpec((None, BLOCK, None), rows)
    ins = [tile(rows)] * 3 + [tile(columns)] * 2 + [docs(rows), docs(columns), steps]
    pair = pl.BlockSpec(
        (None, BLOCK, None, BLOCK), lambda *ids: (*rows(*ids), ids[1] % BLOCKS)
    )
    out = [jax.ShapeDtypeStruct((b, t, h, SIZE), dtype)]
    out += [jax.ShapeDtypeStruct((b, t, h, SIZE), jnp.float32)] * 2
    grid = (b, t // SIZE * pairs, h)
    call = _call(kernel, "pairs", grid, ins, out, [pair] * 3, warps=4)
    return tuple(call(q, k, G, k, G, documents, documents, beta))


def _solves(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    G: Float[Array, "b t h dk"],
    beta: Float[Array, "b t h"],
    documents: Int[Array, "b t"],
    KK: Float[Array, "b t h c"],
    inverses: Float[Array, "b t h c"],
) -> tuple[Float[Array, "b _ h ..."], ...]:
    """Return every chunk's W, Y, Q and K, in v's dtype, D, and T = (I + A)^-1, from
    its pairs' KK and its blocks' inverses: a program a chunk, `SLAB` columns at a
    time, which T takes apart."""
    b, t, h, dk = k.shape
    dv = v.shape[-1]

    def kernel(*refs: Ref) -> None:
        q_ref, k_ref, v_ref, G_ref, beta_ref, docs_ref, KK_ref, inv_ref = refs[:8]
        W_ref, Y_ref, Q_ref, K_ref, D_ref, T_ref = refs[8:]
        beta = beta_ref[...].astype(jnp.float32)[:, None]
        docs, last = docs_ref[...][:, None], docs_ref[SIZE - 1]
        T = _merged(inv_ref[...], beta * KK_ref[...])
        T_ref[...] = T
        for c in range(dk // SLAB):
            at = pl.ds(c * SLAB, SLAB)
            q, k, G = (ref[:, at].astype(jnp.float32) for ref in (q_ref, k_ref, G_ref))
            end = G_ref[SIZE - 1, at]
            decayed = jnp.where(docs == 0, jnp.exp(G), 0.0)  # from the chunk's start
            W_ref[:, at] = _dot(T, beta * k * decayed).astype(W_ref.dtype)
            Q_ref[:, at] = (q * decayed).astype(Q_ref.dtype)
            K = jnp.where(docs == last, k * jnp.exp(end - G), 0.0)  # to its end
            K_ref[:, at] = K.astype(K_ref.dtype)
            D_ref[at] = jnp.where(last == 0, jnp.exp(end), 0.0)
        for c in range(dv // SLAB):
            at = pl.ds(c * SLAB, SLAB)
            values = beta * v_ref[:, at].astype(jnp.float32)
            Y_ref[:, at] = _dot(T, values).astype(Y_ref.dtype)

    xs = (q, k, v, G, beta, documents, KK, inverses)
    ins = [_chunk_spec(x.shape) for x in xs]
    out = [jax.ShapeDtypeStruct(x.shape, v.dtype) for x in (k, v, k, k)]
    out += [
        jax.ShapeDtypeStruct((b, t // SIZE, h, dk), jnp.float32),
        jax.ShapeDtypeStruct(KK.shape, jnp.float32),
    ]
    outs = [_chunk_spec(x.shape) for x in out]
    outs[4] = _chunk_spec(out[4].shape, 1)
    call = _call(kernel, "solves", (b, t // SIZE, h), ins, out, outs, warps=4)
    return tuple(call(*xs))


def _merged(T: Float[Array, "c c"], A: Float[Array, "c c"]) -> Float[Array, "c c"]:
    """Return (I + A)^-1 of a chunk's strictly lower triangular A, from T, the
    inverses of I + A_ii of its blocks, by halves: that of [[I + A1, 0], [A21,
    I + A2]] is [[T1, 0], [-T2 A21 T1, T2]], from blocks of 16 up."""
    half = BLOCK
    while half < SIZE:
        quarter = jnp.where(_across_chunk(half), A, 0.0)
        T = T - _dot(_dot(T, quarter), T)
        half *= 2
    return T


def _across_chunk(half: int) -> Bool[Array, "c c"]:
    """Return the pairs of a chunk's positions of a position in each half of a block
    of 2 `half`, the later in the second."""
    t = jax.lax.broadcasted_iota(jnp.int32, (SIZE, SIZE), 0)
    s = jax.lax.broadcasted_iota(jnp.int32, (SIZE, SIZE), 1)
    same = t // (2 * half) == s // (2 * half)
    return same & (t % (2 * half) >= half) & (s % (2 * half) < half)


def _states(
    parts: Every, *, outputs: bool = True, states: bool
) -> tuple[Float[Array, "b _ h ..."], ...]:
    """Return, of the chunks of `parts` from a zero state each carries to the next,
    their outputs if `outputs`, and the state before each chunk if `states`, in their
    dtype."""
    W, Y, *_, D = parts
    b, _, h, dk = W.shape
    n, dv = D.shape[1], Y.shape[-1]

    def kernel(
        W_ref: Ref,
        Y_ref: Ref,
        Q_ref: Ref,
        K_ref: Ref,
        M_ref: Ref,
        D_ref: Ref,
        *outs: Ref,
    ) -> None:
        def step(i: Int[Array, ""], S: Float[Array, "dk v"]) -> Float[Array, "dk v"]:
            at = pl.ds(i * SIZE, SIZE)
            U = Y_ref[at, :] - _dot(W_ref[at, :], S)
            if outputs:
                o = _dot(Q_ref[at, :], S) + _dot(M_ref[at, :], U)
                outs[0][at, :] = o.astype(outs[0].dtype)
            if states:
                outs[-1][i] = S.astype(outs[-1].dtype)
            return D_ref[i][:, None] * S + _dot(K_ref[at, :].T, U)

        jax.lax.fori_loop(0, n, step, jnp.zeros((dk, COLUMNS), jnp.float32))

    ins = [_row_spec(x.shape, x is Y) for x in parts]
    shapes = [Y.shape] * outputs + [(b, n, h, dk, dv)] * states
    results = [jax.ShapeDtypeStruct(shape, Y.dtype) for shape in shapes]
    specs = [_row_spec(shape, True) for shape in shapes]
    grid = (b, h, dv // COLUMNS)
    call = _call(kernel, "states", grid, ins, results, specs, warps=8, stages=2)
    return tuple(call(*parts))


def _back(d_out: Float[Array, "b t h dv"], parts: Every) -> Float[Array, "b n h dk dv"]:
    """Return the gradient of the state after each chunk, from the outputs' `d_out`,
    carried back from the last chunk through the chunks' `parts`."""
    W, _, Q, K, M, D = parts
    b, _, h, dk = W.shape
    n, dv = D.shape[1], d_out.shape[-1]

    def kernel(
        dO_ref: Ref,
        W_ref: Ref,
        Q_ref: Ref,
        K_ref: Ref,
        M_ref: Ref,
        D_ref: Ref,
        dS_ref: Ref,
    ) -> None:
        def step(j: Int[Array, ""], dS: Float[Array, "dk v"]) -> Float[Array, "dk v"]:
            i = n - 1 - j
            at = pl.ds(i * SIZE, SIZE)
            dS_ref[i] = dS.astype(dS_ref.dtype)
            dO = dO_ref[at, :]
            dU = _dot(M_ref[at, :].T, dO) + _dot(K_ref[at, :], dS)
            dS = D_ref[i][:, None] * dS + _dot(Q_ref[at, :].T, dO)
            return dS - _dot(W_ref[at, :].T, dU)

        jax.lax.fori_loop(0, n, step, jnp.zeros((dk, COLUMNS), jnp.float32))

    xs = (d_out.astype(W.dtype), W, Q, K, M, D)
    ins = [_row_spec(x.shape, x is xs[0]) for x in xs]
    result = jax.ShapeDtypeStruct((b, n, h, dk, dv), W.dtype)
    spec = _row_spec(result.shape, True)
    grid = (b, h, dv // COLUMNS)
    call = _call(kernel, "back", grid, ins, [result], [spec], warps=8, stages=2)
    return call(*xs)[0]


def _back_states(
    parts: Every,
    states: Float[Array, "b n h dk dv"],
    d_states: Float[Array, "b n h dk dv"],
    d_out: Float[Array, "b t h dv"],
) -> Every:
    """Return the gradients of every chunk's parts, in float32, from the state
    before each chunk, the gradient of the one after it, and the outputs'."""
    W, Y, _, K, M, D = parts
    b, t, h, _ = W.shape

    def kernel(*refs: Ref) -> None:
        W, Y, K, M = (ref[...] for ref in refs[:4])
        S, dS, dO = (ref[...] for ref in refs[4:7])
        grads = _parts_grads(W, Y, K, M, S, dS, dO)
        for ref, grad in zip(refs[7:], grads, strict=True):
            ref[...] = grad

    state = _chunk_spec(states.shape, 1)
    xs = (W, Y, K, M, states, d_states, d_out.astype(W.dtype))
    ins = [_chunk_spec(x.shape) for x in xs[:4]]
    ins += [state, state, _chunk_spec(d_out.shape)]
    out = [jax.ShapeDtypeStruct(x.shape, jnp.float32) for x in parts]
    outs = [_chunk_spec(x.shape) for x in parts[:5]] + [_chunk_spec(D.shape, 1)]
    call = _call(kernel, "back_states", (b, t // SIZE, h), ins, out, outs, warps=8)
    return tuple(call(*xs))


def _parts_grads(
    W: Float[Array, "c dk"],
    Y: Float[Array, "c dv"],
    K: Float[Array, "c dk"],
    M: Float[Array, "c c"],
    S: Float[Array, "dk dv"],
    dS: Float[Array, "dk dv"],
    dO: Float[Array, "c dv"],
) -> tuple[Float[Array, "..."], ...]:
    """Return the gradients of a chunk's W, Y, Q, K, M and D, whole, from the state S
    it starts from, the gradient dS of the one it ends with, and its outputs' dO."""
    U = Y.astype(jnp.float32) - _dot(W, S)
    dU = _dot(M.T, dO) + _dot(K, dS)
    U_, dU_ = U.astype(W.dtype), dU.astype(W.dtype)  # as the matmuls take them
    dW, dQ, dK, dM = -_dot(dU_, S.T), _dot(dO, S.T), _dot(U_, dS.T), _dot(dO, U_.T)
    dD = jnp.sum(S.astype(jnp.float32) * dS.astype(jnp.float32), 1)
    return dW, dU, dQ, dK, dM, dD


def _back_solves(
    xs: Inputs,
    parts: Every,
    KK: Float[Array, "b t h c"],
    T: Float[Array, "b t h c"],
    d_parts: tuple[Float[Array, "b _ h ..."], ...],
) -> tuple[tuple[Float[Array, "b t ..."], ...], Float[Array, "b t h c"]]:
    """Return the gradients of q, k, v, G and beta by all but the pairs, and of the
    pairs' KK, in float32, from those of the chunks' W, Y, Q, K and D. Of X = T B,
    B's is dB = T^T dX, a column at a time (`_back_keys`, `_back_values`), and A's
    -dB X^T = -T^T (dX X^T), of all of them (`_back_steps`)."""
    q, k, v, G, beta, documents = xs
    W, Y, *_ = parts
    dW, dY, dQ, dK, dD = d_parts
    dq, dk, dG, dbeta_keys = _back_keys(q, k, G, beta, documents, T, dW, dQ, dK, dD)
    dv, dbeta_values = _back_values(v, beta, T, dY)
    dKK, dbeta = _back_steps(beta, documents, KK, T, W, Y, dW, dY)
    dbeta = dbeta + jnp.sum(dbeta_keys, -1) + jnp.sum(dbeta_values, -1)
    return (dq, dk, dv, dG, dbeta), dKK


def _back_keys(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    G: Float[Array, "b t h dk"],
    beta: Float[Array, "b t h"],
    documents: Int[Array, "b t"],
    T: Float[Array, "b t h c"],
    dW: Float[Array, "b t h dk"],
    dQ: Float[Array, "b t h dk"],
    dK: Float[Array, "b t h dk"],
    dD: Float[Array, "b n h dk"],
) -> tuple[Float[Array, "b t h _"], ...]:
    """Return the gradients of q, k and G by W's B = beta k decayed, Q = q decayed,
    K = k decayed to the chunk's end and D, and beta's by B, `SLAB` of them a
    column: a program a chunk and `SLAB` columns."""
    b, t, h, dk = k.shape

    def kernel(*refs: Ref) -> None:
        q_ref, k_ref, G_ref, beta_ref, docs_ref, T_ref, dW_ref, dQ_ref, dK_ref = refs[
            :9
        ]
        dD_ref, dq_ref, dk_ref, dG_ref, dbeta_ref = refs[9:]
        beta = beta_ref[...].astype(jnp.float32)[:, None]
        docs, last = docs_ref[...][:, None], docs_ref[SIZE - 1]
        q, k, G, dW, dQ, dK = (
            ref[...].astype(jnp.float32)
            for ref in (q_ref, k_ref, G_ref, dW_ref, dQ_ref, dK_ref)
        )
        dB = _dot(T_ref[...].T, dW)
        end = G_ref[SIZE - 1, :]
        decayed = jnp.where(docs == 0, jnp.exp(G), 0.0)
        to_end = jnp.where(docs == last, jnp.exp(end - G), 0.0)
        keys = beta * decayed * dB  # of k, by B = beta k decayed
        reads = decayed * dQ  # of q, by Q = q decayed
        ends = to_end * dK  # of k, by K = k to_end
        d_end = jnp.where(last == 0, jnp.exp(end), 0.0) * dD_ref[...]
        d_end += jnp.sum(k * ends, 0)
        last_row = _chunk_positions() == SIZE - 1
        dG = k * keys + q * reads - k * ends + jnp.where(last_row, d_end, 0.0)
        dq_ref[...], dk_ref[...], dG_ref[...] = reads, keys + ends, dG
        dbeta_ref[...] = jnp.sum(k * decayed * dB, 1)

    slabs = dk // SLAB
    xs = (q, k, G, beta, documents, T, dW, dQ, dK, dD)
    blocks = ["slab"] * 5 + ["whole"] + ["slab"] * 3 + ["chunk"]
    ins = [_slab_spec(x.shape, slabs, at) for x, at in zip(xs, blocks, strict=True)]
    out = [jax.ShapeDtypeStruct(k.shape, jnp.float32)] * 3
    out += [jax.ShapeDtypeStruct((b, t, h, slabs), jnp.float32)]
    outs = [
        _slab_spec(x.shape, slabs, at)
        for x, at in zip(out, ["slab"] * 3 + ["one"], strict=True)
    ]
    grid = (b, t // SIZE * slabs, h)
    call = _call(kernel, "back_keys", grid, ins, out, outs, warps=4)
    return tuple(call(*xs))


def _back_values(
    v: Float[Array, "b t h dv"],
    beta: Float[Array, "b t h"],
    T: Float[Array, "b t h c"],
    dY: Float[Array, "b t h dv"],
) -> tuple[Float[Array, "b t h _"], ...]:
    """Return the gradient of v by Y's B = beta v, and beta's, `SLAB` of them a
    column: a program a chunk and `SLAB` columns."""
    b, t, h, dv = v.shape

    def kernel(
        v_ref: Ref, beta_ref: Ref, T_ref: Ref, dY_ref: Ref, dv_ref: Ref, dbeta_ref: Ref
    ) -> None:
        dB = _dot(T_ref[...].T, dY_ref[...])
        dv_ref[...] = beta_ref[...].astype(jnp.float32)[:, None] * dB
        dbeta_ref[...] = jnp.sum(v_ref[...].astype(jnp.float32) * dB, 1)

    slabs = dv // SLAB
    xs = (v, beta, T, dY)
    ins = [
        _slab_spec(x.shape, slabs, at)
        for x, at in zip(xs, ["slab", "slab", "whole", "slab"], strict=True)
    ]
    out = [
        jax.ShapeDtypeStruct(v.shape, jnp.float32),
        jax.ShapeDtypeStruct((b, t, h, slabs), jnp.float32),
    ]
    outs = [_slab_spec(out[0].shape, slabs), _slab_spec(out[1].shape, slabs, "one")]
    grid = (b, t // SIZE * slabs, h)
    call = _call(kernel, "back_values", grid, ins, out, outs, warps=4)
    return tuple(call(*xs))


def _back_steps(
    beta: Float[Array, "b t h"],
    documents: Int[Array, "b t"],
    KK: Float[Array, "b t h c"],
    T: Float[Array, "b t h c"],
    W: Float[Array, "b t h dk"],
    Y: Float[Array, "b t h dv"],
    dW: Float[Array, "b t h dk"],
    dY: Float[Array, "b t h dv"],
) -> tuple[Float[Array, "b t h c"], Float[Array, "b t h"]]:
    """Return the gradient of the pairs' KK, beta times A's, -T^T (dX X^T) where A
    has entries, and beta's by A: a program a chunk."""
    b, t, h, _ = W.shape

    def kernel(*refs: Ref) -> None:
        beta_ref, docs_ref, KK_ref, T_ref, W_ref, Y_ref, dW_ref, dY_ref = refs[:8]
        dKK_ref, dbeta_ref = refs[8:]
        P = jnp.zeros((SIZE, SIZE), jnp.float32)
        for dX_ref, X_ref in ((dW_ref, W_ref), (dY_ref, Y_ref)):
            for c in range(X_ref.shape[1] // SLAB):
                at = pl.ds(c * SLAB, SLAB)
                X = X_ref[:, at].astype(jnp.float32)
                P += _dot(dX_ref[:, at], X.T)
        dA = -_dot(T_ref[...].T, P)
        docs = docs_ref[...][:, None]
        t = jax.lax.broadcasted_iota(jnp.int32, (SIZE, SIZE), 0)
        s = jax.lax.broadcasted_iota(jnp.int32, (SIZE, SIZE), 1)
        dA = jnp.where((docs == docs.T) & (t > s), dA, 0.0)  # where A has entries
        dKK_ref[...] = beta_ref[...].astype(jnp.float32)[:, None] * dA
        dbeta_ref[...] = jnp.sum(dA * KK_ref[...], 1)

    xs = (beta, documents, KK, T, W, Y, dW, dY)
    ins = [_chunk_spec(x.shape) for x in xs]
    out = [
        jax.ShapeDtypeStruct(KK.shape, jnp.float32),
        jax.ShapeDtypeStruct(beta.shape, jnp.float32),
    ]
    outs = [_chunk_spec(x.shape) for x in out]
    call = _call(kernel, "back_steps", (b, t // SIZE, h), ins, out, outs, warps=4)
    return tuple(call(*xs))


def _slab_spec(shape: tuple[int, ...], slabs: int, block: str = "slab") -> pl.BlockSpec:
    """Return the block, on a grid of (b, n slabs, h), of a chunk and head of an array
    of (b, t, ...) or (b, t): a slab, `SLAB` of its columns, of (b, t, h, e); `whole`,
    all of them; `chunk`, a slab of (b, n, h, e), a chunk's; or `one`, a column of
    (b, t, h, slabs), a slab's."""

    def chunk(p: Any) -> Any:
        return p // slabs

    if len(shape) == 2:
        return pl.BlockSpec((None, SIZE), lambda bb, p, hh: (bb, chunk(p)))
    if len(shape) == 3:
        return pl.BlockSpec((None, SIZE, None), lambda bb, p, hh: (bb, chunk(p), hh))
    if block == "whole":
        whole = (None, SIZE, None, shape[-1])
        return pl.BlockSpec(whole, lambda bb, p, hh: (bb, chunk(p), hh, 0))
    rows = None if block == "chunk" else SIZE
    width = None if block == "one" else SLAB
    return pl.BlockSpec(
        (None, rows, None, width), lambda bb, p, hh: (bb, chunk(p), hh, p % slabs)
    )


def _chunk_positions() -> Int[Array, "c 1"]:
    """Return each position of a chunk, as a column."""
    return jax.lax.broadcasted_iota(jnp.int32, (SIZE, 1), 0)


def _back_pairs(
    xs: Inputs, dM: Float[Array, "b t h c"], dKK: Float[Array, "b t h c"]
) -> tuple[Float[Array, "b t h dk"], ...]:
    """Return the gradients of q, k and G by the pairs, from those of M and KK: a
    program a block, of its pairs with the blocks before it, within it, and with the
    blocks after it."""
    q, k, _, G, _, documents = xs
    b, t, h, dk = k.shape

    def kernel(*refs: Ref) -> None:
        q_ref, k_ref, G_ref, docs_ref, dM_ref, dKK_ref, *outs = refs
        me = pl.program_id(1) % BLOCKS

        def blocks(i: Any) -> tuple[Tile, ...]:
            return tuple(_block(ref, i, jnp.float32) for ref in (q_ref, k_ref, G_ref))

        def cotangents(i: Any, j: Any) -> tuple[Pairs, Pairs]:
            same = _block(docs_ref, i)[:, None] == _block(docs_ref, j)[None, :]
            dM = jnp.where(same, _pair_block(dM_ref, i, j), 0.0)
            return dM, _pair_block(dKK_ref, i, j)

        q, k, G = blocks(me)
        ends = _ends(G_ref, me * BLOCK)
        grads = _in_block_back(q, k, G, ends, *cotangents(me, me))

        def before(j: Any, grads: tuple[Tile, ...]) -> tuple[Tile, ...]:
            _, k_j, G_j = blocks(j)
            back = _across_back(q, k, G, k_j, G_j, *cotangents(me, j))
            return tuple(g + d for g, d in zip(grads, back, strict=True))

        def after(i: Any, grads: tuple[Tile, ...]) -> tuple[Tile, ...]:
            q_i, k_i, G_i = blocks(i)
            dk, dG = _before_back(q_i, k_i, G_i, k, G, *cotangents(i, me))
            return grads[0], grads[1] + dk, grads[2] + dG

        grads = jax.lax.fori_loop(0, me, before, grads)
        grads = jax.lax.fori_loop(me + 1, BLOCKS, after, grads)
        for ref, grad in zip(outs, grads, strict=True):
            ref[...] = grad

    def whole(x: Shaped[Array, "b t ..."]) -> pl.BlockSpec:
        if x.ndim == 2:
            return pl.BlockSpec((None, SIZE), lambda bb, p, hh: (bb, p // BLOCKS))
        block = (None, SIZE, None, x.shape[-1])
        return pl.BlockSpec(block, lambda bb, p, hh: (bb, p // BLOCKS, hh, 0))

    xs_in = (q, k, G, documents, dM, dKK)
    ins = [whole(x) for x in xs_in]
    out = [jax.ShapeDtypeStruct(k.shape, jnp.float32)] * 3
    tile = pl.BlockSpec((None, BLOCK, None, dk), lambda bb, p, hh: (bb, p, hh, 0))
    grid = (b, t // BLOCK, h)
    call = _call(kernel, "back_pairs", grid, ins, out, [tile] * 3, warps=4)
    return tuple(call(*xs_in))


def _block(ref: Ref, i: Any, dtype: Any = jnp.float32) -> Shaped[Array, "c ..."]:
    """Return block i of a chunk's positions in `ref`, floats in `dtype`."""
    x = ref[pl.ds(i * BLOCK, BLOCK)]
    return x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x


def _pair_block(ref: Ref, i: Any, j: Any) -> Pairs:
    """Return the pairs of blocks i and j of a chunk's in `ref`, in float32."""
    pair = ref[pl.ds(i * BLOCK, BLOCK), pl.ds(j * BLOCK, BLOCK)]
    return pair.astype(jnp.float32)


def _chunk_spec(shape: tuple[int, ...], rows: int = SIZE) -> pl.BlockSpec:
    """Return the block of a chunk and head, on a grid of (b, n, h), of an array of
    (b, t, h, ...), (b, t, h) or (b, t); with `rows` 1, of an array of (b, n, h,
    ...), one a chunk."""
    rest = shape[3:]
    if len(shape) == 2:
        return pl.BlockSpec((None, rows), lambda i, j, l: (i, j))
    if len(shape) == 3:
        return pl.BlockSpec((None, rows, None), lambda i, j, l: (i, j, l))
    block = (None, rows if rows > 1 else None, None, *rest)
    zeros = (0,) * len(rest)
    return pl.BlockSpec(block, lambda i, j, l: (i, j, l, *zeros))


def _row_spec(shape: tuple[int, ...], columns: bool) -> pl.BlockSpec:
    """Return the block of a row and head, on a grid of (b, h, blocks of the state's
    columns), of an array of (b, t or n, h, ...): all of its positions, or chunks,
    and of its last axis, `COLUMNS` of it if `columns`."""
    *rest, last = shape[3:]
    width = COLUMNS if columns else last
    block = (None, shape[1], None, *rest, width)
    zeros = (0,) * len(rest)
    return pl.BlockSpec(block, lambda i, l, j: (i, 0, l, *zeros, j if columns else 0))


def _call(
    kernel: Any,
    name: str,
    grid: tuple[int, ...],
    in_specs: list[pl.BlockSpec],
    out_shape: list[jax.ShapeDtypeStruct],
    out_specs: list[pl.BlockSpec],
    *,
    warps: int,
    stages: int = 1,
) -> Any:
    """Return the Pallas call of `kernel` on `grid`, by Triton on a GPU."""
    from jax.experimental.pallas import triton as plgpu

    return pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=kernels.interpreted(),
        name=f"kda_{name}",
    )
