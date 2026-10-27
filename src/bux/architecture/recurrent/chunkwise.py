"""Linear attention's recurrence by chunks of positions, as fla's chunked kernels run it:
matmuls within a chunk, and a scan of the state across chunks.

Over a chunk from a state S_0, with G_t the sum of the log decays of its positions up
to t, the decay from s to t is ``Gamma(t, s) = diag(exp(G_t - G_s))``. The generalized
delta rule (`~bux.architecture.recurrent.linear_attention.recurrence`)::

    S_t = diag(w_t) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T,    o_t = S_t^T q_t,

reads ``x_t = S_{t-1}^T a_t`` of the state, and the x of a chunk solve the unit lower
triangular system ``(I - L) X = Ã S_0 + L_k V``, with
``L[t, s] = a_t^T Gamma(t-1, s) b_s`` and ``L_k[t, s] = a_t^T Gamma(t-1, s) k_s`` for
s < t. So ``X = W S_0 + U`` (the WY form), and the outputs and the state the chunk
ends with are affine in S_0 too::

    o = R S_0 + O,    S_C = A S_0 + B.

Every chunk's R, O, A and B come from matmuls over all chunks at once, and a scan over
the chunks then carries the state; for one decay a head, a matmul over them. Nothing
before a document's start reaches it.

No exponent here is positive: a decay per key channel factors through the end of a
block of `SUB` positions between s and t, and within a block it is taken one pair at a
time. A decay bounded below, as RWKV-7's is, factors through the chunk's middle
position instead, one matmul a pair product, its exponents positive but small.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from bux import parallel
from bux.architecture.recurrent import affine_scan
from bux.parallel.recurrence import Chunk, Positions, Transition, then

SIZE = 64  # positions per chunk
SUB = 16  # positions per block of a chunk, for a decay per key channel
INVERSE = "recurrence.inverse"  # the name of each chunk's (I - L)^-1, which is kept
# The most a chunk may decay a channel, as a log, for its pairs to factor through its
# middle position: a factor's exponent is at most half of it, and exp(80) is finite
# in float32.
FACTORED = 80.0

Chunks = Float[Array, "b n h c _"]  # (b, n, h, C, ...): n chunks of C positions
Grouped = Float[Array, "b n g c _"]  # chunks of g heads, or of groups of h
Pairs = Float[Array, "b n h c c"]  # of each position of a chunk with each
States = Float[Array, "b n h dk dv"]  # one state per chunk
Maps = Float[Array, "b n h dk dk"] | Float[Array, "b n h _"]  # A per chunk, or diagonal


def chunk(
    xs: Positions,
    transition: bool,
    size: int = SIZE,
    sub: int = SUB,
    floor: float | None = None,
) -> Chunk:
    """Run ``linear_attention.chunk``'s `xs` (q, k, v, log decay, continues, and a
    and b if any) from a zero state, in float32, by chunks of `size` positions; with
    `transition`, also return its map of a start state, and the reads. The delta
    rule's xs have a and no b, which is k: `_delta` takes them in half the work. With
    one decay a head and no a or b, q and k may be of fewer heads than v, each shared
    by a group of them, as Mamba-2's C and B: `_scalar` takes them. `floor`, the
    least log decay a position can have, if bounded, lets a decay per channel factor
    through each chunk's middle (`_factored`)."""
    delta = len(xs) == 6
    xs = tuple(shard_whole_positions(x) for x in xs)
    q, k, v, g, *ab = (_split(x.astype(jnp.float32), size) for x in xs[:4] + xs[5:])
    starts = 1 - xs[4].astype(jnp.float32)  # 1 where a document starts
    seg = jnp.cumsum(_split(starts[..., None, None], size)[:, :, :, :, 0], -1)
    if g.shape[-1] == 1 and not ab:
        out, state, P, reads = _scalar(q, k, v, g, seg, transition)
    else:
        affine = _delta if delta else _affine
        ab = ab[: 1 if delta else 2]
        blocks = None if _factors(floor, size) else min(sub, size)
        R, O, A, B = affine(q, k, v, g, seg, *ab, sub=blocks)
        out, state, P, reads = _across(R, O, A, B, transition)
    t = xs[0].shape[1]
    reads = None if reads is None else shard_whole_positions(_merge(reads, t))
    return Chunk(shard_whole_positions(_merge(out, t)), state, P, reads)


def _factors(floor: float | None, size: int) -> bool:
    """Return whether a decay of at least `floor` a position factors through the
    middle of a chunk of `size` positions (`FACTORED`)."""
    return floor is not None and -floor * size <= FACTORED


def shard_whole_positions(x: Float[Array, "b t *e"]) -> Float[Array, "b t *e"]:
    """Gather `x`, of (b, t), (b, t, h) or (b, t, h, e), whole along the positions, as
    the chunks' reshape takes it: with `seq` split under `scan`, an all-gather (of the
    outputs' gradients, too) where XLA would rematerialize it in full. Heads stay
    split."""
    return parallel.shard(x, *("batch", None, "heads", None)[: x.ndim])


def _split(x: Float[Array, "b t *e"], size: int) -> Chunks:
    """Reshape (b, t, h, e) into chunks (b, n, h, C, e), padding the last with
    zeros."""
    b, t = x.shape[:2]
    n = -(-t // size)
    x = jnp.pad(x, [(0, 0), (0, n * size - t)] + [(0, 0)] * (x.ndim - 2))
    return jnp.moveaxis(x.reshape(b, n, size, *x.shape[2:]), 2, 3)


def _merge(x: Chunks, t: int) -> Float[Array, "b t h e"]:
    """Undo `_split`."""
    b, n, h, c, e = x.shape
    return jnp.moveaxis(x, 3, 2).reshape(b, n * c, h, e)[:, :t]


class _Decays(NamedTuple):
    G: Chunks  # the log decays summed up to each position: G_t
    before: Chunks  # G_{t-1}
    end: Float[Array, "b n h 1 _"]  # G at the chunk's end
    reach: Bool[Array, "b n 1 c 1"]  # where no document has started since the chunk's
    same: Bool[Array, "b n 1 c c"]  # pairs of positions of the same document
    tail: Bool[Array, "b n 1 c 1"]  # positions of the chunk-end's document


def _decays(g: Chunks, seg: Float[Array, "b n 1 c"]) -> _Decays:
    """Return each chunk's summed decays and document masks, from its log decays `g`
    and `seg`, the documents started by each position."""
    G = jnp.cumsum(g, -2)
    same = seg[..., :, None] == seg[..., None, :]
    tail = jnp.swapaxes(same[..., -1:, :], -1, -2)
    return _Decays(G, G - g, G[..., -1:, :], (seg == 0)[..., None], same, tail)


def _affine(
    q: Chunks,
    k: Chunks,
    v: Chunks,
    g: Chunks,
    seg: Float[Array, "b n 1 c"],
    a: Chunks | None = None,
    b: Chunks | None = None,
    *,
    sub: int | None,
) -> tuple[Chunks, Chunks, Maps, States]:
    """Return each chunk's R, O, A and B: its outputs R S_0 + O and end state
    A S_0 + B from a start state S_0, where `seg` counts the documents started by each
    position. A is a matrix with a and b, else its diagonal (of dk, or of 1). `sub`
    is `_pairwise`'s."""
    G, before, end, reach, same, tail = _decays(g, seg)
    qd = jnp.where(reach, q * jnp.exp(G), 0)  # Gamma(t, 0) q_t
    kd = jnp.where(tail, k * jnp.exp(end - G), 0)  # Gamma(C, s) k_s
    O = _pairwise(q, k, G, G, same, False, sub) @ v
    A = jnp.where(reach[..., -1, :], jnp.exp(end[..., 0, :]), 0)
    B = jnp.swapaxes(kd, -1, -2) @ v
    if a is None or b is None:
        return qd, O, A, B
    T = _inverse(_pairwise(a, b, before, G, same, True, sub))
    W = T @ jnp.where(reach, a * jnp.exp(before), 0)
    U = T @ (_pairwise(a, k, before, G, same, True, sub) @ v)
    M = _pairwise(q, b, G, G, same, False, sub)
    bd = jnp.swapaxes(jnp.where(tail, b * jnp.exp(end - G), 0), -1, -2)
    A = A[..., None] * jnp.eye(a.shape[-1]) + bd @ W
    return qd + M @ W, O + M @ U, A, B + bd @ U


def _delta(
    q: Chunks,
    k: Chunks,
    v: Chunks,
    g: Chunks,
    seg: Float[Array, "b n 1 c"],
    a: Chunks,
    *,
    sub: int | None,
) -> tuple[Chunks, Chunks, Maps, States]:
    """Compute `_affine` where b is k: L_k = L, so U = T L V = T V - V, and the
    outputs and the end state read the chunk's values through T alone, O = M T V and
    B = K^T T V (Gamma'd), with two pairwise products where `_affine` takes four."""
    G, before, end, reach, same, tail = _decays(g, seg)
    T = _inverse(_pairwise(a, k, before, G, same, True, sub))
    M = _pairwise(q, k, G, G, same, False, sub)
    W, X = T @ jnp.where(reach, a * jnp.exp(before), 0), T @ v
    kd = jnp.swapaxes(jnp.where(tail, k * jnp.exp(end - G), 0), -1, -2)
    A = jnp.where(reach[..., -1, :], jnp.exp(end[..., 0, :]), 0)
    A = A[..., None] * jnp.eye(k.shape[-1]) + kd @ W
    return jnp.where(reach, q * jnp.exp(G), 0) + M @ W, M @ X, A, kd @ X


def _scalar(
    q: Grouped,
    k: Grouped,
    v: Chunks,
    g: Chunks,
    seg: Float[Array, "b n 1 c"],
    transition: bool,
) -> tuple[Chunks, Float[Array, "b h dk dv"], Transition | None, Chunks | None]:
    """Return `_across`'s outputs, end state, P and reads for one decay a head and no
    a or b, where q and k may be of groups of v's heads. The decays scale v and the
    outputs, not q and k, so that a group's q and k reach all its heads at once: what
    a chunk adds to the state is K^T V of every head of a group in one matmul, and so
    is what its outputs read of the state before it, which `_passed` gives."""
    G, _, end, reach, same, tail = _decays(g, seg)
    O = _pairwise(q, k, G, G, same, False, 1) @ v
    vd = jnp.where(tail, v * jnp.exp(end - G), 0)  # Gamma(C, s) v_s
    adds = _by_group(jnp.swapaxes(k, -1, -2), vd)
    # Each chunk's log decay: -inf across a document start, which its log would
    # take from a decay of 0 at an infinite slope.
    log = jnp.where(reach[..., -1, 0], end[..., 0, 0], -jnp.inf)
    before, S, P, starts = _passed(log, adds)
    decay = jnp.where(reach, jnp.exp(G), 0)  # Gamma(t, 0)
    out = O + decay * _by_group(q, before)
    if not transition:
        return out, S, None, None
    return out, S, P, _shared(q, v.shape[2]) * decay * starts[..., None, None]


# A plain matmul: on a B300, XLA's Triton fusions of the einsum's gradients and the
# pairs' took C's and B's gradients wrong at Mamba-2's shapes.
def _by_group(
    x: Float[Array, "b n g i j"], y: Float[Array, "b n h j e"]
) -> Float[Array, "b n h i e"]:
    """Return x @ y for every head of y, x of its group: one matmul a group, its
    heads' columns side by side."""
    b, n, h, j, e = y.shape
    groups = x.shape[2]
    split = y.reshape(b, n, groups, h // groups, j, e)
    columns = jnp.moveaxis(split, 3, 4).reshape(b, n, groups, j, -1)
    out = (x @ columns).reshape(b, n, groups, x.shape[3], h // groups, e)
    return jnp.moveaxis(out, 4, 3).reshape(b, n, h, x.shape[3], e)


def _pairwise(
    x: Grouped,
    y: Grouped,
    gx: Chunks,
    gy: Chunks,
    same: Bool[Array, "b n 1 c c"],
    strict: bool,
    sub: int | None,
) -> Pairs:
    """Return sum_c x[t, c] y[s, c] exp(gx[t, c] - gy[s, c]) for s before t (or at t,
    unless `strict`) of the `same` document, else 0; gx <= gy at every such pair. With
    one decay per head, x and y may be of groups of gx's heads. A decay per channel
    goes by blocks of `sub` positions (`_channels`), or with `sub` None, bounded,
    through the chunk's middle (`_factored`)."""
    c = x.shape[-2]
    index = jnp.arange(c)
    lower = index[:, None] > index[None, :] if strict else index[:, None] >= index
    visible = same & lower
    if gx.shape[-1] == 1:  # one decay per head: a matmul, then the decays
        gap = jnp.where(visible, gx - jnp.swapaxes(gy, -1, -2), -jnp.inf)
        return _shared(x @ jnp.swapaxes(y, -1, -2), gx.shape[2]) * jnp.exp(gap)
    if sub is None:
        return jnp.where(visible, _factored(x, y, gx, gy), 0)
    return jnp.where(visible, _channels(x, y, gx, gy, lower, sub), 0)


def _factored(x: Chunks, y: Chunks, gx: Chunks, gy: Chunks) -> Pairs:
    """Compute `_pairwise` for a decay per key channel that `_factors`, at every pair:
    one matmul of x decayed from the chunk's middle position and y to it, by
    exp(gx - G_mid) and exp(G_mid - gy), whose product is the pair's decay."""
    c = x.shape[-2]
    # The pairs do not depend on G_mid, so neither do their gradients.
    middle = jax.lax.stop_gradient(gy[..., c // 2 : c // 2 + 1, :])
    left = x * jnp.exp(gx - middle)
    right = y * jnp.exp(middle - gy)
    return left @ jnp.swapaxes(right, -1, -2)


def _shared(x: Float[Array, "b n g ..."], heads: int) -> Float[Array, "b n h ..."]:
    """Share each group's x, (b, n, groups, ...), out to its heads."""
    return jnp.repeat(x, heads // x.shape[2], axis=2)


# Rematerialized: a pair's decays, sub times the input, would stay for the backward.
@partial(jax.checkpoint, static_argnums=(5,))
def _channels(
    x: Chunks, y: Chunks, gx: Chunks, gy: Chunks, lower: Bool[Array, "c c"], sub: int
) -> Pairs:
    """Compute `_pairwise` for a decay per key channel, at the pairs `lower` keeps.
    From each block of `sub` positions to those after it, through the decay at the
    block's end, by a matmul; within a block, one decay per pair."""
    *lead, c, e = x.shape
    n = c // sub
    xs, ys, gxs, gys = (z.reshape(*lead, n, sub, e) for z in (x, y, gx, gy))
    ends = gys[..., -1, :]  # (..., n, e): G at each block's end
    later = (jnp.arange(c)[:, None] // sub > jnp.arange(n))[:, :, None]  # (c, n, 1)
    gap = jnp.where(later, gx[..., :, None, :] - ends[..., None, :, :], -jnp.inf)
    left = x[..., :, None, :] * jnp.exp(gap)  # (..., c, n, e)
    right = ys * jnp.exp(ends[..., :, None, :] - gys)  # (..., n, sub, e)
    across = jnp.einsum("...tje,...jse->...tjs", left, right).reshape(*lead, c, c)
    inner = lower[:sub, :sub][..., None]  # (sub, sub, 1)
    gap = jnp.where(inner, gxs[..., :, None, :] - gys[..., None, :, :], -jnp.inf)
    within = jnp.sum(xs[..., :, None, :] * ys[..., None, :, :] * jnp.exp(gap), -1)
    diagonal = jnp.einsum("...jts,jk->...jtks", within, jnp.eye(n))
    return across + diagonal.reshape(*lead, c, c)


@jax.custom_vjp
def _inverse(L: Pairs) -> Pairs:
    """Return T = (I - L)^-1 of a strictly lower triangular L (`_halves`), named
    `INVERSE`. Its backward pass needs T alone."""
    return _inverse_forward(L)[0]


def _inverse_forward(L: Pairs) -> tuple[Pairs, Pairs]:
    T = checkpoint_name(_halves(L), INVERSE)
    return T, T


def _inverse_backward(T: Pairs, dT: Pairs) -> tuple[Pairs]:
    """T moves by T dL T, so L's gradient is T^T dT T^T, below the diagonal: where L
    has entries."""
    Tt = jnp.swapaxes(T, -1, -2)
    return (jnp.tril(_matmul(_matmul(Tt, dT), Tt), -1),)


_inverse.defvjp(_inverse_forward, _inverse_backward)


def _halves(L: Pairs, leaf: int = 16) -> Pairs:
    """Return (I - L)^-1 of a strictly lower triangular L, by halves: that of
    [[L1, 0], [L21, L2]] is [[T1, 0], [T2 L21 T1, T2]], down to blocks of `leaf`
    solved one row at a time. Every product is of entries of the inverse, which stay
    bounded where the powers of L need not (keys alike, steps near 1)."""
    c = L.shape[-1]
    blocks = c // leaf
    size = leaf if c > leaf and c % leaf == 0 and blocks & (blocks - 1) == 0 else c
    T = _substitute(_blocks(L, size, 0))
    while size < c:
        T1, T2 = T[..., 0::2, :, :], T[..., 1::2, :, :]
        X = _matmul(_matmul(T2, _blocks(L, 2 * size, 1)), T1)
        top = jnp.concatenate([T1, jnp.zeros_like(T1)], -1)
        T = jnp.concatenate([top, jnp.concatenate([X, T2], -1)], -2)
        size *= 2
    return T[..., 0, :, :]


def _blocks(L: Pairs, size: int, below: int) -> Float[Array, "b n h m s s"]:
    """Return the blocks of `size` on L's diagonal, (..., c / size, size, size); with
    `below`, each lower left quarter of them instead."""
    *lead, c, _ = L.shape
    m = c // size
    diagonal = jnp.diagonal(L.reshape(*lead, m, size, m, size), axis1=-4, axis2=-2)
    diagonal = jnp.moveaxis(diagonal, -1, -3)  # (..., m, size, size)
    if not below:
        return diagonal
    half = size // 2
    return diagonal[..., half:, :half]


def _substitute(L: Float[Array, "b n h m s s"]) -> Float[Array, "b n h m s s"]:
    """Return (I - L)^-1 of strictly lower triangular blocks, one row at a time: row
    i is e_i + L_i T, of the rows before it."""
    s = L.shape[-1]

    def row(
        i: Int[Array, ""], T: Float[Array, "b n h m s s"]
    ) -> Float[Array, "b n h m s s"]:
        Li = jax.lax.dynamic_index_in_dim(L, i, -2, keepdims=True)
        Ti = jax.nn.one_hot(i, s, dtype=T.dtype) + _matmul(Li, T)
        return jax.lax.dynamic_update_index_in_dim(T, Ti, i, -2)

    eye = jnp.zeros_like(L) + jnp.eye(s, dtype=L.dtype)
    return jax.lax.fori_loop(1, s, row, eye)


def _matmul(
    x: Float[Array, "*b i j"], y: Float[Array, "*b j k"]
) -> Float[Array, "*b i k"]:
    """Return x @ y in full float32: the inverse's rounding grows with its size."""
    return jnp.matmul(x, y, precision=jax.lax.Precision.HIGHEST)


_Carry = tuple[Float[Array, "b h dk dv"], Transition | None]  # the state, and P
_At = tuple[
    Float[Array, "b h c dk"],
    Float[Array, "b h c dv"],
    Transition,
    Float[Array, "b h dk dv"],
]
_Out = tuple[Float[Array, "b h c dv"], Float[Array, "b h c dk"] | None]


def _across(
    R: Chunks, O: Chunks, A: Maps, B: States, transition: bool
) -> tuple[Chunks, Float[Array, "b h dk dv"], Transition | None, Chunks | None]:
    """Return the chunks' outputs, from a zero state that each carries to the next,
    and the state they end with; with `transition`, also the map P of a start state
    that they make together, and the outputs' reads R P. Matrices A carry the state by
    ``affine_scan``'s kernel, and the outputs read it in one batched matmul."""
    if A.ndim == 5 and not transition:
        before, S = affine_scan.states(A, B)
        return O + R @ before, S, None, None
    R, O, A, B = (jnp.moveaxis(x, 1, 0) for x in (R, O, A, B))
    P = None
    if transition:
        diagonal = A.ndim == 4
        P = (
            jnp.ones_like(A[0])
            if diagonal
            else jnp.zeros_like(A[0]) + jnp.eye(A.shape[-1])
        )

    def step(carry: _Carry, at: _At) -> tuple[_Carry, _Out]:
        S, P = carry
        R, O, A, B = at
        out = O + R @ S
        S = then(A, S) + B
        if P is None:
            return (S, None), (out, None)
        reads = R @ P if P.ndim == 4 else R * P[..., None, :]
        return (S, then(A, P)), (out, reads)

    (S, P), (out, reads) = jax.lax.scan(step, (jnp.zeros_like(B[0]), P), (R, O, A, B))
    reads = None if reads is None else jnp.moveaxis(reads, 0, 1)
    return jnp.moveaxis(out, 0, 1), S, P, reads


def _passed(
    log: Float[Array, "b n h"], B: States
) -> tuple[
    States, Float[Array, "b h dk dv"], Float[Array, "b h 1"], Float[Array, "b n h"]
]:
    """Return, from each chunk's log decay, the state before each chunk and after the
    last, and the product of the chunks' decays, and of those before each. The states
    are one matmul of what the chunks add, by the decays of the chunks between (SSD's
    state passing), where a scan would take a chunk at a time."""
    index = jnp.arange(log.shape[1])
    later = (index[:, None] > index)[:, None]  # (n, 1, n): chunk i or l after j
    # The log decays of the chunks l after j up to i, at [i, j], by sums, not
    # differences, which -inf at a document start would make nan; then up to i - 1.
    upto = jnp.cumsum(jnp.where(later, log[..., None], 0), 1)
    between = jnp.pad(upto, [(0, 0), (1, 0), (0, 0), (0, 0)])[:, :-1]
    decays = jnp.where(later, jnp.exp(between), 0)
    before = jnp.einsum("bihj,bjhkv->bihkv", decays, B)
    S = jnp.exp(log[:, -1, :, None, None]) * before[:, -1] + B[:, -1]
    total = jnp.cumsum(log, 1)
    starts = jnp.exp(jnp.pad(total, [(0, 0), (1, 0), (0, 0)])[:, :-1])
    return before, S, jnp.exp(total[:, -1, :, None]), starts
