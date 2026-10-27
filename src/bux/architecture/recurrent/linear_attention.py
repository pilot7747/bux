"""Linear attention: token mixers with one matrix-valued state per head, updated token
by token. Gated DeltaNet, Kimi Delta Attention, Gated DeltaNet-2, DeltaNet, GLA and
Lightning are here, and RWKV-7 is in ``rwkv7``.

All of them run one recurrence (`recurrence`), over a state S of dk by dv per value
head::

    S_t = diag(w_t) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T,    o_t = S_t^T q_t

with a decay w_t in (0, 1] per key channel, or one per head. The ``b a^T`` term is how
the delta rule writes over what S holds at a key (`delta_rule`), and RWKV-7
generalizes it. The state is zero at every document start, so a packed row gives each
document what it gives alone.

The layers, each a config of `bux.config`:

- ``gla``: `GatedLinearAttention`, a decay per key channel, no delta rule;
- ``deltanet``: `DeltaNet`, the delta rule, no decay;
- ``gdn``: `GatedDeltaNet`, the delta rule with a decay per head;
- ``kda``: `KimiDeltaAttention`, the delta rule with a decay per key channel;
- ``gdn2``: `GatedDeltaNet2`, KDA whose step is two gates per channel, an erase and a
  write gate;
- ``lightning``: `LightningAttention`, a fixed decay per head.

Shapes: b rows, t positions, d the width, h heads (value heads, once q and k are shared
out to them), dk and dv the head dims of keys and values.

`recurrence` runs by chunks of positions (``chunkwise``, the form of fla's chunked
kernels), and `stepwise` is its reference, a scan over positions. It runs by the run's
``methods.recurrence``: ``scan``, the default, here, and ``pass_state``, which splits
the positions over ``context``, in `bux.parallel.recurrence`. On a GPU, kernels run
the delta rule (`delta_rule`, `gated_delta_2`) and decayed linear attention
(`decayed`) instead.

Under tensor parallelism (`bux.parallel`) every projection splits over ``heads``, into
the heads or out of them, and so does every parameter a head has: the recurrence runs
on each device's own heads, with no communication.
"""

import functools
import math
from collections.abc import Callable
from types import ModuleType
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux import initializers, parallel
from bux.architecture.common.functional import activation
from bux.architecture.common.segments import Segments
from bux.architecture.common.sharding import projection, shard_heads, shard_layer_input
from bux.architecture.layers.conv import CausalConv
from bux.architecture.layers.norm import Norm
from bux.architecture.recurrent import chunkwise, fused_delta, pallas_delta
from bux.architecture.registry import Context, Taps, register, segments
from bux.architecture.repeats import remat
from bux.lang import ir
from bux.parallel import methods
from bux.parallel.recurrence import Chunk, Positions, Recurrence, Shared, Transition

# Xavier's uniform at a gain of 2^-2.5, a variance of 2^-5 over the fans' mean: where
# Gated DeltaNet-2's projections start.
XAVIER = nnx.initializers.variance_scaling(2**-5, "fan_avg", "uniform")


@methods.recurrence.register("scan")
def scan(r: Recurrence) -> Float[Array, "b t h dv"]:
    """Run every position as one chunk, the default of ``methods.recurrence``. With
    ``seq`` over ``context``, XLA gathers them all onto every device."""
    return r.chunk(r.xs, r.shared, False).out


def _rank_one(
    s: Float[Array, "b h dk n"],
    w: Float[Array, "b h #dk"],
    a: Float[Array, "b h dk"] | None,
    b: Float[Array, "b h dk"] | None,
) -> Float[Array, "b h dk n"]:
    """Return (diag(w) + b a^T) s, or diag(w) s without a and b: one position's map
    applied to the state, or to a transition."""
    decayed = w[..., None] * s
    if a is None or b is None:
        return decayed
    return decayed + b[..., None] * jnp.einsum("bhk,bhkn->bhn", a, s)[..., None, :]


def _reads(transition: Transition, q: Float[Array, "b h dk"]) -> Float[Array, "b h dk"]:
    """Return P^T q, for the transition P from a chunk's start: what the output reads
    of the state the chunk starts from."""
    if transition.ndim == 4:
        return jnp.einsum("bhij,bhi->bhj", transition, q)
    return transition * q


# One position: q, k, v, the log decay, 1 where the token continues a document and 0
# where it starts one, and a and b if any.
_At = tuple[Float[Array, "b ..."], ...]
_Carry = tuple[Float[Array, "b h dk dv"], Transition | None]  # the state, and P


def _step(
    carry: _Carry, at: _At
) -> tuple[_Carry, tuple[Float[Array, "b h dv"], Float[Array, "b h dk"] | None]]:
    state, P = carry
    q, k, v, log_decay, continues, *ab = at
    w, keep = jnp.exp(log_decay), continues[:, None, None]
    a, b = ab or (None, None)
    state = _rank_one(state * keep[..., None], w, a, b) + k[..., None] * v[..., None, :]
    out = jnp.einsum("bhk,bhkv->bhv", q, state)
    if P is None:
        return (state, None), (out, None)
    P = _rank_one(P * keep[..., None], w, a, b) if ab else w * keep * P
    return (state, P), (out, _reads(P, q))


def _identity(
    like: Float[Array, "..."], shape: tuple[int, ...], matrix: bool
) -> Transition:
    """Return the transition of no positions, I of (b, h, dk, dk), or its diagonal of
    (b, h, dk): an array like `like`, so that in a shard_map it varies where `like`
    does."""
    if matrix:
        return jnp.zeros_like(like, shape=(*shape, shape[-1])) + jnp.eye(shape[-1])
    return jnp.ones_like(like, shape=shape)


def chunk(
    xs: Positions, shared: Shared, transition: bool, floor: float | None = None
) -> Chunk:
    """Run `recurrence` over a chunk of positions from a zero state, in float32: by
    smaller chunks, with matmuls within each and a scan across them (``chunkwise``).
    With `transition`, also return the product P of every position's diag(w) + b a^T
    (its diagonal without a and b), and each output's reads P^T q. `floor` is the
    least log decay, if bounded."""
    return chunkwise.chunk(xs, transition, floor=floor)


def stepwise(xs: Positions, shared: Shared, transition: bool) -> Chunk:
    """Run `chunk` one position at a time: a scan in float32, the reference."""
    q, k, v, log_decay, continues, *ab = (
        jnp.moveaxis(y.astype(jnp.float32), 1, 0) for y in xs
    )
    if len(ab) == 1:  # the delta rule's: b is k
        ab = [ab[0], k]
    _, batch, h, dk = k.shape
    zero = jnp.zeros_like(k, shape=(batch, h, dk, v.shape[-1]))
    shape = (batch, h, dk if ab else log_decay.shape[-1])
    P = _identity(k, shape, bool(ab)) if transition else None
    (state, P), (out, reads) = jax.lax.scan(
        _step, (zero, P), (q, k, v, log_decay, continues, *ab)
    )
    out = jnp.moveaxis(out, 0, 1)
    return Chunk(out, state, P, None if reads is None else jnp.moveaxis(reads, 0, 1))


States = Literal["keep", "recompute"]  # what a recurrence's backward pass reads


def recurrence(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    log_decay: Float[Array, "b t h #dk"],
    docs: Segments | None,
    a: Float[Array, "b t h dk"] | None = None,
    b: Float[Array, "b t h dk"] | None = None,
    *,
    floor: float | None = None,
    states: States | None = None,
) -> Float[Array, "b t h dv"]:
    """Return the outputs ``S_t^T q_t`` of the state::

        S_t = diag(w_t) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T,

    with ``w = exp(log_decay)``, one per key channel or one per head, and S zero
    before every document; without a and b, it is decayed linear attention. It runs
    by the run's ``methods.recurrence``, in float32, named ``recurrence``
    (``remat.named``), its chunks' states for the backward pass kept or computed again
    by `states`, by default the run's ``methods.states`` (`recomputed`). Where b is
    k, as in the delta rule, the positions leave b out, (q, k, v, log decay,
    continues, a), which ``chunkwise`` runs in half the work.

    Args:
        q: The queries, ``b t h dk``.
        k: The keys, ``b t h dk``.
        v: The values, ``b t h dv``.
        log_decay: log w, ``b t h dk``, or ``b t h 1`` for one decay a head; at most
            0.
        docs: The documents packed into the rows; None for one a row.
        a: What the rank-one term reads of the state; None for no such term.
        b: Where the rank-one term writes it.
        floor: The least log decay, if bounded: near enough 0, as RWKV-7's, the
            chunks take a decay per channel by fewer, larger matmuls.
        states: Keep the chunks' states for the backward pass, or recompute them;
            None for the run's ``methods.states``.

    Returns:
        The outputs, ``b t h dv``, in v's dtype.
    """
    rank_one = a is not None and b is not None
    family = "the generalized delta rule" if rank_one else "decayed linear attention"
    ab = (a,) if rank_one and b is k else (a, b) if rank_one else ()
    xs = (q, k, v, log_decay, _continues(k, docs), *ab)
    mix = recomputed(functools.partial(_mixed, family, floor), states)
    return remat.named("recurrence", mix, *xs).astype(v.dtype)


def decayed(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    log_decay: Float[Array, "b t h #dk"],
    docs: Segments | None,
) -> Float[Array, "b t h dv"]:
    """Run decayed linear attention, `recurrence` without a rank-one term::

        S_t = diag(w_t) S_{t-1} + k_t v_t^T,    o_t = S_t^T q_t.

    On a B200 or B300, with q, k and v in bfloat16 and heads of 64 or 128, cuDNN's
    GDN-2 kernel runs it instead (`~bux.architecture.recurrent.fused_delta`), keeping
    only its inputs for the backward pass, as the chunks it falls back on do.

    Args:
        q: The queries, ``b t h dk``.
        k: The keys, ``b t h dk``.
        v: The values, ``b t h dv``.
        log_decay: log w, ``b t h dk``, or ``b t h 1`` for one decay a head.
        docs: The documents packed into the rows; None for one a row.

    Returns:
        The outputs ``S_t^T q_t``, ``b t h dv``, in v's dtype.
    """
    if not fused_delta.linear_runs(k.shape[-1], v.shape[-1], v.dtype):
        return recurrence(q, k, v, log_decay, docs)
    otherwise = functools.partial(recurrence, states="recompute")

    def fused(*xs: Float[Array, "b t ..."]) -> Float[Array, "b t h dv"]:
        return fused_delta.linear(*xs, docs, otherwise)

    return remat.named("recurrence", fused, q, k, v, log_decay)


def _states() -> States:
    """Return the run's `methods.states`, or keep outside ``parallel.use``."""
    on = parallel.active()
    return on.methods.states if on else "keep"


def recomputed[**A, R](
    f: Callable[A, R], states: States | None = None
) -> Callable[A, R]:
    """Return `f`, or with `states = recompute` (by default the run's), `f` whose
    backward pass computes it again from its inputs, as fla's kernels do: of all it
    computes, it keeps only each chunk's (I - L)^-1 (``chunkwise.INVERSE``), for one
    more forward pass. So the delta rule keeps a fourteenth of the memory its chunks'
    states and maps take, and Mamba-2's selective scan (``ssm``), which has no
    inverse, a sixteenth."""
    if (states or _states()) == "keep":
        return f
    keep = jax.checkpoint_policies.save_only_these_names(chunkwise.INVERSE)
    return jax.checkpoint(f, policy=keep)


def _mixed(
    family: str, floor: float | None, *xs: Float[Array, "b t ..."]
) -> Float[Array, "b t h dv"]:
    """Run the recurrence of the positions `xs`, its log decays at least `floor`, by
    the run's `methods.recurrence`."""
    chunked = functools.partial(chunk, floor=floor)
    return parallel.method("recurrence")(Recurrence(family, chunked, xs))


def _continues(
    k: Float[Array, "b t ..."], docs: Segments | None
) -> Float[Array, "b t"]:
    """Return 1 where a token continues a document, and 0 where it starts one."""
    if docs is None:
        return jnp.ones(k.shape[:2])
    return (docs.within > 0).astype(jnp.float32)


def delta_rule(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    log_decay: Float[Array, "b t h #dk"],
    beta: Float[Array, "b t h"],
    docs: Segments | None,
    floor: float | None = None,
) -> Float[Array, "b t h dv"]:
    """Run the gated delta rule: each token decays the state, then moves what it holds
    at k_t a step beta_t toward v_t::

        S_t = (I - beta_t k_t k_t^T) diag(w_t) S_{t-1} + beta_t k_t v_t^T.

    That is `recurrence` with ``a = -w k``, and ``beta k`` for both b and k, in
    float32. On a GPU, with q, k and v of 16 bits, kernels run it instead, in their
    dtype: on a B200 or B300 cuDNN's (`~bux.architecture.recurrent.fused_delta`), one
    decay a head at any gate, and a decay per channel where `floor`, the least log
    decay, bounds it at -5; else bux's own
    (`~bux.architecture.recurrent.pallas_delta`), a decay per channel at any gate.
    cuDNN's keep only their inputs for the backward pass, and so do the chunks the
    kernels fall back on; bux's keep their chunks' parts and states too by the run's
    ``methods.states``.

    Args:
        q: The queries, ``b t h dk``.
        k: The keys, ``b t h dk``, L2-normalized.
        v: The values, ``b t h dv``.
        log_decay: log w, ``b t h dk``, or ``b t h 1`` for one decay a head.
        beta: The step of every head at every position, in (0, 1).
        docs: The documents packed into the rows; None for one a row.
        floor: The least log decay the layer can give, if bounded.

    Returns:
        The outputs ``S_t^T q_t``, ``b t h dv``, in v's dtype.
    """
    shapes = (k.shape[-1], v.shape[-1], v.dtype, log_decay.shape[-1])
    kernels = _kernels(*shapes, floor)
    if kernels is not None:
        otherwise = functools.partial(_chunked, states="recompute")

        def fused(*xs: Float[Array, "b t ..."]) -> Float[Array, "b t h dv"]:
            return kernels.delta_rule(*xs, docs, otherwise)

        return remat.named("recurrence", fused, q, k, v, log_decay, beta)
    return _chunked(q, k, v, log_decay, beta, docs)


def _kernels(
    dk: int, dv: int, dtype: Any, decays: int, floor: float | None
) -> ModuleType | None:
    """Return the module whose kernels run a delta rule of these head dims, its q, k
    and v of `dtype` and `decays` log decays a head, never below `floor`: none if
    no kernel does."""
    if fused_delta.runs(dk, dv, dtype, decays, floor):
        return fused_delta
    if pallas_delta.runs(dk, dv, dtype, decays):
        return pallas_delta
    return None


def _chunked(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    log_decay: Float[Array, "b t h #dk"],
    beta: Float[Array, "b t h"],
    docs: Segments | None,
    *,
    states: States | None = None,
) -> Float[Array, "b t h dv"]:
    """Run `delta_rule` by `recurrence`'s chunks, which with `states = recompute` (by
    default the run's) keep q, k, v, the log decay and beta, in their own dtype, for
    the backward pass."""

    def rule(
        q: Float[Array, "b t h dk"],
        k: Float[Array, "b t h dk"],
        v: Float[Array, "b t h dv"],
        log_decay: Float[Array, "b t h #dk"],
        beta: Float[Array, "b t h"],
        continues: Float[Array, "b t"],
    ) -> Float[Array, "b t h dv"]:
        k, log_decay = k.astype(jnp.float32), log_decay.astype(jnp.float32)
        step = beta[..., None] * k
        a = -jnp.exp(log_decay) * k
        family = "the generalized delta rule"
        return _mixed(family, None, q, step, v, log_decay, continues, a)

    xs = (q, k, v, log_decay, beta, _continues(k, docs))
    mix = recomputed(rule, states)
    return remat.named("recurrence", mix, *xs).astype(v.dtype)


def gated_delta_2(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    log_decay: Float[Array, "b t h dk"],
    erase: Float[Array, "b t h dk"],
    write: Float[Array, "b t h dv"],
    docs: Segments | None,
    floor: float | None = None,
) -> Float[Array, "b t h dv"]:
    """Run Gated DeltaNet-2's delta rule: each token decays the state, takes away what
    it reads there along ``b_t * k_t``, and writes ``w_t * v_t`` in its place at k_t::

        S_t = (I - k_t (b_t * k_t)^T) diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T,

    with the erase gate b over the key channels and the write gate w over the value
    channels. That is `recurrence` reading ``a = -exp(g) * b * k`` of the state, which
    it writes at k, and the values ``w * v``, in float32; at b = w = beta it is
    `delta_rule`'s. On a B200 or B300, with q, k and v in bfloat16, heads of 64 or
    128 and `floor`, the least log decay, at -5 or above, cuDNN's GDN-2 kernel runs it
    instead (`~bux.architecture.recurrent.fused_delta`), keeping only its inputs for
    the backward pass, as the chunks it falls back on do.

    Args:
        q: The queries, ``b t h dk``.
        k: The keys, ``b t h dk``, L2-normalized.
        v: The values, ``b t h dv``.
        log_decay: g, the log decay of each key channel, ``b t h dk``; at most 0.
        erase: The erase gate b, ``b t h dk``.
        write: The write gate w, ``b t h dv``.
        docs: The documents packed into the rows; None for one a row.
        floor: The least log decay the layer can give, if bounded.

    Returns:
        The outputs ``S_t^T q_t``, ``b t h dv``, in v's dtype.
    """
    chunked = functools.partial(_gated_chunked, floor=floor)
    if not fused_delta.gated_runs(k.shape[-1], v.shape[-1], v.dtype, floor):
        return chunked(q, k, v, log_decay, erase, write, docs)
    otherwise = functools.partial(chunked, states="recompute")

    def fused(*xs: Float[Array, "b t ..."]) -> Float[Array, "b t h dv"]:
        return fused_delta.gated(*xs, docs, otherwise)

    return remat.named("recurrence", fused, q, k, v, log_decay, erase, write)


def _gated_chunked(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    log_decay: Float[Array, "b t h dk"],
    erase: Float[Array, "b t h dk"],
    write: Float[Array, "b t h dv"],
    docs: Segments | None,
    *,
    floor: float | None = None,
    states: States | None = None,
) -> Float[Array, "b t h dv"]:
    """Run `gated_delta_2` by `recurrence`'s chunks, which with `states = recompute`
    (by default the run's) keep its inputs, in their own dtype, for the backward
    pass."""

    def rule(
        q: Float[Array, "b t h dk"],
        k: Float[Array, "b t h dk"],
        v: Float[Array, "b t h dv"],
        log_decay: Float[Array, "b t h dk"],
        erase: Float[Array, "b t h dk"],
        write: Float[Array, "b t h dv"],
        continues: Float[Array, "b t"],
    ) -> Float[Array, "b t h dv"]:
        k, log_decay = k.astype(jnp.float32), log_decay.astype(jnp.float32)
        a = -jnp.exp(log_decay) * erase * k
        written = write * v.astype(jnp.float32)
        family = "the generalized delta rule"
        return _mixed(family, floor, q, k, written, log_decay, continues, a)

    xs = (q, k, v, log_decay, erase, write, _continues(k, docs))
    mix = recomputed(rule, states)
    return remat.named("recurrence", mix, *xs).astype(v.dtype)


def l2norm(x: Float[Array, "*lead e"], eps: float = 1e-6) -> Float[Array, "*lead e"]:
    """Return x over its L2 norm as fla computes it, ``x / sqrt(sum(x^2) + eps)``: in
    float32, the result in x's dtype."""
    y = x.astype(jnp.float32)
    y = y * jax.lax.rsqrt(jnp.square(y).sum(-1, keepdims=True) + eps)
    return y.astype(x.dtype)


class LowRank(nnx.Module):
    """``x -> act(x A) B + c``, a projection of the width through `rank` dims
    (``latent``), with an optional activation between and bias after.

    Args:
        n_in: The width of the input.
        rank: The dims between A and B.
        n_out: The width of the output.
        rngs: The random streams A and B start from.
        activation: The activation between, by name; None for none.
        bias: Whether to add a bias c, from 0, or where c starts.
        orthogonal: Start it as RWKV-7 does, A at zero and B orthogonal, scaled by
            0.1; otherwise A and B start from `init`.
        out: The name of the outputs' dimension (`bux.parallel`).
        init: Where A and B start if the model sets no ``init``
            (`bux.initializers.weight`); None: as a weight matrix starts.
    """

    def __init__(
        self,
        n_in: int,
        rank: int,
        n_out: int,
        rngs: nnx.Rngs,
        *,
        activation: str | None = None,
        bias: Float[Array, " n"] | bool = False,
        orthogonal: bool = False,
        out: parallel.Name = None,
        init: nnx.Initializer | None = None,
    ):
        if orthogonal:
            gain = 0.1 * max(1.0, math.sqrt(rank / n_out))
            down = jnp.zeros((n_in, rank))
            up = nnx.initializers.orthogonal(gain)(rngs.params(), (rank, n_out))
        else:
            init = initializers.weight(own=init)
            down = init(rngs.params(), (n_in, rank))
            up = init(rngs.params(), (rank, n_out))
        self.down = nnx.Param(down, names=("embed", "latent"))
        self.up = nnx.Param(up, names=("latent", out))
        start = (jnp.zeros(n_out) if bias else None) if isinstance(bias, bool) else bias
        self.bias = None if start is None else nnx.Param(start, names=(out,))
        self.activation = activation

    def __call__(self, x: Float[Array, "*lead d"]) -> Float[Array, "*lead n"]:
        """Return ``act(x A) B + c``."""
        h = x @ self.down[...]
        if self.activation is not None:
            h = activation(self.activation, h)
        y = h @ self.up[...]
        return y if self.bias is None else y + self.bias[...]


def into_heads(
    width: int,
    heads: int,
    dim: int,
    rngs: nnx.Rngs,
    init: nnx.Initializer | None = None,
) -> nnx.Linear:
    """Make a projection of the width to `heads` heads of `dim` channels, side by
    side, its kernel from `init` if the model sets no ``init``; None: as a weight
    matrix starts."""
    start = initializers.weight(own=init)
    named = parallel.named("embed", ("heads", heads), init=start)
    return projection(
        width, heads * dim, "recurrence", use_bias=False, kernel_init=named, rngs=rngs
    )


def _dt_bias(rngs: nnx.Rngs, n: int) -> Float[Array, " n"]:
    """Return softplus^-1 of time steps log-uniform in [1e-3, 0.1], floored at 1e-4:
    how Mamba starts its time steps, and fla the dt_bias of Gated DeltaNet and KDA."""
    log_dt = jax.random.uniform(
        rngs.params(), (n,), minval=math.log(1e-3), maxval=math.log(0.1)
    )
    dt = jnp.maximum(jnp.exp(log_dt), 1e-4)
    return dt + jnp.log(-jnp.expm1(-dt))


def _log_uniform(rngs: nnx.Rngs, n: int, low: float, high: float) -> Float[Array, " n"]:
    """Return the logs of n draws uniform in [low, high]: where A_log starts."""
    return jnp.log(jax.random.uniform(rngs.params(), (n,), minval=low, maxval=high))


def _mixer_depths(arch: ir.Architecture, kind: str) -> tuple[list[int], int]:
    """Return where the `kind` layers are built, as indices among the token mixers
    the model runs (in the order they are built), and how many token mixers it runs.
    A shared layer is built where it first runs."""
    built: list[int] = []
    shared: set[str] = set()
    runs = 0

    def visit(node: ir.Node, fresh: bool) -> None:
        nonlocal runs
        if node.share is not None:
            fresh = fresh and node.share not in shared
            shared.add(node.share)
        if isinstance(node, ir.Layer) and isinstance(node.config, cfg.AttentionConfig):
            if fresh and node.config.type == kind:
                built.append(runs)
            runs += 1
        for _ in range(node.times if isinstance(node, ir.Repeat) else 1):
            for child in ir.children(node):
                visit(child, fresh)

    visit(arch.graph, True)
    return built, runs


def depth(ctx: Context, kind: str) -> tuple[int, int]:
    """Return the index of the `kind` layer being built among the model's token
    mixers, from 0, and how many token mixers the model runs: the layer index and
    count that MiniMax and RWKV-7 initialize a layer by. A layer built outside the
    model's graph counts as the first."""
    starts, layers = _mixer_depths(ctx.arch, kind)
    i = ctx.built.count[kind]
    return (starts[i] if i < len(starts) else 0), max(layers, 1)


class LinearAttention(nnx.Module):
    """The parts the linear-attention layers share: the q, k and v projections, the
    short causal conv on them, and the output's norm, gate and projection.

    With more value heads than q/k heads, each q/k head serves a group of value heads
    next to each other (grouped value attention, Qwen3-Next). The output norm is an
    RMSNorm of each head, or of all heads together (`norm_all_heads`), times the gate's
    activation of the input: the gated norm of Qwen3-Next and fla.

    Args:
        config: The layer's settings (`bux.config.LinearAttentionConfig`).
        width: The width of the stream, d.
        rngs: The random streams the parameters start from.
        silu: SiLU q, k and v without a conv; after one, they always are.
        gate_rank: The rank of a low-rank output gate; None for a dense one.
        gate_bias: Add a bias, from 0, to a low-rank output gate.
        norm_all_heads: Normalize all heads' outputs together, not each on its own.
        init: Where the projections' kernels start if the model sets no ``init``
            (`bux.initializers.weight`); None: as weight matrices start.

    Raises:
        ValueError: num_v_heads is not a multiple of num_heads.
    """

    def __init__(
        self,
        config: cfg.LinearAttentionConfig,
        width: int,
        rngs: nnx.Rngs,
        *,
        silu: bool = True,
        gate_rank: int | None = None,
        gate_bias: bool = False,
        norm_all_heads: bool = False,
        init: nnx.Initializer | None = None,
    ):
        h, hv = config.num_heads, config.num_v_heads or config.num_heads
        if hv % h:
            raise ValueError(f"num_v_heads={hv} is not a multiple of num_heads={h}")
        dk, dv = config.head_dim, config.v_head_dim or config.head_dim
        self.heads, self.v_heads, self.head_dim, self.v_head_dim = h, hv, dk, dv

        self.q, self.k, self.v = (
            into_heads(width, h, dk, rngs, init),
            into_heads(width, h, dk, rngs, init),
            into_heads(width, hv, dv, rngs, init),
        )
        # One conv over q, k and v side by side, which it convolves apart.
        size = config.conv_size
        self.conv = (
            CausalConv(2 * h * dk + hv * dv, size, False, rngs) if size else None
        )
        self.silu = silu  # on q, k and v without a conv; after one, always
        self.gate_activation = config.output_gate
        self.gate = (
            None
            if config.output_gate is None
            else LowRank(
                width,
                gate_rank,
                hv * dv,
                rngs,
                bias=gate_bias,
                out=("heads", hv),
                init=init,
            )
            if gate_rank
            else into_heads(width, hv, dv, rngs, init)
        )
        self.norm_all_heads = norm_all_heads
        rms = cfg.NormalizationConfig(type="rmsnorm", eps=config.norm_eps)
        self.norm = Norm(rms, hv * dv if norm_all_heads else dv)
        self.o = projection(
            hv * dv,
            width,
            "recurrence_out",
            use_bias=False,
            kernel_init=parallel.named(
                ("heads", hv), "embed", init=initializers.weight("out", own=init)
            ),
            rngs=rngs,
        )

    def _qkv(
        self, x: Float[Array, "b t d"], taps: Taps
    ) -> tuple[
        Float[Array, "b t h dk"], Float[Array, "b t h dk"], Float[Array, "b t h dv"]
    ]:
        """Project the queries, keys and values, convolve them and apply SiLU, and
        split them into heads, q and k shared out to the value heads."""
        b, t, _ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        if self.conv is not None:
            q, k, v = map(jax.nn.silu, self.conv.side_by_side((q, k, v), taps))
        elif self.silu:
            q, k, v = map(jax.nn.silu, (q, k, v))
        q, k = (
            self._per_value_head(y.reshape(b, t, self.heads, self.head_dim))
            for y in (q, k)
        )
        return q, k, shard_heads(v.reshape(b, t, self.v_heads, self.v_head_dim))

    def _per_value_head(self, y: Float[Array, "b t h e"]) -> Float[Array, "b t hv e"]:
        """Share each q/k head's y out to its group of value heads."""
        return jnp.repeat(shard_heads(y), self.v_heads // self.heads, axis=2)

    def _output(
        self, x: Float[Array, "b t d"], o: Float[Array, "b t h dv"]
    ) -> Float[Array, "b t d"]:
        """Normalize and gate the heads' outputs, and project them back to the
        width."""
        b, t, _ = x.shape
        o = shard_heads(o)
        o = self.norm(o.reshape(b, t, -1) if self.norm_all_heads else o)
        o = o.reshape(b, t, -1)
        if self.gate is not None and self.gate_activation is not None:
            o = o * activation(self.gate_activation, self.gate(x))
        return self.o(o)


class DeltaNet(LinearAttention):
    """DeltaNet (Schlag et al., 2021; Yang et al., 2024), ``deltanet``, as fla has it:
    the delta rule on L2-normalized q and k, with q scaled by ``dk ** -0.5``, and one
    step per value head, ``beta = sigmoid(W_b x)``. Gated DeltaNet and KDA add a
    decay.

    Args:
        config: The layer's settings (`bux.config.DeltaNetConfig`).
        width: The width of the stream, d.
        rngs: The random streams the parameters start from.
        gate_rank: The rank of a low-rank output gate; None for a dense one.
    """

    def __init__(
        self,
        config: cfg.LinearAttentionConfig,
        width: int,
        rngs: nnx.Rngs,
        *,
        gate_rank: int | None = None,
    ):
        super().__init__(config, width, rngs, gate_rank=gate_rank)
        self.b = into_heads(width, self.v_heads, 1, rngs)

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        """Return the mixer's output at every position of x, ``b t d``.

        Args:
            x: The stream.
            taps: The forward pass's taps, for the documents in the rows.
        """
        x = shard_layer_input(x)
        q, k, v = self._qkv(x, taps)
        q, k = l2norm(q) * self.head_dim**-0.5, l2norm(k)
        beta = shard_heads(jax.nn.sigmoid(self.b(x)))
        g, docs = shard_heads(self._log_decay(x)), segments(taps)
        return self._output(x, delta_rule(q, k, v, g, beta, docs, self._floor()))

    def _log_decay(self, x: Float[Array, "b t d"]) -> Float[Array, "b t h #dk"]:
        """Return the log decay of the state, per key channel or one per head: 0, for
        no decay."""
        b, t, _ = x.shape
        return jnp.zeros((b, t, self.v_heads, 1))

    def _floor(self) -> float | None:
        """Return the least log decay `_log_decay` can give, if it is bounded: 0."""
        return 0.0


class GatedDeltaNet(DeltaNet):
    """Gated DeltaNet (Yang et al., 2025), ``gdn``, as transformers' Qwen3-Next and fla
    have it: DeltaNet with a decay ``exp(g)`` per value head,
    ``g = -exp(A_log) softplus(W_a x + dt_bias)``, with A_log and dt_bias learned per
    value head.

    A_log starts at the logs of draws uniform in [0.01, 16] (transformers), and dt_bias
    at Mamba's time steps (fla and the paper's code; transformers starts it at ones).

    Args:
        config: The layer's settings (`bux.config.GatedDeltaNetConfig`).
        width: The width of the stream, d.
        rngs: The random streams the parameters start from.
    """

    def __init__(self, config: cfg.GatedDeltaNetConfig, width: int, rngs: nnx.Rngs):
        super().__init__(config, width, rngs)
        hv = self.v_heads
        self.a = into_heads(width, hv, 1, rngs)
        self.A_log = nnx.Param(_log_uniform(rngs, hv, 0.01, 16), names=("heads",))
        self.dt_bias = nnx.Param(_dt_bias(rngs, hv), names=("heads",))

    def _log_decay(self, x: Float[Array, "b t d"]) -> Float[Array, "b t h 1"]:
        rate = jax.nn.softplus(self.a(x).astype(jnp.float32) + self.dt_bias[...])
        return (-jnp.exp(self.A_log[...].astype(jnp.float32)) * rate)[..., None]

    def _floor(self) -> float | None:
        return None  # -exp(A_log) softplus(...) is unbounded


class KimiDeltaAttention(DeltaNet):
    """Kimi Delta Attention (Kimi Linear, 2025), ``kda``, as transformers and fla have
    it: Gated DeltaNet with a decay per key channel,
    ``g = -exp(A_log) softplus(f + dt_bias)``, where ``f = W_f2 W_f1 x`` is a low-rank
    projection (`gate_rank`) and A_log is per head. With a lower bound (Kimi K3,
    GLM-5-Next), ``g = lower_bound * sigmoid(exp(A_log) (f + dt_bias))``, and A_log
    starts at 0, as in fla's safe gate (K3's code starts it at ``log U(1, 16)``
    either way). The output gate is low rank too (`output_gate_rank`).

    Args:
        config: The layer's settings (`bux.config.KDAConfig`).
        width: The width of the stream, d.
        rngs: The random streams the parameters start from.
    """

    def __init__(self, config: cfg.KDAConfig, width: int, rngs: nnx.Rngs):
        super().__init__(config, width, rngs, gate_rank=config.output_gate_rank)
        hv, dk, rank = self.v_heads, self.head_dim, config.gate_rank
        self.f = LowRank(width, rank or dk, hv * dk, rngs, out=("heads", hv))
        self.lower_bound = config.gate_lower_bound
        self.A_log, self.dt_bias = _channel_rates(rngs, hv, dk, self.lower_bound)

    def _log_decay(self, x: Float[Array, "b t d"]) -> Float[Array, "b t h dk"]:
        b, t, _ = x.shape
        f = self.f(x).astype(jnp.float32) + self.dt_bias[...]
        f = f.reshape(b, t, self.v_heads, self.head_dim)
        return _channel_decay(f, self.A_log[...], self.lower_bound)

    def _floor(self) -> float | None:
        return self.lower_bound


def _channel_rates(
    rngs: nnx.Rngs, heads: int, dk: int, lower_bound: float | None
) -> tuple[nnx.Param, nnx.Param]:
    """Return KDA's A_log, one a head, and dt_bias, one a key channel: A_log at
    ``log U(1, 16)``, or at 0 for a decay bounded below, as fla's safe gate starts it,
    and dt_bias at Mamba's time steps."""
    unbounded = lower_bound is None
    rates = _log_uniform(rngs, heads, 1, 16) if unbounded else jnp.zeros(heads)
    A_log = nnx.Param(rates, names=("heads",))
    dt_bias = nnx.Param(_dt_bias(rngs, heads * dk), names=(("heads", heads),))
    return A_log, dt_bias


def _channel_decay(
    f: Float[Array, "b t h dk"], A_log: Float[Array, " h"], lower_bound: float | None
) -> Float[Array, "b t h dk"]:
    """Return KDA's log decay per key channel of ``f`` (its projection plus dt_bias,
    in float32): ``-exp(A_log) softplus(f)``, or with a lower bound
    ``lower_bound * sigmoid(exp(A_log) f)``, in float32."""
    rate = jnp.exp(A_log.astype(jnp.float32))[:, None]
    if lower_bound is not None:
        return lower_bound * jax.nn.sigmoid(rate * f)
    return -rate * jax.nn.softplus(f)


class GatedDeltaNet2(LinearAttention):
    """Gated DeltaNet-2 (Hatamizadeh et al., 2026), ``gdn2``, as the paper's code and
    fla have it: KDA whose step beta, which erases and writes alike, becomes two gates
    of channels, ``b = sigmoid(W_b x)`` over the keys' and ``w = sigmoid(W_w x)`` over
    the values' (`gated_delta_2`), on L2-normalized q and k, q scaled by
    ``dk ** -0.5``. The decay is KDA's, per key channel, of a low-rank projection
    (`gate_rank`); it, b and A_log are a q/k head's, shared out to its value heads as
    q and k are.

    Every projection starts from Xavier's uniform at a gain of 2^-2.5 (`XAVIER`), its
    bias at 0, as the paper's code starts them, unless the model sets an ``init``;
    the output gate is low rank, with a bias. The paper gates the output norm by SiLU, fla's port by a sigmoid
    (`output_gate`).

    Args:
        config: The layer's settings (`bux.config.GatedDeltaNet2Config`).
        width: The width of the stream, d.
        rngs: The random streams the parameters start from.
    """

    def __init__(self, config: cfg.GatedDeltaNet2Config, width: int, rngs: nnx.Rngs):
        dv = config.v_head_dim or config.head_dim
        gate_rank = config.output_gate_rank or dv
        super().__init__(
            config, width, rngs, gate_rank=gate_rank, gate_bias=True, init=XAVIER
        )
        h, dk, hv = self.heads, self.head_dim, self.v_heads
        rank = config.gate_rank or dv
        self.f = LowRank(width, rank, h * dk, rngs, out=("heads", h), init=XAVIER)
        self.erase = into_heads(width, h, dk, rngs, XAVIER)
        self.write = into_heads(width, hv, dv, rngs, XAVIER)
        self.lower_bound = config.gate_lower_bound
        self.A_log, self.dt_bias = _channel_rates(rngs, h, dk, self.lower_bound)
        self.erase_scale = 2.0 if config.negative_eigenvalues else 1.0

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        """Return the mixer's output at every position of x, ``b t d``.

        Args:
            x: The stream.
            taps: The forward pass's taps, for the documents in the rows.
        """
        x = shard_layer_input(x)
        q, k, v = self._qkv(x, taps)
        q, k = l2norm(q) * self.head_dim**-0.5, l2norm(k)
        g, erase, write = self._gates(x)
        o = gated_delta_2(q, k, v, g, erase, write, segments(taps), self.lower_bound)
        return self._output(x, o)

    def _gates(
        self, x: Float[Array, "b t d"]
    ) -> tuple[
        Float[Array, "b t h dk"], Float[Array, "b t h dk"], Float[Array, "b t h dv"]
    ]:
        """Return the log decay and the erase gate of every value head, and its write
        gate."""
        b, t, _ = x.shape
        h, dk = self.heads, self.head_dim
        f = (self.f(x).astype(jnp.float32) + self.dt_bias[...]).reshape(b, t, h, dk)
        g = _channel_decay(f, self.A_log[...], self.lower_bound)
        erase = self.erase_scale * jax.nn.sigmoid(self.erase(x).reshape(b, t, h, dk))
        write = jax.nn.sigmoid(self.write(x)).reshape(b, t, self.v_heads, -1)
        return self._per_value_head(g), self._per_value_head(erase), shard_heads(write)


class GatedLinearAttention(LinearAttention):
    """Gated linear attention (Yang et al., 2024), ``gla``, as fla has it: linear
    attention with q scaled by ``dk ** -0.5`` and a decay per key channel,
    ``log w = log sigmoid(f) / 16``, where f is a low-rank projection with a bias
    (`gate_rank`). Without a conv, q, k and v do not go through SiLU.

    Args:
        config: The layer's settings (`bux.config.GLAConfig`).
        width: The width of the stream, d.
        rngs: The random streams the parameters start from.
    """

    def __init__(self, config: cfg.GLAConfig, width: int, rngs: nnx.Rngs):
        super().__init__(config, width, rngs, silu=False)
        hv, rank = self.v_heads, config.gate_rank
        self.f = LowRank(
            width, rank, hv * self.head_dim, rngs, bias=True, out=("heads", hv)
        )

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        """Return the mixer's output at every position of x, ``b t d``.

        Args:
            x: The stream.
            taps: The forward pass's taps, for the documents in the rows.
        """
        x = shard_layer_input(x)
        b, t, _ = x.shape
        q, k, v = self._qkv(x, taps)
        f = self.f(x).reshape(b, t, self.v_heads, self.head_dim)
        g = shard_heads(jax.nn.log_sigmoid(f.astype(jnp.float32)) / 16)
        o = decayed(q * self.head_dim**-0.5, k, v, g, segments(taps))
        return self._output(x, o)


def slopes(heads: int, layer: int, layers: int) -> Float[Array, " h"]:
    """Return MiniMax's decay rates, ``s_i = 2 ** (-8 i / heads)`` for heads i = 1 to
    heads, scaled by ``1 - layer / (layers - 1)`` (and 1e-5 more, as MiniMax has it).
    The decay is ``exp(-s)``, which leaves a plain sum of the past at the last
    layer."""
    factor = 1 - layer / (layers - 1 + 1e-5) + 1e-5
    return 2 ** (-8 / heads * jnp.arange(1, heads + 1)) * factor


class LightningAttention(LinearAttention):
    """Lightning attention (TransNormerLLM; MiniMax-01, as transformers has it),
    ``lightning``: linear attention on SiLU'd q, k and v, unscaled, with a fixed decay
    ``exp(-s)`` per head that weakens with depth (`slopes`), and one RMSNorm of all
    the heads together before the gate.

    Args:
        config: The layer's settings (`bux.config.LightningAttentionConfig`).
        width: The width of the stream, d.
        rngs: The random streams the parameters start from.
        layer: The layer's index among the model's token mixers, from 0.
        layers: How many token mixers the model runs.
    """

    def __init__(
        self,
        config: cfg.LightningAttentionConfig,
        width: int,
        rngs: nnx.Rngs,
        *,
        layer: int = 0,
        layers: int = 1,
    ):
        super().__init__(config, width, rngs, norm_all_heads=True)
        self.slopes = nnx.Variable(
            slopes(self.v_heads, layer, layers), names=("heads",)
        )

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        """Return the mixer's output at every position of x, ``b t d``.

        Args:
            x: The stream.
            taps: The forward pass's taps, for the documents in the rows.
        """
        x = shard_layer_input(x)
        q, k, v = self._qkv(x, taps)
        g = jnp.broadcast_to(-self.slopes[...][:, None], (*q.shape[:3], 1))
        return self._output(x, decayed(q, k, v, g, segments(taps)))


@register("deltanet")
def _(
    c: cfg.DeltaNetConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> DeltaNet:
    return DeltaNet(c, width, ctx.rngs)


@register("gdn")
def _(
    c: cfg.GatedDeltaNetConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> GatedDeltaNet:
    return GatedDeltaNet(c, width, ctx.rngs)


@register("kda")
def _(
    c: cfg.KDAConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> KimiDeltaAttention:
    return KimiDeltaAttention(c, width, ctx.rngs)


@register("gdn2")
def _(
    c: cfg.GatedDeltaNet2Config, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> GatedDeltaNet2:
    return GatedDeltaNet2(c, width, ctx.rngs)


@register("gla")
def _(
    c: cfg.GLAConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> GatedLinearAttention:
    return GatedLinearAttention(c, width, ctx.rngs)


@register("lightning")
def _(
    c: cfg.LightningAttentionConfig,
    width: int,
    ports: tuple[ir.Port, ...],
    ctx: Context,
) -> LightningAttention:
    layer, layers = depth(ctx, "lightning")
    return LightningAttention(c, width, ctx.rngs, layer=layer, layers=layers)
