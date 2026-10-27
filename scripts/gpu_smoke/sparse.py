"""Sparse attention on GPUs: tokamax's flash kernel with the kept keys as a mask
(``bux.architecture.attention.sparse`` under `methods.attention = auto`) against XLA's
chunks of queries (inside ``bux.kernels.off()``), on one GPU and with the positions
split over 2: top-k tokens of GQA and MLA, QSA's blocks, MSA's blocks for each kv
group (a group at a time), CSA's window and latents (512 dims, which XLA runs), on
packed documents with a window, sinks and a soft cap; the output, the logsumexp and
the gradients. Then the indexer's kernels on a B200 or B300: cuDNN's top-k against
XLA's, bitwise, and its loss's log-softmax and gradients. Then the time of a forward
and backward of each, and of dense flash attention, from 4k to 32k tokens.

On a B200 or B300, cuDNN's DSA kernels (``sparse_cudnn``) against XLA the same way:
CSA's window and compressed latents at DeepSeek-V4.1-Flash's shapes and at the edges
of what they take, and the indexer's target at its picks; then their times.

    uv run --extra cuda python scripts/gpu_smoke/sparse.py

It needs tokamax and cudnn-frontend (the cuda extra), and exits 1 if a kernel differs
from XLA.
"""

import sys
import time
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

import bux.architecture  # noqa: F401  (registers the attention methods)
from bux import kernels, parallel
from bux.architecture import Segments
from bux.architecture.attention import (
    indexer,
    indexer_cudnn,
    scoring,
    sparse,
    sparse_cudnn,
)
from bux.parallel import methods
from bux.parallel.context import AttentionArgs
from bux.train import config as tcfg

DTYPE = jnp.bfloat16
TOLERANCE = 2e-2  # of the largest value, in bfloat16
Keys = list[tuple[Float[Array, "b s g dk"], Float[Array, "b s g dv"]]]


class Case(NamedTuple):
    heads: int
    kv_heads: int
    dk: int
    dv: int
    top_k: int  # kept slots, or blocks' slots
    block: int = 1  # QSA's blocks
    window: int | None = None
    csa: bool = False  # a window part, then latents every other slot
    sinks: bool = False
    softcap: float | None = None
    groups: bool = False  # MSA's blocks, picked for each kv head's group


CASES = {
    "gqa": Case(16, 8, 128, 128, 512),
    "gqa_window_sinks": Case(16, 8, 128, 128, 512, window=1000, sinks=True, softcap=30),
    "qsa_blocks": Case(16, 2, 128, 128, 512, block=4),
    "msa_groups": Case(16, 4, 128, 128, 512, block=128, groups=True),
    "mla": Case(16, 16, 192, 128, 512),
    "csa": Case(16, 1, 512, 512, 256, window=128, csa=True, sinks=True),
    "csa_128": Case(16, 1, 128, 128, 256, window=128, csa=True, sinks=True),
}


def documents(b: int, t: int, seed: int = 0) -> Segments:
    """Rows of a dozen documents of random lengths, cut mid-way."""
    rng = np.random.default_rng(seed)
    positions = np.zeros((b, t), np.int32)
    for row in range(b):
        starts = np.unique([0, *rng.integers(1, t, 12)])
        for a, z in zip(starts, [*starts[1:], t]):
            positions[row, a:z] = np.arange(z - a) + (7 if a == 0 else 0)
    return Segments.of(jnp.asarray(positions))


def kept(
    b: int, t: int, top_k: int, block: int, seed: int, groups: int | None = None
) -> sparse.Kept:
    """Each query's top_k random slots at or before it (with repeats), or random
    blocks and its own, for each of `groups` groups if given; the span cuts them down
    to its document's."""
    n, lead = top_k // block, (b, t) if groups is None else (b, t, groups)
    u = jax.random.uniform(jax.random.key(seed), (*lead, n))
    at = jnp.arange(t, dtype=jnp.int32).reshape(1, t, *(1,) * (len(lead) - 1))
    index = jnp.floor(u * (at // block + 1)).astype(jnp.int32)
    if block == 1:
        return sparse.Kept(index)
    own = jnp.broadcast_to(at // block, (*lead, 1))
    blocks = jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32) // block, (b, t))
    return sparse.Kept(jnp.concatenate([index, own], -1), blocks)


def parts(case: Case, rows: Segments, keys: Keys) -> list[sparse.Part]:
    """The case's parts, of the given keys and values."""
    b, t = rows.ids.shape
    span = sparse.token_span(rows, None if case.csa else case.window)
    groups = case.kv_heads if case.groups else None
    chosen = kept(b, t, case.top_k, case.block, seed=3, groups=groups)
    if not case.csa:
        return [sparse.Part(*keys[0], span, None, chosen)]
    window = sparse.Part(*keys[0], sparse.token_span(rows, case.window))
    exists = jnp.broadcast_to(jnp.arange(t) % 2 == 1, (b, t))
    return [window, sparse.Part(*keys[1], span, exists, chosen)]


def inputs(case: Case, b: int, t: int) -> tuple:
    """Queries, keys and values of each part, sinks, and a cotangent."""
    key = iter(jax.random.split(jax.random.key(0), 8))
    h, g = case.heads, case.kv_heads

    def normal(*shape: int) -> Float[Array, "..."]:
        return jax.random.normal(next(key), shape, DTYPE)

    keys = [(normal(b, t, g, case.dk), normal(b, t, g, case.dv))]
    if case.csa:
        latent = normal(b, t, g, case.dk)
        keys.append((latent, latent))
    sinks = jnp.linspace(-1, 1, h) if case.sinks else None
    return normal(b, t, h, case.dk), keys, sinks, normal(b, t, h, case.dv)


Step = jax.stages.Wrapped  # jitted (q, keys, sinks, dout) -> (grads, (out, lse))


def step(case: Case, rows: Segments) -> Step:
    """A jitted forward and backward of sparse attention on the case: the output and
    logsumexp, and the gradients of q, the keys and values, and the sinks."""

    def loss(
        q: Float[Array, "b t h dk"],
        keys: Keys,
        sinks: Float[Array, " h"] | None,
        dout: Float[Array, "b t h dv"],
    ) -> tuple[Float[Array, ""], tuple]:
        given = parts(case, rows, keys)
        scale = case.dk**-0.5
        out, lse = sparse.attend(q, given, sinks, scale, case.softcap)
        return (out.astype(jnp.float32) * dout).sum(), (out, lse)

    return jax.jit(jax.grad(loss, argnums=(0, 1, 2), has_aux=True))


def error(a: Float[Array, "*s"], b: Float[Array, "*s"]) -> float:
    """The largest difference, as a share of the largest value of `b`."""
    x, y = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.abs(x - y).max() / max(np.abs(y).max(), 1e-30))


def errors(got: tuple, want: tuple) -> list[float]:
    """Every output's and gradient's `error`, the -inf logsumexps left out."""
    pairs = list(zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True))
    return [
        error(jnp.where(jnp.isfinite(b), a, 0), jnp.where(jnp.isfinite(b), b, 0))
        for a, b in pairs
    ]


def kernel(f: Step, *args: object) -> str:
    """Which kernel the compiled step calls."""
    text = (f.lower(*args).compile().as_text() or "").lower()
    return "mosaic" if "mosaic" in text else "xla"


def check(name: str, case: Case, context: int) -> bool:
    """The kernel, split over `context` GPUs, against XLA on one; whether they agree."""
    b, t = 2, 4096
    rows, x = documents(b, t), inputs(case, b, t)
    with kernels.off():
        grads, aux = step(case, rows)(*x)
    with parallel.use(tcfg.MeshConfig(data=1, context=context)) as mesh:
        put = NamedSharding(mesh, P(None, "context"))
        split = jax.tree.map(lambda a: jax.device_put(a, put) if a.ndim > 1 else a, x)
        f = step(case, rows)
        got_grads, got_aux = f(*split)
        ran = kernel(f, *split)
    shown = errors((got_aux, got_grads), (aux, grads))
    widest = max(case.dk, case.dv)  # MLA's 192 and CSA's 512 run by XLA
    wanted = "mosaic" if widest <= parallel.flash.TOKAMAX_HEAD_DIM else "xla"
    ok = max(shown) < TOLERANCE and ran == wanted
    print(
        f"context={context} {name:16} {ran:6} out, lse, grads "
        + " ".join(f"{e:.1e}" for e in shown)
        + f"  {'ok' if ok else 'DIFFERS'}",
        flush=True,
    )
    return ok


def milliseconds(f: Step, *args: object) -> float:
    """The median time of a call of `f`, once compiled."""
    jax.block_until_ready(f(*args))
    times = []
    for _ in range(5):
        start = time.perf_counter()
        jax.block_until_ready(f(*args))
        times.append(time.perf_counter() - start)
    return 1e3 * float(np.median(times))


def dense(case: Case, rows: Segments) -> Step:
    """A jitted forward and backward of dense flash attention (`auto`)."""
    attention = methods.attention.find("auto")
    seen = AttentionArgs(rows.ids, case.window, case.dk**-0.5, case.softcap, None)

    def loss(
        q: Float[Array, "b t h dk"],
        keys: Keys,
        sinks: Float[Array, " h"] | None,
        dout: Float[Array, "b t h dv"],
    ) -> Float[Array, ""]:
        return (attention(q, *keys[0], seen).astype(jnp.float32) * dout).sum()

    return jax.jit(jax.grad(loss, argnums=(0, 1)))


def timings(name: str, case: Case) -> None:
    """Forward and backward ms of the kernel, XLA's chunks and dense flash."""
    for t in (4096, 16384, 32768):
        rows, x = documents(1, t), inputs(case._replace(top_k=2048), 1, t)
        sparse_case = case._replace(top_k=2048)
        shown = f"flash {milliseconds(step(sparse_case, rows), *x):8.2f}"
        if not case.csa:
            shown += f"  dense flash {milliseconds(dense(case, rows), *x):8.2f}"
        if t == 4096:
            with kernels.off():
                shown += f"  xla {milliseconds(step(sparse_case, rows), *x):8.2f}"
        print(f"time {name:16} t={t:6} top 2048, ms: {shown}", flush=True)


# --- The indexer's kernels: cuDNN's top-k and its loss's log-softmax ------------------


class Ranked(NamedTuple):
    """A chunk of queries' scores of every key, -inf where they don't see it."""

    scores: Float[Array, "b c n"]
    span: sparse.Span  # the queries' spans


def ranked(t: int, c: int, b: int = 1, docs: bool = False, ties: float = 0) -> Ranked:
    """The indexer's scores of a chunk of c queries (every t // c-th) of b rows of t,
    by `scoring`'s kernel; rounded to multiples of `ties` (0 for none, a huge one for
    all 0), of packed documents if `docs`."""
    key = iter(jax.random.split(jax.random.key(1), 3))
    q = jax.random.normal(next(key), (b, t, 64, 128), DTYPE)
    w = jax.random.normal(next(key), (b, t, 64), DTYPE) * 0.1
    keys = jax.random.normal(next(key), (b, t, 128), DTYPE)
    rows = (
        documents(b, t)
        if docs
        else Segments.of(jnp.broadcast_to(jnp.arange(t), (b, t)))
    )
    span = sparse.token_span(rows, None)
    pick = slice(t // c - 1, None, t // c)
    chunk = sparse.Span(span.start[:, pick], span.end[:, pick])
    scores = jax.jit(scoring.scores)(q[:, pick], w[:, pick], keys, chunk.end)
    if ties:
        scores = jnp.round(scores / ties) * ties
    return Ranked(jnp.where(sparse.mask(chunk, t), scores, -jnp.inf), chunk)


def top_k_check(name: str, case: Ranked, k: int = 2048) -> bool:
    """cuDNN's top-k against XLA's counting, bitwise."""
    with kernels.off():
        want = jax.jit(lambda s: indexer._top(s, k))(case.scores)
    got = jax.jit(lambda s, span: indexer._top(s, k, span))(*case)
    rows = int((got != want).any(-1).sum())
    ok = rows == 0 and indexer_cudnn.runs(min(k, case.scores.shape[-1]))
    print(f"top-k {name:22} k={k:4} rows differing {rows}  {'ok' if ok else 'DIFFERS'}")
    return ok


def picks(b: int, t: int, s: int, k: int, seed: int = 2) -> Array:
    """Each query's k distinct picks among s keys, -1 for those after it (so some
    before a kept one): a query at t's last slot sees every key."""
    u = jax.random.uniform(jax.random.key(seed), (b, t, s))
    index = jnp.argsort(u, -1)[..., :k].astype(jnp.int32)
    at = (jnp.arange(t) + 1) * (s // t) - 1
    return jnp.where(index <= at[None, :, None], index, -1)


def log_softmax_check(name: str, b: int, t: int, s: int, h: int, k: int) -> bool:
    """cuDNN's log-softmax of the picks' scores, and its gradients, against XLA's
    gather in float32: where the picks' probability is above 1e-30."""
    key = iter(jax.random.split(jax.random.key(3), 4))
    q = jax.random.normal(next(key), (b, t, h, 128), DTYPE)
    w = jax.random.normal(next(key), (b, t, h), DTYPE) * 0.1
    keys = jax.random.normal(next(key), (b, s, 128), DTYPE)
    index = picks(b, t, s, k)
    d = jax.random.normal(next(key), (b, t, k)) * (index >= 0)

    def xla(q: Array, w: Array, keys: Array) -> Array:
        rows = jax.vmap(lambda k, i: k[i])(
            keys.astype(jnp.float32), jnp.maximum(index, 0)
        )
        scores = jnp.einsum(
            "bthi,btki->bthk", q, rows, preferred_element_type=jnp.float32
        )
        scores = jnp.einsum("bthk,bth->btk", jax.nn.relu(scores), w)
        return jax.nn.log_softmax(jnp.where(index >= 0, scores, -jnp.inf))

    def both(f: Callable[..., Array]) -> Callable[..., tuple]:
        def run(q: Array, w: Array, keys: Array) -> tuple:
            out, pull = jax.vjp(f, q, w, keys)
            return out, pull(d)

        return jax.jit(run)

    def kernel(q: Array, w: Array, keys: Array) -> Array:
        return indexer_cudnn.log_softmax(q, w, keys, index)

    want, want_grads = both(xla)(q, w, keys)
    got, grads = both(kernel)(q, w, keys)
    seen = want > -69  # e^-69 = 1e-30
    shown = [error(jnp.where(seen, got, 0), jnp.where(seen, want, 0))]
    shown += [error(a, b) for a, b in zip(grads, want_grads, strict=True)]
    ok = max(shown) < TOLERANCE and indexer_cudnn.takes(h, 128, k, DTYPE)
    print(
        f"log-softmax {name:16} log q, dq, dw, dkeys "
        + " ".join(f"{e:.1e}" for e in shown)
        + f"  {'ok' if ok else 'DIFFERS'}",
        flush=True,
    )
    return ok


def indexer_checks() -> list[bool]:
    """The indexer's kernels against XLA: the top-k at 4k and 32k, on packed
    documents, ties at 0 and at the k-th, rows that see fewer than k keys or none, k
    that are no multiple of anything; the loss's log-softmax at 2048 picks a query
    and the edges of what the kernels take."""
    some = ranked(4096, 512)
    no_key = some.scores.at[:, :7].set(-jnp.inf)[..., :3000]
    first = jnp.zeros_like(some.span.end)
    every, some_keys = (
        sparse.Span(first, first + 3000),
        sparse.Span(first, first + 1000),
    )
    agree = [
        top_k_check("32k", ranked(32768, 1024)),
        top_k_check("32k packed documents", ranked(32768, 1024, docs=True)),
        top_k_check("32k ties at k-th", ranked(32768, 1024, ties=0.5)),
        top_k_check("32k all 0", ranked(32768, 1024, docs=True, ties=1e30)),
        top_k_check("4k x 8 rows", ranked(4096, 4096, b=8)),
        top_k_check("4k k 100", ranked(4096, 512), k=100),
        top_k_check("4k k 2047", ranked(4096, 512), k=2047),
        top_k_check("3000 keys, 7 see none", Ranked(no_key, every)),
        top_k_check("1000 keys < k", Ranked(no_key[..., :1000], some_keys)),
    ]
    agree += [
        log_softmax_check("4k x 2, 64 heads", 2, 4096, 4096, 64, 2048),
        log_softmax_check("32k, 1024 queries", 1, 1024, 32768, 64, 2048),
        log_softmax_check("32 heads, 128 picks", 2, 1024, 1024, 32, 128),
        log_softmax_check("64 heads, 512 picks", 1, 2048, 2048, 64, 512),
    ]
    return agree


# --- CSA by cuDNN's DSA kernels (sparse_cudnn) ---------------------------------------


class Latents(NamedTuple):
    """A CSA layer's parts: the window of the tokens' latents, and the compressed
    latents a query keeps."""

    heads: int = 64
    window: int = 128
    top_k: int = 512
    ratio: int = 2  # a compressed latent every `ratio` slots of a document
    sinks: bool = True
    window_only: bool = False
    dtype: Any = jnp.bfloat16


LATENTS = {  # DeepSeek-V4.1-Flash's, then the edges of what the kernels take
    "v4.1_flash": Latents(),
    "16_heads": Latents(heads=16),  # padded to 64
    "96_heads": Latents(heads=96),  # two calls, the second padded
    "top_300_ratio_1": Latents(top_k=300, ratio=1),  # lists of 428, in tiles of 64
    "no_sinks": Latents(sinks=False),
    "window_only": Latents(window_only=True),
    "float16": Latents(dtype=jnp.float16),
}


def latent_kept(case: Latents, rows: Segments) -> sparse.Kept:
    """Each query's top_k of the compressed latents of its document, by random
    scores, each once, as an indexer keeps them."""
    b, t = rows.ids.shape
    exists = (rows.within + 1) % case.ratio == 0
    seen = sparse.mask(sparse.token_span(rows, None), t, exists)
    scores = jax.random.uniform(jax.random.key(3), (b, t, t))
    return sparse.Kept(indexer._top(jnp.where(seen, scores, -jnp.inf), case.top_k))


def latent_parts(
    case: Latents, rows: Segments, keys: Keys, kept: sparse.Kept
) -> list[sparse.Part]:
    """The case's parts, of the given latents and kept ones."""
    kv, latents = keys[0][0], keys[1][0]
    window = sparse.Part(kv, kv, sparse.token_span(rows, case.window))
    if case.window_only:
        return [window]
    exists = (rows.within + 1) % case.ratio == 0
    span = sparse.token_span(rows, None)
    return [window, sparse.Part(latents, latents, span, exists, kept)]


def latent_inputs(case: Latents, b: int, t: int) -> tuple:
    """Queries, the window's and the compressed latents, sinks, and a cotangent."""
    key = iter(jax.random.split(jax.random.key(0), 8))

    def normal(*shape: int) -> Float[Array, "..."]:
        return jax.random.normal(next(key), shape, case.dtype)

    d = sparse_cudnn.DIMS
    kv, latents = normal(b, t, 1, d), normal(b, t, 1, d)
    keys = [(kv, kv), (latents, latents)]
    sinks = jnp.linspace(-1, 1, case.heads) if case.sinks else None
    return normal(b, t, case.heads, d), keys, sinks, normal(b, t, case.heads, d)


def latent_step(
    case: Latents, rows: Segments, kept: sparse.Kept, by_kernels: bool
) -> Step:
    """A jitted forward and backward of the case's attention, by the kernels or by
    `sparse.attend`: the output and logsumexp, and the gradients of q, the latents
    and the sinks."""
    scale = sparse_cudnn.DIMS**-0.5

    def loss(
        q: Float[Array, "b t h dk"],
        keys: Keys,
        sinks: Float[Array, " h"] | None,
        dout: Float[Array, "b t h dv"],
    ) -> tuple[Float[Array, ""], tuple]:
        given = latent_parts(case, rows, keys, kept)
        windows = (case.window, None)[: len(given)]
        if by_kernels:
            assert sparse_cudnn.takes(q, given, windows, None)
            out, lse = sparse_cudnn.attend(q, given, sinks, scale, windows)
        else:
            out, lse = sparse.attend(q, given, sinks, scale, None)
        return (out.astype(jnp.float32) * dout).sum(), (out, lse)

    return jax.jit(jax.grad(loss, argnums=(0, 1, 2), has_aux=True))


def ran_kernels(f: Step, *args: object) -> bool:
    """Whether the compiled step calls CuTe DSL's kernels."""
    return "cutlasscall" in (f.lower(*args).compile().as_text() or "").lower()


def latent_check(name: str, case: Latents, context: int) -> bool:
    """The kernels, split over `context` GPUs, against XLA on one; then the
    indexer's target at the picks against XLA's of every slot. Whether they agree."""
    b, t = 2, 4096
    rows, x = documents(b, t), latent_inputs(case, b, t)
    kept = jax.jit(latent_kept, static_argnums=0)(case, rows)
    with kernels.off():
        grads, aux = latent_step(case, rows, kept, by_kernels=False)(*x)
    with parallel.use(tcfg.MeshConfig(data=1, context=context)) as mesh:
        put = NamedSharding(mesh, P(None, "context"))
        split = jax.tree.map(lambda a: jax.device_put(a, put) if a.ndim > 1 else a, x)
        f = latent_step(case, rows, kept, by_kernels=True)
        got_grads, got_aux = f(*split)
        ran = ran_kernels(f, *split)
    shown = errors((got_aux, got_grads), (aux, grads))
    targets = case.dtype == jnp.bfloat16 and not case.window_only
    if targets:
        part = latent_parts(case, rows, x[1], kept)[1]
        shown.append(target_error(x[0], part, aux[1]))
    ok = max(shown) < TOLERANCE and ran
    print(
        f"context={context} {name:16} {'cudnn' if ran else 'xla':6} out, lse, grads"
        + (", target " if targets else " ")
        + " ".join(f"{e:.1e}" for e in shown)
        + f"  {'ok' if ok else 'DIFFERS'}",
        flush=True,
    )
    return ok


def target_error(
    q: Float[Array, "b t h d"], part: sparse.Part, lse: Float[Array, "b t h"]
) -> float:
    """The kernel's target at the picks against XLA's, taken at the picks and
    normalized."""
    scale = sparse_cudnn.DIMS**-0.5
    got = jax.jit(sparse_cudnn.probabilities, static_argnums=3)(q, part, lse, scale)
    with kernels.off():
        every = jax.jit(sparse.probabilities, static_argnums=(3, 4))
        every = every(q, part, lse, scale, None)
    assert part.kept is not None
    index = part.kept.index
    picked = jnp.take_along_axis(every, jnp.maximum(index, 0), -1)
    picked = jnp.where(index >= 0, picked, 0)
    want = picked / jnp.maximum(picked.sum(-1, keepdims=True), 1e-30)
    return error(got, want)


def latent_timings() -> None:
    """Forward and backward ms of DeepSeek-V4.1-Flash's CSA attention (64 heads, a
    window of 128 and the top 512 compressed latents) by the kernels and by XLA, and
    of the indexer's target at the picks by the kernel and by XLA."""
    case = LATENTS["v4.1_flash"]
    scale = sparse_cudnn.DIMS**-0.5
    for t in (4096, 16384, 32768):
        rows, x = documents(1, t), latent_inputs(case, 1, t)
        kept = jax.jit(latent_kept, static_argnums=0)(case, rows)
        step = latent_step(case, rows, kept, by_kernels=True)
        shown = f"cudnn {milliseconds(step, *x):8.2f}"
        part = latent_parts(case, rows, x[1], kept)[1]
        lse = step(*x)[1][1]
        target = jax.jit(sparse_cudnn.probabilities, static_argnums=3)
        shown += f"  target {milliseconds(target, x[0], part, lse, scale):8.2f}"
        if t <= 16384:
            with kernels.off():
                xla = milliseconds(latent_step(case, rows, kept, False), *x)
                every = jax.jit(sparse.probabilities, static_argnums=(3, 4))
                xla_target = milliseconds(every, x[0], part, lse, scale, None)
            shown += f"  xla {xla:8.2f}  target {xla_target:8.2f}"
        print(f"time csa_cudnn t={t:6} window 128 + top 512, ms: {shown}", flush=True)


def main() -> None:
    gpus = jax.device_count()
    agree = [check(n, c, 1) for n, c in CASES.items()]
    if gpus >= 2:
        agree += [check(n, c, 2) for n, c in CASES.items()]
    agree += indexer_checks()
    if sparse_cudnn.runs():
        agree += [latent_check(n, c, 1) for n, c in LATENTS.items()]
        if gpus >= 2:
            agree += [latent_check(n, c, 2) for n, c in LATENTS.items()]
    for name in ("gqa", "mla", "csa"):
        timings(name, CASES[name])
    if sparse_cudnn.runs():
        latent_timings()
    sys.exit(0 if all(agree) else 1)


if __name__ == "__main__":
    main()
