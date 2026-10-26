"""Sparse attention on GPUs: tokamax's flash kernel with the kept keys as a mask
(``bux.architecture.attention.sparse`` under `methods.attention = auto`) against XLA's
chunks of queries (inside ``bux.kernels.off()``), on one GPU and with the positions
split over 2: top-k tokens of GQA and MLA, QSA's blocks, CSA's window and latents (512
dims, which XLA runs), on packed documents with a window, sinks and a soft cap; the
output, the logsumexp and the gradients. Then the indexer's kernels on a B200 or B300:
cuDNN's top-k against XLA's, bitwise, and its loss's log-softmax and gradients. Then
the time of a forward and backward of each, and of dense flash attention, from 4k to
32k tokens.

    uv run --extra cuda python scripts/gpu_smoke/sparse.py

It needs tokamax and cudnn-frontend (the cuda extra), and exits 1 if a kernel differs
from XLA.
"""

import sys
import time
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

import bux.architecture  # noqa: F401  (registers the attention methods)
from bux import kernels, parallel
from bux.architecture import Segments
from bux.architecture.attention import indexer, indexer_cudnn, scoring, sparse
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


CASES = {
    "gqa": Case(16, 8, 128, 128, 512),
    "gqa_window_sinks": Case(16, 8, 128, 128, 512, window=1000, sinks=True, softcap=30),
    "qsa_blocks": Case(16, 2, 128, 128, 512, block=4),
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


def kept(b: int, t: int, top_k: int, block: int, seed: int) -> sparse.Kept:
    """Each query's top_k random slots at or before it (with repeats), or random
    blocks and its own; the span cuts them down to its document's."""
    n = top_k // block
    u = jax.random.uniform(jax.random.key(seed), (b, t, n))
    upto = jnp.arange(t)[None, :, None] // block + 1
    index = jnp.floor(u * upto).astype(jnp.int32)
    if block == 1:
        return sparse.Kept(index)
    own = jnp.broadcast_to(
        jnp.arange(t, dtype=jnp.int32)[None, :, None] // block, (b, t, 1)
    )
    blocks = jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32) // block, (b, t))
    return sparse.Kept(jnp.concatenate([index, own], -1), blocks)


def parts(case: Case, rows: Segments, keys: Keys) -> list[sparse.Part]:
    """The case's parts, of the given keys and values."""
    b, t = rows.ids.shape
    span = sparse.token_span(rows, None if case.csa else case.window)
    chosen = kept(b, t, case.top_k, case.block, seed=3)
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
    wanted = "mosaic" if case.dk <= sparse.HEAD_DIMS else "xla"  # CSA's 512: XLA
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


def main() -> None:
    gpus = jax.device_count()
    agree = [check(n, c, 1) for n, c in CASES.items()]
    if gpus >= 2:
        agree += [check(n, c, 2) for n, c in CASES.items()]
    agree += indexer_checks()
    for name in ("gqa", "mla", "csa"):
        timings(name, CASES[name])
    sys.exit(0 if all(agree) else 1)


if __name__ == "__main__":
    main()
