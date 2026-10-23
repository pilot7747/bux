"""Sparse attention on GPUs: tokamax's flash kernel with the kept keys as a mask
(``bux.architecture.sparse`` under `methods.attention = auto`) against XLA's chunks of
queries (inside ``bux.kernels.off()``), on one GPU and with the positions split over
2: top-k tokens of GQA and MLA, QSA's blocks, CSA's window and latents (512 dims,
which XLA runs), on packed documents with a window, sinks and a soft cap; the output,
the logsumexp and the gradients. Then the time of a forward and backward of each, and
of dense flash attention, from 4k to 32k tokens.

    uv run --extra cuda python scripts/gpu_smoke/sparse.py

It needs tokamax (the cuda extra), and exits 1 if the kernel differs from XLA.
"""

import sys
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

import bux.architecture  # noqa: F401  (registers the attention methods)
from bux import kernels, parallel
from bux.architecture import Segments, sparse
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


def main() -> None:
    gpus = jax.device_count()
    agree = [check(n, c, 1) for n, c in CASES.items()]
    if gpus >= 2:
        agree += [check(n, c, 2) for n, c in CASES.items()]
    for name in ("gqa", "mla", "csa"):
        timings(name, CASES[name])
    sys.exit(0 if all(agree) else 1)


if __name__ == "__main__":
    main()
