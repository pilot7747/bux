"""WP8 on GPUs: every method of `methods.attention` against `xla` on one GPU, on packed
documents (a dozen a row, more than cuDNN's packed layout takes, as many as FA4's
cu_seqlens has room for and one more, and one), a window, a soft cap, sinks and GQA,
forward and gradients: on whole rows on one GPU, then with the positions split over
`context`; MLA's head dims (192 and 128), gpt-oss's (64 heads of 8 kv heads of 64)
and rows of 1000 positions on whole rows; and the time of a forward and backward of
each.

    uv run --extra cuda python scripts/gpu_smoke/attention.py

It needs 2 GPUs or more (a power of 2), and tokamax (the cuda extra), whose kernels
CPU cannot run: it says which kernels each step calls. It exits 1 if a method differs
from xla.
"""

import sys
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

import bux.architecture  # noqa: F401  (registers xla)
from bux import parallel
from bux.parallel import fa4, methods
from bux.parallel.context import AttentionArgs
from bux.train import config as tcfg

B, T = 2, 4096
DTYPE = jnp.bfloat16
TOLERANCE = 2e-2  # of the largest value, in bfloat16
WHOLE = ("cudnn", "tokamax", "fa4", "auto")  # on whole rows
SPLIT = ("ring", "ulysses", "tokamax", "auto")  # with the positions over context
KERNELS = {
    "cudnn": "cudnn$fmha",
    "mosaic": "mosaic",
    "triton": "triton",
    "fa4": "nvjaxcutlasscall",
}
Step = jax.stages.Wrapped  # jitted (q, k, v, dout) -> ((loss, out), (dq, dk, dv))


class Heads(NamedTuple):
    """A layer's heads: query heads, kv heads, and the keys' and values' dims."""

    h: int
    g: int
    dk: int
    dv: int


GQA, MLA, OSS = Heads(8, 2, 128, 128), Heads(8, 8, 192, 128), Heads(64, 8, 64, 64)


def documents(seed: int, n: int, t: int = T) -> Int[Array, "b t"]:
    """Each row's documents' ids, from 1: about n documents of random lengths."""
    rng = np.random.default_rng(seed)
    starts = np.zeros((B, t), np.int32)
    for row in range(B):
        starts[row, np.unique([0, *rng.integers(1, t, n)])] = 1
    return jnp.asarray(np.cumsum(starts, 1), jnp.int32)


def exactly(n: int) -> Int[Array, "b t"]:
    """Rows of n documents each, the first ones a token long."""
    starts = np.zeros((B, T), np.int32)
    starts[:, : n // 2] = 1
    starts[:, np.linspace(n // 2, T - 1, n - n // 2).astype(int)] = 1
    return jnp.asarray(np.cumsum(starts, 1), jnp.int32)


ROOM = fa4.room(B, T) // B  # documents a row FA4's cu_seqlens has room for
IDS, MANY, ONE = documents(0, 12), documents(1, 60), documents(2, 0)
SINKS = jnp.linspace(-1, 1, 64)
CASES = {  # each scaled for its layer's keys by `seen`
    "documents": AttentionArgs(IDS, None, 1.0, None, None),
    "one_doc": AttentionArgs(ONE, None, 1.0, None, SINKS),
    "many_docs": AttentionArgs(MANY, None, 1.0, None, None),
    "fa4_room": AttentionArgs(exactly(ROOM), None, 1.0, None, None),
    "past_room": AttentionArgs(exactly(ROOM + 1), None, 1.0, None, None),
    "window": AttentionArgs(IDS, 1000, 1.0, None, None),
    "causal": AttentionArgs(None, None, 1.0, None, None),
    "softcap": AttentionArgs(None, None, 1.0, 30.0, None),
    "sinks": AttentionArgs(IDS, None, 1.0, None, SINKS),
}
ODD = 1000  # rows of no whole number of the kernels' blocks of 128
ODD_CASES = {  # as many documents as FA4 has room for in rows of ODD, or fewer
    "odd_documents": AttentionArgs(documents(3, 2, ODD), None, 1.0, None, None),
    "odd_causal": AttentionArgs(None, None, 1.0, None, None),
}
# On whole rows: MLA's and gpt-oss's layers, their documents, windows and sinks; and
# rows of ODD positions.
MORE = (
    (MLA, T, ("documents", "one_doc", "causal", "softcap")),
    (OSS, T, ("documents", "one_doc", "window", "sinks", "past_room")),
    (GQA, ODD, tuple(ODD_CASES)),
)


def seen(case: str, heads: Heads) -> AttentionArgs:
    """The case's mask, scaled for the keys' dims and with a sink a query head."""
    args = (CASES | ODD_CASES)[case]
    sinks = None if args.sinks is None else args.sinks[: heads.h]
    return args._replace(scale=heads.dk**-0.5, sinks=sinks)


def inputs(heads: Heads, t: int = T) -> tuple[Float[Array, "b t _ k"], ...]:
    """Queries, keys, values, and a cotangent for the output."""
    key = jax.random.split(jax.random.key(0), 4)
    h, g, dk, dv = heads
    shapes = [(B, t, h, dk), (B, t, g, dk), (B, t, g, dv), (B, t, h, dv)]
    return tuple(jax.random.normal(k, s, DTYPE) for k, s in zip(key, shapes))


def step(method: str, args: AttentionArgs) -> Step:
    """A jitted forward and backward of `method`: its output, and the gradients."""
    f = methods.attention.find(method)

    def loss(
        q: Float[Array, "b t h k"],
        k: Float[Array, "b t g k"],
        v: Float[Array, "b t g v"],
        dout: Float[Array, "b t h v"],
    ) -> tuple[Float[Array, ""], Float[Array, "b t h v"]]:
        out = f(q, k, v, args)
        return (out.astype(jnp.float32) * dout).sum(), out

    return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))


def error(a: Float[Array, "*s"], b: Float[Array, "*s"]) -> float:
    """The largest difference, as a share of the largest value of `b`."""
    x, y = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.abs(x - y).max() / np.abs(y).max())


def milliseconds(f: Step, *args: Float[Array, "..."]) -> float:
    """The mean time of a call of `f`, once compiled."""
    jax.block_until_ready(f(*args))
    start = time.perf_counter()
    for _ in range(10):
        jax.block_until_ready(f(*args))
    return (time.perf_counter() - start) * 100


def kernels(f: Step, *args: Float[Array, "..."]) -> str:
    """Which GPU kernels the compiled step calls, or xla."""
    text = (f.lower(*args).compile().as_text() or "").lower()
    return "+".join(k for k, name in KERNELS.items() if name in text) or "xla"


def takes(method: str, args: AttentionArgs, heads: Heads) -> bool:
    """Whether `method` takes the case: cudnn has no soft cap, no sinks and no MLA."""
    if method != "cudnn":
        return True
    return args.softcap is None and args.sinks is None and heads.dk == heads.dv


def compare(
    context: int,
    heads: Heads,
    cases: tuple[str, ...],
    x: tuple[Float[Array, "..."], ...],
) -> bool:
    """Every method on `context` GPUs against xla on one, or on whole rows on one GPU
    for context 1; whether all agree."""
    agree = True
    for case in cases:
        args = seen(case, heads)
        (_, out), grads = step("xla", args)(*x)
        for method in WHOLE if context == 1 else SPLIT:
            if not takes(method, args, heads):
                continue
            with parallel.use(tcfg.MeshConfig(data=1, context=context)) as mesh:
                put = NamedSharding(mesh, P(None, "context"))
                split = tuple(jax.device_put(a, put) for a in x)
                f = step(method, args)
                (_, split_out), split_grads = f(*split)
                ms = milliseconds(f, *split)
                ran = kernels(f, *split)
            errors = [error(split_out, out)]
            errors += [error(a, b) for a, b in zip(split_grads, grads)]
            ok = max(errors) < TOLERANCE
            agree &= ok
            shown = " ".join(f"{e:.1e}" for e in errors)
            print(
                f"context={context} {heads} {method:8} {case:10} out, dq, dk, dv "
                f"{shown}  {ms:7.1f} ms {ran:18} {'ok' if ok else 'DIFFERS'}",
                flush=True,
            )
    return agree


def main() -> None:
    gpus = jax.device_count()
    if gpus < 2:
        sys.exit(f"needs 2 GPUs or more, and has {gpus}")
    x = inputs(GQA)
    for case in CASES:
        ms = milliseconds(step("xla", seen(case, GQA)), *x)
        print(f"{jax.devices()[0].device_kind}: xla on one GPU, {case}, {ms:.1f} ms")
    # every mesh, failing or not
    agree = [compare(c, GQA, tuple(CASES), x) for c in sorted({1, 2, gpus})]
    agree += [compare(1, heads, cases, inputs(heads, t)) for heads, t, cases in MORE]
    sys.exit(0 if all(agree) else 1)


if __name__ == "__main__":
    main()
