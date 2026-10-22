"""WP8 on GPUs: every method of `methods.attention` against `xla` on one GPU, with the
positions split over `context` on packed documents, a window, a soft cap, sinks and
GQA, forward and gradients; then the time of a forward and backward of each.

    uv run --extra cuda python scripts/gpu_smoke/attention.py

It needs 2 GPUs or more (a power of 2), and tokamax (the cuda extra), whose kernels
CPU cannot run: it says which of them ran. It exits 1 if a method differs from xla.
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

import bux.architecture  # noqa: F401  (registers xla)
from bux import parallel
from bux.parallel import methods
from bux.parallel.context import AttentionArgs
from bux.train import config as tcfg

B, T, H, G, D = 2, 4096, 8, 2, 128
DTYPE = jnp.bfloat16
TOLERANCE = 2e-2  # of the largest value, in bfloat16
METHODS = ("ring", "ulysses", "tokamax")
Step = jax.stages.Wrapped  # jitted (q, k, v, dout) -> ((loss, out), (dq, dk, dv))


def documents(seed: int) -> Int[Array, "b t"]:
    """Each row's documents' ids, from 1: a dozen documents of random lengths."""
    rng = np.random.default_rng(seed)
    starts = np.zeros((B, T), np.int32)
    for row in range(B):
        starts[row, np.unique([0, *rng.integers(1, T, 12)])] = 1
    return jnp.asarray(np.cumsum(starts, 1), jnp.int32)


IDS = documents(0)
CASES = {
    "documents": AttentionArgs(IDS, None, D**-0.5, None, None),
    "window": AttentionArgs(IDS, 1000, D**-0.5, None, None),
    "softcap": AttentionArgs(None, None, D**-0.5, 30.0, None),
    "sinks": AttentionArgs(IDS, None, D**-0.5, None, jnp.linspace(-1, 1, H)),
}


def inputs() -> tuple[Float[Array, "b t _ k"], ...]:
    """Queries, keys, values, and a cotangent for the output."""
    key = jax.random.split(jax.random.key(0), 4)
    shapes = [(B, T, H, D), (B, T, G, D), (B, T, G, D), (B, T, H, D)]
    return tuple(jax.random.normal(k, s, DTYPE) for k, s in zip(key, shapes))


def step(method: str, seen: AttentionArgs) -> Step:
    """A jitted forward and backward of `method`: its output, and the gradients."""
    f = methods.attention.find(method)

    def loss(
        q: Float[Array, "b t h k"],
        k: Float[Array, "b t g k"],
        v: Float[Array, "b t g k"],
        dout: Float[Array, "b t h k"],
    ) -> tuple[Float[Array, ""], Float[Array, "b t h k"]]:
        out = f(q, k, v, seen)
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


def kernel(f: Step, *args: Float[Array, "..."]) -> str:
    """Which of tokamax's kernels the compiled step calls."""
    text = (f.lower(*args).compile().as_text() or "").lower()
    return next((k for k in ("mosaic", "triton") if k in text), "xla")


def compare(context: int, x: tuple[Float[Array, "b t _ k"], ...]) -> bool:
    """Every method on `context` GPUs against xla on one; whether all agree."""
    agree = True
    for case, seen in CASES.items():
        (_, out), grads = step("xla", seen)(*x)
        for method in METHODS:
            if method == "tokamax" and seen.sinks is not None:
                continue  # tokamax has no sinks
            with parallel.use(tcfg.MeshConfig(data=1, context=context)) as mesh:
                put = NamedSharding(mesh, P(None, "context"))
                split = tuple(jax.device_put(a, put) for a in x)
                f = step(method, seen)
                (_, split_out), split_grads = f(*split)
                ms = milliseconds(f, *split)
                ran = kernel(f, *split) if method == "tokamax" else ""
            errors = [error(split_out, out)]
            errors += [error(a, b) for a, b in zip(split_grads, grads)]
            ok = max(errors) < TOLERANCE
            agree &= ok
            shown = " ".join(f"{e:.1e}" for e in errors)
            print(
                f"context={context} {method:8} {case:10} out, dq, dk, dv {shown}"
                f"  {ms:7.1f} ms {ran:7} {'ok' if ok else 'DIFFERS'}"
            )
    return agree


def main() -> None:
    gpus = jax.device_count()
    if gpus < 2:
        sys.exit(f"needs 2 GPUs or more, and has {gpus}")
    x = inputs()
    for case, seen in CASES.items():
        ms = milliseconds(step("xla", seen), *x)
        print(f"{jax.devices()[0].device_kind}: xla on one GPU, {case}, {ms:.1f} ms")
    agree = [compare(c, x) for c in sorted({2, gpus})]  # every mesh, failing or not
    sys.exit(0 if all(agree) else 1)


if __name__ == "__main__":
    main()
