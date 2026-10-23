"""Linear attention's and Mamba-2's recurrences by chunks (``chunkwise``) against their
scans over positions, on one GPU at a hybrid's shapes, 2 rows of 4096 in packed
documents: the outputs and every input's gradient, in float32. XLA's GPU fusions
can differ from its CPU's, which the tests run: on a B300, a grouped einsum's took
Mamba-2's gradients for B and C wrong.

    uv run --extra cuda python scripts/gpu_smoke/recurrence.py

It exits 1 if a form differs from its scan.
"""

import sys
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from bux.architecture import linear_attention, ssm
from bux.parallel.recurrence import Chunked, Positions, Shared

B, T = 2, 4096
TOLERANCE = 1e-4  # of each gradient's norm, in float32
Inputs = Callable[[np.random.Generator], tuple[Positions, Shared]]


def documents(rng: np.random.Generator) -> Float[Array, "b t"]:
    """1 where a token continues a document, 0 where one starts: a dozen a row."""
    continues = np.ones((B, T), np.float32)
    for row in range(B):
        continues[row, rng.integers(0, T, 12)] = 0
    return jnp.asarray(continues)


def mamba2(heads: int, groups: int) -> Inputs:
    """x, dt, B, C and continues, and A, of heads of 64 and states of 128."""

    def make(rng: np.random.Generator) -> tuple[Positions, Shared]:
        x = rng.normal(size=(B, T, heads, 64))
        dt = np.log1p(np.exp(rng.normal(size=(B, T, heads)) - 3))
        BC = rng.normal(size=(2, B, T, groups, 128))
        A = -rng.uniform(1, 16, size=(heads, 1))
        xs = (x, dt, BC[0], BC[1])
        return (*(jnp.asarray(a, jnp.float32) for a in xs), documents(rng)), (
            jnp.asarray(A, jnp.float32),
        )

    return make


def linear(decay: int, delta: bool) -> Inputs:
    """q, k, v, log decay and continues of 16 heads of 128, and a for the delta rule."""

    def make(rng: np.random.Generator) -> tuple[Positions, Shared]:
        q, k, v = rng.normal(size=(3, B, T, 16, 128)) / np.sqrt(128)
        g = -rng.uniform(0, 0.1, size=(B, T, 16, decay))
        if delta:
            k = k / np.linalg.norm(k, axis=-1, keepdims=True)
            q, k, v, g = (jnp.asarray(a, jnp.float32) for a in (q, k, v, g))
            a = -jnp.exp(g) * k
            return (q, 0.5 * k, v, g, documents(rng), a), ()
        xs = (jnp.asarray(a, jnp.float32) for a in (q, k, v, g))
        return (*xs, documents(rng)), ()

    return make


CASES: dict[str, tuple[Chunked, Chunked, Inputs]] = {
    "mamba2, 64 heads in 8 groups": (ssm.chunk, ssm.stepwise, mamba2(64, 8)),
    "mamba2, 64 heads in 1 group": (ssm.chunk, ssm.stepwise, mamba2(64, 1)),
    "mamba2, 128 heads in 8 groups": (ssm.chunk, ssm.stepwise, mamba2(128, 8)),
    "lightning, a decay a head": (
        linear_attention.chunk,
        linear_attention.stepwise,
        linear(1, False),
    ),
    "gla, a decay a channel": (
        linear_attention.chunk,
        linear_attention.stepwise,
        linear(128, False),
    ),
    "the gated delta rule": (
        linear_attention.chunk,
        linear_attention.stepwise,
        linear(1, True),
    ),
}


def gradients(f: Chunked, xs: Positions, shared: Shared) -> list[np.ndarray]:
    """f's outputs, and the gradients of a weighted sum of them for every input but
    continues (the fifth)."""
    learned = [i for i in range(len(xs)) if i != 4]

    def loss(
        *args: Float[Array, "..."],
    ) -> tuple[Float[Array, ""], Float[Array, "..."]]:
        given = list(xs)
        for i, a in zip(learned, args[: len(learned)]):
            given[i] = a
        out = f(tuple(given), tuple(args[len(learned) :]), False).out
        return (out * jnp.cos(jnp.arange(out.size)).reshape(out.shape)).sum(), out

    args = [xs[i] for i in learned] + list(shared)
    grad = jax.grad(loss, tuple(range(len(args))), has_aux=True)
    grads, out = jax.jit(grad)(*args)
    return [np.asarray(out), *(np.asarray(g) for g in grads)]


def errors(case: str) -> list[float]:
    """How far the case's chunks are from its scan: the outputs, then each gradient,
    as a share of its norm."""
    chunked, scan, make = CASES[case]
    xs, shared = make(np.random.default_rng(list(CASES).index(case)))
    got, want = (gradients(f, xs, shared) for f in (chunked, scan))
    return [float(np.linalg.norm(a - b) / np.linalg.norm(b)) for a, b in zip(got, want)]


def main() -> None:
    jax.config.update("jax_default_matmul_precision", "highest")
    print(f"{jax.devices()[0].device_kind}, {B} x {T} tokens, float32")
    agree = True
    for case in CASES:
        found = errors(case)
        ok = max(found) < TOLERANCE
        agree &= ok
        shown = " ".join(f"{e:.1e}" for e in found)
        print(f"{case:32} out, then grads {shown}  {'ok' if ok else 'DIFFERS'}")
    sys.exit(0 if agree else 1)


if __name__ == "__main__":
    main()
