"""Linear attention's and Mamba-2's recurrences by chunks (``chunkwise``) against their
scans over positions, on one GPU at a hybrid's shapes, 2 rows of 4096 in packed
documents: the outputs and every input's gradient, in float32. XLA's GPU fusions
can differ from its CPU's, which the tests run: on a B300, a grouped einsum's took
Mamba-2's gradients for B and C wrong. Then the delta rule by cuDNN's kernels
(``fused_delta``), in bfloat16, against its chunks in float32: GDN's at any gate,
KDA's at a gate bounded at -5, and unbounded with heads too steep for it.

    uv run --extra cuda python scripts/gpu_smoke/recurrence.py

It exits 1 if a form differs from its scan, or a kernel from the chunks.
"""

import sys
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from bux.architecture import Segments, linear_attention, ssm
from bux.parallel.recurrence import Chunked, Positions, Shared

B, T = 2, 4096
TOLERANCE = 1e-4  # of each gradient's norm, in float32
KERNEL_TOLERANCE = 2e-2  # of a kernel's, in bfloat16
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


# The delta rule's gates: channels, the least log decay, the floor it is bounded at,
# and the heads whose gates are at -12, too steep for KDA's kernel.
KERNELS: dict[str, tuple[int, float, float | None, tuple[int, ...]]] = {
    "gdn's kernel, gates to -20": (1, -20.0, None, ()),
    "kda's kernel, gates to -5": (128, -5.0, -5.0, ()),
    "kda's kernel, 2 heads too steep": (128, -3.0, None, (3, 11)),
}


def delta_inputs(case: str) -> tuple[Float[Array, "..."], ...]:
    """q, k and v in bfloat16, of 16 heads of 128, the log decay and beta."""
    decay, low, _, steep = KERNELS[case]
    rng = np.random.default_rng(list(KERNELS).index(case))
    q, k, v = rng.normal(size=(3, B, T, 16, 128))
    q = q / np.linalg.norm(q, axis=-1, keepdims=True) / np.sqrt(128)
    k = k / np.linalg.norm(k, axis=-1, keepdims=True)
    g = low / (1 + np.exp(-rng.normal(size=(B, T, 16, decay))))
    g[:, :, list(steep)] = -12.0
    beta = 1 / (1 + np.exp(-rng.normal(size=(B, T, 16))))
    qkv = (jnp.asarray(x, jnp.bfloat16) for x in (q, k, v))
    return (*qkv, jnp.asarray(g, jnp.float32), jnp.asarray(beta, jnp.float32))


def kernel_errors(case: str) -> list[float]:
    """How far the delta rule by its kernels is from its chunks: the outputs, then
    each input's gradient, as a share of its norm."""
    floor = KERNELS[case][2]
    positions = jnp.asarray(np.arange(T) % 1000, jnp.int32)  # a document every 1000
    docs = Segments.of(jnp.broadcast_to(positions, (B, T)))

    def kernels(*xs: Float[Array, "..."]) -> Float[Array, "b t h dv"]:
        return linear_attention.delta_rule(*xs, docs, floor).astype(jnp.float32)

    def chunks(*xs: Float[Array, "..."]) -> Float[Array, "b t h dv"]:
        f32 = (x.astype(jnp.float32) for x in xs)
        return linear_attention._chunked(*f32, docs)

    xs = delta_inputs(case)
    got, want = (weighted(f, xs) for f in (kernels, chunks))
    return [float(np.linalg.norm(a - b) / np.linalg.norm(b)) for a, b in zip(got, want)]


def weighted(f: Callable[..., Float[Array, "..."]], xs: tuple) -> list[np.ndarray]:
    """f's outputs, and the gradients of a weighted sum of them for every input."""

    def loss(*xs: Float[Array, "..."]) -> tuple[Float[Array, ""], Float[Array, "..."]]:
        out = f(*xs)
        return (out * jnp.cos(jnp.arange(out.size)).reshape(out.shape)).sum(), out

    grads, out = jax.jit(jax.grad(loss, tuple(range(len(xs))), has_aux=True))(*xs)
    return [np.asarray(out), *(np.asarray(g, np.float32) for g in grads)]


def report(case: str, found: list[float], tolerance: float) -> bool:
    """Print how far a form is from its reference; return whether within
    `tolerance`."""
    ok = max(found) < tolerance
    shown = " ".join(f"{e:.1e}" for e in found)
    print(f"{case:32} out, then grads {shown}  {'ok' if ok else 'DIFFERS'}")
    return ok


def main() -> None:
    jax.config.update("jax_default_matmul_precision", "highest")
    print(f"{jax.devices()[0].device_kind}, {B} x {T} tokens, float32")
    forms = [report(case, errors(case), TOLERANCE) for case in CASES]
    kernels = [report(c, kernel_errors(c), KERNEL_TOLERANCE) for c in KERNELS]
    sys.exit(0 if all(forms) and all(kernels) else 1)


if __name__ == "__main__":
    main()
