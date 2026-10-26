"""Linear attention's and Mamba-2's recurrences by chunks (``chunkwise``) against their
scans over positions, on one GPU at a hybrid's shapes, 2 rows of 4096 in packed
documents: the outputs and every input's gradient, in float32; Mamba-2's also with
its chunks computed again for the backward pass, as `methods(states=recompute)` runs
them, and RWKV-7's with its bounded decay through each chunk's middle. XLA's GPU fusions can differ from its CPU's, which the tests run: on a B300, a
grouped einsum's took Mamba-2's gradients for B and C wrong. Then the delta rule by
its kernels, in bfloat16, against its chunks in float32: cuDNN's (``fused_delta``),
GDN's at steep gates and spikes and KDA's at a gate bounded at -5, and bux's own for
KDA (``pallas_delta``) at steep gates, spikes and Kimi's own gate. Last, decayed
linear attention by cuDNN's GDN-2 kernel against its chunks: GLA's own gate, steep
gates, spikes, heads of 64, rows of more documents than the kernel takes, and
Lightning's decays.

    uv run --extra cuda python scripts/gpu_smoke/recurrence.py

It exits 1 if a form differs from its scan, or a kernel from the chunks.
"""

import sys
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux.architecture import Segments
from bux.architecture.recurrent import fused_delta, linear_attention, rwkv7, ssm
from bux.parallel.recurrence import Chunk, Chunked, Positions, Shared

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


def rwkv(rng: np.random.Generator) -> tuple[Positions, Shared]:
    """r, k, v, log decay and continues of 32 heads of 64, and a and b, as RWKV-7
    gives them: its log decay at least `rwkv7.FLOOR`."""
    r, k, v, kk = rng.normal(size=(4, B, T, 32, 64))
    g = rwkv7.FLOOR / (1 + np.exp(-rng.normal(size=(B, T, 32, 64))))
    kk = kk / np.linalg.norm(kk, axis=-1, keepdims=True)
    rate = 1 / (1 + np.exp(-rng.normal(size=(B, T, 32, 64))))
    xs = (jnp.asarray(x, jnp.float32) for x in (r, 0.3 * k, v, g))
    return (*xs, documents(rng), jnp.asarray(-kk), jnp.asarray(kk * rate)), ()


def recomputed(f: Chunked) -> Chunked:
    """Return `f` whose backward pass computes it again from its inputs, as
    `methods(states=recompute)` runs it."""

    def run(xs: Positions, shared: Shared, transition: bool) -> Chunk:
        def chunk(xs: Positions, shared: Shared) -> Chunk:
            return f(xs, shared, transition)

        return linear_attention.recomputed(chunk, "recompute")(xs, shared)

    return run


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
    "rwkv-7, its decay bounded": (
        partial(linear_attention.chunk, floor=rwkv7.FLOOR),
        linear_attention.stepwise,
        rwkv,
    ),
    "mamba2, 8 groups, recomputed": (
        recomputed(ssm.chunk),
        ssm.stepwise,
        mamba2(64, 8),
    ),
    "mamba2, 1 group, recomputed": (
        recomputed(ssm.chunk),
        ssm.stepwise,
        mamba2(64, 1),
    ),
}


def gradients(f: Chunked, xs: Positions, shared: Shared) -> list[np.ndarray]:
    """f's outputs, and the gradients of a weighted sum of them for every input but
    continues (the fifth)."""
    learned = [i for i in range(len(xs)) if i != 4]

    def run(*args: Float[Array, "..."]) -> Float[Array, "..."]:
        given = list(xs)
        for i, a in zip(learned, args[: len(learned)]):
            given[i] = a
        return f(tuple(given), tuple(args[len(learned) :]), False).out

    return weighted(run, (*(xs[i] for i in learned), *shared))


def errors(case: str) -> list[float]:
    """How far the case's chunks are from its scan: the outputs, then each gradient,
    as a share of its norm."""
    chunked, scan, make = CASES[case]
    xs, shared = make(np.random.default_rng(list(CASES).index(case)))
    got, want = (gradients(f, xs, shared) for f in (chunked, scan))
    return [float(np.linalg.norm(a - b) / np.linalg.norm(b)) for a, b in zip(got, want)]


Gate = Callable[[np.random.Generator, int], np.ndarray]  # (rng, decays) -> log decays


def steep(low: float, spike: float | None = None) -> Gate:
    """Log decays in [low, 0], and `spike` every 16 positions if given."""

    def gate(rng: np.random.Generator, decays: int) -> np.ndarray:
        g = low / (1 + np.exp(-rng.normal(size=(B, T, 16, decays))))
        if spike is not None:
            g[:, 5::16] = spike
        return g

    return gate


def kimi(rng: np.random.Generator, decays: int) -> np.ndarray:
    """The log decays of KDA's own gate, -exp(A_log) softplus(f + dt_bias), as a
    layer starts, of inputs drawn from a normal."""
    config = cfg.KDAConfig(num_heads=16, head_dim=decays)
    layer = linear_attention.KimiDeltaAttention(config, 256, nnx.Rngs(0))
    return np.asarray(layer._log_decay(jnp.asarray(rng.normal(size=(B, T, 256)))))


# The delta rule's gates: decays a head, the log decays, and their floor, if any.
KERNELS: dict[str, tuple[int, Gate, float | None]] = {
    "gdn's kernel, gates to -20": (1, steep(-20.0), None),
    "gdn's kernel, spikes of -1e4": (1, steep(-1.0, -1e4), None),
    "kda's kernel, gates to -5": (128, steep(-5.0), -5.0),
    "kda's own kernels, gates to -20": (128, steep(-20.0), None),
    "kda's own kernels, spikes of -1e4": (128, steep(-1.0, -1e4), None),
    "kda's own kernels, kimi's gate": (128, kimi, None),
}


def delta_inputs(case: str) -> tuple[Float[Array, "..."], ...]:
    """q, k and v in bfloat16, of 16 heads of 128, the log decay and beta."""
    decays, gate, _ = KERNELS[case]
    rng = np.random.default_rng(list(KERNELS).index(case))
    q, k, v = rng.normal(size=(3, B, T, 16, 128))
    q = q / np.linalg.norm(q, axis=-1, keepdims=True) / np.sqrt(128)
    k = k / np.linalg.norm(k, axis=-1, keepdims=True)
    g = gate(rng, decays)
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


def gla(rng: np.random.Generator, decays: int) -> np.ndarray:
    """The log decays of GLA's own gate, log sigmoid(f) / 16, of f from a normal."""
    return -np.logaddexp(0, -2 * rng.normal(size=(B, T, 16, decays))) / 16


def lightning(rng: np.random.Generator, decays: int) -> np.ndarray:
    """The log decays of Lightning's first layer of 16, a head each."""
    slopes = np.asarray(linear_attention.slopes(16, 0, 16))[:, None]
    return np.broadcast_to(-slopes, (B, T, 16, decays))


# Decayed linear attention by GDN-2's kernel: head dims, decays a head, the log decays,
# and a document every so many positions.
DECAYED: dict[str, tuple[int, int, Gate, int]] = {
    "gla by gdn-2, its own gate": (128, 128, gla, 1000),
    "gla by gdn-2, heads of 64": (64, 64, gla, 1000),
    "gla by gdn-2, gates to -5": (128, 128, steep(-5.0), 1000),
    "gla by gdn-2, gates to -20": (128, 128, steep(-20.0), 1000),
    "gla by gdn-2, spikes of -1e4": (128, 128, steep(-1.0, -1e4), 1000),
    "gla by gdn-2, more documents": (128, 128, gla, 41),
    "lightning by gdn-2": (128, 1, lightning, 1000),
}


def decayed_errors(case: str) -> list[float]:
    """How far decayed linear attention by GDN-2's kernel is from its chunks: the
    outputs, then each input's gradient, as a share of its norm. More documents a row
    than the kernel's room run by the chunks."""
    dim, decays, gate, every = DECAYED[case]
    rng = np.random.default_rng(list(DECAYED).index(case))
    positions = jnp.asarray(np.arange(T) % every, jnp.int32)
    docs = Segments.of(jnp.broadcast_to(positions, (B, T)))
    q, k, v = rng.normal(size=(3, B, T, 16, dim))
    qkv = (jnp.asarray(x, jnp.bfloat16) for x in (q / np.sqrt(dim), k, v))
    xs = (*qkv, jnp.asarray(gate(rng, decays), jnp.float32))
    assert fused_delta.linear_runs(dim, dim, jnp.bfloat16)

    def kernel(*xs: Float[Array, "..."]) -> Float[Array, "b t h dv"]:
        return linear_attention.decayed(*xs, docs).astype(jnp.float32)

    def chunks(*xs: Float[Array, "..."]) -> Float[Array, "b t h dv"]:
        q, k, v, g = (x.astype(jnp.float32) for x in xs)
        return linear_attention.recurrence(q, k, v, g, docs)

    got, want = (weighted(f, xs) for f in (kernel, chunks))
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
    kernels += [report(c, decayed_errors(c), KERNEL_TOLERANCE) for c in DECAYED]
    sys.exit(0 if all(forms) and all(kernels) else 1)


if __name__ == "__main__":
    main()
