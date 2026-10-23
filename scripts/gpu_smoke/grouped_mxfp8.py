"""Grouped MXFP8 GEMMs on a Blackwell GPU (bux.architecture.experts.grouped_mxfp8):
each kernel against XLA's dequantize-and-multiply, with empty groups, a group of one
row and rows past the groups; then an MoE layer's experts, forward and backward, in
MXFP8 (``grouped_blockscaled``) against bfloat16 (``grouped``), at the baseline MoE's
shapes (64 experts, d_model 2048, intermediate 704 and 2048, 262144 picks), and the
GPU time of each by CUPTI.

    uv run --extra cuda python scripts/gpu_smoke/grouped_mxfp8.py

It exits 1 if a kernel differs from XLA by more than float32's rounding.
"""

import sys
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.mosaic.gpu import profiler
from jaxtyping import Array, Float

from bux import kernels, runtime
from bux.architecture.experts import grouped, grouped_blockscaled
from bux.architecture.experts import grouped_mxfp8 as gm
from bux.architecture.precision import blockscaled
from bux.architecture.precision.blockscaled import Blocks

TOLERANCE = 1e-5  # of the largest value: float32, summed in another order
# Each kernel's cases: x W (columns) and dy W^T (rows), k not whole tiles; and dW.
ROWS: list[tuple[gm.Stacked, int, int]] = [
    ("columns", 256, 384),
    ("rows", 256, 384),
    ("columns", 704, 256),
]
CONTRACTS = [(256, 384), (704, 256)]
SIZES = jnp.array([0, 130, 1, 3, 256, 0, 300, 100], jnp.int32)  # 1024 rows
Product = Callable[..., Float[Array, "m n"]]  # x, W and maybe sizes -> x W


def mxfp8(shape: tuple[int, int], seed: int) -> Blocks:
    return blockscaled.mx_quantize(jax.random.normal(jax.random.key(seed), shape))


def off(got: Float[Array, "..."], want: Float[Array, "..."]) -> float:
    return float(jnp.abs(got - want).max() / jnp.abs(want).max())


def rows_off(stacked: gm.Stacked, k: int, n: int) -> float:
    """How far `rows` is from XLA's dequantized product, rows past the groups too."""
    g = SIZES.shape[0]
    a = mxfp8((1024, k), 0)
    b = mxfp8((g * n, k) if stacked == "rows" else (n, g * k), 1)
    got = gm.rows(a, b, SIZES, stacked, jnp.float32)
    each = grouped_blockscaled._stacked(grouped_blockscaled.dequantized(b), g, stacked)
    want = jax.lax.ragged_dot(grouped_blockscaled.dequantized(a), each, SIZES)
    return off(got, want)


def contract_off(d: int, f: int) -> float:
    """How far `contract` is from XLA's dequantized product, empty groups too."""
    a, b = mxfp8((d, 1024), 2), mxfp8((f, 1024), 3)
    got = gm.contract(a, b, SIZES, jnp.float32)
    rows = (grouped_blockscaled.dequantized(a).T, grouped_blockscaled.dequantized(b).T)
    want = jax.lax.ragged_dot_general(*rows, SIZES, grouped.RAGGED_CONTRACTING)
    return off(got, want)


def experts(product: Product) -> Product:
    """An MoE's expert FFN of sorted rows: up and gate as one GEMM, then down."""

    def ffn(
        x: Float[Array, "m d"], up: Float[Array, "g d f"], down: Float[Array, "g e d"]
    ) -> Float[Array, "m d"]:
        u, g = jnp.split(product(x, up), 2, axis=-1)
        return product(jax.nn.silu(u) * g, down)

    return ffn


def gpu_ms(product: Product, f: int) -> float:
    """The GPU time of the experts' forward and backward, by CUPTI."""
    m, d, g = 262144, 2048, 64
    keys = jax.random.split(jax.random.key(4), 4)
    x = jax.random.normal(keys[0], (m, d), jnp.bfloat16)
    up = (jax.random.normal(keys[1], (g, d, 2 * f)) / d**0.5).astype(jnp.bfloat16)
    down = (jax.random.normal(keys[2], (g, f, d)) / f**0.5).astype(jnp.bfloat16)
    sizes = jnp.asarray(
        np.random.default_rng(0).multinomial(m, np.ones(g) / g), jnp.int32
    )
    dy = jax.random.normal(keys[3], (m, d), jnp.bfloat16)

    def step(x: Array, up: Array, down: Array) -> tuple[Array, ...]:
        y, vjp = jax.vjp(experts(lambda a, w: product(a, w, sizes)), x, up, down)
        return y, *vjp(dy)

    _, times = profiler.measure(jax.jit(step), iterations=5)(x, up, down)
    assert times is not None
    return float(np.median(np.asarray(times)))


def main() -> int:
    runtime.configure()
    if not kernels.gpu() or "B" not in jax.devices()[0].device_kind:
        print(f"needs a Blackwell GPU: found {jax.devices()}")
        return 1
    worst = 0.0
    for stacked, k, n in ROWS:
        worst = max(worst, e := rows_off(stacked, k, n))
        print(f"rows by {stacked:7} k {k:4} n {n:4}: off by {e:.1e}", flush=True)
    for d, f in CONTRACTS:
        worst = max(worst, e := contract_off(d, f))
        print(f"contract d {d:4} f {f:4}: off by {e:.1e}", flush=True)
    seed = jnp.uint32(0)

    def in_mxfp8(x: Array, w: Array, sizes: Array) -> Array:
        return grouped_blockscaled.ragged_dot(x, w, sizes, blockscaled.MXFP8, seed)

    for f in (704, 2048):
        bf16, mx = gpu_ms(grouped.ragged_dot, f), gpu_ms(in_mxfp8, f)
        print(
            f"experts of {f}, forward and backward: bfloat16 {bf16:.2f} ms, "
            f"MXFP8 {mx:.2f} ms ({bf16 / mx:.2f}x)",
            flush=True,
        )
    return int(worst > TOLERANCE)


if __name__ == "__main__":
    sys.exit(main())
