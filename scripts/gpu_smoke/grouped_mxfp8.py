"""Grouped MXFP8 GEMMs on a Blackwell GPU (bux.architecture.experts.grouped_mxfp8):
each kernel against XLA's dequantize-and-multiply, with empty groups, a group of one
row and rows past the groups; Triton's quantizer (grouped_triton) against XLA's
quantization, laid out for the kernels, and rows it reads by index against their copy;
the quantizer's bandwidth at what one GPU of megatron.bux's `sorted` arm quantizes a
layer; then an MoE layer's experts, forward and backward, in MXFP8
(``grouped_blockscaled``) against bfloat16 (``grouped``), at the baseline MoE's shapes
(64 experts, d_model 2048, intermediate 704 and 2048, 262144 picks), and the GPU time
of each by CUPTI. Last, the experts by cuDNN's fused kernels (``grouped_glu``), each
activation they fuse at a model's widths, against bux's GEMMs with them off, forward
and backward, sorted into their tokens and as an exchange's rows: hidden units padded
(1856, 192), widths of 2880, experts with no rows.

    uv run --extra cuda python scripts/gpu_smoke/grouped_mxfp8.py

It exits 1 if a kernel differs from XLA by more than float32's rounding, the
quantizer's bytes from XLA's, or the fused experts from bux's GEMMs by more than
MXFP8's rounding.
"""

import sys
from collections.abc import Callable
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.mosaic.gpu import profiler
from jaxtyping import Array, Float

from bux import config as cfg
from bux import kernels, runtime
from bux.architecture.experts import (
    grouped,
    grouped_blockscaled,
    grouped_glu,
    grouped_triton,
    moe,
)
from bux.architecture.experts import grouped_mxfp8 as gm
from bux.architecture.precision import blockscaled, matmuls
from bux.architecture.precision.blockscaled import Blocks

TOLERANCE = 1e-5  # of the largest value: float32, summed in another order
# Each kernel's cases, k not whole tiles in some: x W[g] (k, n), and dW (d, f).
ROWS = [(256, 384), (704, 256)]
CONTRACTS = [(256, 384), (704, 256)]
# What the quantizer takes: rows of x, some past the groups, and each W[g].
QUANTIZED = [((1024, 704), 900), ((4, 704, 2048), 704)]
SIZES = jnp.array([0, 130, 1, 3, 256, 0, 300, 100], jnp.int32)  # 1024 rows
# What one GPU of megatron.bux's `sorted` arm quantizes a layer: its 262144 picks' rows,
# read by index from its 32768 tokens, h, dh and dy, and each expert's W.
TOKENS = 32768
LAYER = {
    "x, read by index": (262144, 2048),
    "h": (262144, 768),
    "dh": (262144, 1536),
    "dy": (262144, 2048),
    "W of up and gate": (128, 2048, 1536),
    "W of down": (128, 768, 2048),
}
Product = Callable[..., Float[Array, "m n"]]  # x, W and maybe sizes -> x W


def mxfp8(shape: tuple[int, ...], seed: int) -> Blocks:
    """Random values in MXFP8 along their rows, each matrix of a stack on its own."""
    return quantized(jax.random.normal(jax.random.key(seed), shape))


def quantized(x: Float[Array, "*g r c"]) -> Blocks:
    """x in MXFP8 by XLA along its rows, each matrix of a stack on its own."""
    if x.ndim == 3:
        return jax.vmap(blockscaled.mx_quantize)(x)
    return blockscaled.mx_quantize(x)


def off(got: Float[Array, "..."], want: Float[Array, "..."]) -> float:
    return float(jnp.abs(got - want).max() / jnp.abs(want).max())


def rows_off(k: int, n: int) -> float:
    """How far `rows` is from XLA's dequantized product, rows past the groups too."""
    a, b = mxfp8((1024, k), 0), mxfp8((SIZES.shape[0], n, k), 1)
    got = gm.rows(gm.tiled(a), gm.tiled(b), SIZES, jnp.float32)
    each = jnp.swapaxes(jax.vmap(blockscaled.dequantized)(b), 1, 2)
    want = jax.lax.ragged_dot(blockscaled.dequantized(a), each, SIZES)
    return off(got, want)


def contract_off(d: int, f: int) -> float:
    """How far `contract` is from XLA's dequantized product, empty groups too."""
    a, b = mxfp8((d, 1024), 2), mxfp8((f, 1024), 3)
    got = gm.contract(gm.tiled(a), gm.tiled(b), SIZES, jnp.float32)
    rows = (blockscaled.dequantized(a).T, blockscaled.dequantized(b).T)
    want = jax.lax.ragged_dot_general(*rows, SIZES, grouped.RAGGED_CONTRACTING)
    return off(got, want)


def quantizer_differs(shape: tuple[int, ...], filled: int) -> int:
    """How many bytes of the quantizer's copies differ from XLA's laid out so."""
    x = jax.random.normal(jax.random.key(4), shape, jnp.bfloat16)
    rows, columns = grouped_triton.quantized_twice(x, jnp.int32(filled))
    assert columns is not None
    kept = jnp.where((jnp.arange(shape[-2]) < filled)[:, None], x, 0)
    wants = (quantized(kept), quantized(jnp.swapaxes(kept, -1, -2)))
    differ = 0
    for got, want in zip((rows, columns), wants, strict=True):
        for a, b in zip(got, gm.tiled(want), strict=True):
            differ += int((blockscaled.as_bytes(a) != blockscaled.as_bytes(b)).sum())
    return differ


def gathered_differs(shape: tuple[int, int], filled: int) -> int:
    """How many bytes of the quantizer's copies of rows it reads by index differ from
    those of their copy."""
    m, c = shape
    tokens = jax.random.normal(jax.random.key(5), (m // 3, c), jnp.bfloat16)
    index = jax.random.randint(jax.random.key(6), (m,), 0, m // 3)
    got = grouped_triton.quantized_twice(tokens, jnp.int32(filled), index=index)
    want = grouped_triton.quantized_twice(tokens[index], jnp.int32(filled))
    pairs = zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True)
    return sum(
        int((blockscaled.as_bytes(a) != blockscaled.as_bytes(b)).sum())
        for a, b in pairs
    )


def quantizer_tb_s(name: str, shape: tuple[int, ...]) -> float:
    """The quantizer's bandwidth, by CUPTI: the bytes it reads and writes a second."""
    *stack, r, c = shape
    by_index = "index" in name
    x = jax.random.normal(jax.random.key(7), (TOKENS, c) if by_index else shape)
    index = jax.random.randint(jax.random.key(8), (r,), 0, TOKENS) if by_index else None

    def quantize(x: Array) -> tuple[gm.Tiled, gm.Tiled | None]:
        return grouped_triton.quantized_twice(x, jnp.int32(r), index=index)

    _, times = profiler.measure(jax.jit(quantize), iterations=10)(
        x.astype(jnp.bfloat16)
    )
    assert times is not None
    g, rt, ct = int(np.prod(stack)), -(-r // gm.LANES), -(-c // gm.LANES)
    copies = g * gm.LANES * (r * ct + c * rt) + 2 * g * rt * ct * 512
    return (2 * g * r * c + copies) / float(np.median(np.asarray(times))) / 1e9


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


def ffn(intermediate: int, activation: cfg.Activation, **more: float) -> cfg.MLPConfig:
    """An expert's FFN: ReLU^2's ungated, the others gated."""
    kind = "mlp" if activation == "relu2" else "gated_mlp"
    return cfg.MLPConfig(
        type=kind, intermediate_size=intermediate, activation=activation, **more
    )


# Each activation the fused kernels take, at a model's experts' widths: the FFN, d, the
# experts, k picks a token.
FUSED = {
    "Nemotron 3 Nano's relu2": (ffn(1856, "relu2"), 2688, 16, 6),
    "Nemotron 3 Super's relu2": (ffn(2688, "relu2"), 1024, 64, 22),
    "gpt-oss's GeGLU, unbiased": (
        ffn(2880, "quick_gelu", limit=7.0, up_offset=1.0),
        2880,
        16,
        4,
    ),
    "DeepSeek-V4-Flash's clamped SwiGLU": (ffn(2304, "swish", limit=10.0), 4096, 16, 6),
    "Kimi K3's SiTU-GLU": (ffn(3072, "situ", up_softcap=25.0), 2048, 16, 8),
    "Qwen3-30B-A3B's SwiGLU": (ffn(768, "swish"), 2048, 16, 8),
    "relu2 of 192, padded": (ffn(192, "relu2"), 256, 8, 2),
}
FUSED_TOKENS = 4096
FUSED_TOLERANCE = 0.06  # of the fused output and gradients from bux's GEMMs' (MXFP8)


def fused_off() -> Any:
    """Return a context in which the experts run by bux's own GEMMs."""
    return mock.patch.object(grouped_glu, "runs", return_value=False)


def experts_case(name: str, summed: bool) -> tuple[Callable[..., Array], list, Array]:
    """An MoE layer's experts of FUSED[name] in MXFP8: by `moe.routed`, summed into
    the tokens, or by `moe.experts`, as an exchange's rows lie (the last expert's
    empty)."""
    config, d, experts, k = FUSED[name]
    p = 2 if config.type == "gated_mlp" else 1
    f = config.intermediate_size
    keys = jax.random.split(jax.random.key(9), 5)
    up = jax.random.normal(keys[0], (experts, d, f, p)) / d**0.5
    down = jax.random.normal(keys[1], (experts, f, d)) / f**0.5
    if summed:
        logits = jax.random.normal(keys[2], (FUSED_TOKENS, experts))
        top, idx = jax.lax.top_k(logits, k)
        weights = jax.nn.softmax(top)
        x = jax.random.normal(keys[3], (1, FUSED_TOKENS, d))

        def run(x: Array, w: Array, up: Array, down: Array) -> Array:
            return moe.routed(x, idx[None], w[None], moe.Kernels(up, down), config)

    else:
        rows = FUSED_TOKENS * k
        shares = np.append(np.ones(experts - 1), 0) / (experts - 1)
        sizes = np.random.default_rng(0).multinomial(rows * 4 // 5, shares)
        weights = jax.random.uniform(keys[2], (rows,))
        x = jax.random.normal(keys[3], (rows, d))

        def run(x: Array, w: Array, up: Array, down: Array) -> Array:
            sorted_sizes = jnp.asarray(sizes, jnp.int32)
            return moe.experts(x, sorted_sizes, w, moe.Kernels(up, down), config)

    args = [a.astype(jnp.bfloat16) for a in (x, weights, up, down)]
    dy = jax.random.normal(keys[4], run(*args).shape, jnp.bfloat16)
    return run, args, dy


def with_grads(
    run: Callable[..., Array], args: list, dy: Array, fmt: str
) -> list[np.ndarray]:
    """The experts' output and every gradient, in `fmt`."""

    def step(*args: Array) -> tuple[Array, ...]:
        with matmuls.use(fmt):
            y, vjp = jax.vjp(run, *args)
        return y, *vjp(dy)

    return [np.asarray(a, np.float32) for a in jax.jit(step)(*args)]


def frobenius_off(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def fused_off_by(name: str, summed: bool) -> float:
    """How far the fused experts are from bux's GEMMs, the worst of the output and
    gradients; infinite where they are no nearer bfloat16's than those GEMMs, by a
    margin, or not finite."""
    run, args, dy = experts_case(name, summed)
    fused = with_grads(run, args, dy, "mxfp8")
    with fused_off():
        before = with_grads(run, args, dy, "mxfp8")
        exact = with_grads(run, args, dy, "compute")
    worst = 0.0
    for got, was, want in zip(fused, before, exact, strict=True):
        near = frobenius_off(got, want) < 1.25 * frobenius_off(was, want) + 1e-3
        if not (np.isfinite(got).all() and near):
            return float("inf")
        worst = max(worst, frobenius_off(got, was))
    return worst


def fused_runs(name: str) -> bool:
    """Whether the fused kernels take FUSED[name]'s experts."""
    config, d, experts, _ = FUSED[name]
    p = 2 if config.type == "gated_mlp" else 1
    up = jnp.zeros((experts, d, config.intermediate_size, p))
    return grouped_glu.runs(blockscaled.MXFP8, blockscaled.PASSES, config, up)


def main() -> int:
    runtime.configure()
    if not kernels.gpu() or "B" not in jax.devices()[0].device_kind:
        print(f"needs a Blackwell GPU: found {jax.devices()}")
        return 1
    worst = 0.0
    for k, n in ROWS:
        worst = max(worst, e := rows_off(k, n))
        print(f"rows k {k:4} n {n:4}: off by {e:.1e}", flush=True)
    for d, f in CONTRACTS:
        worst = max(worst, e := contract_off(d, f))
        print(f"contract d {d:4} f {f:4}: off by {e:.1e}", flush=True)
    differ = 0
    for shape, filled in QUANTIZED:
        differ += (n := quantizer_differs(shape, filled))
        print(f"quantizer {shape}, {filled} rows: {n} bytes differ", flush=True)
    differ += (n := gathered_differs((1024, 2048), 900))
    print(f"quantizer of rows read by index: {n} bytes differ", flush=True)
    for name, shape in LAYER.items():
        print(f"quantizer, {name} {shape}: {quantizer_tb_s(name, shape):.2f} TB/s")
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
    fused = 0.0
    for name in FUSED:
        assert fused_runs(name), f"the fused kernels do not take {name}"
        for summed in (True, False):
            fused = max(fused, e := fused_off_by(name, summed))
            lay = "summed into tokens" if summed else "as rows"
            print(f"fused experts, {name}, {lay}: off by {e:.1e}", flush=True)
    return int(worst > TOLERANCE or differ > 0 or fused > FUSED_TOLERANCE)


if __name__ == "__main__":
    sys.exit(main())
