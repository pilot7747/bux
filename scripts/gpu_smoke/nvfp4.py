"""NVFP4's quantizers on a B200 or B300 (bux.architecture.precision.nvfp4): Transformer
Engine's kernels (``nvfp4_te``) and bux's Pallas kernels against XLA's quantizer, at
the 1.3B's projections (32768 rows of 2048, 5632 and 1024, and its weights) and at the
edge of TE's tiles (64 x 128); that shapes outside the tiles, float32, and x of fewer
than 8192 rows go to the Pallas kernels; that x split over the GPUs quantizes as it
does whole; that dy's
stochastic rounding is unbiased; then a projection's y, dx and dW by each, against
bfloat16, and the quantizers' GPU time by CUPTI.

    uv run --extra cuda python scripts/gpu_smoke/nvfp4.py

TE divides exactly where XLA on a GPU does not, and rounds the Hadamard transform to
bfloat16 before it quantizes it: so a few of its values are a step from XLA's. It
exits 1 if more are, or a scale or a tensor scale is further off, or a projection's
error is.
"""

import sys
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.mosaic.gpu import profiler
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

from bux import kernels, runtime
from bux.architecture.precision import blockscaled, matmuls, nvfp4, nvfp4_te
from bux.architecture.precision.blockscaled import Blocks

M = 32768
SHAPES = [(M, 2048), (M, 5632), (M, 1024), (64, 128), (192, 384)]
WEIGHTS = [(2048, 5632), (5632, 2048), (2048, 1024), (64, 128)]
#: Each copy's bounds against XLA's: the share of values a step off, of scales off,
#: and how far the tensor scale is, relatively.
ROWS = (5e-3, 0.0, 0.0)
COLUMNS = (2e-2, 5e-2, 2**-8)  # TE's transform, rounded to bfloat16
Twice = Callable[..., tuple[Blocks, Blocks]]


def operand(shape: tuple[int, int], key: int = 0) -> Float[Array, "r c"]:
    x = jax.random.normal(jax.random.key(key), shape) * 2
    return x.at[7, 9].set(37.0).astype(jnp.bfloat16)  # an outlier sets the scale


def off(got: Blocks, want: Blocks) -> tuple[float, float, float]:
    """Return the share of values that differ, of scales, and how far the tensor
    scales are apart, relatively."""
    values = got.values.astype(jnp.float32) != want.values.astype(jnp.float32)
    scales = got.scales.astype(jnp.float32) != want.scales.astype(jnp.float32)
    assert got.tensor_scale is not None and want.tensor_scale is not None
    tensor = jnp.abs(got.tensor_scale / want.tensor_scale - 1)
    return float(values.mean()), float(scales.mean()), float(tensor)


def within(name: str, got: Blocks, want: Blocks, bounds: tuple[float, ...]) -> bool:
    found = off(got, want)
    ok = all(f <= b for f, b in zip(found, bounds, strict=True))
    print(f"  {name}: values {found[0]:.1e}, scales {found[1]:.1e}, tensor scale "
          f"{found[2]:.1e} off{'' if ok else '  FAILED'}", flush=True)  # fmt: skip
    return ok


def agree(shape: tuple[int, int], tiles: bool) -> bool:
    """Whether TE's and the Pallas kernels' copies of x agree with XLA's."""
    x, ok = operand(shape), True
    xla = jax.jit(lambda x: nvfp4.xla_twice(x, tiles=tiles))(x)
    for name, twice in (("te", nvfp4_te.twice), ("pallas", nvfp4.kernel_twice)):
        rows, cols = jax.jit(lambda x, f=twice: f(x, tiles=tiles))(x)
        print(f"{'W' if tiles else 'x'} {shape} by {name}:", flush=True)
        ok &= within("rows", rows, xla[0], ROWS)
        ok &= within("columns", cols, xla[1], ROWS if tiles else COLUMNS)
    return ok


def routes() -> bool:
    """Whether TE takes the shapes and dtypes it should, and only those: of x, rows
    enough; of W, any."""
    want = {
        ((8192, 128), jnp.bfloat16, False): True,
        ((64, 128), jnp.bfloat16, True): True,
        ((4096, 128), jnp.bfloat16, False): False,
        ((8192, 64), jnp.bfloat16, False): False,
        ((8192, 128), jnp.float32, False): False,
    }
    got = {k: nvfp4_te.takes(jnp.zeros(k[0], k[1]), k[2]) for k in want}
    print(f"TE takes {[k for k, v in got.items() if v]}", flush=True)
    return got == want


def split_as_whole() -> bool:
    """Whether x split over the GPUs' rows quantizes as it does whole."""
    devices = jax.devices()
    mesh = jax.make_mesh((len(devices),), ("data",), devices=devices)
    x = operand((M, 2048))
    whole = jax.jit(nvfp4_te.twice)(x)
    parts = jax.jit(nvfp4_te.twice)(jax.device_put(x, NamedSharding(mesh, P("data"))))
    same = all(
        jnp.array_equal(a.astype(jnp.float32), b.astype(jnp.float32))
        for got, want in zip(parts, whole, strict=True)
        for a, b in ((got.values, want.values), (got.scales, want.scales))
    )
    print(f"x split over {len(devices)} GPUs quantizes as whole: {same}", flush=True)
    return same


def unbiased() -> bool:
    """Whether dy's stochastic rounding is unbiased: 5 between E2M1's 4 and 6, its
    block's largest 6, rounds to either half the time."""
    x = jnp.full((4096, 1024), 5.0).at[:, ::16].set(6.0).astype(jnp.bfloat16)
    rows, _ = jax.jit(lambda x: nvfp4_te.twice(x, jnp.uint32(3), stochastic=True))(x)
    rest = np.arange(1024) % 16 != 0
    mean = float(rows.values.astype(jnp.float32)[:, rest].mean())
    print(f"dy of 5s rounds stochastically to a mean of {mean:.4f}", flush=True)
    return abs(mean - 5) < 2e-3


# NVFP4 by bux's Pallas kernels alone, to compare TE's against.
PALLAS = blockscaled.Recipe(
    nvfp4.BLOCK,
    x=nvfp4.kernel_twice,
    w=partial(nvfp4.kernel_twice, tiles=True),
    dy=partial(nvfp4.kernel_twice, stochastic=True),
)
matmuls.formats.register("nvfp4_pallas")(matmuls.block_scaled(PALLAS))


def products(fmt: str, k: int, n: int) -> tuple[Array, ...]:
    """y, dx and dW of a projection of 32768 rows, k to n, in `fmt`."""
    x, w = operand((M, k)), (operand((k, n), 1) * 0.02).astype(jnp.bfloat16)
    dy = operand((M, n), 2)

    def project(x: Array, w: Array) -> Array:
        with matmuls.use(fmt):
            return matmuls.dot_general(x, w, (((1,), (0,)), ((), ())))

    def outs(x: Array, w: Array) -> tuple[Array, ...]:
        y, vjp = jax.vjp(project, x, w)
        return y, *vjp(dy)

    return jax.jit(outs)(x, w)


def errors(k: int, n: int) -> dict[str, list[float]]:
    """Each NVFP4 route's error in y, dx and dW against bfloat16's."""
    exact = products("compute", k, n)

    def error(fmt: str) -> list[float]:
        pairs = zip(products(fmt, k, n), exact, strict=True)
        return [_relative(a, b) for a, b in pairs]

    return {"te": error("nvfp4"), "pallas": error("nvfp4_pallas")}


def _relative(a: Array, b: Array) -> float:
    a, b = a.astype(jnp.float32), b.astype(jnp.float32)
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def gpu_us(twice: Twice, shape: tuple[int, int], **kind: bool) -> float:
    x = operand(shape)
    _, times = profiler.measure(jax.jit(lambda x: twice(x, **kind)), iterations=10)(x)
    assert times is not None
    return 1e3 * float(np.median(np.asarray(times)))


def main() -> int:
    runtime.configure()
    if not kernels.gpu() or nvfp4_te._te() is None:
        print(f"needs a B200 or B300 with TE's JAX bindings: found {jax.devices()}")
        return 1
    ok = routes()
    for shape in SHAPES:
        ok &= agree(shape, tiles=False)
    for shape in WEIGHTS:
        ok &= agree(shape, tiles=True)
    ok &= split_as_whole() and unbiased()
    for name, (k, n) in {"up": (2048, 5632), "k": (2048, 1024)}.items():
        found = errors(k, n)
        for route, e in found.items():
            print(f"{name} {M}x{k}x{n} by {route}: y, dx and dW off bfloat16's by "
                  + ", ".join(f"{a:.4f}" for a in e), flush=True)  # fmt: skip
        ok &= all(a <= 1.05 * b for a, b in zip(found["te"], found["pallas"]))
    for shape in SHAPES[:3]:
        for kind in ({}, {"stochastic": True}):
            te, pallas = (
                gpu_us(f, shape, **kind) for f in (nvfp4_te.twice, nvfp4.kernel_twice)
            )
            print(f"quantizing {'dy' if kind else 'x'} {shape}: TE {te:.0f} us, "
                  f"Pallas {pallas:.0f} us ({pallas / te:.2f}x)", flush=True)  # fmt: skip
    return int(not ok)


if __name__ == "__main__":
    sys.exit(main())
