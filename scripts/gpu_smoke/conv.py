"""The depthwise causal conv's own backward pass on a GPU, by its kernel
(``pallas_conv``) and by XLA's form of it, against autodiff's gradients: Mamba-2's conv
and GDN's at the benchmarks' widths, in rows of one document and of packed documents,
and at the edges of what the kernel takes (dilated taps, few channels, rows whose
positions take a block of 16). Each runs in bfloat16, autodiff in float32.

    uv run --extra cuda python scripts/gpu_smoke/conv.py

It exits 1 if a gradient differs from autodiff's, or the kernel does not run.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from bux.architecture import Segments
from bux.architecture.layers import conv, pallas_conv

TOLERANCE = 1e-2  # of each gradient's norm, in bfloat16

# rows, positions, channels, taps, dilation, bias, documents a row
CASES = {
    "mamba2": (2, 4096, 6144, 4, 1, True, 0),
    "mamba2, packed documents": (2, 4096, 6144, 4, 1, True, 12),
    "gdn": (2, 4096, 2048, 4, 1, False, 0),
    "gdn, packed documents": (2, 4096, 2048, 4, 1, False, 12),
    "3 taps 2 apart, 784 channels": (3, 1008, 784, 3, 2, True, 40),
    "2 taps 9 apart, a row of 4000": (2, 4000, 512, 2, 9, False, 3),
}


def within(rng: np.random.Generator, b: int, t: int, docs: int) -> Int[Array, "b t"]:
    """How many tokens of its document come before each token, `docs` documents
    starting at random in each row."""
    positions = np.tile(np.arange(t), (b, 1))
    for row in range(b):
        for start in np.sort(rng.integers(1, t, docs)):
            positions[row, start:] = np.arange(t - start)
    return Segments.of(jnp.asarray(positions, jnp.int32)).within


def gradients(case: str, by: str) -> list[np.ndarray]:
    """The gradients of a weighted sum of the conv's outputs for x, its weights and
    bias, all of them rounded to bfloat16: by the `kernel` or `xla` in bfloat16, or
    by `autodiff` in float32."""
    b, t, c, size, dilation, bias, docs = CASES[case]
    rng = np.random.default_rng(list(CASES).index(case))
    x, w = rng.normal(size=(b, t, c)), rng.normal(size=(size, c)) / size
    bias_ = rng.normal(size=c) if bias else None
    seen = within(rng, b, t, docs) if docs else None
    xs = [jnp.asarray(a, jnp.bfloat16) for a in (x, w, bias_) if a is not None]
    if by == "autodiff":
        xs = [a.astype(jnp.float32) for a in xs]

    def loss(
        x: Float[Array, "b t c"], *params: Float[Array, "..."]
    ) -> Float[Array, ""]:
        args = (x, params[0], params[1] if bias else None, dilation, seen)
        if by == "autodiff":
            y = conv._shifted_sum(*args)
        else:
            y = conv._with_own_backward(*args, by == "kernel")
        # In bfloat16, so that both runs take the same gradient of y: the bias's,
        # a sum of it over positions that mostly cancels, would show its rounding.
        weights = jnp.cos(jnp.arange(y.size)).astype(jnp.bfloat16).reshape(y.shape)
        return (y.astype(jnp.float32) * weights).sum()

    grads = jax.jit(jax.grad(loss, tuple(range(len(xs)))))(*xs)
    return [np.asarray(g, np.float32) for g in grads]


def report(case: str, by: str) -> bool:
    """Print how far the gradients `by` the kernel or XLA are from autodiff's; return
    whether within the tolerance, and the kernel runs the case."""
    runs = by == "xla" or pallas_conv.runs(CASES[case][:3])
    got, want = gradients(case, by), gradients(case, "autodiff")
    found = [
        float(np.linalg.norm(a - b) / np.linalg.norm(b)) for a, b in zip(got, want)
    ]
    ok = runs and max(found) < TOLERANCE
    shown = " ".join(f"{e:.1e}" for e in found)
    verdict = "ok" if ok else "DIFFERS" if runs else "NOT RUN"
    print(f"{case:32} by {by:6} grads {shown}  {verdict}")
    return ok


def main() -> None:
    print(f"{jax.devices()[0].device_kind}, bfloat16 against float32")
    found = [report(case, by) for case in CASES for by in ("kernel", "xla")]
    sys.exit(0 if all(found) else 1)


if __name__ == "__main__":
    main()
