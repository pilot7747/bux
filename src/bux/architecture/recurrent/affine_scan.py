"""The states of a chain of affine maps of matrices, S_{n+1} = A_n S_n + B_n from a zero
S_0: how the delta rule carries its state from chunk to chunk (``chunkwise``).

On a GPU a Pallas kernel runs the whole chain, one program per head and block of the
state's columns, which stays in registers from map to map; XLA's scan would launch
kernels for every map. Its gradient runs the same kernel back along the chain, on
the maps transposed. Elsewhere, or at shapes the kernel does not take, `lax.scan`.
On a mesh, which XLA cannot split a kernel over, a `shard_map` gives each device its
own rows and heads.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from bux import kernels, parallel

BLOCK = 32  # columns of the state a program carries
WARPS = 8
Maps = Float[Array, "z n k k"]  # A, a map of a chain: z chains of n
Adds = Float[Array, "z n k v"]  # B, or a state before each map


def states(
    A: Float[Array, "b n h k k"], B: Float[Array, "b n h k v"]
) -> tuple[Float[Array, "b n h k v"], Float[Array, "b h k v"]]:
    """Return the state before each of the n maps of each row and head, and the
    last."""
    if not _kernel_runs(A.shape[-1], B.shape[-1]):
        return _heads(A, B)
    split = parallel.spec(("batch", None, "heads", None, None), B.shape)
    if not any(split):
        return _heads(A, B)
    last = P(split[0], split[2], None, None)
    # Untyped by what varies, which a kernel's outputs do not say; nothing here
    # communicates.
    return jax.shard_map(
        _heads, in_specs=(split, split), out_specs=(split, last), check_vma=False
    )(A, B)


def _heads(
    A: Float[Array, "b n h k k"], B: Float[Array, "b n h k v"]
) -> tuple[Float[Array, "b n h k v"], Float[Array, "b h k v"]]:
    """Run `states` as one chain per row and head."""
    b, n, h, k, v = B.shape
    A, B = (jnp.moveaxis(x, 2, 1).reshape(b * h, n, k, -1) for x in (A, B))
    before, last = _chains(A, B)
    return jnp.moveaxis(before.reshape(b, h, n, k, v), 1, 2), last.reshape(b, h, k, v)


def _chains(A: Maps, B: Adds) -> tuple[Adds, Float[Array, "z k v"]]:
    """Return the state before each map, and the last."""
    if not _kernel_runs(A.shape[-1], B.shape[-1]):
        return _reference(A, B, jnp.zeros_like(B[:, 0]))
    return _kernel_states(A, B)


def _kernel_runs(k: int, v: int) -> bool:
    """Return whether the kernel runs: on a GPU, where its tiles, k by k and k by
    BLOCK, divide as Triton needs."""
    fits = k >= 16 and k & (k - 1) == 0 and v % BLOCK == 0
    return kernels.gpu() and fits


def _reference(
    A: Maps, B: Adds, start: Float[Array, "z k v"]
) -> tuple[Adds, Float[Array, "z k v"]]:
    def step(
        S: Float[Array, "z k v"],
        ab: tuple[Float[Array, "z k k"], Float[Array, "z k v"]],
    ) -> tuple[Float[Array, "z k v"], Float[Array, "z k v"]]:
        a, b = ab
        return a @ S + b, S

    maps = (jnp.moveaxis(A, 1, 0), jnp.moveaxis(B, 1, 0))
    last, before = jax.lax.scan(step, start, maps)
    return jnp.moveaxis(before, 0, 1), last


@jax.custom_vjp
def _kernel_states(A: Maps, B: Adds) -> tuple[Adds, Float[Array, "z k v"]]:
    return _run(A, B, jnp.zeros_like(B[:, 0]))


def _forward(
    A: Maps, B: Adds
) -> tuple[tuple[Adds, Float[Array, "z k v"]], tuple[Maps, Adds]]:
    before, last = _run(A, B, jnp.zeros_like(B[:, 0]))
    return (before, last), (A, before)


def _backward(
    saved: tuple[Maps, Adds], grads: tuple[Adds, Float[Array, "z k v"]]
) -> tuple[Maps, Adds]:
    """Back along the chain: the gradient of S_{n+1} is dB_n, and it reaches S_n
    through A_n^T, where that of S_n itself joins it; dA_n = dB_n S_n^T."""
    A, before = saved
    d_before, d_last = grads
    back = jnp.flip(jnp.swapaxes(A, -1, -2), 1)
    dB = jnp.flip(_run(back, jnp.flip(d_before, 1), d_last)[0], 1)
    return jnp.einsum("znkv,znjv->znkj", dB, before), dB


_kernel_states.defvjp(_forward, _backward)


def _run(
    A: Maps,
    B: Adds,
    start: Float[Array, "z k v"],
    block: int = BLOCK,
    warps: int = WARPS,
    maps: Any = jnp.float32,
) -> tuple[Adds, Float[Array, "z k v"]]:
    """Run the kernel, one program per chain and `block` columns, from `start`, the
    maps read in `maps`; interpreted off a GPU, as tests run it."""
    from jax.experimental.pallas import triton as plgpu

    z, n, k, _ = A.shape
    v = B.shape[-1]

    # Pallas refs, which index and load as arrays do not.
    def kernel(
        a_ref: Any, b_ref: Any, start_ref: Any, before_ref: Any, last_ref: Any
    ) -> None:
        def step(
            i: Int[Array, ""], S: Float[Array, "k block"]
        ) -> Float[Array, "k block"]:
            before_ref[i] = S.astype(before_ref.dtype)
            return plgpu.dot(a_ref[i], S.astype(a_ref.dtype)) + b_ref[i]

        S = start_ref[...].astype(jnp.float32)
        last_ref[...] = jax.lax.fori_loop(0, n, step, S).astype(last_ref.dtype)

    chain = pl.BlockSpec((None, n, k, block), lambda i, j: (i, 0, 0, j))
    state = pl.BlockSpec((None, k, block), lambda i, j: (i, 0, j))
    return pl.pallas_call(
        kernel,
        grid=(z, v // block),
        in_specs=[
            pl.BlockSpec((None, n, k, k), lambda i, j: (i, 0, 0, 0)),
            chain,
            state,
        ],
        out_specs=[chain, state],
        out_shape=[
            jax.ShapeDtypeStruct(B.shape, jnp.float32),
            jax.ShapeDtypeStruct(start.shape, jnp.float32),
        ],
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=2),
        interpret=kernels.interpreted(),
        name="affine_scan",
    )(A.astype(maps), B.astype(jnp.float32), start)
