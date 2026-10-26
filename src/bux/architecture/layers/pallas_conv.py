"""The depthwise causal convolution's backward pass by a Pallas kernel, for an x whose
channels are laid out last: the gradients of x, the weights and the bias in one pass
over x and the output's gradient dy.

The conv's output at position t is ``y_t = sum_i w_i x_{t - s_i} + b``, tap i looking
``s_i = (size - 1 - i) r`` positions back, never before the document. So x's gradient
and the weights' both take dy s_i positions ahead::

    dx_t = sum_i w_i dy_{t + s_i},    dw_i = sum_t x_t dy_{t + s_i},    db = sum_t dy_t,

each term only where the document of position ``t + s_i`` started at t or before. A
program takes a block of `ROWS` positions (of the rows flattened) and `COLUMNS`
channels, `STEP` positions at a time: it writes their dx, and adds their terms of dw
and db into sums of a step's shape, which it adds up once, at its end. The blocks'
sums add up after it. Its loads of dy ahead hit the cache, so it reads x and dy from
memory once.
"""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from bux import kernels, parallel
from bux.parallel.mesh import free_axes

ROWS = 512  # positions of the rows, flattened, that a program takes
STEP = 8  # positions a step of a program's loop takes
COLUMNS = 128  # channels a program takes
WARPS = 2
F32 = jnp.float32

Ref = Any  # a Pallas ref, which indexes and loads as an array does not
Gradients = tuple[Float[Array, "b t c"], Float[Array, "size c"], Float[Array, " c"]]


def runs(shape: tuple[int, ...]) -> bool:
    """Return whether the kernel runs the backward pass of a conv of an x of `shape`,
    ``b t c``: on a GPU, on a mesh that splits nothing but its rows, each device's
    positions and the channels in blocks of a power of two, 16 or more."""
    b, t, c = shape
    if not kernels.gpu() or _rows(shape) is None:
        return False
    rows = b // parallel.splits(("batch", None, None), shape)[0]
    return None not in (_block(rows * t, ROWS), _block(c, COLUMNS))


def backward(
    x: Float[Array, "b t c"],
    weight: Float[Array, "size c"],
    dy: Float[Array, "b t c"],
    within: Int[Array, "b t"] | None,
    dilation: int,
) -> Gradients:
    """Return the gradients of the conv's x, weight and bias from its output's `dy`,
    x's in its dtype and the others in the weight's.

    Args:
        x: The conv's input.
        weight: Its taps' weights.
        dy: The gradient of its output.
        within: How many tokens of its document come before each token; None for
            rows of one document each.
        dilation: The distance between two taps, in positions.
    """
    if within is None:
        within = jnp.broadcast_to(jnp.arange(x.shape[1]), x.shape[:2])
    rows = _rows(x.shape)
    assert rows is not None

    sums: Callable[..., Any] = functools.partial(_block_sums, dilation=dilation)
    if parallel.axes_of(rows):
        ins = (rows, P(), rows, P(*rows[:2]))
        outs = (rows, P(rows[0], None, None), P(rows[0], None))
        sums = kernels.per_device(sums, ins, outs)
    dx, dw, db = sums(x, weight, dy, within.astype(jnp.int32))
    return dx, dw.sum(0).astype(weight.dtype), db.sum(0).astype(weight.dtype)


def _rows(shape: tuple[int, ...]) -> P | None:
    """Return how an x of `shape` splits over the mesh, by its rows alone; None if
    the mesh splits more of it, which XLA then runs."""
    on = parallel.active()
    if on is None:
        return P()
    rows = parallel.spec(("batch", None, None), shape)
    split = {a for a in free_axes() if on.size(a) > 1}
    return rows if split <= set(parallel.axes_of(rows)) else None


def _block(n: int, most: int) -> int | None:
    """Return the largest power of two up to `most`, and 16 or more, that divides n."""
    size = most
    while size >= 16 and n % size:
        size //= 2
    return size if size >= 16 else None


def _block_sums(
    x: Float[Array, "b t c"],
    weight: Float[Array, "size c"],
    dy: Float[Array, "b t c"],
    within: Int[Array, "b t"],
    dilation: int,
) -> tuple[Float[Array, "b t c"], Float[Array, "n size c"], Float[Array, "n c"]]:
    """Return dx, and each block of positions' sums of dw and db, by the kernel;
    interpreted off a GPU, as tests run it."""
    from jax.experimental.pallas import triton as plgpu

    b, t, c = x.shape
    size, n = weight.shape[0], b * t
    rows, columns = _block(n, ROWS), _block(c, COLUMNS)
    assert rows is not None and columns is not None
    step = min(STEP, rows)
    aheads = [(size - 1 - i) * dilation for i in range(size)]  # the last is 0

    def kernel(
        within_ref: Ref, x_ref: Ref, dy_ref: Ref, w_ref: Ref, *outs: Ref
    ) -> None:
        dx_ref, dw_ref, db_ref = outs
        weights = [w_ref[i, :].astype(F32) for i in range(size)]
        start = pl.program_id(0) * rows  # outside the loop, as interpreting takes it

        def run(j: Int[Array, ""], sums: tuple[Array, ...]) -> tuple[Array, ...]:
            first = start + j * step
            x = x_ref[pl.ds(first, step), :].astype(F32)
            dx = jnp.zeros((step, columns), F32)
            dws = []
            for i, (w, ahead) in enumerate(zip(weights, aheads)):
                g = _ahead(within_ref, dy_ref, first, ahead, step, n)
                dws.append(sums[i] + g * x)
                dx += w * g
            dx_ref[pl.ds(j * step, step), :] = dx.astype(dx_ref.dtype)
            return (*dws, sums[-1] + g)  # g of the last tap: dy itself

        zeros = (jnp.zeros((step, columns), F32),) * (size + 1)
        sums = jax.lax.fori_loop(0, rows // step, run, zeros)
        for i in range(size):
            dw_ref[i, :] = jnp.sum(sums[i], axis=0)
        db_ref[...] = jnp.sum(sums[-1], axis=0)

    slab = pl.BlockSpec((n, columns), lambda i, j: (0, j))  # every position
    dx, dw, db = pl.pallas_call(
        kernel,
        grid=(n // rows, c // columns),
        in_specs=[
            pl.BlockSpec((n,), lambda i, j: (0,)),
            slab,
            slab,
            pl.BlockSpec((size, columns), lambda i, j: (0, j)),
        ],
        out_specs=[
            pl.BlockSpec((rows, columns), lambda i, j: (i, j)),
            pl.BlockSpec((None, size, columns), lambda i, j: (i, 0, j)),
            pl.BlockSpec((None, columns), lambda i, j: (i, j)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((n, c), x.dtype),
            jax.ShapeDtypeStruct((n // rows, size, c), F32),
            jax.ShapeDtypeStruct((n // rows, c), F32),
        ],
        compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=2),
        interpret=kernels.interpreted(),
        name="causal_conv_backward",
    )(within.reshape(n), x.reshape(n, c), dy.reshape(n, c), weight)
    return dx.reshape(b, t, c), dw, db


def _ahead(
    within_ref: Ref, dy_ref: Ref, first: Int[Array, ""], ahead: int, step: int, n: int
) -> Float[Array, "step c"]:
    """Return dy `ahead` positions after each of the `step` from `first`, in float32;
    0 where that position's document starts after the step's, or past the last."""
    from jax.experimental.pallas import triton as plgpu

    at = pl.ds(first + ahead, step)
    inside = first + ahead + jnp.arange(step) < n
    seen = plgpu.load(within_ref.at[at], mask=inside, other=-1)
    every = jnp.broadcast_to(inside[:, None], (step, dy_ref.shape[-1]))
    dy = plgpu.load(dy_ref.at[at, :], mask=every, other=0).astype(F32)
    return jnp.where((seen >= ahead)[:, None], dy, 0.0)
