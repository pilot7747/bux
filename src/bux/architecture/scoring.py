"""The lightning indexer's scores of every key, sum_j w_j ReLU(q_j . k), with which it
ranks them (``indexer``).

On a GPU a Pallas kernel computes them, one program per block of queries and of keys,
which keeps the block's scores in registers from head to head and skips keys after
every query's span; XLA would write and read every head's (queries, keys) instead.
Elsewhere, or at shapes the kernel does not take, XLA. The scores pass no gradient:
the indexer's loss scores its kept keys alone. On a mesh a `shard_map` gives each
device its rows, queries and heads, and sums the heads' scores.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from bux import kernels, parallel
from bux.architecture.sparse import axes

QUERIES, KEYS = 64, 128  # the most of each a program takes
WARPS = 4


def runs(t: int, s: int, i: int) -> bool:
    """Return whether the kernel runs: on a GPU, where blocks of a power of two
    divide the t queries and s keys, and the head dims i are one."""
    pow2 = i >= 16 and i & (i - 1) == 0
    return kernels.gpu() and pow2 and None not in (_block(t, QUERIES), _block(s, KEYS))


def scores(
    q: Float[Array, "b t h i"],
    w: Float[Array, "b t h"],
    keys: Float[Array, "b s i"],
    end: Int[Array, "b t"],
) -> Float[Array, "b t s"]:
    """Return every query's score of every key by the kernel: sum over heads of w
    times the ReLU'd dot product. Keys from a query's `end` on score 0 or anything:
    it does not see them."""
    q, w, keys = jax.lax.stop_gradient((q, w, keys))  # nor the parameters behind them
    split = parallel.spec(("batch", "seq", "heads", None), q.shape)
    names = axes(split)
    if not names:
        return _kernel(q, w, keys, end)
    heads = split[2]

    def device(
        q: Float[Array, "b t h i"],
        w: Float[Array, "b t h"],
        keys: Float[Array, "b s i"],
        end: Int[Array, "b t"],
    ) -> Float[Array, "b t s"]:
        own = _kernel(q, w, keys, end)
        return own if heads is None else jax.lax.psum(own, heads)

    rows = P(split[0], split[1])
    return jax.shard_map(
        device,
        in_specs=(split, P(*split[:3]), P(split[0], None, None), rows),
        out_specs=P(split[0], split[1], None),
        axis_names=names,
        check_vma=False,
    )(q, w, keys, end)


def _block(n: int, most: int) -> int | None:
    """Return the largest power of two up to `most`, and 16 or more, that divides n."""
    size = most
    while size >= 16 and n % size:
        size //= 2
    return size if size >= 16 else None


def _kernel(
    q: Float[Array, "b t h i"],
    w: Float[Array, "b t h"],
    keys: Float[Array, "b s i"],
    end: Int[Array, "b t"],
) -> Float[Array, "b t s"]:
    """Run the kernel; interpreted off a GPU, as tests run it."""
    from jax.experimental.pallas import triton as plgpu

    b, t, h, i = q.shape
    s = keys.shape[1]
    bq, bk = _block(t, QUERIES), _block(s, KEYS)
    assert bq is not None and bk is not None

    # Pallas refs, which index and load as arrays do not.
    def kernel(end_ref: Any, q_ref: Any, w_ref: Any, k_ref: Any, out_ref: Any) -> None:
        seen = pl.program_id(2) * bk < jnp.max(end_ref[...])

        @pl.when(~seen)
        def _() -> None:
            out_ref[...] = jnp.zeros((bq, bk), jnp.float32)

        @pl.when(seen)
        def _() -> None:
            k = k_ref[...]

            def head(
                j: Int[Array, ""], acc: Float[Array, "bq bk"]
            ) -> Float[Array, "bq bk"]:
                dims = (((1,), (1,)), ((), ()))
                dots = jax.lax.dot_general(
                    q_ref[j], k, dims, preferred_element_type=jnp.float32
                )
                return acc + jnp.maximum(dots, 0) * w_ref[j][:, None]

            zeros = jnp.zeros((bq, bk), jnp.float32)
            out_ref[...] = jax.lax.fori_loop(0, h, head, zeros)

    queries = (None, bq)
    return pl.pallas_call(
        kernel,
        grid=(b, t // bq, s // bk),
        in_specs=[
            pl.BlockSpec(queries, lambda z, m, n: (z, m)),
            pl.BlockSpec((None, h, bq, i), lambda z, m, n: (z, 0, m, 0)),
            pl.BlockSpec((None, h, bq), lambda z, m, n: (z, 0, m)),
            pl.BlockSpec((None, bk, i), lambda z, m, n: (z, n, 0)),
        ],
        out_specs=pl.BlockSpec((None, bq, bk), lambda z, m, n: (z, m, n)),
        out_shape=jax.ShapeDtypeStruct((b, t, s), jnp.float32),
        compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=2),
        interpret=jax.default_backend() != "gpu",
        name="indexer_scores",
    )(
        end.astype(jnp.int32),
        q.transpose(0, 2, 1, 3),
        w.transpose(0, 2, 1).astype(jnp.float32),
        keys.astype(q.dtype),
    )
