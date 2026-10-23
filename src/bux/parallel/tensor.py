"""Tensor parallelism at a layer's edges: the methods of `methods.tensor`.

A layer's input projections split by columns (q, k, v, an MLP's up and gate), so each
device multiplies the whole input by its own columns. Its output projection splits by
rows, so each device's product is a partial sum of the whole output. Under sequence
parallelism, `layout(stream=tensor)`, the input comes split by positions over
`tensor`: it is gathered before the input projections, and the partial sums
reduce-scatter back to each device's positions. Without it, they all-reduce.

    methods = methods(tensor=chunked, tensor_chunks=2)

`gspmd`, the default, leaves the collectives to the compiler, which puts each at the
boundary of the matmul it feeds or drains, where the matmul waits for it. `chunked`
splits each device's rows into `methods.tensor_chunks` chunks, in a shard_map: chunk
i + 1 gathers while chunk i multiplies, and chunk i reduce-scatters while chunk i + 1
multiplies.

A method is a class of two functions: `project_in(x, linears)` returns the input as
the layer takes it and its projection by each of `linears`, and `project_out(h,
linear)` the output projection, split as the stream is.
"""

import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, NamedTuple, Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

from bux.architecture import blockscaled, matmuls
from bux.parallel import methods
from bux.parallel.layout import NAMES, Name, shard, spec, splits
from bux.parallel.mesh import active

AXIS = "tensor"
Entry = str | tuple[str, ...] | None  # one dimension of a PartitionSpec


@runtime_checkable
class Method(Protocol):
    @staticmethod
    def project_in(
        x: Float[Array, "b t d"], linears: Sequence[nnx.Linear]
    ) -> tuple[Float[Array, "b t d"], list[Float[Array, "b t _"]]]: ...

    @staticmethod
    def project_out(
        h: Float[Array, "b t f"], linear: nnx.Linear
    ) -> Float[Array, "b t d"]: ...


def project_in(
    x: Float[Array, "b t d"], linears: Sequence[nnx.Linear]
) -> tuple[Float[Array, "b t d"], list[Float[Array, "b t _"]]]:
    """Return a layer's input `x` as the layer takes it, gathered under sequence
    parallelism, and its projection by each of `linears`, by the run's method."""
    return _method().project_in(x, linears)


def project_out(h: Float[Array, "b t f"], linear: nnx.Linear) -> Float[Array, "b t d"]:
    """Return the output projection of `h`, split as the stream is, by the run's
    method."""
    return _method().project_out(h, linear)


def _method() -> type[Method]:
    return cast(type[Method], methods.method("tensor"))


@methods.tensor.register("gspmd")
class GSPMD:
    """The compiler's collectives: it gathers the input over `tensor` before the
    projections, and reduces the output projection's partial sums after it."""

    @staticmethod
    def project_in(
        x: Float[Array, "b t d"], linears: Sequence[nnx.Linear]
    ) -> tuple[Float[Array, "b t d"], list[Float[Array, "b t _"]]]:
        x = shard(x, "batch", "seq", None)
        return x, [linear(x) for linear in linears]

    @staticmethod
    def project_out(
        h: Float[Array, "b t f"], linear: nnx.Linear
    ) -> Float[Array, "b t d"]:
        return linear(h)


@methods.tensor.register("chunked")
class Chunked:
    """Each device's rows in chunks, in a shard_map over every axis, so that each
    chunk's collective runs beside another chunk's matmul. It is `gspmd` where there
    is nothing to overlap, where a projection splits otherwise than by `tensor`, and
    where a device's share would multiply otherwise than the whole (`_as_whole`)."""

    @staticmethod
    def project_in(
        x: Float[Array, "b t d"], linears: Sequence[nnx.Linear]
    ) -> tuple[Float[Array, "b t d"], list[Float[Array, "b t _"]]]:
        split = _split(x)
        if split is None or not split.sequence_parallel:
            return GSPMD.project_in(x, linears)
        if not all(_by_columns(linear, split) for linear in linears):
            return GSPMD.project_in(x, linears)
        columns = [_kernel_spec(linear) for linear in linears]
        kernels = [linear.kernel[...] for linear in linears]
        outputs = [P(split.rows, split.layer, c[1]) for c in columns]
        x, ys = _shard_map(
            partial(_gather_and_project, linears, split.chunks),
            in_specs=(split.stream_spec, columns),
            out_specs=(split.layer_spec, outputs),
        )(x, kernels)
        return x, [_biased(linear, y) for linear, y in zip(linears, ys)]

    @staticmethod
    def project_out(
        h: Float[Array, "b t f"], linear: nnx.Linear
    ) -> Float[Array, "b t d"]:
        split = _split(h)
        if split is None or not _by_rows(linear, split):
            return GSPMD.project_out(h, linear)
        body = partial(
            _project_and_reduce, linear, split.chunks, split.sequence_parallel
        )
        y = _shard_map(
            body,
            in_specs=(P(split.rows, split.layer, AXIS), _kernel_spec(linear)),
            out_specs=split.stream_spec,
        )(h, linear.kernel[...])
        return _biased(linear, y)


class Split(NamedTuple):
    """How a layer's (b, t, d) input splits: its rows, its positions in the stream
    and in the layer, the chunks each device's rows split into, and the tokens of
    each."""

    rows: Entry
    stream: Entry
    layer: Entry
    chunks: int
    tokens: int

    @property
    def sequence_parallel(self) -> bool:
        return self.stream != self.layer

    @property
    def stream_spec(self) -> P:
        return P(self.rows, self.stream, None)

    @property
    def layer_spec(self) -> P:
        return P(self.rows, self.layer, None)


def _split(x: Float[Array, "b t d"]) -> Split | None:
    """Return how `x` splits, or None where `tensor` does not split the layer, or the
    stream splits otherwise than the layer's positions and then `tensor`."""
    on = active()
    manual = jax.sharding.get_abstract_mesh().manual_axes
    if on is None or on.size(AXIS) == 1 or AXIS in manual:
        return None
    rows, stream, _ = spec(("batch", "stream", None), x.shape)
    _, layer, _ = spec(("batch", "seq", None), x.shape)
    if _axes(stream) not in (_axes(layer), (*_axes(layer), AXIS)):
        return None
    rows_here = x.shape[0] // on.size(*_axes(rows))
    chunks = math.gcd(on.methods.tensor_chunks, rows_here)
    positions = x.shape[1] // on.size(*_axes(layer))
    return Split(rows, stream, layer, chunks, rows_here // chunks * positions)


def _axes(entry: Entry) -> tuple[str, ...]:
    return () if entry is None else (entry,) if isinstance(entry, str) else entry


def _names(linear: nnx.Linear) -> tuple[Name, ...]:
    """Return the names of the kernel of `linear`: in a scanned pass, its own, not
    those of the stacked `layers` it came from."""
    names = linear.kernel.get_metadata().get(NAMES) or (None, None)
    return tuple(names[-2:])


def _kernel_spec(linear: nnx.Linear) -> P:
    """Return how the kernel of `linear` splits by its names, whole over FSDP's axes:
    as a step uses it."""
    return spec(_names(linear), tuple(linear.kernel.shape))


def _by_columns(linear: nnx.Linear, split: Split) -> bool:
    """Whether the kernel of `linear` splits by columns over `tensor`, or not at all,
    and a device's share multiplies as the whole does."""
    rows, columns = _kernel_spec(linear)
    return rows is None and columns in (None, AXIS) and _as_whole(linear, split)


def _by_rows(linear: nnx.Linear, split: Split) -> bool:
    """Whether the kernel of `linear` splits by rows over `tensor`, and a device's
    share multiplies as the whole does."""
    return _kernel_spec(linear) == P(AXIS, None) and _as_whole(linear, split)


def _as_whole(linear: nnx.Linear, split: Split) -> bool:
    """Whether a chunk times a device's share of the kernel of `linear` multiplies as
    the whole does in the matmul format in force: `compute` does, and MXFP8 where
    the chunk's tokens and the share are whole blocks. NVFP4 does not: it scales by
    the largest value of the whole operand."""
    share = [
        n // s
        for n, s in zip(
            linear.kernel.shape, splits(_names(linear), linear.kernel.shape)
        )
    ]
    match matmuls.FORMAT.get():
        case "compute":
            return True
        case "mxfp8":
            return all(n % blockscaled.MX_BLOCK == 0 for n in (split.tokens, *share))
    return False


def _shard_map(
    body: Callable[..., Any], in_specs: tuple[Any, ...], out_specs: Any
) -> Callable[..., Any]:
    """`body` on each device's part, manual over every axis no shard_map being traced
    maps. Its matmuls may be kernels (MXFP8's), which need check_vma=False."""
    on = active()
    assert on is not None
    free = set(on.mesh.axis_names) - set(jax.sharding.get_abstract_mesh().manual_axes)
    return jax.shard_map(
        body,
        in_specs=in_specs,
        out_specs=out_specs,
        axis_names=free,
        check_vma=False,
    )


# --- On each device --------------------------------------------------------------------


def _gather_and_project(
    linears: Sequence[nnx.Linear],
    chunks: int,
    x: Float[Array, "b t d"],
    kernels: list[Float[Array, "d _"]],
) -> tuple[Float[Array, "b T d"], list[Float[Array, "b T _"]]]:
    """Return a device's rows gathered over `tensor` and their projections, a chunk
    of rows at a time: each chunk multiplies while the next one gathers."""
    pieces = _gathered(jnp.split(x, chunks))
    ys = [
        jnp.concatenate([_multiply(linear, piece, w) for piece in pieces])
        for linear, w in zip(linears, kernels)
    ]
    return jnp.concatenate(pieces), ys


def _gathered(pieces: list[Float[Array, "c t d"]]) -> list[Float[Array, "c T d"]]:
    """Gather each piece over `tensor` along its positions, each gather after the one
    before: an optimization barrier ties a piece to the last one gathered, so that
    XLA neither combines the gathers into one nor runs them side by side."""
    gathered: list[Float[Array, "c T d"]] = []
    for piece in pieces:
        if gathered:
            piece, gathered[-1] = jax.lax.optimization_barrier((piece, gathered[-1]))
        gathered.append(jax.lax.all_gather(piece, AXIS, axis=1, tiled=True))
    return gathered


def _project_and_reduce(
    linear: nnx.Linear,
    chunks: int,
    sequence_parallel: bool,
    h: Float[Array, "b T f"],
    kernel: Float[Array, "f d"],
) -> Float[Array, "b t d"]:
    """Return the sum over `tensor` of a device's rows projected by its rows of the
    kernel, a chunk of rows at a time, reduce-scattered over its positions under
    sequence parallelism: each chunk reduces while the next one multiplies. A barrier
    ties a chunk's partial sum to the last one reduced, as in `_gathered`."""
    reduced: list[Float[Array, "c t d"]] = []
    for piece in jnp.split(h, chunks):
        partial_sum = _multiply(linear, piece, kernel)
        if reduced:
            partial_sum, reduced[-1] = jax.lax.optimization_barrier(
                (partial_sum, reduced[-1])
            )
        reduced.append(_reduce(partial_sum, sequence_parallel))
    return jnp.concatenate(reduced)


def _reduce(
    partial_sum: Float[Array, "c T d"], sequence_parallel: bool
) -> Float[Array, "c t d"]:
    if sequence_parallel:
        return jax.lax.psum_scatter(partial_sum, AXIS, scatter_dimension=1, tiled=True)
    return jax.lax.psum(partial_sum, AXIS)


def _multiply(
    linear: nnx.Linear, x: Float[Array, "*lead i"], kernel: Float[Array, "i o"]
) -> Float[Array, "*lead o"]:
    """Return what `linear(x)` computes without its bias, of a device's part of its
    kernel: in its dtype and its matmul format."""
    x, kernel = linear.promote_dtype((x, kernel), dtype=linear.dtype)
    contract = (((x.ndim - 1,), (0,)), ((), ()))
    return linear.dot_general(x, kernel, contract, precision=linear.precision)


def _biased(linear: nnx.Linear, y: Float[Array, "b t o"]) -> Float[Array, "b t o"]:
    """Return `y` plus the bias of `linear`, if it has one, as `linear(x)` adds it."""
    if linear.bias is None:
        return y
    return y + linear.bias[...].astype(y.dtype)
