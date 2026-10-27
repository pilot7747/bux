"""Tensor parallelism at a layer's edges: the methods of ``methods.tensor``.

A layer's input projections split by columns (q, k, v, an MLP's up and gate), so each
device multiplies the whole input by its own columns. Its output projection splits by
rows, so each device's product is a partial sum of the whole output. Under sequence
parallelism, ``layout(stream=tensor)``, the input comes split by positions over
``tensor``: it is gathered before the input projections, and the partial sums
reduce-scatter back to each device's positions. Without it, they all-reduce.

.. code-block:: bux

    methods = methods(tensor=chunked, tensor_chunks=2)

``gspmd`` (`GSPMD`), the default, leaves the collectives to the compiler, which puts
each at the boundary of the matmul it feeds or drains, where the matmul waits for
it. ``chunked`` (`Chunked`) splits each device's rows into ``methods.tensor_chunks``
chunks, in a shard_map::

    project_in:   gather chunk 0 | gather chunk 1    | ...
                                 | multiply chunk 0  | multiply chunk 1 | ...
    project_out:  multiply chunk 0 | multiply chunk 1 | ...
                                   | reduce chunk 0   | reduce chunk 1   | ...

A method is a class of two functions (`Method`): ``project_in(x, linears)`` returns
the input as the layer takes it and its projection by each of ``linears``, and
``project_out(h, linear)`` the output projection, split as the stream is. A layer
calls the run's by `project_in` and `project_out`.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Any, NamedTuple, Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

from bux.architecture.precision import blockscaled, matmuls
from bux.parallel import methods
from bux.parallel.layout import NAMES, Entry, Name, as_split, shard, spec, splits
from bux.parallel.mesh import active, free_axes, manual_axes

AXIS = "tensor"
# The formats that scale each block of an operand by itself, and their blocks: a
# device's share of whole blocks multiplies as the whole does.
SCALED_BY_BLOCK = {"mxfp8": blockscaled.MX_BLOCK, "mxfp8_2d": blockscaled.MX_BLOCK}


@runtime_checkable
class Method(Protocol):
    """What a method of ``methods.tensor`` is: a class of these two functions."""

    @staticmethod
    def project_in(
        x: Float[Array, "b t d"], linears: Sequence[nnx.Linear]
    ) -> tuple[Float[Array, "b t d"], list[Float[Array, "b t _"]]]:
        """Return the layer's input `x` as the layer takes it, and its projection by
        each of `linears`, split by columns over ``tensor``."""
        ...

    @staticmethod
    def project_out(
        h: Float[Array, "b t f"], linear: nnx.Linear
    ) -> Float[Array, "b t d"]:
        """Return the output projection of `h` by `linear`, split by rows over
        ``tensor``, summed and split as the stream is."""
        ...


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
    """The compiler's collectives: it gathers the input over ``tensor`` before the
    projections, and reduces the output projection's partial sums after it."""

    @staticmethod
    def project_in(
        x: Float[Array, "b t d"], linears: Sequence[nnx.Linear]
    ) -> tuple[Float[Array, "b t d"], list[Float[Array, "b t _"]]]:
        """Constrain `x` to the layer's positions and project it; the compiler
        gathers it."""
        x = shard(x, "batch", "seq", None)
        return x, [linear(x) for linear in linears]

    @staticmethod
    def project_out(
        h: Float[Array, "b t f"], linear: nnx.Linear
    ) -> Float[Array, "b t d"]:
        """Project `h`; the compiler reduces the partial sums."""
        return linear(h)


@methods.tensor.register("chunked")
class Chunked:
    """Each device's rows in chunks, in a shard_map over every axis, so that each
    chunk's collective runs beside another chunk's matmul. It is ``gspmd`` where
    there is nothing to overlap, where a projection splits otherwise than by
    ``tensor`` or the layer's rows or positions split over it, and where a device's
    share would multiply otherwise than the whole:
    in NVFP4, which scales by the largest value of the whole operand, or in MXFP8
    where a chunk or a share is not whole blocks."""

    @staticmethod
    def project_in(
        x: Float[Array, "b t d"], linears: Sequence[nnx.Linear]
    ) -> tuple[Float[Array, "b t d"], list[Float[Array, "b t _"]]]:
        """Gather each chunk of `x` over ``tensor`` beside the projection of the
        chunk before, under sequence parallelism; else as `GSPMD`."""
        split = _input_split(x)
        if split is None or not split.sequence_parallel:
            return GSPMD.project_in(x, linears)
        if not all(_splits_by_columns(linear, split) for linear in linears):
            return GSPMD.project_in(x, linears)
        kernel_specs = [_kernel_spec(linear) for linear in linears]
        outputs = [P(split.rows, split.layer, cols) for _, cols in kernel_specs]
        x, ys = _shard_map(
            partial(_gather_and_project, linears, split.chunks),
            in_specs=(split.stream_spec, kernel_specs),
            out_specs=(split.layer_spec, outputs),
        )(x, [linear.kernel[...] for linear in linears])
        return x, [_biased(linear, y) for linear, y in zip(linears, ys)]

    @staticmethod
    def project_out(
        h: Float[Array, "b t f"], linear: nnx.Linear
    ) -> Float[Array, "b t d"]:
        """Reduce, or reduce-scatter, each chunk's partial sums over ``tensor``
        beside the projection of the chunk after."""
        split = _input_split(h)
        if split is None or not _splits_by_rows(linear, split):
            return GSPMD.project_out(h, linear)
        y = _shard_map(
            partial(_project_and_reduce, linear, split.chunks, split.sequence_parallel),
            in_specs=(P(split.rows, split.layer, AXIS), _kernel_spec(linear)),
            out_specs=split.stream_spec,
        )(h, linear.kernel[...])
        return _biased(linear, y)


class InputSplit(NamedTuple):
    """How a layer's (b, t, d) input splits over the mesh: its rows; its positions as
    the stream splits them and as the layer does, the stream's over `tensor` too
    under sequence parallelism; and the chunks each device's rows split into, of
    `tokens` tokens each."""

    rows: Entry
    stream: Entry
    layer: Entry
    chunks: int
    tokens: int

    @property
    def sequence_parallel(self) -> bool:
        """Whether the stream splits its positions over ``tensor`` too."""
        return self.stream != self.layer

    @property
    def stream_spec(self) -> P:
        """The input's spec as the stream splits it."""
        return P(self.rows, self.stream, None)

    @property
    def layer_spec(self) -> P:
        """The input's spec as the layer takes it."""
        return P(self.rows, self.layer, None)


def _input_split(x: Float[Array, "b t d"]) -> InputSplit | None:
    """Return how `x` splits, or None where `tensor` does not split the layer, as in
    a shard_map already manual over it, or where it splits the layer's rows or
    positions, or where the stream splits otherwise than the layer's positions and
    then `tensor`."""
    on = active()
    if on is None or on.size(AXIS) == 1 or AXIS in manual_axes():
        return None
    rows, stream, _ = spec(("batch", "stream", None), x.shape)
    _, layer, _ = spec(("batch", "seq", None), x.shape)
    if AXIS in (*as_split(rows), *as_split(layer)):
        return None
    if as_split(stream) not in (as_split(layer), (*as_split(layer), AXIS)):
        return None
    rows_here = x.shape[0] // on.size(*as_split(rows))
    chunks = math.gcd(on.methods.tensor_chunks, rows_here)
    positions = x.shape[1] // on.size(*as_split(layer))
    return InputSplit(rows, stream, layer, chunks, rows_here // chunks * positions)


def _names(linear: nnx.Linear) -> tuple[Name, ...]:
    """Return the names of the kernel of `linear`: in a scanned pass, its own, not
    those of the stacked `layers` it came from."""
    names = linear.kernel.get_metadata().get(NAMES) or (None, None)
    return tuple(names[-2:])


def _kernel_spec(linear: nnx.Linear) -> P:
    """Return how the kernel of `linear` splits by its names, whole over FSDP's axes:
    as a step uses it."""
    return spec(_names(linear), tuple(linear.kernel.shape))


def _splits_by_columns(linear: nnx.Linear, split: InputSplit) -> bool:
    """Whether the kernel of `linear` splits by columns over `tensor`, or not at all,
    and a device's share multiplies as the whole does."""
    rows, columns = _kernel_spec(linear)
    by_columns = rows is None and columns in (None, AXIS)
    return by_columns and _multiplies_as_whole(linear, split)


def _splits_by_rows(linear: nnx.Linear, split: InputSplit) -> bool:
    """Whether the kernel of `linear` splits by rows over `tensor`, and a device's
    share multiplies as the whole does."""
    by_rows = _kernel_spec(linear) == P(AXIS, None)
    return by_rows and _multiplies_as_whole(linear, split)


def _multiplies_as_whole(linear: nnx.Linear, split: InputSplit) -> bool:
    """Whether a chunk times a device's share of the kernel of `linear` multiplies as
    the whole does, in the format it multiplies in where it traces (`_format`):
    `compute` does, and MXFP8's where the chunk's tokens and the share are whole
    blocks (`SCALED_BY_BLOCK`). NVFP4's does not: it scales by the largest value of
    the whole operand."""
    shape = tuple(linear.kernel.shape)
    share = [n // s for n, s in zip(shape, splits(_names(linear), shape))]
    match _format(linear):
        case "compute":
            return True
        case str(name) if name in SCALED_BY_BLOCK:
            block = SCALED_BY_BLOCK[name]
            return all(n % block == 0 for n in (split.tokens, *share))
    return False


def _format(linear: nnx.Linear) -> str | None:
    """Return the format `linear` multiplies in, in the branch that is tracing: its
    role's (``matmuls.format_of``); None for a dot_general of another kind."""
    dot = linear.dot_general
    if dot is jax.lax.dot_general:
        return "compute"
    if dot is matmuls.dot_general:
        return matmuls.format_of(None)
    if isinstance(dot, partial) and dot.func is matmuls.dot_general:
        return matmuls.format_of(dot.keywords.get("role"))
    return None


def _shard_map(
    body: Callable[..., Any], in_specs: tuple[Any, ...], out_specs: Any
) -> Callable[..., Any]:
    """`body` on each device's part, manual over every axis no shard_map being traced
    maps. Its matmuls may be kernels (MXFP8's), which need check_vma=False."""
    return jax.shard_map(
        body,
        in_specs=in_specs,
        out_specs=out_specs,
        axis_names=free_axes(),
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
    pieces = _in_turn(_gather, jnp.split(x, chunks))
    ys = [
        jnp.concatenate([_multiply(linear, piece, w) for piece in pieces])
        for linear, w in zip(linears, kernels)
    ]
    return jnp.concatenate(pieces), ys


def _project_and_reduce(
    linear: nnx.Linear,
    chunks: int,
    sequence_parallel: bool,
    h: Float[Array, "b T f"],
    kernel: Float[Array, "f d"],
) -> Float[Array, "b t d"]:
    """Return the sum over `tensor` of a device's rows projected by its rows of the
    kernel, a chunk of rows at a time, reduce-scattered over its positions under
    sequence parallelism: each chunk reduces while the next one multiplies."""
    partial_sums = (_multiply(linear, piece, kernel) for piece in jnp.split(h, chunks))
    reduce = partial(_reduce, sequence_parallel=sequence_parallel)
    return jnp.concatenate(_in_turn(reduce, partial_sums))


def _in_turn(
    collective: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    pieces: Iterable[Float[Array, "..."]],
) -> list[Float[Array, "..."]]:
    """Return `collective` of each of `pieces`, each after the one before: an
    optimization barrier ties a piece to the last one done, so that XLA neither
    combines the collectives into one nor runs them side by side."""
    done: list[Float[Array, "..."]] = []
    for piece in pieces:
        if done:
            piece, done[-1] = jax.lax.optimization_barrier((piece, done[-1]))
        done.append(collective(piece))
    return done


def _gather(piece: Float[Array, "c t d"]) -> Float[Array, "c T d"]:
    """Gather `piece` over `tensor` along its positions."""
    return jax.lax.all_gather(piece, AXIS, axis=1, tiled=True)


def _reduce(
    partial_sum: Float[Array, "c T d"], sequence_parallel: bool
) -> Float[Array, "c t d"]:
    """Sum `partial_sum` over `tensor`, each device keeping its own positions under
    sequence parallelism."""
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
