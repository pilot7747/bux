"""Grouped MXFP8 GEMMs for an MoE's experts by cuDNN-frontend's CuTe DSL kernels, on a
Blackwell GPU, through ``cudnn.jax.call`` on XLA's stream: the gated MLP's SwiGLU, and
its backward, in their epilogues.

Each takes its operands as `~bux.architecture.experts.grouped_mxfp8.Tiled` (E4M3
values, their scales in tensor memory's tiles), each group's rows from a multiple of
256 to its `ends`, and runs tiles of 256 by 256 on pairs of SMs:

- `swiglu`: ``x W[g]^T``, kept for the backward, then its SwiGLU times each row's
  weight, h, quantized along its rows and down its columns.
- `dswiglu`: ``dy W[g]`` times SwiGLU's gradient at the kept product, times each
  row's weight, quantized the same two ways; and each weight's gradient.
- `rows`: ``a b[g]^T``, in bfloat16.
- `contract`: ``a^T b`` over each group's rows, in bfloat16.

The GLU takes each gate 32 columns before its up, ``[32 gates | 32 ups]`` in each 64
of W[g]'s rows (`halves_of`). The epilogues write the copies down the columns as
`Columns`, which `contract` takes as they lie. Rows past the last group are not
written.
"""

import functools
import os
from typing import Any, NamedTuple, no_type_check

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Shaped

from bux.architecture.experts import grouped_mxfp8
from bux.architecture.experts.grouped_mxfp8 import LANES, Tiled

try:  # cudnn-frontend and CuTe DSL, where the cuda extra is installed (Linux)
    import cutlass
    import cutlass.utils
    from cudnn.gemm.cutedsl.grouped import canonical_jax
    from cudnn.gemm.cutedsl.grouped.dswiglu import jax_api as dswiglu_api
    from cudnn.gemm.cutedsl.grouped.dswiglu.api import GroupedGemmDswigluSm100
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode, WGradInputOrder
    from cudnn.gemm.cutedsl.grouped.quant import grouped_gemm_quant
    from cudnn.gemm.cutedsl.grouped.swiglu import jax_api as swiglu_api
    from cudnn.gemm.cutedsl.grouped.wgrad import moe_blockscaled_grouped_gemm_wgrad
    from cudnn.jax import TensorSpec, call, zeros_init
    from cutlass import cute
    from cutlass.cute.nvgpu import OperandMajorMode
except ImportError:
    cutlass = None

PAD = 256  # where each group's rows start: whole tiles of the kernels'
TILE = (256, 256)  # the tiles of the output on a pair of SMs
E4M3, E8M0 = jnp.float8_e4m3fn, jnp.float8_e8m0fnu


class Columns(NamedTuple):
    """An operand in MXFP8 in blocks of 32 down its columns, as the epilogues write
    it."""

    #: E4M3 values (r, c), row-major.
    values: Shaped[Array, "r c"]
    #: The scales of their transpose (c, r), laid out as
    #: `~bux.architecture.experts.grouped_mxfp8.Tiled`'s: (c', r', 32, 16).
    scales: Shaped[Array, "p q 32 16"]


class Glu(NamedTuple):
    """What `swiglu` returns."""

    #: The product ``x W[g]^T``, in bfloat16, kept for the backward.
    product: Float[Array, "m n"]
    #: h, SwiGLU's output times each row's weight, along its rows.
    h: Tiled
    #: h down its columns.
    h_columns: Columns


class GluGrad(NamedTuple):
    """What `dswiglu` returns."""

    #: The product's gradient along its rows.
    product: Tiled
    #: The product's gradient down its columns.
    product_columns: Columns
    #: Each row's weight's gradient.
    weights: Float[Array, " m"]


def runs() -> bool:
    """Return whether the kernels run here: on a Blackwell GPU, with
    cudnn-frontend."""
    return cutlass is not None and grouped_mxfp8.runs()


def fits(d: int, f: int, groups: int) -> bool:
    """Return whether the kernels take `groups` experts of width d and f hidden
    units: each of their GEMMs whole tiles of 256, and 1024 experts at most."""
    return d % TILE[1] == 0 and f % TILE[1] == 0 and groups <= 1024


def halves_of(pairs: Int[Array, " n"]) -> Int[Array, " n"]:
    """Return where columns of pairs (unit u's up at 2u, its gate at 2u + 1) lie in
    the GLU's halves: each 32 units take 64 columns, their gates first, then their
    ups."""
    unit, gate = pairs // 2, pairs % 2
    return unit // 32 * 64 + (1 - gate) * 32 + unit % 32


# --- The kernels ---------------------------------------------------------------------


def swiglu(
    x: Tiled, w: Tiled, ends: Int[Array, " g"], weights: Float[Array, " m"]
) -> Glu:
    """Return ``x W[g]^T`` for each group's rows of x, and h, its SwiGLU times each
    row's weight.

    Args:
        x: the rows, (m, k).
        w: each group's up kernel, (g, n, k), its rows in halves (`halves_of`).
        ends: where each group's rows end, each starting at a multiple of 256.
        weights: each row's weight.

    Returns:
        The product (m, n), and h (m, n / 2) along its rows and down its columns.
    """
    m, n = x.values.shape[0], w.values.shape[1]
    ins = {
        "a": x.values,
        "b": w.values,
        "sfa": x.scales,
        "sfb": w.scales,
        "padded_offsets": ends,
        "alpha": _ones(ends.shape[0]),
        "prob": weights,
        "norm_const": _ones(1),
    }
    outs = {
        "c": _out((m, n), jnp.bfloat16),
        "d": _out((m, n // 2), E4M3),
        "d_col": _out((m, n // 2), E4M3),
        "sfd_row": _out(canonical_jax.sf_shape(m, n // 2), E8M0),
        "sfd_col": _out(canonical_jax.sf_shape(n // 2, m), E8M0),
    }
    kernel, mac = swiglu_api.swiglu_plan(ins, outs, TILE, None)
    adapter = swiglu_api.grouped_swiglu_adapter
    product, h, h_col, h_scales, h_col_scales = _call(adapter, ins, outs, kernel, mac)
    return Glu(product, _tiled(h, h_scales), _columns(h_col, h_col_scales))


def dswiglu(
    dy: Tiled,
    w: Tiled,
    product: Float[Array, "m n2"],
    ends: Int[Array, " g"],
    weights: Float[Array, " m"],
) -> GluGrad:
    """Return the gradient of `swiglu`'s product, and of each row's weight.

    Args:
        dy: the output's gradient, (m, k).
        w: each group's down kernel, (g, n, k): dy times it, over k, is h's
            gradient.
        product: `swiglu`'s product, (m, 2n), in halves.
        ends: where each group's rows end, each starting at a multiple of 256.
        weights: each row's weight.

    Returns:
        The product's gradient along its rows and down its columns, and each row's
        weight's gradient.
    """
    m, n = dy.values.shape[0], w.values.shape[1]
    ins = {
        "a": dy.values,
        "b": w.values,
        "c": product,
        "sfa": dy.scales,
        "sfb": w.scales,
        "padded_offsets": ends,
        "alpha": _ones(ends.shape[0]),
        "beta": _ones(ends.shape[0]),
        "prob": weights,
        "norm_const": _ones(1),
    }
    outs = {
        "d_row": _out((m, 2 * n), E4M3),
        "d_col": _out((m, 2 * n), E4M3),
        "dprob": _out((m,), jnp.float32),
        "sfd_row": _out(canonical_jax.sf_shape(m, 2 * n), E8M0),
        "sfd_col": _out(canonical_jax.sf_shape(2 * n, m), E8M0),
    }
    kernel, mac = canonical_jax.grouped_plan(
        GroupedGemmDswigluSm100, ins, outs, mma_tiler_mn=TILE, cluster_shape_mn=None
    )
    adapter = dswiglu_api.grouped_dswiglu_adapter
    rows_, columns, dweights, scales, column_scales = _call(  # dprob: atomic adds
        adapter, ins, outs, kernel, mac, zeroed=(2,)
    )
    return GluGrad(_tiled(rows_, scales), _columns(columns, column_scales), dweights)


def rows(a: Tiled, b: Tiled, ends: Int[Array, " g"]) -> Float[Array, "m n"]:
    """Return ``a b[g]^T`` for each group's rows of a (m, k), of b's (g, n, k), in
    bfloat16; each group's rows end at `ends`, starting at a multiple of 256."""
    (m, _), (groups, n, _) = a.values.shape, b.values.shape
    kernel, mac = _gemm_kernel(groups)
    ins = (a.values, b.values, a.scales, b.scales, ends, _ones(groups))
    out = _out((m, n), jnp.bfloat16)
    return call(
        _jit(_gemm),
        output_shape_dtype=out,
        input_spec=tuple(map(_row, ins)),
        output_spec=_row(out),
        kernel=kernel,
        mac=mac,
    )(*ins)


def contract(
    a: Tiled | Columns, b: Tiled, ends: Int[Array, " g"]
) -> Float[Array, "g d f"]:
    """Return ``a^T b`` over each group's rows, (g, d, f) in bfloat16: a (d, m) in
    blocks along m as `~bux.architecture.experts.grouped_mxfp8.Tiled`, or as
    `Columns` (m, d); b (f, m) as ``Tiled``."""
    groups, f = ends.shape[0], b.values.shape[0]
    by_rows = isinstance(a, Columns)
    d = a.values.shape[1] if by_rows else a.values.shape[0]
    kernel, mac, workspace = _wgrad_kernel(groups)
    ins = (
        a.values,
        b.values,
        _by_group(a.scales, ends),
        _by_group(b.scales, ends),
        ends,
    )
    a_spec = _transposed() if by_rows else _row(a.values)
    # The kernel's own descriptors of each group's operands go in the second output.
    outs = (_out((groups, d, f), jnp.bfloat16), _out((workspace,), jnp.uint8))
    out, _ = call(
        _jit(_wgrad),
        output_shape_dtype=outs,
        input_spec=(a_spec, _transposed(), *map(_row, ins[2:])),
        output_spec=tuple(map(_row, outs)),
        kernel=kernel,
        mac=mac,
    )(*ins)
    return out


# --- Operands --------------------------------------------------------------------------


def _tiled(values: Shaped[Array, "m n"], scales: Shaped[Array, "*s"]) -> Tiled:
    """Return the epilogue's rows and their scales, (1, m', n', 32, 4, 4), as Tiled."""
    m, n = values.shape
    return Tiled(values, scales.reshape(-(-m // LANES), -(-n // LANES), 32, 16))


def _columns(values: Shaped[Array, "m n"], scales: Shaped[Array, "*s"]) -> Columns:
    """Return the epilogue's copy down the columns and its scales as Columns."""
    m, n = values.shape
    return Columns(values, scales.reshape(-(-n // LANES), -(-m // LANES), 32, 16))


def _by_group(
    scales: Shaped[Array, "p q 32 16"], ends: Int[Array, " g"]
) -> Shaped[Array, "r c"]:
    """Return the tiles of scales of a (p * 128, m) operand of `contract`, q along m,
    with each group's together, its p by its own tiles of m, as the kernel reads them:
    as (p * 128, q * 4) bytes."""
    p, q = scales.shape[:2]
    first, last = jnp.concatenate([jnp.zeros(1, ends.dtype), ends[:-1]]), ends
    a, b = first // LANES, last // LANES  # each group's tiles of m
    at = jnp.arange(p * q)  # each tile, where it goes
    group = jnp.minimum(jnp.searchsorted(p * b, at, side="right"), b.shape[0] - 1)
    within = at - p * a[group]
    count = jnp.maximum(b - a, 1)[group]
    taken = within // count * q + a[group] + within % count
    taken = jnp.where(at < p * b[-1], taken, at)  # past the groups: as they lie
    words = jax.lax.bitcast_convert_type(scales.reshape(p * q, 128, 4), jnp.uint32)
    gathered = jax.lax.bitcast_convert_type(words[taken], scales.dtype)
    return gathered.reshape(p * 128, q * 4)


def _ones(n: int) -> Float[Array, " n"]:
    return jnp.ones(n, jnp.float32)


def _out(shape: tuple[int, ...], dtype: Any) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, dtype)


def _row(a: Any) -> Any:
    """Return the spec of a row-major array as the kernels take it."""
    return TensorSpec(layout=tuple(reversed(range(len(a.shape)))))


def _transposed() -> Any:
    """Return the spec of a row-major (c, r) array as (r, c), r innermost: an operand
    M-major, or K-major as a (k, n) one."""
    return TensorSpec(mode=(1, 0))


def _call(
    adapter: Any,
    ins: dict[str, Any],
    outs: dict[str, jax.ShapeDtypeStruct],
    kernel: Any,
    mac: int,
    zeroed: tuple[int, ...] = (),
) -> Any:
    """Run a fused kernel by its adapter on row-major operands: what it adds into,
    `zeroed` first; nothing else, which it writes where read."""
    return call(
        adapter,
        output_shape_dtype=tuple(outs.values()),
        input_spec=tuple(_row(t) for t in ins.values()),
        output_spec=tuple(_row(t) for t in outs.values()),
        initialized_outputs={i: zeros_init for i in zeroed} or None,
        kernel=kernel,
        mac=mac,
    )(*ins.values())


# --- The plain kernels, and their adapters, which CuTe DSL compiles ------------------


@functools.cache
def _jit(adapter: Any) -> Any:
    """Return `adapter` compiled by CuTe DSL, once."""
    return cute.jit(adapter)


@functools.cache
def _gemm_kernel(groups: int) -> tuple[Any, int]:
    """Return cuDNN's grouped GEMM of `groups` stacked matrices, bfloat16 out, and how
    many pairs of SMs it runs on."""
    assert cutlass is not None
    kernel = grouped_gemm_quant.BlockScaledMoEGroupedGemmQuantKernel(
        sf_vec_size=32,
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=True,
        mma_tiler_mn=TILE,
        cluster_shape_mn=(2, 1),
        vectorized_f32=False,
        generate_sfd=False,
        discrete_col_sfd=False,
        enable_bias=False,
        expert_cnt=groups,
        weight_mode=MoEWeightMode.DENSE,
    )
    return kernel, _pairs_of_sms()


@functools.cache
def _wgrad_kernel(groups: int) -> tuple[Any, int, int]:
    """Return cuDNN's block-scaled wgrad into `groups` matrices, how many pairs of SMs
    it runs on, and the bytes of its descriptors."""
    assert cutlass is not None
    kernel = moe_blockscaled_grouped_gemm_wgrad.BlockScaledMoEGroupedGemmWgradKernel(
        sf_vec_size=32,
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=True,
        mma_tiler_mn=TILE,
        cluster_shape_mn=(2, 1),
        expert_cnt=groups,
        weight_mode=MoEWeightMode.DENSE,
        input_order=WGradInputOrder.Tensor2D,
    )
    return kernel, _pairs_of_sms(), max(kernel.get_workspace_bytes(), 1)


def _pairs_of_sms() -> int:
    """Return how many pairs of SMs a kernel runs on: all of them, but as many as
    CUDNNFE_CLUSTER_OVERLAP_MARGIN leaves to a kernel beside it, as cuDNN's fused
    ones do."""
    assert cutlass is not None
    margin = int(os.environ.get("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    return cutlass.utils.HardwareInfo().get_max_active_clusters(2) - margin


# CuTe DSL takes an adapter's annotations for its own types: these have none.


@no_type_check
def _gemm(stream, *operands, kernel, mac):  # noqa: ANN001, ANN002, ANN202
    """The grouped GEMM on a stack of matrices b, its output d unquantized: so the
    copy down the columns, which it does not write, is d too."""
    a, b, sfa, sfb, ends, alpha, d = operands
    kernel(
        a=a,
        b=b,
        sfb=sfb,
        n=cutlass.Int32(0),
        k=cutlass.Int32(0),
        b_stride_size=cutlass.Int64(0),
        b_major_mode=OperandMajorMode.K,
        workspace_ptr=cute.make_ptr(
            cutlass.Uint8, 0, cute.AddressSpace.gmem, assumed_align=128
        ),
        d=d,
        d_col=d,
        sfa=sfa,
        sfd_row_tensor=None,
        sfd_col_tensor=None,
        amax_tensor=None,
        norm_const_tensor=None,
        padded_offsets=ends,
        alpha=alpha,
        row_scale=None,
        bias=None,
        prob=None,
        max_active_clusters=mac,
        stream=stream,
    )


@no_type_check
def _wgrad(stream, *operands, kernel, mac):  # noqa: ANN001, ANN002, ANN202
    """The block-scaled wgrad into a stack of matrices."""
    a, b, sfa, sfb, ends, out, workspace = operands
    kernel(a, b, sfa, sfb, out, ends, workspace, mac, stream)
