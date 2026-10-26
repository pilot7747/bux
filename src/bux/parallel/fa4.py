"""FlashAttention-4's kernels (flash-attn-4, in CuTe DSL) on a Blackwell GPU, through
``cutlass.jax.cutlass_call`` on XLA's stream: `kernel` makes a
`bux.parallel.flash.Kernel` of them.

They attend causally to whole rows, or to the documents packed into them in FA4's
varlen layout: the rows end to end, and where each document starts (cu_seqlens), with
room for `room` documents. A window is FA4's local mask, a soft cap its score_mod, and
sinks fold into the logsumexp by `bux.parallel.flash.attend`.

A document starts where a query's first key (`bux.parallel.flash.Keys.start`) is its
own position: its document's start, which a window's never is (but in a window of
one, where every query sees only itself either way).

The forward pass is one kernel. The backward pass is four: each query's ``dO . O`` and
its logsumexp in base 2, then the main kernel, which adds dQ (and GQA's dK and dV)
into float32, then one for each that rounds it to the inputs' dtype.

FA4's modules import torch (the cuda extra's CPU build), so they load when a call
first needs them.
"""

import functools
from types import SimpleNamespace
from typing import Any, NamedTuple, no_type_check

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from bux import kernels
from bux.parallel import flash
from bux.parallel.flash import Grads, Keys, Output

TILE = 128  # the kernels' blocks of queries and of keys
DIMS = 32  # the backward kernels pad head dims to a multiple of this
# Documents cu_seqlens has room for, a row of 4096. Each empty one costs time: a
# layer of the 1.3B on 8 rows of 4096 packed (53 documents) takes 2.50 ms forward and
# backward with room for 128, 3.05 for 256 and 3.96 for 512.
DOCS = 16
F32 = jnp.float32
Shape = jax.ShapeDtypeStruct


@functools.cache
def _modules() -> SimpleNamespace | None:
    """Return FA4's kernels and CuTe DSL's call from jax, None where flash-attn-4 is
    not installed (it is on Linux, with the cuda extra)."""
    try:
        import cutlass
        from cutlass import cute
        from cutlass.jax import cutlass_call
        from flash_attn.cute import utils
        from flash_attn.cute.flash_bwd_postprocess import (
            FlashAttentionBackwardPostprocess,
        )
        from flash_attn.cute.flash_bwd_preprocess import (
            FlashAttentionBackwardPreprocess,
        )
        from flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
        from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
    except ImportError:
        return None
    return SimpleNamespace(
        cutlass=cutlass,
        cute=cute,
        call=cutlass_call,
        utils=utils,
        Forward=FlashAttentionForwardSm100,
        Backward=FlashAttentionBackwardSm100,
        Preprocess=FlashAttentionBackwardPreprocess,
        Postprocess=FlashAttentionBackwardPostprocess,
    )


def _fa4() -> SimpleNamespace:
    modules = _modules()
    assert modules is not None
    return modules


def runs() -> bool:
    """Return whether FA4's kernels run here: on a GPU of compute capability 10
    (B200, B300), with flash-attn-4, outside `bux.kernels.off`."""
    return kernels.gpu() and _arch() // 10 == 10 and _modules() is not None


@functools.cache
def _arch() -> int:
    """Return this host's GPUs' compute capability as FA4 numbers it: 103 for 10.3."""
    capability = getattr(jax.local_devices()[0], "compute_capability", "0.0")
    major, minor = str(capability).split(".")
    return int(major) * 10 + int(minor)


def takes(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
) -> bool:
    """Return whether FA4's kernels take q, k and v here (`runs`): bfloat16 or
    float16 alike, and head dims its Blackwell kernels have: keys' and values' equal,
    up to 128 in steps of 8, or MLA's 192 and 128."""
    dk, dv = q.shape[-1], v.shape[-1]
    dims = (dk == dv <= 128 and dk % 8 == 0) or (dk, dv) == (192, 128)
    dtype = q.dtype in (jnp.bfloat16, jnp.float16) and q.dtype == k.dtype == v.dtype
    return dims and dtype and runs()


@functools.cache
def kernel(window: int | None) -> flash.Kernel:
    """Return FA4's kernels as a `bux.parallel.flash.Kernel`, attending within
    `window` (a query sees no key `window` or more behind it) if any. They take
    causal keys with no last key: whole rows, or the documents of a query's first
    key."""
    return flash.Kernel(
        functools.partial(_forward, window), functools.partial(_backward, window)
    )


# --- Documents ----------------------------------------------------------------------


def room(b: int, t: int) -> int:
    """Return how many documents cu_seqlens has room for in b rows of t: `DOCS` a row
    of 4096, and one a row at least."""
    return b * max(1, t * DOCS // 4096)


def starts(start: Int[Array, "b t"]) -> Bool[Array, "b t"]:
    """Return where documents start in rows whose queries' first keys are `start`:
    at a query that is its own first key."""
    return start == jnp.arange(start.shape[1])


def fits(start: Int[Array, "b t"]) -> Bool[Array, ""]:
    """Return whether cu_seqlens has room for the documents of rows whose queries'
    first keys are `start`."""
    return starts(start).sum() <= room(*start.shape)


def cu_seqlens(start: Int[Array, "b t"]) -> Int[Array, " n1"]:
    """Return where each document of the rows end to end starts, and where the last
    one ends: `room` + 1 of them, the last documents empty."""
    b, t = start.shape
    first = jnp.nonzero(starts(start).reshape(-1), size=room(b, t), fill_value=b * t)
    return jnp.append(first[0], b * t).astype(jnp.int32)


def _tokens(x: Float[Array, "b t h e"]) -> Float[Array, "n h e"]:
    """Return rows end to end."""
    return x.reshape(-1, *x.shape[2:])


def _rows_lse(lse: Float[Array, "h n"], b: int) -> Float[Array, "b h t"]:
    """Return the logsumexp of b rows' tokens end to end as the rows'."""
    return lse.reshape(lse.shape[0], b, -1).transpose(1, 0, 2)


def _tokens_lse(lse: Float[Array, "b h t"]) -> Float[Array, "h n"]:
    """Undo `_rows_lse`."""
    return lse.transpose(1, 0, 2).reshape(lse.shape[1], -1)


# --- The passes ---------------------------------------------------------------------


def _forward(
    window: int | None,
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t g dk"],
    v: Float[Array, "b t g dv"],
    keys: Keys,
    scale: float,
    cap: float | None,
) -> Output:
    _check(keys)
    if keys.start is None:
        return _forward_call(window, q, k, v, None, scale, cap, q.shape[1])
    cu = cu_seqlens(keys.start)
    tokens = (_tokens(x) for x in (q, k, v))
    out, lse = _forward_call(window, *tokens, cu, scale, cap, q.shape[1])
    return out.reshape(*q.shape[:3], -1), _rows_lse(lse, q.shape[0])


def _backward(
    window: int | None,
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t g dk"],
    v: Float[Array, "b t g dv"],
    out: Float[Array, "b t h dv"],
    dout: Float[Array, "b t h dv"],
    lse: Float[Array, "b h t"],
    keys: Keys,
    scale: float,
    cap: float | None,
) -> Grads:
    _check(keys)
    t = q.shape[1]
    if keys.start is None:
        return _backward_call(window, q, k, v, out, dout, lse, None, scale, cap, t)
    cu = cu_seqlens(keys.start)
    tokens = (_tokens(x) for x in (q, k, v, out, dout))
    grads = _backward_call(window, *tokens, _tokens_lse(lse), cu, scale, cap, t)
    dq, dk, dv = (d.reshape(x.shape) for d, x in zip(grads, (q, k, v)))
    return dq, dk, dv


def _check(keys: Keys) -> None:
    if not keys.causal or keys.end is not None:
        raise ValueError("FA4's kernels take causal keys with no last key")


class Config(NamedTuple):
    """What a call's kernels are compiled for."""

    dtype: Any  #: q's, k's and v's.
    dk: int  #: The keys' head dim.
    dv: int  #: The values' head dim.
    groups: int  #: Query heads a kv head.
    window: int | None  #: The window, if any.
    cap: float | None  #: The soft cap, if any.
    varlen: bool  #: Whether the rows are end to end, with cu_seqlens.


def _config(
    q: Float[Array, "*r h dk"],
    v: Float[Array, "*r g dv"],
    window: int | None,
    cap: float | None,
    cu: Int[Array, " n1"] | None,
) -> Config:
    groups = q.shape[-2] // v.shape[-2]
    dk, dv = q.shape[-1], v.shape[-1]
    return Config(q.dtype, dk, dv, groups, window, cap, cu is not None)


def _forward_call(
    window: int | None,
    q: Float[Array, "*r h dk"],
    k: Float[Array, "*r g dk"],
    v: Float[Array, "*r g dv"],
    cu: Int[Array, " n1"] | None,
    scale: float,
    cap: float | None,
    t: int,
) -> tuple[Float[Array, "*r h dv"], Float[Array, "..."]]:
    """Return FA4's output and logsumexp of rows of t (b, t, h, d), or of tokens
    (n, h, d) end to end in the documents cu_seqlens bounds, in rows of t."""
    config = _config(q, v, window, cap, cu)
    if cu is None:
        lse = Shape((q.shape[0], q.shape[2], t), F32)  # (b, h, t)
    else:
        lse = Shape((q.shape[1], q.shape[0]), F32)  # (h, n)
    out = Shape((*q.shape[:-1], config.dv), q.dtype)
    call = _fa4().call(
        _adapter(_launch_forward),
        output_shape_dtype=(out, lse),
        kernel=_forward_kernel(config, _q_stage(t * config.groups)),
        scale=scale,
        window=window,
        rows=None if cu is None else t,
    )
    return call(q, k, v, cu)


def _q_stage(rows: int) -> int:
    """Return how many blocks of queries a forward CTA takes of a kv head's `rows`:
    two, but where there is only one."""
    return 2 if rows > TILE else 1


def _backward_call(
    window: int | None,
    q: Float[Array, "*r h dk"],
    k: Float[Array, "*r g dk"],
    v: Float[Array, "*r g dv"],
    out: Float[Array, "*r h dv"],
    dout: Float[Array, "*r h dv"],
    lse: Float[Array, "..."],
    cu: Int[Array, " n1"] | None,
    scale: float,
    cap: float | None,
    t: int,
) -> Grads:
    """Return FA4's gradients of q, k and v, laid out as `_forward_call` takes them,
    given the output and logsumexp of the softmax they are part of."""
    config = _config(q, v, window, cap, cu)
    plan = _backward_plan(config, flash.deterministic())
    scratch = Scratch.of(plan, q, k, v, cu, t)
    zeroed = tuple(jnp.zeros(x.shape, x.dtype) for x in scratch.zeroed)
    ins = (q, k, v, out, dout, lse, cu, *zeroed)
    grads = tuple(Shape(x.shape, x.dtype) for x in (q, k, v))
    outs = (*grads, *scratch.written, *scratch.zeroed)
    # each zeroed input comes back as its output, which the kernels write in place
    into, back = len(ins) - len(zeroed), len(outs) - len(zeroed)
    call = _fa4().call(
        _adapter(_launch_backward),
        output_shape_dtype=outs,
        input_output_aliases={into + i: back + i for i in range(len(zeroed))},
        plan=plan,
        scale=scale,
        window=window,
    )
    dq, dk, dv, *_ = call(*ins)
    return dq, dk, dv


# --- What the backward pass needs beside its inputs ----------------------------------


def _round(x: int, to: int) -> int:
    return -(-x // to) * to


def _slack(n: int, sequences: int, tile: int) -> int:
    """Return n rows and a block of slack for each of the sequences, in whole blocks,
    as FA4's interface sizes the scratch of tokens end to end."""
    return (n + sequences * tile - 1) // tile * tile


class BackwardPlan(NamedTuple):
    """FA4's backward kernels for a call, compiled for its `Config`."""

    pre: Any  #: Each query's dO . O and logsumexp in base 2; dQ's float32 zeroed.
    main: Any  #: dQ (added into float32), dK and dV.
    dq: Any  #: dQ rounded.
    dk: Any  #: GQA's dK rounded.
    dv: Any  #: GQA's dV rounded.
    groups: int  #: Query heads a kv head.
    cluster: int  #: CTAs a cluster: two for head dims of 128 or more.
    deterministic: bool  #: Whether the main kernel orders its adds.

    def key_rows(self, t: int) -> int:
        """Return the rows of a kv head's float32 dK and dV for t keys: whole
        blocks, of whole clusters."""
        rows = _round(t, TILE)
        return rows + TILE if self.cluster == 2 and rows // TILE % 2 else rows

    def semaphores(self, b: int, h: int, g: int, t: int) -> tuple[Shape, ...]:
        """Return the semaphores of a deterministic backward pass of b sequences of
        up to t: dQ's, and GQA's dK's and dV's."""
        dq = Shape((b, h, _round(t, TILE) // TILE, self.cluster), jnp.int32)
        if self.groups == 1:
            return (dq,)
        dkv = Shape((b, g, self.key_rows(t) // TILE, 2), jnp.int32)
        return dq, dkv, dkv


class Scratch(NamedTuple):
    """The buffers the backward kernels pass to each other."""

    #: Those the first kernel writes: dQ in float32, each query's dO . O and its
    #: logsumexp in base 2.
    written: tuple[Shape, ...]
    #: Those the main kernel adds into, which start at 0: GQA's dK and dV in float32,
    #: and the semaphores that order the adds when they are deterministic.
    zeroed: tuple[Shape, ...]

    @classmethod
    def of(
        cls,
        plan: BackwardPlan,
        q: Float[Array, "*r h dk"],
        k: Float[Array, "*r g dk"],
        v: Float[Array, "*r g dv"],
        cu: Int[Array, " n1"] | None,
        t: int,
    ) -> "Scratch":
        """Return the scratch of rows of t (b, t, h, d), or of tokens end to end
        (n, h, d) in the documents cu_seqlens bounds, as FA4's interface sizes it."""
        h, g = q.shape[-2], k.shape[-2]
        dk, dv = _round(q.shape[-1], DIMS), _round(v.shape[-1], DIMS)
        if cu is None:
            lead, sequences = (q.shape[0],), q.shape[0]
            queries, keys = _round(t, TILE), plan.key_rows(t)
        else:
            lead, sequences = (), cu.shape[0] - 1
            queries = _slack(q.shape[0], cu.shape[0], TILE)
            keys = _slack(q.shape[0], cu.shape[0], plan.cluster * TILE)
        written = (
            Shape((*lead, h, queries * dk), F32),
            Shape((*lead, h, queries), F32),
            Shape((*lead, h, queries), F32),
        )
        zeroed: tuple[Shape, ...] = ()
        if plan.groups > 1:
            zeroed = (
                Shape((*lead, g, keys * dk), F32),
                Shape((*lead, g, keys * dv), F32),
            )
        if plan.deterministic:
            zeroed += plan.semaphores(sequences, h, g, t)
        return cls(written, zeroed)


# --- The kernels, made once a configuration ------------------------------------------


@functools.cache
def _softcap(cap: float | None) -> tuple[Any, Any]:
    """Return FA4's score_mod for a soft cap and its backward, (None, None) for
    none."""
    if cap is None:
        return None, None
    utils = _fa4().utils
    return utils.create_softcap_scoremod(cap), utils.create_softcap_scoremod_bwd(cap)


@functools.cache
def _forward_kernel(config: Config, q_stage: int) -> Any:
    """Return FA4's forward kernel for `config`, as its interface configures it on
    SM100: blocks of 128 by 128, a kv head's query heads packed together."""
    local = config.window is not None
    return _fa4().Forward(
        config.dk,
        config.dv,
        qhead_per_kvhead=config.groups,
        is_causal=not local,
        is_local=local,
        pack_gqa=config.groups > 1,
        m_block_size=TILE,
        n_block_size=TILE,
        q_stage=q_stage,
        is_static_persistent=False,
        score_mod=_softcap(config.cap)[0],
        is_varlen_q=config.varlen,
    )


@functools.cache
def _backward_plan(config: Config, deterministic: bool) -> BackwardPlan:
    """Return FA4's backward kernels for `config`, as its interface configures them
    on SM100: blocks of 128 by 128, on pairs of CTAs for head dims of 128 or more."""
    fa4 = _fa4()
    pair = config.dk >= 128
    cluster = 2 if pair else 1
    dq_order, dkv_order = fa4.Backward.accum_tile_orders(pair, TILE)
    local = config.window is not None
    score_mod, score_mod_bwd = _softcap(config.cap)
    main = fa4.Backward(
        config.dk,
        config.dv,
        is_causal=not local,
        is_local=local,
        qhead_per_kvhead=config.groups,
        tile_m=TILE,
        tile_n=TILE,
        cluster_size=cluster,
        use_2cta_instrs=pair,
        deterministic=deterministic,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
    )
    rounds = functools.partial(_postprocess, _cute_dtype(config.dtype))
    return BackwardPlan(
        pre=fa4.Preprocess(_cute_dtype(config.dtype), config.dk, config.dv, TILE),
        main=main,
        dq=rounds(config.dk, pair, 1, dq_order),
        dk=rounds(config.dk, False, cluster, dkv_order),
        dv=rounds(config.dv, False, cluster, dkv_order),
        groups=config.groups,
        cluster=cluster,
        deterministic=deterministic,
    )


def _postprocess(dtype: Any, dim: int, pair: bool, cluster: int, order: Any) -> Any:
    """Return FA4's kernel that rounds a float32 gradient of head dim `dim`, written
    in `order` by the main kernel, as its interface configures it on SM100."""
    return _fa4().Postprocess(
        dtype,
        dim,
        _arch(),
        TILE,
        128,  # threads
        1,  # the MMA's atoms along M
        False,  # its operands not swapped
        use_2cta_instrs=pair,
        cluster_size=cluster,
        accum_order=order,
    )


def _cute_dtype(dtype: Any) -> Any:
    cutlass = _fa4().cutlass
    return cutlass.BFloat16 if dtype == jnp.bfloat16 else cutlass.Float16


@functools.cache
def _adapter(launch: Any) -> Any:
    """Return a launcher compiled by CuTe DSL, once."""
    return _fa4().cute.jit(launch)


# --- The launchers CuTe DSL compiles -------------------------------------------------


def _window(window: int | None) -> tuple[int | None, int | None]:
    """Return FA4's (left, right) of a window: the keys up to `window` - 1 behind a
    query, none ahead; (None, None) for none, under its causal mask."""
    return (None, None) if window is None else (window - 1, 0)


class Buffers(NamedTuple):
    """The backward launcher's buffers, as cutlass_call passes them."""

    q: Any
    k: Any
    v: Any
    out: Any
    dout: Any
    lse: Any
    cu: Any  #: cu_seqlens, or None for whole rows.
    zeroed: tuple  #: GQA's dK and dV in float32, then the semaphores, if any.
    dq: Any
    dk: Any
    dv: Any
    dq_accum: Any  #: dQ in float32.
    dpsum: Any  #: Each query's dO . O.
    lse_log2: Any  #: Each query's logsumexp in base 2.

    @classmethod
    def of(cls, buffers: tuple) -> "Buffers":
        """Return the buffers named: the inputs, the zeroed ones, the outputs."""
        zeroed = len(buffers) - (len(cls._fields) - 1)
        first = cls._fields.index("zeroed")
        ins, rest = buffers[:first], buffers[first:]
        return cls(*ins, tuple(rest[:zeroed]), *rest[zeroed:])


def _run_backward(
    stream: Any, x: Buffers, plan: BackwardPlan, scale: float, window: int | None
) -> None:
    """Launch FA4's backward kernels in turn."""
    f32 = _fa4().cutlass.Float32
    gqa = plan.groups > 1
    accums = x.zeroed[:2] if gqa else ()
    semaphores = x.zeroed[len(accums) :]
    dk, dv = accums or (x.dk, x.dv)
    semaphore = (*semaphores, None, None, None)
    left, right = _window(window)
    plan.pre(
        x.out, x.dout, x.dpsum, x.lse, x.lse_log2, x.dq_accum, x.cu,
        None, None, None, None, f32(1.0), stream=stream,
    )  # fmt: skip
    plan.main(
        x.q, x.k, x.v, x.dout, x.lse_log2, x.dpsum, x.dq_accum, dk, dv, f32(scale),
        x.cu, x.cu, window_size_left=left, window_size_right=right,
        mdQ_semaphore=semaphore[0], mdK_semaphore=semaphore[1],
        mdV_semaphore=semaphore[2], stream=stream,
    )  # fmt: skip
    plan.dq(x.dq_accum, x.dq, f32(scale), x.cu, None, None, stream=stream)
    if gqa:
        plan.dk(accums[0], x.dk, f32(scale), x.cu, None, None, stream=stream)
        plan.dv(accums[1], x.dv, f32(1.0), x.cu, None, None, stream=stream)


# CuTe DSL takes a launcher's annotations for its own types: these have none.


@no_type_check
def _launch_forward(stream, q, k, v, cu, out, lse, *, kernel, scale, window, rows):  # noqa: ANN001, ANN202
    """FA4's forward kernel on XLA's stream."""
    left, right = _window(window)
    scale = _fa4().cutlass.Float32(scale)
    kernel(q, k, v, out, lse, scale, cu, cu, window_size_left=left,
           window_size_right=right, max_seqlen_q=rows, stream=stream)  # fmt: skip


@no_type_check
def _launch_backward(stream, *buffers, plan, scale, window):  # noqa: ANN001, ANN002, ANN202
    """FA4's backward kernels on XLA's stream."""
    _run_backward(stream, Buffers.of(buffers), plan, scale, window)
