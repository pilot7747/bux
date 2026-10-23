"""Flash attention kernels for the methods of `methods.attention`.

A `Kernel` attends to one block of keys. Its forward pass returns each query's
logsumexp beside its output, and its backward pass takes one: given the output and
logsumexp of a larger softmax (more blocks of keys, or a sink beside them), it returns
that softmax's gradients through its own block, as FlashAttention-2's backward pass
recomputes P = exp(s - lse). So blocks merge by their logsumexps (`merge`), and a sink
folds into one (`attend`).

`TOKAMAX` is tokamax's Mosaic kernels, on GPUs; `JNP` the same in jnp, all t by s
logits at once, which a CPU runs and the tests check the rest by. `cudnn` is cuDNN's
fused attention through jax, for whole rows: no softcap, no sinks, no logsumexp.
"""

from collections.abc import Callable
from functools import cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from bux import kernels

try:  # tokamax's kernels, where its extra is installed (Linux)
    from tokamax._src.ops.attention import api as tokamax
    from tokamax._src.ops.attention import base as masks

    MOSAIC = tokamax.IMPLEMENTATIONS.get("mosaic_gpu")
except ImportError:
    MOSAIC = masks = None

F32 = jnp.float32


class Keys(NamedTuple):
    """Which keys of a block each query sees: none ahead of it if `causal` (the
    queries' own positions), none before `start` and none from `end` on, each query's
    index into the block's keys, if any."""

    causal: bool
    start: Int[Array, "b t"] | None = None
    end: Int[Array, "b t"] | None = None


Output = tuple[Float[Array, "b t h dv"], Float[Array, "b h t"]]
Grads = tuple[
    Float[Array, "b t h dk"], Float[Array, "b s g dk"], Float[Array, "b s g dv"]
]


class Kernel(NamedTuple):
    """`forward(q, k, v, keys, scale, cap) -> (out, lse)`, lse in float32 and -inf for
    a query that sees no key; `backward(q, k, v, out, dout, lse, keys, scale, cap)
    -> (dq, dk, dv)`. `cap` soft-caps the logits, if not None."""

    forward: Callable[..., Output]
    backward: Callable[..., Grads]


# --- jnp ----------------------------------------------------------------------------


def _seen(keys: Keys, t: int, s: int) -> Bool[Array, "#b 1 t s"]:
    """Return which of s keys each of t queries sees, by `keys`."""
    index = jnp.arange(s)
    seen = jnp.ones((1, 1, t, s), bool)
    if keys.causal:
        seen &= index[None, :] <= jnp.arange(t)[:, None]
    if keys.start is not None:
        seen = seen & (index >= keys.start[..., None])[:, None]
    if keys.end is not None:
        seen = seen & (index < keys.end[..., None])[:, None]
    return seen


def _logits(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    scale: float,
    cap: float | None,
) -> tuple[Float[Array, "b h t s"], Float[Array, "b h t s"]]:
    """Return the logits in float32, before the soft cap and after it."""
    k = jnp.repeat(k.astype(F32), q.shape[2] // k.shape[2], axis=2)
    raw = jnp.einsum("bthd,bshd->bhts", q.astype(F32), k) * scale
    return raw, raw if cap is None else cap * jnp.tanh(raw / cap)


def _jnp_forward(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    keys: Keys,
    scale: float,
    cap: float | None,
) -> Output:
    seen = _seen(keys, q.shape[1], k.shape[1])
    _, logits = _logits(q, k, scale, cap)
    lse = jax.nn.logsumexp(jnp.where(seen, logits, -jnp.inf), -1)
    p = jnp.where(seen, jnp.exp(logits - _finite(lse)[..., None]), 0)
    v = jnp.repeat(v.astype(F32), q.shape[2] // v.shape[2], axis=2)
    return jnp.einsum("bhts,bshd->bthd", p, v).astype(q.dtype), lse


def _jnp_backward(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    out: Float[Array, "b t h dv"],
    dout: Float[Array, "b t h dv"],
    lse: Float[Array, "b h t"],
    keys: Keys,
    scale: float,
    cap: float | None,
) -> Grads:
    g, r = k.shape[2], q.shape[2] // k.shape[2]
    raw, logits = _logits(q, k, scale, cap)
    seen = _seen(keys, q.shape[1], k.shape[1])
    p = jnp.where(seen, jnp.exp(logits - _finite(lse)[..., None]), 0)
    do, kk = dout.astype(F32), jnp.repeat(k.astype(F32), r, axis=2)
    dp = jnp.einsum("bthd,bshd->bhts", do, jnp.repeat(v.astype(F32), r, axis=2))
    ds = p * (dp - jnp.einsum("bthd,bthd->bht", do, out.astype(F32))[..., None])
    if cap is not None:
        ds = ds * (1 - jnp.tanh(raw / cap) ** 2)
    ds = ds * scale
    dq = jnp.einsum("bhts,bshd->bthd", ds, kk)
    dk = jnp.einsum("bhts,bthd->bshd", ds, q.astype(F32))
    dv = jnp.einsum("bhts,bthd->bshd", p, do)
    group = partial(_sum_groups, g=g)
    return dq.astype(q.dtype), group(dk).astype(k.dtype), group(dv).astype(v.dtype)


def _sum_groups(x: Float[Array, "b s h e"], g: int) -> Float[Array, "b s g e"]:
    """Sum the gradients of a kv head over the query heads of its group."""
    b, s, h, e = x.shape
    return x.reshape(b, s, g, h // g, e).sum(3)


def _finite(lse: Float[Array, "*s"]) -> Float[Array, "*s"]:
    """`lse` with 0 for a query that sees no key: its probabilities are 0 anyway."""
    return jnp.where(jnp.isfinite(lse), lse, 0.0)


JNP = Kernel(_jnp_forward, _jnp_backward)


# --- tokamax ------------------------------------------------------------------------


def tokamax_runs() -> bool:
    """Whether tokamax's Mosaic kernels run here: on a GPU (``bux.kernels``), with
    tokamax installed."""
    return MOSAIC is not None and kernels.gpu()


def _tokamax_args(keys: Keys, scale: float, cap: float | None) -> dict:
    assert masks is not None
    per_head = [None if x is None else x[:, None] for x in (keys.start, keys.end)]
    mask = masks.Mask(k_start=per_head[0], k_end=per_head[1], is_causal=keys.causal)
    return {
        "mask": mask,
        "logits_scale": scale,
        "logits_soft_cap": cap,
        "logits_dtype": F32,
    }


def _tokamax_forward(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    keys: Keys,
    scale: float,
    cap: float | None,
) -> Output:
    """tokamax's residuals are the row's largest logit m and sum of exp(s - m)."""
    assert MOSAIC is not None
    args = _tokamax_args(keys, scale, cap)
    out, (m, total) = MOSAIC(q, k, v, return_residuals=True, **args)
    some = total > 0
    return out, jnp.where(some, m + jnp.log(jnp.where(some, total, 1)), -jnp.inf)


def _tokamax_backward(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    out: Float[Array, "b t h dv"],
    dout: Float[Array, "b t h dv"],
    lse: Float[Array, "b h t"],
    keys: Keys,
    scale: float,
    cap: float | None,
) -> Grads:
    """tokamax's VJP, given the residuals (lse, 1): P = exp(s - lse)."""
    assert MOSAIC is not None
    lse = _finite(lse)
    bound = MOSAIC.bind(q, k, v, **_tokamax_args(keys, scale, cap))
    kwargs = {n: a for n, a in bound.kwargs.items() if n != "return_residuals"}
    grads = MOSAIC.vjp((lse, jnp.ones_like(lse)), out, dout, *bound.args, **kwargs)
    return grads["q"], grads["k"], grads["v"]


TOKAMAX = Kernel(_tokamax_forward, _tokamax_backward)


# --- sinks, and merging blocks --------------------------------------------------------


def attend(
    kernel: Kernel,
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    keys: Keys,
    sinks: Float[Array, " h"] | None,
    scale: float,
    cap: float | None,
) -> Float[Array, "b t h dv"]:
    """Return each query's softmax-weighted values of the `keys` it sees, beside its
    head's sink logit if any, by `kernel`: the sink folds into the logsumexp, with
    which the backward pass runs."""
    static = (kernel, keys.causal, scale, cap)
    return _attend(q, k, v, keys.start, keys.end, sinks, static)


@partial(jax.custom_vjp, nondiff_argnums=(6,))
def _attend(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    start: Int[Array, "b t"] | None,
    end: Int[Array, "b t"] | None,
    sinks: Float[Array, " h"] | None,
    static: tuple[Kernel, bool, float, float | None],
) -> Float[Array, "b t h dv"]:
    return _attend_fwd(q, k, v, start, end, sinks, static)[0]


def _attend_fwd(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    start: Int[Array, "b t"] | None,
    end: Int[Array, "b t"] | None,
    sinks: Float[Array, " h"] | None,
    static: tuple[Kernel, bool, float, float | None],
) -> tuple[Float[Array, "b t h dv"], tuple]:
    kernel, causal, scale, cap = static
    out, lse = kernel.forward(q, k, v, Keys(causal, start, end), scale, cap)
    out, lse = with_sinks(out.astype(F32), lse, sinks)
    out = out.astype(q.dtype)
    return out, (q, k, v, start, end, sinks, out, lse)


def _attend_bwd(
    static: tuple[Kernel, bool, float, float | None],
    saved: tuple,
    dout: Float[Array, "b t h dv"],
) -> tuple:
    kernel, causal, scale, cap = static
    q, k, v, start, end, sinks, out, lse = saved
    keys = Keys(causal, start, end)
    dq, dk, dv = kernel.backward(q, k, v, out, dout, lse, keys, scale, cap)
    return dq, dk, dv, None, None, sink_grads(sinks, lse, out, dout)


_attend.defvjp(_attend_fwd, _attend_bwd)


def _share(
    lse: Float[Array, "b h t"], total: Float[Array, "b h t"]
) -> Float[Array, "b t h 1"]:
    """Return the share exp(lse - total) of a part of a softmax, 0 where neither sees
    a key."""
    share = jnp.where(jnp.isneginf(total), 0.0, jnp.exp(lse - _finite(total)))
    return share.transpose(0, 2, 1)[..., None]


def merge(
    out: Float[Array, "b t h dv"],
    lse: Float[Array, "b h t"],
    block_out: Float[Array, "b t h dv"],
    block_lse: Float[Array, "b h t"],
) -> tuple[Float[Array, "b t h dv"], Float[Array, "b h t"]]:
    """Return the output, in float32, and the logsumexp of the softmax over the keys
    of two blocks, from each one's."""
    total = jnp.logaddexp(lse, block_lse)
    out = out * _share(lse, total) + block_out.astype(F32) * _share(block_lse, total)
    return out, total


def with_sinks(
    out: Float[Array, "b t h dv"],
    lse: Float[Array, "b h t"],
    sinks: Float[Array, " h"] | None,
) -> tuple[Float[Array, "b t h dv"], Float[Array, "b h t"]]:
    """Return the output and logsumexp with each head's sink logit beside the keys: a
    sink takes a share of the probability and attends to nothing."""
    if sinks is None:
        return out, lse
    total = jnp.logaddexp(lse, sinks.astype(F32)[None, :, None])
    return out * _share(lse, total), total


def sink_grads(
    sinks: Float[Array, " h"] | None,
    lse: Float[Array, "b h t"],
    out: Float[Array, "b t h dv"],
    dout: Float[Array, "b t h dv"],
) -> Float[Array, " h"] | None:
    """Return the sinks' gradient: a sink takes exp(sink - lse) of each query's
    probability, so its gradient is minus that times dout . out, summed over the
    queries. A shard_map sums it over the devices that share the sinks."""
    if sinks is None:
        return None
    delta = jnp.einsum("bthd,bthd->bht", dout.astype(F32), out.astype(F32))
    share = jnp.exp(sinks.astype(F32)[None, :, None] - lse)
    return -(share * delta).sum((0, 2)).astype(sinks.dtype)


# --- cuDNN --------------------------------------------------------------------------


def cudnn_runs() -> bool:
    """Whether cuDNN's fused attention runs here: a GPU of compute capability 9 or
    more (the packed layout's), and cuDNN; outside ``bux.kernels.off``."""
    return kernels.gpu() and _cudnn_here()


@cache
def _cudnn_here() -> bool:
    if jax.default_backend() != "gpu":
        return False
    from jax._src.cudnn import fused_attention_stablehlo as fused

    try:
        fused.check_cudnn_version()
    except RuntimeError:
        return False
    return float(getattr(jax.devices()[0], "compute_capability", 0)) >= 9


def cudnn_takes(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    cap: float | None,
    sinks: Float[Array, " h"] | None,
) -> bool:
    """Whether `cudnn` takes the call: in bfloat16 or float16, keys' and values' head
    dims equal (on a B300 its packed layout gets MLA's wrong) and up to 256 in steps
    of 8, no soft cap and no sinks, which cuDNN's attention in jax lacks."""
    d = q.shape[-1]
    return (
        cap is None
        and sinks is None
        and q.dtype in (jnp.bfloat16, jnp.float16)
        and q.dtype == k.dtype == v.dtype
        and d == v.shape[-1] <= 256
        and d % 8 == 0
        and cudnn_runs()
    )


def cudnn(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t g dk"],
    v: Float[Array, "b t g dv"],
    ids: Int[Array, "b t"] | None,
    window: int | None,
    scale: float,
    documents: int,
) -> Float[Array, "b t h dv"]:
    """Return cuDNN's causal attention of whole rows, within `window`, and each of up
    to `documents` documents of a row (by `ids`) by itself, in cuDNN's packed layout:
    cuDNN skips what no query sees, forward and backward."""
    from jax._src.cudnn import fused_attention_stablehlo as fused

    lengths = starts = None
    mask = fused.MaskType.CAUSAL
    if ids is not None:
        lengths, starts = packed_documents(ids, documents)
        mask = fused.MaskType.PADDING_CAUSAL
    return fused.dot_product_attention(
        q,
        k,
        v,
        q_seqlen=lengths,
        kv_seqlen=lengths,
        q_offsets=starts,
        kv_offsets=starts,
        scale=scale,
        mask_type=mask,
        sliding_window_length=window,
    )


def packed_documents(
    ids: Int[Array, "b t"], documents: int
) -> tuple[Int[Array, "b m"], Int[Array, "b m1"]]:
    """Return the length of each of a row's first `documents` documents, and where
    each starts and then -1 (cuDNN's q_offsets: starts only, no end), -1 past the
    row's last document."""
    b, t = ids.shape
    new = jnp.concatenate([jnp.ones((b, 1), bool), ids[:, 1:] != ids[:, :-1]], 1)
    nonzero = partial(jnp.nonzero, size=documents, fill_value=-1)
    starts = jax.vmap(lambda row: nonzero(row)[0])(new).astype(jnp.int32)
    after = jnp.concatenate([starts[:, 1:], jnp.full((b, 1), -1, jnp.int32)], 1)
    lengths = jnp.where(starts < 0, -1, jnp.where(after < 0, t, after) - starts)
    return lengths, jnp.concatenate([starts, jnp.full((b, 1), -1, jnp.int32)], 1)


def documents_in(ids: Int[Array, "b t"]) -> Int[Array, ""]:
    """Return the most documents a row of `ids` has."""
    return (1 + (ids[:, 1:] != ids[:, :-1]).sum(1)).max()
