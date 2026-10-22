"""The gated delta rule by cuDNN's fused KDA kernel (Frost, in cudnn-frontend, which
tokamax's cuda extra installs): forward and backward, 13 times as fast as
``chunkwise`` on a B300 at KDA's shapes.

It factors the decays through blocks of positions, so it overflows where a log decay
falls below -5: it runs a layer whose gate is bounded there, KDA's with
`gate_lower_bound` (Kimi K3), and no other. It runs on one SM100 or SM103 GPU at a
time, on token-packed rows and their documents' bounds (cu_seqlens); on a mesh, a
`shard_map` gives it each device's rows and heads. It takes heads of 64 or 128, and
every position of a row: not `methods.recurrence = pass_state`, which splits them.
"""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from bux import parallel
from bux.architecture.segments import Segments

try:  # cudnn-frontend's JAX kernels, where the cuda extra is installed (Linux)
    from cudnn.linear_attention import jax_api as frost
except ImportError:
    frost = None

DIMS = (64, 128)  # the head dims the kernel takes
SM = ("10.0", "10.3")  # the GPUs it runs on: B200, B300
FLOOR = -5.0  # the least log decay it takes at any position
# Documents a row of 4096 tokens may hold for the kernel: an empty sequence costs as
# much as 10 tokens, so cu_seqlens has room for this many, and more go by `otherwise`.
DOCS = 32

Rule = Callable[..., Float[Array, "b t h dv"]]  # linear_attention's delta_rule


def runs(dk: int, dv: int, floor: float | None) -> bool:
    """Return whether the kernel runs a layer of these head dims whose log decays are
    never below `floor`, by the run's methods, here."""
    on = parallel.active()
    method = on.methods.recurrence if on else "scan"
    bounded = floor is not None and floor >= FLOOR
    shapes = dk in DIMS and dv in DIMS
    return frost is not None and bounded and method == "scan" and shapes and _gpu()


@functools.cache
def _gpu() -> bool:
    """Return whether the default device is a GPU the kernel runs on."""
    device = jax.devices()[0]
    return device.platform == "gpu" and getattr(device, "compute_capability", "") in SM


def delta_rule(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    log_decay: Float[Array, "b t h dk"],
    beta: Float[Array, "b t h"],
    docs: Segments | None,
    otherwise: Rule,
) -> Float[Array, "b t h dv"]:
    """Run `linear_attention.delta_rule` by the kernel, in v's dtype, where `runs`;
    by `otherwise` for rows of more documents than cu_seqlens holds."""
    docs = docs or Segments.of(jnp.broadcast_to(jnp.arange(q.shape[1]), q.shape[:2]))
    xs = (q, k, v, log_decay, beta, *docs)
    if parallel.active() is None:
        return _rows(otherwise, *xs)
    rows = parallel.spec(("batch", "seq", "heads", None), q.shape)
    if rows[1] is not None:  # positions split over devices, which XLA gathers
        return otherwise(q, k, v, log_decay, beta, docs)
    specs = (rows,) * 4 + (P(*rows[:3]),) + (P(*rows[:2]),) * 3
    # Untyped by what varies, which the kernel's outputs do not say; nothing here
    # communicates.
    rule = functools.partial(_rows, otherwise)
    return jax.shard_map(rule, in_specs=specs, out_specs=rows, check_vma=False)(*xs)


def _rows(
    otherwise: Rule,
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    g: Float[Array, "b t h dk"],
    beta: Float[Array, "b t h"],
    positions: Int[Array, "b t"],
    ids: Int[Array, "b t"],
    within: Int[Array, "b t"],
) -> Float[Array, "b t h dv"]:
    """Run one device's rows by the kernel if cu_seqlens holds their documents."""
    b, t = within.shape
    room = b * max(1, t * DOCS // 4096)
    fits = jnp.sum(within == 0) <= room
    docs = Segments(positions, ids, within)

    def fallback(
        q: Float[Array, "b t h dk"],
        k: Float[Array, "b t h dk"],
        v: Float[Array, "b t h dv"],
        g: Float[Array, "b t h dk"],
        beta: Float[Array, "b t h"],
        within: Int[Array, "b t"],
    ) -> Float[Array, "b t h dv"]:
        return otherwise(q, k, v, g, beta, docs).astype(v.dtype)

    kernel = functools.partial(_kernel, room=room)
    return jax.lax.cond(fits, kernel, fallback, q, k, v, g, beta, within)


def _kernel(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    g: Float[Array, "b t h dk"],
    beta: Float[Array, "b t h"],
    within: Int[Array, "b t"],
    *,
    room: int,
) -> Float[Array, "b t h dv"]:
    """Run the kernel on the rows packed end to end, each document a sequence, in
    `room` sequences, the last ones empty."""
    assert frost is not None
    b, t, h, _ = q.shape
    dtype = v.dtype if v.dtype in (jnp.bfloat16, jnp.float32) else jnp.float32
    q, k, v = (x.reshape(b * t, *x.shape[2:]).astype(dtype) for x in (q, k, v))
    starts = jnp.nonzero(within.reshape(-1) == 0, size=room, fill_value=b * t)[0]
    cu = jnp.append(starts, b * t).astype(jnp.int32)
    gates = g.reshape(b * t, h, -1).astype(jnp.float32)
    steps = beta.reshape(b * t, h).astype(jnp.float32)
    out, _ = frost.kimi_delta_attention(q, k, v, gates, steps, cu, scale=1.0)
    return out.reshape(b, t, h, -1)
