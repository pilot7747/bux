"""State space mixers: Mamba as in Jamba, Mamba-2, and LFM2's gated short convolution.

Under tensor parallelism (``bux.parallel``) their channels split, Mamba's and the short
conv's over `mlp` and Mamba-2's over `heads`, and so does every parameter of a channel
or a head: the scan runs on each device's own. Their input projections are fused (u
and the gate side by side), so they stay whole, and every device computes them. The
selective scan runs by the run's `methods.recurrence`, as linear attention does, and
Mamba-2's by linear attention's chunks of positions (``chunkwise``).
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux import parallel
from bux.architecture.common.segments import Segments
from bux.architecture.common.sharding import projection, shard_layer_input
from bux.architecture.layers.conv import CausalConv
from bux.architecture.layers.norm import Norm
from bux.architecture.recurrent import chunkwise
from bux.architecture.registry import Context, Taps, register, segments
from bux.architecture.repeats import remat
from bux.lang import ir
from bux.parallel.recurrence import Chunk, Positions, Recurrence, Shared


def selective_scan(
    x: Float[Array, "b t h p"],
    dt: Float[Array, "b t h"],
    A: Float[Array, " h #n"],
    B: Float[Array, "b t g n"],
    C: Float[Array, "b t g n"],
    docs: Segments | None = None,
) -> Float[Array, "b t h p"]:
    """Run the selective scan. Each of the h heads keeps an n by p state S, and at
    every position

        S = exp(dt A) S + B (dt x)^T,    y = S^T C,

    with the heads in g groups that share B and C, as kv heads share keys in GQA. Mamba
    has one head per channel (p = 1) and one A per channel and state dim, and runs one
    position at a time; Mamba-2 has one A per head (n = 1 in A), and runs by chunks
    (`chunk`). The state starts at zero, and again where a document starts. It runs by
    the run's `methods.recurrence`, in float32, the outputs in x's dtype, named
    `recurrence` (``remat.named``)."""
    b, t = x.shape[:2]
    within = jnp.ones((b, t), jnp.int32) if docs is None else docs.within
    keep = (within > 0).astype(jnp.float32)  # 0 where a document starts
    run = parallel.method("recurrence")

    def scan(xs: Positions, shared: Shared) -> Float[Array, "b t h p"]:
        return run(Recurrence("the selective scan", chunk, xs, shared))

    y = remat.named("recurrence", scan, (x, dt, B, C, keep), (A,))
    return y.astype(x.dtype)


def chunk(xs: Positions, shared: Shared, transition: bool) -> Chunk:
    """Run `selective_scan` over a chunk of positions from a zero state, in float32,
    its state n by p per head, as keys by values. With `transition`, also return the
    product P of every position's exp(dt A), and each output's reads P C.

    With one A per head, as Mamba-2's, it is decayed linear attention with q = C and
    k = B, a group's, v = dt x and a log decay of dt A a head: by smaller chunks, with
    matmuls within each (``chunkwise``). Mamba's, one A per channel and state dim,
    runs `stepwise`."""
    if shared[0].shape[-1] > 1:
        return stepwise(xs, shared, transition)
    x, dt, B, C, keep = (a.astype(jnp.float32) for a in xs)
    log_decay = dt[..., None] * shared[0].astype(jnp.float32)
    return chunkwise.chunk((C, B, dt[..., None] * x, log_decay, keep), transition)


_At = tuple[
    Float[Array, "b h p"],
    Float[Array, "b h"],
    Float[Array, "b g n"],
    Float[Array, "b g n"],
    Float[Array, " b"],
]
_Carry = tuple[Float[Array, "b h n p"], Float[Array, "b h #n"] | None]  # S, and P


def _step(
    A: Float[Array, " h #n"], carry: _Carry, at: _At
) -> tuple[_Carry, tuple[Float[Array, "b h p"], Float[Array, "b h n"] | None]]:
    state, P = carry
    x, dt, B, C, keep = at
    per_group = x.shape[1] // B.shape[1]
    B, C = jnp.repeat(B, per_group, axis=1), jnp.repeat(C, per_group, axis=1)
    decay = jnp.exp(dt[..., None] * A) * keep[:, None, None]
    state = decay[..., None] * state + B[..., None] * (dt[..., None] * x)[:, :, None]
    y = jnp.einsum("bhnp,bhn->bhp", state, C)
    if P is None:
        return (state, None), (y, None)
    P = decay * P
    return (state, P), (y, P * C)


def stepwise(xs: Positions, shared: Shared, transition: bool) -> Chunk:
    """Run `chunk` one position at a time: a scan in float32, Mamba-2's reference."""
    x, dt, B, C, keep = (jnp.moveaxis(a.astype(jnp.float32), 1, 0) for a in xs)
    A = shared[0].astype(jnp.float32)
    _, b, h, p = x.shape
    # Of x's kind, which in a shard_map varies where it does.
    zero = jnp.zeros_like(x, shape=(b, h, B.shape[-1], p))
    P = jnp.ones_like(x, shape=(b, h, A.shape[-1])) if transition else None
    (state, P), (y, reads) = jax.lax.scan(
        partial(_step, A), (zero, P), (x, dt, B, C, keep)
    )
    y = jnp.moveaxis(y, 0, 1)
    return Chunk(y, state, P, None if reads is None else jnp.moveaxis(reads, 0, 1))


def _dt_bias(
    config: cfg.StateSpaceConfig, size: int, rngs: nnx.Rngs
) -> Float[Array, " size"]:
    """Return the time step bias as Mamba initializes it: softplus^-1 of steps
    log-uniform between dt_min and dt_max, floored at 1e-4."""
    low, high = math.log(config.dt_min), math.log(config.dt_max)
    dt = jnp.exp(jax.random.uniform(rngs.params(), (size,), minval=low, maxval=high))
    dt = jnp.maximum(dt, 1e-4)
    return dt + jnp.log(-jnp.expm1(-dt))


def _out(
    n: int, width: int, name: parallel.Name, bias: bool, rngs: nnx.Rngs
) -> nnx.Linear:
    """Make the projection of n channels named `name` back to the width."""
    init = parallel.named(name, "embed")
    return projection(
        n, width, "recurrence_out", use_bias=bias, kernel_init=init, rngs=rngs
    )


def shard_fused(y: Float[Array, "b t c"]) -> Float[Array, "b t c"]:
    """Keep a fused projection's output whole over its channels, whose parts do not
    split on shard boundaries; its rows and positions split as inside a layer."""
    return parallel.shard(y, "batch", "seq", None)


def shard_channels(
    y: Float[Array, "b t c"], name: parallel.Name
) -> Float[Array, "b t c"]:
    """Shard the channels an output projection takes as `name` says, so that it
    reduces where they are, and the stream's split does not reach back."""
    return parallel.shard(y, "batch", "seq", name)


class Mamba(nnx.Module):
    """Mamba's selective SSM (Gu & Dao, 2023) as Jamba has it, with the time steps, B
    and C RMS-normalized.

    The input projects to u and a gate z, both `expand` times the width. u is convolved
    over positions and SiLU'd, and projects to B and C, which all its channels share,
    and to a time step of rank dt_rank, projected up to one per channel. Every channel
    keeps a state of state_size, which decays at its own rate in each state dim. With
    D u added, the output is gated by SiLU(z) and projected back.
    """

    def __init__(self, config: cfg.MambaConfig, width: int, rngs: nnx.Rngs):
        d_inner, n = config.expand * width, config.state_size
        r = config.dt_rank or math.ceil(width / 16)
        self.rank, self.state_size = r, n
        self.in_proj = projection(
            width, 2 * d_inner, "recurrence", use_bias=False, rngs=rngs
        )
        self.conv = CausalConv(d_inner, config.conv_size, True, rngs, channel="mlp")
        self.x_proj = projection(
            d_inner,
            r + 2 * n,
            "recurrence",
            use_bias=False,
            kernel_init=parallel.named("mlp", None),
            rngs=rngs,
        )
        dt_proj = jax.random.uniform(
            rngs.params(), (r, d_inner), minval=-(r**-0.5), maxval=r**-0.5
        )
        self.dt_proj = nnx.Param(dt_proj, names=(None, "mlp"))
        self.dt_bias = nnx.Param(_dt_bias(config, d_inner, rngs), names=("mlp",))
        norm = cfg.NormalizationConfig(type="rmsnorm", eps=1e-6)
        self.dt_norm = Norm(norm, r)
        self.b_norm, self.c_norm = Norm(norm, n), Norm(norm, n)
        # S4D-real: A = -(1, 2, ..., n) in every channel.
        A_log = jnp.tile(jnp.log(jnp.arange(1.0, n + 1)), (d_inner, 1))
        self.A_log = nnx.Param(A_log, names=("mlp", None))
        self.D = nnx.Param(jnp.ones(d_inner), names=("mlp",))
        self.out_proj = _out(d_inner, width, "mlp", False, rngs)

    def __call__(
        self, x: Float[Array, "b t d"], taps: Taps | None = None
    ) -> Float[Array, "b t d"]:
        u, z = jnp.split(shard_fused(self.in_proj(shard_layer_input(x))), 2, axis=-1)
        u = jax.nn.silu(self.conv(u, taps))
        r, n = self.rank, self.state_size
        dt, B, C = jnp.split(self.x_proj(u), [r, r + n], axis=-1)
        dt = jax.nn.softplus(self.dt_norm(dt) @ self.dt_proj[...] + self.dt_bias[...])
        B, C = self.b_norm(B)[:, :, None], self.c_norm(C)[:, :, None]
        A = -jnp.exp(self.A_log[...].astype(jnp.float32))
        y = selective_scan(u[..., None], dt, A, B, C, segments(taps))[..., 0]
        y = (y + u * self.D[...]) * jax.nn.silu(z)
        return self.out_proj(shard_channels(y, "mlp"))


_HEADS = ("batch", "seq", "heads")


class Mamba2(nnx.Module):
    """Mamba-2 (Dao & Gu, 2024), as transformers' Mamba-2, Zamba2 and Nemotron-H
    mixers compute it.

    The input projects to a gate z, to u, B and C, which are convolved over positions
    together and SiLU'd, and to one time step per head. u splits into heads of
    head_dim, whose states decay by one rate per head, and B and C into num_groups
    groups that the heads share. With D u added, the output is gated by SiLU(z),
    RMS-normalized one group of heads at a time (as Mamba-2's reference, Zamba2 and
    Nemotron-H group it) and projected back.
    """

    def __init__(self, config: cfg.Mamba2Config, width: int, rngs: nnx.Rngs):
        d_inner, n, g = config.expand * width, config.state_size, config.num_groups
        h = d_inner // config.head_dim
        if h * config.head_dim != d_inner or h % g:
            raise ValueError(
                f"mamba2 splits {d_inner} channels into heads of {config.head_dim}, "
                f"and those into {g} groups"
            )
        self.inner, self.heads, self.head_dim = d_inner, h, config.head_dim
        self.groups, self.state_size = g, n
        self.in_proj = projection(
            width, 2 * d_inner + 2 * g * n + h, "recurrence", use_bias=False, rngs=rngs
        )
        # Over u, B and C side by side, as the projection gives them: whole.
        self.conv = CausalConv(d_inner + 2 * g * n, config.conv_size, True, rngs)
        self.dt_bias = nnx.Param(_dt_bias(config, h, rngs), names=("heads",))
        # A = -a, with a uniform in [1, 16] per head (Mamba-2's A_init_range).
        a = jax.random.uniform(rngs.params(), (h,), minval=1.0, maxval=16.0)
        self.A_log = nnx.Param(jnp.log(a), names=("heads",))
        self.D = nnx.Param(jnp.ones(h), names=("heads",))
        norm = cfg.NormalizationConfig(type="rmsnorm", eps=1e-5)
        self.norm = Norm(norm, (g, d_inner // g))
        self.out_proj = _out(d_inner, width, ("heads", h), False, rngs)

    def __call__(
        self, x: Float[Array, "b t d"], taps: Taps | None = None
    ) -> Float[Array, "b t d"]:
        b, t, _ = x.shape
        d_inner, g, n = self.inner, self.groups, self.state_size
        x = shard_layer_input(x)
        proj = shard_fused(self.in_proj(x))
        z, ubc, dt = jnp.split(proj, [d_inner, 2 * d_inner + 2 * g * n], axis=-1)
        ubc = jax.nn.silu(self.conv(ubc, taps))
        u, B, C = jnp.split(ubc, [d_inner, d_inner + g * n], axis=-1)
        u = parallel.shard(u.reshape(b, t, self.heads, self.head_dim), *_HEADS, None)
        B, C = B.reshape(b, t, g, n), C.reshape(b, t, g, n)
        dt = parallel.shard(jax.nn.softplus(dt + self.dt_bias[...]), *_HEADS)
        A = -jnp.exp(self.A_log[...].astype(jnp.float32))[:, None]
        y = selective_scan(u, dt, A, B, C, segments(taps)) + u * self.D[...][:, None]
        y = y.reshape(b, t, d_inner) * jax.nn.silu(z)
        y = self.norm(y.reshape(b, t, g, -1)).reshape(b, t, d_inner)
        return self.out_proj(shard_channels(y, ("heads", self.heads)))


class ShortConv(nnx.Module):
    """LFM2's double-gated short convolution: the input projects to B, C and u, and
    the output is C times the causal convolution of B u, projected back."""

    def __init__(self, config: cfg.ShortConvConfig, width: int, rngs: nnx.Rngs):
        bias = config.bias
        self.in_proj = projection(
            width, 3 * width, "recurrence", use_bias=bias, rngs=rngs
        )
        self.conv = CausalConv(width, config.conv_size, bias, rngs, channel="mlp")
        self.out_proj = _out(width, width, "mlp", bias, rngs)

    def __call__(
        self, x: Float[Array, "b t d"], taps: Taps | None = None
    ) -> Float[Array, "b t d"]:
        B, C, u = jnp.split(shard_fused(self.in_proj(shard_layer_input(x))), 3, axis=-1)
        return self.out_proj(shard_channels(C * self.conv(B * u, taps), "mlp"))


@register("mamba")
def _(
    c: cfg.MambaConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Mamba:
    return Mamba(c, width, ctx.rngs)


@register("mamba2")
def _(
    c: cfg.Mamba2Config, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Mamba2:
    return Mamba2(c, width, ctx.rngs)


@register("short_conv")
def _(
    c: cfg.ShortConvConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> ShortConv:
    return ShortConv(c, width, ctx.rngs)
