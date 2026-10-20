"""State space mixers: Mamba as in Jamba, Mamba-2, and LFM2's gated short convolution.

Under tensor parallelism (``bux.parallel``) their channels split, Mamba's and the short
conv's over `mlp` and Mamba-2's over `heads`, and so does every parameter of a channel
or a head: the scan runs on each device's own. Their input projections are fused (u
and the gate side by side), so they stay whole, and every device computes them.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux import parallel
from bux.architecture.conv import CausalConv
from bux.architecture.norm import Norm
from bux.architecture.registry import Context, Taps, register, segments
from bux.architecture.segments import Segments
from bux.lang import ir


def selective_scan(
    x: Float[Array, "b t h p"],
    dt: Float[Array, "b t h"],
    A: Float[Array, " h #n"],
    B: Float[Array, "b t g n"],
    C: Float[Array, "b t g n"],
    docs: Segments | None = None,
) -> Float[Array, "b t h p"]:
    """The selective scan, a position at a time. Each of the h heads keeps a p by n
    state S, and at every position

        S = exp(dt A) S + dt x B^T,    y = S C,

    with the heads in g groups that share B and C, as kv heads share keys in GQA. Mamba
    has a head a channel (p = 1) and an A a channel and state dim; Mamba-2 has an A a
    head (n = 1 in A). The state starts at zero, and again where a document starts."""
    b, t, h, p = x.shape
    n, per_group = B.shape[-1], h // B.shape[2]
    within = jnp.ones((b, t), jnp.int32) if docs is None else docs.within
    keep = (within > 0).astype(x.dtype)  # 0 where a document starts

    def step(
        state: Float[Array, "b h p n"],
        at: tuple[
            Float[Array, "b h p"],
            Float[Array, "b h"],
            Float[Array, "b g n"],
            Float[Array, "b g n"],
            Float[Array, " b"],
        ],
    ) -> tuple[Float[Array, "b h p n"], Float[Array, "b h p"]]:
        x, dt, B, C, keep = at
        B, C = jnp.repeat(B, per_group, axis=1), jnp.repeat(C, per_group, axis=1)
        decay = jnp.exp(dt[..., None] * A) * keep[:, None, None]
        state = (
            decay[:, :, None] * state + (dt[..., None] * x)[..., None] * B[:, :, None]
        )
        return state, jnp.einsum("bhpn,bhn->bhp", state, C)

    time_major = jax.tree.map(lambda a: jnp.moveaxis(a, 1, 0), (x, dt, B, C, keep))
    _, y = jax.lax.scan(step, jnp.zeros((b, h, p, n), x.dtype), time_major)
    return jnp.moveaxis(y, 0, 1)


def _dt_bias(
    c: cfg.StateSpaceConfig, size: int, rngs: nnx.Rngs
) -> Float[Array, " size"]:
    """The time step bias as Mamba initializes it: softplus^-1 of steps log-uniform
    between dt_min and dt_max, floored at 1e-4."""
    low, high = math.log(c.dt_min), math.log(c.dt_max)
    dt = jnp.exp(jax.random.uniform(rngs.params(), (size,), minval=low, maxval=high))
    dt = jnp.maximum(dt, 1e-4)
    return dt + jnp.log(-jnp.expm1(-dt))


def _out(
    n: int, width: int, name: parallel.Name, bias: bool, rngs: nnx.Rngs
) -> nnx.Linear:
    """The projection of n channels named `name` back to the width."""
    init = parallel.named(name, "embed")
    return nnx.Linear(n, width, use_bias=bias, kernel_init=init, rngs=rngs)


def _whole(y: Float[Array, "b t n"]) -> Float[Array, "b t n"]:
    """`y` whole but for its rows and positions, split as inside a layer (`seq`): a
    layer's input, which under sequence parallelism is gathered before the
    projections, not their weights; and what a fused projection gives, whose parts do
    not split on shard boundaries."""
    return parallel.shard(y, "batch", "seq", None)


def _channels(y: Float[Array, "b t e"], name: parallel.Name) -> Float[Array, "b t e"]:
    """`y`, its channels split as `name` says: what an output projection takes, so
    that it reduces where they are, and the stream's split does not reach back."""
    return parallel.shard(y, "batch", "seq", name)


class Mamba(nnx.Module):
    """Mamba's selective SSM (Gu & Dao, 2023) as Jamba has it, with the time steps, B
    and C RMS-normalized.

    The input projects to u and a gate z, both `expand` times the width. u is convolved
    over positions and SiLU'd, and projects to B and C, which all its channels share,
    and to a time step of rank dt_rank, projected up to one a channel. Every channel
    keeps a state of state_size, which decays at its own rate in each state dim. With
    D u added, the output is gated by SiLU(z) and projected back.
    """

    def __init__(self, c: cfg.MambaConfig, width: int, rngs: nnx.Rngs):
        e, n = c.expand * width, c.state_size
        r = c.dt_rank or math.ceil(width / 16)
        self.rank, self.state_size = r, n
        self.in_proj = nnx.Linear(width, 2 * e, use_bias=False, rngs=rngs)
        self.conv = CausalConv(e, c.conv_size, True, rngs, channel="mlp")
        self.x_proj = nnx.Linear(
            e,
            r + 2 * n,
            use_bias=False,
            kernel_init=parallel.named("mlp", None),
            rngs=rngs,
        )
        dt_proj = jax.random.uniform(
            rngs.params(), (r, e), minval=-(r**-0.5), maxval=r**-0.5
        )
        self.dt_proj = nnx.Param(dt_proj, names=(None, "mlp"))
        self.dt_bias = nnx.Param(_dt_bias(c, e, rngs), names=("mlp",))
        norm = cfg.NormalizationConfig(type="rmsnorm", eps=1e-6)
        self.dt_norm = Norm(norm, r)
        self.b_norm, self.c_norm = Norm(norm, n), Norm(norm, n)
        # S4D-real: A = -(1, 2, ..., n) in every channel.
        A_log = jnp.tile(jnp.log(jnp.arange(1.0, n + 1)), (e, 1))
        self.A_log = nnx.Param(A_log, names=("mlp", None))
        self.D = nnx.Param(jnp.ones(e), names=("mlp",))
        self.out_proj = _out(e, width, "mlp", False, rngs)

    def __call__(
        self, x: Float[Array, "b t d"], taps: Taps | None = None
    ) -> Float[Array, "b t d"]:
        u, z = jnp.split(_whole(self.in_proj(_whole(x))), 2, axis=-1)
        u = jax.nn.silu(self.conv(u, taps))
        r, n = self.rank, self.state_size
        dt, B, C = jnp.split(self.x_proj(u), [r, r + n], axis=-1)
        dt = jax.nn.softplus(self.dt_norm(dt) @ self.dt_proj[...] + self.dt_bias[...])
        B, C = self.b_norm(B)[:, :, None], self.c_norm(C)[:, :, None]
        A = -jnp.exp(self.A_log[...])
        y = selective_scan(u[..., None], dt, A, B, C, segments(taps))[..., 0]
        return self.out_proj(_channels((y + u * self.D[...]) * jax.nn.silu(z), "mlp"))


_HEADS = ("batch", "seq", "heads")


class Mamba2(nnx.Module):
    """Mamba-2 (Dao & Gu, 2024), as transformers' Mamba-2, Zamba2 and Nemotron-H
    mixers compute it.

    The input projects to a gate z, to u, B and C, which are convolved over positions
    together and SiLU'd, and to a time step a head. u splits into heads of head_dim,
    whose states decay by one rate a head, and B and C into num_groups groups that the
    heads share. With D u added, the output is gated by SiLU(z), RMS-normalized a group
    of heads at a time (as Mamba-2's reference, Zamba2 and Nemotron-H group it) and
    projected back.
    """

    def __init__(self, c: cfg.Mamba2Config, width: int, rngs: nnx.Rngs):
        e, n, g = c.expand * width, c.state_size, c.num_groups
        h = e // c.head_dim
        if h * c.head_dim != e or h % g:
            raise ValueError(
                f"mamba2 splits {e} channels into heads of {c.head_dim}, and those "
                f"into {g} groups"
            )
        self.inner, self.heads, self.head_dim = e, h, c.head_dim
        self.groups, self.state_size = g, n
        self.in_proj = nnx.Linear(
            width, 2 * e + 2 * g * n + h, use_bias=False, rngs=rngs
        )
        # Over u, B and C side by side, as the projection gives them: whole.
        self.conv = CausalConv(e + 2 * g * n, c.conv_size, True, rngs)
        self.dt_bias = nnx.Param(_dt_bias(c, h, rngs), names=("heads",))
        # A = -a, with a uniform in [1, 16] a head (Mamba-2's A_init_range).
        a = jax.random.uniform(rngs.params(), (h,), minval=1.0, maxval=16.0)
        self.A_log = nnx.Param(jnp.log(a), names=("heads",))
        self.D = nnx.Param(jnp.ones(h), names=("heads",))
        norm = cfg.NormalizationConfig(type="rmsnorm", eps=1e-5)
        self.norm = Norm(norm, (g, e // g))
        self.out_proj = _out(e, width, ("heads", h), False, rngs)

    def __call__(
        self, x: Float[Array, "b t d"], taps: Taps | None = None
    ) -> Float[Array, "b t d"]:
        b, t, _ = x.shape
        e, g, n = self.inner, self.groups, self.state_size
        x = _whole(x)
        proj = _whole(self.in_proj(x))
        z, ubc, dt = jnp.split(proj, [e, 2 * e + 2 * g * n], axis=-1)
        u, B, C = jnp.split(jax.nn.silu(self.conv(ubc, taps)), [e, e + g * n], axis=-1)
        u = parallel.shard(u.reshape(b, t, self.heads, self.head_dim), *_HEADS, None)
        B, C = B.reshape(b, t, g, n), C.reshape(b, t, g, n)
        dt = parallel.shard(jax.nn.softplus(dt + self.dt_bias[...]), *_HEADS)
        A = -jnp.exp(self.A_log[...])[:, None]
        y = selective_scan(u, dt, A, B, C, segments(taps)) + u * self.D[...][:, None]
        y = y.reshape(b, t, e) * jax.nn.silu(z)
        y = self.norm(y.reshape(b, t, g, -1)).reshape(b, t, e)
        return self.out_proj(_channels(y, ("heads", self.heads)))


class ShortConv(nnx.Module):
    """LFM2's double-gated short convolution: the input projects to B, C and u, and
    the output is C times the causal convolution of B u, projected back."""

    def __init__(self, c: cfg.ShortConvConfig, width: int, rngs: nnx.Rngs):
        self.in_proj = nnx.Linear(width, 3 * width, use_bias=c.bias, rngs=rngs)
        self.conv = CausalConv(width, c.conv_size, c.bias, rngs, channel="mlp")
        self.out_proj = _out(width, width, "mlp", c.bias, rngs)

    def __call__(
        self, x: Float[Array, "b t d"], taps: Taps | None = None
    ) -> Float[Array, "b t d"]:
        B, C, u = jnp.split(_whole(self.in_proj(_whole(x))), 3, axis=-1)
        return self.out_proj(_channels(C * self.conv(B * u, taps), "mlp"))


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
