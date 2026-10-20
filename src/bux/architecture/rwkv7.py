"""RWKV-7 "Goose" time mixing (Peng et al., 2025), as the RWKV-LM reference has it.

Under tensor parallelism (``bux.parallel``) its projections and per-head parameters
split over `heads`, as linear attention's do, and the token shift stays whole."""

import functools
import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux import parallel
from bux.architecture.attention import _read, _write
from bux.architecture.linear_attention import (
    LowRank,
    depth,
    heads,
    l2norm,
    layer_input,
    recurrence,
)
from bux.architecture.norm import Norm
from bux.architecture.registry import Context, Taps, register, segments
from bux.architecture.segments import Segments
from bux.lang import ir


def _previous(x: Float[Array, "b t d"], docs: Segments | None) -> Float[Array, "b t d"]:
    """Every token's predecessor in its document, zero at a document's start."""
    prev = jnp.pad(x, ((0, 0), (1, 0), (0, 0)))[:, :-1]
    return prev if docs is None else jnp.where((docs.within > 0)[..., None], prev, 0)


class RWKV7(nnx.Module):
    """RWKV-7 time mixing. Every projection reads the token moved toward the one before
    it by a learned amount a channel (token shift). The state follows the generalized
    delta rule

        S_t = (diag(w_t) - (a_t kk_t) kk_t^T) S_{t-1} + k_t v_t^T,    o_t = S_t^T r_t

    with kk the normalized key it removes from, a the in-context learning rate, and
    log w = -e^-0.5 sigmoid(.) a key channel. Low-rank projections make w, a, the output
    gate g and the value residual's mix. The output is a layer norm of each head, plus a
    bonus (r k r_k) v a head, times g. Parameters start as RWKV-LM starts them, by the
    layer's depth among the model's token mixers.

    Ports: `v` written pushes the values; read moves the values toward the tap's by a
    learned amount a channel, from a projection of rank `v_rank`. That is RWKV-7's value
    residual, with the first layer's values: `rwkv7(v=@v0)` first, then `rwkv7(v=$v0)`.
    """

    def __init__(
        self,
        c: cfg.RWKV7Config,
        width: int,
        ports: Sequence[ir.Port],
        rngs: nnx.Rngs,
        *,
        layer: int = 0,
        layers: int = 1,
    ):
        if c.conv_size is not None:
            raise NotImplementedError("rwkv7 with a short conv (it shifts tokens)")
        h, hv = c.num_heads, c.num_v_heads or c.num_heads
        if hv % h:
            raise ValueError(f"num_v_heads={hv} is not a multiple of num_heads={h}")
        dk, dv = c.head_dim, c.v_head_dim or c.head_dim
        self.heads, self.v_heads, self.head_dim, self.v_head_dim = h, hv, dk, dv
        self.ports = tuple(ports)
        nk, nv = h * dk, hv * dv
        shallow = 1 - layer / layers  # 1 at the first layer, toward 0 at the last
        deep = layer / max(layers - 1, 1)  # 0 at the first layer, 1 at the last

        # Token shift, for r, w, k, v, a and g
        powers = jnp.asarray([0.2, 0.9, 0.7, 0.7, 0.9, 0.2])[:, None] * shallow
        self.shift = nnx.Param(1 - (jnp.arange(width) / width) ** powers)

        n, m = jnp.arange(nk), jnp.arange(nv)
        linear, v_linear = n / max(nk - 1, 1) - 0.5, m / max(nv - 1, 1) - 0.5
        zigzag = ((n % dk) - (dk - 1) / 2) / max((dk - 1) / 2, 1)
        zigzag = zigzag * jnp.abs(zigzag)
        decay = -6 + 6 * (n / max(nk - 1, 1)) ** (1 + deep**0.3)

        def dense(n_out: int, bound: float, n_heads: int) -> nnx.Linear:
            """A projection from the width to `n_heads` heads, uniform in
            +-bound / sqrt(width)."""
            init = functools.partial(
                jax.random.uniform,
                minval=-bound / math.sqrt(width),
                maxval=bound / math.sqrt(width),
            )
            init = parallel.named("embed", ("heads", n_heads), init=init)
            return nnx.Linear(width, n_out, use_bias=False, kernel_init=init, rngs=rngs)

        def lora(
            rank: int,
            n_out: int,
            bias: Float[Array, " n"] | bool,
            fn: str | None,
            n_heads: int,
        ) -> LowRank:
            return LowRank(
                width,
                rank,
                n_out,
                rngs,
                activation=fn,
                bias=bias,
                orthogonal=True,
                out=("heads", n_heads),
            )

        self.r, self.k = dense(nk, 0.5, h), dense(nk, 0.05, h)
        self.v = dense(nv, 0.5, hv)
        self.w = lora(c.decay_rank, nk, decay + 0.5 + zigzag * 2.5, "tanh", h)
        self.a = lora(c.a_rank, nk, -0.19 + zigzag * 0.3 + linear * 0.4, None, h)
        reads_v = any(p.name == "v" and p.mode != "write" for p in ports)
        self.v_mix = (
            lora(c.v_rank, nv, 0.73 - v_linear * 0.4, None, hv) if reads_v else None
        )
        self.g = (
            lora(c.gate_rank, nv, False, c.output_gate, hv)
            if c.output_gate is not None
            else None
        )
        self.k_k = nnx.Param(0.71 - linear * 0.1, names=(("heads", h),))
        self.k_a = nnx.Param(jnp.full(nk, 1.02), names=(("heads", h),))
        self.r_k = nnx.Param(jnp.full((hv, dk), -0.04), names=("heads", None))
        group = cfg.NormalizationConfig(
            type="layernorm", bias=True, eps=dv * c.norm_eps
        )
        self.norm = Norm(group, (hv, dv))
        zeros = parallel.named(("heads", hv), "embed", init=nnx.initializers.zeros)
        self.o = nnx.Linear(nv, width, use_bias=False, kernel_init=zeros, rngs=rngs)

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        x = layer_input(x)
        b, t, _ = x.shape
        h, hv, dk, dv = self.heads, self.v_heads, self.head_dim, self.v_head_dim
        docs = segments(taps)
        xr, xw, xk, xv, xa, xg = (
            x + (_previous(x, docs) - x) * self.shift[:, None, None]
        )
        r, k, v = self.r(xr), self.k(xk), heads(self.v(xv).reshape(b, t, hv, dv))
        if self.v_mix is not None:  # value residual
            mix = jax.nn.sigmoid(self.v_mix(xv)).reshape(v.shape)
            v = v + (_read(taps, self.ports, "v") - v) * mix
        _write(taps, self.ports, "v", v)
        log_w = -math.exp(-0.5) * jax.nn.sigmoid(self.w(xw))
        a = jax.nn.sigmoid(self.a(xa))
        kk = k * self.k_k
        k = k * (1 + (a - 1) * self.k_a)

        def split(y: Float[Array, "b t n"]) -> Float[Array, "b t h k"]:
            """Split into heads, each serving its group of value heads."""
            return jnp.repeat(heads(y.reshape(b, t, h, dk)), hv // h, axis=2)

        r, k, log_w, a, kk = map(split, (r, k, log_w, a, kk))
        kk = l2norm(kk, eps=1e-12)
        o = heads(recurrence(r, k, v, log_w, docs, a=-kk, b=kk * a))
        o = self.norm(o) + (r * k * self.r_k).sum(-1, keepdims=True) * v
        o = o.reshape(b, t, -1)
        if self.g is not None:
            o = o * self.g(xg)
        return self.o(o)


@register("rwkv7")
def _(
    c: cfg.RWKV7Config, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> RWKV7:
    layer, layers = depth(ctx, "rwkv7")
    return RWKV7(c, width, ports, ctx.rngs, layer=layer, layers=layers)
