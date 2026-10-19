"""Causal grouped-query attention."""

import jax
import jax.numpy as jnp
from flax import nnx

from bux import config as cfg
from bux.architecture.conv import CausalConv
from bux.architecture.functional import activation, softcap
from bux.architecture.norm import Norm
from bux.architecture.registry import Context, Taps, register
from bux.architecture.rope import logit_gain, rotate
from bux.lang import ir


def _read(taps: Taps, port: ir.Port):
    stack = taps[port.tap]
    return stack.pop() if port.mode == "pop" else stack[-1]


class Attention(nnx.Module):
    """Causal grouped-query attention.

    Ports: `v` written pushes the values, read mixes the tap's values in with a learned
    weight (value residual); `kv` written pushes (k, v), read uses them instead of
    projecting its own (cross-layer KV sharing).
    """

    def __init__(self, c: cfg.GQAConfig, width: int, ports, rngs: nnx.Rngs):
        if c.indexer is not None or any(p.name == "index" for p in ports):
            raise NotImplementedError("sparse attention indexer")
        h, kv = c.num_heads, c.num_kv_heads
        d = c.head_dim or width // h
        dv = c.v_head_dim or d
        self.heads, self.kv_heads, self.head_dim, self.v_head_dim = h, kv, d, dv
        self.window, self.softcap = c.sliding_window_size, c.softcap
        self.scale = (c.scale or d**-0.5) * logit_gain(c.rope)
        self.log_scale, self.log_scale_floor = c.log_scale, c.log_scale_floor
        self.rope = c.rope
        self.ports = tuple(ports)
        reads_kv = any(p.name == "kv" and p.mode != "write" for p in ports)
        reads_v = any(p.name == "v" and p.mode != "write" for p in ports)

        def linear(n_in: int, n_out: int) -> nnx.Linear:
            return nnx.Linear(n_in, n_out, use_bias=c.bias, rngs=rngs)

        self.q = linear(width, h * d)
        self.k = None if reads_kv else linear(width, kv * d)
        self.v = None if reads_kv else linear(width, kv * dv)
        self.o = linear(h * dv, width)
        conv = c.kv_conv_size if not reads_kv else None
        self.k_conv = CausalConv(kv * d, conv, False, rngs) if conv else None
        self.v_conv = CausalConv(kv * dv, conv, False, rngs) if conv else None
        self.gate = linear(width, h if c.gate_per_head else h * dv) if c.gated else None
        self.gate_activation = c.gate_activation
        self.out_norm = Norm(c.output_norm, dv) if c.output_norm else None
        self.q_norm = Norm(c.qk_norm, d) if c.qk_norm else None
        self.k_norm = Norm(c.qk_norm, d) if c.qk_norm and not reads_kv else None
        self.sinks = nnx.Param(jnp.zeros(h)) if c.sinks else None
        self.v_mix = nnx.Param(jnp.asarray(0.5)) if reads_v else None
        rel = c.relative_bias
        self.rel_proj = linear(width, h * rel.rank) if rel else None
        self.rel_table = nnx.Param(jnp.zeros((rel.rank, rel.extent))) if rel else None

    def __call__(self, x, taps: Taps):
        b, t, _ = x.shape
        ports = {(p.name, p.mode == "write"): p for p in self.ports}
        q = self.q(x).reshape(b, t, self.heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k is None or self.v is None:
            k, v = _read(taps, ports["kv", False])
        else:
            k, v = self.k(x), self.v(x)
            if self.k_conv is not None and self.v_conv is not None:
                k, v = k + self.k_conv(k), v + self.v_conv(v)
            k = k.reshape(b, t, self.kv_heads, self.head_dim)
            v = v.reshape(b, t, self.kv_heads, self.v_head_dim)
            if self.k_norm is not None:
                k = self.k_norm(k)
            if self.rope is not None:
                k = rotate(k, self.rope)
        if self.rope is not None:
            q = rotate(q, self.rope)
        if self.v_mix is not None:
            v = self.v_mix * v + (1 - self.v_mix) * _read(taps, ports["v", False])
        if (port := ports.get(("v", True))) is not None:
            taps.setdefault(port.tap, []).append(v)
        if (port := ports.get(("kv", True))) is not None:
            taps.setdefault(port.tap, []).append((k, v))

        group = self.heads // self.kv_heads
        k, v = jnp.repeat(k, group, axis=2), jnp.repeat(v, group, axis=2)
        logits = jnp.einsum("bthd,bshd->bhts", q, k) * self.scale
        pos = jnp.arange(t)
        if self.rel_proj is not None and self.rel_table is not None:
            r = self.rel_proj(x).reshape(b, t, self.heads, -1)
            offset = pos[:, None] - pos[None, :]
            extent = self.rel_table.shape[-1]
            table = self.rel_table[...][:, jnp.clip(offset, 0, extent - 1)]
            near = (offset >= 0) & (offset < extent)
            logits = logits + jnp.einsum("bthr,rts->bhts", r, table) * near
        if self.log_scale is not None:
            ahead = jnp.maximum((pos + 1) / self.log_scale_floor, 1.0)
            logits = logits * (1 + self.log_scale * jnp.log(ahead))[:, None]
        logits = softcap(logits, self.softcap)
        mask = pos[:, None] >= pos[None, :]
        if self.window is not None:
            mask &= pos[:, None] - pos[None, :] < self.window
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
        if self.sinks is not None:
            sink = jnp.broadcast_to(
                self.sinks[None, :, None, None], (b, self.heads, t, 1)
            )
            probs = jax.nn.softmax(jnp.concatenate([logits, sink], -1))[..., :-1]
        else:
            probs = jax.nn.softmax(logits)
        out = jnp.einsum("bhts,bshd->bthd", probs, v)
        if self.out_norm is not None:
            out = self.out_norm(out)
        if self.gate is not None:
            gate = activation(self.gate_activation, self.gate(x))
            out = out * gate.reshape(b, t, self.heads, -1)
        return self.o(out.reshape(b, t, -1))


@register("gqa")
def _(c: cfg.GQAConfig, width, ports, ctx: Context):
    return Attention(c, width, ports, ctx.rngs)
