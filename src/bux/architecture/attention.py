"""Causal grouped-query attention."""

from collections.abc import Sequence

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


def _read(taps: Taps, ports: tuple[ir.Port, ...], name: str):
    """What the layer reads (or pops) from its port `name`."""
    port = next(p for p in ports if p.name == name and p.mode != "write")
    stack = taps[port.tap]
    return stack.pop() if port.mode == "pop" else stack[-1]


def _write(taps: Taps, ports: tuple[ir.Port, ...], name: str, value) -> None:
    """Push `value` to the tap of port `name`, if the layer writes that port."""
    for port in ports:
        if port.name == name and port.mode == "write":
            taps.setdefault(port.tap, []).append(value)


def _relative_bias(r: jax.Array, table: jax.Array, offset: jax.Array) -> jax.Array:
    """The (b, h, t, s) bias <r[b, t, h], table[:, t - s]>, from r of (b, t, h, rank) and
    a table of (rank, extent); keys ahead of the query, or extent or more behind, get none.
    """
    extent = table.shape[-1]
    rows = table[:, jnp.clip(offset, 0, extent - 1)]
    near = (offset >= 0) & (offset < extent)
    return jnp.einsum("bthr,rts->bhts", r, rows) * near


def _length_scale(t: int, log_scale: float, floor: int) -> jax.Array:
    """The (t, 1) factor on each query's logits: 1 + log_scale * ln(max(n / floor, 1)),
    at position n counted from 1."""
    ahead = jnp.maximum(jnp.arange(1, t + 1) / floor, 1.0)
    return (1 + log_scale * jnp.log(ahead))[:, None]


def _softmax_with_sinks(logits: jax.Array, sinks: jax.Array) -> jax.Array:
    """Softmax over the keys of (b, h, t, s) logits, next to a per-head sink logit that
    takes a share of the probability and attends to nothing."""
    b, h, t, _ = logits.shape
    sink = jnp.broadcast_to(sinks[None, :, None, None], (b, h, t, 1))
    return jax.nn.softmax(jnp.concatenate([logits, sink], -1))[..., :-1]


class Attention(nnx.Module):
    """Causal grouped-query attention.

    Ports: `v` written pushes the values, read mixes the tap's values in with a learned
    weight (value residual); `kv` written pushes (k, v), read uses them instead of
    projecting its own (cross-layer KV sharing).
    """

    def __init__(
        self, c: cfg.GQAConfig, width: int, ports: Sequence[ir.Port], rngs: nnx.Rngs
    ):
        if c.indexer is not None or any(p.name == "index" for p in ports):
            raise NotImplementedError("sparse attention indexer")
        h, kv = c.num_heads, c.num_kv_heads
        d = c.head_dim or width // h
        dv = c.v_head_dim or d
        self.heads, self.kv_heads, self.head_dim, self.v_head_dim = h, kv, d, dv
        self.ports = tuple(ports)
        reads = {p.name for p in ports if p.mode != "write"}
        own_kv = "kv" not in reads  # else the kv tap's keys and values replace its own

        def linear(n_in: int, n_out: int) -> nnx.Linear:
            return nnx.Linear(n_in, n_out, use_bias=c.bias, rngs=rngs)

        # The randomly initialized layers; reordering them changes a seed's init.
        self.q = linear(width, h * d)
        self.k = linear(width, kv * d) if own_kv else None
        self.v = linear(width, kv * dv) if own_kv else None
        self.o = linear(h * dv, width)
        conv = c.kv_conv_size if own_kv else None
        self.k_conv = CausalConv(kv * d, conv, False, rngs) if conv else None
        self.v_conv = CausalConv(kv * dv, conv, False, rngs) if conv else None
        self.gate = linear(width, h if c.gate_per_head else h * dv) if c.gated else None
        rel = c.relative_bias
        self.rel_proj = linear(width, h * rel.rank) if rel else None

        # Queries, keys and values
        self.q_norm = Norm(c.qk_norm, d) if c.qk_norm else None
        self.k_norm = Norm(c.qk_norm, d) if c.qk_norm and own_kv else None
        self.rope = c.rope
        self.v_mix = nnx.Param(jnp.asarray(0.5)) if "v" in reads else None

        # Logits and probabilities
        self.scale = (c.scale or d**-0.5) * logit_gain(c.rope)
        self.rel_table = nnx.Param(jnp.zeros((rel.rank, rel.extent))) if rel else None
        self.log_scale, self.log_scale_floor = c.log_scale, c.log_scale_floor
        self.softcap = c.softcap
        self.window = c.sliding_window_size
        self.sinks = nnx.Param(jnp.zeros(h)) if c.sinks else None

        # Output
        self.out_norm = Norm(c.output_norm, dv) if c.output_norm else None
        self.gate_activation = c.gate_activation

    def __call__(self, x, taps: Taps):
        q = self._queries(x)
        k, v = self._keys_and_values(x, taps)
        if self.v_mix is not None:  # value residual
            v = self.v_mix * v + (1 - self.v_mix) * _read(taps, self.ports, "v")
        _write(taps, self.ports, "v", v)
        _write(taps, self.ports, "kv", (k, v))
        logits = self._logits(x, q, k)
        if self.sinks is not None:
            probs = _softmax_with_sinks(logits, self.sinks[...])
        else:
            probs = jax.nn.softmax(logits)
        out = jnp.einsum("bhts,bshd->bthd", probs, self._per_query_head(v))
        return self._output(x, out)

    def _queries(self, x) -> jax.Array:
        """The (b, t, heads, head_dim) queries."""
        b, t, _ = x.shape
        q = self.q(x).reshape(b, t, self.heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.rope is not None:
            q = rotate(q, self.rope)
        return q

    def _keys_and_values(self, x, taps: Taps) -> tuple[jax.Array, jax.Array]:
        """The (b, t, kv_heads, head_dim) keys and (b, t, kv_heads, v_head_dim) values,
        the layer's own or those it reads from its kv port."""
        if self.k is None or self.v is None:
            return _read(taps, self.ports, "kv")
        b, t, _ = x.shape
        k, v = self.k(x), self.v(x)
        if self.k_conv is not None and self.v_conv is not None:
            k, v = k + self.k_conv(k), v + self.v_conv(v)
        k = k.reshape(b, t, self.kv_heads, self.head_dim)
        v = v.reshape(b, t, self.kv_heads, self.v_head_dim)
        if self.k_norm is not None:
            k = self.k_norm(k)
        if self.rope is not None:
            k = rotate(k, self.rope)
        return k, v

    def _per_query_head(self, kv: jax.Array) -> jax.Array:
        """Keys or values repeated for every query head of their group."""
        return jnp.repeat(kv, self.heads // self.kv_heads, axis=2)

    def _logits(self, x, q: jax.Array, k: jax.Array) -> jax.Array:
        """The (b, heads, t, s) logits, at the dtype's minimum for keys a query does not
        see."""
        b, t, _ = x.shape
        pos = jnp.arange(t)
        offset = pos[:, None] - pos[None, :]  # how far each key is behind each query
        logits = jnp.einsum("bthd,bshd->bhts", q, self._per_query_head(k)) * self.scale
        if self.rel_proj is not None and self.rel_table is not None:
            r = self.rel_proj(x).reshape(b, t, self.heads, -1)
            logits = logits + _relative_bias(r, self.rel_table[...], offset)
        if self.log_scale is not None:
            logits = logits * _length_scale(t, self.log_scale, self.log_scale_floor)
        logits = softcap(logits, self.softcap)
        visible = offset >= 0
        if self.window is not None:
            visible &= offset < self.window
        return jnp.where(visible, logits, jnp.finfo(logits.dtype).min)

    def _output(self, x, out: jax.Array) -> jax.Array:
        """The (b, t, heads, v_head_dim) head outputs, normalized, gated and projected
        back to the width."""
        b, t, _ = x.shape
        if self.out_norm is not None:
            out = self.out_norm(out)
        if self.gate is not None:
            gate = activation(self.gate_activation, self.gate(x))
            out = out * gate.reshape(b, t, self.heads, -1)
        return self.o(out.reshape(b, t, -1))


@register("gqa")
def _(c: cfg.GQAConfig, width, ports, ctx: Context):
    return Attention(c, width, ports, ctx.rngs)
