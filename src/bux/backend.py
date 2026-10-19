"""Reference Flax NNX backend: turns an Architecture into a runnable nnx.Module.

Every IR node kind maps to one small module, so the graph side is complete. Leaves
come from a registry keyed by config type; this file registers the dense layers and
grouped-query attention. Other mixers plug in the same way:

    @register("gdn")
    def _(config: GatedDeltaNetConfig, width: int, ports: tuple[Port, ...], ctx: Context):
        return MyGatedDeltaNet(config, width, rngs=ctx.rngs)

Modules are called as ``module(x, taps)``: ``taps`` holds the tap stacks for one
forward pass, and under reserved keys the input tokens and the logits of every head
run so far. Repeats are unrolled; nodes with a share key are built once and reused.
"""

import functools
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from bux import config as cfg
from bux.lang import ir

Taps = dict[str, list]
# Reserved tap keys, which no tap name can collide with.
TOKENS, OUTPUTS, NEXTN = "<tokens>", "<outputs>", "<nextn>"


@dataclass
class Context:
    arch: ir.Architecture
    rngs: nnx.Rngs
    shared: dict[str, nnx.Module] = field(default_factory=dict)
    embedding: Embedding | None = None
    engrams: int = 0  # engram layers built so far
    primes: set[int] = field(default_factory=set)  # engram table sizes taken


Builder = Callable[..., nnx.Module]
_REGISTRY: dict[str, Builder] = {}


def register(*types: str) -> Callable[[Builder], Builder]:
    def decorate(builder: Builder) -> Builder:
        for t in types:
            _REGISTRY[t] = builder
        return builder

    return decorate


def _activation(name: str, x: jax.Array) -> jax.Array:
    match name:
        case "relu":
            return jax.nn.relu(x)
        case "relu2":
            return jnp.square(jax.nn.relu(x))
        case "gelu":
            return jax.nn.gelu(x, approximate=False)
        case "gelu_tanh":
            return jax.nn.gelu(x, approximate=True)
        case "swish":
            return jax.nn.swish(x)
        case "sigmoid":
            return jax.nn.sigmoid(x)
        case "tanh":
            return jnp.tanh(x)
        case "softplus":
            return jax.nn.softplus(x)
        case "situ":
            return _softcap(x, 4.0) * jax.nn.sigmoid(x)
    raise ValueError(name)


def _softcap(x: jax.Array, cap: float | None) -> jax.Array:
    return x if cap is None else cap * jnp.tanh(x / cap)


# --- Graph nodes ------------------------------------------------------------------


class Sequence(nnx.Module):
    def __init__(self, items: list[nnx.Module]):
        self.items = nnx.List(items)

    def __call__(self, x, taps: Taps):
        for item in self.items:
            x = item(x, taps)
        return x


class ResidualBranch(nnx.Module):
    def __init__(self, body: nnx.Module):
        self.body = body

    def __call__(self, x, taps: Taps):
        return x + self.body(x, taps)


class Parallel(nnx.Module):
    def __init__(self, branches: list[nnx.Module], op: str):
        self.branches = nnx.List(branches)
        self.op = op

    def __call__(self, x, taps: Taps):
        outs = [branch(x, taps) for branch in self.branches]
        return functools.reduce(
            operator.add if self.op == "sum" else operator.mul, outs
        )


class Negate(nnx.Module):
    def __init__(self, body: nnx.Module):
        self.body = body

    def __call__(self, x, taps: Taps):
        return -self.body(x, taps)


class Constant(nnx.Module):
    def __init__(self, value: float):
        self.value = value

    def __call__(self, x, taps: Taps):
        return jnp.asarray(self.value, dtype=x.dtype)


class Identity(nnx.Module):
    def __call__(self, x, taps: Taps):
        return x


class TapWrite(nnx.Module):
    def __init__(self, name: str):
        self.name = name

    def __call__(self, x, taps: Taps):
        taps.setdefault(self.name, []).append(x)
        return x


class TapRead(nnx.Module):
    def __init__(self, name: str, pop: bool):
        self.name = name
        self.pop = pop

    def __call__(self, x, taps: Taps):
        stack = taps[self.name]
        return stack.pop() if self.pop else stack[-1]


def build_node(node: ir.Node, ctx: Context) -> nnx.Module:
    if node.share is not None:
        if node.share not in ctx.shared:
            ctx.shared[node.share] = _build(node, ctx)
        return ctx.shared[node.share]
    return _build(node, ctx)


def _build(node: ir.Node, ctx: Context) -> nnx.Module:
    match node:
        case ir.Layer(config=config, width=width, ports=ports):
            builder = _REGISTRY.get(config.type)
            if builder is None:
                raise NotImplementedError(
                    f"the reference backend has no '{config.type}' layer yet; "
                    f"add one with @bux.backend.register({config.type!r})"
                )
            return builder(config, width, ports, ctx)
        case ir.Seq(items=items):
            return Sequence([build_node(item, ctx) for item in items])
        case ir.Residual(body=body):
            return ResidualBranch(build_node(body, ctx))
        case ir.Sum(terms=terms):
            return Parallel([build_node(t, ctx) for t in terms], "sum")
        case ir.Prod(factors=factors):
            return Parallel([build_node(f, ctx) for f in factors], "prod")
        case ir.Neg(body=body):
            return Negate(build_node(body, ctx))
        case ir.Const(value=value):
            return Constant(value)
        case ir.Identity():
            return Identity()
        case ir.Repeat(body=body, times=times):
            return Sequence([build_node(body, ctx) for _ in range(times)])
        case ir.Tap(name=name):
            return TapWrite(name)
        case ir.Ref(name=name, pop=pop):
            return TapRead(name, pop)
    raise AssertionError(node)


class Model(nnx.Module):
    """tokens (batch, seq) -> logits (batch, seq, vocab).

    Every head is an output. ``outputs`` returns them all in execution order; head j
    predicts the token 1 + k positions ahead, where k counts the nextn layers run before it.
    """

    def __init__(self, arch: ir.Architecture, *, rngs: nnx.Rngs):
        if arch.residual is not None:
            raise NotImplementedError(
                "hyper-connections are not in the reference backend yet"
            )
        self.graph = build_node(arch.graph, Context(arch, rngs))

    def outputs(self, tokens: jax.Array) -> list[jax.Array]:
        taps: Taps = {TOKENS: [tokens]}
        last = self.graph(tokens, taps)
        return taps.get(OUTPUTS) or [last]

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """The first head's logits: the next-token prediction."""
        return self.outputs(tokens)[0]


def num_params(module: nnx.Module) -> int:
    return sum(x.size for x in jax.tree.leaves(nnx.state(module, nnx.Param)))


# --- Leaves ---------------------------------------------------------------------


def _read(taps: Taps, port: ir.Port):
    stack = taps[port.tap]
    return stack.pop() if port.mode == "pop" else stack[-1]


class Embedding(nnx.Module):
    def __init__(self, c: cfg.EmbeddingConfig, vocab: int, d: int, rngs: nnx.Rngs):
        self.tokens = nnx.Embed(vocab, d, rngs=rngs)
        self.positions = (
            nnx.Embed(c.max_positions, d, rngs=rngs) if c.max_positions else None
        )
        self.multiplier = d**0.5 if c.scale_by_sqrt_dim else 1.0
        self.norm = Norm(c.norm, d) if c.norm is not None else None
        self.dropout = nnx.Dropout(c.dropout, rngs=rngs)

    def embed_tokens(self, tokens):
        """The token embedding, scaled and normalized: what nextn sees."""
        x = self.tokens(tokens) * self.multiplier
        return x if self.norm is None else self.norm(x)

    def __call__(self, tokens, taps: Taps):
        x = self.embed_tokens(tokens)
        if self.positions is not None:
            x = x + self.positions(jnp.arange(tokens.shape[-1]))
        return self.dropout(x)


@register("embed")
def _(c: cfg.EmbeddingConfig, width, ports, ctx: Context):
    assert ctx.arch.vocab_size is not None
    embedding = Embedding(c, ctx.arch.vocab_size, ctx.arch.d_model, ctx.rngs)
    if ctx.embedding is None:
        ctx.embedding = embedding
    return embedding


class Head(nnx.Module):
    def __init__(self, c: cfg.LMHeadConfig, width: int, ctx: Context):
        vocab = ctx.arch.vocab_size
        assert vocab is not None
        self.embedding = ctx.embedding if c.tied else None
        self.proj = (
            None if c.tied else nnx.Linear(width, vocab, use_bias=c.bias, rngs=ctx.rngs)
        )
        self.bias = nnx.Param(jnp.zeros(vocab)) if c.tied and c.bias else None
        self.softcap = c.softcap

    def __call__(self, x, taps: Taps):
        if self.embedding is not None:
            logits = self.embedding.tokens.attend(x)
        else:
            assert self.proj is not None
            logits = self.proj(x)
        if self.bias is not None:
            logits = logits + self.bias
        logits = _softcap(logits, self.softcap)
        taps.setdefault(OUTPUTS, []).append(logits)
        return logits


@register("head")
def _(c: cfg.LMHeadConfig, width, ports, ctx: Context):
    return Head(c, width, ctx)


class NextN(nnx.Module):
    """Mixes the stream with the embedding of the token k positions ahead, where this
    is the k-th nextn to run in the forward pass (shared modules run more than once)."""

    def __init__(self, c: cfg.NextNConfig, width: int, ctx: Context):
        assert ctx.embedding is not None
        self.embedding = ctx.embedding
        self.hidden_norm = Norm(c.norm, width)
        self.embed_norm = Norm(c.norm, ctx.arch.d_model)
        self.proj = nnx.Linear(
            ctx.arch.d_model + width, width, use_bias=False, rngs=ctx.rngs
        )

    def __call__(self, x, taps: Taps):
        runs = taps.setdefault(NEXTN, [])
        runs.append(None)
        k, tokens = len(runs), taps[TOKENS][0]
        # The last k positions have no token that far ahead; pad them with 0.
        ahead = jnp.pad(tokens[:, k:], ((0, 0), (0, k)))
        e = self.embed_norm(self.embedding.embed_tokens(ahead))
        return self.proj(jnp.concatenate([e, self.hidden_norm(x)], -1))


@register("nextn")
def _(c: cfg.NextNConfig, width, ports, ctx: Context):
    return NextN(c, width, ctx)


class Norm(nnx.Module):
    def __init__(self, c: cfg.NormalizationConfig, width: int | tuple[int, ...]):
        self.kind = c.type
        self.eps = c.eps
        self.zero_centered = c.zero_centered
        self.scale_input = c.scale_input
        init = jnp.zeros if c.zero_centered else jnp.ones
        self.scale = nnx.Param(init(width)) if c.affine else None
        self.bias = nnx.Param(jnp.zeros(width)) if c.bias else None

    def __call__(self, x, taps: Taps | None = None):
        gain = None
        if self.scale is not None:
            gain = 1 + self.scale if self.zero_centered else self.scale
        if gain is not None and self.scale_input:
            x = x * gain
        if self.kind == "layernorm":
            x = x - x.mean(-1, keepdims=True)
        x = x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)
        if gain is not None and not self.scale_input:
            x = x * gain
        if self.bias is not None:
            x = x + self.bias
        return x


@register("layernorm", "rmsnorm")
def _(c: cfg.NormalizationConfig, width, ports, ctx: Context):
    return Norm(c, width)


class DyT(nnx.Module):
    def __init__(self, c: cfg.DyTConfig, width: int):
        self.alpha = nnx.Param(jnp.asarray(c.alpha_init))
        self.weight = nnx.Param(jnp.ones(width))
        self.bias = nnx.Param(jnp.zeros(width))

    def __call__(self, x, taps: Taps):
        return jnp.tanh(self.alpha * x) * self.weight + self.bias


@register("dyt")
def _(c: cfg.DyTConfig, width, ports, ctx: Context):
    return DyT(c, width)


class MLP(nnx.Module):
    def __init__(self, c: cfg.MLPConfig, width: int, rngs: nnx.Rngs, size: int = 0):
        size = size or c.intermediate_size
        self.up = nnx.Linear(width, size, use_bias=c.bias, rngs=rngs)
        self.gate = (
            nnx.Linear(width, size, use_bias=c.bias, rngs=rngs)
            if c.type == "gated_mlp"
            else None
        )
        self.down = nnx.Linear(size, width, use_bias=c.bias, rngs=rngs)
        self.activation = c.activation
        self.limit, self.up_softcap = c.limit, c.up_softcap
        self.dropout = nnx.Dropout(c.dropout, rngs=rngs)

    def __call__(self, x, taps: Taps | None = None):
        up = _bound(self.up(x), self.limit, self.up_softcap)
        if self.gate is not None:
            gate = self.gate(x)
            if self.limit is not None:
                gate = jnp.minimum(gate, self.limit)
            h = _activation(self.activation, gate) * up
        else:
            h = _activation(self.activation, up)
        return self.dropout(self.down(h))


def _bound(x: jax.Array, limit: float | None, softcap: float | None) -> jax.Array:
    """The up branch of a GLU, clamped to +-limit and/or tanh soft-capped."""
    if limit is not None:
        x = jnp.clip(x, -limit, limit)
    return _softcap(x, softcap)


@register("mlp", "gated_mlp")
def _(c: cfg.MLPConfig, width, ports, ctx: Context):
    return MLP(c, width, ctx.rngs)


class MoE(nnx.Module):
    """Dense reference: every expert runs on every token, weighted by the router."""

    def __init__(self, c: cfg.MoEConfig, width: int, rngs: nnx.Rngs):
        if c.num_groups > 1:
            raise NotImplementedError("group-limited routing")
        if c.shared_expert_sink:
            raise NotImplementedError("routed shared experts (shared_expert_sink)")
        e, f = c.num_experts, c.expert.intermediate_size
        init = nnx.initializers.lecun_normal()
        self.router = nnx.Linear(width, e, use_bias=False, rngs=rngs)
        self.router_bias = (
            nnx.Variable(jnp.zeros(e)) if c.load_balancing == "aux_free" else None
        )
        # LatentMoE: the routed experts work between these two projections.
        latent = c.latent_size or width
        self.to_latent = (
            nnx.Linear(width, latent, use_bias=False, rngs=rngs)
            if c.latent_size
            else None
        )
        self.latent_input_norm = (
            Norm(c.latent_input_norm, latent) if c.latent_input_norm else None
        )
        self.latent_output_norm = (
            Norm(c.latent_output_norm, latent) if c.latent_output_norm else None
        )
        self.expert_norm = Norm(c.expert_norm, latent) if c.expert_norm else None
        self.from_latent = (
            nnx.Linear(latent, width, use_bias=False, rngs=rngs)
            if c.latent_size
            else None
        )
        self.up = nnx.Param(init(rngs.params(), (e, latent, f)))
        self.gate = (
            nnx.Param(init(rngs.params(), (e, latent, f)))
            if c.expert.type == "gated_mlp"
            else None
        )
        self.down = nnx.Param(init(rngs.params(), (e, f, latent)))
        self.top_k, self.router_fn = c.top_k, c.router
        self.router_logit_scale = c.router_logit_scale
        self.normalize, self.scaling = c.normalize_top_k, c.routed_scaling_factor
        self.activation = c.expert.activation
        self.limit, self.up_softcap = c.expert.limit, c.expert.up_softcap
        shared = c.shared_expert or c.expert
        self.shared = (
            MLP(
                shared,
                width,
                rngs,
                size=shared.intermediate_size * c.num_shared_experts,
            )
            if c.num_shared_experts
            else None
        )
        self.shared_gate = (
            nnx.Linear(width, 1, use_bias=False, rngs=rngs)
            if c.shared_expert_gate
            else None
        )

    def __call__(self, x, taps: Taps):
        logits = self.router(x) * self.router_logit_scale
        match self.router_fn:
            case "softmax":
                scores = jax.nn.softmax(logits)
            case "sigmoid":
                scores = jax.nn.sigmoid(logits)
            case "sqrtsoftplus":
                scores = jnp.sqrt(jax.nn.softplus(logits))
        choice = scores if self.router_bias is None else scores + self.router_bias[...]
        _, idx = jax.lax.top_k(choice, self.top_k)
        weights = jnp.take_along_axis(scores, idx, axis=-1)
        if self.normalize:
            weights = weights / weights.sum(-1, keepdims=True)
        weights = weights * self.scaling
        dense = (jax.nn.one_hot(idx, scores.shape[-1]) * weights[..., None]).sum(-2)
        z = x if self.to_latent is None else self.to_latent(x)
        if self.latent_input_norm is not None:
            z = self.latent_input_norm(z)
        h = _bound(
            jnp.einsum("btd,edf->btef", z, self.up[...]), self.limit, self.up_softcap
        )
        if self.gate is not None:
            gate = jnp.einsum("btd,edf->btef", z, self.gate[...])
            if self.limit is not None:
                gate = jnp.minimum(gate, self.limit)
            h = _activation(self.activation, gate) * h
        else:
            h = _activation(self.activation, h)
        if self.expert_norm is not None:
            each = self.expert_norm(jnp.einsum("btef,efd->bted", h, self.down[...]))
            y = jnp.einsum("bted,bte->btd", each, dense)
        else:
            y = jnp.einsum("btef,efd,bte->btd", h, self.down[...], dense)
        if self.latent_output_norm is not None:
            y = self.latent_output_norm(y)
        if self.from_latent is not None:
            y = self.from_latent(y)
        if self.shared is not None:
            s = self.shared(x)
            if self.shared_gate is not None:
                s = s * jax.nn.sigmoid(self.shared_gate(x))
            y = y + s
        return y


@register("moe")
def _(c: cfg.MoEConfig, width, ports, ctx: Context):
    return MoE(c, width, ctx.rngs)


def _attention_factor(y: cfg.YaRNConfig) -> float:
    if y.attention_factor is not None:
        return y.attention_factor
    return 0.1 * math.log(y.factor) + 1 if y.factor > 1 else 1.0


def _frequencies(rope: cfg.RoPEConfig, dims: int) -> tuple[jax.Array, float]:
    """Inverse frequencies of the `dims` rotated dims, and the factor on cos and sin."""
    inv_freq = 1.0 / rope.base ** (jnp.arange(0, dims, 2) / dims)
    match rope.scaling:
        case None:
            return inv_freq, 1.0
        case cfg.PositionInterpolationConfig(factor=factor):
            return inv_freq / factor, 1.0
        case cfg.YaRNConfig() as y:
            # The pair that turns `turns` times over the original context, as
            # transformers computes it (with truncation to whole pairs).
            def pair(turns: float) -> float:
                return (
                    dims
                    * math.log(y.original_max_len / (turns * 2 * math.pi))
                    / (2 * math.log(rope.base))
                )

            low = max(math.floor(pair(y.beta_fast)), 0)
            high = min(math.ceil(pair(y.beta_slow)), dims - 1)
            ramp = jnp.clip(
                (jnp.arange(dims // 2) - low) / ((high - low) or 1e-3), 0, 1
            )
            inv_freq = inv_freq / y.factor * ramp + inv_freq * (1 - ramp)
            on_rotary = y.attention_factor_on == "rotary"
            return inv_freq, _attention_factor(y) if on_rotary else 1.0
        case other:
            raise NotImplementedError(f"{other.type} RoPE scaling")


def _logit_gain(rope: cfg.RoPEConfig | None) -> float:
    """YaRN's attention_factor ** 2 if it goes on the softmax scale, else 1."""
    y = rope.scaling if rope is not None else None
    if isinstance(y, cfg.YaRNConfig) and y.attention_factor_on == "softmax":
        return _attention_factor(y) ** 2
    return 1.0


def _rope(x: jax.Array, rope: cfg.RoPEConfig) -> jax.Array:
    """Rotate the first `fraction` of each head's dims (half-split layout); x is (b, t, h, d)."""
    dims = int(x.shape[-1] * rope.fraction) // 2 * 2
    if dims == 0:
        return x
    inv_freq, gain = _frequencies(rope, dims)
    angles = jnp.arange(x.shape[1])[:, None] * inv_freq
    sin, cos = gain * jnp.sin(angles)[:, None, :], gain * jnp.cos(angles)[:, None, :]
    x1, x2, rest = x[..., : dims // 2], x[..., dims // 2 : dims], x[..., dims:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], -1)


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
        self.scale = (c.scale or d**-0.5) * _logit_gain(c.rope)
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
                k = _rope(k, self.rope)
        if self.rope is not None:
            q = _rope(q, self.rope)
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
        logits = _softcap(logits, self.softcap)
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
            gate = _activation(self.gate_activation, self.gate(x))
            out = out * gate.reshape(b, t, self.heads, -1)
        return self.o(out.reshape(b, t, -1))


@register("gqa")
def _(c: cfg.GQAConfig, width, ports, ctx: Context):
    return Attention(c, width, ports, ctx.rngs)


class CausalConv(nnx.Module):
    """Depthwise causal convolution over positions of (batch, seq, channels)."""

    def __init__(
        self,
        channels: int,
        size: int,
        bias: bool,
        rngs: nnx.Rngs,
        *,
        dilation: int = 1,
        zero: bool = False,
    ):
        init = nnx.initializers.zeros if zero else nnx.initializers.lecun_normal()
        self.weight = nnx.Param(init(rngs.params(), (size, channels)))
        self.bias = nnx.Param(jnp.zeros(channels)) if bias else None
        self.dilation = dilation

    def __call__(self, x, taps: Taps | None = None):
        size, t, r = self.weight.shape[0], x.shape[1], self.dilation
        padded = jnp.pad(x, ((0, 0), ((size - 1) * r, 0), (0, 0)))
        y = sum(self.weight[i] * padded[:, i * r : i * r + t] for i in range(size))
        return y if self.bias is None else y + self.bias


@register("conv")
def _(c: cfg.ConvConfig, width, ports, ctx: Context):
    return CausalConv(width, c.size, c.bias, ctx.rngs)


def _next_primes(start: int, count: int, taken: set[int]) -> list[int]:
    """The `count` smallest primes >= start that are not taken yet, which takes them."""
    found, n = [], max(start, 2)
    while len(found) < count:
        if n not in taken and all(n % p for p in range(2, math.isqrt(n) + 1)):
            found.append(n)
            taken.add(n)
        n += 1
    return found


def _ngram_rows(ids, tokens, multipliers, primes, offsets, boundary, pad):
    """Table rows of the n-grams ending at every position, (b, t) -> (b, t, orders * heads).

    The n-gram of the last j + 1 ids hashes to XOR_{i <= j} ids[t - i] * m_i, in wrapping
    uint32 arithmetic. Positions before the sequence, or at and before the last boundary
    token before t, count as `pad`, so that n-grams do not cross documents.
    """
    b, t = ids.shape
    pos = jnp.arange(t)
    start = jnp.zeros((b, t), jnp.int32)
    if boundary is not None:
        after = jnp.where(tokens == boundary, pos + 1, 0)
        start = jnp.pad(jax.lax.cummax(after, axis=1)[:, :-1], ((0, 0), (1, 0)))
    h, rows = jnp.zeros((b, t), jnp.uint32), []
    for i in range(multipliers.shape[0]):
        prev = jnp.take(ids, jnp.maximum(pos - i, 0), axis=1)
        prev = jnp.where(pos - i >= start, prev, pad).astype(jnp.uint32)
        h = h ^ (prev * multipliers[i])
        if i:
            rows.append(h[..., None] % primes[i - 1] + offsets[i - 1])
    return jnp.concatenate(rows, -1).astype(jnp.int32)


class Engram(nnx.Module):
    """Hashed n-gram rows of the input tokens, gated by the stream and added to it. On a
    widened stream (batch, seq, copies, d) every copy has a key of its own, and they
    share the value."""

    def __init__(self, c: cfg.EngramConfig, width, ctx: Context):
        d, vocab = ctx.arch.d_model, ctx.arch.vocab_size
        assert vocab is not None
        copies = getattr(ctx.arch.residual, "expansion", 1) if width == "streams" else 1
        orders, heads = c.max_ngram_size - 1, c.num_heads
        primes = _next_primes(c.table_size, orders * heads, ctx.primes)
        offsets = np.cumsum([0, *primes[:-1]])
        self.primes = nnx.Variable(
            jnp.asarray(primes, jnp.uint32).reshape(orders, heads)
        )
        self.offsets = nnx.Variable(
            jnp.asarray(offsets, jnp.uint32).reshape(orders, heads)
        )
        ctx.engrams += 1
        rng = np.random.default_rng(10007 * ctx.engrams)
        odd = rng.integers(0, 2**31, c.max_ngram_size) * 2 + 1
        self.multipliers = nnx.Variable(jnp.asarray(odd, jnp.uint32))
        # A stand-in for the tokenizer's normalized classes, which a checkpoint replaces.
        self.classes = (
            nnx.Variable(jnp.arange(vocab) % c.vocab_classes)
            if c.vocab_classes
            else None
        )
        init = nnx.initializers.normal(1.0)
        self.table = nnx.Param(init(ctx.rngs.params(), (sum(primes), c.head_dim)))
        self.kv = nnx.Linear(
            orders * heads * c.head_dim, (copies + 1) * d, use_bias=False, rngs=ctx.rngs
        )
        self.q_norm, self.k_norm = Norm(c.norm, (copies, d)), Norm(c.norm, (copies, d))
        self.conv = (
            CausalConv(
                copies * d,
                c.conv_size,
                False,
                ctx.rngs,
                dilation=c.max_ngram_size,
                zero=True,
            )
            if c.conv_size
            else None
        )
        self.conv_norm = Norm(c.norm, (copies, d)) if c.conv_size else None
        self.boundary = c.boundary_token

    def __call__(self, x, taps: Taps):
        tokens = taps[TOKENS][0]
        ids, pad = tokens, self.boundary or 0
        if self.classes is not None:
            ids, pad = self.classes[...][tokens], self.classes[...][pad]
        rows = _ngram_rows(
            ids,
            tokens,
            self.multipliers[...],
            self.primes[...],
            self.offsets[...],
            self.boundary,
            pad,
        )
        h = x if x.ndim == 4 else x[..., None, :]
        (b, t, n, d), kv = (
            h.shape,
            self.kv(self.table[...][rows].reshape(*rows.shape[:2], -1)),
        )
        key, value = kv[..., : n * d].reshape(b, t, n, d), kv[..., n * d :]
        a = (self.q_norm(h) * self.k_norm(key)).sum(-1) / math.sqrt(d)
        gate = jax.nn.sigmoid(jnp.sign(a) * jnp.sqrt(jnp.maximum(jnp.abs(a), 1e-6)))
        u = gate[..., None] * value[..., None, :]
        if self.conv is not None and self.conv_norm is not None:
            conv = self.conv(self.conv_norm(u).reshape(b, t, n * d))
            u = u + jax.nn.silu(conv).reshape(u.shape)
        y = h + u
        return y if x.ndim == 4 else y[..., 0, :]


@register("engram")
def _(c: cfg.EngramConfig, width, ports, ctx: Context):
    return Engram(c, width, ctx)


class Scale(nnx.Module):
    def __init__(self, c: cfg.ScaleConfig, width: int):
        self.value = nnx.Param(jnp.full(width if c.per_channel else (), c.init))

    def __call__(self, x, taps: Taps):
        return x * self.value


@register("scale")
def _(c: cfg.ScaleConfig, width, ports, ctx: Context):
    return Scale(c, width)


class Linear(nnx.Module):
    def __init__(self, c: cfg.LinearConfig, width: int, rngs: nnx.Rngs):
        self.proj = nnx.Linear(width, c.features, use_bias=c.bias, rngs=rngs)

    def __call__(self, x, taps: Taps):
        return self.proj(x)


@register("linear")
def _(c: cfg.LinearConfig, width, ports, ctx: Context):
    return Linear(c, width, ctx.rngs)


class Activation(nnx.Module):
    def __init__(self, fn: str):
        self.fn = fn

    def __call__(self, x, taps: Taps):
        return _activation(self.fn, x)


@register("act")
def _(c: cfg.ActivationConfig, width, ports, ctx: Context):
    return Activation(c.fn)


class Dropout(nnx.Module):
    def __init__(self, rate: float, rngs: nnx.Rngs):
        self.dropout = nnx.Dropout(rate, rngs=rngs)

    def __call__(self, x, taps: Taps):
        return self.dropout(x)


@register("dropout")
def _(c: cfg.DropoutConfig, width, ports, ctx: Context):
    return Dropout(c.rate, ctx.rngs)
