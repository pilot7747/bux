"""Linear attention: token mixers with a matrix-valued state a head, updated token by
token. Gated DeltaNet, Kimi Delta Attention, DeltaNet, GLA and Lightning are here, and
RWKV-7 is in ``rwkv7``.

All of them run one recurrence (`recurrence`), over a state S of k by v a value head:

    S_t = diag(w_t) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T,    o_t = S_t^T q_t

with a decay w_t in (0, 1] a key channel. The b a^T term is how the delta rule writes
over what S holds at a key (`delta_rule`), and RWKV-7 generalizes it. The state is zero
at every document start, so a packed row gives each document what it gives alone.

Shapes: b rows, t positions, d the width, h heads (value heads, once q and k are shared
out to them), k and v the head dims of keys and values.

`recurrence` is the reference form, a scan over positions; a chunked kernel (fla's)
computes the same.

Under tensor parallelism (``bux.parallel``) every projection splits over `heads`, into
the heads or out of them, and so does every parameter a head has: the recurrence runs
on each device's own heads, with no communication.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux import parallel
from bux.architecture.conv import CausalConv
from bux.architecture.functional import activation
from bux.architecture.norm import Norm
from bux.architecture.registry import Context, Taps, register, segments
from bux.architecture.segments import Segments
from bux.lang import ir

# One position of the recurrence: q, k, v, the log decay, a, b, and 1 where the token
# continues a document, 0 where it starts one.
_Inputs = tuple[
    Float[Array, "b h k"],
    Float[Array, "b h k"],
    Float[Array, "b h v"],
    Float[Array, "b h k"],
    Float[Array, "b h k"],
    Float[Array, "b h k"],
    Float[Array, " b"],
]


def _step(
    state: Float[Array, "b h k v"], inputs: _Inputs
) -> tuple[Float[Array, "b h k v"], Float[Array, "b h v"]]:
    q, k, v, log_decay, a, b, continues = inputs
    state = state * continues[:, None, None, None]
    read = jnp.einsum("bhk,bhkv->bhv", a, state)
    state = (
        jnp.exp(log_decay)[..., None] * state
        + b[..., None] * read[..., None, :]
        + k[..., None] * v[..., None, :]
    )
    return state, jnp.einsum("bhk,bhkv->bhv", q, state)


def recurrence(
    q: Float[Array, "b t h k"],
    k: Float[Array, "b t h k"],
    v: Float[Array, "b t h v"],
    log_decay: Float[Array, "b t h k"],
    docs: Segments | None,
    a: Float[Array, "b t h k"] | None = None,
    b: Float[Array, "b t h k"] | None = None,
) -> Float[Array, "b t h v"]:
    """The outputs S_t^T q_t of the state

        S_t = diag(w_t) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T,

    with w = exp(log_decay), and S zero before every document; without a and b, decayed
    linear attention. A scan over positions, in float32."""
    if a is None or b is None:
        a = b = jnp.zeros_like(k)
    batch, t, h, dk = k.shape
    continues = (
        jnp.ones((batch, t)) if docs is None else (docs.within > 0).astype(jnp.float32)
    )
    xs = jax.tree.map(
        lambda y: jnp.moveaxis(y.astype(jnp.float32), 1, 0),
        (q, k, v, log_decay, a, b, continues),
    )
    state = jnp.zeros((batch, h, dk, v.shape[-1]), jnp.float32)
    _, o = jax.lax.scan(_step, state, xs)
    return jnp.moveaxis(o, 0, 1).astype(v.dtype)


def delta_rule(
    q: Float[Array, "b t h k"],
    k: Float[Array, "b t h k"],
    v: Float[Array, "b t h v"],
    log_decay: Float[Array, "b t h k"],
    beta: Float[Array, "b t h"],
    docs: Segments | None,
) -> Float[Array, "b t h v"]:
    """The gated delta rule: each token decays the state, then moves what it holds at
    k_t a step beta_t toward v_t,

        S_t = (I - beta_t k_t k_t^T) diag(w_t) S_{t-1} + beta_t k_t v_t^T.

    That is `recurrence` with a = -w k, and b = beta k for k."""
    step = beta[..., None] * k
    return recurrence(q, step, v, log_decay, docs, a=-jnp.exp(log_decay) * k, b=step)


def layer_input(x: Float[Array, "b t d"]) -> Float[Array, "b t d"]:
    """A layer's input, its positions split as inside a layer (`seq`), not as the
    stream: under sequence parallelism, gathered before the projections, not their
    weights."""
    return parallel.shard(x, "batch", "seq", None)


def heads(y: Float[Array, "b t h *k"]) -> Float[Array, "b t h *k"]:
    """`y`, something of every head at every position, split as the layout splits
    heads: whole if their axes do not divide them, though a projection's h * k
    channels may split, so that nothing else splits a head."""
    return parallel.shard(y, "batch", "seq", "heads", *(None,) * (y.ndim - 3))


def l2norm(x: Float[Array, "*lead k"], eps: float = 1e-6) -> Float[Array, "*lead k"]:
    """x over its L2 norm, as fla computes it: x / sqrt(|x|^2 + eps)."""
    return x * jax.lax.rsqrt(jnp.square(x).sum(-1, keepdims=True) + eps)


class LowRank(nnx.Module):
    """x -> act(x A) B + c, a projection of the width through `rank` dims (`latent`),
    with an optional activation between and bias after, its outputs named `out`.
    `orthogonal` starts it as RWKV-7 does, A at zero and B orthogonal, scaled by 0.1;
    otherwise A and B start as dense layers do."""

    def __init__(
        self,
        n_in: int,
        rank: int,
        n_out: int,
        rngs: nnx.Rngs,
        *,
        activation: str | None = None,
        bias: Float[Array, " n"] | bool = False,
        orthogonal: bool = False,
        out: parallel.Name = None,
    ):
        if orthogonal:
            gain = 0.1 * max(1.0, math.sqrt(rank / n_out))
            down = jnp.zeros((n_in, rank))
            up = nnx.initializers.orthogonal(gain)(rngs.params(), (rank, n_out))
        else:
            init = nnx.initializers.lecun_normal()
            down = init(rngs.params(), (n_in, rank))
            up = init(rngs.params(), (rank, n_out))
        self.down = nnx.Param(down, names=("embed", "latent"))
        self.up = nnx.Param(up, names=("latent", out))
        start = (jnp.zeros(n_out) if bias else None) if isinstance(bias, bool) else bias
        self.bias = None if start is None else nnx.Param(start, names=(out,))
        self.activation = activation

    def __call__(self, x: Float[Array, "*lead d"]) -> Float[Array, "*lead n"]:
        h = x @ self.down[...]
        if self.activation is not None:
            h = activation(self.activation, h)
        y = h @ self.up[...]
        return y if self.bias is None else y + self.bias[...]


def into_heads(width: int, heads: int, dim: int, rngs: nnx.Rngs) -> nnx.Linear:
    """A projection of the width to `heads` heads of `dim` channels, side by side."""
    init = parallel.named("embed", ("heads", heads))
    return nnx.Linear(width, heads * dim, use_bias=False, kernel_init=init, rngs=rngs)


def _dt_bias(rngs: nnx.Rngs, n: int) -> Float[Array, " n"]:
    """softplus^-1 of steps log-uniform in [1e-3, 0.1], floored at 1e-4: Mamba's start
    for its time steps, and fla's for the dt_bias of Gated DeltaNet and KDA."""
    log_dt = jax.random.uniform(
        rngs.params(), (n,), minval=math.log(1e-3), maxval=math.log(0.1)
    )
    dt = jnp.maximum(jnp.exp(log_dt), 1e-4)
    return dt + jnp.log(-jnp.expm1(-dt))


def _log_uniform(rngs: nnx.Rngs, n: int, low: float, high: float) -> Float[Array, " n"]:
    """The logs of n draws uniform in [low, high]: where A_log starts."""
    return jnp.log(jax.random.uniform(rngs.params(), (n,), minval=low, maxval=high))


def _mixer_depths(arch: ir.Architecture, kind: str) -> tuple[list[int], int]:
    """Where the `kind` layers are built, as indices among the token mixers the model
    runs (in the order they are built), and how many token mixers it runs. A shared
    layer is built where it first runs."""
    built: list[int] = []
    shared: set[str] = set()
    runs = 0

    def visit(node: ir.Node, fresh: bool) -> None:
        nonlocal runs
        if node.share is not None:
            fresh = fresh and node.share not in shared
            shared.add(node.share)
        if isinstance(node, ir.Layer) and isinstance(node.config, cfg.AttentionConfig):
            if fresh and node.config.type == kind:
                built.append(runs)
            runs += 1
        for _ in range(node.times if isinstance(node, ir.Repeat) else 1):
            for child in ir.children(node):
                visit(child, fresh)

    visit(arch.graph, True)
    return built, runs


def depth(ctx: Context, kind: str) -> tuple[int, int]:
    """The index of the `kind` layer being built among the model's token mixers, from 0,
    and how many token mixers it runs: the layer index and count that MiniMax and
    RWKV-7 start a layer by. A layer built outside the model's graph is the first."""
    starts, layers = _mixer_depths(ctx.arch, kind)
    i = ctx.built[kind]
    return (starts[i] if i < len(starts) else 0), max(layers, 1)


class LinearAttention(nnx.Module):
    """What the linear-attention layers share: the q, k and v projections, the short
    causal conv on them, and the output's norm, gate and projection.

    With more value heads than q/k heads, each q/k head serves a group of value heads
    next to each other (grouped value attention, Qwen3-Next). The output norm is an
    RMSNorm of each head, or of all heads together (`norm_all_heads`), times the gate's
    activation of the input: the gated norm of Qwen3-Next and fla.
    """

    def __init__(
        self,
        c: cfg.LinearAttentionConfig,
        width: int,
        rngs: nnx.Rngs,
        *,
        silu: bool = True,
        gate_rank: int | None = None,
        norm_all_heads: bool = False,
    ):
        h, hv = c.num_heads, c.num_v_heads or c.num_heads
        if hv % h:
            raise ValueError(f"num_v_heads={hv} is not a multiple of num_heads={h}")
        dk, dv = c.head_dim, c.v_head_dim or c.head_dim
        self.heads, self.v_heads, self.head_dim, self.v_head_dim = h, hv, dk, dv

        self.q, self.k, self.v = (
            into_heads(width, h, dk, rngs),
            into_heads(width, h, dk, rngs),
            into_heads(width, hv, dv, rngs),
        )
        # One conv over q, k and v side by side, which it convolves apart.
        size = c.conv_size
        self.conv = (
            CausalConv(2 * h * dk + hv * dv, size, False, rngs) if size else None
        )
        self.silu = silu  # on q, k and v without a conv; after one, always
        self.gate_activation = c.output_gate
        self.gate = (
            None
            if c.output_gate is None
            else LowRank(width, gate_rank, hv * dv, rngs, out=("heads", hv))
            if gate_rank
            else into_heads(width, hv, dv, rngs)
        )
        self.norm_all_heads = norm_all_heads
        rms = cfg.NormalizationConfig(type="rmsnorm", eps=c.norm_eps)
        self.norm = Norm(rms, hv * dv if norm_all_heads else dv)
        self.o = nnx.Linear(
            hv * dv,
            width,
            use_bias=False,
            kernel_init=parallel.named(("heads", hv), "embed"),
            rngs=rngs,
        )

    def _qkv(
        self, x: Float[Array, "b t d"], taps: Taps
    ) -> tuple[
        Float[Array, "b t h k"], Float[Array, "b t h k"], Float[Array, "b t h v"]
    ]:
        """The queries, keys and values of every value head: projected, then convolved
        and through SiLU."""
        b, t, _ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        if self.conv is not None:
            q, k, v = map(jax.nn.silu, self.conv.side_by_side((q, k, v), taps))
        elif self.silu:
            q, k, v = map(jax.nn.silu, (q, k, v))
        group = self.v_heads // self.heads
        q, k = (
            jnp.repeat(heads(y.reshape(b, t, self.heads, self.head_dim)), group, axis=2)
            for y in (q, k)
        )
        return q, k, heads(v.reshape(b, t, self.v_heads, self.v_head_dim))

    def _output(
        self, x: Float[Array, "b t d"], o: Float[Array, "b t h v"]
    ) -> Float[Array, "b t d"]:
        """The heads' outputs, normalized, gated and projected back to the width."""
        b, t, _ = x.shape
        o = heads(o)
        o = self.norm(o.reshape(b, t, -1) if self.norm_all_heads else o)
        o = o.reshape(b, t, -1)
        if self.gate is not None and self.gate_activation is not None:
            o = o * activation(self.gate_activation, self.gate(x))
        return self.o(o)


class DeltaNet(LinearAttention):
    """DeltaNet (Schlag et al., 2021; Yang et al., 2024), as fla has it: the delta rule
    on L2-normalized q and k, with q scaled by k ** -0.5, and a step a value head,
    beta = sigmoid(W_b x). Gated DeltaNet and KDA add a decay (`_log_decay`)."""

    def __init__(
        self,
        c: cfg.LinearAttentionConfig,
        width: int,
        rngs: nnx.Rngs,
        *,
        gate_rank: int | None = None,
    ):
        super().__init__(c, width, rngs, gate_rank=gate_rank)
        self.b = into_heads(width, self.v_heads, 1, rngs)

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        x = layer_input(x)
        q, k, v = self._qkv(x, taps)
        q, k = l2norm(q) * self.head_dim**-0.5, l2norm(k)
        beta = heads(jax.nn.sigmoid(self.b(x)))
        o = delta_rule(q, k, v, heads(self._log_decay(x)), beta, segments(taps))
        return self._output(x, o)

    def _log_decay(self, x: Float[Array, "b t d"]) -> Float[Array, "b t h k"]:
        """The log of the decay of the state, a key channel: none."""
        b, t, _ = x.shape
        return jnp.zeros((b, t, self.v_heads, self.head_dim))


class GatedDeltaNet(DeltaNet):
    """Gated DeltaNet (Yang et al., 2025), as transformers' Qwen3-Next and fla have it:
    DeltaNet with a decay exp(g) a value head, g = -exp(A_log) softplus(W_a x + dt_b),
    with A_log and dt_b (dt_bias) learned a value head.

    A_log starts at the logs of draws uniform in [0.01, 16] (transformers), and dt_bias
    at Mamba's time steps (fla and the paper's code; transformers starts it at ones).
    """

    def __init__(self, c: cfg.GatedDeltaNetConfig, width: int, rngs: nnx.Rngs):
        super().__init__(c, width, rngs)
        hv = self.v_heads
        self.a = into_heads(width, hv, 1, rngs)
        self.A_log = nnx.Param(_log_uniform(rngs, hv, 0.01, 16), names=("heads",))
        self.dt_bias = nnx.Param(_dt_bias(rngs, hv), names=("heads",))

    def _log_decay(self, x: Float[Array, "b t d"]) -> Float[Array, "b t h k"]:
        b, t, _ = x.shape
        rate = jax.nn.softplus(self.a(x) + self.dt_bias[...])
        g = -jnp.exp(self.A_log[...]) * rate
        return jnp.broadcast_to(g[..., None], (b, t, self.v_heads, self.head_dim))


class KimiDeltaAttention(DeltaNet):
    """Kimi Delta Attention (Kimi Linear, 2025), as transformers and fla have it: Gated
    DeltaNet with a decay a key channel, g = -exp(A_log) softplus(f + dt_bias), where
    f = W_f2 W_f1 x is a low-rank projection (`gate_rank`) and A_log is one a head. With
    a lower bound (Kimi K3, GLM-5-Next), g = lower_bound * sigmoid(exp(A_log) (f +
    dt_bias)), and A_log starts at 0, as in fla's safe gate (K3's code starts it at
    log U(1, 16) either way). The output gate is low rank too (`output_gate_rank`).
    """

    def __init__(self, c: cfg.KDAConfig, width: int, rngs: nnx.Rngs):
        super().__init__(c, width, rngs, gate_rank=c.output_gate_rank)
        hv, dk = self.v_heads, self.head_dim
        self.f = LowRank(width, c.gate_rank or dk, hv * dk, rngs, out=("heads", hv))
        self.lower_bound = c.gate_lower_bound
        self.A_log = nnx.Param(
            _log_uniform(rngs, hv, 1, 16)
            if c.gate_lower_bound is None
            else jnp.zeros(hv),
            names=("heads",),
        )
        self.dt_bias = nnx.Param(_dt_bias(rngs, hv * dk), names=(("heads", hv),))

    def _log_decay(self, x: Float[Array, "b t d"]) -> Float[Array, "b t h k"]:
        b, t, _ = x.shape
        f = (self.f(x) + self.dt_bias[...]).reshape(b, t, self.v_heads, self.head_dim)
        rate = jnp.exp(self.A_log[...])[:, None]
        if self.lower_bound is not None:
            return self.lower_bound * jax.nn.sigmoid(rate * f)
        return -rate * jax.nn.softplus(f)


class GatedLinearAttention(LinearAttention):
    """Gated linear attention (Yang et al., 2024), as fla has it: linear attention with
    q scaled by k ** -0.5 and a decay a key channel, log w = log sigmoid(f) / 16, where
    f is a low-rank projection with a bias (`gate_rank`). Without a conv, q, k and v do
    not go through SiLU."""

    def __init__(self, c: cfg.GLAConfig, width: int, rngs: nnx.Rngs):
        super().__init__(c, width, rngs, silu=False)
        hv = self.v_heads
        self.f = LowRank(
            width, c.gate_rank, hv * self.head_dim, rngs, bias=True, out=("heads", hv)
        )

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        x = layer_input(x)
        b, t, _ = x.shape
        q, k, v = self._qkv(x, taps)
        f = self.f(x).reshape(b, t, self.v_heads, self.head_dim)
        g = heads(jax.nn.log_sigmoid(f) / 16)
        o = recurrence(q * self.head_dim**-0.5, k, v, g, segments(taps))
        return self._output(x, o)


def slopes(heads: int, layer: int, layers: int) -> Float[Array, " h"]:
    """MiniMax's decay rates, s_i = 2 ** (-8 i / heads) for heads i = 1 to heads, scaled
    by 1 - layer / (layers - 1) (and 1e-5 more, as MiniMax has it). The decay is
    exp(-s), which leaves a plain sum of the past at the last layer."""
    factor = 1 - layer / (layers - 1 + 1e-5) + 1e-5
    return 2 ** (-8 / heads * jnp.arange(1, heads + 1)) * factor


class LightningAttention(LinearAttention):
    """Lightning attention (TransNormerLLM; MiniMax-01, as transformers has it): linear
    attention on SiLU'd q, k and v, unscaled, with a fixed decay exp(-s) a head that
    weakens with depth (`slopes`), and one RMSNorm of all the heads together before the
    gate."""

    def __init__(
        self,
        c: cfg.LightningAttentionConfig,
        width: int,
        rngs: nnx.Rngs,
        *,
        layer: int = 0,
        layers: int = 1,
    ):
        super().__init__(c, width, rngs, norm_all_heads=True)
        self.slopes = nnx.Variable(
            slopes(self.v_heads, layer, layers), names=("heads",)
        )

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        x = layer_input(x)
        q, k, v = self._qkv(x, taps)
        g = jnp.broadcast_to(-self.slopes[...][:, None], q.shape)
        return self._output(x, recurrence(q, k, v, g, segments(taps)))


@register("deltanet")
def _(
    c: cfg.DeltaNetConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> DeltaNet:
    return DeltaNet(c, width, ctx.rngs)


@register("gdn")
def _(
    c: cfg.GatedDeltaNetConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> GatedDeltaNet:
    return GatedDeltaNet(c, width, ctx.rngs)


@register("kda")
def _(
    c: cfg.KDAConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> KimiDeltaAttention:
    return KimiDeltaAttention(c, width, ctx.rngs)


@register("gla")
def _(
    c: cfg.GLAConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> GatedLinearAttention:
    return GatedLinearAttention(c, width, ctx.rngs)


@register("lightning")
def _(
    c: cfg.LightningAttentionConfig,
    width: int,
    ports: tuple[ir.Port, ...],
    ctx: Context,
) -> LightningAttention:
    layer, layers = depth(ctx, "lightning")
    return LightningAttention(c, width, ctx.rngs, layer=layer, layers=layers)
