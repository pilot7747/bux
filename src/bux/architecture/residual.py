"""The residual algebras: what a residual branch [f] outside any other does in place of
x + f(x), and ``merge``, which reads the stream they widen back into one.

hc, mhc and gr widen the stream into n copies of it, an array (b, t, n, d). attnres
keeps the embedding and every branch output so far, (b, t, 1 + L, d), and sums them
into blocks as it reads: so what a branch reads follows from the state alone, at
whatever depth it runs (a shared block runs at several), and taps save and restore it
as it is. vpgate keeps one stream. ``embed`` widens its output with ``widen``, and so
does nextn when it gets one stream. Single-pass hc and mhc hand the read coefficients
from one branch to the next in the taps.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux.architecture.norm import Norm
from bux.architecture.registry import Context, Taps, register
from bux.lang import ir
from bux.parallel import stream

# In Sinkhorn's sums and added to mhc's read: DeepSeek-V4's hc_eps.
_EPS = 1e-6
# The read coefficients the last single-pass branch computed, for the next read.
_PENDING = "<pending read>"


def widening(residual: cfg.ResidualConfig | None) -> int | None:
    """How many copies of one stream the widened stream starts as: n for hc, mhc and gr,
    and 1 for attnres, whose first source is the embedding. None when it is not widened.
    """
    match residual:
        case cfg.HyperConnectionsConfig() | cfg.GatedResidualConfig():
            return residual.expansion
        case cfg.AttentionResidualConfig():
            return 1
    return None


def widen(x: Float[Array, "b t d"], n: int) -> Float[Array, "b t n d"]:
    """The widened stream that starts from x: n copies of it."""
    return jnp.repeat(x[:, :, None], n, axis=2)


def sinkhorn(
    logits: Float[Array, "*lead n n"], iters: int
) -> Float[Array, "*lead n n"]:
    """The doubly stochastic matrix Sinkhorn-Knopp makes of exp(logits), as DeepSeek-V4
    does: a softmax over every row, the columns divided by their sums, then `iters` - 1
    more rounds of rows and columns. Row j is what stream j gives the others, and the
    columns, what every stream receives, sum to 1 last."""
    m = jax.nn.softmax(logits, -1) + _EPS
    m = m / (m.sum(-2, keepdims=True) + _EPS)
    for _ in range(iters - 1):
        m = m / (m.sum(-1, keepdims=True) + _EPS)
        m = m / (m.sum(-2, keepdims=True) + _EPS)
    return m


def sources(
    state: Float[Array, "b t s d"], block_size: int | None
) -> Float[Array, "b t m d"]:
    """What attnres reads from: the embedding, then every branch output or, with
    `block_size`, the sums of the complete blocks of that many and of the block under
    way, once it has an output."""
    if block_size is None:
        return state
    b, t, s, d = state.shape
    outputs = state[:, :, 1:]
    full = (s - 1) // block_size
    blocks = outputs[:, :, : full * block_size].reshape(b, t, full, block_size, d)
    parts = [state[:, :, :1], blocks.sum(3)]
    if full * block_size < s - 1:
        parts.append(outputs[:, :, full * block_size :].sum(2, keepdims=True))
    return jnp.concatenate(parts, 2)


def _pending(taps: Taps, x: Float[Array, "b t n d"]) -> Float[Array, "b t n"]:
    """What a single-pass read uses: the coefficients the last branch computed, or
    stream 0 alone before any branch, as DeepSeek-V4.1 starts (make_identity_pre_mix).
    """
    if _PENDING in taps:
        return taps[_PENDING][-1]
    return jnp.broadcast_to(jax.nn.one_hot(0, x.shape[2]), x.shape[:3])


def _single_pass_read(
    own: Float[Array, "b t n"], x: Float[Array, "b t n d"], taps: Taps
) -> Float[Array, "b t n"]:
    """The `_pending` coefficients, leaving the branch's `own` for the next read."""
    read = _pending(taps, x)
    taps[_PENDING] = [own]
    return read


class _Dynamic(nnx.Module):
    """s tanh(norm(H) W) at every stream of H, where norm is a layernorm without
    parameters: what hyper-connection weights add with `dynamic`. W starts at 0, and the
    scale s at 0.01."""

    def __init__(self, d: int, columns: int, eps: float):
        layernorm = cfg.NormalizationConfig(type="layernorm", affine=False, eps=eps)
        self.norm = Norm(layernorm, d)
        self.proj = nnx.Param(jnp.zeros((d, columns)))
        self.scale = nnx.Param(jnp.asarray(0.01))

    def __call__(self, x: Float[Array, "b t n d"]) -> Float[Array, "b t n k"]:
        return self.scale[...] * jnp.tanh(self.norm(x) @ self.proj[...])


class HyperConnections(nnx.Module):
    """Hyper-connections around one branch f (Zhu et al., 2025). From the n streams H,
    f reads h = A_m^T H, and the branch returns A_r^T H + B^T f(h). With `dynamic`,
    A = [A_m A_r] and B each add a `_Dynamic` term of their own.

    The k-th branch starts as a pre-norm residual, as in the paper: it reads stream
    k mod n (A_m = e_k), keeps the streams (A_r = I) and adds f to every one (B = 1).
    Without f it is the read alone, which starts as the paper's sum of the streams
    (A_m = 1): ``merge``. With `single_pass`, f reads with the A_m the branch before
    computed (`_single_pass_read`).
    """

    def __init__(
        self,
        c: cfg.HyperConnectionsConfig,
        d: int,
        index: int,
        body: nnx.Module | None = None,
    ):
        n, write = c.expansion, body is not None
        self.body, self.single_pass = body, c.single_pass
        a_m = jax.nn.one_hot(index % n, n) if write else jnp.ones(n)
        a = jnp.concatenate([a_m[:, None], jnp.eye(n)], 1)
        self.static_alpha = nnx.Param(a if write else a[:, :1])
        self.static_beta = nnx.Param(jnp.ones(n)) if write else None
        columns, eps = a.shape[1] if write else 1, c.norm_eps
        self.dynamic_alpha = _Dynamic(d, columns, eps) if c.dynamic else None
        self.dynamic_beta = _Dynamic(d, 1, eps) if c.dynamic and write else None

    def _alpha(self, x: Float[Array, "b t n d"]) -> Float[Array, "b t n k"]:
        """A = [A_m A_r] at every position, or A_m alone without f."""
        a = self.static_alpha[...]
        a = jnp.broadcast_to(a, (*x.shape[:3], a.shape[-1]))
        return a if self.dynamic_alpha is None else a + self.dynamic_alpha(x)

    def read(self, x: Float[Array, "b t n d"]) -> Float[Array, "b t d"]:
        return jnp.einsum("btn,btnd->btd", self._alpha(x)[..., 0], x)

    def __call__(
        self, x: Float[Array, "b t n d"], taps: Taps
    ) -> Float[Array, "b t n d"]:
        assert self.body is not None and self.static_beta is not None
        alpha = self._alpha(x)
        read = alpha[..., 0]
        if self.single_pass:
            read = _single_pass_read(read, x, taps)
        y = self.body(jnp.einsum("btn,btnd->btd", read, x), taps)
        beta = jnp.broadcast_to(self.static_beta[...], x.shape[:3])
        if self.dynamic_beta is not None:
            beta = beta + self.dynamic_beta(x)[..., 0]
        kept = jnp.einsum("btjk,btjd->btkd", alpha[..., 1:], x)
        return stream(kept + beta[..., None] * y[:, :, None])


class ManifoldHyperConnections(nnx.Module):
    """Manifold-constrained hyper-connections around one branch f (DeepSeek mHC, 2025),
    as DeepSeek-V4's code computes them. With r the n streams x flattened into one and
    RMS-normalized, f reads h = sum_j pre_j x_j, and stream k becomes
    sum_j res_jk x_j + post_k f(h), where pre = sigmoid(a_pre r W_pre + b_pre),
    post = 2 sigmoid(a_post r W_post + b_post), and res is what `sinkhorn` makes of
    a_res r W_res + b_res.

    The projections W start at 0, and the gates a at 0.01, as the paper's do. The paper
    and DeepSeek's code give no init for the biases b, so these start it as a pre-norm
    residual, as hc starts: reading the mean of the streams (pre = 1/n), adding f to
    each (post = 1) and keeping them (b_res = -8 off the diagonal, so that res is I to
    within e^-8, as arXiv 2605.21724 starts its mHC baselines). Without f it is the
    read alone, as DeepSeek-V4's head reads the streams: ``merge``. Without `dynamic`
    there are no W and a, and the coefficients are the biases' alone.

    With `single_pass`, f reads with the pre the branch before computed, or stream 0
    alone in the first branch, as DeepSeek-V4.1 reads (X_{l+1} = B_l X_l +
    C_l F_l(A_{l-1} X_l)). Whatever runs between two branches, such as engram or nextn,
    leaves that pre as it is, as V4.1's engram does, and taps save the stream alone.
    """

    def __init__(
        self, c: cfg.HyperConnectionsConfig, d: int, body: nnx.Module | None = None
    ):
        n, write = c.expansion, body is not None
        self.body, self.single_pass = body, c.single_pass
        self.iters, self.norm_eps = c.sinkhorn_iters, c.norm_eps
        pre = jnp.full(n, -math.log(max(n - 1, 1)))  # sigmoid(b_pre) = 1 / n
        res = -8.0 * (1 - jnp.eye(n))
        bias = jnp.concatenate([pre, jnp.zeros(n), res.reshape(-1)]) if write else pre
        proj = jnp.zeros((n * d, bias.shape[0]))
        scale = jnp.full(3 if write else 1, 0.01)
        self.proj = nnx.Param(proj) if c.dynamic else None  # W
        self.scale = nnx.Param(scale) if c.dynamic else None  # a
        self.bias = nnx.Param(bias)  # b

    def _logits(self, x: Float[Array, "b t n d"]) -> Float[Array, "b t k"]:
        """a r W + b, or b alone without `dynamic`: pre's columns, then post's, then
        res's row by row."""
        b, t, n, d = x.shape
        bias = self.bias[...]
        if self.proj is None or self.scale is None:
            return jnp.broadcast_to(bias, (b, t, bias.shape[0]))
        flat = x.reshape(b, t, n * d)
        rms = jax.lax.rsqrt(jnp.square(flat).mean(-1, keepdims=True) + self.norm_eps)
        widths = (n, n, n * n)
        a = jnp.concatenate([jnp.full(w, s) for s, w in zip(self.scale[...], widths)])
        return (flat @ self.proj[...]) * rms * a + bias

    def _pre(self, logits: Float[Array, "b t k"], n: int) -> Float[Array, "b t n"]:
        return jax.nn.sigmoid(logits[..., :n]) + _EPS

    def read(self, x: Float[Array, "b t n d"]) -> Float[Array, "b t d"]:
        return jnp.einsum("btn,btnd->btd", self._pre(self._logits(x), x.shape[2]), x)

    def __call__(
        self, x: Float[Array, "b t n d"], taps: Taps
    ) -> Float[Array, "b t n d"]:
        assert self.body is not None
        b, t, n, _ = x.shape
        logits = self._logits(x)
        read = self._pre(logits, n)
        if self.single_pass:
            read = _single_pass_read(read, x, taps)
        post = 2 * jax.nn.sigmoid(logits[..., n : 2 * n])
        res = sinkhorn(logits[..., 2 * n :].reshape(b, t, n, n), self.iters)
        y = self.body(jnp.einsum("btn,btnd->btd", read, x), taps)
        kept = jnp.einsum("btjk,btjd->btkd", res, x)
        return stream(kept + post[..., None] * y[:, :, None])


class GatedResidual(nnx.Module):
    """The gated residual around one branch f (Qwen3.8-Flash-Next, 2026), as its report
    and the transformers code have it. Each of the n streams is RMS-normalized with a
    gain of its own, into r_j. With r the r_j side by side, f reads mean_j g_j * r_j,
    for the gate g = sigmoid(W_u silu(W_d r / n)) of rank `rank`, and the branch adds f
    to stream j with the weight 2 sigmoid(W_w r / n)_j, in (0, 2). The projections
    start as the other layers' do. Without f it is the read alone, which before the
    head replaces the final norm: ``merge``.
    """

    def __init__(
        self,
        c: cfg.GatedResidualConfig,
        d: int,
        rngs: nnx.Rngs,
        body: nnx.Module | None = None,
    ):
        n = c.expansion
        self.body = body
        rmsnorm = cfg.NormalizationConfig(type="rmsnorm", eps=c.norm_eps)
        self.norm = Norm(rmsnorm, (n, d))
        self.down = nnx.Linear(n * d, c.rank, use_bias=False, rngs=rngs)
        self.up = nnx.Linear(c.rank, n * d, use_bias=False, rngs=rngs)
        self.write = (
            nnx.Linear(n * d, n, use_bias=False, rngs=rngs)
            if body is not None
            else None
        )

    def _read(self, r: Float[Array, "b t n d"]) -> Float[Array, "b t d"]:
        """mean_j g_j * r_j, from the normalized streams r."""
        b, t, n, d = r.shape
        gate = self.up(jax.nn.silu(self.down(r.reshape(b, t, n * d)) / n))
        return (jax.nn.sigmoid(gate).reshape(r.shape) * r).mean(2)

    def read(self, x: Float[Array, "b t n d"]) -> Float[Array, "b t d"]:
        return self._read(self.norm(x))

    def __call__(
        self, x: Float[Array, "b t n d"], taps: Taps
    ) -> Float[Array, "b t n d"]:
        assert self.body is not None and self.write is not None
        b, t, n, d = x.shape
        r = self.norm(x)
        y = self.body(self._read(r), taps)
        weight = 2 * jax.nn.sigmoid(self.write(r.reshape(b, t, n * d)) / n)
        return stream(x + weight[..., None] * y[:, :, None])


class AttentionResidual(nnx.Module):
    """Attention residuals around one branch f (Kimi, 2026), as the paper and Kimi K3's
    code have them: f reads the softmax over the `sources` v of w . rmsnorm(v), for a
    query w of its own that starts at 0, so that it starts as their average. f's output
    joins the state. Without f it is the read alone, which the head reads through:
    ``merge``.
    """

    def __init__(
        self, c: cfg.AttentionResidualConfig, d: int, body: nnx.Module | None = None
    ):
        self.body = body
        self.query = nnx.Param(jnp.zeros(d))
        self.key_norm = Norm(cfg.NormalizationConfig(type="rmsnorm", eps=c.norm_eps), d)
        self.block_size = c.block_size

    def read(self, x: Float[Array, "b t s d"]) -> Float[Array, "b t d"]:
        v = sources(x, self.block_size)
        scores = jnp.einsum("btmd,d->btm", self.key_norm(v), self.query[...])
        return jnp.einsum("btm,btmd->btd", jax.nn.softmax(scores, -1), v)

    def __call__(
        self, x: Float[Array, "b t s d"], taps: Taps
    ) -> Float[Array, "b t s+1 d"]:
        assert self.body is not None
        y = self.body(self.read(x), taps)
        return stream(jnp.concatenate([x, y[:, :, None]], 2))


class ResidualGate(nnx.Module):
    """The variance-preserving residual gate around one branch f (Muse Spark): x becomes
    a * x + b * f(x) per channel, with b = sigmoid(g / temperature) and
    a = sqrt(max(1 - b^2, eps)).

    The k-th branch starts at b = 1 / sqrt(k + 2), so that the stream after it is the
    plain residual sum, of the embedding and k + 1 outputs, over sqrt(k + 2): with
    branches that start with a norm, and a norm before the head, the model starts as
    the pre-norm residual one.
    """

    def __init__(self, c: cfg.ResidualGateConfig, d: int, index: int, body: nnx.Module):
        b = 1 / math.sqrt(index + 2)
        self.gate = nnx.Param(jnp.full(d, c.temperature * math.log(b / (1 - b))))
        self.temperature, self.eps = c.temperature, c.eps
        self.body = body

    def weights(self) -> tuple[Float[Array, " d"], Float[Array, " d"]]:
        """a and b, with 1 - b^2 as (1 - b)(1 + b)."""
        w = self.gate[...] / self.temperature
        b = jax.nn.sigmoid(w)
        return jnp.sqrt(jnp.maximum(jax.nn.sigmoid(-w) * (1 + b), self.eps)), b

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        a, b = self.weights()
        return stream(a * x + b * self.body(x, taps))


Reader = HyperConnections | ManifoldHyperConnections | GatedResidual | AttentionResidual


class Merge(nnx.Module):
    """Reads the widened stream back into one: the read a branch of the algebra does,
    without the write, with parameters of its own. Single-pass, without a reader, it
    reads with the coefficients the last branch computed, as DeepSeek-V4.1's head does.
    """

    def __init__(self, reader: Reader | None):
        self.reader = reader

    def __call__(self, x: Float[Array, "b t s d"], taps: Taps) -> Float[Array, "b t d"]:
        if self.reader is None:
            return jnp.einsum("btn,btnd->btd", _pending(taps, x), x)
        return self.reader.read(x)


def _algebra(
    residual: cfg.ResidualConfig | None,
    ctx: Context,
    index: int = 0,
    body: nnx.Module | None = None,
) -> Reader:
    d = ctx.arch.d_model
    match residual:
        case cfg.HyperConnectionsConfig(type="hc"):
            return HyperConnections(residual, d, index, body)
        case cfg.HyperConnectionsConfig():
            return ManifoldHyperConnections(residual, d, body)
        case cfg.GatedResidualConfig():
            return GatedResidual(residual, d, ctx.rngs, body)
        case cfg.AttentionResidualConfig():
            return AttentionResidual(residual, d, body)
    raise ValueError("merge reads a widened residual stream, but there is none")


def branch(residual: cfg.ResidualConfig, body: nnx.Module, ctx: Context) -> nnx.Module:
    """The algebra's residual branch around `body`, the next one at depth 0."""
    index = ctx.branches
    ctx.branches += 1
    if isinstance(residual, cfg.ResidualGateConfig):
        return ResidualGate(residual, ctx.arch.d_model, index, body)
    return _algebra(residual, ctx, index, body)


@register("merge")
def _(
    c: cfg.MergeConfig, width: ir.Width, ports: tuple[ir.Port, ...], ctx: Context
) -> Merge:
    residual = ctx.arch.residual
    if isinstance(residual, cfg.HyperConnectionsConfig) and residual.single_pass:
        return Merge(None)
    return Merge(_algebra(residual, ctx))
