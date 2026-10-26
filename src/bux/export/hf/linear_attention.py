"""Linear attention, exported: bux's ``gdn``, ``kda``, ``deltanet``, ``gla`` and
``lightning``, and the recurrences ``rwkv7`` and the state space models run too.

Each runs bux's recurrence over a state S of dk by dv per value head::

    S_t = diag(w_t) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T,    o_t = S_t^T q_t

in float32. A prefill runs it by chunks of `CHUNK` positions, with matmuls within a
chunk and a loop across them (`delta_rule`, `decayed`, the form of fla's chunked
kernels); a decoding step runs one position (`recurrence`). Between calls, the state
and the short conv's last inputs stay in the layer's cache slot, a
``linear_attention`` layer. A padded token of a left-padded batch leaves the state as
it was: its input is zeroed, and its decay is 1 (`hold`).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .common import ACTIVATIONS, Ctx, Norm, Site
from .conv import CausalConv, keep_state, previous_state, real_tokens, without_padding

CHUNK = 64  # positions per chunk of a prefill


def rmsnorm(eps: float) -> dict:
    """Return the config of an RMSNorm with a gain, as a layer's own norm has it."""
    return {
        "type": "rmsnorm",
        "eps": eps,
        "affine": True,
        "bias": False,
        "zero_centered": False,
        "scale_input": False,
    }


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return x over its L2 norm as bux and fla compute it, ``x / sqrt(sum(x^2) +
    eps)``, in float32; the result in x's dtype."""
    y = x.float()
    return (y * torch.rsqrt(y.square().sum(-1, keepdim=True) + eps)).to(x.dtype)


def hold(keep: torch.Tensor | None, *xs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return each x, b t ..., zeroed at the padded tokens (`keep` 0): what a token
    writes to the state, and its log decay, so that it leaves the state as it was."""
    if keep is None:
        return xs
    return tuple(
        x * keep.view(*keep.shape, *(1,) * (x.ndim - 2)).to(x.dtype) for x in xs
    )


def recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    state: torch.Tensor | None = None,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run bux's recurrence one position at a time, from `state`, in float32.

    Args:
        q: The queries, b t h dk.
        k: The keys, b t h dk.
        v: The values, b t h dv.
        log_decay: log w, b t h dk, or b t h 1 for one decay a head.
        state: S before the first position, b h dk dv; None for zero.
        a: What the rank-one term reads of the state, b t h dk; None for none.
        b: Where the rank-one term writes it, b t h dk.

    Returns:
        The outputs, b t h dv in v's dtype, and the last state, in float32.
    """
    q, k, w, vs = q.float(), k.float(), log_decay.float().exp(), v.float()
    s = _zero_state(k, v) if state is None else state.float()
    outs = []
    for t in range(k.shape[1]):
        new = w[:, t, ..., None] * s
        if a is not None and b is not None:
            read = torch.einsum("bhk,bhkv->bhv", a[:, t].float(), s)
            new = new + b[:, t, ..., None].float() * read[..., None, :]
        s = new + k[:, t, ..., None] * vs[:, t, ..., None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], s))
    return torch.stack(outs, 1).to(v.dtype), s


def delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the gated delta rule, bux's ``delta_rule``, from `state`::

        S_t = (I - beta_t k_t k_t^T) diag(w_t) S_{t-1} + beta_t k_t v_t^T,

    one position as `recurrence` runs it (``a = -w k``, ``b = beta k``), more by
    chunks. Shapes as `recurrence`'s, and beta b t h; returns its outputs and state.
    """
    if q.shape[1] == 1:
        step = beta.float()[..., None] * k.float()
        a = -log_decay.float().exp() * k.float()
        return recurrence(q, step, v, log_decay, state, a, step)
    return _by_chunks(_delta_chunk, (q, k, v, log_decay, beta), v, state)


def decayed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run decayed linear attention, ``S_t = diag(w_t) S_{t-1} + k_t v_t^T``, from
    `state`: one position as `recurrence` runs it, more by chunks. Shapes as
    `recurrence`'s; returns its outputs and state."""
    if q.shape[1] == 1:
        return recurrence(q, k, v, log_decay, state)
    return _by_chunks(_decayed_chunk, (q, k, v, log_decay), v, state)


def _zero_state(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Return the state of no positions, b h dk dv, in float32."""
    b, _, h, dk = k.shape
    return torch.zeros(b, h, dk, v.shape[-1], dtype=torch.float32, device=k.device)


def _chunked(x: torch.Tensor) -> torch.Tensor:
    """Return x, b t h ..., as b h n c ... in float32: n chunks of c positions, the
    last padded with zeros, which leave a state as it was."""
    x = x.float().transpose(1, 2)
    x = F.pad(x, (0, 0) * (x.ndim - 3) + (0, -x.shape[2] % CHUNK))
    return x.reshape(*x.shape[:2], -1, CHUNK, *x.shape[3:])


def _by_chunks(
    chunk: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    xs: tuple[torch.Tensor, ...],
    v: torch.Tensor,
    state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `chunk` over the chunks of the positions `xs`, carrying the state."""
    t = v.shape[1]
    parts = [_chunked(x) for x in xs]
    s = _zero_state(xs[1], v) if state is None else state.float()
    outs = []
    for i in range(parts[0].shape[2]):
        out, s = chunk(s, *(x[:, :, i] for x in parts))
        outs.append(out)
    out = torch.stack(outs, 2).flatten(2, 3)[:, :, :t].transpose(1, 2)
    return out.to(v.dtype), s


def _pairwise(G: torch.Tensor) -> torch.Tensor:
    """Return the decays ``exp(G_t - G_s)`` from position s to t of a chunk, b h c c
    e, 0 where s is after t, from the cumulative log decays G, b h c e. No exponent
    is positive."""
    c = G.shape[-2]
    after = torch.ones(c, c, dtype=torch.bool, device=G.device).triu(1)
    diff = G[..., :, None, :] - G[..., None, :, :]
    return diff.masked_fill(after[..., None], float("-inf")).exp()


def _inner(x: torch.Tensor, y: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Return ``sum_d x_t[d] y_s[d] D[t, s, d]`` for every two positions of a chunk,
    b h c c; D of one decay a head is ``... 1``."""
    if D.shape[-1] == 1:
        return (x @ y.transpose(-1, -2)) * D[..., 0]
    return torch.einsum("bhtd,bhsd,bhtsd->bhts", x, y, D)


def _carry(
    s: torch.Tensor, k: torch.Tensor, G: torch.Tensor, u: torch.Tensor
) -> torch.Tensor:
    """Return the state a chunk ends with, ``diag(exp(G_c)) S + sum_s diag(exp(G_c -
    G_s)) k_s u_s^T``, from the state it starts with and what it writes."""
    last = G[..., -1:, :]
    return (
        last.exp().transpose(-1, -2) * s + (k * (last - G).exp()).transpose(-1, -2) @ u
    )


def _decayed_chunk(
    s: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `decayed` over one chunk from the state s: its outputs and last state."""
    G = g.cumsum(-2)
    D = _pairwise(G)
    out = (q * G.exp()) @ s + _inner(q, k, D) @ v
    return out, _carry(s, k, G, v)


def _delta_chunk(
    s: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `delta_rule` over one chunk from the state s: its outputs and last state.

    What each position writes, ``u_t = beta_t (v_t - S'^T k_t)`` for the decayed
    state S' before it, solves ``(I + diag(beta) L) U = diag(beta) (V - (K exp(G))
    S)``, with ``L[t, s] = k_t^T diag(exp(G_t - G_s)) k_s`` below the diagonal: the
    WY form of fla and transformers.
    """
    G = g.cumsum(-2)
    D = _pairwise(G)
    kb = k * beta[..., None]
    eye = torch.eye(k.shape[-2], dtype=k.dtype, device=k.device)
    system = _inner(kb, k, D).tril(-1) + eye
    rhs = torch.cat([v * beta[..., None], kb * G.exp()], -1)
    solved = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
    w, kc = solved.split([v.shape[-1], k.shape[-1]], -1)
    u = w - kc @ s
    out = (q * G.exp()) @ s + _inner(q, k, D) @ u
    return out, _carry(s, k, G, u)


class LowRank(nn.Module):
    """bux's ``LowRank``: ``act(x A) B + c``, through `rank` dims.

    Args:
        n_in: The width of the input.
        rank: The dims between A and B.
        n_out: The width of the output.
        activation: The activation between, by name; None for none.
        bias: Whether to add a bias c.
    """

    def __init__(
        self,
        n_in: int,
        rank: int,
        n_out: int,
        activation: str | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.down = nn.Parameter(torch.zeros(n_in, rank))
        self.up = nn.Parameter(torch.zeros(rank, n_out))
        self.bias = nn.Parameter(torch.zeros(n_out)) if bias else None
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.down
        if self.activation is not None:
            h = ACTIVATIONS[self.activation](h)
        y = h @ self.up
        return y if self.bias is None else y + self.bias


class LinearAttention(nn.Module):
    """What bux's linear-attention layers share (``LinearAttention``): the q, k and v
    projections, the short conv on them (one over the three side by side, which is
    the same as each on its own), and the output's norm, gate and projection. Each
    q/k head serves a group of value heads next to each other.

    Args:
        c: The layer's config, as JSON.
        width: The width of the stream.
        silu: SiLU q, k and v without a conv; after one, they always are.
        gate_rank: The rank of a low-rank output gate; None for a dense one.
        norm_all_heads: Normalize all heads' outputs together, not each on its own.
    """

    @staticmethod
    def cache_type(c: dict) -> str:
        """The cache layer it keeps, as ``config.layer_types`` names it."""
        return "linear_attention"

    def __init__(
        self,
        c: dict,
        width: int,
        *,
        silu: bool = True,
        gate_rank: int | None = None,
        norm_all_heads: bool = False,
    ):
        super().__init__()
        h, hv = c["num_heads"], c["num_v_heads"] or c["num_heads"]
        dk, dv = c["head_dim"], c["v_head_dim"] or c["head_dim"]
        self.heads, self.v_heads, self.head_dim, self.v_head_dim = h, hv, dk, dv
        self.q = nn.Linear(width, h * dk, bias=False)
        self.k = nn.Linear(width, h * dk, bias=False)
        self.v = nn.Linear(width, hv * dv, bias=False)
        size = c["conv_size"]
        self.conv = CausalConv(2 * h * dk + hv * dv, size, False) if size else None
        self.silu = silu
        self.gate_activation = c["output_gate"]
        self.gate: nn.Module | None = None
        if self.gate_activation is not None:
            self.gate = (
                LowRank(width, gate_rank, hv * dv)
                if gate_rank
                else nn.Linear(width, hv * dv, bias=False)
            )
        self.norm_all_heads = norm_all_heads
        self.norm = Norm(rmsnorm(c["norm_eps"]), hv * dv if norm_all_heads else dv)
        self.o = nn.Linear(hv * dv, width, bias=False)

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        slot = ctx.cache_slot()
        x = without_padding(x, ctx)
        q, k, v = self.qkv(x, ctx, slot)
        keep = real_tokens(ctx, x.shape[1])
        o, state = self.mix(x, q, k, v, keep, previous_state(ctx, slot))
        keep_state(ctx, slot, state)
        return self.output(x, o)

    def mix(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        keep: torch.Tensor | None,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the heads' outputs, b t h dv, and the state they leave, from the
        state before (None for zero) and the tokens to `keep` (None for all)."""
        raise NotImplementedError

    def qkv(
        self, x: torch.Tensor, ctx: Ctx, slot: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project, convolve and SiLU the queries, keys and values, and split them
        into heads, q and k shared out to the value heads."""
        b, t, _ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        if self.conv is not None:
            sizes = [q.shape[-1], k.shape[-1], v.shape[-1]]
            qkv = F.silu(self.conv(torch.cat([q, k, v], -1), ctx, slot))
            q, k, v = qkv.split(sizes, -1)
        elif self.silu:
            q, k, v = F.silu(q), F.silu(k), F.silu(v)
        group = self.v_heads // self.heads
        q, k = (
            y.reshape(b, t, self.heads, self.head_dim).repeat_interleave(group, 2)
            for y in (q, k)
        )
        return q, k, v.reshape(b, t, self.v_heads, self.v_head_dim)

    def output(self, x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        """Normalize and gate the heads' outputs, and project them back."""
        b, t = o.shape[:2]
        o = self.norm(o.reshape(b, t, -1) if self.norm_all_heads else o)
        o = o.reshape(b, t, -1)
        if self.gate is not None:
            o = o * ACTIVATIONS[self.gate_activation](self.gate(x))
        return self.o(o)


class DeltaNet(LinearAttention):
    """bux's ``deltanet``: the delta rule on L2-normalized q and k, q scaled by
    ``dk ** -0.5``, a step ``beta = sigmoid(W_b x)`` per value head, and no decay."""

    bux_types = ("deltanet",)

    def __init__(
        self,
        c: dict,
        width: int,
        site: Site | None = None,
        gate_rank: int | None = None,
    ):
        super().__init__(c, width, gate_rank=gate_rank)
        self.b = nn.Linear(width, self.v_heads, bias=False)

    def mix(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        keep: torch.Tensor | None,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k = l2norm(q) * self.head_dim**-0.5, l2norm(k)
        beta, g = hold(keep, torch.sigmoid(self.b(x)), self.log_decay(x))
        return delta_rule(q, k, v, g, beta, state)

    def log_decay(self, x: torch.Tensor) -> torch.Tensor:
        """The log decay of the state, b t h 1 or b t h dk, in float32: 0 here."""
        b, t, _ = x.shape
        return torch.zeros(b, t, self.v_heads, 1, device=x.device)


class GatedDeltaNet(DeltaNet):
    """bux's ``gdn``: DeltaNet with a decay ``exp(g)`` per value head,
    ``g = -exp(A_log) softplus(W_a x + dt_bias)``."""

    bux_types = ("gdn",)

    def __init__(self, c: dict, width: int, site: Site | None = None):
        super().__init__(c, width)
        self.a = nn.Linear(width, self.v_heads, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.v_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.v_heads))

    def log_decay(self, x: torch.Tensor) -> torch.Tensor:
        rate = F.softplus(self.a(x).float() + self.dt_bias.float())
        return (-self.A_log.float().exp() * rate)[..., None]


class KimiDeltaAttention(DeltaNet):
    """bux's ``kda``: Gated DeltaNet with a decay per key channel,
    ``g = -exp(A_log) softplus(f + dt_bias)`` for a low-rank f, or
    ``lower_bound * sigmoid(exp(A_log) (f + dt_bias))``; its output gate low rank
    too, with ``output_gate_rank``."""

    bux_types = ("kda",)

    def __init__(self, c: dict, width: int, site: Site | None = None):
        super().__init__(c, width, gate_rank=c["output_gate_rank"])
        hv, dk = self.v_heads, self.head_dim
        self.f = LowRank(width, c["gate_rank"] or dk, hv * dk)
        self.lower_bound = c["gate_lower_bound"]
        self.A_log = nn.Parameter(torch.zeros(hv))
        self.dt_bias = nn.Parameter(torch.zeros(hv * dk))

    def log_decay(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        f = self.f(x).float() + self.dt_bias.float()
        f = f.reshape(b, t, self.v_heads, self.head_dim)
        rate = self.A_log.float().exp()[:, None]
        if self.lower_bound is not None:
            return self.lower_bound * torch.sigmoid(rate * f)
        return -rate * F.softplus(f)


class GatedLinearAttention(LinearAttention):
    """bux's ``gla``: linear attention with q scaled by ``dk ** -0.5`` and a decay per
    key channel, ``log w = log sigmoid(f) / 16`` for a low-rank f with a bias."""

    bux_types = ("gla",)

    def __init__(self, c: dict, width: int, site: Site | None = None):
        super().__init__(c, width, silu=False)
        hv, dk = self.v_heads, self.head_dim
        self.f = LowRank(width, c["gate_rank"], hv * dk, bias=True)

    def mix(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        keep: torch.Tensor | None,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.f(x).float().reshape(*k.shape)
        k, g = hold(keep, k, F.logsigmoid(f) / 16)
        return decayed(q * self.head_dim**-0.5, k, v, g, state)


def slopes(heads: int, layer: int, layers: int) -> torch.Tensor:
    """Return MiniMax's decay rates, ``2 ** (-8 i / heads)`` for heads i = 1 to heads,
    scaled by ``1 - layer / (layers - 1)`` (and 1e-5 more), as bux's ``slopes``."""
    factor = 1 - layer / (layers - 1 + 1e-5) + 1e-5
    return 2 ** (-8 / heads * torch.arange(1, heads + 1, dtype=torch.float32)) * factor


class LightningAttention(LinearAttention):
    """bux's ``lightning``: linear attention on SiLU'd q, k and v, unscaled, with a
    fixed decay ``exp(-s)`` per head by the layer's depth among the model's token
    mixers (`slopes`; ``site.extra``, from `bux.export.plan.mixer_depth`), and one
    RMSNorm of all the heads together."""

    bux_types = ("lightning",)
    slopes: torch.Tensor

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__(c, width, norm_all_heads=True)
        self.depth = site.extra["layer"], site.extra["layers"]
        self.register_buffer(
            "slopes", slopes(self.v_heads, *self.depth), persistent=False
        )

    def reset(self) -> None:
        """Compute the rates again, which transformers' meta-device init leaves unset
        (``BuxPreTrainedModel._init_weights`` calls it)."""
        with torch.no_grad():
            self.slopes.copy_(slopes(self.v_heads, *self.depth))

    def mix(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        keep: torch.Tensor | None,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = (-self.slopes[:, None]).expand(*q.shape[:3], 1)
        k, g = hold(keep, k, g)
        return decayed(q, k, v, g, state)
