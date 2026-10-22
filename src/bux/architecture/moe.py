"""Mixture of experts, with shared experts and LatentMoE projections.

The router picks each token's top-k experts, and a method of ``methods.moe`` runs
them (docs/parallelism.md): "dense" runs every expert on every token, the
reference; "sorted", the default, sorts the picks by expert and runs each expert on
its own rows (``experts``), dropping none.
"""

from collections.abc import Callable
from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux import parallel
from bux.architecture.functional import activation, bound
from bux.architecture.grouped import ragged_dot
from bux.architecture.mlp import MLP
from bux.architecture.norm import Norm
from bux.architecture.registry import Context, Taps, register, report, training
from bux.lang import ir
from bux.parallel import methods

Rows = Callable[[Float[Array, "n d"]], Float[Array, "n d"]]  # a norm of rows
# The experts' kernels, stacked: up, gate (None for an mlp) and down.
Kernels = tuple[
    Float[Array, "e d f"], Float[Array, "e d f"] | None, Float[Array, "e f d"]
]


def _limit_groups(
    choice: Float[Array, "b t e"], groups: int, keep: int
) -> Float[Array, "b t e"]:
    """The scores, at -inf for the experts outside the `keep` best of `groups` groups,
    each scored by the sum of its best two (DeepSeek-V3)."""
    b, t, e = choice.shape
    grouped = choice.reshape(b, t, groups, e // groups)
    best = jax.lax.top_k(grouped, min(2, e // groups))[0].sum(-1)
    kept = jax.nn.one_hot(jax.lax.top_k(best, keep)[1], groups).sum(-2) > 0
    return jnp.where(kept[..., None], grouped, -jnp.inf).reshape(b, t, e)


def balance_loss(
    scores: Float[Array, "b t e"], picks: Float[Array, "b t e"], k: int, sequence: bool
) -> Float[Array, ""]:
    """E * sum_e f_e P_e: f_e the fraction of the top-k `picks` that are expert e, P_e
    its mean score normalized over the experts; over the batch, or the mean of it over
    sequences (DeepSeek-V3, eqs. 17-20). 1 when both are uniform."""
    over = 1 if sequence else (0, 1)
    f = picks.mean(over) / k
    p = (scores / scores.sum(-1, keepdims=True)).mean(over)
    return scores.shape[-1] * (f * p).sum(-1).mean()


def mix(
    idx: Int[Array, "*n k"], weights: Float[Array, "*n k"], e: int
) -> Float[Array, "*n e"]:
    """Every expert's weight for each token, 0 outside its top-k picks `idx`."""
    return (jax.nn.one_hot(idx, e, dtype=weights.dtype) * weights[..., None]).sum(-2)


def hidden(
    up: Float[Array, "*n f"], gate: Float[Array, "*n f"] | None, ffn: cfg.MLPConfig
) -> Float[Array, "*n f"]:
    """What an expert's down projection takes, from its up and gate projections."""
    up = bound(up, ffn.limit, ffn.up_softcap)
    if gate is None:
        return activation(ffn.activation, up)
    if ffn.limit is not None:
        gate = jnp.minimum(gate, ffn.limit)
    return activation(ffn.activation, gate) * up


def experts(
    x: Float[Array, "n d"],
    sizes: Int[Array, " e"],
    up: Float[Array, "e d f"],
    gate: Float[Array, "e d f"] | None,
    down: Float[Array, "e f d"],
    ffn: cfg.MLPConfig,
    norm: Rows | None = None,
) -> Float[Array, "n d"]:
    """Each row of `x` through its expert's FFN, the rows sorted by expert: the first
    sizes[0] are expert 0's, the next sizes[1] expert 1's, and so on; rows past them
    come out 0, before `norm`. A function of its arguments alone, so that it runs on
    one device's experts in a shard_map as it does on all of them."""
    g = None if gate is None else ragged_dot(x, gate, sizes)
    y = ragged_dot(hidden(ragged_dot(x, up, sizes), g, ffn), down, sizes)
    return y if norm is None else norm(y)


def by_expert(
    idx: Int[Array, "*n k"], e: int
) -> tuple[Int[Array, " m"], Int[Array, " e"]]:
    """The order that sorts the picks `idx`, flattened, by expert (each expert's in
    the order of its tokens), and how many picks each of the `e` experts has. Picks
    outside [0, e) sort last and count for none."""
    flat = idx.reshape(-1)
    flat = jnp.where((flat >= 0) & (flat < e), flat, e)
    return jnp.argsort(flat, stable=True), jnp.bincount(flat, length=e + 1)[:e]


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def pick_rows(
    z: Float[Array, "n d"], order: Int[Array, " m"], back: Int[Array, " m"], k: int
) -> Float[Array, "m d"]:
    """z[order // k]: the rows of the picks `order`, a permutation of the m = n k
    picks, k of each row of z, which `back` undoes. Its gradient gathers the picks
    back by `back` and sums each row's k, where XLA's would scatter-add, 4 times as
    slow on a B300."""
    return z[order // k]


def _pick_rows_forward(
    z: Float[Array, "n d"], order: Int[Array, " m"], back: Int[Array, " m"], k: int
) -> tuple[Float[Array, "m d"], Int[Array, " m"]]:
    return z[order // k], back


def _pick_rows_backward(
    k: int, back: Int[Array, " m"], g: Float[Array, "m d"]
) -> tuple[Float[Array, "n d"], None, None]:
    rows = g[back].reshape(-1, k, g.shape[-1])
    return jnp.sum(rows, 1, dtype=jnp.float32).astype(g.dtype), None, None


pick_rows.defvjp(_pick_rows_forward, _pick_rows_backward)


@jax.custom_vjp
def permuted(
    x: Float[Array, "m d"], order: Int[Array, " m"], back: Int[Array, " m"]
) -> Float[Array, "m d"]:
    """x[order], for a permutation `order` that `back` undoes: its gradient gathers
    by `back`, where XLA's would scatter-add."""
    return x[order]


def _permuted_forward(
    x: Float[Array, "m d"], order: Int[Array, " m"], back: Int[Array, " m"]
) -> tuple[Float[Array, "m d"], Int[Array, " m"]]:
    return x[order], back


def _permuted_backward(
    back: Int[Array, " m"], g: Float[Array, "m d"]
) -> tuple[Float[Array, "m d"], None, None]:
    return g[back], None, None


permuted.defvjp(_permuted_forward, _permuted_backward)


def routed(
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    up: Float[Array, "e d f"],
    gate: Float[Array, "e d f"] | None,
    down: Float[Array, "e f d"],
    ffn: cfg.MLPConfig,
    norm: Rows | None = None,
) -> Float[Array, "b t d"]:
    """Each token's sum of its experts' outputs by their `weights`, the picks `idx`
    sorted by expert, through ``experts``, and back. A pick outside the experts of
    `up` adds nothing, so that one device's experts can take every pick."""
    b, t, d = z.shape
    k, e = idx.shape[-1], up.shape[0]
    order, sizes = by_expert(idx, e)
    back = jnp.argsort(order)
    rows = pick_rows(z.reshape(b * t, d), order, back, k)
    out = experts(rows, sizes, up, gate, down, ffn, norm)
    each = permuted(out, back, order).reshape(b, t, k, d)
    mine = jnp.where((idx >= 0) & (idx < e), weights, 0)
    return jnp.einsum("btkd,btk->btd", each, mine)


class MoE(nnx.Module):
    """Token-choice top-k experts, run by the run's method of ``methods.moe``."""

    muon_axes: ClassVar = {"up": (1, 2), "gate": (1, 2), "down": (1, 2)}

    def __init__(self, c: cfg.MoEConfig, width: int, rngs: nnx.Rngs):
        e, f = c.num_experts, c.expert.intermediate_size
        if e % c.num_groups or c.top_k > c.top_k_groups * (e // c.num_groups):
            raise ValueError(
                f"{c.num_groups} groups of {e} experts, {c.top_k_groups} of them "
                f"picked, cannot hold the top {c.top_k}"
            )
        init = nnx.initializers.lecun_normal()
        # With shared_expert_sink, the router scores the shared experts too, last.
        sinks = c.num_shared_experts if c.shared_expert_sink else 0
        self.router = _linear(width, e + sinks, ("embed", None), rngs)
        free = c.load_balancing == "aux_free"
        self.router_bias = nnx.Variable(jnp.zeros(e)) if free else None
        # The picks of each expert since the bias last moved; with `update_at_step_end`
        # set (by the trainer), the bias moves in `step_end_update`, once per step.
        self.router_load = nnx.Variable(jnp.zeros(e)) if free else None
        self.update_at_step_end = False
        self.bias_update_rate = c.bias_update_rate
        self.aux_loss_weight = c.aux_loss_weight
        self.per_sequence = c.aux_loss_scope == "sequence"
        # LatentMoE: the routed experts work between these two projections.
        latent = c.latent_size or width
        inner = "latent" if c.latent_size else "embed"  # the width the experts see
        self.to_latent = (
            _linear(width, latent, ("embed", "latent"), rngs) if c.latent_size else None
        )
        self.latent_input_norm = (
            Norm(c.latent_input_norm, latent) if c.latent_input_norm else None
        )
        self.latent_output_norm = (
            Norm(c.latent_output_norm, latent) if c.latent_output_norm else None
        )
        self.expert_norm = Norm(c.expert_norm, latent) if c.expert_norm else None
        self.from_latent = (
            _linear(latent, width, ("latent", "embed"), rngs) if c.latent_size else None
        )
        into = ("experts", inner, "mlp")
        self.up = nnx.Param(init(rngs.params(), (e, latent, f)), names=into)
        self.gate = (
            nnx.Param(init(rngs.params(), (e, latent, f)), names=into)
            if c.expert.type == "gated_mlp"
            else None
        )
        out = ("experts", "mlp", inner)
        self.down = nnx.Param(init(rngs.params(), (e, f, latent)), names=out)
        self.num_experts, self.top_k, self.router_fn = e, c.top_k, c.router
        self.groups, self.top_k_groups = c.num_groups, c.top_k_groups
        self.router_logit_scale = c.router_logit_scale
        self.normalize, self.scaling = c.normalize_top_k, c.routed_scaling_factor
        self.ffn = c.expert  # what each routed expert computes
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
            _linear(width, 1, ("embed", None), rngs) if c.shared_expert_gate else None
        )

    def _route(
        self, x: Float[Array, "b t d"], taps: Taps | None = None
    ) -> tuple[
        Int[Array, "b t k"], Float[Array, "b t k"], Float[Array, "b t s"] | None
    ]:
        """Each token's top-k experts and their weights, and the shared experts'
        weights if the router scores them too; in a training forward, it also
        balances the experts. Scores and top-k are in float32, the weights in x's
        dtype."""
        logits = self.router(x.astype(jnp.float32)) * self.router_logit_scale
        match self.router_fn:
            case "softmax":
                scores = jax.nn.softmax(logits)
            case "sigmoid":
                scores = jax.nn.sigmoid(logits)
            case "sqrtsoftplus":
                scores = jnp.sqrt(jax.nn.softplus(logits))
        scores, sinks = scores[..., : self.num_experts], scores[..., self.num_experts :]
        choice = scores if self.router_bias is None else scores + self.router_bias[...]
        if self.groups > 1:
            choice = _limit_groups(choice, self.groups, self.top_k_groups)
        _, idx = jax.lax.top_k(choice, self.top_k)
        if training(taps):
            self._balance(scores, idx, taps)
        weights = jnp.take_along_axis(scores, idx, axis=-1)
        if self.normalize:
            total = weights.sum(-1, keepdims=True) + sinks.sum(-1, keepdims=True)
            weights, sinks = weights / total, sinks / total
        weights, sinks = (w.astype(x.dtype) * self.scaling for w in (weights, sinks))
        return idx, weights, sinks if sinks.shape[-1] else None

    def _balance(
        self, scores: Float[Array, "b t e"], idx: Int[Array, "b t k"], taps: Taps
    ) -> None:
        """Report the balance loss, and count the picks for the router bias, which
        moves now unless `update_at_step_end`."""
        picks = jax.nn.one_hot(idx, scores.shape[-1]).sum(-2)
        if self.aux_loss_weight:
            loss = balance_loss(scores, picks, self.top_k, self.per_sequence)
            report(taps, "balance_loss", self.aux_loss_weight, loss)
        if self.router_load is not None:
            self.router_load[...] = self.router_load[...] + picks.sum((0, 1))
            if not self.update_at_step_end:
                self.step_end_update()

    def step_end_update(self) -> None:
        """Move the router bias toward the experts picked less than the mean since it
        last moved (DeepSeek-V3), and count again from zero."""
        if self.router_bias is None or self.router_load is None:
            return
        load = self.router_load[...]
        step = self.bias_update_rate * jnp.sign(load.mean() - load)
        self.router_bias[...] = self.router_bias[...] + step
        self.router_load[...] = jnp.zeros_like(load)

    def kernels(self) -> Kernels:
        """The routed experts' kernels, as arrays."""
        gate = None if self.gate is None else self.gate[...]
        return self.up[...], gate, self.down[...]

    def _latent(self, x: Float[Array, "b t d"]) -> Float[Array, "b t l"]:
        """The tokens as the routed experts see them."""
        z = x if self.to_latent is None else self.to_latent(x)
        return z if self.latent_input_norm is None else self.latent_input_norm(z)

    def _shared(
        self, x: Float[Array, "b t d"], sinks: Float[Array, "b t s"] | None
    ) -> Float[Array, "b t d"]:
        """The shared experts' output; if the router scores them, each one's part of
        the hidden units weighted by its `sinks`."""
        assert self.shared is not None
        if sinks is None:
            s = self.shared(x)
        else:
            h = self.shared.hidden(x)
            h = h * jnp.repeat(sinks, h.shape[-1] // sinks.shape[-1], axis=-1)
            s = self.shared.dropout(self.shared.down(h))
        if self.shared_gate is not None:
            s = s * jax.nn.sigmoid(self.shared_gate(x))
        return s

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        idx, weights, sinks = self._route(x, taps)
        y = parallel.method("moe")(self, self._latent(x), idx, weights)
        if self.latent_output_norm is not None:
            y = self.latent_output_norm(y)
        if self.from_latent is not None:
            y = self.from_latent(y)
        return y if self.shared is None else y + self._shared(x, sinks)


def _linear(
    i: int, o: int, names: tuple[str | None, str | None], rngs: nnx.Rngs
) -> nnx.Linear:
    """A linear map with no bias, its kernel's dimensions `names`."""
    init = parallel.named(*names)  # nnx.Linear's lecun_normal
    return nnx.Linear(i, o, use_bias=False, kernel_init=init, rngs=rngs)


# --- Methods: (moe, z, idx, weights) -> the routed sum, in the experts' width -------


@methods.moe.register("dense")
def run_dense(
    moe: MoE,
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
) -> Float[Array, "b t d"]:
    """Every expert on every token, weighted by the router: the reference. Each
    device runs every expert on its own tokens."""
    up, gate, down = moe.kernels()
    dense = mix(idx, weights, moe.num_experts)

    def project(w: Float[Array, "e d f"]) -> Float[Array, "b t e f"]:
        h = jnp.einsum("btd,edf->btef", z, w)
        return parallel.shard(h, "batch", "seq", "experts", "mlp")

    h = hidden(project(up), None if gate is None else project(gate), moe.ffn)
    if moe.expert_norm is None:
        return jnp.einsum("btef,efd,bte->btd", h, down, dense)
    each = moe.expert_norm(jnp.einsum("btef,efd->bted", h, down))
    return jnp.einsum("bted,bte->btd", each, dense)


@methods.moe.register("sorted")
def run_sorted(
    moe: MoE,
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
) -> Float[Array, "b t d"]:
    """Each expert on its own rows (``routed``), each device on its own tokens with
    every expert's kernels: a shard_map over the axes that split the tokens, so that
    no device sorts another's."""

    def local(
        z: Float[Array, "b t d"],
        idx: Int[Array, "b t k"],
        weights: Float[Array, "b t k"],
        kernels: Kernels,
    ) -> Float[Array, "b t d"]:
        return routed(z, idx, weights, *kernels, moe.ffn, moe.expert_norm)

    tokens = parallel.spec(("batch", "seq", None), z.shape)
    if not _axes(tokens):
        return local(z, idx, weights, moe.kernels())
    # Untyped by what varies: tokamax's kernel takes jnp.searchsorted, which fails in
    # a shard_map on jax 0.12 where it is typed; nothing here communicates.
    return jax.shard_map(
        local,
        in_specs=(tokens, tokens, tokens, P()),
        out_specs=tokens,
        axis_names=_axes(tokens),
        check_vma=False,
    )(z, idx, weights, moe.kernels())


def _axes(spec: P) -> set[str]:
    """The mesh axes `spec` splits over."""
    return {a for s in spec if s for a in ((s,) if isinstance(s, str) else s)}


@register("moe")
def _(c: cfg.MoEConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context) -> MoE:
    return MoE(c, width, ctx.rngs)
