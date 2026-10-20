"""Mixture of experts, with shared experts and LatentMoE projections."""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux.architecture.functional import activation, bound
from bux.architecture.mlp import MLP
from bux.architecture.norm import Norm
from bux.architecture.registry import Context, Taps, register
from bux.lang import ir


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


class MoE(nnx.Module):
    """Dense reference: every expert runs on every token, weighted by the router."""

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
        self.router = nnx.Linear(width, e + sinks, use_bias=False, rngs=rngs)
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
        self.groups, self.top_k_groups = c.num_groups, c.top_k_groups
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

    def _route(
        self, x: Float[Array, "b t d"]
    ) -> tuple[Float[Array, "b t e"], Float[Array, "b t s"] | None]:
        """Every routed expert's weight for each token, 0 outside its top-k, and the
        shared experts' weights if the router scores them too."""
        logits = self.router(x) * self.router_logit_scale
        match self.router_fn:
            case "softmax":
                scores = jax.nn.softmax(logits)
            case "sigmoid":
                scores = jax.nn.sigmoid(logits)
            case "sqrtsoftplus":
                scores = jnp.sqrt(jax.nn.softplus(logits))
        e = self.up.shape[0]
        scores, sinks = scores[..., :e], scores[..., e:]
        choice = scores if self.router_bias is None else scores + self.router_bias[...]
        if self.groups > 1:
            choice = _limit_groups(choice, self.groups, self.top_k_groups)
        _, idx = jax.lax.top_k(choice, self.top_k)
        weights = jnp.take_along_axis(scores, idx, axis=-1)
        if self.normalize:
            total = weights.sum(-1, keepdims=True) + sinks.sum(-1, keepdims=True)
            weights, sinks = weights / total, sinks / total
        dense = (jax.nn.one_hot(idx, e) * weights[..., None]).sum(-2) * self.scaling
        return dense, sinks * self.scaling if sinks.shape[-1] else None

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        dense, sinks = self._route(x)
        z = x if self.to_latent is None else self.to_latent(x)
        if self.latent_input_norm is not None:
            z = self.latent_input_norm(z)
        h = bound(
            jnp.einsum("btd,edf->btef", z, self.up[...]), self.limit, self.up_softcap
        )
        if self.gate is not None:
            gate = jnp.einsum("btd,edf->btef", z, self.gate[...])
            if self.limit is not None:
                gate = jnp.minimum(gate, self.limit)
            h = activation(self.activation, gate) * h
        else:
            h = activation(self.activation, h)
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
            if sinks is None:
                s = self.shared(x)
            else:  # each shared expert's part of the hidden units, by its weight
                h = self.shared.hidden(x)
                h = h * jnp.repeat(sinks, h.shape[-1] // sinks.shape[-1], axis=-1)
                s = self.shared.dropout(self.shared.down(h))
            if self.shared_gate is not None:
                s = s * jax.nn.sigmoid(self.shared_gate(x))
            y = y + s
        return y


@register("moe")
def _(c: cfg.MoEConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context) -> MoE:
    return MoE(c, width, ctx.rngs)
