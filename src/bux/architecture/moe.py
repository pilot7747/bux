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

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
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
            s = self.shared(x)
            if self.shared_gate is not None:
                s = s * jax.nn.sigmoid(self.shared_gate(x))
            y = y + s
        return y


@register("moe")
def _(c: cfg.MoEConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context) -> MoE:
    return MoE(c, width, ctx.rngs)
