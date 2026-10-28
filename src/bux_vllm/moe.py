"""Mixtures of experts in vLLM: bux's ``moe``, its routed experts by vLLM's fused
MoE kernels (`FusedMoEFactory`), routed by bux's own rule.

The router runs as bux's does (`bux_vllm.routing`): vLLM takes the picks and
weights from it (``custom_routing_function``) and runs the experts. In LatentMoE
the experts run in the latent, as vLLM's fused MoE of the latent's width, between
the projections to and from it. Under ``expert_norm`` each expert's output is
normalized before the router weighs it: each of a token's picks runs as a row of its
own, which vLLM's fused MoE leaves unsummed (`bux_vllm.routing.apart`). Under tensor
parallelism each rank keeps every expert's share of the hidden units, as vLLM splits
them, and the projections whole. Zero experts (``num_zero_experts``) run as vLLM
runs LongCat-Flash's, by its ``ZeroExpertRouter``: vLLM's fused top-k with the
bias, then a Triton kernel that adds each token times its zero picks' weights,
after the routed experts' sum over the ranks. The routed experts, the shared
experts and the latent's projections take the engine's quantization config
(`bux_vllm.common.quantization`); the router, which the layer computes itself in
float32, and the shared experts' gate of one output stay in the model's dtype, as
vLLM keeps Qwen's.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.linear import ReplicatedLinear

from .basic import MLP
from .common import Ctx, Layer, Norm, Site, unsupported
from .routing import Router, apart, expert_shards, pick_rows, zero_split

#: vLLM's MoE activations, by a bux expert's type and activation, without a limit
#: or an offset.
ACTIVATIONS = {
    ("gated_mlp", "swish"): "silu",
    ("gated_mlp", "gelu"): "gelu",
    ("gated_mlp", "gelu_tanh"): "gelu_tanh",
    ("gated_mlp", "relu2"): "relu2",
    ("mlp", "swish"): "silu_no_mul",
    ("mlp", "gelu"): "gelu_no_mul",
    ("mlp", "gelu_tanh"): "gelu_tanh_no_mul",
    ("mlp", "relu2"): "relu2_no_mul",
}

#: The gate's activation ``x sigmoid(alpha x)``, by its name, of vLLM's clamped
#: SwiGLU: ``act(min(gate, limit)) * (clip(up, limit) + offset)``.
SIGMOID_GATES = {"swish": 1.0, "quick_gelu": 1.702}
FLOAT32_MAX = 3.4028234e38  # a limit that clips nothing
#: bux's situ, ``4 tanh(x / 4) sigmoid(x)``: vLLM's SiTU of this beta.
SITU_BETA = 4.0


def experts_activation(expert: dict) -> dict:
    """vLLM's activation of the routed experts `expert` (an MLP's config), with its
    clamp and offset, or SiTU-GLU's soft cap of up, as `FusedMoEFactory`'s
    arguments.

    Raises:
        NotImplementedError: vLLM has no such activation.
    """
    kind = (expert["type"], expert["activation"])
    clamped = expert["limit"] is not None or expert["up_offset"]
    if kind == ("gated_mlp", "situ") and not clamped:
        return {
            "activation": "situ",  # Kimi's SiTU-GLU, the gate first as w13 holds it
            "activation_situ_beta": SITU_BETA,
            "activation_situ_linear_beta": expert["up_softcap"],
        }
    unsupported(expert, "up_softcap")
    if not clamped and kind in ACTIVATIONS:
        return {"activation": ACTIVATIONS[kind]}
    if kind[0] != "gated_mlp" or kind[1] not in SIGMOID_GATES:
        raise NotImplementedError(f"moe: {kind} experts, so clamped, not in vLLM yet")
    limit = FLOAT32_MAX if expert["limit"] is None else expert["limit"]
    return {
        "activation": "swigluoai_uninterleave",  # gate then up, as w13 holds them
        "swiglu_limit": limit,
        "swiglu_alpha": SIGMOID_GATES[kind[1]],
        "swiglu_beta": expert["up_offset"],
    }


class MoE(Layer):
    """bux's ``moe``: the router (bux's rule, `Router`), the routed experts by vLLM's
    fused kernels, in LatentMoE's latent (`latent_size`) with its norms, and the
    shared experts as one MLP, gated by a sigmoid under ``shared_expert_gate``, or
    each one's share of the hidden units weighted by its score under
    ``shared_expert_sink``; with ``expert_norm``, each routed expert's output
    normalized before its weight (`apart`); zero experts by vLLM's own
    (`zero_routing`), or by `zero_split` with ``expert_norm``. Training's settings
    (the balance loss, the bias's update, random logits) change nothing here."""

    bux_types = ("moe",)
    once = True
    sink_of: torch.Tensor | None

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        expert, self.top_k = c["expert"], c["top_k"]
        n, self.gated = c["num_experts"], expert["type"] == "gated_mlp"
        latent = c["latent_size"] or width
        self.num_experts, self.expert_norm = n, _norm(c["expert_norm"], latent)
        apart_ = self.expert_norm is not None  # each pick a row of its own
        self.to_latent, self.from_latent = _latent(c, width, site)
        self.latent_input_norm = _norm(c["latent_input_norm"], latent)
        self.latent_output_norm = _norm(c["latent_output_norm"], latent)
        self.router = Router(c)
        self.zero = c["num_zero_experts"]
        sinks = c["num_shared_experts"] if c["shared_expert_sink"] else 0
        scored = n + self.zero + sinks
        weight = torch.empty(scored, width)
        self.router_weight = nn.Parameter(weight, requires_grad=False)
        self.router_proj_bias = (
            nn.Parameter(torch.empty(scored), requires_grad=False)
            if c["router_bias"]
            else None
        )
        self.experts = FusedMoEFactory(
            num_experts=n,
            top_k=1 if apart_ else c["top_k"],
            hidden_size=latent,
            intermediate_size=expert["intermediate_size"],
            renormalize=c["normalize_top_k"],
            quant_config=site.quant,
            prefix=f"{site.path}.experts",
            **self._routing(c, latent, apart_),
            **experts_activation(expert),
            has_bias=expert["bias"],
            router_logits_dtype=torch.float32,
            # vLLM's names of the shards, which a non-gated MoE has no up of
            ckpt_names=("gate_proj", "down_proj", "up_proj" if self.gated else None),
        )
        self.shared = None
        if c["num_shared_experts"]:
            shared = c["shared_expert"] or expert
            size = shared["intermediate_size"] * c["num_shared_experts"]
            inner = Site(f"{site.path}.shared", site.vllm_config, site.config)
            self.shared = MLP(shared, width, inner, size=size)
        self.shared_gate = (  # unquantized, as vLLM's Qwen MoE keeps its own
            ReplicatedLinear(width, 1, bias=False, return_bias=False)
            if c["shared_expert_gate"]
            else None
        )
        sinks = _sink_of(c) if c["shared_expert_sink"] else None
        self.register_buffer("sink_of", sinks, persistent=False)

    def _routing(self, c: dict, width: int, apart_: bool) -> dict:
        """How vLLM's fused MoE routes: by bux's rule (`Router`), by one-hot rows
        (`pick_rows`) for `apart`, or by vLLM's own with zero experts."""
        if self.zero and not apart_:
            return self.zero_routing(c, width)
        return {"custom_routing_function": pick_rows if apart_ else self.router}

    def zero_routing(self, c: dict, width: int) -> dict:
        """`FusedMoEFactory`'s arguments that route by vLLM's ``ZeroExpertRouter``,
        LongCat-Flash's: its fused top-k of the scores plus the bias (a bias of 0
        without one) over the routed and zero experts, normalized and scaled, and
        its Triton kernel of the zero experts' output.

        Raises:
            NotImplementedError: Settings its top-k does not take, or experts whose
                width is no multiple of 256, the blocks its kernel covers whole.
        """
        unsupported(c, "shared_expert_sink")
        if width % 256:
            raise NotImplementedError(
                f"moe: zero experts {width} wide, no multiple of 256: vLLM's kernel of "
                "their output takes whole blocks of 256"
            )
        bias = self.router.bias
        if bias is None:
            bias = torch.zeros(c["num_experts"] + self.zero, dtype=torch.float32)
            self.register_buffer("zero_bias", bias, persistent=False)
        return {
            "custom_routing_function": None,
            "zero_expert_type": "identity",
            "e_score_correction_bias": bias,
            "scoring_func": c["router"],
            "routed_scaling_factor": c["routed_scaling_factor"],
        }

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        out = {"router.weight": (self.router_weight, ())}
        if self.router_proj_bias is not None:
            out["router.bias"] = (self.router_proj_bias, ())
        if self.router.bias is not None:
            out["router_bias"] = (self.router.bias, ())
        if self.shared is not None:
            out |= {f"shared.{k}": v for k, v in self.shared.targets().items()}
        for name in ("shared_gate", "to_latent", "from_latent"):
            if (module := getattr(self, name)) is not None:
                out[f"{name}.weight"] = (module.weight, ())
        for name in ("latent_input_norm", "latent_output_norm"):
            if (norm := getattr(self, name)) is not None:
                out |= {f"{name}.{k}": v for k, v in norm.targets().items()}
        if self.expert_norm is not None:  # the export's name of it
            norm = self.expert_norm.targets().items()
            out |= {f"experts.post_expert_norm.{k}": v for k, v in norm}
        return out

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        shards = expert_shards(name, tensor, self.gated)
        if shards is None:
            return super().load(name, tensor)
        routed = self.experts.routed_experts
        bias = name.endswith("bias")
        stem = "w2" if shards[0][0] == "w2" else "w13"
        param = getattr(routed, f"{stem}_{'bias' if bias else 'weight'}")
        rank = self.experts.moe_config.moe_parallel_config.tp_rank
        if bias and stem == "w2" and rank != 0:  # the ranks' outputs sum: one adds it
            shards = [(shard, torch.zeros_like(t)) for shard, t in shards]
        for shard, stacked in shards:
            for e in range(stacked.shape[0]):
                where = f"experts.{e}.{shard}.{'bias' if bias else 'weight'}"
                # The parameter's loader: vLLM's online quantization wraps it
                param.weight_loader(param, stacked[e].contiguous(), where, shard, e)
        return [param]

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        logits = F.linear(x.float(), self.router_weight.float())
        if self.router_proj_bias is not None:
            logits = logits + self.router_proj_bias.float()
        if self.zero and self.expert_norm is None:  # vLLM's top-k scores them
            logits = logits * self.router.c["router_logit_scale"]
        y = self.routed(x, logits)
        return y if self.shared is None else y + self._shared(x, logits)

    def _shared(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """The shared experts' output; where the router scores them, each one's share
        of the hidden units weighted by its score."""
        assert self.shared is not None
        if self.sink_of is None:
            y = self.shared(x)
        else:
            sinks = self.router.route(logits, self.top_k)[2].to(x.dtype)
            y = self.shared.down(self.shared.hidden(x) * sinks[..., self.sink_of])
        if self.shared_gate is not None:
            y = y * torch.sigmoid(self.shared_gate(x))
        return y

    def routed(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """The routed experts' output, by the router's `logits`: in LatentMoE, run
        on x projected to the latent and normalized, and projected back; each
        expert's normalized before its weight under ``expert_norm``."""
        z = x if self.to_latent is None else self.to_latent(x)
        if self.latent_input_norm is not None:
            z = self.latent_input_norm(z)
        if self.expert_norm is None:
            y = self.experts(z, logits)
        else:
            weights, idx, _ = self.router.route(logits, self.top_k)
            zero, weights, idx = zero_split(z, weights, idx, self.num_experts)
            y = apart(self.experts, self.expert_norm, z, weights, idx, self.num_experts)
            y = y + zero if self.zero else y
        if self.latent_output_norm is not None:
            y = self.latent_output_norm(y)
        return y if self.from_latent is None else self.from_latent(y)


def _latent(c: dict, width: int, site: Site) -> tuple[nn.Module | None, ...]:
    """LatentMoE's projections to the latent and back, whole on every rank, or
    none without a latent."""
    latent = c["latent_size"]
    if latent is None:
        return None, None
    path, kw = site.path, {"bias": False, "return_bias": False}
    kw["quant_config"] = site.quant
    to = ReplicatedLinear(width, latent, prefix=f"{path}.to_latent", **kw)
    back = ReplicatedLinear(latent, width, prefix=f"{path}.from_latent", **kw)
    return to, back


def _sink_of(c: dict) -> torch.Tensor:
    """The shared expert of each of this rank's shared hidden units, as the
    column-parallel projections of `MLP` split them: a rank holds a slice of each."""
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    shared = c["shared_expert"] or c["expert"]
    each = shared["intermediate_size"]
    units = each * c["num_shared_experts"] // get_tensor_model_parallel_world_size()
    start = get_tensor_model_parallel_rank() * units
    return torch.arange(start, start + units) // each


def _norm(c: dict | None, width: int) -> Norm | None:
    """A norm of `c`, the config of one, or None."""
    return None if c is None else Norm(c, width)
