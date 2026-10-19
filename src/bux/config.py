"""Building blocks of language models, as plain data.

Every concrete layer has a globally unique ``type`` tag, so any of them can be
validated from a dict through the ``LayerConfig`` discriminated union, and the
``.bux`` language (``bux.lang``) can call them by that name: ``gqa(num_heads=8, ...)``.
Abstract bases (``AttentionConfig``, ``LinearAttentionConfig``, ...) only group
shared fields and carry no tag.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Internal tensors a layer can export to a tap (``attn(v=@v1)``) or take
    # from one (``attn(v=$v1)``). Their meaning is documented per layer.
    ports: ClassVar[frozenset[str]] = frozenset()


Activation = Literal["relu", "relu2", "gelu", "gelu_tanh", "swish", "sigmoid", "tanh"]


# --- Normalization ------------------------------------------------------------


class NormalizationConfig(Config):
    type: Literal["layernorm", "rmsnorm"]
    eps: float = 1e-6
    affine: bool = True
    bias: bool = False  # layernorm only
    # Scale parametrized as (1 + w) with w initialized at 0 (Gemma, Qwen3-Next).
    zero_centered: bool = False


class DyTConfig(Config):
    """Dynamic Tanh, gamma * tanh(alpha * x) + beta (Zhu et al., 2025)."""

    type: Literal["dyt"] = "dyt"
    alpha_init: float = 0.5


# --- Positional encodings -----------------------------------------------------


class PositionInterpolationConfig(Config):
    """Linear RoPE interpolation (Chen et al., 2023)."""

    type: Literal["pi"] = "pi"
    factor: float


class YaRNConfig(Config):
    """YaRN RoPE extension (Peng et al., 2023), used by DeepSeek-V3 and Qwen."""

    type: Literal["yarn"] = "yarn"
    factor: float
    original_max_len: int
    beta_fast: float = 32.0
    beta_slow: float = 1.0


class Llama3RoPEScalingConfig(Config):
    type: Literal["llama3"] = "llama3"
    factor: float
    original_max_len: int = 8192
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0


RoPEScalingConfig = Annotated[
    PositionInterpolationConfig | YaRNConfig | Llama3RoPEScalingConfig,
    Field(discriminator="type"),
]


class RoPEConfig(Config):
    type: Literal["rope"] = "rope"
    base: float = 10000.0
    # Partial rotary: only this fraction of each head's dims is rotated (GLM, Qwen3-Next).
    fraction: float = 1.0
    scaling: RoPEScalingConfig | None = None


# --- Channel mixers -----------------------------------------------------------


class MLPConfig(Config):
    type: Literal["mlp", "gated_mlp"]
    intermediate_size: int
    activation: Activation
    dropout: float = 0.0
    bias: bool = False


class MoEConfig(Config):
    """Token-choice top-k mixture of experts (Mixtral, DeepSeekMoE, Qwen3-MoE)."""

    type: Literal["moe"] = "moe"
    num_experts: int
    top_k: int
    expert: MLPConfig
    num_shared_experts: int = 0
    shared_expert: MLPConfig | None = None  # None -> same shape as `expert`
    shared_expert_gate: bool = False  # sigmoid gate on the shared expert (Qwen)
    router: Literal["softmax", "sigmoid"] = "softmax"
    normalize_top_k: bool = True
    routed_scaling_factor: float = 1.0
    # Group-limited routing: pick top_k_groups of num_groups first (DeepSeek-V3).
    num_groups: int = 1
    top_k_groups: int = 1
    # "aux_free" balances with a per-expert router bias instead of a loss (DeepSeek-V3).
    load_balancing: Literal["aux_loss", "aux_free"] = "aux_loss"
    aux_loss_coef: float = 1e-2
    capacity_factor: float | None = None  # None -> dropless


# --- Token mixers ---------------------------------------------------------------


class AttentionConfig(Config):
    """Anything that mixes information across sequence positions."""


class SparseIndexerConfig(Config):
    """Lightning indexer of DeepSeek Sparse Attention: each query only attends to its top-k keys."""

    type: Literal["indexer"] = "indexer"
    num_heads: int = 64
    head_dim: int = 128
    top_k: int = 2048


class QuadraticAttentionConfig(AttentionConfig):
    """Softmax attention."""

    sliding_window_size: int | None = None
    attention_dropout: float = 0.0
    output_dropout: float = 0.0
    # Elementwise sigmoid gate on the attention output (Qwen3-Next gated attention).
    gated: bool = False
    qk_norm: NormalizationConfig | None = None
    softcap: float | None = None  # tanh soft-capping of the logits (Gemma 2)
    sinks: bool = False  # learned per-head attention sink logit (gpt-oss)
    scale: float | None = None  # None -> head_dim ** -0.5
    indexer: SparseIndexerConfig | None = None


class GQAConfig(QuadraticAttentionConfig):
    """Grouped-query attention; MHA when num_kv_heads == num_heads, MQA when it is 1.

    Ports: ``v`` exports the value heads, or imports them for value-residual mixing
    (ResFormer); ``kv`` exports keys and values, or imports them instead of computing
    its own (cross-layer KV sharing, YOCO / CLA).
    """

    ports: ClassVar[frozenset[str]] = frozenset({"v", "kv"})

    type: Literal["gqa"] = "gqa"
    num_heads: int
    num_kv_heads: int
    head_dim: int | None = None  # None -> width // num_heads
    bias: bool = False  # qkv bias (GPT-2, Qwen2)
    rope: RoPEConfig | None = None


class MLAConfig(QuadraticAttentionConfig):
    """Multi-head latent attention (DeepSeek-V2/V3, Kimi K2, Kimi Linear)."""

    type: Literal["mla"] = "mla"
    num_heads: int
    q_lora_rank: int | None = None  # None -> full-rank query projection
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope: RoPEConfig | None = RoPEConfig()  # on the decoupled qk_rope_head_dim part


class LinearAttentionConfig(AttentionConfig):
    """Recurrent token mixers with a matrix-valued state of shape (head_dim, v_head_dim) per value head."""

    num_heads: int
    head_dim: int
    # More value heads than q/k heads shares each q/k head across a group (Qwen3-Next).
    num_v_heads: int | None = None  # None -> num_heads
    v_head_dim: int | None = None  # None -> head_dim
    conv_size: int | None = 4  # causal depthwise conv on q, k, v; None disables it
    output_gate: bool = True  # swish-gated output norm before the output projection


class GLAConfig(LinearAttentionConfig):
    """Gated linear attention: channel-wise data-dependent decay, no delta rule (Yang et al., 2024)."""

    type: Literal["gla"] = "gla"
    gate_rank: int = 16
    conv_size: int | None = None


class DeltaNetConfig(LinearAttentionConfig):
    """Delta rule S <- S (I - b k k^T) + b v k^T (Schlag et al., 2021; Yang et al., 2024)."""

    type: Literal["deltanet"] = "deltanet"


class GatedDeltaNetConfig(LinearAttentionConfig):
    """Gated DeltaNet: the delta rule with a scalar per-head decay (Yang et al., 2025)."""

    type: Literal["gdn"] = "gdn"


class KDAConfig(LinearAttentionConfig):
    """Kimi Delta Attention: Gated DeltaNet with a channel-wise decay (Kimi Linear, 2025)."""

    type: Literal["kda"] = "kda"
    gate_rank: int | None = None  # rank of the decay projection; None -> head_dim


class LightningAttentionConfig(LinearAttentionConfig):
    """Linear attention with a fixed per-head exponential decay (TransNormerLLM, MiniMax-01)."""

    type: Literal["lightning"] = "lightning"
    conv_size: int | None = None


class RWKV7Config(LinearAttentionConfig):
    """RWKV-7 "Goose" time mixing: generalized delta rule with vector-valued gates (Peng et al., 2025)."""

    type: Literal["rwkv7"] = "rwkv7"
    conv_size: int | None = None
    decay_rank: int = 64
    a_rank: int = 64
    v_rank: int = 32
    gate_rank: int = 128


class StateSpaceConfig(AttentionConfig):
    state_size: int
    expand: int = 2
    conv_size: int = 4
    dt_min: float = 1e-3
    dt_max: float = 1e-1


class MambaConfig(StateSpaceConfig):
    """Selective SSM with a diagonal state matrix (Gu & Dao, 2023), as in Jamba."""

    type: Literal["mamba"] = "mamba"
    state_size: int = 16
    dt_rank: int | None = None  # None -> ceil(width / 16)


class Mamba2Config(StateSpaceConfig):
    """Mamba-2 / SSD: scalar per-head decay (Dao & Gu, 2024), as in Zamba2 and Nemotron-H."""

    type: Literal["mamba2"] = "mamba2"
    state_size: int = 128
    head_dim: int = 64
    num_groups: int = 1  # groups sharing B and C, like kv heads in GQA


class ShortConvConfig(AttentionConfig):
    """Double-gated short causal convolution (LFM2; cf. H3, Hyena)."""

    type: Literal["short_conv"] = "short_conv"
    conv_size: int = 3
    bias: bool = False


# --- Elementwise and plumbing ---------------------------------------------------


class LinearConfig(Config):
    type: Literal["linear"] = "linear"
    features: int
    bias: bool = False


class ActivationConfig(Config):
    type: Literal["act"] = "act"
    fn: Activation


class ScaleConfig(Config):
    """Learnable multiplier: ReZero (init=0), LayerScale (per_channel), residual lambdas."""

    type: Literal["scale"] = "scale"
    init: float = 1.0
    per_channel: bool = False


class DropoutConfig(Config):
    type: Literal["dropout"] = "dropout"
    rate: float


class EmbeddingConfig(Config):
    type: Literal["embed"] = "embed"
    max_positions: int | None = None  # learned absolute positions (GPT-2)
    scale_by_sqrt_dim: bool = False  # Gemma
    dropout: float = 0.0


class LMHeadConfig(Config):
    type: Literal["head"] = "head"
    tied: bool = False  # reuse the embedding matrix
    bias: bool = False
    softcap: float | None = None  # tanh soft-capping of the logits (Gemma 2)


# --- Residual stream ------------------------------------------------------------


class HyperConnectionsConfig(Config):
    """Widen the residual stream into `expansion` streams with learned read/write/mix
    connections around every residual branch (Zhu et al., 2025). "mhc" projects the
    stream mixing onto doubly stochastic matrices with Sinkhorn (DeepSeek mHC, 2025)."""

    type: Literal["hc", "mhc"]
    expansion: int = 4
    dynamic: bool = True
    sinkhorn_iters: int = 20


LayerConfig = Annotated[
    NormalizationConfig
    | DyTConfig
    | MLPConfig
    | MoEConfig
    | GQAConfig
    | MLAConfig
    | GLAConfig
    | DeltaNetConfig
    | GatedDeltaNetConfig
    | KDAConfig
    | LightningAttentionConfig
    | RWKV7Config
    | MambaConfig
    | Mamba2Config
    | ShortConvConfig
    | LinearConfig
    | ActivationConfig
    | ScaleConfig
    | DropoutConfig
    | EmbeddingConfig
    | LMHeadConfig,
    Field(discriminator="type"),
]

# Configs that parametrize a layer or the model but are not layers themselves.
SubConfig = Annotated[
    RoPEConfig
    | PositionInterpolationConfig
    | YaRNConfig
    | Llama3RoPEScalingConfig
    | SparseIndexerConfig
    | HyperConnectionsConfig,
    Field(discriminator="type"),
]
