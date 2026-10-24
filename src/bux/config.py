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


# situ is 4 * tanh(x / 4) * sigmoid(x): SiLU with its linear factor soft-capped (Kimi K3).
Activation = Literal[
    "relu", "relu2", "gelu", "gelu_tanh", "swish", "sigmoid", "tanh", "softplus", "situ"
]


# --- Normalization ------------------------------------------------------------


class NormalizationConfig(Config):
    type: Literal["layernorm", "rmsnorm"]
    eps: float = 1e-6
    affine: bool = True
    bias: bool = False  # layernorm only
    # Scale parametrized as (1 + w) with w initialized at 0 (Gemma, Qwen3-Next).
    zero_centered: bool = False
    scale_input: bool = False  # scale before normalizing, norm(w * x) (Muse Spark)


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
    """YaRN RoPE extension (Peng et al., 2023), as in transformers. Rotary pairs that
    make more than beta_fast turns over original_max_len keep their frequency, those
    that make fewer than beta_slow turns have it divided by factor, and a linear ramp
    blends the ones between. q.k is scaled by attention_factor ** 2."""

    type: Literal["yarn"] = "yarn"
    factor: float
    original_max_len: int
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attention_factor: float | None = None  # None -> 0.1 * ln(factor) + 1; 1 disables it
    # "rotary" multiplies cos and sin by it: only the rotated dims (transformers);
    # "softmax" multiplies the softmax scale by its square: all qk dims (DeepSeek MLA).
    attention_factor_on: Literal["rotary", "softmax"] = "rotary"


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
    # Which rotated dims pair up: "half" rotates dim i with i + dims / 2 (GPT-NeoX,
    # Llama), "interleaved" rotates dim 2i with 2i + 1 (GPT-J, Cohere).
    layout: Literal["half", "interleaved"] = "half"
    scaling: RoPEScalingConfig | None = None


class RelativeBiasConfig(Config):
    """Data-dependent relative position bias (Inkling): every head adds <W_h x_i, P[i - j]>
    to its logit for key j, with a learned table P of `extent` offsets, `rank` wide."""

    type: Literal["relbias"] = "relbias"
    rank: int = 16
    extent: int = 1024  # offsets with a bias; farther keys get none


# --- Channel mixers -----------------------------------------------------------


class MLPConfig(Config):
    type: Literal["mlp", "gated_mlp"]
    intermediate_size: int
    activation: Activation
    dropout: float = 0.0
    bias: bool = False
    # Clamped SwiGLU: the gate input is capped at limit, the up branch at +-limit
    # (gpt-oss, DeepSeek-V4).
    limit: float | None = None
    up_softcap: float | None = None  # tanh soft-capping of the up branch (Kimi K3)


class MoEConfig(Config):
    """Token-choice top-k mixture of experts (Mixtral, DeepSeekMoE, Qwen3-MoE)."""

    type: Literal["moe"] = "moe"
    num_experts: int
    top_k: int
    expert: MLPConfig
    num_shared_experts: int = 0
    shared_expert: MLPConfig | None = None  # None -> same shape as `expert`
    shared_expert_gate: bool = False  # sigmoid gate on the shared expert (Qwen)
    # sqrtsoftplus scores experts with sqrt(softplus(logit)) (DeepSeek-V4).
    router: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "softmax"
    router_logit_scale: float = 1.0  # multiplies the logits before scoring (Muse Spark)
    # For benchmarks: a training forward routes by random logits, N(0, 1), so that each
    # expert takes an even share of the picks in expectation and a capacity drops few;
    # their gradient goes to the router's own logits (Megatron-Core's
    # moe_router_force_load_balancing).
    random_logits: bool = False
    normalize_top_k: bool = True
    routed_scaling_factor: float = 1.0
    # Group-limited routing: pick top_k_groups of num_groups first (DeepSeek-V3).
    num_groups: int = 1
    top_k_groups: int = 1
    # Load balancing by a loss, a bias or both. aux_free adds a bias per expert to the
    # scores that pick the top-k, not to the weights. Every training step moves it by
    # bias_update_rate: up for experts picked less than the mean, down for those picked
    # more (DeepSeek-V3).
    load_balancing: Literal["aux_loss", "aux_free"] = "aux_loss"
    bias_update_rate: Annotated[float, Field(ge=0)] = 1e-3
    # Weight of each layer's balance loss, E * sum_e f_e P_e, where f_e is the fraction
    # of top-k picks that go to expert e and P_e its mean score, normalized over the
    # experts (Switch; DeepSeek-V3, eqs. 17-20). None -> 1e-2, or 0 with aux_free. Every
    # layer adds its own, as in Megatron-LM. transformers pools the layers into one loss
    # and counts tokens, not picks, so its coefficient c is about c * top_k / layers.
    aux_loss_coef: Annotated[float, Field(ge=0)] | None = None
    # Over the whole batch (Switch, Qwen), or per sequence and then averaged (the
    # complementary sequence-wise loss of DeepSeek-V3).
    aux_loss_scope: Literal["batch", "sequence"] = "batch"
    # LatentMoE: routed experts run in a narrower space, between a down and an up
    # projection; the router and the shared experts see the full width (Nemotron 3).
    latent_size: int | None = None
    latent_input_norm: NormalizationConfig | None = None  # after the down projection
    latent_output_norm: NormalizationConfig | None = None  # on the routed sum (Kimi K3)
    expert_norm: NormalizationConfig | None = None  # on each routed expert's output
    # The shared experts are weighted by the router too, normalized with the top-k
    # (Inkling); otherwise they are added with weight 1.
    shared_expert_sink: bool = False

    @property
    def aux_loss_weight(self) -> float:
        """aux_loss_coef, or its default."""
        if self.aux_loss_coef is not None:
            return self.aux_loss_coef
        return 0.0 if self.load_balancing == "aux_free" else 1e-2


# --- Token mixers ---------------------------------------------------------------


class AttentionConfig(Config):
    """Anything that mixes information across sequence positions."""


class SparseIndexerConfig(Config):
    """Lightning indexer of DeepSeek Sparse Attention: each query only attends to its top-k keys."""

    type: Literal["indexer"] = "indexer"
    num_heads: int = 64
    head_dim: int = 128
    top_k: int = 2048
    # Mean-pool keys in blocks of this many and pick whole blocks, top_k keys in total,
    # plus the query's own unfinished block (Qwen3.8 QSA).
    block_size: int = 1
    weighted: bool = True  # learned per-head score weights (DSA); else a plain sum
    # Norm of each query head and of the keys (after block pooling). None: no query
    # norm, and the layer's own key norm (DSA: a layernorm). Qwen3.8 QSA: an rmsnorm.
    norm: NormalizationConfig | None = None
    # How the indexer pairs RoPE dims when its keys are the tokens, whatever the layer's
    # layout (DeepSeek-V3.2: half, GLM-5.3: interleaved).
    rope_layout: Literal["half", "interleaved"] = "half"
    # What a `candidates` port exports: the best `candidate_top_k` blocks of
    # `candidate_block_size` keys, each scored by its best key (DeepSeek-V4.1).
    candidate_block_size: int = 8
    candidate_top_k: int = 2048  # blocks in the candidate pool
    # Weight of the indexer's loss, its only training signal: the KL divergence from the
    # attention over the kept keys to the softmax of its scores (DSA). 0 leaves it
    # untrained.
    loss_weight: Annotated[float, Field(ge=0)] = 1.0


class QuadraticAttentionConfig(AttentionConfig):
    """Softmax attention.

    Ports, where a layer lists them: ``index`` exports the indexer's top-k, or imports
    it instead of running an indexer of its own (GLM-5, DeepSeek-V4.1).
    """

    sliding_window_size: int | None = None
    attention_dropout: float = 0.0
    output_dropout: float = 0.0
    # Elementwise sigmoid gate on the attention output (Qwen3-Next gated attention).
    gated: bool = False
    gate_per_head: bool = False  # one gate value per head instead (Laguna)
    gate_activation: Activation = "sigmoid"  # of the gate (Laguna: softplus)
    output_norm: NormalizationConfig | None = None  # per head, before the gate (Muse)
    qk_norm: NormalizationConfig | None = None
    softcap: float | None = None  # tanh soft-capping of the logits (Gemma 2)
    sinks: bool = False  # learned per-head attention sink logit (gpt-oss)
    # None -> head_dim ** -0.5 (MLA: of qk_nope_head_dim + qk_rope_head_dim); YaRN with
    # attention_factor_on=softmax multiplies it by attention_factor ** 2.
    scale: float | None = None
    # Logits scaled by 1 + log_scale * ln(max(position / log_scale_floor, 1)), so that
    # attention stays sharp at long range (Inkling). Not Llama 4's temperature tuning,
    # 1 + s * ln(1 + floor(position / floor)): a step function, on its NoPE layers.
    log_scale: float | None = None
    log_scale_floor: int = 8192
    relative_bias: RelativeBiasConfig | None = None
    indexer: SparseIndexerConfig | None = None


class GQAConfig(QuadraticAttentionConfig):
    """Grouped-query attention; MHA when num_kv_heads == num_heads, MQA when it is 1.

    Ports: ``v`` exports the value heads, or imports them for value-residual mixing
    (ResFormer); ``kv`` exports keys and values, or imports them instead of computing
    its own (cross-layer KV sharing, YOCO / CLA); ``index`` shares the indexer's top-k.
    """

    ports: ClassVar[frozenset[str]] = frozenset({"v", "kv", "index"})

    type: Literal["gqa"] = "gqa"
    num_heads: int
    num_kv_heads: int
    head_dim: int | None = None  # None -> width // num_heads
    v_head_dim: int | None = None  # None -> head_dim (MiMo-V2: 128 against 192)
    bias: bool = False  # on the projections of the input: q, k, v, gate (GPT-2, Qwen2)
    output_bias: bool | None = None  # None -> bias (GPT-2); Qwen2 has none
    # Make q, k and v with one projection and split it, instead of one projection
    # each: the same function and init, held as one matrix (GPT-2, MiMo-V2.6-Pro).
    fused_qkv: bool = False
    rope: RoPEConfig | None = None
    # Causal depthwise conv on keys and values, added to them: k + conv(k) (Inkling).
    kv_conv_size: int | None = None
    # How a layer that reads the v port mixes its values v1 into its own v, as
    # a v + b v1 starting from a = b = 0.5: "fixed" keeps a and b (ResFormer's code),
    # "learned" learns both (Learnable-ResFormer), "convex" learns a, with b = 1 - a.
    value_residual: Literal["fixed", "learned", "convex"] = "fixed"


class MLAConfig(QuadraticAttentionConfig):
    """Multi-head latent attention (DeepSeek-V2/V3, Kimi K2, Kimi Linear).

    Ports: ``index`` shares the indexer's top-k across layers (GLM-5 IndexCache).
    """

    ports: ClassVar[frozenset[str]] = frozenset({"index"})

    type: Literal["mla"] = "mla"
    num_heads: int
    q_lora_rank: int | None = None  # None -> full-rank query projection
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope: RoPEConfig | None = RoPEConfig()  # on the decoupled qk_rope_head_dim part


class CSAConfig(QuadraticAttentionConfig):
    """Compressed sparse attention (DeepSeek-V4): every query head attends to a sliding
    window of per-token latents and to the indexer's top-k compressed latents. A latent
    is the key and the value at once, a single KV head shared by all query heads, and
    each `compress_ratio` tokens are pooled into one compressed latent.

    Ports: ``kv`` exports the compressed latents (with the indexer's keys), or imports
    them instead of compressing its own; ``index`` exports the indexer's top-k, or
    imports it instead of running an indexer; ``candidates`` exports coarse blocks of
    compressed positions, and an indexer that imports them selects only inside them.
    """

    ports: ClassVar[frozenset[str]] = frozenset({"kv", "index", "candidates"})

    type: Literal["csa"] = "csa"
    num_heads: int
    head_dim: int = 512
    q_lora_rank: int | None = None  # None -> full-rank query projection
    # Grouped low-rank output projection: o_groups groups of heads, each projected to
    # o_lora_rank, then mixed back to the width; None -> one dense projection.
    o_lora_rank: int | None = None
    o_groups: int = 1
    # Tokens pooled into one compressed latent; None -> window only.
    compress_ratio: int | None = None
    rope: RoPEConfig | None = RoPEConfig()
    # Norm of the query latent, the latents, the compressed ones and indexer keys.
    latent_norm: NormalizationConfig = NormalizationConfig(type="rmsnorm")


class LinearAttentionConfig(AttentionConfig):
    """Recurrent token mixers with a matrix-valued state of shape (head_dim, v_head_dim) per value head."""

    num_heads: int
    head_dim: int
    # With more value heads than q/k heads, each q/k head serves a group (Qwen3-Next).
    num_v_heads: int | None = None  # None -> num_heads
    v_head_dim: int | None = None  # None -> head_dim
    conv_size: int | None = 4  # causal depthwise conv on q, k, v; None disables it
    # Gate on the output norm before the output projection; None disables it.
    output_gate: Literal["swish", "sigmoid"] | None = "swish"
    # Epsilon of the output norm (Qwen3-Next: 1e-6; Kimi Linear and fla: 1e-5).
    norm_eps: float = 1e-6


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
    output_gate: Literal["swish", "sigmoid"] | None = "sigmoid"
    output_gate_rank: int | None = None  # rank of the output gate; None -> full rank
    # Bounded decay lower_bound * sigmoid(.), in (lower_bound, 0), instead of
    # -exp(A) * softplus(.) (Kimi K3: -5).
    gate_lower_bound: float | None = None


class LightningAttentionConfig(LinearAttentionConfig):
    """Linear attention with a fixed per-head exponential decay (TransNormerLLM, MiniMax-01)."""

    type: Literal["lightning"] = "lightning"
    conv_size: int | None = None
    output_gate: Literal["swish", "sigmoid"] | None = "sigmoid"  # MiniMax-01


class RWKV7Config(LinearAttentionConfig):
    """RWKV-7 "Goose" time mixing: generalized delta rule with vector-valued gates (Peng et al., 2025).

    Ports: ``v`` exports the values, or imports them for RWKV-7's value residual: the
    first layer exports its values, and every later one mixes them into its own, by a
    learned amount per channel, from a projection of rank `v_rank`.
    """

    ports: ClassVar[frozenset[str]] = frozenset({"v"})

    type: Literal["rwkv7"] = "rwkv7"
    conv_size: int | None = None
    # The gate is a low-rank projection with this activation between, act(x G1) G2.
    output_gate: Literal["swish", "sigmoid"] | None = "sigmoid"
    norm_eps: float = 1e-5  # times the value head dim, as in fla (RWKV-LM: 64e-5)
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


class ConvConfig(AttentionConfig):
    """Causal depthwise convolution over positions, with no gate or activation. As a
    residual branch, [conv] is a short-conv shortcut (Inkling; Canon layers)."""

    type: Literal["conv"] = "conv"
    size: int = 4
    bias: bool = False


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
    norm: NormalizationConfig | None = None  # after the scale (Inkling); nextn too
    dropout: float = 0.0


class LMHeadConfig(Config):
    type: Literal["head"] = "head"
    tied: bool = False  # reuse the embedding matrix
    bias: bool = False
    softcap: float | None = None  # tanh soft-capping of the logits (Gemma 2)


class NextNConfig(Config):
    """Multi-token prediction input (DeepSeek-V3): h -> W [norm(e_{i+k}); norm(h)].
    For the k-th nextn in a forward pass, e_{i+k} is `embed`'s output for the token k
    positions ahead (scaled and normalized, without positions). So every head after it
    predicts one token further ahead, and a shared MTP module run twice looks 1, then 2
    tokens ahead (Nemotron 3, GLM-5). An MTP module is ``$h! nextn block norm @h head``,
    run after the main head.

    On a widened stream it mixes the embedding into each copy, W_e norm(e) + W_h
    norm(R)_j, normalizing the copies together (Qwen3.8-Flash-Next). On one stream
    under a widening algebra it widens its output, as `embed` does.
    """

    type: Literal["nextn"] = "nextn"
    norm: NormalizationConfig = NormalizationConfig(type="rmsnorm")  # on both inputs


class EngramConfig(Config):
    """Conditional memory (DeepSeek Engram, 2026): hashed n-gram embeddings of the input
    tokens, gated by the stream. Every n-gram of 2 to `max_ngram_size` tokens ending
    at a position is hashed by `num_heads` heads, each into its own table, and the
    rows are concatenated into e. With a = <norm(x), norm(W_k e)> / sqrt(d), the layer
    adds u = sigmoid(sign(a) sqrt(|a|)) W_v e to the stream x, then silu(conv(norm(u))).
    Write it bare, not in [...]: on a widened stream it adds to each copy, gated by
    its own key (DeepSeek-V4.1, Qwen3.8-Flash-Next). It reads the model's input tokens
    itself, as RoPE reads positions.
    """

    type: Literal["engram"] = "engram"
    head_dim: int  # width of a table row
    table_size: int  # rows per head: the next prime >= this not used by an earlier head
    max_ngram_size: int = 3  # n-grams of 2 to this many tokens
    num_heads: int = 8  # hash heads per n-gram size
    norm: NormalizationConfig = NormalizationConfig(type="rmsnorm")  # query, key, conv
    conv_size: int | None = 4  # zero-initialized, dilated by max_ngram_size; None: off
    vocab_classes: int | None = None  # hash normalized token classes, not raw ids
    boundary_token: int | None = None  # n-grams stop at this token, e.g. EOS (Qwen)


# --- Residual stream ------------------------------------------------------------


class HyperConnectionsConfig(Config):
    """Widen the residual stream into `expansion` streams with learned read/write/mix
    connections around every residual branch (Zhu et al., 2025). "mhc" projects the
    stream mixing onto doubly stochastic matrices with Sinkhorn (DeepSeek mHC, 2025).
    With `single_pass`, each branch reads with the coefficients that the branch before
    it computed (the first reads stream 0 alone), and so does the parameter-free read
    before the head (DeepSeek-V4.1's Single-Pass mHC)."""

    type: Literal["hc", "mhc"]
    expansion: int = 4
    dynamic: bool = True  # coefficients that depend on the stream; else static ones
    sinkhorn_iters: int = 20
    single_pass: bool = False
    norm_eps: float = 1e-6  # of the norm before the dynamic coefficients (V4.1: 1e-20)


class GatedResidualConfig(Config):
    """Gated residual (Qwen3.8-Flash-Next, 2026): `expansion` streams, each RMS-normalized
    when read. A branch reads their mean under a low-rank sigmoid gate and writes its
    output back to each stream with its own weight in (0, 2). The read normalizes,
    so branches need no pre-norm, and the read before the head replaces the final norm.
    """

    type: Literal["gr"] = "gr"
    rank: int  # of the read gate (Qwen3.8-Flash-Next: d_model / 8)
    expansion: int = 4
    norm_eps: float = 1e-6  # of the streams' RMS norm


class AttentionResidualConfig(Config):
    """Attention residuals (Kimi, 2026): each branch reads a softmax over depth of the
    embedding and the branch outputs so far, scored against its own learned query,
    instead of their plain sum. Outputs are summed in blocks of `block_size` branches,
    and each block is one source; the head reads the same way.
    """

    type: Literal["attnres"] = "attnres"
    block_size: int | None = None  # None -> each branch output is its own source
    norm_eps: float = 1e-6  # of the keys' RMS norm (Kimi K3: 1e-5)


class ResidualGateConfig(Config):
    """Variance-preserving residual gate (Muse Spark): x <- a * x + b * f(x) per channel,
    with b = sigmoid(g / temperature) for a learned g, and a = sqrt(1 - b^2)."""

    type: Literal["vpgate"] = "vpgate"
    temperature: float = 0.3
    eps: float = 1e-3  # a = sqrt(max(1 - b^2, eps)): the stream is never cut off


class MergeConfig(Config):
    """Reads a widened residual stream (hc, mhc, gr, attnres) back into one: the read the
    algebra does for a branch, without the write. The compiler puts one before the
    first layer outside a branch that needs one stream, such as the final norm."""

    type: Literal["merge"] = "merge"


ResidualConfig = Annotated[
    HyperConnectionsConfig
    | GatedResidualConfig
    | AttentionResidualConfig
    | ResidualGateConfig,
    Field(discriminator="type"),
]


LayerConfig = Annotated[
    NormalizationConfig
    | DyTConfig
    | MLPConfig
    | MoEConfig
    | GQAConfig
    | MLAConfig
    | CSAConfig
    | GLAConfig
    | DeltaNetConfig
    | GatedDeltaNetConfig
    | KDAConfig
    | LightningAttentionConfig
    | RWKV7Config
    | MambaConfig
    | Mamba2Config
    | ConvConfig
    | ShortConvConfig
    | LinearConfig
    | ActivationConfig
    | ScaleConfig
    | DropoutConfig
    | EmbeddingConfig
    | LMHeadConfig
    | NextNConfig
    | EngramConfig
    | MergeConfig,
    Field(discriminator="type"),
]

# Configs that parametrize a layer or the model but are not layers themselves.
SubConfig = Annotated[
    RoPEConfig
    | PositionInterpolationConfig
    | YaRNConfig
    | Llama3RoPEScalingConfig
    | RelativeBiasConfig
    | SparseIndexerConfig
    | HyperConnectionsConfig
    | GatedResidualConfig
    | AttentionResidualConfig
    | ResidualGateConfig,
    Field(discriminator="type"),
]
