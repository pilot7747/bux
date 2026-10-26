"""Building blocks of language models, as plain data: every layer and setting of the
``.bux`` language.

Every concrete config has a globally unique ``type`` tag, which is its name in the
language, so ``gqa(num_heads=8, num_kv_heads=2)`` is a `GQAConfig`, and any config
validates from a dict through the `LayerConfig` (layers) and `SubConfig` (settings)
discriminated unions. Abstract bases (`AttentionConfig`, `LinearAttentionConfig`,
...) only group shared fields and carry no tag.

A config documents itself here: its docstring says what it is, and the comment above
or beside each field says what the field does. ``bux-lsp`` shows them on hover, and
the language reference is generated from them (``bux.lang.reference``).

A new layer is a config class here, added to `LayerConfig`, and a builder registered
for its tag in the backend (`bux.architecture.register`).
"""

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class Config(BaseModel):
    """The base of every config: frozen pydantic data that rejects unknown fields.

    A config with a ``type`` tag is a layer or a setting that a ``.bux`` file calls
    by that tag; one without only groups the fields its subclasses share.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Internal tensors a layer can export to a tap (``attn(v=@v1)``) or take from
    #: one (``attn(v=$v1)``). Each layer's docstring says what its ports carry.
    ports: ClassVar[frozenset[str]] = frozenset()


#: The activations a layer or a setting can name, bare (``linear(features=h) gelu``)
#: or as ``act(fn=gelu)``. ``relu2`` is ReLU squared, ``gelu`` is exact and
#: ``gelu_tanh`` its tanh approximation, ``swish`` is SiLU, and ``situ`` is
#: ``4 * tanh(x / 4) * sigmoid(x)``: SiLU with its linear factor soft-capped (Kimi K3).
Activation = Literal[
    "relu", "relu2", "gelu", "gelu_tanh", "swish", "sigmoid", "tanh", "softplus", "situ"
]


# --- Normalization ------------------------------------------------------------
# Norms over the width of the stream, or of a head. Their statistics are in float32
# whatever dtype the run computes in, and the result in the input's dtype.


class NormalizationConfig(Config):
    """Layer normalization (Ba et al., 2016) or RMSNorm (Zhang & Sennrich, 2019).

    ``layernorm`` subtracts the mean over the width and divides by the standard
    deviation; ``rmsnorm`` only divides by the root mean square. Then each multiplies
    by a learned gain per channel (`affine`) and adds a learned bias (`bias`). A norm
    is a layer, ``[norm attn]``, and a setting of layers that normalize something of
    their own, as ``gqa(qk_norm=rmsnorm)``.
    """

    type: Literal["layernorm", "rmsnorm"]
    eps: float = 1e-6  # Added to the variance (rmsnorm: mean square) before its root.
    affine: bool = True  # A learned gain per channel, the norm's scale.
    bias: bool = False  # A learned bias per channel, added last (GPT-2's layernorms).
    # The gain as 1 + w, with w learned from 0 (Gemma, Qwen3-Next).
    zero_centered: bool = False
    # Multiply by the gain before normalizing, norm(w * x), not after (Muse Spark).
    scale_input: bool = False


class DyTConfig(Config):
    """Dynamic Tanh (Zhu et al., 2025), a norm without statistics:
    ``gamma * tanh(alpha * x) + beta``, with a learned scalar alpha and a learned gain
    gamma (from 1) and bias beta (from 0) per channel. It replaces a norm one for one:
    ``[dyt attn] [dyt ffn]``."""

    type: Literal["dyt"] = "dyt"
    alpha_init: float = 0.5  # Where the learned scalar alpha starts.


# --- Positional encodings -----------------------------------------------------
# Rotary embeddings, their scalings for a longer context, and a learned relative
# bias: settings of attention layers, as `gqa(rope=rope(base=500000))`, not layers.
# Positions are those given with the tokens, each token's offset in its document
# when rows are packed, else its index in the row.


class PositionInterpolationConfig(Config):
    """Linear RoPE interpolation (Chen et al., 2023): every rotary frequency divided by
    `factor`, so that `factor` times as many positions turn through the angles the
    model trained on. A RoPE scaling: ``rope(scaling=pi(factor=8))`` (Gemma 3)."""

    type: Literal["pi"] = "pi"
    factor: float  # How many times longer the context gets: frequencies over it.


class YaRNConfig(Config):
    """YaRN RoPE extension (Peng et al., 2023), as in transformers. A RoPE scaling:
    ``rope(scaling=yarn(factor=40, original_max_len=4096))``.

    Rotary pairs that make more than `beta_fast` turns over `original_max_len`
    positions keep their frequency, those that make fewer than `beta_slow` turns have
    it divided by `factor`, and a linear ramp blends the ones between. ``q . k`` is
    scaled by ``attention_factor ** 2``.
    """

    type: Literal["yarn"] = "yarn"
    factor: float  # How many times longer than original_max_len the context gets.
    original_max_len: int  # The context length the model was trained at.
    # Pairs that turn more times than this keep their frequency.
    beta_fast: float = 32.0
    beta_slow: float = 1.0  # Pairs that turn fewer times have it divided by factor.
    # YaRN's temperature: q . k is scaled by its square. None -> 0.1 * ln(factor) + 1
    # (1 for a factor of 1 or less); 1 disables it.
    attention_factor: float | None = None
    # "rotary" multiplies cos and sin by it: only the rotated dims (transformers);
    # "softmax" multiplies the softmax scale by its square: all qk dims (DeepSeek MLA).
    attention_factor_on: Literal["rotary", "softmax"] = "rotary"


class Llama3RoPEScalingConfig(Config):
    """Llama 3.1's RoPE scaling, as in transformers. A RoPE scaling:
    ``rope(scaling=llama3(factor=8))``.

    Rotary pairs that make fewer than `low_freq_factor` turns over `original_max_len`
    positions have their frequency divided by `factor`, those that make more than
    `high_freq_factor` turns keep it, and a linear ramp in the turns blends the ones
    between.
    """

    type: Literal["llama3"] = "llama3"
    factor: float  # What the slowest pairs' frequencies are divided by.
    original_max_len: int = 8192  # The context length the model was trained at.
    low_freq_factor: float = 1.0  # Pairs that turn fewer times are divided by factor.
    high_freq_factor: float = 4.0  # Pairs that turn more times keep their frequency.


#: The RoPE scalings, `pi`, `yarn` and `llama3`, by their ``type``.
RoPEScalingConfig = Annotated[
    PositionInterpolationConfig | YaRNConfig | Llama3RoPEScalingConfig,
    Field(discriminator="type"),
]


class RoPEConfig(Config):
    """Rotary position embeddings (Su et al., 2021): each pair of a head's rotated
    dims turns by its position times its frequency, ``base ** (-2i / dims)`` for pair
    i of the dims rotated. A setting of attention layers: ``gqa(rope=rope())``; an
    attention layer without one has no positions (NoPE)."""

    type: Literal["rope"] = "rope"
    base: float = 10000.0  # Theta, the base of the frequencies (Llama 3: 500000).
    # Partial rotary: only this fraction of each head's dims is rotated, the first
    # ones, in whole pairs (GLM, Qwen3-Next).
    fraction: float = 1.0
    # Which rotated dims pair up: "half" rotates dim i with i + dims / 2 (GPT-NeoX,
    # Llama), "interleaved" rotates dim 2i with 2i + 1 (GPT-J, Cohere).
    layout: Literal["half", "interleaved"] = "half"
    scaling: RoPEScalingConfig | None = None  # Context extension; None: none.


class RelativeBiasConfig(Config):
    """Data-dependent relative position bias (Inkling): every head adds
    ``<W_h x_i, P[i - j]>`` to the logit of query i for key j, with a learned table P
    of `extent` offsets, `rank` wide, that starts at 0. A setting of softmax
    attention, ``gqa(relative_bias=relbias())``; it needs all t by t logits, so it
    runs under ``methods.attention`` ``xla`` or ``auto`` alone."""

    type: Literal["relbias"] = "relbias"
    rank: int = 16  # Width of the table's entries and of each head's projection.
    extent: int = 1024  # Offsets with a bias, from 0; farther keys get none.


# --- Channel mixers -----------------------------------------------------------
# Layers that mix the channels of each position on its own: dense MLPs, and
# mixtures of experts that route each token to a few of many MLPs. Llama 3's is
# `ffn = gated_mlp(intermediate_size=14336, activation=swish)`.


class MLPConfig(Config):
    """A dense MLP, ``mlp``, or a gated one, ``gated_mlp`` (GLU variants; Shazeer,
    2020): ``down(act(up(x)))``, or ``down(act(gate(x)) * up(x))``. SwiGLU is
    ``gated_mlp(activation=swish)``. It is also the shape of an MoE's experts
    (`MoEConfig.expert`)."""

    type: Literal["mlp", "gated_mlp"]
    intermediate_size: int  # Hidden units: the output width of up (and of gate).
    activation: Activation  # Of the hidden units, or of the gate when gated.
    dropout: float = 0.0  # Dropout rate of the output, in training.
    bias: bool = False  # Biases on the up, gate and down projections.
    # Clamped SwiGLU: the gate input is capped at limit, the up branch at +-limit
    # (gpt-oss, DeepSeek-V4); None: no clamp.
    limit: float | None = None
    # Soft-capping of the up branch, cap * tanh(up / cap) (Kimi K3); None: none.
    up_softcap: float | None = None


class MoEConfig(Config):
    """Token-choice top-k mixture of experts (Mixtral, DeepSeekMoE, Qwen3-MoE).

    A router scores the `num_experts` routed experts for every token, and the token
    goes through its `top_k` best, their outputs summed by the router's weights; the
    shared experts, if any, take every token. A loss, a bias or both balance the
    experts' loads (`load_balancing`). The run's ``methods.moe`` says how the experts
    run: every expert on every token (``dense``), each on its own tokens
    (``sorted``, the default), or over devices by expert parallelism.
    """

    type: Literal["moe"] = "moe"
    num_experts: int  # Routed experts.
    top_k: int  # Routed experts each token goes through.
    # Each routed expert: its type, intermediate_size and activation, and limit and
    # up_softcap. Routed experts have no bias or dropout.
    expert: MLPConfig
    # Experts every token goes through, run as one MLP of num_shared_experts times
    # the shared expert's intermediate_size.
    num_shared_experts: int = 0
    shared_expert: MLPConfig | None = None  # A shared expert; None -> like `expert`.
    # A sigmoid gate on the shared experts' output, from the token (Qwen).
    shared_expert_gate: bool = False
    # How the router scores the experts from its logits: by a softmax over them, a
    # sigmoid each (DeepSeek-V3), or sqrt(softplus(logit)) each (DeepSeek-V4).
    router: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "softmax"
    router_logit_scale: float = 1.0  # Multiplies the logits before scoring (Muse Spark)
    # For benchmarks: a training forward routes by random logits, N(0, 1), so that each
    # expert takes an even share of the picks in expectation and a capacity drops few;
    # their gradient goes to the router's own logits (Megatron-Core's
    # moe_router_force_load_balancing).
    random_logits: bool = False
    # Divide the top-k scores by their sum, so the weights sum to 1; else the weights
    # are the scores.
    normalize_top_k: bool = True
    routed_scaling_factor: float = 1.0  # Multiplies the weights (DeepSeek-V3: 2.5).
    # Group-limited routing: pick top_k_groups of num_groups first (DeepSeek-V3).
    # Each group is scored by the sum of its best two experts' scores.
    num_groups: int = 1
    top_k_groups: int = 1  # Groups each token picks its experts from.
    # Load balancing by a loss, a bias or both. aux_free adds a bias per expert to the
    # scores that pick the top-k, not to the weights. Every training step moves it by
    # bias_update_rate: up for experts picked less than the mean, down for those picked
    # more (DeepSeek-V3).
    load_balancing: Literal["aux_loss", "aux_free"] = "aux_loss"
    # How far aux_free's bias moves each training step, for each expert.
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
    # LatentMoE: routed experts run in a narrower space of this width, between a down
    # and an up projection; the router and the shared experts see the full width
    # (Nemotron 3). None: the full width.
    latent_size: int | None = None
    # LatentMoE's norm of the tokens after the down projection; None: none.
    latent_input_norm: NormalizationConfig | None = None
    # A norm of the routed sum (Kimi K3), before LatentMoE's up projection; None: none.
    latent_output_norm: NormalizationConfig | None = None
    expert_norm: NormalizationConfig | None = None  # Of each routed expert's output.
    # The shared experts are weighted by the router too, normalized with the top-k
    # (Inkling); otherwise they are added with weight 1.
    shared_expert_sink: bool = False

    @property
    def aux_loss_weight(self) -> float:
        """`aux_loss_coef`, or its default: 0 with ``aux_free``, else 1e-2."""
        if self.aux_loss_coef is not None:
            return self.aux_loss_coef
        return 0.0 if self.load_balancing == "aux_free" else 1e-2


# --- Token mixers ---------------------------------------------------------------
# Layers that mix information across positions, all causal: softmax attention
# (`gqa`, `mla`, `csa`), linear attention with a matrix-valued state (`gla`,
# `deltanet`, `gdn`, `kda`, `lightning`, `rwkv7`), state space models (`mamba`,
# `mamba2`) and short causal convolutions (`conv`, `short_conv`). In a packed row,
# each sees its own document alone.


class AttentionConfig(Config):
    """Anything that mixes information across sequence positions: the base of every
    token mixer's config."""


class SparseIndexerConfig(Config):
    """Lightning indexer of DeepSeek Sparse Attention (DSA, DeepSeek-V3.2): each query
    only attends to its top-k keys.

    A small attention of `num_heads` query heads and one key head scores every key a
    query may see, ``sum_j w_j ReLU(q_j . k)``, and the layer attends to the `top_k`
    best alone. A setting of softmax attention: ``mla(indexer=indexer(top_k=2048))``.
    The keep mask passes no gradient, so only the indexer's own loss trains it.
    """

    type: Literal["indexer"] = "indexer"
    num_heads: int = 64  # Query heads of the indexer.
    head_dim: int = 128  # Width of an indexer query head and of its keys.
    top_k: int = 2048  # Keys each query keeps (in whole blocks, with block_size).
    # Mean-pool keys in blocks of this many and pick whole blocks, top_k keys in total,
    # plus the query's own unfinished block (Qwen3.8 QSA).
    block_size: int = 1
    weighted: bool = True  # Learned per-head score weights (DSA); else a plain sum.
    # Norm of each query head and of the keys (after block pooling). None: no query
    # norm, and the layer's own key norm (DSA: a layernorm). Qwen3.8 QSA: an rmsnorm.
    norm: NormalizationConfig | None = None
    # How the indexer pairs RoPE dims when its keys are the tokens, whatever the layer's
    # layout (DeepSeek-V3.2: half, GLM-5.3: interleaved).
    rope_layout: Literal["half", "interleaved"] = "half"
    # What a `candidates` port exports: the best `candidate_top_k` blocks of
    # `candidate_block_size` keys, each scored by its best key (DeepSeek-V4.1).
    candidate_block_size: int = 8
    candidate_top_k: int = 2048  # Blocks in the candidate pool.
    # Weight of the indexer's loss, its only training signal: the KL divergence from the
    # attention over the kept keys to the softmax of its scores (DSA). 0 leaves it
    # untrained.
    loss_weight: Annotated[float, Field(ge=0)] = 1.0


class QuadraticAttentionConfig(AttentionConfig):
    """Softmax attention: what gqa, mla and csa share.

    Ports, where a layer lists them: ``index`` exports the indexer's top-k, or imports
    it instead of running an indexer of its own (GLM-5, DeepSeek-V4.1).
    """

    # Each query attends to this many keys: itself and those just before it (local
    # attention). None: every key before it.
    sliding_window_size: int | None = None
    # Dropout rate of the attention probabilities, in training; it needs all t by t
    # of them, so it runs under methods.attention xla or auto alone.
    attention_dropout: float = 0.0
    output_dropout: float = 0.0  # Dropout rate of the output, in training.
    # Elementwise gate act(W_g x) on each head's output, before the output projection
    # (Qwen3-Next's gated attention; the activation is gate_activation).
    gated: bool = False
    gate_per_head: bool = False  # One gate value per head instead (Laguna).
    gate_activation: Activation = "sigmoid"  # Of the gate (Laguna: softplus).
    # A norm of each head's output, before the gate (Muse); None: none.
    output_norm: NormalizationConfig | None = None
    # Norm of each query and key head, before RoPE (Qwen3); None: none.
    qk_norm: NormalizationConfig | None = None
    # Soft-capping of the logits, softcap * tanh(logits / softcap) (Gemma 2); None:
    # none.
    softcap: float | None = None
    # A learned sink logit per head, in every query's softmax beside its keys', that
    # takes a share of the attention and gives no value (gpt-oss).
    sinks: bool = False
    # The logits' scale. None -> head_dim ** -0.5 (MLA: of qk_nope_head_dim +
    # qk_rope_head_dim); YaRN with attention_factor_on=softmax multiplies it by
    # attention_factor ** 2.
    scale: float | None = None
    # Logits scaled by 1 + log_scale * ln(max(n / log_scale_floor, 1)) at the query's
    # position n, counted from 1, so that attention stays sharp at long range
    # (Inkling); None: no scaling. It runs under methods.attention xla or auto alone.
    # Not Llama 4's temperature tuning, 1 + s * ln(1 + floor(position / floor)): a step
    # function, on its NoPE layers.
    log_scale: float | None = None
    log_scale_floor: int = 8192  # Positions up to this one keep their logits' scale.
    relative_bias: RelativeBiasConfig | None = None  # A learned relative position bias.
    # A lightning indexer that picks the keys each query attends to; None: all of them.
    indexer: SparseIndexerConfig | None = None


class GQAConfig(QuadraticAttentionConfig):
    """Grouped-query attention; MHA when num_kv_heads == num_heads, MQA when it is 1.

    Causal softmax attention (Vaswani et al., 2017) of `num_heads` query heads and
    `num_kv_heads` key and value heads, each shared by a group of
    ``num_heads / num_kv_heads`` query heads (Ainslie et al., 2023):
    ``gqa(num_heads=32, num_kv_heads=8, rope=rope())``.

    Ports: ``v`` exports the value heads, or imports them for value-residual mixing
    (ResFormer); ``kv`` exports keys and values, or imports them instead of computing
    its own (cross-layer KV sharing, YOCO / CLA); ``index`` shares the indexer's top-k.
    """

    ports: ClassVar[frozenset[str]] = frozenset({"v", "kv", "index"})

    type: Literal["gqa"] = "gqa"
    num_heads: int  # query heads
    num_kv_heads: int  # key and value heads, each shared by a group of query heads
    head_dim: int | None = None  # dims of each head; None -> width // num_heads

    @field_validator("num_kv_heads")
    @classmethod
    def _groups(cls, kv: int, info: ValidationInfo) -> int:
        """Check that the query heads split into a group for each key and value head."""
        heads = info.data.get("num_heads")
        if kv < 1 or (heads is not None and heads % kv):
            raise ValueError(
                f"{kv} key and value heads do not divide {heads} query heads"
            )
        return kv

    # Dims of each value head; None -> head_dim (MiMo-V2: 128 against 192).
    v_head_dim: int | None = None
    # Biases on the projections of the input: q, k, v and the gate (GPT-2, Qwen2).
    bias: bool = False
    # A bias on the output projection; None -> bias (GPT-2). Qwen2 has none.
    output_bias: bool | None = None
    # Make q, k and v with one projection and split it, instead of one projection
    # each: the same function and init, held as one matrix (GPT-2, MiMo-V2.6-Pro).
    fused_qkv: bool = False
    rope: RoPEConfig | None = None  # Rotary embedding of q and k; None: no positions.
    # Causal depthwise conv on keys and values, of this many taps, added to them:
    # k + conv(k) (Inkling); None: none.
    kv_conv_size: int | None = None
    # How a layer that reads the v port mixes its values v1 into its own v, as
    # a v + b v1 starting from a = b = 0.5: "fixed" keeps a and b (ResFormer's code),
    # "learned" learns both (Learnable-ResFormer), "convex" learns a, with b = 1 - a.
    value_residual: Literal["fixed", "learned", "convex"] = "fixed"


class MLAConfig(QuadraticAttentionConfig):
    """Multi-head latent attention (DeepSeek-V2/V3, Kimi K2, Kimi Linear).

    Every head's keys and values are projected up from one `kv_lora_rank` wide latent
    per token, and its queries from one `q_lora_rank` wide latent (or from the input
    at full rank). A head's query and key have `qk_nope_head_dim` dims without RoPE
    and `qk_rope_head_dim` more with it, the keys' of which all heads share.

    Ports: ``index`` shares the indexer's top-k across layers (GLM-5 IndexCache).
    """

    ports: ClassVar[frozenset[str]] = frozenset({"index"})

    type: Literal["mla"] = "mla"
    num_heads: int  # Query heads, each with keys and values of its own.
    # Width of the latent the queries come from; None -> full-rank query projection.
    q_lora_rank: int | None = None
    kv_lora_rank: int = 512  # Width of the latent that keys and values come from.
    qk_nope_head_dim: int = 128  # Dims of each query and key head without RoPE.
    qk_rope_head_dim: int = 64  # Dims with RoPE: the keys' shared by all heads.
    v_head_dim: int = 128  # Dims of each value head.
    # On the decoupled qk_rope_head_dim part; None: no dims rotated (Kimi Linear).
    rope: RoPEConfig | None = RoPEConfig()


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
    num_heads: int  # Query heads, which share the latents as their one KV head.
    head_dim: int = 512  # Width of a query head and of a latent.
    # Width of the latent the queries come from; None -> full-rank query projection.
    q_lora_rank: int | None = None
    # Grouped low-rank output projection: o_groups groups of heads, each projected to
    # o_lora_rank, then mixed back to the width; None -> one dense projection.
    o_lora_rank: int | None = None
    o_groups: int = 1  # Groups of heads of the low-rank output projection.
    # Tokens pooled into one compressed latent; None -> window only.
    compress_ratio: int | None = None
    rope: RoPEConfig | None = RoPEConfig()  # Of the queries and latents; None: none.
    # Norm of the query latent, the latents, the compressed ones and indexer keys.
    latent_norm: NormalizationConfig = NormalizationConfig(type="rmsnorm")


class LinearAttentionConfig(AttentionConfig):
    """Recurrent token mixers with a matrix-valued state per value head.

    Linear attention: every value head keeps a state of `head_dim` by `v_head_dim`
    that each token decays, updates with its key and value, and reads with its query.
    The state is zero at every document's start. What the layers share: the q, k and
    v projections, a short conv on them, and the output's norm, gate and projection.
    """

    num_heads: int  # Query and key heads.
    head_dim: int  # Width of a query and a key head: the state's rows.
    # Value heads. With more of them than q/k heads, each q/k head serves a group of
    # them (Qwen3-Next).
    num_v_heads: int | None = None  # None -> num_heads
    # Width of a value head: the state's columns.
    v_head_dim: int | None = None  # None -> head_dim
    # Causal depthwise conv on q, k and v, of this many taps, each then SiLU'd; None
    # disables it.
    conv_size: int | None = 4
    # Activation of a gate on the output, after its norm and before the output
    # projection; None disables it.
    output_gate: Literal["swish", "sigmoid"] | None = "swish"
    # Epsilon of the output norm (Qwen3-Next: 1e-6; Kimi Linear and fla: 1e-5).
    norm_eps: float = 1e-6


class GLAConfig(LinearAttentionConfig):
    """Gated linear attention (Yang et al., 2024): a data-dependent decay per channel.

    Linear attention with no delta rule, as fla has it: q scaled by
    ``head_dim ** -0.5``, and a decay per key channel, ``log w = log sigmoid(f) / 16``,
    for a low-rank projection f of the input. Without a conv, q, k and v are not
    SiLU'd.
    """

    type: Literal["gla"] = "gla"
    gate_rank: int = 16  # Rank of the decay's projection f.
    conv_size: int | None = None


class DeltaNetConfig(LinearAttentionConfig):
    """DeltaNet: the delta rule (Schlag et al., 2021; Yang et al., 2024).

    Linear attention in which each token writes its value over what the state holds
    at its key, a step ``beta = sigmoid(W_b x)`` per value head:
    ``S_t = (I - beta k k^T) S_{t-1} + beta k v^T``, read as ``o_t = S_t^T q_t``.
    q and k are L2-normalized, and q is scaled by ``head_dim ** -0.5``.
    """

    type: Literal["deltanet"] = "deltanet"


class GatedDeltaNetConfig(LinearAttentionConfig):
    """Gated DeltaNet: the delta rule with a scalar per-head decay (Yang et al., 2025).

    DeltaNet whose state decays by ``exp(g)`` before each token's update, with
    ``g = -exp(A_log) softplus(W_a x + dt_bias)`` per value head, as in Qwen3-Next:
    ``gdn(num_heads=16, num_v_heads=32, head_dim=128)``.
    """

    type: Literal["gdn"] = "gdn"


class KDAConfig(LinearAttentionConfig):
    """Kimi Delta Attention (Kimi Linear, 2025): Gated DeltaNet, a decay per channel.

    The decay is per key channel, ``g = -exp(A_log) softplus(f + dt_bias)`` for a
    low-rank projection f of the input (`gate_rank`), or bounded below
    (`gate_lower_bound`). The output gate is low-rank too (`output_gate_rank`).
    """

    type: Literal["kda"] = "kda"
    gate_rank: int | None = None  # Rank of the decay's projection; None -> head_dim.
    output_gate: Literal["swish", "sigmoid"] | None = "sigmoid"
    output_gate_rank: int | None = None  # Rank of the output gate; None -> full rank.
    # Bounded decay lower_bound * sigmoid(.), in (lower_bound, 0), instead of
    # -exp(A) * softplus(.) (Kimi K3: -5); None: unbounded. At -5 or above, cuDNN's
    # fused kernel can run the layer on a B200 or B300, else bux's own kernel does.
    gate_lower_bound: float | None = None


class LightningAttentionConfig(LinearAttentionConfig):
    """Lightning attention (TransNormerLLM, MiniMax-01): a fixed decay per head.

    Linear attention as MiniMax-01 has it: q, k and v SiLU'd and unscaled, a decay
    ``exp(-s)`` per head whose rates s weaken with the layer's depth among the
    model's token mixers, and one RMSNorm of all the heads together before the gate.
    """

    type: Literal["lightning"] = "lightning"
    conv_size: int | None = None
    # Activation of the gate on the output norm, sigmoid as in MiniMax-01; None
    # disables it.
    output_gate: Literal["swish", "sigmoid"] | None = "sigmoid"


class RWKV7Config(LinearAttentionConfig):
    """RWKV-7 "Goose" time mixing (Peng et al., 2025): a generalized delta rule.

    A delta rule with vector-valued gates, as the RWKV-LM reference has it. Every
    projection reads the token shifted toward the one before it, by a learned
    amount per channel. The state follows ``S_t = (diag(w_t) - (a_t kk_t) kk_t^T)
    S_{t-1} + k_t v_t^T``, with kk the normalized key, a the in-context learning rate
    and w a decay per key channel, each from a low-rank projection.

    Ports: ``v`` exports the values, or imports them for RWKV-7's value residual: the
    first layer exports its values, and every later one mixes them into its own, by a
    learned amount per channel, from a projection of rank `v_rank`.
    """

    ports: ClassVar[frozenset[str]] = frozenset({"v"})

    type: Literal["rwkv7"] = "rwkv7"
    conv_size: int | None = None  # None only: RWKV-7 shifts tokens instead.
    # The gate is a low-rank projection with this activation between, act(x G1) G2.
    output_gate: Literal["swish", "sigmoid"] | None = "sigmoid"
    # Epsilon of the output's layernorm of each head, times the value head dim, as in
    # fla (RWKV-LM: 64e-5).
    norm_eps: float = 1e-5
    decay_rank: int = 64  # Rank of the decay w's projection.
    a_rank: int = 64  # Rank of the in-context learning rate a's projection.
    v_rank: int = 32  # Rank of the value residual's mix, with the v port.
    gate_rank: int = 128  # Rank of the output gate's projection.


class StateSpaceConfig(AttentionConfig):
    """Selective state space models: what mamba and mamba2 share.

    The input projects to `expand` times the width of inner channels, convolved over
    positions and SiLU'd, and each inner channel keeps a state of `state_size` that
    decays and takes in the input by a time step of its own (Mamba) or of its head
    (Mamba-2). The state is zero at every document's start.
    """

    state_size: int  # State dims of each inner channel: the N of the papers.
    expand: int = 2  # Inner channels, as a multiple of the width.
    conv_size: int = 4  # Taps of the causal depthwise conv before the scan.
    # The least time step at the start: they start log-uniform in [dt_min, dt_max].
    dt_min: float = 1e-3
    dt_max: float = 1e-1  # The greatest time step at the start.


class MambaConfig(StateSpaceConfig):
    """Selective SSM with a diagonal state matrix (Gu & Dao, 2023), as in Jamba.

    Every inner channel's state decays at its own rate in each state dim, by a time
    step from a projection of rank `dt_rank`; B and C, which select what the state
    takes in and gives out, are shared by all channels. As in Jamba, the time steps,
    B and C are RMS-normalized. ``mamba(state_size=16)``.
    """

    type: Literal["mamba"] = "mamba"
    state_size: int = 16
    # Rank of the projection the time steps come from. None -> ceil(width / 16).
    dt_rank: int | None = None


class Mamba2Config(StateSpaceConfig):
    """Mamba-2 / SSD (Dao & Gu, 2024): a scalar decay per head (Zamba2, Nemotron-H).

    The inner channels split into heads of `head_dim` channels each, whose states
    decay by one rate per head. B and C are shared by the heads of each of
    `num_groups` groups, as keys are by the query heads of a group in GQA. The output
    is gated by SiLU(z) and RMS-normalized a group of heads at a time:
    ``mamba2(state_size=128)``.
    """

    type: Literal["mamba2"] = "mamba2"
    state_size: int = 128
    # Channels of each head, which its state of state_size by head_dim holds.
    head_dim: int = 64
    num_groups: int = 1  # groups sharing B and C, like kv heads in GQA


class ConvConfig(AttentionConfig):
    """Causal depthwise convolution over positions, with no gate or activation. As a
    residual branch, ``[conv]`` is a short-conv shortcut (Inkling; Canon layers)."""

    type: Literal["conv"] = "conv"
    size: int = 4  # Taps: each position sees itself and the size - 1 before it.
    bias: bool = False  # A learned bias per channel.


class ShortConvConfig(AttentionConfig):
    """Double-gated short causal convolution (LFM2; cf. H3, Hyena).

    The input projects to B, C and u, each the width, and the output is
    ``C * conv(B * u)``, projected back.
    """

    type: Literal["short_conv"] = "short_conv"
    conv_size: int = 3  # Taps of the causal depthwise conv.
    bias: bool = False  # Biases on the projections and the conv.


# --- Elementwise and plumbing ---------------------------------------------------
# The embedding and the head, multi-token prediction (`nextn`) and Engram's memory,
# and the small layers blocks are made of: `linear`, `act`, `scale` and `dropout`.


class LinearConfig(Config):
    """A dense projection to `features` channels, ``x W + b``: with embed and head, the
    layers that change the stream's width. ``linear(features=4 * d_model)``."""

    type: Literal["linear"] = "linear"
    features: int  # Output channels.
    bias: bool = False  # A learned bias per output channel.


class ActivationConfig(Config):
    """An activation, applied elementwise. Usually written bare, by its name:
    ``linear(features=h) swish`` is ``linear(features=h) act(fn=swish)``."""

    type: Literal["act"] = "act"
    fn: Activation  # Which activation.


class ScaleConfig(Config):
    """A learnable multiplier: ReZero, LayerScale, residual lambdas.

    ``[norm attn scale(init=0)]`` is ReZero, ``scale(per_channel=true)`` LayerScale,
    and ``[$x0 scale]`` adds a learned multiple of the tap x0 to the stream.
    """

    type: Literal["scale"] = "scale"
    init: float = 1.0  # Where the multiplier starts.
    per_channel: bool = False  # One multiplier per channel; else one for the stream.


class DropoutConfig(Config):
    """Dropout of the stream, in training: ``[norm attn dropout(rate=0.1)]``."""

    type: Literal["dropout"] = "dropout"
    rate: float  # Fraction of entries zeroed; the rest are scaled by 1 / (1 - rate).


class EmbeddingConfig(Config):
    """The token embedding: token ids to the stream, by a table of `vocab_size` rows
    of `d_model`. A model starts with one, and a tied head and nextn read the table
    of its first. Under a residual algebra that widens the stream, it returns that
    many copies of the embedding."""

    type: Literal["embed"] = "embed"
    # Learned absolute positions (GPT-2): a table of this many, added to the
    # embedding; None: none.
    max_positions: int | None = None
    scale_by_sqrt_dim: bool = False  # Multiply the embedding by sqrt(d_model) (Gemma).
    # A norm of the embedding, after the scale and before positions are added
    # (Inkling); nextn reads the embedding normalized too.
    norm: NormalizationConfig | None = None
    dropout: float = 0.0  # Dropout rate of the embedding, in training.


class LMHeadConfig(Config):
    """The language-model head: the stream to logits over the vocabulary,
    ``softcap(x W + b)``. Every head is an output of the model; nothing follows one
    but a tap read, as a multi-token prediction module does (`NextNConfig`)."""

    type: Literal["head"] = "head"
    tied: bool = False  # Use the model's first embedding table, transposed, as W.
    bias: bool = False  # A learned bias per token of the vocabulary.
    # Soft-capping of the logits, softcap * tanh(logits / softcap) (Gemma 2); None:
    # none.
    softcap: float | None = None


class NextNConfig(Config):
    """Multi-token prediction input (DeepSeek-V3): ``h -> W [norm(e_{i+k}); norm(h)]``.
    For the k-th nextn in a forward pass, ``e_{i+k}`` is `embed`'s output for the
    token k positions ahead (scaled and normalized, without positions). So every head
    after it predicts one token further ahead, and a shared MTP module run twice
    looks 1, then 2 tokens ahead (Nemotron 3, GLM-5). An MTP module is
    ``$h! nextn block norm @h head``, run after the main head.

    On a widened stream it mixes the embedding into each copy, ``W_e norm(e) + W_h
    norm(R)_j``, normalizing the copies together (Qwen3.8-Flash-Next). On one stream
    under a widening algebra it widens its output, as `embed` does.
    """

    type: Literal["nextn"] = "nextn"
    # The norm of both inputs, the embedding and the stream, each its own.
    norm: NormalizationConfig = NormalizationConfig(type="rmsnorm")


class EngramConfig(Config):
    """Conditional memory (DeepSeek Engram, 2026): hashed n-gram embeddings of the input
    tokens, gated by the stream.

    Every n-gram of 2 to `max_ngram_size` tokens ending at a position is hashed by
    `num_heads` heads, each into its own table, and the rows are concatenated into e.
    With ``a = <norm(x), norm(W_k e)> / sqrt(d)``, the layer adds
    ``u = sigmoid(sign(a) sqrt(abs(a))) W_v e`` to the stream x, then
    ``silu(conv(norm(u)))``. N-grams stop at a document's start.

    Write it bare, not in ``[...]``: on a widened stream it adds to each copy, gated
    by its own key (DeepSeek-V4.1, Qwen3.8-Flash-Next). It reads the model's input
    tokens itself, as RoPE reads positions.
    """

    type: Literal["engram"] = "engram"
    head_dim: int  # Width of a table row.
    # Rows of each head's table: the least prime >= this that no earlier head (of any
    # engram layer) has taken.
    table_size: int
    max_ngram_size: int = 3  # N-grams of 2 to this many tokens.
    num_heads: int = 8  # Hash heads per n-gram size, each with a table of its own.
    # Of the stream (the query), of the keys, and of u before the conv.
    norm: NormalizationConfig = NormalizationConfig(type="rmsnorm")
    # Taps of the causal conv of u, zero-initialized and dilated by max_ngram_size;
    # None: no conv.
    conv_size: int | None = 4
    # Hash each token's class, of this many, not its id; until a checkpoint brings
    # the tokenizer's normalized classes, id i is class i mod vocab_classes. None:
    # the ids.
    vocab_classes: int | None = None
    # N-grams do not reach back past this token, such as EOS (Qwen); None: none.
    boundary_token: int | None = None


# --- Residual stream ------------------------------------------------------------
# Residual algebras: what a residual branch `[f]` does instead of `x + f(x)`, set for
# the whole model by the top-level setting `residual`, as `residual = mhc()`. Only
# branches outside any other change; one inside another stays `x + f(x)`. `hc`,
# `mhc`, `gr` and `attnres` widen the stream, and `merge` reads it back into one.


class HyperConnectionsConfig(Config):
    """Hyper-connections (Zhu et al., 2025), or manifold-constrained ones (DeepSeek mHC,
    2025): a residual algebra, ``residual = mhc(expansion=4)``.

    The stream widens into `expansion` copies, and every residual branch reads its
    input from them, writes its output back to each, and mixes them, by learned
    read/write/mix connections. "mhc" projects the stream mixing onto doubly
    stochastic matrices with Sinkhorn. Both start as the plain residual stream. With
    `single_pass`, each branch reads with the coefficients that the branch before it
    computed (the first reads stream 0 alone), and so does the parameter-free read
    before the head (DeepSeek-V4.1's Single-Pass mHC)."""

    type: Literal["hc", "mhc"]
    expansion: int = 4  # Copies of the stream.
    # Coefficients that depend on the stream, by a projection that starts at 0, on
    # top of static ones; else static ones alone.
    dynamic: bool = True
    sinkhorn_iters: int = 20  # Sinkhorn-Knopp rounds of mhc's mixing matrix.
    single_pass: bool = False  # Read with the coefficients the branch before computed.
    # Epsilon of the norm before the dynamic coefficients (DeepSeek-V4.1: 1e-20).
    norm_eps: float = 1e-6


class GatedResidualConfig(Config):
    """Gated residual (Qwen3.8-Flash-Next, 2026): a residual algebra,
    ``residual = gr(rank=320)``.

    The stream widens into `expansion` copies, each RMS-normalized when read. A branch
    reads their mean under a low-rank sigmoid gate and writes its output back to each
    stream with its own weight in (0, 2). The read normalizes, so branches need no
    pre-norm, and the read before the head replaces the final norm.
    """

    type: Literal["gr"] = "gr"
    rank: int  # Rank of the read gate (Qwen3.8-Flash-Next: d_model / 8).
    expansion: int = 4  # Copies of the stream.
    norm_eps: float = 1e-6  # Epsilon of the streams' RMS norm.


class AttentionResidualConfig(Config):
    """Attention residuals (Kimi, 2026): a residual algebra,
    ``residual = attnres(block_size=24)``.

    Each branch reads a softmax over depth of the embedding and the branch outputs so
    far, scored against its own learned query, instead of their plain sum. Outputs
    are summed in blocks of `block_size` branches, and each block is one source; the
    head reads the same way. The stream holds the sources, not copies of it, so
    `engram` and `nextn` need one stream under it.
    """

    type: Literal["attnres"] = "attnres"
    # Branch outputs summed into one source; None -> each is its own source.
    block_size: int | None = None
    norm_eps: float = 1e-6  # Epsilon of the keys' RMS norm (Kimi K3: 1e-5).


class ResidualGateConfig(Config):
    """Variance-preserving residual gate (Muse Spark): a residual algebra,
    ``residual = vpgate()``.

    Each branch makes ``x <- a * x + b * f(x)`` per channel, with
    ``b = sigmoid(g / temperature)`` for a learned g, and ``a = sqrt(1 - b^2)``. The
    k-th branch starts at ``b = 1 / sqrt(k + 2)``, so that the model starts as the
    pre-norm residual one, scaled. The stream stays one.
    """

    type: Literal["vpgate"] = "vpgate"
    temperature: float = 0.3  # Divides the learned g in the gate's sigmoid.
    eps: float = 1e-3  # a = sqrt(max(1 - b^2, eps)): the stream is never cut off.


class MergeConfig(Config):
    """Reads a widened residual stream back into one (hc, mhc, gr, attnres).

    The read the algebra does for a branch, without the write. The compiler puts one
    before the first layer outside a branch that needs one stream, such as the final
    norm, and ``bux fmt`` prints it.
    """

    type: Literal["merge"] = "merge"


#: The residual algebras, `hc`, `mhc`, `gr`, `attnres` and `vpgate`, by their
#: ``type``, the values of the top-level ``residual`` setting.
ResidualConfig = Annotated[
    HyperConnectionsConfig
    | GatedResidualConfig
    | AttentionResidualConfig
    | ResidualGateConfig,
    Field(discriminator="type"),
]


#: Every layer's config, by its ``type``. An ``ir.Layer`` holds one, and the
#: compiler knows layer names from it.
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

#: Configs that parametrize a layer or the model but are not layers themselves.
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
