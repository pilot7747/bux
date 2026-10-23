"""Training settings, as plain data.

Every field of ``TrainConfig`` is a top-level setting of a .bux file, next to
``d_model``, and every config here with a ``type`` tag is called by that name, as
layers are:

    tokenizer = tokenizer(name="gpt2")
    data      = text(path="data/fineweb/sample/10BT/*.parquet")
    packing   = best_fit
    optimizer = muon(lr=0.02)
    schedule  = wsd(warmup=500)
    seq_len   = 1024
    batch     = 512
    steps     = 10B // (batch * seq_len)

A data mixture is a weighted sum of sources, ``0.7 * web + 0.3 * code``, which the
compiler turns into a ``MixtureConfig``.
"""

from typing import Annotated, Literal

from pydantic import AfterValidator, BeforeValidator, Field

from bux.config import Config

Positive = Annotated[int, Field(gt=0)]
Fraction = Annotated[float, Field(ge=0, le=1)]


# --- Data ---------------------------------------------------------------------------
# A source is an endless stream of rows of seq_len tokens: input_ids, position_ids
# (where each token is in its document, so one starts wherever they stop increasing)
# and loss_mask (whether each token is a target). Documents, `text` and `chat`,
# become rows once, when bux.data tokenizes and packs them; `tokens` and `synthetic`
# are rows already.


def _ordered(split: tuple[float, float]) -> tuple[float, float]:
    if split[0] >= split[1]:
        raise ValueError("a split is [start, end], with start < end")
    return split


# The documents whose hash falls in [start, end) of [0, 1]: a sample by content, so
# that split=[0, 0.99] and split=[0.99, 1] are disjoint whatever else changes.
Split = Annotated[tuple[Fraction, Fraction], AfterValidator(_ordered)]


class TextConfig(Config):
    """Raw text, a document a row of Parquet files, tokenized with the run's tokenizer
    and packed by its packing."""

    type: Literal["text"] = "text"
    path: str  # a glob of Parquet files, relative to the working directory
    column: str = "text"
    bos: bool = False  # start every document with the tokenizer's bos
    eos: bool = True  # end it with its eos, so that the model learns where text ends
    split: Split = (0.0, 1.0)  # the documents whose hash falls in [start, end)


class ChatConfig(Config):
    """Conversations in the OpenAI format, a list of {role, content} messages a row of
    Parquet files, rendered with a chat template. The loss is on the replies, each
    with the token that ends it."""

    type: Literal["chat"] = "chat"
    path: str
    column: str = "messages"
    template: Literal["chatml", "llama3"] = "chatml"
    train_on: Literal["assistant", "last", "all"] = "assistant"
    split: Split = (0.0, 1.0)  # the documents whose hash falls in [start, end)


class TokensConfig(Config):
    """Pre-tokenized shards: flat binary files of token ids, read through memory maps.
    A row is a window of the shard; windows overlap by a token, so that every token
    is a target once, and come in a new order every epoch."""

    type: Literal["tokens"] = "tokens"
    path: str  # a glob of shards, relative to the working directory
    dtype: Literal["uint16", "uint32"] = "uint16"
    header: int = 0  # bytes before the tokens in every shard (modded-nanogpt: 1024)


class SyntheticConfig(Config):
    """Uniformly random tokens: for smoke tests and for measuring throughput."""

    type: Literal["synthetic"] = "synthetic"


Source = Annotated[
    TextConfig | ChatConfig | TokensConfig | SyntheticConfig,
    Field(discriminator="type"),
]


class Weighted(Config):
    weight: Annotated[float, Field(gt=0)]
    source: Source


class MixtureConfig(Config):
    """Sources in proportion to their weights: every batch has round(weight * batch)
    rows of each. It is written as a weighted sum, ``0.7 * web + 0.3 * code``;
    ``web + code`` is an even mix."""

    type: Literal["mix"] = "mix"
    sources: tuple[Weighted, ...]


DataConfig = Annotated[
    TextConfig | ChatConfig | TokensConfig | SyntheticConfig | MixtureConfig,
    Field(discriminator="type"),
]


class TokenizerConfig(Config):
    """A BPE tokenizer, run by gigatoken: a Hugging Face Hub repo or a tokenizer.json.
    Special tokens written in documents stay text; only bos, eos and chat templates
    put them in."""

    type: Literal["tokenizer"] = "tokenizer"
    name: str  # "gpt2", "Qwen/Qwen3-0.6B", or a path to a tokenizer.json
    bos: str = "<|endoftext|>"
    eos: str = "<|endoftext|>"
    # Special tokens after the vocabulary, for a template it lacks: GPT-2's 50257 tokens
    # padded to 50304 leave room for ["<|im_start|>", "<|im_end|>"].
    added: tuple[str, ...] = ()


# --- Packing ----------------------------------------------------------------------------
# How documents become rows. Documents are shuffled by a hash of their content and
# packed `chunk` tokens at a time. With doc_mask, positions start again at every
# document and attention stays within it; without, a row reads as one text.


class ConcatPacking(Config):
    """Documents end to end, cut into rows (GPT-2): no padding, and only documents
    that happen to fit are whole. Rows overlap by a token, so that every token is a
    target once."""

    type: Literal["concat"] = "concat"
    doc_mask: bool = False  # positions start at every document, attention within it
    chunk: Positive = 16_000_000  # tokens of documents packed together


class GreedyPacking(Config):
    """Whole documents in their shuffled order, a row padded and closed when the next
    one does not fit (classic SFT). max_docs=1 is one document a row."""

    type: Literal["greedy"] = "greedy"
    doc_mask: bool = True  # positions start at every document, attention within it
    long: Literal["drop", "truncate", "split"] = "drop"  # documents longer than a row
    max_docs: Positive | None = None  # a row's documents and pieces of them
    chunk: Positive = 16_000_000  # tokens of documents packed together


class BestFitPacking(Config):
    """Best-fit decreasing (Ding et al., 2024): the longest documents first, each into
    the open row that it leaves least room in. Nearly no padding, and documents are
    only cut when they are longer than a row, into pieces that fill gaps."""

    type: Literal["best_fit"] = "best_fit"
    doc_mask: bool = True  # positions start at every document, attention within it
    long: Literal["drop", "truncate", "split"] = "split"  # documents longer than a row
    max_docs: Positive | None = None  # a row's documents and pieces of them
    sort: bool = True  # longest first; false keeps the shuffled order (online best fit)
    chunk: Positive = 16_000_000  # tokens of documents packed together


PackingConfig = Annotated[
    ConcatPacking | GreedyPacking | BestFitPacking, Field(discriminator="type")
]


# --- Optimizers -----------------------------------------------------------------------


class AdamWConfig(Config):
    """AdamW (Loshchilov & Hutter, 2019). Weight decay applies to matrices, not to norm
    scales, biases or other vectors."""

    type: Literal["adamw"] = "adamw"
    lr: float  # the peak learning rate, which the schedule scales
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1


class MuonConfig(Config):
    """Muon (Jordan et al., 2024): momentum orthogonalized by Newton-Schulz iterations,
    for the matrices inside the blocks (stacked expert weights one expert at a time).
    The embedding, the LM head and every parameter that is not a matrix train with
    `adam` instead."""

    type: Literal["muon"] = "muon"
    lr: float  # the peak learning rate, which the schedule scales
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5  # Newton-Schulz iterations
    # Their quintics, by name in bux.train.optim.ns_coefficients: Keller Jordan's, the
    # same at every iteration, or polar_express's (Amsel et al., 2025), one for each.
    ns_coefficients: str = "jordan"
    ns_dtype: Literal["float32", "bfloat16"] = "bfloat16"  # what Newton-Schulz runs in
    weight_decay: float = 0.1
    # Updates scaled to this RMS, as AdamW's roughly are, so that one lr suits both
    # (Moonlight: 0.2); none -> by sqrt(max(1, fan_out / fan_in)) (Keller Jordan).
    consistent_rms: float | None = 0.2
    adam: AdamWConfig | None = None  # for the rest; None -> adamw(lr=lr)


OptimizerConfig = Annotated[AdamWConfig | MuonConfig, Field(discriminator="type")]


# --- Learning-rate schedules ------------------------------------------------------------
# A schedule multiplies the optimizer's lr. Warmup rises linearly from lr / warmup at
# the first step to lr.


class ConstantSchedule(Config):
    """The peak learning rate throughout, after the warmup."""

    type: Literal["constant"] = "constant"
    warmup: int = 0  # steps


class CosineSchedule(Config):
    """Warmup, then a cosine decay to `final` times the peak at the last step."""

    type: Literal["cosine"] = "cosine"
    warmup: int = 0  # steps
    final: float = 0.1


class WSDSchedule(Config):
    """Warmup-stable-decay (Hu et al., 2024): warmup, the peak, then a cooldown to
    `final` times the peak over the last `decay` of the steps. Checkpoints from the
    stable phase can be cooled down later, to study several run lengths at once."""

    type: Literal["wsd"] = "wsd"
    warmup: int = 0  # steps
    decay: Annotated[float, Field(ge=0, le=1)] = 0.2  # the fraction of steps it takes
    final: float = 0.0
    # The cooldown's curve; sqrt is 1 - sqrt(t) (Hägele et al., 2024).
    shape: Literal["linear", "sqrt", "cosine"] = "linear"

    def cooldown(self, steps: int) -> int:
        """The steps of the cooldown, in a run of `steps`."""
        return round(self.decay * steps)


ScheduleConfig = Annotated[
    ConstantSchedule | CosineSchedule | WSDSchedule, Field(discriminator="type")
]


# --- Everything else ---------------------------------------------------------------------


class LossConfig(Config):
    """Cross-entropy of the first head, plus these terms."""

    type: Literal["loss"] = "loss"
    z: float = 0.0  # weight of the z-loss, log(Z)^2 of the first head (PaLM: 1e-4)
    # Weight of the mean loss of the heads that predict further ahead (DeepSeek-V3).
    mtp: float = 0.3


class EvalConfig(Config):
    """Held-out loss every `every` steps, over the same `batches` batches each time."""

    type: Literal["eval"] = "eval"
    data: DataConfig
    every: Positive = 1000
    batches: Positive = 10


class CheckpointConfig(Config):
    """Model and optimizer state, saved in the run's directory every `every` steps and
    at the end. A run that finds a checkpoint there resumes from the latest one."""

    type: Literal["checkpoint"] = "checkpoint"
    every: Positive = 1000
    keep: Positive = 2  # the most recent checkpoints to keep


class LogConfig(Config):
    """Training metrics every `every` steps, to the console and to metrics.jsonl."""

    type: Literal["log"] = "log"
    every: Positive = 10


class ProfileConfig(Config):
    """A jax.profiler trace of `steps` steps, those after step `start`, in the run's
    directory under profile/, which TensorBoard's profile plugin and xprof read."""

    type: Literal["profile"] = "profile"
    start: Annotated[int, Field(ge=0)] = 10
    steps: Positive = 5


# --- Precision ---------------------------------------------------------------------------
# What a run computes in, and the format each projection inside a block multiplies in
# (bux.architecture.matmuls): by the kind of projection, by the residual branch it is
# in, and by the GEMM.

# The GEMMs of a projection y = x W: y's (fprop), and the gradient's by x (dgrad) and
# by W (wgrad).
Pass = Literal["fprop", "dgrad", "wgrad"]
PASSES: tuple[Pass, ...] = ("fprop", "dgrad", "wgrad")


def _in_compute(fmt: str | None) -> str | None:
    if fmt not in (None, "compute"):
        raise ValueError(
            "the routed experts multiply in compute: a grouped block-scaled GEMM is "
            "not in bux yet"
        )
    return fmt


# A number of residual branches, or a share of the model's, rounded up.
Branches = Annotated[int, Field(ge=0)] | Annotated[float, Field(gt=0, lt=1)]


class RolesConfig(Config):
    """The format each kind of projection multiplies in, where it is not the
    precision's `matmuls`; none is `matmuls`. The embedding, the head, the router and
    softmax attention's own matmuls always stay in compute."""

    type: Literal["roles"] = "roles"
    q: str | None = None  # attention's queries and gate (gqa, mla, csa, the indexer)
    kv: str | None = None  # attention's keys and values
    o: str | None = None  # attention's output projection
    mlp: str | None = None  # MLPs, the shared experts' too
    # The routed experts, in compute until bux has a grouped block-scaled GEMM.
    experts: Annotated[str | None, AfterValidator(_in_compute)] = None
    latent: str | None = None  # LatentMoE's projections to and from its latent
    recurrence: str | None = None  # the recurrent mixers' projections, but the output
    recurrence_out: str | None = None  # their output projection, Mamba's out_proj
    mtp: str | None = None  # nextn's projection, and every branch after the first head


class SwitchConfig(Config):
    """From `at` of the steps on, only `passes` multiply in the format: the end of a
    run in a higher precision. NVIDIA's NVFP4 recipe takes fprop to bfloat16 for the
    last 18%: passes=[dgrad, wgrad]."""

    type: Literal["switch"] = "switch"
    at: Annotated[float, Field(gt=0, lt=1)]
    passes: tuple[Pass, ...] = ()


class PrecisionConfig(Config):
    """The dtype the forward and backward compute in, and the format the projections
    inside blocks multiply in (bux.architecture.matmuls): `matmuls`, or what `roles`
    gives a kind of projection, and compute in the `first` and `last` residual
    branches of the model. Of each projection's GEMMs, only `passes` multiply in the
    format. Parameters stay float32, and so do the optimizer's `moments`, unless
    bfloat16, norm statistics, router scores and top-k, recurrent states and the loss;
    gradients are summed over rows in `reduce`. Unlike the three settings below, it is
    in the run's id, but for `eval`."""

    type: Literal["precision"] = "precision"
    compute: Literal["float32", "bfloat16"] = "bfloat16"
    matmuls: str = "compute"  # a format of bux.architecture.matmuls: mxfp8, nvfp4
    roles: RolesConfig = RolesConfig()
    # The residual branches outside any other, first and last before the first head,
    # that multiply in compute: a number of them, or a share.
    first: Branches = 0
    last: Branches = 0
    passes: tuple[Pass, ...] = PASSES  # the GEMMs that multiply in the format
    switch: SwitchConfig | None = None
    # Evaluation multiplies as training does, or in compute.
    eval: Literal["train", "compute"] = "train"
    moments: Literal["float32", "bfloat16"] = "float32"  # the optimizer's state
    # The dtype gradients are summed over the batch's rows in: compute, or float32 as
    # Megatron's are, which a step sums once (methods.accumulate = unreduced).
    reduce: Literal["compute", "float32"] = "compute"


# --- Parallelism -----------------------------------------------------------------------
# How a run splits over devices, which changes how it computes, not what: none of the
# three is in the run's id. bux.parallel reads them; docs/parallelism.md explains them.

Axis = Literal["data", "stage", "fsdp", "context", "expert", "tensor"]
Axes = Axis | tuple[Axis, ...] | None  # the mesh axes a named dimension splits over


class MeshConfig(Config):
    """The device grid: a size per axis, the slowest links first."""

    type: Literal["mesh"] = "mesh"
    data: Positive | None = None  # None: the devices the other axes leave
    stage: Positive = 1
    fsdp: Positive = 1
    context: Positive = 1
    expert: Positive = 1
    tensor: Positive = 1


class LayoutConfig(Config):
    """Which mesh axes each named dimension of parameters and activations splits over.
    A dimension the axes' sizes do not divide stays whole."""

    type: Literal["layout"] = "layout"
    batch: Axes = ("data", "fsdp", "expert")  # rows
    seq: Axes = "context"  # positions, inside a layer
    stream: Axes = "context"  # positions of the residual stream, between branches
    embed: Axes = None  # the model's width
    vocab: Axes = "tensor"
    heads: Axes = "tensor"  # query heads, or a mixer's heads
    kv_heads: Axes = "tensor"
    mlp: Axes = "tensor"  # hidden units
    experts: Axes = "expert"
    latent: Axes = None  # low-rank bottlenecks: MLA, LatentMoE
    layers: Axes = None  # the stacked passes of a scanned repeat
    stages: Axes = "stage"  # those of a pipelined one, a share a stage
    # What FSDP splits a parameter's largest whole dimension over: the axes that copy
    # dense parameters but `data`, which HSDP keeps as copies.
    fsdp: Axes = ("fsdp", "context", "expert")
    zero: Literal[1, 3] = 3  # 1: FSDP splits the optimizer state alone


def _named_none(v: object) -> object:
    """A method named none, which a .bux file writes as its none."""
    return "none" if v is None else v


def _listed(v: object) -> object:
    """One name, as a list of one."""
    return (v,) if isinstance(v, str) else v


Names = Annotated[tuple[str, ...], BeforeValidator(_listed)]


class RematConfig(Config):
    """What a rematerialized branch or scanned pass keeps for the backward pass, by the
    names layers give what they make (bux.architecture.remat); the backward computes
    the rest again. A preset is a name: save_attention is remat(save=[attention,
    recurrence])."""

    type: Literal["remat"] = "remat"
    save: Names = ()  # kept on the device: attention, qkv, mlp, dots, ...
    offload: Names = ()  # moved to the host's memory until the backward reads them
    # The passes of each repeat it rematerializes, from the first; the rest keep
    # everything. None: every pass.
    first: Annotated[int, Field(ge=0)] | None = None


class MethodsConfig(Config):
    """How each part that communicates is computed, by the name of a function in its
    registry in bux.parallel.methods: another method is a function, not a field."""

    type: Literal["methods"] = "methods"
    attention: str = "auto"
    moe: str = "sorted"
    # The picks a device of all_gather or the ragged MoE methods takes, a multiple of
    # its own; what does not fit drops, which `dropped` reports. None: all that come.
    capacity: Annotated[float, Field(gt=0)] | None = None
    recurrence: str = "scan"
    # What the backward pass of a recurrence by chunks reads of its chunks' states:
    # kept from the forward pass, or computed again from its inputs, in a fourteenth
    # of the memory and a fifth more time. The delta rule by cuDNN's kernels keeps only
    # its inputs either way.
    states: Literal["keep", "recompute"] = "keep"
    repeat: str = "unroll"
    # A policy by name, or as data: remat(save=[attention], first=8).
    remat: Annotated[str | RematConfig, BeforeValidator(_named_none)] = "none"
    pipeline: str = "gpipe"
    rounds: Positive = 1  # times a circular pipeline sends a microbatch round
    # The passes of the pipelined repeat each stage runs, first to last: fewer on the
    # first and last, which run the embedding and the head. None: an even share.
    stages: list[Positive] | None = None
    orthogonalize: str = "owner"  # how Muon reaches whole matrices
    loss: str = "chunks"  # how the loss takes a head's logits: by chunks, or whole
    tensor: str = "gspmd"  # how TP's collectives meet the projections they feed
    tensor_chunks: Positive = 2  # of each device's rows, under tensor = chunked
    microbatches: Positive = 1
    # Where a step runs its layers more than once (microbatches, a pipeline's ticks),
    # it casts and gathers the parameters FSDP splits once, or as each layer runs
    # (ZeRO-3), which holds one layer's at a time.
    gather: Literal["step", "layer"] = "step"
    # How its microbatches' gradients add up: unreduced, each device's own reduced once
    # a step; or reduced each microbatch. auto: unreduced where the devices along the
    # rows' axes span more than one process (between hosts), and no layer learns from
    # the whole batch as an MoE's router does; else reduced.
    accumulate: Literal["auto", "unreduced", "reduced"] = "auto"
    # A step's new state takes the old one's memory: a lower peak, in fewer pieces.
    donate: bool = True


def _flags(options: str) -> str:
    for option in options.split():
        if "=" not in option:
            raise ValueError(f"an option is --name=value, not {option!r}")
    return options


class CompilerConfig(Config):
    """XLA's options for the training step: a preset's, by name in bux.train.xla, and
    options of one's own over them. Like the three settings above, it changes how a
    run computes, not what, and is not in the run's id."""

    type: Literal["compiler"] = "compiler"
    preset: str = "default"  # XLA's own; o1, lhs, maxtext, collective_matmul
    # Over the preset's, as XLA_FLAGS writes them: "--xla_gpu_enable_triton_gemm=false
    # --xla_gpu_threshold_for_windowed_einsum_mib=0". XLA's debug options, and jax's
    # optimization_level and memory_fitting_level.
    options: Annotated[str, AfterValidator(_flags)] = ""


class TrainConfig(Config):
    """The training settings of a run: one field per top-level setting of a .bux file."""

    data: DataConfig
    tokenizer: TokenizerConfig | None = None  # for text and chat data
    packing: PackingConfig = BestFitPacking()
    optimizer: OptimizerConfig
    schedule: ScheduleConfig = ConstantSchedule()
    seq_len: Annotated[int, Field(ge=2)]  # tokens per row
    batch: Positive  # rows per step
    steps: Positive
    seed: int = 0
    clip: float | None = 1.0  # clip gradients to this global norm; none disables it
    loss: LossConfig = LossConfig()
    eval: EvalConfig | None = None
    checkpoint: CheckpointConfig | None = None
    log: LogConfig = LogConfig()
    profile: ProfileConfig | None = None
    precision: PrecisionConfig = PrecisionConfig()
    mesh: MeshConfig = MeshConfig()
    layout: LayoutConfig = LayoutConfig()
    methods: MethodsConfig = MethodsConfig()
    compiler: CompilerConfig = CompilerConfig()


# The configs a .bux file calls by name, besides layers and their settings.
Setting = Annotated[
    TextConfig
    | ChatConfig
    | TokensConfig
    | SyntheticConfig
    | TokenizerConfig
    | ConcatPacking
    | GreedyPacking
    | BestFitPacking
    | AdamWConfig
    | MuonConfig
    | ConstantSchedule
    | CosineSchedule
    | WSDSchedule
    | LossConfig
    | EvalConfig
    | CheckpointConfig
    | LogConfig
    | ProfileConfig
    | PrecisionConfig
    | RolesConfig
    | SwitchConfig
    | MeshConfig
    | LayoutConfig
    | RematConfig
    | MethodsConfig
    | CompilerConfig,
    Field(discriminator="type"),
]

SOURCES = (TextConfig, ChatConfig, TokensConfig, SyntheticConfig, MixtureConfig)
DOCUMENTS = (TextConfig, ChatConfig)


def leaves(c: DataConfig) -> tuple[Source, ...]:
    """The sources of `c`, a mixture's in order."""
    return tuple(w.source for w in c.sources) if isinstance(c, MixtureConfig) else (c,)
