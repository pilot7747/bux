"""Training settings, as plain data.

Every field of `TrainConfig` is a top-level setting of a ``.bux`` file, next to
``d_model``, and every config here with a ``type`` tag is called by that name, as
layers are::

    tokenizer = tokenizer(name="gpt2")
    data      = text(path="data/fineweb/sample/10BT/*.parquet")
    packing   = best_fit
    optimizer = muon(lr=0.02)
    schedule  = wsd(warmup=500)
    seq_len   = 1024
    batch     = 512
    steps     = 10B // (batch * seq_len)

A data mixture is a weighted sum of sources, ``0.7 * web + 0.3 * code``, which the
compiler turns into a `MixtureConfig`.

The module needs no JAX, so the compiler and the editor import it. Each config's
docstring and the comments on its fields are the training half of the language
reference and what the editor shows on hover (`bux.lang.reference`).
"""

from typing import Annotated, Literal

from pydantic import AfterValidator, BeforeValidator, Field

from bux.config import Config

Positive = Annotated[int, Field(gt=0)]
Fraction = Annotated[float, Field(ge=0, le=1)]


# --- Data ---------------------------------------------------------------------------
# A source is an endless stream of rows of `seq_len` tokens: `input_ids`,
# `position_ids` (where each token is in its document, so one starts wherever they
# stop increasing) and `loss_mask` (whether each token is a target), in a new order
# every epoch. Documents, `text` and `chat`, become rows once, when bux.data tokenizes
# and packs them; `tokens` and `synthetic` are rows already.


def _ordered(split: tuple[float, float]) -> tuple[float, float]:
    if split[0] >= split[1]:
        raise ValueError("a split is [start, end], with start < end")
    return split


# The documents whose hash falls in [start, end) of [0, 1]: a sample by content, so
# that split=[0, 0.99] and split=[0.99, 1] are disjoint whatever else changes.
Split = Annotated[tuple[Fraction, Fraction], AfterValidator(_ordered)]


class TextConfig(Config):
    """Raw text: a document per row of Parquet files, tokenized by the run's
    `tokenizer` and packed into rows by its `packing`.

    A run prepares its documents once, before its first step, into a cache of rows
    (`data/packed/` by default) that every run with the same source, tokenizer,
    packing and `seq_len` reads again; `bux prepare` does it ahead of time. Hold
    documents out for eval by their hash: `text(path=..., split=[0, 0.99])` to train
    on and `split=[0.99, 1]` to evaluate on."""

    type: Literal["text"] = "text"
    path: str  # a glob of Parquet files, relative to the working directory
    column: str = "text"  # the column that holds each document's text
    bos: bool = False  # start every document with the tokenizer's `bos`
    eos: bool = True  # end it with its `eos`, so that the model learns where text ends
    # The documents whose hash falls in [start, end): [0.99, 1] is a 1% held out.
    split: Split = (0.0, 1.0)


class ChatConfig(Config):
    """Conversations in the OpenAI format: a list of `{role, content}` messages per row
    of Parquet files, `content` a string or a list of `{type: text, text}` parts,
    rendered with a chat template and tokenized by the run's `tokenizer`.

    The loss is on the replies `train_on` picks, each with the token that ends it
    (`<|im_end|>`, `<|eot_id|>`), so that the model learns to stop. A conversation
    with nothing to learn is dropped. Like `text`, it is prepared once into rows by
    the run's `packing`."""

    type: Literal["chat"] = "chat"
    path: str  # a glob of Parquet files, relative to the working directory
    column: str = "messages"  # the column that holds each conversation's messages
    template: Literal["chatml", "llama3"] = "chatml"  # chatml: Qwen's, SmolLM's
    # assistant: every reply; last: the last reply alone; all: every token, prompts
    # and template too.
    train_on: Literal["assistant", "last", "all"] = "assistant"
    split: Split = (0.0, 1.0)  # the conversations whose hash falls in [start, end)


class TokensConfig(Config):
    """Pre-tokenized shards: flat binary files of token ids, read through memory maps,
    as modded-nanogpt's FineWeb shards. They need no tokenizer and no packing.

    A row is a window of `seq_len` tokens of a shard that reads as one text, every
    token a target. Windows overlap by a token, so that every token is a target once,
    and come in a new order every epoch."""

    type: Literal["tokens"] = "tokens"
    path: str  # a glob of shards, relative to the working directory
    dtype: Literal["uint16", "uint32"] = "uint16"  # a token id's type in the shards
    header: int = 0  # bytes before the tokens in every shard (modded-nanogpt: 1024)


class SyntheticConfig(Config):
    """Uniformly random tokens over the model's vocabulary: for smoke tests and for
    measuring throughput. Every row is drawn from a seed of its own, and nothing is
    read from disk."""

    type: Literal["synthetic"] = "synthetic"


Source = Annotated[
    TextConfig | ChatConfig | TokensConfig | SyntheticConfig,
    Field(discriminator="type"),
]


class Weighted(Config):
    """A source of a mixture and its weight: the term ``0.7 * web`` of
    ``0.7 * web + 0.3 * code``."""

    weight: Annotated[float, Field(gt=0)]  # its share, relative to the others' weights
    source: Source


class MixtureConfig(Config):
    """Sources in proportion to their weights, in every batch: a batch's rows are
    dealt out by the weights, normalized, the largest remainders first. It is written
    as a weighted sum, ``0.7 * web + 0.3 * code``; ``web + code`` is an even mix.
    A source whose weight leaves it no row of a batch is an error."""

    type: Literal["mix"] = "mix"
    sources: tuple[Weighted, ...]  # the sources and their weights, as written


DataConfig = Annotated[
    TextConfig | ChatConfig | TokensConfig | SyntheticConfig | MixtureConfig,
    Field(discriminator="type"),
]


class TokenizerConfig(Config):
    """A BPE tokenizer, run by gigatoken: a Hugging Face Hub repo or a
    `tokenizer.json`. `text` and `chat` data need one.

    Special tokens written in documents stay text, as tiktoken's `encode_ordinary`
    keeps them: a web page that mentions `<|endoftext|>` does not end there. Only
    `bos`, `eos` and chat templates put them in."""

    type: Literal["tokenizer"] = "tokenizer"
    name: str  # "gpt2", "Qwen/Qwen3-0.6B", or a path to a tokenizer.json
    # The token a document starts with, where its source sets `bos`; Llama 3's is
    # "<|begin_of_text|>".
    bos: str = "<|endoftext|>"
    # The token a document ends with, where its source sets `eos`; Llama 3's is
    # "<|end_of_text|>".
    eos: str = "<|endoftext|>"
    # Special tokens after the vocabulary, for a template it lacks: GPT-2's 50257 tokens
    # padded to 50304 leave room for ["<|im_start|>", "<|im_end|>"].
    added: tuple[str, ...] = ()


# --- Packing ----------------------------------------------------------------------------
# How `text` and `chat` documents become rows. Documents are shuffled by a hash of their
# content and packed `chunk` tokens at a time, each chunk a random sample of them. With
# `doc_mask`, positions start again at every document and attention stays within it;
# without, a row reads as one text.


class ConcatPacking(Config):
    """Documents end to end, cut into rows (GPT-2): no padding, and only documents
    that happen to fit are whole. Rows overlap by a token, so that every token is a
    target once."""

    type: Literal["concat"] = "concat"
    doc_mask: bool = False  # positions start at every document, attention within it
    chunk: Positive = 16_000_000  # about how many tokens of documents pack together


class GreedyPacking(Config):
    """Whole documents in their shuffled order, a row padded and closed when the next
    one does not fit (classic SFT). `max_docs=1` is one document a row."""

    type: Literal["greedy"] = "greedy"
    doc_mask: bool = True  # positions start at every document, attention within it
    # A document longer than a row: dropped, truncated to a row, or split into pieces
    # that fill rows. drop by default, as a cut conversation may lose its answer.
    long: Literal["drop", "truncate", "split"] = "drop"
    # The most documents and pieces of them in a row, for attention kernels that take
    # a fixed number of them. None: as many as fit.
    max_docs: Positive | None = None
    chunk: Positive = 16_000_000  # about how many tokens of documents pack together


class BestFitPacking(Config):
    """Best-fit decreasing (Ding et al., 2024): the longest documents first, each into
    the open row that it leaves least room in. Nearly no padding, and documents are
    only cut when they are longer than a row, into pieces that fill gaps."""

    type: Literal["best_fit"] = "best_fit"
    doc_mask: bool = True  # positions start at every document, attention within it
    # A document longer than a row: dropped, truncated to a row, or split into pieces
    # that fill gaps.
    long: Literal["drop", "truncate", "split"] = "split"
    # The most documents and pieces of them in a row, for attention kernels that take
    # a fixed number of them. None: as many as fit.
    max_docs: Positive | None = None
    sort: bool = True  # longest first; false keeps the shuffled order (online best fit)
    chunk: Positive = 16_000_000  # about how many tokens of documents pack together


PackingConfig = Annotated[
    ConcatPacking | GreedyPacking | BestFitPacking, Field(discriminator="type")
]


# --- Optimizers -----------------------------------------------------------------------
# The update rule. Its `lr` is the peak learning rate, which the `schedule` multiplies
# at every step. Weight decay is decoupled, times the learning rate, and applies to
# matrices only: not to norm scales, biases, other vectors or Engram's tables.


class AdamWConfig(Config):
    """AdamW (Loshchilov & Hutter, 2019), by optax. Weight decay applies to matrices
    and stacks of them, not to norm scales, biases, other vectors or what a module
    names in `no_weight_decay`."""

    type: Literal["adamw"] = "adamw"
    lr: float  # the peak learning rate, which the schedule scales
    b1: float = 0.9  # the decay of the first moment, the gradients' average
    b2: float = 0.95  # the decay of the second moment, the squared gradients' average
    eps: float = 1e-8  # added to the second moment's square root, against zero
    weight_decay: float = 0.1  # decoupled, of the matrices, times the learning rate


class MuonConfig(Config):
    """Muon (Jordan et al., 2024): momentum orthogonalized by Newton-Schulz iterations,
    for the matrices inside the blocks (stacked expert weights one expert at a time).
    The embedding, the LM head and every parameter that is not a matrix train with
    `adam` instead.

    With `consistent_rms` (Moonlight's 0.2), one learning rate suits both. How Muon
    reaches whole matrices that are split over devices is `methods.orthogonalize`."""

    type: Literal["muon"] = "muon"
    lr: float  # the peak learning rate, which the schedule scales
    momentum: float = 0.95  # the decay of the gradients' average it orthogonalizes
    nesterov: bool = True  # Nesterov's look-ahead, as Keller Jordan's Muon
    ns_steps: int = 5  # Newton-Schulz iterations
    # Their quintics, by name in bux.train.optim.ns_coefficients: Keller Jordan's, the
    # same at every iteration, or polar_express's (Amsel et al., 2025), one for each.
    ns_coefficients: str = "jordan"
    # What Newton-Schulz runs in, and what the updates move between devices in.
    ns_dtype: Literal["float32", "bfloat16"] = "bfloat16"
    weight_decay: float = 0.1  # decoupled, of its matrices, times the learning rate
    # Updates scaled to this RMS, as AdamW's roughly are, so that one lr suits both
    # (Moonlight: 0.2); none -> by sqrt(max(1, fan_out / fan_in)) (Keller Jordan).
    consistent_rms: float | None = 0.2
    adam: AdamWConfig | None = None  # for the rest; None -> adamw(lr=lr)


OptimizerConfig = Annotated[AdamWConfig | MuonConfig, Field(discriminator="type")]


# --- Learning-rate schedules ------------------------------------------------------------
# A schedule multiplies the optimizer's lr at every step. Each starts with `warmup`
# steps that rise linearly from lr / warmup at the first step to lr.


class ConstantSchedule(Config):
    """The peak learning rate throughout, after the warmup."""

    type: Literal["constant"] = "constant"
    warmup: int = 0  # steps of linear warmup


class CosineSchedule(Config):
    """Warmup, then a cosine decay to `final` times the peak at the last step."""

    type: Literal["cosine"] = "cosine"
    warmup: int = 0  # steps of linear warmup
    final: float = 0.1  # the multiplier at the last step: a fraction of the peak


class WSDSchedule(Config):
    """Warmup-stable-decay (Hu et al., 2024): warmup, the peak, then a cooldown to
    `final` times the peak over the last `decay` of the steps. Checkpoints from the
    stable phase can be cooled down later, to study several run lengths at once."""

    type: Literal["wsd"] = "wsd"
    warmup: int = 0  # steps of linear warmup
    # The share of the steps the cooldown takes, rounded to a number of steps.
    decay: Annotated[float, Field(ge=0, le=1)] = 0.2
    final: float = 0.0  # the multiplier at the last step: a fraction of the peak
    # The cooldown's curve; sqrt is 1 - sqrt(t) (Hägele et al., 2024).
    shape: Literal["linear", "sqrt", "cosine"] = "linear"

    def cooldown(self, steps: int) -> int:
        """Return the steps of the cooldown, in a run of `steps`."""
        return round(self.decay * steps)


ScheduleConfig = Annotated[
    ConstantSchedule | CosineSchedule | WSDSchedule, Field(discriminator="type")
]


# --- Loss, evaluation, checkpoints and logs ------------------------------------
# What a run minimizes, and what it does every so many steps: evaluate, save, log and
# profile. None of these but the loss is in the run's id.


class LossConfig(Config):
    """The loss a run minimizes: the cross-entropy of the first head, plus `z` times
    its z-loss, plus `mtp` times the mean cross-entropy of the heads after it, each of
    which predicts one token further ahead for every `nextn` before it. The losses
    layers report, as an MoE's balance loss, add with the weights set on the layer."""

    type: Literal["loss"] = "loss"
    z: float = 0.0  # weight of the z-loss, log(Z)^2 of the first head (PaLM: 1e-4)
    # Weight of the mean loss of the heads that predict further ahead (DeepSeek-V3).
    mtp: float = 0.3


class EvalConfig(Config):
    """Held-out loss every `every` steps and after the last: the first head's mean
    cross-entropy over the same `batches` batches of `data` each time, logged as
    `eval_loss`. A batch has the run's `batch` rows of `seq_len` tokens.

    Held-out batches are a stream of their own seed, not other documents: hold
    documents out of training by their hash, as `eval(data=web(split=[0.99, 1]))`
    beside `data = web(split=[0, 0.99])`."""

    type: Literal["eval"] = "eval"
    data: DataConfig  # a source or a mixture, as `data` takes
    every: Positive = 1000  # steps between evaluations
    batches: Positive = 10  # batches an evaluation reads, the same ones each time


class CheckpointConfig(Config):
    """Model and optimizer state, saved in the run's directory every `every` steps, at
    the end, and at the step a run stops at early (as at SIGTERM). A run that finds a
    checkpoint there resumes from the latest one, and its data from the batch after.

    Every process of a job saves its part, so the run's directory must be one that
    every node sees. A run resumes only with the methods it was saved with where they
    lay out its state: `repeat`, a circular pipeline's `rounds` and an uneven
    pipeline's `stages`."""

    type: Literal["checkpoint"] = "checkpoint"
    every: Positive = 1000  # steps between saves
    keep: Positive = 2  # the most recent checkpoints to keep


class LogConfig(Config):
    """Training metrics every `every` steps and after the last, to the console and to
    the run's metrics.jsonl: the loss and its terms, `grad_norm`, `lr`, `tokens`,
    `tokens_per_s`, and `mfu` on devices whose peak FLOPs bux knows."""

    type: Literal["log"] = "log"
    every: Positive = 10  # steps between logs


class ProfileConfig(Config):
    """A jax.profiler trace of `steps` steps, those after step `start`, in the run's
    directory under profile/, which TensorBoard's profile plugin and xprof read. On
    GPUs it also logs what a step spent in collectives (`comm_ms`), in collectives with
    no compute beside them (`exposed_comm_ms`), and running nothing (`idle_ms`)."""

    type: Literal["profile"] = "profile"
    start: Annotated[int, Field(ge=0)] = 10  # the step after which the trace starts
    steps: Positive = 5  # the steps it traces


# --- Precision ---------------------------------------------------------------------------
# What a run computes in, and the format each projection inside a block multiplies in
# (bux.architecture.precision.matmuls): by the kind of projection, by the residual
# branch it is in, and by the GEMM.

# The GEMMs of a projection y = x W: y's (fprop), and the gradient's by x (dgrad) and
# by W (wgrad).
Pass = Literal["fprop", "dgrad", "wgrad"]
PASSES: tuple[Pass, ...] = ("fprop", "dgrad", "wgrad")


# The formats of bux.architecture.precision.matmuls the routed experts' grouped GEMM
# multiplies in (bux.architecture.experts.grouped_blockscaled.RECIPES).
GROUPED = ("compute", "mxfp8", "mxfp8_2d")


def _grouped(fmt: str | None) -> str | None:
    if fmt not in (None, *GROUPED):
        raise ValueError(
            f"the routed experts multiply in {', '.join(GROUPED)}: bux has no grouped "
            f"GEMM in {fmt}"
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
    # The routed experts, by a grouped GEMM: compute, mxfp8 or mxfp8_2d; none is
    # `matmuls` where it is one of those, else compute, as no grouped GEMM takes nvfp4.
    experts: Annotated[str | None, AfterValidator(_grouped)] = None
    latent: str | None = None  # LatentMoE's projections to and from its latent
    recurrence: str | None = None  # the recurrent mixers' projections, but the output
    recurrence_out: str | None = None  # their output projection, Mamba's out_proj
    mtp: str | None = None  # nextn's projection, and every branch after the first head


class SwitchConfig(Config):
    """From `at` of the steps on, only `passes` multiply in the format: the end of a
    run in a higher precision. The steps compile again at the switch. NVIDIA's NVFP4
    recipe takes fprop to bfloat16 for the last 18%: `switch(at=0.82,
    passes=[dgrad, wgrad])`."""

    type: Literal["switch"] = "switch"
    at: Annotated[float, Field(gt=0, lt=1)]  # the share of the steps it switches at
    passes: tuple[Pass, ...] = ()  # the GEMMs in the format after it; [] is none


class PrecisionConfig(Config):
    """The dtype the forward and backward passes compute in, `compute`, and the format
    the projections inside blocks multiply in, `matmuls` (a format of
    bux.architecture.precision.matmuls: compute, mxfp8, mxfp8_2d, nvfp4).

    `roles` gives a kind of projection a format of its own, `first` and `last` keep
    residual branches at the ends of the model in compute, and of each projection's
    GEMMs only `passes` multiply in the format, until a `switch`. Parameters stay
    float32, and so do the optimizer's `moments` unless bfloat16, norm statistics,
    router scores and top-k, recurrent states and the loss; gradients are summed over
    rows in `reduce`. Unlike `mesh`, `layout`, `methods` and `compiler`, it is in the
    run's id, all but `eval`."""

    type: Literal["precision"] = "precision"
    compute: Literal["float32", "bfloat16"] = "bfloat16"  # the passes' dtype
    # The format the projections inside blocks multiply in, by name in
    # bux.architecture.precision.matmuls: compute, mxfp8, mxfp8_2d or nvfp4.
    matmuls: str = "compute"
    roles: RolesConfig = RolesConfig()  # a format by the kind of projection
    # The residual branches outside any other at the start of the model that multiply
    # in compute: a number of them, or a share of them, rounded up.
    first: Branches = 0
    last: Branches = 0  # as `first`, the last ones before the first head
    passes: tuple[Pass, ...] = PASSES  # the GEMMs that multiply in the format
    switch: SwitchConfig | None = None  # other `passes` late in the run; None: none
    # Evaluation multiplies as training does, or in compute.
    eval: Literal["train", "compute"] = "train"
    # The dtype the optimizer's moments are kept in between steps; they update in
    # float32 (DeepSeek-V3's AdamW in bfloat16).
    moments: Literal["float32", "bfloat16"] = "float32"
    # The dtype gradients are summed over the batch's rows in: compute, or float32 as
    # Megatron's are, which a step sums once (methods.accumulate = unreduced).
    reduce: Literal["compute", "float32"] = "compute"
    # What the MoE's exchanges send tokens to their experts in: compute, or MXFP8
    # (DeepSeek-V3's FP8 dispatch); what comes back, and gradients, go in compute.
    dispatch: Literal["compute", "mxfp8"] = "compute"

    def experts(self) -> str | None:
        """Return the format the routed experts multiply in: `roles.experts`, or where
        it is none, `matmuls` if a grouped GEMM multiplies in it; none is compute."""
        if self.roles.experts is None and self.matmuls in GROUPED:
            return None if self.matmuls == "compute" else self.matmuls
        return self.roles.experts


# --- Parallelism -----------------------------------------------------------------------
# How a run splits over devices, which changes how it computes, not what: none of
# `mesh`, `layout` and `methods` is in the run's id. bux.parallel reads them, and the
# parallelism guide explains them.

Axis = Literal["data", "stage", "fsdp", "context", "expert", "tensor"]
Axes = Axis | tuple[Axis, ...] | None  # the mesh axes a named dimension splits over


class MeshConfig(Config):
    """The device grid: a size for each of its six axes, whose product is the number
    of devices a run takes. It lays them out the slowest links first: stage, data,
    fsdp, context, expert, tensor, so that `tensor` stays inside a node. Which
    dimensions each axis splits is the `layout`'s to say."""

    type: Literal["mesh"] = "mesh"
    data: Positive | None = None  # copies of the model; None: the devices left
    stage: Positive = 1  # pipeline stages
    fsdp: Positive = 1  # devices FSDP splits parameters over, rows too
    context: Positive = 1  # devices a row's positions split over
    expert: Positive = 1  # devices the routed experts split over, rows too
    tensor: Positive = 1  # devices heads, hidden units and the vocabulary split over


class LayoutConfig(Config):
    """Which mesh axes each named dimension of parameters and activations splits over:
    an axis, a list of them, or none. A dimension the axes' sizes do not divide stays
    whole, and an axis of size 1 splits nothing, so the defaults only take effect as
    `mesh` sizes their axes. `layout(stream=[context, tensor])` is sequence
    parallelism."""

    type: Literal["layout"] = "layout"
    batch: Axes = ("data", "fsdp", "expert")  # rows
    seq: Axes = "context"  # positions, inside a layer
    stream: Axes = "context"  # positions of the residual stream, between branches
    embed: Axes = None  # the model's width
    vocab: Axes = "tensor"  # the embedding's and the head's vocabulary
    heads: Axes = "tensor"  # query heads, or a mixer's heads
    kv_heads: Axes = "tensor"  # key and value heads
    mlp: Axes = "tensor"  # hidden units
    experts: Axes = "expert"  # the routed experts
    latent: Axes = None  # low-rank bottlenecks: MLA, LatentMoE
    layers: Axes = None  # the stacked passes of a scanned repeat
    stages: Axes = "stage"  # those of a pipelined one, a share a stage
    # What FSDP splits a parameter's first whole dimension over: the axes that copy
    # dense parameters but `data`, which HSDP keeps as copies.
    fsdp: Axes = ("fsdp", "context", "expert")
    # 3: FSDP splits parameters, gradients and the optimizer state; 1: it splits the
    # optimizer state alone.
    zero: Literal[1, 3] = 3


def _named_none(v: object) -> object:
    """A method named none, which a .bux file writes as its none."""
    return "none" if v is None else v


def _listed(v: object) -> object:
    """One name, as a list of one."""
    return (v,) if isinstance(v, str) else v


Names = Annotated[tuple[str, ...], BeforeValidator(_listed)]


class RematConfig(Config):
    """What a rematerialized branch or scanned pass keeps for the backward pass, by the
    names layers give what they make (bux.architecture.repeats.remat); the backward
    computes the rest again. It is a value of `methods.remat`, where a preset is a
    name: save_attention is remat(save=[attention, recurrence])."""

    type: Literal["remat"] = "remat"
    save: Names = ()  # kept on the device: attention, qkv, mlp, dots, ...
    offload: Names = ()  # moved to the host's memory until the backward reads them
    # The passes of each repeat it rematerializes, from the first; the rest keep
    # everything. None: every pass.
    first: Annotated[int, Field(ge=0)] | None = None


class MethodsConfig(Config):
    """How each part that communicates is computed, by the name of a function in its
    registry in bux.parallel.methods (the registries reference lists them), and how a
    step takes its batch: in `microbatches`, its parameters gathered and its gradients
    reduced how often. Another method is a function registered under a new name, not
    a field."""

    type: Literal["methods"] = "methods"
    # Attention: auto (the fastest kernel that takes the call), xla, cudnn, tokamax;
    # with `seq` over `context`, also ring and ulysses.
    attention: str = "auto"
    # The MoE's experts: sorted or dense on each device's tokens; over the `expert`
    # axis, all_gather, all_to_all, ragged_all_to_all, ragged_tokens or nccl_ep.
    moe: str = "sorted"
    # The picks a device of all_gather or the ragged MoE methods takes, or all_to_all
    # sends each peer, a multiple of its own (of its share for a peer); what does not
    # fit drops, which `dropped` reports. None: all that come.
    capacity: Annotated[float, Field(gt=0)] | None = None
    # The slices of each device's tokens that expert parallelism sends one after
    # another, so that one's exchange runs beside another's experts.
    moe_chunks: Positive = 1
    # nccl_ep's kernels on a GPU stream of their own, beside the compute stream's.
    moe_stream: bool = False
    # The linear attentions' recurrence: scan, all of a row's positions at once on each
    # device; pass_state, with `seq` over `context`, each device's chunk of positions
    # from the state the chunks before pass it (LASP-2).
    recurrence: str = "scan"
    # What the backward pass of a recurrence by chunks reads of its chunks' states:
    # kept from the forward pass, or computed again from its inputs, in a fourteenth
    # of the memory and a fifth more time. The delta rule by cuDNN's kernels keeps only
    # its inputs either way.
    states: Literal["keep", "recompute"] = "keep"
    # A repeat's passes: unroll, a module each; scan, stacked into one module that a
    # loop runs, which compiles once for all of them.
    repeat: str = "unroll"
    # A policy by name, or as data: remat(save=[attention], first=8). auto: the first
    # of none, dots_attention, dots, save_attention and full whose step fits.
    remat: Annotated[str | RematConfig, BeforeValidator(_named_none)] = "auto"
    # 1f1b, zbh1, gpipe or circular: 1f1b runs where gpipe does, but in rounds and
    # where a layer outside the pipelined repeat learns from the whole batch.
    pipeline: str = "1f1b"
    rounds: Positive = 1  # times a circular pipeline sends a microbatch round
    # The passes of the pipelined repeat each stage runs, first to last. None: under
    # 1f1b and zbh1, the split that balances the stages' FLOPs, the embedding's on the
    # first and the head's on the last (bux.train.stages); else as even as it goes.
    stages: list[Positive] | None = None
    # How Muon reaches whole matrices: gspmd, on the shards the compiler gives; gather,
    # each whole on every device that holds it; owner, each whole on one device.
    orthogonalize: str = "owner"
    # How the loss takes a head's logits: chunks, a chunk of positions at a time, so
    # that it never makes them all; logits, whole.
    loss: str = "chunks"
    # How TP's collectives meet the projections they feed: gspmd, as the compiler
    # places them; chunked, each device's rows in chunks, so that a chunk's collective
    # runs beside another's matmul.
    tensor: str = "gspmd"
    tensor_chunks: Positive = 2  # the chunks of each device's rows, under chunked
    # The microbatches a step takes its batch in, adding up their gradients; under a
    # pipeline, those it sends through its stages.
    microbatches: Positive = 1
    # Each microbatch's forward beside the backward of the one before, a layer at a
    # time, so that one's collectives run beside the other's compute. Off: it holds
    # half a microbatch more and copies it across the loop, which pays only beside an
    # exchange left exposed, as NCCL EP's with moe_stream (the parallelism guide).
    interleave: bool = False
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
    options of one's own over them. Like `mesh`, `layout` and `methods`, it changes
    how a run computes, not what, and is not in the run's id."""

    type: Literal["compiler"] = "compiler"
    preset: str = "default"  # XLA's own; o1, lhs, maxtext, collective_matmul
    # Over the preset's, as XLA_FLAGS writes them: "--xla_gpu_enable_triton_gemm=false
    # --xla_gpu_threshold_for_windowed_einsum_mib=0". XLA's debug options, and jax's
    # optimization_level and memory_fitting_level.
    options: Annotated[str, AfterValidator(_flags)] = ""
    # Where the step's gradient reductions start: early, each as its gradient is
    # computed (bux.train.reorder, a pass over XLA's schedule); or where XLA puts them.
    reductions: Literal["early", "xla"] = "early"


class TrainConfig(Config):
    """The training settings of a run: one field per top-level setting of a .bux file.
    A file that sets any of them is a run, and needs the required ones: `data`,
    `optimizer`, `seq_len`, `batch` and `steps`."""

    data: DataConfig  # what the run trains on: a source, or a weighted sum of them
    tokenizer: TokenizerConfig | None = None  # for text and chat data
    packing: PackingConfig = BestFitPacking()  # how text and chat documents fill rows
    optimizer: OptimizerConfig  # the update rule, with its peak learning rate
    schedule: ScheduleConfig = ConstantSchedule()  # the learning rate's multiplier
    seq_len: Annotated[int, Field(ge=2)]  # tokens per row
    batch: Positive  # rows per step, over all devices
    steps: Positive  # optimizer steps
    seed: int = 0  # of the model's initialization and of the data's order
    clip: float | None = 1.0  # clip gradients to this global norm; none disables it
    loss: LossConfig = LossConfig()  # the loss's terms and their weights
    eval: EvalConfig | None = None  # held-out loss every so many steps; none: no eval
    checkpoint: CheckpointConfig | None = None  # saves a run resumes from; none: none
    log: LogConfig = LogConfig()  # metrics to the console and to metrics.jsonl
    profile: ProfileConfig | None = None  # a trace of a few steps; none: no trace
    precision: PrecisionConfig = PrecisionConfig()  # what the run computes in
    mesh: MeshConfig = MeshConfig()  # the device grid; by default every device on data
    layout: LayoutConfig = LayoutConfig()  # which axes each named dimension splits over
    methods: MethodsConfig = MethodsConfig()  # how what communicates is computed
    compiler: CompilerConfig = CompilerConfig()  # XLA's options for the training step


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
    """Return the sources of `c`, a mixture's in order."""
    return tuple(w.source for w in c.sources) if isinstance(c, MixtureConfig) else (c,)
