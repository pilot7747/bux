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

import itertools
import math
from typing import Annotated, Literal

from pydantic import AfterValidator, BeforeValidator, Field, model_validator

from bux.config import Config, RoPEScalingConfig

Positive = Annotated[int, Field(gt=0)]
Fraction = Annotated[float, Field(ge=0, le=1)]


def _listed(v: object) -> object:
    """One value, a name or a number, as a list of one."""
    return (v,) if isinstance(v, str | int) else v


# --- Data ---------------------------------------------------------------------------
# A source is an endless stream of rows of `seq_len` tokens: `input_ids`,
# `position_ids` (where each token is in its document, so one starts wherever they
# stop increasing) and `loss_mask` (whether each token is a target), in a new order
# every epoch. Documents (`text` and `chat`) become rows once, when bux.data tokenizes
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

    The template is a Hugging Face tokenizer's Jinja chat template, rendered as
    transformers renders it: the run's tokenizer's own, or another's. The loss is on
    the replies `train_on` picks, each from where the template's prompt for it ends
    through the special token that ends it (`<|im_end|>`, `<|eot_id|>`), so that the
    model learns to stop. A conversation with nothing to learn is dropped. Like
    `text`, it is prepared once into rows by the run's `packing`."""

    type: Literal["chat"] = "chat"
    path: str  # a glob of Parquet files, relative to the working directory
    column: str = "messages"  # the column that holds each conversation's messages
    # The Hub repo or folder whose chat template renders the conversations, as
    # "HuggingFaceTB/SmolLM-135M-Instruct"; none: the tokenizer's own.
    template: str | None = None
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
    """A Hugging Face tokenizer: a Hub repo or a `tokenizer.json`, run by gigatoken,
    or by Hugging Face's tokenizers where gigatoken cannot run it (SmolLM2's, or one
    that is not BPE). `text` and `chat` data need one.

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
    one does not fit (classic SFT). `max_docs=1` puts one document in each row."""

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
    # How it iterates, by name in bux.train.optim.ns_algorithms: standard, optax's, or
    # gram, on the Gram matrix of each matrix's shorter side (Zhang et al., 2026),
    # with fewer products the further a matrix is from square.
    ns_algorithm: str = "standard"
    # The iterations after which gram restarts from the matrix so far, before rounding
    # grows; none: after the second and every third after it, 2, 5, 8, ...
    ns_restarts: Annotated[tuple[Positive, ...], BeforeValidator(_listed)] | None = None
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


def _from_zero(warmup: float) -> float:
    if warmup < 0:
        raise ValueError(
            "a warmup is a count of steps from 0, or a share of them below 1"
        )
    return warmup


# Steps of warmup, or below 1 a share of the steps, rounded to a number of them.
Warmup = Annotated[
    int | Annotated[float, Field(gt=0, lt=1)], AfterValidator(_from_zero)
]

# How a cooldown or a piece goes from where it starts to where it ends: sqrt is
# 1 - sqrt(t) (Hägele et al., 2024); exp falls by e over it, moved to end where it
# should (IBM's); geometric by the same factor every step (MiniCPM's), so it starts
# and ends above 0; and constant is at its end from its first step.
Shape = Literal["linear", "sqrt", "cosine", "exp", "geometric", "constant"]


def _increasing(shares: tuple[float, ...]) -> tuple[float, ...]:
    if any(a >= b for a, b in itertools.pairwise(shares)):
        raise ValueError("the shares of the steps must increase")
    return shares


def _above_zero(shape: Shape, *ends: float) -> None:
    if shape == "geometric" and min(ends) <= 0:
        raise ValueError("a geometric cooldown or piece starts and ends above 0")


class ConstantSchedule(Config):
    """The peak learning rate throughout, after the warmup."""

    type: Literal["constant"] = "constant"
    warmup: Warmup = 0  # steps of linear warmup, or a share of them below 1


class CosineSchedule(Config):
    """Warmup, then a cosine decay to `final` times the peak at the last step, or at
    the `horizon` of the steps."""

    type: Literal["cosine"] = "cosine"
    warmup: Warmup = 0  # steps of linear warmup, or a share of them below 1
    final: float = 0.1  # the multiplier at the last step: a fraction of the peak
    # The share of the steps the cosine reaches `final` at, rounded to a step: above
    # 1, the run stops before it does, as OLMo 2's and 3's do; below, it holds there.
    horizon: Annotated[float, Field(gt=0)] = 1.0


class _Cooled(Config):
    """A schedule whose cooldown takes the last `decay` of the steps, which its
    subclasses declare."""

    def cooldown(self, steps: int) -> int:
        """Return the steps of the cooldown, in a run of `steps`."""
        return round(getattr(self, "decay") * steps)  # noqa: B009


class WSDSchedule(_Cooled):
    """Warmup-stable-decay (Hu et al., 2024): warmup, the peak, then a cooldown to
    `final` times the peak over the last `decay` of the steps. Checkpoints from the
    stable phase can be cooled down later, to study several run lengths at once."""

    type: Literal["wsd"] = "wsd"
    warmup: Warmup = 0  # steps of linear warmup, or a share of them below 1
    # The share of the steps the cooldown takes, rounded to a number of steps.
    decay: Annotated[float, Field(ge=0, le=1)] = 0.2
    final: float = 0.0  # the multiplier at the last step: a fraction of the peak
    shape: Shape = "linear"  # the cooldown's curve

    @model_validator(mode="after")
    def _ends(self) -> "WSDSchedule":
        """Check that a geometric cooldown ends above 0."""
        _above_zero(self.shape, self.final)
        return self


class LinearDecaySchedule(Config):
    """Warmup, then a linear decay to `final` times the peak at the last step:
    straight to zero by default, which Bergsma et al. (2025) find best."""

    type: Literal["linear_decay"] = "linear_decay"
    warmup: Warmup = 0  # steps of linear warmup, or a share of them below 1
    final: float = 0.0  # the multiplier at the last step: a fraction of the peak


class MultiStepSchedule(Config):
    """Warmup, then the peak, times `factor` from each share of the steps in `at`
    on: DeepSeek LLM's 0.316 from 80% of the steps and again from 90%."""

    type: Literal["multistep"] = "multistep"
    warmup: Warmup = 0  # steps of linear warmup, or a share of them below 1
    # The shares of the steps it drops at, in order, each rounded to a step.
    at: Annotated[tuple[Fraction, ...], AfterValidator(_increasing)] = (0.8, 0.9)
    factor: float = 0.316  # what each drop multiplies by; 0.316 is 1 / sqrt(10)


class PowerSchedule(_Cooled):
    """IBM's Power scheduler (Shen et al., 2024): warmup, then `a` times the batch
    times n to the `b`, n the tokens trained, capped at the peak; then a cooldown to
    `final` times where it starts, over the last `decay` of the steps. The warmup
    rises to the power law's value at the warmup's last step. The peak is the
    optimizer's lr: the paper's is 0.02, under µP."""

    type: Literal["power"] = "power"
    warmup: Warmup = 0  # steps of linear warmup, or a share of them below 1
    a: float = 4.0  # the power law's amplitude, for a batch of one row
    b: float = -0.51  # its exponent, of the tokens trained
    # The share of the steps the cooldown takes, rounded to a number of steps.
    decay: Fraction = 0.1
    final: float = 0.0  # the multiplier at the last step: a fraction of the start
    shape: Shape = "exp"  # the cooldown's curve

    @model_validator(mode="after")
    def _ends(self) -> "PowerSchedule":
        """Check that a geometric cooldown ends above 0."""
        _above_zero(self.shape, self.final)
        return self


class Piece(Config):
    """A piece of a `piecewise` schedule: from the multiplier the piece before ends
    at, the peak for the first, to `to`, along `shape`, until `until` of the
    steps."""

    type: Literal["piece"] = "piece"
    until: Fraction = 1.0  # the share of the steps it ends at, rounded to a step
    to: float | None = None  # the multiplier it ends at; none holds where it starts
    shape: Shape = "constant"  # how it gets there; constant is there at once


def _in_order(pieces: tuple[Piece, ...]) -> tuple[Piece, ...]:
    if not pieces or pieces[-1].until != 1:
        raise ValueError("the last piece ends at 1, the end of the run")
    _increasing(tuple(p.until for p in pieces))
    for p, start in zip(pieces, levels(pieces), strict=False):
        _above_zero(p.shape, start, start if p.to is None else p.to)
    return pieces


def levels(pieces: tuple[Piece, ...]) -> list[float]:
    """Return the multiplier each piece starts at, then the one the last ends at."""
    found = [1.0]
    for p in pieces:
        found.append(found[-1] if p.to is None else p.to)
    return found


class PiecewiseSchedule(Config):
    """Warmup, then `pieces`, one after the other from the end of the warmup: holds,
    decays, drops and re-warmups, as tech reports chain them. DeepSeek-V3's holds
    the peak to 10T of its 14.8T tokens, decays along a cosine to 10% of it by
    14.3T, holds that to 14.633T, and drops to 3.3% for the rest::

        piecewise(warmup=2k, pieces=[piece(until=10 / 14.8),
                                     piece(until=14.3 / 14.8, to=0.1, shape=cosine),
                                     piece(until=14.633 / 14.8),
                                     piece(to=0.033)])
    """

    type: Literal["piecewise"] = "piecewise"
    warmup: Warmup = 0  # steps of linear warmup, or a share of them below 1
    # The pieces in order, each ending later than the one before; the last at 1.
    pieces: Annotated[tuple[Piece, ...], AfterValidator(_in_order)]

    def cooldown(self, steps: int) -> int:
        """Return the steps after the first piece, in a run of `steps`."""
        return steps - round(self.pieces[0].until * steps)


ScheduleConfig = Annotated[
    ConstantSchedule
    | CosineSchedule
    | WSDSchedule
    | LinearDecaySchedule
    | MultiStepSchedule
    | PowerSchedule
    | PiecewiseSchedule,
    Field(discriminator="type"),
]
# The schedules with a part at their end that the warmup must leave room for.
COOLED = (WSDSchedule, PowerSchedule, PiecewiseSchedule)


def warmup_steps(c: ScheduleConfig, steps: int) -> int:
    """Return the steps of the warmup of `c`, in a run of `steps`."""
    return round(c.warmup * steps) if isinstance(c.warmup, float) else c.warmup


# --- Curriculum and ramps ---------------------------------------------------------------
# What a run reads can change as it trains: each phase of a `curriculum` takes rows of
# another `seq_len`, another `batch` or other `data` from a step on, and its full
# attention another RoPE; a `batch_ramp` grows the batch over the first steps. The
# steps compile again at every new shape of batch and RoPE (bux.train.curriculum).

# A step, or a share of the steps below 1, rounded to a step.
At = Annotated[int, Field(gt=0)] | Annotated[float, Field(gt=0, lt=1)]


class PhaseConfig(Config):
    """From step `at` on, rows of `seq_len` tokens, `batch` of them a step, from
    `data`, each as the phase before where it is none: the run's own settings before
    the first phase. ``phase(at=0.9, seq_len=8192, batch=64, data=long)`` reads the
    last tenth of the run at 8k tokens a row from `long`.

    A source goes on, from one phase to the next, from the row it stopped at, where
    it is read at the same `seq_len`; at another, it is a dataset of its own, which
    starts from its first row. An eval reads rows of the run's own `seq_len` and
    `batch`, whichever phase it is in.

    `rope_scaling` and `rope_base` extend the context as DeepSeek-V3 and Qwen do:
    from the phase's first step on, every attention layer that rotates and has no
    `sliding_window_size` takes them in place of its own, as does its indexer; so do
    the evals made from that step on, and an export of a checkpoint from then.
    ``phase(at=0.9, seq_len=16384, rope_scaling=yarn(factor=8, original_max_len=2048))``
    trains the last tenth at 16k tokens with YaRN."""

    type: Literal["phase"] = "phase"
    at: At  # the step it starts at, or a share of the steps: 0.9 starts the last tenth
    # Tokens per row; none: as before.
    seq_len: Annotated[int, Field(ge=2)] | None = None
    batch: Positive | None = None  # rows per step; none: as before
    data: DataConfig | None = None  # a source or a mixture; none: as before
    # The RoPE scaling of full-attention layers (yarn, pi, llama3); none: as before.
    rope_scaling: RoPEScalingConfig | None = None
    # The RoPE base of full-attention layers, raised for long context (Llama 3's
    # ABF); none: as before.
    rope_base: Annotated[float, Field(gt=0)] | None = None

    def start(self, steps: int) -> int:
        """Return the step it starts at, in a run of `steps`."""
        return round(self.at * steps) if isinstance(self.at, float) else self.at


class RampConfig(Config):
    """A batch that grows from `start` rows at the first step to the run's `batch`
    over `steps` steps, as GPT-3's and DeepSeek-V3's do: linearly, rounded down to a
    multiple of `multiple`, and never below `start`. Each batch size it takes
    compiles the step again, so a coarser `multiple` compiles fewer."""

    type: Literal["ramp"] = "ramp"
    start: Positive  # rows per step at the first step
    steps: Positive  # the steps it takes to reach the run's batch
    multiple: Positive | None = None  # of the batch sizes it takes; none: `start`


def _phases(v: object) -> object:
    """One phase, as a list of one."""
    return (v,) if isinstance(v, PhaseConfig) else v


# A phase, or a list of them in the order they start.
Phases = Annotated[tuple[PhaseConfig, ...], BeforeValidator(_phases)]


# --- Loss, evaluation, checkpoints and logs ------------------------------------
# What a run minimizes, and what it does every so many steps: evaluate, save, log,
# observe and profile. None of these but the loss is in the run's id.


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
    """Held-out loss every `every` steps and after the last, over the same `batches`
    batches of `data` each time: the first head's mean cross-entropy over their
    targets, logged as `eval_loss`, and with a `tokenizer`, its bits per byte,
    `eval_bpb`. A batch has the run's `batch` rows of `seq_len` tokens, before any
    phase of its curriculum, or the eval's own: `eval(name="long", data=long,
    seq_len=16k, batch=16)` holds a model to long rows, as a long-context phase
    trains on.

    Bits per byte compare across tokenizers: the targets' nats over their bytes of
    text, in bits. A special token stands for no text, so they leave it out.

    A run takes a list of them, a held-out set each under a name of its own, which
    its metrics start with: `eval = [eval(name="web", data=web(split=[0.99, 1])),
    eval(name="code", data=code(split=[0.99, 1]))]` logs `web_loss`, `web_bpb`,
    `code_loss` and `code_bpb`.

    Held-out batches are a stream of their own seed, not other documents: hold
    documents out of training by their hash, as `eval(data=web(split=[0.99, 1]))`
    beside `data = web(split=[0, 0.99])`."""

    type: Literal["eval"] = "eval"
    # What its metrics' names start with: eval_loss, eval_bpb.
    name: Annotated[str, Field(min_length=1)] = "eval"
    data: DataConfig  # a source or a mixture, as `data` takes
    every: Positive = 1000  # steps between evaluations
    batches: Positive = 10  # batches an evaluation reads, the same ones each time
    # Tokens per row, at least 2; none: the run's seq_len.
    seq_len: Annotated[int, Field(ge=2)] | None = None
    batch: Positive | None = None  # rows per batch; none: the run's batch


def _each(evals: EvalConfig | tuple[EvalConfig, ...] | None) -> tuple[EvalConfig, ...]:
    return () if evals is None else evals if isinstance(evals, tuple) else (evals,)


def _apart(
    evals: EvalConfig | tuple[EvalConfig, ...] | None,
) -> EvalConfig | tuple[EvalConfig, ...] | None:
    names = [e.name for e in _each(evals)]
    if len(set(names)) < len(names):
        raise ValueError('each eval needs a name of its own: eval(name="web", ...)')
    return evals


# An eval, a list of them under names of their own, or none.
Evals = Annotated[EvalConfig | tuple[EvalConfig, ...] | None, AfterValidator(_apart)]


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


class WandbConfig(Config):
    """Weights & Biases: every metric and histogram of the run, to a run of `project`
    named as the run is, with the run's settings as its config and its sweep's
    choices among them. A run that resumes from a checkpoint goes on logging to the
    W&B run it started, whose id its directory keeps (wandb_id), and W&B's own files
    go there too. Only a job's first process logs. It needs the `wandb` extra
    (`uv sync --extra wandb`) and a key: `wandb login`, or `WANDB_API_KEY`."""

    type: Literal["wandb"] = "wandb"
    project: str = "bux"  # the W&B project the runs go to
    entity: str | None = None  # the team or user it is under; none: your default one
    group: str | None = None  # none: the file's name, which a sweep's runs share
    tags: tuple[str, ...] = ()  # tags of the W&B run, to filter runs by
    # online: as it trains; offline: into the run's directory, for `wandb sync`
    # later; disabled: nowhere.
    mode: Literal["online", "offline", "disabled"] = "online"


class TensorBoardConfig(Config):
    """TensorBoard: every metric and histogram of the run, as event files in the
    run's directory under tensorboard/, which `tensorboard --logdir runs` reads. A
    run that resumes drops the events past its checkpoint. Only a job's first process
    writes. It needs the `tensorboard` extra (`uv sync --extra tensorboard`)."""

    type: Literal["tensorboard"] = "tensorboard"


Tracker = Annotated[WandbConfig | TensorBoardConfig, Field(discriminator="type")]


class LogConfig(Config):
    """Training metrics every `every` steps and after the last, to the console and to
    the run's metrics.jsonl, and to each of `trackers`: the loss and its terms,
    `grad_norm`, `lr`, `tokens`, `tokens_per_s`, `mfu` on devices whose peak FLOPs
    bux knows, the time a step takes and where the host spends it, and the device's
    memory."""

    type: Literal["log"] = "log"
    every: Positive = 10  # steps between logs
    # Where else the metrics go: `[wandb(project="ablations"), tensorboard]`.
    trackers: tuple[Tracker, ...] = ()


class ObserveConfig(Config):
    """Every `every` steps, a training step that also measures the model by layer:
    the residual stream after each branch, each attention's logits on a sample of
    queries, each MoE's routing, each head's z-loss, and the norms of each layer's
    gradient, parameters and update. With `histograms`, histograms of ``log2 |x|`` of
    the stream after each branch and of each layer's gradient too. The metrics go
    wherever the log's do, at those steps.

    That step compiles apart, once more; the others run as they would without it.
    It makes the same update as they do, and takes longer: the more so with
    `histograms`."""

    type: Literal["observe"] = "observe"
    every: Positive = 100  # steps between the steps that observe the model
    histograms: bool = False  # take histograms of the stream and the gradients too


class ProfileConfig(Config):
    """A trace of `steps` steps, those after step `start`, in the run's directory under
    profile/. jax.profiler's, by default, is for TensorBoard's profile plugin and
    xprof; on GPUs it also logs what a step spent in collectives (`comm_ms`), in
    collectives with no compute beside them (`exposed_comm_ms`), and running nothing
    (`idle_ms`). nsys's and ncu's are the Nsight reports of a process that bux launch
    runs under the tool (bux.train.profilers)."""

    type: Literal["profile"] = "profile"
    start: Annotated[int, Field(ge=0)] = 10  # the step after which the trace starts
    steps: Positive = 5  # the steps it traces
    # What traces them, by name in bux.train.profilers: jax (jax.profiler), nsys
    # (Nsight Systems) or ncu (Nsight Compute)
    tool: str = "jax"
    # More flags for nsys or ncu, as its command line writes them:
    # "--cuda-memory-usage=true"
    options: str = ""


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
# parallelism guide (docs/parallelism/index.md) explains them.

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
    # Attention: auto (the fastest kernel that takes the call), xla, cudnn, tokamax,
    # fa4; with `seq` over `context`, also ring and ulysses.
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
    # A repeat's passes: unroll, a module each, run in a row; scan, stacked into one
    # module that a loop runs, where the passes can stack (else in a row).
    repeat: str = "unroll"
    # A policy by name, or as data: remat(save=[attention], first=8). auto: the first
    # of none, dots_attention, dots, save_attention and full whose step fits.
    remat: Annotated[str | RematConfig, BeforeValidator(_named_none)] = "auto"
    # 1f1b, zbh1, gpipe or circular. 1f1b runs wherever gpipe does, except with rounds
    # (circular) or a layer outside the pipelined repeat that learns from the whole
    # batch (gpipe).
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
    # Each microbatch's forward beside the backward of the one before, a piece of the
    # model at a time, so that one's collectives run beside the other's compute. Off
    # by default: it holds half a microbatch more and copies it across the loop, which
    # pays only beside an exchange left exposed, such as NCCL EP's with moe_stream
    # (docs/parallelism/microbatches.md).
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


# --- Search ----------------------------------------------------------------------------
# A setting written as a distribution, `optimizer.lr = log_uniform(1e-4, 1e-2)`, is a
# dimension of a search (bux.search): each trial of the search is a run that takes a
# value of it, which the search's method proposes, and the search keeps the trial whose
# objective is least. Outside a search, as `bux check` and `bux plan` show a run, a
# dimension takes its middle value: halfway along its scale, or its first option.

#: The significant digits of a value a trial takes, which name its run exactly.
DIGITS = 3


def significant(x: float) -> float:
    """Return `x` to `DIGITS` significant digits."""
    return float(f"{x:.{DIGITS}g}")


class _Range(Config):
    """A dimension between two bounds, `low` < `high`, which its subclasses declare."""

    @model_validator(mode="after")
    def _increasing(self) -> "_Range":
        low, high = getattr(self, "low"), getattr(self, "high")  # noqa: B009
        if not low < high:
            raise ValueError("low must be less than high")
        return self

    def _rounded(self, x: float) -> float:
        """Return `x` to `DIGITS` significant digits, but within the bounds."""
        low, high = getattr(self, "low"), getattr(self, "high")  # noqa: B009
        return min(max(significant(x), low), high)


class UniformConfig(_Range):
    """A dimension of a search: a number between `low` and `high`, uniformly, as
    ``optimizer.weight_decay = uniform(0, 0.2)``. Its bounds can be written in order,
    without their names."""

    type: Literal["uniform"] = "uniform"
    low: float  # the least value
    high: float  # the greatest value

    def value(self, u: float) -> float:
        """Return the value a fraction `u` of the way from `low` to `high`."""
        return self._rounded(self.low + u * (self.high - self.low))

    def unit(self, x: float) -> float:
        """Return how far `x` is from `low` to `high`, as a fraction: `value`'s inverse."""
        return (x - self.low) / (self.high - self.low)


class LogUniformConfig(_Range):
    """A dimension of a search: a number between `low` and `high`, uniformly on a log
    scale, as a learning rate is searched: ``optimizer.lr = log_uniform(1e-4, 1e-2)``.
    Its middle is the geometric mean."""

    type: Literal["log_uniform"] = "log_uniform"
    low: Annotated[float, Field(gt=0)]  # the least value, above 0
    high: float  # the greatest value

    def value(self, u: float) -> float:
        """Return the value a fraction `u` of the way from `low` to `high`, in logs."""
        return self._rounded(self.low * (self.high / self.low) ** u)

    def unit(self, x: float) -> float:
        """Return how far `x` is from `low` to `high` in logs: `value`'s inverse."""
        return math.log(x / self.low) / math.log(self.high / self.low)


class IntUniformConfig(_Range):
    """A dimension of a search: an integer from `low` to `high`, both included,
    uniformly, or on a log scale with `log`: ``schedule.warmup = int_uniform(100, 2k,
    log=true)``. For a few values that must divide others, as a number of heads,
    `choice` lists them."""

    type: Literal["int_uniform"] = "int_uniform"
    low: int  # the least value
    high: int  # the greatest value
    log: bool = False  # uniform on a log scale, for 0 < low

    def value(self, u: float) -> int:
        """Return the integer nearest to a fraction `u` of the way from `low` to
        `high`, each integer as likely, or in logs with `log`."""
        if self.log:
            # Each integer takes the stretch of the log scale nearest to it.
            low, high = math.log(self.low - 0.5), math.log(self.high + 0.5)
            x = math.exp(low + u * (high - low))
        else:
            x = self.low - 0.5 + u * (self.high - self.low + 1)
        return min(max(round(x), self.low), self.high)

    def unit(self, x: float) -> float:
        """Return the fraction of the way from `low` to `high` at the middle of the
        stretch that `x` takes: `value`'s inverse."""
        if self.log:
            low, high = math.log(self.low - 0.5), math.log(self.high + 0.5)
            return (math.log(x) - low) / (high - low)
        return (x - self.low + 0.5) / (self.high - self.low + 1)

    @model_validator(mode="after")
    def _positive_if_log(self) -> "IntUniformConfig":
        if self.log and self.low < 1:
            raise ValueError("int_uniform(log=true) needs low >= 1")
        return self


class ChoiceConfig(Config):
    """A dimension of a search: one of its options, each as likely, which can be any
    values, settings or blocks: ``ffn.activation = choice(swish, gelu)``, or with
    names, ``block = choice(pre=[norm attn] [norm ffn], par=[norm (attn + ffn)])``,
    which name the trials that take them. Where ``a | b`` is an axis of a sweep, a
    search over each of its runs, a choice is the search's to make."""

    type: Literal["choice"] = "choice"
    # The options' names, or the options as written: "swish", "gelu".
    options: Annotated[tuple[str, ...], Field(min_length=2)]

    def value(self, u: float) -> int:
        """Return the index of the option a fraction `u` of the way along them."""
        return min(int(u * len(self.options)), len(self.options) - 1)

    def unit(self, x: int) -> float:
        """Return the middle of the stretch of option `x`: `value`'s inverse."""
        return (x + 0.5) / len(self.options)


Distribution = Annotated[
    UniformConfig | LogUniformConfig | IntUniformConfig | ChoiceConfig,
    Field(discriminator="type"),
]


class RandomSearch(Config):
    """Random search (Bergstra & Bengio, 2012): every trial a point drawn at random,
    each dimension on its own scale. A baseline that is hard to beat in few
    dimensions, and whose trials need nothing of each other."""

    type: Literal["random"] = "random"


class HaltonSearch(Config):
    """Quasi-random search: the points of a scrambled Halton sequence, which cover the
    space more evenly than random ones, with no clumps and no gaps, however many
    trials there are, and need nothing of each other's results. What the Deep
    Learning Tuning Playbook (Godbole et al., 2023) explores a space with, and what
    `tpe` and `gp` start with."""

    type: Literal["halton"] = "halton"


class SobolSearch(Config):
    """Quasi-random search: the points of a scrambled Sobol sequence, evenest of all
    when the trials are a power of two, 8, 16 or 32: other counts lose its balance
    (Owen, 2020), and `halton` suits them."""

    type: Literal["sobol"] = "sobol"


class TPESearch(Config):
    """The tree-structured Parzen estimator (Bergstra et al., 2011), as Optuna's: after
    `startup` quasi-random trials, the trials are split into the best ``gamma`` of
    them and the rest, a density is fit to each, and of `candidates` points drawn
    from the best's, the one most likely under it and least under the rest's is
    tried. Trials still training count among the rest, so that concurrent trials
    differ.
    It handles choices and many dimensions well, and costs nothing to fit."""

    type: Literal["tpe"] = "tpe"
    startup: Positive = 10  # quasi-random trials before the densities propose
    candidates: Positive = 24  # points drawn from the best's density, to pick from
    # The share of the trials that are the best: ceil(gamma * n), at most 25.
    gamma: Annotated[float, Field(gt=0, lt=1)] = 0.1
    # One density over every dimension, which sees how they interact; or one each.
    multivariate: bool = True


class GPSearch(Config):
    """Bayesian optimization with a Gaussian process (Snoek et al., 2012), as BoTorch's
    and Vizier's: after `startup` quasi-random trials, a GP with a Matérn-5/2 kernel,
    a length scale for each dimension, is fit to the trials' objectives, and the point
    of most expected improvement over the best (log EI, Ament et al., 2023) is tried.
    Trials still training count as done at the GP's prediction for them (the kriging
    believer), so that concurrent trials differ. The most of each trial in few
    dimensions, for runs that take long."""

    type: Literal["gp"] = "gp"
    startup: Positive = 5  # quasi-random trials before the GP proposes
    # What the next point maximizes: expected improvement, in logs, or the upper
    # confidence bound of the improvement, mean + beta * deviation.
    acquisition: Literal["log_ei", "ucb"] = "log_ei"
    beta: Annotated[float, Field(gt=0)] = 2.0  # ucb's weight of the deviation


SearchMethod = Annotated[
    RandomSearch | HaltonSearch | SobolSearch | TPESearch | GPSearch,
    Field(discriminator="type"),
]


class HalvingConfig(Config):
    """Asynchronous successive halving (ASHA; Li et al., 2020): every trial trains a
    short run first, the bottom rung, and the best ``1 / eta`` of each rung train
    again, `eta` times longer, up to the run's own steps. A GPU that frees up takes
    the next trial up, if a rung has one, else a new trial at the bottom. Each rung
    is a run of its own, from scratch, with its schedule over its own steps: so a
    short run's warmup and decay are the long run's, shrunk, and its objective is
    the run's at that length.

    With `brackets` above 1 it is Hyperband (Li et al., 2018): the trials take turns
    starting at the bottom rung, the one above, and so on, which hedges against a
    short run ranking trials as a long one would not."""

    type: Literal["halving"] = "halving"
    # Each rung trains the best 1/eta of the one below, eta times longer.
    eta: Annotated[int, Field(ge=2)] = 3
    # Rungs of steps, the last the run's own: steps / eta^(rungs - 1), ..., steps.
    rungs: Annotated[int, Field(ge=2)] = 3
    brackets: Positive = 1  # Hyperband's brackets: the rungs trials start at, from 0

    @model_validator(mode="after")
    def _brackets(self) -> "HalvingConfig":
        if self.brackets > self.rungs:
            raise ValueError("halving needs brackets <= rungs")
        return self

    def steps(self, steps: int, rung: int) -> int:
        """Return the steps of `rung`, from 0, of a run of `steps`."""
        return max(1, round(steps / self.eta ** (self.rungs - 1 - rung)))


class SearchConfig(Config):
    """How a run with dimensions (`uniform`, `log_uniform`, `int_uniform`, `choice`)
    is searched: `trials` points, each a run of its own, proposed by `method`, ranked
    by the run's `objective`. Each run of a sweep is a search of its own, so
    ``attn.num_kv_heads = 1 | 4`` with ``optimizer.lr = log_uniform(1e-4, 1e-2)`` tunes
    the learning rate of each attention apart, and compares them at their best.

    `bux launch` trains the trials side by side, a GPU group each, and proposes the
    next as one ends. The search keeps its trials in its own directory, so a search
    launched again goes on where it stopped, and one given more trials adds them."""

    type: Literal["search"] = "search"
    method: SearchMethod = GPSearch()  # how the next trial is proposed
    trials: Positive = 16  # the points it tries, at the bottom rung with halving
    # Trials at once, at most; none: as many as the GPUs hold.
    parallel: Positive | None = None
    # Short runs first, and only the best trained to the end; none: every trial all.
    halving: HalvingConfig | None = None
    # Of the proposals, from 0, not of the runs, which keep the run's own.
    seed: Annotated[int, Field(ge=0)] = 0


class Term(Config):
    """A term of an objective: a metric and its weight, ``0.7 * web_bpb``."""

    metric: str  # a metric the run logs, as metrics.jsonl names it
    weight: float  # what its value is multiplied by


class ObjectiveConfig(Config):
    """What a search minimizes, and what ranks the runs of a sweep: a weighted sum of
    metrics a run logs, each at its latest value, written as one:
    ``objective = 0.7 * web_bpb + 0.3 * code_bpb``, ``(web_bpb + code_bpb) / 2``, or
    ``-tokens_per_s`` for the most tokens per second. Its names are metrics, not
    definitions, except for definitions of numbers: with ``w = 0.7``,
    ``w * web_bpb + (1 - w) * code_bpb`` is the same objective. A metric with a slash
    is a string, ``"time/step_ms"``."""

    type: Literal["objective"] = "objective"
    terms: Annotated[tuple[Term, ...], Field(min_length=1)]  # the metrics, weighted
    constant: float = 0.0  # added to the sum, which changes no ranking


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
    # A smaller batch at the first steps, which grows to `batch`; none: none.
    batch_ramp: RampConfig | None = None
    steps: Positive  # optimizer steps
    # Phases that each change seq_len, batch or data from a step on: a phase, or a list
    # of them; [] is none.
    curriculum: Phases = ()
    seed: int = 0  # of the model's initialization and of the data's order
    # Clip gradients to this global norm, above 0; none disables it.
    clip: Annotated[float | None, Field(gt=0)] = 1.0
    loss: LossConfig = LossConfig()  # the loss's terms and their weights
    # Held-out loss and bits per byte every so many steps: an eval, or a list of them
    # under names of their own; none: no eval.
    eval: Evals = None
    checkpoint: CheckpointConfig | None = None  # saves a run resumes from; none: none
    log: LogConfig = LogConfig()  # metrics to the console, metrics.jsonl and trackers
    # Metrics of the model by layer every so many steps; none: none.
    observe: ObserveConfig | None = None
    profile: ProfileConfig | None = None  # a trace of a few steps; none: no trace
    precision: PrecisionConfig = PrecisionConfig()  # what the run computes in
    mesh: MeshConfig = MeshConfig()  # the device grid; by default every device on data
    layout: LayoutConfig = LayoutConfig()  # which axes each named dimension splits over
    methods: MethodsConfig = MethodsConfig()  # how what communicates is computed
    compiler: CompilerConfig = CompilerConfig()  # XLA's options for the training step
    # What a search minimizes and a sweep's table ranks by: a weighted sum of metrics,
    # 0.7 * web_bpb + 0.3 * code_bpb; none: the first eval's bpb, else its loss, else
    # the training loss.
    objective: ObjectiveConfig | None = None
    search: SearchConfig = SearchConfig()  # how a run with dimensions is searched


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
    | LinearDecaySchedule
    | MultiStepSchedule
    | PowerSchedule
    | PiecewiseSchedule
    | Piece
    | PhaseConfig
    | RampConfig
    | LossConfig
    | EvalConfig
    | CheckpointConfig
    | LogConfig
    | WandbConfig
    | TensorBoardConfig
    | ObserveConfig
    | ProfileConfig
    | PrecisionConfig
    | RolesConfig
    | SwitchConfig
    | MeshConfig
    | LayoutConfig
    | RematConfig
    | MethodsConfig
    | CompilerConfig
    | UniformConfig
    | LogUniformConfig
    | IntUniformConfig
    | ChoiceConfig
    | RandomSearch
    | HaltonSearch
    | SobolSearch
    | TPESearch
    | GPSearch
    | HalvingConfig
    | SearchConfig,
    Field(discriminator="type"),
]

SOURCES = (TextConfig, ChatConfig, TokensConfig, SyntheticConfig, MixtureConfig)
DOCUMENTS = (TextConfig, ChatConfig)


def leaves(c: DataConfig) -> tuple[Source, ...]:
    """Return the sources of `c`, a mixture's in order."""
    return tuple(w.source for w in c.sources) if isinstance(c, MixtureConfig) else (c,)


def evals(t: TrainConfig) -> tuple[EvalConfig, ...]:
    """Return the run's evals, in order: none, one, or those of its list."""
    return _each(t.eval)


def evaluated(t: TrainConfig, e: EvalConfig) -> TrainConfig:
    """Return `t` with the shape of eval `e`'s batches: its `seq_len` and `batch`
    where it sets them."""
    shape = {k: v for k in ("seq_len", "batch") if (v := getattr(e, k)) is not None}
    return t.model_copy(update=shape) if shape else t


def objective(t: TrainConfig) -> ObjectiveConfig:
    """Return the run's objective: its own, or the first eval's bits per byte where
    it has a tokenizer, else that eval's loss, else the training loss."""
    if t.objective is not None:
        return t.objective
    first = next(iter(evals(t)), None)
    if first is None:
        metric = "loss"
    else:
        metric = f"{first.name}_{'bpb' if t.tokenizer is not None else 'loss'}"
    return ObjectiveConfig(terms=(Term(metric=metric, weight=1.0),))
