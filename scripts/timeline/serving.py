"""Price a .bux model's serving under an agentic coding load, without a GPU: an
analytic model of vLLM on one B300, calibrated on the tutorial "Old against new at
the same serving cost" (docs/tutorials/old-vs-new.md), whose load it prices.

A session is 2k shared tokens, 14k of its own, 4 turns of 512 output tokens and a
2k tool result between turns; C sessions run at once in a closed loop, 2C in all,
as ``scripts/old_vs_new/agentic.py`` runs them. What a model costs is read from
bux's compiler: its parameters by module (`bux.export.plan` places them, as vLLM
builds them up to the first head), the experts a batch of tokens touches, the
bytes each layer keeps per token or per sequence, and the FLOPs of a token and of
attention by its mask (`bux.train.flops`). A simulation of vLLM's scheduler
(chunked prefill, prefix caching in blocks, LRU eviction, preemption) turns them
into each engine step's work; a step takes a fitted overhead plus the larger of its
bytes over HBM bandwidth and its FLOPs over the peak, each at a fitted efficiency.

    python scripts/timeline/serving.py check
    python scripts/timeline/serving.py fit
    python scripts/timeline/serving.py cost examples/experiments/old_vs_new/new_2026.bux 'd_model = 3072'
    python scripts/timeline/serving.py match examples/experiments/old_vs_new/new_2026.bux --budget B1a
    python scripts/timeline/serving.py table
    python scripts/timeline/serving.py export examples/experiments/timeline/moe.bux --arm moe_8x2 --out exported/moe8x2 'd_model = 2048'

From Python: ``cost(file, settings)`` and ``match(file, budget)``.
"""

import argparse
import contextlib
import functools
import json
import math
import statistics
import sys
import traceback
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # imported where used: JAX is slow to import
    from bux.export.plan import LayerCall, Plan
    from bux.lang.ir import Architecture

ROOT = Path(__file__).resolve().parents[2]
OVN = ROOT / "examples" / "experiments" / "old_vs_new"
TIMELINE = ROOT / "examples" / "experiments" / "timeline"
RESULTS = Path("serving-out")  # where the table goes, under the working directory
BF16 = 2
#: Sessions at once that `cost` reports, and the grid `best` searches.
REPORTED = (8, 16, 32, 64, 128)
GRID = (1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128)
MOST_AT_ONCE = 256  # sessions at once an engine of tp GPUs is searched to
#: The reference of budget B1: a GPT-3 6.7B's serving shape (32 layers, d 4096, MHA).
REFERENCE = (OVN / "old_2022.bux", ("d_model = 4096",))
#: Layer types that mix nothing across positions and keep no state.
STATELESS = {
    "embed", "head", "layernorm", "rmsnorm", "dyt", "mlp", "gated_mlp", "moe",
    "linear", "act", "scale", "dropout", "nextn",
}  # fmt: skip
LINEAR = {"gla", "deltanet", "gdn", "kda", "gdn2", "lightning", "rwkv7"}
#: What bux_vllm runs (docs/export/vllm.md, "What runs").
VLLM_TYPES = STATELESS - {"dyt", "nextn"} | {
    "gqa", "mla", "csa", "mamba2", "gdn", "kda", "gdn2", "rwkv7", "engram", "merge",
}  # fmt: skip
VLLM_UNSUPPORTED = {"gqa": ("relative_bias",), "moe": ("expert_norm",)}
#: The recurrent layers vLLM keeps in its Mamba cache, of one type and shape a model.
VLLM_RECURRENT = ("mamba2", "gdn", "kda", "gdn2", "rwkv7")


class Unpriced(Exception):
    """A model the serving model cannot price, and why."""


# --- The hardware, the engine and the load ---------------------------------------


@dataclass(frozen=True)
class Hardware:
    """One GPU: a B300's HBM, bandwidth and dense bf16 peak, and its NVLink."""

    hbm: float = 288e9  # bytes, as vLLM's kv_blocks in the tutorial imply
    bandwidth: float = 8e12  # bytes/s
    peak: float = 2.25e15  # FLOP/s, dense bf16
    nvlink: float = 900e9  # bytes/s per direction, for tensor parallelism


@dataclass(frozen=True)
class Engine:
    """vLLM's settings, as agentic.py gives them."""

    utilization: float = 0.9  # gpu_memory_utilization
    batched_tokens: int = 16384  # max_num_batched_tokens
    tp: int = 1  # tensor_parallel_size


@dataclass(frozen=True)
class Load:
    """The agentic load of agentic.py's defaults."""

    system: int = 2048
    context: int = 14336
    turns: int = 4
    output: int = 512
    tool: int = 2048
    sessions: int = 2  # sessions in all, per session at once

    def prompts(self) -> list[int]:
        """Each turn's prompt."""
        first = self.system + self.context
        return [first + t * (self.output + self.tool) for t in range(self.turns)]


B300, VLLM, AGENTIC = Hardware(), Engine(), Load()


# --- What a model costs, from the compiler ----------------------------------------


@dataclass(frozen=True)
class Cache:
    """What one layer call keeps: bytes a token adds (up to `window` tokens), bytes
    of each token's index keys, read whole, and bytes of a sequence's state."""

    token: float = 0.0
    window: int | None = None
    top_k: int | None = None  # tokens of `token` a decoding query reads at most
    index: float = 0.0
    state: float = 0.0


@dataclass(frozen=True)
class Profile:
    """A model's serving costs, all per GPU at tensor parallelism 1."""

    d_model: int
    params: int  # all of them
    served: int  # those vLLM builds: up to the first head
    nonembed: int  # served, besides the embedding tables and the head
    active: float  # nonembed a token goes through (routed experts by top_k / E)
    dense_read: float  # bytes every step reads, but the routed experts
    experts: tuple[tuple[int, int, float], ...]  # (E, top_k, bytes) per MoE call
    caches: tuple[Cache, ...]
    light: int  # calls a step makes of attention, MLPs and other plain layers
    heavy: int  # calls of MoE and recurrent layers, each many kernels
    flops: float  # FLOPs of a token in dense matrix products, the head's apart
    moe_flops: float  # in the MoE layers
    mix_flops: float  # in recurrent mixing, besides its projections
    head_flops: float  # FLOPs of a sampled token's logits
    masks: tuple[
        tuple[int | None, int | None, float], ...
    ]  # (window, top_k, FLOPs/key)
    vllm: str  # "" if bux_vllm serves it, else why not
    name: str = field(default="", compare=False)

    @property
    def weights(self) -> float:
        return self.served * BF16

    def kv_per_token(self) -> float:
        """Bytes of cache a token adds at long context: windows and states left out."""
        return sum(c.token + c.index for c in self.caches if c.window is None)

    def state(self) -> float:
        return sum(c.state for c in self.caches)


def compile_arms(
    file: Path, settings: tuple[str, ...] = ()
) -> dict[str, "Architecture"]:
    """Every distinct architecture of `file` under `settings`, by its arm: the
    choices that every run with it took (``attention[gqa4]``), or the file's stem.
    An arm whose rules reject these settings (heads that do not split into its
    groups at this width) is left out, not the file."""
    from bux.lang import compile

    source = file.read_text()
    runs = _runs(source, file, settings)
    if not runs:  # not a run: its one model
        return {
            file.stem: compile(source, filename=str(file), overrides=list(settings))
        }
    groups: dict[str, list] = {}
    for run in runs:
        groups.setdefault(run.arch.model_dump_json(), []).append(run)
    arms = {}
    for same in groups.values():
        common = [c for c in same[0].choices if all(c in r.choices for r in same)]
        arms[f"{file.stem}[{', '.join(common)}]" if common else file.stem] = same[
            0
        ].arch
    return arms


def _runs(source: str, file: Path, settings: tuple[str, ...]) -> list:
    """`bux.lang.compile_runs`, but leaving out the combinations that fail; none for
    a file that is not a run. It walks the sweep as the compiler's own explorer does
    (private API: a research script's shortcut)."""
    from bux.lang import compiler as bc

    program = bc._load(source, str(file), list(settings), None, recover=False)
    found: list = []

    def explore(choices: dict) -> None:
        ev = bc._Evaluator(program, choices=choices, point=None)
        try:
            arch = bc._compile(ev, "model")
            train = bc._train(ev, arch, require=True)
            ok = True
        except bc.BuxError:  # this combination, or one of the axes met so far
            ok = False
        reached = sorted(ev.axes.values(), key=lambda axis: axis.span.start)
        free = [axis for axis in reached if axis.key not in choices]
        if free:
            for option in range(len(free[0].options)):
                explore({**choices, free[0].key: free[0].pick(option)})
        elif ok and train is not None:
            # every definition, those of other arms too: not this arm's to fail
            with contextlib.suppress(bc.BuxError):
                ev.evaluate_all()
            dims = list(ev.dims.values())
            found.append(bc._Explored(choices, reached, arch, train, dims))

    try:
        explore({})
    except bc.BuxError:
        return []
    return [bc._run(f, str(file)) for f in found]


@functools.cache
def _profile_cached(key: str, name: str) -> Profile:
    from bux.lang import ir

    return _profile(ir.Architecture.model_validate_json(key), name)


def profile(arch: "Architecture", name: str = "") -> Profile:
    """Return what serving `arch` costs (cached by its IR)."""
    return _profile_cached(arch.model_dump_json(), name)


def _profile(arch: "Architecture", name: str) -> Profile:
    from flax import nnx

    from bux import kernels
    from bux.architecture import Model, num_params
    from bux.architecture.repeats import remat
    from bux.export.plan import plan

    with remat.off(), kernels.off():
        model = nnx.eval_shape(lambda: Model(arch, rngs=nnx.Rngs(0)))
        p = plan(arch)
        calls = p.generation()
        sizes = _module_sizes(p, model, calls)
        flops, masks = _flops(arch, model)
    served = sum(s["all"] for s in sizes.values()) + _branches(p, model)
    embed = sum(s["all"] for s in sizes.values() if s["type"] == "embed")
    head = sum(s["all"] for s in sizes.values() if s["type"] == "head")
    tied = head < arch.d_model * (arch.vocab_size or 0)
    routed = sum(s["routed"] for s in sizes.values())
    idle = sum(s["routed"] * (1 - s["k"] / s["E"]) for s in sizes.values() if s["E"])
    tables = sum(s["all"] for s in sizes.values() if s["type"] == "engram")
    dense = (
        served
        - routed
        - embed
        - tables
        + (arch.d_model * (arch.vocab_size or 0) if tied else 0)
    )
    experts = tuple(
        (
            sizes[c.path]["E"],
            sizes[c.path]["k"],
            sizes[c.path]["routed"] * BF16 / sizes[c.path]["E"],
        )
        for c in calls
        if sizes[c.path]["E"]
    )
    caches = tuple(x for c in calls for x in layer_caches(c))
    total = num_params(model)
    after = total - served  # multi-token prediction's, which vLLM does not run
    head_flops = 2.0 * arch.d_model * (arch.vocab_size or 0)
    active = served - embed - head - tables - idle
    moe = sum(_moe_active(sizes[c.path]) for c in calls if sizes[c.path]["E"])
    return Profile(
        name=name,
        d_model=arch.d_model,
        params=total,
        served=served,
        nonembed=served - embed - head - tables,
        active=active,
        dense_read=dense * BF16,
        experts=experts,
        caches=caches,
        light=sum(_kind(c) == "light" for c in calls),
        heavy=sum(_kind(c) == "heavy" for c in calls),
        flops=2.0 * (active - moe),
        moe_flops=2.0 * moe,
        mix_flops=max(flops - 2.0 * after * _active_share(sizes) - 2.0 * active, 0.0),
        head_flops=head_flops,
        masks=masks,
        vllm=vllm_support(arch, calls),
    )


_CHEAP = {"embed", "head", "layernorm", "rmsnorm", "dyt", "act", "scale", "dropout"}
_HEAVY = LINEAR | {"moe", "mamba", "mamba2"}


def _kind(call: "LayerCall") -> str:
    """A call's overhead class: cheap (fused away), light, or heavy."""
    t = call.layer.config.type
    return "cheap" if t in _CHEAP else "heavy" if t in _HEAVY else "light"


def _moe_active(size: dict) -> float:
    """The parameters of an MoE layer a token goes through."""
    return size["all"] - size["routed"] * (1 - size["k"] / size["E"])


def _active_share(sizes: dict) -> float:
    """The share of the served parameters a token goes through, for MTP's FLOPs."""
    total = sum(s["all"] for s in sizes.values()) or 1
    idle = sum(s["routed"] * (1 - s["k"] / s["E"]) for s in sizes.values() if s["E"])
    return 1 - idle / total


def _module_sizes(p: "Plan", model: Any, calls: list["LayerCall"]) -> dict[str, dict]:
    """Each served module's parameters, its routed experts' apart, by its path."""
    import jax
    from flax import nnx

    from bux.export.weights import module_at

    out = {}
    for call in calls:
        if not call.first or call.path in out:
            continue
        module = module_at(p.arch.graph, model.graph, call.trail)
        c = call.layer.config
        E = c.num_experts if c.type == "moe" else 0
        size = {
            "type": c.type,
            "all": 0,
            "routed": 0,
            "E": E,
            "k": getattr(c, "top_k", 0),
        }
        leaves = jax.tree_util.tree_leaves_with_path(nnx.state(module, nnx.Param))
        for path, leaf in leaves:
            n = math.prod(leaf.shape)
            size["all"] += n
            top = str(getattr(path[0], "key", path[0]))
            if (
                E
                and leaf.shape
                and leaf.shape[0] == E
                and top not in ("router", "shared", "shared_gate")
            ):
                size["routed"] += n
        out[call.path] = size
    return out


def _branches(p: "Plan", model: Any) -> int:
    """The parameters of the residual algebra's branches."""
    import jax
    from flax import nnx

    from bux.export.weights import module_at

    n = 0
    for b in p.branches:
        if b.first:
            m = module_at(p.arch.graph, model.graph, b.trail)
            own = {k: v for k, v in vars(m).items() if k != "body"}
            for v in own.values():
                if isinstance(v, nnx.Module | nnx.Param):
                    state = nnx.state(v, nnx.Param) if isinstance(v, nnx.Module) else v
                    n += sum(math.prod(x.shape) for x in jax.tree.leaves(state))
    return n


def _flops(arch: "Architecture", model: Any, t: int = 4096) -> tuple[float, tuple]:
    """Return a token's FLOPs to the heads' input, and attention's FLOPs per key a
    query sees, by window and top_k (`bux.train.flops`)."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from bux.train import flops

    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def forward(params: Any, rest: Any, ids: Any) -> list:
        m = nnx.merge(graphdef, params, rest, copy=True)
        return [h.hidden for h in m.heads(ids)]

    ids = jax.ShapeDtypeStruct((1, t), jnp.int32)
    work = flops.count(forward, params, rest, ids)
    masks = []
    for mask, f in work.attention.items():  # f as if every query saw every key
        keys = mask.top_k if mask.picks and mask.top_k else t
        masks.append((mask.window, mask.top_k, f / (t * keys)))
    return work.fixed / t, tuple(masks)


def layer_caches(call: "LayerCall") -> list[Cache]:
    """What a layer call keeps in vLLM's cache.

    Raises:
        Unpriced: A token mixer of a type this model does not know, or on a widened
            residual stream.
    """
    from bux.config import AttentionConfig

    c, width = call.layer.config, call.layer.width
    if c.type in STATELESS or not (
        isinstance(c, AttentionConfig) or c.type == "engram"
    ):
        return []  # only token mixers keep a cache
    if any(port.name == "kv" and port.mode != "write" for port in call.layer.ports):
        return []  # it reads another layer's keys and values
    if not isinstance(width, int):
        raise Unpriced(f"no cache model of {c.type!r} on {width!r}")
    match c.type:
        case "gqa":
            dk = c.head_dim or width // c.num_heads
            dv = c.v_head_dim or dk
            return [_attention(c, c.num_kv_heads * (dk + dv), c.num_kv_heads)]
        case "mla":
            return [_attention(c, c.kv_lora_rank + c.qk_rope_head_dim, 1)]
        case "csa":
            return _csa(c)
        case t if t in LINEAR:
            return [Cache(state=_linear_state(c, width))]
        case "mamba2":
            inner = c.num_heads * c.head_dim if c.num_heads else c.expand * width
            ssm = inner * c.state_size
            conv = c.conv_size * (inner + 2 * c.num_groups * c.state_size)
            return [Cache(state=(ssm + conv) * BF16)]
        case "mamba":
            inner = c.expand * width
            return [Cache(state=(inner * c.state_size + c.conv_size * inner) * BF16)]
        case "conv":
            return [Cache(state=c.size * width * BF16)]
        case "short_conv":
            return [Cache(state=c.conv_size * width * BF16)]
        case "engram":
            return [Cache(state=(c.conv_size or 0) * width * BF16)]
    raise Unpriced(f"no cache model of {c.type!r}")


def _attention(c: Any, elements: int, kv_heads: int) -> Cache:
    """A softmax attention's cache: `elements` a token, and its indexer's keys."""
    index, top_k = 0.0, None
    if (ix := c.indexer) is not None:
        index = (kv_heads if ix.per_group else 1) * ix.head_dim * BF16
        top_k = ix.top_k
    return Cache(elements * BF16, c.sliding_window_size, top_k, index)


def _csa(c: Any) -> list[Cache]:
    """DeepSeek-V4's: a window of latents, and a latent per compress_ratio tokens."""
    window = Cache(c.head_dim * BF16, c.sliding_window_size)
    if c.compress_ratio is None:
        return [window]
    ix = c.indexer
    index = ix.head_dim * BF16 / c.compress_ratio if ix else 0.0
    top_k = ix.top_k if ix else None
    return [window, Cache(c.head_dim * BF16 / c.compress_ratio, None, top_k, index)]


def _linear_state(c: Any, width: int) -> float:
    """A linear attention's state and conv taps, in bf16 as vLLM keeps them."""
    hv = c.num_v_heads or c.num_heads
    dk, dv = c.head_dim, c.v_head_dim or c.head_dim
    conv = (c.conv_size or 0) * (2 * c.num_heads * dk + hv * dv)
    shift = width if c.type == "rwkv7" else 0
    return (hv * dk * dv + conv + shift) * BF16


def vllm_support(arch: "Architecture", calls: list["LayerCall"]) -> str:
    """Return "" if bux_vllm builds every layer `arch` serves with, else why not
    (as docs/export/vllm.md lists them)."""
    recurrent = set()
    once = Counter(call.path for call in calls)
    for call in calls:
        c = call.layer.config
        if c.type not in VLLM_TYPES:
            return f"{c.type}"
        if c.type == "gqa" and any(port.name != "v" for port in call.layer.ports):
            return "gqa kv or index ports"
        for name in VLLM_UNSUPPORTED.get(c.type, ()):
            value = getattr(c, name, None)
            if value not in (None, False) and not (
                name == "gate_head_bias" and not getattr(c, "gated", False)
            ):
                return f"{c.type} {name}"
        if c.type == "moe" and c.expert.up_softcap is not None:
            return "moe experts' up_softcap"
        if c.type in ("moe", *VLLM_RECURRENT) and once[call.path] > 1:
            return f"{c.type} called more than once"
        if c.type in VLLM_RECURRENT:
            recurrent.add(c.model_dump_json())
            if c.type == "gdn" and (c.conv_size is None or c.output_gate is None):
                return "gdn without conv or gate"
    return "recurrent layers of several shapes" if len(recurrent) > 1 else ""


# --- vLLM's memory --------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """How vLLM lays out a model's cache: its block of tokens, bytes per token and
    per state slot, and the bytes left for all of it."""

    block: int
    token: float  # bytes a token adds, but windowed layers'
    windows: tuple[tuple[int, float], ...]  # (window, bytes a token) of windowed layers
    slot: float  # bytes of one state slot of every recurrent layer together
    capacity: float
    attn_read: tuple[
        tuple[int | None, int | None, float], ...
    ]  # (window, top_k, bytes)
    state_rw: float  # bytes a decoding sequence's states take to read and write


#: vLLM's memory besides weights and cache: a fixed part and a share of the weights,
#: set to give the tutorial's caches (2022 7B: 463k tokens; 2026 d3072: 3589 blocks
#: of 816 = 2.93M; d3328 at utilization 0.94: 1.3M); they give 463k, 2.91M and 1.24M.
WORKSPACE = (2.5e9, 0.016)


def layout(p: Profile, hw: Hardware, e: Engine) -> Layout:
    """vLLM's cache layout of `p` on one GPU of `e.tp`."""
    tp = e.tp
    attn = [c for c in p.caches if c.token or c.index]  # heads split over the ranks
    states = [c.state for c in p.caches if c.state]
    full = sum((c.token + c.index) / tp for c in attn if c.window is None)
    windows = tuple((c.window, (c.token + c.index) / tp) for c in attn if c.window)
    block = 16
    slot = 0.0
    if states:
        # vLLM pads a layer's state to an attention layer's page, block tokens of it;
        # 816 for new_2026 at 3072, as measured. Without attention, a guess.
        per_layer = max((c.token / tp for c in attn), default=0.0)
        biggest = max(states) / tp
        block = 16 * math.ceil(biggest / per_layer / 16) if per_layer else 512
        slot = len(states) * max(biggest, block * per_layer)
    weights = p.weights / tp
    free = hw.hbm * e.utilization - weights - WORKSPACE[0] - WORKSPACE[1] * weights
    read = [(c.window, c.top_k, c.token / tp) for c in attn]
    read += [(None, None, sum(c.index / tp for c in attn))]
    return Layout(
        block, full, _grouped(windows), slot, free, _grouped(read), 2 * sum(states) / tp
    )


def _grouped(items: Iterable[tuple]) -> tuple:
    """(key..., bytes) items with their bytes summed by key, zeros left out."""
    out: dict = {}
    for *key, b in items:
        out[tuple(key)] = out.get(tuple(key), 0.0) + b
    return tuple((*k, b) for k, b in out.items() if b)


def footprint(lay: Layout, n: float) -> float:
    """Bytes of the attention cache of a sequence of n tokens."""
    return lay.token * n + sum(b * min(n, w) for w, b in lay.windows)


# --- The scheduler ----------------------------------------------------------------

# A step's features, which the coefficients weigh into seconds.
FEATURES = (
    "step", "light", "heavy", "prefill", "dense", "experts", "kv",
    "flops", "moe_flops", "mix_flops", "attn_flops", "allreduce", "nvlink",
)  # fmt: skip
F = {name: i for i, name in enumerate(FEATURES)}


@dataclass(eq=False)
class Request:
    """A turn of a session, as vLLM schedules it (equal only to itself)."""

    session: int
    turn: int
    prompt: int  # its prompt's tokens
    arrived: int  # the step it arrived after
    target: int = -1  # tokens to compute before it samples
    done: int = 0  # tokens of KV it has
    base: int = 0  # of them, the shared prefix's, which it does not hold
    output: int = 0
    first: int | None = None
    hit: int | None = None
    held: float = 0.0  # bytes of cache it holds, its state slots too

    def __post_init__(self) -> None:
        self.target = self.prompt if self.target < 0 else self.target


@dataclass
class Idle:
    """A session's cache between turns, or a preempted turn's: evictable. A hybrid's
    is of no use once any of it is evicted: its last state goes first."""

    tokens: int
    base: int
    held: float
    intact: bool = True


@dataclass
class Run:
    """What a simulation leaves: its steps' features and multiplicities, and each
    turn's marks, as indices of steps."""

    rows: list = field(default_factory=list)
    counts: list = field(default_factory=list)
    turns: list = field(default_factory=list)  # (arrived, first token, last token)
    prompts: int = 0
    hits: int = 0
    preemptions: int = 0


class Pool:
    """vLLM's blocks, in bytes: free ones, and idle sessions' cached ones in LRU
    order, evicted oldest first and from their tails."""

    def __init__(self, capacity: float):
        self.free = capacity
        self.idle: OrderedDict[int, Idle] = OrderedDict()
        self.idle_bytes = 0.0

    def room(self, keep: int | None = None) -> float:
        """Bytes it can give, without evicting `keep`'s blocks."""
        own = self.idle[keep].held if keep in self.idle else 0.0
        return self.free + self.idle_bytes - own

    def take(self, need: float) -> bool:
        """Take `need` bytes, evicting idle blocks if short."""
        if need > self.room():
            return False
        short = need - self.free
        while short > 0:
            sid, entry = next(iter(self.idle.items()))
            gone = min(entry.held, short)
            entry.tokens = int(entry.tokens * (1 - gone / entry.held))
            entry.held -= gone
            entry.intact = False
            self.idle_bytes -= gone
            self.free += gone
            short -= gone
            if entry.held <= 0:
                del self.idle[sid]
        self.free -= need
        return True

    def park(self, sid: int, entry: Idle) -> None:
        """Keep a session's blocks as idle, the most recent."""
        if entry.held <= 0:
            self.free += entry.held
            return
        self.idle[sid] = entry
        self.idle_bytes += entry.held

    def claim(self, sid: int) -> Idle | None:
        """Take a session's idle blocks out of the LRU, for it to use."""
        entry = self.idle.pop(sid, None)
        if entry is not None:
            self.idle_bytes -= entry.held
        return entry


class Scheduler:
    """vLLM v1's scheduler over a closed loop of `c` sessions: running requests
    first (a token each, or a chunk of their prefill), preempting the newest when
    blocks run out; then waiting ones, first come first served, with their prefix
    cache hits. Each turn's request arrives as the last one ends."""

    def __init__(self, p: Profile, lay: Layout, load: Load, e: Engine, c: int):
        self.p, self.lay, self.load, self.e, self.c = p, lay, load, e, c
        self.pool = Pool(lay.capacity)
        self.prompts = load.prompts()
        self.shared = 0  # tokens of the shared prefix the cache holds
        self.running: list[Request] = []
        self.waiting: list[Request] = []
        self.todo = list(range(load.sessions * c))
        self.out = Run()
        if (
            lay.capacity
            <= self._bytes(self.prompts[-1] + load.output, 0) + 2 * lay.slot
        ):
            raise Unpriced("its weights leave no room for one session's cache")
        for _ in range(c):
            self._start()

    def run(self) -> Run:
        while self.running or self.waiting:
            self._advance()
        return self.out

    def _bytes(self, done: float, base: float) -> float:
        return footprint(self.lay, done) - footprint(self.lay, base)

    def _start(self) -> None:
        if self.todo:
            sid = self.todo.pop(0)
            arrived = len(self.out.rows)
            self.waiting.append(Request(sid, 0, self.prompts[0], arrived))

    # A step ----------------------------------------------------------------

    def _advance(self) -> None:
        k = self._decode_stretch()
        if k > 1:
            self._decode(k)
            return
        budget, work = self.e.batched_tokens, []
        preempted = self._schedule_running(budget, work)
        if not preempted:
            self._schedule_waiting(budget - sum(n for _, n in work), work)
        self._record(work, 1)
        self._finish(work)

    def _decode_stretch(self) -> int:
        """How many steps every running request may decode with nothing else to
        schedule; 0 if not that many."""
        if not self.running or any(r.done < r.target for r in self.running):
            return 0
        if self.waiting and self._admissible(self.waiting[0], self.e.batched_tokens):
            return 0
        k = min(self.load.output - r.output for r in self.running)
        growth = sum(self._growth(r, 1) for r in self.running)
        if growth > 0:
            k = min(k, int(self.pool.room() / growth))
        return k

    def _decode(self, k: int) -> None:
        """Decode `k` steps of every running request at once."""
        for r in self.running:
            grow = self._growth(r, k)
            assert self.pool.take(grow)
            r.held += grow
        work = [(r, 1) for r in self.running]
        self._record(work, k)
        for r in self.running:
            r.done += k - 1
            r.output += k - 1
        self._finish(work)

    def _growth(self, r: Request, n: int) -> float:
        """Bytes `r` takes for n more tokens."""
        return footprint(self.lay, r.done + n) - footprint(self.lay, r.done)

    def _schedule_running(self, budget: int, work: list) -> bool:
        preempted = False
        for r in list(self.running):
            if r not in self.running:
                continue
            n = 1 if r.done >= r.target else min(r.target - r.done, budget)
            if n <= 0:
                continue
            while not self.pool.take(self._growth(r, n)):
                victim = self.running[-1]
                self._preempt(victim)
                preempted = True
                if victim is r:
                    break
            if r in self.running:
                r.held += self._growth(r, n)
                work.append((r, n))
                budget -= n
        return preempted

    def _preempt(self, r: Request) -> None:
        """Free `r`'s blocks to the cache, and queue it first, to compute again."""
        self.running.remove(r)
        self._park(r)
        r.target = r.prompt + r.output
        self.waiting.insert(0, r)
        self.out.preemptions += 1

    def _park(self, r: Request) -> None:
        """Leave `r`'s blocks, its last state among them, to the cache; free its
        state slot in use."""
        self.pool.free += self.lay.slot
        self.pool.park(r.session, Idle(r.done, r.base, r.held - self.lay.slot))
        r.held, r.done = 0.0, 0

    def _schedule_waiting(self, budget: int, work: list) -> None:
        while self.waiting and budget > 0 and len(self.running) < self.c:
            r = self.waiting[0]
            if not self._admissible(r, budget):
                return
            self.waiting.pop(0)
            n = min(r.target - self._hit(r), budget)
            self._admit(r, n)
            work.append((r, n))
            budget -= n

    def _plan(self, r: Request) -> tuple[int, int, float, float]:
        """Its hit, base, the bytes of its own blocks it reuses, and those of its
        idle entry."""
        entry = self.pool.idle.get(r.session)
        hit = self._hit(r)
        base = self.shared if self.shared and hit >= self.shared else 0
        keep = self._bytes(hit, base)
        have = entry.held if entry is not None else 0.0
        return hit, base, keep, have

    def _admissible(self, r: Request, budget: int) -> bool:
        hit, _, keep, have = self._plan(r)
        n = min(r.target - hit, budget)
        need = keep + self._bytes(hit + n, hit) + 2 * self.lay.slot
        return need <= self.pool.room(keep=r.session) + have

    def _admit(self, r: Request, n: int) -> None:
        hit, base, keep, have = self._plan(r)
        self.pool.claim(r.session)
        self.pool.free += have
        need = keep + self._bytes(hit + n, hit) + 2 * self.lay.slot
        assert self.pool.take(need)
        r.done, r.base, r.held = hit, base, need
        r.hit = hit if r.hit is None else r.hit
        self.running.append(r)

    def _hit(self, r: Request) -> int:
        """Tokens of `r`'s prompt the prefix cache serves, in whole blocks."""
        entry = self.pool.idle.get(r.session)
        own = 0
        if entry is not None and (entry.intact or not self.lay.slot):
            own = entry.tokens
        hit = min(max(own, self.shared), r.target - 1)
        return hit // self.lay.block * self.lay.block

    # Bookkeeping ----------------------------------------------------------

    def _record(self, work: list, k: int) -> None:
        """Add the features of k steps of `work`, (request, tokens) pairs."""
        self.out.rows.append(self._features(work, k))
        self.out.counts.append(k)

    def _features(self, work: list, k: int) -> np.ndarray:
        p, lay, tp = self.p, self.lay, self.e.tp
        f = np.zeros(len(FEATURES))
        tokens = sum(n for _, n in work)
        sampled = sum(1 for r, n in work if r.done + n >= r.target)
        f[F["step"]] = 1.0
        f[F["light"]], f[F["heavy"]] = p.light, p.heavy
        f[F["prefill"]] = (p.light + p.heavy) * any(n > 1 for _, n in work)
        f[F["dense"]] = p.dense_read / tp
        f[F["experts"]] = tokens  # made bytes by `expert_bytes`, at the fitted routing
        f[F["flops"]] = (tokens * p.flops + sampled * p.head_flops) / tp
        f[F["moe_flops"]] = tokens * p.moe_flops / tp
        f[F["mix_flops"]] = tokens * p.mix_flops / tp
        for r, n in work:
            start = r.done + (k - 1) / 2 if n == 1 else r.done
            f[F["kv"]] += _read(lay.attn_read, start + n) + lay.state_rw
            f[F["attn_flops"]] += _attention_flops(p.masks, start, n) / tp
        if tp > 1:
            calls = p.light + p.heavy
            f[F["allreduce"]] = calls
            f[F["nvlink"]] = calls * tokens * p.d_model * BF16 * 2 * (tp - 1) / tp
        return f

    def _finish(self, work: list) -> None:
        last = len(self.out.rows)
        for r, n in work:
            r.done += n
            if r.done < r.target:
                continue
            if r.first is None:
                r.first = last
                self._share()
            r.output += 1
            if r.output >= self.load.output:
                self._end_turn(r, last)

    def _share(self) -> None:
        """Pin the shared system prompt's blocks, once a request has computed it."""
        if not self.shared:
            self.shared = self.load.system // self.lay.block * self.lay.block
            assert self.pool.take(footprint(self.lay, self.shared))

    def _end_turn(self, r: Request, last: int) -> None:
        self.running.remove(r)
        self.out.turns.append((r.arrived, r.first, last))
        self.out.prompts += r.prompt
        self.out.hits += r.hit or 0
        self._park(r)
        if r.turn + 1 < self.load.turns:
            prompt = self.prompts[r.turn + 1]
            self.waiting.append(Request(r.session, r.turn + 1, prompt, last))
        else:
            self._start()


def _read(read: tuple, n: float) -> float:
    """Bytes a decoding query at n tokens reads from the cache."""
    total = 0.0
    for window, top_k, b in read:
        seen = n
        for most in (window, top_k):
            if most is not None:
                seen = min(seen, most)
        total += b * seen
    return total


def _attention_flops(masks: tuple, start: float, n: int) -> float:
    """Attention's FLOPs for n queries after `start` tokens: each sees itself and
    what is before it, within its window and top_k."""
    total = 0.0
    for window, top_k, per_key in masks:
        most = min(x for x in (window, top_k, math.inf) if x is not None)
        total += per_key * _seen(start, n, most)
    return total


def _seen(start: float, n: int, most: float) -> float:
    """Sum over i in [start, start + n) of min(i + 1, most)."""
    lo, hi = start + 1, start + n  # keys of the first and last query
    if hi <= most:
        return n * (lo + hi) / 2
    if lo >= most:
        return n * most
    below = most - lo
    return below * (lo + most - 1) / 2 + (n - below) * most


@functools.cache
def _simulate_cached(p: Profile, load: Load, e: Engine, hw: Hardware, c: int) -> Run:
    return Scheduler(p, layout(p, hw, e), load, e, c).run()


def simulate(
    p: Profile, c: int, load: Load = AGENTIC, e: Engine = VLLM, hw: Hardware = B300
) -> Run:
    """The steps vLLM takes to serve 2c sessions of `load`, c at a time."""
    return _simulate_cached(p, load, e, hw, c)


# --- From steps to seconds ---------------------------------------------------------


@dataclass(frozen=True)
class Coefficients:
    """Seconds per unit of each feature: overheads, and the efficiencies the
    bandwidth and the peak run at."""

    step: float = 1e-3  # seconds a step
    light: float = 10e-6  # seconds a light layer call takes a step
    heavy: float = 30e-6  # a heavy one (MoE, recurrent)
    prefill: float = 50e-6  # more for every call in a step with prefill
    dense_bw: float = 0.85  # of HBM bandwidth, reading dense weights
    moe_bw: float = 0.6  # reading routed experts
    kv_bw: float = 0.8  # reading the cache
    mfu: float = 0.6  # of the peak, on dense matrix products
    moe_mfu: float = 0.3  # on the MoE's
    mix_mfu: float = 0.1  # on recurrent mixing
    attn_mfu: float = 0.4  # on attention
    # A batch's T tokens pick experts as T ** routing independent ones would: below 1
    # if they agree, as vLLM's random weights may make them.
    routing: float = 1.0
    # Seconds a turn waits besides its GPU time (the frontend, the client), which
    # adds to its first token's time and to its session's, not to other sessions'.
    latency: float = 0.05
    allreduce: float = 10e-6  # seconds an all-reduce (a prior: tensor parallelism)
    nvlink: float = 0.7  # of NVLink's bandwidth (a prior)

    def vector(self, hw: Hardware) -> np.ndarray:
        """Seconds per unit of each of FEATURES."""
        bw, peak = hw.bandwidth, hw.peak
        per = {
            "step": self.step, "light": self.light, "heavy": self.heavy,
            "prefill": self.prefill, "dense": 1 / (bw * self.dense_bw),
            "experts": 1 / (bw * self.moe_bw), "kv": 1 / (bw * self.kv_bw),
            "flops": 1 / (peak * self.mfu), "moe_flops": 1 / (peak * self.moe_mfu),
            "mix_flops": 1 / (peak * self.mix_mfu), "attn_flops": 1 / (peak * self.attn_mfu),
            "allreduce": self.allreduce, "nvlink": 1 / (hw.nvlink * self.nvlink),
        }  # fmt: skip
        return np.array([per[name] for name in FEATURES])


PRIORS = Coefficients()
#: Fitted by ``serving.py fit`` to the tutorial's 6 models (2026-10-27).
FITTED = Coefficients(
    step=4.47e-4, light=7.95e-6, heavy=5.41e-5, prefill=8.00e-4, dense_bw=0.890,
    moe_bw=0.95, kv_bw=0.95, mfu=0.80, moe_mfu=0.503, mix_mfu=0.0331,
    attn_mfu=0.643, latency=0.154,
)  # fmt: skip
#: Exponent of the smooth maximum of a step's memory and compute times.
SHARPNESS = 4.0
FIT_NAMES = (
    "step", "light", "heavy", "prefill", "dense_bw", "moe_bw", "kv_bw",
    "mfu", "moe_mfu", "mix_mfu", "attn_mfu", "latency",
)  # fmt: skip


def expert_bytes(
    p: Profile, tokens: np.ndarray, routing: float, tp: int = 1
) -> np.ndarray:
    """Bytes of routed experts a step of `tokens` reads: of E experts, those that
    tokens ** routing independent tokens pick, top_k each, E (1 - (1 - k/E)^T)."""
    t = np.power(np.maximum(tokens, 1.0), routing)
    return sum(E * (1 - (1 - k / E) ** t) * b for E, k, b in p.experts) / tp + 0 * t


def step_times(rows: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Seconds of each row's step: overheads, then the smooth maximum of its memory
    time and its compute time. Rows' experts column holds bytes."""
    t = rows * w

    def total(*names: str) -> np.ndarray:
        return sum((t[:, F[n]] for n in names), np.zeros(len(t)))

    fixed = total("step", "light", "heavy", "prefill", "allreduce", "nvlink")
    memory = total("dense", "experts", "kv")
    compute = total("flops", "moe_flops", "mix_flops", "attn_flops")
    s = SHARPNESS
    return fixed + (memory**s + compute**s) ** (1 / s)


@dataclass
class Point:
    """A concurrency's predicted numbers."""

    concurrency: int
    sessions_per_hour: float
    tpot_ms: float  # p50 of each turn's time per output token
    ttft_s: float  # p50
    cache_hit: float
    preemptions: int
    seconds: float


def measure(
    run: Run,
    c: int,
    load: Load,
    coef: Coefficients,
    hw: Hardware,
    p: Profile,
    tp: int = 1,
) -> Point:
    """The numbers agentic.py would print for `run` of `p`, at `coef`, per GPU."""
    rows, counts = np.array(run.rows), np.array(run.counts, dtype=float)
    rows[:, F["experts"]] = expert_bytes(p, rows[:, F["experts"]], coef.routing, tp)
    w = coef.vector(hw)
    times = step_times(rows, w) * counts
    clock = np.concatenate([[0.0], np.cumsum(times)])
    # each of c workers runs load.sessions sessions of load.turns turns in a chain
    total = clock[-1] + coef.latency * load.sessions * load.turns
    tpot = [
        (clock[last] - clock[first]) / max(load.output - 1, 1)
        for _, first, last in run.turns
    ]
    ttft = [
        clock[first] - clock[arrived] + coef.latency for arrived, first, _ in run.turns
    ]
    sessions = load.sessions * c
    return Point(
        concurrency=c,
        sessions_per_hour=sessions / total * 3600 / tp,
        tpot_ms=statistics.median(tpot) * 1e3,
        ttft_s=statistics.median(ttft),
        cache_hit=run.hits / max(run.prompts, 1),
        preemptions=run.preemptions,
        seconds=total,
    )


# --- The tutorial's measurements, and the fit ---------------------------------------

#: Sessions per hour by concurrency, and p50 TPOT (ms) where the tutorial gives it:
#: vLLM 0.32 on one B300, docs/tutorials/old-vs-new.md, step 2.
MEASURED = {
    "old-4096": (OVN / "old_2022.bux", 4096, 0.9,
                 {8: 892, 16: 992, 32: 873, 64: 881, 12: 956, 20: 1012, 24: 948}, {8: 15.2}),
    "new-1536": (OVN / "new_2026.bux", 1536, 0.9, {8: 2672, 16: 4338, 32: 5736, 64: 7433}, {8: 4.0}),
    "new-2048": (OVN / "new_2026.bux", 2048, 0.9, {8: 1817, 16: 2709, 32: 3647, 64: 4808}, {8: 6.7}),
    "new-2560": (OVN / "new_2026.bux", 2560, 0.9, {8: 1256, 16: 1808, 32: 2425, 64: 3218}, {8: 9.9}),
    "new-3072": (OVN / "new_2026.bux", 3072, 0.9,
                 {8: 894, 16: 1332, 32: 1776, 64: 2335}, {8: 14.0, 16: 18.45}),
    "new-3328": (OVN / "new_2026.bux", 3328, 0.94, {8: 748, 16: 1140, 32: 1496, 64: 1052}, {8: 16.6}),
}  # fmt: skip
#: Measured, not fitted: cache hits and first-token times the tutorial gives.
CHECKS = {
    ("old-4096", 8): {"cache_hit": 0.75, "ttft_s": 0.14},
    ("old-4096", 64): {"cache_hit": 0.10, "ttft_s": 45.0},
    ("new-3072", 8): {"ttft_s": 0.39},
    ("new-3072", 16): {"cache_hit": 0.7149, "ttft_s": 0.387},
}


def _calibration(name: str) -> tuple[Profile, Engine, dict, dict]:
    file, d, util, sph, tpot = MEASURED[name]
    (arch,) = compile_arms(file, (f"d_model = {d}",)).values()
    return profile(arch, name), Engine(utilization=util), sph, tpot


def _residuals(
    coef: Coefficients, names: tuple[str, ...], hw: Hardware, load: Load
) -> list[tuple]:
    """(model, kind, concurrency, predicted, measured) for every measurement:
    sessions per hour, p50 TPOT and p50 TTFT."""
    out = []
    for name in names:
        p, e, sph, tpot = _calibration(name)
        ttft = {
            c: v["ttft_s"]
            for (n, c), v in CHECKS.items()
            if n == name and "ttft_s" in v
        }
        for c in sorted(set(sph) | set(tpot) | set(ttft)):
            point = measure(simulate(p, c, load, e, hw), c, load, coef, hw, p)
            for kind, measured in (
                ("sessions/h", sph),
                ("tpot_ms", tpot),
                ("ttft_s", ttft),
            ):
                if c in measured:
                    predicted = getattr(
                        point, kind.replace("sessions/h", "sessions_per_hour")
                    )
                    out.append((name, kind, c, predicted, measured[c]))
    return out


#: Weights of each kind of measurement's log error in the fit.
WEIGHTS = {"sessions/h": 1.0, "tpot_ms": 1.0, "ttft_s": 0.3}


#: Upper bounds of the efficiencies the fit may reach.
CAPS = {"dense_bw": 0.95, "moe_bw": 0.95, "kv_bw": 0.95, "mfu": 0.8, "moe_mfu": 0.8,
        "mix_mfu": 0.8, "attn_mfu": 0.8}  # fmt: skip


def fit(
    names: tuple[str, ...] = tuple(MEASURED),
    hw: Hardware = B300,
    load: Load = AGENTIC,
    start: Coefficients = PRIORS,
    free: tuple[str, ...] = FIT_NAMES,
) -> Coefficients:
    """Fit the coefficients `free` to the measurements of `names`, by least squares
    on log errors, the efficiencies within CAPS, pulled weakly toward `start`."""
    from scipy.optimize import least_squares

    x0 = np.log([getattr(start, n) for n in free])
    upper = np.array(
        [math.log(CAPS.get(n, 1.0)) if n in CAPS else np.inf for n in free]
    )
    x0 = np.minimum(x0, upper - 1e-6)

    def coef(x: np.ndarray) -> Coefficients:
        return replace(start, **dict(zip(free, map(float, np.exp(x)), strict=True)))

    def loss(x: np.ndarray) -> np.ndarray:
        rows = _residuals(coef(x), names, hw, load)
        errors = [WEIGHTS[k] * math.log(pr / me) for _, k, _, pr, me in rows]
        return np.array(errors + list(0.05 * (x - x0)))

    result = least_squares(loss, x0, bounds=(-np.inf, upper), diff_step=1e-3)
    return coef(result.x)


def _fit_report(coef: Coefficients, hw: Hardware, load: Load) -> str:
    rows = _residuals(coef, tuple(MEASURED), hw, load)
    lines = [
        f"{'model':10} {'what':11} {'c':>4} {'predicted':>10} {'measured':>9} {'error':>7}"
    ]
    for name, kind, c, pr, me in rows:
        lines.append(
            f"{name:10} {kind:11} {c:4} {pr:10.1f} {me:9.1f} {pr / me - 1:+7.1%}"
        )
    errs = [abs(math.log(pr / me)) for *_, pr, me in rows]
    sph = [abs(math.log(pr / me)) for _, k, _, pr, me in rows if k == "sessions/h"]
    lines.append(
        f"mean |log error|: all {np.mean(errs):.3f}, sessions/h {np.mean(sph):.3f}"
    )
    return "\n".join(lines)


def leave_one_out(hw: Hardware = B300, load: Load = AGENTIC) -> str:
    """Fit without each model in turn, and price it: its errors by kind."""
    lines, all_sph = [], []
    for out in MEASURED:
        rest = tuple(n for n in MEASURED if n != out)
        rows = _residuals(fit(rest, hw, load, FITTED), (out,), hw, load)
        sph = [pr / me - 1 for _, k, _, pr, me in rows if k == "sessions/h"]
        all_sph += sph
        detail = " ".join(f"{k}@{c} {pr / me - 1:+.0%}" for _, k, c, pr, me in rows)
        lines.append(
            f"{out}: sessions/h mean |error| {np.mean(np.abs(sph)):.1%}; {detail}"
        )
    lines.append(f"all: sessions/h mean |error| {np.mean(np.abs(all_sph)):.1%}")
    return "\n".join(lines)


# --- Cost and match ------------------------------------------------------------------


def _fits(p: Profile, hw: Hardware, e: Engine, load: Load) -> bool:
    lay = layout(p, hw, e)
    return (
        lay.capacity > footprint(lay, load.prompts()[-1] + load.output) + 2 * lay.slot
    )


def choose_tp(p: Profile, hw: Hardware, e: Engine, load: Load) -> int:
    """The least tensor parallelism (1, 2, 4, 8) whose GPUs hold the weights and a
    session's cache."""
    for tp in (1, 2, 4, 8):
        if _fits(p, hw, replace(e, tp=tp), load):
            return tp
    raise Unpriced("it does not fit on 8 GPUs")


def price(
    p: Profile,
    slo_ms: float = 30.0,
    concurrencies: tuple[int, ...] = REPORTED,
    best: bool = True,
    load: Load = AGENTIC,
    e: Engine | None = None,
    hw: Hardware = B300,
    coef: Coefficients | None = None,
) -> dict:
    """Return `p`'s numbers at each concurrency, per GPU (c sessions at once for
    each of its tp GPUs), and if `best`, its best on GRID: the most sessions per
    hour at a p50 TPOT within `slo_ms`."""
    e = e or replace(Engine(), tp=choose_tp(p, hw, Engine(), load))
    coef = coef or FITTED

    def at(c: int) -> Point:
        run = simulate(p, c * e.tp, load, e, hw)
        return replace(measure(run, c * e.tp, load, coef, hw, p, e.tp), concurrency=c)

    points = {c: at(c) for c in concurrencies}
    out = {"tp": e.tp, "points": {c: asdict(pt) for c, pt in points.items()}}
    if best:
        grid = [c for c in GRID if c * e.tp <= MOST_AT_ONCE]
        out["best"] = _best([points.get(c) or at(c) for c in grid], slo_ms)
    return out


def _best(points: list[Point], slo_ms: float) -> dict | None:
    """The point of the most sessions per hour within the SLO; None if none is."""
    ok = [pt for pt in points if pt.tpot_ms <= slo_ms]
    top = max(ok, key=lambda pt: pt.sessions_per_hour, default=None)
    return asdict(top) if top is not None else None


def cost(
    file: str | Path,
    settings: tuple[str, ...] = (),
    arm: str | None = None,
    slo_ms: float = 30.0,
) -> dict:
    """Price the model of `file` under `settings` (an arm of its sweep by name).

    Returns:
        ``{concurrency: {sessions_per_hour, tpot_ms, cache_hit, ...}, "best": ...,
        "tp": ..., "profile": ...}``.
    """
    p = _arm_profile(Path(file), tuple(settings), arm)
    priced = price(p, slo_ms)
    return {
        **priced["points"],
        "best": priced["best"],
        "tp": priced["tp"],
        "profile": summary(p),
    }


def _arm_profile(file: Path, settings: tuple[str, ...], arm: str | None) -> Profile:
    arms = compile_arms(file, settings)
    if arm is None:
        if len(arms) > 1:
            raise ValueError(
                f"{file} has {len(arms)} arms; pick one: {', '.join(arms)}"
            )
        arm = next(iter(arms))
    if arm not in arms:
        found = [a for a in arms if arm in a]
        if len(found) != 1:
            raise ValueError(f"no single arm {arm!r} in {file}: {', '.join(arms)}")
        arm = found[0]
    return profile(arms[arm], arm)


def summary(p: Profile) -> dict:
    """What the table shows of a profile."""
    return {
        "d_model": p.d_model,
        "params": p.params,
        "served": p.served,
        "nonembed": p.nonembed,
        "active": p.active,
        "kv_bytes_per_token": p.kv_per_token(),
        "state_bytes": p.state(),
        "vllm": p.vllm or "yes",
    }


@functools.cache
def budgets(slo_ms: float = 30.0) -> dict[str, float]:
    """Sessions per hour per GPU of the budgets: B1a, the reference's at 8 at once;
    B1b, its best within the SLO; B2a and B2b, a quarter of each (4 times the cost
    of a session)."""
    file, settings = REFERENCE
    (arch,) = compile_arms(file, settings).values()
    priced = price(profile(arch, "reference"), slo_ms)
    a = priced["points"][8]["sessions_per_hour"]
    b = priced["best"]["sessions_per_hour"]
    return {"B1a": a, "B1b": b, "B2a": a / 4, "B2b": b / 4}


def _value(p: Profile, point: str, slo_ms: float) -> float:
    """Sessions per hour per GPU at an operating point: "a" at 8 at once, "b" the
    best within the SLO (0 if none is)."""
    if point == "a":
        return price(p, slo_ms, (8,), best=False)["points"][8]["sessions_per_hour"]
    best = price(p, slo_ms, (), best=True)["best"]
    return best["sessions_per_hour"] if best else 0.0


def match(
    file: str | Path,
    budget: str | float = "B1b",
    settings: tuple[str, ...] = (),
    arm: str | None = None,
    slo_ms: float = 30.0,
    step: int = 128,
    start: float | None = None,
) -> dict:
    """Find the d_model at which `file`'s family costs `budget`: a name of
    `budgets` (its letter the operating point), or sessions per hour per GPU at the
    best within the SLO.

    Widths are tried at multiples of `step` that the family's own rules accept, and
    the match interpolates between the two nearest that bracket the budget: log
    d_model in log sessions per hour, then the parameters, log in log d_model.

    Returns:
        The matched d_model, its total, served, non-embedding and active
        parameters, and the sessions per hour of every width it priced.
    """
    file = Path(file)
    point = budget[-1] if isinstance(budget, str) else "b"
    target = budgets(slo_ms)[budget] if isinstance(budget, str) else float(budget)
    tried: dict[int, tuple[float, Profile | None]] = {}

    rejected: set[int] = set()

    def value(d: int) -> float | None:
        """Sessions per hour at width d; None if the family rejects it."""
        if d in rejected:
            return None
        if d not in tried:
            try:
                p = _arm_profile(file, (*settings, f"d_model = {d}"), arm)
            except Unpriced:
                raise
            except Exception:  # noqa: BLE001  (the family's rules reject the width)
                rejected.add(d)
                return None
            try:
                tried[d] = (_value(p, point, slo_ms), p)
            except Unpriced:
                tried[d] = (0.0, p)
        return tried[d][0]

    given = [x for x in settings if x.replace(" ", "").startswith("d_model=")]
    d0 = start or (_arm_profile(file, tuple(settings), arm).d_model if given else 3072)
    lo, hi = _bracket(value, round(d0), target, step)
    return _interpolated(tried, lo, hi, target, budget)


def _bracket(value: Callable, d0: int, target: float, step: int) -> tuple[int, int]:
    """The nearest two valid widths whose sessions per hour bracket `target`: the
    narrower at or above it, the wider below. Extrapolates as a power of d."""
    found: dict[int, float] = {}
    got = _valid(value, max(step, round(d0 / step) * step), step, found, +1)
    for _ in range(30):
        if got is None:
            break
        found[got[0]] = got[1]
        above = [x for x, v in found.items() if v >= target]
        below = [x for x, v in found.items() if v < target]
        if above and below and max(above) < min(below):
            lo, hi = max(above), min(below)
            got = _between(value, lo, hi, found, target, step)
            if got is None:
                return lo, hi
            continue
        wider = (
            bool(above) and not below
        )  # every width tried is cheaper than the budget
        nxt = _guess(found, target, step)
        got = _valid(value, nxt, step, found, +1 if wider else -1)
    raise Unpriced(f"no width brackets the budget near d_model {d0}: {found}")


def _between(
    value: Callable, lo: int, hi: int, found: dict[int, float], target: float, step: int
) -> tuple[int, float] | None:
    """A valid width strictly between lo and hi, the nearest the log interpolation
    first, and its value; None if the family accepts none."""
    vlo, vhi = found[lo], found[hi]
    t = math.log(vlo / target) / math.log(vlo / vhi) if vhi > 0 else 0.5
    guess = math.exp(math.log(lo) + t * (math.log(hi) - math.log(lo)))
    for d in sorted(range(lo + step, hi, step), key=lambda x: abs(x - guess)):
        if (v := value(d)) is not None:
            return d, v
    return None


def _guess(found: dict[int, float], target: float, step: int) -> int:
    """The next width to try: by the last two widths' slope in log-log, or a power
    of -2.5, from the nearest to the target."""
    ds = sorted(found, key=lambda x: abs(math.log(max(found[x], 1e-9) / target)))
    d = ds[0]
    slope = -2.5
    if len(ds) > 1 and found[ds[0]] > 0 and found[ds[1]] > 0 and ds[0] != ds[1]:
        fitted = math.log(found[ds[0]] / found[ds[1]]) / math.log(ds[0] / ds[1])
        slope = min(-0.5, max(-6.0, fitted))
    v = max(found[d], target / 64)
    guess = d * (target / v) ** (1 / slope)
    nxt = round(guess / step) * step
    if nxt in found:
        nxt = d + step if found[d] >= target else d - step
        while nxt in found:
            nxt += step if found[d] >= target else -step
    return max(step, nxt)


def _valid(
    value: Callable, d: int, step: int, found: dict, direction: int, reach: int = 8
) -> tuple[int, float] | None:
    """The first width from d on in `direction`, within `reach` steps, that the
    family accepts and has not been priced, and its value."""
    for i in range(reach + 1):
        x = d + direction * i * step
        if x >= step and x not in found and (v := value(x)) is not None:
            return x, v
    return None


def _interpolated(
    tried: dict, lo: int, hi: int, target: float, budget: str | float
) -> dict:
    """The match between widths lo and hi."""
    (vlo, plo), (vhi, phi) = tried[lo], tried[hi]
    t = math.log(vlo / target) / math.log(vlo / vhi) if vhi > 0 else 0.0
    t = min(max(t, 0.0), 1.0)

    def interp(a: float, b: float) -> float:
        return math.exp(math.log(a) + t * (math.log(b) - math.log(a)))

    out = {
        "budget": budget,
        "target": target,
        "d_model": interp(lo, hi),
        "bracket": [lo, hi],
    }
    if phi is not None:
        out["tp"] = choose_tp(phi if t > 0.5 else plo, Hardware(), Engine(), Load())
    if plo is not None and phi is not None:
        for key in ("params", "served", "nonembed", "active"):
            out[key] = interp(getattr(plo, key), getattr(phi, key))
        out["kv_bytes_per_token"] = interp(
            plo.kv_per_token() or 1, phi.kv_per_token() or 1
        )
    out["tried"] = {str(d): round(v, 1) for d, (v, _) in sorted(tried.items())}
    return out


# --- The table -------------------------------------------------------------------


#: The width at which a file's arms are listed, and exported to check the export.
LISTING = ("d_model = 1024",)
BUDGETS = ("B1a", "B1b", "B2b")


def families(folder: Path = TIMELINE) -> list[Path]:
    """The timeline's files, then the tutorial's two families."""
    files = sorted(folder.rglob("*.bux")) if folder.exists() else []
    return files + [OVN / "old_2022.bux", OVN / "new_2026.bux"]


def table(
    folder: Path = TIMELINE, slo_ms: float = 30.0, out: Path = RESULTS, workers: int = 3
) -> list[dict]:
    """Price every arm of every file in `folder` and the tutorial's families, in
    `workers` processes, write the table as Markdown and JSON to `out` as it goes,
    and return its rows. A file without a model (a module) is left out; an arm that
    cannot be priced is a row that says why."""
    import multiprocessing

    out.mkdir(parents=True, exist_ok=True)
    tasks, rows = [], []
    for file in families(folder):
        try:
            tasks += [(file, arm, slo_ms) for arm in compile_arms(file, LISTING)]
        except Exception as err:  # noqa: BLE001
            if "no definition named 'model'" not in str(err):
                rows.append(
                    {"file": _rel(file), "arm": file.stem, "error": _short(err)}
                )
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for row in pool.imap(_task, tasks):
            rows.append(row)
            print(_line(row), file=sys.stderr, flush=True)
            _write(rows, out, slo_ms)
    return rows


def _task(task: tuple[Path, str, float]) -> dict:
    """A table row, in a worker."""
    file, arm, slo_ms = task
    return _row(file, arm, compile_arms(file, LISTING)[arm], slo_ms)


def _row(file: Path, arm: str, arch: "Architecture", slo_ms: float) -> dict:
    """An arm's row: its cache and support at the listing width, its match to each
    budget, and its numbers at the width that matches B1b."""
    row: dict = {"file": _rel(file), "arm": arm}
    try:
        p = profile(arch, arm)
        row["listing"] = summary(p)
        row["export"] = exportable(arch)
        start = None
        for budget in BUDGETS:
            try:
                row[budget] = match(file, budget, arm=arm, slo_ms=slo_ms, start=start)
                start = row[budget]["d_model"]
            except Exception as err:  # noqa: BLE001
                row[budget] = {"error": _short(err)}
        if "d_model" in row["B1b"]:
            row["at_B1b"] = _at(file, arm, row["B1b"], slo_ms)
    except Exception as err:  # noqa: BLE001
        row["error"] = _short(err)
        if not isinstance(err, Unpriced):
            traceback.print_exc(file=sys.stderr)
    return row


def _at(file: Path, arm: str, matched: dict, slo_ms: float) -> dict:
    """The numbers of the priced width nearest the match."""
    d = min(matched["bracket"], key=lambda x: abs(math.log(x / matched["d_model"])))
    p = _arm_profile(file, (f"d_model = {d}",), arm)
    priced = price(p, slo_ms, (8, 32))
    return {"d_model": d, **summary(p), "tp": priced["tp"], **priced}


def exportable(arch: "Architecture") -> str:
    """Return "yes" if bux exports `arch` without weights (``bux export FILE.bux``),
    else why not."""
    import tempfile

    import torch

    from bux.export.huggingface import save

    with tempfile.TemporaryDirectory() as out:
        try:
            save(arch, None, out, dtype=torch.bfloat16)
        except (NotImplementedError, ValueError) as err:
            return _short(err)
    return "yes"


def export(
    file: str | Path,
    out: str | Path,
    settings: tuple[str, ...] = (),
    arm: str | None = None,
) -> str:
    """Write an arm's model without weights, as ``bux export FILE.bux OUT`` writes a
    file's first alternatives, for vLLM to time on random weights."""
    import torch

    from bux.export.huggingface import save

    file = Path(file)
    arms = compile_arms(file, tuple(settings))
    name = arm or next(iter(arms))
    found = [a for a in arms if a == name] or [a for a in arms if name in a]
    if len(found) != 1:
        raise ValueError(f"no single arm {name!r} in {file}: {', '.join(arms)}")
    save(arms[found[0]], None, Path(out), dtype=torch.bfloat16)
    return f"{found[0]}: exported to {out}, without weights"


def _line(row: dict) -> str:
    """A row, short, for the log."""
    if "error" in row:
        return f"{row['arm']}: {row['error']}"
    got = [
        f"{b} d={row[b]['d_model']:.0f}" if "d_model" in row[b] else f"{b} -"
        for b in BUDGETS
    ]
    return f"{row['arm']}: {', '.join(got)}; export {row['export']}; vLLM {row['listing']['vllm']}"


def _rel(file: Path) -> str:
    try:
        return str(file.relative_to(ROOT))
    except ValueError:
        return str(file)


def _short(err: Exception) -> str:
    return f"{type(err).__name__}: {str(err).splitlines()[0][:160] if str(err) else ''}"


def _si(n: float | None) -> str:
    if n is None:
        return ""
    for suffix, k in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= k:
            return f"{n / k:.3g}{suffix}"
    return f"{n:.3g}"


def _write(rows: list[dict], out: Path, slo_ms: float) -> None:
    (out / "serving-table.json").write_text(json.dumps(
        {"budgets": budgets(slo_ms), "slo_ms": slo_ms, "coefficients": asdict(FITTED), "rows": rows},
        indent=1, default=str))  # fmt: skip
    (out / "serving-table.md").write_text(markdown(rows, slo_ms))


def markdown(rows: list[dict], slo_ms: float) -> str:
    """The table as Markdown: a row per arm."""
    b = budgets(slo_ms)
    head = (
        "# Serving cost of the timeline's variants, by the model\n\n"
        "Sessions per hour per B300 (vLLM, bf16, prefix caching) under the tutorial's "
        "agentic load, by `scripts/timeline/serving.py` (not measured). Budgets, of "
        f"old_2022 at d_model 4096 (GPT-3 6.7B's serving shape): B1a = {b['B1a']:.0f}/h "
        f"at 8 at once; B1b = {b['B1b']:.0f}/h, its best at p50 TPOT <= {slo_ms:g} ms; "
        f"B2b = B1b / 4. Each 'B..' column is the d_model of equal cost and its parameters "
        "(all / active besides embeddings), and its GPUs if more than one (tensor "
        "parallelism: a prior, not fitted); 'at B1b' columns price the "
        "priced width nearest that match. KV is bytes a token adds at long context "
        "(windows and recurrent states apart), at that width.\n\n"
        "| arm | B1a: d, params | B1b: d, params | B2b: d, params | at B1b: d "
        "| KV/token | state/seq | s/h at 8 | s/h at 32 | best: c, s/h, TPOT ms "
        "| vLLM | export |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    return head + "\n".join(_cells(r) for r in rows) + "\n"


def _cells(r: dict) -> str:
    """A row of the Markdown table."""
    if "error" in r:
        return f"| {r['arm'] or r['file']} | unpriced: {r['error']} |" + " |" * 10
    cells = [r["arm"]]
    for budget in BUDGETS:
        m = r[budget]
        if "d_model" not in m:
            cells.append(m["error"][:50])
            continue
        gpus = f", {m['tp']} GPUs" if m.get("tp", 1) > 1 else ""
        cells.append(
            f"{m['d_model']:.0f}, {_si(m.get('params'))} / {_si(m.get('active'))}{gpus}"
        )
    at = r.get("at_B1b")
    if at is None:
        cells += [""] * 6
    else:
        best = at["best"]
        cells += [
            f"{at['d_model']}" + (f" (tp {at['tp']})" if at["tp"] > 1 else ""),
            f"{at['kv_bytes_per_token'] / 1024:.1f} KiB",
            _si(at["state_bytes"]) + "B" if at["state_bytes"] else "0",
            f"{at['points'][8]['sessions_per_hour']:.0f}",
            f"{at['points'][32]['sessions_per_hour']:.0f}",
            f"{best['concurrency']}, {best['sessions_per_hour']:.0f}, {best['tpot_ms']:.1f}"
            if best
            else "none",
        ]
    cells += [r["listing"]["vllm"], r["export"]]
    return "| " + " | ".join(cells) + " |"


# --- Checks against the tutorial -------------------------------------------------------


#: What the tutorial says of its models, which the compiler must give: parameters,
#: active ones besides embeddings, KiB of cache a token, the hybrid's block, and the
#: tokens of cache (the 2026 d3072's 3589 blocks of 816; d3328's at 0.94).
FACTS = {
    "old-4096": (6.86e9, 6.44e9, 512, 16, 463e3),
    "new-1536": (11.5e9, 0.83e9, 12, None, None),
    "new-2048": (27.1e9, 1.95e9, 16, None, None),
    "new-2560": (52.8e9, 3.80e9, 20, None, None),
    "new-3072": (91.1e9, 6.54e9, 24, 816, 2.93e6),
    "new-3328": (116e9, 8.31e9, 26, None, 1.3e6),
}


def check() -> tuple[str, list[str]]:
    """Compare what the compiler gives with the tutorial's facts, and the measured
    cache hits and first-token times with the model's. Return the report and what
    disagrees: parameters or bytes off by more than 1%, cache tokens by more than
    6%, cache hits by more than 0.02."""
    lines, wrong = [], []
    for name, (params, active, kib, block, tokens) in FACTS.items():
        p, e, *_ = _calibration(name)
        lay = layout(p, B300, e)
        got = (
            p.params,
            p.active,
            p.kv_per_token() / 1024,
            lay.block,
            lay.capacity / lay.token,
        )
        lines.append(
            f"{name}: {_si(got[0])} params ({_si(got[1])} active), {got[2]:.1f} KiB/token, "
            f"block {got[3]}, cache {_si(got[4])} tokens; tutorial: {_si(params)} "
            f"({_si(active)}), {kib} KiB, block {block or '-'}, {_si(tokens) or '-'}"
        )
        for what, want, have, tol in zip(
            ("params", "active", "KiB/token", "block", "tokens"),
            (params, active, kib, block, tokens),
            got,
            (0.01, 0.01, 0.01, 0.0, 0.06),
            strict=True,
        ):
            if want is not None and abs(have / want - 1) > tol + 0.005 * (
                what == "active"
            ):
                wrong.append(f"{name} {what}: {have:.4g}, the tutorial {want:.4g}")
    for (name, c), facts in CHECKS.items():
        p, e, *_ = _calibration(name)
        pt = measure(simulate(p, c, AGENTIC, e, B300), c, AGENTIC, FITTED, B300, p)
        got = ", ".join(
            f"{k} {getattr(pt, k):.3g} (measured {v:g})" for k, v in facts.items()
        )
        lines.append(f"{name} at {c}: {got}")
        if "cache_hit" in facts and abs(pt.cache_hit - facts["cache_hit"]) > 0.02:
            wrong.append(f"{name} at {c}: cache hit {pt.cache_hit:.3f}")
    return "\n".join(lines), wrong


# --- Command line ---------------------------------------------------------------------


def main() -> None:
    a = parser().parse_args()
    if a.command == "check":
        report, wrong = check()
        print(report)
        print("\n".join(["disagrees:", *wrong]) if wrong else "all agree")
        sys.exit(1 if wrong else 0)
    elif a.command == "fit":
        coef = fit()
        print(coef)
        print(_fit_report(coef, Hardware(), Load()))
        if a.loo:
            global FITTED
            FITTED = coef
            print(leave_one_out())
    elif a.command == "cost":
        print(json.dumps(cost(a.file, a.settings, a.arm, a.slo), indent=1))
    elif a.command == "match":
        budget = a.budget if a.budget.startswith("B") else float(a.budget)
        print(json.dumps(match(a.file, budget, a.settings, a.arm, a.slo), indent=1))
    elif a.command == "export":
        print(export(a.file, a.out, tuple(a.settings), a.arm))
    elif a.command == "table":
        rows = table(Path(a.dir), a.slo, Path(a.out), a.workers)
        print(markdown(rows, a.slo))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "command", choices=["check", "fit", "cost", "match", "table", "export"]
    )
    p.add_argument("file", nargs="?", help="a .bux file (cost, match)")
    p.add_argument("settings", nargs="*", help="settings, as for bux: 'd_model = 3072'")
    p.add_argument("--arm", help="the arm of a sweep, by (part of) its run's name")
    p.add_argument("--budget", default="B1b", help="B1a, B1b, B2a, B2b or sessions/h")
    p.add_argument("--slo", type=float, default=30.0, help="p50 TPOT limit, ms")
    p.add_argument("--dir", default=str(TIMELINE), help="the table's folder of files")
    p.add_argument(
        "--out", default=str(RESULTS), help="where the table, or the export, goes"
    )
    p.add_argument("--loo", action="store_true", help="fit: and leave each model out")
    p.add_argument("--workers", type=int, default=3, help="table: processes")
    return p


if __name__ == "__main__":
    main()
