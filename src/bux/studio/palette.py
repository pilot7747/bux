"""How the studio names and shows each layer: its title, category and chips, and the
text it inserts for it.

Each is a registry keyed by the layer's type, so a new layer shows up with defaults
(a title from its config's docstring, a category from its config's base class) and
can register better ones in a few lines::

    from bux.studio import palette

    palette.names["mixer"] = "Token mixer"           # its title in the palette
    palette.categories["mixer"] = "recurrent"

    @palette.chips.register("mixer")
    def _(c: MixerConfig, ctx: palette.Context) -> list[str]:
        \"\"\"Its heads.\"\"\"
        return [palette.many(c.num_heads, "head")]

    @palette.templates.register("mixer")
    def _(tag: str, ctx: palette.Context) -> str:
        \"\"\"Heads of 128 dims.\"\"\"
        return f"mixer(num_heads={max(1, ctx.d_model // 128)})"

`titles` takes a config and says what that one is ("Multi-query attention" for a
``gqa`` of one key head); `names` is the title of a type alone. A chip is short: a
number and its unit, ``"8 KV heads"``, ``"RoPE 500k"``.
"""

import inspect
import re
import typing
from dataclasses import dataclass
from typing import Any, Literal

from bux import config as cfg
from bux.lang import ir
from bux.lang.compiler import ACTIVATIONS, LAYERS, TYPES
from bux.lang.reference import activation_doc
from bux.train.registry import Registry

#: What a layer is, which picks its colors: the token embedding and heads (io), norms,
#: softmax attention, recurrent mixers (linear attention, state spaces, convolutions),
#: dense feed-forwards (ffn), mixtures of experts, small ops, and the rest.
Category = Literal[
    "io", "norm", "attention", "recurrent", "ffn", "experts", "op", "other"
]


@dataclass(frozen=True)
class Context:
    """What a layer's chips and templates may read of the model around it.

    Attributes:
        d_model: The width of the residual stream.
        vocab_size: The vocabulary; None for a model without one.
        width: What the layer reads, as the compiler inferred it; None for a
            template, which no layer reads yet.
    """

    d_model: int
    vocab_size: int | None = None
    width: ir.Width | None = None


#: Each layer's title by type, as Raschka's figures name them; a type left out is
#: named by the first clause of its config's docstring. Activations too, by name.
names: dict[str, str] = {
    "embed": "Token embedding",
    "head": "Linear output layer",
    "nextn": "Multi-token prediction",
    "engram": "Engram memory",
    "rmsnorm": "RMSNorm",
    "layernorm": "LayerNorm",
    "dyt": "Dynamic Tanh",
    "gqa": "Grouped-query attention",
    "mla": "Multi-head latent attention",
    "csa": "Compressed sparse attention",
    "gla": "Gated linear attention",
    "deltanet": "DeltaNet",
    "gdn": "Gated DeltaNet",
    "kda": "Kimi Delta Attention",
    "gdn2": "Gated DeltaNet-2",
    "lightning": "Lightning attention",
    "rwkv7": "RWKV-7 time mixing",
    "mamba": "Mamba",
    "mamba2": "Mamba-2",
    "conv": "Causal convolution",
    "short_conv": "Short convolution",
    "mlp": "Feed forward",
    "gated_mlp": "Feed forward (SwiGLU)",
    "moe": "Mixture of experts",
    "linear": "Linear layer",
    "act": "Activation",
    "scale": "Scale",
    "dropout": "Dropout",
    "merge": "Merge streams",
    "relu": "ReLU",
    "relu2": "ReLU²",
    "gelu": "GELU",
    "gelu_tanh": "GELU (tanh)",
    "swish": "SiLU",
    "sigmoid": "Sigmoid",
    "tanh": "Tanh",
    "softplus": "Softplus",
    "situ": "SiTU",
    "quick_gelu": "Quick GELU",
}

#: Categories by type, where the config's base class does not say (`category`).
categories: dict[str, Category] = {}

# A type's category by its config's class, the first base that matches.
_BASES: tuple[tuple[type[cfg.Config], Category], ...] = (
    (cfg.EmbeddingConfig, "io"),
    (cfg.LMHeadConfig, "io"),
    (cfg.NextNConfig, "io"),
    (cfg.NormalizationConfig, "norm"),
    (cfg.DyTConfig, "norm"),
    (cfg.QuadraticAttentionConfig, "attention"),
    (cfg.AttentionConfig, "recurrent"),
    (cfg.MLPConfig, "ffn"),
    (cfg.MoEConfig, "experts"),
    (cfg.LinearConfig, "op"),
    (cfg.ActivationConfig, "op"),
    (cfg.ScaleConfig, "op"),
    (cfg.DropoutConfig, "op"),
    (cfg.MergeConfig, "op"),
)

#: Titles that depend on the config, by type: a function of the config.
titles = Registry[str]("title", "bux.studio.palette.titles")
#: The chips of a layer, by type: a function of the config and its `Context`.
chips = Registry[list[str]]("chips", "bux.studio.palette.chips")
#: What the palette inserts for a type, by type: a function of the type and a
#: `Context`.
templates = Registry[str]("template", "bux.studio.palette.templates")


# --- Looking a layer up ------------------------------------------------------------


def name(tag: str) -> str:
    """Return the title of the type or activation `tag` alone."""
    if tag in names:
        return names[tag]
    return _clause(TYPES[tag]) if tag in TYPES else tag


def title(config: cfg.LayerConfig) -> str:
    """Return what the studio calls this layer: its type's title, or what its
    registered `titles` entry makes of its settings."""
    tag = config.type
    if tag in titles.builders:
        return titles.build(config)
    return name(tag)


def category(tag: str) -> Category:
    """Return the category of the layer type `tag`: registered, else by its config's
    base class, else ``"other"``."""
    if tag in categories:
        return categories[tag]
    if tag in ACTIVATIONS:
        return "op"
    cls = LAYERS.get(tag)
    for base, found in _BASES:
        if cls is not None and issubclass(cls, base):
            return found
    return "other"


def chips_of(config: cfg.LayerConfig, ctx: Context) -> list[str]:
    """Return the layer's chips, by its registered `chips` entry; none without one."""
    return chips.build(config, ctx) if config.type in chips.builders else []


def template(tag: str, ctx: Context) -> str:
    """Return the text that inserts a layer of type `tag`: its registered
    `templates` entry, else the bare name."""
    return templates.find(tag)(tag, ctx) if tag in templates.builders else tag


def palette(ctx: Context) -> list[dict[str, Any]]:
    """Return the palette, what ``/api/session`` sends: every layer type, by category,
    then every activation, each with its kind, category, title, doc and insert
    template.

    Args:
        ctx: The model the templates are for: their sizes follow its ``d_model``.
    """
    order = typing.get_args(Category)
    tags = sorted(LAYERS, key=lambda tag: (order.index(category(tag)), tag))
    layers = [_item(tag, "layer", _doc(LAYERS[tag]), ctx) for tag in tags]
    acts = [
        _item(fn, "activation", activation_doc(fn).split("\n\n")[1], ctx)
        for fn in sorted(ACTIVATIONS)
    ]
    return layers + acts


def _item(tag: str, kind: str, doc: str, ctx: Context) -> dict[str, Any]:
    return {
        "name": tag,
        "kind": kind,
        "category": category(tag),
        "title": name(tag),
        "doc": doc,
        "template": template(tag, ctx),
    }


def _doc(cls: type[cfg.Config]) -> str:
    """Return the first paragraph of a config's docstring, on one line."""
    doc = inspect.cleandoc(cls.__doc__ or "")
    return " ".join(doc.split("\n\n")[0].split())


def _clause(cls: type[cfg.Config]) -> str:
    """Return the first clause of a config's docstring, without its article."""
    first = re.split(r"[;:,.(]", _doc(cls), maxsplit=1)[0].strip()
    first = re.sub(r"^(An?|The) ", "", first)
    return first[:1].upper() + first[1:] or cls.__name__


# --- Numbers, short ------------------------------------------------------------------


def si(n: float) -> str:
    """Return `n` in three digits and a suffix, as ``bux check`` writes tokens:
    ``25.6k``, ``3.07B``."""
    for suffix, scale in (("", 1), ("k", 1e3), ("M", 1e6), ("B", 1e9)):
        if "e" not in (shown := f"{n / scale:.3g}"):
            return shown + suffix
    return f"{n / 1e12:.3g}T"


def count(n: int) -> str:
    """Return a count with thousands separated: ``14,336``."""
    return f"{n:,}"


def many(n: int, unit: str) -> str:
    """Return `n` of `unit`, plural but for one: ``8 KV heads``, ``1 KV head``."""
    return f"{n:,} {unit}{'' if n == 1 else 's'}"


def _rope(rope: cfg.RoPEConfig | None) -> list[str]:
    if rope is None:
        return ["NoPE"]
    text = f"RoPE {si(rope.base)}"
    if rope.fraction < 1:
        text += f" · {rope.fraction:.0%}"
    if rope.scaling is not None:
        text += f" · {rope.scaling.type} ×{rope.scaling.factor:g}"
    return [text]


def _heads(width: ir.Width | None, heads: int, head_dim: int | None) -> int | None:
    if head_dim is not None:
        return head_dim
    return width // heads if isinstance(width, int) else None


def _first(found: list[str], most: int = 4) -> list[str]:
    return [c for c in found if c][:most]


# --- Titles --------------------------------------------------------------------------

_GLU = {"swish": "SwiGLU", "gelu": "GeGLU", "gelu_tanh": "GeGLU", "relu": "ReGLU"}


def glu(c: cfg.MLPConfig) -> str:
    """Return what a gated MLP is called by its gate's activation: SwiGLU, GeGLU,
    ReGLU, or GLU."""
    return _GLU.get(c.activation, "GLU")


@titles.register("gqa")
def _(c: cfg.GQAConfig) -> str:
    """Multi-head attention with a key head a query head, multi-query with one, else
    grouped-query."""
    if c.num_kv_heads == c.num_heads:
        return "Multi-head attention"
    return "Multi-query attention" if c.num_kv_heads == 1 else names["gqa"]


@titles.register("gated_mlp")
def _(c: cfg.MLPConfig) -> str:
    """Feed forward, by its gate: SwiGLU, GeGLU, ReGLU, or GLU."""
    return f"Feed forward ({glu(c)})"


@titles.register("act")
def _(c: cfg.ActivationConfig) -> str:
    """The activation's own name: SiLU, GELU."""
    return name(c.fn)


@titles.register("head")
def _(c: cfg.LMHeadConfig) -> str:
    """The output layer, tied to the embedding or not."""
    return "Linear output layer (tied)" if c.tied else names["head"]


# --- Chips ---------------------------------------------------------------------------


@chips.register("embed")
def _(c: cfg.EmbeddingConfig, ctx: Context) -> list[str]:
    """Its vocabulary, and learned positions or a scale where it has them."""
    vocab = f"vocab {si(ctx.vocab_size)}" if ctx.vocab_size else ""
    positions = f"{si(c.max_positions)} positions" if c.max_positions else ""
    return _first([vocab, positions, "× √d" if c.scale_by_sqrt_dim else ""])


@chips.register("head")
def _(c: cfg.LMHeadConfig, ctx: Context) -> list[str]:
    """Its vocabulary, tied or not, and its softcap."""
    vocab = f"vocab {si(ctx.vocab_size)}" if ctx.vocab_size else ""
    cap = f"softcap {c.softcap:g}" if c.softcap else ""
    return _first([vocab, "tied" if c.tied else "", cap])


@chips.register("rmsnorm", "layernorm")
def _(c: cfg.NormalizationConfig, ctx: Context) -> list[str]:
    """Only what is unusual: a zero-centered gain, 1 + w, none, or a bias."""
    return _first(
        [
            "zero-centered" if c.zero_centered else "",
            "no gain" if not c.affine else "",
            "bias" if c.bias else "",
        ]
    )


@chips.register("gqa")
def _(c: cfg.GQAConfig, ctx: Context) -> list[str]:
    """Heads, key and value heads, head dims, and positions."""
    dim = _heads(ctx.width, c.num_heads, c.head_dim)
    kv = "" if c.num_kv_heads == c.num_heads else many(c.num_kv_heads, "KV head")
    window = f"window {si(c.sliding_window_size)}" if c.sliding_window_size else ""
    return _first(
        [
            many(c.num_heads, "head"),
            kv,
            f"head dim {dim}" if dim else "",
            window,
            *_rope(c.rope),
        ]
    )


@chips.register("mla")
def _(c: cfg.MLAConfig, ctx: Context) -> list[str]:
    """Heads, the latents' ranks, and positions."""
    q = f"Q rank {count(c.q_lora_rank)}" if c.q_lora_rank else ""
    return _first(
        [
            many(c.num_heads, "head"),
            f"KV rank {count(c.kv_lora_rank)}",
            q,
            *_rope(c.rope),
        ]
    )


@chips.register("csa")
def _(c: cfg.CSAConfig, ctx: Context) -> list[str]:
    """Heads, head dims, and how many tokens a compressed latent pools."""
    ratio = f"compress ×{c.compress_ratio}" if c.compress_ratio else "window only"
    return _first([many(c.num_heads, "head"), f"head dim {c.head_dim}", ratio])


@chips.register("gla", "deltanet", "gdn", "kda", "gdn2", "lightning", "rwkv7")
def _(c: cfg.LinearAttentionConfig, ctx: Context) -> list[str]:
    """Query and key heads, value heads where there are more, and head dims."""
    v = c.num_v_heads
    return _first(
        [
            many(c.num_heads, "head"),
            many(v, "V head") if v and v != c.num_heads else "",
            f"head dim {c.head_dim}",
        ]
    )


@chips.register("mamba", "mamba2")
def _(c: cfg.StateSpaceConfig, ctx: Context) -> list[str]:
    """The state's dims, the inner channels' expansion, and Mamba-2's heads."""
    found = [f"state {c.state_size}", f"expand ×{c.expand}"]
    if isinstance(c, cfg.Mamba2Config):
        found.append(f"head dim {c.head_dim}")
    return found


@chips.register("conv", "short_conv")
def _(c: cfg.ConvConfig | cfg.ShortConvConfig, ctx: Context) -> list[str]:
    """Its taps."""
    taps = c.size if isinstance(c, cfg.ConvConfig) else c.conv_size
    return [f"{taps} taps"]


@chips.register("mlp", "gated_mlp")
def _(c: cfg.MLPConfig, ctx: Context) -> list[str]:
    """Its hidden units, and a plain MLP's activation."""
    act = name(c.activation) if c.type == "mlp" else ""
    return _first([f"{count(c.intermediate_size)} hidden", act])


@chips.register("moe")
def _(c: cfg.MoEConfig, ctx: Context) -> list[str]:
    """Experts, top-k and shared experts in one chip, then each expert's hidden
    units and the router's scores."""
    routed = f"{c.num_experts} experts · top {c.top_k}"
    if c.num_shared_experts:
        routed += f" · {c.num_shared_experts} shared"
    hidden = f"{count(c.expert.intermediate_size)} hidden"
    return [routed, hidden, c.router]


@chips.register("linear")
def _(c: cfg.LinearConfig, ctx: Context) -> list[str]:
    """Its output channels."""
    return [f"{count(c.features)} out"]


@chips.register("scale")
def _(c: cfg.ScaleConfig, ctx: Context) -> list[str]:
    """Where it starts, and whether each channel has its own."""
    return _first([f"init {c.init:g}", "per channel" if c.per_channel else ""])


@chips.register("dropout")
def _(c: cfg.DropoutConfig, ctx: Context) -> list[str]:
    """Its rate."""
    return [f"rate {c.rate:g}"]


@chips.register("engram")
def _(c: cfg.EngramConfig, ctx: Context) -> list[str]:
    """Hash heads and the n-grams they hash."""
    return [many(c.num_heads, "head"), f"2–{c.max_ngram_size}-grams"]


# --- Templates -----------------------------------------------------------------------


def _heads_of(ctx: Context) -> tuple[int, int]:
    """Return heads of 128 dims or fewer for the width, and their dims."""
    dim = min(128, ctx.d_model)
    return max(1, ctx.d_model // dim), dim


@templates.register("gqa")
def _(tag: str, ctx: Context) -> str:
    """Heads of 128 dims, a key and value head for each four of them, and RoPE."""
    heads, _ = _heads_of(ctx)
    kv = heads // 4 if heads % 4 == 0 else heads
    return f"gqa(num_heads={heads}, num_kv_heads={kv}, rope=rope())"


@templates.register("mla", "csa")
def _(tag: str, ctx: Context) -> str:
    """Heads of 128 dims."""
    return f"{tag}(num_heads={_heads_of(ctx)[0]})"


@templates.register("gla", "deltanet", "gdn", "kda", "gdn2", "lightning", "rwkv7")
def _(tag: str, ctx: Context) -> str:
    """Heads of 128 dims."""
    heads, dim = _heads_of(ctx)
    return f"{tag}(num_heads={heads}, head_dim={dim})"


@templates.register("mlp")
def _(tag: str, ctx: Context) -> str:
    """Four times the width of hidden units, GELU'd."""
    return f"mlp(intermediate_size={4 * ctx.d_model}, activation=gelu)"


@templates.register("gated_mlp")
def _(tag: str, ctx: Context) -> str:
    """SwiGLU with 8/3 of the width of hidden units, to a multiple of 64."""
    hidden = -(-8 * ctx.d_model // 3 // 64) * 64
    return f"gated_mlp(intermediate_size={hidden}, activation=swish)"


@templates.register("moe")
def _(tag: str, ctx: Context) -> str:
    """Eight SwiGLU experts, two a token."""
    return (
        "moe(num_experts=8, top_k=2, "
        f"expert=gated_mlp(intermediate_size={2 * ctx.d_model}, activation=swish))"
    )


@templates.register("linear")
def _(tag: str, ctx: Context) -> str:
    """The width of the stream."""
    return f"linear(features={ctx.d_model})"


@templates.register("act")
def _(tag: str, ctx: Context) -> str:
    """SiLU."""
    return "act(fn=swish)"


@templates.register("dropout")
def _(tag: str, ctx: Context) -> str:
    """A tenth."""
    return "dropout(rate=0.1)"
