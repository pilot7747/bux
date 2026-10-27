"""What is inside a layer: the small diagrams the studio opens beside its box, as
Raschka zooms into the feed-forward module of his figures.

Each is a tree of the view's nodes (`bux.studio.view`), read-only, whose leaves are
the layer's parts: its projections, activations, norms and products. A builder is
registered by the layer's type and returns the tree without ids, from helpers that
make each kind of node; `numbered` gives it ids under the layer's (``n.1:i.0``)::

    from bux.studio import internals
    from bux.studio.internals import op, seq

    @internals.internals.register("mixer")
    def _(c: MixerConfig, ctx: Context) -> Part:
        \"\"\"Its projection, then its scan.\"\"\"
        return seq(op("Linear ↑", [f"{c.num_heads} heads"]), op("Scan", [], "recurrent"))

A ``prod`` of columns is a gate (``⊗``), and a ``sum`` adds them (``⊕``).
"""

from typing import Any

from bux import config as cfg
from bux.studio.palette import Category, Context, count, glu, many, name, si
from bux.train.registry import Registry

#: A node of an inside diagram before `numbered` numbers it: the view's `Node`
#: without ids.
type Part = dict[str, Any]

#: Inside diagrams by layer type: a function of the config and its `Context`.
internals = Registry[Part]("inside diagram", "bux.studio.internals.internals")


def numbered(part: Part, id: str) -> dict[str, Any]:
    """Return `part` with the fields every node has, read-only, and ids by path
    under `id`."""
    node = {
        "id": id,
        "label": None,
        "shared": False,
        "span": None,
        "origin": None,
        "editable": False,
        **part,
    }
    for key in ("items", "terms", "factors"):
        if key in part:
            node[key] = [numbered(c, f"{id}.{k}") for k, c in enumerate(part[key])]
    if (inner := part.get("internals")) is not None:
        node["internals"] = numbered(inner, f"{id}:i")
    return node


# --- Parts ---------------------------------------------------------------------------


def op(
    title: str,
    chips: list[str] | None = None,
    category: Category = "op",
    type: str | None = None,
    internals: Part | None = None,
) -> Part:
    """Return a box of an inside diagram: a part of the layer, with its chips."""
    return {
        "kind": "layer",
        "type": type or title.split(" ")[0].lower(),
        "title": title,
        "category": category,
        "chips": [c for c in chips or [] if c],
        "width": None,
        "ports": [],
        "written": None,
        "internals": internals,
    }


def seq(*items: Part | None) -> Part:
    """Return the parts one after another, the Nones left out and sequences spliced
    in; one part alone."""
    kept = [
        part
        for item in items
        if item is not None
        for part in (item["items"] if item["kind"] == "seq" else [item])
    ]
    return kept[0] if len(kept) == 1 else {"kind": "seq", "items": kept}


def prod(*factors: Part) -> Part:
    """Return columns multiplied elementwise, ``⊗``: a gate."""
    return {"kind": "prod", "factors": list(factors)}


def add(*terms: Part) -> Part:
    """Return columns summed, ``⊕``."""
    return {"kind": "sum", "terms": list(terms)}


def _gated(body: Part, gate: Part | None) -> Part:
    return body if gate is None else prod(body, gate)


def _width(ctx: Context) -> int:
    return ctx.width if isinstance(ctx.width, int) else ctx.d_model


def _act(fn: str) -> Part:
    return op(name(fn), type="act")


def _norm(title: str, norm: cfg.NormalizationConfig | None = None) -> Part:
    return op(title, [name(norm.type)] if norm is not None else [], "norm", "norm")


# --- Feed-forwards -------------------------------------------------------------------


@internals.register("mlp")
def _(c: cfg.MLPConfig, ctx: Context) -> Part:
    """Up to the hidden units, the activation, and back down."""
    d, h = count(_width(ctx)), count(c.intermediate_size)
    return seq(
        op("Linear ↑", [f"{d} → {h}"], type="linear"),
        _act(c.activation),
        op("Linear ↓", [f"{h} → {d}"], type="linear"),
    )


@internals.register("gated_mlp")
def _(c: cfg.MLPConfig, ctx: Context) -> Part:
    """The activated gate times the up projection, then down: SwiGLU's three
    matrices."""
    d, h = count(_width(ctx)), count(c.intermediate_size)
    limit = f"clamp {c.limit:g}" if c.limit is not None else ""
    gate = seq(op("Linear ↑ (gate)", [f"{d} → {h}"], type="linear"), _act(c.activation))
    up = op("Linear ↑", [f"{d} → {h}", limit], type="linear")
    return seq(prod(gate, up), op("Linear ↓", [f"{h} → {d}"], type="linear"))


@internals.register("moe")
def _(c: cfg.MoEConfig, ctx: Context) -> Part:
    """The router's top-k weights times the routed experts' outputs, summed, beside
    the shared experts."""
    expert = internals.build(c.expert, ctx)
    kind = name(c.expert.activation) if c.expert.type == "mlp" else glu(c.expert)
    routed = op(
        "Routed experts",
        [f"×{c.num_experts}", f"{kind} · {count(c.expert.intermediate_size)} hidden"],
        "ffn",
        "experts",
        expert,
    )
    groups = f"{c.top_k_groups} of {c.num_groups} groups" if c.num_groups > 1 else ""
    router = seq(
        op("Router", [f"{_width(ctx)} → {c.num_experts}", c.router], "op", "router"),
        op("Top-k", [f"top {c.top_k}", groups], "op", "topk"),
    )
    scale = c.routed_scaling_factor
    weighted = seq(
        prod(routed, router), op("Weighted sum", [f"× {scale:g}" if scale != 1 else ""])
    )
    if not c.num_shared_experts:
        return weighted
    each = (c.shared_expert or c.expert).intermediate_size
    shared = op(
        "Shared expert",
        [f"{c.num_shared_experts} × {count(each)} hidden"],
        "ffn",
        "shared",
        internals.build(c.shared_expert or c.expert, ctx),
    )
    return add(weighted, shared)


# --- Attention -----------------------------------------------------------------------


def _sdpa(c: cfg.QuadraticAttentionConfig) -> Part:
    window = f"window {si(c.sliding_window_size)}" if c.sliding_window_size else ""
    cap = f"softcap {c.softcap:g}" if c.softcap else ""
    top = f"top {si(c.indexer.top_k)} keys" if c.indexer is not None else ""
    return op(
        "Scaled dot-product attention",
        ["causal", window, cap, top, "sinks" if c.sinks else ""],
        "attention",
        "sdpa",
    )


def _rope(rope: cfg.RoPEConfig | None, dims: str = "") -> Part | None:
    if rope is None:
        return None
    scaling = f"{rope.scaling.type} ×{rope.scaling.factor:g}" if rope.scaling else ""
    return op("RoPE", [f"base {si(rope.base)}", dims, scaling], type="rope")


def _gate(c: cfg.QuadraticAttentionConfig) -> Part | None:
    if not c.gated:
        return None
    per = "per head" if c.gate_per_head else ""
    return seq(op("Linear ↑ (gate)", [per], type="linear"), _act(c.gate_activation))


@internals.register("gqa")
def _(c: cfg.GQAConfig, ctx: Context) -> Part:
    """The q, k and v projections, the queries' and keys' norm and RoPE, attention,
    its gate, and the output projection."""
    dim = c.head_dim or _width(ctx) // c.num_heads
    qkv = op(
        "Q, K, V projections",
        [
            "Q " + many(c.num_heads, "head"),
            "K, V " + many(c.num_kv_heads, "head"),
            f"head dim {dim}",
        ],
        type="linear",
    )
    norm = _norm("QK-norm", c.qk_norm) if c.qk_norm is not None else None
    body = seq(qkv, norm, _rope(c.rope), _sdpa(c))
    out = op("Linear output", [f"{count(c.num_heads * dim)} → {count(_width(ctx))}"])
    return seq(_gated(body, _gate(c)), out)


@internals.register("mla")
def _(c: cfg.MLAConfig, ctx: Context) -> Part:
    """The low-rank latents of the queries and of keys and values, their norms, the
    heads' up projections, RoPE on its own dims, attention and the output."""
    q = f"Q rank {count(c.q_lora_rank)}" if c.q_lora_rank else "Q full rank"
    latents = op(
        "Latent projections",
        [q, f"KV rank {count(c.kv_lora_rank)}", f"RoPE key {c.qk_rope_head_dim}"],
        type="linear",
    )
    up = op(
        "Up projections",
        [
            many(c.num_heads, "head"),
            f"QK {c.qk_nope_head_dim} + {c.qk_rope_head_dim}",
            f"V {c.v_head_dim}",
        ],
        type="linear",
    )
    rope = _rope(c.rope, f"{c.qk_rope_head_dim} dims")
    out = op("Linear output", [f"{count(c.num_heads * c.v_head_dim)} → {_width(ctx)}"])
    body = seq(latents, _norm("Latent norms"), up, rope, _sdpa(c))
    return seq(_gated(body, _gate(c)), out)


# --- Recurrent mixers ----------------------------------------------------------------

_RULES = {
    "deltanet": ("Delta rule", ""),
    "gdn": ("Gated delta rule", "decay per head"),
    "kda": ("Gated delta rule", "decay per channel"),
    "gdn2": ("Erase-write delta rule", "decay per channel"),
}


@internals.register(*_RULES)
def _(
    c: cfg.DeltaNetConfig
    | cfg.GatedDeltaNetConfig
    | cfg.KDAConfig
    | cfg.GatedDeltaNet2Config,
    ctx: Context,
) -> Part:
    """The q, k and v projections, their short conv, the delta rule's state, the
    output's norm and gate, and the output projection."""
    v = c.num_v_heads or c.num_heads
    vdim = c.v_head_dim or c.head_dim
    qkv = op(
        "Q, K, V projections",
        [
            "Q, K " + many(c.num_heads, "head"),
            "V " + many(v, "head"),
            f"head dim {c.head_dim}",
        ],
        type="linear",
    )
    conv = op("Short conv", [f"{c.conv_size} taps", "SiLU"]) if c.conv_size else None
    rule, decay = _RULES[c.type]
    state = op(rule, [f"state {c.head_dim} × {vdim}", decay], "recurrent", "rule")
    gate = None
    if c.output_gate is not None:
        gate = seq(op("Linear ↑ (gate)", type="linear"), _act(c.output_gate))
    body = seq(qkv, conv, op("L2 norm", ["q, k"], "norm"), state, _norm("Output norm"))
    out = op("Linear output", [f"{count(v * vdim)} → {count(_width(ctx))}"])
    return seq(_gated(body, gate), out)


@internals.register("mamba2")
def _(c: cfg.Mamba2Config, ctx: Context) -> Part:
    """The input projection, its short conv, the SSD scan, the SiLU(z) gate, the
    grouped norm and the output projection."""
    inner = c.expand * _width(ctx)
    heads = c.num_heads or inner // c.head_dim
    proj = op("In projection", ["x, B, C, dt", f"expand ×{c.expand}"], type="linear")
    conv = op("Short conv", [f"{c.conv_size} taps", "SiLU"])
    scan = op(
        "SSD scan",
        [many(heads, "head"), f"state {c.state_size}", many(c.num_groups, "group")],
        "recurrent",
        "scan",
    )
    gate = seq(op("Linear ↑ (z)", type="linear"), _act("swish"))
    out = op("Linear output", [f"{count(inner)} → {count(_width(ctx))}"])
    return seq(prod(seq(proj, conv, scan), gate), _norm("Group RMSNorm"), out)
