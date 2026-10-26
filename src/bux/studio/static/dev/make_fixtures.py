"""Write the diagram harness's fixtures: example models as the studio's ``Model``.

A stand-in for the backend's view model: it runs ``bux json`` on each example and
gives every node an id by path, and every layer an approximate title, category,
chips and, for feed-forwards and MoEs, internals. ``written`` is faked per label, so
uses of one definition share it, as they do in a real view. Run it from the
repository root::

    uv run python src/bux/studio/static/dev/make_fixtures.py
"""

import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from itertools import count
from pathlib import Path
from typing import Any

type Json = dict[str, Any]

HERE = Path(__file__).parent / "fixtures"
EXAMPLES = Path("examples/architectures")
FIXTURES = {
    "llama3_8b": "llama3_8b.bux",
    "qwen3_next": "qwen3_next.bux",
    "hymba": "hymba.bux",
    "mhc": "mhc.bux",
    "deepseek_v3": "deepseek_v3.bux",
    "gpt2": "gpt2.bux",
    "ngpt": "ngpt.bux",
    "modded_nanogpt": "modded_nanogpt.bux",
    "primitives": "primitives.bux",
    "resformer": "resformer.bux",
    "zamba2": "zamba2.bux",
    "qwen3_8_flash_next": "qwen/qwen3.8_flash_next.bux",
    "nemotron_super": "nemotron/nemotron_3_super_120b_a12b.bux",
}

CATEGORY = {
    "embed": "io", "head": "io", "nextn": "io",
    "rmsnorm": "norm", "layernorm": "norm", "dyt": "norm",
    "gqa": "attention", "mla": "attention", "csa": "attention",
    "gla": "recurrent", "deltanet": "recurrent", "gdn": "recurrent", "kda": "recurrent",
    "gdn2": "recurrent", "lightning": "recurrent", "rwkv7": "recurrent",
    "mamba": "recurrent", "mamba2": "recurrent",
    "mlp": "ffn", "gated_mlp": "ffn", "moe": "experts",
    "linear": "op", "act": "op", "scale": "op", "dropout": "op", "merge": "op",
    "conv": "op", "short_conv": "op",
}  # fmt: skip
TITLE = {
    "embed": "Token embedding", "head": "Linear output layer",
    "nextn": "Next-token projection", "rmsnorm": "RMSNorm", "layernorm": "LayerNorm",
    "dyt": "Dynamic tanh", "gqa": "Grouped-query attention",
    "mla": "Multi-head latent attention", "csa": "Compressed sparse attention",
    "gla": "Gated linear attention", "deltanet": "DeltaNet", "gdn": "Gated DeltaNet",
    "kda": "Kimi delta attention", "gdn2": "Gated DeltaNet-2",
    "lightning": "Lightning attention", "rwkv7": "RWKV-7", "mamba": "Mamba",
    "mamba2": "Mamba-2", "moe": "Mixture of experts", "linear": "Linear",
    "scale": "Scale", "dropout": "Dropout", "merge": "Merge streams",
    "engram": "Engram memory", "conv": "Convolution", "short_conv": "Short convolution",
}  # fmt: skip
GLU = {"swish": "SwiGLU", "gelu": "GeGLU", "gelu_tanh": "GeGLU", "relu": "ReGLU"}
ACT = {"swish": "SiLU", "gelu": "GELU", "gelu_tanh": "GELU (tanh)", "relu2": "ReLU²"}


def num(value: float) -> str:
    """14336 -> '14,336'; 500000 -> '500k'; 1e7 -> '10M'."""
    if value >= 1e6:
        return f"{value / 1e6:.3g}M"
    if value >= 1e4 and value % 100 == 0:
        return f"{value / 1e3:.3g}k"
    return f"{value:,.0f}" if value == int(value) else f"{value:g}"


def title(c: Json) -> str:
    match c["type"]:
        case "gated_mlp":
            return f"Feed forward ({GLU.get(c['activation'], 'GLU')})"
        case "mlp":
            return f"Feed forward ({ACT.get(c['activation'], c['activation'])})"
        case "act":
            return ACT.get(c["fn"], c["fn"].capitalize())
        case "gqa" if c["num_kv_heads"] == c["num_heads"]:
            return "Multi-head attention"
        case "gqa" if c.get("sliding_window_size"):
            return "Sliding-window attention"
        case t:
            return TITLE.get(t, t)


def rope_chip(rope: Json | None) -> list[str]:
    if not rope:
        return []
    chip = f"RoPE {num(rope['base'])}"
    if rope.get("fraction", 1.0) < 1.0:
        chip += f", {rope['fraction']:.0%}"
    scaling = rope.get("scaling")
    return [chip, f"YaRN ×{scaling['factor']:g}"] if scaling else [chip]


def attention_chips(c: Json) -> list[str]:
    chips = [f"{c['num_heads']} heads"]
    if c.get("num_kv_heads", c["num_heads"]) != c["num_heads"]:
        chips.append(f"{c['num_kv_heads']} KV heads")
    if c.get("kv_lora_rank"):
        chips.append(f"KV rank {c['kv_lora_rank']}")
    if c.get("sliding_window_size"):
        chips.append(f"window {num(c['sliding_window_size'])}")
    if c.get("gated"):
        chips.append("gated")
    return chips + rope_chip(c.get("rope"))


def moe_chips(c: Json) -> list[str]:
    chips = [f"{c['num_experts']} experts, top {c['top_k']}"]
    if c.get("num_shared_experts"):
        chips.append(f"{c['num_shared_experts']} shared")
    return [*chips, f"{num(c['expert']['intermediate_size'])} hidden"]


def chips(c: Json) -> list[str]:
    match c["type"]:
        case "gqa" | "mla" | "csa":
            return attention_chips(c)
        case "gdn" | "kda" | "gdn2" | "deltanet":
            return [f"{c['num_heads']} heads", f"head dim {c.get('head_dim', 128)}"]
        case "mamba" | "mamba2":
            return [f"state {c['state_size']}"]
        case "moe":
            return moe_chips(c)
        case "mlp" | "gated_mlp":
            return [f"{num(c['intermediate_size'])} hidden"]
        case "embed" if c.get("max_positions"):
            return [f"{num(c['max_positions'])} positions"]
        case "head":
            return (["tied"] if c.get("tied") else []) + (
                [f"softcap {c['softcap']:g}"] if c.get("softcap") else []
            )
        case "rmsnorm" | "layernorm":
            return ["zero-centered"] if c.get("zero_centered") else []
        case "scale":
            return ["per channel"] * bool(c.get("per_channel")) + [
                f"init {c['init']:g}"
            ]
        case "linear":
            return [f"→ {num(c['features'])}"]
        case _:
            return []


class Builder:
    """Turns IR nodes into the studio's Nodes, numbering ``written`` by label."""

    def __init__(self, d_model: int):
        self.d_model = d_model
        self.spans: dict[str, list[int]] = {}
        self.fresh = count(1)

    def written(self, label: str | None) -> list[int]:
        key = label or f"#{next(self.fresh)}"
        if key not in self.spans:
            start = 10 * len(self.spans)
            self.spans[key] = [start, start + 6]
        return self.spans[key]

    def node(self, ir: Json, id: str) -> Json:
        out: Json = {
            "id": id, "kind": ir["kind"], "label": ir["label"],
            "shared": ir["share"] is not None, "span": None, "origin": None,
            "editable": True,
        }  # fmt: skip
        match ir["kind"]:
            case "layer":
                out |= self.layer(ir, id)
            case "seq" | "sum" | "prod":
                key = {"seq": "items", "sum": "terms", "prod": "factors"}[ir["kind"]]
                out[key] = [self.node(c, f"{id}.{i}") for i, c in enumerate(ir[key])]
            case "residual" | "neg" | "repeat":
                out["body"] = self.node(ir["body"], f"{id}.0")
                if "times" in ir:
                    out["times"] = ir["times"]
            case "const":
                out["value"] = ir["value"]
            case "tap" | "ref":
                out["name"] = ir["name"]
                out["pop"] = ir.get("pop", False)
        return out

    def layer(self, ir: Json, id: str) -> Json:
        c = ir["config"]
        return {
            "type": c["type"] if c["type"] != "act" else c["fn"],
            "title": title(c),
            "category": CATEGORY.get(c["type"], "other"),
            "chips": chips(c),
            "width": ir["width"],
            "ports": ir["ports"],
            "written": None if c["type"] == "merge" else self.written(ir["label"]),
            "internals": self.internals(c, f"{id}/i"),
            "inserted": c["type"] == "merge" and ir["label"] is None,
        }

    def internals(self, c: Json, id: str) -> Json | None:
        match c["type"]:
            case "gated_mlp" | "mlp":
                return internal_ffn(c, self.d_model, id)
            case "moe":
                return internal_moe(c, id)
        return None


def op(title: str, chips: list[str], category: str = "op") -> Json:
    return {"kind": "layer", "type": title.lower(), "title": title,
            "category": category, "chips": chips, "ports": [], "written": None}  # fmt: skip


def seq(*items: Json) -> Json:
    return {"kind": "seq", "items": list(items)}


def prod(*factors: Json) -> Json:
    return {"kind": "prod", "factors": list(factors)}


def numbered(node: Json, id: str) -> Json:
    """An internals tree with ids by path under `id`, read-only."""
    out = {"id": id, "label": None, "shared": False, "span": None, "origin": None,
           "editable": False, **node}  # fmt: skip
    for key in ("items", "terms", "factors"):
        if key in node:
            out[key] = [numbered(c, f"{id}.{i}") for i, c in enumerate(node[key])]
    return out


def internal_ffn(c: Json, d_model: int, id: str) -> Json:
    hidden, act = num(c["intermediate_size"]), ACT.get(c["activation"], c["activation"])
    up = op("Linear", [f"{num(d_model)} → {hidden}"])
    down = op("Linear", [f"{hidden} → {num(d_model)}"])
    if c["type"] == "mlp":
        return numbered(seq(up, op(act, []), down), id)
    return numbered(seq(prod(seq(up, op(act, [])), up), down), id)


def internal_moe(c: Json, id: str) -> Json:
    expert = c["expert"]
    kind = f"{GLU.get(expert['activation'], 'GLU')}, {num(expert['intermediate_size'])}"
    router = op("Router", [f"{c.get('router', 'softmax')}, top {c['top_k']}"])
    routed = op("Routed experts", [f"{c['num_experts']} × {kind}"], "ffn")
    mixed = prod(routed, router)
    if not c.get("num_shared_experts"):
        return numbered(mixed, id)
    shared = op("Shared expert", [kind], "ffn")
    return numbered({"kind": "sum", "terms": [mixed, shared]}, id)


def walk(node: Json, times: int = 1) -> Iterator[tuple[Json, int]]:
    yield node, times
    for key in ("items", "terms", "factors"):
        for child in node.get(key, []):
            yield from walk(child, times)
    if "body" in node:
        yield from walk(node["body"], times * node.get("times", 1))


def residual_text(residual: Json | None) -> str | None:
    if residual is None:
        return None
    expansion = residual.get("expansion")
    return (
        f"{residual['type']}(expansion={expansion})" if expansion else residual["type"]
    )


def model(name: str, arch: Json) -> Json:
    graph = Builder(arch["d_model"]).node(arch["graph"], "n")
    layers = Counter[str]()
    for node, times in walk(graph):
        if node["kind"] == "layer":
            layers[node["type"]] += times
    return {
        "name": name,
        "facts": {
            "d_model": arch["d_model"],
            "vocab_size": arch["vocab_size"],
            "layers": layers.total(),
            "residual": residual_text(arch["residual"]),
            "init": None,
            "by_type": layers.most_common(),
        },
        "graph": graph,
    }


def compile_json(path: Path) -> Json:
    command = [sys.executable, "-m", "bux", "json", str(path)]
    return json.loads(subprocess.run(command, capture_output=True, check=True).stdout)


def main() -> None:
    HERE.mkdir(exist_ok=True)
    for name, file in FIXTURES.items():
        data = model(
            re.sub(r"\.bux$", "", Path(file).name), compile_json(EXAMPLES / file)
        )
        (HERE / f"model-{name}.json").write_text(json.dumps(data, indent=1) + "\n")
        print(f"model-{name}.json")


if __name__ == "__main__":
    main()
