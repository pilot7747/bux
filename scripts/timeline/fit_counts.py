"""What a family's model holds and costs at a width, from bux's compiler, cached in a
JSON file by file, arm and width.

Its parameters (``total``), those besides the embeddings, the head, Engram's tables
and what comes after the first head, as multi-token prediction (``nonembed``), and
those a token goes through, an MoE's routed experts by top_k / num_experts
(``active``); and its model FLOPs per training token, as ``bux plan`` counts them.
Nothing is materialized: the model is built on shapes alone.
"""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from fit_points import classify, parse_name

REPO = Path(__file__).resolve().parents[2]
FOLDER = REPO / "examples" / "experiments" / "timeline"
EMBEDDING = {"embed", "head", "engram"}
LINEAR = {
    "gla",
    "deltanet",
    "gdn",
    "kda",
    "gdn2",
    "lightning",
    "rwkv7",
    "mamba",
    "mamba2",
}
ARM = re.compile(r"^\s*(?:[\w.]+\s*=\s*)?\|?\s*(\w+)\s*:")


class Counts:
    """Counts of the families' models, compiled on demand and cached in `path`."""

    def __init__(self, path: Path, folders: tuple[Path, ...] = (FOLDER,)) -> None:
        self.path = path
        self.folders = folders
        self.table: dict = json.loads(path.read_text()) if path.exists() else {}
        self._runs: dict[tuple, list] = {}

    def at(self, stem: str, arms: tuple[str, ...], width: int) -> dict:
        """The counts of `stem`'s arm at the rung labelled ``d<width>``."""
        key = f"{stem}|{'+'.join(arms)}|{width}"
        if key not in self.table:
            run = self.rung(stem, arms, width)
            self._store(key, lambda: count(run.arch) if run else {"error": "no run"})
        return self.table[key]

    def flops(self, stem: str, arms: tuple[str, ...], d_model: float) -> dict:
        """The FLOPs per training token and active parameters at the valid width
        nearest `d_model`, the multiple of 128 the family's rules accept."""
        for width in _near(d_model):
            key = f"{stem}|{'+'.join(arms)}|{width}|flops"
            if key not in self.table:
                self._store(key, lambda w=width: _flops(self.sized(stem, arms, w)))
            if "error" not in self.table[key]:
                return self.table[key]
        return {"error": f"no width near {d_model:.0f} compiles"}

    def rung_flops(self, stem: str, arms: tuple[str, ...], width: int) -> dict:
        """The FLOPs per training token of `stem`'s arm at its rung ``d<width>``."""
        key = f"{stem}|{'+'.join(arms)}|{width}|rung_flops"
        if key not in self.table:
            self._store(
                key, lambda: {"flops": _flops(self.rung(stem, arms, width))["flops"]}
            )
        return self.table[key]

    def rung(self, stem: str, arms: tuple[str, ...], width: int) -> Any | None:
        """The compiled run of `stem`'s arm at its rung ``d<width>``."""
        for run in self.runs(stem, ()):
            w, _, _, a = classify(parse_name(run.name)[1])
            if w == width and a == arms:
                return run
        return self.sized(stem, arms, width)

    def sized(self, stem: str, arms: tuple[str, ...], d_model: int) -> Any | None:
        """A compiled run of `stem`'s arm with ``d_model = <d_model>`` set."""
        for run in self.runs(stem, (f"d_model = {d_model}",)):
            if classify(parse_name(run.name)[1])[3] == arms:
                return run
        return None

    def runs(self, stem: str, overrides: tuple[str, ...]) -> list:
        if (stem, overrides) not in self._runs:
            from bux.lang import compile_runs

            file = find(stem, self.folders)
            self._runs[(stem, overrides)] = (
                compile_runs(
                    file.read_text(), filename=str(file), overrides=list(overrides)
                )
                if file
                else []
            )
        return self._runs[(stem, overrides)]

    def _store(self, key: str, make: Any) -> None:
        try:
            self.table[key] = make()
        except Exception as err:  # noqa: BLE001  (a width the rules reject, a missing layer)
            self.table[key] = {"error": f"{type(err).__name__}: {str(err)[:200]}"}
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.table, indent=1, sort_keys=True))


def find(stem: str, folders: tuple[Path, ...]) -> Path | None:
    """The .bux file named `stem` under `folders`."""
    for folder in folders:
        if hits := sorted(folder.rglob(f"{stem}.bux")):
            return hits[0]
    return None


def _near(d_model: float) -> list[int]:
    base = max(128, round(d_model / 128) * 128)
    return [base + s for s in (0, 128, -128, 256, -256) if base + s >= 128]


def count(arch: Any) -> dict:
    """The parameters of `arch`'s model, besides the embeddings and a token's idle
    experts, its width, its kind and a hash of its architecture."""
    from flax import nnx

    from bux.architecture import Model

    model = nnx.eval_shape(lambda: Model(arch, rngs=nnx.Rngs(0)))
    sizes = _param_sizes(model)
    embedding, before, after, idle, types = set(), set(), set(), 0.0, set()
    for layer, module in _layers(arch.graph, model.graph):
        c, ids = layer.config, set(_param_sizes(module))
        types.add(c.type)
        if c.type in EMBEDDING:
            embedding |= ids
        elif after or "head" in types:
            after |= ids - before
        elif not ids <= before:  # a shared module counts once
            idle += _idle(module, c)
            before |= ids
    total = sum(sizes.values())
    dropped = sum(sizes[i] for i in embedding | after)
    return {
        "d_model": arch.d_model,
        "total": total,
        "nonembed": total - dropped,
        "active": total - dropped - idle,
        "moe": "moe" in types,
        "hybrid": bool(types & LINEAR),
        "arch": hashlib.sha1(arch.model_dump_json().encode()).hexdigest()[:12],
    }


def _param_sizes(module: Any) -> dict[int, int]:
    """The size of each parameter under `module`, by its identity."""
    from flax import nnx

    return {
        id(v): math.prod(v.shape)
        for _, v in nnx.iter_graph(module)
        if isinstance(v, nnx.Param)
    }


def _layers(node: Any, module: Any) -> Any:
    """Each layer of an IR graph beside its module, in order (as `module_at`)."""
    from bux.lang import ir

    match node:
        case ir.Layer():
            yield node, module
        case ir.Seq(items=items):
            for n, m in zip(items, module.items, strict=True):
                yield from _layers(n, m)
        case ir.Sum(terms=items) | ir.Prod(factors=items):
            for n, m in zip(items, module.branches, strict=True):
                yield from _layers(n, m)
        case ir.Residual(body=body) | ir.Neg(body=body):
            yield from _layers(body, module.body)
        case ir.Repeat(body=body):
            for m in module.items:
                yield from _layers(body, m)


def _idle(module: Any, c: Any) -> float:
    """The parameters of an MoE's routed experts a token skips."""
    if c.type != "moe":
        return 0.0
    import jax
    from flax import nnx

    routed = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(nnx.state(module, nnx.Param)):
        top = str(getattr(path[0], "key", path[0]))
        if (
            leaf.shape
            and leaf.shape[0] == c.num_experts
            and top
            not in (
                "router",
                "shared",
                "shared_gate",
            )
        ):
            routed += math.prod(leaf.shape)
    return routed * (1 - c.top_k / c.num_experts)


def _flops(run: Any) -> dict:
    if run is None:
        return {"error": "no run"}
    from bux.__main__ import flops_per_token

    work = flops_per_token(run)
    if work is None or not math.isfinite(work):
        return {"error": "a layer has no FLOP count"}
    return {"flops": work, **count(run.arch)}


def notes(stem: str, arms: tuple[str, ...], folders: tuple[Path, ...]) -> dict:
    """The era, source, change and claim of an arm, from the comment above it in its
    file (``# 2023 Llama 2, ...: heads of 128; throughput``)."""
    label = arms[0] if arms else ""
    if m := re.fullmatch(r"m(\d{4})\w*", label):
        return {
            "era": int(m.group(1)),
            "source": label,
            "change": "milestone",
            "claim": "",
        }
    file = find(stem, folders)
    if not arms or file is None:
        return {"era": None, "source": "", "change": "", "claim": ""}
    lines = file.read_text().splitlines()
    for i, line in enumerate(lines):
        if (
            (m := ARM.match(line))
            and m.group(1) == label
            and (above := _above(lines, i))
        ):
            return _comment(above)
    return {"era": None, "source": "", "change": "", "claim": ""}


def _above(lines: list[str], i: int) -> str:
    """The comment lines right above line `i`, joined."""
    block = []
    while i > 0 and lines[i - 1].strip().startswith("#"):
        i -= 1
        block.insert(0, lines[i].strip().lstrip("#").strip())
    return " ".join(block)


def _comment(text: str) -> dict:
    year = re.search(r"\b(19|20)\d\d\b", text)
    source, _, rest = text.partition(": ") if ": " in text else ("", "", text)
    change, _, claim = rest.rpartition("; ") if "; " in rest else (rest, "", "")
    return {
        "era": int(year.group(0)) if year else None,
        "source": source,
        "change": change,
        "claim": claim,
    }
