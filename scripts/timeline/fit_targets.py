"""The big targets of equal agentic serving cost, and what training them costs.

A target is a family at the width that serves an agentic coding load as fast as the
budget (``serving.py``'s ``match``): its parameters a token goes through besides the
embeddings, which its law reads, all its parameters, and its FLOPs per training
token. Its GPU-hours on B300s are tokens x FLOPs / (MFU x 2.25e15 x 3600).
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fit_points import classify, family_key, parse_name

PEAK = 2.25e15  # a B300's dense bf16 FLOP/s
#: Placeholder MFUs where no measured throughput is given: dense, and MoE or hybrid.
MFU_DENSE, MFU_SPARSE = 0.35, 0.25


@dataclass
class Target:
    """A family at the size of equal serving cost."""

    n_active: float
    n_total: float
    flops: float  # per training token
    mfu: float
    mfu_from: str  # "measured", "kin <family>", "ancestor <family>" or "placeholder"
    d_model: float | None = None
    cost_from: str = ""  # how its size was had: measured, via a copy, kernel-limited...
    # its size at its lower-bound cost, where its own size bounds the cost from above
    lower: "Target | None" = None
    reason: str | None = None  # why its size is not measured outright

    def hours(self, tokens: float) -> float:
        """B300 GPU-hours to train on `tokens`."""
        return tokens * self.flops / (self.mfu * PEAK * 3600)

    def tokens(self, hours: float) -> float:
        """The tokens `hours` of B300 GPU time train on."""
        return hours * self.mfu * PEAK * 3600 / self.flops


PRECISIONS = ("fp8", "bf16")  # serving precisions a targets file may hold


def load(path: Path, budget: str = "B1b", serve: str | None = None) -> dict[str, dict]:
    """The targets by family from a JSON file: ``{family: {"N_active", "N_total",
    "flops_per_token"?, "d_model"?}}`` (targets-measured.json's, with ``measured``
    and its flags: see `measured`), or ``serving.py table``'s ``serving-table.json``
    (its matches at `budget`). A file of several serving precisions (`serve`, see
    `served`) gives that precision's."""
    raw = json.loads(path.read_text())
    if "rows" not in raw:
        chosen = serve or _precisions(raw)[0]
        return {
            f: _entry(v) for f, v in served(raw, chosen).items() if isinstance(v, dict)
        }
    out: dict[str, dict] = {}
    for row in raw["rows"]:
        match = row.get(budget) or {}
        if "d_model" in match and "active" in match:
            row_of = {"file": row["file"], "arm": row["arm"]}
            if family := serving_family(row["file"], row["arm"]):
                out.setdefault(family, _aliases(match) | row_of)
    return out


def _entry(v: dict) -> dict:
    return measured(v) if "measured" in v else _aliases(v)


def precisions(path: Path) -> list[str]:
    """The serving precisions a targets file holds, the default first: fp8 once
    present, else bf16 (a file of one precision holds bf16's)."""
    return _precisions(json.loads(path.read_text()))


def _precisions(raw: dict) -> list[str]:
    if "rows" in raw:
        return ["bf16"]
    if any(isinstance(raw.get(p), dict) for p in PRECISIONS):
        return [p for p in PRECISIONS if isinstance(raw.get(p), dict)]
    fp8 = any(
        isinstance(v, dict) and isinstance(v.get("fp8"), dict) for v in raw.values()
    )
    return ["fp8", "bf16"] if fp8 else ["bf16"]


def served(raw: dict, serve: str) -> dict:
    """The entries of one serving precision (`serve`), from a file keyed by
    precision (``{"fp8": {family: ...}, "bf16": ...}``) or by family, each entry
    its bf16 measurement with the fp8 one under ``fp8`` (``{family: {..., "fp8":
    {...}}}``); an entry without that precision keeps bf16's, marked."""
    if any(isinstance(raw.get(p), dict) for p in PRECISIONS):
        return raw.get(serve, {})
    out = {}
    for f, v in raw.items():
        if not isinstance(v, dict):
            continue
        bf16 = (
            v["bf16"]
            if isinstance(v.get("bf16"), dict)
            else {k: x for k, x in v.items() if not k.startswith("fp8")}
        )
        if serve == "bf16":
            out[f] = bf16
        elif isinstance(v.get(serve), dict):
            out[f] = v[serve]
        else:
            out[f] = bf16 | {"fallback": "bf16"}
    return out


def by_architecture(raw: dict, file: Path, arch: Any) -> dict | None:
    """The serving table's target of the arm of `file` whose architecture is
    `arch()` at ``d_model = 1024``: for arms serving.py names apart from their
    family, as those that share one architecture."""
    rows = [r for r in raw.values() if "arm" in r and Path(r["file"]).stem == file.stem]
    if not rows:
        return None
    import serving

    arms = serving.compile_arms(file, ("d_model = 1024",))
    try:
        name = arm_named(arms, arch())
    except LookupError, AttributeError:
        return None
    return next((r for r in rows if r["arm"] == name), None)


def measured(v: dict) -> dict:
    """A target of targets-measured.json: its top-level size (the measured match, or
    for a kernel-limited family the anchored or analytic one), how it was had, and,
    where the top level bounds the cost from above (fused-residual families), the
    size at the lower bound."""
    out = _aliases(v) | {
        "cost_from": cost_label(v),
        "reason": v.get("reason") or v.get("note"),
    }
    if v.get("upper_bound") and v.get("anchored"):
        out["lower"] = _aliases(v["anchored"])
    return out


def cost_label(v: dict) -> str:
    """How a measured target's size was had, in a few words."""
    if v.get("fallback"):
        return f"{v['fallback']} only ({_label(v)})"
    return _label(v)


def _label(v: dict) -> str:
    if v.get("kernel_limited"):
        return "kernel-limited, " + ("anchored" if v.get("anchored") else "analytic")
    if "+dsdims" in str(v.get("via", "")):
        return "measured, DeepSeek dims"
    if v.get("upper_bound"):
        return "measured upper bound of cost"
    if v.get("via"):
        return f"measured via {v['via']}"
    return "measured" if v.get("measured") else "analytic"


def _aliases(v: dict) -> dict:
    return {
        "n_active": v.get("N_active", v.get("active")),
        "n_total": v.get("N_total", v.get("params")),
        "flops": v.get("flops_per_token"),
        "d_model": v.get("d_model"),
    }


def serving_family(file: str, arm: str) -> str:
    """The family of a serving table's row: its file's stem and its arm's labels;
    none for an axis file's base arm, which repeats its axis' parent's row."""
    stem, labels = parse_name(arm)
    arms = classify(labels)[3]
    stem = Path(file).stem if not labels else stem
    return "" if arms == ("base",) else family_key(stem, arms)


def match(file: Path, arch: Any, budget: str, cache: dict) -> dict:
    """The target of `file`'s arm whose architecture is `arch` (at ``d_model =
    1024``), at `budget`, by serving.py's match: cached in `cache` by that
    architecture, so a file that changes is matched again."""
    from fit_counts import digest

    key = f"match|{digest(arch)}|{budget}"
    if key not in cache:
        import serving

        try:
            name = arm_named(serving.compile_arms(file, ("d_model = 1024",)), arch)
            cache[key] = _aliases(serving.match(file, budget, arm=name))
        except Exception as err:  # noqa: BLE001  (an arm serving.py cannot price)
            cache[key] = {"error": f"{type(err).__name__}: {str(err)[:200]}"}
    return cache[key]


def arm_named(arms: dict, arch: Any) -> str:
    """The name serving.py gives the arm whose architecture is `arch`."""
    want = arch.model_dump_json()
    for name, a in arms.items():
        if a.model_dump_json() == want:
            return name
    raise LookupError(f"no arm of {len(arms)} has this architecture")


def mfus(path: Path | None) -> dict[str, dict]:
    """Measured training throughput at target scale by family: ``{family: mfu}``,
    ``{family: {"mfu": ...}}`` or ``{family: {"tokens_per_second_per_gpu": ...}}``."""
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    return {f: v if isinstance(v, dict) else {"mfu": v} for f, v in raw.items()}


def target(
    raw: dict, flops: float, sparse: bool, measured: dict | None, source: str
) -> Target:
    """A target from its counts, its FLOPs per token and the measured MFU it takes
    (from `source`), or the placeholder for its kind."""
    mfu = MFU_SPARSE if sparse else MFU_DENSE
    if measured and "mfu" in measured:
        mfu = float(measured["mfu"])
    elif measured and "tokens_per_second_per_gpu" in measured:
        mfu = (
            float(measured["tokens_per_second_per_gpu"])
            * measured.get("flops_per_token", flops)
            / PEAK
        )
    else:
        source = "placeholder"
    return Target(
        n_active=float(raw["n_active"]),
        n_total=float(raw.get("n_total") or math.nan),
        flops=flops,
        mfu=mfu,
        mfu_from=source,
        d_model=raw.get("d_model"),
    )


def as_dict(t: Target) -> dict:
    return asdict(t)
