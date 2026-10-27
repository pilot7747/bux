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

from fit_points import ROOT, classify, family_key, parse_name

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
    mfu_measured: bool
    d_model: float | None = None

    def hours(self, tokens: float) -> float:
        """B300 GPU-hours to train on `tokens`."""
        return tokens * self.flops / (self.mfu * PEAK * 3600)

    def tokens(self, hours: float) -> float:
        """The tokens `hours` of B300 GPU time train on."""
        return hours * self.mfu * PEAK * 3600 / self.flops


def load(path: Path, budget: str = "B1b") -> dict[str, dict]:
    """The targets by family from a JSON file: ``{family: {"N_active", "N_total",
    "flops_per_token"?, "d_model"?}}``, or ``serving.py table``'s
    ``serving-table.json`` (its matches at `budget`)."""
    raw = json.loads(path.read_text())
    if "rows" not in raw:
        return {f: _aliases(v) for f, v in raw.items()}
    out: dict[str, dict] = {}
    for row in raw["rows"]:
        match = row.get(budget) or {}
        if "d_model" in match and "active" in match:
            row_of = {"file": row["file"], "arm": row["arm"]}
            out.setdefault(
                serving_family(row["file"], row["arm"]), _aliases(match) | row_of
            )
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


def _aliases(v: dict) -> dict:
    return {
        "n_active": v.get("N_active", v.get("active")),
        "n_total": v.get("N_total", v.get("params")),
        "flops": v.get("flops_per_token"),
        "d_model": v.get("d_model"),
    }


def serving_family(file: str, arm: str) -> str:
    """The family of a serving table's row: its file's stem and its arm's labels."""
    stem, labels = parse_name(arm)
    arms = classify(labels)[3]
    stem = Path(file).stem if not labels else stem
    return ROOT if arms == ("base",) else family_key(stem, arms)


def match(file: Path, family: str, arch: Any, budget: str, cache: dict) -> dict:
    """The target of `file`'s arm whose architecture is `arch()` (at ``d_model =
    1024``), at `budget`, by serving.py's match: cached in `cache` by `family`."""
    key = f"{family}|{budget}|match"
    if key not in cache:
        import serving

        try:
            name = arm_named(serving.compile_arms(file, ("d_model = 1024",)), arch())
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


def target(raw: dict, flops: float, sparse: bool, measured: dict | None) -> Target:
    """A target from its counts, its FLOPs per token and its measured MFU, if any."""
    mfu, real = (MFU_SPARSE if sparse else MFU_DENSE), False
    if measured and "mfu" in measured:
        mfu, real = float(measured["mfu"]), True
    elif measured and "tokens_per_second_per_gpu" in measured:
        mfu, real = float(measured["tokens_per_second_per_gpu"]) * flops / PEAK, True
    return Target(
        n_active=float(raw["n_active"]),
        n_total=float(raw.get("n_total") or math.nan),
        flops=flops,
        mfu=mfu,
        mfu_measured=real,
        d_model=raw.get("d_model"),
    )


def as_dict(t: Target) -> dict:
    return asdict(t)
