"""Turn a serving sweep's measurements into each family's target of equal serving
cost: the sizes `--targets` of fit.py takes, measured in vLLM rather than modelled.

A sweep times each family at a few widths under the agentic load of
``scripts/old_vs_new/agentic.py``, a JSON line per width and concurrency, with the
family, the width, its tensor parallelism and its parameters (``profile``, from
``serving.py cost``). A width's cost is its best sessions per hour per GPU (the
engine's, over its TP GPUs) at a median time per output token within the SLO. The
budget is the ``budget`` family's (GPT-3 6.7B's serving shape, old_2022 at d_model
4096), averaged over its seeds. A family's target is where its sessions per hour
cross the budget, interpolated in log sessions per hour against log active
parameters between the two widths that bracket it (extrapolated from the nearest
two otherwise, and marked so); its width and total parameters are interpolated the
same way.

A family timed at one width only, with the serving shape of a reference family
(``--like FAMILY=REFERENCE``), takes the reference's curve, scaled by their ratio
at that width.

    python scripts/timeline/sweep_fit.py sweep.jsonl --out targets-measured.json \\
        --markdown serving-measured.md --analytic serving-table.json
"""

import argparse
import itertools
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

BUDGET = "budget"
#: Positions the 2020 baseline was trained with, where its serving export has more.
TRAINED_POSITIONS = 16384


@dataclass
class Width:
    """A family at one width: its best point within the SLO per lane (seed)."""

    d_model: int
    tp: int  # GPUs of the engines: tensor parallel times data parallel
    params: float
    active: float
    best: list[dict] = field(default_factory=list)
    model: float | None = None  # the model's best sessions per hour per GPU

    def cost(self) -> float:
        """Sessions per hour per GPU, the mean of its lanes' bests (0 if none)."""
        values = [b["sessions_per_hour"] / self.tp for b in self.best if b]
        return statistics.mean(values) if values else 0.0


def read(paths: list[Path]) -> list[dict]:
    """The sweep's rows, from JSON-lines files."""
    rows = []
    for path in paths:
        rows += [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return rows


def widths(rows: list[dict], slo: float) -> dict[str, dict[int, Width]]:
    """Every family's widths, each with its lanes' best points within `slo`."""
    lanes: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        lanes[(r["family"], r["d_model"], r["name"])].append(r)
    out: dict[str, dict[int, Width]] = defaultdict(dict)
    for (family, d, _), points in lanes.items():
        first = points[0]
        gpus = first.get("gpus", first["tp"])
        w = out[family].setdefault(d, Width(d, gpus, *_params(first)))
        w.model = (first.get("predicted_best") or {}).get("sessions_per_hour")
        ok = [p["measured"] for p in points if p["measured"]["tpot_p50_ms"] <= slo]
        w.best.append(max(ok, key=lambda m: m["sessions_per_hour"], default={}))
    return out


def _params(row: dict) -> tuple[float, float]:
    """All parameters and active ones besides embeddings, as trained: an export
    with more learned positions than training had counts them as training's."""
    p = row["profile"] or {}
    params = p.get("params", math.nan)
    for s in row["settings"]:
        key, _, value = s.partition("=")
        if key.strip() == "positions":
            params -= (int(value) - TRAINED_POSITIONS) * row["d_model"]
    return params, p.get("active", math.nan)


def match(ws: dict[int, Width], budget: float) -> dict:
    """Where a family's cost crosses `budget`: interpolated in log-log of sessions
    per hour against active parameters between the bracketing widths."""
    pts = sorted((w for w in ws.values() if w.cost() > 0), key=lambda w: w.active)
    if len(pts) < 2:
        return {"error": f"{len(pts)} width(s) within the SLO"}
    lo, hi, kind = _bracket(pts, budget)
    t = math.log(lo.cost() / budget) / math.log(lo.cost() / hi.cost())
    out = {
        "N_active": _lerp(lo.active, hi.active, t),
        "N_total": _lerp(lo.params, hi.params, t),
        "d_model": _lerp(lo.d_model, hi.d_model, t),
        "tp": hi.tp if t > 0.5 else lo.tp,
        "bracket": [lo.d_model, hi.d_model],
        "interpolation": kind,
    }
    return out


def _bracket(pts: list[Width], budget: float) -> tuple[Width, Width, str]:
    """The two widths to interpolate between: the narrower at or above the budget
    and the next wider below it, else the nearest two (extrapolation)."""
    for a, b in itertools.pairwise(pts):
        if a.cost() >= budget > b.cost():
            return a, b, "interpolated"
    if pts[-1].cost() >= budget:
        return pts[-2], pts[-1], "extrapolated wider"
    return pts[0], pts[1], "extrapolated narrower"


def _lerp(a: float, b: float, t: float) -> float:
    """a to b at t, in log."""
    return math.exp(math.log(a) + t * (math.log(b) - math.log(a)))


def like(ws: Width, ref: dict[int, Width]) -> dict[int, Width]:
    """A family timed at one width, as its reference's widths scaled by their
    ratio of cost and of parameters there."""
    r = ref.get(ws.d_model)
    if r is None or not r.cost():
        raise LookupError(f"the reference has no width {ws.d_model}")
    cost, params, active = (
        ws.cost() / r.cost(),
        ws.params / r.params,
        ws.active / r.active,
    )
    out = {}
    for d, w in ref.items():
        scaled = Width(d, w.tp, w.params * params, w.active * active)
        scaled.best = [{"sessions_per_hour": w.cost() * w.tp * cost}]
        out[d] = scaled
    return out


def targets(fams: dict[str, dict[int, Width]], likes: dict[str, str]) -> dict:
    """Every family's measured target, by `match` (via its reference for `likes`)."""
    budget = statistics.mean(w.cost() for w in fams[BUDGET].values())
    out = {}
    for family, ws in sorted(fams.items()):
        if family == BUDGET:
            continue
        via = likes.get(family)
        try:
            curve = like(next(iter(ws.values())), fams[via]) if via else ws
        except LookupError as err:
            out[family] = {"measured": False, "error": str(err)}
            continue
        m = match(curve, budget)
        entry = {**m, "measured": "error" not in m, "budget": budget}
        entry["sessions_per_hour"] = {
            str(d): round(w.cost(), 1) for d, w in sorted(ws.items())
        }
        if via:
            entry["via"] = via
        if len({w.tp for w in curve.values()}) > 1:
            entry["by_gpus"] = by_gpus(curve, budget)
        out[family] = entry
    return out


def by_gpus(ws: dict[int, Width], budget: float) -> dict[str, dict]:
    """A family whose widths take different GPU counts, each count apart: its
    widths' costs, and their own match where two or more widths have it. Tensor
    parallelism comes in steps, so the family's cost jumps between them."""
    out: dict[str, dict] = {}
    for gpus in sorted({w.tp for w in ws.values()}):
        group = {d: w for d, w in ws.items() if w.tp == gpus}
        entry: dict = {
            "sessions_per_hour": {str(d): round(w.cost(), 1) for d, w in group.items()}
        }
        for d, w in group.items():
            entry.setdefault("widths", {})[str(d)] = {
                "N_active": w.active,
                "N_total": w.params,
            }
        if len(group) > 1:
            entry["match"] = match(group, budget)
        out[str(gpus)] = entry
    return out


def markdown(fams: dict, out: dict, analytic: dict, slo: float) -> str:
    """The table: a family per row, its widths, its match and the model's."""
    budget = statistics.mean(w.cost() for w in fams[BUDGET].values())
    lanes = [b for w in fams[BUDGET].values() for b in w.best if b]
    seeds = ", ".join(
        f"{b['sessions_per_hour']:.0f} at {b['concurrency']}" for b in lanes
    )
    spread = (
        max(b["sessions_per_hour"] for b in lanes)
        - min(b["sessions_per_hour"] for b in lanes)
    ) / budget
    intro = (
        f"Budget: old_2022 at d_model 4096 (GPT-3 6.7B's serving shape), its best "
        f"{budget:.0f} sessions/h per B300 at p50 TPOT <= {slo:g} ms: the mean of "
        f"{len(lanes)} lanes ({seeds}; spread {spread:.1%}). Each width: its best "
        "sessions/h per GPU within the SLO (at sessions at once per GPU, its GPUs; the "
        "model's best there). "
        "Match: where the family's cost crosses the budget, in log-log against active "
        "parameters. Analytic: serving.py's match to its own budget (985/h), the "
        "targets the fits took before.\n"
    )
    head = (
        "| family | widths: best s/h per GPU (c per GPU, GPUs) | match: d, active / total, GPUs "
        "| analytic: d, active | active, measured / analytic |"
    )
    lines = [
        "# Serving cost of the timeline's families, measured in vLLM\n",
        intro,
        head,
        "|---|---|---|---|---|",
    ]
    for family, entry in out.items():
        lines.append(_row(family, fams[family], entry, analytic.get(family, {})))
    return "\n".join([*lines, "", errors(fams)]) + "\n"


def errors(fams: dict[str, dict[int, Width]]) -> str:
    """The model's error: measured over modelled best sessions per hour, by width."""
    ratios = {
        f"{family} {d}": w.cost() / w.model
        for family, ws in fams.items()
        for d, w in ws.items()
        if w.model and w.cost()
    }
    if not ratios:
        return ""
    values = sorted(ratios.values())
    worst = sorted(ratios.items(), key=lambda kv: -abs(math.log(kv[1])))[:8]
    return (
        f"Measured over modelled best sessions/h, {len(values)} widths: median "
        f"{statistics.median(values):.2f}, from {values[0]:.2f} to {values[-1]:.2f}; "
        "farthest: " + ", ".join(f"{k} {v:.2f}" for k, v in worst) + "."
    )


def _row(family: str, ws: dict[int, Width], e: dict, a: dict) -> str:
    cells = []
    for d, w in sorted(ws.items()):
        best = [b for b in w.best if b]
        c = "/".join(str(b["concurrency"] // w.tp) for b in best) or "-"
        tp = f", {w.tp} GPUs" if w.tp > 1 else ""
        model = f", model {w.model:.0f}" if w.model else ""
        cells.append(f"{d}: {w.cost():.0f} ({c}{tp}{model})")
    via = f" via {e['via']}" if "via" in e else ""
    if "by_gpus" in e:
        via += f" (across {'/'.join(e['by_gpus'])} GPUs: a step)"
    if "N_active" in e:
        tp = f", {e['tp']} GPUs" if e["tp"] > 1 else ""
        flag = (
            "" if e["interpolation"] == "interpolated" else f" ({e['interpolation']})"
        )
        got = f"{e['d_model']:.0f}, {_si(e['N_active'])} / {_si(e['N_total'])}{tp}{flag}{via}"
    else:
        got = e.get("error", "-")
    ana = f"{a['d_model']:.0f}, {_si(a['active'])}" if "active" in a else "-"
    ratio = (
        f"{e['N_active'] / a['active']:.2f}"
        if "N_active" in e and "active" in a
        else "-"
    )
    return f"| {family} | {'; '.join(cells)} | {got} | {ana} | {ratio} |"


def _si(n: float) -> str:
    for suffix, k in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(n) >= k:
            return f"{n / k:.3g}{suffix}"
    return f"{n:.3g}"


def analytic_matches(path: Path | None) -> dict[str, dict]:
    """The model's matches by family: a fit's ``fit.json`` (its targets), or
    serving.py table's ``serving-table.json`` (its B1b matches)."""
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    if "targets" in raw:
        return {
            f: {"d_model": t["d_model"], "active": t["n_active"]}
            for f, t in raw["targets"].items()
            if t.get("d_model")
        }
    from fit_targets import serving_family

    out = {}
    for row in raw["rows"]:
        m = row.get("B1b") or {}
        if "d_model" in m and (f := serving_family(row["file"], row["arm"])):
            out.setdefault(f, m)
    return out


def main() -> None:
    a = parser().parse_args()
    fams = widths(read(a.sweeps), a.slo)
    likes = dict(x.split("=", 1) for x in a.like)
    out = targets(fams, likes)
    analytic = analytic_matches(a.analytic)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    if a.markdown:
        a.markdown.write_text(markdown(fams, out, analytic, a.slo))
    print(markdown(fams, out, analytic, a.slo))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("sweeps", nargs="+", type=Path, help="sweep.jsonl files")
    p.add_argument("--out", type=Path, default=Path("targets-measured.json"))
    p.add_argument("--markdown", type=Path, help="where the table goes")
    p.add_argument("--analytic", type=Path, help="a fit.json, or serving-table.json")
    p.add_argument("--slo", type=float, default=30.0, help="p50 TPOT limit, ms")
    p.add_argument(
        "--like",
        action="append",
        default=[],
        metavar="FAMILY=REFERENCE",
        help="a family timed at one width, with REFERENCE's serving shape",
    )
    return p


if __name__ == "__main__":
    main()
