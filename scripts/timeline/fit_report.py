"""The fit's tables: rows of numbers (for JSON and W&B), and the same as Markdown.

A number with an uncertainty comes with ``<key>_lo`` and ``<key>_hi``: the 5th and
95th percentiles of the noise bootstrap (the leave-one-rung-out range is in the
JSON and the laws' table).
"""

import math
from typing import Any

from fit_analysis import LIMITED, REFERENCES, Analysis, tokens_label, verdict
from fit_points import Point

#: The leaderboard's columns: header, key, format.
LEADERBOARD = [
    ("variant", "family", "s"), ("era", "era", "s"), ("change", "change", "s"),
    ("claim", "claim", "s"), ("verdict", "verdict", "s"), ("Δ d512 %", "delta_512", "+.2f"),
    ("Δ d768 %", "delta_768", "+.2f"), ("Δ d1024 %", "delta_1024", "+.2f"),
    ("multiplier", "mult", ".3g"), ("slope", "mult_slope", "+.3f"),
    ("× at target", "mult_target", ".3g"), ("vs", "reference", "s"),
    ("(a) speedup to vs@1T", "speedup", ".3g"),
    ("(a) Δ at vs@1T GPU-h %", "delta_hours", "+.2f"), ("(b) Δ 1T vs %", "delta_1T", "+.2f"),
    ("(b) Δ 1T vs root %", "delta_1T_root", "+.2f"),
]  # fmt: skip
UNLIMITED = [
    ("family", "family", "s"), ("metric", "metric", "s"), ("N active", "N_active", "si"),
    ("N total", "N_total", "si"), ("FLOPs/token", "flops_per_token", "si"), ("MFU", "mfu", ".2f"),
    ("MFU from", "mfu_from", "s"), ("loss, unlimited tokens", "asymptote", ".4f"),
    ("vs", "reference", "s"),
    ("GPU-h to vs@20tpp", "hours_20tpp", "si"), ("speedup", "speedup_20tpp", ".3g"),
    ("GPU-h to vs@1T", "hours_1T", "si"), ("speedup", "speedup_1T", ".3g"),
    ("Δ at vs@1T GPU-h %", "delta_hours_1T", "+.2f"),
]  # fmt: skip
LEARNING_RATES = [
    ("family", "family", "s"), ("rung", "rung", "s"), ("trials", "trials", "s"),
    ("learning rates", "lrs", "s"), ("best", "best", ".3g"), ("vertex", "vertex", ".3g"),
    ("best at the scan's edge", "edge", "s"),
]  # fmt: skip
LIMITED_COLUMNS = [
    ("family", "family", "s"), ("metric", "metric", "s"), ("tokens", "tokens", "s"),
    ("loss", "loss", ".4f"), ("leave one rung out", "loro", "s"),
    ("Δ vs root %", "delta", "+.2f"), ("Δ vs parent %", "delta_parent", "+.2f"),
]  # fmt: skip


def boot(a: Analysis, f: str, path: tuple, row: dict, key: str) -> None:
    """Set `row[key]` from family `f`'s quantity at `path`, with its bootstrap range."""
    q = a.q.get(f, {})
    for k in path:
        q = q.get(k, {}) if isinstance(q, dict) else {}
    if isinstance(q, int | float):
        row[key] = float(q)
        if (r := a.boot.get(f, {}).get(path)) is not None:
            row[f"{key}_lo"], row[f"{key}_hi"] = r


def leaderboard(a: Analysis, notes: dict[str, dict], root: str) -> list[dict]:
    """A row per family of `a`'s metric, the root's first, the rest by their
    multiplier at the target (or over the rungs)."""
    rows = []
    for f in a.fams:
        row: dict[str, Any] = {
            "family": f,
            **{k: notes.get(f, {}).get(k) for k in ("era", "change", "claim")},
        }
        for w in (512, 768, 1024):
            boot(a, f, ("rungs", w, "delta"), row, f"delta_{w}")
        boot(a, f, ("mult",), row, "mult")
        boot(a, f, ("mult_slope",), row, "mult_slope")
        boot(a, f, ("mult_target",), row, "mult_target")
        boot(a, f, ("unlimited", "1T", "speedup"), row, "speedup")
        boot(a, f, ("unlimited", "1T", "delta_equal_hours"), row, "delta_hours")
        boot(a, f, ("limited", "1T", "delta_parent"), row, "delta_1T")
        boot(a, f, ("limited", "1T", "delta"), row, "delta_1T_root")
        row["reference"] = a.q.get(f, {}).get("reference")
        row["verdict"] = verdict(a.q.get(f, {}), a.sigma)
        rows.append(row)
    rest = sorted(rows[1:], key=lambda r: -(r.get("mult_target") or r.get("mult") or 0))
    return rows[:1] + rest


def multipliers(analyses: list[Analysis]) -> list[dict]:
    """A row per family, metric and rung: its loss, against its parent's, and its
    multiplier from its loss and from its law."""
    rows = []
    for a in analyses:
        for f, q in a.q.items():
            for w in q["rungs"]:
                row = {"family": f, "metric": a.metric, "rung": w}
                row |= {k: q["rungs"][w].get(k) for k in ("n", "d", "loss")}
                boot(a, f, ("rungs", w, "delta"), row, "delta")
                boot(a, f, ("rungs", w, "mult"), row, "mult")
                boot(a, f, ("rungs", w, "mult_law"), row, "mult_law")
                boot(a, f, ("mult_target",), row, "mult_target")
                rows.append(row)
    return rows


def unlimited(analyses: list[Analysis], targets: dict) -> list[dict]:
    """A row per family and metric: its target, and the GPU-hours it takes to reach
    the root target's losses."""
    rows = []
    for a in analyses:
        for f in a.q:
            if f not in targets:
                continue
            t = targets[f]
            row = {"family": f, "metric": a.metric, "N_active": t.n_active,
                   "N_total": t.n_total, "flops_per_token": t.flops, "mfu": t.mfu,
                   "mfu_from": t.mfu_from, "reference": a.q[f].get("reference")}  # fmt: skip
            boot(a, f, ("asymptote", "loss"), row, "asymptote")
            boot(a, f, ("asymptote", "delta"), row, "asymptote_delta")
            for ref in REFERENCES:
                boot(a, f, ("unlimited", ref, "loss_ref"), row, f"loss_{ref}")
                boot(a, f, ("unlimited", ref, "hours"), row, f"hours_{ref}")
                boot(a, f, ("unlimited", ref, "speedup"), row, f"speedup_{ref}")
                boot(
                    a,
                    f,
                    ("unlimited", ref, "delta_equal_hours"),
                    row,
                    f"delta_hours_{ref}",
                )
            rows.append(row)
    return rows


def limited(analyses: list[Analysis]) -> list[dict]:
    """A row per family, metric and token budget: its target's predicted loss, with
    the leave-one-rung-out range, against the root target's."""
    rows = []
    for a in analyses:
        for f, q in a.q.items():
            for d in map(tokens_label, LIMITED):
                if d not in q.get("limited", {}):
                    continue
                row = {"family": f, "metric": a.metric, "tokens": d}
                boot(a, f, ("limited", d, "loss"), row, "loss")
                if (r := a.loro.get(f, {}).get(("limited", d, "loss"))) is not None:
                    row["loro"] = f"[{r[0]:.4f}, {r[1]:.4f}]"
                boot(a, f, ("limited", d, "delta"), row, "delta")
                boot(a, f, ("limited", d, "delta_parent"), row, "delta_parent")
                rows.append(row)
    return rows


def laws(analyses: list[Analysis]) -> list[dict]:
    """A row per family and metric: its law, its parent and its own terms."""
    rows = []
    for a in analyses:
        for f in a.laws:
            fam = a.fams[f]
            row = {"family": f, "metric": a.metric, "parent": fam.parent or "-",
                   "own": " ".join(fam.free), "points": len(fam.y)}  # fmt: skip
            for k in ("E", "A", "alpha", "B", "beta"):
                boot(a, f, ("law", k), row, k)
            rows.append(row)
    return rows


def points_table(points: list[Point], metrics: list[str]) -> list[dict]:
    """A row per point: where it sits, its counts, its learning rates and metrics."""
    rows = []
    for p in points:
        row = {"family": p.family, "rung": p.width, "t": p.per_param, "tokens": p.tokens,
               "N_active": p.n, "N_total": p.counts.get("total"),
               "flops_per_token": p.counts.get("flops"), "lr": p.lr,
               "lrs": len(p.lrs), "lr_edge": p.edge, "runs": p.replicates}  # fmt: skip
        rows.append(row | {m: p.metrics.get(m) for m in metrics})
    return rows


def markdown(rows: list[dict], columns: list[tuple] | None = None) -> str:
    """`rows` as a Markdown table; a number with a range as ``x [lo, hi]``."""
    if not rows:
        return "(none)\n"
    if columns is None:
        keys = [k for k in rows[0] if not k.endswith(("_lo", "_hi"))]
        columns = [(k, k, _default(rows, k)) for k in keys]
    head = "| " + " | ".join(h for h, _, _ in columns) + " |"
    rule = "|" + "---|" * len(columns)
    body = [
        "| " + " | ".join(cell(r, k, f) for _, k, f in columns) + " |" for r in rows
    ]
    return "\n".join([head, rule, *body]) + "\n"


def _default(rows: list[dict], key: str) -> str:
    v = next((r[key] for r in rows if r.get(key) is not None), None)
    return ".4g" if isinstance(v, float) else "s"


def cell(row: dict, key: str, fmt: str) -> str:
    v = row.get(key)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    text = _format(v, fmt)
    if f"{key}_lo" in row:
        plain = fmt.lstrip("+")
        text += (
            f" [{_format(row[f'{key}_lo'], plain)}, {_format(row[f'{key}_hi'], plain)}]"
        )
    return text


def _format(v: Any, fmt: str) -> str:
    if fmt == "s" or not isinstance(v, int | float) or isinstance(v, bool):
        return str(v)
    return si(v) if fmt == "si" else format(v, fmt)


def si(n: float) -> str:
    if not math.isfinite(n):
        return "never" if n > 0 else str(n)
    for suffix, k in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= k:
            return f"{n / k:.3g}{suffix}"
    return f"{n:.3g}"


LONG_CONTEXT = [
    ("family", "family", "s"), ("vs", "parent", "s"), ("rung", "rung", "s"),
    ("Δ repo_bpb_0 %", "delta_0", "+.2f"), ("Δ repo_bpb_8k %", "delta_8k", "+.2f"),
    ("8k minus 0, points", "gain", "+.2f"), ("8k / 0", "ratio", ".3f"),
    ("parent's 8k / 0", "parent_ratio", ".3f"),
]  # fmt: skip


def long_context(
    analyses: list[Analysis], first: str = "repo_bpb_0", last: str = "repo_bpb_8k"
) -> list[dict]:
    """A row per family and rung: its deltas against its parent on the repositories'
    first tokens and on those after 8k, and how much more it gains late (negative:
    the change helps long context most), by how much its late tokens gain overall."""
    by = {a.metric: a for a in analyses}
    if first not in by or last not in by:
        return []
    a0, a8 = by[first], by[last]
    rows = []
    for f, q in a8.q.items():
        for w, r in q["rungs"].items():
            r0 = a0.q.get(f, {}).get("rungs", {}).get(w, {})
            if "delta" in r and "delta" in r0:
                rows.append(_long_row(f, w, r0, r, a0, a8))
    return sorted(rows, key=lambda r: r["gain"])


def _long_row(f: str, w: int, r0: dict, r8: dict, a0: Analysis, a8: Analysis) -> dict:
    parent = a8.fams[f].parent
    p0, p8 = (
        x.q.get(parent, {}).get("rungs", {}).get(w, {}).get("loss") for x in (a0, a8)
    )
    return {
        "family": f, "parent": parent, "rung": w, "delta_0": r0["delta"], "delta_8k": r8["delta"],
        "gain": r8["delta"] - r0["delta"], "ratio": r8["loss"] / r0["loss"],
        "parent_ratio": p8 / p0 if p0 and p8 else None,
    }  # fmt: skip


#: The report's one row per family of the extrapolations, and its columns.
UNLIMITED_SUMMARY = [
    ("family", "family", "s"), ("vs", "reference", "s"), ("N active", "N_active", "si"),
    ("N total", "N_total", "si"), ("FLOPs/token", "flops_per_token", "si"), ("MFU", "mfu", ".2f"),
    ("MFU from", "mfu_from", "s"), ("objective: speedup to vs@1T", "speedup_objective", ".3g"),
    ("Δ at vs@1T GPU-h %", "delta_hours_objective", "+.2f"),
    ("repo: speedup to vs@1T", "speedup_repo_bpb", ".3g"),
    ("Δ at vs@1T GPU-h %", "delta_hours_repo_bpb", "+.2f"),
]  # fmt: skip
LIMITED_SUMMARY = [
    ("family", "family", "s"), ("vs", "reference", "s"), ("objective at 1T", "loss_1T", ".4f"),
    ("leave one rung out", "loro_1T", "s"), ("Δ 130B %", "delta_130B", "+.2f"),
    ("Δ 1T %", "delta_1T", "+.2f"), ("Δ 10T %", "delta_10T", "+.2f"),
    ("Δ 1T vs root %", "delta_1T_root", "+.2f"), ("repo Δ 1T %", "repo_delta_1T", "+.2f"),
    ("repo after 8k Δ 1T %", "repo8k_delta_1T", "+.2f"),
]  # fmt: skip
LONG_SUMMARY = [
    ("family", "family", "s"), ("vs", "parent", "s"), ("rungs", "rungs", "s"),
    ("Δ repo_bpb_0 %", "delta_0", "+.2f"), ("Δ repo_bpb_8k %", "delta_8k", "+.2f"),
    ("8k minus 0, points", "gain", "+.2f"), ("by rung", "by_rung", "s"),
]  # fmt: skip


def unlimited_summary(analyses: list[Analysis], targets: dict) -> list[dict]:
    """A row per family: its target, and for the objective and the repositories the
    speedup to its reference's 1T-token loss and the delta in the GPU-hours it takes."""
    by = {a.metric: a for a in analyses}
    main = by.get("objective") or analyses[0]
    rows = []
    for f in main.q:
        if f not in targets:
            continue
        t = targets[f]
        row = {"family": f, "reference": main.q[f].get("reference"), "N_active": t.n_active,
               "N_total": t.n_total, "flops_per_token": t.flops, "mfu": t.mfu, "mfu_from": t.mfu_from}  # fmt: skip
        for m in ("objective", "repo_bpb"):
            if m in by:
                boot(by[m], f, ("unlimited", "1T", "speedup"), row, f"speedup_{m}")
                boot(
                    by[m],
                    f,
                    ("unlimited", "1T", "delta_equal_hours"),
                    row,
                    f"delta_hours_{m}",
                )
        rows.append(row)
    return rows


def limited_summary(analyses: list[Analysis]) -> list[dict]:
    """A row per family: its target's objective at 1T tokens, and its deltas
    against its reference's target at each of `LIMITED` tokens."""
    by = {a.metric: a for a in analyses}
    main = by.get("objective") or analyses[0]
    rows = []
    for f, q in main.q.items():
        if "limited" not in q:
            continue
        row = {"family": f, "reference": q.get("reference")}
        boot(main, f, ("limited", "1T", "loss"), row, "loss_1T")
        if (r := main.loro.get(f, {}).get(("limited", "1T", "loss"))) is not None:
            row["loro_1T"] = f"[{r[0]:.4f}, {r[1]:.4f}]"
        for d in map(tokens_label, LIMITED):
            boot(main, f, ("limited", d, "delta_parent"), row, f"delta_{d}")
        boot(main, f, ("limited", "1T", "delta"), row, "delta_1T_root")
        for m, key in (
            ("repo_bpb", "repo_delta_1T"),
            ("repo_bpb_8k", "repo8k_delta_1T"),
        ):
            if m in by:
                boot(by[m], f, ("limited", "1T", "delta_parent"), row, key)
        rows.append(row)
    return rows


def long_summary(rows: list[dict]) -> list[dict]:
    """The long-context rows, one per family: their means over its rungs."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["family"], []).append(r)
    out = []
    for f, rs in by.items():
        rs = sorted(rs, key=lambda r: r["rung"])
        mean = {
            k: sum(r[k] for r in rs) / len(rs) for k in ("delta_0", "delta_8k", "gain")
        }
        out.append({"family": f, "parent": rs[0]["parent"], "rungs": len(rs), **mean,
                    "by_rung": " ".join(f"d{r['rung']} {r['gain']:+.1f}" for r in rs)})  # fmt: skip
    return sorted(out, key=lambda r: r["gain"])
