"""The fit's tables: rows of numbers (for JSON and W&B), and the same as Markdown.

A number with an uncertainty comes with ``<key>_lo`` and ``<key>_hi``: the 5th and
95th percentiles of the noise bootstrap (the leave-one-rung-out range is in the
JSON and the laws' table).
"""

import math
from typing import Any

from fit_analysis import LIMITED, REFERENCES, Analysis, tokens_label
from fit_points import Point

#: The leaderboard's columns: header, key, format.
LEADERBOARD = [
    ("variant", "family", "s"), ("era", "era", "s"), ("change", "change", "s"),
    ("claim", "claim", "s"), ("Δ d512 %", "delta_512", "+.2f"),
    ("Δ d768 %", "delta_768", "+.2f"), ("Δ d1024 %", "delta_1024", "+.2f"),
    ("multiplier", "mult", ".3g"), ("slope", "mult_slope", "+.3f"),
    ("× at target", "mult_target", ".3g"), ("vs", "reference", "s"),
    ("(a) speedup at vs@20tpp", "speedup", ".3g"),
    ("(a) Δ at vs@1T GPU-h %", "delta_hours", "+.2f"), ("(b) Δ 1T vs %", "delta_1T", "+.2f"),
    ("(b) Δ 1T vs root %", "delta_1T_root", "+.2f"),
]  # fmt: skip
UNLIMITED = [
    ("family", "family", "s"), ("metric", "metric", "s"), ("N active", "N_active", "si"),
    ("N total", "N_total", "si"), ("FLOPs/token", "flops_per_token", "si"), ("MFU", "mfu", ".2f"),
    ("measured", "mfu_measured", "s"), ("loss, unlimited tokens", "asymptote", ".4f"),
    ("vs", "reference", "s"),
    ("GPU-h to vs@20tpp", "hours_20tpp", "si"), ("speedup", "speedup_20tpp", ".3g"),
    ("GPU-h to vs@1T", "hours_1T", "si"), ("speedup", "speedup_1T", ".3g"),
    ("Δ at vs@1T GPU-h %", "delta_hours_1T", "+.2f"),
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
        row = {
            "family": f,
            **{k: notes.get(f, {}).get(k) for k in ("era", "change", "claim")},
        }
        for w in (512, 768, 1024):
            boot(a, f, ("rungs", w, "delta"), row, f"delta_{w}")
        boot(a, f, ("mult",), row, "mult")
        boot(a, f, ("mult_slope",), row, "mult_slope")
        boot(a, f, ("mult_target",), row, "mult_target")
        boot(a, f, ("unlimited", "20tpp", "speedup"), row, "speedup")
        boot(a, f, ("unlimited", "1T", "delta_equal_hours"), row, "delta_hours")
        boot(a, f, ("limited", "1T", "delta_parent"), row, "delta_1T")
        boot(a, f, ("limited", "1T", "delta"), row, "delta_1T_root")
        row["reference"] = a.q.get(f, {}).get("reference")
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
                   "mfu_measured": t.mfu_measured, "reference": a.q[f].get("reference")}  # fmt: skip
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
