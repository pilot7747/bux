"""Write the W&B report's text (dashboard.py --notes) from the timeline tutorial and
the fit's report, so that a new fit refreshes both by rerunning this script after
doc_tables.py.

The prose and its tables come from docs/tutorials/timeline.md, its includes inlined
and its Sphinx markup dropped; the per-axis facts, the winners at equal serving cost
and the axes' summary come from fit.py's ``report.md``. The output is Markdown in
``## name`` sections, the names dashboard.py reads.

    python scripts/timeline/dashboard_notes.py fit/report.md notes.md \\
        [--secondary secondary/report.md]

``--secondary`` adds the parts of the secondary metrics' report (stability,
outliers, routing, the 16k switch) as ``secondary: ...`` notes.
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from doc_tables import NAMES

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = ROOT / "docs" / "tutorials" / "timeline.md"
LEADER = "## One-change leaderboard (objective)"
OPENING = "# 2020 to 2026, one change at a time"
ANSWER = "By the end you will have the answer."  # where the opening's headline starts
STEPS = "Getting there"  # and where it ends, at the list of the tutorial's steps
# Each note, by the heading of the tutorial's part it takes, which runs to the next
# heading; a list takes several parts.
PARTS = {
    "best": ["### The candidate at equal serving cost"],
    "factorial": ["### The factorial screen"],
    "swa": ["### The sliding-window frontier against the KDA hybrid"],
    "eras": ["## 7. The milestones"],
    "stability": ["## 4. Test the stability claims", "### Secondary metrics"],
    "long": ["### Long context"],
    "results": ["### Each change alone", "### Short notes", "### Model shape"],
    "caveats": ["### Caveats"],
}
VERDICTS = ("better", "no measurable difference", "worse", "mixed")
# The secondary metrics' notes, by the heading of their report's part.
SECONDARY = {
    "secondary: summary": "## Summary",
    "secondary: flags": "## Flags that touch the claims",
    "secondary: stability": "## 1. Stability (d512 seed 0 lr scans)",
    "secondary: candidates": "## 2. Candidates vs the hybrid (paired medians)",
    "secondary: fp8": "## 3. FP8: claims paired with the 2021 reference",
    "secondary: moe": "## 4. MoE (last observe before 16k)",
    "secondary: switch": "## 5. 16k switch (d512, median of seeds)",
}


def part(lines: list[str], heading: str) -> list[str]:
    """The lines under `heading`, to the next heading."""
    start = lines.index(heading) + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("#")), None)
    return lines[start:end]


def clean(lines: list[str], here: Path) -> str:
    """`lines` as plain Markdown: includes inlined (from `here`, the tutorial's
    folder), images, notes, comments and labels dropped, links and roles as their
    text."""
    out: list[str] = []
    fence = None
    for line in lines:
        if fence is not None:
            if line.startswith(("```", ":::")):
                fence = None
            continue
        if m := re.match(r"^```\{include\} (\S+)", line):
            out += _table(here / m.group(1))
            fence = "include"
            continue
        if line.startswith(("```{", ":::{")):
            fence = line
            continue
        if line.startswith("%") or re.match(r"^\(.*\)=$", line):
            continue
        line = re.sub(r"\s*\(\[[^\]]+\]\(#[^)]*\)\)", "", line)  # ([step 6](#...))
        line = re.sub(r"\s*\(step \d+\)", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        out.append(re.sub(r"\{[a-z:]+\}`([^`]+)`", r"`\1`", line))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _table(path: Path) -> list[str]:
    """An included table's lines, without the generator's comment."""
    return [line for line in path.read_text().splitlines() if not line.startswith("%")]


def opening(lines: list[str], here: Path) -> dict[str, str]:
    """The tutorial's opening: its question, and its answer to the list of steps."""
    body = part(lines, OPENING)
    answer = next(i for i, line in enumerate(body) if line.startswith(ANSWER))
    steps = next(i for i, line in enumerate(body) if line.startswith(STEPS))
    question, headline = body[:answer], body[answer:steps]
    headline[0] = headline[0].removeprefix(ANSWER).lstrip()
    return {"intro": clean(question, here), "headline": clean(headline, here)}


def leaderboard(report: str) -> list[dict[str, str]]:
    """The rows of the report's one-change leaderboard, by column."""
    lines = report.splitlines()
    i = lines.index(LEADER) + 2
    head = [c.strip() for c in lines[i].strip("|").split("|")]
    rows = []
    for line in lines[i + 2 :]:
        if not line.startswith("|"):
            break
        rows.append(dict(zip(head, (c.strip() for c in line.strip("|").split("|")))))
    return rows


def pct(cell: str) -> str:
    """A cell of a percent and its range, ``-1.2 [-2.0, -0.5]``, as ``-1.2% [-2.0, -0.5]``."""
    value, _, rest = cell.partition(" ")
    return f"{value}% {rest}".strip()


def name(family: str) -> str:
    """How the page names a family: the 2021 reference, or as the fit does."""
    return NAMES.get(family, family)


def num(cell: str) -> float | None:
    """The number a cell starts with."""
    m = re.match(r"[+-]?\d[\d.]*(?:e[+-]?\d+)?", cell or "")
    return float(m.group(0)) if m else None


def winners(rows: list[dict[str, str]]) -> str:
    """The changes whose limited-data gain excludes zero, best first."""
    cols = ("verdict", "multiplier", "(b) Δ 1T vs %", "(a) speedup to vs@1T")
    head = "| change | at the rungs | × per parameter | 1T tokens, Δ % | GPU-hours, × | against |"
    kept = [r for r in rows if _excludes_zero(r["(b) Δ 1T vs %"])]
    kept.sort(key=lambda r: num(r["(b) Δ 1T vs %"]) or 0)
    body = [
        f"| {name(r['variant'])} | {' | '.join(r[c] for c in cols)} | {name(r['vs'])} |"
        for r in kept
    ]
    intro = "Every change whose gain on 1T tokens at equal serving cost excludes zero:"
    return "\n".join([intro, "", head, "|---" * 6 + "|", *body])


def _excludes_zero(cell: str) -> bool:
    """Whether a cell's range, ``-1.2 [-2.0, -0.5]``, lies below zero."""
    m = re.search(r"\[([^,]+),\s*([^\]]+)\]", cell or "")
    return bool(m) and float(m.group(2)) < 0


def axes(rows: list[dict[str, str]]) -> dict[str, str]:
    """A note per axis file: its verdicts, its best changes, and its worse ones."""
    by: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by[r["variant"].split("/")[0]].append(r)
    return {f"axis: {name}": _axis(rs) for name, rs in by.items()}


def _axis(rows: list[dict[str, str]]) -> str:
    """One axis file's facts, from its leaderboard rows."""
    counts = Counter(r["verdict"] for r in rows if r["verdict"])
    said = ", ".join(f"{counts[v]} {v}" for v in VERDICTS if counts[v])
    vs = name(Counter(r["vs"] for r in rows).most_common(1)[0][0])
    best = max(rows, key=lambda r: num(r["multiplier"]) or 0)
    fast = max(rows, key=lambda r: num(r["(a) speedup to vs@1T"]) or 0)
    low = min(rows, key=lambda r: num(r["(b) Δ 1T vs %"]) or 0)
    lines = [
        f"**{len(rows)} arms against {vs}: {said}.**",
        "",
        (
            f"- Per parameter, at the rungs: **{best['variant']}**, ×"
            f" {best['multiplier']}: {pct(best['Δ d512 %']) or '–'} at d512,"
            f" {pct(best['Δ d768 %']) or '–'} at d768."
        ),
        (
            f"- Unlimited data: **{fast['variant']}** reaches its parent's 1T-token"
            f" loss in {fast['(a) speedup to vs@1T']} times fewer GPU-hours."
        ),
        f"- Limited data: **{low['variant']}**, {pct(low['(b) Δ 1T vs %'])} on 1T tokens.",
    ]
    worse = [r["variant"] for r in rows if r["verdict"] == "worse"]
    if worse:
        lines.append(f"- Worse at the rungs: {', '.join(worse)}.")
    return "\n".join(lines)


def summary(rows: list[dict[str, str]]) -> str:
    """A table of each axis file's best change on 1T tokens, and its verdicts."""
    by: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by[r["variant"].split("/")[0]].append(r)
    head = (
        "| axis | best on 1T tokens | Δ % | GPU-hours, × | × per parameter | verdicts |"
    )
    body = []
    for name, rs in by.items():
        low = min(rs, key=lambda r: num(r["(b) Δ 1T vs %"]) or 0)
        counts = Counter(r["verdict"] for r in rs if r["verdict"])
        said = ", ".join(f"{counts[v]} {v}" for v in VERDICTS if counts[v])
        cells = (low["(b) Δ 1T vs %"], low["(a) speedup to vs@1T"], low["multiplier"])
        body.append(
            f"| {name} | {low['variant']} | " + " | ".join(cells) + f" | {said} |"
        )
    return "\n".join([head, "|---" * 6 + "|", *body])


def secondary(report: Path) -> dict[str, str]:
    """The secondary metrics' notes: their report's parts, without its paths to its
    own figures and scripts."""
    lines = report.read_text().splitlines()
    out = {}
    for name, heading in SECONDARY.items():
        if heading in lines:
            kept = [
                line for line in part(lines, heading) if not line.startswith("Re-run")
            ]
            text = re.sub(
                r"\s*\(`(figures|scripts|tables)/[^)]*\)", "", "\n".join(kept)
            )
            out[name] = text.strip()
    return out


def notes(report: Path, tutorial: Path, more: Path | None = None) -> str:
    """The notes, in ``## name`` sections, with the secondary metrics' of `more`."""
    lines = tutorial.read_text().splitlines()
    here = tutorial.parent
    out = opening(lines, here)
    for name, headings in PARTS.items():
        out[name] = "\n\n".join(clean(part(lines, h), here) for h in headings)
    rows = leaderboard(report.read_text())
    out |= {"winners": winners(rows), "axes": summary(rows), **axes(rows)}
    out |= secondary(more) if more else {}
    head = "# The W&B report's notes, written by scripts/timeline/dashboard_notes.py\n"
    return head + "".join(f"\n## {name}\n\n{text}\n" for name, text in out.items())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("report", type=Path, help="fit.py's report.md")
    p.add_argument("out", type=Path, help="the notes to write")
    p.add_argument("--tutorial", type=Path, default=TUTORIAL)
    p.add_argument("--secondary", type=Path, help="the secondary metrics' report.md")
    args = p.parse_args()
    args.out.write_text(notes(args.report, args.tutorial, args.secondary))


if __name__ == "__main__":
    main()
