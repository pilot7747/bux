"""How the timeline's families relate: each one's parent, read from the modules its
file uses; the base arms that repeat another family's runs; and whose measured
training MFU a family without its own takes.
"""

import re
from pathlib import Path

from fit_counts import Counts, find
from fit_points import ROOT, Run

#: The family a module's model is: an axis file that uses it changes that family.
MODULES = {"gpt3_rope": "positions/rope", "gpt3_2020": ROOT}
MILESTONE = re.compile(r"m\d{4}_\w+")  # a milestone's module, an arm of milestones.bux


def infer_parents(
    families: set[str], folders: tuple[Path, ...], root: str
) -> dict[str, str]:
    """Each family's parent, from the modules its file uses: one milestone's
    (``use m2026_hybrid``: that milestone), else `MODULES`' (``use gpt3_rope``:
    ``positions/rope``), else the root, as for milestones.bux, which uses them all."""
    out = {}
    for f in families - {root}:
        file = find(f.partition("/")[0], folders)
        parent = file_parent(file.read_text(), root) if file else root
        out[f] = root if parent == f else parent
    return out


def file_parent(text: str, root: str) -> str:
    """The family a file's arms change, from the modules it uses."""
    uses = re.findall(r"^use\s+(\w+)", text, re.MULTILINE)
    milestones = [u for u in uses if MILESTONE.fullmatch(u)]
    if len(milestones) == 1:
        return f"milestones/{milestones[0]}"
    return next((MODULES[u] for u in uses if u in MODULES), root)


def alias_bases(runs: list[Run], counts: Counts, parents: dict, root: str) -> list[Run]:
    """`runs` with each axis file's ``base`` arm as the run it repeats: the parent
    of the axis' other arms, where the two have one architecture at that rung."""
    for r in runs:
        if r.arms != ("base",):
            continue
        axis = [
            v
            for f, v in parents.items()
            if f.startswith(f"{r.stem}/") and f != r.family
        ]
        parent = max(set(axis), key=axis.count) if axis else root
        stem, _, arm = parent.partition("/")
        arms = (arm,) if arm else ()
        mine, theirs = (
            counts.at(r.stem, r.arms, r.width),
            counts.at(stem, arms, r.width),
        )
        if mine.get("arch") and mine.get("arch") == theirs.get("arch"):
            r.stem, r.arms, r.replicate = stem, arms, True
    return runs


def kind(counts: dict) -> str:
    """``moe``, ``hybrid`` (a linear-time mixer) or ``dense``."""
    return "moe" if counts.get("moe") else "hybrid" if counts.get("hybrid") else "dense"


def mfu_source(
    f: str, measured: dict, parents: dict, kinds: dict
) -> tuple[dict | None, str]:
    """The measured MFU family `f` takes, and whose it is: its own; for an MoE or a
    hybrid, that of a measured family of its file and kind (``kin``); else its
    nearest measured ancestor's; else none (the placeholder)."""
    if f in measured:
        return measured[f], "measured"
    stem = f.partition("/")[0]
    if kinds.get(f) in ("moe", "hybrid"):
        kin = sorted(
            g
            for g in measured
            if g.partition("/")[0] == stem and kinds.get(g) == kinds[f]
        )
        if kin:
            return measured[kin[0]], f"kin {kin[0]}"
    seen, g = {f}, parents.get(f)
    while g and g not in seen:
        if g in measured:
            return measured[g], f"ancestor {g}"
        seen.add(g)
        g = parents.get(g)
    return None, "placeholder"
