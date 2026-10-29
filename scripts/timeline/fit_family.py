"""How the timeline's families relate: each one's parent, read from the modules its
file uses; the base arms that repeat another family's runs; and whose measured
training MFU a family without its own takes.
"""

import re
from pathlib import Path
from typing import Any

from fit_counts import Counts, find
from fit_points import ROOT, Run

#: The family a module's model is: an axis file that uses it changes that family.
MODULES = {"gpt3_rope": "positions/rope", "gpt3_2020": ROOT}
MILESTONE = re.compile(r"m\d{4}_\w+")  # a milestone's module, an arm of milestones.bux


def infer_parents(
    families: set[str], folders: tuple[Path, ...], root: str, kind_of: Any = None
) -> dict[str, str]:
    """Each family's parent, from the modules its file uses: one milestone's
    (``use m2026_hybrid``: that milestone), `MODULES`' (``use gpt3_rope``:
    ``positions/rope``), or the baseline, as for milestones.bux, which uses them
    all. A file that uses two (sparse.bux: gpt3_rope and m2026_hybrid) gives each
    arm the one of its kind (`kind_of`: dense, MoE or hybrid), else the first. A
    family that would be its own parent gets `root` (the baseline, if the root is
    another)."""
    out = {}
    for f in families - {root}:
        file = find(f.partition("/")[0], folders)
        options = file_parents(file.read_text(), folders) if file else [ROOT]
        parent = pick(f, options, kind_of) if len(options) > 1 else options[0]
        out[f] = root if parent == f else parent
    return out


def file_parents(
    text: str, folders: tuple[Path, ...] = (), seen: frozenset = frozenset()
) -> list[str]:
    """The families a file's arms may change, from the modules it uses, in order;
    through a module that is no family (factorial_mhc.bux's ``use factorial``), the
    families that module's file names."""
    uses = re.findall(r"^use\s+(\w+)", text, re.MULTILINE)
    milestones = [u for u in uses if MILESTONE.fullmatch(u)]
    if len(milestones) > 1:  # the milestones' own sweep
        return [ROOT]
    found: list[str] = []
    for u in uses:
        if MILESTONE.fullmatch(u):
            found.append(f"milestones/{u}")
        elif u in MODULES:
            found.append(MODULES[u])
        elif u not in seen and (file := find(u, folders)):
            found += [
                f
                for f in file_parents(file.read_text(), folders, seen | {u})
                if f != ROOT
            ]
    return list(dict.fromkeys(found)) or [ROOT]


def pick(f: str, options: list[str], kind_of: Any) -> str:
    """The option of `f`'s kind, if just one is; else the first."""
    if kind_of is None:
        return options[0]
    same = [o for o in options if kind_of(o) == kind_of(f)]
    return same[0] if len(same) == 1 else options[0]


#: Settings of a run that do not change what it computes, left out of "same config".
INERT = {"log", "checkpoint", "mesh", "methods", "observe", "seed", "search"}


def alias_bases(runs: list[Run], counts: Counts, parents: dict, root: str) -> list[Run]:
    """`runs` with every run that repeats another family's as that family's
    replicate, where the two have one architecture and one training at that rung:
    an axis file's ``base`` arm, as the parent of the axis' other arms; and any
    family's run, as its parent (the factorial's all-zeros arm is the hybrid)."""
    for r in runs:
        if r.arms == ("base",):
            axis = [
                v
                for f, v in parents.items()
                if f.startswith(f"{r.stem}/") and f != r.family
            ]
            parent = max(set(axis), key=axis.count) if axis else root
            same = _same_config(counts, r, parent, base=True)
        else:
            parent = parents.get(r.family)
            same = parent is not None and _same_config(counts, r, parent)
        if same and parent is not None:
            stem, _, arm = parent.partition("/")
            r.stem, r.arms, r.replicate = stem, (arm,) if arm else (), True
    return runs


def _same_config(counts: Counts, r: Run, family: str, base: bool = False) -> bool:
    """Whether `r`'s family and `family` compile to one run at `r`'s rung: one
    architecture (its source's labels aside) and one training. A base arm whose
    files are not at hand is matched by its counted architecture alone."""
    stem, _, arm = family.partition("/")
    arms = (arm,) if arm else ()
    mine, theirs = (
        counts.rung(r.stem, r.arms, r.width),
        counts.rung(stem, arms, r.width),
    )
    if mine is None or theirs is None:
        ours, other = counts.at(r.stem, r.arms, r.width), counts.at(stem, arms, r.width)
        return base and bool(ours.get("arch")) and ours.get("arch") == other.get("arch")
    same_arch = _unlabelled(mine.arch.model_dump(mode="json")) == _unlabelled(
        theirs.arch.model_dump(mode="json")
    )
    return same_arch and _train(mine) == _train(theirs)


def _unlabelled(x: Any) -> Any:
    """`x` without its ``label`` keys: the source text a layer came from."""
    if isinstance(x, dict):
        return {k: _unlabelled(v) for k, v in x.items() if k != "label"}
    return [_unlabelled(v) for v in x] if isinstance(x, list) else x


def _train(run: Any) -> dict:
    return {
        k: v for k, v in run.train.model_dump(mode="json").items() if k not in INERT
    }


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
