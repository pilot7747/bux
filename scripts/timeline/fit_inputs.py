"""Read the timeline's runs: from run directories as bux writes them
(``status.json``, ``run.json``, ``metrics.jsonl``), or from files of RESULT lines
(``RESULT {"dir": ..., "state": ..., <final metrics>}``, one per run).

A run directory's name is its run's name from ``run.json`` (``positions[rope,
d512, seed=1]``), else from ``run.bux``'s first line, else the directory's own.
Its tokens are the last ``tokens`` it logged, and its metrics their last values.
"""

import json
import re
from pathlib import Path

from fit_points import Run, parse_row

#: The logged metrics a fit can read, as the cluster's collect script keeps them.
KEEP = re.compile(r"^(loss|.*_bpb.*|.*_loss|objective)$")


def read(paths: list[Path]) -> list[Run]:
    """Every finished run under `paths`: run directories, folders of them, or
    files of RESULT lines."""
    rows = []
    for path in paths:
        if path.is_dir():
            rows += [run_row(d) for d in run_dirs(path)]
        else:
            rows += result_rows(path)
    return [r for r in map(parse_row, rows) if r is not None]


def run_dirs(path: Path) -> list[Path]:
    """`path` if it is a run's directory, else the run directories in it."""
    if (path / "status.json").exists():
        return [path]
    return sorted(d for d in path.iterdir() if (d / "status.json").exists())


def result_rows(path: Path) -> list[dict]:
    """The rows of a file's RESULT lines."""
    rows = []
    for line in path.read_text().splitlines():
        if (at := line.find("RESULT {")) >= 0:
            rows.append(json.JSONDecoder().raw_decode(line, at + len("RESULT "))[0])
    return rows


def run_row(folder: Path) -> dict:
    """A run directory as a RESULT row: its name, state, lr, tokens and metrics."""
    status = json.loads((folder / "status.json").read_text())
    run = (
        json.loads((folder / "run.json").read_text())
        if (folder / "run.json").exists()
        else {}
    )
    last, tokens = last_metrics(folder / "metrics.jsonl")
    lr = run.get("train", {}).get("optimizer", {}).get("lr")
    row = {**last, "dir": run.get("name") or header_name(folder) or folder.name}
    return row | {"state": status.get("state"), "lr": lr, "budget": tokens}


def last_metrics(path: Path) -> tuple[dict, float | None]:
    """The last value of each metric `KEEP` names in a metrics.jsonl, and the last
    tokens it logged."""
    last: dict = {}
    tokens = None
    for line in path.read_text().splitlines() if path.exists() else ():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # a line cut short while the run wrote it
            continue
        tokens = record.get("tokens", tokens)
        last.update({k: v for k, v in record.items() if KEEP.match(k)})
    return last, tokens


def header_name(folder: Path) -> str | None:
    """The run's name from run.bux's first line, ``# <name>, run <id>``."""
    path = folder / "run.bux"
    first = path.read_text().splitlines()[0] if path.exists() else ""
    m = re.fullmatch(r"#\s*(.+), run [0-9a-f]+\s*", first)
    return m.group(1) if m else None
