"""scripts/benchmark.py: the suites are the scale they say, a suite trains a worker a
run, and compare prints the table of two results files."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from bux.lang import compile_runs
from bux.launch import need

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "benchmark.py"
TINY = """
d_model    = 32
vocab_size = 64
norm  = rmsnorm
attn  = gqa(num_heads=2, num_kv_heads=2)
ffn   = gated_mlp(intermediate_size=64, activation=swish)
block = [norm attn] [norm ffn]
model = small: embed block^2 norm head | deep: embed block^3 norm head
data      = synthetic
optimizer = adamw(lr=1e-3)
seq_len   = 16
batch     = 8
steps     = 6
log       = log(every=2)
"""


def benchmark(*args: str | Path) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT), *map(str, args)]
    return subprocess.run(
        command, check=False, capture_output=True, text=True, cwd=ROOT
    )


@pytest.mark.parametrize(
    ("suite", "gpus"),
    [("one_gpu", 1), ("one_node", 8), ("two_nodes", 16), ("four_nodes", 32)],
)
def test_every_run_of_a_suite_takes_the_gpus_of_its_scale(suite, gpus):
    path = ROOT / "examples" / "benchmarks" / f"{suite}.bux"
    runs = compile_runs(path.read_text(), filename=str(path))
    assert len(runs) > 2
    assert {need(run.train.mesh, 8, alone=False) for run in runs} == {gpus}


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """A tiny suite's results, trained with this checkout's bux by --tree, and those
    of its first run where it cannot train."""
    folder = tmp_path_factory.mktemp("benchmark")
    suite = folder / "tiny.bux"
    suite.write_text(TINY)
    good, bad = folder / "good.json", folder / "bad.json"
    trained = benchmark("run", suite, "--out", good, "--tree", ROOT)
    assert trained.returncode == 0, trained.stdout + trained.stderr
    failed = benchmark(
        "run", suite, "mesh = mesh(fsdp=16)", "--only", "small", "--out", bad
    )
    assert failed.returncode == 1, failed.stdout + failed.stderr
    return good, bad, trained.stdout


def test_a_suite_trains_its_runs_and_keeps_their_logs(results):
    good, _, said = results
    runs = json.loads(good.read_text())["runs"]
    assert list(runs) == ["tiny[small]", "tiny[deep]"]
    for result in runs.values():
        assert result["steps"] == [2, 4, 6] and len(result["losses"]) == 3
        assert result["tokens_per_s"] > 0 and result["peak_gib"] is None  # a CPU
        assert result["bux"] == str(ROOT / "src" / "bux")
    assert said.count(" tok/s, ") == 2
    folder = good.with_suffix("")  # the runs' directories, and their workers' output
    assert len([*folder.glob("tiny_*/metrics.jsonl")]) == 2
    assert len([*folder.glob("tiny_*.txt")]) == 2


def test_a_run_that_fails_keeps_its_error(results):
    _, bad, _ = results
    [failed] = json.loads(bad.read_text())["runs"].values()
    assert failed["error"].startswith("exit 1: ") and Path(failed["log"]).exists()


def slowed(path: Path, tmp_path: Path) -> Path:
    """Return a copy of the results at `path`, its first run 10% slower and the loss
    of its second 0.1 higher at the last log."""
    written = json.loads(path.read_text())
    small, deep = written["runs"].values()
    small["tokens_per_s"] *= 0.9
    deep["losses"][-1] += 0.1
    copy = tmp_path / "slowed.json"
    copy.write_text(json.dumps(written))
    return copy


def test_compare_prints_a_line_a_run_and_flags_what_changed(results, tmp_path):
    good, bad, _ = results
    same = benchmark("compare", good, good)
    assert same.returncode == 0
    head, *lines = same.stdout.splitlines()
    assert head.split()[:4] == ["run", "old", "tok/s", "new"]
    assert [line.split()[0] for line in lines] == ["tiny[small]", "tiny[deep]"]
    assert all("+0.0%" in line and "+0.0000" in line for line in lines)

    changed = benchmark("compare", good, slowed(good, tmp_path))
    assert changed.returncode == 1
    small, deep = changed.stdout.splitlines()[1:]
    assert "-10.0%" in small and small.endswith("slower")
    assert "+0.1000" in deep and deep.endswith("loss")

    broken = benchmark("compare", good, bad)
    small, deep = broken.stdout.splitlines()[1:]
    assert broken.returncode == 1 and "failed" in small and small.endswith("failed")
    assert deep.split()[2] == "-"  # in the old results alone
