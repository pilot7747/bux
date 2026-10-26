"""bux.studio.training: what the page shows of a run's training."""

from pathlib import Path

import pytest

from bux.lang.compiler import Trace, trace
from bux.studio import training
from bux.studio.picks import traced
from bux.studio.training import GROUPS, regions, run_view, samples
from bux.train import config as tcfg

EXAMPLES = Path(__file__).parent.parent / "examples"
MODEL = """\
d_model = 32
vocab_size = 64
model = embed [rmsnorm gqa(num_heads=2, num_kv_heads=1)] rmsnorm head
data = synthetic
seq_len = 8
batch = 4
"""
MIXTURE = """\
use llama3_8b
use fineweb_wsd

d_model = 768
ffn.intermediate_size = 2048
attn.num_heads    = 12
attn.num_kv_heads = 4
model = embed block^12 norm head(tied=true)

math = text(path="data/finemath/*.parquet", bos=true)
long = text(path="data/long-docs/*.parquet", bos=true)
data = 0.6 * web(split=[0, 0.999]) + 0.25 * code + 0.15 * math
batch_ramp = ramp(start=64, steps=1k, multiple=64)
curriculum = [phase(at=0.8, data=0.5 * web + 0.3 * math + 0.2 * code),
              phase(at=0.95, seq_len=8192, batch=64, data=long)]
budget = 10B
"""

SCHEDULES = {
    "constant(warmup=10)": [("warmup", 0, 10), ("stable", 10, 100)],
    "cosine(warmup=10)": [("warmup", 0, 10), ("decay", 10, 100)],
    "cosine(horizon=0.5)": [("decay", 0, 50), ("stable", 50, 100)],
    "cosine(warmup=0.1, horizon=1.25)": [("warmup", 0, 10), ("decay", 10, 100)],
    "wsd(warmup=10, decay=0.2)": [
        ("warmup", 0, 10),
        ("stable", 10, 80),
        ("decay", 80, 100),
    ],
    "linear_decay(warmup=0.1)": [("warmup", 0, 10), ("decay", 10, 100)],
    "multistep(warmup=5, at=[0.8, 0.9])": [
        ("warmup", 0, 5),
        ("stable", 5, 80),
        ("drop", 80, 90),
        ("drop", 90, 100),
    ],
    "power(warmup=10, decay=0.1)": [
        ("warmup", 0, 10),
        ("stable", 10, 90),
        ("decay", 90, 100),
    ],
    "piecewise(warmup=10, pieces=[piece(until=0.5), piece(until=0.75, to=0.1, "
    "shape=cosine), piece(to=0.05)])": [
        ("warmup", 0, 10),
        ("piece 1", 10, 50),
        ("piece 2", 50, 75),
        ("piece 3", 75, 100),
    ],
}


def run_of(schedule: str, steps: int = 100) -> Trace:
    return trace(
        MODEL + f"optimizer = adamw(lr=1e-2)\nsteps = {steps}\nschedule = {schedule}\n"
    )


def named(found: list[dict]) -> list[tuple[str, int, int]]:
    return [(r["name"], r["start"], r["stop"]) for r in found]


@pytest.mark.parametrize("schedule", SCHEDULES, ids=lambda s: s.split("(")[0])
def test_every_schedule_has_its_regions(schedule):
    t = run_of(schedule)
    assert t.train is not None
    assert named(regions(t.train.schedule, t.train.steps)) == SCHEDULES[schedule]
    view = run_view(t)["schedule"]
    assert view["type"] == t.train.schedule.type
    assert view["regions"] == regions(t.train.schedule, 100)
    assert view["warmup"] == tcfg.warmup_steps(t.train.schedule, 100)


def test_a_cosines_horizon_is_a_step():
    t = run_of("cosine(horizon=1.25)")
    assert t.train is not None
    assert training.schedule_view(t.train)["horizon"] == 125


def test_samples_keep_every_edge_and_at_most_their_count():
    found = samples(10_000, [0, 100, 8000])
    assert len(found) <= training.SAMPLES and found == sorted(set(found))
    assert {0, 99, 100, 7999, 8000, 9999} <= set(found)
    assert samples(20, [5]) == list(range(20))


def test_a_run_its_budget_optimizer_and_settings():
    path = EXAMPLES / "experiments/smoke.bux"
    run = run_view(traced(path.read_text(), {}, path=str(path)), name="smoke")
    assert run["name"] == "smoke" and len(run["id"]) == 10
    assert run["budget"] == {
        "steps": 50,
        "batch": 8,
        "seq_len": 64,
        "tokens": 25600,
        "tokens_text": "25.6k",
        "varies": False,
    }
    assert run["optimizer"] == {"type": "adamw", "lr": 3e-3, "text": "adamw(lr=0.003)"}
    assert run["data"] == [{"name": "synthetic", "weight": 1.0, "text": "synthetic()"}]
    assert run["evals"] == [] and run["checkpoint"] is None and run["ramp"] is None
    settings = {s["name"]: s for s in run["settings"]}
    assert set(settings) == set(tcfg.TrainConfig.model_fields)
    assert settings["steps"] == {
        "name": "steps",
        "text": "50",
        "written": True,
        "group": "budget",
        "origin": None,
    }
    assert (
        not settings["mesh"]["written"] and settings["mesh"]["group"] == "parallelism"
    )


def test_every_setting_has_a_group():
    grouped = [f for fields in GROUPS.values() for f in fields]
    assert sorted(grouped) == sorted(tcfg.TrainConfig.model_fields)


def test_a_curriculum_its_stretches_phases_and_ramp():
    path = EXAMPLES / "experiments/curriculum.bux"
    run = run_view(traced(path.read_text(), {}, path=str(path)))
    assert run["budget"]["varies"]
    assert run["ramp"] == {"start": 4, "steps": 20, "multiple": None}
    assert [(p["n"], p["start"], p["seq_len"], p["batch"]) for p in run["phases"]] == [
        (1, 40, 128, 8),
        (2, 60, 256, 4),
    ]
    stretches = run["stretches"]
    assert stretches[0]["start"] == 0 and stretches[-1]["stop"] == 80
    assert stretches[-1]["tokens_per_step"] == 4 * 256
    assert (
        sum((s["stop"] - s["start"]) * s["tokens_per_step"] for s in stretches)
        == (run["budget"]["tokens"])
    )


def test_sources_are_named_by_their_definitions():
    path = EXAMPLES / "experiments/mixture.bux"  # not on disk: use finds by name
    run = run_view(traced(MIXTURE, {}, path=str(path)))
    assert [(d["name"], round(d["weight"], 2)) for d in run["data"]] == [
        ("web", 0.6),
        ("code", 0.25),
        ("math", 0.15),
    ]
    assert run["data"][0]["text"].startswith('text(path="data/fineweb-edu/')
    phases = [p["data"] for p in run["phases"]]
    assert phases == ["0.5 * web + 0.3 * math + 0.2 * code", "long"]
    data = [s["data"] for s in run["stretches"]]
    assert data[0] == "0.6 * web + 0.25 * code + 0.15 * math" and data[-1] == "long"
    assert [e["name"] for e in run["evals"]] == ["eval"]
    assert run["checkpoint"] == {"every": 2000}
    settings = {s["name"]: s for s in run["settings"]}
    assert settings["optimizer"]["origin"] == "fineweb_wsd"
    assert settings["data"]["origin"] is None and settings["data"]["written"]


# --- JAX: the learning rate and the parameters ------------------------------------


@pytest.mark.parametrize("schedule", SCHEDULES, ids=lambda s: s.split("(")[0])
def test_the_learning_rate_is_the_trainers_at_region_edges(schedule):
    from bux.train.optim import lr_multiplier

    t = run_of(schedule, steps=1000)
    assert t.train is not None
    found = training.schedule(t)
    assert found["peak"] == 1e-2 and found["steps"] == 1000
    points = dict(found["points"])
    assert len(points) <= training.SAMPLES
    m = lr_multiplier(t.train)
    for region in regions(t.train.schedule, 1000):
        for step in (region["start"], region["stop"] - 1):
            assert points[step] == pytest.approx(1e-2 * float(m(step)), rel=1e-5)


def test_a_plan_counts_parameters_as_bux_plan_does():
    from bux.__main__ import abstract
    from bux.architecture import num_params
    from bux.lang import compile_runs

    path = EXAMPLES / "experiments/smoke.bux"
    found = training.plan(traced(path.read_text(), {}, path=str(path)))
    [run] = compile_runs(path.read_text(), filename=str(path))
    made = abstract(run)
    assert made is not None and found["params"] == num_params(made[0])
    assert found["params_text"] == training.si(found["params"])
    assert found["flops_per_token"] is None or found["flops_per_token"] > 0
    model = training.plan(trace(MODEL.split("data")[0]))
    assert model["params"] > 0 and model["flops_per_token"] is None
