"""Write the training and sweep panels' fixtures, ``dev/fixtures/run-*.json`` and
``dev/fixtures/sweep-*.json``, from real files: what ``/api/view`` and
``/api/schedule`` would answer, near enough for a harness. Needs no JAX: the learning
rates are plain Python, close to `bux.train.optim`'s schedules::

    uv run python src/bux/studio/static/dev/panels_fixtures.py
"""

import json
import math
import re
from pathlib import Path

from bux.lang import check, compile_spaces
from bux.lang.compiler import Axis, Dimension, Report, Space
from bux.lang.printer import config_to_source, value_to_source
from bux.train import config as cfg
from bux.train import curriculum
from bux.train.config import TrainConfig

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).parent / "fixtures"
EXPERIMENTS = ROOT / "examples" / "experiments"

# A run whose data changes between phases, for the data bands and the mixture bar.
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
schedule = wsd(warmup=0.02, decay=0.2, shape=sqrt)
budget = 10B
eval = [eval(name="web", data=web(split=[0.999, 1]), every=1k),
        eval(name="code", data=code(split=[0.999, 1]), every=2k)]
"""

GROUPS = {
    "budget": ("seq_len", "batch", "batch_ramp", "steps", "curriculum", "seed"),
    "optimizer": ("optimizer", "schedule", "clip"),
    "data": ("data", "tokenizer", "packing"),
    "loss": ("loss",),
    "eval": ("eval", "checkpoint", "objective"),
    "logging": ("log", "observe", "profile"),
    "precision": ("precision",),
    "parallelism": ("mesh", "layout", "methods", "compiler"),
    "search": ("search",),
}


# --- Learning rates ---------------------------------------------------------------


def shaped(shape: str, t: float) -> float:
    """Return a cooldown's curve at `t`, from 1 at 0 to 0 at 1."""
    match shape:
        case "sqrt":
            return 1 - math.sqrt(t)
        case "cosine":
            return 0.5 * (1 + math.cos(math.pi * t))
        case "exp":
            return (math.exp(-t) - math.exp(-1)) / (1 - math.exp(-1))
        case "constant":
            return 0.0 if t > 0 else 1.0
    return 1 - t


def toward(start: float, end: float, shape: str, t: float) -> float:
    """Return the multiplier `t` of the way from `start` to `end` along `shape`."""
    if shape == "geometric":
        return start * (end / start) ** t
    return end + (start - end) * shaped(shape, t)


def through(step: int, start: int, length: int) -> float:
    """Return how far `step` is through the `length` steps from `start`."""
    return min(max((step + 1 - start) / max(length, 1), 0.0), 1.0)


def curve(c: cfg.ScheduleConfig, t: TrainConfig, step: int) -> float:
    """Return the schedule's multiplier at `step`, before the warmup."""
    steps, warm = t.steps, cfg.warmup_steps(c, t.steps)
    match c:
        case cfg.CosineSchedule():
            span = round(c.horizon * steps)
            x = min(max((step + 1 - warm) / max(span - warm, 1), 0), 1)
            return c.final + (1 - c.final) * 0.5 * (1 + math.cos(math.pi * x))
        case cfg.WSDSchedule() | cfg.PowerSchedule():
            cool = c.cooldown(steps)
            return toward(1.0, c.final, c.shape, through(step, steps - cool, cool))
        case cfg.LinearDecaySchedule():
            return toward(1.0, c.final, "linear", through(step, warm, steps - warm))
        case cfg.MultiStepSchedule():
            return c.factor ** sum(step >= round(a * steps) for a in c.at)
        case cfg.PiecewiseSchedule():
            return piecewise(c, t, step)
    return 1.0


def piecewise(c: cfg.PiecewiseSchedule, t: TrainConfig, step: int) -> float:
    """Return a piecewise schedule's multiplier at `step`."""
    ends = [round(p.until * t.steps) for p in c.pieces]
    starts = [cfg.warmup_steps(c, t.steps), *ends[:-1]]
    values = cfg.levels(c.pieces)
    k = max([0] + [i for i, b in enumerate(starts) if step >= b])
    p, b, e = c.pieces[k], starts[k], ends[k]
    return toward(values[k], values[k + 1], p.shape, through(step, b, e - b))


def lr(t: TrainConfig, step: int) -> float:
    """Return the learning rate at `step`."""
    warm = cfg.warmup_steps(t.schedule, t.steps)
    w = min((step + 1) / warm, 1.0) if warm else 1.0
    return t.optimizer.lr * w * curve(t.schedule, t, step)


def sampled(t: TrainConfig, corners: list[int]) -> list[int]:
    """Return the steps to draw: every one of a short run, else about 600 of them and
    the steps either side of each corner."""
    every = max(1, t.steps // 600)
    found = set(range(0, t.steps, every)) | {t.steps - 1}
    found |= {s + d for s in corners for d in (-1, 0, 1) if 0 <= s + d < t.steps}
    return sorted(found)


def schedule(t: TrainConfig) -> dict:
    """Return what ``/api/schedule`` answers for `t`."""
    corners = [r["start"] for r in regions(t)]
    points = [[s, float(f"{lr(t, s):.4g}")] for s in sampled(t, corners)]
    return {"points": points, "peak": t.optimizer.lr, "steps": t.steps}


def regions(t: TrainConfig) -> list[dict]:
    """Return the named stretches of the schedule: warmup, stable, decay, ..."""
    c, steps = t.schedule, t.steps
    warm = cfg.warmup_steps(c, steps)
    found = [("warmup", 0, warm)] if warm else []
    match c:
        case cfg.CosineSchedule():
            end = min(round(c.horizon * steps), steps)
            found += [("cosine", warm, end), ("hold", end, steps)]
        case cfg.WSDSchedule():
            start = steps - c.cooldown(steps)
            found += [("stable", warm, start), ("decay", start, steps)]
        case cfg.PowerSchedule():
            start = steps - c.cooldown(steps)
            found += [("power law", warm, start), ("decay", start, steps)]
        case cfg.LinearDecaySchedule():
            found += [("decay", warm, steps)]
        case cfg.MultiStepSchedule():
            cuts = [warm, *(round(a * steps) for a in c.at), steps]
            names = ["peak", *(f"× {c.factor:g}" for _ in c.at)]
            found += list(zip(names, cuts, cuts[1:], strict=False))
        case cfg.PiecewiseSchedule():
            found += pieces(c, steps, warm)
        case _:
            found += [("peak", warm, steps)]
    return [{"name": n, "start": a, "stop": b} for n, a, b in found if b > a]


def pieces(c: cfg.PiecewiseSchedule, steps: int, warm: int) -> list[tuple]:
    """Return a piecewise schedule's pieces as regions: hold, decay or drop."""
    ends = [round(p.until * steps) for p in c.pieces]
    starts = [warm, *ends[:-1]]
    names = [
        "hold" if p.to is None else "drop" if p.shape == "constant" else "decay"
        for p in c.pieces
    ]
    return list(zip(names, starts, ends, strict=True))


# --- The run ----------------------------------------------------------------------


def si(n: float) -> str:
    """Return `n` as bux check writes it: 3.07B."""
    for suffix, scale in (("", 1), ("k", 1e3), ("M", 1e6), ("B", 1e9)):
        if "e" not in (shown := f"{n / scale:.3g}"):
            return shown + suffix
    return f"{n / 1e12:.3g}T"


def named(data: cfg.DataConfig, report: Report) -> str:
    """Return `data` as source, its sources by the names they are defined under."""
    text = value_to_source(data)
    for name, value in report.values.items():
        if isinstance(value, cfg.TextConfig):
            path = re.escape(value.path)
            text = re.sub(rf'text\(path="{path}"[^()]*\)', name, text)
    return text.replace(", split=[0.0, 0.999]", "")


def mixture(data: cfg.DataConfig, report: Report) -> list[dict]:
    """Return the sources of `data` with their weights, as the Run's `data`."""
    if not isinstance(data, cfg.MixtureConfig):
        text = value_to_source(data)
        return [{"name": named(data, report), "weight": 1.0, "text": text}]
    total = sum(w.weight for w in data.sources)
    return [
        {
            "name": named(w.source, report),
            "weight": w.weight / total,
            "text": value_to_source(w.source),
        }
        for w in data.sources
    ]


def budget(t: TrainConfig) -> dict:
    """Return the Run's budget."""
    tokens = curriculum.tokens(curriculum.stretches(t), t.steps)
    return {
        "steps": t.steps,
        "batch": t.batch,
        "seq_len": t.seq_len,
        "tokens": tokens,
        "tokens_text": si(tokens),
        "varies": curriculum.varies(t),
    }


def stretches(t: TrainConfig, report: Report) -> list[dict]:
    """Return the Run's stretches."""
    return [
        {
            "start": s.start,
            "stop": s.stop,
            "batch": s.config.batch,
            "seq_len": s.config.seq_len,
            "tokens_per_step": s.config.batch * s.config.seq_len,
            "data": named(s.config.data, report),
        }
        for s in curriculum.stretches(t)
    ]


def phases(t: TrainConfig, report: Report) -> list[dict]:
    """Return the Run's phases."""
    return [
        {
            "n": n,
            "start": p.start(t.steps),
            "seq_len": p.seq_len,
            "batch": p.batch,
            "data": None if p.data is None else named(p.data, report),
        }
        for n, p in enumerate(t.curriculum, 1)
    ]


def settings(t: TrainConfig, report: Report) -> list[dict]:
    """Return every top-level setting: its value, whether a file writes it, its group."""
    written = {s.name.split(".")[0] for s in report.statements} | set(report.inherited)
    return [
        {
            "name": name,
            "text": value_to_source(getattr(t, name)),
            "written": name in written,
            "group": group,
        }
        for group, names in GROUPS.items()
        for name in names
    ]


def run(space: Space, report: Report) -> dict:
    """Return the Run of `space`, as the View has it."""
    t = space.run.train
    ramp = t.batch_ramp
    return {
        "name": space.run.name,
        "id": space.run.id,
        "budget": budget(t),
        "optimizer": {
            "type": t.optimizer.type,
            "lr": t.optimizer.lr,
            "text": config_to_source(t.optimizer),
        },
        "schedule": {
            "type": t.schedule.type,
            "text": config_to_source(t.schedule),
            "warmup": cfg.warmup_steps(t.schedule, t.steps),
            "regions": regions(t),
        },
        "stretches": stretches(t, report),
        "phases": phases(t, report),
        "ramp": None
        if ramp is None
        else {"start": ramp.start, "steps": ramp.steps, "multiple": ramp.multiple},
        "data": mixture(curriculum.stretches(t)[0].config.data, report),
        "evals": [{"name": e.name, "every": e.every} for e in cfg.evals(t)],
        "checkpoint": None if t.checkpoint is None else {"every": t.checkpoint.every},
        "settings": settings(t, report),
    }


# --- The sweep --------------------------------------------------------------------


def alternatives(text: str) -> list[str]:
    """Split `a | b | c` at its top-level bars."""
    found, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        depth += (ch in "([") - (ch in ")]")
        if ch == "|" and depth == 0:
            found.append(text[start:i])
            start = i + 1
    return [" ".join(a.split()) for a in [*found, text[start:]]]


def options(axis: Axis, report: Report) -> list[str]:
    """Return an axis's alternatives as source, labeled ones as ``mqa: 1``."""
    if not axis.labeled:
        span = axis.key
        assert not isinstance(span, frozenset)
        return alternatives(report.source[span.start : span.end])
    begin = report.source.rindex("=", 0, axis.span.start) + 1
    return alternatives(report.source[begin : axis.span.end])


def span(start: int, end: int, size: int) -> list[int] | None:
    """Return a span in the file, or None for one in a module."""
    return [start, end] if end <= size else None


def picks(space: Space, axes: list[Axis]) -> dict[str, int]:
    """Return the option `space` takes on each axis, by index."""
    found = {}
    for i, axis in enumerate(axes):
        took = space.choices.get(axis.key, 0)
        found[f"a{i}"] = axis.options.index(took) if isinstance(took, str) else took
    return found


def dimension(d: Dimension, size: int) -> dict:
    """Return a dimension of the search, as the Sweep has it."""
    s = d.space
    choice = isinstance(s, cfg.ChoiceConfig)
    return {
        "where": d.where,
        "type": s.type,
        "low": None if choice else s.low,
        "high": None if choice else s.high,
        "options": list(s.options) if choice else None,
        "log": isinstance(s, cfg.LogUniformConfig) or getattr(s, "log", False),
        "span": span(d.key.start, d.key.end, size),
    }


def search(t: TrainConfig) -> dict:
    """Return the search's method, trials and halving rungs."""
    s, h = t.search, t.search.halving
    halving = None
    if h is not None:
        rungs = [h.steps(t.steps, r) for r in range(h.rungs)]
        halving = f"rungs of {', '.join(map(si, rungs))} steps, eta {h.eta}"
    return {"method": s.method.type, "trials": s.trials, "halving": halving}


def sweep(spaces: list[Space], report: Report, size: int, pick: int) -> dict:
    """Return the Sweep of a file, its run `pick` picked."""
    axes = report.axes
    chosen = picks(spaces[pick], axes) if spaces else {}
    return {
        "axes": [
            {
                "id": f"a{i}",
                "where": a.where,
                "options": options(a, report),
                "labeled": a.labeled,
                "span": span(a.span.start, a.span.end, size),
                "pick": chosen.get(f"a{i}", 0),
            }
            for i, a in enumerate(axes)
        ],
        "runs": [
            {"name": s.run.name, "id": s.run.id, "picks": picks(s, axes)}
            for s in spaces[:256]
        ],
        "total": len(spaces),
        "truncated": len(spaces) > 256,
        "dimensions": [dimension(d, size) for d in report.dimensions],
        "search": search(spaces[0].run.train) if report.dimensions else None,
    }


# --- Files ------------------------------------------------------------------------


def write(name: str, value: dict) -> None:
    """Write one fixture."""
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    print(path.relative_to(ROOT))


def fixture(
    name: str,
    source: str,
    filename: str,
    *,
    overrides: tuple[str, ...] = (),
    runs: range = range(1),
    swept: bool = True,
) -> None:
    """Write the run fixtures of `runs` and the sweep fixture of one file."""
    report = check(source, filename=filename, overrides=list(overrides))
    if report.errors:
        raise SystemExit("\n".join(map(str, report.errors)))
    spaces = compile_spaces(source, filename=filename, overrides=list(overrides))
    if swept:
        write(f"sweep-{name}", sweep(spaces, report, len(source), 0))
    for k in runs:
        train = spaces[k].run.train
        value = {"run": run(spaces[k], report), "schedule": schedule(train)}
        write(f"run-{name}" + (f"-{k}" if k else ""), value)


def example(name: str, runs: range = range(1)) -> None:
    """Write the fixtures of examples/experiments/NAME.bux."""
    path = EXPERIMENTS / f"{name}.bux"
    fixture(name, path.read_text(encoding="utf-8"), str(path), runs=runs)


def main() -> None:
    """Write every fixture."""
    OUT.mkdir(exist_ok=True)
    for name in ("smoke", "curriculum", "attention_ablation", "attention_search"):
        example(name)
    example("schedule_sweep", runs=range(6))
    example("width_sweep")
    fixture("mixture", MIXTURE, str(EXPERIMENTS / "mixture.bux"))
    grid = ("optimizer.lr = 1e-3 | 3e-3", "attn.num_kv_heads = 4 | 2 | 1")
    grid += ("schedule = wsd(warmup=10) | cosine(warmup=10)",)
    smoke = EXPERIMENTS / "smoke.bux"
    fixture("grid", smoke.read_text(encoding="utf-8"), str(smoke), overrides=grid)
    power = ("schedule = power(warmup=500, decay=0.1)",)
    path = EXPERIMENTS / "attention_ablation.bux"
    source = path.read_text(encoding="utf-8")
    fixture("power", source, str(path), overrides=power, swept=False)
    wide = ("ffn.activation = choice(swish, gelu)",)
    wide += ("schedule.warmup = int_uniform(100, 2k, log=true)",)
    path = EXPERIMENTS / "attention_search.bux"
    source = path.read_text(encoding="utf-8")
    fixture("search_wide", source, str(path), overrides=wide, runs=range(0))


if __name__ == "__main__":
    main()
