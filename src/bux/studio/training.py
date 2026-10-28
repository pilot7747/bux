"""The training view: what the page shows of a run, as the plan's ``Run``.

`run_view` reads a run's settings from its trace: its budget over the stretches of
its curriculum (`bux.train.curriculum`), its optimizer, its schedule's regions, its
data by the names the file gives its sources, its evals and checkpoints, and every
top-level setting with whether a file writes it. It needs no JAX; `schedule` and
`plan` do, and import it when they are called::

    t = traced(source, {}, path="examples/experiments/smoke.bux")
    run_view(t)["schedule"]["regions"]
    # [{'name': 'warmup', 'start': 0, 'stop': 10}, {'name': 'stable', ...}, ...]
"""

import hashlib
import math
from pathlib import Path
from typing import Any

from bux.lang import ir
from bux.lang.compiler import TRAIN_SETTINGS, Trace, short
from bux.lang.printer import config_to_source, value_to_source
from bux.studio.palette import as_source, si
from bux.train import config as tcfg
from bux.train import curriculum

#: A JSON object, as the API sends it.
type Json = dict[str, Any]

#: The group of each top-level training setting, as the page sorts them.
GROUPS: dict[str, tuple[str, ...]] = {
    "budget": (
        "seq_len",
        "batch",
        "batch_ramp",
        "steps",
        "curriculum",
        "seed",
        "weights",
        "patches_per_row",
    ),
    "optimizer": ("optimizer", "schedule", "clip", "frozen"),
    "data": ("data", "tokenizer", "packing"),
    "loss": ("loss",),
    "eval": ("eval", "checkpoint", "objective"),
    "logging": ("log", "observe", "profile"),
    "precision": ("precision",),
    "parallelism": ("mesh", "layout", "methods", "compiler"),
    "search": ("search",),
}

#: The most steps `schedule` samples the learning rate at.
SAMPLES = 512


def run_view(t: Trace, *, name: str = "run") -> Json:
    """Return the plan's ``Run`` of the run that `t` traced.

    Args:
        t: The trace of a run: its ``train`` is not None.
        name: The run's name, as the sweep names it.

    Returns:
        Its name and id, budget, optimizer, schedule, stretches, phases, batch ramp,
        data, evals, checkpoints and settings.
    """
    train = _train(t)
    parts = curriculum.stretches(train)
    names = _Names(t)
    ramp = train.batch_ramp
    return {
        "name": name,
        "id": ir.Run(name=name, arch=t.arch, train=train).id,
        "budget": budget(train),
        "optimizer": {
            "type": train.optimizer.type,
            "lr": train.optimizer.lr,
            "text": config_to_source(train.optimizer),
        },
        "schedule": schedule_view(train),
        "stretches": [_stretch(s, names) for s in parts],
        "phases": [
            _phase(n, p, train, names) for n, p in enumerate(train.curriculum, 1)
        ],
        "ramp": None
        if ramp is None
        else {"start": ramp.start, "steps": ramp.steps, "multiple": ramp.multiple},
        "data": names.mixture(parts[0].config.data),
        "evals": [{"name": e.name, "every": e.every} for e in tcfg.evals(train)],
        "checkpoint": None
        if train.checkpoint is None
        else {"every": train.checkpoint.every},
        "settings": settings(t),
    }


def _train(t: Trace) -> tcfg.TrainConfig:
    if t.train is None:
        raise ValueError("a model, not a run: it sets no training")
    return t.train


def budget(train: tcfg.TrainConfig) -> Json:
    """Return the run's budget: its steps and their shape, and its tokens over the
    stretches of its curriculum."""
    tokens = curriculum.tokens(curriculum.stretches(train), train.steps)
    return {
        "steps": train.steps,
        "batch": train.batch,
        "seq_len": train.seq_len,
        "tokens": tokens,
        "tokens_text": si(tokens),
        "varies": curriculum.varies(train),
    }


def _stretch(s: curriculum.Stretch, names: "_Names") -> Json:
    c = s.config
    return {
        "start": s.start,
        "stop": s.stop,
        "batch": c.batch,
        "seq_len": c.seq_len,
        "tokens_per_step": c.batch * c.seq_len,
        "data": names.of(c.data),
    }


def _phase(
    n: int, p: tcfg.PhaseConfig, train: tcfg.TrainConfig, names: "_Names"
) -> Json:
    return {
        "n": n,
        "start": p.start(train.steps),
        "seq_len": p.seq_len,
        "batch": p.batch,
        "data": None if p.data is None else names.of(p.data),
    }


# --- Data, by name ---------------------------------------------------------------------


class _Names:
    """Names a run's sources by the definitions that make them: ``web`` for
    ``web = text(...)`` and for ``web(split=[0, 0.99])``; else by their source."""

    def __init__(self, t: Trace):
        self.t = t
        # The definition that each body is, by its span, but for training settings.
        self.bodies = {
            st.body.span: name
            for name, st in t.statements.items()
            if st.params is None and name not in TRAIN_SETTINGS
        }

    def name(self, data: tcfg.DataConfig) -> str | None:
        """Return the definition that names `data`, or None."""
        w = self.t.written(data)
        if w is None:
            return None
        if w.call is not None and w.call.name in self.t.statements:
            return w.call.name
        return self.bodies.get(w.span)

    def of(self, data: tcfg.DataConfig) -> str:
        """Return `data` by name: its own, a mixture's by its sources', else as
        source, ``synthetic`` without its brackets."""
        if (found := self.name(data)) is not None:
            return found
        if isinstance(data, tcfg.MixtureConfig):
            return " + ".join(
                f"{short(w.weight)} * {self.of(w.source)}" for w in data.sources
            )
        return value_to_source(data).removesuffix("()")

    def mixture(self, data: tcfg.DataConfig) -> list[Json]:
        """Return the sources of `data`, each by name, with its weight normalized and
        its source."""
        sources = data.sources if isinstance(data, tcfg.MixtureConfig) else ()
        if not sources:
            return [
                {"name": self.of(data), "weight": 1.0, "text": value_to_source(data)}
            ]
        total = sum(w.weight for w in sources)
        return [
            {
                "name": self.of(w.source),
                "weight": w.weight / total,
                "text": value_to_source(w.source),
            }
            for w in sources
        ]


# --- Settings --------------------------------------------------------------------------


def settings(t: Trace) -> list[Json]:
    """Return every top-level training setting, by group: its value as source,
    whether a file writes it, and the module that does if not this file."""
    train = _train(t)
    size = len(t.program.units[0].text)
    found = []
    for group, fields in GROUPS.items():
        for field in fields:
            st = t.statements.get(field)
            module = None
            if st is not None and st.span.start >= size:
                module = t.program.unit(st.span.start).filename
            found.append(
                {
                    "name": field,
                    "text": as_source(getattr(train, field)),
                    "written": st is not None,
                    "group": group,
                    "origin": None if module is None else _stem(module),
                }
            )
    return found


def _stem(filename: str) -> str:
    return Path(filename).stem


# --- The schedule ----------------------------------------------------------------------


def regions(schedule: tcfg.ScheduleConfig, steps: int) -> list[Json]:
    """Return the named parts of a schedule over `steps` steps, each from its first
    step to the step after its last, as `bux.train.optim.lr_multiplier` runs it.

    ``warmup`` rises to the peak; ``stable`` holds it, or a cosine's end, or follows
    the power law; ``decay`` is a cooldown, a cosine's or a linear decay; ``drop`` a
    multistep's lower level; ``piece 1`` to ``piece n`` a piecewise schedule's pieces.
    Parts without steps are left out.
    """
    warm = min(tcfg.warmup_steps(schedule, steps), steps)
    found = [("warmup", 0, warm), *_after_warmup(schedule, steps, warm)]
    return [
        {"name": n, "start": a, "stop": min(b, steps)}
        for n, a, b in found
        if min(b, steps) > a
    ]


def _after_warmup(
    c: tcfg.ScheduleConfig, steps: int, warm: int
) -> list[tuple[str, int, int]]:
    match c:
        case tcfg.CosineSchedule():
            end = round(c.horizon * steps)
            return [("decay", warm, end), ("stable", end, steps)]
        case tcfg.WSDSchedule() | tcfg.PowerSchedule():
            start = steps - c.cooldown(steps)
            return [("stable", warm, start), ("decay", max(start, warm), steps)]
        case tcfg.LinearDecaySchedule():
            return [("decay", warm, steps)]
        case tcfg.MultiStepSchedule():
            cuts = [warm, *(max(warm, round(a * steps)) for a in c.at), steps]
            names = ["stable", *("drop" for _ in c.at)]
            return list(zip(names, cuts, cuts[1:], strict=False))
        case tcfg.PiecewiseSchedule():
            ends = [round(p.until * steps) for p in c.pieces]
            starts = [warm, *ends[:-1]]
            return [
                (f"piece {k}", a, b)
                for k, (a, b) in enumerate(zip(starts, ends, strict=True), 1)
            ]
    return [("stable", warm, steps)]


def schedule_view(train: tcfg.TrainConfig) -> Json:
    """Return the run's schedule: its type and source, its warmup in steps, its
    regions, and for a cosine the step it reaches its end at (`horizon`)."""
    c, steps = train.schedule, train.steps
    horizon = round(c.horizon * steps) if isinstance(c, tcfg.CosineSchedule) else None
    return {
        "type": c.type,
        "text": config_to_source(c),
        "warmup": tcfg.warmup_steps(c, steps),
        "regions": regions(c, steps),
        "horizon": horizon,
    }


def samples(steps: int, edges: list[int], most: int = SAMPLES) -> list[int]:
    """Return at most `most` steps to draw a schedule at: every step of a short run;
    else the steps either side of each edge and the rest evenly apart."""
    if steps <= most:
        return list(range(steps))
    keep = {0, steps - 1}
    keep |= {s for e in edges for s in (e - 1, e) if 0 <= s < steps}
    room = max(2, most - len(keep))
    keep |= {round(k * (steps - 1) / (room - 1)) for k in range(room)}
    return sorted(keep)[:most]


def key(t: Trace) -> str:
    """Return what `schedule` and `plan` of `t` depend on, to keep them by: the
    run's id, or a hash of a model that is not a run."""
    if t.train is not None:
        return ir.Run(name="run", arch=t.arch, train=t.train).id
    return hashlib.sha256(t.arch.model_dump_json().encode()).hexdigest()[:10]


def schedule(t: Trace) -> Json:
    """Return the run's learning rate at up to `SAMPLES` steps, as the trainer's
    `bux.train.optim.lr_multiplier` makes it, times the optimizer's peak. Imports
    JAX.

    Returns:
        ``points``, ``[step, lr]`` pairs, the ``peak`` and the run's ``steps``.
    """
    import jax
    import jax.numpy as jnp

    from bux.train.optim import lr_multiplier

    train = _train(t)
    found = regions(train.schedule, train.steps)
    steps = samples(train.steps, [r["start"] for r in found])
    multiplier = jax.jit(jax.vmap(lr_multiplier(train)))
    values = multiplier(jnp.asarray(steps, jnp.float32)).tolist()
    peak = train.optimizer.lr
    points = [[s, float(f"{peak * v:.6g}")] for s, v in zip(steps, values, strict=True)]
    return {"points": points, "peak": peak, "steps": train.steps}


def plan(t: Trace) -> Json:
    """Return the parameters of the model, as ``bux plan`` counts them on shapes
    alone, and a run's model FLOPs a training token; None where the backend lacks a
    layer. Imports JAX.

    Returns:
        ``params``, ``params_text`` (``8.03B``) and ``flops_per_token``.
    """
    from bux.__main__ import flops_per_token

    n = _params(t)
    flops = None
    if t.train is not None:
        flops = flops_per_token(ir.Run(name="run", arch=t.arch, train=t.train))
    return {
        "params": n,
        "params_text": None if n is None else si(n),
        "flops_per_token": None if flops is None or math.isnan(flops) else flops,
    }


def _params(t: Trace) -> int | None:
    """Return the parameters of the trace's model, built on shapes alone."""
    from flax import nnx

    from bux.__main__ import abstract
    from bux.architecture import Model, num_params

    if t.train is not None:
        made = abstract(ir.Run(name="run", arch=t.arch, train=t.train))
        return None if made is None else num_params(made[0])
    try:
        model = nnx.eval_shape(
            lambda: Model(t.arch, rngs=nnx.Rngs(0)), graph_updates=False
        )
    except NotImplementedError:
        return None
    return num_params(model)
