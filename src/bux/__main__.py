"""bux check|plan|prepare|train|launch|render|fmt|json FILE [SETTING=VALUE ...] [--entry NAME]

Settings after the file redefine the file's own, as a last layer over it would:
``bux train speedrun.bux optimizer.lr=0.03 'steps = 2k'``. Alternatives there make a
sweep, as in a file: ``'optimizer.lr = 0.01 | 0.03'``.

``bux train`` trains the runs in this process, one after another. ``bux launch`` is
the entrypoint for production (``bux.launch``): it trains a sweep a run on the GPUs
its mesh needs, and a run alone on many hosts, records what each run did, trains
again only what is not done, and stops at SIGTERM with a checkpoint. ``bux plan``
says what each run takes: its GPUs, and its bytes on each.
"""

import argparse
import json
import math
import sys
from collections import Counter
from typing import TYPE_CHECKING

from bux.lang import Architecture, BuxError, Run, check, compile_runs
from bux.lang.printer import config_to_source, run_to_source, to_source, value_to_source
from bux.lang.render import render

if TYPE_CHECKING:  # imported where used: jax and the backend are slow to import
    from flax import nnx


def _si(n: float) -> str:
    for suffix, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= scale:
            return f"{n / scale:.3g}{suffix}"
    return f"{n:g}"


def _bytes(n: int | None) -> str:
    if n is None:
        return "?"
    for suffix, power in (("TiB", 40), ("GiB", 30), ("MiB", 20), ("KiB", 10)):
        if n >= 2**power:
            return f"{n / 2**power:.3g}{suffix}"
    return f"{n}B"


def summary(arch: Architecture) -> str:
    counts = Counter(layer.config.type for layer in arch.layers())
    head = (
        f"d_model={arch.d_model} vocab_size={arch.vocab_size} layers={counts.total()}"
    )
    if arch.residual is not None:
        head += f" residual={config_to_source(arch.residual)}"
    return head + "\n" + "  ".join(f"{t}×{n}" for t, n in counts.most_common())


def training(run: Run) -> str:
    t = run.train
    tokens = t.steps * t.batch * t.seq_len
    return (
        f"{t.steps:,} steps × {t.batch} × {t.seq_len} tokens = {_si(tokens)} tokens\n"
        f"optimizer={config_to_source(t.optimizer)} "
        f"schedule={config_to_source(t.schedule)}\n"
        f"data={value_to_source(t.data)}"
    ) + (
        f"\ntokenizer={config_to_source(t.tokenizer)} "
        f"packing={config_to_source(t.packing)}"
        if t.tokenizer is not None
        else ""
    )


def abstract(run: Run) -> "tuple[nnx.Module, nnx.Optimizer] | None":
    """The run's model and its optimizer, from shapes alone, or None if the backend
    lacks a layer."""
    import jax.numpy as jnp
    from flax import nnx

    from bux.architecture import Model
    from bux.train.optim import optimizer

    t = run.train

    def make() -> tuple[Model, nnx.Optimizer]:
        model = Model(run.arch, rngs=nnx.Rngs(0))
        # The schedule shapes no state: a constant one, whatever the run's says.
        tx = optimizer(t.optimizer, lambda step: jnp.ones(()), model, t.clip)
        return model, nnx.Optimizer(model, tx, wrt=nnx.Param)

    try:
        return nnx.eval_shape(make)
    except NotImplementedError:
        return None


def per_device(state: "nnx.State") -> int:
    """The bytes of `state` on each device, as ``parallel.shardings`` splits it."""
    import jax
    from flax import nnx

    from bux import parallel

    variables = jax.tree.leaves(state, is_leaf=lambda x: isinstance(x, nnx.Variable))
    splits = jax.tree.leaves(parallel.shardings(state))
    return sum(
        math.prod(split.shard_shape(v.shape)) * v.dtype.itemsize
        for v, split in zip(variables, splits, strict=True)
        if hasattr(v.get_value(), "shape")
    )


def memory(
    run: Run, made: "tuple[nnx.Module, nnx.Optimizer]", gpus: int
) -> tuple[int, int] | None:
    """The bytes of the parameters and of the optimizer's state that `run` `made` on
    each of `gpus` GPUs, as its mesh and layout split them; None if its mesh does not
    fit them."""
    from flax import nnx
    from jax.sharding import AbstractMesh, AxisType

    from bux import parallel

    t = run.train
    try:
        sizes = parallel.sizes(t.mesh, gpus)
    except ValueError:
        return None
    auto = (AxisType.Auto,) * len(parallel.AXES)
    grid = AbstractMesh(tuple(sizes.values()), parallel.AXES, axis_types=auto)
    model, opt = made
    with parallel.use(grid, t.layout, t.methods):
        return per_device(nnx.state(model, nnx.Param)), per_device(nnx.state(opt))


def _row(run: Run, gpus: int, alone: bool) -> tuple[str, ...]:
    """What `plan` says of `run`, of `gpus` there are."""
    from bux.architecture import num_params
    from bux.launch import need

    t = run.train
    tokens = t.steps * t.batch * t.seq_len
    made = abstract(run)
    n = None if made is None else num_params(made[0])
    used = need(t.mesh, gpus, alone=alone)
    params, opt = (made and memory(run, made, used)) or (None, None)
    return (
        run.id,
        _si(n) if n else "?",
        _si(tokens),
        f"{tokens / n:.1f}" if n else "?",
        str(used),
        _bytes(params),
        _bytes(opt),
        run.name,
    )


def plan(runs: list[Run], gpus: int | None = None) -> str:
    """A line a run: its parameters and tokens, the GPUs it trains on of `gpus`
    (default: the visible ones), and the bytes of its parameters and of its
    optimizer's state on each."""
    from bux.launch import visible_gpus

    there = len(visible_gpus()) if gpus is None else gpus
    head = ("id", "params", "tokens", "tok/param", "gpus", "params/gpu", "optim/gpu")
    rows = [(*head, "run"), *(_row(run, there, len(runs) == 1) for run in runs)]
    widths = [max(len(row[i]) for row in rows) for i in range(len(head))]
    lines = [
        "  ".join(c.ljust(w) for c, w in zip(row, widths)) + "  " + row[-1]
        for row in rows
    ]
    return f"{len(runs)} run{'s' if len(runs) != 1 else ''}\n" + "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bux", description="Compile, inspect and train .bux models and runs."
    )
    parser.add_argument(
        "command",
        choices=[
            "check",
            "plan",
            "prepare",
            "train",
            "launch",
            "render",
            "fmt",
            "json",
        ],
    )
    parser.add_argument("file")
    parser.add_argument(
        "settings",
        nargs="*",
        metavar="SETTING=VALUE",
        help="statements that redefine the file's: optimizer.lr=1e-3, 'use recipe'",
    )
    parser.add_argument(
        "--entry", default="model", help="definition to compile (default: model)"
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="render: draw blocks inline instead of separately",
    )
    parser.add_argument(
        "--out",
        default="runs",
        help="train, launch: where run directories go (default: runs)",
    )
    parser.add_argument(
        "--only",
        metavar="TEXT",
        help="train, launch: only the runs whose name contains TEXT",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="train, launch: only the run at INDEX, from 0, in the order bux plan lists "
        "them; $SLURM_ARRAY_TASK_ID or $JOB_COMPLETION_INDEX in an array job",
    )
    parser.add_argument(
        "--cache",
        default="data/packed",
        help="prepare, train, launch: where prepared datasets go (default: data/packed)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="prepare, launch: processes that prepare data (default: one a core)",
    )
    args = parser.parse_args(argv)
    for setting in args.settings:
        if "=" not in setting and not setting.startswith("use "):
            parser.error(f"settings are name=value or 'use module', not {setting!r}")
    with open(args.file, encoding="utf-8") as f:
        source = f.read()
    where = {"entry": args.entry, "filename": args.file, "overrides": args.settings}
    report = check(source, **where)
    if report.errors:
        print("\n\n".join(map(str, report.errors)), file=sys.stderr)
        return 1
    if report.arch is None:
        defined = ", ".join(s.name for s in report.statements)
        print(f"a module: it defines {defined or 'nothing'}, and no model")
        return 0
    arch = report.arch
    try:
        runs = compile_runs(source, **where) if report.run is not None else []
    except BuxError as err:
        print(err, file=sys.stderr)
        return 1
    match args.command:
        case "check":
            print(summary(arch))
            if report.run is not None:
                print(training(report.run))
            if len(runs) > 1:
                axes = " × ".join(
                    f"{a.where} ({' | '.join(a.options)})" for a in report.axes
                )
                print(f"sweep: {len(runs)} runs over {axes}")
        case "plan":
            if not runs:
                print("this file does not say how to train its model", file=sys.stderr)
                return 1
            print(plan(runs))
        case "prepare":
            from bux.data.pipeline import jobs, prepare

            todo = dict.fromkeys(job for run in runs for job in jobs(run.train))
            if not todo:
                print("this file reads no documents to prepare", file=sys.stderr)
                return 1
            for job in todo:
                print(prepare(job, args.cache, workers=args.workers))
        case "train" | "launch":
            if not runs:
                print("this file does not say how to train its model", file=sys.stderr)
                return 1
            if args.index is not None:
                if not 0 <= args.index < len(runs):
                    s = "s" if len(runs) != 1 else ""
                    print(
                        f"--index {args.index}, but the file makes {len(runs)} run{s}",
                        file=sys.stderr,
                    )
                    return 1
                runs = runs[args.index : args.index + 1]
            if args.only is not None:
                runs = [run for run in runs if args.only in run.name]
                if not runs:
                    print(f"no run's name contains {args.only!r}", file=sys.stderr)
                    return 1
            if args.command == "launch":
                from bux.launch import launch

                return launch(runs, args.out, cache=args.cache, workers=args.workers)
            from bux.train.trainer import train

            for run in runs:
                train(run, args.out, cache=args.cache)
        case "render":
            print(render(arch, expand=args.expand), end="")
        case "fmt":
            if runs:
                print("\n".join(map(run_to_source, runs)), end="")
            else:
                print(to_source(arch), end="")
        case "json":
            if runs:
                print(json.dumps([r.model_dump(mode="json") for r in runs], indent=2))
            else:
                print(arch.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
