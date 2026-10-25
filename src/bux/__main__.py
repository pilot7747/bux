"""The ``bux`` command line: check, plan, prepare, train and print .bux files.

Usage::

    bux COMMAND FILE [SETTING=VALUE ...] [--entry NAME] [options]

    check     the model's layers, the run's training and the sweep's axes
    plan      a line a run: parameters, tokens, FLOPs a token, GPUs, bytes a GPU
    prepare   tokenize and pack the documents the runs read
    train     train the runs in this process, one after another
    launch    train what is left of the runs, a sweep's side by side on the GPUs
    render    draw the model as a circuit of the residual stream
    fmt       print the file back, a run at a time, with nothing left to resolve
    json      print the runs' IR (or the model's) as JSON

Settings after the file override the file's own, as if written at its end:
``bux train speedrun.bux optimizer.lr=0.03 'steps = 2k'``. Alternatives there make a
sweep, as in a file: ``'optimizer.lr = 0.01 | 0.03'``.

``bux train`` trains the runs in this process, one after another. ``bux launch`` is
the entrypoint for production (`bux.launch`): it trains each run of a sweep on the
GPUs its mesh needs, or a run alone on many hosts, records what each run did, trains
again only what is not done, and stops at SIGTERM with a checkpoint. ``bux plan``
shows what each run takes: its FLOPs a token, its GPUs, and its bytes on each.
"""

import argparse
import json
import math
import sys
from collections import Counter
from typing import TYPE_CHECKING

from bux.lang import Architecture, BuxError, Report, Run, check, compile_runs
from bux.lang.printer import config_to_source, run_to_source, to_source, value_to_source
from bux.lang.render import render

if TYPE_CHECKING:  # imported where used: jax and the backend are slow to import
    from flax import nnx
    from jax.sharding import AbstractMesh


def _si(n: float) -> str:
    for suffix, scale in (("", 1), ("k", 1e3), ("M", 1e6), ("B", 1e9)):
        if "e" not in (shown := f"{n / scale:.3g}"):
            return shown + suffix
    return f"{n / 1e12:.3g}T"


def _bytes(n: int | None) -> str:
    if n is None:
        return "?"
    for suffix, power in (("B", 0), ("KiB", 10), ("MiB", 20), ("GiB", 30)):
        if "e" not in (shown := f"{n / 2**power:.3g}"):
            return shown + suffix
    return f"{n / 2**40:.3g}TiB"


def summary(arch: Architecture) -> str:
    """Return what ``bux check`` says of a model: its settings and its layers, counted
    by type, the most common first."""
    counts = Counter(layer.config.type for layer in arch.layers())
    head = (
        f"d_model={arch.d_model} vocab_size={arch.vocab_size} layers={counts.total()}"
    )
    if arch.residual is not None:
        head += f" residual={config_to_source(arch.residual)}"
    return head + "\n" + "  ".join(f"{t}×{n}" for t, n in counts.most_common())


def training(run: Run) -> str:
    """Return what ``bux check`` says of a run's training: its steps and tokens, its
    optimizer, schedule and data, and its tokenizer and packing if it has them."""
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


def abstract(
    run: Run, gpus: int | None = None
) -> "tuple[nnx.Module, nnx.Optimizer] | None":
    """Build the run's model and optimizer from shapes alone; None if the backend
    lacks a layer. Under the run's mesh on `gpus` GPUs if it fits, with repeats
    scanned and pipelined as the methods say, on the stages training balances
    (``stages.balanced``), else outside any mesh.

    ``graph_updates=False`` skips flax's sharding of the shapes, which under a mesh
    fails on the masked nodes in Muon's state; ``memory`` shards them itself."""
    import jax.numpy as jnp
    from flax import nnx

    from bux import parallel
    from bux.architecture import Model
    from bux.train.optim import optimizer
    from bux.train.stages import balanced

    t = run.train

    def make() -> tuple[Model, nnx.Optimizer]:
        model = Model(run.arch, rngs=nnx.Rngs(0))
        # The schedule has no state, so a constant one stands in for the run's.
        tx = optimizer(t.optimizer, lambda step: jnp.ones(()), model, t.clip)
        return model, nnx.Optimizer(model, tx, wrt=nnx.Param)

    grid = None if gpus is None else _grid(run, gpus)
    try:
        if grid is None:
            return nnx.eval_shape(make, graph_updates=False)
        with parallel.use(grid, t.layout, balanced(run)):
            return nnx.eval_shape(make, graph_updates=False)
    except NotImplementedError:
        return None


def _grid(run: Run, gpus: int) -> "AbstractMesh | None":
    """Return the run's mesh on `gpus` GPUs, without devices; None if it won't fit."""
    from jax.sharding import AbstractMesh, AxisType

    from bux import parallel

    try:
        sizes = parallel.sizes(run.train.mesh, gpus)
    except ValueError:
        return None
    auto = (AxisType.Auto,) * len(parallel.AXES)
    return AbstractMesh(tuple(sizes.values()), parallel.AXES, axis_types=auto)


def per_device(state: "nnx.State") -> int:
    """Return the bytes of `state` per device, as ``parallel.shardings`` splits it."""
    import jax
    from flax import nnx

    from bux import parallel

    variables = jax.tree.leaves(state, is_leaf=lambda x: isinstance(x, nnx.Variable))
    shardings = jax.tree.leaves(parallel.shardings(state))
    return sum(
        math.prod(sharding.shard_shape(v.shape)) * v.dtype.itemsize
        for v, sharding in zip(variables, shardings, strict=True)
        if hasattr(v.get_value(), "shape")
    )


def memory(
    run: Run, made: "tuple[nnx.Module, nnx.Optimizer]", gpus: int
) -> tuple[int, int] | None:
    """Return the bytes per GPU of the parameters and of the optimizer state in `made`,
    split over `gpus` GPUs by the run's mesh and layout; None if the mesh does not fit.
    """
    from flax import nnx

    from bux import parallel

    t = run.train
    if (grid := _grid(run, gpus)) is None:
        return None
    model, opt = made
    with parallel.use(grid, t.layout, t.methods):
        return per_device(nnx.state(model, nnx.Param)), per_device(nnx.state(opt))


def flops_per_token(run: Run) -> float | None:
    """Return the model FLOPs a training token of `run` takes in rows of one document
    (``trainer.step_flops``); None if the backend lacks a layer."""
    from bux.train.trainer import step_flops

    t = run.train
    try:
        return step_flops(run).one_document(t.seq_len) / (t.batch * t.seq_len)
    except NotImplementedError:
        return None


def _row(run: Run, gpus: int, alone: bool) -> tuple[str, ...]:
    """Return the cells of `run`'s line in the plan, out of `gpus` GPUs."""
    from bux.architecture import num_params
    from bux.launch import need
    from bux.parallel.pipeline import bubble

    t = run.train
    tokens = t.steps * t.batch * t.seq_len
    used = need(t.mesh, gpus, alone=alone)
    made = abstract(run, used)
    n = None if made is None else num_params(made[0])
    param_bytes, opt_bytes = (made and memory(run, made, used)) or (None, None)
    idle = bubble(
        t.mesh.stage, t.methods.microbatches, t.methods.rounds, t.methods.pipeline
    )
    work = flops_per_token(run)
    return (
        run.id,
        _si(n) if n else "?",
        _si(tokens),
        f"{tokens / n:.1f}" if n else "?",
        _si(work) if work else "?",
        str(used),
        _bytes(param_bytes),
        _bytes(opt_bytes),
        f"{idle:.0%}",
        run.name,
    )


def plan(runs: list[Run], gpus: int | None = None) -> str:
    """Return a table with one line per run: its parameters and tokens, its FLOPs a
    token, the GPUs it trains on out of `gpus` (default: the visible ones), the bytes
    of its parameters and of its optimizer state per GPU, and the share of a step its
    pipeline idles."""
    from bux.launch import visible_gpus

    available = len(visible_gpus()) if gpus is None else gpus
    head = (
        *("id", "params", "tokens", "tok/param", "flops/tok"),
        *("gpus", "params/gpu", "optim/gpu", "bubble"),
    )
    alone = len(runs) == 1
    rows = [(*head, "run"), *(_row(run, available, alone) for run in runs)]
    widths = [max(len(row[i]) for row in rows) for i in range(len(head))]
    lines = [
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths)) + "  " + row[-1]
        for row in rows
    ]
    return f"{len(runs)} run{'s' if len(runs) != 1 else ''}\n" + "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    """Return the parser of bux's command line."""
    p = argparse.ArgumentParser(
        prog="bux", description="Compile, inspect and train .bux models and runs."
    )
    p.add_argument(
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
    p.add_argument("file")
    p.add_argument(
        "settings",
        nargs="*",
        metavar="SETTING=VALUE",
        help="statements that redefine the file's: optimizer.lr=1e-3, 'use recipe'",
    )
    p.add_argument(
        "--entry", default="model", help="definition to compile (default: model)"
    )
    p.add_argument(
        "--expand",
        action="store_true",
        help="render: draw blocks inline instead of separately",
    )
    p.add_argument(
        "--out",
        default="runs",
        help="train, launch: where run directories go (default: runs)",
    )
    p.add_argument(
        "--only",
        metavar="TEXT",
        help="train, launch: only the runs whose name contains TEXT",
    )
    p.add_argument(
        "--index",
        type=int,
        help="train, launch: only the run at INDEX, from 0, in the order bux plan lists "
        "them; $SLURM_ARRAY_TASK_ID or $JOB_COMPLETION_INDEX in an array job",
    )
    p.add_argument(
        "--cache",
        default="data/packed",
        help="prepare, train, launch: where prepared datasets go (default: data/packed)",
    )
    p.add_argument(
        "--workers",
        type=int,
        help="prepare, launch: processes that prepare data (default: one a core)",
    )
    return p


def _error(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _check(report: Report, arch: Architecture, runs: list[Run]) -> None:
    print(summary(arch))
    if report.run is not None:
        print(training(report.run))
    if len(runs) > 1:
        axes = " × ".join(f"{a.where} ({' | '.join(a.options)})" for a in report.axes)
        print(f"sweep: {len(runs)} runs over {axes}")


def _prepare(runs: list[Run], cache: str, workers: int | None) -> int:
    from bux.data.pipeline import jobs, prepare

    todo = dict.fromkeys(job for run in runs for job in jobs(run.train))
    if not todo:
        return _error("this file reads no documents to prepare")
    for job in todo:
        print(prepare(job, cache, workers=workers))
    return 0


def _select(runs: list[Run], index: int | None, only: str | None) -> list[Run]:
    if index is not None:
        if not 0 <= index < len(runs):
            s = "s" if len(runs) != 1 else ""
            raise ValueError(f"--index {index}, but the file makes {len(runs)} run{s}")
        runs = runs[index : index + 1]
    if only is not None:
        runs = [run for run in runs if only in run.name]
        if not runs:
            raise ValueError(f"no run's name contains {only!r}")
    return runs


def _train(args: argparse.Namespace, runs: list[Run]) -> int:
    try:
        runs = _select(runs, args.index, args.only)
    except ValueError as err:
        return _error(str(err))
    from bux.runtime import configure

    configure()
    if args.command == "launch":
        from bux.launch import launch

        return launch(runs, args.out, cache=args.cache, workers=args.workers)
    from bux.train.trainer import train

    for run in runs:
        train(run, args.out, cache=args.cache)
    return 0


def _run(
    args: argparse.Namespace, report: Report, arch: Architecture, runs: list[Run]
) -> int:
    match args.command:
        case "check":
            _check(report, arch, runs)
        case "plan" | "train" | "launch" if not runs:
            return _error("this file does not say how to train its model")
        case "plan":
            print(plan(runs))
        case "prepare":
            return _prepare(runs, args.cache, args.workers)
        case "train" | "launch":
            return _train(args, runs)
        case "render":
            print(render(arch, expand=args.expand), end="")
        case "fmt":
            text = "\n".join(map(run_to_source, runs)) if runs else to_source(arch)
            print(text, end="")
        case "json":
            if runs:
                print(json.dumps([r.model_dump(mode="json") for r in runs], indent=2))
            else:
                print(arch.model_dump_json(indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command line, ``bux``, and return its exit code.

    The file is checked first, with every error printed to stderr (exit 1). A file
    with no model is a module: it says what it defines, and exits 0.

    Args:
        argv: The arguments after ``bux``; None for ``sys.argv[1:]``.

    Returns:
        0 when it succeeds, 1 on errors; ``bux launch`` returns its own (see
        `bux.launch.launch`).
    """
    cli = parser()
    args = cli.parse_args(argv)
    for setting in args.settings:
        if "=" not in setting and not setting.startswith("use "):
            cli.error(f"settings are name=value or 'use module', not {setting!r}")
    with open(args.file, encoding="utf-8") as f:
        source = f.read()
    where = {"entry": args.entry, "filename": args.file, "overrides": args.settings}
    report = check(source, **where)
    if report.errors:
        return _error("\n\n".join(map(str, report.errors)))
    if report.arch is None:
        defined = ", ".join(s.name for s in report.statements)
        print(f"a module: it defines {defined or 'nothing'}, and no model")
        return 0
    try:
        runs = compile_runs(source, **where) if report.run is not None else []
    except BuxError as err:
        return _error(str(err))
    return _run(args, report, report.arch, runs)


if __name__ == "__main__":
    sys.exit(main())
