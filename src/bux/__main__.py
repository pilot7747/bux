"""The ``bux`` command line: check, plan, prepare, train and print .bux files.

Usage::

    bux COMMAND FILE [SETTING=VALUE ...] [--entry NAME] [options]

    check     the model's layers, the run's training and the sweep's axes
    plan      a line per run: parameters, tokens, FLOPs per token, GPUs, bytes per GPU
    prepare   tokenize and pack the documents the runs read
    train     train the runs in this process, one after another
    launch    train what is left of the runs, a sweep's side by side on the GPUs
    results   the runs' states and objectives, a search's trials the best first
    render    draw the model as a circuit of the residual stream
    fmt       print the file back, a run at a time, with nothing left to resolve
    json      print the runs' IR (or the model's) as JSON

    bux export RUN_DIR OUT [--step N] [--dtype DTYPE] [--check]
    bux export FILE.bux OUT [SETTING=VALUE ...] [--entry NAME]
    bux studio FILE [--port PORT] [--host HOST] [--no-browser]

Settings after the file override the file's own, as if written at its end:
``bux train speedrun.bux optimizer.lr=0.03 'steps = 2k'``. Alternatives there make a
sweep, as in a file: ``'optimizer.lr = 0.01 | 0.03'``.

``bux train`` trains the runs in this process, one after another. ``bux launch`` is
the entrypoint for production (`bux.launch`): it trains each run of a sweep on the
GPUs its mesh needs, or a run alone on many hosts, records what each run did, trains
again only what is not done, and stops at SIGTERM with a checkpoint. ``bux plan``
shows what each run takes: its FLOPs a token, its GPUs, and its bytes on each.

A file with dimensions, ``optimizer.lr = log_uniform(1e-4, 1e-2)``, is a search
(`bux.search`): ``bux train`` and ``bux launch`` train its trials, a run each, and
``bux results`` prints them, the best first.

``bux export`` takes a trained run's directory and writes its model as a transformers
model directory (`bux.export.runs`). Given a ``.bux`` file, it writes the file's model
without weights, which vLLM serves on random ones (``--load-format dummy``): to time
an architecture before training it. ``bux studio`` serves a page that draws a file's
model, training and sweep, and edits them (`bux.studio`).
"""

import argparse
import json
import math
import sys
from collections import Counter
from typing import TYPE_CHECKING

from bux.lang import Architecture, BuxError, Report, Run, Space, check, compile_spaces
from bux.lang.compiler import Dimension, short
from bux.lang.printer import config_to_source, run_to_source, to_source, value_to_source
from bux.lang.render import render
from bux.train import config as cfg
from bux.train import curriculum
from bux.train.config import TrainConfig

if TYPE_CHECKING:  # imported where used: jax and the backend are slow to import
    from flax import nnx
    from jax.sharding import AbstractMesh


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
    if arch.init is not None:
        head += f" init={config_to_source(arch.init)}"
    return head + "\n" + "  ".join(f"{t}×{n}" for t, n in counts.most_common())


def training(run: Run) -> str:
    """Return what ``bux check`` says of a run's training: its steps and tokens, its
    phases, its optimizer, schedule and data, and its tokenizer and packing if it has
    them."""
    t = run.train
    phases = "".join(f"{line}\n" for line in curriculum.lines(t))
    return (
        f"{curriculum.summary(t)}\n{phases}"
        f"optimizer={config_to_source(t.optimizer)} "
        f"schedule={config_to_source(t.schedule)}\n"
        f"data={value_to_source(t.data)}"
    ) + (
        f"\ntokenizer={config_to_source(t.tokenizer)} "
        f"packing={config_to_source(t.packing)}"
        if t.tokenizer is not None
        else ""
    )


def dimension(dim: Dimension) -> str:
    """Return a dimension as ``bux check`` shows it, its bounds in order:
    ``optimizer.lr (log_uniform(1e-4, 0.01))``."""
    match dim.space:
        case cfg.ChoiceConfig(options=options):
            written = ", ".join(options)
        case cfg.IntUniformConfig(low=low, high=high, log=log):
            written = f"{low}, {high}" + (", log=true" if log else "")
        case space:
            written = f"{short(space.low)}, {short(space.high)}"
    return f"{dim.where} ({dim.space.type}({written}))"


def searching(run: Run, dims: list[Dimension]) -> str:
    """Return what ``bux check`` says of a run's search: its method and trials, its
    dimensions, its objective, and its rungs of steps with halving."""
    from bux.search import halving

    search = run.train.search
    line = f"search: {config_to_source(search.method)}, {search.trials} trials over "
    line += " × ".join(map(dimension, dims))
    line += f"\nobjective = {value_to_source(cfg.objective(run.train))}"
    if (h := search.halving) is not None:
        rungs = [
            halving.steps(h, run.train.steps, r) or run.train.steps
            for r in range(h.rungs)
        ]
        sizes = ", ".join(map(curriculum.si, rungs))
        line += f"\nhalving: rungs of {sizes} steps, eta {h.eta}"
        line += f", about {_worth(search):.3g} runs' steps in all"
    return line


def _worth(search: cfg.SearchConfig) -> float:
    """Return about how many runs' steps a search with halving takes in all: every
    rung as many, of 1 / eta of the trials of the rung before, each for eta times the
    steps."""
    h = search.halving
    assert h is not None
    return h.rungs * search.trials / h.eta ** (h.rungs - 1)


def _tokens(t: TrainConfig) -> int:
    """Return the tokens of a run's steps, phase by phase."""
    return curriculum.tokens(curriculum.stretches(t), t.steps)


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
    from bux.train import ema
    from bux.train.optim import optimizer
    from bux.train.stages import balanced

    t = run.train

    def make() -> tuple[Model, nnx.Optimizer]:
        model = Model(run.arch, rngs=nnx.Rngs(0))
        # The schedule has no state, so a constant one stands in for the run's.
        tx = optimizer(t.optimizer, lambda step: jnp.ones(()), model, t.clip)
        tx = ema.beside(tx, t)  # the average is state as the moments are
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
    (``trainer.step_flops``), averaged over the seq_len of its phases by their
    tokens; None if the backend lacks a layer."""
    from bux.train.trainer import step_flops

    t = run.train
    tokens: Counter[int] = Counter()  # by seq_len
    for s in curriculum.stretches(t):
        tokens[s.config.seq_len] += (
            (s.stop - s.start) * s.config.batch * s.config.seq_len
        )
    try:
        each = {n: step_flops(run, t.batch, n).one_document(n) for n in tokens}
    except NotImplementedError:
        return None
    return sum(each[n] / (t.batch * n) * k for n, k in tokens.items()) / tokens.total()


def _row(run: Run, gpus: int, alone: bool) -> tuple[str, ...]:
    """Return the cells of `run`'s line in the plan, out of `gpus` GPUs."""
    from bux.architecture import num_params
    from bux.launch import need
    from bux.parallel.pipeline import bubble

    t = run.train
    tokens = _tokens(t)
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
        curriculum.si(n) if n else "?",
        curriculum.si(tokens),
        f"{tokens / n:.1f}" if n else "?",
        curriculum.si(work) if work else "?",
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
    from bux.launch import table, visible_gpus

    available = len(visible_gpus()) if gpus is None else gpus
    head = (
        *("id", "params", "tokens", "tok/param", "flops/tok"),
        *("gpus", "params/gpu", "optim/gpu", "bubble"),
    )
    alone = len(runs) == 1
    rows = [(*head, "run"), *(_row(run, available, alone) for run in runs)]
    lines = "\n".join(table(rows))
    return f"{len(runs)} run{'s' if len(runs) != 1 else ''}\n" + lines


def plan_searches(spaces: list[Space]) -> str:
    """Return a line for each run with dimensions: its search's trials, dimensions
    and method, and with halving, how many runs' steps they take in all."""
    lines = []
    for space in spaces:
        if not space.dimensions:
            continue
        search = space.run.train.search
        line = f"{space.run.name}: {search.trials} trials of "
        line += (
            f"{len(space.dimensions)} dimensions by {config_to_source(search.method)}"
        )
        if (h := search.halving) is not None:
            line += f", {h.rungs} rungs: about {_worth(search):.3g} runs' steps"
        lines.append(line)
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    """Return the parser of bux's command line."""
    p = argparse.ArgumentParser(
        prog="bux",
        description="Compile, inspect and train .bux models and runs.",
        epilog="bux export RUN_DIR OUT writes a trained run as a transformers model: "
        "bux export --help. bux studio FILE edits a file in a browser: bux studio --help",
    )
    p.add_argument(
        "command",
        choices=[
            "check",
            "plan",
            "prepare",
            "train",
            "launch",
            "results",
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
        help="train, launch, results: where run directories go (default: runs)",
    )
    p.add_argument(
        "--only",
        metavar="TEXT",
        help="train, launch, results: only the runs whose name contains TEXT",
    )
    p.add_argument(
        "--index",
        type=int,
        help="train, launch, results: only the run at INDEX, from 0, in the order bux "
        "plan lists them; $SLURM_ARRAY_TASK_ID or $JOB_COMPLETION_INDEX in an array job",
    )
    p.add_argument(
        "--cache",
        default="data/packed",
        help="prepare, train, launch: where prepared datasets go (default: data/packed)",
    )
    p.add_argument(
        "--workers",
        type=int,
        help="prepare, launch: processes that prepare data (default: one per core)",
    )
    return p


def export_parser() -> argparse.ArgumentParser:
    """Return the parser of ``bux export``."""
    p = argparse.ArgumentParser(
        prog="bux export",
        description="Write a trained run's model as a transformers model directory: "
        "its code, its weights, its tokenizer and a model card. Given a .bux file, "
        "write its model's code and config alone, without weights.",
    )
    p.add_argument(
        "run",
        metavar="RUN_DIR",
        help="the run's directory, runs/<name>-<id>; or a .bux file",
    )
    p.add_argument("out", metavar="OUT", help="the directory to write")
    p.add_argument(
        "settings",
        nargs="*",
        metavar="SETTING=VALUE",
        help="a .bux file's: statements that redefine the file's own",
    )
    p.add_argument(
        "--entry", default="model", help="a .bux file's definition (default: model)"
    )
    p.add_argument(
        "--step",
        type=int,
        metavar="N",
        help="the checkpoint's step (default: the latest)",
    )
    p.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        help="the weights' dtype (default: the parameters', float32)",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="compare the export with bux on the run's weights, in float32, and fail "
        "if they disagree",
    )
    return p


def studio_parser() -> argparse.ArgumentParser:
    """Return the parser of ``bux studio``."""
    p = argparse.ArgumentParser(
        prog="bux studio",
        description="Serve a page that shows a .bux file's source, its model as a "
        "diagram, and its training and sweep, and edits them all in the file.",
    )
    p.add_argument(
        "file", metavar="FILE", help="the file; a new one starts from a template"
    )
    p.add_argument(
        "--port",
        type=int,
        default=8765,
        help="the port to try first, then the next few (default: 8765)",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="the address to listen on (default: 127.0.0.1)",
    )
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="print the page's URL without opening a browser",
    )
    return p


def _studio(args: argparse.Namespace) -> int:
    from bux.studio.server import serve

    return serve(args.file, host=args.host, port=args.port, browser=not args.no_browser)


def _export(args: argparse.Namespace) -> int:
    if args.run.endswith(".bux"):
        return _export_file(args)
    if args.settings:
        return _error("bux export: settings go with a .bux file, not a run")
    from transformers.utils import logging

    from bux.export.runs import describe, export

    logging.disable_progress_bar()  # a bar each time --check loads the export
    try:
        done = export(
            args.run, args.out, step=args.step, dtype=args.dtype, check=args.check
        )
    except (FileNotFoundError, ValueError, NotImplementedError) as err:
        return _error(f"bux export: {err}")
    run, step = done.restored.run, done.restored.step
    print(
        f"{run.name}: step {step} exported to {done.out}, its weights in {done.dtype}"
    )
    if done.report is None:
        return 0
    print(describe(done.report))
    try:
        done.report.check()
    except AssertionError as err:
        return _error(f"bux export: the export disagrees with bux: {err}")
    return 0


def _export_file(args: argparse.Namespace) -> int:
    if args.step is not None or args.check:
        return _error("bux export: a .bux file has no weights to --step or --check")
    if (source := _read(args.run)) is None:
        return 1
    import torch

    from bux.export.huggingface import save
    from bux.lang import compile

    try:
        arch = compile(
            source, entry=args.entry, filename=args.run, overrides=args.settings
        )
        save(arch, None, args.out, dtype=getattr(torch, args.dtype or "bfloat16"))
    except BuxError as err:
        return _error(str(err))
    except NotImplementedError as err:
        return _error(f"bux export: {err}")
    from flax import nnx

    from bux.architecture import Model, num_params

    n = curriculum.si(num_params(nnx.eval_shape(lambda: Model(arch, rngs=nnx.Rngs(0)))))
    print(f"{args.run}: {n} parameters exported to {args.out}, without weights")
    return 0


def _error(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _read(path: str) -> str | None:
    """Return the text of the file at `path`; None, saying why, if it cannot be read."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as err:
        _error(f"error: cannot read {path}: {err.strerror or err}")
        return None


def _check(report: Report, arch: Architecture, runs: list[Run]) -> None:
    print(summary(arch))
    if report.run is not None:
        print(training(report.run))
        if report.dimensions:
            print(searching(report.run, report.dimensions))
    if len(runs) > 1:
        taken = {part for run in runs for part in run.choices}
        live = [
            a
            for a in report.axes
            if any(a.part(a.pick(i)) in taken for i in range(len(a.options)))
        ]
        axes = " × ".join(f"{a.where} ({' | '.join(a.options)})" for a in live)
        print(f"sweep: {len(runs)} runs over {axes}")


def _prepare(runs: list[Run], cache: str, workers: int | None) -> int:
    from bux.data.pipeline import jobs, prepare

    todo = dict.fromkeys(job for run in runs for job in jobs(run.train))
    if not todo:
        return _error("this file reads no documents to prepare")
    for job in todo:
        print(prepare(job, cache, workers=workers))
    return 0


def _select(spaces: list[Space], index: int | None, only: str | None) -> list[Space]:
    if index is not None:
        if not 0 <= index < len(spaces):
            n, s = len(spaces), "s" if len(spaces) != 1 else ""
            raise ValueError(f"--index {index}, but the file makes {n} run{s}")
        spaces = spaces[index : index + 1]
    if only is not None:
        spaces = [space for space in spaces if only in space.run.name]
        if not spaces:
            raise ValueError(f"no run's name contains {only!r}")
    return spaces


def _train(args: argparse.Namespace, spaces: list[Space]) -> int:
    try:
        spaces = _select(spaces, args.index, args.only)
    except ValueError as err:
        return _error(str(err))
    from bux.runtime import configure

    configure()
    runs = [space.run for space in spaces]
    if any(space.dimensions for space in spaces):
        from bux.launch import search

        here = args.command == "train"
        return search(
            spaces, args.out, cache=args.cache, workers=args.workers, here=here
        )
    if args.command == "launch":
        from bux.launch import launch

        return launch(runs, args.out, cache=args.cache, workers=args.workers)
    from bux.train.trainer import train

    for run in runs:
        train(run, args.out, cache=args.cache)
    return 0


def _results(args: argparse.Namespace, spaces: list[Space]) -> int:
    try:
        spaces = _select(spaces, args.index, args.only)
    except ValueError as err:
        return _error(str(err))
    from bux.launch import summary

    if not any(space.dimensions for space in spaces):
        print(summary([space.run for space in spaces], args.out))
        return 0
    from bux.search.search import Search, Searches

    searches = Searches([Search(space, args.out) for space in spaces])
    print(searches.summary())
    return 0


def _run(
    args: argparse.Namespace, report: Report, arch: Architecture, spaces: list[Space]
) -> int:
    runs = [space.run for space in spaces]
    match args.command:
        case "check":
            _check(report, arch, runs)
        case "plan" | "train" | "launch" | "results" if not runs:
            return _error("this file does not say how to train its model")
        case "plan":
            print(plan(runs))
            if searches := plan_searches(spaces):
                print(searches)
        case "prepare":
            return _prepare(runs, args.cache, args.workers)
        case "train" | "launch":
            return _train(args, spaces)
        case "results":
            return _results(args, spaces)
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
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["export"]:
        return _export(export_parser().parse_args(argv[1:]))
    if argv[:1] == ["studio"]:
        return _studio(studio_parser().parse_args(argv[1:]))
    cli = parser()
    args = cli.parse_args(argv)
    for setting in args.settings:
        if "=" not in setting and not setting.startswith("use "):
            cli.error(f"settings are name=value or 'use module', not {setting!r}")
    if (source := _read(args.file)) is None:
        return 1
    where = {"entry": args.entry, "filename": args.file, "overrides": args.settings}
    report = check(source, **where)
    if report.errors:
        return _error("\n\n".join(map(str, report.errors)))
    if report.arch is None:
        defined = ", ".join(s.name for s in report.statements)
        print(f"a module: it defines {defined or 'nothing'}, and no model")
        return 0
    try:
        spaces = compile_spaces(source, **where) if report.run is not None else []
    except BuxError as err:
        return _error(str(err))
    return _run(args, report, report.arch, spaces)


if __name__ == "__main__":
    sys.exit(main())
