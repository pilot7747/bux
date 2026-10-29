"""Trained runs, exported: a run's directory read back into a bux model, and written
as a transformers model directory that loads as it is, its tokenizer with it.

::

    from bux.export.runs import export

    export("runs/smoke-5a4f0f7d85", "exported/", check=True)

`restore` reads the run's ``run.json`` and one of its checkpoints onto this
process's CPU, whatever mesh, layout and methods trained it: FSDP's and tensor
parallelism's shards restore as whole arrays, and a scanned repeat's passes, stacked
along a leading dimension, or split over a pipeline's stages, come apart into the
passes of the model ``methods.repeat = unroll`` builds, which the exporter takes.
The rest of the model's state comes with it, an MoE's router bias too.
"""

import contextlib
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import torch
from flax import nnx
from jax.sharding import AbstractMesh, AxisType, SingleDeviceSharding
from jaxtyping import Shaped

from bux import parallel
from bux.architecture import Model
from bux.architecture.repeats.repeat import Scanned
from bux.export import codegen, huggingface, tokenizer
from bux.export.check import Report, compare
from bux.export.plan import plan
from bux.lang import ir
from bux.lang.printer import run_to_source
from bux.train import curriculum, hooks
from bux.train.config import TrainConfig
from bux.train.stages import balanced

#: The name of the model's part of a checkpoint, which keeps
#: ``nnx.state((model, optimizer))``.
MODEL = 0


@dataclass
class Restored:
    """A run's model, as one of its checkpoints holds it.

    Attributes:
        run: The run, from its ``run.json``.
        model: Its model, built with ``methods.repeat`` ``unroll``, in eval mode, its
            state the checkpoint's, on this process's CPU, as its weights trained
            (`arch`).
        step: The checkpoint's step: the steps its weights took.
    """

    run: ir.Run
    model: Model
    step: int

    @property
    def arch(self) -> ir.Architecture:
        """The model as its weights trained: with the RoPE of the curriculum's phase
        of the last step they took (`last_step`, `bux.lang.ir.Run.arch_at`)."""
        return self.run.arch_at(last_step(self.step))


def last_step(step: int) -> int:
    """Return the last step the weights of a checkpoint of `step` took, from 0: a
    checkpoint of a phase's first step holds weights the phase before trained; step
    0's, for the weights a run starts from."""
    return max(step, 1) - 1


@dataclass(frozen=True)
class Layout:
    """How a run laid out a pipelined repeat: its mesh's `stage` axis, and the passes
    of each stage of an uneven split, as its checkpoint keeps them (``hooks.layout``);
    None for an even one."""

    stage: int
    stages: tuple[int, ...] | None

    def shares(self, passes: int) -> tuple[int, ...]:
        """Return the passes of a round each stage ran (``pipeline.shares``)."""
        return self.stages or parallel.pipeline.even(passes, self.stage)


def read_run(folder: Path | str) -> ir.Run:
    """Return the run whose directory is `folder`, from its ``run.json``.

    Raises:
        FileNotFoundError: `folder` is not a run's directory.
    """
    path = Path(folder) / "run.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{folder} is not a run's directory: it has no run.json"
        )
    return ir.Run.model_validate_json(path.read_text(encoding="utf-8"))


def restore(folder: Path | str, step: int | None = None) -> Restored:
    """Restore the model of the run in `folder` from its checkpoint of `step`.

    Args:
        folder: The run's directory, ``runs/<name>-<id>``.
        step: The checkpoint's step; None for the latest.

    Returns:
        The run, and its model as the checkpoint holds it, on this process's CPU.

    Raises:
        FileNotFoundError: `folder` is not a run's directory, or has no checkpoint
            (of `step`).
        ValueError: The checkpoint holds other arrays than bux builds for the run.
    """
    folder = Path(folder)
    run = read_run(folder)
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu), _manager(folder / "checkpoints") as manager:
        step = _step(manager, step, folder)
        layout = _layout(run, manager.metadata(step).custom_metadata or {})
        at = run.model_copy(update={"arch": run.arch_at(last_step(step))})
        trained = as_trained(at, layout)
        nnx.update(trained, _read(manager, step, nnx.state(trained), cpu))
        model = unrolled(at, trained, layout)
    model.eval()
    return Restored(run, model, step)


def _layout(run: ir.Run, saved: dict) -> Layout:
    """Return how the run laid out its pipelined repeat, as its checkpoint says; one
    saved before checkpoints said is taken to have the split training balanced."""
    if "methods.stages" in saved:
        stages = saved["methods.stages"]
    else:
        stages = balanced(run).stages
    split = tuple(stages) if stages and len(set(stages)) > 1 else None
    return Layout(run.train.mesh.stage, split)


@contextlib.contextmanager
def _manager(folder: Path) -> Generator[ocp.CheckpointManager]:
    """Open the checkpoints in `folder` to read."""
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder.parent} has no checkpoints")
    manager = ocp.CheckpointManager(
        folder.resolve(),
        options=ocp.CheckpointManagerOptions(
            read_only=True, save_interval_steps=0, create=False
        ),
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    try:
        yield manager
    finally:
        manager.close()


def _step(manager: ocp.CheckpointManager, step: int | None, folder: Path) -> int:
    """Return `step`, or the latest, once it is a step the manager holds."""
    steps = manager.all_steps()
    if not steps:
        raise FileNotFoundError(f"{folder} has no checkpoints")
    if step is None:
        return max(steps)
    if step not in steps:
        have = ", ".join(map(str, sorted(steps)))
        raise FileNotFoundError(f"{folder} has no checkpoint of step {step}: {have}")
    return step


def as_trained(run: ir.Run, layout: Layout) -> Model:
    """Return the run's model from shapes alone, as it trained: under its methods and
    its mesh, without devices, its repeats scanned and pipelined as they were."""
    t = run.train
    stages = None if layout.stages is None else list(layout.stages)
    methods = t.methods.model_copy(update={"stages": stages})
    sizes = {axis: getattr(t.mesh, axis) or 1 for axis in parallel.AXES}
    auto = (AxisType.Auto,) * len(parallel.AXES)
    grid = AbstractMesh(tuple(sizes.values()), parallel.AXES, axis_types=auto)
    with parallel.use(grid, t.layout, methods):
        return nnx.eval_shape(lambda: Model(run.arch, rngs=nnx.Rngs(t.seed)))


def _read(
    manager: ocp.CheckpointManager, step: int, shapes: nnx.State, device: jax.Device
) -> nnx.State:
    """Read the model's part of the checkpoint of `step`, whole arrays on `device`,
    and not the optimizer's.

    Raises:
        ValueError: The checkpoint's arrays are not those of `shapes`.
    """
    here = SingleDeviceSharding(device)
    target = {
        MODEL: jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=here), shapes
        )
    }
    mine = hooks._arrays(target)
    saved = {
        path: array
        for path, array in hooks._arrays(manager.item_metadata(step)).items()
        if path.startswith(f"{MODEL}/")
    }
    if misfit := hooks._differs(saved, mine):
        raise ValueError(
            f"the checkpoint of step {step} does not hold the model bux builds for "
            f"this run: {misfit}"
        )
    args = ocp.args.PyTreeRestore(
        item=target,
        restore_args=ocp.checkpoint_utils.construct_restore_args(target),
        partial_restore=True,
    )
    with ocp.PyTreeCheckpointer() as reader:
        return reader.restore(manager.directory / str(step) / "default", args=args)[
            MODEL
        ]


def unrolled(run: ir.Run, trained: Model, layout: Layout) -> Model:
    """Return the run's model built with ``methods.repeat`` ``unroll``, its state that
    of `trained`, as it trained under `layout`: each pass of a scanned repeat its own
    module, its variables a slice of the repeat's stacked ones.

    Raises:
        ValueError: A variable of the model has no match in `trained`.
    """
    model = nnx.eval_shape(lambda: Model(run.arch, rngs=nnx.Rngs(run.train.seed)))
    nodes = dict(nnx.iter_graph(trained))
    scanned = {p: m for p, m in nodes.items() if isinstance(m, Scanned)}
    for path, variable in nnx.iter_graph(model):
        if not isinstance(variable, nnx.Variable):
            continue
        where, index = source(path, scanned, layout)
        value = nodes[where].get_value()[index]
        if value.shape != variable.shape:
            raise ValueError(
                f"{'/'.join(map(str, path))} is {list(variable.shape)}, and its pass of "
                f"{'/'.join(map(str, where))} {list(value.shape)}"
            )
        variable.set_value(value)
    return model


def source(
    path: tuple, scanned: dict[tuple, Scanned], layout: Layout
) -> tuple[tuple, tuple[int, ...]]:
    """Return where the variable of an unrolled model at `path` is in the model that
    trained, whose scanned repeats are `scanned` by path: its path there, and its
    index in the dimensions that stack passes, outermost first."""
    where, index, rest = (), (), list(path)
    while rest:
        key = rest.pop(0)
        if key == "items" and (repeat := scanned.get(where)) is not None:
            keys, at = _pass(repeat, rest.pop(0), layout)
            where, index = (*where, *keys), (*index, *at)
        else:
            where = (*where, key)
    return where, index


def _pass(
    repeat: Scanned, i: int, layout: Layout
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Return where pass `i` of a scanned repeat is: the path from the repeat to its
    stacked pass, and its index in each stacked dimension (``Scanned.pipe``)."""
    if repeat.rounds > 1:  # rounds of passes, a scanned repeat in each
        each = repeat.layers
        assert isinstance(each, Scanned)
        return ("layers", "layers"), divmod(i, each.times)
    if repeat.rounds == 1:  # a stage's passes, then copies of its first
        return ("layers",), (_row(i, layout.shares(repeat.times)),)
    return ("layers",), (i,)


def _row(i: int, shares: tuple[int, ...]) -> int:
    """Return the row of pass `i` in a stacked variable spread over stages that run
    `shares` of the passes, as many rows a stage as the most any runs
    (``pipeline.spread``)."""
    start = 0
    for stage, n in enumerate(shares):
        if i < start + n:
            return stage * max(shares) + i - start
        start += n
    raise IndexError(f"pass {i} of {start}")


@dataclass
class Exported:
    """What `export` wrote, and what its check found.

    Attributes:
        restored: The run and its model, as the checkpoint held them.
        out: The directory.
        dtype: The weights' dtype in it.
        report: The comparison of the export with bux, if asked for.
    """

    restored: Restored
    out: Path
    dtype: str
    report: Report | None = None


def export(
    folder: Path | str,
    out: Path | str,
    *,
    step: int | None = None,
    dtype: str | None = None,
    check: bool = False,
) -> Exported:
    """Write the model of the run in `folder`, from its checkpoint of `step`, to `out`
    as a transformers model directory that loads as it is: the model as its weights
    trained (`Restored.arch`), its ``max_position_embeddings`` the longest rows the
    run trained on by then, both by the phase of the last step the weights took
    (`last_step`), the run's tokenizer and generation config
    (`bux.export.tokenizer.save`), and a model card, ``README.md``.

    With `check`, the export is compared with bux (`bux.export.check.compare`) on the
    restored weights as `out` holds them, rounded to `dtype`, both in float32.

    Args:
        folder: The run's directory, ``runs/<name>-<id>``.
        out: The directory to write, made if missing.
        step: The checkpoint's step; None for the latest.
        dtype: The weights' dtype, ``float32``, ``bfloat16`` or ``float16``; None for
            the parameters' own, float32 as bux keeps them.
        check: Compare the export with bux.

    Returns:
        What it wrote, and the comparison's `Report` with `check`, which
        ``Report.check`` asserts on.

    Raises:
        FileNotFoundError: `folder` is not a run's directory, or has no checkpoint
            (of `step`).
        NotImplementedError: A layer of the model, or a setting of one, is not
            exported yet.
    """
    codegen.modeling(
        plan(read_run(folder).arch), []
    )  # a missing layer, before a restore
    restored = restore(folder, step)
    run, arch, model, out = restored.run, restored.arch, restored.model, Path(out)
    dtype = dtype or param_dtype(model)
    torch_dtype, length = getattr(torch, dtype), _longest(run.train, restored.step)
    huggingface.save(arch, model, out, dtype=torch_dtype, max_positions=length)
    if (config := run.train.tokenizer) is not None:
        assert run.arch.vocab_size is not None, "a run has a vocabulary"
        tokenizer.save(config, out, run.arch.vocab_size)
    (out / "README.md").write_text(card(restored, _source(folder), dtype, out))
    report = None
    if check:
        with jax.default_device(jax.devices("cpu")[0]):
            report = compare(arch, rounded(model, dtype))
    return Exported(restored, out, dtype, report)


def _longest(t: TrainConfig, step: int) -> int:
    """Return the longest rows a run of `t` trained on in its first `step` steps:
    those of the phases up to that of their last (`last_step`)."""
    return max(
        c.seq_len for start, c in curriculum.phases(t) if start <= last_step(step)
    )


def _source(folder: Path | str) -> str:
    """Return the run's ``run.bux``, or its IR printed back if it has none."""
    path = Path(folder) / "run.bux"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return run_to_source(read_run(folder))


def param_dtype(model: Model) -> str:
    """Return the widest dtype of `model`'s floating-point parameters."""
    params = jax.tree.leaves(nnx.state(model, nnx.Param))
    floats = {
        jnp.dtype(x.dtype) for x in params if jnp.issubdtype(x.dtype, jnp.floating)
    }
    return max(floats, key=lambda d: d.itemsize, default=jnp.dtype("float32")).name


def card(restored: Restored, source: str, dtype: str, out: Path) -> str:
    """Return the model card of an export: where it comes from, how to load it, and
    the run that trained it."""
    run = restored.run
    tokenizer = (
        f'tokenizer = AutoTokenizer.from_pretrained("{out}")\n'
        if run.train.tokenizer is not None
        else ""
    )
    return (
        f"---\nlibrary_name: transformers\npipeline_tag: text-generation\n"
        f"tags:\n- bux\n---\n\n# {run.name}\n\n"
        f"Exported from bux: the run `{run.name}`, id `{run.id}`, at step "
        f"{restored.step} of {run.train.steps}, its weights in {dtype}. Its model "
        "code is generated from its architecture (`modeling_bux.py`), so "
        "transformers loads it as remote code:\n\n"
        "```python\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
        f'model = AutoModelForCausalLM.from_pretrained("{out}", trust_remote_code=True)'
        f"\n{tokenizer}```\n\n## The run\n\n```bux\n{source.rstrip()}\n```\n"
    )


def rounded(model: Model, dtype: str) -> Model:
    """Return a copy of `model` whose floating-point parameters are rounded to `dtype`
    and kept in their own: the weights an export in `dtype` holds."""
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def round_to(x: Shaped[jax.Array, "*shape"]) -> Shaped[jax.Array, "*shape"]:
        if not jnp.issubdtype(x.dtype, jnp.floating):
            return x
        return x.astype(dtype).astype(x.dtype)

    return nnx.merge(graphdef, jax.tree.map(round_to, params), rest)


def describe(report: Report) -> str:
    """Return what `report` found, a line for each attention implementation."""
    lines = []
    for r in report.results:
        want = r.greedy["bux"]
        greedy = ", ".join(
            f"{how} {'as bux' if r.greedy[how] == want else 'unlike bux'}"
            for how in ("cache", "no cache")
        )
        lines.append(
            f"{r.implementation}: logits within {r.error:.1e} of bux's largest "
            f"(tolerance {report.tolerance:.0e}); greedy decoding with {greedy}; "
            f"a left-padded batch {'as' if r.padded else 'unlike'} its prompts alone"
        )
    return "\n".join(lines)
