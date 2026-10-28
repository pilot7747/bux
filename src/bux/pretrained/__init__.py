"""Pretrained weights: a released checkpoint loaded into a bux model, and a bux model
saved back as one.

A checkpoint loads in the order the model's layers run. A family's table
(`register`) says what its layers are, in that order: for each, its bux type and its
parts (`Part`), each some of the layer's tensors as the export names them
(`bux.export.weights`: ``q.weight``, ``norm.weight``, a linear's weight ``(out,
in)``), made of some of the checkpoint's tensors, and the way back. `read` makes the
layers' tensors (`Layer`), and `load` pairs them with the model's layers that hold
parameters, one for one, failing on any difference of type, name or shape. So a
table knows nothing of how a ``.bux`` file names its definitions::

    from bux import pretrained

    model = Model(arch, rngs=nnx.Rngs(0))
    pretrained.load(model, arch, pretrained.read("Qwen/Qwen3.5-0.8B"))

`save` runs the table the other way: a model of the released architecture, trained
on, written as the released checkpoint is, with its config and tokenizer, for
transformers' and vLLM's own code of the family.

A checkpoint's optional parts, as a multi-token prediction module, are groups of
layers (`Layer.group`): a model that leaves a group out loads the rest. A model that
changes the released architecture loads what still matches: `skip` names the layers,
by type or by definition, that start fresh instead (`load`).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from jaxtyping import Shaped

from bux.architecture import Model
from bux.export import plan as plans
from bux.export import weights
from bux.lang import ir

#: A layer's tensors, by their names under its module.
Tensors = dict[str, Shaped[np.ndarray, "..."]]


class Layer(NamedTuple):
    """One layer's tensors in a checkpoint: its bux layer type, as ``gdn``, and its
    tensors as the export names them."""

    kind: str
    tensors: Tensors
    #: The optional part of the checkpoint the layer is in, as ``mtp``, which a model
    #: may leave out; None for a layer every model has.
    group: str | None = None


class Checkpoint(Mapping[str, Shaped[np.ndarray, "..."]]):
    """A checkpoint's tensors by name, read from its safetensors files as they are
    asked for."""

    def __init__(self, folder: Path):
        from safetensors import safe_open

        self.folder, self.files = folder, {}
        for f in sorted(folder.glob("*.safetensors")):
            with safe_open(f, "flax") as handle:
                self.files |= dict.fromkeys(handle.keys(), f)

    def __getitem__(self, name: str) -> Shaped[np.ndarray, "..."]:
        from safetensors import safe_open

        if name not in self.files:
            raise KeyError(f"the checkpoint in {self.folder} has no tensor '{name}'")
        with safe_open(self.files[name], "flax") as handle:
            return np.asarray(handle.get_tensor(name))

    def __iter__(self) -> Iterator[str]:
        return iter(self.files)

    def __len__(self) -> int:
        return len(self.files)


class Part(NamedTuple):
    """Tensors of a bux layer, `names`, made of tensors of the checkpoint, `keys`.

    ``read(*checkpoint's) -> bux's`` makes them, and ``write(bux's, released) ->
    checkpoint's`` the checkpoint's back, given the released ones, which keep what bux
    does not tell apart (as two frames of a patch embedding that bux sums).
    """

    names: tuple[str, ...]
    keys: tuple[str, ...]
    read: Callable[..., tuple]
    write: Callable[[tuple, tuple], tuple]


def same(name: str, key: str) -> Part:
    """Return the part of the bux tensor `name`, the checkpoint's `key` as it is."""
    return Part((name,), (key,), lambda x: (x,), lambda ys, _: ys)


class Spec(NamedTuple):
    """One layer of a checkpoint: its bux type, its parts, and the optional group it
    is in, as `Layer` has it."""

    kind: str
    parts: tuple[Part, ...]
    group: str | None = None


#: A family's table: ``(config) -> layers``, `config` its config.json.
Table = Callable[[dict], list[Spec]]

_TABLES: dict[str, Table] = {}


def register(*model_types: str) -> Callable[[Table], Table]:
    """Register the table of checkpoints whose config.json has a ``model_type``
    among `model_types`, replacing any before it."""

    def decorate(f: Table) -> Table:
        for t in model_types:
            _TABLES[t] = f
        return f

    return decorate


def read(source: str | Path) -> list[Layer]:
    """Return a checkpoint's layers, by the table of its ``model_type``.

    Args:
        source: A folder with a config.json and safetensors files, or a Hugging Face
            Hub repo, as ``Qwen/Qwen3.5-0.8B``, downloaded to its cache.

    Raises:
        ValueError: No table is registered for the checkpoint's model type, or the
            checkpoint has tensors no layer reads.
    """
    folder = _folder(source)
    specs, ckpt = _table(folder, source), Checkpoint(folder)
    out = []
    for spec in specs:
        tensors: Tensors = {}
        for part in spec.parts:
            made = part.read(*(ckpt[k] for k in part.keys))
            tensors |= dict(zip(part.names, made, strict=True))
        out.append(Layer(spec.kind, tensors, spec.group))
    if rest := sorted(set(ckpt) - {k for s in specs for p in s.parts for k in p.keys}):
        raise ValueError(f"the checkpoint has tensors no layer reads: {rest[:5]}")
    return out


def _table(folder: Path, source: str | Path) -> list[Spec]:
    """Return the layers of the checkpoint in `folder`, by its family's table."""
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    kind = config.get("model_type")
    if kind not in _TABLES:
        raise ValueError(
            f"{source} is a '{kind}' checkpoint, which no table reads; known: "
            f"{', '.join(sorted(_TABLES))}. Add one with "
            f"@bux.pretrained.register({kind!r})"
        )
    return _TABLES[kind](config)


def load(
    model: Model,
    arch: ir.Architecture,
    layers: Sequence[Layer],
    skip: Sequence[str] = (),
) -> None:
    """Set the parameters of `model`, built from `arch`, from a checkpoint's layers.

    The model's layers that hold parameters take the checkpoint's in order, but those
    whose type is in `skip`, which keep their fresh parameters, and the checkpoint's
    optional groups (`Layer.group`) the model leaves out are not read. Every parameter of a
    layer that loads must be in the checkpoint, and every tensor of the checkpoint
    must go somewhere.

    Args:
        model: The model, built with ``methods.repeat`` ``unroll`` (the default).
        arch: Its architecture.
        layers: The checkpoint's layers, in the order they run (`read`).
        skip: The model's layers that start fresh, by type, as ``gla`` for a model
            whose attention is swapped for gated linear attention, or by the name of
            their definition, as ``pool`` for ``pool = linear(...)``. Each takes the
            place of the checkpoint's layer where it is; or, if the layers line up
            only without the skipped ones, they are new.

    Raises:
        ValueError: The model's layers and the checkpoint's differ: in number, type,
            names or shapes. The message names the first difference.
    """
    targets = _targets(model, arch)
    have = [c.layer.config.type for c, _ in targets]
    fresh = [c.layer.config.type in skip or c.layer.label in skip for c, _ in targets]
    for (call, refs), layer in zip(
        targets, _matching(have, layers, fresh), strict=True
    ):
        if layer is not None:
            _assign(call.path, call.layer.config.type, refs, layer)


def save(
    model: Model,
    arch: ir.Architecture,
    source: str | Path,
    out: str | Path,
    shard: int = 5 << 30,
) -> list[str]:
    """Write `model`, built from `arch`, as the checkpoint `source` is: its tensors,
    each in the released one's dtype, under its names, beside the released config,
    tokenizer and processor files. transformers and vLLM then load it with their own
    code of the family.

    The model must be of the released architecture: its layers that hold parameters
    are the checkpoint's, in order, by type, name and shape. An optional group it
    leaves out (`Layer.group`), as an MTP module, is written as released.

    Args:
        model: The model, built with ``methods.repeat`` ``unroll`` (the default).
        arch: Its architecture.
        source: The released checkpoint: a Hub repo or a folder, as `read` takes it.
        out: The folder to write, made if missing.
        shard: The most bytes of tensors a safetensors file holds.

    Returns:
        The optional groups written as released.

    Raises:
        ValueError: The model's layers differ from the checkpoint's: the message
            names the first difference. A model whose architecture changed exports
            by `bux.export`'s own code instead.
    """
    from safetensors.numpy import save_file

    folder = _folder(source, files=True)
    specs, ckpt = _table(folder, source), Checkpoint(folder)
    targets = _targets(model, arch)
    have = [c.layer.config.type for c, _ in targets]
    shapes = [Layer(s.kind, {}, s.group) for s in specs]  # kinds and groups alone
    paired = _matching(have, shapes, [False] * len(have))
    written: dict[str, Shaped[np.ndarray, "..."]] = {}
    for (call, refs), layer in zip(targets, paired, strict=True):
        assert layer is not None
        spec = specs[next(i for i, x in enumerate(shapes) if x is layer)]
        names = {n for p in spec.parts for n in p.names}
        _check(call.path, call.layer.config.type, refs, spec.kind, names)
        mine = {name: ref.array() for name, ref in refs.items()}
        for part in spec.parts:
            released = tuple(ckpt[k] for k in part.keys)
            made = part.write(tuple(mine[n] for n in part.names), released)
            for key, x, old in zip(part.keys, made, released, strict=True):
                if x.shape != old.shape:
                    raise ValueError(
                        f"{call.path} makes {key} of shape {x.shape}, where the "
                        f"checkpoint's is {old.shape}"
                    )
                written[key] = np.ascontiguousarray(x, old.dtype)
    left = sorted({x.group for x in shapes if x.group} - {x.group for x in paired if x})
    for key in ckpt:  # the groups the model leaves out, as released
        if key not in written:
            written[key] = ckpt[key]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in folder.iterdir():
        if (
            f.is_file()
            and f.suffix != ".safetensors"
            and f.name != "model.safetensors.index.json"
        ):
            (out / f.name).write_bytes(f.read_bytes())
    files = _shards(written, shard)
    index: dict[str, str] = {}
    for i, keys in enumerate(files):
        name = f"model-{i + 1:05d}-of-{len(files):05d}.safetensors"
        save_file({k: written[k] for k in keys}, out / name, metadata={"format": "pt"})
        index |= dict.fromkeys(keys, name)
    size = sum(x.nbytes for x in written.values())
    meta = {"metadata": {"total_size": size}, "weight_map": dict(sorted(index.items()))}
    (out / "model.safetensors.index.json").write_text(json.dumps(meta, indent=2))
    return left


def _targets(
    model: Model, arch: ir.Architecture
) -> list[tuple[plans.LayerCall, dict[str, weights.Ref]]]:
    """Return the model's layers that hold parameters, in the order they run, each
    its call and its variables by name."""
    out = []
    seen: set[int] = set()
    for call in plans.plan(arch).calls:
        if not call.first:
            continue
        module = weights.module_at(arch.graph, model.graph, call.trail)
        if refs := dict(weights.refs(module, "", seen)):
            out.append((call, refs))
    return out


def _shards(
    tensors: Mapping[str, Shaped[np.ndarray, "..."]], most: int
) -> list[list[str]]:
    """Return the names of `tensors`, in order, in runs of at most `most` bytes."""
    out: list[list[str]] = [[]]
    size = 0
    for k in sorted(tensors):
        n = tensors[k].nbytes
        if out[-1] and size + n > most:
            out.append([])
            size = 0
        out[-1].append(k)
        size += n
    return out


def _matching(
    have: Sequence[str], layers: Sequence[Layer], fresh: Sequence[bool]
) -> list[Layer | None]:
    """Return the checkpoint's layer each of the model's layers, of types `have`,
    takes; None for a fresh one.

    The optional groups the model leaves out are dropped: the most groups are kept
    for which the types line up. A fresh layer takes the place of the checkpoint's
    layer where it is, whatever its type; or, if the layers line up only without it,
    it is new."""
    groups = list(dict.fromkeys(x.group for x in layers if x.group is not None))
    for kept in sorted(_subsets(groups), key=len, reverse=True):
        xs = [x for x in layers if x.group is None or x.group in kept]
        pairs = list(zip(have, fresh, strict=True))
        replaced = len(xs) == len(have) and all(
            f or h == x.kind for (h, f), x in zip(pairs, xs, strict=True)
        )
        if replaced:
            return [None if f else x for f, x in zip(fresh, xs, strict=True)]
        if [x.kind for x in xs] == [h for h, f in pairs if not f]:
            it = iter(xs)
            return [None if f else next(it) for f in fresh]
    for kept in sorted(_subsets(groups), key=len, reverse=True):
        xs = [x for x in layers if x.group is None or x.group in kept]
        if len(xs) == len(have) and not any(fresh):  # name the first that differs
            return list(xs)
    starting = [h for h, f in zip(have, fresh, strict=True) if f]
    raise ValueError(
        f"the model's layers with parameters are {_counts(have)}, the checkpoint's "
        f"{_counts([x.kind for x in layers])}"
        + (f", of which the groups {groups} are optional" if groups else "")
        + (f"; the model's {_counts(starting)} start fresh" if starting else "")
    )


def _subsets(xs: Sequence[str]) -> list[tuple[str, ...]]:
    """Return every subset of `xs`."""
    return [
        tuple(x for i, x in enumerate(xs) if m >> i & 1) for m in range(1 << len(xs))
    ]


def _assign(
    path: str, kind: str, refs: Mapping[str, weights.Ref], layer: Layer
) -> None:
    """Set the variables `refs` of the layer at `path` from `layer`'s tensors."""
    _check(path, kind, refs, layer.kind, set(layer.tensors))
    for name, ref in refs.items():
        try:
            ref.set(layer.tensors[name])
        except ValueError as e:
            raise ValueError(f"{path}.{name}: {e}") from None


def _check(
    path: str,
    kind: str,
    refs: Mapping[str, weights.Ref],
    theirs: str,
    names: set[str],
) -> None:
    """Raise if the layer at `path`, of type `kind` and variables `refs`, is not the
    checkpoint's, of type `theirs` and tensors `names`."""
    if theirs != kind:
        raise ValueError(
            f"{path} is a '{kind}' layer, but the checkpoint's is a '{theirs}'"
        )
    if set(refs) != names:
        raise ValueError(
            f"{path} ({kind}) has the parameters {sorted(refs)}, but the checkpoint's "
            f"layer has {sorted(names)}"
        )


def _counts(types: Sequence[str]) -> str:
    """Return `types` as runs, ``embed, (gdn x 3, gqa) ...`` compressed to counts."""
    out: list[str] = []
    for t in types:
        if out and out[-1].split(" x ")[0] == t:
            n = int(out[-1].split(" x ")[1]) if " x " in out[-1] else 1
            out[-1] = f"{t} x {n + 1}"
        else:
            out.append(t)
    return ", ".join(out)


def _folder(source: str | Path, files: bool = False) -> Path:
    """Return the folder of `source`, downloading a Hub repo's config and safetensors;
    with `files`, its other files too: its tokenizer's, its chat template."""
    if Path(source).is_dir():
        return Path(source)
    from huggingface_hub import snapshot_download

    patterns = None if files else ["*.json", "*.safetensors"]
    return Path(snapshot_download(str(source), allow_patterns=patterns))


from bux.pretrained import qwen3_5 as qwen3_5  # registers its table
