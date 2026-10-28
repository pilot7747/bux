"""An exported bux architecture as a vLLM model: ``BuxForCausalLM``, built from
``config.json`` alone.

The model is the plan's tree of blocks (`bux_vllm.plan`): the model's, each
repeat's body, each shared definition's, with every layer, and every branch of a
residual algebra, at the path its weights have in the checkpoint. Its forward runs
the IR graph over the batch's tokens, as the generated ``modeling_bux.py`` does over
a batch's rows, up to the first head: it returns the head's input, and vLLM takes the
head's logits of the tokens it samples (``compute_logits``). Under a residual
algebra, a branch outside any other is its module's read, the body, and its write
back (`bux_vllm.residual`), and the embedding's output outside any branch widens.
The layers after the first head (multi-token prediction's) are not built, and their
weights are skipped.
"""

from __future__ import annotations

import importlib
import operator
from collections import Counter
from collections.abc import Iterable
from functools import cache, reduce
from typing import Any, ClassVar

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import IsHybrid
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import MambaSpec

from . import conv, residual, ssm
from .common import Ctx, Layer, Site
from .config import states
from .plan import Call, Node, Plan, Pos, Scope, at, children, is_call, live, plan

#: The modules of this package that hold layers.
LAYER_MODULES = ("common", "basic", "attention", "mla", "moe", "ssm", "kda", "residual")


@cache
def layer_classes() -> dict[str, type[Layer]]:
    """Return every layer's class, by the bux layer type it implements."""
    out = {}
    for name in LAYER_MODULES:
        module = importlib.import_module(f"{__package__}.{name}")
        for value in vars(module).values():
            if isinstance(value, type) and value.__module__ == module.__name__:
                out |= dict.fromkeys(getattr(value, "bux_types", ()), value)
    return out


class Unused(Layer):
    """A layer that runs only after the first head: not built, its weights skipped."""

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        return []


def cache_name(slot: int, path: str) -> str:
    """vLLM's name of the `slot`-th state the model keeps, of the layer at `path`:
    with one integer, the slot, as vLLM orders its caches by it."""
    return f"caches.{slot}.{path.replace('.', '_')}"


def state_names(calls: list[Call]) -> dict[str, list[str]]:
    """The names of the states of each stateful layer's calls, by its path: each
    call's in turn, its first named by the path, the others by the path and their
    kind (`bux_vllm.config.states`)."""
    out: dict[str, list[str]] = {}
    for call in calls:
        if _class(call).stateful:
            kinds = states(call.layer["config"]) or ["state"]
            names = out.setdefault(call.path, [])
            for i, kind in enumerate(kinds):
                path = call.path if i == 0 else f"{call.path}.{kind}"
                names.append(cache_name(sum(map(len, out.values())), path))
    return out


def build_layers(vllm_config: VllmConfig, p: Plan) -> dict[str, nn.Module]:
    """Build every layer of `p`, by its path, in the order of their first calls (as
    vLLM calls its MoE layers in the order they are built).

    Raises:
        NotImplementedError: A layer type, or a setting of one, is not in vLLM.
    """
    config = vllm_config.model_config.hf_config
    calls = live(p.calls)
    states, count = state_names(calls), Counter(c.path for c in calls)
    layers: dict[str, nn.Module] = {}
    embedding = None
    for call in (c for c in p.calls if c.first):
        if call.path not in count:
            layers[call.path] = Unused()
            continue
        node, cls = call.layer, _class(call)
        if cls.once and count[call.path] > 1:
            raise NotImplementedError(f"{cls.__name__}: called more than once")
        site = Site(
            call.path,
            vllm_config,
            config,
            states.get(call.path, []),
            embedding,
            node["ports"],
        )
        layer = layers[call.path] = cls(node["config"], node["width"], site)
        if embedding is None and node["config"]["type"] == "embed":
            embedding = layer
    return layers | build_branches(config.bux, p, len(calls))


def build_branches(arch: Node, p: Plan, calls: int) -> dict[str, nn.Module]:
    """Build the module of each residual algebra's branch of `p`, by its path: those
    that run before the first head, the first `calls` layer calls; the others are
    not built.

    Raises:
        NotImplementedError: The residual algebra is not in vLLM yet.
    """
    out: dict[str, nn.Module] = {}
    algebra = arch["residual"]
    for b in (b for b in p.branches if b.first):
        if b.calls < calls:
            out[b.path] = residual.algebra(algebra)(algebra, arch["d_model"])
        else:
            out[b.path] = Unused()
    return out


def _class(call: Call) -> type[Layer]:
    """The class of `call`'s layer type.

    Raises:
        NotImplementedError: No layer of the vLLM model implements it.
    """
    kind = call.layer["config"]["type"]
    if kind not in layer_classes():
        raise NotImplementedError(f"bux's {kind!r} is not in vLLM yet")
    return layer_classes()[kind]


class Scoped(nn.Module):
    """A module of a scope of the plan: its layers, repeats and residual algebra's
    branches by their names."""

    def __init__(self, scope: Scope, prefix: str, layers: dict[str, nn.Module]):
        super().__init__()
        self.scope = scope
        for pos, name in scope.names.items():
            if pos in scope.blocks:
                inner = scope.blocks[pos]
                times = at(scope.root, pos)["times"]
                self.add_module(
                    name,
                    nn.ModuleList(
                        Block(inner, f"{prefix}{name}.{i}.", layers)
                        for i in range(times)
                    ),
                )
            else:
                self.add_module(name, layers[prefix + name])


class Block(Scoped):
    """A repeat's body or a shared definition: its scope's graph on the stream."""

    def forward(self, x: Any, ctx: Ctx) -> Any:
        return run(self, self.scope.root, (), x, ctx, self.scope.depth)


def run(block: Scoped, node: Node, pos: Pos, x: Any, ctx: Ctx, depth: int) -> Any:
    """Run `node`, at `pos` in `block`'s scope inside `depth` residual branches, on
    `x`: nothing once the first head has run."""
    if ctx.hidden is not None:
        return x
    if is_call(ctx.plan, block.scope, node, pos):
        y = ctx.shared[node["share"]](x, ctx)
        return _widened(node, y, ctx.plan.shared[node["share"]].depth, ctx)
    kind = node["kind"]
    if kind == "layer":
        y = getattr(block, block.scope.names[pos])(x, ctx)
        return _widened(node, y, depth, ctx)
    if kind == "repeat":
        return _repeat(block, node, pos, x, ctx)
    if kind == "seq":
        for i, item in enumerate(node["items"]):
            x = run(block, item, (*pos, i), x, ctx, depth)
        return x
    if kind == "residual" and pos in block.scope.names:
        return _branch(block, node, pos, x, ctx)
    if kind in LEAVES:
        return LEAVES[kind](node, x, ctx)
    inner = depth + (kind == "residual")
    ys = [run(block, c, (*pos, i), x, ctx, inner) for i, c in enumerate(children(node))]
    if kind == "residual":
        return x + ys[0]
    if kind == "neg":
        return -ys[0]
    return reduce(operator.add if kind == "sum" else operator.mul, ys)


def _branch(block: Scoped, node: Node, pos: Pos, x: Any, ctx: Ctx) -> Any:
    """A residual algebra's branch: its module's read of the stream `x`, the body on
    it, and its write back."""
    algebra = getattr(block, block.scope.names[pos])
    h, kept = algebra.read(x, ctx)
    y = run(block, node["body"], (*pos, 0), h, ctx, 1)  # inside this one branch
    return algebra.write_back(kept, y)


def _widened(node: Node, y: Any, depth: int, ctx: Ctx) -> Any:
    """`y`, the output of `node`, widened by the residual algebra if `node` is an
    embedding outside any branch, as bux's embed widens it."""
    if node["kind"] != "layer" or node["config"]["type"] != "embed" or depth > 0:
        return y
    return residual.widen(ctx.plan.residual, y)


#: The nodes that read no layer: ``value``, ``_``, ``@name``, ``$name``.
LEAVES: dict[str, Any] = {
    "const": lambda node, x, ctx: node["value"],
    "identity": lambda node, x, ctx: x,
    "tap": lambda node, x, ctx: ctx.push(node["name"], x),
    "ref": lambda node, x, ctx: ctx.read(node["name"], node["pop"]),
}


def _repeat(block: Scoped, node: Node, pos: Pos, x: Any, ctx: Ctx) -> Any:
    """``body^n``: its blocks one after another, or a shared body n times."""
    key = node["body"].get("share")
    if key is not None:
        passes = [ctx.shared[key]] * node["times"]
    else:
        passes = getattr(block, block.scope.names[pos])
    for inner in passes:
        x = inner(x, ctx)
    return x


# Its annotations are strings, which vLLM's decorator does not read dims from.
@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class BuxModel(Scoped):
    """The model's block, at ``model``: every layer, and the graph from token ids to
    the first head's input.

    Raises:
        NotImplementedError: A layer, a setting of one or the residual algebra is
            not in vLLM yet.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        p = plan(config.bux, config.bux_reserved)
        layers = build_layers(vllm_config, p)
        super().__init__(p.top, "model.", layers)
        self.plan = p
        self.layers_by_path: dict[str, Any] = layers  # the blocks hold them
        self.shared = nn.ModuleDict(
            {
                key: layers[f"model.shared.{key}"]
                if s.root["kind"] == "layer"
                else Block(s, f"model.shared.{key}.", layers)
                for key, s in p.shared.items()
            }
        )
        embeds = [c.path for c in p.calls if c.layer["config"]["type"] == "embed"]
        self.head_path = live(p.calls)[-1].path  # the first head's
        self.embed_path = embeds[0] if embeds else None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The model's first embedding of the ids, before its learned positions."""
        assert self.embed_path is not None, "the model embeds no tokens"
        return self.layers_by_path[self.embed_path].embed(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ctx = Ctx(positions, self.shared, inputs_embeds, self.plan)
        run(self, self.plan.top.root, (), input_ids, ctx, 0)
        assert ctx.hidden is not None, "the model ran no head"
        return ctx.hidden


class BuxForCausalLM(nn.Module, IsHybrid):
    """A bux model in vLLM: `BuxModel`, and its first head's logits.

    It is a hybrid for vLLM where it has recurrent layers or convs on keys and
    values (``config.layer_types`` says so), whose states' shapes and dtypes it
    gives from the config (`bux_vllm.ssm`, `bux_vllm.conv`).
    """

    is_hybrid: ClassVar = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = BuxModel(vllm_config=vllm_config, prefix="model")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.model.layers_by_path[self.model.head_path].logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the checkpoint's weights, each by the layer at the longest prefix of
        its name; return the names of the parameters filled.

        Raises:
            ValueError: A weight belongs to no layer, or its layer has no such weight.
        """
        layers = self.model.layers_by_path
        filled: set[int] = set()
        for name, tensor in weights:
            path, rest = _owner(name, layers)
            try:
                filled |= {id(p) for p in layers[path].load(rest, tensor)}
            except KeyError as e:
                raise ValueError(f"{name}: {e}") from e
        return {n for n, p in self.named_parameters() if id(p) in filled}

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> tuple:
        return ssm.state_shapes(vllm_config)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple:
        return ssm.state_dtypes(vllm_config)

    @classmethod
    def get_mamba_specs_from_config(cls, vllm_config: VllmConfig) -> tuple:
        """Every shape of state the model keeps in vLLM's Mamba cache, by which vLLM
        sizes its pages: its recurrent layers', and its convs' on keys and values."""
        shapes = ssm.state_shapes(vllm_config)
        dtypes = ssm.state_dtypes(vllm_config) if shapes else ()
        specs = [MambaSpec(shapes=shapes, dtypes=dtypes, block_size=-1)]
        dtype = conv.state_dtype(vllm_config)
        for shape in conv.state_shapes(vllm_config):
            specs.append(MambaSpec(shapes=shape, dtypes=dtype, block_size=-1))
        return tuple(specs)

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple:
        return ssm.copy_funcs("MAMBA2")

    @classmethod
    def get_mamba_state_copy_funcs(cls, mamba_types: set) -> dict:
        return {kind: ssm.copy_funcs(kind.name) for kind in mamba_types}


def _owner(name: str, layers: dict[str, nn.Module]) -> tuple[str, str]:
    """The path of the layer whose weight `name` is, and the name under it.

    Raises:
        ValueError: No layer's path prefixes it.
    """
    parts = name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        path = ".".join(parts[:i])
        if path in layers:
            return path, ".".join(parts[i:])
    raise ValueError(f"the checkpoint's {name} belongs to no layer of the model")
