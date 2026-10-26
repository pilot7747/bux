"""The generated ``modeling_bux.py``: an architecture's graph as PyTorch code that
calls the exported layers (`bux.export.hf`).

Every scope of the plan (`bux.export.plan`) becomes a class whose ``__init__`` builds
its layers and whose ``forward`` runs its graph, a line per node: a sequence one
after another, ``[f]`` as ``x + f(x)``, a sum or product of its branches, a repeat as
a loop over a ``ModuleList`` of its body's class, ``@name`` and ``$name`` as pushes
and reads of the pass's taps, and a shared definition as a call of
``ctx.shared[key]``. Each distinct layer config is a constant at the top of the file.
"""

import importlib
import inspect
import pkgutil
import pprint
from dataclasses import dataclass, field
from functools import cache

from bux.export import hf
from bux.export.plan import Plan, Pos, Scope
from bux.lang import ir


@cache
def layer_classes() -> dict[str, type]:
    """Return every exported layer's class, by the bux layer type it implements."""
    out = {}
    for info in pkgutil.iter_modules(hf.__path__):
        module = importlib.import_module(f"{hf.__name__}.{info.name}")
        for value in vars(module).values():
            if isinstance(value, type) and value.__module__ == module.__name__:
                out |= dict.fromkeys(getattr(value, "bux_types", ()), value)
    return out


def layer_class(layer: ir.Layer) -> type:
    """Return the exported class of `layer`'s type.

    Raises:
        NotImplementedError: No exported layer implements the type yet.
    """
    cls = layer_classes().get(layer.config.type)
    if cls is None:
        raise NotImplementedError(
            f"no exported layer implements bux's {layer.config.type!r} yet"
        )
    return cls


@dataclass
class _Class:
    """The lines of one generated class: its ``__init__``'s and its ``forward``'s."""

    init: list[str] = field(default_factory=list)
    body: list[str] = field(default_factory=list)
    temps: int = 0

    def let(self, expr: str) -> str:
        """Append ``h<n> = expr`` to the forward, and return ``h<n>``."""
        self.temps += 1
        self.body.append(f"h{self.temps} = {expr}")
        return f"h{self.temps}"


@dataclass
class Generator:
    """Generates the classes of a plan's scopes, and the constants they use."""

    plan: Plan
    classes: list[str] = field(default_factory=list)
    constants: dict[str, str] = field(default_factory=dict)  # config source -> name

    def constant(self, layer: ir.Layer) -> str:
        """Return the name of the constant holding `layer`'s config, made once."""
        source = pprint.pformat(layer.config.model_dump(mode="json"), sort_dicts=False)
        if source not in self.constants:
            hint = (layer.label or layer.config.type).upper()
            hint = hint if hint.isidentifier() else layer.config.type.upper()
            taken = set(self.constants.values())
            name, i = hint, 1
            while name in taken:
                name, i = f"{hint}_{i}", i + 1
            self.constants[source] = name
        return self.constants[source]

    def leaf(self, layer: ir.Layer, path: str) -> str:
        """Return the expression that builds `layer`'s module at `path`, an
        expression of the class's ``prefix``."""
        cls = layer_class(layer).__name__
        return f"{cls}({self.constant(layer)}, {layer.width!r}, site(config, {path}))"

    def scope(self, s: Scope, name: str, doc: str) -> str:
        """Generate the class `name` of the scope `s`; return its name."""
        index = len(self.classes)
        self.classes.append("")  # its place, so that a repeat inside is named after
        c = _Class()
        out = self.emit(s.root, "x", (), s, c, 0)
        init = "\n        ".join(c.init) or "pass"
        body = "\n        ".join([*c.body, f"return {out}"])
        self.classes[index] = (
            f"class {name}(nn.Module):\n"
            f'    """bux: {doc}."""\n\n'
            f"    def __init__(self, config: BuxConfig, prefix: str):\n"
            f"        super().__init__()\n        {init}\n\n"
            f"    def forward(self, x: Any, ctx: Ctx) -> Any:\n        {body}\n"
        )
        return name

    def emit(
        self, node: ir.Node, x: str, pos: Pos, s: Scope, c: _Class, depth: int
    ) -> str:
        """Append to `c` the code of `node`, at `pos` in `s`, applied to the variable
        `x`; return the variable or expression of its output."""
        if node.share is not None and not (
            pos == () and s is self.plan.shared.get(node.share)
        ):
            return c.let(f"ctx.shared[{node.share!r}]({x}, ctx)")
        match node:
            case ir.Layer():
                attr = s.names[pos]
                c.init.append(f"self.{attr} = {self.leaf(node, f'prefix + {attr!r}')}")
                return c.let(f"self.{attr}({x}, ctx)")
            case ir.Seq(items=items):
                for i, item in enumerate(items):
                    x = self.emit(item, x, (*pos, i), s, c, depth)
                return x
            case ir.Residual(body=body):
                return self.residual(body, x, pos, s, c, depth)
            case ir.Sum(terms=items) | ir.Prod(factors=items):
                op = " + " if isinstance(node, ir.Sum) else " * "
                ys = [
                    self.emit(t, x, (*pos, i), s, c, depth) for i, t in enumerate(items)
                ]
                return c.let(op.join(ys))
            case ir.Neg(body=body):
                return c.let(f"-{self.emit(body, x, (*pos, 0), s, c, depth)}")
            case ir.Const(value=value):
                return repr(value)
            case ir.Identity():
                return x
            case ir.Tap(name=name):
                c.body.append(f"ctx.push({name!r}, {x})")
                return x
            case ir.Ref(name=name, pop=pop):
                return c.let(f"ctx.read({name!r}, {pop})")
            case ir.Repeat():
                return self.repeat(node, x, pos, s, c)
        raise AssertionError(node)

    def residual(
        self, body: ir.Node, x: str, pos: Pos, s: Scope, c: _Class, depth: int
    ) -> str:
        """``[body]``: ``x + body(x)``."""
        if self.plan.arch.residual is not None and depth == 0:
            kind = self.plan.arch.residual.type
            raise NotImplementedError(
                f"the residual algebra {kind!r} is not exported yet"
            )
        return c.let(f"{x} + {self.emit(body, x, (*pos, 0), s, c, depth + 1)}")

    def repeat(self, node: ir.Repeat, x: str, pos: Pos, s: Scope, c: _Class) -> str:
        """``body^n``: a loop over a list of the body's class, or over calls of a
        shared body."""
        y = c.let(x)
        if node.body.share is not None:
            c.body += [
                f"for _ in range({node.times}):",
                f"    {y} = ctx.shared[{node.body.share!r}]({y}, ctx)",
            ]
            return y
        attr = s.names[pos]
        label = node.body.label or "a repeated block"
        cls = self.scope(s.blocks[pos], f"Block{len(self.classes)}", label)
        passes = (
            f'[{cls}(config, f"{{prefix}}{attr}.{{i}}.") for i in range({node.times})]'
        )
        c.init.append(f"self.{attr} = nn.ModuleList({passes})")
        c.body += [f"for block in self.{attr}:", f"    {y} = block({y}, ctx)"]
        return y

    def shared(self) -> str:
        """Return the expression of the ModuleDict of every shared definition."""
        items = []
        for key, s in self.plan.shared.items():
            path = f"model.shared.{key}"
            if isinstance(s.root, ir.Layer):
                items.append(f"{key!r}: {self.leaf(s.root, repr(path))}")
            else:
                name = self.scope(s, f"Shared{len(self.classes)}", key)
                items.append(f'{key!r}: {name}(config, "{path}.")')
        return "nn.ModuleDict({" + ", ".join(items) + "})"


HEADER = '''"""A bux architecture as PyTorch: generated by `bux export` from its IR; do not
edit. The layers are bux's exported layers, in the files beside this one."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_bux import BuxConfig
'''

MODEL = '''
def seen_tokens(cache: Any, t: int) -> int:
    """Return how many tokens `cache` holds before the t of this forward: as its
    attention layers count them, or, where it has none, as the model counts them."""
    if cache is None or COUNTS_TOKENS:
        return 0 if cache is None else cache.get_seq_length()
    seen = getattr(cache, "bux_seen", 0)
    cache.bux_seen = seen + t
    return seen


class BuxPreTrainedModel(PreTrainedModel):
    config_class = BuxConfig
    base_model_prefix = "model"
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module: nn.Module) -> None:
        """Set what no checkpoint holds; a converted one brings every weight."""
        if hasattr(module, "reset"):
            module.reset()


class BuxModel(BuxPreTrainedModel):
    """bux: the graph of the architecture, from token ids to every head's logits."""

    def __init__(self, config: BuxConfig):
        super().__init__(config)
        prefix = "model."
        self.shared = {shared}
        {init}
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> tuple[list[torch.Tensor], Any]:
        b, t = input_ids.shape
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        seen = seen_tokens(past_key_values, t)
        if position_ids is None:
            position_ids = torch.arange(seen, seen + t, device=input_ids.device)
            position_ids = position_ids[None].expand(b, t)
        ctx = Ctx(input_ids, position_ids, attention_mask, past_key_values, self.shared, make_mask)
        ctx.dtype, ctx.config = self.dtype, self.config
        for kind, arg in MASKS:
            ctx.mask(kind, arg)
        x = input_ids
        {body}
        return ctx.heads, past_key_values


class BuxForCausalLM(BuxPreTrainedModel, GenerationMixin):
    """The model, and the first head's logits as transformers' causal LM output."""

    def __init__(self, config: BuxConfig):
        super().__init__(config)
        self.model = BuxModel(config)
        self.post_init()

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        """Whether generate may make the cache: not where no cache layer is an
        attention layer's, which would know how many tokens it holds; the model's
        first forward makes it then."""
        return COUNTS_TOKENS

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = None,
        logits_to_keep: int = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        heads, cache = self.model(
            input_ids, attention_mask, position_ids, past_key_values, use_cache, **kwargs
        )
        logits = heads[0][:, -logits_to_keep:] if logits_to_keep else heads[0]
        return CausalLMOutputWithPast(logits=logits, past_key_values=cache)
'''


def modeling(plan: Plan, masks: list[tuple[str, object]]) -> str:
    """Return the source of ``modeling_bux.py`` for `plan`.

    Args:
        plan: The exported model's module tree.
        masks: The attention masks its layers read, as ``(kind, arg)``: the model
            makes them before any layer runs, since they depend on the cache.
    """
    g = Generator(plan)
    shared = g.shared()
    top = _Class()
    out = g.emit(plan.top.root, "x", (), plan.top, top, 0)
    del out  # the heads are the outputs
    code = MODEL.replace("{shared}", shared)
    code = code.replace("{init}", "\n        ".join(top.init) or "pass")
    code = code.replace("{body}", "\n        ".join(top.body))
    imports = sorted(
        {(c.__module__.rsplit(".", 1)[1], c.__name__) for c in _used(plan)}
    )
    lines = [f"from .{m} import {n}" for m, n in imports]
    lines += ["from .common import Ctx, site", "from .attention import make_mask"]
    consts = [f"{name} = {source}" for source, name in g.constants.items()]
    return "\n".join(
        [
            HEADER + "\n".join(sorted(set(lines))),
            "",
            "\n\n".join(consts),
            f"MASKS = {masks!r}",
            f"COUNTS_TOKENS = {_counts_tokens(plan)!r}",
            "",
            "\n\n".join(g.classes),
            code,
        ]
    )


def _used(plan: Plan) -> set[type]:
    """Return the exported classes of every layer in `plan`."""
    return {layer_class(call.layer) for call in plan.calls}


def cache_layer(layer: ir.Layer) -> tuple[str, dict] | None:
    """Return the kind of the cache layer each call of `layer` keeps, and what the
    config sets for it alone (``per_layer_config``), or None for none.

    Its class's ``cache_type(c)``, or ``cache_type(c, ports)`` where its ports decide
    (as ``[name, tap, mode]``), returns the kind, ``(kind, settings)``, or None.
    """
    fn = getattr(layer_class(layer), "cache_type", None)
    if fn is None:
        return None
    c = layer.config.model_dump(mode="json")
    ports = [[port.name, port.tap, port.mode] for port in layer.ports]
    kind = fn(c, ports) if len(inspect.signature(fn).parameters) > 1 else fn(c)
    if kind is None or isinstance(kind, tuple | list):
        return None if kind is None else (kind[0], dict(kind[1]))
    return kind, {}


def _counts_tokens(plan: Plan) -> bool:
    """Whether a cache layer of the model counts the tokens it holds: an attention
    layer's does, a recurrent layer's (``conv``, ``linear_attention``) does not."""
    for call in plan.calls:
        kept = cache_layer(call.layer)
        if kept is not None and kept[0] not in ("conv", "linear_attention"):
            return True
    return False
