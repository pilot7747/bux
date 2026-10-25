"""Print an Architecture, or a whole Run, back to .bux source.

Every labeled node becomes a definition, so the output reads like hand-written source:
leaf definitions, then blocks in dependency order, then the model. Macro calls are
inlined, since the IR only keeps what they produced. ``compile(to_source(a))``
equals ``a`` up to labels, and a run's source compiles back to a run with its id.

`to_source` prints an architecture and `run_to_source` a run (``bux fmt`` prints
either); `config_to_source` and `value_to_source` print one config or one value, as
hovers and the language reference show them.
"""

import re
from functools import cache

from pydantic import BaseModel

from bux import config as cfg
from bux.lang import ir
from bux.lang.compiler import ACTIVATIONS, SETTINGS, TRAIN_SETTINGS, TYPES
from bux.lang.syntax import KEYWORDS
from bux.train import config as tcfg

_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
SUM, PROD, SEQ, UNARY, POWER, ATOM = range(6)


@cache
def field_order(cls: type[BaseModel]) -> tuple[str, ...]:
    """Return the class's own fields, then those it inherits: gqa(num_heads=...)
    before window sizes."""
    order: list[str] = []
    for klass in cls.__mro__:
        is_model = isinstance(klass, type) and issubclass(klass, BaseModel)
        if not is_model or klass is BaseModel:
            continue
        parent = next(
            (b for b in klass.__bases__ if issubclass(b, BaseModel)), BaseModel
        )
        order += [
            name
            for name in klass.model_fields
            if name not in parent.model_fields and name not in order
        ]
    return tuple(order)


def _key(node: ir.Node) -> str:
    if isinstance(node, ir.Layer):
        return f"{node.share}:{node.config.model_dump_json()}"
    return ir.strip(node).model_dump_json()


class _Printer:
    def __init__(self, arch: ir.Architecture):
        self.arch = arch
        # Definitions may not shadow builtins or settings, so that inline configs and
        # bare strings keep their meaning.
        taken = (
            {*SETTINGS, *TRAIN_SETTINGS, "model"} | KEYWORDS | set(TYPES) | ACTIVATIONS
        )
        self.names: dict[str, str] = {}
        for node in ir.walk(arch.graph):
            if node is arch.graph or node.label is None or not _IDENT.match(node.label):
                continue
            key = _key(node)
            if key not in self.names:
                name, i = node.label, 2
                while name in taken:
                    name, i = f"{node.label}_{i}", i + 1
                taken.add(name)
                self.names[key] = name
        self.defined = set(self.names.values())
        # Settings equal to a leaf definition print as its name: qk_norm=norm.
        self.leaf_names = {
            node.config.model_dump_json(): self.names[_key(node)]
            for node in ir.walk(arch.graph)
            if isinstance(node, ir.Layer)
            and node.share is None
            and _key(node) in self.names
        }
        self.emitted: set[str] = set()
        self.leaves: list[str] = []
        self.blocks: list[str] = []

    def architecture(self) -> str:
        settings = [f"d_model = {self.arch.d_model}"]
        if self.arch.vocab_size is not None:
            settings.append(f"vocab_size = {self.arch.vocab_size}")
        if self.arch.residual is not None:
            settings.append(f"residual = {self.config(self.arch.residual)}")
        model = f"model = {self.expr(self.arch.graph, top=True)[0]}"
        sections = [settings, self.leaves, self.blocks, [model]]
        return "\n\n".join("\n".join(lines) for lines in sections if lines) + "\n"

    def name(self, node: ir.Node) -> str | None:
        return self.names.get(_key(node)) if node.label is not None else None

    def reference(self, node: ir.Node) -> str | None:
        name = self.name(node)
        if name is None or name in self.emitted:
            return name
        self.emitted.add(name)
        prefix = "shared " if node.share else ""
        if isinstance(node, ir.Layer):
            self.leaves.append(f"{prefix}{name} = {self.config(node.config)}")
        else:
            body = node.model_copy(update={"label": None, "share": None})
            self.blocks.append(f"{prefix}{name} = {self.expr(body, top=True)[0]}")
        return name

    def expr(self, node: ir.Node, top: bool = False) -> tuple[str, int]:
        if not top and (name := self.reference(node)) is not None:
            if isinstance(node, ir.Layer) and node.ports:
                return f"{name}({', '.join(self.ports(node))})", ATOM
            return name, ATOM
        match node:
            case ir.Layer(config=config):
                return self.config(config, self.ports(node)), ATOM
            case ir.Seq(items=items):
                # Only the first item may start with "-": `a -b` would read as a - b.
                parts = [
                    self.wrap(item, UNARY if i == 0 else POWER)
                    for i, item in enumerate(self.spliced(items))
                ]
                return " ".join(parts), SEQ
            case ir.Residual(body=body):
                return f"[{self.expr(body)[0]}]", ATOM
            case ir.Sum(terms=terms):
                parts = []
                for i, term in enumerate(terms):
                    if isinstance(term, ir.Neg) and term.label is None:
                        parts.append(
                            ("-" if i == 0 else " - ")
                            + self.wrap(term.body, PROD if i else UNARY)
                        )
                    elif isinstance(term, ir.Const) and term.value < 0 and i:
                        parts.append(f" - {-term.value!r}")
                    else:
                        parts.append(("" if i == 0 else " + ") + self.wrap(term, PROD))
                return "".join(parts), SUM
            case ir.Prod(factors=factors):
                return " * ".join(self.wrap(f, SEQ) for f in factors), PROD
            case ir.Neg(body=body):
                return f"-{self.wrap(body, UNARY)}", UNARY
            case ir.Const(value=value):
                return repr(value), (UNARY if value < 0 else ATOM)
            case ir.Identity():
                return "_", ATOM
            case ir.Repeat(body=body, times=times):
                return f"{self.wrap(body, ATOM)}^{times}", POWER
            case ir.Tap(name=name):
                return f"@{name}", ATOM
            case ir.Ref(name=name, pop=pop):
                return f"${name}{'!' if pop else ''}", ATOM
        raise AssertionError(node)

    def spliced(self, items: tuple[ir.Node, ...]) -> list[ir.Node]:
        """Splice unnamed sequences inside a sequence, such as macro calls, inline."""
        out: list[ir.Node] = []
        for item in items:
            if isinstance(item, ir.Seq) and self.name(item) is None:
                out += self.spliced(item.items)
            else:
                out.append(item)
        return out

    def wrap(self, node: ir.Node, min_prec: int) -> str:
        text, prec = self.expr(node)
        return text if prec >= min_prec else f"({text})"

    def ports(self, node: ir.Layer) -> list[str]:
        sigil = {"write": "@", "read": "$", "pop": "$"}
        return [
            f"{p.name}={sigil[p.mode]}{p.tap}{'!' if p.mode == 'pop' else ''}"
            for p in node.ports
        ]

    def config(
        self,
        config: cfg.Config,
        extra: list[str] | None = None,
        *,
        as_value: bool = False,
    ) -> str:
        if isinstance(config, cfg.ActivationConfig) and not extra and not as_value:
            return config.fn
        if isinstance(config, tcfg.MixtureConfig):
            return self.value(config)
        args = []
        fields = type(config).model_fields
        for name in field_order(type(config)):
            field, value = fields[name], getattr(config, name)
            if name == "type" or (
                not field.is_required()
                and value == field.get_default(call_default_factory=True)
            ):
                continue
            args.append(f"{name}={self.value(value)}")
        args += extra or []
        tag = getattr(config, "type")  # noqa: B009
        return f"{tag}({', '.join(args)})" if args or as_value else tag

    def value(self, v: object) -> str:
        match v:
            case bool():
                return "true" if v else "false"
            case None:
                return "none"
            case int():
                return str(v)
            case float():
                return repr(v)
            case str():
                bare = (
                    _IDENT.match(v)
                    and v not in KEYWORDS
                    and v not in TYPES
                    and v not in self.defined
                )
                return v if bare else ('"' + v + '"' if '"' not in v else "'" + v + "'")
            case tcfg.MixtureConfig(sources=sources):
                return " + ".join(
                    f"{w.weight!r} * {self.value(w.source)}" for w in sources
                )
            case cfg.Config():
                return self.leaf_names.get(v.model_dump_json()) or self.config(
                    v, as_value=True
                )
            case list() | tuple():
                items = [self.value(x) for x in v]
                return f"[{items[0]},]" if len(items) == 1 else f"[{', '.join(items)}]"
        raise TypeError(f"cannot print {v!r}")


def to_source(arch: ir.Architecture) -> str:
    """Print an architecture as .bux source, which compiles back to it.

    The settings come first, then a definition for each labeled leaf and block, in
    dependency order, then the model. Settings at their defaults are left out, and
    arithmetic shows its result (``4 * d_model`` prints as ``256``).

    Args:
        arch: What `bux.lang.compile` returned, or any architecture.

    Returns:
        The source, ending with a newline. ``compile(to_source(arch))`` equals
        `arch` up to labels.

    Example::

        print(to_source(compile(open("examples/experiments/smoke.bux").read())))
        # d_model = 64
        # vocab_size = 512
        #
        # norm = rmsnorm
        # attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
        # ffn = gated_mlp(intermediate_size=256, activation=swish)
        #
        # block = [norm attn] [norm ffn]
        #
        # model = embed block^2 norm head(tied=true)
    """
    return _Printer(arch).architecture()


def run_to_source(run: ir.Run) -> str:
    """Print a run as one file: no modules, no alternatives, nothing left to resolve.

    This is the ``run.bux`` a run's directory keeps, and what ``bux fmt`` prints for
    each run of a sweep. A first comment names the run and its id; the architecture
    follows as `to_source` prints it, then the training settings. Those at their
    defaults are left out, as config settings are.

    Args:
        run: One run, as `bux.lang.compile_runs` returns them.

    Returns:
        The source, which compiles back to a run with the same id.
    """
    printer = _Printer(run.arch)
    arch = printer.architecture()
    fields = tcfg.TrainConfig.model_fields
    settings = [
        f"{name} = {printer.value(getattr(run.train, name))}"
        for name in TRAIN_SETTINGS
        if fields[name].is_required()
        or getattr(run.train, name)
        != fields[name].get_default(call_default_factory=True)
    ]
    return f"# {run.name}, run {run.id}\n{arch}\n" + "\n".join(settings) + "\n"


# A printer with no definitions to refer to, so that values print in full.
_PLAIN = ir.Architecture(d_model=1, graph=ir.Identity())


def config_to_source(config: cfg.Config) -> str:
    """Print one config as source, without its defaults.

    Args:
        config: A layer's or a setting's config.

    Returns:
        A call such as ``gqa(num_heads=8, num_kv_heads=8)`` or ``adamw(lr=0.003)``,
        or the bare name where every setting is at its default, ``rmsnorm``.
    """
    return _Printer(_PLAIN).config(config)


def value_to_source(value: object) -> str:
    """Print a setting's value as source, as the language reference shows defaults.

    Args:
        value: A bool, None, a number, a string, a config, a mix of sources, or a
            list or tuple of them.

    Returns:
        The value as a ``.bux`` file writes it, such as ``1e-05``, ``swish``, ``none``,
        ``rope()``, ``[1, 2]``.

    Raises:
        TypeError: For a value that has no source, such as a dict.
    """
    return _Printer(_PLAIN).value(value)
