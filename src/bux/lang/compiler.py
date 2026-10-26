"""Compile .bux source into the architecture IR.

Two passes. Evaluation resolves names, macros, layer settings and patterns into an
untyped graph. Lowering then walks it in execution order, with repeats unrolled, to
infer every layer's input width, check residuals and parallel branches for width
agreement, simulate the tap stacks, and check that shared blocks are used consistently.

A program can span files (``use``), and its alternatives (``a | b``) make it a sweep
of several runs: evaluation takes one alternative of each `Axis`.

`compile` stops at the first error. `check` is for editors: it recovers from
syntax errors and reports every definition that fails, in every alternative.
`compile_runs` builds every run of a sweep, with its training settings.

The tables of names below (`TYPES`, `LAYERS`, `ACTIVATIONS`, ...) are what a
``.bux`` file can call. They come from the config unions of `bux.config` and
`bux.train.config`, so a config added to one is a name of the language at once.
"""

import builtins
import difflib
import operator
import re
import typing
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from bux import config as cfg
from bux.lang import ir, modules
from bux.lang import syntax as ast
from bux.lang.syntax import BuxError, Span
from bux.train import config as tcfg


def _tags(union: object) -> dict[str, type[cfg.Config]]:
    classes = typing.get_args(typing.get_args(union)[0])
    return {
        tag: cls
        for cls in classes
        for tag in typing.get_args(cls.model_fields["type"].annotation)
    }


#: The layers' config classes, by the names a ``.bux`` file calls them
#: (``gqa``, ``moe``, ...).
#:
#: :meta hide-value:
LAYERS = _tags(cfg.LayerConfig)
#: The training settings' config classes by name (``adamw``, ``wsd``, ``text``).
#:
#: :meta hide-value:
TRAINING = _tags(tcfg.Setting)
#: Every config class a ``.bux`` file can call by name, in one table.
#:
#: The layers, their parts (``rope``), the residual algebras and the training
#: settings. The ``{bux}`` role of these docs links to any name in it.
#:
#: :meta hide-value:
TYPES = LAYERS | _tags(cfg.SubConfig) | TRAINING
_SOURCES = frozenset(t for t, cls in TRAINING.items() if issubclass(cls, tcfg.SOURCES))
#: The activations, each a layer (``linear(features=h) gelu``) or a setting's value.
ACTIVATIONS = frozenset(typing.get_args(cfg.Activation))
#: The residual algebras that ``residual = ...`` takes, by name.
#:
#: :meta hide-value:
RESIDUALS = _tags(cfg.ResidualConfig)
#: The top-level settings of a model.
SETTINGS = ("d_model", "vocab_size", "residual", "init")
#: The top-level settings that make a file a run, the fields of
#: `bux.train.config.TrainConfig` (``data``, ``optimizer``, ``schedule``, ...).
TRAIN_SETTINGS = tuple(tcfg.TrainConfig.model_fields)
_RESERVED = (*SETTINGS, *TRAIN_SETTINGS, "model")
_REQUIRED = tuple(
    name for name, f in tcfg.TrainConfig.model_fields.items() if f.is_required()
)
# Settings of names that layers give what they make, which the compiler reads as
# written (``_Evaluator.names``).
_REMAT_NAMES = (("save",), ("offload",))
_EXAMPLES = {
    "data": 'data = text(path="data/fineweb/*.parquet")',
    "optimizer": "optimizer = adamw(lr=3e-4)",
    "seq_len": "seq_len = 1024",
    "batch": "batch = 256",
    "steps": "steps = 10k",
}


@dataclass(frozen=True)
class _Named:
    """A value reached through a definition or a macro call: its graphs carry the label."""

    label: str
    value: object
    share: str | None = None


@dataclass(frozen=True)
class _Builtin:
    name: str  # a config type tag or an activation


@dataclass(frozen=True)
class _Bare:
    """An unresolved name: a string in argument position, an error anywhere else."""

    name: str


@dataclass(frozen=True)
class _Macro:
    statement: ast.Statement


@dataclass(frozen=True)
class _PortValue:
    port: str


@dataclass(frozen=True)
class Axis:
    """One axis of a sweep: one ``a | b`` as written, or all the alternatives with the
    same labels, which are picked together.

    ``optimizer.lr = 1e-3 | 3e-3`` is an axis of its own. Labeled alternatives with
    the same labels are one axis wherever they are written: with ``attn = mha: ... |
    mqa: ...`` and ``optimizer.lr = mha: 1e-3 | mqa: 3e-3``, a run takes ``mha`` in
    both places or ``mqa`` in both. `Report.axes` lists a file's axes.

    Attributes:
        key: The span of the ``a | b`` for an unlabeled axis, the set of its labels
            for a labeled one.
        where: The definition and setting of the first alternative met, which
            names the runs of an unlabeled axis: ``"optimizer.lr"``.
        options: The labels, or the alternatives as written: ``("1e-3", "3e-3")``.
        span: Where the first alternative met is, in the joined source.
    """

    key: Span | frozenset[str]
    where: str
    options: tuple[str, ...]
    span: Span

    @property
    def labeled(self) -> bool:
        """Whether the alternatives have labels, and are picked by them."""
        return isinstance(self.key, frozenset)

    def pick(self, option: int) -> int | str:
        """Return what ``choices`` holds for `option`: its label or index."""
        return self.options[option] if self.labeled else option

    def part(self, pick: int | str) -> str:
        """Return how a run that took `pick` is named: mqa, or optimizer.lr=1e-3."""
        if isinstance(pick, str):
            return pick
        return f"{self.where}={self.options[pick]}"


class _Reported(BuxError):
    """A repeat, at a use, of an error already reported; check() drops it."""

    def __init__(self) -> None:
        super().__init__("already reported")


_ARITHMETIC: dict[str, Callable[[float, float], float]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "^": operator.pow,
}


def _tag(config: cfg.Config) -> str:
    # Every concrete config has a type tag; the base class does not.
    return getattr(config, "type")  # noqa: B009


def _unwrap(value: object) -> object:
    while isinstance(value, _Named):
        value = value.value
    return value


def _config_type(value: object) -> type[cfg.Config] | None:
    """Return the class of a config, or of a bare layer or setting type."""
    match value:
        case cfg.Config():
            return type(value)
        case _Builtin(name=name) if name in TYPES:
            return TYPES[name]
    return None


def _is_number(value: object) -> bool:
    value = _unwrap(value)
    return isinstance(value, int | float) and not isinstance(value, bool)


def _activation(name: str) -> cfg.ActivationConfig:
    return cfg.ActivationConfig(fn=typing.cast(cfg.Activation, name))


def _flatten(nodes: list[ir.Node], cls: type, children: str) -> tuple[ir.Node, ...]:
    out: list[ir.Node] = []
    for node in nodes:
        if isinstance(node, cls) and node.label is None and node.share is None:
            out.extend(getattr(node, children))
        else:
            out.append(node)
    return tuple(out)


def _describe(value: object) -> str:
    match _unwrap(value):
        case bool():
            return "a boolean"
        case int() | float():
            return "a number"
        case str():
            return "a string"
        case None:
            return "none"
        case list():
            return "a list"
        case cfg.Config() as config:
            return f"a {_tag(config)} config"
        case _Macro(statement=st):
            return f"the macro '{st.name}'"
        case _:
            return "a block"


def _weighted(source: tcfg.DataConfig) -> tuple[tcfg.Weighted, ...]:
    if isinstance(source, tcfg.MixtureConfig):
        return source.sources
    return (tcfg.Weighted(weight=1.0, source=source),)


def _assign(data: dict, key: tuple[str, ...], value: object) -> None:
    for part in key[:-1]:
        inner = data.get(part)
        if isinstance(inner, cfg.Config):
            inner = inner.model_dump()
        elif not isinstance(inner, dict):
            inner = {}
        data[part] = inner
        data = inner
    data[key[-1]] = value


def _port(
    key: tuple[str, ...], tap: ir.Tap | ir.Ref, cls: type[cfg.Config], span: Span
) -> ir.Port:
    if len(key) > 1 or key[0] not in cls.ports:
        available = ", ".join(sorted(cls.ports)) or "none"
        raise BuxError(f"no port '{'.'.join(key)}' (ports: {available})", span)
    mode = "write" if isinstance(tap, ir.Tap) else ("pop" if tap.pop else "read")
    return ir.Port(name=key[0], tap=tap.name, mode=mode)


_Fields = list[tuple[tuple[str, ...], object, Span]]


class _Evaluator:
    def __init__(
        self,
        program: modules.Program,
        *,
        report: Callable[[BuxError], None] | None = None,
        choices: Mapping[Span | frozenset[str], int | str] | None = None,
    ):
        """`report` collects statement errors instead of raising them, for check().
        `choices` picks alternatives by Axis key; the others take their first."""
        self.source = program.source
        # Names of definitions that failed to parse, whose uses are not errors.
        self.broken = set(program.broken)
        # A module failed to load: names that it might have defined are not errors.
        self.incomplete = program.incomplete
        # Every definition of each name, lowest layer first.
        self.chain: dict[str, list[ast.Statement]] = {}
        for layer in program.layers:
            self.add_layer(layer, report)
        # The definition in effect for each name: the one in the highest layer.
        self.statements = {name: chain[-1] for name, chain in self.chain.items()}
        self.values: dict[str, object] = {}
        # Values of the definitions that later layers amend or replace.
        self.replaced: dict[tuple[str, int], object] = {}
        # The definitions being evaluated, with their places in their chains.
        self.evaluating: list[str] = []
        self.levels: list[int] = []
        self.depth = 0
        self.choices = dict(choices or {})
        self.axes: dict[Span | frozenset[str], Axis] = {}  # met so far, in order
        # Source spans of graph nodes, for errors found while lowering. The list keeps
        # the nodes alive so that their ids stay unique.
        self.node_spans: dict[int, Span] = {}
        self._alive: list[object] = []
        # Settings given as an undefined bare name, which became a string: arg span ->
        # (name, its span, names in scope), reported as unknown names if invalid.
        self.bare: dict[Span, tuple[str, Span, tuple[str, ...]]] = {}

    def add_layer(
        self, layer: list[ast.Statement], report: Callable[[BuxError], None] | None
    ) -> None:
        names: set[str] = set()
        for st in layer:
            if st.name in names:
                error = BuxError(f"'{st.name}' is defined twice", st.name_span)
            elif st.params is not None and st.name in _RESERVED:
                error = BuxError(f"'{st.name}' cannot be a macro", st.name_span)
                self.broken.add(st.name)
            else:
                names.add(st.name)
                chain = self.chain.setdefault(st.name, [])
                if st.amends and chain:
                    st = replace(st, shared=chain[-1].shared)
                chain.append(st)
                continue
            if report is None:
                raise error
            report(error)

    def node[N](self, node: N, span: Span) -> N:
        self.node_spans[id(node)] = span
        self._alive.append(node)
        return node

    def text(self, span: Span) -> str:
        return " ".join(self.source[span.start : span.end].split())

    # --- Names ------------------------------------------------------------------

    def definition(self, name: str, level: int | None = None) -> object:
        """Evaluate the definition of `name` in effect, or the one at `level` of its
        chain, once, and return its value."""
        chain = self.chain[name]
        top = len(chain) - 1
        level = top if level is None else level
        st = chain[level]
        if st.params is not None:
            return _Macro(st)
        cached = name in self.values if level == top else (name, level) in self.replaced
        if not cached:
            self.evaluating.append(name)
            self.levels.append(level)
            try:
                value = self.eval(st.body, {})
            finally:
                self.evaluating.pop()
                self.levels.pop()
            if level == top:
                self.values[name] = value
            else:
                self.replaced[name, level] = value
        value = self.values[name] if level == top else self.replaced[name, level]
        return _Named(name, value, name if st.shared else None)

    def lookup(self, name: str, span: Span, scope: dict[str, object]) -> object:
        if name in scope:
            return scope[name]
        if name in self.broken and name not in self.statements:
            raise _Reported()
        if name in self.statements:
            if name not in self.evaluating:
                return self.definition(name)
            # Inside its own definition a name means what it meant before it: the
            # definition this one amends or replaces, else the builtin: mlp = mlp(...).
            inner = len(self.evaluating) - 1 - self.evaluating[::-1].index(name)
            level = self.levels[inner]
            if level > 0:
                return self.definition(name, level - 1)
            if name not in TYPES and name not in ACTIVATIONS:
                if self.incomplete:  # the module that failed may have defined it
                    raise _Reported()
                if self.chain[name][level].amends:
                    raise BuxError(
                        f"nothing defines '{name}' for this to amend",
                        span,
                        f"define it first: {name} = ...",
                    )
                cycle = self.evaluating[self.evaluating.index(name) :] + [name]
                raise BuxError(f"definition cycle: {' -> '.join(cycle)}", span)
        if name in TYPES or name in ACTIVATIONS:
            return _Builtin(name)
        return _Bare(name)

    def unknown(
        self,
        name: str,
        span: Span,
        scope: typing.Iterable[str] = (),
        what: str = "layer or block",
        called: bool = False,
    ) -> BuxError:
        if self.incomplete:
            return _Reported()
        known = [
            *scope,
            *self.statements,
            *TYPES,
            *ACTIVATIONS,
            "true",
            "false",
            "none",
        ]
        # The name itself can be in scope where it was not visible: blk(a=b, b=1).
        close = difflib.get_close_matches(name, [k for k in known if k != name], n=1)
        hint = f"did you mean '{close[0]}'?" if close else None
        if called:
            new = (
                f"a new layer '{name}' needs a config in the LayerConfig union of "
                f"bux/config.py and a builder under @bux.architecture.register({name!r})"
            )
            hint = f"{hint} Or: {new}" if hint else new
        return BuxError(f"unknown {what} '{name}'", span, hint)

    # --- Expressions --------------------------------------------------------------

    def eval(self, expr: ast.Expr, scope: dict[str, object]) -> object:
        match expr:
            case ast.Lit(value=value):
                return value
            case ast.Name(name=name):
                return self.lookup(name, expr.span, scope)
            case ast.Identity():
                return self.node(ir.Identity(), expr.span)
            case ast.Tap(name=name):
                return self.node(ir.Tap(name=name), expr.span)
            case ast.Ref(name=name, pop=pop):
                return self.node(ir.Ref(name=name, pop=pop), expr.span)
            case ast.ListExpr(items=items):
                return [self.data(self.eval(item, scope), item.span) for item in items]
            case ast.Residual(body=body):
                return self.node(
                    ir.Residual(body=self.graph(self.eval(body, scope), body.span)),
                    expr.span,
                )
            case ast.Seq(items=items):
                nodes = [
                    self.graph(self.eval(item, scope), item.span) for item in items
                ]
                return self.node(
                    ir.Seq(items=_flatten(nodes, ir.Seq, "items")), expr.span
                )
            case ast.Neg(operand=operand):
                value = self.eval(operand, scope)
                if _is_number(value):
                    return -typing.cast(float, _unwrap(value))
                return self.node(
                    ir.Neg(body=self.graph(value, operand.span)), expr.span
                )
            case ast.BinOp():
                return self.binop(expr, scope)
            case ast.Call():
                return self.call(expr, scope)
            case ast.Attr():
                return self.attr(expr, scope)
            case ast.Alt():
                return self.eval(self.choose(expr), scope)
        raise AssertionError(expr)

    def choose(self, alt: ast.Alt) -> ast.Expr:
        """Return the alternative that `choices` picks, or the first."""
        key: Span | frozenset[str] = alt.span
        if alt.labels:
            key = frozenset(alt.labels)
            for other in self.axes:
                if isinstance(other, frozenset) and other != key and other & key:
                    raise BuxError(
                        f"these alternatives share some labels with others, but not "
                        f"all: {', '.join(sorted(key))} against {', '.join(sorted(other))}",
                        alt.span,
                        "alternatives with the same labels are picked together, so "
                        "give them all the same labels, or none in common",
                    )
        if key not in self.axes:
            options = alt.labels or tuple(self.text(o.span) for o in alt.options)
            self.axes[key] = Axis(key, alt.where, options, alt.span)
        pick = self.choices.get(key, 0)
        return alt.options[alt.labels.index(pick) if isinstance(pick, str) else pick]

    def attr(self, expr: ast.Attr, scope: dict[str, object]) -> object:
        """Read ``base.field`` as written: width-dependent defaults read as none."""
        base = _unwrap(self.eval(expr.base, scope))
        match base:
            case _Builtin(name=name) if name in TYPES:
                base = self.construct(name, [], expr.base.span)
            case _Builtin(name=name):
                base = _activation(name)
            case ir.Layer(config=config):
                base = config
            case _Bare(name=name):
                raise self.unknown(name, expr.base.span, scope, "name")
        if not isinstance(base, cfg.Config):
            raise BuxError(
                f"only layers and settings have fields, but this is {_describe(base)}",
                expr.base.span,
            )
        fields = type(base).model_fields
        if expr.field not in fields:
            close = difflib.get_close_matches(expr.field, list(fields), n=1)
            raise BuxError(
                f"{_tag(base)} has no setting '{expr.field}'",
                expr.field_span,
                f"did you mean '{close[0]}'?" if close else None,
            )
        return getattr(base, expr.field)

    def binop(self, expr: ast.BinOp, scope: dict[str, object]) -> object:
        left, right = self.eval(expr.left, scope), self.eval(expr.right, scope)
        self.reject_bare_numbers(expr, left, right, scope)
        if _is_number(left) and _is_number(right):
            return self.arithmetic(expr, left, right)
        if (mixture := self.mixture(expr, left, right, scope)) is not None:
            return mixture
        match expr.op:
            case "+" | "-":
                return self.branch_sum(expr, left, right)
            case "*":
                factors = [
                    self.factor(left, expr.left.span),
                    self.factor(right, expr.right.span),
                ]
                return self.node(
                    ir.Prod(factors=_flatten(factors, ir.Prod, "factors")), expr.span
                )
            case "^":
                return self.repeat(expr, left, right)
            case op:
                raise BuxError(
                    f"'{op}' only divides numbers",
                    expr.span,
                    "scale a branch with `f * 0.5`",
                )

    def reject_bare_numbers(
        self, expr: ast.BinOp, left: object, right: object, scope: dict[str, object]
    ) -> None:
        # In arithmetic an undefined name is a typo for a number, not a block.
        for value, other, side in ((left, right, expr.left), (right, left, expr.right)):
            if isinstance(value, _Bare) and (
                _is_number(other)
                or expr.op in ("/", "//")
                or (expr.op == "^" and side is expr.right)
            ):
                raise self.unknown(value.name, side.span, scope, "name")

    def arithmetic(self, expr: ast.BinOp, left: object, right: object) -> float:
        a, b = typing.cast(float, _unwrap(left)), typing.cast(float, _unwrap(right))
        try:
            return _ARITHMETIC[expr.op](a, b)
        except ZeroDivisionError:
            raise BuxError("division by zero", expr.span) from None

    def branch_sum(self, expr: ast.BinOp, left: object, right: object) -> ir.Node:
        terms = [self.term(left, expr.left.span), self.term(right, expr.right.span)]
        if expr.op == "-":
            last = terms[-1]
            terms[-1] = (
                ir.Const(value=-last.value)
                if isinstance(last, ir.Const)
                else self.node(ir.Neg(body=last), expr.right.span)
            )
        return self.node(ir.Sum(terms=_flatten(terms, ir.Sum, "terms")), expr.span)

    def repeat(self, expr: ast.BinOp, left: object, right: object) -> ir.Node:
        times = _unwrap(right)
        if isinstance(times, bool) or not isinstance(times, int) or times < 1:
            raise BuxError("a repeat count must be a positive integer", expr.right.span)
        body = self.graph(left, expr.left.span)
        if times == 1:
            return body
        return self.node(ir.Repeat(body=body, times=times), expr.span)

    def mixture(
        self, expr: ast.BinOp, left: object, right: object, scope: dict[str, object]
    ) -> tcfg.MixtureConfig | None:
        """Mix data sources with + and weight them with *: 0.7 * web + 0.3 * code.
        Return None unless a side is a data source."""

        def operand(value: object, side: ast.Expr) -> object:
            value = _unwrap(value)
            # The bare name of a kind of source is one with its defaults: web + synthetic.
            if isinstance(value, _Builtin) and value.name in _SOURCES:
                return self.construct(value.name, [], side.span)
            return value

        a, b = operand(left, expr.left), operand(right, expr.right)
        if not (isinstance(a, tcfg.SOURCES) or isinstance(b, tcfg.SOURCES)):
            return None
        for value, side in ((a, expr.left), (b, expr.right)):
            if isinstance(value, _Bare):
                raise self.unknown(value.name, side.span, scope, "name")
        if (
            expr.op == "+"
            and isinstance(a, tcfg.SOURCES)
            and isinstance(b, tcfg.SOURCES)
        ):
            return tcfg.MixtureConfig(sources=_weighted(a) + _weighted(b))
        if expr.op == "*":
            (weight, side), source = (
                ((a, expr.left), b)
                if isinstance(b, tcfg.SOURCES)
                else ((b, expr.right), a)
            )
            if _is_number(weight) and isinstance(source, tcfg.SOURCES):
                return self.weighted(source, typing.cast(float, weight), side.span)
        raise BuxError(
            "data is mixed with + and weighted with *",
            expr.span,
            "0.7 * web + 0.3 * code, or web + code for an even mix",
        )

    def weighted(
        self, source: tcfg.DataConfig, weight: float, span: Span
    ) -> tcfg.MixtureConfig:
        if weight <= 0:
            raise BuxError("a data weight must be positive", span)
        return tcfg.MixtureConfig(
            sources=tuple(
                part.model_copy(update={"weight": part.weight * weight})
                for part in _weighted(source)
            )
        )

    def term(self, value: object, span: Span) -> ir.Node:
        if _is_number(value):
            return self.node(
                ir.Const(value=float(typing.cast(float, _unwrap(value)))), span
            )
        return self.graph(value, span)

    def factor(self, value: object, span: Span) -> ir.Node:
        if _is_number(value) and isinstance(_unwrap(value), int):
            n = _unwrap(value)
            raise BuxError(
                f"multiplying a block by the integer {n} is ambiguous",
                span,
                f"write blk^{n} to repeat it, or {n}.0 to scale its output",
            )
        return self.term(value, span)

    # --- Coercions ------------------------------------------------------------------

    def graph(self, value: object, span: Span) -> ir.Node:
        """Return `value` as a layer or block."""
        match value:
            case _Named(label=label, value=inner, share=share):
                node = self.graph(inner, span)
                return self.node(
                    node.model_copy(
                        update={"label": label, "share": share or node.share}
                    ),
                    span,
                )
            case _ if isinstance(value, ir.NODE_TYPES):
                return value
            case cfg.Config():
                if _tag(value) not in LAYERS:
                    kind = (
                        "a training setting"
                        if _tag(value) in TRAINING
                        else "a setting of a layer"
                    )
                    raise BuxError(f"{_tag(value)}(...) is {kind}, not a layer", span)
                config = typing.cast(cfg.LayerConfig, value)
                return self.node(ir.Layer(config=config), span)
            case _Builtin(name=name) if name in ACTIVATIONS:
                return self.graph(_activation(name), span)
            case _Builtin(name=name):
                return self.graph(self.construct(name, [], span), span)
            case str():
                return self.pattern(value, span)
            case _Bare(name=name):
                raise self.unknown(name, span)
            case _Macro(statement=st):
                raise BuxError(
                    f"'{st.name}' is a macro and needs arguments: {st.name}({', '.join(st.params or ())})",
                    span,
                )
            case int() | float() if not isinstance(value, bool):
                raise BuxError(
                    "a number is not a layer",
                    span,
                    "repeat a block with blk^3, scale a branch with f * 0.5",
                )
        raise BuxError(f"expected a layer or block, found {_describe(value)}", span)

    def pattern(self, text: str, span: Span) -> ir.Node:
        """Expand a pattern: "MMMA" is the sequence M M M A of one-letter names."""
        nodes = []
        for k, letter in enumerate(text):
            if letter.isspace():
                continue
            letter_span = Span(span.start + 1 + k, span.start + 2 + k)
            if letter in self.broken and letter not in self.statements:
                raise _Reported()
            if not letter.isalpha() or letter not in self.statements:
                raise BuxError(
                    f"pattern letter '{letter}' is not a one-letter definition",
                    letter_span,
                )
            definition = self.lookup(letter, letter_span, {})
            nodes.append(self.graph(definition, letter_span))
        if not nodes:
            raise BuxError("empty pattern", span)
        return (
            nodes[0] if len(nodes) == 1 else self.node(ir.Seq(items=tuple(nodes)), span)
        )

    def data(self, value: object, span: Span) -> object:
        """Return `value` as a setting value."""
        match value:
            case _Named(value=inner):
                return self.data(inner, span)
            case _Builtin(name=name):
                return name if name in ACTIVATIONS else self.construct(name, [], span)
            case _Bare(name=name):
                return name
            case None | bool() | int() | float() | str() | list() | cfg.Config():
                return value
        raise BuxError(f"expected a value, found {_describe(value)}", span)

    # --- Calls ----------------------------------------------------------------------

    def call(self, expr: ast.Call, scope: dict[str, object]) -> object:
        callee = self.lookup(expr.name, expr.name_span, scope)
        match callee:
            case _Macro(statement=st):
                return self.apply(st, expr, scope)
            case _Builtin(name=name) if name in TYPES:
                fields, ports = self.arguments(expr, scope, TYPES[name])
                config = self.construct(name, fields, expr.span)
                return self.with_ports(config, ports, None, None, expr.span)
            case _Named(label=label, share=share) if cls := _config_type(
                base := _unwrap(callee)
            ):
                fields, ports = self.arguments(expr, scope, cls)
                if not fields:
                    config = self.amended(base, fields, expr.span)
                    return self.with_ports(config, ports, label, share, expr.span)
                # An amendment redefines a shared layer, which stays one module.
                if share and not expr.amend:
                    raise BuxError(
                        f"'{label}' is shared, so it cannot be changed here",
                        expr.span,
                        "define a separate layer",
                    )
                config = self.amended(base, fields, expr.span)
                return self.with_ports(
                    config, ports, self.text(expr.span), None, expr.span
                )
            case _Bare(name=name):
                raise self.unknown(name, expr.name_span, called=True)
        raise self.not_callable(expr, callee)

    def amended(self, base: object, fields: _Fields, span: Span) -> cfg.Config:
        """Return the config `base` with `fields` set; a bare type, as in
        ``norm = rmsnorm``, is its builtin called with them."""
        if isinstance(base, _Builtin):
            return self.construct(base.name, fields, span)
        base = typing.cast(cfg.Config, base)
        return self.override(base, fields, span) if fields else base

    def not_callable(self, expr: ast.Call, callee: object) -> BuxError:
        builtin = isinstance(_unwrap(callee), _Builtin)
        what = "an activation" if builtin else _describe(callee)
        if expr.amend:
            return BuxError(
                f"'{expr.name}' is {what}, which has no settings to amend",
                expr.name_span,
                f"redefine it instead: {expr.name} = ...",
            )
        return BuxError(
            f"'{expr.name}' is {what} and takes no arguments",
            expr.name_span,
            f"to parametrize a block, make it a macro: {expr.name}(x) = ..."
            if what == "a block"
            else None,
        )

    def arguments(
        self, expr: ast.Call, scope: dict[str, object], cls: type[cfg.Config]
    ) -> tuple[_Fields, list[ir.Port]]:
        fields: _Fields = []
        ports: list[ir.Port] = []
        seen: set[tuple[str, ...]] = set()
        for arg in expr.args:
            if arg.key is None:
                raise BuxError(
                    "layer settings are keyword arguments: name=value", arg.span
                )
            if arg.key in seen:
                raise BuxError(f"'{'.'.join(arg.key)}' is given twice", arg.span)
            seen.add(arg.key)
            if cls is tcfg.RematConfig and arg.key in _REMAT_NAMES:
                fields.append((arg.key, self.names(arg.value), arg.span))
                continue
            value = self.eval(arg.value, scope)
            if isinstance(value, ir.Tap | ir.Ref):
                ports.append(_port(arg.key, value, cls, arg.span))
                continue
            # A macro body runs once per call, so the same argument can be a bare
            # name in one call and not in the next.
            if isinstance(value, _Bare):
                self.bare[arg.span] = (value.name, arg.value.span, tuple(scope))
            else:
                self.bare.pop(arg.span, None)
            fields.append((arg.key, self.data(value, arg.value.span), arg.span))
        return fields, ports

    def names(self, expr: ast.Expr) -> list[str]:
        """Read names of what layers make, as `remat` takes them: attention,
        attention.in, "mlp", or a list of them. They are not definitions, so `mlp` is
        no layer type here, and `[attention]` a list of one."""
        match expr:
            case ast.Name(name=name) | ast.Lit(value=str() as name):
                return [name]
            case ast.Attr(base=ast.Name(name=base), field=field):
                return [f"{base}.{field}"]
            case ast.ListExpr(items=items):
                return [name for item in items for name in self.names(item)]
            case ast.Residual(body=body):
                return self.names(body)
            case ast.Alt():
                return self.names(self.choose(expr))
        raise BuxError(
            "expected a name of what layers make, or a list of them", expr.span
        )

    def with_ports(
        self,
        config: cfg.Config,
        ports: list[ir.Port],
        label: str | None,
        share: str | None,
        span: Span,
    ) -> object:
        if not ports:
            return config if label is None else _Named(label, config, share)
        layer = typing.cast(ir.Layer, self.graph(config, span))
        return self.node(
            layer.model_copy(
                update={"ports": tuple(ports), "label": label, "share": share}
            ),
            span,
        )

    def construct(self, tag: str, fields: _Fields, span: Span) -> cfg.Config:
        data: dict = {"type": tag}
        for key, value, arg_span in fields:
            if key == ("type",):
                raise BuxError("the type is set by the name of the call", arg_span)
            _assign(data, key, value)
        return self.validate(TYPES[tag], data, fields, span)

    def override(self, base: cfg.Config, fields: _Fields, span: Span) -> cfg.Config:
        data = base.model_dump()
        for key, value, _ in fields:
            _assign(data, key, value)
        return self.validate(type(base), data, fields, span)

    def validate(
        self, cls: type[cfg.Config], data: dict, fields: _Fields, span: Span
    ) -> cfg.Config:
        try:
            return cls.model_validate(data)
        except ValidationError as err:
            raise self.invalid(cls, data["type"], err, fields, span) from None

    def invalid(
        self,
        cls: type[cfg.Config],
        tag: str,
        err: ValidationError,
        fields: _Fields,
        span: Span,
    ) -> BuxError:
        errors = err.errors()
        forbidden = [error for error in errors if error["type"] == "extra_forbidden"]
        error = (forbidden or errors)[0]
        loc = tuple(str(p) for p in error["loc"])
        arg_span = next(
            (s for k, _, s in fields if loc[: len(k)] == k or k[: len(loc)] == loc),
            span,
        )
        path = ".".join(loc)
        match error["type"]:
            case "extra_forbidden":
                close = difflib.get_close_matches(loc[-1], list(cls.model_fields), n=1)
                hint = (
                    f"did you mean '{close[0]}'?" if close and len(loc) == 1 else None
                )
                return BuxError(f"{tag} has no setting '{path}'", arg_span, hint)
            case "missing":
                return BuxError(f"{tag} needs '{path}'", arg_span, f"{tag}({path}=...)")
        hint = None
        if error["type"] == "literal_error":
            expected = error.get("ctx", {}).get("expected", "")
            options = re.findall(r"'([^']*)'", str(expected))
            close = difflib.get_close_matches(str(error["input"]), options, n=1)
            hint = f"did you mean '{close[0]}'?" if close else None
        elif arg_span in self.bare and error["type"] != "value_error":
            # A bare name only works as a string, and this setting wants something
            # else, so the name was meant to refer to a definition. A value error is
            # the setting's own check, of a string it takes.
            name, name_span, scope = self.bare[arg_span]
            return self.unknown(name, name_span, scope, "name")
        return BuxError(
            f"{tag}: {path}: {error['msg']}" if path else f"{tag}: {error['msg']}",
            arg_span,
            hint,
        )

    def apply(
        self, st: ast.Statement, expr: ast.Call, scope: dict[str, object]
    ) -> object:
        macro_scope = self.bind(st, expr, scope)
        self.depth += 1
        if self.depth > 64:
            raise BuxError(f"'{st.name}' recurses too deeply", expr.span)
        try:
            return _Named(self.text(expr.span), self.eval(st.body, macro_scope))
        finally:
            self.depth -= 1

    def bind(
        self, st: ast.Statement, expr: ast.Call, scope: dict[str, object]
    ) -> dict[str, object]:
        params = st.params or ()
        args: dict[str, object] = {}
        positional = [a for a in expr.args if a.key is None]
        if len(positional) > len(params):
            raise BuxError(
                f"'{st.name}' takes {len(params)} argument(s)",
                positional[len(params)].span,
            )
        for param, arg in zip(params, positional):
            args[param] = self.eval(arg.value, scope)
        for arg in expr.args:
            if arg.key is None:
                continue
            name = ".".join(arg.key)
            if name not in params:
                raise BuxError(f"'{st.name}' has no parameter '{name}'", arg.span)
            if name in args:
                raise BuxError(f"'{name}' is given twice", arg.span)
            args[name] = self.eval(arg.value, scope)
        defaults = dict(st.defaults)
        if missing := [p for p in params if p not in args and p not in defaults]:
            raise BuxError(
                f"'{st.name}' is missing argument(s): {', '.join(missing)}", expr.span
            )
        # A default sees the parameters before it: blk(n, kv=n // 4) = ...
        for i, param in enumerate(params):
            if param not in args:
                earlier = {p: args[p] for p in params[:i]}
                args[param] = self.eval(defaults[param], earlier)
        return args

    # --- Settings -------------------------------------------------------------------

    def setting(self, name: str) -> tuple[object, Span | None]:
        st = self.statements.get(name)
        if st is None and name in self.broken:
            raise _Reported()
        if st is None:
            return None, None
        return self.data(self.definition(name), st.body.span), st.body.span


@dataclass(eq=False)
class _Frame:
    """One use of a block (a definition, a macro call, or one pass of a repeat) while lowering."""

    label: str
    used: list[_Entry] = field(default_factory=list)


@dataclass(eq=False)
class _Entry:
    """A value on a tap stack."""

    value: ir.Width | _PortValue
    frames: tuple[_Frame, ...]  # the blocks being lowered when it was written
    # The innermost block that both wrote and read it. Once that block ends, the
    # value is its leftover: `owner` is set, unless it was popped.
    used_in: _Frame | None = None
    owner: str | None = None
    popped: bool = False


# Layers that act on every copy of a widened residual stream instead of reading it.
_ON_COPIES = (cfg.EngramConfig, cfg.NextNConfig)


class _Lowering:
    def __init__(
        self, ev: _Evaluator, d_model: int, residual: cfg.ResidualConfig | None
    ):
        self.ev = ev
        self.d_model = d_model
        # Whether the residual algebra widens the stream between branches, and whether
        # into copies of it (attnres keeps the branch outputs so far instead).
        self.widened = residual is not None and residual.type != "vpgate"
        self.copies = residual is not None and residual.type in ("hc", "mhc", "gr")
        self.depth = 0  # residual branches the node being lowered is inside
        self.stacks: dict[str, list[_Entry]] = {}
        self.frames: list[_Frame] = []
        self.shared: dict[str, ir.Node] = {}
        self.spans: list[Span] = []
        self.embeddings = 0
        # Layers that reuse the embedding matrix (tied heads, nextn), with their spans.
        self.borrowers: list[tuple[str, Span]] = []
        self.heads = 0

    def error(self, message: str, hint: str | None = None) -> BuxError:
        return BuxError(message, self.spans[-1] if self.spans else None, hint)

    @contextmanager
    def frame(self, label: str) -> Generator[None]:
        frame = _Frame(label)
        self.frames.append(frame)
        try:
            yield
        finally:
            self.frames.pop()
            for entry in frame.used:
                if entry.used_in is frame and not entry.popped:
                    entry.owner = label

    def lower_model(self, graph: ir.Node) -> ir.Node:
        first = next(ir.unrolled_layers(graph), None)
        width: ir.Width = (
            "tokens"
            if first is not None and first.config.type == "embed"
            else ("streams" if self.widened else self.d_model)
        )
        typed, _ = self.lower(graph, width)
        if self.borrowers and self.embeddings != 1:
            what, span = self.borrowers[0]
            raise BuxError(
                f"{what} needs exactly one embedding, found {self.embeddings}", span
            )
        return typed

    def lower(self, node: ir.Node, width: ir.Width) -> tuple[ir.Node, ir.Width | None]:
        span = self.ev.node_spans.get(id(node))
        if span is not None:
            self.spans.append(span)
        try:
            if node.label is not None and not isinstance(node, ir.Layer):
                with self.frame(node.label):
                    typed, out = self._lower(node, width)
            else:
                typed, out = self._lower(node, width)
            if node.share is not None:
                first = self.shared.setdefault(node.share, typed)
                if ir.strip(first, widths=False) != ir.strip(typed, widths=False):
                    raise self.error(
                        f"shared '{node.share}' is used where it would need different widths"
                    )
            return typed, out
        finally:
            if span is not None:
                self.spans.pop()

    def _lower(self, node: ir.Node, width: ir.Width) -> tuple[ir.Node, ir.Width | None]:
        match node:
            case ir.Layer():
                return self.layer(node, width)
            case ir.Seq(items=items):
                return self.seq(node, items, width)
            case ir.Residual(body=body):
                return self.residual(node, body, width)
            case ir.Sum(terms=terms) | ir.Prod(factors=terms):
                return self.branches(node, terms, width)
            case ir.Neg(body=body):
                typed, out = self.lower(body, width)
                return ir.replace_children(node, (typed,)), out
            case ir.Const():
                return node, None
            case ir.Identity():
                return node, width
            case ir.Repeat(body=body, times=times):
                return self.repeat(node, body, times, width)
            case ir.Tap(name=name):
                self.write(name, width)
                return node, width
            case ir.Ref(name=name, pop=pop):
                return node, self.read_hidden(name, pop)
        raise AssertionError(node)

    def seq(
        self, node: ir.Node, items: tuple[ir.Node, ...], width: ir.Width
    ) -> tuple[ir.Node, ir.Width]:
        typed = []
        for item in items:
            typed_item, out = self.lower(item, width)
            # A layer that needed a merge came back as an unlabeled (merge, layer).
            if (
                isinstance(typed_item, ir.Seq)
                and typed_item.label is None
                and typed_item.share is None
            ):
                typed.extend(typed_item.items)
            else:
                typed.append(typed_item)
            width = typing.cast(ir.Width, out)
        return ir.replace_children(node, tuple(typed)), width

    def residual(
        self, node: ir.Node, body: ir.Node, width: ir.Width
    ) -> tuple[ir.Node, ir.Width]:
        if isinstance(body, ir.Layer) and isinstance(body.config, cfg.EngramConfig):
            raise self.error(
                "engram adds to the stream itself, so [engram] would add the "
                "stream twice",
                "write it bare: ... engram ...",
            )
        inner = width
        if self.widened and self.depth == 0:
            if width != "streams":
                raise self.error(
                    "a residual branch needs the widened residual stream, but "
                    "it was read back into one before this",
                    f"only branches, taps{', engram and nextn' if self.copies else ''}"
                    " keep it; save it with @h and go on from $h",
                )
            inner = self.d_model
        elif not isinstance(width, int):
            raise self.error(
                f"a residual branch needs hidden states, but receives {width}"
            )
        self.depth += 1
        try:
            typed, out = self.lower(body, inner)
        finally:
            self.depth -= 1
        if out != inner:
            raise self.error(
                f"a residual branch must keep the width, but this one maps {inner} to {out}"
            )
        return ir.replace_children(node, (typed,)), width

    def branches(
        self, node: ir.Node, terms: tuple[ir.Node, ...], width: ir.Width
    ) -> tuple[ir.Node, ir.Width]:
        typed, outs = [], []
        for term in terms:
            typed_term, out = self.lower(term, width)
            typed.append(typed_term)
            if out is not None and out not in outs:
                outs.append(out)
        if not outs:
            raise self.error("parallel branches need at least one layer")
        if len(outs) > 1:
            raise self.error(
                f"parallel branches disagree on width: {', '.join(map(str, outs))}"
            )
        return ir.replace_children(node, tuple(typed)), outs[0]

    def repeat(
        self, node: ir.Node, body: ir.Node, times: int, width: ir.Width
    ) -> tuple[ir.Node, ir.Width]:
        span = self.ev.node_spans.get(id(body))
        text = self.ev.text(span) if span is not None else "the block"
        label = f"one pass of {f'({text})' if ' ' in text else text}^{times}"
        with self.frame(label):
            typed, out = self.lower(body, width)
        if out != width:
            raise self.error(
                f"a repeated block must keep the width, but this one maps {width} to {out}"
            )
        for _ in range(times - 1):
            with self.frame(label):
                self.lower(body, width)
        return ir.replace_children(node, (typed,)), width

    def write(self, tap: str, value: ir.Width | _PortValue) -> None:
        self.stacks.setdefault(tap, []).append(_Entry(value, tuple(self.frames)))

    def read_hidden(self, tap: str, pop: bool) -> ir.Width:
        value = self.read(tap, pop)
        if isinstance(value, _PortValue):
            raise self.error(
                f"tap '{tap}' holds the '{value.port}' port of a layer",
                f"read it through the same port: layer({value.port}=${tap})",
            )
        if value == "streams" and self.depth > 0:
            raise self.error(
                f"tap '{tap}' holds the widened residual stream, which only "
                "exists outside residual branches",
                f"read it back into one first: merge @{tap}",
            )
        return value

    def read(self, tap: str, pop: bool) -> ir.Width | _PortValue:
        stack = self.stacks.get(tap)
        if stack is None:
            raise self.error(
                f"tap '{tap}' is read before anything writes it",
                f"write it first with @{tap}",
            )
        if not stack:
            raise self.error(
                f"tap '{tap}' is empty: everything written to it was already popped"
            )
        entry = stack[-1]
        if entry.owner is not None and any(other.owner is None for other in stack[:-1]):
            # A block wrote the tap for its own use and left the value behind, where
            # it now hides what the reader most likely wants.
            raise self.error(
                f"tap '{tap}' here reads the value that '{entry.owner}' wrote and "
                "read for itself, not the one written before it",
                f"pop it in '{entry.owner}' with ${tap}! where it is read last, "
                "or give one of the two taps another name",
            )
        if entry.owner is None:
            self.mark_used(entry)
        if pop:
            stack.pop()
            entry.popped = True
        return entry.value

    def mark_used(self, entry: _Entry) -> None:
        common = None
        for written, current in zip(entry.frames, self.frames):
            if written is not current:
                break
            common = written
        if common is not None:
            entry.used_in = common
            common.used.append(entry)

    def layer(self, node: ir.Layer, width: ir.Width) -> tuple[ir.Node, ir.Width]:
        config = node.config
        if width == "logits":
            raise self.error(
                "nothing can follow the LM head",
                "to go on after a head, read a tap: @h norm head $h ...",
            )
        on_streams = isinstance(config, cfg.MergeConfig) or (
            self.copies and isinstance(config, _ON_COPIES)
        )
        if width == "streams" and not on_streams:
            # The layer needs one stream: read the widened one back into one first.
            merge = ir.Layer(config=cfg.MergeConfig(), width="streams")
            typed, out = self.layer(node, self.d_model)
            return ir.Seq(items=(merge, typed)), out
        out = self.layer_output(config, width)
        self.connect(node.ports)
        return node.model_copy(update={"width": width}), out

    def layer_output(self, config: cfg.LayerConfig, width: ir.Width) -> ir.Width:
        if isinstance(config, cfg.MergeConfig):
            if width != "streams":
                raise self.error(
                    "merge reads a widened residual stream back into one, but there "
                    "is none here",
                    "widen it with residual = hc(...), mhc(...), gr(...) or attnres(...)"
                    if not self.widened
                    else None,
                )
            return self.d_model
        if config.type == "embed":
            if width != "tokens":
                raise self.error("an embedding takes token ids, so it must come first")
            self.embeddings += 1
            return "streams" if self.widened and self.depth == 0 else self.d_model
        if width == "tokens":
            raise self.error(
                f"{config.type} needs hidden states, but receives token ids",
                "start with an embedding",
            )
        if isinstance(config, cfg.LMHeadConfig):
            self.heads += 1
            if config.tied and self.spans:
                self.borrowers.append(("a tied head", self.spans[-1]))
            return "logits"
        if isinstance(config, cfg.LinearConfig):
            return config.features
        if isinstance(config, cfg.NextNConfig):
            if self.spans:
                self.borrowers.append(("nextn", self.spans[-1]))
            return "streams" if self.widened and self.depth == 0 else width
        return width

    def connect(self, ports: tuple[ir.Port, ...]) -> None:
        for port in sorted(ports, key=lambda p: p.mode == "write"):
            if port.mode == "write":
                self.write(port.tap, _PortValue(port.name))
                continue
            value = self.read(port.tap, port.mode == "pop")
            if not isinstance(value, _PortValue) or value.port != port.name:
                held = (
                    f"the '{value.port}' port"
                    if isinstance(value, _PortValue)
                    else "hidden states"
                )
                raise self.error(
                    f"port '{port.name}' reads tap '{port.tap}', which holds {held}"
                )


def _compile(ev: _Evaluator, entry: str) -> ir.Architecture:
    if entry not in ev.statements:
        if entry in ev.broken or ev.incomplete:
            raise _Reported()
        raise BuxError(
            f"no definition named '{entry}'",
            hint=f"define it: {entry} = embed ... head",
        )
    settings = _model_settings(ev)
    d_model = typing.cast(int | None, settings["d_model"])
    vocab_size = typing.cast(int | None, settings["vocab_size"])
    residual = typing.cast(cfg.ResidualConfig | None, settings["residual"])
    init = typing.cast(cfg.NormalInitConfig | None, settings["init"])
    if d_model is None:
        if ev.incomplete:
            raise _Reported()
        raise BuxError("'d_model' is not defined", hint="d_model = 4096")
    lowering = _Lowering(ev, d_model, residual)
    model = ev.graph(ev.definition(entry), ev.statements[entry].name_span)
    graph = lowering.lower_model(model)
    if vocab_size is None and (lowering.embeddings or lowering.heads):
        if ev.incomplete:
            raise _Reported()
        raise BuxError("'vocab_size' is not defined", hint="vocab_size = 32000")
    return ir.Architecture(
        d_model=d_model,
        vocab_size=vocab_size,
        residual=residual,
        init=init,
        graph=graph,
    )


# What each top-level setting of a model takes, and how that is written, where it is
# not an integer.
_SETTING_TYPES: dict[str, tuple[tuple[type, ...], str]] = {
    "residual": (
        tuple(RESIDUALS.values()),
        " or ".join(f"{t}(...)" for t in RESIDUALS),
    ),
    "init": ((cfg.NormalInitConfig,), "normal(...)"),
}


def _model_settings(ev: _Evaluator) -> dict[str, object]:
    settings = {}
    for name in SETTINGS:
        value, span = ev.setting(name)
        expected, kind = _SETTING_TYPES.get(name, ((int,), "an integer"))
        if value is not None and (
            not isinstance(value, expected) or isinstance(value, bool)
        ):
            raise BuxError(f"'{name}' must be {kind}, found {_describe(value)}", span)
        settings[name] = value
    return settings


def _expects(annotation: object) -> str:
    """Say how a setting of this type is written: adamw(...) or muon(...), an integer."""
    kinds = []
    for member in typing.get_args(annotation) or (annotation,):
        if member is type(None):
            kinds.append("none")
        elif member is int:
            kinds.append("an integer")
        elif member is float:
            kinds.append("a number")
        elif member is tcfg.MixtureConfig:
            kinds.append("a mix of sources (0.7 * web + 0.3 * code)")
        elif typing.get_origin(member) is tuple:
            kinds.append(f"a list of {_expects(typing.get_args(member)[0])}")
        elif isinstance(member, type) and issubclass(member, cfg.Config):
            kinds += [
                f"{t}(...)"
                for t in typing.get_args(member.model_fields["type"].annotation)
            ]
    return ", ".join(kinds[:-1]) + " or " + kinds[-1] if len(kinds) > 1 else kinds[0]


_ADAPTERS = {
    name: TypeAdapter(f.rebuild_annotation())
    for name, f in tcfg.TrainConfig.model_fields.items()
}


def _train(
    ev: _Evaluator, arch: ir.Architecture | None, *, require: bool
) -> tcfg.TrainConfig | None:
    """Return the validated training settings, or None if there are none. Without
    `require`, a module may set only some of them; they are checked all the same."""
    values: dict[str, object] = {}
    spans: dict[str, Span] = {}
    for name in TRAIN_SETTINGS:
        value, span = ev.setting(name)
        if span is None:
            continue
        values[name] = _train_setting(name, value, span)
        spans[name] = span
    if not values:
        return None
    if missing := [n for n in _REQUIRED if n not in values]:
        if not require or ev.incomplete:
            return None
        raise _missing_settings(missing)
    train = tcfg.TrainConfig.model_validate(values)
    _check_schedule(train, spans)
    _check_tokenizer(train, spans)
    _check_positions(train, arch, spans)
    return train


def _train_setting(name: str, value: object, span: Span) -> object:
    annotation = tcfg.TrainConfig.model_fields[name].annotation
    kinds = typing.get_args(annotation) or (annotation,)
    if isinstance(value, bool) and (int in kinds or float in kinds):
        raise BuxError(
            f"'{name}' must be {_expects(annotation)}, found a boolean", span
        )
    try:
        return _ADAPTERS[name].validate_python(value)
    except ValidationError as err:
        errors = err.errors()
        if errors and errors[0]["type"].startswith(("greater", "less")):
            message = errors[0]["msg"].replace("Input should be", "must be")
            raise BuxError(f"'{name}' {message}, found {value}", span) from None
        if errors and errors[0]["type"] == "value_error":  # the setting's own check
            raise BuxError(str(errors[0]["ctx"]["error"]), span) from None
        found = value if _is_number(value) else _describe(value)
        raise BuxError(
            f"'{name}' must be {_expects(annotation)}, found {found}", span
        ) from None


def _missing_settings(missing: list[str]) -> BuxError:
    listed = (
        ", ".join(missing[:-1]) + " and " + missing[-1]
        if len(missing) > 1
        else missing[0]
    )
    return BuxError(
        f"this file sets how to train, so it is a run, and a run needs {listed}",
        hint=_EXAMPLES[missing[0]],
    )


def _check_schedule(train: tcfg.TrainConfig, spans: dict[str, Span]) -> None:
    schedule = train.schedule
    cooldown = (
        schedule.cooldown(train.steps) if isinstance(schedule, tcfg.WSDSchedule) else 0
    )
    if schedule.warmup + cooldown > train.steps:
        what = f"{schedule.warmup} of warmup" + (
            f" and {cooldown} of cooldown" if cooldown else ""
        )
        raise BuxError(
            f"the schedule takes {schedule.warmup + cooldown} steps ({what}), but the "
            f"run has {train.steps}",
            spans.get("schedule", spans["steps"]),
        )


def _check_tokenizer(train: tcfg.TrainConfig, spans: dict[str, Span]) -> None:
    reads = [("data", train.data)] + [("eval", e.data) for e in tcfg.evals(train)]
    for name, data in reads:
        kinds = sorted({s.type for s in tcfg.leaves(data)} & {"text", "chat"})
        if kinds and train.tokenizer is None:
            raise BuxError(
                f"{' and '.join(kinds)} data is tokenized, so the run needs a tokenizer",
                spans[name],
                'tokenizer = tokenizer(name="gpt2")',
            )


def _check_positions(
    train: tcfg.TrainConfig, arch: ir.Architecture | None, spans: dict[str, Span]
) -> None:
    for layer in arch.layers() if arch is not None else ():
        if isinstance(layer.config, cfg.EmbeddingConfig):
            positions = layer.config.max_positions
            if positions is not None and train.seq_len > positions:
                raise BuxError(
                    f"sequences of {train.seq_len} tokens are longer than the "
                    f"{positions} positions the embedding learns",
                    spans["seq_len"],
                    f"seq_len = {positions}, or embed(max_positions={train.seq_len})",
                )


#: Where ``use`` finds modules.
#:
#: None for files on the search path (`bux.lang.modules.find`), a mapping of
#: module names to their sources, or a `~bux.lang.modules.Loader`.
Modules = Mapping[str, str] | modules.Loader | None


def _load(
    source: str,
    filename: str,
    overrides: Sequence[str],
    found: Modules,
    *,
    recover: bool,
) -> modules.Program:
    loader = modules.from_mapping(found) if isinstance(found, Mapping) else found
    return modules.load(
        source, filename, overrides=overrides, loader=loader, recover=recover
    )


def _run_name(filename: str, parts: tuple[str, ...]) -> str:
    stem = "run" if filename.startswith("<") else Path(filename).stem
    return stem + (f"[{', '.join(parts)}]" if parts else "")


def compile(
    source: str,
    *,
    entry: str = "model",
    filename: str = "<string>",
    overrides: Sequence[str] = (),
    modules: Modules = None,
) -> ir.Architecture:
    """Compile .bux source into the architecture of one definition.

    Every definition is evaluated, so a mistake in any of them raises, even in one
    that `entry` does not use. In a sweep, every alternative takes its first option;
    `compile_runs` builds them all. Training settings are evaluated but not returned:
    a run's are in the `~bux.lang.ir.Run` that `compile_runs` returns.

    Args:
        source: The text of a ``.bux`` file.
        entry: The definition to build: the whole model by default, or a block
            (``entry="block"``), which then needs no ``embed`` or ``head``.
        filename: Where `source` is from: errors name it, and ``use`` searches its
            folder first.
        overrides: Statements that redefine the file's own, as if written at its
            end, as the command line gives them: ``"optimizer.lr=1e-3"``,
            ``"steps = 2k"``, ``"use recipe"``.
        modules: Where ``use`` finds modules (`Modules`): by default, files on the
            search path; a mapping holds their sources by module name.

    Returns:
        The architecture of `entry`, with every layer's input width inferred.

    Raises:
        BuxError: At the first mistake: a syntax error, an unknown name, a setting
            that a config rejects, widths that do not agree, ... It is attached to
            the file it is in, and prints that line with a caret under the mistake.

    Example::

        arch = compile(
            "d_model = 64\\n"
            "vocab_size = 512\\n"
            "block = [rmsnorm gqa(num_heads=4, num_kv_heads=2)]\\n"
            "model = embed block^2 rmsnorm head\\n"
        )
        len(list(arch.layers()))  # 7: embed, 2 x (rmsnorm, gqa), rmsnorm, head
    """
    program = _load(source, filename, overrides, modules, recover=False)
    try:
        ev = _Evaluator(program)
        for name, st in ev.statements.items():
            if st.params is None:
                ev.definition(name)
        return _compile(ev, entry)
    except BuxError as err:
        raise program.locate(err) from None


def compile_runs(
    source: str,
    *,
    entry: str = "model",
    filename: str = "<string>",
    overrides: Sequence[str] = (),
    modules: Modules = None,
) -> list[ir.Run]:
    """Compile every run of a file: one per combination of the alternatives it reaches.

    Each combination compiles to its architecture and its training settings, the
    first axis written varying slowest. Alternatives in definitions that the run does
    not reach make no runs. A run is named for the file and the alternatives it took,
    ``speedrun[mqa, optimizer.lr=0.01]``; a file without alternatives makes one run,
    named for its stem: ``speedrun``.

    Args:
        source: The text of a ``.bux`` file that says how to train its model.
        entry: The definition to train: ``model`` by default.
        filename: Where `source` is from, as for `compile`. Runs are named for its
            stem, or ``run`` for a name in angle brackets such as ``<string>``.
        overrides: Statements that redefine the file's own, as for `compile`;
            alternatives there (``"optimizer.lr = 0.01 | 0.03"``) make a sweep too.
        modules: Where ``use`` finds modules, as for `compile`.

    Returns:
        The runs, in the order that ``bux plan`` lists them and ``--index`` counts.

    Raises:
        BuxError: At the first mistake in any run, or if the file sets only some of
            ``data``, ``optimizer``, ``seq_len``, ``batch`` and ``steps``, or none
            (a model, not a run).

    Example::

        # smoke.bux, with optimizer = adamw(lr=3e-3 | 1e-2)
        runs = compile_runs(source, filename="smoke.bux")
        [run.name for run in runs]
        # ['smoke[optimizer.lr=3e-3]', 'smoke[optimizer.lr=1e-2]']
    """
    program = _load(source, filename, overrides, modules, recover=False)
    found: list[_Explored] = []

    def explore(choices: dict[Span | frozenset[str], int | str]) -> None:
        ev = _Evaluator(program, choices=choices)
        arch = _compile(ev, entry)
        train = _train(ev, arch, require=True)
        # The axes met so far are those the run depends on; the rest are in
        # definitions that it does not use.
        reached = sorted(ev.axes.values(), key=lambda axis: axis.span.start)
        free = [axis for axis in reached if axis.key not in choices]
        if free:
            for option in range(len(free[0].options)):
                explore({**choices, free[0].key: free[0].pick(option)})
            return
        for name, st in ev.statements.items():
            if st.params is None:
                ev.definition(name)
        if train is None:
            raise BuxError(
                "this file does not say how to train its model",
                hint="a run sets data, optimizer, seq_len, batch and steps",
            )
        found.append(_Explored(choices, reached, arch, train))

    try:
        explore({})
    except BuxError as err:
        raise program.locate(err) from None
    for axis in {e.key: e for f in found for e in f.reached}.values():
        found = _without_if_alike(found, axis)
    runs = []
    for f in found:
        parts = tuple(axis.part(f.choices[axis.key]) for axis in f.reached)
        name = _run_name(filename, parts)
        runs.append(ir.Run(name=name, arch=f.arch, train=f.train, choices=parts))
    return runs


@dataclass
class _Explored:
    """One combination of a sweep's alternatives, and the run it compiles to."""

    choices: dict[Span | frozenset[str], int | str]
    reached: list[Axis]  # the axes it met, in the order written
    arch: ir.Architecture
    train: tcfg.TrainConfig


def _without_if_alike(found: list[_Explored], axis: Axis) -> list[_Explored]:
    """Drop `axis` where no run depends on which of its alternatives it takes, as one
    that a later amendment overrides (``optimizer.lr = 1e-2`` on the command line,
    over a file's ``optimizer.lr = 1e-3 | 3e-3``): its runs but those of its first
    alternative, and its part of their names."""
    groups: dict[frozenset[object], list[_Explored]] = {}
    for f in found:
        if axis.key in f.choices:
            rest = frozenset((k, v) for k, v in f.choices.items() if k != axis.key)
            groups.setdefault(rest, []).append(f)
    # The runs of a group differ in this axis alone, one for each alternative; an
    # axis nested in another's alternative has groups of one, which prove nothing.
    alike = all(
        len(group) == len(axis.options)
        and all((f.arch, f.train) == (group[0].arch, group[0].train) for f in group)
        for group in groups.values()
    )
    if not groups or not alike:
        return found
    first = axis.pick(0)
    return [
        replace(
            f,
            choices={k: v for k, v in f.choices.items() if k != axis.key},
            reached=[a for a in f.reached if a.key != axis.key],
        )
        for f in found
        if f.choices.get(axis.key, first) == first
    ]


@dataclass
class Report:
    """The result of `check` on a source file: its errors, and what evaluated.

    Attributes:
        statements: The file's own statements, not those of the modules it uses.
        errors: Every error found, each attached to the file it is in, with a span
            into that file. Empty if the file compiles.
        arch: The architecture, only when there are no errors: then it equals what
            `compile` returns. None also for a module, which defines no model.
        values: The definitions that evaluated, by name: numbers, strings, lists,
            configs or IR nodes.
        run: The first run of the sweep, when the file is a run and has no errors.
        axes: The sweep's axes, in the order written, when the file has a model and
            no errors.
        via: For every module, by its filename, the ``use`` in this file that
            brought it in, perhaps through other modules.
        source: Every file joined, this one first: the spans of all statements
            index into it.
        inherited: The definitions in effect that other files make, by name.
        origin: The file that each of `inherited` is in, by name.
    """

    statements: list[ast.Statement]
    errors: list[BuxError]
    arch: ir.Architecture | None
    values: dict[str, object]
    run: ir.Run | None = None
    axes: list[Axis] = field(default_factory=list)
    via: dict[str, Span] = field(default_factory=dict)
    source: str = ""
    inherited: dict[str, ast.Statement] = field(default_factory=dict)
    origin: dict[str, str] = field(default_factory=dict)


def check(
    source: str,
    *,
    entry: str = "model",
    filename: str = "<string>",
    overrides: Sequence[str] = (),
    modules: Modules = None,
) -> Report:
    """Compile like `compile`, but collect every error instead of raising the first.

    This is what editors and ``bux check`` call. Syntax errors lose only their own
    statement, each definition is checked on its own, and an error is not repeated at
    every use of a broken definition. Every alternative of a sweep is checked once,
    with the others at their first option. A file without `entry` and without
    ``d_model`` is a module of settings, and no error.

    Args:
        source: The text of a ``.bux`` file, perhaps half typed.
        entry: The definition to build: ``model`` by default.
        filename: Where `source` is from, as for `compile`.
        overrides: Statements that redefine the file's own, as for `compile`.
        modules: Where ``use`` finds modules, as for `compile`.

    Returns:
        The errors, and what evaluated. The architecture and the first run are
        there only when there are no errors, the definitions' values always.

    Example::

        report = check("d_model = 64\\nmodel = embed nrom head\\n", filename="t.bux")
        [error.message for error in report.errors]
        # ["unknown layer or block 'nrom'"]
        report.arch  # None
    """
    errors: list[BuxError] = []

    def report(err: BuxError) -> None:
        if isinstance(err, _Reported):
            return
        err = program.locate(err)
        if all(
            (known.message, known.span, known.filename)
            != (err.message, err.span, err.filename)
            for known in errors
        ):
            errors.append(err)

    program = _load(source, filename, overrides, modules, recover=True)
    for err in program.errors:
        report(err)
    ev, arch, train = _check_alternatives(program, entry, report)
    values = {
        name: public
        for name, value in ev.values.items()
        if (public := _public(value)) is not None
    }
    clean = not errors
    axes: list[Axis] = []
    run = None
    if clean and arch is not None:
        axes = _reached_axes(program, entry)
        if train is not None:
            parts = tuple(axis.part(axis.pick(0)) for axis in axes)
            name = _run_name(filename, parts)
            run = ir.Run(name=name, arch=arch, train=train, choices=parts)
    inherited = {
        name: st for name, st in ev.statements.items() if st.span.start > len(source)
    }
    return Report(
        statements=program.main,
        errors=errors,
        arch=arch if clean else None,
        values=values,
        run=run,
        axes=axes,
        via={u.filename: u.via for u in program.units[1:] if u.via is not None},
        source=program.source,
        inherited=inherited,
        origin={
            name: program.unit(st.span.start).filename for name, st in inherited.items()
        },
    )


def _check_alternatives(
    program: modules.Program, entry: str, report: Callable[[BuxError], None]
) -> tuple[_Evaluator, ir.Architecture | None, tcfg.TrainConfig | None]:
    first: tuple[_Evaluator, ir.Architecture | None, tcfg.TrainConfig | None] | None
    first = None
    todo: list[dict[Span | frozenset[str], int | str]] = [{}]
    seen: set[tuple[Span | frozenset[str], int | str]] = set()
    while todo:
        choices = todo.pop(0)
        ev = _Evaluator(program, report=report, choices=choices)
        arch, train = _check_once(ev, entry, report)
        first = first or (ev, arch, train)
        for key, axis in ev.axes.items():
            for option in range(1, len(axis.options)):
                if (key, axis.pick(option)) not in seen:
                    seen.add((key, axis.pick(option)))
                    todo.append({**choices, key: axis.pick(option)})
    assert first is not None
    return first


def _check_once(
    ev: _Evaluator, entry: str, report: Callable[[BuxError], None]
) -> tuple[ir.Architecture | None, tcfg.TrainConfig | None]:
    for name, st in ev.statements.items():
        if st.params is None:
            try:
                ev.definition(name)
            except BuxError as err:
                report(err)
    arch = train = None
    if entry in ev.statements or "d_model" in ev.statements:
        try:
            arch = _compile(ev, entry)
        except BuxError as err:
            report(err)
    try:
        train = _train(ev, arch, require=arch is not None)
    except BuxError as err:
        report(err)
    return arch, train


def _reached_axes(program: modules.Program, entry: str) -> list[Axis]:
    # The sweep's axes are those the run reaches, which evaluating every definition,
    # as check does, does not tell apart from the others.
    ev = _Evaluator(program)
    _train(ev, _compile(ev, entry), require=True)
    return sorted(ev.axes.values(), key=lambda axis: axis.span.start)


def _public(value: object) -> object | None:
    """Return a definition's value as Report.values shows it, without private types."""
    value = _unwrap(value)
    match value:
        case _Builtin(name=name) if name in ACTIVATIONS:
            return _activation(name)
        case _Builtin(name=name):
            try:
                return TYPES[name].model_validate({"type": name})
            except ValidationError:
                return None
        case _Bare(name=name):
            return name
    return value


def compile_file(path: str, *, entry: str = "model") -> ir.Architecture:
    """Compile the .bux file at `path`, as `compile` compiles its text.

    Args:
        path: The file. Errors name it, and ``use`` searches its folder first.
        entry: The definition to build: ``model`` by default.

    Returns:
        The architecture of `entry`.

    Raises:
        BuxError: At the first mistake in the file or in the modules it uses.
        OSError: If the file cannot be read.
    """
    with builtins.open(path, encoding="utf-8") as f:
        return compile(f.read(), entry=entry, filename=path)
