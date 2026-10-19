"""Compiles .bux source into the architecture IR.

Two passes. Evaluation resolves names, macros, layer settings and patterns into an
untyped graph. Lowering then walks it in execution order, with repeats unrolled, to
infer every layer's input width, check residuals and parallel branches for width
agreement, simulate the tap stacks, and verify that shared blocks are used consistently.

``compile`` stops at the first error. ``check`` is for editors: it recovers from
syntax errors and reports every definition that fails.
"""

import builtins
import difflib
import re
import typing
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from pydantic import ValidationError

from bux import config as cfg
from bux.lang import ir
from bux.lang import syntax as ast
from bux.lang.syntax import BuxError, Span


def _tags(union: object) -> dict[str, type[cfg.Config]]:
    classes = typing.get_args(typing.get_args(union)[0])
    return {
        tag: cls
        for cls in classes
        for tag in typing.get_args(cls.model_fields["type"].annotation)
    }


LAYERS = _tags(cfg.LayerConfig)
TYPES = LAYERS | _tags(cfg.SubConfig)
ACTIVATIONS = frozenset(typing.get_args(cfg.Activation))
RESIDUALS = _tags(cfg.ResidualConfig)
SETTINGS = ("d_model", "vocab_size", "residual")


@dataclass(frozen=True)
class _Named:
    """A value reached through a definition or macro call; graphs built from it carry the label."""

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


class _Reported(BuxError):
    """Raised on a use of a definition whose own error was already reported; check() drops it."""

    def __init__(self) -> None:
        super().__init__("already reported")


def _tag(config: cfg.Config) -> str:
    # Every concrete config has a type tag; the base class does not.
    return getattr(config, "type")  # noqa: B009


def _unwrap(v: object) -> object:
    while isinstance(v, _Named):
        v = v.value
    return v


def _is_number(v: object) -> bool:
    v = _unwrap(v)
    return isinstance(v, int | float) and not isinstance(v, bool)


def _flatten(nodes: list[ir.Node], cls: type, field: str) -> tuple[ir.Node, ...]:
    out: list[ir.Node] = []
    for node in nodes:
        if isinstance(node, cls) and node.label is None and node.share is None:
            out.extend(getattr(node, field))
        else:
            out.append(node)
    return tuple(out)


def _describe(v: object) -> str:
    match _unwrap(v):
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
        case cfg.Config() as c:
            return f"a {_tag(c)} config"
        case _Macro(statement=st):
            return f"the macro '{st.name}'"
        case _:
            return "a block"


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


class _Evaluator:
    def __init__(
        self,
        statements: list[ast.Statement],
        source: str,
        *,
        report: Callable[[BuxError], None] | None = None,
        broken: set[str] | None = None,
    ):
        """`report` collects statement errors instead of raising them (for check());
        `broken` names definitions that failed to parse, whose uses are not errors."""
        self.source = source
        self.statements: dict[str, ast.Statement] = {}
        self.broken = set(broken or ())
        for st in statements:
            error = None
            if st.name in self.statements:
                error = BuxError(f"'{st.name}' is defined twice", st.name_span)
            elif st.params is not None and st.name in SETTINGS + ("model",):
                error = BuxError(f"'{st.name}' cannot be a macro", st.name_span)
                self.broken.add(st.name)
            if error is None:
                self.statements[st.name] = st
            elif report is None:
                raise error
            else:
                report(error)
        self.values: dict[str, object] = {}
        self.evaluating: list[str] = []
        self.depth = 0
        # Source spans of graph nodes, for errors found while lowering. The list keeps
        # the nodes alive so that their ids stay unique.
        self.spans: dict[int, Span] = {}
        self._alive: list[object] = []
        # Settings given as an undefined bare name, which became a string: arg span ->
        # (name, span of the name, names in scope). Validation errors on them are
        # reported as unknown names.
        self.bare: dict[Span, tuple[str, Span, tuple[str, ...]]] = {}

    def node[N](self, node: N, span: Span) -> N:
        self.spans[id(node)] = span
        self._alive.append(node)
        return node

    def text(self, span: Span) -> str:
        return " ".join(self.source[span.start : span.end].split())

    # --- Names ------------------------------------------------------------------

    def definition(self, name: str) -> object:
        st = self.statements[name]
        if st.params is not None:
            return _Macro(st)
        if name not in self.values:
            self.evaluating.append(name)
            try:
                self.values[name] = self.eval(st.body, {})
            finally:
                self.evaluating.pop()
        return _Named(name, self.values[name], name if st.shared else None)

    def lookup(self, name: str, span: Span, scope: dict[str, object]) -> object:
        if name in scope:
            return scope[name]
        if name in self.broken and name not in self.statements:
            raise _Reported()
        if name in self.statements:
            if name not in self.evaluating:
                return self.definition(name)
            # Inside its own definition a name means the builtin it shadows: mlp = mlp(...).
            if name not in TYPES and name not in ACTIVATIONS:
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
    ) -> BuxError:
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
        return BuxError(
            f"unknown {what} '{name}'",
            span,
            f"did you mean '{close[0]}'?" if close else None,
        )

    # --- Expressions --------------------------------------------------------------

    def eval(self, e: ast.Expr, scope: dict[str, object]) -> object:
        match e:
            case ast.Lit(value=value):
                return value
            case ast.Name(name=name):
                return self.lookup(name, e.span, scope)
            case ast.Identity():
                return self.node(ir.Identity(), e.span)
            case ast.Tap(name=name):
                return self.node(ir.Tap(name=name), e.span)
            case ast.Ref(name=name, pop=pop):
                return self.node(ir.Ref(name=name, pop=pop), e.span)
            case ast.ListExpr(items=items):
                return [self.data(self.eval(item, scope), item.span) for item in items]
            case ast.Residual(body=body):
                return self.node(
                    ir.Residual(body=self.graph(self.eval(body, scope), body.span)),
                    e.span,
                )
            case ast.Seq(items=items):
                nodes = [
                    self.graph(self.eval(item, scope), item.span) for item in items
                ]
                return self.node(ir.Seq(items=_flatten(nodes, ir.Seq, "items")), e.span)
            case ast.Neg(operand=operand):
                v = self.eval(operand, scope)
                if _is_number(v):
                    return -typing.cast(float, _unwrap(v))
                return self.node(ir.Neg(body=self.graph(v, operand.span)), e.span)
            case ast.BinOp():
                return self.binop(e, scope)
            case ast.Call():
                return self.call(e, scope)
            case ast.Attr():
                return self.attr(e, scope)
        raise AssertionError(e)

    def attr(self, e: ast.Attr, scope: dict[str, object]) -> object:
        """`base.field`: a setting of a config, as written (width-dependent defaults read as none)."""
        base = _unwrap(self.eval(e.base, scope))
        match base:
            case _Builtin(name=name) if name in TYPES:
                base = self.construct(name, [], e.base.span)
            case _Builtin(name=name):
                base = cfg.ActivationConfig(fn=typing.cast(cfg.Activation, name))
            case ir.Layer(config=config):
                base = config
            case _Bare(name=name):
                raise self.unknown(name, e.base.span, scope, "name")
        if not isinstance(base, cfg.Config):
            raise BuxError(
                f"only layers and settings have fields, but this is {_describe(base)}",
                e.base.span,
            )
        fields = type(base).model_fields
        if e.field not in fields:
            close = difflib.get_close_matches(e.field, list(fields), n=1)
            raise BuxError(
                f"{_tag(base)} has no setting '{e.field}'",
                e.field_span,
                f"did you mean '{close[0]}'?" if close else None,
            )
        return getattr(base, e.field)

    def binop(self, e: ast.BinOp, scope: dict[str, object]) -> object:
        left, right = self.eval(e.left, scope), self.eval(e.right, scope)
        # In arithmetic an undefined name is a typo for a number, not a block.
        for v, other, side in ((left, right, e.left), (right, left, e.right)):
            if isinstance(v, _Bare) and (
                _is_number(other)
                or e.op in ("/", "//")
                or (e.op == "^" and side is e.right)
            ):
                raise self.unknown(v.name, side.span, scope, "name")
        if _is_number(left) and _is_number(right):
            a, b = typing.cast(float, _unwrap(left)), typing.cast(float, _unwrap(right))
            try:
                match e.op:
                    case "+":
                        return a + b
                    case "-":
                        return a - b
                    case "*":
                        return a * b
                    case "/":
                        return a / b
                    case "//":
                        return a // b
                    case "^":
                        return a**b
            except ZeroDivisionError:
                raise BuxError("division by zero", e.span) from None
        match e.op:
            case "+" | "-":
                terms = [self.term(left, e.left.span), self.term(right, e.right.span)]
                if e.op == "-":
                    last = terms[-1]
                    terms[-1] = (
                        ir.Const(value=-last.value)
                        if isinstance(last, ir.Const)
                        else self.node(ir.Neg(body=last), e.right.span)
                    )
                return self.node(ir.Sum(terms=_flatten(terms, ir.Sum, "terms")), e.span)
            case "*":
                factors = [
                    self.factor(left, e.left.span),
                    self.factor(right, e.right.span),
                ]
                return self.node(
                    ir.Prod(factors=_flatten(factors, ir.Prod, "factors")), e.span
                )
            case "^":
                times = _unwrap(right)
                if isinstance(times, bool) or not isinstance(times, int) or times < 1:
                    raise BuxError(
                        "a repeat count must be a positive integer", e.right.span
                    )
                body = self.graph(left, e.left.span)
                return (
                    body
                    if times == 1
                    else self.node(ir.Repeat(body=body, times=times), e.span)
                )
            case op:
                raise BuxError(
                    f"'{op}' only divides numbers",
                    e.span,
                    "scale a branch with `f * 0.5`",
                )

    def term(self, v: object, span: Span) -> ir.Node:
        if _is_number(v):
            return self.node(
                ir.Const(value=float(typing.cast(float, _unwrap(v)))), span
            )
        return self.graph(v, span)

    def factor(self, v: object, span: Span) -> ir.Node:
        if _is_number(v) and isinstance(_unwrap(v), int):
            n = _unwrap(v)
            raise BuxError(
                f"multiplying a block by the integer {n} is ambiguous",
                span,
                f"write blk^{n} to repeat it, or {n}.0 to scale its output",
            )
        return self.term(v, span)

    # --- Coercions ------------------------------------------------------------------

    def graph(self, v: object, span: Span) -> ir.Node:
        """`v` used as a layer or block."""
        match v:
            case _Named(label=label, value=inner, share=share):
                node = self.graph(inner, span)
                return self.node(
                    node.model_copy(
                        update={"label": label, "share": share or node.share}
                    ),
                    span,
                )
            case _ if isinstance(v, ir.NODE_TYPES):
                return typing.cast(ir.Node, v)
            case cfg.Config():
                if _tag(v) not in LAYERS:
                    raise BuxError(
                        f"{_tag(v)}(...) is a setting of a layer, not a layer", span
                    )
                return self.node(ir.Layer(config=typing.cast(cfg.LayerConfig, v)), span)
            case _Builtin(name=name):
                if name in ACTIVATIONS:
                    return self.node(
                        ir.Layer(
                            config=cfg.ActivationConfig(
                                fn=typing.cast(cfg.Activation, name)
                            )
                        ),
                        span,
                    )
                return self.graph(self.construct(name, [], span), span)
            case str():
                return self.pattern(v, span)
            case _Bare(name=name):
                raise self.unknown(name, span)
            case _Macro(statement=st):
                raise BuxError(
                    f"'{st.name}' is a macro and needs arguments: {st.name}({', '.join(st.params or ())})",
                    span,
                )
            case int() | float() if not isinstance(v, bool):
                raise BuxError(
                    "a number is not a layer",
                    span,
                    "repeat a block with blk^3, scale a branch with f * 0.5",
                )
        raise BuxError(f"expected a layer or block, found {_describe(v)}", span)

    def pattern(self, text: str, span: Span) -> ir.Node:
        """ "MMMA" is the sequence M M M A of one-letter top-level definitions."""
        nodes = []
        for k, ch in enumerate(text):
            if ch.isspace():
                continue
            where = Span(span.start + 1 + k, span.start + 2 + k)
            if ch in self.broken and ch not in self.statements:
                raise _Reported()
            if not ch.isalpha() or ch not in self.statements:
                raise BuxError(
                    f"pattern letter '{ch}' is not a one-letter definition", where
                )
            nodes.append(self.graph(self.lookup(ch, where, {}), where))
        if not nodes:
            raise BuxError("empty pattern", span)
        return (
            nodes[0] if len(nodes) == 1 else self.node(ir.Seq(items=tuple(nodes)), span)
        )

    def data(self, v: object, span: Span) -> object:
        """`v` used as a setting value."""
        match v:
            case _Named(value=inner):
                return self.data(inner, span)
            case _Builtin(name=name):
                return name if name in ACTIVATIONS else self.construct(name, [], span)
            case _Bare(name=name):
                return name
            case None | bool() | int() | float() | str() | list() | cfg.Config():
                return v
        raise BuxError(f"expected a value, found {_describe(v)}", span)

    # --- Calls ----------------------------------------------------------------------

    def call(self, e: ast.Call, scope: dict[str, object]) -> object:
        callee = self.lookup(e.name, e.name_span, scope)
        match callee:
            case _Macro(statement=st):
                return self.apply(st, e, scope)
            case _Builtin(name=name) if name in TYPES:
                fields, ports = self.arguments(e, scope, TYPES[name])
                config = self.construct(name, fields, e.span)
                return self.with_ports(config, ports, None, None, e.span)
            case _Named(label=label, share=share) if isinstance(
                _unwrap(callee), cfg.Config
            ):
                base = typing.cast(cfg.Config, _unwrap(callee))
                fields, ports = self.arguments(e, scope, type(base))
                if not fields:
                    return self.with_ports(base, ports, label, share, e.span)
                if share:
                    raise BuxError(
                        f"'{label}' is shared, so it cannot be changed here",
                        e.span,
                        "define a separate layer",
                    )
                config = self.override(base, fields, e.span)
                return self.with_ports(config, ports, self.text(e.span), None, e.span)
            case _Bare(name=name):
                raise self.unknown(name, e.name_span)
        what = "an activation" if isinstance(callee, _Builtin) else _describe(callee)
        raise BuxError(
            f"'{e.name}' is {what} and takes no arguments",
            e.name_span,
            f"to parametrize a block, make it a macro: {e.name}(x) = ..."
            if what == "a block"
            else None,
        )

    def arguments(
        self, e: ast.Call, scope: dict[str, object], cls: type[cfg.Config]
    ) -> tuple[list[tuple[tuple[str, ...], object, Span]], list[ir.Port]]:
        fields: list[tuple[tuple[str, ...], object, Span]] = []
        ports: list[ir.Port] = []
        seen: set[tuple[str, ...]] = set()
        for arg in e.args:
            if arg.key is None:
                raise BuxError(
                    "layer settings are keyword arguments: name=value", arg.span
                )
            if arg.key in seen:
                raise BuxError(f"'{'.'.join(arg.key)}' is given twice", arg.span)
            seen.add(arg.key)
            v = self.eval(arg.value, scope)
            if isinstance(v, ir.Tap | ir.Ref):
                port = arg.key[0]
                if len(arg.key) > 1 or port not in cls.ports:
                    available = ", ".join(sorted(cls.ports)) or "none"
                    raise BuxError(
                        f"no port '{'.'.join(arg.key)}' (ports: {available})", arg.span
                    )
                mode = (
                    "write" if isinstance(v, ir.Tap) else ("pop" if v.pop else "read")
                )
                ports.append(ir.Port(name=port, tap=v.name, mode=mode))
            else:
                # A macro body runs once per call, so the same argument can be a bare
                # name in one call and not in the next.
                if isinstance(v, _Bare):
                    self.bare[arg.span] = (v.name, arg.value.span, tuple(scope))
                else:
                    self.bare.pop(arg.span, None)
                fields.append((arg.key, self.data(v, arg.value.span), arg.span))
        return fields, ports

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

    def construct(
        self, tag: str, fields: list[tuple[tuple[str, ...], object, Span]], span: Span
    ) -> cfg.Config:
        data: dict = {"type": tag}
        for key, value, where in fields:
            if key == ("type",):
                raise BuxError("the type is set by the name of the call", where)
            _assign(data, key, value)
        return self.validate(TYPES[tag], data, fields, span)

    def override(
        self,
        base: cfg.Config,
        fields: list[tuple[tuple[str, ...], object, Span]],
        span: Span,
    ) -> cfg.Config:
        data = base.model_dump()
        for key, value, _ in fields:
            _assign(data, key, value)
        return self.validate(type(base), data, fields, span)

    def validate(
        self,
        cls: type[cfg.Config],
        data: dict,
        fields: list[tuple[tuple[str, ...], object, Span]],
        span: Span,
    ) -> cfg.Config:
        try:
            return cls.model_validate(data)
        except ValidationError as err:
            errors = err.errors()
            error = next(
                (e for e in errors if e["type"] == "extra_forbidden"), errors[0]
            )
            loc = tuple(str(p) for p in error["loc"])
            where = next(
                (s for k, _, s in fields if loc[: len(k)] == k or k[: len(loc)] == loc),
                span,
            )
            path = ".".join(loc)
            tag = data["type"]
            match error["type"]:
                case "extra_forbidden":
                    close = difflib.get_close_matches(
                        loc[-1], list(cls.model_fields), n=1
                    )
                    hint = (
                        f"did you mean '{close[0]}'?"
                        if close and len(loc) == 1
                        else None
                    )
                    raise BuxError(
                        f"{tag} has no setting '{path}'", where, hint
                    ) from None
                case "missing":
                    raise BuxError(
                        f"{tag} needs '{path}'", where, f"{tag}({path}=...)"
                    ) from None
            hint = None
            if error["type"] == "literal_error":
                expected = error.get("ctx", {}).get("expected", "")
                options = re.findall(r"'([^']*)'", str(expected))
                close = difflib.get_close_matches(str(error["input"]), options, n=1)
                hint = f"did you mean '{close[0]}'?" if close else None
            elif where in self.bare:
                # A bare name only works as a string, and this setting wants something
                # else, so the name was meant to refer to a definition.
                name, name_span, scope = self.bare[where]
                raise self.unknown(name, name_span, scope, "name") from None
            raise BuxError(
                f"{tag}: {path}: {error['msg']}" if path else f"{tag}: {error['msg']}",
                where,
                hint,
            ) from None

    def apply(self, st: ast.Statement, e: ast.Call, scope: dict[str, object]) -> object:
        params = st.params or ()
        bound: dict[str, object] = {}
        positional = [a for a in e.args if a.key is None]
        if len(positional) > len(params):
            raise BuxError(
                f"'{st.name}' takes {len(params)} argument(s)",
                positional[len(params)].span,
            )
        for param, arg in zip(params, positional):
            bound[param] = self.eval(arg.value, scope)
        for arg in e.args:
            if arg.key is None:
                continue
            name = ".".join(arg.key)
            if name not in params:
                raise BuxError(f"'{st.name}' has no parameter '{name}'", arg.span)
            if name in bound:
                raise BuxError(f"'{name}' is given twice", arg.span)
            bound[name] = self.eval(arg.value, scope)
        defaults = dict(st.defaults)
        if missing := [p for p in params if p not in bound and p not in defaults]:
            raise BuxError(
                f"'{st.name}' is missing argument(s): {', '.join(missing)}", e.span
            )
        # A default sees the parameters before it: blk(n, kv=n // 4) = ...
        for i, param in enumerate(params):
            if param not in bound:
                earlier = {p: bound[p] for p in params[:i]}
                bound[param] = self.eval(defaults[param], earlier)
        self.depth += 1
        if self.depth > 64:
            raise BuxError(f"'{st.name}' recurses too deeply", e.span)
        try:
            return _Named(self.text(e.span), self.eval(st.body, bound))
        finally:
            self.depth -= 1

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
        self.where: list[Span] = []
        self.embeddings = 0
        # Layers that reuse the embedding matrix (tied heads, nextn), with their spans.
        self.borrowers: list[tuple[str, Span]] = []
        self.heads = 0

    def error(self, message: str, hint: str | None = None) -> BuxError:
        return BuxError(message, self.where[-1] if self.where else None, hint)

    @contextmanager
    def frame(self, label: str) -> Iterator[None]:
        frame = _Frame(label)
        self.frames.append(frame)
        try:
            yield
        finally:
            self.frames.pop()
            for entry in frame.used:
                if entry.used_in is frame and not entry.popped:
                    entry.owner = label

    def run(self, graph: ir.Node) -> ir.Node:
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
        span = self.ev.spans.get(id(node))
        if span is not None:
            self.where.append(span)
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
                self.where.pop()

    def _lower(self, node: ir.Node, width: ir.Width) -> tuple[ir.Node, ir.Width | None]:
        match node:
            case ir.Layer():
                return self.layer(node, width)
            case ir.Seq(items=items):
                typed = []
                for item in items:
                    t, out = self.lower(item, width)
                    # A layer that needed a merge came back as an unlabeled (merge, layer).
                    if isinstance(t, ir.Seq) and t.label is None and t.share is None:
                        typed.extend(t.items)
                    else:
                        typed.append(t)
                    width = typing.cast(ir.Width, out)
                return ir.replace_children(node, tuple(typed)), width
            case ir.Residual(body=body):
                if isinstance(body, ir.Layer) and isinstance(
                    body.config, cfg.EngramConfig
                ):
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
                            "only branches, taps, engram and nextn keep it; save it "
                            "with @h and go on from $h",
                        )
                    inner = self.d_model
                elif not isinstance(width, int):
                    raise self.error(
                        f"a residual branch needs hidden states, but receives {width}"
                    )
                self.depth += 1
                try:
                    t, out = self.lower(body, inner)
                finally:
                    self.depth -= 1
                if out != inner:
                    raise self.error(
                        f"a residual branch must keep the width, but this one maps {inner} to {out}"
                    )
                return ir.replace_children(node, (t,)), width
            case ir.Sum(terms=terms) | ir.Prod(factors=terms):
                typed, outs = [], []
                for term in terms:
                    t, out = self.lower(term, width)
                    typed.append(t)
                    if out is not None and out not in outs:
                        outs.append(out)
                if not outs:
                    raise self.error("parallel branches need at least one layer")
                if len(outs) > 1:
                    raise self.error(
                        f"parallel branches disagree on width: {', '.join(map(str, outs))}"
                    )
                return ir.replace_children(node, tuple(typed)), outs[0]
            case ir.Neg(body=body):
                t, out = self.lower(body, width)
                return ir.replace_children(node, (t,)), out
            case ir.Const():
                return node, None
            case ir.Identity():
                return node, width
            case ir.Repeat(body=body, times=times):
                span = self.ev.spans.get(id(body))
                text = self.ev.text(span) if span is not None else "the block"
                label = f"one pass of {f'({text})' if ' ' in text else text}^{times}"
                with self.frame(label):
                    t, out = self.lower(body, width)
                if out != width:
                    raise self.error(
                        f"a repeated block must keep the width, but this one maps {width} to {out}"
                    )
                for _ in range(times - 1):
                    with self.frame(label):
                        self.lower(body, width)
                return ir.replace_children(node, (t,)), width
            case ir.Tap(name=name):
                self.write(name, width)
                return node, width
            case ir.Ref(name=name, pop=pop):
                value = self.read(name, pop)
                if isinstance(value, _PortValue):
                    raise self.error(
                        f"tap '{name}' holds the '{value.port}' port of a layer",
                        f"read it through the same port: layer({value.port}=${name})",
                    )
                if value == "streams" and self.depth > 0:
                    raise self.error(
                        f"tap '{name}' holds the widened residual stream, which only "
                        "exists outside residual branches",
                        f"read it back into one first: merge @{name}",
                    )
                return node, value
        raise AssertionError(node)

    def write(self, tap: str, value: ir.Width | _PortValue) -> None:
        self.stacks.setdefault(tap, []).append(_Entry(value, tuple(self.frames)))

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
        if entry.owner is not None and any(e.owner is None for e in stack[:-1]):
            # A block wrote the tap for its own use and left the value behind, where
            # it now hides what the reader most likely wants.
            raise self.error(
                f"tap '{tap}' here reads the value that '{entry.owner}' wrote and "
                "read for itself, not the one written before it",
                f"pop it in '{entry.owner}' with ${tap}! where it is read last, "
                "or give one of the two taps another name",
            )
        if entry.owner is None:
            common = None
            for written, current in zip(entry.frames, self.frames):
                if written is not current:
                    break
                common = written
            if common is not None:
                entry.used_in = common
                common.used.append(entry)
        if pop:
            stack.pop()
            entry.popped = True
        return entry.value

    def layer(self, node: ir.Layer, width: ir.Width) -> tuple[ir.Node, ir.Width]:
        config = node.config
        out: ir.Width
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
        if isinstance(config, cfg.MergeConfig):
            if width != "streams":
                raise self.error(
                    "merge reads a widened residual stream back into one, but there "
                    "is none here",
                    "widen it with residual = hc(...), mhc(...), gr(...) or attnres(...)"
                    if not self.widened
                    else None,
                )
            out = self.d_model
        elif config.type == "embed":
            if width != "tokens":
                raise self.error("an embedding takes token ids, so it must come first")
            self.embeddings += 1
            out = "streams" if self.widened and self.depth == 0 else self.d_model
        elif width == "tokens":
            raise self.error(
                f"{config.type} needs hidden states, but receives token ids",
                "start with an embedding",
            )
        elif isinstance(config, cfg.LMHeadConfig):
            self.heads += 1
            if config.tied and self.where:
                self.borrowers.append(("a tied head", self.where[-1]))
            out = "logits"
        elif isinstance(config, cfg.LinearConfig):
            out = config.features
        elif isinstance(config, cfg.NextNConfig):
            if self.where:
                self.borrowers.append(("nextn", self.where[-1]))
            out = "streams" if self.widened and self.depth == 0 else width
        else:
            out = width
        for port in sorted(node.ports, key=lambda p: p.mode == "write"):
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
        return node.model_copy(update={"width": width}), out


def _compile(ev: _Evaluator, entry: str) -> ir.Architecture:
    if entry not in ev.statements:
        if entry in ev.broken:
            raise _Reported()
        raise BuxError(
            f"no definition named '{entry}'",
            hint=f"define it: {entry} = embed ... head",
        )
    settings = {}
    for name in SETTINGS:
        value, span = ev.setting(name)
        expected = tuple(RESIDUALS.values()) if name == "residual" else int
        if value is not None and (
            not isinstance(value, expected) or isinstance(value, bool)
        ):
            kind = (
                " or ".join(f"{t}(...)" for t in RESIDUALS)
                if name == "residual"
                else "an integer"
            )
            raise BuxError(f"'{name}' must be {kind}, found {_describe(value)}", span)
        settings[name] = value
    if settings["d_model"] is None:
        raise BuxError("'d_model' is not defined", hint="d_model = 4096")
    st = ev.statements[entry]
    lowering = _Lowering(
        ev,
        typing.cast(int, settings["d_model"]),
        typing.cast(cfg.ResidualConfig | None, settings["residual"]),
    )
    graph = lowering.run(ev.graph(ev.definition(entry), st.name_span))
    if settings["vocab_size"] is None and (lowering.embeddings or lowering.heads):
        raise BuxError("'vocab_size' is not defined", hint="vocab_size = 32000")
    return ir.Architecture(
        d_model=typing.cast(int, settings["d_model"]),
        vocab_size=typing.cast(int | None, settings["vocab_size"]),
        residual=typing.cast(cfg.ResidualConfig | None, settings["residual"]),
        graph=graph,
    )


def compile(
    source: str, *, entry: str = "model", filename: str = "<string>"
) -> ir.Architecture:
    """Compile .bux source; `entry` names the definition to build (a whole model by default)."""
    try:
        ev = _Evaluator(ast.parse(source), source)
        for name, st in ev.statements.items():
            if st.params is None:
                ev.definition(name)
        return _compile(ev, entry)
    except BuxError as err:
        raise err.attach(source, filename) from None


@dataclass
class Report:
    """What `check` found in a source file."""

    statements: list[ast.Statement]
    errors: list[BuxError]
    # Only when there are no errors; then it equals what compile() returns.
    arch: ir.Architecture | None
    # Definitions that evaluated, by name: numbers, strings, lists, configs or IR nodes.
    values: dict[str, object]


def check(source: str, *, entry: str = "model", filename: str = "<string>") -> Report:
    """Compile like `compile`, but collect every error instead of raising the first.

    Syntax errors lose only their own statement, each definition is checked on its
    own, and an error is not repeated at every use of a broken definition.
    """
    errors: list[BuxError] = []

    def report(err: BuxError) -> None:
        if isinstance(err, _Reported):
            return
        if all((e.message, e.span) != (err.message, err.span) for e in errors):
            errors.append(err.attach(source, filename))

    statements, syntax_errors, broken = ast.parse_recovering(source)
    for err in syntax_errors:
        report(err)
    ev = _Evaluator(statements, source, report=report, broken=broken)
    for name, st in ev.statements.items():
        if st.params is None:
            try:
                ev.definition(name)
            except BuxError as err:
                report(err)
    arch = None
    try:
        arch = _compile(ev, entry)
    except BuxError as err:
        report(err)
    values = {
        name: public
        for name, v in ev.values.items()
        if (public := _public(v)) is not None
    }
    return Report(statements, errors, None if errors else arch, values)


def _public(v: object) -> object | None:
    """A definition's value as Report.values shows it: no private evaluation types."""
    v = _unwrap(v)
    match v:
        case _Builtin(name=name) if name in ACTIVATIONS:
            return cfg.ActivationConfig(fn=typing.cast(cfg.Activation, name))
        case _Builtin(name=name):
            try:
                return TYPES[name].model_validate({"type": name})
            except ValidationError:
                return None
        case _Bare(name=name):
            return name
    return v


def compile_file(path: str, *, entry: str = "model") -> ir.Architecture:
    with builtins.open(path, encoding="utf-8") as f:
        return compile(f.read(), entry=entry, filename=path)
