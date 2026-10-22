"""Tokens, AST and parser of the .bux language.

Precedence, loosest first::

    a | b   x: a | y: b  alternatives: one run for each (a sweep), optionally labeled
    a + b   a - b        parallel branches, summed
    a * b                parallel branches, multiplied (gating) or scaled by a number
    a b     a -> b       composition, left to right
    -a                   negation
    a^n                  n fresh copies of a in sequence (or a number power)
    atoms                name, call(k=v), name.field, 1.5, 10B, "pattern", _, @tap, $tap,
                         $tap!, (a), [a]

Statements are ``name = expr``, ``name(param, param=default, ...) = expr`` (a macro),
``shared name = expr``, ``name.key = expr`` (an amendment: ``name = name(key=expr)``)
or ``use module``. A statement ends at a newline unless the next line is indented or
a bracket is still open; ``;`` also ends one. ``#`` starts a comment.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def to(self, other: Span) -> Span:
        return Span(self.start, other.end)


class BuxError(Exception):
    def __init__(self, message: str, span: Span | None = None, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.span = span
        self.hint = hint
        self.source: str | None = None
        self.filename = "<string>"

    def attach(self, source: str, filename: str) -> BuxError:
        if self.source is None:
            self.source, self.filename = source, filename
        return self

    def __str__(self) -> str:
        lines = [f"error: {self.message}"]
        if self.source is not None and self.span is not None:
            src, start = self.source, min(self.span.start, len(self.source))
            line_no = src.count("\n", 0, start) + 1
            line_start = src.rfind("\n", 0, start) + 1
            line_end = src.find("\n", start)
            line_end = len(src) if line_end == -1 else line_end
            col = start - line_start
            carets = "^" * max(1, min(self.span.end, line_end) - start)
            pad = " " * len(str(line_no))
            lines += [
                f"{pad}--> {self.filename}:{line_no}:{col + 1}",
                f"{pad} |",
                f"{line_no} | {src[line_start:line_end]}",
                f"{pad} | {' ' * col}{carets}",
            ]
            if self.hint:
                lines.append(f"{pad} = hint: {self.hint}")
        elif self.hint:
            lines.append(f"hint: {self.hint}")
        return "\n".join(lines)


# --- Tokens ---------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    kind: str  # NAME, NUMBER, STRING, OP, NEWLINE, EOF
    text: str
    span: Span
    space_before: bool
    value: int | float | str | None = None


# A count can end in k, M, B or T: 128k, 1.5B, 10T. The suffix must end the token, so
# 4kv is the number 4 followed by the name kv.
_NUMBER = re.compile(
    r"\d[\d_]*(\.\d[\d_]*)?([eE][+-]?\d+)?(?:([kKMBT])(?![A-Za-z0-9_]))?"
)
_SUFFIXES = {"k": 3, "K": 3, "M": 6, "B": 9, "T": 12}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_OPS = (
    "->",
    "//",
    "→",
    "|",
    ":",
    "+",
    "-",
    "*",
    "/",
    "^",
    "=",
    ",",
    "(",
    ")",
    "[",
    "]",
    "@",
    "$",
    "!",
    ".",
    ";",
)
KEYWORDS = frozenset({"true", "false", "none", "shared", "_"})


def _continues(src: str, i: int, stop: int) -> bool:
    """Return whether the next non-blank, non-comment line after `i` is indented."""
    while i < stop:
        j = i
        while j < stop and src[j] in " \t\r":
            j += 1
        if j < stop and src[j] not in "\n#":
            return j > i
        newline = src.find("\n", j, stop)
        if newline == -1:
            return False
        i = newline + 1
    return False


def _token(
    src: str, i: int, stop: int, space: bool, fail: Callable[[BuxError], None]
) -> Token | None:
    if m := _NUMBER.match(src, i, stop):
        return _number(m, space)
    if m := _NAME.match(src, i, stop):
        return Token("NAME", m.group(), Span(i, m.end()), space)
    if src[i] in "\"'":
        return _string(src, i, stop, space, fail)
    op = next((op for op in _OPS if src.startswith(op, i)), None)
    return None if op is None else Token("OP", op, Span(i, i + len(op)), space)


def _number(m: re.Match[str], space: bool) -> Token:
    text = m.group()
    clean = text.replace("_", "")
    value: int | float
    if suffix := m.group(3):
        # Exact, so that 1.1B is the integer 1_100_000_000.
        exact = Decimal(clean[:-1]).scaleb(_SUFFIXES[suffix])
        value = int(exact) if exact == exact.to_integral() else float(exact)
    else:
        value = float(clean) if (m.group(1) or m.group(2)) else int(clean)
    return Token("NUMBER", text, Span(m.start(), m.end()), space, value)


def _string(
    src: str, i: int, stop: int, space: bool, fail: Callable[[BuxError], None]
) -> Token:
    end = src.find(src[i], i + 1, stop)
    if end == -1 or "\n" in src[i:end]:
        fail(BuxError("unterminated string", Span(i, i + 1)))
        # Recovering: the string runs to the end of the line.
        end = src.find("\n", i, stop)
        end = stop if end == -1 else end
        return Token("STRING", src[i:end], Span(i, end), space, src[i + 1 : end])
    return Token("STRING", src[i : end + 1], Span(i, end + 1), space, src[i + 1 : end])


def tokenize(
    src: str,
    start: int = 0,
    stop: int | None = None,
    errors: list[BuxError] | None = None,
) -> list[Token]:
    """Split ``src[start:stop]`` into tokens, with spans into all of `src`.

    Errors raise, unless an `errors` list is given: then they are appended to it and
    tokenizing goes on, which editors need for half-typed source.
    """

    def fail(error: BuxError) -> None:
        if errors is None:
            raise error
        errors.append(error)

    stop = len(src) if stop is None else stop
    tokens: list[Token] = []
    i, depth, space = start, 0, True
    while i < stop:
        ch = src[i]
        if ch == "\n":
            if (
                depth == 0
                and tokens
                and tokens[-1].kind != "NEWLINE"
                and not _continues(src, i + 1, stop)
            ):
                tokens.append(Token("NEWLINE", "\n", Span(i, i + 1), space))
            i, space = i + 1, True
        elif ch in " \t\r":
            i, space = i + 1, True
        elif ch == "#":
            newline = src.find("\n", i, stop)
            i, space = (stop if newline == -1 else newline), True
        elif (token := _token(src, i, stop, space, fail)) is None:
            fail(BuxError(f"unexpected character {ch!r}", Span(i, i + 1)))
            i, space = i + 1, False
        else:
            if token.kind == "OP" and token.text in ("(", "["):
                depth += 1
            elif token.kind == "OP" and token.text in (")", "]"):
                depth = max(0, depth - 1)
            tokens.append(token)
            i, space = token.span.end, False
    # "found end of input" points just after the last token, not past trailing lines.
    last = next((t.span.end for t in reversed(tokens) if t.kind != "NEWLINE"), stop)
    tokens.append(Token("EOF", "", Span(last, last), space))
    return tokens


# --- AST ------------------------------------------------------------------------


@dataclass(frozen=True)
class Expr:
    span: Span


@dataclass(frozen=True)
class Name(Expr):
    name: str


@dataclass(frozen=True)
class Lit(Expr):
    value: int | float | str | bool | None


@dataclass(frozen=True)
class ListExpr(Expr):
    items: tuple[Expr, ...]


@dataclass(frozen=True)
class Identity(Expr):
    pass


@dataclass(frozen=True)
class Tap(Expr):
    name: str


@dataclass(frozen=True)
class Ref(Expr):
    name: str
    pop: bool


@dataclass(frozen=True)
class Residual(Expr):
    body: Expr


@dataclass(frozen=True)
class Seq(Expr):
    items: tuple[Expr, ...]


@dataclass(frozen=True)
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Neg(Expr):
    operand: Expr


@dataclass(frozen=True)
class Arg:
    span: Span
    key: tuple[str, ...] | None  # dotted keyword, or None for a positional argument
    value: Expr
    key_spans: tuple[Span, ...] = ()  # one per part of the key


@dataclass(frozen=True)
class Call(Expr):
    name: str
    name_span: Span
    args: tuple[Arg, ...]
    amend: bool = False  # the body of `name.key = value`, which calls what it amends


@dataclass(frozen=True)
class Alt(Expr):
    """Alternatives, ``a | b``: each run of the program takes one of them.

    Unlabeled alternatives are an axis of their own. Labeled ones are an axis named by
    their labels, so alternatives elsewhere with the same labels are picked together.
    """

    options: tuple[Expr, ...]
    labels: tuple[str, ...]  # one per option, or none
    where: str  # the definition and setting it is written in: "optimizer.lr"


@dataclass(frozen=True)
class Attr(Expr):
    """A setting read off a config: ``attn.num_heads``."""

    base: Expr
    field: str
    field_span: Span


@dataclass(frozen=True)
class Statement:
    span: Span
    name: str
    name_span: Span
    params: tuple[str, ...] | None  # None for plain definitions, a tuple for macros
    body: Expr
    shared: bool = False
    defaults: tuple[tuple[str, Expr], ...] = ()
    param_spans: tuple[Span, ...] = ()
    amends: bool = False  # `name.key = value`: its body is an amending Call


@dataclass(frozen=True)
class Use:
    """``use module``: the definitions of another file, which this one can redefine."""

    span: Span
    module: str
    module_span: Span


# --- Parser ---------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        self.uses: list[Use] = []
        # The definition and setting keys being parsed, which name alternatives.
        self.context: list[str] = []

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]

    def next(self) -> Token:
        tok = self.peek()
        self.pos += 1
        return tok

    def at(self, *ops: str) -> bool:
        tok = self.peek()
        return tok.kind == "OP" and tok.text in ops

    def expect(self, op: str) -> Token:
        if not self.at(op):
            raise self.unexpected(f"expected '{op}'")
        return self.next()

    def expect_name(self, what: str) -> Token:
        tok = self.peek()
        if tok.kind != "NAME" or tok.text in KEYWORDS:
            raise self.unexpected(f"expected {what}")
        return self.next()

    def unexpected(self, message: str) -> BuxError:
        tok = self.peek()
        found = {"NEWLINE": "end of line", "EOF": "end of input"}.get(
            tok.kind, repr(tok.text)
        )
        return BuxError(f"{message}, found {found}", tok.span)

    # program := { statement }
    def program(self) -> list[Statement]:
        statements = []
        while True:
            while self.peek().kind == "NEWLINE" or self.at(";"):
                self.next()
            if self.peek().kind == "EOF":
                return statements
            first = self.peek()
            if first.text == "use" and self.peek(1).kind in ("NAME", "STRING"):
                self.next()
                module = self.next()
                name = str(module.value) if module.kind == "STRING" else module.text
                self.uses.append(Use(first.span.to(module.span), name, module.span))
            else:
                statements.append(self.statement())
            if not (self.peek().kind in ("NEWLINE", "EOF") or self.at(";")):
                raise self.unexpected("expected end of statement")

    def statement(self) -> Statement:
        first = self.peek()
        shared = (
            first.kind == "NAME"
            and first.text == "shared"
            and self.peek(1).kind == "NAME"
        )
        if shared:
            self.next()
        name = self.expect_name("a definition name")
        self.context = [name.text]
        if self.at(".") and not self.peek().space_before:
            if shared:
                raise BuxError(
                    "an amendment keeps whether the definition is shared",
                    first.span,
                    f"leave out 'shared': {name.text}.key = ...",
                )
            return self.amendment(name)
        params, defaults, param_spans = None, (), ()
        if self.at("(") and not self.peek().space_before:
            if shared:
                raise BuxError(
                    "a macro cannot be shared; share the blocks it builds instead",
                    name.span,
                )
            params, defaults, param_spans = self.parameters()
        self.expect("=")
        body = self.expr()
        return Statement(
            first.span.to(body.span),
            name.text,
            name.span,
            params,
            body,
            shared,
            defaults,
            param_spans,
        )

    def parameters(
        self,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, Expr], ...], tuple[Span, ...]]:
        self.expect("(")
        params: list[str] = []
        defaults: list[tuple[str, Expr]] = []
        spans: list[Span] = []
        while not self.at(")"):
            param = self.expect_name("a parameter name")
            if param.text in params:
                raise BuxError(f"parameter '{param.text}' is listed twice", param.span)
            params.append(param.text)
            spans.append(param.span)
            if self.at("="):
                self.next()
                defaults.append((param.text, self.expr()))
            elif defaults:
                raise BuxError(
                    "parameters without a default must come first", param.span
                )
            if not self.at(","):
                break
            self.next()
        self.expect(")")
        return tuple(params), tuple(defaults), tuple(spans)

    # amendment := NAME "." NAME { "." NAME } "=" expr, which means NAME = NAME(key=expr)
    def amendment(self, name: Token) -> Statement:
        parts: list[Token] = []
        while self.at(".") and not self.peek().space_before:
            self.next()
            parts.append(self.expect_name("a setting name after '.'"))
        key = tuple(p.text for p in parts)
        self.context += key
        self.expect("=")
        value = self.expr()
        span = name.span.to(value.span)
        arg = Arg(
            parts[0].span.to(value.span), key, value, tuple(p.span for p in parts)
        )
        body = Call(span, name.text, name.span, (arg,), amend=True)
        return Statement(span, name.text, name.span, None, body, amends=True)

    # expr := option { "|" option }      option := [NAME ":"] sum
    def expr(self) -> Expr:
        labels = [self.label()]
        options = [self.sum()]
        while self.at("|"):
            self.next()
            labels.append(self.label())
            options.append(self.sum())
        span = options[0].span.to(options[-1].span)
        if len(options) == 1:
            if labels[0] is not None:
                raise BuxError(
                    "a label names one of several alternatives",
                    span,
                    f"{labels[0]}: ... | other: ...",
                )
            return options[0]
        named = [label for label in labels if label is not None]
        if named and len(named) < len(labels):
            raise BuxError("label every alternative or none of them", span)
        if len(set(named)) < len(named):
            raise BuxError("two alternatives have the same label", span)
        return Alt(span, tuple(options), tuple(named), ".".join(self.context))

    def label(self) -> str | None:
        tok, colon = self.peek(), self.peek(1)
        if tok.kind != "NAME" or tok.text in KEYWORDS:
            return None
        if colon.kind != "OP" or colon.text != ":":
            return None
        self.pos += 2
        return tok.text

    # sum := prod { ("+" | "-") prod }
    def sum(self) -> Expr:
        left = self.prod()
        while self.at("+", "-"):
            op = self.next().text
            right = self.prod()
            left = BinOp(left.span.to(right.span), op, left, right)
        return left

    # prod := chain { ("*" | "/" | "//") chain }
    def prod(self) -> Expr:
        left = self.chain()
        while self.at("*", "/", "//"):
            op = self.next().text
            right = self.chain()
            left = BinOp(left.span.to(right.span), op, left, right)
        return left

    # chain := unary { ["->"] unary }
    def chain(self) -> Expr:
        items = [self.unary()]
        while True:
            if self.at("->", "→"):
                self.next()
            elif not self.starts_atom():
                break
            items.append(self.unary())
        if len(items) == 1:
            return items[0]
        return Seq(items[0].span.to(items[-1].span), tuple(items))

    def starts_atom(self) -> bool:
        tok = self.peek()
        return tok.kind in ("NAME", "NUMBER", "STRING") or self.at("(", "[", "@", "$")

    # unary := "-" unary | power
    def unary(self) -> Expr:
        if self.at("-"):
            start = self.next().span
            operand = self.unary()
            return Neg(start.to(operand.span), operand)
        return self.power()

    # power := atom [ "^" unary ]
    def power(self) -> Expr:
        base = self.atom()
        if self.at("^"):
            self.next()
            exponent = self.unary()
            return BinOp(base.span.to(exponent.span), "^", base, exponent)
        return base

    def atom(self) -> Expr:
        tok = self.peek()
        if tok.kind == "NUMBER" or tok.kind == "STRING":
            self.next()
            return Lit(tok.span, tok.value)
        if tok.kind == "NAME":
            self.next()
            match tok.text:
                case "_":
                    return Identity(tok.span)
                case "true" | "false":
                    return Lit(tok.span, tok.text == "true")
                case "none":
                    return Lit(tok.span, None)
                case "shared":
                    raise BuxError(
                        "'shared' only starts a definition: shared name = ...", tok.span
                    )
            if self.at("(") and not self.peek().space_before:
                node: Expr = self.call(tok)
            else:
                node = Name(tok.span, tok.text)
            # name.field, with no space before the dot: attn.num_heads
            while self.at(".") and not self.peek().space_before:
                self.next()
                field = self.expect_name("a setting name after '.'")
                node = Attr(node.span.to(field.span), node, field.text, field.span)
            return node
        if self.at("@"):
            self.next()
            name = self.expect_name("a tap name after '@'")
            return Tap(tok.span.to(name.span), name.text)
        if self.at("$"):
            self.next()
            name = self.expect_name("a tap name after '$'")
            end, pop = name.span, self.at("!") and not self.peek().space_before
            if pop:
                end = self.next().span
            return Ref(tok.span.to(end), name.text, pop)
        if self.at("("):
            self.next()
            inner = self.expr()
            self.expect(")")
            return inner
        if self.at("["):
            return self.bracket()
        raise self.unexpected("expected a layer, block or value")

    # "[" expr "]" is a residual branch; with commas (or empty) it is a list.
    def bracket(self) -> Expr:
        start = self.expect("[").span
        if self.at("]"):
            return ListExpr(start.to(self.next().span), ())
        items = [self.expr()]
        is_list = False
        while self.at(","):
            self.next()
            is_list = True
            if self.at("]"):
                break
            items.append(self.expr())
        span = start.to(self.expect("]").span)
        return ListExpr(span, tuple(items)) if is_list else Residual(span, items[0])

    # call := NAME "(" [ arg { "," arg } [","] ] ")"   with no space before "("
    def call(self, name: Token) -> Call:
        self.expect("(")
        args = []
        while not self.at(")"):
            start = self.peek().span
            key, key_spans = None, []
            if self.is_keyword():
                parts = [self.next()]
                while self.at("."):
                    self.next()
                    parts.append(self.expect_name("a field name"))
                self.expect("=")
                key = tuple(p.text for p in parts)
                key_spans = [p.span for p in parts]
            outer = self.context
            self.context = outer + list(key or ())
            try:
                value = self.expr()
            finally:
                self.context = outer
            args.append(Arg(start.to(value.span), key, value, tuple(key_spans)))
            if not self.at(","):
                break
            self.next()
        end = self.expect(")").span
        return Call(name.span.to(end), name.text, name.span, tuple(args))

    def is_keyword(self) -> bool:
        i = 0
        while True:
            if self.peek(i).kind != "NAME":
                return False
            sep = self.peek(i + 1)
            if sep.kind == "OP" and sep.text == "=":
                return True
            if not (sep.kind == "OP" and sep.text == "."):
                return False
            i += 2


@dataclass
class Parsed:
    """The result of parse_file: the definitions and the modules used. A recovering
    parse adds the syntax errors and the names of definitions that failed to parse."""

    statements: list[Statement]
    uses: list[Use]
    errors: list[BuxError] = field(default_factory=list)
    broken: set[str] = field(default_factory=set)


def parse(source: str) -> list[Statement]:
    return parse_file(source).statements


_STATEMENT_START = re.compile(r"^[^\s#)\]]", re.MULTILINE)
_HEADER = re.compile(r"(?:shared[ \t]+)?([A-Za-z_][A-Za-z0-9_]*)")


def statement_start(source: str, offset: int) -> int:
    """Return where the statement around `offset` starts, as parse_file recovers."""
    return max(
        (m.start() for m in _STATEMENT_START.finditer(source, 0, offset)), default=0
    )


def parse_file(
    source: str, start: int = 0, stop: int | None = None, *, recover: bool = False
) -> Parsed:
    """Parse ``source[start:stop]``, with spans into all of `source`.

    Without `recover`, the first syntax error raises. With it, as much as possible is
    parsed, for editors: when the whole does not parse, every statement is parsed on
    its own. A statement then starts at each line that begins in the first column, so
    an unclosed bracket loses only its own statement.
    """
    stop = len(source) if stop is None else stop
    if not recover:
        parser = _Parser(tokenize(source, start, stop))
        return Parsed(parser.program(), parser.uses)
    try:
        parser = _Parser(tokenize(source, start, stop))
        return Parsed(parser.program(), parser.uses)
    except BuxError as err:
        whole = err
    return _parse_each(source, start, stop, whole)


def _parse_each(source: str, start: int, stop: int, whole: BuxError) -> Parsed:
    parsed = Parsed([], [])
    starts = sorted(
        {start} | {m.start() for m in _STATEMENT_START.finditer(source, start, stop)}
    )
    for begin, end in zip(starts, [*starts[1:], stop]):
        tokenizer_errors: list[BuxError] = []
        parser = _Parser(tokenize(source, begin, end, tokenizer_errors))
        try:
            if tokenizer_errors:
                raise tokenizer_errors[0]
            parsed.statements += parser.program()
        except BuxError as err:
            parsed.errors.append(err)
            if m := _HEADER.match(source, begin):
                parsed.broken.add(m.group(1))
        parsed.uses += parser.uses
    # Every statement parsed on its own, but the whole did not: report why.
    parsed.errors = parsed.errors or [whole]
    return parsed


def parse_recovering(source: str) -> tuple[list[Statement], list[BuxError], set[str]]:
    """Parse as much as possible, for editors: see parse_file."""
    parsed = parse_file(source, recover=True)
    return parsed.statements, parsed.errors, parsed.broken
