(lang-grammar)=
# Lexical rules and grammar

This page is the exact syntax of a `.bux` file: its statements and tokens, where
statements end, the places where a space changes the meaning, and the grammar. Read
it when a line does not parse as you expect. The other pages of this guide explain
what each construct means.

## Statements

A file is a sequence of statements:

| Statement | Example |
|-----------|---------|
| a definition | `norm = rmsnorm(eps=1e-5)` |
| a shared definition | `shared out = head` |
| a macro | `layer(mixer, ffn=dense) = [norm mixer] [norm ffn]` |
| an amendment | `attn.num_kv_heads = 1` |
| a module | `use llama3_8b`, `use "qwen3.8_27b"` |

A statement ends at a newline, unless a bracket, `(` or `[`, is still open, or the
next line that is not blank or a comment is indented. A `;` also ends a statement,
so several fit on one line: `d_model = 512; vocab_size = 32000`.

```bux
model = embed norm @x0
        (block @skip)^6              # indented: the statement continues
        ([$skip! scale] block)^6
        norm head
attn  = gqa(num_heads=8,             # an open bracket: the statement continues
num_kv_heads=2)
```

## Tokens

| Token | Form | Examples |
|-------|------|----------|
| name | a letter or `_`, then letters, digits and `_` | `attn`, `D`, `d_model`, `_x` |
| number | digits with `_` between them, an optional fraction and exponent, and an optional suffix | `4096`, `128_256`, `1e-5`, `0.5`, `32k`, `1.5B` |
| string | text between `"` or `'`, on one line, without escapes | `"gpt2"`, `'it"s'` |
| operator | `->` `→` `//` `\|` `:` `+` `-` `*` `/` `^` `=` `,` `(` `)` `[` `]` `@` `$` `!` `.` `;` | |
| comment | `#` to the end of the line | `# pre-norm` |

- **Numbers.** A number has digits before any `.`, so `.5` and `1.` are not numbers,
  and a `_` goes between two digits, so `1_` and `1__000` are errors. One with a `.`
  or an exponent is a float; otherwise it is an integer. The suffix
  `k` or `K`, `M`, `B` or `T` multiplies by 10^3, 10^6, 10^9 or 10^12 exactly, and
  must end the token: `4kv` is the number `4` and the name `kv`
  ([Numbers and suffixes](values.md#numbers-and-suffixes)).
- **Keywords.** `true`, `false`, `none`, `shared` and `_` are reserved: they cannot
  name a definition, a parameter or a tap. `use` is not reserved; it starts a
  module statement only when a name or a string follows it.
- **Any other character**, such as `%`, is an error: `unexpected character '%'`.

## Where a space matters

Four constructs need their punctuation right after a name, with no space:

| Written | Means | With a space |
|---------|-------|--------------|
| `norm(eps=1e-5)` | a call: `norm` configured | `norm (attn + ffn)`: `norm`, then a group |
| `layer(mixer) = ...` | a macro | `layer (mixer) = ...` is an error |
| `attn.num_heads`, `attn.num_heads = 8`, `gqa(rope.base=1e6)` | a field read, an amendment, a dotted key | `attn .num_heads` and `attn. num_heads` are errors |
| `$h!` | read and pop | `$h !` is an error |

A `-` between two items of a composition is a subtraction whatever the spacing:
`f -g` is `f - g`. Inside a composition, it negates only after an arrow: `f -> -g`.
Elsewhere, such as at the start of an expression or after `+`, `*` or `(`, it
negates: `-f g`, `f + -g`.

## Grammar

The grammar below matches the parser, `bux/lang/syntax.py`. `NAME`, `NUMBER` and
`STRING` are the tokens above. `{ x }` repeats `x` zero or more times, and `[ x ]`
makes it optional.

```text
program    := [ item ] { ( NEWLINE | ";" ) [ item ] }
item       := use | definition | macro | amendment
use        := "use" ( NAME | STRING )
definition := [ "shared" ] NAME "=" expr
macro      := NAME "(" [ param { "," param } [ "," ] ] ")" "=" expr
param      := NAME [ "=" expr ]                    # parameters with defaults come last
amendment  := NAME "." NAME { "." NAME } "=" expr  # NAME = NAME(key.key=expr)

expr       := option { "|" option }                # alternatives
option     := [ NAME ":" ] sum                     # label all options, or none
sum        := prod { ( "+" | "-" ) prod }
prod       := chain { ( "*" | "/" | "//" ) chain }
chain      := unary { ( "->" | "→" ) unary | power }
unary      := "-" unary | power
power      := atom [ "^" unary ]

atom       := NAME [ call ] { "." NAME }
            | NUMBER | STRING | "_" | "true" | "false" | "none"
            | "@" NAME | "$" NAME [ "!" ]
            | "(" expr ")"
            | "[" expr "]"                         # a residual branch
            | "[" [ expr { "," expr } [ "," ] ] "]" # a list: it has a comma, or is empty
call       := "(" [ arg { "," arg } [ "," ] ] ")"
arg        := [ NAME { "." NAME } "=" ] expr
```

The parser also enforces:

- `shared` starts only a definition: a shared macro (`a macro cannot be shared;
  share the blocks it builds instead`) and a shared amendment (`an amendment keeps
  whether the definition is shared`) are errors.
- A label needs alternatives: `x = a: 1` is an error.
- No two parameters of a macro have the same name, and no two alternatives of one
  `|` the same label.

What a construct means, and the errors the compiler finds after parsing, are on the
other pages: [Operators](operators.md), [Skip connections](skip-connections.md),
[Definitions](definitions.md), [Settings and values](values.md),
[Modules](modules.md) and [Sweeps](sweeps.md).
