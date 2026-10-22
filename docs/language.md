# The `.bux` architecture language

A `.bux` file describes a language model as an **algebra over the residual stream**.
Layers are functions of the stream. The language composes them, runs them in parallel
and sums or multiplies the results, and draws skip connections. The result is a typed IR
(`bux.lang.ir`) that a backend turns into code.

```
d_model    = 4096
vocab_size = 128256

norm  = rmsnorm(eps=1e-5)
attn  = gqa(num_heads=32, num_kv_heads=8, rope=rope(base=500000))
ffn   = gated_mlp(intermediate_size=14336, activation=swish)

block = [norm attn] [norm ffn]
model = embed block^32 norm head
```

That is all of Llama 3 8B. The core idea is that **`[f]` is a skip arc**: `x + f(x)`.
The bracket reads as the bypass wire drawn over `f`, and `bux render` draws it that way:

```
block
──┬────────────⊕──┬───────────⊕──
  └─norm──attn─┘  └─norm──ffn─┘
```

## Operators

Loosest first:

| syntax              | meaning                                                     |
|---------------------|-------------------------------------------------------------|
| `f \| g`, `a: f \| b: g` | alternatives: a run for each, a sweep ([training](training.md)) |
| `f + g`, `f - g`    | parallel branches on the same input, summed: `f(x) + g(x)`  |
| `f * g`, `f * 0.5`  | parallel branches multiplied (gating), or scaled by a number |
| `f g`, `f -> g`     | composition, left to right: `g(f(x))`                        |
| `-f`                | negation                                                    |
| `f^n`               | `n` fresh copies of `f` in a row (composition power)        |
| `[f]`               | residual branch, `x + f(x)`                                 |
| `_`                 | identity, so `[f]` behaves like `(_ + f)`                   |
| `(f)`               | grouping                                                    |
| `@name`             | tap: push the stream onto the stack `name`                  |
| `$name`, `$name!`   | the top of the tap `name` (ignoring the input); `!` pops it |
| `"MMMA"`            | pattern: the sequence `M M M A` of one-letter definitions   |

Composition binds tighter than `*` and `+`, as concatenation does in regular
expressions. So `norm attn + ffn` means `(norm attn) + ffn`, and SwiGLU is written as
it reads:

```
swiglu = (linear(features=h) swish * linear(features=h)) linear(features=d_model)
```

`f * 3` is an error, because it is unclear whether you meant `f^3` or `f * 3.0`.

## Common shapes

```
[norm attn] [norm ffn]                       # pre-norm (Llama, GPT-2)
[attn] norm [ffn] norm                       # post-norm (original Transformer)
[norm attn norm] [norm ffn norm]             # sandwich norm (Gemma 2/3)
[norm (attn + ffn)]                          # parallel block (GPT-J, PaLM)
[norm (attn norm scale + ssm norm scale)]    # hybrid heads (Hymba)
[norm attn scale(init=0)]                    # ReZero; scale(per_channel=true) for LayerScale
(_ * 2.0 + attn) norm                        # DeepNorm-style weighted skip
[(attn unit - _) step] unit                  # nGPT: h <- norm(h + a * (norm(f(h)) - h))
gqa(gated=true)                              # gated attention, before o_proj (Qiu et al. G1)
gqa(...) * (linear(features=d_model) sigmoid) # gated after o_proj instead (their G5)
```

## Skip connections: local, long-range, internal

- **Local**: `[f]` adds the stream around `f`. Brackets nest.
- **Long-range**: drop a pin with `@name`, then add it back anywhere with `[$name]`.
  Since `$name` ignores its input, `[$x0 scale]` is `x + w * x0`.
  - x0 mixing in every block: `blk = (scale + $x0 scale) [norm attn] [norm ffn]`
  - **U-net**: taps are stacks, and `$name!` pops. The mirror pairing of a U-net
    falls out of last-in-first-out order with no index arithmetic:
    ```
    model = embed @x0 (block @skip)^6 ([$skip! scale] block)^6 norm head
    ```
- **Internal tensors** go through layer *ports*: `attn(v=@v1)` exports the attention
  values, and `attn(v=$v1)` mixes them back in later (value residual / ResFormer).
  `attn(kv=@kv)` and `attn(kv=$kv)` share K/V across layers (YOCO, CLA), and
  `attn(index=@k)` / `attn(index=$k)` share a sparse attention indexer's top-k
  (GLM-5's IndexCache, DeepSeek-V4.1). A config class lists its ports in `ports`.

The compiler simulates every stack through the unrolled model. Reading a tap before it
is written, popping an empty one, or reading a port tap as hidden states are compile
errors.

Tap names are global, so a block can leave a value behind that hides one its caller
wrote. A value that a block (a definition, a macro call, or one pass of `^n`) both
writes and reads is that block's own. If it is still on the stack when the block ends,
and something outside reads the tap while an older value lies under it, that is an
error:

```
mix(f) = @h [f] [$h scale]                # reads h but leaves it on the stack
model = embed @h mix(attn) [$h! scale]    # error: this would pop mix's h, not the first
```

Pop it inside the block on its last read (`[$h! scale]`), or use another name. Exports
(a block that writes a tap and leaves the reading to others) and chains (each block
reads the previous block's value and writes its own) are unaffected.

## Definitions, macros, patterns, sharing

```
norm = rmsnorm                         # a layer: a template, fresh parameters per use
local  = attn(sliding_window_size=1024)  # call a layer to override settings
global = attn(rope.base=1000000)         # dotted keys reach into nested settings
mlp  = mlp(intermediate_size=3072, activation=gelu)  # self-reference means the builtin

layer(mixer, ffn=experts) = [norm mixer] [norm ffn]  # a macro, with a default
attn_kv(n, kv=n // 4) = gqa(num_heads=n, num_kv_heads=kv)  # defaults see earlier parameters
D = layer(gdn)
A = layer(attn)
model = embed "DDDA"^12 norm head      # Qwen3-Next: 3 Gated DeltaNet : 1 attention

shared sa = [norm attn] [norm ffn]     # one module, reused (Zamba, ALBERT)
model = embed (m^6 sa)^9 norm head
```

- Definitions can appear in any order. Cycles are errors.
- Macros take positional or keyword arguments and are values themselves, so a macro can
  take another: `twice(f, x) = f(x) f(x)`. A default can use the parameters before it.
- Patterns use one-letter top-level definitions. Spaces inside a pattern are ignored,
  and patterns repeat like anything else: `"mMmM aMmM"^4` is Jamba.
- `shared` makes a definition a single module, so every use, including inside `^n`, has
  the same parameters. Each use of a plain definition gets fresh parameters.
- `head(tied=true)` reuses the embedding matrix.

A file can build on others by name, `use llama3_8b`, and change one setting of a
definition, `attn.num_kv_heads = 1`, which means `attn = attn(num_kv_heads=1)`. A
definition then replaces the one it shadows everywhere, and inside its own definition
a name means the definition it replaces (the builtin, when there is none). The
[training guide](training.md) describes both, with runs and sweeps.

## Several outputs: multi-token prediction

Every `head` is an output. A head ends the stream, so nothing can follow it but a tap
read, and that is how a model goes on to a second output. `nextn` mixes the stream with
the next token's embedding, `W [norm(e); norm(h)]`, where `e` is what `embed` makes of
the token (scaled and normalized, without positions), so a head after it predicts the
token after next. This is DeepSeek-V3's multi-token prediction (MTP) module, with the
main head shared:

```
shared out = head
mtp = $h nextn layer(experts) norm out
model = embed layer(dense)^3 layer(experts)^58 norm @h out mtp
```

The k-th `nextn` to run embeds the token k positions ahead. A chain of modules, each
reading the previous one's hidden state, is a repeat; with `shared`, one module runs at
every depth and still looks one token further each time (GLM-5, Nemotron 3); and with
`$h` instead of `$h!`, every depth reads the main model's state (MiMo-V2):

```
mtp = $h! nextn layer(experts) norm @h out
model = embed block^40 norm @h out mtp^3       # 4 outputs: 1, 2, 3 and 4 tokens ahead
```

A backend returns the heads' logits in execution order; head j predicts the token
`1 + k` positions ahead, where k counts the `nextn` layers run before it.

## Settings

A layer call takes keyword arguments, which map one-to-one to the pydantic fields in
`bux/config.py`. Every config's `type` tag is also its name in the language:
`gqa`, `mla`, `csa`, `mamba`, `mamba2`, `gla`, `deltanet`, `gdn`, `kda`, `lightning`,
`rwkv7`, `conv`, `short_conv`, `mlp`, `gated_mlp`, `moe`, `layernorm`, `rmsnorm`, `dyt`,
`linear`, `act`, `scale`, `dropout`, `embed`, `head`, `nextn`, `engram`, `merge`.
Settings that are not layers: `rope`, `pi`, `yarn`, `llama3`, `relbias`, `indexer`, and
the residual algebras `hc`, `mhc`, `gr`, `attnres`, `vpgate`. Activation names
(`swish`, `gelu`, `relu2`, `sigmoid`, `situ`, ...) are also leaves:
`linear(features=h) swish`.

Values are numbers (`1e-5`, `128_256`, and counts with a suffix `k`, `M`, `B` or `T`:
`128k`, `1.5B`), strings, `true`/`false`/`none`, lists
(`[1, 2]`, or `[1,]` for one element), configs (`rope(base=1e6)`), or arithmetic over
numeric definitions (`4 * d_model // 3`). A bare name that is not defined is a string,
so `activation=swish` and `router=sigmoid` work without quotes. Where a setting wants
something other than a string, or in arithmetic, an undefined name is reported as
`unknown name`, with a suggestion: `intermediate_size=4 * dmodel` asks whether you meant
`d_model`. Unknown or missing settings are reported against the pydantic schema, with
suggestions.

`name.field` reads a setting back: `num_kv_heads=attn.num_heads // 4`,
`experts.expert` (a layer), `rope.base` (a builtin's default). It reads the value as
written, so a default that depends on the width, like `head_dim`, reads as `none`.

Top-level settings: `d_model` (required), `vocab_size`, and `residual`. Training
settings (`data`, `optimizer`, `schedule`, `steps`, ...) make a file a run: see
[training](training.md).

## Widths

Every leaf knows its input width. `embed` maps token ids to `d_model` (or to the widened
stream, see below), `linear` maps to `features`, `head` maps to logits, and everything
else keeps the width. The compiler checks:

- a residual branch and a repeated block return to the width they started at;
- parallel branches agree on their width;
- `embed` comes first, and nothing comes after `head`;
- every use of a `shared` block has the same widths.

## Changing the residual algebra

`residual = mhc(expansion=4)` keeps the wiring and changes what `[...]` means: the
stream is widened into 4 streams, and every residual branch reads, writes, and mixes them
through learned (for `mhc`, doubly stochastic) connections. The IR keeps `Residual`
separate from `_ + f` for this reason. `[f]` is the residual stream's own connection,
so backends can replace it, while `_ + f` is plain arithmetic.

`mhc(single_pass=true)` is DeepSeek-V4.1's variant: each branch reads with the
coefficients the branch before it computed (the first reads stream 0), and the read
before the head has no parameters. Every algebra with a norm of its own takes
`norm_eps`.

The other algebras:

| setting                    | `[f]` becomes                                                           |
|----------------------------|-------------------------------------------------------------------------|
| `hc(expansion=4)`          | hyper-connections, as `mhc` but without the doubly stochastic mixing     |
| `gr(rank=320)`             | a gated read of 4 normalized streams and a weighted write to each; branches need no pre-norm (Qwen3.8-Flash-Next) |
| `attnres(block_size=24)`   | `f` reads a learned softmax over depth of earlier block outputs instead of their sum (Kimi K3) |
| `vpgate(temperature=0.3)`  | `a x + b f(x)` with `a^2 + b^2 = 1` per channel (Muse Spark)             |

### Outside the brackets

`hc`, `mhc`, `gr` and `attnres` widen the stream: between the branches it is several
copies (for `attnres`, the outputs so far), and only a few things act on it as a whole:

- `embed` widens it, and so does `nextn` when its input is one stream;
- residual branches `[f]` read it, run `f` on one stream and write back; brackets
  inside `f` are plain skips;
- taps `@h` and `$h` save and restore it as it is;
- `engram` adds its memory to every copy, with a key of its own, and `nextn` mixes the
  next token's embedding into every copy (Qwen3.8-Flash-Next). Under `attnres`, whose
  state is the outputs so far rather than copies, both need one stream.

Any other layer outside a branch, such as the final norm or the head, needs one stream,
so the compiler puts a `merge` before it: the read that a branch does, without the
write. `bux fmt` prints it. A residual branch after that is an error, since the copies
are gone, and so is reading a tap that holds the widened stream inside a branch:

```
residual = gr(expansion=4, rank=320)
model = embed D ple "DDA" "DDDA"^11 @h out mtp      # = ... @h merge out $h nextn ...
```

`engram` is written bare, because it adds to the stream itself; `[engram]` is an error.

## The IR and backends

`bux.lang.compile(source)` returns an `Architecture` with `d_model`, `vocab_size`,
`residual` and a `graph`: a tree of 11 node kinds (`Layer`, `Seq`, `Residual`, `Sum`,
`Prod`, `Neg`, `Const`, `Identity`, `Repeat`, `Tap`, `Ref`), documented in
`bux/lang/ir.py`. It is plain pydantic, so it serializes to JSON and goes into a
checkpoint as-is. A backend is one function per node kind plus a registry of leaves.
`bux/architecture/` is a working Flax NNX one. `Repeat` is kept rather than unrolled,
so a backend can `nnx.scan` over identical blocks.

`bux.lang.printer.to_source(arch)` prints any IR back to readable source, including IR
that was generated or mutated programmatically, e.g. by an architecture search.
`compile(to_source(a))` equals `a` up to labels.

### Adding a layer

A new layer type takes two pieces. First, its config, a class in `bux/config.py` added
to the `LayerConfig` union. The union is what the compiler knows layer names from, and
what the IR, checkpoints, run ids, the printer and the editor read configs by:

```python
class DiffAttentionConfig(Config):
    type: Literal["diffattn"] = "diffattn"
    num_heads: int
```

Second, a builder, registered for the config's `type` in any module imported before
the model is built. It takes the config, the width the layer reads, its ports, and a
`Context` with the architecture and the `rngs`. The module it returns is called as
`module(x, taps)`; `segments(taps)`, `positions(taps)`, `training(taps)` and
`report(taps, name, weight, value)` are what a layer reads and reports:

```python
from bux.architecture import Context, register


@register("diffattn")
def _(c: DiffAttentionConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context):
    return DiffAttention(c, width, rngs=ctx.rngs)
```

Registering a builder for an existing type replaces bux's own. A config can only come
from `bux/config.py`: the union is a pydantic discriminated union that `ir.Layer` and
the compiler fix as they load.

## Command line

```
bux check  examples/architectures/qwen3_next.bux  # compile, count layers by type
bux render examples/architectures/qwen3_next.bux  # circuit diagram; --expand to inline blocks
bux fmt    examples/architectures/qwen3_next.bux  # canonical source
bux json   examples/architectures/qwen3_next.bux  # the IR
bux plan   examples/experiments/smoke.bux         # the runs of a sweep
bux train  examples/experiments/smoke.bux         # train them
```

`bux fmt` prints the IR, not your file: it inlines macros and numeric definitions and
drops comments. Statements after the file redefine the file's:
`bux check model.bux d_model=1024 'attn.num_kv_heads = 1'`.

Errors point at the source. Every command reports all the errors it finds: a syntax
error loses only its own statement, and each definition is checked on its own.

```
error: gqa has no setting 'num_kv_head'
 --> model.bux:3:33
  |
3 | blk = [rmsnorm gqa(num_heads=2, num_kv_head=2)]
  |                                 ^^^^^^^^^^^^^
  = hint: did you mean 'num_kv_heads'?
```

## Editor support

`bux-lsp` is a language server (LSP over stdio). Install it with the `lsp` extra:
`uv sync --extra lsp`, which puts it at `.venv/bin/bux-lsp`. It provides:

- **diagnostics**: every error `bux check` finds, as you type;
- **completion**: the settings of the layer being called, including through definitions
  (`local = attn(` offers `gqa`'s settings) and dotted keys (`rope.`); the values a
  setting's type allows (`activation=` offers the activations, `qk_norm=` the norms
  and your norm definitions); macro parameters; fields after `name.`, with their current
  values; tap names after `@` and `$`; one-letter definitions inside patterns;
- **hover**: a layer's documentation and settings, taken from `bux/config.py`; what a
  definition is written as and what it evaluates to, and which module defines it;
  whether a bare name is a reference or a string.

Modules resolve from the file's own folder, and errors in them show at the `use`.

PyCharm (or any JetBrains IDE) needs the LSP4IJ plugin. `src/bux/lang/lsp/pycharm`
holds a TextMate grammar for highlighting and an LSP4IJ template that runs the
project's `.venv/bin/bux-lsp`. After `uv sync`:

1. Settings | Plugins | Marketplace: install **LSP4IJ**.
2. Settings | Editor | TextMate Bundles: `+`, choose `src/bux/lang/lsp/pycharm`.
3. Settings | Languages & Frameworks | Language Servers: `+`, Template: *Import from
   custom template...*, choose the same directory, OK.

In Neovim 0.11:

```lua
vim.filetype.add({ extension = { bux = "bux" } })
vim.lsp.config("bux", { cmd = { "bux-lsp" }, filetypes = { "bux" } })
vim.lsp.enable("bux")
```

The features themselves are plain functions in `bux.lang.lsp.features`, so they can be
tested without an editor.

## Lexical rules

A statement is `name = expr`, `name(params) = expr`, `shared name = expr`,
`name.key = expr` (an amendment) or `use module`. It ends at
a newline unless the next line is indented or a bracket is still open; `;` also ends a
statement. `#` starts a comment. A call needs its `(` right after the name, so
`norm (attn + ffn)` composes two things, while `norm(eps=1e-5)` configures one. In the
same way `attn.num_heads` needs its `.` right after the name.

## Grammar

```
statement := ["shared"] NAME ["(" params ")"] "=" expr
           | NAME "." NAME { "." NAME } "=" expr       # an amendment
           | "use" (NAME | STRING)
params    := NAME ["=" expr] { "," NAME ["=" expr] } [","]
expr      := option { "|" option }                     # alternatives
option    := [NAME ":"] sum                            # all labeled, or none
sum       := prod { ("+" | "-") prod }
prod      := chain { ("*" | "/" | "//") chain }
chain     := unary { ["->" | "→"] unary }
unary     := "-" unary | power
power     := atom [ "^" unary ]
atom      := NAME [call] { "." NAME } | NUMBER | STRING | "_" | "true" | "false" | "none"
           | "@" NAME | "$" NAME ["!"] | "(" expr ")" | "[" expr "]"
           | "[" [expr { "," expr } [","]] "]"          # a list: has a comma, or is empty
call      := "(" [arg { "," arg } [","]] ")"
arg       := [NAME { "." NAME } "="] expr
```

## Design notes

Alternatives considered:

- **Arrows with labels everywhere**, e.g. `@a ln1 attn +a @b ln2 mlp +b`. This is fully general,
  but every residual needs a name. Brackets cover the common case, and taps remain for the rest.
- **Drawn 2D ASCII graphs** as the input. This is the most visual option, but painful to edit
  and diff. Instead the drawing is the *output* (`bux render`), and the 1D syntax is chosen
  so that it already looks like the drawing.
- **Edge lists** such as NAS-Bench-201 cell strings, or HF-style `layer_types` arrays.
  They are precise but unreadable, and per-layer lists say nothing about wiring.

Possible extensions: a `cat` merge (Zamba concatenates x0 into the shared block);
recursive macros for nested structures; and parsing the rendered diagrams back, so a
drawing can be edited directly.

The example files at the top of `examples/architectures/` transcribe public configs
from memory. Ones marked "-style" or "approximate" simplify details, so verify the
numbers before relying on them. The folders, one per model family (`cohere/`, `deepseek/`, `glm/`,
`inkling/`, `kimi/`, `laguna/`, `mellum/`, `mimo/`, `muse/`, `nemotron/`, `qwen/`), transcribe
recent releases from their published `config.json` and modeling code. Their comments
name what is left out (vision encoders, speculative drafters) or approximated.
