(lang-tooling)=
# Tools and editor support

Four commands show what a `.bux` file means without training it: `bux check`,
`bux render`, `bux fmt` and `bux json`. None of them imports JAX, so each answers in
well under a second. The language server `bux-lsp` brings the same checks, with
completion and hover, into your editor. This page covers the commands, the errors
they report, and how to set up your editor.

Training commands are in [Running](../training/running.md), and every option is in
the [command-line reference](../reference/cli.md). To see a file drawn as a diagram
and edit it in a browser, use [bux studio](studio.md).

## Commands

```sh
bux check  examples/architectures/qwen3_next.bux  # compile, count layers by type
bux render examples/architectures/qwen3_next.bux  # circuit diagram; --expand to inline blocks
bux fmt    examples/architectures/qwen3_next.bux  # canonical source
bux json   examples/architectures/qwen3_next.bux  # the IR
```

Every command takes [statements after the file](modules.md#the-command-line), which
redefine the file's definitions:
`bux check model.bux d_model=1024 'attn.num_kv_heads = 1'`. `--entry NAME` builds a
definition other than `model`, to look at one block:
`bux render model.bux --entry block`.

### `bux check`

Compiles the file and prints a summary: the width, the vocabulary and the layers by
type. For a run it adds the budget and settings, and for a sweep the axes.

```console
$ bux check examples/architectures/qwen3_next.bux
d_model=2048 vocab_size=151936 layers=202
rmsnorm×100  moe×49  gdn×36  gqa×13  head×2  embed×1  nextn×1
```

```console
$ bux check examples/experiments/smoke.bux 'optimizer.lr = 1e-3 | 3e-3'
d_model=64 vocab_size=512 layers=11
rmsnorm×5  gqa×2  gated_mlp×2  embed×1  head×1
50 steps × 8 × 64 tokens = 25.6k tokens
optimizer=adamw(lr=0.001) schedule=wsd(warmup=10)
data=synthetic()
sweep: 2 runs over optimizer.lr (1e-3 | 3e-3)
```

A file without a model is a module, and `bux check` lists what it defines
([Runs and modules](modules.md#runs-and-modules)). When there are errors, it prints
all of them and exits with status 1.

### `bux render`

Draws the model as a circuit of the residual stream, with each labeled block it uses
drawn underneath:

```console
$ bux render examples/architectures/qwen3_next.bux
model
         ╭─ ×12 ──────────────╮
──embed──┤─┤D├──┤D├──┤D├──┤A├─├──norm──@h──out──┤mtp├──
         ╰────────────────────╯

D
──┬─────────────⊕──┬───────────────⊕──
  └─norm──delta─┘  └─norm──experts─┘

A
──┬────────────⊕──┬───────────────⊕──
  └─norm──attn─┘  └─norm──experts─┘

mtp
──$h──nextn──┤A├──norm──out──
```

| Mark | Means |
|------|-------|
| `──` | the stream, left to right |
| `┬ ... ⊕` | a fork into branches that are summed: a residual branch, or `f + g` |
| `┬ ... ⊗` | a fork into branches that are multiplied: `f * g` |
| `┤name├` | a labeled block (a definition or a macro call), drawn on its own below |
| `name×n`, a box `×n` | a repeat: of one block, or of a group in a frame |
| `×0.5` | scaling by a number |
| `−` | negation |
| `@h`, `$h`, `$h!` | taps; a branch that starts from a tap is not joined to the fork |

`--expand` draws every block inline instead, in one diagram:

```console
$ bux render examples/architectures/llama3_8b.bux --expand
model
         ╭─ ×32 ─────────────────────────╮
──embed──┤─┬────────────⊕──┬───────────⊕─├──norm──head──
         │ └─norm──attn─┘  └─norm──ffn─┘ │
         ╰───────────────────────────────╯
```

### `bux fmt`

Prints the file in canonical form. It prints the compiled IR, not your text:

- macros are inlined, and numeric definitions replaced by their values;
- patterns are spelled out, `"DDDA"^12` as `(D D D A)^12`;
- settings at their defaults are left out, and comments dropped;
- a definition that shadows a builtin is renamed, `mlp` to `mlp_2`, so that
  every name keeps one meaning;
- a [`merge`](residual-algebras.md#merge) the compiler inserted is written out.

```console
$ bux fmt examples/architectures/qwen3_next.bux
d_model = 2048
vocab_size = 151936

norm = rmsnorm(zero_centered=true)
delta = gdn(num_heads=16, head_dim=128, num_v_heads=32)
experts = moe(num_experts=512, top_k=10, expert=gated_mlp(intermediate_size=512, activation=swish), num_shared_experts=1, shared_expert_gate=true, aux_loss_coef=0.001)
attn = gqa(num_heads=16, num_kv_heads=2, head_dim=256, rope=rope(base=10000000.0, fraction=0.25), gated=true, qk_norm=norm)
shared out = head

D = [norm delta] [norm experts]
A = [norm attn] [norm experts]
mtp = $h nextn(norm=norm) A norm out

model = embed (D D D A)^12 norm @h out mtp
```

For a run, `bux fmt` prints the whole run as one file. A header names the run and its
id, and there are no modules and no alternatives: every training setting is resolved,
and those at their defaults are left out. For a sweep it prints every run, one after
another.

Printing is exact. Compiling the printed source gives the same IR up to labels, and a
printed run compiles back to a run with the same id. The printer is
{py:func}`bux.lang.printer.to_source`. It prints any IR, including IR that a program
generated or mutated, such as an architecture search.

### `bux json`

Prints the IR as JSON: the `Architecture` of a model, or a list of every run of a
file that trains. The IR is plain pydantic, so it serializes as is and goes into a
checkpoint unchanged. See [The IR and backends](../architectures/ir-and-backends.md).

```console
$ bux json examples/architectures/llama3_8b.bux | head -5
{
  "d_model": 4096,
  "vocab_size": 128256,
  "residual": null,
  "init": null,
```

(lang-errors)=
## Errors

Errors point at the source:

```text
error: gqa has no setting 'num_kv_head'
 --> model.bux:3:33
  |
3 | blk = [rmsnorm gqa(num_heads=2, num_kv_head=2)]
  |                                 ^^^^^^^^^^^^^
  = hint: did you mean 'num_kv_heads'?
```

Each error has:

- the message, after `error:`;
- the place, `file:line:column`, in the file the error is in. An error in a module
  points into the module, and one in a command-line statement at `<command line>`;
- the line, with carets under the span at fault;
- often a hint: a suggestion for a misspelled name, or how to write what was meant.

An error about the file as a whole has no place, only a hint:

```text
error: 'd_model' is not defined
hint: d_model = 4096
```

**Every command reports all the errors it finds**, not just the first:

- a syntax error loses only its own statement. A statement starts at each line that
  begins in the first column, so an unclosed bracket costs one definition;
- each definition is checked on its own;
- an error is not repeated at every use of a broken definition;
- every alternative of a sweep is checked ([Checking a sweep](sweeps.md#checking-a-sweep)).

In Python, {py:func}`bux.lang.compiler.check` returns them all in a
{py:class}`bux.lang.compiler.Report`, and {py:func}`bux.lang.compiler.compile`
raises the first as a {py:class}`bux.lang.syntax.BuxError`.

(lang-editor)=
## Editor support

`bux-lsp` is a language server that speaks the Language Server Protocol over stdio.
It needs the `lsp` extra. In a checkout of bux:

```sh
uv sync --extra lsp        # puts bux-lsp at .venv/bin/bux-lsp
```

It provides:

- **diagnostics**: every error `bux check` finds, as you type. Errors in a module
  show at the `use` that brought it in, prefixed with the module's file and line.
- **completion**:
  - the settings of the layer being called, including through definitions
    (`local = attn(` offers `gqa`'s settings) and dotted keys (`rope.`);
  - the values a setting's type allows (`activation=` offers the activations,
    `qk_norm=` the norms and your norm definitions, `router=` the router's names);
  - macro parameters;
  - fields after `name.`, with their current values;
  - tap names after `@` and `$`;
  - one-letter definitions inside patterns.
- **hover**:
  - a layer's documentation and settings, taken from `bux/config.py`;
  - what a definition is written as, what it evaluates to, and which module
    defines it;
  - whether a bare name is a reference or a string;
  - for a tap, where it is written, read and popped.

Modules resolve as on the command line: from the file's own folder, then the module
path ([The module path](modules.md#the-module-path)). The features are plain
functions in `bux.lang.lsp.features`, so you can test them without an editor.

### PyCharm and other JetBrains IDEs

JetBrains IDEs need the LSP4IJ plugin. `src/bux/lang/lsp/pycharm` holds a TextMate
grammar for highlighting and an LSP4IJ template that runs the project's
`.venv/bin/bux-lsp`. After `uv sync --extra lsp`:

1. Settings | Plugins | Marketplace: install **LSP4IJ**.
2. Settings | Editor | TextMate Bundles: `+`, choose `src/bux/lang/lsp/pycharm`.
3. Settings | Languages & Frameworks | Language Servers: `+`, Template: *Import from
   custom template...*, choose the same directory, OK.

### Neovim

In Neovim 0.11:

```lua
vim.filetype.add({ extension = { bux = "bux" } })
vim.lsp.config("bux", { cmd = { "bux-lsp" }, filetypes = { "bux" } })
vim.lsp.enable("bux")
```

`bux-lsp` must be on Neovim's `PATH`, for example with the project's `.venv`
activated; otherwise give its full path in `cmd`.

### Other editors

Any editor with an LSP client can use `bux-lsp`: register a server that runs the
command `bux-lsp` (no arguments, stdio) for files ending in `.bux`, with the
language id `bux`. Editors that read TextMate grammars can highlight `.bux` files with
`src/bux/lang/lsp/pycharm/bux.tmLanguage.json`.
