(lang-studio)=
# bux studio

`bux studio FILE` opens a page in your browser that shows a `.bux` file three ways at
once: its source, its model drawn as an architecture diagram in the style of
Sebastian Raschka's figures, and, for a run, its training and its sweep. Edit any of
them, and the change lands in the file's text, where the file sets what you changed.
Comments, macros, patterns and sweeps stay as you wrote them.

```console
$ bux studio examples/experiments/attention_ablation.bux
bux studio: /home/me/bux/examples/experiments/attention_ablation.bux at http://127.0.0.1:8765/?t=8Jc...
```

```{figure} ../_static/studio/studio.png
:alt: bux studio on attention_ablation.bux: the source on the left, the model in the
:     middle with its attention selected, and that layer's settings on the right.

`examples/experiments/attention_ablation.bux` in the studio, its `mha` run picked.
The source highlights where the settings of the selected attention are written: the
two amendments of `attn`. The inspector shows the alternatives of `num_kv_heads` in
the sweep.
```

The studio needs no npm, no internet connection and no extra dependencies. Python's
standard library serves the page, and the page is plain ES modules, SVG and CSS that
ship with bux.

## Starting it

```sh
bux studio FILE [--port 8765] [--host 127.0.0.1] [--no-browser]
```

- The page opens in your browser. Its URL carries the session's token, so keep it to
  yourself. With `--no-browser`, the studio only prints the URL.
- If the port is taken, the studio tries the next few.
- If `FILE` does not exist, the page starts from a small template of a model and its
  run, and the first save creates the file.
- Over SSH (`$SSH_CONNECTION` is set), the studio opens no browser. It prints the
  `ssh -L PORT:127.0.0.1:PORT` command that forwards the port to your machine
  instead. Open the URL there.
- The parameter count and the learning-rate chart need JAX. The studio runs JAX on
  the CPU unless `JAX_PLATFORMS` says otherwise, so it never takes a GPU from a run.

Every option is in the [command-line reference](../reference/cli.md#bux-studio).

## What it shows

The page draws what the compiler says the file means: the IR of the run you picked,
never a separate reading of the source. As you type, it checks the source with
{py:func}`bux.lang.compiler.check`. While the source has errors, the page underlines
them, as [`bux check`](tooling.md#bux-check) reports them, and dims the drawing.

- **Source**: the file, with completion and hover as
  [bux-lsp](tooling.md#editor-support) gives them. Moving the cursor selects the node
  written there, and selecting a node shows where it is written.
- **Model**: the diagram. Tokens enter at the bottom and logits leave at the top.
  `[f]` is a ⊕ over `f` with its skip wire beside it, `f + g` and `f * g` are columns
  merged in ⊕ and ⊗, and a repeat is a container marked `×32`. A block, the value of
  a definition, is a container captioned with the definition's name. A layer is a box
  with a title ("Grouped-query attention"), the numbers that matter as chips
  ("32 heads", "8 KV heads", "RoPE 500k"), and the name of its definition.
  Feed-forwards, mixtures of experts, attention (`gqa`, `mla`), the delta-rule mixers
  (`deltanet`, `gdn`, `kda`, `gdn2`) and Mamba-2 open into a diagram of what is
  inside them. A card gives the width, vocabulary, layers and parameters of the
  model. The diagram exports as SVG or PNG.
- **Training**: the budget of the run in steps and tokens; the learning rate over the
  steps, with its warmup, stable and decay parts marked; the stretches of its
  curriculum and batch ramp; its data mixture, by the names the file gives its
  sources; its evals and checkpoints; and every top-level setting.
- **Sweep**: the axes of `a | b`, each with its alternatives as written, the runs
  they make, and the dimensions of a search. Picking a run redraws the model and the
  training as that run has them.

```{figure} ../_static/studio/qwen3_next.svg
:alt: Qwen3-Next drawn by bux studio: twelve repeats of three Gated DeltaNet blocks
:     and one attention block, each with a mixture of experts, then the head and a
:     multi-token prediction layer fed from the tap @h.
:width: 520px

`examples/architectures/qwen3_next.bux`, exported from the studio as SVG. `"DDDA"^12`
is a container `×12` with `D` folded `×3`. The multi-token prediction layer reads the
stream saved by `@h`.
```

## Editing

Every panel edits the file:

- **Source**: type. Completion opens after `(`, `,`, `.`, `=`, `@`, `$` and `"`, or
  on Ctrl-Space.
- **Diagram**: hover a layer or a block for its toolbar. Insert a layer above or
  below it from the palette of every layer and activation, wrap it in a residual
  branch `[f]` or unwrap one, or delete it (Delete or Backspace deletes the selected
  node too). Double-click a repeat's `×n` to change the count.
- **Inspector**: the settings of what you selected, field by field. A field that
  cannot be set where it is written, such as one set by a macro's parameter, says
  why.
- **Training**: edit the steps, batch and sequence length in the budget. Drag the
  end of the warmup or the start of the decay on the learning-rate chart, or pick
  another schedule type.
- **Sweep**: click an alternative to pick a run that takes it. Double-click one to
  edit it, click its `×` to remove it, or `+` to add one.

Each edit is a splice of the file's text. It lands where the file writes what you
changed, and the rest of the file stays as it was. Each new source goes through the
compiler again before the page draws it. A few edits have rules:

- A setting written in a module, such as the `attn` of `llama3_8b`, is changed by an
  amendment in your file: `attn.num_heads = 12`.
- One letter of a pattern string, `"DDDA"`, cannot be changed alone. Change the
  definition it names, or the whole pattern.
- In `a - b`, the diagram shows `b` negated, but the file has no separate text for
  that negation: the `-` belongs to the subtraction. To wrap, repeat or extend the
  negation itself, write `a + -b`.
- Editing an axis of the sweep rewrites only the alternatives you changed. The
  others keep their lines, alignment and comments.

### Saving and undo

Saving is explicit: Cmd-S or Ctrl-S. The studio writes the file in one step and keeps
its line endings, `\r\n` too. Undo and redo (Cmd-Z, Shift-Cmd-Z) step through your
edits: each visual edit is one step, and each burst of typing another.

The page watches the file on disk:

- If it changes there and you have no unsaved edits, the page reloads it. With
  unsaved edits, it asks whether to reload or keep yours.
- A save refuses to overwrite a file that changed on disk since the page loaded it,
  as another editor's save would. The page then offers to reload from disk or to
  overwrite.
- If the file is deleted, the page keeps its text, unsaved, and a save writes it
  again.

| Shortcut (Ctrl on Linux and Windows) | Does |
|---|---|
| Cmd-S | save |
| Cmd-Z, Shift-Cmd-Z | undo, redo |
| Cmd-1, Cmd-2, Cmd-3 | show the model, training or sweep |
| Cmd-Backslash | show or hide the source |

## Safety

The studio answers only on `127.0.0.1` (or `--host`). Every API call must carry the
session's token, which only the page's URL holds, so other pages open in your
browser cannot read or write the file. The server refuses a request whose `Host`
header names another machine, which blocks DNS rebinding.

## How a layer is drawn

The title, category, chips and inside diagram of each layer come from registries
keyed by the layer's type, in {py:mod}`bux.studio.palette` and
{py:mod}`bux.studio.internals`. A new layer is drawn from defaults:

- its title is the first clause of its config's docstring;
- its category comes from its config's base class: an `AttentionConfig` that is not
  softmax attention is `recurrent`, and an `MLPConfig` is `ffn`;
- it has no chips and no inside diagram.

Register better ones in a few lines:

```python
from bux.studio import internals, palette
from bux.studio.internals import op, seq

palette.names["mixer"] = "Token mixer"  # the title, and the palette's name
palette.categories["mixer"] = "recurrent"  # its colors


@palette.chips.register("mixer")
def _(c: MixerConfig, ctx: palette.Context) -> list[str]:
    """Its heads."""
    return [palette.many(c.num_heads, "head")]


@palette.templates.register("mixer")
def _(tag: str, ctx: palette.Context) -> str:
    """Heads of 128 dims."""
    return f"mixer(num_heads={max(1, ctx.d_model // 128)})"


@internals.internals.register("mixer")
def _(c: MixerConfig, ctx: palette.Context) -> internals.Part:
    """Its projection, then its scan."""
    return seq(op("Linear ↑", [f"{c.num_heads} heads"]), op("Scan", [], "recurrent"))
```

`palette.titles` takes the place of the name where the title depends on the
settings: a `gqa` with one key head is "Multi-query attention". In an inside diagram,
`internals.prod` of columns is a gate (⊗) and `internals.add` is a sum (⊕). The
[registries' reference](../reference/language/registries.md) lists every entry.

## How it works

The page reads the file through pure functions of its text, which need no JAX.
{py:func}`bux.studio.view.view` returns what the page draws of a file, and
{py:func}`bux.studio.training.run_view` returns the training of a run:

```python
from bux.studio.view import view

path = "examples/experiments/attention_ablation.bux"
v = view(open(path).read(), {"gqa|mha|mqa": 2}, path=path)  # the mqa runs
v["sweep"]["axes"][0]["options"]  # ['mha: 12', 'gqa: 4', 'mqa: 1']
v["run"]["name"]  # 'attention_ablation[mqa, optimizer.lr=1e-3]'
```

A node's id is its path from the root, `n.1.0.2`, where each step is a child's index
in the order {py:func}`bux.lang.ir.children` gives. Each node also says where it is
written in the file. Both come from {py:func}`bux.lang.compiler.trace`. `picks` names
each axis by an id that survives edits elsewhere in the file:
{py:func}`bux.studio.picks.axis_id`.

The server's endpoints all speak JSON, and each request carries the token in
`X-Bux-Token`:

| Endpoint | What it answers |
|---|---|
| `GET /api/session` | the file's path, name, text and modification time, and the layer palette |
| `GET /api/file` | the file's modification time, and its text with `?source=1` |
| `POST /api/view` | the view of a source and a run of its sweep |
| `POST /api/inspect` | the inspector's form for a node or a setting |
| `POST /api/edit` | the source with a list of edits applied |
| `POST /api/schedule` | the learning rate of the run at up to 512 steps (JAX) |
| `POST /api/plan` | the parameters of the model and its FLOPs per token, as `bux plan` counts them (JAX) |
| `POST /api/complete`, `/api/hover` | completion and hover at an offset |
| `POST /api/save` | writes the file; `409` with what is on disk if it changed there |

Offsets, in requests and replies, count UTF-16 code units, as the page's JavaScript
does. Past an emoji, they are one more than Python's string indices.

A view of the file as you type takes milliseconds. These are medians over 20 views on
an Apple M3 Max, measured 2026-10-26 with Python 3.14:

| File | Layers | View |
|---|---|---|
| `architectures/llama3_8b.bux` | 131 | 3 ms |
| `architectures/qwen3_next.bux` | 202 | 5 ms |
| `architectures/deepseek_v3.bux` | 247 | 6 ms |
| `experiments/attention_ablation.bux`, 6 runs | 51 | 12 ms |
| `experiments/schedule_sweep.bux`, 6 runs | 51 | 19 ms |

The first view of a sweep also compiles all of its runs, and the studio caches them
by source. The first `/api/plan` or `/api/schedule` call imports JAX. After that,
each takes about 0.3 s per run on the CPU, and the studio caches the result by the
run's id.
