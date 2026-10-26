(lang-studio)=
# bux studio

`bux studio FILE` opens a local web page that shows a `.bux` file three ways at once:
its source, its model drawn as an architecture diagram in the style of Sebastian
Raschka's figures, and, for a run, its training and its sweep. You can edit all
three, and every edit is a change to the file's text, made where the file sets what
you changed.

```console
$ bux studio examples/experiments/attention_ablation.bux
bux studio: /home/me/bux/examples/experiments/attention_ablation.bux at http://127.0.0.1:8765/?t=8Jc...
```

```{figure} ../_static/studio/studio.png
:alt: bux studio on attention_ablation.bux: the source on the left, the model in the
:     middle with its attention selected, and that layer's settings on the right.

`examples/experiments/attention_ablation.bux` in the studio, its `mha` run picked.
The source highlights where the selected attention's settings are written, the two
amendments of `attn`, and the inspector shows the sweep's alternatives of
`num_kv_heads`.
```

The studio needs no npm and no internet connection, and it adds no dependencies.
Python's standard library serves the page, and the page is plain ES modules, SVG and
CSS that ship with bux.

## Starting it

```sh
bux studio FILE [--port 8765] [--host 127.0.0.1] [--no-browser]
```

- The page opens in your browser. Its URL carries the session's token; keep it to
  yourself. With `--no-browser`, the studio only prints the URL.
- If the port is taken, the studio tries the next few ports.
- If `FILE` does not exist, the page starts from a small template of a model and its
  run, and the first save creates the file.
- On a cluster's login node (`$SSH_CONNECTION` is set), the studio opens no browser.
  It prints the `ssh -L PORT:127.0.0.1:PORT` command that forwards the port to your
  machine instead. Open the URL there.
- The parameter count and the learning-rate chart need JAX. The studio runs JAX on
  the CPU unless `JAX_PLATFORMS` says otherwise, so it never takes a GPU from a run.

Every option is in the [command-line reference](../reference/cli.md#bux-studio).

## What it shows

The page draws what the compiler says the file means: the IR of the run you picked,
not a model of its own. As you type, it checks the source with
{py:func}`bux.lang.compiler.check`. While the source has errors, it underlines them,
as [`bux check`](tooling.md#bux-check) reports them, and dims the drawing.

- **Source**: the file, with completion and hover as
  [bux-lsp](tooling.md#editor-support) gives them. Moving the cursor selects the node
  written there, and selecting a node shows where it is written.
- **Model**: the diagram. Tokens enter at the bottom and logits leave at the top.
  `[f]` is a ⊕ over `f` with its skip wire beside it, `f + g` and `f * g` are columns
  merged in ⊕ and ⊗, and a repeat is a container marked `×32`. A block, the value
  of a definition, is a container captioned with the definition's name. A layer
  is a box with a title ("Grouped-query attention"), the numbers that matter as
  chips ("32 heads", "8 KV heads", "RoPE 500k"), and the name of its definition.
  Feed-forwards, mixtures of experts, attention, Gated DeltaNet and Mamba-2 open
  into what is inside them. A card gives the model's width, vocabulary, layers and
  parameters. The diagram exports as SVG or PNG, as below.
- **Training**: the run's budget in steps and tokens, the learning rate over the
  steps with its warmup, stable and decay parts marked, the stretches of its
  curriculum and batch ramp, its data mixture by the names the file gives its
  sources, its evals and checkpoints, and every top-level setting.
- **Sweep**: the axes of `a | b`, each with its options as written, the runs they
  make, and a search's dimensions. Picking a run redraws the model and the training
  as that run has them.

```{figure} ../_static/studio/qwen3_next.svg
:alt: Qwen3-Next drawn by bux studio: twelve repeats of three Gated DeltaNet blocks
:     and one attention block, each with a mixture of experts, then the head and a
:     multi-token prediction layer fed from the tap @h.
:width: 520px

`examples/architectures/qwen3_next.bux`, exported from the studio as SVG. `"DDDA"^12`
is a container `×12` with `D` folded `×3`; the multi-token prediction layer reads the
stream saved by `@h`.
```

## Editing

An edit made on the page, in the inspector or in the diagram, is a splice of the
file's text. The edit lands where the file writes what you changed, and the rest of
the file stays as it was: comments, macros, patterns and sweeps. Each new source goes
through the compiler again before the page draws it.

- A setting written in a module, such as `llama3_8b`'s `attn`, is changed by an
  amendment in your file, `attn.num_heads = 12`.
- One letter of a pattern string, `"DDDA"`, cannot be changed alone. Change the
  definition it names, or the whole pattern.
- The negation that `a - b` writes is written as `b` alone, so it cannot be
  wrapped, repeated or followed by a layer without `b`. Write `a + -b` for that.
- Editing an axis of the sweep rewrites only the alternatives you changed. The
  others keep their lines, alignment and comments.
- Saving is explicit (Cmd-S or Ctrl-S). The studio writes the file in one step, and
  refuses to write it if the file changed on disk since the page loaded it, as
  another editor's save would. The page then offers what is on disk. If the file is
  deleted on disk, the page keeps its text, unsaved, and a save writes it again. The
  file keeps its line endings, `\r\n` too.

## Safety

The studio answers only on `127.0.0.1` (or `--host`). Every API call must carry the
session's token, which only the page's URL holds, so other pages open in your
browser cannot read or write the file. The server refuses a request whose `Host`
header names another machine, which blocks DNS rebinding.

## How a layer is drawn

Each layer's title, category, chips and inside diagram come from registries keyed by
the layer's type, in {py:mod}`bux.studio.palette` and {py:mod}`bux.studio.internals`.
A new layer is drawn from defaults:

- its title is the first clause of its config's docstring;
- its category comes from its config's base class: an `AttentionConfig` that is not
  softmax attention is `recurrent`, an `MLPConfig` is `ffn`;
- it has no chips and no inside diagram.

It can register better ones in a few lines:

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

`palette.titles` takes the place of the name where it depends on the settings, as a
`gqa` with one key head is "Multi-query attention". In an inside diagram,
`internals.prod` of columns is a gate (⊗) and `internals.add` is a sum (⊕). The
[registries' reference](../reference/language/registries.md) lists every entry.

## The view model in Python

The page reads the file through pure functions of its text, which need no JAX.
{py:func}`bux.studio.view.view` returns what the page draws of a file, and
{py:func}`bux.studio.training.run_view` returns a run's training:

```python
from bux.studio.view import view

path = "examples/experiments/attention_ablation.bux"
v = view(open(path).read(), {"gqa|mha|mqa": 2}, path=path)  # the mqa runs
v["sweep"]["axes"][0]["options"]  # ['mha: 12', 'gqa: 4', 'mqa: 1']
v["run"]["name"]  # 'attention_ablation[mqa, optimizer.lr=1e-3]'
```

A node's id is its path from the root, `n.1.0.2`: each step is a child's index, as
{py:func}`bux.lang.ir.children` orders them. Each node also says where it is written
in the file. Both come from {py:func}`bux.lang.compiler.trace`. `picks` names each
axis by an id that survives edits elsewhere in the file:
{py:func}`bux.studio.picks.axis_id`.

The server's endpoints, all JSON, each with the token in `X-Bux-Token`:

| Endpoint | What it answers |
|---|---|
| `GET /api/session` | the file's path, name, text, modification time, and the layer palette |
| `GET /api/file` | the file's modification time, and its text with `?source=1` |
| `POST /api/view` | the view of a source and a run of its sweep |
| `POST /api/inspect` | the inspector's form for a node or a setting |
| `POST /api/edit` | the source with a list of edits applied |
| `POST /api/schedule` | the run's learning rate at up to 512 steps (JAX) |
| `POST /api/plan` | the model's parameters and its FLOPs per token, as `bux plan` counts them (JAX) |
| `POST /api/complete`, `/api/hover` | completion and hover at an offset |
| `POST /api/save` | writes the file; `409` with what is on disk if it changed there |

Offsets, in requests and replies, count UTF-16 code units, as the page's JavaScript
does: past an emoji they are one more than Python's string indices.

## Speed

A view of the file as you type takes milliseconds. These are medians over 20 views
on an Apple M3 Max, measured 2026-10-26 with Python 3.14:

| File | Layers | View |
|---|---|---|
| `architectures/llama3_8b.bux` | 131 | 3 ms |
| `architectures/qwen3_next.bux` | 202 | 5 ms |
| `architectures/deepseek_v3.bux` | 247 | 6 ms |
| `experiments/attention_ablation.bux`, 6 runs | 51 | 12 ms |
| `experiments/schedule_sweep.bux`, 6 runs | 51 | 19 ms |

The first view of a sweep also compiles all of its runs, which the studio then
caches by source.
The first `/api/plan` or `/api/schedule` call imports JAX. After that, each takes
about 0.3 s per run on the CPU, and the studio caches the result by the run's id.
