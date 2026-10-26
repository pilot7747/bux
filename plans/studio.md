# bux studio: a visual editor for .bux files

`bux studio FILE` opens a local web page showing the file three ways at once: its
source, its model drawn as a Sebastian Raschka-style architecture diagram, and, for a
run, its training (budget, LR schedule, curriculum, data mixture, evals) and its sweep
(axes, runs, search dimensions). Every one of them is editable, and every edit is an
edit of the file.

## Principles

1. **The file is the truth.** The studio draws what the compiler says the file means
   (the IR of the selected run), never a model of its own. Every visual edit becomes a
   *splice* of the user's text at the place the setting is written, or where the
   language would write it, so comments, macros, patterns and modules survive. The
   edited source goes through the compiler again before anything is drawn.
2. **No toolchain.** Python's stdlib `http.server` serves a build-free front end:
   ES modules, SVG and CSS shipped as package data in `src/bux/studio/static/`. No
   npm, no bundler, no CDN, no new Python dependency. It works offline, and on a
   cluster login node through `ssh -L`.
3. **Small pieces.** The Python side is pure functions of the source text (tested
   with pytest); the server only routes. The JS side is a few ES modules with narrow
   interfaces (below); the layout of the diagram is done in the browser, where text
   can be measured.
4. **Hackable.** A layer's title, category, chips and inside diagram come from
   registries (`bux.studio.palette`, `bux.studio.internals`), so a new layer shows up
   with sensible defaults and can register a better picture in a few lines.

## Pieces and owners

```
src/bux/lang/compiler.py      trace(): where every node and config is written   (lead)
src/bux/studio/
  __init__.py                 package docs: what it is, how the pieces fit      (serve)
  server.py                   stdlib HTTP server, the JSON API, `bux studio`    (serve)
  view.py                     the view model: diagram tree, facts, outline, sweep (serve)
  training.py                 the training view: budget, schedule, stretches... (serve)
  palette.py                  titles, categories, chips, insert templates        (serve)
  internals.py                registry of inside-a-layer diagrams (Raschka's zooms) (serve)
  targets.py                  a target (node path, setting) -> where it is written (edit)
  forms.py                    the inspector's forms                               (edit)
  edits.py                    edit ops -> new source                              (edit)
  static/
    tokens.css                design tokens: colors, type, radii (lead; all read it)
    index.html app.css app.js api.js editor.js inspector.js                     (shell)
    diagram.js diagram.css                                                      (diagram)
    training.js sweep.js charts.js panels.css                                   (panels)
    dev/                      harness pages + fixtures for each component (each its own)
tests/test_studio_*.py        (edit: targets, forms, edits; serve: view, training, server)
docs/                         a page for the studio + CLI reference                (serve)
```

Agents work on branches `studio-<role>` in `~/bux-worktrees/studio-<role>`, merged
into `studio` by the lead.

## The compiler's trace (lead, in `bux.lang.compiler`)

```python
@dataclass(frozen=True)
class Written:
    """Where a config is written, in the joined source (Trace.program.source)."""
    config: cfg.Config           # the config it describes
    span: Span                   # the call `gqa(...)`, or the bare name `rmsnorm`
    call: ast.Call | None        # the call; None for a bare name
    base: Written | None = None  # what this call amends: for `attn.num_kv_heads = 1`,
                                 # the `gqa(...)` that attn is

@dataclass
class Trace:
    program: modules.Program     # .source: every file joined, this one first; .units
    arch: ir.Architecture | None
    train: tcfg.TrainConfig | None   # None for a model
    axes: list[Axis]             # the sweep's axes met under `choices`, in order written
    dimensions: list[Dimension]  # the search's, met under `choices`
    nodes: dict[tuple[int, ...], Span]  # where each node of arch.graph is written
    statements: dict[str, ast.Statement]  # the definition in effect for each name
    def written(self, config: cfg.Config) -> Written | None: ...  # by identity
    def written_setting(self, name: str) -> Written | None: ...  # top-level config setting

def trace(source, *, choices=None, point=None, entry="model",
          filename="<string>", modules=None) -> Trace   # raises BuxError like compile
```

A node's path is its child indices from `arch.graph`: `Seq.items`, `Sum.terms`,
`Prod.factors` by index; `Residual.body`, `Neg.body`, `Repeat.body` are child 0. A
span at or after `len(source)` of the file is in a module (read-only: edit it with an
amendment in this file). Merges the compiler inserts have no span.

## HTTP API (serve)

`bux studio FILE [--port 8765] [--host 127.0.0.1] [--no-browser]` serves
`http://127.0.0.1:PORT/?t=TOKEN`. Every `/api/*` call carries header `X-Bux-Token:
TOKEN` (a random token per session, so another web page cannot write the file). All
bodies and replies are JSON. `picks` selects a run of the sweep: `{axisId: option}`
(missing axes take option 0). Offsets are into the source the client sent.

| Endpoint | Body | Reply |
|---|---|---|
| `GET /api/session` | | `{path, name, source, mtime, palette: [PaletteItem]}` |
| `GET /api/file` | | `{mtime}`; `?source=1` adds `source` (poll for edits made elsewhere) |
| `POST /api/view` | `{source, picks}` | `View` |
| `POST /api/inspect` | `{source, picks, target}` | `Form` or `{error}` |
| `POST /api/edit` | `{source, picks, ops: [Op]}` | `{source}` or `{error}` |
| `POST /api/schedule` | `{source, picks}` | `{points: [[step, lr]], peak, steps}` (imports JAX, lazily) |
| `POST /api/plan` | `{source, picks}` | `{params, params_text, flops_per_token}` (JAX, lazily) |
| `POST /api/complete` | `{source, offset}` | `[{label, kind, detail, documentation, insert_text}]` |
| `POST /api/hover` | `{source, offset}` | `{start, end, markdown}` or `null` |
| `POST /api/save` | `{source, mtime}` | `{mtime}`; 409 `{source, mtime}` if the file changed on disk |

### View

```ts
type View = {
  ok: boolean,                  // false: errors; model/run/sweep may be null
  errors: Error[],
  model: Model | null,          // null for a module, or while there are errors
  run: Run | null,              // null for a model without training
  sweep: Sweep,
  outline: {name: string, kind: "layer"|"block"|"macro"|"setting"|"value",
            span: [number, number] | null, origin: string | null}[],
}
type Error = {start: number, end: number, line: number, col: number,
              message: string, hint: string | null, file: string | null}
type Model = {
  name: string,                 // the file's stem
  facts: {d_model: number, vocab_size: number | null, layers: number,
          residual: string | null, init: string | null,
          by_type: [string, number][]},       // [["rmsnorm", 65], ["gqa", 32], ...]
  graph: Node,
}
type Node = {                   // a node of the IR graph, as drawn
  id: string,                   // "n" for the root, "n.0.2" for child 2 of child 0
  kind: "layer"|"seq"|"residual"|"sum"|"prod"|"neg"|"const"|"identity"
      | "repeat"|"tap"|"ref",
  label: string | null,         // the definition it came from; a labeled non-layer
                                // node is a block: drawn as a container, collapsible
  shared: boolean,              // `shared name = ...`: one module at every use
  span: [number, number] | null,// where it is written in this file; null: elsewhere
  origin: string | null,        // the module it is written in, if not this file
  editable: boolean,            // can be inserted around, deleted, wrapped
  // layer:
  type?: string,                // "gqa"
  title?: string,               // "Grouped-query attention"
  category?: Category,
  chips?: string[],             // ["32 heads", "8 KV heads", "RoPE 500k"]
  width?: number | "tokens" | "streams",
  ports?: {name: string, tap: string, mode: "write"|"read"|"pop"}[],
  written?: [number, number] | null,  // where its config is written (the call)
  internals?: Node | null,      // what is inside it, Raschka's zoom; same schema
  inserted?: boolean,           // a merge the compiler inserted
  // seq: items; sum: terms; prod: factors; residual/neg/repeat: body
  items?: Node[], terms?: Node[], factors?: Node[], body?: Node,
  times?: number,               // repeat
  value?: number,               // const
  name?: string, pop?: boolean, // tap, ref
}
type Category = "io"|"norm"|"attention"|"recurrent"|"ffn"|"experts"|"op"|"other"
type Run = {
  name: string, id: string,
  budget: {steps: number, batch: number, seq_len: number, tokens: number,
           tokens_text: string, varies: boolean},
  optimizer: {type: string, lr: number, text: string},
  schedule: {type: string, text: string, warmup: number,   // warmup in steps
             regions: {name: string, start: number, stop: number}[]},  // warmup, decay...
  stretches: {start: number, stop: number, batch: number, seq_len: number,
              tokens_per_step: number, data: string}[],
  phases: {n: number, start: number, seq_len: number|null, batch: number|null,
           data: string|null}[],
  ramp: {start: number, steps: number, multiple: number|null} | null,
  data: {name: string, weight: number, text: string}[],   // the first stretch's mixture
  evals: {name: string, every: number}[],
  checkpoint: {every: number} | null,
  settings: {name: string, text: string, written: boolean, group: string}[],
                                // every top-level training setting: its value as
                                // source, whether the file writes it, its group
                                // ("budget", "optimizer", "data", "loss", "eval",
                                //  "logging", "precision", "parallelism", "search")
}
type Sweep = {
  axes: {id: string, where: string, options: string[], labeled: boolean,
         span: [number, number] | null, pick: number}[],
  runs: {name: string, id: string, picks: {[axisId: string]: number}}[],
  total: number, truncated: boolean,          // runs lists at most 256
  dimensions: {where: string, type: string, low: number|null, high: number|null,
               options: string[]|null, log: boolean, span: [number, number] | null}[],
  search: {method: string, trials: number|null, halving: string|null} | null,
}
type PaletteItem = {name: string, kind: "layer"|"activation", category: Category,
                    title: string, doc: string, template: string}
```

### Inspector forms and edits

```ts
type Target = {node: string}             // a node of model.graph
            | {setting: string}          // a top-level setting: "d_model", "optimizer"
            | {model: true}              // d_model, vocab_size, residual, init
type Form = {title: string, subtitle: string | null, doc: string,
             where: Where | null, fields: Field[]}
type Where = {span: [number, number] | null, file: string | null, line: number}
type Field = {
  key: string,                  // dotted path within the target: "rope.base"; for
                                // {setting} / {model} the setting's own name
  type: string,                 // as the reference writes it: "int | none"
  doc: string, required: boolean, default: string | null,
  value: string,                // the effective value, as source
  written: string | null,       // the source text where the file writes it
  where: Where | null,
  input: "int"|"float"|"bool"|"choice"|"text"|"config",
  options: string[] | null,     // for a choice
  sweep: string[] | null,       // the alternatives, when written as a | b
  editable: boolean, note: string | null,   // why not: "set by the macro's parameter h"
  fields: Field[] | null,       // a config's own fields
}
type Op =
  | {op: "set", target: Target, key: string | null, value: string | null}
        // value is source text ("16", "1e-3 | 3e-3", "rope(base=1e6)"); null removes
        // the setting (back to its default). key null with {setting}: the whole value.
        // A setting written in a module is set by an amendment in this file.
  | {op: "repeat", node: string, times: number}
  | {op: "insert", node: string, where: "before"|"after", text: string}
  | {op: "delete", node: string}
  | {op: "wrap", node: string}            // f  ->  [f]
  | {op: "unwrap", node: string}          // [f] ->  f
  | {op: "retype", node: string, type: string}
  | {op: "axis", axis: string, options: string[]}
        // rewrite a sweep axis's `a | b` (sweep.axes[].id) with these alternatives,
        // each source text, labeled ones as "mqa: 1"; one option ends the sweep there
```

`edits.apply(source, ops, *, picks, path) -> str` and
`forms.inspect(source, target, *, picks, path) -> Form` are the Python entry points the
server calls (edit owns them). They raise `EditError` (a message for the user) when an
op cannot apply, such as an insert inside a pattern string.

## Front-end modules

```js
// api.js (shell): the token is read from the page URL
export async function api(path, body) -> json          // POST with body, GET without

// diagram.js (diagram)
export class Diagram {
  constructor(el, {onSelect(id|null), onAction(op), onHover(id|null), palette})
  render(model)          // keeps zoom, pan and which blocks are open across renders
  select(id)             // highlight; pan to it if off screen
  setDim(dim)            // dimmed while the source has errors
  fit()
  exportSVG() -> string  // a standalone SVG, styles inlined
}
// training.js, sweep.js (panels)
export function renderTraining(el, run, {onEdit(ops), onInspect(target), schedule()})
                          // schedule() -> Promise<{points, peak, steps}>
export function renderSweep(el, sweep, {onPick(picks), onEdit(ops), onInspect(target)})
// inspector.js, editor.js (shell)
export class Inspector { constructor(el, {onEdit(ops), onReveal(span)}); show(form); clear() }
export class CodeEditor {
  constructor(el, {onChange(source), onCursor(offset), complete(offset), hover(offset)})
  setSource(source)  setErrors(errors)  reveal(span)  get source()
}
```

The shell owns the page: header (file, saved state, run picker, theme), the code
pane, the canvas tabs (Model, Training, Sweep), the inspector, undo/redo of both kinds
of edit, save (Cmd/Ctrl+S, explicit), reload when the file changes on disk, and the
link between the two views: a cursor in the code selects the innermost node whose
`span` or `written` holds it, and selecting a node reveals its span.

## Look

Clean and modern, light and dark (`data-theme` on `<html>`), every color from
`static/tokens.css`. The diagram follows Raschka's LLM architecture figures:

- Flow bottom to top: the tokens enter at the bottom, the logits leave at the top.
- Rounded boxes with a pastel fill and a darker border by category (`--cat-*`);
  a human title ("Grouped-query attention"), the key numbers as small chips under it,
  and the definition's name as a small mono tag.
- `[f]` is a ⊕ above `f`, with the skip wire drawn around it on one side.
- `f + g` and `f * g` are columns side by side, forked below and merged in ⊕ / ⊗.
- A repeat is a soft rounded container with `×32` in its margin; a block (a labeled
  node) is a container captioned with its name. The first occurrence of a block is
  open; later ones, and runs of the same block (`D D D`), are folded into a card
  ("D ×3").
- Inside a layer (`internals`), opened on demand, is a zoom panel beside it with a
  leader line, as Raschka draws the feed-forward module.
- A card at the top left: the model's name, parameters (when `/api/plan` answers),
  d_model, vocabulary, layers, and context (`seq_len`) for a run.

Charts (panels) follow the same tokens: thin lines, no chart junk, direct labels.

## Verification

- Python: pytest for trace, targets, forms, edits (each op, on real examples), view,
  training, and the server (spun up on a free port in a thread). The examples sweep:
  every file under `examples/` gives a `View` with `ok` and no exception.
- JS: each component has a harness page in `static/dev/` that loads fixtures; check
  it in headless Chrome and read the screenshot:
  `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new
  --disable-gpu --screenshot=/tmp/x.png --window-size=1600,1000 URL`.
  `node --check file.js` for syntax, if a node is on the PATH (not a dependency).
- Lint: `uv run ruff check . && uv run ruff format --check .`; types:
  `uv run ty check src tests`.
