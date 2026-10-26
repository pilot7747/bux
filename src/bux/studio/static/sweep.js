// The sweep panel of bux studio: a file's axes (its `a | b`s) as rows of chips, its
// runs as a matrix or a table, and its search's dimensions as ranges on their own
// scales. Picking a run is `onPick(picks)`; editing an axis is an `axis` op.

import {
  card, compact, emptyCard, exponent, h, label, linear, log, logTicks, niceTicks, sci,
  svg, textWidth, tooltip,
} from "./charts.js";

// The top-level settings a `where` can start with, which the inspector opens.
const SETTINGS = new Set([
  "data", "tokenizer", "packing", "optimizer", "schedule", "seq_len", "batch", "batch_ramp",
  "steps", "curriculum", "seed", "clip", "loss", "eval", "checkpoint", "log", "observe",
  "profile", "precision", "mesh", "layout", "methods", "compiler", "objective", "search",
  "d_model", "vocab_size", "residual", "init",
]);

/**
 * Draw `sweep` into `el`, replacing what it held.
 *
 * @param {HTMLElement} el
 * @param {object} sweep the View's `sweep` (plans/studio.md)
 * @param {{onPick?: Function, onEdit?: Function, onInspect?: Function}} options
 */
export function renderSweep(el, sweep, options = {}) {
  el.classList.add("bux-panels", "sweep-panel");
  el.replaceChildren();
  const axes = sweep?.axes ?? [];
  const dims = sweep?.dimensions ?? [];
  if (!axes.length && !dims.length) return emptySweep(el);
  const ctx = { sweep, options, tip: tooltip(el), picks: currentPicks(axes) };
  summary(el, ctx);
  const grid = h("div", { cls: "panel-grid" }, el);
  if (axes.length) axesCard(grid, ctx);
  if (axes.length) runsCard(grid, ctx);
  if (dims.length) searchCard(grid, ctx);
}

function emptySweep(el) {
  emptyCard(el, "This file trains one run.",
    "Write alternatives with | anywhere, and each combination is a run; labels name them.",
    "optimizer.lr      = 1e-3 | 3e-3          # two runs\nattn.num_kv_heads = mha: 12 | mqa: 1    # times two, named mha and mqa");
}

function currentPicks(axes) {
  return Object.fromEntries(axes.map((a) => [a.id, a.pick ?? 0]));
}

function pickOf(run, axis) {
  return run.picks?.[axis.id] ?? 0;
}

function isCurrent(ctx, run) {
  return ctx.sweep.axes.every((a) => pickOf(run, a) === ctx.picks[a.id]);
}

function pick(ctx, picks) {
  ctx.options.onPick?.(picks);
}

/** Open the setting `where` is written in, if its first part is one. */
function inspectWhere(ctx, where) {
  const setting = where.split(/[.#\s]/)[0];
  if (SETTINGS.has(setting)) ctx.options.onInspect?.({ setting });
}

// --- Summary --------------------------------------------------------------------------

function summary(el, ctx) {
  const { axes, total, truncated, dimensions, search } = ctx.sweep;
  const bar = h("div", { cls: "panel-toolbar sweep-summary" }, el);
  const box = h("div", { cls: "sweep-count" }, bar);
  const product = axes.reduce((n, a) => n * a.options.length, 1);
  h("span", { cls: "sweep-total", text: `${total} run${total === 1 ? "" : "s"}` }, box);
  if (axes.length > 1 || total < product) {
    const sizes = axes.map((a) => a.options.length).join(" × ");
    h("span", { cls: "sweep-product", text: `${total < product ? "of" : "="} ${sizes}` }, box);
  }
  const notes = h("div", { cls: "sweep-notes" }, bar);
  if (total < product) h("span", { text: `${product - total} alike, so one run` }, notes);
  if (truncated) h("span", { text: `listing the first ${ctx.sweep.runs.length}` }, notes);
  if (dimensions.length) h("span", { text: searchLine(search, dimensions.length, total) }, notes);
}

function searchLine(search, dims, runs) {
  const trials = search?.trials ? `${search.trials} trials` : "trials";
  const each = runs > 1 ? `, each of the ${runs} runs a search` : "";
  return `${trials} over ${dims} dimension${dims === 1 ? "" : "s"}${each}`;
}

// --- Axes -----------------------------------------------------------------------------

function axesCard(grid, ctx) {
  const c = card(grid, { title: "Axes", wide: true, subtitle: "click to pick a run · double-click to edit an alternative" });
  const list = h("div", { cls: "axes" }, c.body);
  for (const axis of ctx.sweep.axes) axisRow(list, ctx, axis);
}

function axisRow(list, ctx, axis) {
  const row = h("div", { cls: "axis-row" }, list);
  const head = h("div", { cls: "axis-head" }, row);
  const where = h("button", { cls: "axis-where", text: axis.where, title: `${axis.where}: open its setting` }, head);
  where.addEventListener("click", () => inspectWhere(ctx, axis.where));
  h("span", { cls: "axis-count", text: `${axis.options.length} options${axis.labeled ? ", labeled" : ""}` }, head);
  const chips = h("div", { cls: "chips" }, row);
  axis.options.forEach((text, i) => chip(chips, ctx, axis, i, text));
  if (editableAxis(axis)) addChip(chips, ctx, axis);
}

/** Split a labeled option, "mqa: 1", into its label and value. */
export function splitOption(text, labeled) {
  const found = labeled ? /^\s*([A-Za-z_][\w-]*)\s*:\s*([\s\S]*)$/.exec(text) : null;
  return found ? { label: found[1], value: found[2].trim() } : { label: null, value: text };
}

/** An axis is edited whole, as source: a labeled one needs its options as "mqa: 1". */
function editableAxis(axis) {
  if (!axis.labeled) return true;
  return axis.options.every((o) => splitOption(o, true).value !== "");
}

function chip(chips, ctx, axis, i, text) {
  const { label: name, value } = splitOption(text, axis.labeled);
  const on = ctx.picks[axis.id] === i;
  const node = h("button", { cls: "chip" + (on ? " on" : ""), title: text, "aria-pressed": String(on) }, chips);
  if (name) h("span", { cls: "chip-label", text: name }, node);
  if (value) h("span", { cls: "chip-value" + (name ? " muted" : ""), text: value }, node);
  node.addEventListener("click", () => pick(ctx, { ...ctx.picks, [axis.id]: i }));
  if (!editableAxis(axis)) return;
  node.addEventListener("dblclick", () => editChip(node, ctx, axis, i));
  if (axis.options.length > 1) removeButton(node, ctx, axis, i);
}

function removeButton(node, ctx, axis, i) {
  const x = h("span", { cls: "chip-remove", role: "button", title: "remove this alternative", text: "×" }, node);
  x.addEventListener("click", (e) => {
    e.stopPropagation();
    editAxis(ctx, axis, axis.options.filter((_, k) => k !== i));
  });
}

function addChip(chips, ctx, axis) {
  const node = h("button", { cls: "chip chip-add", title: "add an alternative", text: "+" }, chips);
  node.addEventListener("click", () => editChip(node, ctx, axis, axis.options.length));
}

/** Turn a chip into an input for option `i`, or a new one past the last. */
function editChip(node, ctx, axis, i) {
  const old = axis.options[i] ?? "";
  const input = h("input", { cls: "chip-input", value: old, spellcheck: "false", "aria-label": `alternative of ${axis.where}` });
  input.style.width = `${Math.max(10, old.length + 2)}ch`;
  node.replaceWith(input);
  input.focus();
  input.select();
  let done = false;
  const finish = (save) => {
    if (done) return;
    done = true;
    input.replaceWith(node);
    if (save) commitOption(ctx, axis, i, input.value.trim());
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") finish(true);
    else if (e.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(false));
}

function commitOption(ctx, axis, i, text) {
  const options = [...axis.options];
  if (!text) options.splice(i, 1);
  else if (text === options[i]) return;
  else options[i] = text;
  if (options.length) editAxis(ctx, axis, options);
}

function editAxis(ctx, axis, options) {
  ctx.options.onEdit?.([{ op: "axis", axis: axis.id, options }]);
}

// --- Runs -----------------------------------------------------------------------------

function runsCard(grid, ctx) {
  const { axes, runs } = ctx.sweep;
  const subtitle = axes.length === 2 ? `${axes[0].where} by ${axes[1].where}` : axes.map((a) => a.where).join(" × ");
  const c = card(grid, { title: "Runs", wide: true, subtitle });
  if (axes.length === 1) return runRow(c.body, ctx, axes[0], runs);
  if (axes.length === 2) return runMatrix(c.body, ctx, axes, runs);
  runTable(c.body, ctx, axes, runs);
}

function optionName(axis, i) {
  const { label: name, value } = splitOption(axis.options[i] ?? "", axis.labeled);
  return name ?? value;
}

/** The class of an option's name: source text in mono, a label in the UI's font. */
function optionClass(axis) {
  return axis.labeled ? "" : " mono";
}

function runCell(parent, ctx, run, tag = "button") {
  const node = h(tag, { cls: "run-cell" + (isCurrent(ctx, run) ? " on" : "") }, parent);
  h("span", { cls: "run-id", text: run.id }, node);
  node.addEventListener("click", () => pick(ctx, { ...ctx.picks, ...run.picks }));
  node.addEventListener("pointermove", (e) => ctx.tip.show(e, [{ value: run.id, name: "run id", mono: true }], run.name));
  node.addEventListener("pointerleave", () => ctx.tip.hide());
  return node;
}

function missingCell(parent, ctx) {
  const node = h("div", { cls: "run-cell missing", text: "—" }, parent);
  node.addEventListener("pointermove", (e) => ctx.tip.show(e, [{ value: "alike", name: "the same run as another, so not a run of its own" }]));
  node.addEventListener("pointerleave", () => ctx.tip.hide());
}

function runRow(body, ctx, axis, runs) {
  const row = h("div", { cls: "run-row" }, body);
  axis.options.forEach((_, i) => {
    const box = h("div", { cls: "run-col" }, row);
    h("div", { cls: "run-head" + optionClass(axis), text: optionName(axis, i), title: axis.options[i] }, box);
    const run = runs.find((r) => pickOf(r, axis) === i);
    if (run) runCell(box, ctx, run);
    else missingCell(box, ctx);
  });
}

function runMatrix(body, ctx, [rows, cols], runs) {
  const table = h("div", { cls: "run-matrix", style: `grid-template-columns: auto repeat(${cols.options.length}, minmax(104px, 168px))` }, body);
  const corner = h("div", { cls: "run-corner" }, table);
  h("span", { cls: "mono", text: `${rows.where} ↓` }, corner);
  h("span", { cls: "mono", text: `${cols.where} →` }, corner);
  cols.options.forEach((_, j) => h("div", { cls: "run-head col" + optionClass(cols), text: optionName(cols, j), title: cols.options[j] }, table));
  rows.options.forEach((_, i) => {
    h("div", { cls: "run-head row" + optionClass(rows), text: optionName(rows, i), title: rows.options[i] }, table);
    cols.options.forEach((_, j) => {
      const run = runs.find((r) => pickOf(r, rows) === i && pickOf(r, cols) === j);
      if (run) runCell(table, ctx, run);
      else missingCell(table, ctx);
    });
  });
}

function runTable(body, ctx, axes, runs) {
  const table = h("table", { cls: "run-table" }, body);
  const head = h("tr", {}, h("thead", {}, table));
  for (const a of axes) h("th", { text: a.where }, head);
  h("th", { text: "run id" }, head);
  const tbody = h("tbody", {}, table);
  for (const run of runs) {
    const tr = h("tr", { cls: isCurrent(ctx, run) ? "on" : "" }, tbody);
    for (const a of axes) h("td", { cls: optionClass(a).trim(), text: optionName(a, pickOf(run, a)), title: a.options[pickOf(run, a)] }, tr);
    h("td", { cls: "mono run-id", text: run.id }, tr);
    tr.addEventListener("click", () => pick(ctx, { ...ctx.picks, ...run.picks }));
    tr.addEventListener("pointermove", (e) => ctx.tip.show(e, [{ value: run.id, name: "run id", mono: true }], run.name));
    tr.addEventListener("pointerleave", () => ctx.tip.hide());
  }
}

// --- Search ---------------------------------------------------------------------------

function searchCard(grid, ctx) {
  const { dimensions, search } = ctx.sweep;
  const c = card(grid, { title: "Search", wide: true, onTitle: () => ctx.options.onInspect?.({ setting: "search" }), subtitle: methodLine(search) });
  const list = h("div", { cls: "dims" }, c.body);
  for (const d of dimensions) dimensionRow(list, ctx, d);
  if (search?.halving) h("div", { cls: "card-note", text: `Successive halving: ${search.halving}.` }, c.body);
}

function methodLine(search) {
  if (!search) return null;
  const parts = [search.method];
  if (search.trials) parts.push(`${search.trials} trials`);
  if (search.halving) parts.push("halving");
  return parts.join(" · ");
}

function dimensionRow(list, ctx, d) {
  const row = h("div", { cls: "dim-row" }, list);
  const head = h("button", { cls: "dim-head", title: `${d.where}: open its setting` }, row);
  h("span", { cls: "dim-where", text: d.where }, head);
  h("span", { cls: "dim-type", text: d.type + (d.log && d.type !== "log_uniform" ? ", log" : "") }, head);
  head.addEventListener("click", () => inspectWhere(ctx, d.where));
  const body = h("div", { cls: "dim-body" }, row);
  if (d.options) choiceChips(body, d.options);
  else rangeBar(body, ctx, d);
}

function choiceChips(body, options) {
  const chips = h("div", { cls: "chips" }, body);
  for (const o of options) h("span", { cls: "chip static", text: o, title: o }, chips);
}

/** Draw a dimension's range on its own scale: its bounds labeled, ticks between, its
 *  middle (what a run takes outside a search) marked. */
function rangeBar(body, ctx, d) {
  const width = Math.max(240, body.clientWidth || 480);
  const root = svg("svg", { class: "chart range", width, height: 46, viewBox: `0 0 ${width} 46` }, body);
  const [pad, y] = [Math.max(14, textWidth(fmtValue(d.high, d), 12, true) / 2 + 2), 20];
  const scale = (d.log ? log : linear)([d.low, d.high], [pad, width - pad]);
  svg("rect", { x: pad - 4, y: y - 4, width: width - 2 * pad + 8, height: 8, rx: 4, class: "range-bar" }, root);
  for (const v of rangeTicks(d)) {
    svg("line", { x1: scale(v), x2: scale(v), y1: y + 6, y2: y + 9, class: "range-tick" }, root);
    if (v !== d.low && v !== d.high) label(root, scale(v), y + 21, fmtValue(v, d), { class: "chart-tick", "text-anchor": "middle" });
  }
  label(root, scale(d.low), y + 21, fmtValue(d.low, d), { class: "range-bound", "text-anchor": "middle" });
  label(root, scale(d.high), y + 21, fmtValue(d.high, d), { class: "range-bound", "text-anchor": "middle" });
  middleDot(root, ctx, d, scale(middle(d)), y);
}

function rangeTicks(d) {
  if (d.log) return logTicks(d.low, d.high);
  const ticks = niceTicks(d.low, d.high, 4);
  return d.type === "int_uniform" ? ticks.filter(Number.isInteger) : ticks;
}

/** Return a dimension's middle: halfway along its scale, as a run outside a search takes. */
export function middle(d) {
  const m = d.log ? Math.sqrt(d.low * d.high) : (d.low + d.high) / 2;
  return d.type === "int_uniform" ? Math.round(m) : Number(m.toPrecision(3));
}

function middleDot(root, ctx, d, x, y) {
  const dot = svg("circle", { cx: x, cy: y, r: 4.5, class: "range-mid" }, root);
  const hit = svg("circle", { cx: x, cy: y, r: 12, class: "chart-hit" }, root);
  for (const node of [dot, hit]) {
    node.addEventListener("pointermove", (e) => ctx.tip.show(e, [{ value: fmtValue(middle(d), d), name: "its middle: what a run takes outside the search" }], d.where));
    node.addEventListener("pointerleave", () => ctx.tip.hide());
  }
}

function fmtValue(v, d) {
  if (d.type === "int_uniform") return compact(v);
  if (d.log && v > 0 && v < 1) return exponent(v);
  return Math.abs(v) >= 1000 ? compact(v) : sci(v);
}

