// Stand-ins for the diagram and the panels while their modules are missing (or fail
// to load): an outline of the model, the run's settings, the sweep's runs. They take
// the same calls as the real ones, so the page works with any of them absent.

import { h } from "./dom.js";
import { children } from "./nodes.js";

const KIND_TEXT = { residual: "[ ]  residual", sum: "+  sum", prod: "×  product", neg: "−  negation", seq: "sequence", identity: "_  identity" };

function missing(file) {
  return h("div.fb-note", {}, h("code", {}, file), " is not loaded: a plain outline stands in.");
}

/** The model as an indented outline, with the `Diagram` class's interface. */
export class OutlineDiagram {
  constructor(el, { onSelect }) {
    this.el = el;
    this.onSelect = onSelect;
    this.rows = new Map();
    el.classList.add("fb");
  }

  render(model) {
    this.rows.clear();
    const tree = model ? this.nodeEl(model.graph) : h("div.fb-empty", {}, "No model to draw.");
    this.el.replaceChildren(missing("diagram.js"), h("div.fb-tree", {}, tree));
    this.select(this.selected);
  }

  nodeEl(node) {
    const row = h("button.fb-row", { onclick: () => this.onSelect?.(node.id) }, ...rowText(node));
    if (node.kind === "layer") row.style.setProperty("--cat", `var(--cat-${node.category ?? "other"}-line)`);
    this.rows.set(node.id, row);
    const kids = children(node).map((c) => this.nodeEl(c));
    return h(`div.fb-node.k-${node.kind}`, {}, row, kids.length ? h("div.fb-kids", {}, kids) : null);
  }

  select(id) {
    this.selected = id;
    for (const [rid, row] of this.rows) row.classList.toggle("on", rid === id);
    this.rows.get(id)?.scrollIntoView({ block: "nearest" });
  }

  setDim(dim) {
    this.el.classList.toggle("dim", dim);
  }

  fit() {}

  exportSVG() {
    return "";
  }
}

function rowText(n) {
  const tag = n.label ? h("span.fb-tag", {}, n.label) : null;
  if (n.kind === "layer") return [h("span.fb-swatch"), h("span.fb-title", {}, n.title ?? n.type), tag, ...(n.chips ?? []).map((c) => h("span.fb-chip", {}, c))];
  if (n.kind === "repeat") return [h("span.fb-title.muted", {}, `×${n.times}`), tag];
  if (n.kind === "tap" || n.kind === "ref") return [h("span.fb-title.mono", {}, `${n.kind === "tap" ? "@" : "$"}${n.name}${n.pop ? "!" : ""}`), tag];
  if (n.kind === "const") return [h("span.fb-title.mono", {}, String(n.value))];
  return [h("span.fb-title.muted", {}, KIND_TEXT[n.kind] ?? n.kind), tag];
}

/** The run's settings by group, standing in for `renderTraining`. */
export function renderTraining(el, run, { onInspect }) {
  const groups = Map.groupBy(run.settings ?? [], (s) => s.group);
  const sections = [...groups].map(([group, settings]) =>
    h("section.fb-group", {}, h("h3", {}, group), settings.map((s) =>
      h("button.fb-setting", { class: s.written ? "written" : "", onclick: () => onInspect?.({ setting: s.name }) },
        h("span.name", {}, s.name), h("code", {}, s.text)))));
  el.replaceChildren(h("div.fb", {}, missing("training.js"), h("div.fb-cols", {}, sections)));
}

/** The sweep's runs, standing in for `renderSweep`. */
export function renderSweep(el, sweep, { onPick }) {
  const current = Object.fromEntries(sweep.axes.map((a) => [a.id, a.pick]));
  const rows = sweep.runs.map((r) =>
    h("button.fb-run", { class: sameRun(r.picks, current) ? "on" : "", onclick: () => onPick?.(r.picks) },
      h("span", {}, r.name), h("code", {}, r.id)));
  const more = sweep.truncated ? h("div.fb-empty", {}, `and ${sweep.total - sweep.runs.length} more`) : null;
  el.replaceChildren(h("div.fb", {}, missing("sweep.js"), h("div.fb-runs", {}, rows, more)));
}

function sameRun(a, b) {
  return Object.keys(b).every((k) => (a[k] ?? 0) === b[k]);
}
