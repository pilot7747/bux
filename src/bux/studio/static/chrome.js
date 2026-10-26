// What the page shows around the views: the file and its saved state, the run picker,
// the error list over the canvas, and the status bar.

import { h, icon } from "./dom.js";

/** Show the file's name and whether it has unsaved edits. */
export function showFile({ name, path, dirty }) {
  const el = document.getElementById("file-name");
  el.textContent = name;
  el.title = path;
  document.getElementById("dirty").hidden = !dirty;
  document.getElementById("file").title = dirty ? "Unsaved edits (⌘S saves)" : "Saved";
  document.getElementById("save").disabled = !dirty;
  document.title = `${dirty ? "● " : ""}${name} · bux studio`;
}

/** One segmented control an axis of the sweep; `onPick(axisId, option)`. */
export function renderRunPicker(el, axes, onPick) {
  el.replaceChildren(...(axes ?? []).map((axis) => axisEl(axis, onPick)));
  el.hidden = !axes?.length;
}

function axisEl(axis, onPick) {
  const label = h("span.axis-name", { title: axis.where }, axis.labeled ? axis.where.split(".").at(-1) : axis.where);
  if (axis.options.length > 5 || axis.options.join("").length > 40) {
    const select = h("select", { onchange: () => onPick(axis.id, Number(select.value)) },
      axis.options.map((o, i) => h("option", { value: i, selected: i === axis.pick }, o)));
    return h("div.axis", {}, label, h("div.select.small", {}, select));
  }
  const buttons = axis.options.map((o, i) =>
    h("button", { "aria-pressed": String(i === axis.pick), title: `${axis.where}: ${o}`, onclick: () => onPick(axis.id, i) }, o));
  return h("div.axis", {}, label, h("div.seg", { role: "group" }, buttons));
}

/** The errors over the canvas: a card listing each one; `onReveal(span)`. */
export function renderErrors(el, errors, { stale, onReveal }) {
  el.hidden = !errors.length;
  if (!errors.length) return;
  const open = el.dataset.closed !== "1";
  const count = `${errors.length} error${errors.length > 1 ? "s" : ""}`;
  const sub = stale ? "showing the last model that compiled" : "";
  const head = h("button.errors-head", { onclick: () => ((el.dataset.closed = open ? "1" : "0"), renderErrors(el, errors, { stale, onReveal })) },
    icon("error"), h("b", {}, count), h("span.sub", {}, sub), h("span.grow"), h("span.toggle", {}, open ? "Hide" : "Show"));
  const rows = errors.map((e) =>
    h("button.error-row", { onclick: () => onReveal([e.start, e.end]), title: "Show in the code" },
      h("span.where", {}, e.file ? `${e.file}:${e.line}` : `${e.line}:${e.col}`),
      h("span.text", {}, h("span.msg", {}, e.message), e.hint ? h("span.hint", {}, e.hint) : null)));
  el.replaceChildren(head, open ? h("div.error-rows", {}, rows) : null);
}

/** The status bar: errors, tokens, parameters on the left; run and cursor on the right. */
export function renderStatus(el, { errors, run, plan, planning, cursor, onErrors }) {
  const n = errors?.length ?? 0;
  const status = h("button.st.st-errors", { class: n ? "bad" : "ok", onclick: n ? onErrors : null },
    h("span.st-dot"), n ? `${n} error${n > 1 ? "s" : ""}` : "No errors");
  const tokens = run?.budget?.tokens_text;
  const params = plan?.params_text ?? (planning ? "…" : null);
  const parts = [
    status,
    tokens ? h("span.st", { title: budgetTitle(run.budget) }, withUnit(tokens, "tokens")) : null,
    params ? h("span.st", { title: "Parameters, from bux plan" }, withUnit(params, "params")) : null,
    h("span.grow"),
    run ? h("span.st.st-run", { title: "The run shown" }, run.name, h("code", {}, run.id)) : null,
    cursor ? h("span.st", {}, `Ln ${cursor.line}, Col ${cursor.col}`) : null,
  ];
  el.replaceChildren(...parts.filter(Boolean));
}

function withUnit(text, unit) {
  if (text === "…") return `${unit} …`;
  return text.includes(unit.slice(0, 5)) ? text : `${text} ${unit}`;
}

function budgetTitle(b) {
  return `${b.steps.toLocaleString()} steps × ${b.batch} × ${b.seq_len.toLocaleString()} tokens${b.varies ? " (varies)" : ""}`;
}
