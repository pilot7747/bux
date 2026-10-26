// One field of an inspector form: its key, whether the file writes it, its input by
// kind, its sweep's alternatives, its doc and default. Every change is `set(value)`:
// source text, or null to go back to the default.

import { h, icon, html } from "./dom.js";
import { inline } from "./markdown.js";

/**
 * Build a field's element.
 * @param f the plan's Field
 * @param ctx {set(field, value) -> Promise<string | null>, reveal(span), open: Set<key>}
 */
export function fieldEl(f, ctx) {
  const el = h("div.field", { dataset: { key: f.key } });
  el.classList.toggle("written", f.written != null);
  el.classList.toggle("readonly", !f.editable);
  const error = h("div.f-error", { hidden: true });
  const set = async (value) => {
    el.classList.add("busy");
    const message = await ctx.set(f, value);
    el.classList.remove("busy");
    error.hidden = !message;
    error.textContent = message ?? "";
  };
  const input = inputEl(f, set);
  el.append(top(f, ctx, set, input.inline ? input.el : null));
  if (!input.inline) el.append(h("div.f-input", {}, input.el));
  if (f.sweep?.length) el.append(sweepEl(f, set));
  el.append(error, ...docEl(f).filter(Boolean));
  if (f.fields?.length) el.append(nested(f, ctx));
  return el;
}

function top(f, ctx, set, inlineInput) {
  const here = f.where?.span;
  const key = h(here ? "button.f-key.link" : "span.f-key", {
    title: here ? `Written at line ${f.where.line}: reveal it` : f.key,
    onclick: here ? () => ctx.reveal(here) : null,
  }, lastPart(f.key));
  const dot = f.written != null ? h("span.f-dot", { title: `Written in ${f.where?.file ?? "this file"}` }) : null;
  const reset = f.written != null && f.editable
    ? h("button.f-reset.ghost", { title: `Reset to the default${f.default ? ` (${f.default})` : ""}`, onclick: () => set(null) }, icon("reset"))
    : null;
  return h("div.f-top", {}, key, dot, h("span.f-type", { title: f.type }, f.type), h("span.grow"), reset, inlineInput);
}

function lastPart(key) {
  return key.split(".").at(-1);
}

// --- Inputs --------------------------------------------------------------------------

function inputEl(f, set) {
  const disabled = !f.editable;
  if (f.input === "bool") return { inline: true, el: switchEl(f, set, disabled) };
  if (f.input === "choice" && f.options?.length && !f.sweep) return { inline: false, el: choiceEl(f, set, disabled) };
  return { inline: false, el: textEl(f, set, disabled) };
}

function switchEl(f, set, disabled) {
  const on = (f.written ?? f.value) === "true";
  const input = h("input", { type: "checkbox", checked: on, disabled, role: "switch" });
  input.addEventListener("change", () => set(input.checked ? "true" : "false"));
  return h("label.switch", { title: on ? "true" : "false" }, input, h("span.track"));
}

function choiceEl(f, set, disabled) {
  const current = f.written ?? f.value;
  const options = f.options.includes(current) ? f.options : [current, ...f.options];
  const select = h("select", { disabled }, options.map((o) => h("option", { value: o, selected: o === current }, o)));
  select.addEventListener("change", () => set(select.value));
  return h("div.select", {}, select);
}

function textEl(f, set, disabled) {
  const initial = f.written ?? (f.input === "config" ? f.value : "");
  const input = h("input.mono", {
    type: "text", value: initial, disabled, spellcheck: false,
    placeholder: f.written == null ? f.value : (f.default ?? ""),
  });
  input.classList.add(`in-${f.input}`);
  const commit = () => {
    const text = input.value.trim();
    if (text === initial.trim()) return;
    set(text === "" ? null : text);
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") (e.preventDefault(), commit());
    if (e.key === "Escape") (input.value = initial, input.blur());
  });
  input.addEventListener("blur", commit);
  return input;
}

// --- Sweep, doc, nested --------------------------------------------------------------

function sweepEl(f, set) {
  const alts = f.sweep;
  const write = (list) => set(list.join(" | "));
  const chips = alts.map((alt, i) =>
    h("span.chip", {}, chipText(alt),
      f.editable && h("button.x", { title: "Remove this alternative", onclick: () => write(alts.filter((_, j) => j !== i)) }, icon("close"))));
  const add = f.editable ? addChip((text) => write([...alts, text])) : null;
  return h("div.f-sweep", { title: "Alternatives: one run each" }, h("span.f-sweep-label", {}, "sweep"), chips, add);
}

function chipText(alt) {
  const m = alt.match(/^(\w+):\s*(.*)$/);
  return m ? [h("b", {}, m[1]), " ", m[2]] : alt;
}

function addChip(add) {
  const input = h("input.mono.chip-input", { type: "text", placeholder: "value", hidden: true, spellcheck: false });
  const button = h("button.chip.add", { title: "Add an alternative" }, icon("plus"));
  button.onclick = () => ((button.hidden = true), (input.hidden = false), input.focus());
  const done = () => ((input.hidden = true), (button.hidden = false), (input.value = ""));
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && input.value.trim()) (e.preventDefault(), add(input.value.trim()), done());
    if (e.key === "Escape") done();
  });
  input.addEventListener("blur", done);
  return [button, input];
}

function docEl(f) {
  const parts = [];
  if (f.doc) parts.push(html("span.f-doc-text", inline(f.doc)));
  if (f.default != null && f.default !== "") parts.push(h("span.f-default", {}, "default ", h("code", {}, f.default)));
  if (f.required) parts.push(h("span.f-default", {}, "required"));
  const note = !f.editable && f.note ? h("div.f-note", {}, icon("lock"), f.note) : null;
  const doc = h("div.f-doc.clamped", { title: "" }, parts);
  doc.addEventListener("click", () => doc.classList.toggle("clamped"));
  return [note, parts.length ? doc : null];
}

function nested(f, ctx) {
  const open = ctx.open.has(f.key) || f.fields.some((g) => g.written != null);
  const details = h("details.f-group", { open }, h("summary", {}, icon("chevron"), `${f.fields.length} settings of `, h("code", {}, lastPart(f.key))));
  details.addEventListener("toggle", () => (details.open ? ctx.open.add(f.key) : ctx.open.delete(f.key)));
  details.append(...f.fields.map((g) => fieldEl(g, ctx)));
  return details;
}
