// The inspector: a form for what is selected (a node, a setting, or the model), whose
// edits become `set` ops on the file.

import { h, html, icon } from "./dom.js";
import { markdown } from "./markdown.js";
import { fieldEl } from "./fields.js";

const FILTER_AT = 10;

export class Inspector {
  /**
   * @param el the element to fill
   * @param opts {onEdit(ops) -> Promise<string | null> (an error message, or null),
   *   onReveal(span)}
   */
  constructor(el, opts) {
    this.el = el;
    this.opts = opts;
    this.open = new Set();
    this.filter = "";
    this.more = false;
    el.classList.add("insp");
    this.clear();
  }

  /** Show a form; `target` is what its `set` ops change. */
  show(form, target) {
    const key = JSON.stringify(target);
    if (this.busyTyping()) return void (this.pending = [form, target]);
    if (key !== this.key) Object.assign(this, { key, filter: "", more: false, open: new Set() });
    this.form = form;
    this.target = target;
    const scroll = this.el.scrollTop;
    this.el.replaceChildren(head(form, this.opts.onReveal), this.fieldsEl(form.fields));
    this.el.scrollTop = scroll;
  }

  /** Show what to do when nothing is selected, or a message. */
  clear(message = null) {
    this.key = this.form = this.target = null;
    this.el.replaceChildren(h("div.insp-empty", {}, h("div.title", {}, message ?? "Nothing selected"),
      message ? null : h("div", {}, "Select a layer in the diagram, or put the cursor on it in the code.")));
  }

  /** An error in place of the form, as `/api/inspect` returns. */
  showError(message) {
    this.key = null;
    this.el.replaceChildren(h("div.insp-empty.error", {}, icon("error"), h("div", {}, message)));
  }

  // While the user types in one of its inputs, a new form waits for the blur.
  busyTyping() {
    const a = document.activeElement;
    if (!this.el.contains(a) || a.tagName !== "INPUT" || a.type !== "text") return false;
    if (!this.waiting) {
      this.waiting = true;
      a.addEventListener("blur", () => {
        this.waiting = false;
        const p = this.pending;
        this.pending = null;
        if (p) setTimeout(() => this.show(...p));
      }, { once: true });
    }
    return true;
  }

  fieldsEl(fields) {
    const ctx = { set: (f, value) => this.set(f, value), reveal: (span) => this.opts.onReveal?.(span), open: this.open };
    const shown = fields.filter((f) => matches(f, this.filter));
    const main = shown.filter((f) => f.written != null || f.required || this.filter);
    const rest = shown.filter((f) => !main.includes(f));
    const box = h("div.insp-fields");
    if (fields.length >= FILTER_AT) box.append(this.filterEl());
    box.append(...main.map((f) => fieldEl(f, ctx)));
    if (rest.length) box.append(this.moreEl(rest, ctx, main.length === 0));
    if (!shown.length) box.append(h("div.insp-none", {}, fields.length ? "No setting matches." : "No settings."));
    return box;
  }

  filterEl() {
    const input = h("input.insp-filter", { type: "search", placeholder: "Filter settings", value: this.filter });
    input.addEventListener("input", () => {
      this.filter = input.value;
      this.el.querySelector(".insp-fields").replaceWith(this.fieldsEl(this.form.fields));
      this.el.querySelector(".insp-filter").focus();
    });
    return input;
  }

  moreEl(rest, ctx, alone) {
    const details = h("details.insp-more", { open: this.more || alone },
      h("summary", {}, icon("chevron"), alone ? "Settings" : "Defaults", h("span.count", {}, String(rest.length))));
    details.addEventListener("toggle", () => (this.more = details.open));
    details.append(...rest.map((f) => fieldEl(f, ctx)));
    return details;
  }

  async set(field, value) {
    const op = { op: "set", target: this.target, key: field.key, value };
    try {
      return (await this.opts.onEdit([op])) ?? null;
    } catch (e) {
      return e.message;
    }
  }
}

function matches(f, filter) {
  const q = filter.trim().toLowerCase();
  return !q || f.key.toLowerCase().includes(q) || (f.fields ?? []).some((g) => matches(g, filter));
}

function head(form, reveal) {
  const where = whereEl(form.where, reveal);
  const sub = form.subtitle ? h("div.insp-sub", {}, h("code", {}, form.subtitle)) : null;
  const doc = form.doc ? html("div.insp-doc.md.clamped", markdown(form.doc)) : null;
  doc?.addEventListener("click", () => doc.classList.toggle("clamped"));
  return h("div.insp-head", {}, h("div.insp-title", {}, form.title), sub, where, doc);
}

function whereEl(where, reveal) {
  if (!where) return null;
  if (where.file) return h("div.insp-where", {}, `Defined in ${where.file}, line ${where.line}`);
  return h("button.insp-where.link", { onclick: () => reveal?.(where.span) }, `Defined at line ${where.line}`);
}
