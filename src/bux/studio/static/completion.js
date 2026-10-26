// The code pane's completion popup: asks the server what fits at the caret, filters
// the items as the word grows, and inserts the one picked.

import { escape } from "./syntax.js";
import { markdown } from "./markdown.js";

const WORD = /[A-Za-z0-9_]*$/;
const KIND_LETTERS = {
  setting: "s", value: "v", layer: "L", config: "C", activation: "a", definition: "d",
  macro: "m", parameter: "p", tap: "@", keyword: "k",
};

export class Completion {
  constructor(editor) {
    this.editor = editor;
    this.items = [];
    this.shown = [];
    this.index = 0;
    this.seq = 0;
    this.el = document.createElement("div");
    this.el.className = "ed-complete popover";
    this.el.hidden = true;
    this.el.addEventListener("mousedown", (e) => this.click(e));
    editor.el.append(this.el);
  }

  get open() {
    return !this.el.hidden;
  }

  /** Ask for the items at the caret, then show them. */
  async request() {
    const ed = this.editor;
    if (!ed.opts.complete) return;
    const seq = ++this.seq;
    const offset = ed.offset;
    const items = await ed.opts.complete(offset).catch(() => []);
    if (seq !== this.seq || !items?.length) return;
    this.anchor = offset - ed.source.slice(0, offset).match(WORD)[0].length;
    this.items = items;
    this.refilter();
  }

  /** Filter by the word typed since the popup opened; close if the caret left it. */
  refilter() {
    const ed = this.editor;
    const at = ed.offset;
    const word = ed.source.slice(this.anchor, at);
    if (at < this.anchor || !/^[A-Za-z0-9_]*$/.test(word)) return this.close();
    const w = word.toLowerCase();
    const starts = this.items.filter((i) => i.label.toLowerCase().startsWith(w));
    const inside = this.items.filter((i) => !starts.includes(i) && i.label.toLowerCase().includes(w));
    this.shown = [...starts, ...inside].slice(0, 200);
    if (!this.shown.length || (this.shown.length === 1 && this.shown[0].label === word)) return this.close();
    this.index = 0;
    this.render();
  }

  close() {
    this.seq++;
    this.el.hidden = true;
  }

  /** Handle a key while open; returns whether it was used. */
  keydown(e) {
    if (!this.open) return false;
    const moves = { ArrowDown: 1, ArrowUp: -1, PageDown: 8, PageUp: -8 };
    if (e.key in moves) {
      this.index = Math.max(0, Math.min(this.shown.length - 1, this.index + moves[e.key]));
      this.render();
    } else if (e.key === "Enter" || e.key === "Tab") {
      this.accept(this.shown[this.index]);
    } else if (e.key === "Escape") {
      this.close();
    } else {
      return false;
    }
    e.preventDefault();
    return true;
  }

  accept(item) {
    if (!item) return;
    this.close();
    this.editor.insert(item.insert_text || item.label, this.anchor, this.editor.offset);
  }

  click(e) {
    const row = e.target.closest("[data-i]");
    if (!row) return;
    e.preventDefault();
    this.accept(this.shown[Number(row.dataset.i)]);
  }

  render() {
    const rows = this.shown.map((item, i) => row(item, i, i === this.index)).join("");
    const doc = this.shown[this.index]?.documentation;
    this.el.innerHTML = `<div class="rows">${rows}</div>` + (doc ? `<div class="doc">${markdown(doc)}</div>` : "");
    this.el.hidden = false;
    this.place();
    this.el.querySelector(".on")?.scrollIntoView({ block: "nearest" });
  }

  place() {
    const r = this.editor.caretRect(this.anchor);
    const el = this.el;
    const below = r.bottom + 4 + el.offsetHeight < innerHeight;
    el.style.left = `${Math.max(8, Math.min(r.left - 28, innerWidth - el.offsetWidth - 8))}px`;
    el.style.top = `${below ? r.bottom + 4 : r.top - 4 - el.offsetHeight}px`;
  }
}

function row(item, i, on) {
  const letter = KIND_LETTERS[item.kind] ?? "·";
  return `<div class="row${on ? " on" : ""}" data-i="${i}">
    <span class="kind k-${escape(item.kind ?? "")}">${escape(letter)}</span>
    <span class="label">${escape(item.label)}</span>
    <span class="detail">${escape(item.detail ?? "")}</span></div>`;
}
