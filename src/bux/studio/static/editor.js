// The code pane: a <textarea> over a highlighted <pre>, with line numbers, error
// underlines, marks, completion and hover. One scroller holds them all, so they move
// together without syncing.

import { highlight, escape, CONFIGS } from "./syntax.js";
import { markdown } from "./markdown.js";
import { Completion } from "./completion.js";

const INDENT = "    ";
const TRIGGERS = new Set(["(", ",", ".", "=", "@", "$", '"']);
const HOVER_MS = 400;

export class CodeEditor {
  /**
   * @param el the element to fill
   * @param opts {onChange(source), onCursor(offset), complete(offset) -> Promise<items>,
   *   hover(offset) -> Promise<{start, end, markdown} | null>, onHistory(kind)}
   */
  constructor(el, opts) {
    this.opts = opts;
    this.builtins = new Set(CONFIGS);
    this.errors = [];
    this.marks = [];
    this.lines = 0;
    this.cursor = -1;
    this.build(el);
    this.completion = new Completion(this);
    this.listen();
    document.fonts?.ready.then(() => this.measure());
  }

  build(host) {
    const el = document.createElement("div");
    el.className = "ed";
    host.replaceChildren(el);
    el.innerHTML = `
      <div class="ed-scroll">
        <div class="ed-gutter" aria-hidden="true"></div>
        <div class="ed-body">
          <pre class="ed-decor" aria-hidden="true"></pre>
          <pre class="ed-code" aria-hidden="true"></pre>
          <textarea class="ed-input" spellcheck="false" wrap="off" autocomplete="off"
            autocapitalize="off" autocorrect="off" aria-label="Source"></textarea>
        </div>
      </div>
      <div class="ed-tip" hidden></div>`;
    const $ = (s) => el.querySelector(s);
    Object.assign(this, { el, scroller: $(".ed-scroll"), gutter: $(".ed-gutter"), body: $(".ed-body") });
    Object.assign(this, { decor: $(".ed-decor"), code: $(".ed-code"), input: $(".ed-input"), tip: $(".ed-tip") });
    this.measure();
  }

  listen() {
    const input = this.input;
    input.addEventListener("input", (e) => this.typed(e));
    input.addEventListener("keydown", (e) => this.keydown(e));
    input.addEventListener("beforeinput", (e) => this.beforeinput(e));
    for (const ev of ["keyup", "mouseup", "focus", "select"]) input.addEventListener(ev, () => this.moved());
    document.addEventListener("selectionchange", () => document.activeElement === input && this.moved());
    input.addEventListener("mousemove", (e) => this.pointer(e));
    input.addEventListener("mouseleave", () => this.hideTip());
    input.addEventListener("blur", () => this.completion.close());
    this.scroller.addEventListener("scroll", () => (this.hideTip(), this.completion.close()));
  }

  // --- The interface ---------------------------------------------------------------

  get source() {
    return this.input.value;
  }

  get offset() {
    return this.input.selectionStart;
  }

  /** Replace the text, keeping the caret where the same text is. */
  setSource(source) {
    const old = this.input.value;
    if (source === old) return;
    const move = shifter(old, source);
    const [a, b] = [this.input.selectionStart, this.input.selectionEnd];
    this.input.value = source;
    this.input.setSelectionRange(move(a), move(b));
    this.moveRanges(move);
    this.render();
  }

  /** Underline the view's errors: `[{start, end, message, hint}]`. */
  setErrors(errors) {
    this.errors = errors ?? [];
    this.renderDecor();
  }

  /** Highlight a primary span and a secondary one (`[start, end]` or null). */
  setMarks(primary, secondary = null) {
    this.marks = [primary && { span: primary, kind: "primary" }, secondary && { span: secondary, kind: "secondary" }].filter(Boolean);
    this.renderDecor();
  }

  /** Scroll to a span and select it. */
  reveal(span) {
    if (!span) return;
    const [start, end] = clampSpan(span, this.source.length);
    this.input.setSelectionRange(start, end);
    this.cursor = start;
    this.scrollTo(start, true);
  }

  /** The names highlighted as builtins: layers, activations and configs. */
  setBuiltins(names) {
    this.builtins = new Set([...CONFIGS, ...names]);
    this.renderCode();
  }

  focus() {
    this.input.focus({ preventScroll: true });
  }

  // --- Editing -----------------------------------------------------------------------

  typed(e) {
    const before = this.lastSource ?? "";
    this.moveRanges(shifter(before, this.source));
    this.render();
    this.opts.onChange?.(this.source);
    this.moved();
    this.scrollTo(this.offset);
    const ch = e.data?.at(-1);
    if (this.completion.open) this.completion.refilter();
    if (!this.completion.open && ch && TRIGGERS.has(ch)) this.completion.request();
  }

  /** Replace `[start, end)` with text, as typing would. */
  insert(text, start = this.input.selectionStart, end = this.input.selectionEnd) {
    this.input.setRangeText(text, start, end, "end");
    this.typed({});
  }

  beforeinput(e) {
    if (e.inputType === "historyUndo" || e.inputType === "historyRedo") {
      e.preventDefault();
      this.opts.onHistory?.(e.inputType === "historyUndo" ? "undo" : "redo");
    }
  }

  keydown(e) {
    if (this.completion.keydown(e)) return;
    if (e.key === " " && e.ctrlKey) return this.prevent(e, () => this.completion.request());
    if (e.key === "Tab") return this.prevent(e, () => (e.shiftKey ? this.dedent() : this.indent()));
    if (e.key === "Enter" && !e.metaKey && !e.ctrlKey) return this.prevent(e, () => this.newline());
    if (e.key === "Escape") this.hideTip();
  }

  prevent(e, fn) {
    e.preventDefault();
    fn();
  }

  newline() {
    const src = this.source;
    const at = this.input.selectionStart;
    const line = src.slice(src.lastIndexOf("\n", at - 1) + 1, at);
    const indent = line.match(/^[ \t]*/)[0] + (/[([]\s*$/.test(line) ? INDENT : "");
    this.insert("\n" + indent);
  }

  indent() {
    const [a, b] = [this.input.selectionStart, this.input.selectionEnd];
    if (!this.source.slice(a, b).includes("\n")) return this.insert(INDENT);
    this.editLines((line) => INDENT + line);
  }

  dedent() {
    this.editLines((line) => line.replace(/^ {1,4}|^\t/, ""));
  }

  /** Rewrite every line the selection touches, keeping them selected. */
  editLines(fn) {
    const src = this.source;
    const start = src.lastIndexOf("\n", this.input.selectionStart - 1) + 1;
    const stop = lineEnd(src, Math.max(start, this.input.selectionEnd - 1));
    const text = src.slice(start, stop).split("\n").map(fn).join("\n");
    this.input.setRangeText(text, start, stop, "select");
    this.typed({});
  }

  moved() {
    const at = this.input.selectionStart;
    if (at === this.cursor) return;
    this.cursor = at;
    this.markLine();
    this.opts.onCursor?.(at);
  }

  // --- Drawing -----------------------------------------------------------------------

  render() {
    this.lastSource = this.source;
    this.renderCode();
    this.renderGutter();
    this.renderDecor();
  }

  renderCode() {
    // The trailing blank keeps a last empty line its height.
    this.code.innerHTML = highlight(this.source, this.builtins) + " ";
  }

  renderGutter() {
    const n = this.source.split("\n").length;
    if (n === this.lines) return this.markLine();
    this.lines = n;
    this.gutter.innerHTML = Array.from({ length: n }, (_, i) => `<div>${i + 1}</div>`).join("");
    this.markLine();
  }

  markLine() {
    const line = lineOf(this.source, this.input.selectionStart);
    this.gutter.querySelector(".on")?.classList.remove("on");
    this.gutter.children[line]?.classList.add("on");
  }

  renderDecor() {
    const ranges = [
      ...this.marks.map((m) => ({ span: m.span, cls: `mark-${m.kind}` })),
      ...this.errors.map((e) => ({ span: [e.start, e.end], cls: "err" })),
    ];
    this.decor.innerHTML = decorate(this.source, ranges) + " ";
  }

  moveRanges(move) {
    for (const e of this.errors) [e.start, e.end] = [move(e.start), move(e.end)];
    for (const m of this.marks) m.span = [move(m.span[0]), move(m.span[1])];
  }

  // --- Geometry ----------------------------------------------------------------------

  measure() {
    const probe = document.createElement("span");
    probe.textContent = "M".repeat(100);
    this.code.append(probe);
    this.charWidth = probe.getBoundingClientRect().width / 100 || 7.5;
    probe.remove();
    const style = getComputedStyle(this.code);
    this.lineHeight = parseFloat(style.lineHeight) || 20;
    this.pad = { x: parseFloat(style.paddingLeft) || 0, y: parseFloat(style.paddingTop) || 0 };
  }

  /** Where the caret at an offset is on screen: {left, top, bottom}. */
  caretRect(offset) {
    const src = this.source;
    const line = lineOf(src, offset);
    const col = visualColumn(src.slice(src.lastIndexOf("\n", offset - 1) + 1, offset));
    const box = this.code.getBoundingClientRect();
    const top = box.top + this.pad.y + line * this.lineHeight;
    return { left: box.left + this.pad.x + col * this.charWidth, top, bottom: top + this.lineHeight };
  }

  /** The offset under a point on screen. */
  offsetAt(x, y) {
    const box = this.code.getBoundingClientRect();
    const line = Math.floor((y - box.top - this.pad.y) / this.lineHeight);
    const lines = this.source.split("\n");
    if (line < 0 || line >= lines.length) return -1;
    const col = (x - box.left - this.pad.x) / this.charWidth;
    const start = lines.slice(0, line).reduce((n, l) => n + l.length + 1, 0);
    return start + charIndex(lines[line], col);
  }

  /** Scroll so the offset is visible; its line centered if it was far away. */
  scrollTo(offset, center = false) {
    const s = this.scroller;
    const top = this.pad.y + lineOf(this.source, offset) * this.lineHeight;
    const view = s.clientHeight;
    if (top < s.scrollTop || top + this.lineHeight > s.scrollTop + view) {
      s.scrollTop = center ? top - view / 3 : top < s.scrollTop ? top - 8 : top + this.lineHeight - view + 8;
    }
    this.scrollColumn(offset);
  }

  scrollColumn(offset) {
    const s = this.scroller;
    const src = this.source;
    const gutter = this.gutter.offsetWidth;
    const x = gutter + this.pad.x + visualColumn(src.slice(src.lastIndexOf("\n", offset - 1) + 1, offset)) * this.charWidth;
    if (x < s.scrollLeft + gutter + 8) s.scrollLeft = Math.max(0, x - gutter - 48);
    else if (x > s.scrollLeft + s.clientWidth - 16) s.scrollLeft = x - s.clientWidth + 48;
  }

  // --- Hover -------------------------------------------------------------------------

  pointer(e) {
    const offset = this.offsetAt(e.clientX, e.clientY);
    clearTimeout(this.hoverTimer);
    const err = this.errors.find((r) => offset >= r.start && offset < Math.max(r.end, r.start + 1));
    if (err) return this.showTip(errorHTML(err), e.clientX, this.caretRect(offset).bottom, "error");
    if (this.tipAt !== undefined && offset >= this.tipAt[0] && offset < this.tipAt[1]) return;
    this.hideTip();
    if (offset < 0 || !this.opts.hover) return;
    this.hoverTimer = setTimeout(() => this.hover(offset, e.clientX), HOVER_MS);
  }

  async hover(offset, x) {
    const source = this.source;
    const h = await this.opts.hover(offset).catch(() => null);
    if (!h?.markdown || source !== this.source) return;
    this.showTip(markdown(h.markdown), x, this.caretRect(h.start).bottom);
    this.tipAt = [h.start, h.end];
  }

  showTip(html, x, y, kind = "") {
    const tip = this.tip;
    tip.innerHTML = html;
    tip.className = `ed-tip popover ${kind}`;
    tip.hidden = false;
    const w = tip.offsetWidth;
    tip.style.left = `${Math.max(8, Math.min(x - 12, innerWidth - w - 8))}px`;
    const below = y + 6 + tip.offsetHeight < innerHeight;
    tip.style.top = `${below ? y + 6 : y - this.lineHeight - 6 - tip.offsetHeight}px`;
  }

  hideTip() {
    clearTimeout(this.hoverTimer);
    this.tip.hidden = true;
    this.tipAt = undefined;
  }
}

// --- Helpers -------------------------------------------------------------------------

function errorHTML(err) {
  const hint = err.hint ? `<div class="hint">hint: ${escape(err.hint)}</div>` : "";
  return `<div class="msg">${escape(err.message)}</div>${hint}`;
}

/**
 * Return a function that moves an offset of `old` to the same text in `next`. An
 * offset inside the text that changed goes to its end, or to null when `strict`.
 */
export function shifter(old, next, strict = false) {
  let p = 0;
  const max = Math.min(old.length, next.length);
  while (p < max && old[p] === next[p]) p++;
  let s = 0;
  while (s < max - p && old[old.length - 1 - s] === next[next.length - 1 - s]) s++;
  const delta = next.length - old.length;
  const inside = strict ? null : next.length - s;
  return (i) => (i <= p ? i : i >= old.length - s ? i + delta : inside);
}

function clampSpan([a, b], n) {
  return [Math.max(0, Math.min(a, n)), Math.max(0, Math.min(Math.max(a, b), n))];
}

function lineOf(src, offset) {
  let n = 0;
  for (let i = src.indexOf("\n"); i !== -1 && i < offset; i = src.indexOf("\n", i + 1)) n++;
  return n;
}

function lineEnd(src, i) {
  const n = src.indexOf("\n", i);
  return n === -1 ? src.length : n;
}

function visualColumn(text) {
  let col = 0;
  for (const ch of text) col = ch === "\t" ? col + 4 - (col % 4) : col + 1;
  return col;
}

function charIndex(line, col) {
  let at = 0;
  for (let i = 0; i < line.length; i++) {
    const next = line[i] === "\t" ? at + 4 - (at % 4) : at + 1;
    if (col < (at + next) / 2) return i;
    at = next;
  }
  return line.length;
}

/** The source as transparent text, with `<span>`s over the ranges, which may overlap. */
function decorate(src, ranges) {
  const live = ranges
    .map((r) => ({ cls: r.cls, span: widen(clampSpan(r.span, src.length), src) }))
    .filter((r) => r.span[1] > r.span[0]);
  const cuts = [...new Set([0, src.length, ...live.flatMap((r) => r.span)])].sort((a, b) => a - b);
  let html = "";
  for (let k = 0; k < cuts.length - 1; k++) {
    const [a, b] = [cuts[k], cuts[k + 1]];
    const cls = live.filter((r) => r.span[0] <= a && b <= r.span[1]).map((r) => r.cls);
    const text = escape(src.slice(a, b));
    html += cls.length ? `<span class="${cls.join(" ")}">${text}</span>` : text;
  }
  return html;
}

/** An empty span covers the character after it, or before it at a line's end. */
function widen([a, b], src) {
  if (b > a) return [a, b];
  if (a < src.length && src[a] !== "\n") return [a, a + 1];
  return a > 0 ? [a - 1, a] : [a, a];
}
