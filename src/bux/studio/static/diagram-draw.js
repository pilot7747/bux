/* Drawing of the model diagram: sized frames (diagram-layout.js) -> SVG markup.

   Everything is drawn into an Ink: wires go into a few long paths, boxes and marks
   into layers of markup, and what can be selected into `rects`, so that rings,
   toolbars and panels can find it without touching the DOM. */

import { FONT, G, measure } from "./diagram-layout.js";

const r1 = (v) => Math.round(v * 10) / 10;
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ESC[c]);

function el(tag, attrs, inner) {
  let s = `<${tag}`;
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    s += ` ${k}="${typeof v === "number" ? r1(v) : esc(v)}"`;
  }
  return inner === undefined ? `${s}/>` : `${s}>${inner}</${tag}>`;
}

/** A line of text centered on `cy`. */
function text(cls, x, cy, content, attrs = {}) {
  return el("text", { class: cls, x, y: cy, dy: ".35em", ...attrs }, esc(content));
}

/** Path data, built a command at a time: path().M(0, 0).H(10).toString(). */
class Path {
  d = "";
  M(x, y) { return this.add(`M${r1(x)} ${r1(y)}`); } // prettier-ignore
  L(x, y) { return this.add(`L${r1(x)} ${r1(y)}`); } // prettier-ignore
  H(x) { return this.add(`H${r1(x)}`); } // prettier-ignore
  V(y) { return this.add(`V${r1(y)}`); } // prettier-ignore
  Q(cx, cy, x, y) { return this.add(`Q${r1(cx)} ${r1(cy)} ${r1(x)} ${r1(y)}`); } // prettier-ignore
  Z() { return this.add("Z"); } // prettier-ignore
  add(s) {
    this.d += s;
    return this;
  }
  toString() {
    return this.d;
  }
}
const path = () => new Path();

const ARROW = { up: [0, 1], left: [1, 0], right: [-1, 0] };
const LANE = 1.2; // the width of a lane of a widened wire, and of the gap between two
const TIP = 7; // how far a widened wire stops short of its arrowhead's tip

export class Ink {
  constructor(lanes, zooms) {
    this.lanes = lanes;
    this.zooms = zooms; // ids of the boxes whose internals are open
    this.back = [];
    this.front = [];
    this.leaders = []; // under every panel, so a zoom's leaders pass behind the zoom it starts in
    this.over = [];
    this.thin = [];
    this.wide = new Map(); // background -> path data of widened wires on it
    this.heads = [];
    this.rects = new Map(); // id -> {x, y, w, h} of what can be selected
    this.nodes = new Map(); // id -> the node drawn there
    this.extents = []; // [x0, y0, x1, y1] of what is drawn
    this.inside = new Map(); // id -> {node, box} of what the zoom panels hold
  }

  line(d, wide, bg) {
    if (!this.widens(wide)) return this.thin.push(String(d));
    if (!this.wide.has(bg)) this.wide.set(bg, []);
    this.wide.get(bg).push(String(d));
  }

  widens(wide) {
    return wide && this.lanes > 0;
  }

  arrow(x, y, dir, wide = false) {
    const [ux, uy] = ARROW[dir];
    const [len, half] = this.widens(wide) ? [8, 5.5] : [6, 3.5];
    const [bx, by] = [x + ux * len, y + uy * len];
    const [px, py] = [uy * half, ux * half];
    this.heads.push(path().M(bx - px, by - py).L(x, y).L(bx + px, by + py).Z());
  }

  hit(node, x, y, w, h) {
    if (!node.id) return;
    if (!this.rects.has(node.id)) this.rects.set(node.id, { x, y, w, h });
    this.nodes.set(node.id, node);
  }

  extent(x0, y0, x1, y1) {
    this.extents.push([x0, y0, x1, y1]);
  }

  bounds() {
    const [x0, y0, x1, y1] = [0, 1, 2, 3].map((i) => this.extents.map((e) => e[i]));
    const [x, y] = [Math.min(...x0), Math.min(...y0)];
    return { x, y, w: Math.max(...x1) - x, h: Math.max(...y1) - y };
  }

  svg() {
    const lanes = [...this.wide].map(([bg, ds]) => this.laneMarkup(bg, ds.join(""))).join("");
    return [
      el("g", { class: "dg-back" }, this.back.join("")),
      el("path", { class: "dg-wire", d: this.thin.join("") }),
      lanes,
      el("path", { class: "dg-arrow", d: this.heads.join("") }),
      el("g", { class: "dg-front" }, this.front.join("")),
      el("path", { class: "dg-leader", d: this.leaders.join("") }),
      el("g", { class: "dg-over" }, this.over.join("")),
    ].join("");
  }

  /** A widened wire: strokes of falling widths, wire and background in turn, so
      that `lanes` thin lines run side by side along any path. */
  laneMarkup(bg, d) {
    const total = (2 * this.lanes - 1) * LANE;
    let s = "";
    for (let k = 0; k < this.lanes; k++) {
      const cls = k % 2 ? `dg-lane-gap bg-${bg}` : "dg-lane";
      s += el("path", { class: cls, d, style: `stroke-width:${r1(total - 2 * LANE * k)}px` });
    }
    return s;
  }
}

// ---------------------------------------------------------------- wires

/** A wire up from `y0` to `y1` (above it), with an arrowhead at its end. */
function rise(ink, x, y0, y1, wide, bg, arrow) {
  if (y0 - y1 < 0.5) return;
  ink.line(path().M(x, y0).V(arrow && ink.widens(wide) ? y1 + TIP : y1), wide, bg);
  if (arrow) ink.arrow(x, y1, "up", wide);
}

/** The start of a wire that leaves the spine at (x, y) to the right: it curves
    off the wire coming up, as a railway switch does. */
function switchOut(x, y) {
  return path().M(x, y + G.corner).Q(x, y, x + G.corner, y);
}

/** From the fork at (x, y) right to a branch at bx, and up to its bottom. */
function forkTo(ink, x, y, bx, bottom, bg, arrow) {
  if (Math.abs(bx - x) < 0.5) return rise(ink, x, y, bottom, false, bg, arrow);
  const c = Math.min(G.corner, (y - bottom) / 2);
  ink.line(switchOut(x, y).H(bx - c).Q(bx, y, bx, y - c).V(bottom), false, bg);
  if (arrow) ink.arrow(bx, bottom, "up");
}

/** From the top of a branch at bx up to the merge's height, and left into it. */
function joinTo(ink, bx, top, cy, edge, bg) {
  const c = Math.min(G.corner, (top - cy) / 2);
  ink.line(path().M(bx, top).V(cy + c).Q(bx, cy, bx - c, cy).H(edge), false, bg);
}

/** A residual's skip: off the spine at its fork, up its lane, into its ⊕. */
function skip(ink, f, x, bottom, cy, bg) {
  const lane = x + f.skipX;
  const c = G.corner;
  const edge = x + G.circle;
  const end = ink.widens(f.wideIn) ? edge + TIP : edge;
  const d = switchOut(x, bottom).H(lane - c).Q(lane, bottom, lane, bottom - c);
  ink.line(d.V(cy + c).Q(lane, cy, lane - c, cy).H(end), f.wideIn, bg);
  ink.arrow(edge, cy, "left", f.wideIn);
}

function mergeMark(ink, node, x, cy, op) {
  const a = G.circle * (op === "prod" ? 0.42 : 0.55);
  const cross =
    op === "prod"
      ? path().M(x - a, cy - a).L(x + a, cy + a).M(x + a, cy - a).L(x - a, cy + a)
      : path().M(x - a, cy).H(x + a).M(x, cy - a).V(cy + a);
  const inner = el("circle", { cx: x, cy, r: G.circle }) + el("path", { d: cross });
  ink.front.push(el("g", { class: "dg-merge dg-hit", "data-id": node.id }, inner));
  ink.hit(node, x - G.circle, cy - G.circle, 2 * G.circle, 2 * G.circle);
}

function takesArrow(f) {
  if (f.t === "seq") return takesArrow(f.items[0]);
  return f.t === "box" || f.t === "card" || f.t === "pill";
}

// ---------------------------------------------------------------- frames

const DRAW = {
  box: drawBox,
  card: drawCard,
  seq: drawSeq,
  cols: drawCols,
  res: drawRes,
  par: drawPar,
  cont: drawCont,
  tap: drawTap,
  ref: drawRef,
  mark: drawMark,
  pill: drawPill,
  wire: () => {},
};

/** Draws frame `f` with its spine at `x` and its top at `y`. `ctx.bg` is what it
    is drawn on, `ctx.tail` whether nothing follows it. */
export function draw(f, x, y, ink, ctx = { bg: "canvas", tail: true }) {
  DRAW[f.t](f, x, y, ink, ctx);
}

function drawSeq(f, x, y, ink, ctx) {
  f.items.forEach((item, i) => {
    const top = y + item.dy;
    const next = f.items[i + 1];
    draw(item, x + item.dx, top, ink, { ...ctx, tail: ctx.tail && !next });
    if (next?.fed) rise(ink, x, top, y + next.dy + next.h, item.wideOut, ctx.bg, takesArrow(next));
  });
}

function drawCols(f, x, y, ink, ctx) {
  for (const seg of f.segs) draw(seg, x + seg.dx, y + seg.dy, ink, { ...ctx, tail: true });
  for (const link of f.links) {
    const [x1, x2, ly] = [x + link.x1, x + link.x2, y + link.y];
    ink.line(path().M(x1, ly).H(ink.widens(link.wide) ? x2 - TIP : x2), link.wide, ctx.bg);
    ink.arrow(x2, ly, "right", link.wide);
  }
}

function drawRes(f, x, y, ink, ctx) {
  const { body } = f;
  const cy = y + G.circle;
  const top = y + body.dy;
  const bottom = y + f.h;
  draw(body, x, top, ink, { ...ctx, tail: false });
  rise(ink, x, bottom, top + body.h, false, ctx.bg, takesArrow(body));
  rise(ink, x, top, cy + G.circle, false, ctx.bg, true);
  skip(ink, f, x, bottom, cy, ctx.bg);
  mergeMark(ink, f.node, x, cy, "sum");
  ink.extent(x - f.l, y, x + f.r, bottom);
}

function drawPar(f, x, y, ink, ctx) {
  const cy = y + G.circle;
  const bottom = y + f.h;
  f.cols.forEach((col, i) => {
    const cx = x + col.dx;
    const top = y + col.dy;
    draw(col, cx, top, ink, { ...ctx, tail: false });
    if (col.fed) forkTo(ink, x, bottom, cx, top + col.h, ctx.bg, takesArrow(col));
    if (i === 0) rise(ink, x, top, cy + G.circle, false, ctx.bg, true);
    else joinTo(ink, cx, top, cy, x + G.circle, ctx.bg);
  });
  if (f.cols.length > 1) ink.arrow(x + G.circle, cy, "left");
  mergeMark(ink, f.node, x, cy, f.op);
  ink.extent(x - f.l, y, x + f.r, bottom);
}

function drawCont(f, x, y, ink, ctx) {
  const [x0, w] = [x - f.l, f.l + f.r];
  const fill = f.depth % 2 ? "canvas" : "container";
  const rect = { class: `dg-cont dg-fill-${fill}`, x: x0, y, width: w, height: f.h, rx: G.cont.radius };
  ink.back.push(el("rect", { ...rect, "data-id": f.node.id }));
  ink.hit(f.node, x0, y, w, f.h);
  if (f.repeat) ink.hit(f.repeat, x0, y, w, f.h);
  if (f.caption) ink.back.push(caption(f, x0 + 12, y + G.cont.caption / 2 + 1));
  if (f.times > 1) ink.back.push(badge(f, x0 + f.padL / 2, y + f.h / 2));
  draw(f.body, x, y + f.padT, ink, { bg: fill, tail: ctx.tail });
  if (f.fed) rise(ink, x, y + f.h, y + f.h - f.padB, f.wideIn, fill, takesArrow(f.body));
  if (!ctx.tail) rise(ink, x, y + f.padT, y, f.body.wideOut, fill, false);
  ink.extent(x0, y, x + f.r, y + f.h);
}

function chevron(x, cy, open) {
  const [a, b] = [3.5, 1.8];
  const d = open
    ? path().M(x - a, cy - b).L(x, cy + b).L(x + a, cy - b)
    : path().M(x - b, cy - a).L(x + b, cy).L(x - b, cy + a);
  return el("path", { class: "dg-chevron", d });
}

/** A block's name at its container's top left; its chevron folds it. */
function caption(f, x, cy) {
  const w = 16 + measure(f.caption, FONT.caption);
  const hit = el("rect", { class: "dg-caption-hit dg-ui", x: x - 6, y: cy - 10, width: w + 10, height: 20, rx: 5 });
  const name = text("dg-caption-text", x + 11, cy, f.caption);
  const note = f.node.shared ? text("dg-caption-note", x + w + 6, cy, "shared") : "";
  const attrs = { class: "dg-caption", "data-toggle": f.key, "data-id": f.node.id };
  return el("g", attrs, hit + el("g", { class: "dg-ui" }, chevron(x + 3, cy, true)) + name + note);
}

/** A repeat's count in its container's left margin; double-click edits it. */
function badge(f, x, cy) {
  const repeat = f.repeat?.editable ? f.repeat.id : null;
  return text("dg-badge", x, cy, `×${f.times}`, { "data-id": (f.repeat ?? f.node).id, "data-repeat": repeat });
}

function drawBox(f, x, y, ink) {
  const n = f.node;
  const x0 = x - f.w / 2;
  let s = el("rect", { class: "dg-shape", x: x0, y, width: f.w, height: f.h, rx: G.box.radius });
  s += text("dg-title", x, y + G.box.padY + G.box.title / 2, f.text.title);
  f.text.rows.forEach((row, i) => {
    s += metaRow(row, x, y + G.box.padY + G.box.title + (i + 0.5) * G.box.meta);
  });
  if (n.internals) s += zoomIcon(n, x0 + f.w - 15, y + 15, ink.zooms.has(n.id));
  const cls = ["dg-box", "dg-hit", `cat-${n.category ?? "other"}`, n.inserted && "dg-inserted"];
  ink.front.push(el("g", { class: cls.filter(Boolean).join(" "), "data-id": n.id }, s));
  ink.hit(n, x0, y, f.w, f.h);
  ink.extent(x - f.l, y, x + f.r, y + f.h);
  drawPorts(f, x0, y, ink);
}

/** The definition's name as a tag, then the chips, centered on `x`. */
function metaRow(row, x, cy) {
  let sx = x - row.w / 2;
  let s = "";
  if (row.label) {
    const h = G.label.h;
    s += el("rect", { class: "dg-label-bg", x: sx, y: cy - h / 2, width: row.labelW, height: h, rx: 4 });
    s += text("dg-label", sx + row.labelW / 2, cy, row.label);
    sx += row.labelW + G.label.gap;
  }
  return row.text ? s + text("dg-chips", sx, cy, row.text) : s;
}

function zoomIcon(node, x, cy, open) {
  const a = 4;
  const d = path()
    .M(x - a, cy + a).L(x + a, cy - a)
    .M(x + 0.5, cy - a).H(x + a).V(cy - 0.5)
    .M(x - a, cy + 0.5).V(cy + a).H(x - 0.5); // prettier-ignore
  const hit = el("rect", { class: "dg-zoom-hit", x: x - 8, y: cy - 8, width: 16, height: 16, rx: 4 });
  const attrs = { class: `dg-zoom dg-ui${open ? " dg-open" : ""}`, "data-zoom": node.id };
  return el("g", attrs, el("title", {}, "Inside this layer") + hit + el("path", { d }));
}

/** A box's ports, as tags left of it: `kv=@kv` exports, `kv=$kv` imports. */
function drawPorts(f, x0, y, ink) {
  const mid = y + f.h / 2;
  const step = G.tag.h + 4;
  f.ports.forEach((p, i) => portTag(ink, f.node, p, x0, mid + (i - (f.ports.length - 1) / 2) * step));
}

function portTag(ink, node, p, boxX, cy) {
  const x1 = boxX - G.ports;
  ink.line(path().M(x1, cy).H(boxX), false, "canvas");
  if (p.mode === "write") ink.arrow(x1, cy, "left");
  else ink.arrow(boxX, cy, "right");
  ink.front.push(tagPill(node, p.text, p.tap, x1 - p.w, cy, p.w));
}

function tagPill(node, content, tap, x0, cy, w) {
  const h = G.tag.h;
  const rect = el("rect", { x: x0, y: cy - h / 2, width: w, height: h, rx: h / 2 });
  return el("g", { class: "dg-tag dg-hit", "data-id": node.id, "data-tap": tap }, rect + text(null, x0 + w / 2, cy, content));
}

/** A folded block: its name, how many in a row, and what it holds. */
function drawCard(f, x, y, ink) {
  const n = f.node;
  const t = f.text;
  const x0 = x - f.w / 2;
  const rx = G.box.radius;
  let s = f.times > 1 ? el("rect", { class: "dg-card-stack", x: x0 + 4, y: y - 4, width: f.w, height: f.h, rx }) : "";
  s += el("rect", { class: "dg-shape", x: x0, y, width: f.w, height: f.h, rx });
  s += cardHead(f, x, y + G.box.padY + G.box.title / 2);
  if (t.about) s += text("dg-chips dg-mid", x, y + G.box.padY + G.box.title + G.box.meta / 2, t.about);
  ink.front.push(el("g", { class: "dg-card dg-hit", "data-id": n.id, "data-key": f.key }, s));
  ink.hit(n, x0, y, f.w, f.h);
  if (f.repeat) ink.hit(f.repeat, x0, y, f.w, f.h);
  ink.extent(x - f.l, y - 4, x0 + f.w + 4, y + f.h);
  drawPorts(f, x0, y, ink);
}

function cardHead(f, x, cy) {
  const t = f.text;
  const labelW = measure(t.label, FONT.card);
  const sx = x - (14 + labelW + (t.count ? 6 + measure(t.count, FONT.times) : 0)) / 2;
  const hit = el("rect", { class: "dg-caption-hit", x: sx - 6, y: cy - 9, width: 18, height: 18, rx: 4 });
  let s = el("g", { class: "dg-ui dg-card-toggle", "data-toggle": f.key }, hit + chevron(sx + 3, cy, false));
  s += text("dg-card-label", sx + 14, cy, t.label);
  const repeat = f.repeat?.editable ? f.repeat.id : null;
  return t.count ? s + text("dg-card-count", sx + 20 + labelW, cy, t.count, { "data-repeat": repeat }) : s;
}

function drawTap(f, x, y, ink, ctx) {
  const cy = y + f.h / 2;
  const x1 = x - G.tag.gap;
  ink.line(path().M(x, y + f.h).V(y), f.wideIn, ctx.bg);
  ink.line(path().M(x1, cy).H(x), false, ctx.bg);
  ink.front.push(tagPill(f.node, f.text, f.node.name, x1 - f.w, cy, f.w));
  ink.front.push(el("circle", { class: "dg-tap-dot", cx: x, cy, r: 2.6 }));
  ink.hit(f.node, x1 - f.w, y, f.w, f.h);
  ink.extent(x - f.l, y, x + f.r, y + f.h);
}

function drawRef(f, x, y, ink) {
  ink.front.push(tagPill(f.node, f.text, f.node.name, x - f.w / 2, y + f.h / 2, f.w));
  ink.hit(f.node, x - f.w / 2, y, f.w, f.h);
  ink.extent(x - f.l, y, x + f.r, y + f.h);
}

function drawMark(f, x, y, ink) {
  const rect = el("rect", { x: x - f.w / 2, y, width: f.w, height: f.h, rx: f.h / 2 });
  ink.front.push(el("g", { class: "dg-mark dg-hit", "data-id": f.node.id }, rect + text(null, x, y + f.h / 2, f.text)));
  ink.hit(f.node, x - f.w / 2, y, f.w, f.h);
  ink.extent(x - f.l, y, x + f.r, y + f.h);
}

function drawPill(f, x, y, ink) {
  const rect = el("rect", { x: x - f.w / 2, y, width: f.w, height: f.h, rx: f.h / 2 });
  ink.front.push(el("g", { class: "dg-io" }, rect + text(null, x, y + f.h / 2, f.text)));
  ink.extent(x - f.l, y, x + f.r, y + f.h);
}

// ---------------------------------------------------------------- panels

const PANEL = { pad: 18, title: 30, stub: 18, gap: 56 };

export function panelSize(frame) {
  const w = Math.max(frame.l + frame.r, measure("Inside", FONT.title) + 60) + 2 * PANEL.pad;
  return { w, h: PANEL.title + 2 * PANEL.stub + frame.h + PANEL.pad };
}

/** Where a box's zoom goes: beside it, right of whatever is level with it. */
export function placePanel(box, size, extents) {
  const y = box.y + box.h / 2 - size.h / 2;
  const level = extents.filter(([, y0, , y1]) => y1 > y - 8 && y0 < y + size.h + 8);
  const right = Math.max(box.x + box.w, ...level.map((e) => e[2]));
  return { x: right + PANEL.gap, y };
}

/** Raschka's zoom: the inside of a layer in a card, with leader lines to it. */
export function drawPanel(frame, node, box, at, ink) {
  const { w, h } = panelSize(frame);
  const { x, y } = at;
  const right = box.x + box.w;
  ink.leaders.push(path().M(right, box.y).L(x, y).M(right, box.y + box.h).L(x, y + h));
  const card = el("rect", { class: "dg-panel", x, y, width: w, height: h, rx: G.cont.radius });
  const title = text("dg-panel-title", x + PANEL.pad, y + PANEL.title / 2 + 3, `Inside: ${node.title ?? node.type}`);
  const sub = panelBody(frame, x + PANEL.pad + (w - 2 * PANEL.pad - frame.r + frame.l) / 2, y, h, ink.zooms);
  const s = card + title + panelClose(node, x + w - 26, y + 8) + sub.svg();
  ink.over.push(el("g", { class: "dg-zoom-panel", "data-panel": node.id }, s));
  ink.extent(x, y, x + w, y + h);
  for (const [id, n] of sub.nodes) ink.inside.set(id, { node: n, box: sub.rects.get(id) });
}

function panelBody(frame, spine, y, h, zooms) {
  const sub = new Ink(0, zooms);
  const top = y + PANEL.title + PANEL.stub;
  draw(frame, spine, top, sub, { bg: "surface", tail: false });
  rise(sub, spine, y + h - PANEL.pad / 2, top + frame.h, false, "surface", takesArrow(frame));
  rise(sub, spine, top, y + PANEL.title, false, "surface", false);
  return sub;
}

function panelClose(node, x, y) {
  const hit = el("rect", { x, y, width: 18, height: 18, rx: 4 });
  const cross = el("path", { d: path().M(x + 5, y + 5).L(x + 13, y + 13).M(x + 13, y + 5).L(x + 5, y + 13) });
  return el("g", { class: "dg-ui dg-panel-close", "data-zoom": node.id }, el("title", {}, "Close") + hit + cross);
}

// ---------------------------------------------------------------- header

const HEADER = { pad: 16, name: 34, row: 21, gap: 18 };
const num = (v) => (typeof v === "number" ? v.toLocaleString("en-US") : v);
const SHELL = new Set(["params", "params_text", "flops_per_token", "context"]); // what the shell sets

/** 1.6e10 -> "16G", as a count of FLOPs is read. */
function si(v) {
  if (typeof v !== "number") return v;
  const [k, unit] = [[1e12, "T"], [1e9, "G"], [1e6, "M"], [1e3, "k"]].find(([k]) => v >= k) ?? [1, ""];
  return `${(v / k).toPrecision(3).replace(/\.?0+$/, "")}${unit}`;
}

/** The header's rows: [name, value as text, whether it is unknown yet]. The
    shell's facts are params (or params_text), context and flops_per_token. */
function headerRows(model, facts) {
  const f = model.facts ?? {};
  const rows = [
    ["Parameters", facts.params_text ?? si(facts.params)],
    ["d_model", num(f.d_model)],
    ["Vocabulary", num(f.vocab_size)],
    ["Layers", num(f.layers)],
    ["Residual", f.residual ?? "x + f(x)"],
  ];
  if (facts.context != null) rows.splice(1, 0, ["Context", num(facts.context)]);  // a run's seq_len
  if (facts.flops_per_token != null) rows.splice(1, 0, ["FLOPs / token", si(facts.flops_per_token)]);
  for (const [k, v] of Object.entries(facts)) if (!SHELL.has(k) && v != null) rows.push([k, v]);
  return rows.map(([k, v]) => (v == null || v === "" ? [k, "—", true] : [k, String(v), false]));
}

export function headerSize(model, facts) {
  const rows = headerRows(model, facts);
  const keyW = Math.max(...rows.map(([k]) => measure(k, FONT.chip)));
  const valW = Math.max(...rows.map(([, v]) => measure(v, FONT.tag)));
  const nameW = measure(model.name ?? "model", FONT.name);
  const w = Math.max(nameW, keyW + HEADER.gap + valW) + 2 * HEADER.pad;
  return { w, h: HEADER.name + rows.length * HEADER.row + HEADER.pad - 4, keyW, rows };
}

/** The card at the figure's top left: the model's name and its facts. */
export function drawHeader(model, facts, x, y, ink) {
  const { w, h, keyW, rows } = headerSize(model, facts);
  let s = el("rect", { class: "dg-header-card", x, y, width: w, height: h, rx: G.cont.radius });
  s += text("dg-header-name", x + HEADER.pad, y + HEADER.pad + 6, model.name ?? "model");
  rows.forEach(([k, v, missing], i) => {
    const cy = y + HEADER.name + (i + 0.5) * HEADER.row + 2;
    s += text("dg-header-key", x + HEADER.pad, cy, k);
    s += text(missing ? "dg-header-value dg-missing" : "dg-header-value", x + HEADER.pad + keyW + HEADER.gap, cy, v);
  });
  ink.over.push(el("g", { class: "dg-header" }, s));
  ink.extent(x, y, x + w, y + h);
}
