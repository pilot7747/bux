/* Drawing of the model diagram: sized frames (diagram-layout.js) -> SVG markup.

   Everything is drawn into an Ink: wires go into a few long paths, boxes and marks
   into layers of markup, and what can be selected into `rects`, so that rings,
   toolbars and panels can find it without touching the DOM. */

import { FONT, G, measure } from "./diagram-layout.js";

const r1 = (v) => Math.round(v * 10) / 10;
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
export const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ESC[c]);

export function el(tag, attrs, inner) {
  let s = `<${tag}`;
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    s += ` ${k}="${typeof v === "number" ? r1(v) : esc(v)}"`;
  }
  return inner === undefined ? `${s}/>` : `${s}>${inner}</${tag}>`;
}

const ARROW = { up: [0, 1], left: [1, 0], right: [-1, 0] };
const LANE = 1.2; // the width of a lane of a widened wire, and of the gap between two

export class Ink {
  constructor(lanes, zooms) {
    this.lanes = lanes;
    this.zooms = zooms; // ids of the boxes whose internals are open
    this.back = [];
    this.front = [];
    this.over = [];
    this.thin = [];
    this.wide = new Map(); // background -> path data of widened wires on it
    this.heads = [];
    this.rects = new Map(); // id -> {x, y, w, h} of what can be selected
    this.nodes = new Map(); // id -> the node drawn there
    this.extents = []; // [x0, y0, x1, y1] of what is drawn
  }

  line(d, wide, bg) {
    if (!wide || !this.lanes) return this.thin.push(d);
    if (!this.wide.has(bg)) this.wide.set(bg, []);
    this.wide.get(bg).push(d);
  }

  arrow(x, y, dir, wide = false) {
    const [ux, uy] = ARROW[dir];
    const len = wide && this.lanes ? 8 : 6;
    const half = wide && this.lanes ? 5.5 : 3.5;
    const bx = x + ux * len;
    const by = y + uy * len;
    const [px, py] = [uy * half, ux * half];
    this.heads.push(`M${r1(bx - px)} ${r1(by - py)}L${r1(x)} ${r1(y)}L${r1(bx + px)} ${r1(by + py)}Z`);
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
    return { x: Math.min(...x0), y: Math.min(...y0), w: Math.max(...x1) - Math.min(...x0), h: Math.max(...y1) - Math.min(...y0) };
  }

  svg() {
    const lanes = [...this.wide].map(([bg, ds]) => this.laneMarkup(bg, ds.join(""))).join("");
    return [
      el("g", { class: "dg-back" }, this.back.join("")),
      el("path", { class: "dg-wire", d: this.thin.join("") }),
      lanes,
      el("path", { class: "dg-arrow", d: this.heads.join("") }),
      el("g", { class: "dg-front" }, this.front.join("")),
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
  const end = arrow && wide && ink.lanes ? y1 + 7 : y1;
  ink.line(`M${r1(x)} ${r1(y0)}V${r1(end)}`, wide, bg);
  if (arrow) ink.arrow(x, y1, "up", wide);
}

/** From the fork at (x, y) right to a branch at bx, and up to its bottom. */
function forkTo(ink, x, y, bx, bottom, bg, arrow) {
  if (Math.abs(bx - x) < 0.5) return rise(ink, x, y, bottom, false, bg, arrow);
  const c = Math.min(G.corner, (y - bottom) / 2);
  ink.line(`M${r1(x)} ${r1(y)}H${r1(bx - c)}Q${r1(bx)} ${r1(y)} ${r1(bx)} ${r1(y - c)}V${r1(bottom)}`, false, bg);
  if (arrow) ink.arrow(bx, bottom, "up");
}

/** From the top of a branch at bx up to the merge's height, and left into it. */
function joinTo(ink, bx, top, cy, edge, bg, wide = false) {
  const c = Math.min(G.corner, (top - cy) / 2);
  const end = wide && ink.lanes ? edge + 7 : edge;
  ink.line(`M${r1(bx)} ${r1(top)}V${r1(cy + c)}Q${r1(bx)} ${r1(cy)} ${r1(bx - c)} ${r1(cy)}H${r1(end)}`, wide, bg);
}

function mergeMark(ink, node, x, cy, op) {
  const a = G.circle * (op === "prod" ? 0.42 : 0.55);
  const cross =
    op === "prod"
      ? `M${r1(x - a)} ${r1(cy - a)}L${r1(x + a)} ${r1(cy + a)}M${r1(x + a)} ${r1(cy - a)}L${r1(x - a)} ${r1(cy + a)}`
      : `M${r1(x - a)} ${r1(cy)}H${r1(x + a)}M${r1(x)} ${r1(cy - a)}V${r1(cy + a)}`;
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
    const end = link.wide && ink.lanes ? x2 - 7 : x2;
    ink.line(`M${r1(x1)} ${r1(ly)}H${r1(end)}`, link.wide, ctx.bg);
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
  const lane = x + f.skipX;
  const c = G.corner;
  const end = f.wideIn && ink.lanes ? x + G.circle + 7 : x + G.circle;
  const d = `M${r1(x)} ${r1(bottom)}H${r1(lane - c)}Q${r1(lane)} ${r1(bottom)} ${r1(lane)} ${r1(bottom - c)}V${r1(cy + c)}Q${r1(lane)} ${r1(cy)} ${r1(lane - c)} ${r1(cy)}H${r1(end)}`;
  ink.line(d, f.wideIn, ctx.bg);
  ink.arrow(x + G.circle, cy, "left", f.wideIn);
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
  const x0 = x - f.l;
  const fill = f.depth % 2 ? "canvas" : "container";
  const id = f.node.id;
  const rect = { class: `dg-cont dg-fill-${fill}`, x: x0, y, width: f.l + f.r, height: f.h, rx: G.cont.radius };
  ink.back.push(el("rect", { ...rect, "data-id": id }));
  ink.hit(f.node, x0, y, f.l + f.r, f.h);
  if (f.repeat) ink.hit(f.repeat, x0, y, f.l + f.r, f.h);
  if (f.caption) ink.back.push(caption(f, x0 + 12, y + G.cont.caption / 2 + 1));
  if (f.times > 1) ink.back.push(badge(f, x0 + f.padL / 2, y + f.h / 2));
  const inner = { bg: fill, tail: ctx.tail };
  draw(f.body, x, y + f.padT, ink, inner);
  if (f.fed) rise(ink, x, y + f.h, y + f.h - f.padB, f.wideIn, fill, takesArrow(f.body));
  if (!ctx.tail) rise(ink, x, y + f.padT, y, f.body.wideOut, fill, false);
  ink.extent(x0, y, x + f.r, y + f.h);
}

function chevron(x, cy, open) {
  const d = open
    ? `M${r1(x - 3.5)} ${r1(cy - 1.8)}L${r1(x)} ${r1(cy + 1.8)}L${r1(x + 3.5)} ${r1(cy - 1.8)}`
    : `M${r1(x - 1.8)} ${r1(cy - 3.5)}L${r1(x + 1.8)} ${r1(cy)}L${r1(x - 1.8)} ${r1(cy + 3.5)}`;
  return el("path", { class: "dg-chevron", d });
}

function caption(f, x, cy) {
  const w = 16 + measure(f.caption, FONT.caption);
  const hit = el("rect", { class: "dg-caption-hit", x: x - 6, y: cy - 10, width: w + 10, height: 20, rx: 5 });
  const text = el("text", { class: "dg-caption-text", x: x + 11, y: cy, dy: ".35em" }, esc(f.caption));
  const attrs = { class: "dg-caption dg-ui", "data-toggle": f.key, "data-id": f.node.id };
  return el("g", attrs, hit + chevron(x + 3, cy, true) + text);
}

function badge(f, x, cy) {
  const repeat = f.repeat?.editable ? f.repeat.id : null;
  const attrs = { class: "dg-badge", x, y: cy, dy: ".35em", "data-id": (f.repeat ?? f.node).id, "data-repeat": repeat };
  return el("text", attrs, `×${f.times}`);
}

function drawBox(f, x, y, ink) {
  const n = f.node;
  const t = f.text;
  const x0 = x - f.w / 2;
  let s = el("rect", { class: "dg-shape", x: x0, y, width: f.w, height: f.h, rx: G.box.radius });
  s += el("text", { class: "dg-title", x, y: y + G.box.padY + G.box.title / 2, dy: ".35em" }, esc(t.title));
  t.rows.forEach((row, i) => {
    s += metaRow(row, x, y + G.box.padY + G.box.title + (i + 0.5) * G.box.meta);
  });
  if (n.internals) s += zoomIcon(n, x0 + f.w - 15, y + 15, ink.zooms.has(n.id));
  const cls = ["dg-box", "dg-hit", `cat-${n.category ?? "other"}`, n.inserted && "dg-inserted"];
  ink.front.push(el("g", { class: cls.filter(Boolean).join(" "), "data-id": n.id }, s));
  ink.hit(n, x0, y, f.w, f.h);
  ink.extent(x - f.l, y, x + f.r, y + f.h);
  const mid = y + f.h / 2;
  f.ports.forEach((p, i) => portTag(ink, n, p, x0, mid + (i - (f.ports.length - 1) / 2) * (G.tag.h + 4)));
}

function metaRow(row, x, cy) {
  let sx = x - row.w / 2;
  let s = "";
  if (row.label) {
    const h = G.label.h;
    s += el("rect", { class: "dg-label-bg", x: sx, y: cy - h / 2, width: row.labelW, height: h, rx: 4 });
    s += el("text", { class: "dg-label", x: sx + row.labelW / 2, y: cy, dy: ".35em" }, esc(row.label));
    sx += row.labelW + G.label.gap;
  }
  if (row.text) s += el("text", { class: "dg-chips", x: sx, y: cy, dy: ".35em" }, esc(row.text));
  return s;
}

function zoomIcon(node, x, cy, open) {
  const a = 4;
  const d =
    `M${r1(x - a)} ${r1(cy + a)}L${r1(x + a)} ${r1(cy - a)}` +
    `M${r1(x + 0.5)} ${r1(cy - a)}H${r1(x + a)}V${r1(cy - 0.5)}` +
    `M${r1(x - a)} ${r1(cy + 0.5)}V${r1(cy + a)}H${r1(x - 0.5)}`;
  const hit = el("rect", { class: "dg-zoom-hit", x: x - 8, y: cy - 8, width: 16, height: 16, rx: 4 });
  const attrs = { class: `dg-zoom dg-ui${open ? " dg-open" : ""}`, "data-zoom": node.id };
  return el("g", attrs, el("title", {}, "Inside this layer") + hit + el("path", { d }));
}

function portTag(ink, node, p, boxX, cy) {
  const x1 = boxX - G.ports;
  const x0 = x1 - p.w;
  ink.line(`M${r1(x1)} ${r1(cy)}H${r1(boxX)}`, false, "canvas");
  if (p.mode === "write") ink.arrow(x1, cy, "left");
  else ink.arrow(boxX, cy, "right");
  ink.front.push(tagPill(node, p.text, p.tap, x0, cy, p.w));
}

function tagPill(node, text, tap, x0, cy, w) {
  const h = G.tag.h;
  const rect = el("rect", { x: x0, y: cy - h / 2, width: w, height: h, rx: h / 2 });
  const label = el("text", { x: x0 + w / 2, y: cy, dy: ".35em" }, esc(text));
  return el("g", { class: "dg-tag dg-hit", "data-id": node.id, "data-tap": tap }, rect + label);
}

function drawCard(f, x, y, ink) {
  const n = f.node;
  const t = f.text;
  const x0 = x - f.w / 2;
  const rx = G.box.radius;
  let s = f.times > 1 ? el("rect", { class: "dg-card-stack", x: x0 + 4, y: y - 4, width: f.w, height: f.h, rx }) : "";
  s += el("rect", { class: "dg-shape", x: x0, y, width: f.w, height: f.h, rx });
  const cy = y + G.box.padY + G.box.title / 2;
  const labelW = measure(t.label, FONT.card);
  const headW = 14 + labelW + (t.count ? 6 + measure(t.count, FONT.times) : 0);
  const sx = x - headW / 2;
  const toggle = el("rect", { class: "dg-caption-hit", x: sx - 6, y: cy - 9, width: 18, height: 18, rx: 4 });
  s += el("g", { class: "dg-ui dg-card-toggle", "data-toggle": f.key }, toggle + chevron(sx + 3, cy, false));
  s += el("text", { class: "dg-card-label", x: sx + 14, y: cy, dy: ".35em" }, esc(t.label));
  if (t.count) s += countText(f, sx + 20 + labelW, cy, t.count);
  if (t.about) s += el("text", { class: "dg-chips dg-mid", x, y: cy + G.box.title / 2 + G.box.meta / 2, dy: ".35em" }, esc(t.about));
  ink.front.push(el("g", { class: "dg-card dg-hit", "data-id": n.id, "data-key": f.key }, s));
  ink.hit(n, x0, y, f.w, f.h);
  if (f.repeat) ink.hit(f.repeat, x0, y, f.w, f.h);
  ink.extent(x0, y - 4, x0 + f.w + 4, y + f.h);
}

function countText(f, x, cy, text) {
  const repeat = f.repeat?.editable ? f.repeat.id : null;
  return el("text", { class: "dg-card-count", x, y: cy, dy: ".35em", "data-repeat": repeat }, text);
}

function drawTap(f, x, y, ink, ctx) {
  const cy = y + f.h / 2;
  const x1 = x - G.tag.gap;
  ink.line(`M${r1(x)} ${r1(y + f.h)}V${r1(y)}`, f.wideIn, ctx.bg);
  ink.line(`M${r1(x1)} ${r1(cy)}H${r1(x)}`, false, ctx.bg);
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
  const text = el("text", { x, y: y + f.h / 2, dy: ".35em" }, esc(f.text));
  ink.front.push(el("g", { class: "dg-mark dg-hit", "data-id": f.node.id }, rect + text));
  ink.hit(f.node, x - f.w / 2, y, f.w, f.h);
  ink.extent(x - f.l, y, x + f.r, y + f.h);
}

function drawPill(f, x, y, ink) {
  const rect = el("rect", { x: x - f.w / 2, y, width: f.w, height: f.h, rx: f.h / 2 });
  const text = el("text", { x, y: y + f.h / 2, dy: ".35em" }, esc(f.text));
  ink.front.push(el("g", { class: "dg-io" }, rect + text));
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
  const leaders = `M${r1(box.x + box.w)} ${r1(box.y)}L${r1(x)} ${r1(y)}M${r1(box.x + box.w)} ${r1(box.y + box.h)}L${r1(x)} ${r1(y + h)}`;
  ink.over.push(el("path", { class: "dg-leader", d: leaders }));
  const sub = new Ink(0, new Set());
  const spine = x + PANEL.pad + frame.l + (w - 2 * PANEL.pad - frame.l - frame.r) / 2;
  const top = y + PANEL.title + PANEL.stub;
  draw(frame, spine, top, sub, { bg: "surface", tail: false });
  rise(sub, spine, y + h - PANEL.pad / 2, top + frame.h, false, "surface", takesArrow(frame));
  rise(sub, spine, top, y + PANEL.title, false, "surface", false);
  const card = el("rect", { class: "dg-panel", x, y, width: w, height: h, rx: G.cont.radius });
  const title = el("text", { class: "dg-panel-title", x: x + PANEL.pad, y: y + PANEL.title / 2 + 3, dy: ".35em" }, esc(`Inside: ${node.title ?? node.type}`));
  const close = el("g", { class: "dg-ui dg-panel-close", "data-zoom": node.id }, el("rect", { x: x + w - 26, y: y + 8, width: 18, height: 18, rx: 4 }) + el("path", { d: `M${r1(x + w - 21)} ${r1(y + 13)}l8 8m0 -8l-8 8` }));
  ink.over.push(el("g", { class: "dg-zoom-panel", "data-panel": node.id }, card + title + close + sub.svg()));
  ink.extent(x, y, x + w, y + h);
}

// ---------------------------------------------------------------- header

const HEADER = { pad: 16, name: 34, row: 21, gap: 18 };
const num = (v) => (typeof v === "number" ? v.toLocaleString("en-US") : v);

function headerRows(model, facts) {
  const f = model.facts ?? {};
  const known = { params: "Parameters", context: "Context" };
  const rows = [
    ["Parameters", facts.params ?? null],
    ["Context", facts.context ?? null],
    ["d_model", num(f.d_model)],
    ["Vocabulary", f.vocab_size == null ? null : num(f.vocab_size)],
    ["Layers", num(f.layers)],
    ["Residual", f.residual ?? "x + f(x)"],
  ];
  for (const [k, v] of Object.entries(facts)) if (!(k in known)) rows.push([k, v]);
  return rows.map(([k, v]) => [k, v == null || v === "" ? "—" : String(v), v == null || v === ""]);
}

export function headerSize(model, facts) {
  const rows = headerRows(model, facts);
  const keyW = Math.max(...rows.map(([k]) => measure(k, FONT.chip)));
  const valW = Math.max(...rows.map(([, v]) => measure(v, FONT.tag)));
  const nameW = measure(model.name ?? "model", FONT.name);
  const w = Math.max(nameW, keyW + HEADER.gap + valW) + 2 * HEADER.pad;
  return { w, h: HEADER.name + rows.length * HEADER.row + HEADER.pad - 4, keyW, rows };
}

export function drawHeader(model, facts, x, y, ink) {
  const { w, h, keyW, rows } = headerSize(model, facts);
  let s = el("rect", { class: "dg-header-card", x, y, width: w, height: h, rx: G.cont.radius });
  s += el("text", { class: "dg-header-name", x: x + HEADER.pad, y: y + HEADER.pad + 6, dy: ".35em" }, esc(model.name ?? "model"));
  rows.forEach(([k, v, missing], i) => {
    const cy = y + HEADER.name + (i + 0.5) * HEADER.row + 2;
    s += el("text", { class: "dg-header-key", x: x + HEADER.pad, y: cy, dy: ".35em" }, esc(k));
    const cls = missing ? "dg-header-value dg-missing" : "dg-header-value";
    s += el("text", { class: cls, x: x + HEADER.pad + keyW + HEADER.gap, y: cy, dy: ".35em" }, esc(v));
  });
  ink.over.push(el("g", { class: "dg-header" }, s));
  ink.extent(x, y, x + w, y + h);
}
