/* The model diagram of bux studio: a Model drawn as Raschka draws an LLM, with
   pan, zoom, selection and edits as ops.

     const diagram = new Diagram(el, {onSelect, onAction, onHover, palette});
     diagram.render(view.model);

   Layout is diagram-layout.js, drawing diagram-draw.js, the overlays (toolbar,
   palette, controls) diagram-ui.js; this module holds the state and the events. */

import { Ink, draw, drawHeader, drawPanel, headerSize, panelSize, placePanel } from "./diagram-draw.js";
import { plan, planInternals, setFonts } from "./diagram-layout.js";
import { Controls, CountEditor, Palette, Toolbar } from "./diagram-ui.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const ZOOM = { min: 0.1, max: 4, fit: 1.15, pad: 40 };
const HOVER_GRACE = 350; // ms a toolbar stays after the pointer leaves its node
const EXPORT_PROPS = [
  "fill", "fill-opacity", "stroke", "stroke-width", "stroke-opacity", "stroke-dasharray",
  "stroke-linecap", "stroke-linejoin", "opacity", "font-family", "font-size", "font-weight",
  "letter-spacing", "text-anchor",
]; // prettier-ignore

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const noop = () => {};
const spanKey = (n) => (n?.written ? `w${n.written}` : n?.span ? `s${n.span}` : null);

export class Diagram {
  /**
   * @param {HTMLElement} el where to draw; the diagram fills it
   * @param {object} options onSelect(id|null), onAction(op), onHover(id|null),
   *   palette: [PaletteItem] offered by "insert" (also settable as `.palette`)
   */
  constructor(el, { onSelect, onAction, onHover, palette } = {}) {
    this.el = el;
    this.on = { select: onSelect ?? noop, action: onAction ?? noop, hover: onHover ?? noop };
    this.palette = palette ?? [];
    this.view = { x: 0, y: 0, k: 1 };
    this.folds = { mode: null, keys: new Map() }; // which blocks the user opened or closed
    this.zooms = new Set(); // boxes whose internals are open
    this.facts = {};
    this.model = null;
    this.selected = null;
    this.hovered = null;
    this.mount();
    this.listen();
  }

  // ---------------------------------------------------------------- the interface

  /** Draws `model`, keeping zoom, pan, selection and open blocks. */
  render(model) {
    const start = performance.now();
    this.model = model;
    this.plan = plan(model, this.folds);
    const ink = new Ink(this.plan.lanes, this.zooms);
    draw(this.plan.root, 0, 0, ink);
    this.drawPanels(ink);
    this.drawHeader(ink);
    this.content.innerHTML = ink.svg();
    this.ink = ink;
    if (!this.fitted) this.fit();
    this.fitted = true;
    this.paint();
    this.renderMs = performance.now() - start;
  }

  /** Highlights the node `id` (or what holds it, when folded); pans to it if
      it is off screen. Does not call onSelect. */
  select(id) {
    this.selected = id ?? null;
    this.paint();
    if (id) this.reveal(id);
  }

  setDim(dim) {
    this.el.classList.toggle("dg-dim", !!dim);
  }

  /** Fits the whole figure in view. */
  fit(animate = false) {
    if (!this.ink) return;
    const b = this.ink.bounds();
    const { width, height } = this.el.getBoundingClientRect();
    const k = clamp(Math.min((width - 2 * ZOOM.pad) / b.w, (height - 2 * ZOOM.pad) / b.h), ZOOM.min, ZOOM.fit);
    this.moveTo({ k, x: (width - b.w * k) / 2 - b.x * k, y: (height - b.h * k) / 2 - b.y * k }, animate);
  }

  /** A standalone SVG of the figure, styles inlined, without the editing marks. */
  exportSVG() {
    const b = this.ink.bounds();
    const pad = 24;
    const clone = this.content.cloneNode(true);
    inlineStyles(this.content, clone);
    for (const ui of clone.querySelectorAll(".dg-ui")) ui.remove();
    for (const e of [clone, ...clone.querySelectorAll("*")]) stripAttributes(e);
    const bg = getComputedStyle(this.el).getPropertyValue("--canvas").trim() || "#fff";
    const [x, y, w, h] = [b.x - pad, b.y - pad, b.w + 2 * pad, b.h + 2 * pad].map(Math.round);
    const rect = `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${bg}"/>`;
    return `<svg xmlns="${SVG_NS}" width="${w}" height="${h}" viewBox="${x} ${y} ${w} ${h}">${rect}${clone.innerHTML}</svg>`;
  }

  /** Facts for the header card beyond the model's own: {params: "8.03B",
      context: "8,192"}, or any other name -> value; null removes one. */
  setFacts(facts) {
    for (const [k, v] of Object.entries(facts)) {
      if (v == null) delete this.facts[k];
      else this.facts[k] = v;
    }
    if (this.model) this.render(this.model);
  }

  /** Opens or closes the block `key` ("D#0": its label, then which occurrence). */
  setOpen(key, open = !this.plan?.open.get(key)) {
    this.folds.keys.set(key, open);
    this.rerender();
  }

  expandAll() {
    this.folds = { mode: "open", keys: new Map() };
    this.rerender();
  }

  collapseAll() {
    this.folds = { mode: "closed", keys: new Map() };
    this.rerender();
  }

  /** Opens or closes the zoom of a layer's internals. */
  setInternals(id, open = !this.zooms.has(id)) {
    if (open) this.zooms.add(id);
    else this.zooms.delete(id);
    this.rerender();
  }

  /** Opens the palette to insert before or after `id`, as its toolbar does. */
  openPalette(id, where = "after") {
    this.ui.toolbar.show(id);
    const anchor = this.ui.toolbar.el.querySelector(".dg-btn") ?? this.el;
    this.ui.palette.open(anchor, id, where);
  }

  // ---------------------------------------------------------------- drawing

  mount() {
    const css = getComputedStyle(document.documentElement);
    const font = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
    setFonts(font("--font-ui", "sans-serif"), font("--font-mono", "monospace"));
    this.el.classList.add("dg-host");
    this.el.tabIndex = 0;
    this.svg = document.createElementNS(SVG_NS, "svg");
    this.svg.classList.add("dg-svg");
    this.world = document.createElementNS(SVG_NS, "g");
    this.world.classList.add("dg-world");
    this.content = document.createElementNS(SVG_NS, "g");
    this.rings = document.createElementNS(SVG_NS, "g");
    this.rings.classList.add("dg-rings");
    this.world.append(this.content, this.rings);
    this.svg.append(this.world);
    this.el.append(this.svg);
    this.ui = {};
    this.ui.toolbar = new Toolbar(this);
    this.ui.palette = new Palette(this);
    this.ui.controls = new Controls(this);
    this.ui.count = new CountEditor(this);
  }

  rerender(anchor = this.selected) {
    if (!this.model) return;
    const before = anchor && this.screenRect(this.drawnId(anchor));
    this.render(this.model);
    const after = before && this.screenRect(this.drawnId(anchor));
    if (after) this.moveTo({ ...this.view, x: this.view.x + before.x - after.x, y: this.view.y + before.y + before.h - after.y - after.h });
  }

  drawPanels(ink) {
    for (const id of this.zooms) {
      const node = ink.nodes.get(id);
      const box = ink.rects.get(id);
      if (!node?.internals || !box) continue;
      const frame = planInternals(node);
      drawPanel(frame, node, box, placePanel(box, panelSize(frame), ink.extents), ink);
    }
  }

  drawHeader(ink) {
    const b = ink.bounds();
    const { w } = headerSize(this.model, this.facts);
    drawHeader(this.model, this.facts, b.x - w - 64, b.y, ink);
  }

  /** The selection's rings, and the toolbar. */
  paint() {
    if (!this.ink) return;
    const id = this.drawnId(this.selected);
    let s = "";
    if (id) {
      const key = spanKey(this.ink.nodes.get(id));
      for (const [other, n] of this.ink.nodes) {
        if (key && other !== id && spanKey(n) === key) s += ring(this.ink.rects.get(other), "dg-ring dg-ring-soft");
      }
      s += ring(this.ink.rects.get(id), "dg-ring");
    }
    this.rings.innerHTML = s;
    this.ui.toolbar.show(this.toolbarTarget());
  }

  /** What is drawn for `id`: itself, its twin in a run's first block, or the
      folded card or container that holds it. */
  drawnId(id) {
    if (!id || !this.ink) return null;
    if (this.ink.rects.has(id)) return id;
    for (const [alias, first] of this.plan.aliases) {
      if (id.startsWith(`${alias}.`) || id === alias) return this.drawnId(first + id.slice(alias.length));
    }
    for (let cut = id; cut.includes("."); ) {
      cut = cut.slice(0, cut.lastIndexOf("."));
      if (this.ink.rects.has(cut)) return cut;
    }
    return null;
  }

  // ---------------------------------------------------------------- the view

  moveTo(view, animate = false) {
    cancelAnimationFrame(this.frame);
    if (!animate) return this.applyView(view);
    const from = { ...this.view };
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / 220);
      const e = 1 - (1 - t) ** 3;
      this.applyView({ k: from.k + (view.k - from.k) * e, x: from.x + (view.x - from.x) * e, y: from.y + (view.y - from.y) * e });
      if (t < 1) this.frame = requestAnimationFrame(step);
    };
    this.frame = requestAnimationFrame(step);
  }

  applyView(view) {
    this.view = view;
    const { x, y, k } = view;
    this.world.setAttribute("transform", `translate(${x} ${y}) scale(${k})`);
    this.el.style.backgroundSize = `${20 * k}px ${20 * k}px`;
    this.el.style.backgroundPosition = `${x}px ${y}px`;
    this.ui.controls.update(k);
    this.ui.toolbar.place();
  }

  zoomAt(px, py, factor) {
    const { x, y, k } = this.view;
    const next = clamp(k * factor, ZOOM.min, ZOOM.max);
    const f = next / k;
    this.applyView({ k: next, x: px - (px - x) * f, y: py - (py - y) * f });
  }

  zoomBy(factor) {
    const { width, height } = this.el.getBoundingClientRect();
    this.zoomAt(width / 2, height / 2, factor);
  }

  /** The rect of a drawn node in the host's pixels. */
  screenRect(id) {
    const r = id && this.ink?.rects.get(id);
    if (!r) return null;
    const { x, y, k } = this.view;
    return { x: r.x * k + x, y: r.y * k + y, w: r.w * k, h: r.h * k };
  }

  reveal(id) {
    const r = this.screenRect(this.drawnId(id));
    if (!r) return;
    const { width, height } = this.el.getBoundingClientRect();
    const m = 48;
    if (r.x > m && r.y > m && r.x + r.w < width - m && r.y + r.h < height - m) return;
    const { k } = this.view;
    const world = this.ink.rects.get(this.drawnId(id));
    this.moveTo({ k, x: width / 2 - (world.x + world.w / 2) * k, y: height / 2 - (world.y + world.h / 2) * k }, true);
  }

  async download(kind) {
    const svg = this.exportSVG();
    const name = this.model?.name ?? "model";
    const blob = kind === "svg" ? new Blob([svg], { type: "image/svg+xml" }) : await rasterize(svg, 2);
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `${name}.${kind}` });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  // ---------------------------------------------------------------- events

  listen() {
    const svg = this.svg;
    svg.addEventListener("pointerdown", (e) => this.down(e));
    svg.addEventListener("pointermove", (e) => this.drag(e));
    svg.addEventListener("pointerup", (e) => this.up(e));
    svg.addEventListener("click", (e) => this.click(e));
    svg.addEventListener("dblclick", (e) => this.dblclick(e));
    svg.addEventListener("pointerover", (e) => this.over(e));
    svg.addEventListener("pointerleave", () => this.hoverLater(null));
    this.el.addEventListener("wheel", (e) => this.wheel(e), { passive: false });
    this.el.addEventListener("keydown", (e) => this.key(e));
    new ResizeObserver(() => this.ui.toolbar.place()).observe(this.el);
  }

  down(e) {
    if (e.button !== 0) return;
    this.el.focus({ preventScroll: true });
    this.press = { x: e.clientX, y: e.clientY, view: { ...this.view }, id: e.pointerId, panning: false };
  }

  drag(e) {
    const p = this.press;
    if (!p || p.id !== e.pointerId) return;
    const [dx, dy] = [e.clientX - p.x, e.clientY - p.y];
    if (!p.panning && Math.hypot(dx, dy) < 4) return;
    if (!p.panning) {
      p.panning = true;
      this.svg.setPointerCapture(e.pointerId);
      this.el.classList.add("dg-panning");
    }
    this.applyView({ ...p.view, x: p.view.x + dx, y: p.view.y + dy });
  }

  up(e) {
    const p = this.press;
    this.press = null;
    if (!p?.panning) return;
    this.panned = true;
    this.el.classList.remove("dg-panning");
    this.svg.releasePointerCapture(e.pointerId);
  }

  click(e) {
    if (this.panned) return void (this.panned = false);
    const t = e.target;
    const zoom = t.closest("[data-zoom]");
    if (zoom) return this.setInternals(zoom.dataset.zoom);
    const toggle = t.closest("[data-toggle]");
    if (toggle) return this.setOpen(toggle.dataset.toggle);
    this.choose(t.closest("[data-id]")?.dataset.id ?? null);
  }

  dblclick(e) {
    const repeat = e.target.closest("[data-repeat]");
    if (repeat) return this.ui.count.open(repeat, repeat.dataset.repeat);
    const card = e.target.closest("[data-key]");
    if (card) this.setOpen(card.dataset.key, true);
  }

  /** A selection by the user: drawn, and reported. */
  choose(id) {
    this.select(id);
    this.on.select(id);
  }

  over(e) {
    this.hotTap(e.target.closest("[data-tap]")?.dataset.tap ?? null);
    const id = e.target.closest("[data-id]")?.dataset.id ?? null;
    if (id) this.hover(id);
    else this.hoverLater(null);
  }

  hover(id) {
    clearTimeout(this.hoverTimer);
    if (id === this.hovered) return;
    this.hovered = id;
    this.on.hover(id);
    this.ui.toolbar.show(this.toolbarTarget());
  }

  hoverLater(id) {
    clearTimeout(this.hoverTimer);
    this.hoverTimer = setTimeout(() => !this.ui.toolbar.held && this.hover(id), HOVER_GRACE);
  }

  toolbarTarget() {
    const editable = (id) => this.ink?.nodes.get(id)?.editable && id;
    return editable(this.hovered) || editable(this.drawnId(this.selected)) || null;
  }

  hotTap(name) {
    if (name === this.tap) return;
    this.tap = name;
    for (const el of this.content.querySelectorAll(".dg-hot")) el.classList.remove("dg-hot");
    if (!name) return;
    for (const el of this.content.querySelectorAll(`[data-tap="${CSS.escape(name)}"]`)) el.classList.add("dg-hot");
  }

  wheel(e) {
    e.preventDefault();
    const r = this.el.getBoundingClientRect();
    const delta = e.deltaY * (e.deltaMode === 1 ? 16 : 1);
    this.zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-delta * (e.ctrlKey ? 0.01 : 0.002)));
  }

  key(e) {
    if (e.target !== this.el) return;
    const id = this.drawnId(this.selected);
    const steps = { ArrowUp: 1, ArrowDown: -1 };
    if ((e.key === "Delete" || e.key === "Backspace") && this.ink?.nodes.get(id)?.editable) {
      this.on.action({ op: "delete", node: id });
    } else if (e.key === "Escape") this.choose(null);
    else if (e.key in steps) this.step(steps[e.key]);
    else return;
    e.preventDefault();
  }

  /** Moves the selection along the flow: up is downstream. */
  step(d) {
    const order = this.plan.order.filter((id) => this.ink.rects.has(id));
    if (!order.length) return;
    const i = order.indexOf(this.drawnId(this.selected));
    const next = order[i < 0 ? (d > 0 ? 0 : order.length - 1) : clamp(i + d, 0, order.length - 1)];
    this.choose(next);
    this.reveal(next);
  }
}

function ring(r, cls) {
  if (!r) return "";
  const pad = 3.5;
  const rx = Math.min(r.h / 2 + pad, 11);
  return `<rect class="${cls}" x="${r.x - pad}" y="${r.y - pad}" width="${r.w + 2 * pad}" height="${r.h + 2 * pad}" rx="${rx}"/>`;
}

function inlineStyles(live, clone) {
  const from = live.querySelectorAll("*");
  const to = clone.querySelectorAll("*");
  from.forEach((el, i) => {
    const cs = getComputedStyle(el);
    const style = EXPORT_PROPS.map((p) => `${p}:${cs.getPropertyValue(p)}`).join(";");
    to[i].setAttribute("style", style);
  });
}

function stripAttributes(el) {
  for (const name of [...el.getAttributeNames()]) {
    if (name === "class" || name.startsWith("data-")) el.removeAttribute(name);
  }
}

async function rasterize(svg, scale) {
  const img = new Image();
  img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
  await img.decode();
  const canvas = document.createElement("canvas");
  canvas.width = img.width * scale;
  canvas.height = img.height * scale;
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  ctx.drawImage(img, 0, 0);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}
