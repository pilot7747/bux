// A tiny chart toolkit for the training and sweep panels: scales (linear, log), nice
// ticks, axes, line / step / area paths, bands, a tooltip, a crosshair, dragging and
// number formats. Plain SVG; every color comes from a CSS class in panels.css.

const SVG = "http://www.w3.org/2000/svg";

// --- Elements -------------------------------------------------------------------------

/** Make an SVG element with `attrs`, appended to `parent` if given. */
export function svg(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}

/** Make an SVG text element: `content` set as text, never parsed. */
export function label(parent, x, y, content, attrs = {}) {
  const node = svg("text", { x, y, ...attrs }, parent);
  node.textContent = content;
  return node;
}

/** Make an HTML element: `props.cls` its classes, `props.text` its text. */
export function h(tag, props = {}, parent = null) {
  const node = document.createElement(tag);
  const { cls, text, ...rest } = props;
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  for (const [k, v] of Object.entries(rest)) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on")) node.addEventListener(k.slice(2).toLowerCase(), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  if (parent) parent.appendChild(node);
  return node;
}

// --- Scales and ticks -----------------------------------------------------------------

/** A linear scale from `domain` to `range`, with `invert` and `ticks(n)`. */
export function linear([d0, d1], [r0, r1]) {
  const span = d1 - d0 || 1;
  const scale = (v) => r0 + ((v - d0) / span) * (r1 - r0);
  scale.invert = (p) => d0 + ((p - r0) / (r1 - r0 || 1)) * span;
  scale.domain = [d0, d1];
  scale.range = [r0, r1];
  scale.ticks = (n = 5) => niceTicks(d0, d1, n);
  return scale;
}

/** A log scale (base 10) from `domain`, both above 0, to `range`. */
export function log([d0, d1], [r0, r1]) {
  const [l0, l1] = [Math.log10(d0), Math.log10(d1)];
  const inner = linear([l0, l1], [r0, r1]);
  const scale = (v) => inner(Math.log10(Math.max(v, 1e-300)));
  scale.invert = (p) => 10 ** inner.invert(p);
  scale.domain = [d0, d1];
  scale.range = [r0, r1];
  scale.ticks = () => logTicks(d0, d1);
  return scale;
}

/** Return a step of 1, 2 or 5 times a power of ten near `raw`. */
function niceStep(raw) {
  const power = 10 ** Math.floor(Math.log10(raw));
  const f = raw / power;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * power;
}

/** Return about `n` round values from `lo` to `hi`, both included where round. */
export function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const step = niceStep((hi - lo) / Math.max(n, 1));
  const found = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    found.push(Number(v.toPrecision(12)));
  }
  return found;
}

/** Return powers of ten from `lo` to `hi`, with 2 and 5 between them in a short span. */
export function logTicks(lo, hi) {
  const [a, b] = [Math.floor(Math.log10(lo)), Math.ceil(Math.log10(hi))];
  const steps = b - a <= 2 ? [1, 2, 5] : [1];
  const found = [];
  for (let p = a; p <= b; p++) {
    for (const m of steps) {
      const v = Number((m * 10 ** p).toPrecision(12));
      if (v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9)) found.push(v);
    }
  }
  return found.length >= 2 ? found : [lo, hi];
}

// --- Numbers --------------------------------------------------------------------------

/** 5722 -> "5,722": a count, grouped. */
export function grouped(n) {
  return Math.round(n).toLocaleString("en-US");
}

/** 1500 -> "1.5k", 3e9 -> "3B": a count in at most 3 digits, as bux writes them. */
export function compact(n) {
  const a = Math.abs(n);
  if (a < 1000) return String(Number(n.toPrecision(3)));
  for (const [suffix, scale] of [["k", 1e3], ["M", 1e6], ["B", 1e9], ["T", 1e12]]) {
    if (a < scale * 1000 || suffix === "T") return `${Number((n / scale).toPrecision(3))}${suffix}`;
  }
  return String(n);
}

/** Return a formatter for the values of one axis, all in one form: "1e-3, 2e-3" or
 *  "0.01, 0.02", never both. */
export function rateFormat(values) {
  const top = Math.max(...values.map(Math.abs));
  return top < 0.01 ? (v) => (v === 0 ? "0" : exponent(v)) : sci;
}

/** 0.01 -> "1e-2", 0.0003 -> "3e-4". */
export function exponent(v) {
  const [m, e] = v.toExponential().split("e");
  return `${Number(Number(m).toPrecision(3))}e${Number(e)}`;
}

/** 0.003 -> "3e-3", 0.02 -> "0.02": a rate, as bux writes them. */
export function sci(x) {
  if (x === 0) return "0";
  if (Math.abs(x) >= 0.01 && Math.abs(x) < 1e4) return String(Number(x.toPrecision(3)));
  const [m, e] = x.toExponential().split("e");
  return `${Number(Number(m).toPrecision(3))}e${Number(e)}`;
}

// --- Text -----------------------------------------------------------------------------

let measurer = null;

/** Return the width in pixels of `text` at `size` px, in the UI or the mono font. */
export function textWidth(text, size = 11, mono = false) {
  measurer ??= document.createElement("canvas").getContext("2d");
  const root = getComputedStyle(document.documentElement);
  const family = root.getPropertyValue(mono ? "--font-mono" : "--font-ui") || "sans-serif";
  measurer.font = `${size}px ${family}`;
  return measurer.measureText(text).width;
}

// --- Axes -----------------------------------------------------------------------------

/** Draw an x axis at `y`: a hairline, tick labels below it, gridlines up to `top`. */
export function axisBottom(g, ticks, { y, left, right, top = null }) {
  const axis = svg("g", { class: "chart-axis" }, g);
  svg("line", { x1: left, x2: right, y1: y + 0.5, y2: y + 0.5, class: "chart-baseline" }, axis);
  for (const t of ticks) {
    if (top !== null) {
      const x = Math.round(t.px) + 0.5;
      svg("line", { x1: x, x2: x, y1: top, y2: y, class: "chart-grid" }, axis);
    }
    const anchor = t.px - left < 12 ? "start" : right - t.px < 12 ? "end" : "middle";
    label(axis, t.px, y + 14, t.label, { "text-anchor": anchor, class: "chart-tick" });
  }
  return axis;
}

/** Draw a y axis: tick labels right-aligned at `x`, gridlines from `left` to `right`. */
export function axisLeft(g, ticks, { x, left, right }) {
  const axis = svg("g", { class: "chart-axis" }, g);
  for (const t of ticks) {
    const y = Math.round(t.px) + 0.5;
    svg("line", { x1: left, x2: right, y1: y, y2: y, class: "chart-grid" }, axis);
    label(axis, x, t.px + 3.5, t.label, { "text-anchor": "end", class: "chart-tick" });
  }
  return axis;
}

// --- Marks ----------------------------------------------------------------------------

/** Return the path of a line through `points` ([x, y] in data), placed by `x`, `y`. */
export function linePath(points, x, y) {
  return points.map(([a, b], i) => `${i ? "L" : "M"}${x(a).toFixed(1)},${y(b).toFixed(1)}`).join("");
}

/** Return the path of the area under `points` down to `y0` px. */
export function areaPath(points, x, y, y0) {
  if (!points.length) return "";
  const first = x(points[0][0]).toFixed(1);
  const last = x(points.at(-1)[0]).toFixed(1);
  return `M${first},${y0}${linePath(points, x, y).replace("M", "L")}L${last},${y0}Z`;
}

/** Return the path of a step line: `segments` [{start, stop, value}] held flat, joined
 *  by risers where the value changes. */
export function stepPath(segments, x, y) {
  return segments
    .map((s, i) => {
      const [x0, x1, v] = [x(s.start).toFixed(1), x(s.stop).toFixed(1), y(s.value).toFixed(1)];
      return `${i ? "L" : "M"}${x0},${v}L${x1},${v}`;
    })
    .join("");
}

/** Draw a band from `x0` to `x1` px, from `top` to `bottom` px. */
export function band(g, x0, x1, top, bottom, cls) {
  return svg("rect", { x: x0, y: top, width: Math.max(0, x1 - x0), height: bottom - top, class: cls }, g);
}

// --- Tooltip and crosshair ------------------------------------------------------------

/** A tooltip inside `host` (positioned): `show(event, rows, title)` and `hide()`.
 *  Rows are {value, name, color?}: the value strong, the name after it. */
export function tooltip(host) {
  const tip = h("div", { cls: "chart-tip", role: "tooltip" }, host);
  tip.hidden = true;
  function show(event, rows, title = null) {
    tip.replaceChildren();
    if (title) h("div", { cls: "chart-tip-title", text: title }, tip);
    for (const row of rows) tipRow(tip, row);
    tip.hidden = false;
    place(tip, host, event);
  }
  return { show, hide: () => (tip.hidden = true), node: tip };
}

function tipRow(tip, { value, name, color, mono }) {
  const row = h("div", { cls: "chart-tip-row" }, tip);
  if (color) h("span", { cls: "chart-tip-key", style: `background:${color}` }, row);
  h("span", { cls: "chart-tip-value" + (mono ? " mono" : ""), text: value }, row);
  if (name) h("span", { cls: "chart-tip-name", text: name }, row);
}

function place(tip, host, event) {
  const box = host.getBoundingClientRect();
  const [w, ht] = [tip.offsetWidth, tip.offsetHeight];
  let left = event.clientX - box.left + 14;
  let top = event.clientY - box.top + 14;
  if (left + w > host.clientWidth - 4) left = event.clientX - box.left - w - 14;
  if (top + ht > host.scrollHeight - 4) top = event.clientY - box.top - ht - 14;
  tip.style.left = `${Math.max(4, left)}px`;
  tip.style.top = `${Math.max(4, top)}px`;
}

/** A crosshair over the plot `area` {left, right, top, bottom} of `g`: a hairline that
 *  snaps to `snap(px)` (a px or null) and shows `rows(px)` in `tip`. */
export function crosshair(g, area, { snap, rows, title, tip }) {
  const hair = svg("line", { y1: area.top, y2: area.bottom, class: "chart-hair" }, g);
  hair.style.display = "none";
  const hit = svg("rect", {
    x: area.left, y: area.top, width: area.right - area.left,
    height: area.bottom - area.top, class: "chart-hit",
  }, g);
  hit.addEventListener("pointermove", (event) => {
    const px = snap(pointerX(event, g));
    if (px === null) return hideHair(hair, tip);
    hair.setAttribute("x1", px);
    hair.setAttribute("x2", px);
    hair.style.display = "";
    tip.show(event, rows(px), title ? title(px) : null);
  });
  hit.addEventListener("pointerleave", () => hideHair(hair, tip));
  return hit;
}

function hideHair(hair, tip) {
  hair.style.display = "none";
  tip.hide();
}

/** Return the pointer's x in the coordinates of `g`'s SVG. */
export function pointerX(event, g) {
  const root = g.ownerSVGElement ?? g;
  return event.clientX - root.getBoundingClientRect().left;
}

// --- Dragging -------------------------------------------------------------------------

/** Make `node` draggable: `move(x)` with the pointer's x in its SVG while dragging,
 *  `end(x)` on release. The listeners outlive a redraw of the node itself. */
export function draggable(node, { move, end }) {
  node.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    event.stopPropagation();
    const root = node.ownerSVGElement;
    const at = (e) => e.clientX - root.getBoundingClientRect().left;
    const onMove = (e) => move(at(e));
    const onUp = (e) => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      document.body.classList.remove("chart-dragging");
      end(at(e));
    };
    document.body.classList.add("chart-dragging");
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  });
}

// --- Colors ---------------------------------------------------------------------------

/** Return the color of series `i`, from 0: tokens.css's --series-1..6, in order. */
export function series(i) {
  return `var(--series-${(i % 6) + 1})`;
}

/** Call `draw(width)` now and whenever `el`'s width changes. */
export function onWidth(el, draw) {
  let width = el.clientWidth;
  const observer = new ResizeObserver(() => {
    if (el.clientWidth === width || !el.isConnected) return;
    width = el.clientWidth;
    draw(width);
  });
  observer.observe(el);
  return observer;
}

// --- Cards ----------------------------------------------------------------------------

/** Append a card to `grid`: a title (a button when `onTitle`), a subtitle, a slot for
 *  tools on the right, and a body. `wide` spans the grid's width. */
export function card(grid, { title, subtitle = null, wide = false, onTitle = null, mono = false }) {
  const root = h("section", { cls: "panel-card" + (wide ? " wide" : "") }, grid);
  const head = h("header", { cls: "card-head" }, root);
  const titles = h("div", { cls: "card-titles" }, head);
  const tag = onTitle ? "button" : "h3";
  const name = h(tag, { cls: "card-title" + (onTitle ? " clickable" : ""), text: title }, titles);
  if (onTitle) name.addEventListener("click", onTitle);
  if (subtitle) h("div", { cls: "card-subtitle" + (mono ? " mono" : ""), text: subtitle, title: subtitle }, titles);
  const tools = h("div", { cls: "card-tools" }, head);
  const body = h("div", { cls: "card-body" }, root);
  return { root, head, tools, body };
}

/** Fill `el` with one calm card saying there is nothing to draw, and how to change it. */
export function emptyCard(el, text, hint, code = null) {
  const box = h("div", { cls: "panel-empty" }, el);
  h("div", { cls: "panel-empty-title", text }, box);
  if (hint) h("div", { cls: "panel-empty-hint", text: hint }, box);
  if (code) h("pre", { cls: "panel-empty-code", text: code }, box);
  return box;
}
