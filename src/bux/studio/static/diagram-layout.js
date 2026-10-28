/* Layout of the model diagram: a Node tree -> a tree of sized frames.

   The stream flows bottom to top along a frame's spine. A frame reaches `l` left
   and `r` right of its spine and is `h` tall; a child sits at `dx` (its spine from
   its parent's) and `dy` (its top from its parent's top). `fed` says whether the
   stream enters it at the bottom, `wideIn`/`wideOut` whether that stream is the
   widened one of a residual algebra. Drawing them is diagram-draw.js's job. */

export const G = {
  gap: 22, // the wire between two items of a column
  tight: 12, // the wire beside a tap or a mark
  box: { padX: 14, padY: 9, title: 18, meta: 16, minW: 176, maxW: 268, icon: 18, radius: 8 },
  label: { padX: 5, gap: 6, h: 15 },
  circle: 8, // a merge, ⊕ or ⊗
  stub: 16, // from a fork up to a branch
  merge: 16, // from a branch up to its merge
  lane: 18, // a skip wire, from what it passes
  branch: 28, // between two branches
  cont: { pad: 14, caption: 28, badge: 52, right: 14, radius: 12 },
  tag: { h: 18, padX: 7, gap: 10 },
  mark: { h: 17, padX: 7 },
  pill: { h: 24, padX: 14 },
  ports: 10, // between a port's tag and its box
  segment: 72, // between the columns of a stream cut by a head
  corner: 7, // the radius of a wire's turn
};

export const FONT = {};

export function setFonts(ui, mono) {
  Object.assign(FONT, {
    title: `600 13px ${ui}`,
    chip: `400 11px ${ui}`,
    label: `500 10.5px ${mono}`,
    card: `600 12.5px ${mono}`,
    times: `700 12.5px ${ui}`,
    caption: `500 11.5px ${mono}`,
    badge: `700 16px ${ui}`,
    tag: `500 11px ${mono}`,
    pill: `500 12px ${ui}`,
    mark: `600 11px ${ui}`,
    name: `600 16px ${ui}`,
  });
  widths.clear();
}

let pen = null;
const widths = new Map();

export function measure(text, font) {
  const key = `${font}\u0000${text}`;
  let w = widths.get(key);
  if (w === undefined) {
    pen ??= document.createElement("canvas").getContext("2d");
    pen.font = font;
    w = pen.measureText(text).width;
    widths.set(key, w);
  }
  return w;
}

// ---------------------------------------------------------------- the IR

const BLOCK_KINDS = new Set(["seq", "residual", "sum", "prod", "neg", "repeat"]);

export const isBlock = (node) => !!node.label && BLOCK_KINDS.has(node.kind);
const isHead = (node) => node?.kind === "layer" && node.type === "head";

export function children(node) {
  return node.items ?? node.terms ?? node.factors ?? (node.body ? [node.body] : []);
}

function startsWithRef(node) {
  switch (node.kind) {
    case "ref":
      return true;
    case "seq":
      return startsWithRef(node.items[0]);
    case "neg":
    case "repeat":
      return startsWithRef(node.body);
    default:
      return false;
  }
}

function* layers(node) {
  if (node.kind === "layer") yield node;
  for (const child of children(node)) yield* layers(child);
}

function lastLayer(node) {
  if (node.kind === "layer") return node;
  const kids = children(node);
  return node.kind === "seq" || node.kind === "repeat" ? lastLayer(kids.at(-1)) : null;
}

/** What a folded block holds: its main layers' types, "gdn · moe". */
export function summary(node) {
  const all = [...layers(node)];
  const main = all.filter((n) => !["norm", "op", "io"].includes(n.category));
  const types = [...new Set((main.length ? main : all).map((n) => n.type))];
  return types.length > 4 ? `${types.slice(0, 4).join(" · ")} …` : types.join(" · ");
}

const signatures = new WeakMap();

/** The structure of a node, without its ids: equal for `D D D`. */
function signature(node) {
  let sig = signatures.get(node);
  if (sig === undefined) {
    const own = [node.kind, node.label, node.type, node.times, node.value, node.name, node.pop];
    const extra = node.kind === "layer" ? [node.chips, node.ports] : [];
    sig = JSON.stringify([...own, ...extra, children(node).map(signature)]);
    signatures.set(node, sig);
  }
  return sig;
}

function flatten(nodes) {
  return nodes.flatMap((n) => (n.kind === "seq" && !n.label ? flatten(n.items) : [n]));
}

const PERIOD = 4; // the longest group of blocks folded into a run, as (E M) ×4

/** Splits a sequence into runs: `D D D` is D three times, `E M E M` (E M) twice. */
function runs(nodes) {
  const out = [];
  for (let i = 0; i < nodes.length; ) {
    const { k, n } = period(nodes, i);
    out.push({ nodes: nodes.slice(i, i + k), n, members: nodes.slice(i, i + k * n) });
    i += k * n;
  }
  return out;
}

/** The group of blocks at `i` that repeats over the most items, and how often. */
function period(nodes, i) {
  let best = { k: 1, n: 1 };
  for (let k = 1; k <= PERIOD && i + 2 * k <= nodes.length; k++) {
    if (!nodes.slice(i, i + k).every(isBlock)) break;
    let n = 1;
    while (i + (n + 1) * k <= nodes.length && repeats(nodes, i, i + n * k, k)) n++;
    if (n > 1 && k * n > best.k * best.n) best = { k, n };
  }
  return best;
}

function repeats(nodes, a, b, k) {
  for (let j = 0; j < k; j++) {
    if (nodes[a + j].label !== nodes[b + j].label) return false;
    if (signature(nodes[a + j]) !== signature(nodes[b + j])) return false;
  }
  return true;
}

/** Whether a residual algebra widens the stream, and into how many lanes. */
export function lanesOf(residual) {
  const type = /^\s*(\w+)/.exec(residual ?? "")?.[1];
  if (!["hc", "mhc", "gr", "attnres"].includes(type)) return 0;
  const expansion = Number(/expansion\s*=\s*(\d+)/.exec(residual)?.[1] ?? 4);
  return Math.max(2, Math.min(4, type === "attnres" ? 3 : expansion));
}

// ---------------------------------------------------------------- texts

function wrapMeta(label, chips) {
  const max = G.box.maxW - 2 * G.box.padX;
  const labelW = label ? measure(label, FONT.label) + 2 * G.label.padX : 0;
  const rows = [];
  let row = { label, labelW, text: "", w: labelW };
  for (const chip of chips) {
    const text = row.text ? `${row.text} · ${chip}` : chip;
    const w = (row.label ? labelW + G.label.gap : 0) + measure(text, FONT.chip);
    if (row.text && w > max) {
      rows.push(row);
      row = { label: null, labelW: 0, text: chip, w: measure(chip, FONT.chip) };
    } else Object.assign(row, { text, w });
  }
  if (row.label || row.text) rows.push(row);
  return rows;
}

function boxText(node) {
  const title = node.title ?? node.type ?? "layer";
  const chips = [...(node.shared ? ["shared"] : []), ...(node.chips ?? [])];
  const rows = wrapMeta(node.label, chips);
  const icon = node.internals ? 2 * G.box.icon : 0;
  const inner = Math.max(measure(title, FONT.title) + icon, ...rows.map((r) => r.w));
  const h = 2 * G.box.padY + G.box.title + rows.length * G.box.meta;
  return { title, rows, w: Math.max(G.box.minW, inner + 2 * G.box.padX), h };
}

function cardText(node, times) {
  const label = node.label;
  const count = times > 1 ? `×${times}` : "";
  const head = 14 + measure(label, FONT.card) + (count ? 6 + measure(count, FONT.times) : 0);
  const about = node.shared ? `shared · ${summary(node)}` : summary(node);
  const inner = Math.max(head, measure(about, FONT.chip));
  const h = 2 * G.box.padY + G.box.title + (about ? G.box.meta : 0);
  return { label, count, about, w: Math.max(G.box.minW, inner + 2 * G.box.padX), h };
}

const MODE = { write: "@", read: "$", pop: "$" };

function portTags(node) {
  return (node.ports ?? []).map((p) => {
    const text = `${p.name}=${MODE[p.mode]}${p.tap}${p.mode === "pop" ? "!" : ""}`;
    return { text, tap: p.tap, mode: p.mode, w: measure(text, FONT.tag) + 2 * G.tag.padX };
  });
}

const fmt = (value) => String(value);

// ---------------------------------------------------------------- planning

/** Turns Nodes into frames, deciding what is folded, what is a run, and which
    boxes share a column (and so a width). */
class Planner {
  constructor(folds, lanes, ns) {
    this.folds = folds;
    this.lanes = lanes;
    this.ns = ns;
    this.seen = new Map(); // label -> its occurrences so far
    this.widths = new Map(); // column -> its widest box
    this.order = []; // what arrow keys walk, in flow order
    this.aliases = new Map(); // a block folded into a run -> the run's first block
    this.decided = new Map(); // a block's key -> whether it is drawn open
    this.taps = new Map(); // a tap -> whether it holds the widened stream
  }

  widen(col, w) {
    this.widths.set(col, Math.max(this.widths.get(col) ?? 0, w));
  }

  fold(label) {
    const k = this.seen.get(label) ?? 0;
    this.seen.set(label, k + 1);
    const key = `${label}#${k}`;
    const { mode, keys } = this.folds;
    const open = keys.get(key) ?? (mode ? mode === "open" : k === 0);
    this.decided.set(key, open);
    return { key, open };
  }

  root(graph) {
    const c = { col: this.ns, wide: false, depth: 0 };
    const top = graph.kind === "seq" ? this.seq(graph.items, c) : this.chain([this.inner(graph, c)]);
    const segs = top.t === "cols" ? top.segs : [top];
    const tokens = firstIsEmbed(graph);
    if (segs[0].fed) segs[0].items.unshift(pill(tokens ? "Tokens" : "Input"));
    for (const seg of segs) if (endsInHead(seg)) seg.items.push(pill("Logits"));
    if (!tokens && !endsInHead(segs.at(-1))) segs.at(-1).items.push(pill("Output"));
    return top;
  }

  build(node, c) {
    return isBlock(node) ? this.block(node, c, 1, null) : this.inner(node, c);
  }

  inner(node, c) {
    switch (node.kind) {
      case "layer":
        return this.box(node, c);
      case "seq":
        return this.seq(node.items, c);
      case "residual":
        return this.residual(node, c);
      case "sum":
        return this.parallel(node, node.terms, "sum", c);
      case "prod":
        return this.prod(node, c);
      case "neg":
        return this.chain([this.build(node.body, c), mark(node, "−", true, c)]);
      case "image":
        return this.chain([this.build(node.body, c), mark(node, "image", true, c)]);
      case "const":
        return mark(node, fmt(node.value), false, c);
      case "repeat":
        return this.repeat(node, c);
      case "tap":
        return this.tap(node, c);
      case "ref":
        return this.ref(node, c);
      default:
        return wire(c.wide);
    }
  }

  seq(nodes, c) {
    const frames = [];
    let wide = c.wide;
    for (const run of runs(flatten(nodes))) {
      const f = this.run(run, { ...c, wide });
      wide = f.wideOut;
      if (f.t !== "wire") frames.push(f);
    }
    return frames.length ? this.chain(frames) : wire(c.wide);
  }

  /** A column of frames; one that ignores its input starts a column of its own. */
  chain(frames) {
    const segs = [];
    const solid = frames.filter((f) => f.t !== "wire");
    for (const f of solid.length ? solid : frames.slice(0, 1)) {
      if (!segs.length || (!f.fed && segs.at(-1).length)) segs.push([]);
      segs.at(-1).push(f);
    }
    const seqs = segs.map((items) => ({ t: "seq", items, ...ends(items) }));
    return seqs.length === 1 ? seqs[0] : { t: "cols", segs: seqs, ...ends(seqs) };
  }

  run({ nodes, n, members }, c) {
    members.slice(nodes.length).forEach((m, i) => this.aliases.set(m.id, nodes[i % nodes.length].id));
    if (nodes.length > 1) return container(GROUP, null, null, n, this.seq(nodes, { ...c, depth: c.depth + 1 }), c);
    return n > 1 ? this.block(nodes[0], c, n, null) : this.build(nodes[0], c);
  }

  block(node, c, times, repeat) {
    const { key, open } = this.fold(node.label);
    if (!open) return this.card(node, c, times, repeat, key);
    const body = this.inner(node, { ...c, depth: c.depth + 1 });
    return container(node, repeat, key, times, body, c);
  }

  repeat(node, c) {
    if (isBlock(node.body)) return this.block(node.body, c, node.times, node);
    const body = this.build(node.body, { ...c, depth: c.depth + 1 });
    return container(node, node, null, node.times, body, c);
  }

  tap(node, c) {
    this.order.push(node.id);
    this.taps.set(node.name, c.wide);
    return { t: "tap", node, text: `@${node.name}`, fed: true, ...stream(c.wide) };
  }

  /** A read of a tap; the stream it starts is as wide as the one written. */
  ref(node, c) {
    this.order.push(node.id);
    const text = `$${node.name}${node.pop ? "!" : ""}`;
    return { t: "ref", node, text, fed: false, ...stream(this.taps.get(node.name) ?? c.wide) };
  }

  box(node, c) {
    this.order.push(node.id);
    const text = boxText(node);
    this.widen(c.col, text.w);
    const wideIn = c.wide || node.width === "streams";
    const out = { wideIn, wideOut: this.layerWide(node, wideIn) };
    return { t: "box", node, text, col: c.col, ports: portTags(node), fed: true, ...out };
  }

  card(node, c, times, repeat, key) {
    this.order.push(node.id);
    const text = cardText(node, times);
    this.widen(c.col, text.w);
    const out = { wideIn: c.wide, wideOut: this.wideAfter(node, c.wide) };
    const ports = [...new Map([...layers(node)].flatMap(portTags).map((p) => [p.text, p])).values()];
    return { t: "card", node, repeat, key, times, text, col: c.col, ports, fed: !startsWithRef(node), ...out };
  }

  /** `[f]`; a branch that reads a tap, `[$h f]`, is drawn beside the stream. */
  residual(node, c) {
    const side = startsWithRef(node.body);
    const body = this.build(node.body, { ...c, wide: false, col: side ? `${node.id}:1` : c.col });
    if (side) return this.branches(node, [wire(false), body], "sum", c);
    return { t: "res", node, body, fed: true, ...stream(c.wide) };
  }

  parallel(node, terms, op, c) {
    const fed = terms.filter((t) => !startsWithRef(t));
    const sorted = [...fed, ...terms.filter(startsWithRef)];
    const cols = sorted.map((t, i) => this.build(t, { ...c, wide: false, col: i ? `${node.id}:${i}` : c.col }));
    return this.branches(node, cols, op, c);
  }

  branches(node, cols, op, c) {
    return { t: "par", node, op, cols, fed: cols[0].fed, ...stream(c.wide) };
  }

  prod(node, c) {
    const scales = node.factors.filter((f) => f.kind === "const");
    const others = node.factors.filter((f) => f.kind !== "const");
    const core = others.length > 1 ? this.parallel(node, others, "prod", c) : this.build(others[0], c);
    const marks = scales.map((k) => mark(k, `×${fmt(k.value)}`, true, c));
    return marks.length ? this.chain([core, ...marks]) : core;
  }

  layerWide(node, wide) {
    if (node.type === "merge") return false;
    return this.lanes > 0 && (node.type === "embed" || node.type === "nextn") ? true : wide;
  }

  wideAfter(node, wide) {
    switch (node.kind) {
      case "layer":
        return this.layerWide(node, wide || node.width === "streams");
      case "seq":
        return node.items.reduce((w, n) => this.wideAfter(n, w), wide);
      case "repeat":
        return this.wideAfter(node.body, wide);
      default:
        return wide;
    }
  }
}

const GROUP = { id: null, kind: "group", label: null }; // what a run of a group is drawn for
const stream = (wide) => ({ wideIn: wide, wideOut: wide });
const wire = (wide) => ({ t: "wire", fed: true, ...stream(wide) });
const pill = (text) => ({ t: "pill", text, fed: true, ...stream(false) });
const ends = (items) => ({ fed: items[0].fed, wideIn: items[0].wideIn, wideOut: items.at(-1).wideOut });

function mark(node, text, fed, c) {
  return { t: "mark", node, text, fed, ...stream(c.wide) };
}

function container(node, repeat, key, times, body, c) {
  const caption = isBlock(node) ? node.label : null;
  const base = { t: "cont", node, repeat, key, caption, times, body, depth: c.depth };
  return { ...base, fed: body.fed, wideIn: body.t === "wire" ? c.wide : body.wideIn, wideOut: body.wideOut };
}

function firstIsEmbed(node) {
  const first = layers(node).next().value;
  return first?.type === "embed";
}

function endsInHead(frame) {
  switch (frame.t) {
    case "seq":
      return endsInHead(frame.items.at(-1));
    case "cont":
      return endsInHead(frame.body);
    case "box":
      return isHead(frame.node);
    case "card":
      return isHead(lastLayer(frame.node));
    default:
      return false;
  }
}

// ---------------------------------------------------------------- sizing

const THIN = new Set(["tap", "mark", "ref"]);

function gapBetween(a, b) {
  return THIN.has(a.t) || THIN.has(b.t) ? G.tight : G.gap;
}

const SIZE = {
  box: sizeBox,
  card: sizeBox,
  seq: sizeSeq,
  cols: sizeCols,
  res: sizeRes,
  par: sizePar,
  cont: sizeCont,
  tap: sizeTap,
  ref: sizePill,
  mark: sizePill,
  pill: sizePill,
  wire: sizeWire,
};

function size(f, widths) {
  SIZE[f.t](f, widths);
  return f;
}

function sizeBox(f, widths) {
  f.w = widths.get(f.col);
  f.h = f.text.h;
  f.portW = f.ports?.length ? Math.max(...f.ports.map((p) => p.w)) + G.ports : 0;
  f.l = f.w / 2 + f.portW;
  f.r = f.w / 2;
}

const PILLS = {
  ref: ["tag", G.tag.padX, G.tag.h],
  mark: ["mark", G.mark.padX, G.mark.h],
  pill: ["pill", G.pill.padX, G.pill.h],
};

function sizePill(f) {
  const [font, pad, h] = PILLS[f.t];
  f.w = Math.max(h, measure(f.text, FONT[font]) + 2 * pad);
  f.h = h;
  f.l = f.r = f.w / 2;
}

function sizeTap(f) {
  f.w = measure(f.text, FONT.tag) + 2 * G.tag.padX;
  f.h = G.tag.h;
  f.l = G.tag.gap + f.w;
  f.r = 4;
}

function sizeWire(f) {
  f.l = f.r = f.h = 0;
}

function sizeSeq(f, widths) {
  let y = 0;
  f.items.forEach((item, i) => {
    size(item, widths);
    if (i) y += gapBetween(f.items[i - 1], item);
    item.bottom = y;
    y += item.h;
  });
  f.h = y;
  for (const item of f.items) Object.assign(item, { dx: 0, dy: y - item.bottom - item.h });
  stretch(f.items.filter((i) => i.t === "cont"));
  f.l = Math.max(...f.items.map((i) => i.l));
  f.r = Math.max(...f.items.map((i) => i.r));
}

/** Gives the containers of one column one width, as Raschka's blocks have. */
function stretch(conts) {
  const l = Math.max(...conts.map((c) => c.l));
  const r = Math.max(...conts.map((c) => c.r));
  for (const c of conts) Object.assign(c, { l, r });
}

function sizeRes(f, widths) {
  const body = size(f.body, widths);
  f.skipX = body.r + G.lane;
  f.h = G.stub + body.h + G.merge + 2 * G.circle;
  Object.assign(body, { dx: 0, dy: 2 * G.circle + G.merge });
  f.l = Math.max(body.l, G.circle);
  f.r = f.skipX + 1;
}

function sizePar(f, widths) {
  let x = 0;
  f.cols.forEach((col, i) => {
    size(col, widths);
    if (i) x += f.cols[i - 1].r + G.branch + col.l;
    col.dx = x;
  });
  // Branches start level: each first box at the height of the highest one.
  const most = Math.max(...f.cols.map(lift));
  const tallest = Math.max(...f.cols.map((c) => c.h + most - lift(c)));
  f.h = G.stub + tallest + G.merge + 2 * G.circle;
  for (const col of f.cols) col.dy = f.h - G.stub - col.h - (most - lift(col));
  f.l = Math.max(f.cols[0].l, G.circle);
  f.r = Math.max(G.circle, ...f.cols.map((c) => c.dx + c.r));
}

/** How far above a frame's bottom its first box is. */
function lift(f) {
  switch (f.t) {
    case "seq":
      return lift(f.items[0]);
    case "res":
    case "par":
      return f.fed ? G.stub + lift(f.t === "res" ? f.body : f.cols[0]) : 0;
    case "cont":
      return f.padB + lift(f.body);
    default:
      return 0;
  }
}

function sizeCont(f, widths) {
  const body = size(f.body, widths);
  const note = f.node.shared ? 6 + measure("shared", FONT.chip) : 0;
  const captionW = f.caption ? 16 + measure(f.caption, FONT.caption) + note + 14 : 0;
  const badge = f.times > 1 ? G.cont.badge : G.cont.pad;
  f.padL = Math.max(badge, captionW - body.l);
  f.padT = f.caption ? G.cont.caption : G.cont.pad;
  f.padB = G.cont.pad;
  f.l = body.l + f.padL;
  f.r = body.r + G.cont.right;
  f.h = f.padT + body.h + f.padB;
  Object.assign(body, { dx: 0, dy: f.padT });
}

function sizeCols(f, widths) {
  let x = 0;
  f.segs.forEach((seg, i) => {
    size(seg, widths);
    if (i) x += f.segs[i - 1].r + G.segment + seg.l;
    Object.assign(seg, { dx: x, dy: 0 });
  });
  f.links = [];
  f.segs.forEach((seg, i) => i && alignToTap(f, seg, i));
  const top = Math.min(...f.segs.map((s) => s.dy));
  for (const seg of f.segs) seg.dy -= top;
  for (const link of f.links) link.y -= top;
  f.h = Math.max(...f.segs.map((s) => s.dy + s.h));
  f.l = f.segs[0].l;
  f.r = Math.max(...f.segs.map((s) => s.dx + s.r));
}

/** Puts a column that starts by reading a tap level with where the tap is
    written, and links the two; else bottom-aligns it with the column before. */
function alignToTap(f, seg, i) {
  const ref = locate(seg, (k) => k.t === "ref");
  const taps = f.segs.slice(0, i).map((s) => [s, locate(s, (k) => k.t === "tap" && k.node.name === ref?.f.node.name, true)]);
  const [from, tap] = taps.reverse().find(([, t]) => t) ?? [];
  if (!ref || !tap) {
    seg.dy = f.segs[i - 1].dy + f.segs[i - 1].h - seg.h;
    return;
  }
  seg.dy = from.dy + tap.y + tap.f.h / 2 - (ref.y + ref.f.h / 2);
  const y = seg.dy + ref.y + ref.f.h / 2;
  f.links.push({ x1: from.dx + tap.x, x2: seg.dx + ref.x - ref.f.w / 2, y, wide: tap.f.wideIn });
}

export function kids(f) {
  switch (f.t) {
    case "seq":
      return f.items;
    case "cols":
      return f.segs;
    case "par":
      return f.cols;
    case "res":
    case "cont":
      return [f.body];
    default:
      return [];
  }
}

/** The first frame (or the last, with `last`) that `test` accepts, with its
    spine and top relative to `f`'s. */
export function locate(f, test, last = false, x = 0, y = 0) {
  let found = test(f) ? { f, x, y } : null;
  for (const k of kids(f)) {
    if (found && !last) break;
    found = locate(k, test, last, x + k.dx, y + k.dy) ?? found;
  }
  return found;
}

// ---------------------------------------------------------------- entry points

/** Lays out a model: its frames, sized, and what the diagram needs to walk them. */
export function plan(model, folds) {
  const lanes = lanesOf(model.facts?.residual);
  const planner = new Planner(folds, lanes, "c");
  const root = size(planner.root(model.graph), planner.widths);
  return { root, lanes, order: planner.order, aliases: planner.aliases, open: planner.decided };
}

/** Lays out a layer's internals, its own column of the same renderer. */
export function planInternals(node) {
  const planner = new Planner({ mode: "open", keys: new Map() }, 0, `${node.id}/i`);
  const frame = planner.chain([planner.build(node.internals, { col: planner.ns, wide: false, depth: 1 })]);
  return size(frame, planner.widths);
}
