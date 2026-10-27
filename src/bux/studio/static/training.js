// The training panel of bux studio: a run's budget, learning rate, curriculum, data,
// evals and settings, as cards. Every edit is an op for the backend (plans/studio.md);
// the LR, curriculum and eval cards share one x axis, in steps or in tokens.

import {
  areaPath, axisBottom, axisLeft, band, card, compact, crosshair, draggable, emptyCard,
  grouped, h, label, linePath, linear, log, niceTicks, onWidth, rateFormat, sci, series,
  stepPath, svg, textWidth, tooltip,
} from "./charts.js";

// The left margin of every timeline chart, so that their x axes line up.
const GUTTER = 92;
const PAD_RIGHT = 18;
const TYPES = ["constant", "cosine", "wsd", "linear_decay", "multistep", "power", "piecewise"];
const PLAIN = new Set(["stable", "peak", "hold", "constant", "power law"]);
const states = new WeakMap();

/**
 * Draw `run`'s training into `el`, replacing what it held.
 *
 * @param {HTMLElement} el
 * @param {object|null} run the View's `run` (plans/studio.md), null for a model
 * @param {{onEdit?: Function, onInspect?: Function, schedule?: () => Promise<object>,
 *          params?: number|{params: number, params_text?: string},
 *          axis?: "steps"|"tokens", onAxis?: Function}} options `axis` is the x axis
 *   the first time `el` is drawn; the toggle keeps it after, and tells `onAxis`
 */
export function renderTraining(el, run, options = {}) {
  const state = stateOf(el, options.axis);
  if (state.run !== run) {
    state.previous = state.draft ?? state.schedule ?? state.previous;
    state.draft = null;
    state.schedule = null;
    state.run = run;
    if (run) load(state, run, options);
  }
  state.options = options;
  draw(el, state);
}

function stateOf(el, axis) {
  if (!states.has(el)) {
    const mode = axis === "tokens" ? "tokens" : "steps";
    const state = { mode, run: null, schedule: null, previous: null, draft: null };
    states.set(el, state);
    onWidth(el, () => draw(el, state));
  }
  return states.get(el);
}

function load(state, run, options) {
  const pending = options.schedule?.();
  if (!pending) return;
  state.loading = true;
  pending.then(
    (schedule) => loaded(state, run, schedule),
    () => loaded(state, run, null),
  );
}

function loaded(state, run, schedule) {
  if (state.run !== run) return;
  state.loading = false;
  state.schedule = schedule;
  state.failed = schedule === null;
  state.redrawLR?.();
}

function draw(el, state) {
  el.classList.add("bux-panels", "training-panel");
  el.replaceChildren();
  const { run, options } = state;
  if (!run) return emptyCard(el, "This file is a model, and trains nothing.",
    "Set data, optimizer, seq_len, batch and steps to make it a run.");
  const ctx = { run, options, state, tip: tooltip(el), colors: sourceColors(run) };
  toolbar(el, ctx);
  const grid = h("div", { cls: "panel-grid" }, el);
  ctx.grid = h("div", { cls: "panel-row" }, grid);
  budgetCard(ctx);
  dataCard(ctx);
  ctx.grid = grid;
  lrCard(ctx);
  curriculumCard(ctx);
  eventsCard(ctx);
  settingsCard(ctx);
}

// --- The shared x axis ----------------------------------------------------------------

/** Return the x axis of a timeline `width` px wide: `x(step)` in px, `step(px)`, its
 *  ticks, in steps or in tokens trained (`state.mode`). */
function timeline(ctx, width) {
  const steps = ctx.run.budget.steps;
  const [left, right] = [GUTTER, Math.max(GUTTER + 40, width - PAD_RIGHT)];
  const n = Math.max(2, Math.floor((right - left) / 90));
  if (ctx.state.mode === "tokens") return tokenTimeline(ctx.run, left, right, n);
  const lin = linear([0, steps], [left, right]);
  const ticks = lin.ticks(n).map((v) => ({ px: lin(v), label: compact(v) }));
  return { x: lin, step: lin.invert, ticks, left, right, unit: "steps" };
}

function tokenTimeline(run, left, right, n) {
  const { at, inverse } = tokenMap(run);
  const lin = linear([0, at(run.budget.steps)], [left, right]);
  const ticks = lin.ticks(n).map((v) => ({ px: lin(v), label: compact(v) }));
  return { x: (s) => lin(at(s)), step: (px) => inverse(lin.invert(px)), ticks, left, right, unit: "tokens" };
}

/** Return the tokens trained before a step, `at(step)`, and its inverse. */
function tokenMap(run) {
  const parts = run.stretches.length ? run.stretches
    : [{ start: 0, stop: run.budget.steps, tokens_per_step: run.budget.batch * run.budget.seq_len }];
  const before = [];
  let sum = 0;
  for (const s of parts) {
    before.push(sum);
    sum += (s.stop - s.start) * s.tokens_per_step;
  }
  const at = (step) => {
    const k = Math.max(0, parts.findLastIndex((s) => s.start <= step));
    return before[k] + (step - parts[k].start) * parts[k].tokens_per_step;
  };
  const inverse = (tokens) => {
    const k = Math.max(0, before.findLastIndex((b) => b <= tokens));
    return parts[k].start + (tokens - before[k]) / parts[k].tokens_per_step;
  };
  return { at, inverse };
}

function stepTitle(ctx, step) {
  const tokens = tokenMap(ctx.run).at(step);
  return `step ${grouped(step)} · ${compact(tokens)} tokens`;
}

function toolbar(el, ctx) {
  const bar = h("div", { cls: "panel-toolbar" }, el);
  const name = h("div", { cls: "panel-run" }, bar);
  h("span", { cls: "panel-run-name", text: ctx.run.name }, name);
  h("span", { cls: "panel-run-id", text: ctx.run.id }, name);
  const toggle = h("div", { cls: "segmented", role: "group", "aria-label": "x axis" }, bar);
  h("span", { cls: "segmented-label", text: "x axis" }, toggle);
  for (const mode of ["steps", "tokens"]) {
    h("button", {
      cls: "segmented-option" + (ctx.state.mode === mode ? " on" : ""), text: mode,
      "aria-pressed": String(ctx.state.mode === mode),
      onClick: () => setMode(el, ctx.state, mode),
    }, toggle);
  }
}

function setMode(el, state, mode) {
  state.mode = mode;
  state.options.onAxis?.(mode);
  draw(el, state);
}

// --- Budget ---------------------------------------------------------------------------

function budgetCard(ctx) {
  const { budget } = ctx.run;
  const c = card(ctx.grid, { title: "Budget", onTitle: inspect(ctx, "steps") });
  const row = h("div", { cls: "figures" }, c.body);
  figure(row, ctx, "steps", budget.steps);
  h("span", { cls: "figure-op", text: "×" }, row);
  figure(row, ctx, "batch", budget.batch, budget.varies ? "batch at first" : "batch");
  h("span", { cls: "figure-op", text: "×" }, row);
  figure(row, ctx, "seq_len", budget.seq_len, budget.varies ? "seq_len at first" : "seq_len");
  h("span", { cls: "figure-op", text: "=" }, row);
  result(row, budget.tokens_text, "tokens");
  budgetNotes(c.body, ctx);
}

function figure(row, ctx, name, value, caption = name) {
  const box = h("div", { cls: "figure" }, row);
  const written = ctx.run.settings.find((s) => s.name === name)?.text ?? String(value);
  editable(box, {
    text: grouped(value), value: written, title: `Edit ${name}: a number, or 2k, 1.5B`,
    commit: (text) => emit(ctx, [{ op: "set", target: { setting: name }, key: null, value: text }]),
  });
  h("div", { cls: "figure-caption", text: caption }, box);
}

function result(row, text, caption) {
  const box = h("div", { cls: "figure figure-result" }, row);
  h("div", { cls: "figure-value static", text }, box);
  h("div", { cls: "figure-caption", text: caption }, box);
}

function budgetNotes(body, ctx) {
  const { budget, stretches } = ctx.run;
  const notes = h("div", { cls: "figure-notes" }, body);
  const per = stretches.at(-1)?.tokens_per_step ?? budget.batch * budget.seq_len;
  h("span", { text: `${compact(per)} tokens a step${budget.varies ? " at the end" : ""}` }, notes);
  const params = paramsOf(ctx.options.params);
  if (params) {
    h("span", { text: `${compact(budget.tokens / params.n)} tokens per parameter` }, notes);
    h("span", { cls: "muted", text: `${params.text} parameters` }, notes);
  }
  if (budget.varies) h("span", { cls: "muted", text: "the curriculum changes the shape" }, notes);
}

function paramsOf(params) {
  if (!params) return null;
  if (typeof params === "number") return { n: params, text: compact(params) };
  return { n: params.params, text: params.params_text ?? compact(params.params) };
}

/** A number that turns into an input on click: Enter commits, Escape cancels. */
function editable(parent, { text, value, title, commit }) {
  const button = h("button", { cls: "figure-value", text, title }, parent);
  button.addEventListener("click", () => {
    const input = h("input", { cls: "figure-input", value, spellcheck: "false", "aria-label": title });
    input.style.width = `${Math.max(4, value.length + 1)}ch`;
    button.replaceWith(input);
    input.focus();
    input.select();
    let done = false;
    const finish = (save) => {
      if (done) return;
      done = true;
      const next = input.value.trim();
      input.replaceWith(button);
      if (save && next && next !== value) {
        button.textContent = next;
        button.classList.add("pending");
        commit(next);
      }
    };
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") finish(true);
      else if (e.key === "Escape") finish(false);
    });
    input.addEventListener("blur", () => finish(false));
  });
}

// --- Data -----------------------------------------------------------------------------

/** Give each source a series color, in the order met: the first mixture's, then those
 *  the phases bring in. Color follows the source, everywhere in the panel. */
function sourceColors(run) {
  const names = run.data.map((d) => d.name);
  for (const s of run.stretches) {
    for (const part of parseMixture(s.data)) if (!names.includes(part.name)) names.push(part.name);
  }
  const map = new Map(names.map((n, i) => [n, series(i)]));
  return (name) => map.get(name) ?? "var(--text-3)";
}

function dataCard(ctx) {
  const { data, stretches } = ctx.run;
  const c = card(ctx.grid, { title: "Data", onTitle: inspect(ctx, "data"), subtitle: dataSubtitle(data) });
  stackedBar(c.body, ctx, data);
  const legend = h("div", { cls: "mix-legend" }, c.body);
  for (const d of data) legendRow(legend, ctx, d);
  const changes = stretches.findIndex((s, i) => i > 0 && s.data !== stretches[0].data);
  if (changes > 0) {
    h("div", { cls: "card-note", text: `Phases change it from step ${grouped(stretches[changes].start)}: see the curriculum.` }, c.body);
  }
}

function dataSubtitle(data) {
  return data.length === 1 ? "one source" : `a mixture of ${data.length} sources`;
}

function stackedBar(body, ctx, data) {
  const bar = h("div", { cls: "mix-bar", role: "img", "aria-label": data.map(mixText).join(", ") }, body);
  for (const d of data) {
    const seg = h("div", { cls: "mix-seg", style: `flex-grow:${d.weight};background:${ctx.colors(d.name)}` }, bar);
    seg.addEventListener("pointermove", (e) => showSource(ctx, e, d));
    seg.addEventListener("pointerleave", () => ctx.tip.hide());
    seg.addEventListener("click", () => ctx.options.onInspect?.({ setting: "data" }));
  }
}

function legendRow(legend, ctx, d) {
  const row = h("button", { cls: "mix-row", title: d.text }, legend);
  h("span", { cls: "swatch", style: `background:${ctx.colors(d.name)}` }, row);
  h("span", { cls: "mix-name", text: d.name }, row);
  h("span", { cls: "mix-share", text: percent(d.weight) }, row);
  if (d.text !== d.name) h("span", { cls: "mix-text", text: d.text }, row);
  row.addEventListener("click", () => ctx.options.onInspect?.({ setting: "data" }));
}

function showSource(ctx, event, d) {
  ctx.tip.show(event, [{ value: percent(d.weight), name: d.name, color: ctx.colors(d.name) },
    { value: d.text, mono: true }]);
}

function mixText(d) {
  return `${d.name} ${percent(d.weight)}`;
}

function percent(w) {
  const p = w * 100;
  return `${p >= 10 || Number.isInteger(p) ? Math.round(p) : p.toPrecision(2)}%`;
}

// --- Learning rate --------------------------------------------------------------------

function lrCard(ctx) {
  const c = card(ctx.grid, {
    title: "Learning rate", wide: true, onTitle: inspect(ctx, "schedule"),
    subtitle: ctx.run.schedule.text, mono: true,
  });
  typeSelect(c.tools, ctx);
  const root = svg("svg", { class: "chart", role: "img", "aria-label": "learning rate over the steps" }, c.body);
  const redraw = () => drawLR(root, ctx, c.body.clientWidth || 800);
  ctx.state.redrawLR = redraw;
  redraw();
}

function typeSelect(tools, ctx) {
  const select = h("select", { cls: "panel-select", "aria-label": "schedule type" }, tools);
  for (const t of TYPES) h("option", { value: t, text: t, selected: t === ctx.run.schedule.type }, select);
  select.addEventListener("change", () => {
    const value = scheduleTemplate(select.value, writtenWarmup(ctx.run.schedule.text));
    emit(ctx, [{ op: "set", target: { setting: "schedule" }, key: null, value }]);
  });
}

/** Return the warmup as the schedule's text writes it, "1000" or "0.15", or null. */
function writtenWarmup(text) {
  const found = /\bwarmup\s*=\s*([^,()\s]+)/.exec(text ?? "");
  return found ? found[1] : null;
}

/** Return a schedule of `type` as source, with `warmup` carried over. */
export function scheduleTemplate(type, warmup) {
  const args = warmup && Number(warmup) !== 0 ? [`warmup=${warmup}`] : [];
  if (type === "piecewise") args.push("pieces=[piece(until=0.8), piece(to=0.1, shape=cosine)]");
  return `${type}(${args.join(", ")})`;
}

/** Return what the LR chart draws: the server's points, or a draft being dragged. */
function lrModel(ctx) {
  if (ctx.state.draft) return ctx.state.draft;
  const sched = ctx.state.schedule;
  if (!sched) return null;
  const { schedule, budget } = ctx.run;
  return { points: sched.points, peak: sched.peak, steps: budget.steps, regions: schedule.regions, warmup: schedule.warmup };
}

function drawLR(root, ctx, width) {
  const height = 212;
  sized(root, width, height);
  const time = timeline(ctx, width);
  const area = { left: time.left, right: time.right, top: 26, bottom: height - 24 };
  const model = lrModel(ctx);
  if (!model) return lrPlaceholder(root, ctx, time, area);
  const top = Math.max(model.peak, ...model.points.map((p) => p[1])) * 1.16;
  const y = linear([0, top], [area.bottom, area.top]);
  const values = niceTicks(0, top, 4).filter((v) => v <= top);
  const fmt = rateFormat(values);
  const yTicks = values.map((v) => ({ px: y(v), label: fmt(v) }));
  regionLayer(root, model, time, area);
  axisLeft(root, yTicks, { x: GUTTER - 10, left: area.left, right: area.right });
  axisBottom(root, time.ticks, { y: area.bottom, left: area.left, right: area.right });
  unitLabel(root, time, area.bottom);
  lrCurve(root, model, time, y, area);
  peakLabel(root, model, time, y, area);
  lrHover(root, ctx, model, time, y, area);
  handles(root, ctx, model, time, y, area);
}

function sized(root, width, height) {
  root.replaceChildren();
  root.setAttribute("width", width);
  root.setAttribute("height", height);
  root.setAttribute("viewBox", `0 0 ${width} ${height}`);
}

function lrPlaceholder(root, ctx, time, area) {
  const held = ctx.state.previous;
  axisBottom(root, time.ticks, { y: area.bottom, left: area.left, right: area.right });
  unitLabel(root, time, area.bottom);
  if (ctx.state.failed || !ctx.options.schedule) {
    const text = ctx.state.failed ? "The schedule could not be computed." : "No schedule to draw.";
    label(root, (area.left + area.right) / 2, (area.top + area.bottom) / 2, text, { class: "chart-empty", "text-anchor": "middle" });
    return;
  }
  if (held && held.steps === ctx.run.budget.steps) return heldCurve(root, held, time, area);
  const g = svg("g", { class: "chart-skeleton" }, root);
  band(g, area.left, area.right, area.top, area.bottom, "skeleton-block");
  label(root, (area.left + area.right) / 2, (area.top + area.bottom) / 2 + 4, "computing the schedule…", { class: "chart-empty", "text-anchor": "middle" });
}

function heldCurve(root, held, time, area) {
  const top = Math.max(...held.points.map((p) => p[1])) * 1.16;
  const y = linear([0, top], [area.bottom, area.top]);
  const g = svg("g", { class: "chart-held" }, root);
  svg("path", { d: linePath(held.points, time.x, y), class: "lr-line" }, g);
}

function unitLabel(root, time, y) {
  label(root, GUTTER - 10, y + 14, time.unit, { class: "chart-unit", "text-anchor": "end" });
}

function regionLayer(root, model, time, area) {
  const g = svg("g", { class: "lr-regions" }, root);
  for (const r of model.regions) {
    const [x0, x1] = [time.x(r.start), time.x(r.stop)];
    if (!PLAIN.has(r.name)) band(g, x0, x1, area.top - 18, area.bottom, "region-band");
    if (r.start > 0) svg("line", { x1: x0, x2: x0, y1: area.top - 18, y2: area.bottom, class: "region-edge" }, g);
    if (textWidth(r.name, 11) + 10 < x1 - x0) {
      label(g, (x0 + x1) / 2, area.top - 6, r.name, { class: "region-label", "text-anchor": "middle" });
    }
  }
}

function lrCurve(root, model, time, y, area) {
  svg("path", { d: areaPath(model.points, time.x, y, area.bottom), class: "lr-area" }, root);
  svg("path", { d: linePath(model.points, time.x, y), class: "lr-line" }, root);
}

function peakLabel(root, model, time, y, area) {
  const top = Math.max(...model.points.map((p) => p[1]));
  const at = model.points.find((p) => p[1] >= top * 0.9999);
  if (!at) return;
  const x = time.x(Math.max(at[0], model.warmup));
  const text = `peak ${sci(model.peak)}`;
  const right = x + 10 + textWidth(text, 11) > area.right;
  label(root, right ? x - 10 : x + 10, y(top) - 8, text, { class: "chart-annotation", "text-anchor": right ? "end" : "start" });
}

function lrHover(root, ctx, model, time, y, area) {
  const pts = model.points;
  const xs = pts.map((p) => time.x(p[0]));
  const nearest = (px) => xs.reduce((best, x, i) => (Math.abs(x - px) < Math.abs(xs[best] - px) ? i : best), 0);
  crosshair(root, area, {
    tip: ctx.tip,
    snap: (px) => (pts.length ? xs[nearest(px)] : null),
    title: (px) => stepTitle(ctx, pts[nearest(px)][0]),
    rows: (px) => lrRows(model, pts[nearest(px)]),
  });
}

function lrRows(model, [step, value]) {
  const region = model.regions.find((r) => r.start <= step && step < r.stop);
  const rows = [{ value: sci(value), name: "learning rate", color: "var(--series-1)" }];
  if (model.peak) rows.push({ value: `${Number((value / model.peak).toPrecision(3))}×`, name: "the peak" });
  if (region) rows.push({ value: region.name, name: `steps ${grouped(region.start)}–${grouped(region.stop)}` });
  return rows;
}

// --- Dragging the warmup and the cooldown ---------------------------------------------

/** Draw the handles that map to one setting each: the warmup's end, and the start of a
 *  wsd or power cooldown. */
function handles(root, ctx, model, time, y, area) {
  const decay = model.regions.find((r) => r.name === "decay");
  handle(root, ctx, { model, time, y, area, step: model.warmup, kind: "warmup" });
  if (decay && ["wsd", "power"].includes(ctx.run.schedule.type)) {
    handle(root, ctx, { model, time, y, area, step: decay.start, kind: "decay" });
  }
}

function handle(root, ctx, h0) {
  const { model, time, y, area, step, kind } = h0;
  const x = time.x(step);
  const g = svg("g", { class: "lr-handle", "aria-label": `drag the ${kind}` }, root);
  svg("line", { x1: x, x2: x, y1: area.top - 18, y2: area.bottom, class: "handle-line" }, g);
  svg("rect", { x: x - 12, y: area.top - 18, width: 24, height: area.bottom - area.top + 18, class: "chart-hit handle-hit" }, g);
  svg("circle", { cx: x, cy: y(valueAt(model.points, step)), r: 5, class: "handle-knob" }, g);
  g.addEventListener("pointerenter", (e) => ctx.tip.show(e, [{ value: handleText(ctx, kind, step), name: "drag to change" }]));
  g.addEventListener("pointerleave", () => ctx.tip.hide());
  draggable(g, {
    move: (px) => dragTo(ctx, h0, px, false),
    end: (px) => dragTo(ctx, h0, px, true),
  });
}

function dragTo(ctx, { model, time, kind }, px, release) {
  const { steps, written } = dragged(ctx, model, kind, time.step(px));
  ctx.state.draft = kind === "warmup" ? withWarmup(model, steps) : withDecay(model, steps);
  ctx.tip.hide();
  ctx.state.redrawLR?.();
  if (release) emit(ctx, [{ op: "set", target: { setting: "schedule" }, key: kind, value: written }]);
}

/** Return where a drag to `step` lands, snapped to what it writes: the steps, and the
 *  setting's new text. */
function dragged(ctx, model, kind, step) {
  const total = model.steps;
  if (kind === "decay") {
    const start = clamp(Math.round(step), model.warmup + 1, total - 1);
    const share = Number(((total - start) / total).toPrecision(2));
    return { steps: total - Math.round(share * total), written: String(share) };
  }
  const limit = (model.regions.find((r) => r.name === "decay")?.start ?? total) - 1;
  const raw = clamp(step, 0, limit);
  if (isShare(writtenWarmup(ctx.run.schedule.text))) {
    const share = Number((raw / total).toPrecision(2));
    return { steps: Math.round(share * total), written: String(share) };
  }
  const grain = 10 ** Math.max(0, Math.floor(Math.log10(total)) - 2);
  const steps = Math.round(raw / grain) * grain;
  return { steps, written: String(steps) };
}

function isShare(written) {
  const v = Number(written);
  return written !== null && v > 0 && v < 1;
}

function handleText(ctx, kind, step) {
  const total = ctx.run.budget.steps;
  if (kind === "decay") return `decay ${Number(((total - step) / total).toPrecision(2))} of the steps`;
  return `warmup ${grouped(step)} steps`;
}

/** Return `model` with its warmup ending at `w`: the warmup a straight ramp, what came
 *  after it stretched to start at `w`, up to the end of the region after it. */
export function withWarmup(model, w) {
  const old = model.warmup;
  const next = model.regions.find((r) => r.start === old && r.name !== "warmup");
  const from = [old, next?.stop ?? model.steps, model.steps];
  const to = [w, from[1], model.steps];
  const after = model.points.filter(([s]) => s >= old).map(([s, v]) => [shifted(s, from, to), v]);
  const peak = after[0]?.[1] ?? model.peak;
  const ramp = w > 0 ? [[0, peak / w], [Math.max(0, w - 1), peak]] : [];
  const regions = model.regions.filter((r) => r.name !== "warmup").map((r) => warpRegion(r, from, to));
  if (w > 0) regions.unshift({ name: "warmup", start: 0, stop: w });
  return { ...model, points: [...ramp, ...after], regions, warmup: w };
}

/** Return `model` with its cooldown starting at `d`, the stable part stretched to it. */
export function withDecay(model, d) {
  const decay = model.regions.find((r) => r.name === "decay");
  const from = [model.warmup, decay.start, model.steps];
  const to = [model.warmup, d, model.steps];
  const points = model.points.map(([s, v]) => [s < model.warmup ? s : shifted(s, from, to), v]);
  const regions = model.regions.map((r) => (r.name === "warmup" ? r : warpRegion(r, from, to)));
  return { ...model, points, regions };
}

function warpRegion(r, from, to) {
  return { ...r, start: warp(r.start, from, to), stop: warp(r.stop, from, to) };
}

/** Move the point at step `s`: a schedule goes by how far through its part a step's
 *  end, s + 1, is. */
function shifted(s, from, to) {
  return warp(s + 1, from, to) - 1;
}

/** Move `s` piecewise linearly from the breakpoints `from` to `to`. */
function warp(s, from, to) {
  for (let i = 0; i < from.length - 1; i++) {
    if (s <= from[i + 1]) {
      const t = (s - from[i]) / (from[i + 1] - from[i] || 1);
      return to[i] + Math.max(0, t) * (to[i + 1] - to[i]);
    }
  }
  return s;
}

function valueAt(points, step) {
  let best = points[0];
  for (const p of points) if (Math.abs(p[0] - step) < Math.abs(best[0] - step)) best = p;
  return best?.[1] ?? 0;
}

function clamp(v, lo, hi) {
  return Math.min(Math.max(v, lo), Math.max(lo, hi));
}

// --- Curriculum -----------------------------------------------------------------------

function curriculumCard(ctx) {
  const { run } = ctx;
  const c = card(ctx.grid, { title: "Curriculum", wide: true, onTitle: inspect(ctx, "curriculum"), subtitle: curriculumSubtitle(run) });
  if (!run.phases.length && !run.ramp) {
    const s = run.stretches[0] ?? { batch: run.budget.batch, seq_len: run.budget.seq_len };
    h("div", { cls: "card-note", text: `One shape throughout: ${grouped(s.batch)} rows of ${grouped(s.seq_len)} tokens a step.` }, c.body);
    h("div", { cls: "card-hint", text: "curriculum = phase(at=0.9, seq_len=8192, batch=64) reads the last tenth at 8k tokens a row; batch_ramp = ramp(start=64, steps=2k) grows the batch." }, c.body);
    return;
  }
  const root = svg("svg", { class: "chart", role: "img", "aria-label": "the curriculum over the steps" }, c.body);
  drawCurriculum(root, ctx, c.body.clientWidth || 800);
  sourceLegend(c.body, ctx);
}

function curriculumSubtitle(run) {
  const parts = [];
  if (run.phases.length) parts.push(`${run.phases.length} phase${run.phases.length > 1 ? "s" : ""}`);
  if (run.ramp) parts.push(`batch ramp from ${run.ramp.start} rows over ${grouped(run.ramp.steps)} steps`);
  return parts.join(" · ") || "none";
}

/** The lanes of the curriculum: what each stretch takes. */
function lanes(run) {
  const found = [
    { name: "seq_len", value: (s) => s.seq_len, fmt: grouped },
    { name: "batch", value: (s) => s.batch, fmt: grouped },
    { name: "tokens / step", value: (s) => s.tokens_per_step, fmt: compact },
  ];
  return found.filter((lane) => lane.name !== "seq_len" || varies(run, lane.value) || run.phases.length);
}

function varies(run, value) {
  return new Set(run.stretches.map(value)).size > 1;
}

function drawCurriculum(root, ctx, width) {
  const { run } = ctx;
  const time = timeline(ctx, width);
  const all = lanes(run);
  const data = varies(run, (s) => s.data);
  const [strip, laneH, dataH] = [24, 44, 40];
  const bottom = strip + all.length * laneH + (data ? dataH + 8 : 0);
  sized(root, width, bottom + 26);
  const area = { left: time.left, right: time.right, top: strip, bottom };
  rampBand(root, run, time, area);
  all.forEach((lane, i) => drawLane(root, run, lane, time, strip + i * laneH, laneH));
  if (data) dataLane(root, ctx, time, strip + all.length * laneH + 8, dataH);
  axisBottom(root, time.ticks, { y: bottom, left: area.left, right: area.right });
  unitLabel(root, time, bottom);
  curriculumHover(root, ctx, time, area);
  phaseMarkers(root, ctx, time, area);
}

function rampBand(root, run, time, area) {
  if (!run.ramp) return;
  const [x0, x1] = [time.x(0), time.x(run.ramp.steps)];
  band(root, x0, x1, area.top - 20, area.bottom, "region-band");
  if (textWidth("ramp", 11) + 8 < x1 - x0) label(root, (x0 + x1) / 2, area.top - 8, "ramp", { class: "region-label", "text-anchor": "middle" });
}

/** Merge the stretches into runs of one value of `value`. */
function runsOf(stretches, value) {
  const found = [];
  for (const s of stretches) {
    const v = value(s);
    if (found.length && found.at(-1).value === v) found.at(-1).stop = s.stop;
    else found.push({ start: s.start, stop: s.stop, value: v });
  }
  return found;
}

function drawLane(root, run, lane, time, top, height) {
  const segs = runsOf(run.stretches, lane.value);
  const values = segs.map((s) => s.value);
  const [lo, hi] = [Math.min(...values), Math.max(...values)];
  const y = hi > lo ? log([lo, hi], [top + height - 8, top + 16]) : () => top + height / 2 + 4;
  svg("line", { x1: time.left, x2: time.right, y1: top + height + 0.5, y2: top + height + 0.5, class: "chart-grid" }, root);
  label(root, GUTTER - 10, top + height / 2 + 4, lane.name, { class: "lane-label", "text-anchor": "end" });
  svg("path", { d: stepPath(segs, time.x, y), class: "lane-line" }, root);
  for (const s of segs) laneValue(root, s, lane, time, y);
}

function laneValue(root, s, lane, time, y) {
  const text = lane.fmt(s.value);
  const [x0, x1] = [time.x(s.start), time.x(s.stop)];
  if (textWidth(text, 11) + 8 > x1 - x0) return;
  label(root, x0 + 4, y(s.value) - 5, text, { class: "lane-value" });
}

function dataLane(root, ctx, time, top, height) {
  label(root, GUTTER - 10, top + height / 2 + 4, "data", { class: "lane-label", "text-anchor": "end" });
  for (const s of runsOf(ctx.run.stretches, (st) => st.data)) {
    const [x0, x1] = [time.x(s.start) + (s.start > 0 ? 1 : 0), time.x(s.stop) - 1];
    let at = top;
    for (const part of parseMixture(s.value)) {
      const hgt = part.weight * height;
      svg("rect", { x: x0, y: at, width: Math.max(0, x1 - x0), height: Math.max(0, hgt - 2), fill: ctx.colors(part.name), class: "data-band" }, root);
      at += hgt;
    }
  }
}

function sourceLegend(body, ctx) {
  if (!varies(ctx.run, (s) => s.data)) return;
  const names = [...new Set(ctx.run.stretches.flatMap((s) => parseMixture(s.data).map((p) => p.name)))];
  const legend = h("div", { cls: "inline-legend" }, body);
  for (const name of names) {
    const item = h("span", { cls: "inline-legend-item" }, legend);
    h("span", { cls: "swatch", style: `background:${ctx.colors(name)}` }, item);
    h("span", { text: name }, item);
  }
}

/** Draw a marker at each phase's start, labeled as long as the labels have room; a
 *  click opens the curriculum in the inspector. */
function phaseMarkers(root, ctx, time, area) {
  const { phases } = ctx.run;
  const xs = phases.map((p) => time.x(p.start));
  phases.forEach((p, i) => {
    const g = svg("g", { class: "phase" }, root);
    const x = Math.round(xs[i]) + 0.5;
    svg("line", { x1: x, x2: x, y1: area.top - 20, y2: area.bottom, class: "phase-marker" }, g);
    const room = (xs[i + 1] ?? time.right + 1) - xs[i] - 8;
    const text = [`phase ${p.n} · step ${grouped(p.start)}`, `phase ${p.n}`, `${p.n}`]
      .find((t) => textWidth(t, 11) <= room);
    if (text) label(g, x + 4, area.top - 8, text, { class: "phase-label" });
    const width = Math.max(12, text ? textWidth(text, 11) + 12 : 12);
    const hit = band(g, x - 6, x - 6 + width, area.top - 22, area.top - 2, "chart-hit");
    hit.addEventListener("click", () => ctx.options.onInspect?.({ setting: "curriculum" }));
    hit.addEventListener("pointermove", (e) => ctx.tip.show(e, phaseRows(p), `phase ${p.n} · from step ${grouped(p.start)}`));
    hit.addEventListener("pointerleave", () => ctx.tip.hide());
  });
}

function phaseRows(p) {
  const rows = [["seq_len", p.seq_len], ["batch", p.batch], ["data", p.data]]
    .filter(([, v]) => v !== null && v !== undefined)
    .map(([name, v]) => ({ value: typeof v === "number" ? grouped(v) : v, name, mono: name === "data" }));
  return rows.length ? rows : [{ value: "as before", name: "changes nothing" }];
}

function curriculumHover(root, ctx, time, area) {
  const steps = ctx.run.budget.steps;
  const stepOf = (px) => clamp(Math.floor(time.step(px)), 0, steps - 1);
  const at = (px) => ctx.run.stretches.find((s) => s.start <= stepOf(px) && stepOf(px) < s.stop);
  crosshair(root, area, {
    tip: ctx.tip,
    snap: (px) => time.x(stepOf(px) + 0.5),
    title: (px) => stepTitle(ctx, stepOf(px)),
    rows: (px) => stretchRows(at(px)),
  });
}

function stretchRows(s) {
  if (!s) return [];
  return [
    { value: grouped(s.seq_len), name: "seq_len" },
    { value: grouped(s.batch), name: "batch" },
    { value: compact(s.tokens_per_step), name: "tokens a step" },
    { value: s.data, name: "data", mono: true },
  ];
}

// --- Evals and checkpoints ------------------------------------------------------------

function eventsCard(ctx) {
  const rows = eventRows(ctx.run);
  const c = card(ctx.grid, { title: "Evals & checkpoints", wide: true, onTitle: inspect(ctx, rows[0]?.setting ?? "eval"), subtitle: eventsSubtitle(rows) });
  if (!rows.length) {
    h("div", { cls: "card-note", text: "No evals and no checkpoints." }, c.body);
    h("div", { cls: "card-hint", text: "eval = eval(data=web(split=[0.99, 1]), every=1k) logs a held-out loss; checkpoint = checkpoint(every=2k) saves the run." }, c.body);
    return;
  }
  const root = svg("svg", { class: "chart", role: "img", "aria-label": "evals and checkpoints over the steps" }, c.body);
  drawEvents(root, ctx, rows, c.body.clientWidth || 800);
}

function eventRows(run) {
  const rows = run.evals.map((e) => ({ name: e.name, every: e.every, setting: "eval" }));
  if (run.checkpoint) rows.push({ name: "checkpoint", every: run.checkpoint.every, setting: "checkpoint" });
  return rows;
}

function eventsSubtitle(rows) {
  return rows.map((r) => `${r.name} every ${compact(r.every)}`).join(" · ");
}

function drawEvents(root, ctx, rows, width) {
  const time = timeline(ctx, width);
  const laneH = 26;
  const bottom = 6 + rows.length * laneH;
  sized(root, width, bottom + 26);
  rows.forEach((row, i) => eventLane(root, ctx, row, time, 6 + i * laneH, laneH));
  axisBottom(root, time.ticks, { y: bottom, left: time.left, right: time.right });
  unitLabel(root, time, bottom);
}

/** The steps an eval or a save happens at: every `every`, and after the last step. */
function eventSteps(every, steps) {
  const found = [];
  for (let s = every; s < steps; s += every) found.push(s);
  return [...found, steps];
}

/** Return the steps to draw a tick at: the last one only if it is clear of the one
 *  before it. */
function drawnSteps(steps, time) {
  const n = steps.length;
  if (n < 2 || time.x(steps[n - 1]) - time.x(steps[n - 2]) >= 5) return steps;
  return steps.slice(0, -1);
}

function eventLane(root, ctx, row, time, top, height) {
  const steps = eventSteps(row.every, ctx.run.budget.steps);
  const name = label(root, GUTTER - 10, top + height / 2 + 4, row.name, { class: "lane-label clickable", "text-anchor": "end" });
  name.addEventListener("click", () => ctx.options.onInspect?.({ setting: row.setting }));
  const gap = (time.right - time.left) / Math.max(1, steps.length);
  const mid = top + height / 2;
  if (gap < 4) band(root, time.left, time.right, mid - 5, mid + 5, "event-dense");
  else for (const s of drawnSteps(steps, time)) svg("line", { x1: time.x(s), x2: time.x(s), y1: mid - 6, y2: mid + 6, class: "event-tick" }, root);
  const hit = band(root, time.left, time.right, top, top + height, "chart-hit");
  hit.addEventListener("pointermove", (e) => eventTip(ctx, e, row, steps, time));
  hit.addEventListener("pointerleave", () => ctx.tip.hide());
}

function eventTip(ctx, event, row, steps, time) {
  const px = event.clientX - event.target.ownerSVGElement.getBoundingClientRect().left;
  const step = time.step(px);
  const near = steps.reduce((a, b) => (Math.abs(b - step) < Math.abs(a - step) ? b : a));
  const what = row.setting === "checkpoint" ? "saves" : "evaluates";
  ctx.tip.show(event, [{ value: `step ${grouped(near)}`, name: `${what}, every ${grouped(row.every)}` }], row.name);
}

// --- All settings ---------------------------------------------------------------------

function settingsCard(ctx) {
  const c = card(ctx.grid, { title: "All settings", wide: true, subtitle: settingsSubtitle(ctx.run.settings) });
  const groups = h("div", { cls: "settings-groups" }, c.body);
  for (const [group, items] of groupBy(ctx.run.settings)) {
    const box = h("div", { cls: "settings-group" }, groups);
    h("div", { cls: "settings-heading", text: group }, box);
    for (const s of items) settingRow(box, ctx, s);
  }
}

function settingsSubtitle(settings) {
  const written = settings.filter((s) => s.written).length;
  return `${written} written, ${settings.length - written} at their defaults`;
}

function groupBy(settings) {
  const found = new Map();
  for (const s of settings) {
    if (!found.has(s.group)) found.set(s.group, []);
    found.get(s.group).push(s);
  }
  return found;
}

function settingRow(box, ctx, s) {
  const row = h("button", { cls: "setting" + (s.written ? " written" : ""), title: `${s.name} = ${s.text}` }, box);
  h("span", { cls: "setting-name", text: s.name }, row);
  h("span", { cls: "setting-value", text: s.text }, row);
  row.addEventListener("click", () => ctx.options.onInspect?.({ setting: s.name }));
}

// --- Helpers --------------------------------------------------------------------------

/** Split a stretch's data, "0.5 * web + 0.5 * math", into its sources and their shares;
 *  anything else is one source, all of it. */
export function parseMixture(text) {
  const terms = splitTop(text ?? "", "+").map((t) => /^\s*([0-9.eE+-]+)\s*\*\s*(.+?)\s*$/.exec(t));
  if (terms.length < 2 || terms.some((m) => !m)) return [{ name: (text ?? "").trim(), weight: 1 }];
  const total = terms.reduce((sum, m) => sum + Number(m[1]), 0) || 1;
  return terms.map((m) => ({ name: m[2], weight: Number(m[1]) / total }));
}

/** Split `text` at `sep` outside brackets and strings. */
export function splitTop(text, sep) {
  const found = [];
  let [depth, quote, start] = [0, null, 0];
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quote) quote = ch === quote ? null : quote;
    else if (ch === '"' || ch === "'") quote = ch;
    else if ("([{".includes(ch)) depth++;
    else if (")]}".includes(ch)) depth--;
    else if (ch === sep && depth === 0) {
      found.push(text.slice(start, i));
      start = i + 1;
    }
  }
  return [...found, text.slice(start)];
}

function inspect(ctx, setting) {
  return ctx.options.onInspect ? () => ctx.options.onInspect({ setting }) : null;
}

function emit(ctx, ops) {
  ctx.options.onEdit?.(ops);
}
