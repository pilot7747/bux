// The page: one source string is the state. Typing and visual edits both make a new
// source, which the server compiles into a view; the diagram, panels, inspector and
// errors draw that view. Saving is explicit; the file on disk is watched.

import { api } from "./api.js";
import { CodeEditor, shifter } from "./editor.js";
import { Inspector } from "./inspector.js";
import { History } from "./history.js";
import { findNode, nodeAt } from "./nodes.js";
import { initTheme, initPanes, Tabs, banner, toast } from "./layout.js";
import { showFile, renderRunPicker, renderErrors, renderStatus } from "./chrome.js";
import * as fallback from "./fallback.js";

const VIEW_MS = 150;
const PLAN_MS = 1200;
const POLL_MS = 2000;
const CURSOR_MS = 60;
const MODEL_KEYS = new Set(["d_model", "vocab_size", "residual", "init"]);

const $ = (id) => document.getElementById(id);
const state = {
  session: null, source: "", saved: "", mtime: 0, picks: {},
  view: null, good: null, goodSource: "", plan: null, planKey: "", planning: false,
  selected: null, target: { model: true }, cursor: null, drawn: {}, seq: { view: 0, inspect: 0, plan: 0 },
};
const ui = {};

// --- Starting --------------------------------------------------------------------------

async function boot() {
  const session = await api("/api/session");
  Object.assign(state, { session, source: session.source, saved: session.source, mtime: session.mtime });
  ui.history = new History(session.source);
  initFrame();
  ui.editor = makeEditor(session);
  ui.inspector = new Inspector($("inspector"), { onEdit: edit, onReveal: revealSpan });
  await loadComponents();
  showDirty();
  refresh();
  setInterval(poll, POLL_MS);
}

function initFrame() {
  initTheme($("theme"));
  Object.assign(ui, initPanes($("workspace"), $("resizer")));
  ui.tabs = new Tabs($("tabs"), { model: $("view-model"), training: $("view-training"), sweep: $("view-sweep") }, renderPanel);
  ui.tabs.setAvailable("training", false);
  ui.tabs.setAvailable("sweep", false);
  for (const id of ["code-toggle", "code-close"]) $(id).onclick = () => ui.toggleCode();
  for (const id of ["inspector-toggle", "inspector-close"]) $(id).onclick = () => ui.toggleInspector();
  $("save").onclick = () => save();
  if (!/Mac|iPhone|iPad/.test(navigator.platform)) $("save").querySelector("kbd").textContent = "Ctrl+S";
  window.addEventListener("keydown", shortcut, true);
  window.addEventListener("beforeunload", (e) => dirty() && e.preventDefault());
}

function makeEditor(session) {
  const editor = new CodeEditor($("editor"), {
    onChange: (source) => change(source, "typing"),
    onCursor: (offset) => cursorMoved(offset),
    complete: (offset) => api("/api/complete", { source: state.source, offset }),
    hover: (offset) => api("/api/hover", { source: state.source, offset }),
    onHistory: (kind) => step(kind),
  });
  editor.setBuiltins((session.palette ?? []).map((p) => p.name));
  editor.setSource(session.source);
  return editor;
}

// The diagram and the panels are other modules; a missing one gets a stand-in.
async function loadComponents() {
  const [diagram, training, sweep] = await Promise.all([
    load("./diagram.js", "diagram.css"), load("./training.js", "panels.css"), load("./sweep.js", "panels.css"),
  ]);
  const Diagram = diagram?.Diagram ?? fallback.OutlineDiagram;
  ui.renderTraining = training?.renderTraining ?? fallback.renderTraining;
  ui.renderSweep = sweep?.renderSweep ?? fallback.renderSweep;
  ui.diagram = makeDiagram(Diagram) ?? makeDiagram(fallback.OutlineDiagram);
}

function makeDiagram(Diagram) {
  $("view-model").replaceChildren();
  const opts = {
    onSelect: (id) => select(id, "diagram"),
    onAction: (op) => editOrToast([op]),
    onHover: () => {},
    palette: state.session.palette ?? [],
  };
  try {
    return new Diagram($("view-model"), opts);
  } catch (e) {
    console.error("bux studio: the diagram failed to start", e);
    return null;
  }
}

async function load(path, css) {
  try {
    const module = await import(path);
    if (!document.querySelector(`link[href="${css}"]`)) document.head.append(Object.assign(document.createElement("link"), { rel: "stylesheet", href: css }));
    return module;
  } catch (e) {
    console.info(`bux studio: ${path} not loaded (${e.message}); using a stand-in`);
    return null;
  }
}

// --- The source --------------------------------------------------------------------------

const dirty = () => state.source !== state.saved;

/** Take a new source: from typing, a visual edit, undo/redo or the disk. */
function change(source, how) {
  if (source === state.source) return;
  state.source = source;
  if (how !== "typing") ui.editor.setSource(source);
  if (how !== "history") ui.history.push(source, { typing: how === "typing" });
  showDirty();
  clearTimeout(state.viewTimer);
  state.viewTimer = setTimeout(refresh, how === "typing" ? VIEW_MS : 0);
}

function showDirty() {
  const { name, path } = state.session;
  showFile({ name, path, dirty: dirty() });
}

function step(kind) {
  const source = kind === "undo" ? ui.history.undo() : ui.history.redo();
  if (source !== null) change(source, "history");
}

// --- The view ------------------------------------------------------------------------------

async function refresh() {
  const seq = ++state.seq.view;
  const source = state.source;
  let view;
  try {
    view = await api("/api/view", { source, picks: state.picks });
  } catch (e) {
    if (seq === state.seq.view) toast(`The server did not answer: ${e.message}`, "error");
    return;
  }
  if (seq === state.seq.view) applyView(view, source);
}

function applyView(view, source) {
  state.view = view;
  ui.editor.setErrors(view.errors.map((e) => ({ ...e })));
  if (view.ok) Object.assign(state, { good: view, goodSource: source });
  const good = state.good;
  if (view.ok) drawModel(view.model);
  else if (!good) note("The model is drawn once the file compiles.");
  ui.diagram.setDim(!view.ok && !!good);
  renderErrors($("errors"), view.errors, { stale: !view.ok && !!good?.model, onReveal: revealError });
  drawRun(good);
  renderPanel(ui.tabs.current);
  showStatus();
  if (view.ok) (reinspect(), schedulePlan());
}

// The diagram owns its element; a file without a model gets a note over it.
function drawModel(model) {
  note(model ? null : "This file is a module: it defines no model to draw.");
  if (model) ui.diagram.render(model);
}

function note(text) {
  $("canvas-note").hidden = !text;
  $("canvas-note").textContent = text ?? "";
}

function drawRun(good) {
  const sweep = good?.sweep;
  const runs = sweep?.total ?? 0;
  ui.tabs.setAvailable("training", !!good?.run);
  ui.tabs.setAvailable("sweep", !!(sweep?.axes.length || sweep?.dimensions.length));
  ui.tabs.setBadge("training", good?.run?.budget?.tokens_text ?? null);
  ui.tabs.setBadge("sweep", runs ? `${runs}${sweep.truncated ? "+" : ""} run${runs > 1 ? "s" : ""}` : sweep?.search ? "search" : null);
  renderRunPicker($("runpicker"), sweep?.axes ?? [], pick);
}

// A panel draws again only when what it shows changed.
function renderPanel(tab) {
  const good = state.good;
  const key = JSON.stringify(tab === "training" ? good?.run : good?.sweep);
  if (state.drawn[tab] === key) return;
  state.drawn[tab] = key;
  if (tab === "training" && good?.run) {
    ui.renderTraining($("view-training"), good.run, { onEdit: editOrToast, onInspect: inspectFromPanel, schedule });
  } else if (tab === "sweep" && good?.sweep) {
    ui.renderSweep($("view-sweep"), good.sweep, { onPick: pickRun, onEdit: editOrToast, onInspect: inspectFromPanel });
  }
}

function schedule() {
  return api("/api/schedule", { source: state.goodSource, picks: state.picks });
}

function showStatus() {
  renderStatus($("status"), {
    errors: state.view?.errors, run: state.good?.run, plan: state.plan, planning: state.planning,
    cursor: state.cursor, onErrors: () => revealError(errorSpan(state.view.errors[0])),
  });
}

const errorSpan = (e) => [e.start, e.end];

// --- The run picked ----------------------------------------------------------------------

function pick(axisId, option) {
  pickRun({ ...state.picks, [axisId]: option });
}

function pickRun(picks) {
  state.picks = { ...picks };
  refresh();
}

// --- Plan: parameters, once JAX has imported on the server ----------------------------------

function schedulePlan() {
  clearTimeout(state.planTimer);
  state.planTimer = setTimeout(plan, PLAN_MS);
}

async function plan() {
  const key = state.goodSource + JSON.stringify(state.picks);
  if (key === state.planKey) return;
  const seq = ++state.seq.plan;
  state.planning = true;
  showStatus();
  const reply = await api("/api/plan", { source: state.goodSource, picks: state.picks }).catch(() => null);
  if (seq !== state.seq.plan) return;
  Object.assign(state, { plan: reply, planKey: key, planning: false });
  showStatus();
  const context = state.good?.run?.budget?.seq_len ?? null;
  const shown = context === null ? null : context.toLocaleString("en-US");
  if (reply) ui.diagram.setFacts?.({ params: reply.params_text, context: shown });
}

// --- Selection: the code and the diagram, linked -------------------------------------------

function cursorMoved(offset) {
  const src = state.source;
  state.cursor = position(src, offset);
  showStatus();
  clearTimeout(state.cursorTimer);
  const at = shifter(src, state.goodSource, true)(offset);
  if (at !== null) state.cursorTimer = setTimeout(() => selectAt(at), CURSOR_MS);
}

function selectAt(offset) {
  const node = nodeAt(state.good?.model?.graph, offset);
  if (node) return select(node.id, "code");
  const def = (state.good?.outline ?? []).find((o) => o.span && o.span[0] <= offset && offset <= o.span[1]);
  if (def && (def.kind === "setting" || def.kind === "value")) {
    select(null, "code", MODEL_KEYS.has(def.name) || def.kind === "value" ? { model: true } : { setting: def.name });
  }
}

/** Select a node (or none) from the code, the diagram, or a panel. */
function select(id, from, target = null) {
  const node = id ? findNode(state.good?.model?.graph, id) : null;
  const next = target ?? (node ? { node: id } : { model: true });
  const same = from === "code" && state.selected === (node ? id : null) && sameTarget(next, state.target);
  state.selected = node ? id : null;
  if (same) return;
  ui.diagram.select(state.selected);
  const move = mover(state.goodSource, state.source);
  if (from === "diagram" && node?.span) ui.editor.reveal(move(node.span));
  ui.editor.setMarks(move(node?.span), move(node?.written));
  inspect(next);
}

const sameTarget = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** Move spans of one source onto another; null for a span in text that changed. */
function mover(from, to) {
  const at = shifter(from, to, true);
  return (span) => {
    const [a, b] = span ? [at(span[0]), at(span[1])] : [null, null];
    return a === null || b === null ? null : [a, b];
  };
}

function revealSpan(span) {
  if (!span) return;
  ui.toggleCode(true);
  ui.editor.reveal(span);
  ui.editor.setMarks(span);
}

function revealError(span) {
  revealSpan(span);
  ui.editor.focus();
}

function position(src, offset) {
  const before = src.slice(0, offset);
  const line = before.split("\n").length;
  return { line, col: offset - before.lastIndexOf("\n") };
}

// --- The inspector -----------------------------------------------------------------------

async function inspect(target) {
  state.target = target;
  const seq = ++state.seq.inspect;
  const reply = await api("/api/inspect", { source: state.goodSource, picks: state.picks, target }).catch((e) => ({ error: e.body?.error ?? e.message }));
  if (seq !== state.seq.inspect) return;
  if (reply?.error) ui.inspector.showError(reply.error);
  else ui.inspector.show(reply, target);
}

function reinspect() {
  const t = state.target;
  if (t?.node && !findNode(state.good?.model?.graph, t.node)) return select(null, "view");
  inspect(t);
}

function inspectFromPanel(target) {
  ui.toggleInspector(true);
  select(null, "panel", target);
}

// --- Edits ---------------------------------------------------------------------------------

/** Apply visual edit ops; resolves to an error message, or null when applied. */
async function edit(ops) {
  const sent = state.source;
  let reply;
  try {
    reply = await api("/api/edit", { source: sent, picks: state.picks, ops });
  } catch (e) {
    return e.body?.error ?? e.message;
  }
  if (reply.error) return reply.error;
  if (state.source !== sent) return "The code changed while the edit was on its way; try again.";
  change(reply.source, "edit");
  return null;
}

async function editOrToast(ops) {
  const error = await edit(ops);
  if (error) toast(error, "error");
  return error;
}

// --- Saving and the file on disk ----------------------------------------------------------------

async function save(mtime = state.mtime) {
  const source = state.source;
  try {
    const reply = await api("/api/save", { source, mtime });
    Object.assign(state, { saved: source, mtime: reply.mtime });
    showDirty();
    toast("Saved");
  } catch (e) {
    if (e.status === 409) conflict(e.body);
    else toast(`Could not save: ${e.message}`, "error");
  }
}

function conflict(disk) {
  banner($("banners"), {
    id: "disk", kind: "danger", message: "The file changed on disk since it was opened here.",
    actions: [
      { label: "Reload from disk", run: () => loadDisk(disk.source, disk.mtime) },
      { label: "Overwrite", primary: true, run: () => save(disk.mtime) },
    ],
  });
}

async function poll() {
  if (document.hidden || state.polling) return;
  state.polling = true;
  try {
    const { mtime } = await api("/api/file");
    if (mtime !== state.mtime && mtime !== state.ignored) await diskChanged();
  } catch {
    // The server is gone or busy; the next poll tries again.
  } finally {
    state.polling = false;
  }
}

async function diskChanged() {
  const disk = await api("/api/file?source=1");
  if (disk.source === state.saved) return void (state.mtime = disk.mtime);
  if (!dirty()) return loadDisk(disk.source, disk.mtime);
  state.ignored = disk.mtime;
  banner($("banners"), {
    id: "disk", message: "The file changed on disk, and you have unsaved edits here.",
    actions: [
      { label: "Reload from disk", run: () => loadDisk(disk.source, disk.mtime) },
      { label: "Keep mine", primary: true, run: () => (state.mtime = disk.mtime) },
    ],
  });
}

function loadDisk(source, mtime) {
  Object.assign(state, { saved: source, mtime });
  change(source, "disk");
  showDirty();
}

// --- Keys --------------------------------------------------------------------------------------

function shortcut(e) {
  if (!(e.metaKey || e.ctrlKey) || e.altKey) return;
  const key = e.key.toLowerCase();
  const inField = e.target.matches?.("input, select");
  const run = {
    s: () => save(),
    z: () => !inField && step(e.shiftKey ? "redo" : "undo"),
    y: () => !inField && e.ctrlKey && step("redo"),
    "\\": () => ui.toggleCode(),
    1: () => ui.tabs.show("model"),
    2: () => ui.tabs.show("training"),
    3: () => ui.tabs.show("sweep"),
  }[key];
  if (!run || (inField && (key === "z" || key === "y"))) return;
  e.preventDefault();
  e.stopPropagation();
  run();
}

boot().catch((e) => {
  console.error(e);
  document.body.classList.add("failed");
  toast(`bux studio could not start: ${e.message}`, "error");
});
