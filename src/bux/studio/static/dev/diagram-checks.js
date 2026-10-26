/* Interaction checks of the diagram, run by the harness with ?checks=1: each
   drives the real DOM with synthetic events and logs "check PASS|FAIL name". */

const calls = [];
let diagram = null;

function fire(el, type, init = {}) {
  const Event = type.startsWith("pointer") ? PointerEvent : type === "wheel" ? WheelEvent : type.startsWith("key") ? KeyboardEvent : MouseEvent;
  el.dispatchEvent(new Event(type, { bubbles: true, cancelable: true, button: 0, pointerId: 1, ...init }));
}

function click(el) {
  fire(el, "pointerdown");
  fire(el, "pointerup");
  fire(el, "click");
}

const $ = (sel) => diagram.el.querySelector(sel);
const last = (op) => calls.filter((c) => c[0] === op).at(-1)?.[1];
const key = (k) => fire(diagram.el, "keydown", { key: k });

const CHECKS = {
  "click selects": () => {
    click($('[data-id="n.1.0.0.0.1"]'));
    return last("select") === "n.1.0.0.0.1" && $(".dg-ring:not(.dg-ring-soft)") !== null;
  },
  "other uses get a soft ring": () => {
    click($('[data-id="n.1.0.0.0.0"]'));
    return diagram.rings.querySelectorAll(".dg-ring-soft").length === 2;
  },
  "arrow up walks the flow": () => {
    key("ArrowUp");
    return last("select") === "n.1.0.0.0.1";
  },
  "delete sends an op": () => {
    key("Delete");
    return last("action")?.op === "delete" && last("action").node === "n.1.0.0.0.1";
  },
  "escape clears": () => {
    key("Escape");
    return last("select") === null && !$(".dg-ring");
  },
  "toolbar wraps": () => {
    click($('[data-id="n.1.0.1.0.1"]'));
    click(diagram.ui.toolbar.el.querySelectorAll(".dg-btn")[2]);
    return last("action")?.op === "wrap" && last("action").node === "n.1.0.1.0.1";
  },
  "a residual's toolbar unwraps": () => {
    click($('.dg-merge[data-id="n.1.0.1"]'));
    const title = diagram.ui.toolbar.el.querySelectorAll(".dg-btn")[2].title;
    click(diagram.ui.toolbar.el.querySelectorAll(".dg-btn")[2]);
    return title.startsWith("Unwrap") && last("action")?.op === "unwrap" && last("action").node === "n.1.0.1";
  },
  "insert above picks from the palette": () => {
    click($('[data-id="n.1.0.1.0.1"]'));
    click(diagram.ui.toolbar.el.querySelectorAll(".dg-btn")[0]);
    const input = diagram.ui.palette.input;
    input.value = "mamba";
    input.dispatchEvent(new Event("input"));
    fire(input, "keydown", { key: "Enter" });
    const op = last("action");
    return op?.op === "insert" && op.where === "after" && op.text.startsWith("mamba2") && diagram.ui.palette.el.hidden;
  },
  "double-click edits a repeat": () => {
    fire($(".dg-badge[data-repeat]"), "dblclick");
    const input = diagram.ui.count.el;
    const shown = !input.hidden;
    input.value = "16";
    fire(input, "keydown", { key: "Enter" });
    const op = last("action");
    return shown && op?.op === "repeat" && op.node === "n.1" && op.times === 16;
  },
  "a caption folds its block": () => {
    click($('.dg-caption[data-toggle="block#0"]'));
    const folded = $('.dg-card[data-key="block#0"]') !== null;
    click($('.dg-card-toggle[data-toggle="block#0"]'));
    return folded && $('.dg-caption[data-toggle="block#0"]') !== null;
  },
  "collapse and expand all": () => {
    diagram.collapseAll();
    const cards = diagram.el.querySelectorAll(".dg-card").length;
    diagram.expandAll();
    return cards === 1 && diagram.el.querySelectorAll(".dg-card").length === 0;
  },
  "zoom opens the internals": () => {
    click($('[data-zoom="n.1.0.1.0.1"]'));
    const open = $(".dg-zoom-panel") !== null;
    click($('.dg-panel-close[data-zoom="n.1.0.1.0.1"]'));
    return open && $(".dg-zoom-panel") === null;
  },
  "drag pans and does not select": () => {
    const before = { ...diagram.view };
    const box = $('[data-id="n.0"]');
    const n = calls.length;
    fire(box, "pointerdown", { clientX: 100, clientY: 100 });
    fire(diagram.svg, "pointermove", { clientX: 160, clientY: 130 });
    fire(diagram.svg, "pointerup", { clientX: 160, clientY: 130 });
    fire(box, "click");
    return diagram.view.x === before.x + 60 && diagram.view.y === before.y + 30 && calls.length === n;
  },
  "wheel zooms around the pointer": () => {
    const k = diagram.view.k;
    fire(diagram.el, "wheel", { deltaY: -100, clientX: 300, clientY: 300 });
    return diagram.view.k > k;
  },
  "render keeps the view": () => {
    const before = JSON.stringify(diagram.view);
    diagram.render(diagram.model);
    return JSON.stringify(diagram.view) === before;
  },
  "facts fill the header": () => {
    diagram.setFacts({ params: 8030261248, params_text: "8.03B", flops_per_token: 4.8e10, context: 8192 });
    const header = diagram.content.querySelector(".dg-header").textContent;
    return header.includes("8.03B") && header.includes("48G") && header.includes("8,192");
  },
  "export is standalone": () => {
    const svg = diagram.exportSVG();
    const doc = new DOMParser().parseFromString(svg, "image/svg+xml");
    const ok = !doc.querySelector("parsererror") && svg.includes("Grouped-query attention");
    return ok && !/data-|class=|var\(--/.test(svg) && !svg.includes("dg-zoom");
  },
  "png export": async () => {
    const png = await diagram.exportPNG(1);
    return png?.type === "image/png" && png.size > 10000;
  },
  "odd graphs render": () => {
    for (const graph of ODD) diagram.render({ name: "odd", facts: {}, graph });
    return diagram.content.querySelectorAll(".dg-box").length > 0;
  },
  "hovering a tap lights its namesakes": async () => {
    const model = await (await fetch("fixtures/model-qwen3_next.json")).json();
    diagram.folds = { mode: null, keys: new Map() };
    diagram.render(model);
    fire($('.dg-tag[data-tap="h"]'), "pointerover");
    return diagram.el.querySelectorAll(".dg-tag.dg-hot").length === 2;
  },
  "select finds a block folded into a run": () => {
    diagram.select("n.1.0.2.0.0.1");
    return diagram.drawnId("n.1.0.2.0.0.1") === "n.1.0.0.0.0.1" && $(".dg-ring") !== null;
  },
  "an internals id selects its box": () => {
    diagram.select("n.1.0.0.0.0.1:i.0.1");
    return diagram.drawnId("n.1.0.0.0.0.1:i.0.1") === "n.1.0.0.0.0.1";
  },
  "select finds what a folded card holds": () => {
    diagram.select("n.5.2.1.0.1");
    return diagram.drawnId("n.5.2.1.0.1") === "n.5.2";
  },
};

const L = (id, type, extra = {}) => ({ id, kind: "layer", label: null, type, editable: true, ...extra });
const ODD = [
  L("n", "gqa"),
  { id: "n", kind: "seq", items: [L("n.0", "rmsnorm"), { id: "n.1", kind: "identity" }] },
  { id: "n", kind: "sum", terms: [{ id: "n.0", kind: "const", value: 1 }, L("n.1", "mlp"), { id: "n.2", kind: "neg", body: { id: "n.2.0", kind: "ref", name: "h" } }] },
  { id: "n", kind: "prod", factors: [{ id: "n.0", kind: "identity" }, { id: "n.1", kind: "const", value: 0.5 }] },
  { id: "n", kind: "seq", items: [L("n.0", "head"), L("n.1", "mlp"), { id: "n.2", kind: "repeat", times: 2, body: { id: "n.2.0", kind: "tap", name: "x" } }] },
  { id: "n", kind: "seq", label: "model", items: [{ id: "n.0", kind: "residual", body: { id: "n.0.0", kind: "seq", label: "b", items: [L("n.0.0.0", "gqa")] } }] },
];

export async function runChecks(d, model) {
  diagram = d;
  Object.assign(d.on, {
    select: (id) => calls.push(["select", id]),
    action: (op) => calls.push(["action", op]),
    hover: (id) => calls.push(["hover", id]),
  });
  d.render(model);
  let failed = 0;
  for (const [name, check] of Object.entries(CHECKS)) {
    let ok = false;
    try {
      ok = await check();
    } catch (e) {
      console.log(`check ERROR ${name}: ${e.message}`);
    }
    failed += !ok;
    console.log(`check ${ok ? "PASS" : "FAIL"} ${name}`);
  }
  console.log(`check DONE ${failed} failed`);
}
