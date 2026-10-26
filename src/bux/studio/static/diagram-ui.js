/* The diagram's HTML overlays: the floating toolbar of a node, the palette of what
   to insert, the zoom and export controls, and the editor of a repeat's count. */

const ICONS = {
  above: '<path d="M3 13.5h10M8 2.5v6M5 5.5h6"/>',
  below: '<path d="M3 2.5h10M8 7.5v6M5 10.5h6"/>',
  wrap: '<path d="M5.5 3H3.5v10h2M10.5 3h2v10h-2"/><path d="M8 5.5v5M5.5 8h5"/>',
  unwrap: '<path d="M5.5 3H3.5v10h2M10.5 3h2v10h-2"/><path d="M6.5 8h3"/>',
  delete: '<path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.7 8.5h5.6l.7-8.5"/>',
  minus: '<path d="M4 8h8"/>',
  plus: '<path d="M4 8h8M8 4v8"/>',
  fit: '<path d="M3 6V3h3M10 3h3v3M13 10v3h-3M6 13H3v-3"/>',
  expand: '<path d="M5 6.5l3-3 3 3M5 9.5l3 3 3-3"/>',
  collapse: '<path d="M5 3.5l3 3 3-3M5 12.5l3-3 3 3"/>',
  svg: '<path d="M8 2.5v7M5 6.5l3 3 3-3M3 12.5h10"/>',
};

const CATEGORIES = ["attention", "recurrent", "ffn", "experts", "norm", "io", "op", "other"];

function icon(name) {
  return `<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">${ICONS[name]}</svg>`;
}

function button(name, title, onClick, text = "") {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "dg-btn";
  b.title = title;
  b.innerHTML = text ? `${icon(name)}<span>${text}</span>` : icon(name);
  b.addEventListener("click", (e) => {
    e.stopPropagation();
    onClick(b);
  });
  return b;
}

function div(className) {
  const d = document.createElement("div");
  d.className = className;
  return d;
}

/** Stops pointer and wheel events from reaching the canvas under an overlay. */
function shield(el) {
  for (const type of ["pointerdown", "wheel", "dblclick", "click"]) {
    el.addEventListener(type, (e) => e.stopPropagation());
  }
}

export class Toolbar {
  constructor(diagram) {
    this.d = diagram;
    this.el = div("dg-toolbar");
    this.el.hidden = true;
    shield(this.el);
    this.el.addEventListener("pointerenter", () => (this.held = true));
    this.el.addEventListener("pointerleave", () => {
      this.held = false;
      this.d.hoverLater(null);
    });
    diagram.el.append(this.el);
    this.id = null;
  }

  /** Shows the toolbar of `id` (a drawn, editable node), or hides it. */
  show(id) {
    const node = id && this.d.ink?.nodes.get(id);
    if (!node?.editable) {
      this.el.hidden = true;
      this.id = this.node = null;
      return;
    }
    if (node !== this.node) this.fill(node);
    Object.assign(this, { id, node });
    this.el.hidden = false;
    this.place();
  }

  fill(node) {
    const act = (op, extra = {}) => this.d.on.action({ op, node: node.id, ...extra });
    const insert = (where) => (b) => this.d.ui.palette.open(b, node.id, where);
    const residual = node.kind === "residual";
    this.el.replaceChildren(
      button("above", "Insert above", insert("after")),
      button("below", "Insert below", insert("before")),
      residual
        ? button("unwrap", "Unwrap the residual: [f] -> f", () => act("unwrap"))
        : button("wrap", "Wrap in a residual: f -> [f]", () => act("wrap")),
      button("delete", "Delete", () => act("delete")),
    );
  }

  place() {
    if (this.el.hidden || !this.id) return;
    const r = this.d.screenRect(this.id);
    if (!r) return void (this.el.hidden = true);
    const [w, h] = [this.el.offsetWidth, this.el.offsetHeight];
    const small = r.w < 40;
    const left = small ? r.x + r.w + 6 : r.x + r.w - w - 6;
    const top = small ? r.y + r.h / 2 - h / 2 : r.y - h + 6;
    this.el.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
  }
}

export class Palette {
  constructor(diagram) {
    this.d = diagram;
    this.el = div("dg-palette");
    this.el.hidden = true;
    this.input = document.createElement("input");
    this.input.type = "search";
    this.input.placeholder = "Insert a layer…";
    this.input.spellcheck = false;
    this.list = div("dg-palette-list");
    this.el.append(this.input, this.list);
    shield(this.el);
    this.input.addEventListener("input", () => this.fill());
    this.input.addEventListener("keydown", (e) => this.key(e));
    document.addEventListener("pointerdown", (e) => {
      if (!this.el.hidden && !this.el.contains(e.target)) this.close();
    });
    diagram.el.append(this.el);
  }

  open(anchor, node, where) {
    Object.assign(this, { node, where, active: 0 });
    this.input.value = "";
    this.el.hidden = false;
    this.fill();
    this.place(anchor);
    this.input.focus({ preventScroll: true });
  }

  close() {
    if (this.el.hidden) return;
    this.el.hidden = true;
    this.d.el.focus({ preventScroll: true });
  }

  place(anchor) {
    const host = this.d.el.getBoundingClientRect();
    const a = anchor.getBoundingClientRect();
    const w = this.el.offsetWidth;
    const h = this.el.offsetHeight;
    const left = Math.min(Math.max(8, a.left - host.left), host.width - w - 8);
    const below = a.bottom - host.top + 6;
    const top = below + h > host.height - 8 ? a.top - host.top - h - 6 : below;
    this.el.style.transform = `translate(${Math.round(left)}px, ${Math.round(Math.max(8, top))}px)`;
  }

  matches() {
    const q = this.input.value.trim().toLowerCase();
    const hit = (item) => !q || [item.name, item.title, item.doc].some((s) => s?.toLowerCase().includes(q));
    const items = (this.d.palette ?? []).filter(hit);
    const rank = (item) => (CATEGORIES.indexOf(item.category) + 9) % 9;
    return items.sort((a, b) => rank(a) - rank(b));
  }

  fill() {
    this.items = this.matches();
    this.active = Math.min(this.active, Math.max(0, this.items.length - 1));
    this.list.replaceChildren();
    let category = null;
    this.items.forEach((item, i) => {
      if (item.category !== category) {
        category = item.category;
        const head = div("dg-palette-group");
        head.textContent = category;
        this.list.append(head);
      }
      this.list.append(this.row(item, i));
    });
    if (!this.items.length) this.list.append(Object.assign(div("dg-palette-empty"), { textContent: "Nothing matches" }));
  }

  row(item, i) {
    const row = div(`dg-palette-item${i === this.active ? " dg-active" : ""}`);
    row.title = item.doc ?? "";
    const dot = div(`dg-dot cat-${item.category}`);
    const name = Object.assign(document.createElement("code"), { textContent: item.name });
    const title = Object.assign(document.createElement("span"), { textContent: item.title ?? "" });
    row.append(dot, name, title);
    row.addEventListener("pointerenter", () => this.activate(i));
    row.addEventListener("click", () => this.pick(item));
    return row;
  }

  activate(i) {
    this.active = i;
    this.list.querySelectorAll(".dg-palette-item").forEach((row, k) => {
      row.classList.toggle("dg-active", k === i);
      if (k === i) row.scrollIntoView({ block: "nearest" });
    });
  }

  key(e) {
    const moves = { ArrowDown: 1, ArrowUp: -1 };
    if (e.key in moves) {
      e.preventDefault();
      const n = this.items.length;
      if (n) this.activate((this.active + moves[e.key] + n) % n);
    } else if (e.key === "Enter" && this.items[this.active]) this.pick(this.items[this.active]);
    else if (e.key === "Escape") this.close();
    e.stopPropagation();
  }

  pick(item) {
    this.close();
    this.d.on.action({ op: "insert", node: this.node, where: this.where, text: item.template });
  }
}

export class Controls {
  constructor(diagram) {
    const d = diagram;
    this.el = div("dg-controls");
    shield(this.el);
    this.zoom = Object.assign(document.createElement("button"), { type: "button", className: "dg-btn dg-zoom-level", title: "Actual size" });
    this.zoom.addEventListener("click", () => d.zoomBy(1 / d.view.k));
    const wrap = (items) => {
      const g = div("dg-group");
      g.append(...items);
      return g;
    };
    this.el.append(
      wrap([button("minus", "Zoom out", () => d.zoomBy(1 / 1.25)), this.zoom, button("plus", "Zoom in", () => d.zoomBy(1.25))]),
      wrap([button("fit", "Fit to screen", () => d.fit(true))]),
      wrap([button("expand", "Expand all blocks", () => d.expandAll()), button("collapse", "Collapse all blocks", () => d.collapseAll())]),
      wrap([button("svg", "Export SVG", () => d.download("svg"), "SVG"), button("svg", "Export PNG", () => d.download("png"), "PNG")]),
    );
    d.el.append(this.el);
  }

  update(k) {
    this.zoom.textContent = `${Math.round(k * 100)}%`;
  }
}

export class CountEditor {
  constructor(diagram) {
    this.d = diagram;
    this.el = Object.assign(document.createElement("input"), { type: "text", className: "dg-count", hidden: true });
    this.el.inputMode = "numeric";
    shield(this.el);
    this.el.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter") this.commit();
      if (e.key === "Escape") this.close();
    });
    this.el.addEventListener("blur", () => this.close());
    diagram.el.append(this.el);
  }

  open(target, node) {
    const host = this.d.el.getBoundingClientRect();
    const r = target.getBoundingClientRect();
    this.node = node;
    this.el.value = target.textContent.replace("×", "");
    this.el.hidden = false;
    const w = Math.max(48, r.width + 20);
    Object.assign(this.el.style, {
      width: `${w}px`,
      transform: `translate(${Math.round(r.left - host.left + r.width / 2 - w / 2)}px, ${Math.round(r.top - host.top + r.height / 2 - 14)}px)`,
    });
    this.el.focus();
    this.el.select();
  }

  commit() {
    const times = Number(this.el.value.trim().replace(/^[×x*]/, ""));
    const node = this.node;
    this.close();
    if (Number.isInteger(times) && times >= 1) this.d.on.action({ op: "repeat", node, times });
  }

  close() {
    if (this.el.hidden) return;
    this.el.hidden = true;
    this.d.el.focus({ preventScroll: true });
  }
}
