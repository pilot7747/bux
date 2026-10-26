// The page's frame: theme, the code pane's width, collapsing panes, tabs, banners
// and toasts. Layout choices are remembered in localStorage.

import { h, icon, ICONS } from "./dom.js";

const KEY = "bux-studio-";
const stored = (name, fallback) => localStorage.getItem(KEY + name) ?? fallback;
const store = (name, value) => localStorage.setItem(KEY + name, value);

/** Wire the theme toggle; the theme itself was set before the first paint. */
export function initTheme(button) {
  const draw = () => {
    const dark = document.documentElement.dataset.theme === "dark";
    button.innerHTML = dark ? ICONS.sun : ICONS.moon;
    button.title = dark ? "Light theme" : "Dark theme";
  };
  button.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    store("theme", next);
    draw();
    // For components that inline colors, such as an exported SVG.
    window.dispatchEvent(new CustomEvent("bux-theme", { detail: next }));
  });
  draw();
}

/** Make the side panes collapsible and the code pane resizable; returns toggles. */
export function initPanes(workspace, resizer) {
  const width = Number(stored("code-width", 440));
  workspace.style.setProperty("--code-w", `${width}px`);
  for (const pane of ["code", "inspector"]) workspace.classList.toggle(`no-${pane}`, stored(`no-${pane}`, "0") === "1");
  const toggle = (pane, show) => {
    const hidden = !(show ?? workspace.classList.contains(`no-${pane}`));
    workspace.classList.toggle(`no-${pane}`, hidden);
    store(`no-${pane}`, hidden ? "1" : "0");
    window.dispatchEvent(new Event("resize"));
  };
  resizer.addEventListener("pointerdown", (e) => drag(e, workspace, resizer));
  resizer.addEventListener("dblclick", () => setWidth(workspace, 440));
  return { toggleCode: (show) => toggle("code", show), toggleInspector: (show) => toggle("inspector", show) };
}

function drag(e, workspace, resizer) {
  e.preventDefault();
  resizer.setPointerCapture(e.pointerId);
  const left = workspace.getBoundingClientRect().left;
  const move = (ev) => setWidth(workspace, ev.clientX - left);
  const up = () => {
    resizer.removeEventListener("pointermove", move);
    document.body.classList.remove("resizing");
    window.dispatchEvent(new Event("resize"));
  };
  document.body.classList.add("resizing");
  resizer.addEventListener("pointermove", move);
  resizer.addEventListener("pointerup", up, { once: true });
}

function setWidth(workspace, px) {
  const w = Math.round(Math.max(260, Math.min(px, workspace.clientWidth * 0.6)));
  workspace.style.setProperty("--code-w", `${w}px`);
  store("code-width", w);
}

/** Tabs over views: `show(name)`, `setAvailable(name, bool)`, `setBadge(name, text)`. */
export class Tabs {
  constructor(nav, views, onShow) {
    this.nav = nav;
    this.views = views;
    this.onShow = onShow;
    this.current = "model";
    nav.addEventListener("click", (e) => {
      const tab = e.target.closest("[data-tab]");
      if (tab) this.show(tab.dataset.tab);
    });
  }

  button(name) {
    return this.nav.querySelector(`[data-tab="${name}"]`);
  }

  show(name) {
    const b = this.button(name);
    if (!b || b.hidden) return;
    this.current = name;
    for (const t of this.nav.querySelectorAll("[data-tab]")) t.setAttribute("aria-selected", String(t === b));
    for (const [n, el] of Object.entries(this.views)) el.hidden = n !== name;
    this.onShow(name);
  }

  setAvailable(name, available) {
    this.button(name).hidden = !available;
    if (!available && this.current === name) this.show("model");
  }

  setBadge(name, text) {
    const badge = this.button(name).querySelector(".badge");
    badge.textContent = text ?? "";
    badge.hidden = !text;
  }
}

/** Show a banner under the header with action buttons; returns a function that removes it. */
export function banner(host, { kind = "warning", message, actions = [], id }) {
  host.querySelector(`[data-id="${id}"]`)?.remove();
  const close = () => el.remove();
  const buttons = actions.map((a) => h(`button.btn${a.primary ? ".primary" : ""}`, { onclick: () => (close(), a.run()) }, a.label));
  const el = h(`div.banner.${kind}`, { dataset: { id: id ?? "" } }, icon("error"), h("span.msg", {}, message), h("span.grow"), buttons);
  host.append(el);
  return close;
}

/** A short message at the bottom of the page. */
export function toast(message, kind = "info") {
  const host = document.getElementById("toasts");
  const el = h(`div.toast.${kind}`, {}, kind === "error" ? icon("error") : icon("check"), message);
  host.append(el);
  setTimeout(() => el.classList.add("out"), kind === "error" ? 5000 : 2200);
  setTimeout(() => el.remove(), kind === "error" ? 5400 : 2600);
}
