// Building DOM in a line: h("button.primary", {onclick}, "Save").

/**
 * Make an element. `tag` may carry classes ("div.row.on"); `attrs` sets properties
 * (`onclick`, `hidden`, `title`), `dataset` entries, or attributes for the rest;
 * children are nodes, strings, or arrays of them (false and null are skipped).
 */
export function h(tag, attrs = {}, ...children) {
  const [name, ...classes] = tag.split(".");
  const el = document.createElement(name || "div");
  if (classes.length) el.className = classes.join(" ");
  for (const [k, v] of Object.entries(attrs ?? {})) set(el, k, v);
  el.append(...children.flat(Infinity).filter((c) => c !== false && c != null));
  return el;
}

function set(el, key, value) {
  if (value === undefined || value === null || value === false) return;
  if (key === "dataset") Object.assign(el.dataset, value);
  else if (key === "class") el.classList.add(...String(value).split(" ").filter(Boolean));
  else if (key in el && !key.includes("-")) el[key] = value;
  else el.setAttribute(key, value === true ? "" : value);
}

/** Make an element from trusted HTML (icons, rendered Markdown). */
export function html(tag, markup) {
  const el = h(tag);
  el.innerHTML = markup;
  return el;
}

/** The icons of the shell, 16px, drawn with currentColor. */
export const ICONS = {
  sun: `<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="3"/><path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1 1M11.6 11.6l1 1M3.4 12.6l1-1M11.6 4.4l1-1"/></svg>`,
  moon: `<svg viewBox="0 0 16 16"><path d="M13.5 9.5A5.5 5.5 0 0 1 6.5 2.5a5.5 5.5 0 1 0 7 7z"/></svg>`,
  code: `<svg viewBox="0 0 16 16"><rect x="1.5" y="2.5" width="13" height="11" rx="2"/><path d="M6 2.5v11"/></svg>`,
  panel: `<svg viewBox="0 0 16 16"><rect x="1.5" y="2.5" width="13" height="11" rx="2"/><path d="M10 2.5v11"/></svg>`,
  close: `<svg viewBox="0 0 16 16"><path d="M4.5 4.5l7 7M11.5 4.5l-7 7"/></svg>`,
  plus: `<svg viewBox="0 0 16 16"><path d="M8 3.5v9M3.5 8h9"/></svg>`,
  chevron: `<svg viewBox="0 0 16 16"><path d="M6 4l4 4-4 4"/></svg>`,
  lock: `<svg viewBox="0 0 16 16"><rect x="3.5" y="7" width="9" height="6.5" rx="1.5"/><path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2"/></svg>`,
  reset: `<svg viewBox="0 0 16 16"><path d="M3 8a5 5 0 1 0 1.6-3.7"/><path d="M3 2.5v3h3"/></svg>`,
  error: `<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6"/><path d="M8 5v3.5M8 10.8v.2"/></svg>`,
  check: `<svg viewBox="0 0 16 16"><path d="M3.5 8.5l3 3 6-7"/></svg>`,
};

/** An icon as an element. */
export function icon(name) {
  return html("span.icon", ICONS[name]);
}
