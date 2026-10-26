// Highlighting of .bux source: a small tokenizer that mirrors bux.lang.syntax.

const NUMBER = /\d[\d_]*(\.\d[\d_]*)?([eE][+-]?\d+)?(?:[kKMBT](?![A-Za-z0-9_]))?/y;
const NAME = /[A-Za-z_][A-Za-z0-9_]*/y;
const OPS = ["->", "//", "→", "|", ":", "+", "-", "*", "/", "^", "=", ",", "(", ")", "[", "]", "@", "$", "!", ".", ";"];
const KEYWORDS = new Set(["true", "false", "none", "shared", "use", "_"]);

// The configs a file writes as values (`rope(...)`, `adamw(...)`), which the session's
// palette of layers and activations does not list: TYPES minus LAYERS in bux.lang.
export const CONFIGS = [
  "adamw", "attnres", "best_fit", "chat", "checkpoint", "choice", "compiler", "concat",
  "constant", "cosine", "eval", "fan_avg", "fan_in", "fan_out", "gp", "gr", "greedy",
  "halton", "halving", "hc", "indexer", "int_uniform", "layout", "linear_decay", "llama3",
  "log", "log_uniform", "loss", "mesh", "methods", "mhc", "multistep", "muon", "normal",
  "observe", "phase", "pi", "piece", "piecewise", "power", "precision", "profile", "ramp",
  "random", "relbias", "remat", "roles", "rope", "search", "sobol", "switch", "synthetic",
  "tensorboard", "text", "tokenizer", "tokens", "tpe", "uniform", "vpgate", "wandb", "wsd",
  "yarn", "zeros",
];

/** Split source into tokens `{start, end, kind}`; blanks make none. */
export function tokenize(src, builtins = new Set()) {
  const out = [];
  let i = 0;
  while (i < src.length) {
    const t = token(src, i, builtins, out.at(-1));
    if (t) out.push(t);
    i = t ? t.end : i + 1;
  }
  return out;
}

function token(src, i, builtins, prev) {
  const ch = src[i];
  if (ch === " " || ch === "\t" || ch === "\n" || ch === "\r") return null;
  if (ch === "#") return { start: i, end: lineEnd(src, i), kind: "comment" };
  if (ch === '"' || ch === "'") return string(src, i);
  if ((ch === "@" || ch === "$") && match(NAME, src, i + 1)) return tap(src, i);
  const number = match(NUMBER, src, i);
  if (number) return { start: i, end: i + number.length, kind: "number" };
  const name = match(NAME, src, i);
  if (name) return { start: i, end: i + name.length, kind: nameKind(src, i, name, builtins, prev) };
  const op = OPS.find((o) => src.startsWith(o, i));
  return op ? { start: i, end: i + op.length, kind: "op" } : null;
}

function match(re, src, i) {
  re.lastIndex = i;
  return re.exec(src)?.[0] ?? null;
}

function lineEnd(src, i) {
  const n = src.indexOf("\n", i);
  return n === -1 ? src.length : n;
}

function string(src, i) {
  const close = src.indexOf(src[i], i + 1);
  const end = close === -1 || src.slice(i, close).includes("\n") ? lineEnd(src, i) : close + 1;
  return { start: i, end, kind: "string" };
}

function tap(src, i) {
  let end = i + 1 + match(NAME, src, i + 1).length;
  if (src[i] === "$" && src[end] === "!") end += 1;
  return { start: i, end, kind: "tap" };
}

function nameKind(src, i, name, builtins, prev) {
  if (KEYWORDS.has(name)) return "keyword";
  if (prev && src.slice(prev.start, prev.end) === "." && prev.end === i) return "field";
  const next = src.slice(i + name.length).match(/^[ \t]*(\S)(\S?)/);
  const after = next?.[1] ?? "";
  if (startsStatement(src, i, prev) && (after === "=" || after === "(" || after === ".")) return "def";
  if (after === ":") return "label";
  if (after === "=" && next[2] !== "=") return "key";
  return builtins.has(name) ? "builtin" : "name";
}

function startsStatement(src, i, prev) {
  const lineStart = src.lastIndexOf("\n", i - 1) + 1;
  if (i === lineStart) return true;
  const before = src.slice(lineStart, i);
  return /^(shared|use)\s+$/.test(before) || (prev && src.slice(prev.start, prev.end) === ";");
}

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };

/** Escape text for HTML. */
export function escape(text) {
  return text.replace(/[&<>"]/g, (c) => ESCAPES[c]);
}

/** Return the source as HTML, each token a `<span class="t-KIND">`. */
export function highlight(src, builtins) {
  let html = "";
  let at = 0;
  for (const t of tokenize(src, builtins)) {
    html += escape(src.slice(at, t.start));
    html += `<span class="t-${t.kind}">${escape(src.slice(t.start, t.end))}</span>`;
    at = t.end;
  }
  return html + escape(src.slice(at));
}
