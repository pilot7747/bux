// The little Markdown that hovers and config docs use: paragraphs, "- " lists,
// fenced code, `code`, **bold** and [links](url) (shown as their text).

import { escape } from "./syntax.js";

/** Render Markdown to HTML. */
export function markdown(text) {
  return blocks(text ?? "").map(block).join("");
}

function blocks(text) {
  const out = [];
  let fence = null;
  for (const line of text.replace(/\r/g, "").split("\n")) {
    if (line.startsWith("```")) {
      fence = fence ? (out.push({ code: fence.join("\n") }), null) : [];
    } else if (fence) {
      fence.push(line);
    } else if (!line.trim()) {
      out.push(null);
    } else {
      out.push(line);
    }
  }
  return group(out);
}

function group(lines) {
  const out = [];
  let para = [];
  const flush = () => para.length && (out.push(para), (para = []));
  for (const line of lines) {
    if (line === null || typeof line === "object") {
      flush();
      if (line) out.push(line);
    } else {
      para.push(line);
    }
  }
  flush();
  return out;
}

function block(b) {
  if (b.code !== undefined) return `<pre><code>${escape(b.code)}</code></pre>`;
  if (b.every((l) => /^\s*[-*] /.test(l))) {
    return `<ul>${b.map((l) => `<li>${inline(l.replace(/^\s*[-*] /, ""))}</li>`).join("")}</ul>`;
  }
  return `<p>${inline(b.join(" "))}</p>`;
}

/** Render inline Markdown: code spans (RST's ``double`` too), bold and links. */
export function inline(text) {
  return text
    .split(/(``[^`]+``|`[^`]+`)/)
    .map((part) =>
      /^`.+`$/.test(part)
        ? `<code>${escape(part.replace(/^``?|``?$/g, ""))}</code>`
        : escape(part)
            .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
            .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1"),
    )
    .join("");
}
