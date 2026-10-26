// Walking the view's model tree: children, lookup by id, the node at an offset.

/** A node's children as drawn: items, terms, factors or body (not its internals). */
export function children(node) {
  return [...(node.items ?? []), ...(node.terms ?? []), ...(node.factors ?? []), ...(node.body ? [node.body] : [])];
}

/** Every node under `root` with its depth, depth first. */
export function* walk(root, depth = 0) {
  if (!root) return;
  yield [root, depth];
  for (const child of children(root)) yield* walk(child, depth + 1);
}

/** The node with an id, or null. */
export function findNode(root, id) {
  for (const [node] of walk(root)) if (node.id === id) return node;
  return null;
}

/**
 * The innermost node whose span or written holds the offset; null for none or the
 * root. A caret just after a name holds it, unless another node starts there.
 */
export function nodeAt(root, offset) {
  let best = null;
  for (const [node, depth] of walk(root)) {
    const fit = Math.min(score(node.span, offset), score(node.written, offset));
    if (fit === Infinity || node === root) continue;
    if (!best || depth > best.depth || (depth === best.depth && fit < best.fit)) best = { node, depth, fit };
  }
  return best?.node ?? null;
}

// Smaller fits better: the span's length, past every strict fit when only touched.
function score(span, offset) {
  if (!span || offset < span[0] || offset > span[1]) return Infinity;
  return span[1] - span[0] + (offset === span[1] && span[1] > span[0] ? 1e9 : 0);
}
