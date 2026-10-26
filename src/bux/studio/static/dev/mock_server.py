"""A stand-in for ``bux studio``'s server, to develop the page without the backend.

It serves the static directory and answers ``/api/*`` from ``fixtures/``: the view
whose source is closest to the one sent (its spans moved to match), the forms made
for it, canned completion and hover, and a ``set`` that splices the value where the
form says it is written. Saving keeps the file in memory. Stdlib only::

    python3 src/bux/studio/static/dev/mock_server.py 9411 [FIXTURE]

then open ``http://127.0.0.1:9411/?t=dev``. FIXTURE is ``llama3_8b`` (the default),
``qwen3_next``, ``attention_ablation`` or ``errors``. ``POST /dev/touch`` edits the
file "on disk", as another editor would.
"""

import difflib
import json
import re
import sys
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# A JSON object, as the API sends and fixtures hold.
Json = dict[str, Any]

STATIC = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
TOKEN = "dev"


def load(name: str) -> Json:
    """Read one fixture."""
    return json.loads((FIXTURES / name).read_text())


VIEWS = {p.stem[5:]: load(p.name) for p in FIXTURES.glob("view-*.json")}
SESSION = load("session.json")
DISK: Json = {"source": "", "mtime": 0.0, "path": ""}


# --- Views -----------------------------------------------------------------------------


def closest(source: str) -> str:
    """Return the name of the view fixture whose source is most like `source`."""

    def ratio(name: str) -> float:
        return difflib.SequenceMatcher(None, VIEWS[name]["source"], source).ratio()

    return max(VIEWS, key=ratio)


def mover(old: str, new: str) -> Callable[[int], int]:
    """Return a function that moves an offset of `old` to where it is in `new`."""
    ops = difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes()

    def move(i: int) -> int:
        for _tag, a0, a1, b0, b1 in ops:
            if a0 <= i < a1 or i == a1 == len(old):
                return b0 + min(i - a0, b1 - b0)
        return len(new)

    return move


def moved(value: Any, move: Callable[[int], int]) -> Any:
    """Move every span (a ``[start, end]`` under ``span`` or ``written``) and error."""
    if isinstance(value, list):
        return [moved(v, move) for v in value]
    if not isinstance(value, dict):
        return value
    out = {k: moved(v, move) for k, v in value.items()}
    for key in ("span", "written"):
        if isinstance(out.get(key), list) and len(out[key]) == 2:
            out[key] = [move(out[key][0]), move(out[key][1])]
    if "message" in out and "start" in out:
        out["start"], out["end"] = move(out["start"]), move(out["end"])
    return out


def unbalanced(source: str) -> Json | None:
    """Return an error for the first bracket that does not close, as the parser would."""
    stack: list[int] = []
    pairs = {")": "(", "]": "["}
    text = re.sub(r"#[^\n]*|\"[^\"\n]*\"", lambda m: " " * len(m.group()), source)
    for i, ch in enumerate(text):
        if ch in "([":
            stack.append(i)
        elif ch in pairs and (not stack or text[stack.pop()] != pairs[ch]):
            return error(source, i, f"unmatched '{ch}'")
    return (
        error(source, stack[-1], f"'{text[stack[-1]]}' was never closed")
        if stack
        else None
    )


def error(source: str, i: int, message: str) -> Json:
    """Return the plan's `Error` at offset `i`."""
    line = source.count("\n", 0, i) + 1
    col = i - source.rfind("\n", 0, i)
    return {
        "start": i,
        "end": i + 1,
        "line": line,
        "col": col,
        "message": message,
        "hint": None,
        "file": None,
    }


def view(body: Json) -> Json:
    """Answer ``/api/view``: the closest fixture, moved onto the source sent."""
    source = body["source"]
    name = closest(source)
    fixture = {k: v for k, v in VIEWS[name].items() if k not in ("source", "path")}
    out = moved(fixture, mover(VIEWS[name]["source"], source))
    if err := unbalanced(source):
        out |= {"ok": False, "errors": [err], "model": None, "run": None}
    return picked(out, body.get("picks") or {})


def picked(v: Json, picks: Json) -> Json:
    """Select the run of the sweep that `picks` takes."""
    for axis in v["sweep"]["axes"]:
        axis["pick"] = int(picks.get(axis["id"], 0))
    want = {a["id"]: a["pick"] for a in v["sweep"]["axes"]}
    for r in v["sweep"]["runs"]:
        if r["picks"] == want and v["run"]:
            v["run"] |= {"name": r["name"], "id": r["id"]}
    return v


# --- Forms and edits -------------------------------------------------------------------


def target_key(target: Json) -> str:
    """Return a target's key in a forms fixture: model, node:ID or setting:NAME."""
    if target.get("model"):
        return "model"
    return (
        f"node:{target['node']}"
        if "node" in target
        else f"setting:{target.get('setting')}"
    )


def inspect(body: Json) -> Json:
    """Answer ``/api/inspect`` from the forms of the closest fixture."""
    name = closest(body["source"])
    path = FIXTURES / f"form-{name}.json"
    forms = json.loads(path.read_text()) if path.exists() else {}
    key = target_key(body["target"])
    if key in forms:
        return moved(forms[key], mover(VIEWS[name]["source"], body["source"]))
    return generic(VIEWS[name], body["target"])


def generic(fixture: Json, target: Json) -> Json:
    """Return a form with no fields, titled by the node or setting it is of."""
    node = find_node((fixture.get("model") or {}).get("graph"), target.get("node"))
    title = (
        (node or {}).get("label")
        or (node or {}).get("kind")
        or target.get("setting")
        or "Model"
    )
    subtitle = (node or {}).get("kind")
    return {
        "title": title,
        "subtitle": subtitle,
        "doc": "",
        "where": None,
        "fields": [],
    }


def find_node(node: Json | None, node_id: str | None) -> Json | None:
    """Return the node with an id in a fixture's graph."""
    if not node or not node_id:
        return None
    if node["id"] == node_id:
        return node
    kids = [*node.get("items", []), *node.get("terms", []), *node.get("factors", [])]
    for kid in [*kids, *([node["body"]] if "body" in node else [])]:
        if found := find_node(kid, node_id):
            return found
    return None


def edit(body: Json) -> Json:
    """Answer ``/api/edit``: splice a ``set`` where its field is written, else no-op."""
    source = body["source"]
    for op in body["ops"]:
        if op["op"] == "set" and op.get("value") is not None:
            source = splice(source, body, op)
    return {"source": source}


def splice(source: str, body: Json, op: Json) -> str:
    """Write ``key=value`` over the field's argument, where the form has it."""
    form = inspect(body | {"source": source, "target": op["target"]})
    field = find(form.get("fields") or [], op.get("key") or "")
    where = (field or {}).get("where") or {}
    if not where.get("span"):
        return source
    start, end = where["span"]
    old = source[start:end]
    sep = re.search(r"\s*=\s*", old)
    return (
        source[:start] + (old[: sep.end()] if sep else "") + op["value"] + source[end:]
    )


def find(fields: list, key: str) -> Json | None:
    """Return the field with `key`, among nested fields too."""
    for f in fields:
        if f["key"] == key:
            return f
        if inner := find(f.get("fields") or [], key):
            return inner
    return None


# --- Completion, hover, plan ---------------------------------------------------------


def complete(body: Json) -> list:
    """Answer ``/api/complete`` with what fits the character before the cursor."""
    source, offset = body["source"], body["offset"]
    before = source[:offset].rstrip(" ")[-1:] if offset else ""
    if before in "(,":
        form = load("form-llama3_8b.json")["node:n.1.0.0.0.1"]
        return [setting(f) for f in form["fields"]]
    if before == "=":
        return [item(p) for p in SESSION["palette"] if p["kind"] == "activation"]
    if before in "@$":
        return [
            {
                "label": t,
                "kind": "tap",
                "detail": "tap",
                "documentation": "",
                "insert_text": None,
            }
            for t in ("h", "x0", "v1")
        ]
    return [item(p) for p in SESSION["palette"]]


def setting(f: Json) -> Json:
    """Return a completion item for a setting."""
    detail = f["type"] + (f" = {f['default']}" if f["default"] else " (required)")
    return {
        "label": f["key"],
        "kind": "setting",
        "detail": detail,
        "documentation": f["doc"],
        "insert_text": f["key"] + "=",
    }


def item(p: Json) -> Json:
    """Return a completion item for a palette entry."""
    return {
        "label": p["name"],
        "kind": p["kind"],
        "detail": p["title"],
        "documentation": p["doc"],
        "insert_text": None,
    }


def hover(body: Json) -> Json | None:
    """Answer ``/api/hover`` on the name at the offset: a layer's doc or a definition."""
    source, offset = body["source"], body["offset"]
    for m in re.finditer(r"[A-Za-z_]\w*", source):
        if m.start() <= offset < m.end():
            return describe(source, m)
    return None


def describe(source: str, m: re.Match) -> Json | None:
    """Return the hover of one name."""
    name = m.group()
    found = next((p for p in SESSION["palette"] if p["name"] == name), None)
    if found:
        markdown = (
            f"**{name}** — {found['kind']}\n\n{found['doc']}\n\n- `{found['template']}`"
        )
    elif d := re.search(rf"^{name}\s*=\s*(.+)$", source, re.MULTILINE):
        markdown = f"**{name}** = `{d.group(1).strip()}`\n\nA definition of this file."
    else:
        return None
    return {"start": m.start(), "end": m.end(), "markdown": markdown}


def plan(body: Json) -> Json:
    """Answer ``/api/plan``, as slowly as importing JAX would."""
    time.sleep(0.6)
    facts = (view(body).get("model") or {}).get("facts") or {}
    params = 12 * facts.get("layers", 0) * facts.get("d_model", 0) ** 2 // 4
    return {
        "params": params,
        "params_text": f"{params / 1e9:.2f}B",
        "flops_per_token": 6 * params,
    }


def schedule(body: Json) -> Json:
    """Answer ``/api/schedule`` with a WSD curve of the run's steps."""
    r = view(body).get("run") or {
        "budget": {"steps": 1000},
        "optimizer": {"lr": 1e-3},
        "schedule": {"warmup": 100},
    }
    steps, peak = int(r["budget"]["steps"]), float(r["optimizer"]["lr"])
    warm = int(r["schedule"]["warmup"])
    stop = int(steps * 0.8)

    def lr(s: int) -> float:
        return peak * min(
            s / max(warm, 1),
            1,
            ((steps - s) / (steps - stop)) ** 0.5 if s > stop else 1,
        )

    points = [[s, lr(s)] for s in range(0, steps + 1, max(1, steps // 200))]
    return {"points": points, "peak": peak, "steps": steps}


def save(body: Json) -> tuple[int, Json]:
    """Answer ``/api/save``: 409 with the file when it changed since the client read it."""
    if body.get("mtime") != DISK["mtime"]:
        return HTTPStatus.CONFLICT, {"source": DISK["source"], "mtime": DISK["mtime"]}
    DISK["source"], DISK["mtime"] = body["source"], time.time()
    return HTTPStatus.OK, {"mtime": DISK["mtime"]}


def touch(_body: Json) -> Json:
    """Edit the file as another editor would: add a comment line."""
    DISK["source"] += f"# edited elsewhere at {time.strftime('%H:%M:%S')}\n"
    DISK["mtime"] = time.time()
    return {"mtime": DISK["mtime"]}


POSTS = {
    "/api/view": view,
    "/api/inspect": inspect,
    "/api/edit": edit,
    "/api/complete": complete,
    "/api/hover": hover,
    "/api/plan": plan,
    "/api/schedule": schedule,
    "/dev/touch": touch,
}


class Handler(SimpleHTTPRequestHandler):
    """Static files from the studio's directory, and the JSON API."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        """Serve a file, the session, or the file's mtime."""
        if not self.path.startswith("/api/"):
            return super().do_GET()
        if not self.authorized():
            return None
        if self.path == "/api/session":
            return self.reply(SESSION | DISK | {"name": Path(DISK["path"]).name})
        if self.path.startswith("/api/file"):
            extra = {"source": DISK["source"]} if "source=1" in self.path else {}
            return self.reply({"mtime": DISK["mtime"]} | extra)
        return self.reply({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        """Answer an API call."""
        if not self.path.startswith("/dev/") and not self.authorized():
            return
        body = json.loads(
            self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"
        )
        if self.path == "/api/save":
            status, reply = save(body)
            self.reply(reply, status)
        elif handler := POSTS.get(self.path):
            self.reply(handler(body))
        else:
            self.reply({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def authorized(self) -> bool:
        """Refuse a call without the session's token."""
        if self.headers.get("X-Bux-Token") == TOKEN:
            return True
        self.reply({"error": "bad token"}, HTTPStatus.FORBIDDEN)
        return False

    def reply(self, data: object, status: int = HTTPStatus.OK) -> None:
        """Send JSON."""
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: object) -> None:
        """Log only failures."""
        if args and str(args[1])[:1] not in "23":
            super().log_message(format, *args)


def main() -> None:
    """Serve on the port given, with the fixture given as the file."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9411
    name = sys.argv[2] if len(sys.argv) > 2 else "llama3_8b"
    DISK.update(
        source=VIEWS[name]["source"], mtime=time.time(), path=VIEWS[name]["path"]
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"bux studio (mock): http://127.0.0.1:{port}/?t={TOKEN}  [{name}]")
    server.serve_forever()


if __name__ == "__main__":
    main()
