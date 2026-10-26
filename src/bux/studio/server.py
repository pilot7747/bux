"""The studio's HTTP server: the page's static files and its JSON API, on localhost.

`serve` is ``bux studio FILE``. Its `Studio` holds one file's session, and every
endpoint of the plan's table is a method of it that takes the request's JSON and
returns the reply's; the handler only routes, checks the request, and sends. The
pure work is in `bux.studio.view` and `bux.studio.training`, and the edits and forms
in ``bux.studio.edits`` and ``bux.studio.forms``.

Every ``/api/*`` call carries the session's token in ``X-Bux-Token``, which the page
reads from its URL, so that no other page the browser has open can read or write the
file; the ``Host`` header must name this machine, against DNS rebinding.
"""

import importlib
import json
import os
import secrets
import sys
import tempfile
import threading
import typing
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from bux.lang.compiler import check
from bux.lang.lsp import features
from bux.lang.syntax import BuxError
from bux.studio import palette, training
from bux.studio.picks import traced
from bux.studio.view import view

#: A JSON object, as the API sends it.
type Json = dict[str, Any]

#: Where the page's files are: ``static/`` beside this module.
#:
#: :meta hide-value:
STATIC = Path(__file__).parent / "static"

#: What a file that does not exist yet starts as: a small model and its run.
TEMPLATE = """\
# A new model: tokens come in at the bottom, logits leave at the top.
d_model    = 256
vocab_size = 32k

norm  = rmsnorm
attn  = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn   = gated_mlp(intermediate_size=4 * d_model, activation=swish)
block = [norm attn] [norm ffn]
model = embed block^4 norm head

# How to train it.
data      = synthetic
optimizer = adamw(lr=3e-3)
schedule  = wsd(warmup=0.1)
seq_len   = 256
batch     = 16
steps     = 1k
"""

_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json",
    ".png": "image/png",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}
_MOST = 8 << 20  # the largest body read, in bytes


class Reply(Exception):
    """A reply other than 200: its status and its JSON body."""

    def __init__(self, status: int, body: Json):
        super().__init__(body.get("error", status))
        self.status, self.body = status, body


def _bad(message: str, status: int = HTTPStatus.BAD_REQUEST) -> Reply:
    return Reply(status, {"error": message})


@dataclass
class Studio:
    """One file's session: what every endpoint reads and writes.

    Attributes:
        path: The file, absolute; it need not exist until the first save.
        token: The session's secret, which every API call must carry.
    """

    path: Path
    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    _jax: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cache: dict[tuple[str, str], Json] = field(default_factory=dict, repr=False)

    # --- The file --------------------------------------------------------------------

    def mtime(self) -> float | None:
        """Return when the file last changed, or None if it does not exist."""
        try:
            return self.path.stat().st_mtime
        except FileNotFoundError:
            return None

    def source(self) -> str:
        """Return the file's text, or the template for a file not created yet."""
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return TEMPLATE

    def session(self, body: Json, query: dict[str, str]) -> Json:
        """``GET /api/session``: the file, its text and the palette."""
        source = self.source()
        return {
            "path": str(self.path),
            "name": self.path.name,
            "source": source,
            "mtime": self.mtime(),
            "palette": palette.palette(_context(source, self.where)),
        }

    def file(self, body: Json, query: dict[str, str]) -> Json:
        """``GET /api/file``: when the file last changed, and its text with
        ``?source=1``."""
        found: Json = {"mtime": self.mtime()}
        if query.get("source") == "1":
            found["source"] = self.source()
        return found

    def save(self, body: Json, query: dict[str, str]) -> Json:
        """``POST /api/save``: write the text over the file, at once, unless it
        changed on disk since the page loaded it (409, with what is on disk)."""
        source, loaded = _text(body, "source"), body.get("mtime")
        now = self.mtime()
        if now != loaded:
            disk = self.source() if now is not None else None
            raise Reply(HTTPStatus.CONFLICT, {"source": disk, "mtime": now})
        _write(self.path, source)
        return {"mtime": self.mtime()}

    @property
    def where(self) -> str:
        """The file's name, as the compiler takes it: ``use`` searches from it."""
        return str(self.path)

    # --- The language ----------------------------------------------------------------

    def view(self, body: Json, query: dict[str, str]) -> Json:
        """``POST /api/view``: the plan's ``View`` of the text, the run picked."""
        return view(_text(body, "source"), _picks(body), path=self.where)

    def complete(self, body: Json, query: dict[str, str]) -> list[Json]:
        """``POST /api/complete``: what completion offers at the offset."""
        items = features.complete(_text(body, "source"), _offset(body), self.where)
        items = sorted(items, key=lambda c: (c.sort, c.label))
        return [
            {
                "label": c.label,
                "kind": c.kind,
                "detail": c.detail,
                "documentation": c.documentation,
                "insert_text": c.insert_text,
            }
            for c in items
        ]

    def hover(self, body: Json, query: dict[str, str]) -> Json | None:
        """``POST /api/hover``: what hovering the offset shows, or null."""
        found = features.hover(_text(body, "source"), _offset(body), self.where)
        if found is None:
            return None
        return {"start": found.start, "end": found.end, "markdown": found.markdown}

    def inspect(self, body: Json, query: dict[str, str]) -> Json:
        """``POST /api/inspect``: the inspector's form of a target."""
        forms, error = _module("forms"), _edit_error()
        target = body.get("target")
        if not isinstance(target, dict):
            raise _bad("target is an object: {node}, {setting} or {model}")
        try:
            return forms.inspect(
                _text(body, "source"), target, picks=_picks(body), path=self.where
            )
        except (error, BuxError) as err:
            raise _bad(_message(err)) from None

    def edit(self, body: Json, query: dict[str, str]) -> Json:
        """``POST /api/edit``: the text with the ops applied."""
        edits, error = _module("edits"), _edit_error()
        ops = body.get("ops")
        if not isinstance(ops, list):
            raise _bad("ops is a list of edits")
        try:
            new = edits.apply(
                _text(body, "source"), ops, picks=_picks(body), path=self.where
            )
        except (error, BuxError) as err:
            raise _bad(_message(err)) from None
        return {"source": new}

    # --- What needs JAX ----------------------------------------------------------------

    def schedule(self, body: Json, query: dict[str, str]) -> Json:
        """``POST /api/schedule``: the run's learning rate, step by step."""
        return self._jax_work("schedule", body, training.schedule)

    def plan(self, body: Json, query: dict[str, str]) -> Json:
        """``POST /api/plan``: the model's parameters and FLOPs a token."""
        return self._jax_work("plan", body, training.plan)

    def _jax_work(self, what: str, body: Json, work: Callable[..., Json]) -> Json:
        """Return `work` of the run picked, one at a time, kept by the run's id."""
        try:
            t = traced(_text(body, "source"), _picks(body), path=self.where)
        except BuxError as err:
            raise _bad(err.message) from None
        if what == "schedule" and t.train is None:
            raise _bad("a model, not a run: it sets no schedule")
        key = (what, training.key(t))
        with self._jax:
            if key not in self._cache:
                self._cache[key] = work(t)
            return self._cache[key]


def _context(source: str, filename: str) -> palette.Context:
    """Return the model the palette's templates are for: the file's width, or 768."""
    d_model = check(source, filename=filename).values.get("d_model")
    width = d_model if isinstance(d_model, int) and d_model > 0 else 768
    return palette.Context(d_model=width)


def _module(name: str) -> ModuleType:
    """Return ``bux.studio.<name>``, imported when first asked for."""
    try:
        return importlib.import_module(f"bux.studio.{name}")
    except ModuleNotFoundError as err:
        if err.name != f"bux.studio.{name}":
            raise
        raise _bad(f"bux.studio.{name} is not built yet", HTTPStatus.NOT_IMPLEMENTED)


def _edit_error() -> type[Exception]:
    """Return the error an edit or a form raises for the user: ``EditError``."""
    try:
        return _module("edits").EditError
    except Reply:
        return ValueError


def _message(err: Exception) -> str:
    return err.message if isinstance(err, BuxError) else str(err)


def _text(body: Json, key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str):
        raise _bad(f"{key} is a string")
    return value


def _offset(body: Json) -> int:
    value = body.get("offset")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _bad("offset is a number of characters, from 0")
    return value


def _picks(body: Json) -> dict[str, int]:
    picks = body.get("picks") or {}
    if not isinstance(picks, dict) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in picks.values()
    ):
        raise _bad("picks maps an axis's id to the option it takes")
    return {str(k): v for k, v in picks.items()}


def _write(path: Path, source: str) -> None:
    """Write `source` over `path` at once: to a file beside it, then in its place,
    keeping its permissions."""
    mode = path.stat().st_mode if path.exists() else None
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(source)
        if mode is not None:
            os.chmod(temp, mode)
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


# --- Routing ---------------------------------------------------------------------------

type Endpoint = Callable[[Studio, Json, dict[str, str]], object]

#: The API, by method and path, as the plan's table lists it.
ROUTES: dict[tuple[str, str], Endpoint] = {
    ("GET", "/api/session"): Studio.session,
    ("GET", "/api/file"): Studio.file,
    ("POST", "/api/view"): Studio.view,
    ("POST", "/api/inspect"): Studio.inspect,
    ("POST", "/api/edit"): Studio.edit,
    ("POST", "/api/schedule"): Studio.schedule,
    ("POST", "/api/plan"): Studio.plan,
    ("POST", "/api/complete"): Studio.complete,
    ("POST", "/api/hover"): Studio.hover,
    ("POST", "/api/save"): Studio.save,
}


class Handler(BaseHTTPRequestHandler):
    """Answers one request: a static file, or an endpoint of `ROUTES`."""

    protocol_version = "HTTP/1.1"

    @property
    def studio(self) -> Studio:
        """The session this request is of."""
        return typing.cast(Server, self.server).studio

    @property
    def static(self) -> Path:
        """Where the page's files are."""
        return typing.cast(Server, self.server).static

    def do_GET(self) -> None:
        """Answer a GET: a static file or an endpoint."""
        self.answer("GET")

    def do_POST(self) -> None:
        """Answer a POST: an endpoint."""
        self.answer("POST")

    def answer(self, method: str) -> None:
        url = urlsplit(self.path)
        try:
            self.check_host()
            if url.path.startswith("/api/"):
                self.send_json(HTTPStatus.OK, self.call(method, url.path, url.query))
            elif method == "GET":
                self.send_static(url.path)
            else:
                raise _bad("not found", HTTPStatus.NOT_FOUND)
        except Reply as reply:
            self.send_json(reply.status, reply.body)
        except Exception as err:  # noqa: BLE001 (a bug: say what, not a traceback page)
            self.log_error("%s %s: %r", method, url.path, err)
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": repr(err)})

    def check_host(self) -> None:
        """Refuse a Host that is not this machine's, as a rebound DNS name's."""
        host = (self.headers.get("Host") or "").lower()
        name = host.rsplit(":", 1)[0] if not host.endswith("]") else host
        if name.strip("[]") not in typing.cast(Server, self.server).hosts:
            raise _bad("this server answers localhost only", HTTPStatus.FORBIDDEN)

    def call(self, method: str, path: str, query: str) -> object:
        sent = self.headers.get("X-Bux-Token") or ""
        if not secrets.compare_digest(sent.encode(), self.studio.token.encode()):
            raise _bad("a wrong or missing X-Bux-Token", HTTPStatus.FORBIDDEN)
        endpoint = ROUTES.get((method, path))
        if endpoint is None:
            raise _bad(f"no endpoint {method} {path}", HTTPStatus.NOT_FOUND)
        body = self.body() if method == "POST" else {}
        args = {k: v[-1] for k, v in parse_qs(query).items()}
        return endpoint(self.studio, body, args)

    def body(self) -> Json:
        """Return the request's JSON object."""
        size = int(self.headers.get("Content-Length") or 0)
        if size > _MOST:
            raise _bad("the body is too large", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        try:
            found = json.loads(self.rfile.read(size) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError) as err:
            raise _bad(f"the body is not JSON: {err}") from None
        if not isinstance(found, dict):
            raise _bad("the body is a JSON object")
        return found

    def send_static(self, path: str) -> None:
        """Send a file of `STATIC`, ``index.html`` for ``/``; none outside it."""
        root = self.static.resolve()
        relative = unquote(path).lstrip("/") or "index.html"
        file = (root / relative).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise _bad("not found", HTTPStatus.NOT_FOUND)
        kind = _TYPES.get(file.suffix, "application/octet-stream")
        self.send(HTTPStatus.OK, file.read_bytes(), kind)

    def send_json(self, status: int, value: object) -> None:
        data = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
        self.send(status, data, "application/json")

    def send(self, status: int, data: bytes, kind: str) -> None:
        self.send_response(status)
        if status >= 400:  # a body it refused may be unread: it ends the connection
            self.send_header("Connection", "close")
            self.close_connection = True
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        """Log failed requests alone."""
        if isinstance(code, int) and code >= 400:
            super().log_request(code, size)


class Server(ThreadingHTTPServer):
    """The studio's server: its session, its static files, and the hosts it answers
    to."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        studio: Studio,
        static: Path = STATIC,
    ):
        super().__init__(address, Handler)
        self.studio, self.static = studio, static
        self.hosts = {"localhost", "127.0.0.1", "::1", address[0].lower()}

    @property
    def url(self) -> str:
        """The page's address, with the session's token."""
        host, port = self.server_address[:2]
        return f"http://{_shown(str(host))}:{port}/?t={self.studio.token}"


def _shown(host: str) -> str:
    if host in ("0.0.0.0", "::"):
        return "127.0.0.1"
    return f"[{host}]" if ":" in host else host


def bind(
    studio: Studio, host: str = "127.0.0.1", port: int = 8765, tries: int = 10
) -> Server:
    """Return a server of `studio` on `port`, or the first of the next `tries` ports
    that is free; port 0 takes any.

    Raises:
        OSError: If none is.
    """
    last = tries - 1 if port else 0
    for k in range(last + 1):
        try:
            return Server((host, port + k if port else 0), studio)
        except OSError:
            if k == last:
                raise
    raise AssertionError("unreachable")


def serve(
    file: str, *, host: str = "127.0.0.1", port: int = 8765, browser: bool = True
) -> int:
    """Serve the studio of `file` until interrupted: ``bux studio FILE``.

    JAX runs on the CPU in this process, unless ``JAX_PLATFORMS`` says otherwise,
    so that the studio takes no GPU from a run.

    Args:
        file: The ``.bux`` file; one that does not exist starts from `TEMPLATE`, and
            is created by the first save.
        host: The address to listen on.
        port: The port to try first; the next few are tried after it.
        browser: Open the page in a browser; never over SSH, where the hint to
            forward the port is printed instead.

    Returns:
        The exit code: 0 once interrupted, 1 if no port was free.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    studio = Studio(Path(file).resolve())
    try:
        server = bind(studio, host, port)
    except OSError as err:
        print(f"bux studio: no free port from {port}: {err}", file=sys.stderr)
        return 1
    print(f"bux studio: {studio.path} at {server.url}", flush=True)
    _open(server, browser)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _open(server: Server, browser: bool) -> None:
    """Open the page, or say how to reach it over SSH."""
    if os.environ.get("SSH_CONNECTION"):
        port = server.server_address[1]
        print(
            f"over SSH: ssh -L {port}:127.0.0.1:{port} THIS_HOST, then open the URL",
            flush=True,
        )
    elif browser:
        webbrowser.open(server.url)
