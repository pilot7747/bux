"""bux.studio.server: the studio's HTTP server, on a free port in a thread."""

import http.client
import json
import os
import re
import socket
import sys
import threading
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from bux.__main__ import studio_parser
from bux.studio import server as studio_server
from bux.studio import training
from bux.studio.server import TEMPLATE, Server, Studio, bind

EXAMPLES = Path(__file__).parent.parent / "examples"
SOURCE = (EXAMPLES / "experiments/smoke.bux").read_text()


class Client:
    """Calls the server as the page does, with the token, or as anyone else."""

    def __init__(self, server: Server):
        self.port, self.token = server.server_address[1], server.studio.token

    def call(
        self,
        path: str,
        body: object = None,
        *,
        token: str | None = "",
        host: str = "127.0.0.1",
    ) -> tuple[int, Any, str]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        headers = {"Host": f"{host}:{self.port}"}
        if token is not None:
            headers["X-Bux-Token"] = token or self.token
        data = None if body is None else json.dumps(body).encode()
        if data is not None:
            headers["Content-Type"] = "application/json"
        conn.request("GET" if body is None else "POST", path, data, headers)
        reply = conn.getresponse()
        raw, kind = reply.read(), reply.getheader("Content-Type") or ""
        conn.close()
        value = json.loads(raw) if "json" in kind else raw.decode()
        return reply.status, value, kind


@pytest.fixture
def file(tmp_path: Path) -> Path:
    path = tmp_path / "smoke.bux"
    path.write_text(SOURCE)
    return path


@pytest.fixture
def static(tmp_path: Path) -> Path:
    root = tmp_path / "static"
    (root / "dev").mkdir(parents=True)
    for name in ("index.html", "app.js", "app.css", "mark.svg", "dev/fixture.json"):
        (root / name).write_text(json.dumps(name) if name.endswith(".json") else name)
    (tmp_path / "secret.txt").write_text("secret")
    return root


@pytest.fixture
def client(file: Path, static: Path) -> Iterator[Client]:
    server = Server(("127.0.0.1", 0), Studio(file), static)
    thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    yield Client(server)
    server.shutdown()
    server.server_close()


def test_the_api_needs_the_token(client):
    assert client.call("/api/session", token=None)[0] == 403
    status, body, kind = client.call("/api/session", token="guess")
    assert (status, kind) == (403, "application/json")
    assert "X-Bux-Token" in body["error"]
    assert client.call("/api/view", {"source": "", "picks": {}}, token=None)[0] == 403


def test_only_localhost_is_answered(client):
    assert client.call("/api/session", host="evil.example")[0] == 403
    assert client.call("/index.html", host="rebound.example")[0] == 403
    assert client.call("/api/session", host="localhost")[0] == 200


def test_static_files_with_their_types_and_nothing_else(client):
    status, text, kind = client.call("/", token=None)
    assert (status, text, kind) == (200, "index.html", "text/html; charset=utf-8")
    assert client.call("/app.js", token=None)[2] == "text/javascript; charset=utf-8"
    assert client.call("/app.css", token=None)[2] == "text/css; charset=utf-8"
    assert client.call("/mark.svg", token=None)[2] == "image/svg+xml"
    assert client.call("/dev/fixture.json", token=None)[2] == "application/json"
    for path in (
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/dev/../../secret.txt",
        "/nope.js",
    ):
        assert client.call(path, token=None)[0] == 404


def test_the_session_and_the_file(client, file):
    status, session, _ = client.call("/api/session")
    assert status == 200
    assert session["path"] == str(file) and session["name"] == "smoke.bux"
    assert session["source"] == SOURCE and session["mtime"] == file.stat().st_mtime
    assert {"gqa", "swish"} <= {p["name"] for p in session["palette"]}
    assert client.call("/api/file")[1] == {"mtime": file.stat().st_mtime}
    assert client.call("/api/file?source=1")[1]["source"] == SOURCE


def test_a_view(client):
    status, v, _ = client.call("/api/view", {"source": SOURCE, "picks": {}})
    assert status == 200 and v["ok"]
    assert v["model"]["name"] == "smoke" and v["run"]["budget"]["steps"] == 50
    status, v, _ = client.call("/api/view", {"source": "model = nrom\n", "picks": {}})
    assert status == 200 and not v["ok"] and v["errors"]


def test_bad_requests_are_json_errors(client):
    assert client.call("/api/view", {"picks": {}})[0] == 400
    assert client.call("/api/view", {"source": "", "picks": {"a": "x"}})[0] == 400
    assert client.call("/api/complete", {"source": "", "offset": -1})[0] == 400
    status, body, kind = client.call("/api/nothing", {})
    assert (status, kind) == (404, "application/json") and "error" in body


def test_saving_writes_the_file_unless_it_changed_on_disk(client, file):
    mtime = client.call("/api/session")[1]["mtime"]
    status, saved, _ = client.call("/api/save", {"source": "# one\n", "mtime": mtime})
    assert status == 200 and file.read_text() == "# one\n"
    assert saved["mtime"] == file.stat().st_mtime
    os.utime(file, (0, 12345.0))  # another editor wrote it
    file.write_text("# theirs\n")
    os.utime(file, (0, 12345.0))
    status, body, _ = client.call(
        "/api/save", {"source": "# two\n", "mtime": saved["mtime"]}
    )
    assert status == 409 and body == {"source": "# theirs\n", "mtime": 12345.0}
    assert file.read_text() == "# theirs\n"
    status, _, _ = client.call("/api/save", {"source": "# two\n", "mtime": 12345.0})
    assert status == 200 and file.read_text() == "# two\n"
    assert [p.name for p in file.parent.iterdir() if p.name.startswith(".")] == []


def test_a_new_file_starts_from_the_template(tmp_path, static):
    path = tmp_path / "new" / "model.bux"
    path.parent.mkdir()
    server = Server(("127.0.0.1", 0), Studio(path), static)
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    client = Client(server)
    try:
        session = client.call("/api/session")[1]
        assert (session["source"], session["mtime"]) == (TEMPLATE, None)
        v = client.call("/api/view", {"source": TEMPLATE, "picks": {}})[1]
        assert v["ok"] and v["run"] is not None
        assert client.call("/api/save", {"source": TEMPLATE, "mtime": None})[0] == 200
        assert path.read_text() == TEMPLATE
    finally:
        server.shutdown()
        server.server_close()


def test_completion_and_hover(client):
    source = "d_model = 64\nattn = gqa("
    status, items, _ = client.call(
        "/api/complete", {"source": source, "offset": len(source)}
    )
    assert status == 200 and items[0]["label"] == "num_heads"
    assert set(items[0]) == {"label", "kind", "detail", "documentation", "insert_text"}
    at = SOURCE.index("rmsnorm") + 2
    status, hover, _ = client.call("/api/hover", {"source": SOURCE, "offset": at})
    assert status == 200 and SOURCE[hover["start"] : hover["end"]] == "rmsnorm"
    assert "rmsnorm" in hover["markdown"]
    assert client.call("/api/hover", {"source": SOURCE, "offset": 0})[1] is None


def test_forms_and_edits_are_called_once_they_exist(client, monkeypatch):
    class EditError(Exception):
        pass

    def inspect(source, target, *, picks, path):
        if "node" in target:
            raise EditError("no such node")
        return {"title": "Model", "picks": picks, "path": path}

    def apply(source, ops, *, picks, path):
        if not ops:
            raise EditError("nothing to do")
        return source + "# edited\n"

    forms = types.SimpleNamespace(inspect=inspect)
    edits = types.SimpleNamespace(apply=apply, EditError=EditError)
    monkeypatch.setitem(sys.modules, "bux.studio.forms", forms)
    monkeypatch.setitem(sys.modules, "bux.studio.edits", edits)
    body = {"source": "x", "picks": {"a": 1}, "target": {"model": True}}
    status, form, _ = client.call("/api/inspect", body)
    assert (
        status == 200
        and form["picks"] == {"a": 1}
        and form["path"].endswith("smoke.bux")
    )
    status, error, _ = client.call("/api/inspect", body | {"target": {"node": "n.9"}})
    assert (status, error) == (400, {"error": "no such node"})
    status, edited, _ = client.call(
        "/api/edit", {"source": "x\n", "picks": {}, "ops": [{}]}
    )
    assert (status, edited) == (200, {"source": "x\n# edited\n"})
    assert client.call("/api/edit", {"source": "x", "picks": {}, "ops": []})[0] == 400


def test_jax_work_is_kept_by_run(client, monkeypatch):
    calls = []

    def schedule(t):
        calls.append(t)
        return {"points": [[0, 1.0]], "peak": 1.0, "steps": t.train.steps}

    monkeypatch.setattr(training, "schedule", schedule)
    body = {"source": SOURCE, "picks": {}}
    first = client.call("/api/schedule", body)
    assert first[:2] == (200, {"points": [[0, 1.0]], "peak": 1.0, "steps": 50})
    assert client.call("/api/schedule", body)[1] == first[1] and len(calls) == 1
    client.call("/api/schedule", {"source": SOURCE.replace("50", "60"), "picks": {}})
    assert len(calls) == 2
    model = (EXAMPLES / "architectures/llama3_8b.bux").read_text()
    assert client.call("/api/schedule", {"source": model, "picks": {}})[0] == 400
    assert client.call("/api/plan", {"source": "model = nrom", "picks": {}})[0] == 400


def test_a_taken_port_is_passed_over(file):
    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen()
    port = taken.getsockname()[1]
    try:
        server = bind(Studio(file), "127.0.0.1", port, tries=20)
        assert server.server_address[1] != port
        assert (
            server.url
            == f"http://127.0.0.1:{server.server_address[1]}/?t={server.studio.token}"
        )
        server.server_close()
    finally:
        taken.close()


def test_the_command_line(monkeypatch, file, capsys):
    args = studio_parser().parse_args([str(file), "--port", "9000", "--no-browser"])
    assert (args.port, args.host, args.no_browser) == (9000, "127.0.0.1", True)
    served = {}

    def serve_forever(self: Server) -> None:
        served["url"] = self.url
        raise KeyboardInterrupt

    monkeypatch.setattr(Server, "serve_forever", serve_forever)
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 22 10.0.0.2 22")
    opened = []
    monkeypatch.setattr(studio_server.webbrowser, "open", opened.append)
    from bux.__main__ import main

    assert main(["studio", str(file), "--port", "0"]) == 0
    out = capsys.readouterr().out
    assert served["url"] in out and "ssh -L" in out and not opened


def test_the_page_names_only_files_that_ship():
    static = studio_server.STATIC
    named = re.findall(r'(?:href|src)="([\w.-]+)"', (static / "index.html").read_text())
    for script in static.glob("*.js"):
        text = script.read_text()
        named += re.findall(r'(?:from|import)\s*\(?\s*"\./([\w.-]+)"', text)
        named += re.findall(r'load\("\./([\w.-]+)", "([\w.-]+)"\)', text)
    files = {n for item in named for n in ((item,) if isinstance(item, str) else item)}
    assert {"app.js", "diagram.js", "training.js", "sweep.js", "panels.css"} <= files
    assert [f for f in sorted(files) if not (static / f).is_file()] == []
