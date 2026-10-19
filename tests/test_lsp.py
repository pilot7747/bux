import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from bux.lang.lsp import features

EXAMPLES = sorted(Path(__file__).parent.parent.joinpath("examples").glob("*.bux"))

PRELUDE = """d_model = 16
vocab_size = 32
norm = rmsnorm
attn = gqa(num_heads=2, num_kv_heads=2)
ffn = gated_mlp(intermediate_size=32, activation=swish)
"""


def cursor(text: str) -> tuple[str, int]:
    """The text without its `|`, and where the `|` was."""
    return text.replace("|", "", 1), text.index("|")


def labels(text: str) -> list[str]:
    return [c.label for c in features.complete(*cursor(text))]


def hover(text: str) -> str:
    found = features.hover(*cursor(text))
    assert found is not None
    return found.markdown


# --- Completion ---------------------------------------------------------------------


def test_completes_the_settings_of_a_layer():
    got = labels(PRELUDE + "x = gqa(num_heads=2, |)")
    assert "num_kv_heads" in got and "rope" in got and "num_heads" not in got
    assert got.index("num_kv_heads") < got.index("rope")  # required first
    # Through a definition, and while the statement is still unfinished.
    assert "sliding_window_size" in labels(PRELUDE + "x = attn(|\nmodel = embed head")
    assert labels(PRELUDE + "x = gqa(rope.|)") == ["base", "fraction", "scaling"]
    item = next(
        c
        for c in features.complete(*cursor(PRELUDE + "x = gqa(|)"))
        if c.label == "head_dim"
    )
    assert (item.insert_text, item.detail) == ("head_dim=", "int | none = none")
    assert item.documentation == "None -> width // num_heads"


def test_completes_values_by_the_type_of_the_setting():
    got = labels(PRELUDE + "x = gated_mlp(intermediate_size=8, activation=|)")
    assert got[:7] == ["relu", "relu2", "gelu", "gelu_tanh", "swish", "sigmoid", "tanh"]
    assert "gqa" not in got and "d_model" in got
    got = labels(
        PRELUDE + "blk = [attn]\nx = gqa(num_heads=2, num_kv_heads=2, qk_norm=|)"
    )
    assert {"layernorm", "rmsnorm", "none", "norm"} <= set(got) and "blk" not in got


def test_completes_names_parameters_fields_taps_and_pattern_letters():
    macro = PRELUDE + "layer(mixer, f=ffn) = [norm mixer] [norm f]\n"
    got = labels(macro + "model = embed layer(|) head")
    assert got[:2] == ["mixer", "f"] and "attn" in got and "gdn" in got
    assert {"mixer", "f", "attn", "gqa", "swish", "_"} <= set(
        labels(macro.replace("[norm f]", "[norm |]"))
    )
    got = labels(PRELUDE + "x = attn.|")
    assert got[:2] == ["num_heads", "num_kv_heads"] and "type" not in got
    assert labels(PRELUDE + "model = embed @x0 [attn] [$|") == ["x0"]
    assert labels(PRELUDE + 'A = [attn]\nF = [ffn]\nmodel = embed "A|') == ["A", "F"]
    assert labels(PRELUDE + "# x = gqa(|") == []
    assert labels(PRELUDE + "|") == []  # a new statement's name


# --- Hover --------------------------------------------------------------------------


def test_hover_documents_layers_and_their_settings():
    text = hover(PRELUDE + "x = g|qa(num_heads=2, num_kv_heads=2)")
    assert text.startswith("**gqa** — layer\n\nGrouped-query attention")
    assert "- `num_kv_heads: int` (required)" in text
    assert "- `head_dim: int | none = none` — None -> width // num_heads" in text
    assert (
        hover(PRELUDE + "x = gqa(num_heads=2, num_kv|_heads=2)")
        == "`num_kv_heads: int` (required)"
    )
    assert hover(PRELUDE + "x = attn.num_h|eads") == "`num_heads: int` (required)"
    # Inside its own definition a name means the builtin.
    assert hover(
        PRELUDE + "mlp = ml|p(intermediate_size=8, activation=gelu)"
    ).startswith("**mlp**")


def test_hover_shows_what_names_mean():
    text = hover(
        PRELUDE + "local = attn(sliding_window_size=4)\nmodel = embed [lo|cal] head"
    )
    assert text == (
        "```bux\nlocal = attn(sliding_window_size=4)\n```\n\n"
        "= `gqa(num_heads=2, num_kv_heads=2, sliding_window_size=4)`"
    )
    assert hover(
        PRELUDE
        + "e = moe(num_experts=4, top_k=2, expert=ffn, load_balancing=aux_|free)"
    ) == ('`aux_free` is not defined, so it is the string `"aux_free"`')
    assert (
        hover("layer(mixer, f=1) = [rmsnorm mi|xer]")
        == "parameter **mixer** of `layer(mixer, f=1)`"
    )
    assert "written in 1 place, popped in 1 place" in hover(
        PRELUDE + "model = embed @s|kip [$skip!] head"
    )
    found = features.hover(*cursor(PRELUDE + "model = embed [no|rm attn] head"))
    assert found is not None and (found.end - found.start) == len("norm")


# --- Diagnostics and robustness -------------------------------------------------------


def test_diagnostics_point_at_the_source():
    source = PRELUDE + "model = embed [norm atn] head\n"
    [d] = features.diagnostics(source)
    assert source[d.start : d.end] == "atn"
    assert d.message == "unknown layer or block 'atn'\nhint: did you mean 'attn'?"
    assert features.diagnostics(EXAMPLES[0].read_text()) == []


@pytest.mark.parametrize("path", EXAMPLES[::3], ids=lambda p: p.stem)
def test_features_never_fail_while_typing(path):
    """Every prefix of a file, as if typed, gets diagnostics and completions; every point hovers."""
    text = path.read_text()
    for k in range(0, len(text), 3):
        features.diagnostics(text[:k])
        features.complete(text[:k], k)
        features.complete(text, k)
        features.hover(text, k)


# --- The server ---------------------------------------------------------------------


class _Client:
    """A minimal LSP client over the server's stdio."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "bux.lang.lsp.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self.inbox: queue.Queue[dict] = queue.Queue()
        self.ids = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        stdout = self.proc.stdout
        assert stdout is not None
        while header := stdout.readline():
            length = (
                int(header.split(b":")[1])
                if header.startswith(b"Content-Length")
                else None
            )
            while stdout.readline().strip():
                pass
            if length is not None:
                self.inbox.put(json.loads(stdout.read(length)))

    def send(
        self, method: str, params: dict | None = None, request: bool = False
    ) -> int | None:
        message: dict = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if request:
            self.ids += 1
            message["id"] = self.ids
        body = json.dumps(message).encode()
        stdin = self.proc.stdin
        assert stdin is not None
        stdin.write(b"Content-Length: %d\r\n\r\n%s" % (len(body), body))
        stdin.flush()
        return message.get("id")

    def wait(self, *, id: int | None = None, method: str | None = None) -> dict:
        while True:
            message = self.inbox.get(timeout=20)
            if (id is not None and message.get("id") == id) or (
                method is not None and message.get("method") == method
            ):
                return message

    def request(self, method: str, params: dict) -> dict:
        return self.wait(id=self.send(method, params, request=True))["result"]


def test_server_speaks_lsp_over_stdio():
    pytest.importorskip("pygls")
    client = _Client()
    try:
        caps = client.request(
            "initialize", {"processId": None, "rootUri": None, "capabilities": {}}
        )
        assert "(" in caps["capabilities"]["completionProvider"]["triggerCharacters"]
        assert caps["capabilities"]["hoverProvider"]
        client.send("initialized")

        # The emoji is two UTF-16 units: positions must count them that way.
        bad_line = 'p = "😀"; model = embed [norm atn] head'
        source = PRELUDE + bad_line + "\nx = gqa(num_heads=2, num_kv_heads=2)\n"
        uri = "file:///model.bux"
        doc = {"uri": uri, "languageId": "bux", "version": 1, "text": source}
        client.send("textDocument/didOpen", {"textDocument": doc})
        diagnostics = client.wait(method="textDocument/publishDiagnostics")["params"][
            "diagnostics"
        ]
        assert [d["message"].split("\n")[0] for d in diagnostics] == [
            "unknown layer or block 'atn'"
        ]
        start = len(bad_line[: bad_line.index("atn")].encode("utf-16-le")) // 2
        assert diagnostics[0]["range"]["start"] == {"line": 5, "character": start}

        where = {"textDocument": {"uri": uri}, "position": {"line": 6, "character": 8}}
        items = client.request("textDocument/completion", where)["items"]
        assert "rope" in {item["label"] for item in items}
        where["position"]["character"] = 5
        result = client.request("textDocument/hover", where)
        assert result["contents"]["value"].startswith("**gqa** — layer")

        fixed = source.replace("atn]", "attn]")
        client.send(
            "textDocument/didChange",
            {
                "textDocument": {"uri": uri, "version": 2},
                "contentChanges": [{"text": fixed}],
            },
        )
        assert (
            client.wait(method="textDocument/publishDiagnostics")["params"][
                "diagnostics"
            ]
            == []
        )
        client.request("shutdown", {})
        client.send("exit")
        assert client.proc.wait(timeout=20) == 0
    finally:
        client.proc.kill()
