"""bux.studio.view, palette and internals: what the page draws of a file."""

import inspect
import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from bux.lang import compile
from bux.lang.compiler import ACTIVATIONS, LAYERS
from bux.studio import internals, palette
from bux.studio.palette import Context
from bux.studio.view import RUNS, view

EXAMPLES = Path(__file__).parent.parent / "examples"
LLAMA = EXAMPLES / "architectures/llama3_8b.bux"
ABLATION = EXAMPLES / "experiments/attention_ablation.bux"
#: Examples that compile only where others use them, after a model: the
#: timeline's rungs set d_model, so alone they are a model file without a model.
PARTS = {"experiments/timeline/rungs.bux"}


def view_of(path: Path, picks: dict[str, int] | None = None) -> dict:
    return view(path.read_text(), picks or {}, path=str(path))


def nodes(node: dict) -> Iterator[dict]:
    yield node
    for key in ("items", "terms", "factors"):
        for child in node.get(key, []):
            yield from nodes(child)
    if "body" in node:
        yield from nodes(node["body"])


def layers(v: dict, kind: str | None = None) -> list[dict]:
    found = [n for n in nodes(v["model"]["graph"]) if n["kind"] == "layer"]
    return [n for n in found if kind is None or n["type"] == kind]


def text(source: str, span: list[int]) -> str:
    return source[span[0] : span[1]]


@pytest.mark.parametrize(
    "path",
    [
        p
        for p in sorted(EXAMPLES.rglob("*.bux"))
        if str(p.relative_to(EXAMPLES)) not in PARTS
    ],
    ids=lambda p: str(p.relative_to(EXAMPLES)),
)
def test_every_example_has_a_view(path):
    v = view_of(path)
    assert v["ok"], v["errors"]
    json.dumps(v, allow_nan=False)
    if v["model"] is None:  # a module: a recipe of settings
        assert path.parent.name == "recipes" and v["run"] is None
    else:
        assert v["model"]["facts"]["layers"] >= len({n["type"] for n in layers(v)})


def test_nodes_have_ids_by_path_and_spans_where_they_are_written():
    source = LLAMA.read_text()
    graph = view_of(LLAMA)["model"]["graph"]
    assert graph["id"] == "n" and graph["label"] == "model"
    embed, repeat, _, head = graph["items"]
    assert [embed["id"], repeat["id"], head["id"]] == ["n.0", "n.1", "n.3"]
    assert repeat["kind"] == "repeat" and repeat["times"] == 32
    block = repeat["body"]
    assert block["id"] == "n.1.0" and block["label"] == "block"
    attn = block["items"][0]["body"]["items"][1]
    assert attn["id"] == "n.1.0.0.0.1"
    assert text(source, attn["span"]) == "attn"  # where it is used
    assert text(source, attn["written"]).startswith("gqa(num_heads=32")
    assert attn["editable"] and not attn["inserted"]
    for n in nodes(graph):
        assert n["span"] is None or n["origin"] is None


def test_layers_have_titles_categories_and_chips():
    v = view_of(LLAMA)
    [attn] = {n["id"]: n for n in layers(v, "gqa")}.values()
    assert attn["title"] == "Grouped-query attention"
    assert attn["category"] == "attention"
    assert attn["chips"] == ["32 heads", "8 KV heads", "head dim 128", "RoPE 500k"]
    [ffn] = layers(v, "gated_mlp")
    assert ffn["title"] == "Feed forward (SwiGLU)"
    assert ffn["chips"] == ["14,336 hidden"]
    [embed] = layers(v, "embed")
    assert (embed["title"], embed["category"], embed["chips"]) == (
        "Token embedding",
        "io",
        ["vocab 128k"],
    )
    assert layers(v, "head")[0]["title"] == "Linear output layer"


def test_facts_are_bux_checks_summary():
    facts = view_of(LLAMA)["model"]["facts"]
    assert facts["d_model"] == 4096 and facts["vocab_size"] == 128256
    assert facts["layers"] == 1 + 32 * 4 + 2
    assert facts["by_type"][0] == ["rmsnorm", 65]
    assert facts["residual"] is None


def test_what_a_module_writes_is_not_editable_here():
    v = view_of(ABLATION)
    source = ABLATION.read_text()
    [attn] = {n["id"]: n for n in layers(v, "gqa")}.values()
    assert attn["span"] is None and attn["origin"] == "llama3_8b"
    assert not attn["editable"]
    # attn.num_heads = 12 and attn.num_kv_heads = mha: 12 | ..., in this file
    assert text(source, attn["written"]).startswith("attn.num_heads")
    assert attn["title"] == "Multi-head attention" and attn["chips"][0] == "12 heads"


def test_pattern_letters_are_not_edited_alone():
    path = EXAMPLES / "architectures/qwen3_next.bux"
    source = path.read_text()
    pattern = view_of(path)["model"]["graph"]["items"][1]["body"]
    assert text(source, pattern["span"]) == '"DDDA"' and pattern["editable"]
    letters = pattern["items"]
    assert [n["label"] for n in letters] == ["D", "D", "D", "A"]
    assert not any(n["editable"] for n in letters)
    inner = letters[0]["items"][0]  # [norm mixer], in layer(mixer)'s body
    assert inner["editable"] and text(source, inner["span"]) == "[norm mixer]"


def test_merges_the_compiler_inserted_are_marked():
    merges = layers(view_of(EXAMPLES / "architectures/mhc.bux"), "merge")
    assert merges and all(m["inserted"] and m["span"] is None for m in merges)
    assert not any(m["editable"] for m in merges)


def test_errors_say_where_with_hints():
    v = view("d_model = 64\nmodel = embed nrom head\n", {}, path=None)
    assert not v["ok"] and v["model"] is None and v["run"] is None
    [error] = v["errors"]
    assert (error["line"], error["col"], error["file"]) == (2, 15, None)
    assert error["start"] == 27 and error["end"] == 31
    assert error["hint"].startswith("did you mean")
    assert [o["name"] for o in v["outline"]] == ["d_model", "model"]


def test_errors_in_a_module_are_at_its_use(tmp_path):
    (tmp_path / "parts.bux").write_text("attn = gqa(num_heads=4, num_kv_head=2)\n")
    main = tmp_path / "main.bux"
    source = "use parts\nd_model = 64\nvocab_size = 8\nmodel = embed [attn] head\n"
    main.write_text(source)
    [error] = view(source, {}, path=str(main))["errors"]
    assert text(source, [error["start"], error["end"]]) == "use parts"
    assert error["file"] == str(tmp_path / "parts.bux")
    assert error["message"].startswith("parts.bux:1: gqa has no setting")
    assert (error["line"], error["col"]) == (1, 1)


def test_the_outline_lists_this_files_definitions_then_inherited_ones():
    outline = {o["name"]: o for o in view_of(ABLATION)["outline"]}
    assert outline["d_model"]["kind"] == "setting"
    assert outline["model"]["kind"] == "block" and outline["model"]["origin"] is None
    assert outline["attn"]["span"] is not None  # amended here
    assert outline["block"] == {
        "name": "block",
        "kind": "block",
        "span": None,
        "origin": "llama3_8b",
    }
    assert outline["norm"]["kind"] == "layer"
    assert outline["steps"]["origin"] == "fineweb_wsd"


def test_a_sweeps_axes_and_runs():
    sweep = view_of(ABLATION)["sweep"]
    kv, lr = sweep["axes"]
    assert kv["id"] == "gqa|mha|mqa" and kv["labeled"]
    assert kv["options"] == ["mha: 12", "gqa: 4", "mqa: 1"]
    assert kv["labels"] == ["mha", "gqa", "mqa"]
    assert (lr["id"], lr["options"], lr["labels"]) == (
        "optimizer.lr=1e-3|3e-3",
        ["1e-3", "3e-3"],
        None,
    )
    assert text(ABLATION.read_text(), lr["span"]) == "1e-3 | 3e-3"
    assert sweep["total"] == 6 and not sweep["truncated"]
    picks = [tuple(r["picks"].values()) for r in sweep["runs"]]
    assert picks == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert len({r["id"] for r in sweep["runs"]}) == 6


@pytest.mark.parametrize("k", range(6))
def test_picks_select_the_run_the_sweep_lists(k):
    listed = view_of(ABLATION)["sweep"]["runs"][k]
    v = view_of(ABLATION, listed["picks"])
    assert (v["run"]["name"], v["run"]["id"]) == (listed["name"], listed["id"])
    assert [a["pick"] for a in v["sweep"]["axes"]] == list(listed["picks"].values())
    heads = {0: "12 heads", 1: "4 KV heads", 2: "1 KV head"}[
        listed["picks"]["gqa|mha|mqa"]
    ]
    assert heads in layers(v, "gqa")[0]["chips"]
    lr = (1e-3, 3e-3)[listed["picks"]["optimizer.lr=1e-3|3e-3"]]
    assert v["run"]["optimizer"]["lr"] == lr


def test_a_schedule_sweeps_alternatives_are_written_as_in_the_file():
    path = EXAMPLES / "experiments/schedule_sweep.bux"
    sweep = view_of(path)["sweep"]
    [axis] = sweep["axes"]
    assert axis["options"] == [
        "wsd(warmup=1k, decay=0.2, shape=sqrt)",
        "deepseek_v3",
        "kimi_k2",
        "olmo_2",
        "d2z",
        "deepseek_llm",
    ]
    assert sweep["total"] == 6
    types = [
        view_of(path, run["picks"])["run"]["schedule"]["type"] for run in sweep["runs"]
    ]
    assert types == ["wsd", "piecewise", "wsd", "cosine", "linear_decay", "multistep"]


def test_a_searchs_dimensions():
    sweep = view_of(EXAMPLES / "experiments/attention_search.bux")["sweep"]
    lr, wd = sweep["dimensions"]
    assert (lr["where"], lr["type"], lr["low"], lr["high"], lr["log"]) == (
        "optimizer.lr",
        "log_uniform",
        1e-4,
        1e-2,
        True,
    )
    assert (wd["type"], wd["log"], wd["options"]) == ("uniform", False, None)
    assert sweep["search"]["trials"] == 27
    assert sweep["search"]["halving"].startswith("rungs of")


def test_a_sweep_lists_at_most_its_cap_of_runs():
    source = (
        "d_model = 64\nvocab_size = 64\n"
        "model = embed [rmsnorm gqa(num_heads=2, num_kv_heads=1 | 2)]^(1 | 2 | 3 | 4)"
        " head\ndata = synthetic\noptimizer = adamw(lr=1e-3 | 2e-3 | 3e-3 | 4e-3)\n"
        "seq_len = 8\nbatch = 2\nsteps = 10\n"
        "schedule = wsd(warmup=1 | 2 | 3 | 4)\nseed = 0 | 1 | 2 | 3\n"
    )
    v = view(source, {}, path=None)
    assert v["ok"], v["errors"]
    sweep = v["sweep"]
    assert sweep["total"] == 2 * 4 * 4 * 4 * 4 > RUNS
    assert sweep["truncated"] and len(sweep["runs"]) == RUNS


@pytest.mark.parametrize("name", ["deepseek_v3", "qwen3_next"])
def test_a_view_is_fast(name):
    source = (EXAMPLES / f"architectures/{name}.bux").read_text()
    view(source, {}, path=None)
    start = time.perf_counter()
    for _ in range(5):
        view(source, {}, path=None)
    each = (time.perf_counter() - start) / 5
    assert each < 0.2, f"{each * 1e3:.0f} ms"  # ~10 ms on an M-series Mac


# --- The palette and internals -------------------------------------------------------


def test_the_palette_has_every_layer_and_activation():
    items = palette.palette(Context(d_model=512))
    assert {i["name"] for i in items if i["kind"] == "layer"} == set(LAYERS)
    assert {i["name"] for i in items if i["kind"] == "activation"} == ACTIVATIONS
    assert all(i["title"] and i["doc"] and i["template"] for i in items)
    by = {i["name"]: i for i in items}
    assert by["gqa"]["template"] == "gqa(num_heads=4, num_kv_heads=1, rope=rope())"
    assert by["swish"]["title"] == "SiLU" and by["moe"]["category"] == "experts"


@pytest.mark.parametrize(
    "item",
    [
        i
        for i in palette.palette(Context(d_model=256))
        if i["category"] in ("attention", "recurrent", "ffn", "experts", "norm")
        or i["kind"] == "activation"
    ],
    ids=lambda i: i["name"],
)
def test_templates_compile_in_a_branch(item):
    compile(
        "d_model = 256\nvocab_size = 64\n"
        f"model = embed [rmsnorm {item['template']}] rmsnorm head\n"
    )


@pytest.mark.parametrize(
    ("kv", "title"),
    [
        (8, "Multi-head attention"),
        (2, "Grouped-query attention"),
        (1, "Multi-query attention"),
    ],
)
def test_attention_is_named_by_its_key_heads(kv, title):
    source = f"d_model = 64\nmodel = gqa(num_heads=8, num_kv_heads={kv})\n"
    [attn] = layers(view(source, {}, path=None))
    assert attn["title"] == title


def test_a_new_layer_registers_its_look(monkeypatch):
    monkeypatch.setitem(palette.names, "gqa", "Attention")
    monkeypatch.setitem(palette.categories, "rmsnorm", "other")
    monkeypatch.setitem(palette.titles.builders, "gqa", lambda c: "Mine")
    monkeypatch.setitem(palette.chips.builders, "rmsnorm", lambda c, ctx: ["mine"])
    v = view(
        "d_model = 64\nmodel = rmsnorm gqa(num_heads=2, num_kv_heads=2)\n",
        {},
        path=None,
    )
    norm, attn = layers(v)
    assert (norm["category"], norm["chips"], attn["title"]) == (
        "other",
        ["mine"],
        "Mine",
    )
    assert palette.name("gqa") == "Attention"


@pytest.mark.parametrize(
    "registry",
    [palette.titles, palette.chips, palette.templates, internals.internals],
    ids=lambda r: r.kind,
)
def test_every_entry_says_what_it_does(registry):
    """Its first paragraph is its row in the registries' reference."""
    assert all(inspect.getdoc(b) for b in registry.builders.values())
    assert set(registry.builders) <= set(LAYERS)


def test_inside_a_gated_mlp():
    [ffn] = layers(view_of(LLAMA), "gated_mlp")
    inside = ffn["internals"]
    assert inside["id"] == ffn["id"] + ":i" and inside["kind"] == "seq"
    gate, down = inside["items"]
    assert gate["kind"] == "prod" and gate["id"] == ffn["id"] + ":i.0"
    act = gate["factors"][0]["items"][1]
    assert act["title"] == "SiLU" and act["id"] == ffn["id"] + ":i.0.0.1"
    assert gate["factors"][1]["chips"] == ["4,096 → 14,336"]
    assert down["title"] == "Linear ↓" and down["chips"] == ["14,336 → 4,096"]
    assert not any(n["editable"] for n in nodes(inside))


def test_inside_a_mixture_of_experts_and_attention():
    v = view_of(EXAMPLES / "architectures/qwen3_next.bux")
    moe = layers(v, "moe")[0]["internals"]
    assert moe["kind"] == "sum"
    routed, shared = moe["terms"]
    weighted = routed["items"][0]
    assert weighted["kind"] == "prod"
    experts, router = weighted["factors"]
    assert experts["chips"][0] == "×512" and experts["internals"]["kind"] == "seq"
    assert [n["title"] for n in router["items"]] == ["Router", "Top-k"]
    assert shared["title"] == "Shared expert"
    attn = layers(v, "gqa")[0]["internals"]
    gated, out = attn["items"]
    assert gated["kind"] == "prod" and out["title"] == "Linear output"
    titles = [n["title"] for n in gated["factors"][0]["items"]]
    assert titles == [
        "Q, K, V projections",
        "QK-norm",
        "RoPE",
        "Scaled dot-product attention",
    ]
    delta = layers(v, "gdn")[0]["internals"]
    assert "Gated delta rule" in [n.get("title") for n in nodes(delta)]


def test_inside_latent_attention():
    v = view_of(EXAMPLES / "architectures/deepseek_v3.bux")
    mla = layers(v, "mla")[0]
    assert mla["title"] == "Multi-head latent attention"
    titles = [n["title"] for n in mla["internals"]["items"]]
    assert titles[0] == "Latent projections" and titles[-1] == "Linear output"
