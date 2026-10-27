"""bux.studio.edits: each op of the page, as a splice of the file's own text."""

import re
from pathlib import Path

import pytest

from bux import config as cfg
from bux.lang import check, ir
from bux.lang.printer import field_order, value_to_source
from bux.studio.edits import EditError, apply
from bux.studio.targets import File, node_at, place, value_at

EXAMPLES = Path(__file__).parent.parent / "examples"
LLAMA = EXAMPLES / "architectures/llama3_8b.bux"
QWEN = EXAMPLES / "architectures/qwen3_next.bux"
ABLATION = EXAMPLES / "experiments/attention_ablation.bux"
SMOKE = EXAMPLES / "experiments/smoke.bux"
CURRICULUM = EXAMPLES / "experiments/curriculum.bux"
ATTN = "n.1.0.0.0.1"  # model -> block^n -> block -> [norm attn] -> norm attn -> attn
QWEN_ATTN = "n.1.0.3.0.0.1"  # the A of "DDDA": layer(attn)'s mixer
BASE = "d_model = 64\nvocab_size = 512\n"


def edit(path: Path, *ops: dict, picks: dict[str, int] | None = None) -> str:
    return apply(path.read_text(), list(ops), picks=picks or {}, path=str(path))


def changed(path: Path, old: str, new: str) -> str:
    source = path.read_text()
    assert old in source
    return source.replace(old, new, 1)


def set_(target: dict | str, key: str | None, value: object) -> dict:
    target = {"node": target} if isinstance(target, str) else target
    return {"op": "set", "target": target, "key": key, "value": value}


def setting(name: str, key: str | None, value: str | None) -> dict:
    return set_({"setting": name}, key, value)


def on(source: str, *ops: dict, path: str | None = None) -> str:
    return apply(source, list(ops), picks={}, path=path)


# --- set, on layers ---------------------------------------------------------------

LLAMA_ATTN = "attn = gqa(num_heads=32, num_kv_heads=8, rope=rope(base=500000))"


@pytest.mark.parametrize(
    ("key", "value", "new"),
    [
        ("num_kv_heads", "4", "num_heads=32, num_kv_heads=4, rope=rope(base=500000)"),
        ("head_dim", "128", "num_heads=32, num_kv_heads=8, rope=rope(base=500000), head_dim=128"),
        ("rope.fraction", "0.5", "num_heads=32, num_kv_heads=8, rope=rope(base=500000, fraction=0.5)"),
        ("rope.base", None, "num_heads=32, num_kv_heads=8, rope=rope"),
        ("rope", "none", "num_heads=32, num_kv_heads=8, rope=none"),
    ],
)  # fmt: skip
def test_set_replaces_or_writes_where_the_call_is(key, value, new):
    expected = changed(LLAMA, LLAMA_ATTN, f"attn = gqa({new})")
    assert edit(LLAMA, set_(ATTN, key, value)) == expected


def test_set_turns_a_bare_name_into_a_call():
    expected = changed(LLAMA, "norm head\n", "norm head(tied=true)\n")
    assert edit(LLAMA, set_("n.3", "tied", "true")) == expected


def test_set_keeps_a_multi_line_call_and_its_alignment():
    expected = changed(
        QWEN,
        "rope=rope(base=10000000, fraction=0.25))",
        "rope=rope(base=10000000, fraction=0.25),\n              softcap=30.0)",
    )
    assert edit(QWEN, set_(QWEN_ATTN, "softcap", "30.0")) == expected
    expected = changed(QWEN, "head_dim=256, gated", "head_dim=128, gated")
    assert edit(QWEN, set_(QWEN_ATTN, "head_dim", "128")) == expected


def test_set_under_a_reference_writes_the_dotted_key_here():
    # qk_norm=norm: the eps of this attention's norm, not of every norm.
    new = edit(QWEN, set_(QWEN_ATTN, "qk_norm.eps", "1e-6"))
    assert "fraction=0.25),\n              qk_norm.eps=1e-6)" in new
    assert "norm    = rmsnorm(zero_centered=true)\n" in new


@pytest.mark.parametrize(
    ("source", "new"),
    [
        ("gqa(num_heads=4, num_kv_heads=2)", "gqa(num_heads=4, num_kv_heads=2, head_dim=8)"),
        ("gqa(\n    num_heads=4,\n    num_kv_heads=2,\n)", "gqa(\n    num_heads=4,\n    num_kv_heads=2,\n    head_dim=8,\n)"),
        ("gqa(\n    num_heads=4,\n    num_kv_heads=2\n)", "gqa(\n    num_heads=4,\n    num_kv_heads=2,\n    head_dim=8\n)"),
        ("gqa(num_heads=4, num_kv_heads=2,)", "gqa(num_heads=4, num_kv_heads=2, head_dim=8,)"),
    ],
)  # fmt: skip
def test_a_new_argument_follows_the_calls_style(source, new):
    text = f"{BASE}attn = {source}\nmodel = embed [rmsnorm attn] head\n"
    assert on(text, set_("n.1.0.1", "head_dim", "8")) == text.replace(source, new)


def test_set_none_removes_the_argument_with_its_comma():
    text = f"{BASE}attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8)\nmodel = embed [rmsnorm attn] head\n"
    expected = text.replace(", head_dim=8", "")
    assert on(text, set_("n.1.0.1", "head_dim", None)) == expected
    assert on(expected, set_("n.1.0.1", "head_dim", None)) == expected  # a default
    with pytest.raises(EditError, match="needs 'num_heads'"):
        on(text, set_("n.1.0.1", "num_heads", None))


def test_set_in_a_macro_body_that_its_parameter_writes_is_refused():
    text = f"{BASE}b(h) = [rmsnorm gqa(num_heads=h, num_kv_heads=2)]\nmodel = embed b(4) head\n"
    with pytest.raises(EditError, match="parameter h of the macro b"):
        on(text, set_("n.1.0.1", "num_heads", "8"))
    expected = text.replace("num_kv_heads=2", "num_kv_heads=4")
    assert on(text, set_("n.1.0.1", "num_kv_heads", "4")) == expected


def test_set_of_a_layer_in_a_module_amends_it_here():
    # attn.num_heads    = 12
    # attn.num_kv_heads = mha: 12 | gqa: 4 | mqa: 1
    line = "attn.num_kv_heads = mha: 12 | gqa: 4 | mqa: 1\n"
    expected = changed(ABLATION, line, f"{line}attn.head_dim     = 64\n")
    assert edit(ABLATION, set_(ATTN, "head_dim", "64")) == expected
    expected = changed(ABLATION, line, f"{line}attn.rope.base    = 1e6\n")
    assert edit(ABLATION, set_(ATTN, "rope.base", "1e6")) == expected
    expected = changed(ABLATION, "mha: 12 | gqa: 4 | mqa: 1", "2")
    assert edit(ABLATION, set_(ATTN, "num_kv_heads", "2")) == expected
    expected = changed(ABLATION, "ffn.intermediate_size = 2048\n", "")
    assert edit(ABLATION, set_("n.1.0.1.0.1", "intermediate_size", None)) == expected
    # norm = rmsnorm(eps=1e-5) in llama3_8b, which this file does not amend yet.
    expected = ABLATION.read_text() + "\nnorm.eps = 1e-6\n"
    assert edit(ABLATION, set_("n.1.0.0.0.0", "eps", "1e-6")) == expected


def test_set_none_of_a_modules_setting_is_refused():
    with pytest.raises(EditError, match="written in llama3_8b.bux"):
        edit(ABLATION, set_(ATTN, "rope", None))


def test_a_module_layer_without_a_name_is_not_amended(tmp_path):
    (tmp_path / "lib.bux").write_text(
        f"{BASE}norm = rmsnorm\nattn = gqa(num_heads=4, num_kv_heads=2)\n"
        "model = embed [norm attn]^2 norm head\n"
    )
    main = tmp_path / "main.bux"
    main.write_text("use lib\n\nattn.num_kv_heads = 1\n")
    expected = "use lib\n\nattn.num_kv_heads = 1\nattn.num_heads = 8\n"
    assert edit(main, set_("n.1.0.0.1", "num_heads", "8")) == expected
    with pytest.raises(EditError, match="lib.bux with no name to amend"):
        edit(main, set_("n.3", "tied", "true"))  # head, in lib's model
    assert edit(main, set_("n.1.0.0.1", "num_kv_heads", None)) == "use lib\n"


# --- set, on settings --------------------------------------------------------------


def test_set_of_training_settings():
    assert edit(SMOKE, setting("steps", None, "100")) == changed(
        SMOKE, "steps     = 50", "steps     = 100"
    )
    assert edit(SMOKE, setting("loss", "z", "1e-4")) == changed(
        SMOKE, "steps     = 50\n", "steps     = 50\nloss      = loss(z=1e-4)\n"
    )
    assert edit(SMOKE, setting("schedule", "warmup", None)) == changed(
        SMOKE, "wsd(warmup=10)", "wsd"
    )
    assert edit(SMOKE, setting("optimizer", None, "muon(lr=0.02)")) == changed(
        SMOKE, "adamw(lr=3e-3)", "muon(lr=0.02)"
    )
    assert edit(SMOKE, setting("optimizer", "optimizer", "muon(lr=0.02)")) == changed(
        SMOKE, "adamw(lr=3e-3)", "muon(lr=0.02)"
    )
    with pytest.raises(EditError, match="needs data"):
        edit(SMOKE, setting("data", None, None))


def test_set_of_a_modules_setting_amends_or_replaces_it_here():
    line = "optimizer.lr      = 1e-3 | 3e-3\n"
    expected = changed(ABLATION, line, f"{line}optimizer.weight_decay = 0.0\n")
    assert edit(ABLATION, setting("optimizer", "weight_decay", "0.0")) == expected
    expected = changed(ABLATION, line, "optimizer         = muon(lr=0.02)\n")
    assert edit(ABLATION, setting("optimizer", None, "muon(lr=0.02)")) == expected
    with pytest.raises(EditError, match="the schedule takes 1200 steps"):
        edit(ABLATION, setting("steps", None, "1000"))
    expected = changed(ABLATION, line, f"{line}steps             = 2000\n")
    assert edit(ABLATION, setting("steps", None, "2000")) == expected
    assert edit(ABLATION, setting("budget", None, "6B")) == changed(
        ABLATION, "budget = 3B", "budget = 6B"
    )


def test_set_of_model_settings():
    model = {"model": True}
    expected = changed(
        ABLATION, "d_model = 768\n", "d_model = 768\nvocab_size = 32000\n"
    )
    assert edit(ABLATION, set_(model, "vocab_size", "32000")) == expected
    assert edit(ABLATION, set_(model, "d_model", "1536")) == changed(
        ABLATION, "d_model = 768", "d_model = 1536"
    )


def test_set_of_a_curriculums_phases():
    expected = changed(CURRICULUM, "at=0.5, seq_len=128", "at=0.5, seq_len=96")
    assert edit(CURRICULUM, setting("curriculum", "0.seq_len", "96")) == expected
    expected = changed(CURRICULUM, "batch=4)]", "batch=4, data=synthetic)]")
    assert edit(CURRICULUM, setting("curriculum", "1.data", "synthetic")) == expected
    expected = changed(CURRICULUM, "batch_ramp = ramp(start=4, steps=20)\n", "")
    assert edit(CURRICULUM, setting("batch_ramp", None, None)) == expected
    source = CURRICULUM.read_text()
    whole = source[source.index("[phase") :].rstrip("\n")
    expected = changed(CURRICULUM, whole, "[phase(at=0.5, seq_len=128),]")
    curriculum = setting("curriculum", None, "[phase(at=0.5, seq_len=128),]")
    assert edit(CURRICULUM, curriculum) == expected


# --- repeat, insert, delete, wrap, unwrap --------------------------------------------


def test_repeat():
    assert edit(LLAMA, {"op": "repeat", "node": "n.1", "times": 16}) == changed(
        LLAMA, "block^32", "block^16"
    )
    assert edit(LLAMA, {"op": "repeat", "node": "n.1", "times": 1}) == changed(
        LLAMA, "block^32", "block"
    )
    assert edit(LLAMA, {"op": "repeat", "node": "n.2", "times": 2}) == changed(
        LLAMA, "block^32 norm", "block^32 norm^2"
    )
    assert edit(QWEN, {"op": "repeat", "node": "n.1", "times": 6}) == changed(
        QWEN, '"DDDA"^12', '"DDDA"^6'
    )
    assert edit(LLAMA, set_("n.1", "times", "8")) == changed(LLAMA, "^32", "^8")


@pytest.mark.parametrize(
    ("model", "times", "result"),
    [
        ("n = 3\nmodel = embed rmsnorm^n head", 4, "n = 4\nmodel = embed rmsnorm^n head"),
        ("model = embed (rmsnorm rmsnorm)^2 head", 1, "model = embed (rmsnorm rmsnorm) head"),
        ("model = embed [rmsnorm + gelu] head", 2, "model = embed [rmsnorm + gelu]^2 head"),
        ("x = rmsnorm gelu\nmodel = embed x head", 3, "x = rmsnorm gelu\nmodel = embed x^3 head"),
        ("n = 3\nmodel = embed rmsnorm^(n + 1) head", 4, "written as (n + 1)"),
        ("b(k) = rmsnorm^k\nmodel = embed b(3) head", 4, "parameter k of the macro b"),
    ],
)  # fmt: skip
def test_repeat_counts_as_written(model, times, result):
    op = {"op": "repeat", "node": "n.1", "times": times}
    if "model" not in result:
        with pytest.raises(EditError, match=re.escape(result)):
            on(BASE + model + "\n", op)
    else:
        assert on(BASE + model + "\n", op) == BASE + result + "\n"


def insert(node: str, where: str, text: str) -> dict:
    return {"op": "insert", "node": node, "where": where, "text": text}


def test_insert_in_compositions():
    assert edit(LLAMA, insert("n.1", "after", "rmsnorm")) == changed(
        LLAMA, "block^32 norm", "block^32 rmsnorm norm"
    )
    assert edit(LLAMA, insert("n.1", "before", "rmsnorm")) == changed(
        LLAMA, "embed block^32", "embed rmsnorm block^32"
    )
    # Inside the repeat: composition binds looser than ^.
    assert edit(LLAMA, insert("n.1.0", "after", "rmsnorm")) == changed(
        LLAMA, "block^32", "(block rmsnorm)^32"
    )
    assert edit(LLAMA, insert(ATTN, "after", "rmsnorm")) == changed(
        LLAMA, "[norm attn]", "[norm attn rmsnorm]"
    )
    assert edit(QWEN, insert("n.1.0.0.0.0.0", "after", "rmsnorm")) == changed(
        QWEN, "[norm mixer]", "[norm rmsnorm mixer]"
    )


@pytest.mark.parametrize(
    ("model", "node", "where", "text", "result"),
    [
        ("x = rmsnorm + rmsnorm\nmodel = embed [x] head", "n.1.0", "after", "rmsnorm", "model = embed [x rmsnorm] head"),
        ("model = embed [rmsnorm + rmsnorm] head", "n.1.0", "before", "rmsnorm", "[rmsnorm (rmsnorm + rmsnorm)]"),
        ("model = embed (rmsnorm rmsnorm)^2 head", "n.1.0.1", "after", "rmsnorm | gelu", "(rmsnorm rmsnorm (rmsnorm | gelu))^2"),
        ("model = embed rmsnorm -> head", "n.1", "after", "rmsnorm", "embed rmsnorm -> rmsnorm -> head"),
        ("model = embed [rmsnorm] head", "n.1.0", "after", "-gelu", "[rmsnorm (-gelu)]"),
        ("model = embed [-gelu] head", "n.1.0", "before", "rmsnorm", "[rmsnorm (-gelu)]"),
    ],
)  # fmt: skip
def test_insert_keeps_precedence(model, node, where, text, result):
    assert result in on(BASE + model + "\n", insert(node, where, text))


def test_insert_of_a_layer_without_its_required_settings_is_left_to_fill():
    new = edit(LLAMA, insert("n.2", "before", "gqa"))
    assert new == changed(LLAMA, "block^32 norm", "block^32 gqa norm")
    assert check(new).errors[0].message == "gqa needs 'num_heads'"
    with pytest.raises(EditError, match="cannot read"):
        edit(LLAMA, insert("n.2", "before", "gqa(num_heads=4"))


def test_structure_in_modules_and_patterns_is_refused():
    with pytest.raises(EditError, match="llama3_8b.bux"):
        edit(ABLATION, insert("n.1.0.0", "after", "rmsnorm"))
    with pytest.raises(EditError, match="pattern"):
        edit(QWEN, {"op": "delete", "node": "n.1.0.0"})
    with pytest.raises(EditError, match="llama3_8b.bux"):
        edit(ABLATION, {"op": "retype", "node": ATTN, "type": "mla"})


def delete(node: str) -> dict:
    return {"op": "delete", "node": node}


def test_delete():
    assert edit(LLAMA, delete("n.1.0.0.0.0")) == changed(LLAMA, "[norm attn]", "[attn]")
    assert edit(LLAMA, delete("n.1")) == changed(
        LLAMA, "embed block^32 norm", "embed norm"
    )
    with pytest.raises(EditError, match="only part of the definition of model"):
        edit(LLAMA, delete("n"))


@pytest.mark.parametrize(
    ("model", "node", "result"),
    [
        ("x = rmsnorm + gelu\nmodel = embed [x] head", "n.1.0.0", "x = gelu"),
        ("model = embed [rmsnorm - gelu] head", "n.1.0.0", "[-gelu]"),
        ("model = embed [rmsnorm - gelu] head", "n.1.0.1", "[rmsnorm]"),
        ("model = embed\n        rmsnorm gelu\n        rmsnorm head", "n.3", "model = embed\n        rmsnorm gelu\n        head"),
        ("model = embed\n        rmsnorm gelu\n        rmsnorm head", "n.2", "model = embed\n        rmsnorm\n        rmsnorm head"),
    ],
)  # fmt: skip
def test_delete_keeps_the_rest(model, node, result):
    assert result in on(BASE + model + "\n", delete(node))
    with pytest.raises(EditError, match="only part of its residual branch"):
        on(BASE + "model = embed [rmsnorm] head\n", delete("n.1.0"))


def test_wrap_and_unwrap():
    wrap = {"op": "wrap", "node": "n.2"}
    assert edit(LLAMA, wrap) == changed(
        LLAMA, "block^32 norm head", "block^32 [norm] head"
    )
    unwrap = {"op": "unwrap", "node": "n.1.0.0"}
    assert edit(LLAMA, unwrap) == changed(
        LLAMA, "[norm attn] [norm ffn]", "norm attn [norm ffn]"
    )
    with pytest.raises(EditError, match="not a residual"):
        edit(LLAMA, {"op": "unwrap", "node": "n.2"})
    text = BASE + "model = embed [rmsnorm + gelu] head\n"
    assert on(text, {"op": "unwrap", "node": "n.1"}) == text.replace(
        "[rmsnorm + gelu]", "(rmsnorm + gelu)"
    )
    text = BASE + "r = [rmsnorm gelu]\nmodel = embed r^2 head\n"
    assert on(text, {"op": "unwrap", "node": "n.1.0"}) == text.replace(
        "[rmsnorm gelu]", "rmsnorm gelu"
    )


def test_retype_keeps_the_settings_the_new_type_takes():
    new = edit(LLAMA, {"op": "retype", "node": ATTN, "type": "mla"})
    assert new == changed(
        LLAMA, LLAMA_ATTN, "attn = mla(num_heads=32, rope=rope(base=500000))"
    )
    new = edit(LLAMA, {"op": "retype", "node": "n.2", "type": "layernorm"})
    assert new == changed(LLAMA, "rmsnorm(eps=1e-5)", "layernorm(eps=1e-5)")
    with pytest.raises(EditError, match="unknown layer"):
        edit(LLAMA, {"op": "retype", "node": ATTN, "type": "gqaa"})
    text = BASE + "model = embed [rmsnorm swish] head\n"
    assert on(
        text, {"op": "retype", "node": "n.1.0.1", "type": "gelu"}
    ) == text.replace("swish", "gelu")


# --- axis --------------------------------------------------------------------------


def axis(name: str, *options: str) -> dict:
    return {"op": "axis", "axis": name, "options": list(options)}


def test_axis_rewrites_the_alternatives():
    sweep = "mha: 12 | gqa: 4 | mqa: 1"
    assert edit(ABLATION, axis("gqa|mha|mqa", "mha: 12", "mqa: 1")) == changed(
        ABLATION, sweep, "mha: 12 | mqa: 1"
    )
    assert edit(ABLATION, axis("gqa|mha|mqa", "gqa: 4")) == changed(
        ABLATION, sweep, "4"
    )
    lr = "optimizer.lr=1e-3|3e-3"
    assert edit(ABLATION, axis(lr, "1e-3", "3e-3", "1e-2")) == changed(
        ABLATION, "1e-3 | 3e-3", "1e-3 | 3e-3 | 1e-2"
    )
    assert edit(ABLATION, axis(lr, "1e-3")) == changed(ABLATION, "1e-3 | 3e-3", "1e-3")
    with pytest.raises(EditError, match="no axis"):
        edit(ABLATION, axis("nothing", "1"))


def test_axis_labels_written_in_several_places_change_together():
    text = (
        f"{BASE}attn = gqa(num_heads=4, num_kv_heads=s: 2 | l: 4)\n"
        "model = embed [rmsnorm attn] head\n"
        "data = synthetic\noptimizer = adamw(lr=s: 1e-3 | l: 3e-3)\n"
        "seq_len = 16\nbatch = 2\nsteps = 2\n"
    )
    with pytest.raises(EditError, match="more than one place"):
        on(text, axis("l|s", "s: 2", "m: 3"))
    new = on(text, axis("l|s", "l: 1", "s: 2"))  # the same labels: one site
    assert new == text.replace("s: 2 | l: 4", "l: 1 | s: 2")


def test_axis_of_a_whole_part_keeps_its_brackets():
    text = BASE + "model = embed (rmsnorm | gelu) head\n"
    assert on(
        text, axis("model=rmsnorm|gelu", "rmsnorm", "gelu", "swish")
    ) == text.replace("(rmsnorm | gelu)", "(rmsnorm | gelu | swish)")
    assert on(text, axis("model=rmsnorm|gelu", "gelu")) == text.replace(
        "(rmsnorm | gelu)", "gelu"
    )


# --- apply -------------------------------------------------------------------------


def test_ops_apply_one_after_another():
    ops = [set_(ATTN, "num_kv_heads", "4"), {"op": "repeat", "node": "n.1", "times": 2}]
    expected = changed(LLAMA, "num_kv_heads=8", "num_kv_heads=4").replace("^32", "^2")
    assert edit(LLAMA, *ops) == expected
    with pytest.raises(EditError, match="unknown edit"):
        edit(LLAMA, {"op": "paint"})
    with pytest.raises(EditError):
        on("d_model = 64\nmodel = nrom\n", set_("n", "x", "1"))


def test_picks_select_the_run_that_node_ids_name():
    picks = {"gqa|mha|mqa": 2}  # mqa
    new = edit(ABLATION, set_(ATTN, "num_kv_heads", "1"), picks=picks)
    assert new == changed(ABLATION, "mha: 12 | gqa: 4 | mqa: 1", "1")


# --- Every layer of the zoo ------------------------------------------------------------


def _layers(f: File):
    def walk(node: ir.Node, path: tuple[int, ...]):
        yield path, node
        for k, child in enumerate(ir.children(node)):
            yield from walk(child, (*path, k))

    seen = set()
    for path, node in walk(f.trace.arch.graph, ()):
        span = f.trace.nodes.get(path)
        if not isinstance(node, ir.Layer) or span is None or span in seen:
            continue
        if not isinstance(node.config, cfg.ActivationConfig):
            seen.add(span)
            yield ".".join(["n", *map(str, path)]), node


def _field_to_set(f: File, node: ir.Layer) -> tuple[tuple[str, ...], str]:
    """Return a field of the layer and its value as source: the first that this file
    writes, else the first with a default."""
    written = f.trace.written(node.config)
    fields = [k for k in field_order(type(node.config)) if k != "type"]
    for name in fields:
        found = place(f, written, (name,))
        if found.found is not None and f.local(found.found.arg.span):
            return (name,), value_to_source(getattr(node.config, name))
    info = type(node.config).model_fields
    name = next(k for k in fields if not info[k].is_required())
    return (name,), value_to_source(getattr(node.config, name))


@pytest.mark.parametrize(
    "path", sorted(EXAMPLES.glob("architectures/**/*.bux")), ids=lambda p: p.stem
)
def test_set_of_a_field_of_every_layer_lands_in_its_trace(path):
    source = path.read_text()
    f = File(source, {}, str(path))
    for node_id, node in _layers(f):
        key, value = _field_to_set(f, node)
        op = set_(node_id, ".".join(key), value)
        try:
            new = apply(source, [op], picks={}, path=str(path))
        except EditError as err:
            assert "macro" in str(err) or "no name to amend" in str(err), (node_id, err)
            continue
        g = File(new, {}, str(path))
        layer = node_at(g, node_id)
        assert isinstance(layer, ir.Layer)
        assert value_at(layer.config, key) == value_at(node.config, key)
        found = place(g, g.trace.written(layer.config), key)
        assert found.found is not None, (node_id, key)
        assert g.text(found.found.arg.value.span) == value


def test_edits_at_the_end_of_a_file_without_a_newline(tmp_path):
    (tmp_path / "lib.bux").write_text(f"{BASE}model = embed [rmsnorm] head\n")
    main = tmp_path / "main.bux"
    main.write_text("use lib\n\nd_model = 32")
    assert edit(main, set_({"model": True}, "d_model", None)) == "use lib\n"
    assert edit(main, set_({"model": True}, "vocab_size", "64")) == (
        "use lib\n\nd_model = 32\nvocab_size = 64"
    )


def test_values_of_the_page_may_be_json():
    assert edit(LLAMA, set_("n.3", "tied", True)) == changed(
        LLAMA, "norm head\n", "norm head(tied=true)\n"
    )
    assert edit(LLAMA, set_(ATTN, "num_kv_heads", 4)) == changed(
        LLAMA, "num_kv_heads=8", "num_kv_heads=4"
    )


def test_a_whole_item_of_a_list():
    new = edit(CURRICULUM, setting("curriculum", "1", "phase(at=0.8, seq_len=256)"))
    assert new == changed(
        CURRICULUM, "phase(at=0.75, seq_len=256, batch=4)", "phase(at=0.8, seq_len=256)"
    )


# --- Comments and layout survive -------------------------------------------------------

INIT = EXAMPLES / "experiments/init_ablation.bux"
SCHEDULES = EXAMPLES / "experiments/schedule_sweep.bux"
NGPT = EXAMPLES / "architectures/ngpt.bux"


def attn(call: str) -> str:
    return f"{BASE}attn = gqa({call})\nmodel = embed [rmsnorm attn] head\n"


@pytest.mark.parametrize(
    ("call", "key", "value", "new"),
    [
        ("num_heads=4,\n    num_kv_heads=2  # kv\n", "head_dim", "8", "num_heads=4,\n    num_kv_heads=2,  # kv\n    head_dim=8\n"),
        ("\n    num_heads=4,  # q\n    num_kv_heads=2,  # kv\n", "head_dim", "8", "\n    num_heads=4,  # q\n    num_kv_heads=2,  # kv\n    head_dim=8,\n"),
        ("num_heads=4,  # q\n    num_kv_heads=2,  # kv\n    head_dim=8  # hd\n", "head_dim", None, "num_heads=4,  # q\n    num_kv_heads=2  # kv\n"),
        ("num_heads=4, num_kv_heads=2,  # kv\n    head_dim=8", "head_dim", None, "num_heads=4, num_kv_heads=2  # kv\n    "),
        ("\n    num_heads=4,\n    head_dim=8,\n    # two groups\n    num_kv_heads=2,\n", "head_dim", None, "\n    num_heads=4,\n    # two groups\n    num_kv_heads=2,\n"),
        ("\n    head_dim=8,  # hd\n    num_heads=4,  # q\n    num_kv_heads=2,\n", "head_dim", None, "\n    num_heads=4,  # q\n    num_kv_heads=2,\n"),
        ("num_heads=4, head_dim=8,  # hd\n    num_kv_heads=2", "head_dim", None, "num_heads=4, num_kv_heads=2"),
        ("num_heads=4, num_kv_heads=2, head_dim=8  # hd\n", "head_dim", None, "num_heads=4, num_kv_heads=2\n"),
    ],
)  # fmt: skip
def test_comments_stay_with_their_arguments(call, key, value, new):
    assert on(attn(call), set_("n.1.0.1", key, value)) == attn(new)


SEQ = "model = embed  # in\n        rmsnorm  # first\n        gelu  # second\n        head  # out\n"


@pytest.mark.parametrize(
    ("model", "node", "result"),
    [
        (SEQ, "n.1", "model = embed  # in\n        gelu  # second\n        head  # out\n"),
        (SEQ, "n.3", "model = embed  # in\n        rmsnorm  # first\n        gelu  # second\n"),
        ("model = embed  # in\n     -> rmsnorm  # norm\n     -> head\n", "n.2", "model = embed  # in\n     -> rmsnorm  # norm\n"),
        ("model = embed ->\n        rmsnorm ->\n        head\n", "n.2", "model = embed ->\n        rmsnorm\n"),
        ("b = [rmsnorm  # pre-norm\n     gelu]\nmodel = embed b head\n", "n.1.0.1", "b = [rmsnorm  # pre-norm\n     ]\nmodel = embed b head\n"),
        ("b = [gelu\n     -> rmsnorm  # norm\n     -> gelu]\nmodel = embed b head\n", "n.1.0.2", "b = [gelu\n     -> rmsnorm  # norm\n     ]\nmodel = embed b head\n"),
    ],
)  # fmt: skip
def test_comments_stay_with_their_items(model, node, result):
    assert on(BASE + model, delete(node)) == BASE + result


def test_statements_on_one_line_lose_their_semicolon():
    text = (
        attn("num_heads=4, num_kv_heads=2")
        + "attn.num_kv_heads = 1; attn.num_heads = 8\n"
    )
    assert on(text, set_("n.1.0.1", "num_kv_heads", None)) == text.replace(
        "attn.num_kv_heads = 1; ", ""
    )
    assert on(text, set_("n.1.0.1", "num_heads", None)) == text.replace(
        "; attn.num_heads = 8", ""
    )


def test_an_axis_keeps_its_lines_and_the_options_it_keeps():
    options = [
        "normal: normal",
        "olmo2: normal(truncate=3)",
        "gpt2: normal(out=normal(std=0.02 / (2 * layers) ^ 0.5))",
        "lecun: fan_in",
        "zero_out: normal(out=zeros)",
    ]
    labels = "gpt2|lecun|normal|olmo2|zero_out"
    dropped = [o for o in options if not o.startswith("lecun")]
    expected = changed(INIT, "\n     | lecun: fan_in", "")
    assert edit(INIT, axis(labels, *dropped)) == expected
    olmo2 = [o.replace("truncate=3", "truncate=2") for o in options]
    assert edit(INIT, axis(labels, *olmo2)) == changed(INIT, "truncate=3", "truncate=2")
    added = edit(INIT, axis(labels, *options, "small: normal(std=0.01)"))
    assert added == changed(
        INIT, "zero_out: normal(out=zeros)\n", "zero_out: normal(out=zeros)\n     | small: normal(std=0.01)\n"
    )  # fmt: skip
    where = "schedule=wsd(warmup=1k, decay=0.2, shape=sqrt)|deepseek_v3|kimi_k2|olmo_2|d2z|deepseek_llm"
    schedules = where.removeprefix("schedule=").split("|")
    assert edit(SCHEDULES, axis(where, *schedules, "wsd")) == changed(
        SCHEDULES, "| d2z | deepseek_llm\n", "| d2z | deepseek_llm | wsd\n"
    )


def test_an_axis_keeps_comments_and_nested_alternatives():
    text = (
        f"{BASE}model = embed [rmsnorm gqa(num_heads=4, num_kv_heads=2)] head\n"
        "data = synthetic\noptimizer = adamw(lr=1e-3  # small\n    | 3e-3  # big\n    )\n"
        "seq_len = 16\nbatch = 2\nsteps = 2\n"
    )
    lr = "optimizer.lr=1e-3|3e-3"
    # One left ends the sweep; the option kept keeps its comment.
    assert on(text, axis(lr, "1e-3")) == text.replace("\n    | 3e-3  # big", "")
    assert on(text, axis(lr, "1e-3", "3e-3", "1e-2")) == text.replace(
        "# big\n", "# big\n    | 1e-2\n"
    )
    assert on(text, axis(lr, "3e-3")) == text.replace("1e-3  # small\n    | ", "")
    nested = text.replace("[rmsnorm gqa", "(rmsnorm | (gelu | swish)) [rmsnorm gqa")
    alts = "model=rmsnorm|gelu | swish"  # as the view writes the option: gelu | swish
    assert on(nested, axis(alts, "rmsnorm", "gelu | swish", "relu")) == nested.replace(
        "(gelu | swish))", "(gelu | swish) | relu)"
    )
    assert on(nested, axis(alts, "rmsnorm", "gelu | relu")) == nested.replace(
        "(gelu | swish))", "(gelu | relu))"
    )


def test_the_last_line_of_a_file_that_uses_a_module_goes_without_a_final_newline(
    tmp_path,
):
    (tmp_path / "base.bux").write_text(BASE)
    path = str(tmp_path / "main.bux")
    run = "use base\nnorm = rmsnorm\ndata = synthetic\noptimizer = adamw(lr=1e-3)\n"
    run += "seq_len = 16\nbatch = 2\nsteps = 2\n"
    model = run + "model = embed ->\n  norm ->\n  head"
    assert on(model, delete("n.2"), path=path) == run + "model = embed ->\n  norm\n"
    sweep = model + "\noptimizer.lr = 1e-3 |\n  3e-3 |\n  1e-2"
    lr = axis("optimizer.lr=1e-3|3e-3|1e-2", "1e-3", "3e-3")
    assert on(sweep, lr, path=path) == sweep.removesuffix(" |\n  1e-2") + "\n"


# --- What an op means ------------------------------------------------------------------


def test_the_negation_of_a_minus_b_is_not_edited_inside_b():
    # block = [(attn unit - _) step] ...: n.2.0.0.0.0.1 is the -_ of attn unit - _
    neg = "n.2.0.0.0.0.1"
    ops = [insert(neg, "after", "gelu"), {"op": "wrap", "node": neg}]
    for op in [*ops, {"op": "repeat", "node": neg, "times": 2}]:
        with pytest.raises(EditError, match="a - b"):
            edit(NGPT, op)
    assert edit(NGPT, {"op": "wrap", "node": f"{neg}.0"}) == changed(
        NGPT, "(attn unit - _)", "(attn unit - [_])"
    )  # the _ itself, inside the negation


def test_a_number_is_not_repeated():
    text = BASE + "model = embed [gelu * 0.5] head\n"
    with pytest.raises(EditError, match="number"):
        on(text, {"op": "repeat", "node": "n.1.0.1", "times": 3})


def test_a_retype_to_an_activation_drops_the_settings_written_after():
    text = attn("num_heads=4, num_kv_heads=2") + "attn.num_kv_heads = 1  # mqa\n"
    retype = {"op": "retype", "node": "n.1.0.1", "type": "swish"}
    assert on(text, retype) == attn("num_heads=4, num_kv_heads=2").replace(
        "gqa(num_heads=4, num_kv_heads=2)", "swish"
    )
