"""bux.studio.forms: the inspector's forms of nodes and settings."""

from pathlib import Path

import pytest

from bux.lang import ir
from bux.studio.edits import apply
from bux.studio.forms import inspect
from bux.studio.targets import EditError, File

EXAMPLES = Path(__file__).parent.parent / "examples"
LLAMA = EXAMPLES / "architectures/llama3_8b.bux"
QWEN = EXAMPLES / "architectures/qwen3_next.bux"
ABLATION = EXAMPLES / "experiments/attention_ablation.bux"
SMOKE = EXAMPLES / "experiments/smoke.bux"
CURRICULUM = EXAMPLES / "experiments/curriculum.bux"
NEMOTRON = EXAMPLES / "architectures/nemotron/nemotron_3.5_bolt_74b_a9b.bux"
ATTN = "n.1.0.0.0.1"  # model -> block^n -> block -> [norm attn] -> norm attn -> attn
KEYS = {
    "key", "type", "doc", "required", "default", "value", "written", "where",
    "input", "options", "sweep", "editable", "note", "fields",
}  # fmt: skip


def form(path: Path, target: dict, picks: dict[str, int] | None = None) -> dict:
    return inspect(path.read_text(), target, picks=picks or {}, path=str(path))


def fields(f: dict) -> dict[str, dict]:
    """Return every field of a form by key, nested ones too."""
    out = {}
    todo = list(f["fields"])
    while todo:
        field = todo.pop(0)
        assert set(field) == KEYS
        out[field["key"]] = field
        todo += field["fields"] or []
    return out


def test_a_layers_form_has_every_field_in_order():
    f = form(LLAMA, {"node": ATTN})
    assert (f["title"], f["subtitle"]) == ("gqa", "attn")
    assert f["doc"].startswith("Grouped-query attention; MHA when")
    assert f["where"] == {"span": [130, 187], "file": None, "line": 6}
    keys = [field["key"] for field in f["fields"]]
    assert keys[:3] == ["num_heads", "num_kv_heads", "head_dim"]
    by = fields(f)
    assert by["num_kv_heads"] | {"doc": ""} == {
        "key": "num_kv_heads", "type": "int", "doc": "", "required": True,
        "default": None, "value": "8", "written": "8",
        "where": {"span": [148, 162], "file": None, "line": 6},
        "input": "int", "options": None, "sweep": None, "editable": True,
        "note": None, "fields": None,
    }  # fmt: skip
    assert by["head_dim"]["default"] == "none" and by["head_dim"]["written"] is None
    assert (
        by["rope"]["input"] == "config" and by["rope"]["written"] == "rope(base=500000)"
    )
    assert by["rope.base"]["written"] == "500000"
    assert by["rope.layout"]["input"] == "choice"
    assert by["rope.layout"]["options"] == ["half", "interleaved"]
    assert by["bias"]["input"] == "bool" and by["bias"]["doc"]


def test_a_sweep_and_a_modules_settings_say_where_they_are():
    by = fields(form(ABLATION, {"node": ATTN}, {"gqa|mha|mqa": 2}))
    kv = by["num_kv_heads"]
    assert kv["value"] == "1" and kv["written"] == "mha: 12 | gqa: 4 | mqa: 1"
    assert kv["sweep"] == ["mha: 12", "gqa: 4", "mqa: 1"]
    assert kv["where"]["line"] == 18 and kv["editable"]
    base = by["rope.base"]
    assert base["where"]["span"] is None and base["where"]["line"] == 6
    assert base["where"]["file"].endswith("llama3_8b.bux")
    assert base["note"] == "written in llama3_8b.bux; set here, in this file"


def test_what_a_macros_parameter_sets_is_not_editable():
    by = fields(form(NEMOTRON, {"node": "n.2.0.1"}))  # E(240, 2048)'s experts
    assert by["num_experts"]["written"] == "n"
    assert not by["num_experts"]["editable"]
    assert by["num_experts"]["note"] == "set by the parameter n of the macro E"
    assert not by["expert.intermediate_size"]["editable"]
    assert by["top_k"]["editable"] and by["top_k"]["written"] == "9"


def test_a_layer_of_a_module_without_a_name_is_read_only(tmp_path):
    (tmp_path / "lib.bux").write_text(
        "d_model = 64\nvocab_size = 512\nmodel = embed [rmsnorm] head\n"
    )
    (tmp_path / "main.bux").write_text("use lib\n")
    by = fields(form(tmp_path / "main.bux", {"node": "n.2"}))
    assert not by["tied"]["editable"]
    assert by["tied"]["note"] == "written in lib.bux, with no name to amend it by"


def test_repeats_blocks_and_residuals():
    f = form(LLAMA, {"node": "n.1"})
    assert (f["title"], f["subtitle"]) == ("repeat", None)
    [times] = f["fields"]
    assert (times["key"], times["value"], times["written"]) == ("times", "32", "32")
    assert times["input"] == "int" and times["editable"]
    f = form(ABLATION, {"node": "n.1.0"})
    assert (f["title"], f["subtitle"]) == ("block", "sequence")
    [block] = f["fields"]
    assert block["value"] == "[norm attn] [norm ffn]"
    assert block["where"]["line"] == 9 and block["where"]["span"] is None
    assert (
        block["note"] == "written in llama3_8b.bux; a definition here takes its place"
    )
    f = form(LLAMA, {"node": "n.1.0.0"})
    assert (f["title"], f["fields"]) == ("residual", [])
    f = form(QWEN, {"node": "n.1.0.0"})  # the D of "DDDA"
    assert f["title"] == "D" and fields(f)["D"]["value"] == "layer(delta)"
    source = "d_model = 8\nb(k) = rmsnorm^k\nmodel = b(3)\n"
    times = fields(inspect(source, {"node": "n"}, picks={}, path=None))["times"]
    assert not times["editable"]
    assert times["note"] == "the count is set by the parameter k of the macro b"


def test_the_models_form():
    by = fields(form(ABLATION, {"model": True}))
    assert list(by)[:4] == ["d_model", "vocab_size", "residual", "init"]
    assert by["d_model"]["written"] == "768" and by["d_model"]["where"]["line"] == 12
    assert by["vocab_size"]["note"] == (
        "written in llama3_8b.bux; a value here takes its place"
    )
    assert by["residual"]["value"] == "none" and by["residual"]["written"] is None


def test_training_settings():
    f = form(SMOKE, {"setting": "optimizer"})
    assert (f["title"], f["subtitle"]) == ("adamw", "optimizer")
    by = fields(f)
    assert by["lr"]["written"] == "3e-3" and by["lr"]["input"] == "float"
    assert by["b1"]["default"] == "0.9" and by["b1"]["written"] is None
    f = form(SMOKE, {"setting": "steps"})
    [steps] = f["fields"]
    assert (steps["key"], steps["value"], steps["input"]) == ("steps", "50", "int")
    assert f["doc"] == "optimizer steps"
    [steps] = form(ABLATION, {"setting": "steps"})["fields"]
    assert (
        steps["value"] == "5722" and steps["written"] == "budget // (batch * seq_len)"
    )
    [budget] = form(ABLATION, {"setting": "budget"})["fields"]
    assert budget["value"] == "3B" and budget["where"]["line"] == 21
    with pytest.raises(EditError, match="no setting"):
        form(SMOKE, {"setting": "nothing"})
    with pytest.raises(EditError, match="not a run"):
        form(LLAMA, {"setting": "optimizer"})


def test_a_mix_of_data_is_listed_read_only():
    f = form(ABLATION, {"setting": "data"})
    assert f["subtitle"] == "mix"
    [data] = f["fields"]
    assert data["written"] == "0.8 * web(split=[0, 0.999]) + 0.2 * code"
    assert data["input"] == "text" and data["editable"]
    assert [s["value"][:16] for s in data["fields"]] == [
        "0.8 * text(path=",
        "0.2 * text(path=",
    ]
    assert not any(s["editable"] for s in data["fields"])


def test_a_curriculums_phases():
    [curriculum] = form(CURRICULUM, {"setting": "curriculum"})["fields"]
    assert [phase["key"] for phase in curriculum["fields"]] == ["0", "1"]
    by = fields({"fields": curriculum["fields"]})
    assert by["0"]["written"] == "phase(at=0.5, seq_len=128, batch=8)"
    assert by["1.seq_len"]["written"] == "256" and by["1.seq_len"]["editable"]
    assert by["1.data"]["written"] is None
    f = form(CURRICULUM, {"setting": "batch_ramp"})
    assert (f["title"], fields(f)["steps"]["written"]) == ("ramp", "20")


def test_a_mix_of_data_inside_a_setting_is_a_field_of_text():
    source = SMOKE.read_text() + "web = synthetic\ncode = synthetic\n"
    source += 'eval = eval(name="mix", data=0.5 * web + 0.5 * code, every=10)\n'
    data = fields(inspect(source, {"setting": "eval"}, picks={}, path=str(SMOKE)))["data"]
    assert data["input"] == "text" and data["fields"] is None
    assert data["written"] == "0.5 * web + 0.5 * code"


def _layer_ids(f: File) -> list[str]:
    """Return the ids of the layers of `f`'s graph, one a place they are written."""
    found, seen = [], set()
    todo: list[tuple[ir.Node, tuple[int, ...]]] = [(f.trace.arch.graph, ())]
    while todo:
        node, path = todo.pop(0)
        span = f.trace.nodes.get(path)
        if isinstance(node, ir.Layer) and span is not None and span not in seen:
            seen.add(span)
            found.append(".".join(["n", *map(str, path)]))
        todo += [(c, (*path, k)) for k, c in enumerate(ir.children(node))]
    return found


def _agree(path: Path, target: dict) -> None:
    """Check that a set of each field of the form does what the form says: raises
    where it is not editable, and changes nothing where it sets what is written."""
    source = path.read_text()
    for field in fields(inspect(source, target, picks={}, path=str(path))).values():
        op = {"op": "set", "target": target, "key": field["key"]}
        value = field["written"] or field["value"]
        if not field["editable"]:
            with pytest.raises(EditError):
                apply(source, [op | {"value": value}], picks={}, path=str(path))
        elif field["written"] is not None and field["where"]["span"] is not None:
            new = apply(source, [op | {"value": value}], picks={}, path=str(path))
            assert new == source, (target, field["key"])


@pytest.mark.parametrize("path", [QWEN, ABLATION, NEMOTRON], ids=lambda p: p.stem)
def test_forms_of_layers_say_what_a_set_does(path):
    for node_id in _layer_ids(File(path.read_text(), {}, str(path))):
        _agree(path, {"node": node_id})


@pytest.mark.parametrize(
    ("path", "name"),
    [
        (SMOKE, "optimizer"),
        (SMOKE, "steps"),
        (ABLATION, "optimizer"),
        (ABLATION, "data"),
        (ABLATION, "budget"),
        (CURRICULUM, "curriculum"),
        (CURRICULUM, "batch_ramp"),
        (ABLATION, "d_model"),
    ],
)
def test_forms_of_settings_say_what_a_set_does(path, name):
    _agree(path, {"setting": name})
    _agree(path, {"model": True})
