"""bux.studio.targets: nodes, settings and fields, found in the file's text."""

from pathlib import Path

import pytest

from bux import config as cfg
from bux.lang import ir
from bux.lang import syntax as ast
from bux.studio.targets import (
    EditError,
    File,
    count,
    expression,
    extent,
    macro_parameter,
    node_at,
    node_path,
    options,
    place,
)

EXAMPLES = Path(__file__).parent.parent / "examples"
LLAMA = EXAMPLES / "architectures/llama3_8b.bux"
QWEN = EXAMPLES / "architectures/qwen3_next.bux"
ABLATION = EXAMPLES / "experiments/attention_ablation.bux"
NEMOTRON = EXAMPLES / "architectures/nemotron/nemotron_3.5_bolt_74b_a9b.bux"
ATTN = "n.1.0.0.0.1"  # model -> block^n -> block -> [norm attn] -> norm attn -> attn


def file_of(path: Path, picks: dict[str, int] | None = None) -> File:
    return File(path.read_text(), picks or {}, str(path))


def text(f: File, expr: ast.Expr) -> str:
    return f.text(extent(f.source, expr))


def layer(f: File, node: str) -> cfg.Config:
    found = node_at(f, node)
    assert isinstance(found, ir.Layer)
    return found.config


def test_node_ids_are_paths():
    assert node_path("n") == ()
    assert node_path("n.1.0.2") == (1, 0, 2)
    for bad in ("x.1", "n.a", "n.1."):
        with pytest.raises(EditError):
            node_path(bad)


def test_nodes_are_found_as_written():
    f = file_of(LLAMA)
    assert text(f, expression(f, "n").expr) == "embed block^32 norm head"
    assert text(f, expression(f, "n.1").expr) == "block^32"
    attn = expression(f, ATTN)
    assert text(f, attn.expr) == "attn"
    assert isinstance(attn.parent, ast.Seq) and text(f, attn.parent) == "norm attn"
    assert attn.statement.name == "block"
    with pytest.raises(EditError, match="no node"):
        node_at(f, "n.9")


def test_letters_of_patterns_and_modules_are_not_editable():
    f = file_of(QWEN)
    assert text(f, expression(f, "n.1").expr) == '"DDDA"^12'
    with pytest.raises(EditError, match="pattern"):
        expression(f, "n.1.0.0")  # the D of "DDDA"
    # The macro's body is: layer(mixer) = [norm mixer] [norm experts]
    assert text(f, expression(f, "n.1.0.0.0").expr) == "[norm mixer]"
    g = file_of(ABLATION)
    with pytest.raises(EditError, match="llama3_8b.bux"):
        expression(g, "n.1.0.0")  # [norm attn], in llama3_8b


def test_extents_hold_brackets_and_labels():
    source = "x = (a b)^2\ny = mha: 12 | mqa: 1\n"
    [x, y] = ast.parse(source)
    assert isinstance(x.body, ast.BinOp) and isinstance(y.body, ast.Alt)
    span = extent(source, x.body)
    assert source[span.start : span.end] == "(a b)^2"
    span = extent(source, x.body.left)
    assert source[span.start : span.end] == "(a b)"
    span = extent(source, y.body)
    assert source[span.start : span.end] == "mha: 12 | mqa: 1"
    assert options(source, y.body) == ["mha: 12", "mqa: 1"]
    [call] = ast.parse("z = f(a)\n")
    assert isinstance(call.body, ast.Call)
    span = extent("z = f(a)\n", call.body.args[0].value)
    assert (span.start, span.end) == (6, 7)  # a call's bracket is not a group


def test_a_field_is_placed_where_it_is_written():
    f = file_of(LLAMA)
    w = f.trace.written(layer(f, ATTN))
    found = place(f, w, ("num_kv_heads",))
    assert found.found is not None and text(f, found.found.arg.value) == "8"
    # Not written: into the latest call of this file, rope=rope(...) followed inline.
    found = place(f, w, ("rope", "fraction"))
    assert found.found is None and found.into is not None
    assert f.text(found.into.span) == "rope(base=500000)"
    assert found.rest == ("fraction",)


def test_a_field_written_in_a_module_is_placed_after_this_files_amendment():
    f = file_of(ABLATION)
    w = f.trace.written(layer(f, ATTN))
    found = place(f, w, ("rope", "base"))
    assert found.found is not None and not f.local(found.found.arg.span)
    assert found.into is not None and found.into.call is not None
    assert found.into.call.amend and found.rest == ("rope", "base")
    found = place(f, w, ("num_kv_heads",))
    assert found.found is not None and f.local(found.found.arg.span)
    assert text(f, found.found.arg.value) == "mha: 12 | gqa: 4 | mqa: 1"


def test_a_prefix_set_by_a_name_sets_what_is_under_it():
    f = file_of(QWEN)
    attn = "n.1.0.3.0.0.1"  # the A of "DDDA": layer(attn)'s mixer
    w = f.trace.written(layer(f, attn))
    found = place(f, w, ("qk_norm", "eps"))  # qk_norm=norm: not followed into norm
    assert found.found is None and found.into is w


def test_a_macros_parameter_is_found():
    f = file_of(NEMOTRON)
    w = f.trace.written(layer(f, "n.2.0.1"))  # experts(num_experts=n, ...)
    found = place(f, w, ("num_experts",))
    assert found.found is not None
    assert macro_parameter(f, found.found.arg) == ("n", "E")
    found = place(f, w, ("top_k",))  # written in experts = moe(...), below
    assert found.found is not None and macro_parameter(f, found.found.arg) is None


def test_counts_of_repeats():
    f = file_of(LLAMA)
    found = count(f, "n.1")
    assert found is not None and found.number is not None
    assert text(f, found.number) == "32"
    assert count(f, "n.2") is None  # norm
    source = "d_model = 8\nb(k) = rmsnorm^k\nmodel = b(3)\n"
    found = count(File(source, {}, None), "n")
    assert found is not None and found.number is None
    assert found.why == "set by the parameter k of the macro b"


def test_where_names_modules_and_lines():
    f = file_of(ABLATION)
    span = f.trace.nodes[(1, 0, 0)]
    assert f.where(span) == {
        "span": None,
        "file": f.origin(span),
        "line": 9,
    }
    assert f.origin(span).endswith("llama3_8b.bux")
    assert f.where(f.trace.nodes[()])["span"] == [485, 490]  # model
    with pytest.raises(EditError):
        File("d_model = 64\nmodel = nrom\n", {}, None)
