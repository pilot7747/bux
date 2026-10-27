"""bux.lang.compiler.trace: where each node and config of a compiled file is written."""

from pathlib import Path

import pytest

from bux import config as cfg
from bux.lang import compile, ir
from bux.lang.compiler import Trace, Written, trace
from bux.train import config as tcfg

EXAMPLES = Path(__file__).parent.parent / "examples"

SOURCE = """\
d_model    = 64
vocab_size = 512

norm  = rmsnorm
attn  = gqa(num_heads=4, num_kv_heads=2, rope=rope(base=1e4))
attn.num_kv_heads = 1
block = [norm attn]
model = embed block^2 norm head
"""
ATTN = (1, 0, 0, 1)  # model -> block^2 -> [norm attn] -> norm attn -> attn


def node_at(t: Trace, path: tuple[int, ...]) -> ir.Node:
    node = t.arch.graph
    for k in path:
        node = ir.children(node)[k]
    return node


def config_at(t: Trace, path: tuple[int, ...]) -> cfg.Config:
    node = node_at(t, path)
    assert isinstance(node, ir.Layer)
    return node.config


def where(t: Trace, config: object) -> Written:
    assert isinstance(config, cfg.Config)
    w = t.written(config)
    assert w is not None
    return w


def text(t: Trace, w: Written | None) -> str:
    assert w is not None
    return t.program.source[w.span.start : w.span.end]


def used(t: Trace, path: tuple[int, ...]) -> str:
    span = t.nodes[path]
    return t.program.source[span.start : span.end]


def train_of(t: Trace) -> tcfg.TrainConfig:
    assert t.train is not None
    return t.train


def test_nodes_are_found_where_they_are_used():
    t = trace(SOURCE)
    assert used(t, ()) == "model"
    assert used(t, (1,)) == "block^2"
    assert used(t, (1, 0)) == "block"  # [norm attn], at its use, which its label names
    assert used(t, (1, 0, 0)) == "norm attn"
    assert used(t, ATTN) == "attn"
    assert used(t, (3,)) == "head"


def test_configs_are_found_where_they_are_written():
    t = trace(SOURCE)
    assert text(t, where(t, config_at(t, (0,)))) == "embed"
    assert text(t, where(t, config_at(t, (2,)))) == "rmsnorm"  # bare, where defined
    w = where(t, config_at(t, ATTN))
    assert text(t, w) == "attn.num_kv_heads = 1"
    assert w.call is not None and w.call.amend
    assert text(t, w.base).startswith("gqa(num_heads=4")
    assert w.base is not None and w.base.base is None


def test_nested_configs_keep_where_they_are_written():
    t = trace(SOURCE.replace("attn.num_kv_heads = 1\n", ""))
    attn = config_at(t, ATTN)
    assert isinstance(attn, cfg.GQAConfig)
    assert text(t, where(t, attn.rope)) == "rope(base=1e4)"


def test_the_trace_compiles_what_compile_does():
    assert trace(SOURCE).arch == compile(SOURCE)


def test_training_settings_are_found_and_runs_are_picked():
    source = (
        SOURCE
        + "data = synthetic\noptimizer = adamw(lr=1e-3 | 3e-3)\n"
        + "seq_len = 64\nbatch = 8\nsteps = 10\n"
        + "curriculum = [phase(at=0.5, seq_len=128),]\n"
    )
    [axis] = trace(source).axes
    assert axis.where == "optimizer.lr"
    t = trace(source, choices={axis.key: axis.pick(1)})
    train = train_of(t)
    assert train.optimizer.lr == 3e-3
    assert text(t, where(t, train.optimizer)) == "adamw(lr=1e-3 | 3e-3)"
    assert where(t, train.data).call is None  # bare: synthetic
    [phase] = train.curriculum
    call = where(t, phase).call
    assert call is not None and call.name == "phase"


def test_modules_are_found_past_the_file():
    path = EXAMPLES / "experiments/attention_ablation.bux"
    source = path.read_text()
    t = trace(source, filename=str(path))
    assert t.nodes[(1, 0, 0)].start >= len(source)  # [norm attn], in llama3_8b
    w = where(t, config_at(t, (1, 0, 0, 0, 1)))
    # attn.num_heads = 12 and attn.num_kv_heads = mha: 12 | ..., one amendment
    assert w.span.start < len(source)
    assert w.base is not None and w.base.span.start >= len(source)  # gqa(...)


def _paths(node: ir.Node, path: tuple[int, ...] = ()):
    yield path, node
    for k, child in enumerate(ir.children(node)):
        yield from _paths(child, (*path, k))


def test_inserted_merges_have_no_span():
    t = trace((EXAMPLES / "architectures/mhc.bux").read_text())
    merges = [
        path
        for path, node in _paths(t.arch.graph)
        if isinstance(node, ir.Layer) and node.config.type == "merge"
    ]
    assert merges and all(path not in t.nodes for path in merges)


@pytest.mark.parametrize(
    "path",
    sorted(EXAMPLES.glob("architectures/**/*.bux")),
    ids=lambda p: p.stem,
)
def test_every_layer_of_the_zoo_is_found(path):
    source = path.read_text()
    t = trace(source, filename=str(path))
    for found, node in _paths(t.arch.graph):
        if isinstance(node, ir.Layer) and node.config.type != "merge":
            assert found in t.nodes, (found, node.config.type)
