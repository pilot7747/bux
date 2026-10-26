"""bux.studio.picks: the run of a sweep that the page picked."""

from bux import config as cfg
from bux.lang import ir
from bux.lang.compiler import Trace
from bux.studio.picks import axis_id, traced

SOURCE = """\
d_model = 64
vocab_size = 512
attn = small: gqa(num_heads=4, num_kv_heads=2) | big: gqa(num_heads=8, num_kv_heads=2)
model = embed [rmsnorm attn] rmsnorm head
data = synthetic
optimizer = adamw(lr=1e-3 | 3e-3)
seq_len = 64
batch = 8
steps = 10
"""


def _attn(t: Trace) -> cfg.GQAConfig:
    [attn] = [n for n in ir.walk(t.arch.graph) if isinstance(n, ir.Layer)][2:3]
    assert isinstance(attn.config, cfg.GQAConfig)
    return attn.config


def test_axes_have_ids_by_labels_or_by_where_and_options():
    t = traced(SOURCE, {}, path=None)
    assert [axis_id(a) for a in t.axes] == ["big|small", "optimizer.lr=1e-3|3e-3"]


def test_picks_select_the_run():
    t = traced(SOURCE, {"big|small": 1, "optimizer.lr=1e-3|3e-3": 1}, path=None)
    assert t.train is not None and t.train.optimizer.lr == 3e-3
    assert _attn(t).num_heads == 8


def test_unknown_picks_and_options_are_ignored():
    t = traced(SOURCE, {"nothing": 3, "big|small": 7}, path=None)
    assert _attn(t).num_heads == 4


def test_an_axis_inside_an_option_is_met_once_it_is_picked():
    source = SOURCE.replace(
        "big: gqa(num_heads=8, num_kv_heads=2)",
        "big: gqa(num_heads=8, num_kv_heads=2 | 4)",
    )
    assert len(traced(source, {}, path=None).axes) == 2
    t = traced(source, {"big|small": 1, "attn.num_kv_heads=2|4": 1}, path=None)
    assert [axis_id(a) for a in t.axes] == [
        "big|small",
        "attn.num_kv_heads=2|4",
        "optimizer.lr=1e-3|3e-3",
    ]
    assert _attn(t).num_kv_heads == 4
