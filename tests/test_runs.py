"""Runs and sweeps: training settings, modules, amendments, alternatives and mixtures."""

from pathlib import Path

import pytest

from bux import config as cfg
from bux.lang import BuxError, Run, check, compile, compile_runs
from bux.lang.lsp import features
from bux.lang.printer import run_to_source
from bux.lang.syntax import tokenize
from bux.train import config as tcfg

ROOT = Path(__file__).parent.parent
EXPERIMENTS = sorted(ROOT.joinpath("examples", "experiments").rglob("*.bux"))

ARCH = """
d_model = 16
vocab_size = 32
norm = rmsnorm
attn = gqa(num_heads=2, num_kv_heads=2)
ffn = gated_mlp(intermediate_size=32, activation=swish)
block = [norm attn] [norm ffn]
model = embed block^2 norm head
"""

TRAIN = """
data = synthetic
optimizer = adamw(lr=3e-3)
seq_len = 16
batch = 4
steps = 100
"""

MODULES = {"arch": ARCH, "recipe": TRAIN}
# A run made of the two, which the source after it redefines.
RUN = "use arch\nuse recipe\n"


def runs(source: str, **kw) -> list[Run]:
    kw.setdefault("modules", MODULES)
    return compile_runs(source, filename="exp.bux", **kw)


def names(source: str, **kw) -> list[str]:
    return [r.name for r in runs(source, **kw)]


def error(source: str, **kw) -> str:
    with pytest.raises(BuxError) as info:
        runs(source, **kw)
    return str(info.value)


def layers[T](run: Run, cls: type[T]) -> list[T]:
    return [l.config for l in run.arch.layers() if isinstance(l.config, cls)]


def layer[T](run: Run, cls: type[T]) -> T:
    return layers(run, cls)[0]


# --- Numbers ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "value"),
    [("128k", 128_000), ("10B", 10**10), ("1.1B", 1_100_000_000), ("2.5k", 2500)],
)
def test_counts_take_suffixes_exactly(text, value):
    [token, _] = tokenize(text)
    assert token.value == value and type(token.value) is int


def test_a_suffix_must_end_the_number():
    assert [t.text for t in tokenize("4kv")[:-1]] == ["4", "kv"]
    assert compile("d_model = 2k\nmodel = rmsnorm").d_model == 2000


# --- Training settings ------------------------------------------------------------------


def test_a_file_with_training_settings_is_a_run():
    [run] = runs(RUN + "steps = 20B // (batch * seq_len) // 1M")
    assert run.name == "exp" and run.choices == ()
    assert run.train.steps == 20_000_000_000 // 64 // 1_000_000
    assert run.train.schedule == tcfg.ConstantSchedule()
    assert isinstance(run.train.optimizer, tcfg.AdamWConfig)


def test_an_architecture_alone_is_not_a_run():
    assert check(ARCH).run is None and check(ARCH).errors == []
    assert "does not say how to train" in error(ARCH)


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ("steps = true", "'steps' must be an integer, found a boolean"),
        ("steps = 1.5", "'steps' must be an integer, found 1.5"),
        ("steps = 0", "'steps' must be greater than 0, found 0"),
        ("optimizer = rmsnorm", "'optimizer' must be adamw(...) or muon(...), found"),
        (
            "data = 3",
            "'data' must be text(...), chat(...), tokens(...), synthetic(...) or a mix",
        ),
        ("schedule = wsd(warmup=90)", "the schedule takes 110 steps (90 of warmup"),
    ],
)
def test_settings_are_checked_where_they_are_written(setting, message):
    text = error(RUN + setting)
    assert message in text and setting.split(" = ")[1] in text


def test_a_run_lists_what_it_lacks():
    text = error(ARCH + "data = synthetic\nbatch = 4")
    assert "a run needs optimizer, seq_len and steps" in text
    assert "optimizer = adamw(lr=3e-4)" in text


def test_sequences_must_fit_learned_positions():
    source = ARCH.replace("embed block", "embed(max_positions=8) block") + TRAIN
    assert "longer than the 8 positions the embedding learns" in error(source)


def test_a_training_config_is_not_a_layer():
    assert "adamw(...) is a training setting, not a layer" in error(
        RUN + "model = embed adamw(lr=1) head"
    )


# --- Modules --------------------------------------------------------------------------


def test_modules_are_used_by_name_and_redefined_everywhere():
    # `block`, written in arch, picks up the attention defined here.
    [run] = runs("use arch\nuse recipe\nattn = gqa(num_heads=2, num_kv_heads=1)")
    assert layer(run, cfg.GQAConfig).num_kv_heads == 1
    assert run.train.steps == 100


def test_inside_its_definition_a_name_means_the_one_it_replaces():
    [run] = runs("use arch\nuse recipe\nffn = ffn(intermediate_size=64)")
    assert layer(run, cfg.MLPConfig).intermediate_size == 64
    assert layer(run, cfg.MLPConfig).activation == "swish"


def test_a_module_used_twice_is_placed_once_below_both():
    modules = {
        "base": "x = 1\ny = x",
        "a": "use base\nx = 2",
        "b": "use base\nz = 3",
    }
    report = check("use a\nuse b", modules=modules)
    assert report.values["y"] == 2 and report.errors == []


def test_module_errors_point_into_the_module():
    text = error(
        "use arch\nuse broken", modules={**MODULES, "broken": "a = 1\nb = gqa(x=1)"}
    )
    assert "broken.bux:2:" in text and "gqa has no setting 'x'" in text


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("use archh", "no module named 'archh'"),
        ("use 'models/arch.bux'", "modules are found by name, not by path"),
        ("use loop", "modules use each other: exp -> loop -> exp"),
    ],
)
def test_module_errors(source, message):
    found = {**MODULES, "loop": "use exp", "exp": source}
    with pytest.raises(BuxError) as info:
        compile_runs(source, filename="exp.bux", modules=found)
    assert message in str(info.value)


def test_modules_are_found_on_the_search_path(tmp_path, monkeypatch):
    (tmp_path / "zoo" / "family").mkdir(parents=True)
    (tmp_path / "zoo" / "family" / "tiny.bux").write_text(ARCH)
    (tmp_path / "pyproject.toml").write_text('[tool.bux]\npath = ["zoo"]\n')
    (tmp_path / "runs").mkdir()
    run_file = tmp_path / "runs" / "exp.bux"
    run_file.write_text("use tiny\n" + TRAIN)
    monkeypatch.delenv("BUX_PATH", raising=False)
    [run] = compile_runs(run_file.read_text(), filename=str(run_file))
    assert run.arch.d_model == 16
    (tmp_path / "zoo" / "tiny.bux").write_text(ARCH)
    with pytest.raises(BuxError, match="there are 2 modules named 'tiny'"):
        compile_runs(run_file.read_text(), filename=str(run_file))


# --- Amendments and the command line ------------------------------------------------------


def test_amendments_set_one_setting_and_merge():
    source = "use arch\nuse recipe\noptimizer.lr = 1e-2\noptimizer.weight_decay = 0"
    [run] = runs(source)
    assert run.train.optimizer == tcfg.AdamWConfig(lr=1e-2, weight_decay=0.0)


def test_an_amendment_applies_over_the_same_files_definition():
    [run] = runs(ARCH + TRAIN + "attn.rope = rope(base=1e6)")
    rope = layer(run, cfg.GQAConfig).rope
    assert rope is not None and rope.base == 1e6


def test_an_amendment_keeps_a_layer_shared():
    shared = ARCH.replace("attn = ", "shared attn = ") + TRAIN
    [err] = check(shared + "x = attn(bias=true)").errors
    assert "'attn' is shared, so it cannot be changed here" in err.message
    [run] = runs(shared + "attn.num_kv_heads = 1")
    shares = {l.share for l in run.arch.layers() if l.config.type == "gqa"}
    assert shares == {"attn"}
    assert {c.num_kv_heads for c in layers(run, cfg.GQAConfig)} == {1}


def test_an_amendment_reaches_a_layer_or_setting_defined_by_a_bare_type():
    source = ARCH + TRAIN + "layout = layout\nnorm.eps = 1e-3\nlayout.zero = 1"
    [run] = runs(source)
    assert {c.eps for c in layers(run, cfg.NormalizationConfig)} == {1e-3}
    assert run.train.layout.zero == 1
    bare = "attn = gqa\nattn.num_heads = 2\nattn.num_kv_heads = 1"
    [run] = runs(RUN + bare)
    assert layer(run, cfg.GQAConfig).num_kv_heads == 1
    swept = ARCH + TRAIN + "methods = a: methods | b: methods(microbatches=4)"
    amended = runs(swept, overrides=["methods = methods(microbatches=2)"])
    # Both alternatives compute the same run now, so they make one.
    assert [r.train.methods.microbatches for r in amended] == [2]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("foo.bar = 1", "nothing defines 'foo' for this to amend"),
        ("block.bar = 1", "'block' is a block, which has no settings to amend"),
        ("act = gelu\nact.x = 1", "'act' is an activation, which has no settings"),
        ("optimizer.lr = 1\noptimizer.lr = 2", "'lr' is given twice"),
    ],
)
def test_amendment_errors(source, message):
    assert message in str(check(ARCH + TRAIN + source).errors[0])


def test_command_line_settings_redefine_the_file():
    [run] = runs(RUN, overrides=["optimizer.lr=1e-2", "steps = 2k"])
    assert run.train.optimizer.lr == 1e-2 and run.train.steps == 2000
    assert names(RUN, overrides=["seed = 1 | 2"]) == ["exp[seed=1]", "exp[seed=2]"]


def test_command_line_errors_say_so():
    with pytest.raises(BuxError) as info:
        runs(RUN, overrides=["steps = 1.5"])
    assert "<command line>" in str(info.value)


# --- Alternatives ------------------------------------------------------------------------


def test_alternatives_make_a_grid_in_the_order_written():
    source = RUN + "attn.num_kv_heads = 2 | 1\noptimizer.lr = 1e-3 | 3e-3 | 1e-2"
    assert names(source) == [
        f"exp[attn.num_kv_heads={kv}, optimizer.lr={lr}]"
        for kv in (2, 1)
        for lr in ("1e-3", "3e-3", "1e-2")
    ]


def test_alternatives_with_the_same_labels_are_picked_together():
    source = RUN + "d_model = s: 16 | l: 32\noptimizer.lr = l: 1e-3 | s: 3e-3"
    picked = {r.name: (r.arch.d_model, r.train.optimizer.lr) for r in runs(source)}
    assert picked == {"exp[s]": (16, 3e-3), "exp[l]": (32, 1e-3)}


def test_alternatives_can_change_the_architecture():
    source = RUN + "block = pre: [norm attn] [norm ffn] | parallel: [norm (attn + ffn)]"
    pre, parallel = runs(source)
    assert parallel.name == "exp[parallel]" and pre.id != parallel.id
    assert sum(1 for l in parallel.arch.layers() if l.config.type == "rmsnorm") == 3


def test_only_the_alternatives_a_run_reaches_make_runs():
    assert names(RUN + "spare = 1 | 2 | 3") == ["exp"]
    report = check(RUN + "spare = 1 | 2\nseed = 1 | 2", modules=MODULES)
    assert [a.where for a in report.axes] == ["seed"]
    assert report.run is not None and report.run.name == "run[seed=1]"
    assert names(
        RUN + "attn = attn | gqa(num_heads=2, num_kv_heads=1)\nmodel = embed head"
    ) == ["exp"]
    nested = runs(RUN + "seed = 0 | (10 | 20)")
    assert [r.train.seed for r in nested] == [0, 10, 20]


def test_an_alternative_a_later_amendment_overrides_makes_no_runs():
    swept = RUN + "optimizer.lr = 1e-3 | 3e-3\nseed = 1 | 2"
    [one, two] = runs(swept, overrides=["optimizer.lr = 1e-2"])
    assert (one.name, two.name) == ("exp[seed=1]", "exp[seed=2]")
    assert one.train.optimizer.lr == two.train.optimizer.lr == 1e-2
    # Alternatives that compute the same run elsewhere still make runs where they differ.
    labeled = RUN + "d_model = s: 16 | l: 32\noptimizer.lr = s: 1e-3 | l: 1e-3"
    assert names(labeled) == ["exp[s]", "exp[l]"]


def test_a_macro_takes_the_same_alternative_at_every_call():
    source = (
        RUN
        + "layer(m) = [norm m] [norm (ffn | attn)]\nmodel = embed layer(attn)^2 norm head"
    )
    first, second = runs(source)
    assert [l.config.type for l in second.arch.layers()].count("gated_mlp") == 0
    assert [l.config.type for l in first.arch.layers()].count("gated_mlp") == 2


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("seed = a: 1", "a label names one of several alternatives"),
        ("seed = a: 1 | 2", "label every alternative or none of them"),
        ("seed = a: 1 | a: 2", "two alternatives have the same label"),
        ("seed = a: 1 | b: 2\nsteps = a: 10 | c: 20", "share some labels with others"),
    ],
)
def test_label_errors(source, message):
    assert message in error(RUN + source)


def test_check_looks_at_every_alternative():
    report = check(
        RUN + "attn = gqa(num_heads=2, num_kv_heads=2) | gqa(num_head=2)",
        modules=MODULES,
    )
    assert [e.message for e in report.errors] == ["gqa has no setting 'num_head'"]


# --- Data mixtures -------------------------------------------------------------------------


def test_data_mixes_as_a_weighted_sum():
    source = RUN + 'web = tokens(path="w")\ncode = tokens(path="c")\n'
    source += "data = 0.5 * (0.5 * web + 0.5 * synthetic) + 0.5 * code"
    [run] = runs(source)
    data = run.train.data
    assert isinstance(data, tcfg.MixtureConfig)
    assert [w.weight for w in data.sources] == [0.25, 0.25, 0.5]
    assert [w.source.type for w in data.sources] == ["tokens", "synthetic", "tokens"]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ("web * web", "data is mixed with + and weighted with *"),
        ("web + 1", "data is mixed with + and weighted with *"),
        ("-0.5 * web", "a data weight must be positive"),
        ("web + cod", "unknown name 'cod'"),
    ],
)
def test_mixture_errors(data, message):
    source = RUN + f'web = tokens(path="w")\ndata = {data}'
    assert message in error(source)


# --- Identity and printing -------------------------------------------------------------------


def test_the_id_follows_what_a_run_computes():
    [run] = runs(RUN)
    [renamed] = runs(ARCH.replace("ffn", "mlp_block") + TRAIN)
    [watched] = runs(RUN + "log = log(every=1)\ncheckpoint = checkpoint(every=5)")
    [changed] = runs(RUN + "seed = 1")
    assert run.id == renamed.id == watched.id != changed.id


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_experiments_compile_and_print_back_to_the_same_runs(path):
    report = check(path.read_text(), filename=str(path))
    assert report.errors == []
    if report.arch is None:  # a recipe: settings without a model
        return
    for run in compile_runs(path.read_text(), filename=str(path)):
        [again] = compile_runs(run_to_source(run))
        assert again.id == run.id and again.train == run.train


def test_experiments_sweep_as_documented():
    def count(stem: str) -> int:
        path = ROOT / "examples" / "experiments" / f"{stem}.bux"
        return len(compile_runs(path.read_text(), filename=str(path)))

    assert count("attention_ablation") == 6 and count("width_sweep") == 3


# --- Editor support -------------------------------------------------------------------------


def test_editor_shows_module_errors_at_the_use(tmp_path):
    (tmp_path / "base.bux").write_text("d_model = 8\nx = gqa(num_head=1)\n")
    path = tmp_path / "exp.bux"  # not saved yet: its folder is still where to look
    source = "use base\nmodel = rmsnorm\n"
    [d] = features.diagnostics(source, str(path))
    assert source[d.start : d.end] == "use base"
    assert d.message.startswith("base.bux:2: gqa has no setting 'num_head'")


def test_editor_knows_inherited_definitions(tmp_path):
    (tmp_path / "base.bux").write_text(ARCH)
    path = str(tmp_path / "exp.bux")
    source = "use base\nmodel = embed block norm head\n"
    found = features.hover(source, source.index("block"), path)
    assert found is not None and "Defined in `base.bux`" in found.markdown
    labels = [
        c.label
        for c in features.complete(source + "x = attn(num_", len(source) + 13, path)
    ]
    assert labels[:2] == ["num_heads", "num_kv_heads"]


@pytest.mark.parametrize("path", EXPERIMENTS[::2], ids=lambda p: p.stem)
def test_features_never_fail_on_runs_while_typing(path):
    text, name = path.read_text(), str(path)
    for k in range(0, len(text), 7):
        features.diagnostics(text[:k], name)
        features.complete(text, k, name)
        features.hover(text, k, name)
