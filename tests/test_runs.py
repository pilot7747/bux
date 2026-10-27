"""Runs and sweeps: training settings, modules, amendments, alternatives and mixtures."""

from pathlib import Path

import pytest

from bux import config as cfg
from bux.data.pipeline import jobs
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

# How every weight matrix started before the default init was a normal.
LECUN = "init = fan_in(embed=fan_in(distribution=gaussian))\n"

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
        ("clip = 0", "'clip' must be greater than 0, found 0"),  # not off: no updates
        ("schedule = wsd(warmup=-5)", "a warmup is a count of steps from 0"),
        ("eval = eval(data=synthetic, seq_len=1)", "seq_len: Input should be greater"),
        ("schedule = wsd(warmup=90)", "the schedule takes 110 steps (90 of warmup"),
        ("schedule = power(warmup=0.95)", "takes 105 steps (95 of warmup and 10 of"),
        (
            "schedule = piecewise(warmup=50, pieces=[piece(until=0.4), piece()])",
            "the schedule takes 110 steps (50 of warmup and 60 of cooldown)",
        ),
    ],
)
def test_settings_are_checked_where_they_are_written(setting, message):
    text = error(RUN + setting)
    assert message in text and setting.split(" = ")[1] in text


def test_a_bare_name_a_setting_takes_is_that_string():
    """cosine is a schedule, and one of the shapes wsd's cooldown takes."""
    [run] = runs(RUN + "schedule = wsd(shape=cosine)")
    assert run.train.schedule == tcfg.WSDSchedule(shape="cosine")
    [run] = runs(RUN + "schedule = piecewise(pieces=[piece(to=0.5, shape=constant),])")
    assert isinstance(schedule := run.train.schedule, tcfg.PiecewiseSchedule)
    assert schedule.pieces == (tcfg.Piece(to=0.5, shape="constant"),)


def test_a_run_lists_what_it_lacks():
    text = error(ARCH + "data = synthetic\nbatch = 4")
    assert "a run needs optimizer, seq_len and steps" in text
    assert "optimizer = adamw(lr=3e-4)" in text


def test_sequences_must_fit_learned_positions():
    source = ARCH.replace("embed block", "embed(max_positions=8) block") + TRAIN
    assert "longer than the 8 positions the embedding learns" in error(source)
    # An eval's own rows too, which would read past the table.
    learned = ARCH.replace("embed block", "embed(max_positions=16) block") + TRAIN
    text = error(learned + "eval = eval(data=synthetic, seq_len=64)")
    assert "sequences of 64 tokens are longer than the 16 positions" in text
    assert "eval = eval(data=synthetic, seq_len=64)" in text  # where the error is


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


def test_a_missing_module_is_its_only_error():
    # ffn = ffn(...) means the missing module's ffn, not a cycle.
    source = "use archh\nffn = ffn(intermediate_size=8)\nattn.num_heads = 2"
    report = check(source, filename="exp.bux", modules=MODULES)
    assert [e.message for e in report.errors] == ["no module named 'archh'"]


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
    # A key inside one set before it changes it.
    [run] = runs(ARCH + TRAIN + "attn.rope = rope(base=1e6)\nattn.rope.fraction = 0.5")
    rope = layer(run, cfg.GQAConfig).rope
    assert rope is not None and (rope.base, rope.fraction) == (1e6, 0.5)


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
        (
            "attn.rope.fraction = 0.5\nattn.rope = rope(base=1e6)",
            "'rope' is given after 'rope.fraction', which it would undo",
        ),
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
    # Outside the grid, each place takes the label met first, wherever it lists it.
    assert check(source, filename="exp.bux", modules=MODULES).run == runs(source)[0]


def test_outside_the_grid_labels_start_where_the_grid_does():
    # steps comes first in the file's definitions, d_model first in compiling a run.
    source = "use recipe\nuse arch\nsteps = l: 10 | s: 20\nd_model = s: 16 | l: 32"
    first = runs(source)[0]
    assert (first.name, first.arch.d_model, first.train.steps) == ("exp[s]", 16, 20)
    assert check(source, filename="exp.bux", modules=MODULES).run == first
    assert compile(source, modules=MODULES) == first.arch


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


def test_a_list_setting_reads_off_a_config():
    source = (
        RUN
        + 'tokenizer = tokenizer(name="gpt2")\nweb = text(path="w", split=[0, 0.9])\n'
    )
    [run] = runs(source + 'data = text(path="v", split=web.split)')
    assert isinstance(run.train.data, tcfg.TextConfig)
    assert run.train.data.split == (0.0, 0.9)


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


# --- Evals ---------------------------------------------------------------------------------


def test_a_run_evaluates_one_held_out_set_or_a_list_of_named_ones():
    source = RUN + 'tokenizer = tokenizer(name="gpt2")\nweb = text(path="w")\n'
    [one] = runs(source + "eval = eval(data=web(split=[0.99, 1]))")
    named = (
        'eval(name="web", data=web(split=[0.99, 1])), eval(name=noise, data=synthetic)'
    )
    [many] = runs(source + f"eval = [{named}]")
    assert [e.name for e in tcfg.evals(one.train)] == ["eval"]
    assert [(e.name, e.data.type) for e in tcfg.evals(many.train)] == [
        ("web", "text"),
        ("noise", "synthetic"),
    ]
    assert [job.source.split for job in jobs(many.train)] == [(0.99, 1.0)]
    assert one.id == many.id == runs(source)[0].id  # evals only watch a run
    for run in (one, many):
        [again] = compile_runs(run_to_source(run))
        assert again.train == run.train


@pytest.mark.parametrize(
    ("evals", "message"),
    [
        (
            "[eval(data=synthetic), eval(data=synthetic, every=5)]",
            'each eval needs a name of its own: eval(name="web", ...)',
        ),
        (
            "[eval(name=a, data=synthetic), eval(name=b, data=web)]",
            "text data is tokenized, so the run needs a tokenizer",
        ),
        ("3", "'eval' must be eval(...), a list of eval(...) or none, found 3"),
    ],
)
def test_eval_errors(evals, message):
    assert message in error(RUN + f'web = text(path="w")\neval = {evals}')


# --- Identity and printing -------------------------------------------------------------------


def test_the_id_follows_what_a_run_computes():
    [run] = runs(RUN)
    [renamed] = runs(ARCH.replace("ffn", "mlp_block") + TRAIN)
    [watched] = runs(RUN + "log = log(every=1)\ncheckpoint = checkpoint(every=5)")
    [changed] = runs(RUN + "seed = 1")
    assert run.id == renamed.id == watched.id != changed.id


def test_runs_that_start_as_before_the_normal_init_keep_their_ids_but_moes():
    [normal] = runs(ARCH + TRAIN)
    [before] = runs(LECUN + ARCH + TRAIN)
    assert before.id == "b26aa7149f" != normal.id  # main 7ffc128's
    # Routed experts started sqrt(E) times smaller then, so an MoE's id is new.
    moe = ARCH.replace("[norm ffn]", "[norm moe(num_experts=4, top_k=2, expert=ffn)]")
    assert runs(LECUN + moe + TRAIN)[0].id != "ba4e9ec7b8"


MOE = "moe(num_experts=4, top_k=2, expert=ffn)"
MLA = (
    "mla(num_heads=2, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4,"
    " v_head_dim=4)"
)
CSA = "csa(num_heads=2, head_dim=8, sliding_window_size=4)"
# An indexer, open for the settings after it.
INDEXER = "indexer(num_heads=2, head_dim=8, "


@pytest.mark.parametrize(
    ("layer", "field", "before", "after"),
    [
        ("ffn", "up_offset", "0.0", "1.0"),
        (MOE, "router_bias", "false", "true"),
        (MOE, "center_bias", "false", "true"),
        (MOE, "bias_momentum", "0.5", "0.9"),
        ("mamba2(state_size=8, head_dim=8)", "num_heads", "none", "4"),
        ("gqa(num_heads=2, num_kv_heads=2)", "power_scale", "none", "0.1"),
        (
            "gqa(num_heads=2, num_kv_heads=2, gated=true)",
            "gate_head_bias",
            "false",
            "true",
        ),
        (
            "gqa(num_heads=2, num_kv_heads=2, qk_norm=norm)",
            "qk_norm_all_heads",
            "false",
            "true",
        ),
        (
            f"gqa(num_heads=2, num_kv_heads=2, indexer={INDEXER}top_k=4))",
            "relu",
            "true",
            "false",
        ),
        (
            f"gqa(num_heads=2, num_kv_heads=2, indexer={INDEXER}top_k=8, block_size=4))",
            "pool",
            "mean",
            "max",
        ),
        (
            f"gqa(num_heads=2, num_kv_heads=2, indexer={INDEXER}top_k=4, weighted=false))",
            "per_group",
            "false",
            "true",
        ),
        # With the gate it biases, which they have not by default.
        (MLA, "gate_head_bias", "false", "true, gated=true"),
        (CSA, "gate_head_bias", "false", "true, gated=true"),
    ],
)
def test_a_layer_field_newer_than_ids_keeps_them_where_it_computes_as_before(
    layer, field, before, after
):
    def run_of(value: str | None) -> Run:
        given = "" if value is None else f"{field}={value}"
        # Into the first settings that close: the indexer's, inside the layer's.
        new = (
            layer.replace(")", f", {given})", 1)
            if "(" in layer
            else f"{layer}({given})"
        )
        block = f"block = [norm {new}]"
        source = LECUN + ARCH.replace("block = [norm attn] [norm ffn]", block)
        [run] = runs(source + TRAIN)
        return run

    # The ids before these fields: of main 0f0cfe2, of d4a8e27 for the attention's,
    # and of a4268fb for the indexer's; the MoE's since its experts start by their
    # own fan in (init).
    old = {
        ("ffn", "up_offset"): "a8fa406e9b",
        ("moe", "router_bias"): "087eeedead",
        ("moe", "center_bias"): "087eeedead",
        ("moe", "bias_momentum"): "087eeedead",
        ("mamba2", "num_heads"): "ccdcaff2f8",
        ("gqa", "power_scale"): "4de8bfe6fd",
        ("gqa", "gate_head_bias"): "370e873062",
        ("gqa", "qk_norm_all_heads"): "9a5d6c3ed2",
        ("mla", "gate_head_bias"): "fa0bdd5558",
        ("csa", "gate_head_bias"): "e1e9cc3d90",
        ("gqa", "relu"): "610bb9cf51",
        ("gqa", "pool"): "b7e22463de",
        ("gqa", "per_group"): "73f1c2e501",
    }[layer.split("(")[0], field]
    assert run_of(None).id == run_of(before).id == old != run_of(after).id


def test_strings_that_name_a_setting_print_quoted():
    """A bare `data` would be the run's data, not the mesh axis."""
    [run] = runs(RUN + 'layout.batch = ["data", "expert"]')
    assert 'layout(batch=["data", expert])' in run_to_source(run)
    [again] = compile_runs(run_to_source(run))
    assert again.train == run.train


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
    assert count("init_ablation") == 5


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


def test_editor_quotes_an_option_that_names_a_definition():
    source = (
        ARCH + TRAIN + "e = moe(num_experts=4, top_k=2, expert=ffn, aux_loss_scope="
    )
    items = {c.label: c for c in features.complete(source, len(source))}
    assert items["batch"].insert_text == '"batch"'  # a bare batch is the batch size
    assert items["sequence"].insert_text is None


@pytest.mark.parametrize("path", EXPERIMENTS[::2], ids=lambda p: p.stem)
def test_features_never_fail_on_runs_while_typing(path):
    text, name = path.read_text(), str(path)
    for k in range(0, len(text), 7):
        features.diagnostics(text[:k], name)
        features.complete(text, k, name)
        features.hover(text, k, name)
