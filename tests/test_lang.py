from pathlib import Path

import pytest

from bux import config as cfg
from bux.lang import Architecture, BuxError, check, compile, ir
from bux.lang.printer import to_source
from bux.lang.render import render

EXAMPLES = sorted(
    Path(__file__).parent.parent.joinpath("examples", "architectures").rglob("*.bux")
)

PRELUDE = """
d_model = 16
vocab_size = 32
norm = rmsnorm
attn = gqa(num_heads=2, num_kv_heads=2)
ffn = gated_mlp(intermediate_size=32, activation=swish)
"""


def build(body: str, entry: str = "model") -> Architecture:
    return compile(PRELUDE + body, entry=entry)


def shape(node: ir.Node) -> str:
    """A compact rendering of the tree, for asserting on structure. Labeled blocks show their label."""
    if node.label is not None and not isinstance(node, ir.Layer):
        return node.label
    match node:
        case ir.Layer():
            name = node.label or node.config.type
            return name + "".join(f"{{{p.name}:{p.mode}:{p.tap}}}" for p in node.ports)
        case ir.Seq(items=items):
            return "(" + " ".join(map(shape, items)) + ")"
        case ir.Residual(body=body):
            return f"[{shape(body)}]"
        case ir.Sum(terms=terms):
            return "sum(" + ", ".join(map(shape, terms)) + ")"
        case ir.Prod(factors=factors):
            return "prod(" + ", ".join(map(shape, factors)) + ")"
        case ir.Neg(body=body):
            return f"-{shape(body)}"
        case ir.Const(value=value):
            return repr(value)
        case ir.Identity():
            return "_"
        case ir.Repeat(body=body, times=times):
            return f"{shape(body)}^{times}"
        case ir.Tap(name=name):
            return f"@{name}"
        case ir.Ref(name=name, pop=pop):
            return f"${name}" + ("!" if pop else "")
    raise AssertionError(node)


def unlabeled(node: ir.Node) -> str:
    return shape(node.model_copy(update={"label": None}))


def graph(expr: str) -> str:
    return unlabeled(build(f"b = {expr}", entry="b").graph)


def error(body: str, entry: str = "model") -> BuxError:
    with pytest.raises(BuxError) as info:
        build(body, entry=entry)
    return info.value


# --- Syntax and evaluation ------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("[norm attn] [norm ffn]", "([(norm attn)] [(norm ffn)])"),
        ("norm -> attn → ffn", "(norm attn ffn)"),
        ("[norm (attn + ffn)]", "[(norm sum(attn, ffn))]"),
        ("norm attn + ffn", "sum((norm attn), ffn)"),
        (
            "linear(features=16) swish * linear(features=16)",
            "prod((linear act), linear)",
        ),
        ("_ * 2.0 + attn", "sum(prod(_, 2.0), attn)"),
        ("attn - _", "sum(attn, -_)"),
        ("-attn + ffn", "sum(-attn, ffn)"),
        ("[norm attn]^3", "[(norm attn)]^3"),
        ("(norm attn)^2 ffn", "((norm attn)^2 ffn)"),
        ("@x [attn] [$x scale]", "(@x [attn] [($x scale)])"),
        ("$x", None),
    ],
)
def test_precedence(expr, expected):
    if expected is None:
        error(f"b = {expr}", entry="b")
    else:
        assert graph(expr) == expected


def test_call_needs_adjacent_paren():
    assert graph("norm (attn + ffn)") == "(norm sum(attn, ffn))"
    assert (
        graph("norm attn(sliding_window_size=4)")
        == "(norm attn(sliding_window_size=4))"
    )


def test_continuation_lines_comments_and_semicolons():
    a = compile(
        """
d_model = 16; vocab_size = 32   # two statements
model = embed
    # a comment line inside a statement
    rmsnorm
        head
"""
    )
    assert unlabeled(a.graph) == "(embed rmsnorm head)"


def test_arithmetic_and_constants():
    a = build(
        "h = 4 * d_model // 3 + 2^2\nmodel = embed [linear(features=h) linear(features=d_model)] head"
    )
    widths = [layer.width for layer in a.layers()]
    assert widths == ["tokens", 16, 25, 16]
    [err] = check("d_model = 10.0 ^ 400\nmodel = rmsnorm").errors
    assert err.message == "this number is too large"


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("d_model = 0\nmodel = rmsnorm", "'d_model' must be greater than 0, found 0"),
        (
            "d_model = 16\nvocab_size = -1\nmodel = embed head",
            "'vocab_size' must be greater than 0, found -1",
        ),
        (
            "d_model = 16\nmodel = gqa(num_heads=true, num_kv_heads=1)",
            "'num_heads' takes a number, found a boolean",
        ),
        (
            "d_model = 16\nx = rope(base=false)\nmodel = rmsnorm",
            "'base' takes a number",
        ),
    ],
)
def test_counts_and_numbers_are_not_zero_or_booleans(source, message):
    assert message in str(check(source).errors[0])


def test_overrides_including_dotted_keys():
    a = build("model = embed [attn(rope.base=1e6, sliding_window_size=8)] [attn] head")
    first, second = [
        layer.config for layer in a.layers() if isinstance(layer.config, cfg.GQAConfig)
    ]
    assert first.rope == cfg.RoPEConfig(base=1e6) and first.sliding_window_size == 8
    assert second.rope is None and second.sliding_window_size is None


def test_a_definition_may_shadow_the_builtin_it_calls():
    a = build(
        "mlp = mlp(intermediate_size=8, activation=gelu)\nmodel = embed [mlp] head"
    )
    assert any(
        layer.config
        == cfg.MLPConfig(type="mlp", intermediate_size=8, activation="gelu")
        for layer in a.layers()
    )


def test_definitions_are_order_independent_and_cycles_are_errors():
    assert (
        unlabeled(build("model = embed blk head\nblk = [attn]").graph)
        == "(embed blk head)"
    )
    assert "cycle" in error("a = [b]\nb = [a]\nmodel = embed a head").message


@pytest.mark.parametrize("heads_first", [True, False])
def test_a_replaced_name_that_a_definition_uses_back_is_a_cycle(heads_first):
    """Only in its own definition does `heads` mean the one it replaces: `attn`
    means the new one, whichever is evaluated first."""
    lines = ["heads = 2", "attn = gqa(num_heads=heads, num_kv_heads=heads)"]
    lib = "\n".join(lines if heads_first else lines[::-1])
    lib += "\nd_model = 16\nvocab_size = 32\nmodel = embed [attn] head"
    source = "use lib\nheads = attn.num_heads * 2"
    with pytest.raises(BuxError, match="definition cycle"):
        compile(source, modules={"lib": lib})
    errors = check(source, modules={"lib": lib}).errors
    assert errors and all("definition cycle" in e.message for e in errors)


def test_bare_names_in_settings_are_strings_unless_defined():
    a = build(
        "e = moe(num_experts=4, top_k=2, router=sigmoid, expert=ffn, load_balancing=aux_free)\nmodel = embed [e] head"
    )
    moe = next(
        layer.config for layer in a.layers() if isinstance(layer.config, cfg.MoEConfig)
    )
    assert moe.router == "sigmoid" and moe.load_balancing == "aux_free"
    assert moe.expert == cfg.MLPConfig(
        type="gated_mlp", intermediate_size=32, activation="swish"
    )


def test_patterns_expand_one_letter_definitions():
    a = build('A = [attn]\nF = [ffn]\nmodel = embed "AAF A"^2 head')
    assert unlabeled(a.graph) == "(embed (A A F A)^2 head)"
    assert "pattern letter 'Z'" in error('A = [attn]\nmodel = embed "AZ" head').message
    # A pattern reached through a name points at the name, not past it.
    err = error('A = [attn]\np = "AZA"\nmodel = embed p head')
    assert err.span is not None and err.source is not None
    assert err.source[err.span.start : err.span.end] == "p"


def test_macros():
    a = build(
        "layer(mixer, mlp=ffn) = [norm mixer] [norm mlp]\n"
        "model = embed layer(attn) layer(mlp=attn, mixer=ffn) head"
    )
    assert unlabeled(a.graph) == "(embed layer(attn) layer(mlp=attn, mixer=ffn) head)"
    assert isinstance(a.graph, ir.Seq)
    first, second = a.graph.items[1:3]
    assert unlabeled(first) == "([(norm attn)] [(norm ffn)])"
    assert unlabeled(second) == "([(norm ffn)] [(norm attn)])"
    assert "missing" in error("layer(m) = [m]\nmodel = embed layer() head").message
    assert "is a macro" in error("layer(m) = [m]\nmodel = embed layer head").message


def test_a_macro_that_recursed_too_deeply_leaves_the_depth_as_it_was():
    # m63(x) calls 64 macros deep, which is the limit, after r's error.
    chain = "".join(f"m{i}(x) = m{i - 1}(x)\n" for i in range(1, 64))
    source = PRELUDE + "model = embed head\nr(x) = r(x)\nbad = r(1)\nm0(x) = x\n"
    source += chain + "ok = m63(1)"
    assert [e.message for e in check(source).errors] == ["'r' recurses too deeply"]


def test_shared_definitions():
    a = build("shared blk = [norm attn]\nmodel = embed blk^4 [norm attn] head")
    assert isinstance(a.graph, ir.Seq)
    blk = a.graph.items[1]
    assert isinstance(blk, ir.Repeat) and isinstance(blk.body, ir.Residual)
    assert isinstance(blk.body.body, ir.Seq)
    shares = [layer.share for layer in blk.body.body.items]
    assert shares == [None, None] and blk.body.share == "blk"
    assert (
        "shared"
        in error(
            "shared a = attn\nmodel = embed [a(sliding_window_size=2)] head"
        ).message
    )


def test_macro_defaults_see_earlier_parameters():
    a = build(
        "blk(n, kv=n // 2) = [gqa(num_heads=n, num_kv_heads=kv)]\n"
        "model = embed blk(4) blk(4, kv=4) head"
    )
    heads = [
        (layer.config.num_heads, layer.config.num_kv_heads)
        for layer in a.layers()
        if isinstance(layer.config, cfg.GQAConfig)
    ]
    assert heads == [(4, 2), (4, 4)]
    err = error(
        "blk(a=b, b=2) = [gqa(num_heads=a, num_kv_heads=b)]\nmodel = embed blk() head"
    )
    assert err.message == "unknown name 'b'" and err.hint != "did you mean 'b'?"


def test_field_access_reads_settings():
    a = build(
        "e = moe(num_experts=4, top_k=2, expert=ffn)\n"
        "wide = gqa(num_heads=attn.num_heads * 2, num_kv_heads=attn.num_kv_heads)\n"
        "model = embed [wide] [e.expert] [scale(init=rope.base)] head"
    )
    configs = [layer.config for layer in a.layers()][1:-1]
    assert configs[0] == cfg.GQAConfig(num_heads=4, num_kv_heads=2)
    assert configs[1] == cfg.MLPConfig(
        type="gated_mlp", intermediate_size=32, activation="swish"
    )
    assert configs[2] == cfg.ScaleConfig(init=10000.0)
    err = error("x = attn.num_head\nmodel = embed head")
    assert err.message == "gqa has no setting 'num_head'"
    assert err.hint == "did you mean 'num_heads'?"
    # Each key and value head serves a group of the query heads.
    err = error("x = gqa(num_heads=32, num_kv_heads=12)\nmodel = embed [x] head")
    assert "12 key and value heads do not divide 32 query heads" in err.message
    assert err.span is not None and err.source is not None
    assert err.source[err.span.start : err.span.end] == "num_kv_heads=12"
    assert (
        "only layers and settings"
        in error("b = [attn]\nx = b.a\nmodel = embed head").message
    )
    assert (
        error("x = atn.num_heads\nmodel = embed head").message == "unknown name 'atn'"
    )


def test_unknown_names_are_reported_as_names_with_suggestions():
    err = error(
        "f = mlp(intermediate_size=4 * dmodel, activation=gelu)\nmodel = embed [f] head"
    )
    assert (err.message, err.hint) == (
        "unknown name 'dmodel'",
        "did you mean 'd_model'?",
    )
    assert err.source is not None and err.span is not None
    assert err.source[err.span.start : err.span.end] == "dmodel"
    err = error(
        "hidden = 64\nf = mlp(intermediate_size=hiden, activation=gelu)\nmodel = embed [f] head"
    )
    assert (err.message, err.hint) == ("unknown name 'hiden'", "did you mean 'hidden'?")
    assert error("model = embed [attn]^n head").message == "unknown name 'n'"
    err = error(
        "f = mlp(intermediate_size=8, activation=swihs)\nmodel = embed [f] head"
    )
    assert err.hint == "did you mean 'swish'?"
    # A bare name is still a string where a string is wanted.
    build(
        "e = moe(num_experts=4, top_k=2, expert=ffn, router=sigmoid)\nmodel = embed [e] head"
    )


def test_a_value_a_block_kept_for_itself_cannot_hide_an_earlier_one():
    err = error(
        "mix(f) = @h [f] [$h scale]\nmodel = embed @h mix(attn) [$h! scale] head"
    )
    assert "'mix(attn)' wrote and read for itself" in err.message
    assert "$h!" in (err.hint or "")
    err = error("model = embed @h (@h [attn] [$h scale])^2 [$h! scale] head")
    assert "'one pass of (@h [attn] [$h scale])^2'" in err.message
    # Popping inside the block, exporting, chaining, and reading what is the only
    # value are all fine.
    build("mix(f) = @h [f] [$h! scale]\nmodel = embed @h mix(attn) [$h! scale] head")
    build("export = [attn] @h\nmodel = embed @h export [$h! scale] [$h! scale] head")
    build("chain = [$prev scale] @prev\nmodel = embed @prev chain^3 head")
    build("kv = [attn(kv=@kv)] [attn(kv=$kv)]\nmodel = embed kv [attn(kv=$kv)] head")


def test_check_reports_every_error_and_recovers_from_syntax_errors():
    report = check(
        PRELUDE + "a = gqa(num_heads=2)\nblk = [norm attn\n"
        "b = mlp(intermdiate_size=3, activation=gelu)\nc = [norm atn]\n"
        # A definition that uses a broken one is not checked further.
        "model = embed blk [norm atn2] head\n"
    )
    assert [e.message for e in report.errors] == [
        "expected ']', found end of input",
        "gqa needs 'num_kv_heads'",
        "mlp has no setting 'intermdiate_size'",
        "unknown layer or block 'atn'",
    ]
    assert report.arch is None and "a" not in report.values
    assert report.values["attn"] == cfg.GQAConfig(num_heads=2, num_kv_heads=2)
    # A macro argument that was a bare name in one call is not blamed in the next.
    report = check(
        PRELUDE + "blk(h) = gated_mlp(intermediate_size=h, activation=swish)\n"
        "a = blk(nope)\nc = blk(1.5)\nmodel = embed head\n"
    )
    assert [e.message.split(":")[0] for e in report.errors] == [
        "unknown name 'nope'",
        "gated_mlp",
    ]
    # An unparsable setting is not then reported as missing.
    assert [e.message for e in check("d_model = (16\nmodel = rmsnorm\n").errors] == [
        "expected ')', found end of input"
    ]


def test_integer_factor_is_rejected_with_a_hint():
    err = error("model = embed ([attn] * 3) head")
    assert "ambiguous" in err.message and "^3" in (err.hint or "")
    assert "^3" in (error("model = embed [attn] * 3 head").hint or "")


# --- Lowering: widths, taps, ports ------------------------------------------------


def test_widths_are_inferred_and_checked():
    assert "keep the width" in error("model = embed [linear(features=8)] head").message
    assert (
        "disagree on width"
        in error("model = embed (linear(features=8) + _) head").message
    )
    assert "token ids" in error("model = rmsnorm embed head").message
    assert "follow the LM head" in error("model = embed head rmsnorm").message


def test_tied_head_needs_one_embedding():
    assert (
        "exactly one embedding"
        in error("b = rmsnorm head(tied=true)", entry="b").message
    )


def test_unet_taps_balance():
    a = build("model = embed (attn @skip)^3 ([$skip! scale] ffn)^3 head")
    assert sum(1 for layer in a.layers() if layer.config.type == "gqa") == 3
    assert (
        "empty" in error("model = embed (attn @skip)^2 ([$skip!] ffn)^3 head").message
    )
    assert "before anything writes" in error("model = embed [$nope] head").message


def test_ports():
    a = build("model = embed [norm attn(v=@v1)] [norm attn(v=$v1)] head")
    ports = [layer.ports for layer in a.layers() if layer.ports]
    assert ports == [
        (ir.Port(name="v", tap="v1", mode="write"),),
        (ir.Port(name="v", tap="v1", mode="read"),),
    ]
    assert "no port 'q'" in error("model = embed [attn(q=@x)] head").message
    assert "port of a layer" in error("model = embed [attn(v=@v)] [$v] head").message
    assert "holds hidden states" in error("model = embed @x [attn(v=$x)] head").message


def test_shared_blocks_must_agree_on_width():
    err = error(
        "shared n = norm\nmodel = embed [n] [linear(features=8) n linear(features=16)] head"
    )
    assert "different widths" in err.message
    err = error("shared a = attn\nmodel = embed [a(v=@v)] [a(v=$v)] head")
    assert "every use of it takes the same ports" in err.message


@pytest.mark.parametrize(
    "residual", ["", "residual = hc(expansion=2)\n", "residual = vpgate\n"]
)
def test_a_shared_block_with_branches_stays_outside_branches_or_inside(residual):
    # A branch outside any other is the residual algebra's, inside one x + f(x).
    err = error(
        residual + "shared blk = [norm attn]\nmodel = embed blk [blk] norm head"
    )
    assert "used both outside residual branches and inside one" in err.message
    build(residual + "shared blk = [norm attn]\nmodel = embed blk blk norm head")
    build(residual + "shared n = norm\nmodel = embed [n attn] [[n attn]] norm head")


def test_a_shared_layer_is_shared_whether_or_not_a_merge_goes_before_it():
    a = build(
        "residual = gr(expansion=2, rank=4)\nshared out = head\n"
        "mtp = $h nextn [attn] out\nmodel = embed [attn] @h merge out mtp"
    )
    heads = [l.width for l in a.layers() if l.config.type == "head"]
    assert heads == [16, 16]
    build("residual = hc(expansion=2)\nshared n = norm\nmodel = embed [n attn] n head")


def test_heads_are_outputs_and_mtp_reads_on_from_a_tap():
    a = build(
        "shared out = head\n"
        "mtp = $h! nextn [norm attn] @h norm out\n"
        "model = embed [norm attn] norm @h out mtp^2"
    )
    kinds = [layer.config.type for layer in a.layers()]
    assert kinds.count("head") == 3 and kinds.count("nextn") == 2
    err = error("model = embed head norm")
    assert "follow the LM head" in err.message and "$h" in (err.hint or "")
    assert "exactly one embedding" in error("b = nextn", entry="b").message


def test_residual_algebras_and_their_errors():
    for setting in (
        "mhc(expansion=4)",
        "gr(rank=4)",
        "attnres(block_size=2)",
        "vpgate",
    ):
        a = build(f"residual = {setting}\nmodel = embed [norm attn] [norm ffn] head")
        assert a.residual is not None and a.residual.type == setting.split("(")[0]
        merges = [layer for layer in a.layers() if layer.config.type == "merge"]
        assert len(merges) == (0 if setting == "vpgate" else 1)
    err = error("residual = rope()\nmodel = embed head")
    assert "hc(...)" in err.message and "vpgate(...)" in err.message


def test_init_starts_the_model_by_role_and_its_errors():
    a = build("init = normal(std=0.5 * d_model ^ -0.5, truncate=3)\nmodel = embed head")
    assert a.init == cfg.NormalInitConfig(std=0.125, truncate=3)
    assert "init = normal(std=0.125, truncate=3.0)" in to_source(a)
    plain = build("model = embed head")  # normal(std=0.02), but layers' own starts
    assert plain.init is None and "init" not in to_source(plain)
    roles = build(
        "init = fan_in(scale=2, out=normal(std=0.02 / 4 ^ 0.5), head=zeros)\n"
        "model = embed head"
    )
    assert roles.init == cfg.FanInitConfig(
        type="fan_in",
        scale=2,
        out=cfg.NormalInitConfig(std=0.01),
        head=cfg.ZerosInitConfig(),
    )
    assert compile(to_source(roles)).init == roles.init
    err = error("init = 0.02\nmodel = embed head")
    assert "'init' must be normal(...) or fan_in(...)" in err.message
    err = error("init = normal(out=normal(head=zeros))\nmodel = embed head")
    assert "the init of 'out' sets roles of its own" in err.message


MEMORY = "mem = engram(head_dim=4, table_size=50)\n"


def test_engram_is_bare_and_adds_to_every_stream():
    body = MEMORY + "model = embed [norm attn] mem [norm ffn] norm head"
    widened = build("residual = mhc(expansion=4)\n" + body)
    widths = {layer.config.type: layer.width for layer in widened.layers()}
    assert widths["engram"] == "streams" and widths["rmsnorm"] == 16
    assert unlabeled(widened.graph).endswith("mem [(norm ffn)] merge norm head)")
    sources = build("residual = attnres\n" + MEMORY + "model = embed mem norm head")
    assert unlabeled(sources.graph) == "(embed merge mem norm head)"
    plain = build(body)
    assert [layer.width for layer in plain.layers() if layer.label == "mem"] == [16]
    assert "merge" not in unlabeled(plain.graph)


def test_widened_streams_are_read_back_once():
    mhc = "residual = mhc(expansion=4)\n"
    assert "twice" in error(MEMORY + "model = embed [mem] head").message
    err = error(mhc + "model = embed [attn] norm [ffn] head")
    assert "read back into one" in err.message and "engram and nextn" in (
        err.hint or ""
    )
    # attnres keeps the outputs so far, not copies: engram and nextn need one stream.
    err = error("residual = attnres\nmodel = embed [attn] norm [ffn] head")
    assert "read back into one" in err.message and "engram" not in (err.hint or "")
    assert "there is none here" in error("model = embed merge head").message
    err = error(mhc + "model = embed @x0 [norm attn] [$x0 scale] head")
    assert "only exists outside residual branches" in err.message


def test_mtp_continues_from_the_widened_stream():
    a = build(
        "residual = gr(rank=4)\nshared out = head\nmtp = $h nextn [attn] out\n"
        "model = embed [attn] @h out mtp"
    )
    kinds = [(layer.config.type, layer.width) for layer in a.layers()]
    assert ("nextn", "streams") in kinds and kinds.count(("merge", "streams")) == 2


def test_attention_shares_indexer_selections_through_a_port():
    a = build(
        "a = mla(num_heads=2, indexer=indexer(num_heads=2, head_dim=4, top_k=8))\n"
        "model = embed [norm a(index=@k)] [norm a(index=$k)] head"
    )
    assert [p.mode for layer in a.layers() for p in layer.ports] == ["write", "read"]
    assert (
        "ports: index" in error("model = embed [mla(num_heads=2, kv=@x)] head").message
    )


# --- Errors -------------------------------------------------------------------------


def test_errors_point_at_the_source():
    err = error("model = embed [norm atn] head")
    text = str(err)
    assert "unknown layer or block 'atn'" in text and "did you mean 'attn'?" in text
    assert "model = embed [norm atn] head" in text and "^^^" in text


def test_an_unknown_layer_called_with_settings_says_how_to_add_one():
    err = error("model = embed [norm diffattn(num_heads=12)] head")
    assert err.hint is not None and err.hint.startswith("did you mean 'attn'?")
    assert "LayerConfig" in err.hint and "register('diffattn')" in err.hint


def test_unknown_setting_suggests_the_right_one():
    err = error("model = embed [gqa(num_heads=2, num_kv_head=2)] head")
    assert (
        err.message == "gqa has no setting 'num_kv_head'"
        and err.hint == "did you mean 'num_kv_heads'?"
    )


# --- Examples, printer, renderer ----------------------------------------------------


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_examples_round_trip(path):
    arch = compile(path.read_text(), filename=str(path))
    assert check(path.read_text()).arch == arch
    source = to_source(arch)
    again = compile(source)
    assert ir.strip(again.graph) == ir.strip(arch.graph)
    assert again.model_copy(update={"graph": None}) == arch.model_copy(
        update={"graph": None}
    )
    assert to_source(again) == source
    assert Architecture.model_validate_json(arch.model_dump_json()) == arch
    assert render(arch)


def test_render_draws_residual_branches_under_the_stream():
    arch = build(
        "block = [norm attn] [norm (attn + ffn)]\nmodel = embed block^2 norm head"
    )
    assert render(arch) == (
        "model\n"
        "──embed──┤block├×2──norm──head──\n"
        "\n"
        "block\n"
        "──┬────────────⊕──┬────────────────⊕──\n"
        "  └─norm──attn─┘  └─norm──┬─attn─⊕─┘\n"
        "                          └─ffn──┘\n"
    )


def test_printer_keeps_negation_inside_sequences_unary():
    arch = build("model = embed (-attn) [ffn] head")
    assert "embed (-attn) [ffn] head" in to_source(arch)


def test_a_macro_call_in_a_sequence_prints_inline_and_round_trips():
    arch = build("layer(f) = [norm f] [norm ffn]\nmodel = embed layer(attn) norm head")
    source = to_source(arch)
    assert "model = embed [norm attn] [norm ffn] norm head" in source
    assert ir.strip(compile(source).graph) == ir.strip(arch.graph)


FUZZ_PIECES = [
    "norm",
    "attn",
    "ffn",
    "[",
    "]",
    "(",
    ")",
    "+",
    "-",
    "*",
    "^",
    "2",
    "3",
    "0.5",
    "-0.5",
    "@x",
    "$x",
    "$x!",
    "_",
    '"AB"',
    "->",
    "scale",
    "A",
    "B",
    "s",
    "linear(features=8)",
    "attn(v=@v)",
    "attn(v=$v)",
    "f(attn)",
    "f(B)",
    "attn(sliding_window_size=4)",
]


def test_random_programs_fail_cleanly_and_round_trip():
    import random

    rng = random.Random(0)
    prelude = (
        PRELUDE + "A = [attn]\nB = [ffn]\nf(x) = [norm x]\nshared s = [norm attn]\n"
    )
    compiled = 0
    for _ in range(3000):
        body = " ".join(rng.choices(FUZZ_PIECES, k=rng.randint(1, 10)))
        program = f"{prelude}model = embed {body} head\n"
        report = check(program)
        try:
            arch = compile(program)
        except BuxError:
            assert report.errors, body
            continue
        assert not report.errors and report.arch == arch, body
        compiled += 1
        source = to_source(arch)
        assert ir.strip(compile(source).graph) == ir.strip(arch.graph), body
        assert render(arch) and render(arch, expand=True)
    assert compiled > 100
