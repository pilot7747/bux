"""Exports of the graph's other constructs compute what bux does (bux.export): the
residual algebras and their widened streams, multi-token prediction (nextn), engram,
and the graph's edge cases, alone and in the zoo's models that use them, shrunk."""

import numpy as np
import pytest
from flax import nnx
from test_examples import INDEXER, MLA, ROOT, SHRUNK, SMALL, YARN

from bux.architecture import Model
from bux.architecture.layers.engram import Engram
from bux.architecture.layers.mlp import MLP
from bux.export import check
from bux.export.huggingface import cache_layers
from bux.export.plan import engram_hashes, plan
from bux.export.weights import module_at
from bux.lang import compile, ir

HEAD = "d_model = 32\nvocab_size = 256\n"
ATTN = "attn = gqa(num_heads=4, num_kv_heads=2, rope=rope(fraction=0.5))\n"
FFN = "ffn = gated_mlp(intermediate_size=64, activation=swish)\n"
BLOCK = HEAD + ATTN + FFN + "block = [rmsnorm attn] [rmsnorm ffn]\n"
PLAIN = "model = embed block^2 rmsnorm head"


def under(residual: str, model: str = PLAIN) -> str:
    """Return the source of `model` under the residual algebra `residual`."""
    return BLOCK + f"residual = {residual}\n" + model


ALGEBRAS = {
    "hc": under("hc(expansion=2)"),
    "hc static": under("hc(expansion=3, dynamic=false)"),
    "hc single pass": under("hc(expansion=2, single_pass=true, norm_eps=1e-3)"),
    "mhc": under("mhc(expansion=2)"),
    "mhc static": under("mhc(expansion=4, dynamic=false, sinkhorn_iters=3)"),
    "mhc single pass": under("mhc(expansion=4, single_pass=true, norm_eps=1e-20)"),
    "gr": under("gr(expansion=2, rank=8)"),
    "gr eps": under("gr(expansion=3, rank=4, norm_eps=1e-3)"),
    "attnres": under("attnres()"),
    "attnres blocks": under("attnres(block_size=2)", "model = embed block^3 head"),
    "attnres block under way": under(
        "attnres(block_size=4, norm_eps=1e-5)", "model = embed block^3 rmsnorm head"
    ),
    "msar": under("msar(kernel_sizes=[2, 3])"),
    "msar blocks": under("msar(block_size=2)", "model = embed block^3 head"),
    "msar block under way": under(
        "msar(block_size=3, norm_eps=1e-5)", "model = embed block^3 rmsnorm head"
    ),
    "vpgate": under("vpgate(temperature=0.5)"),
    "vpgate clamped": under("vpgate(eps=0.6)"),
    "taps of the streams": under(
        "mhc(expansion=2, single_pass=true)",
        "model = embed @x block $x block rmsnorm head",
    ),
    "branches inside a branch": under(
        "hc(expansion=2)", "model = embed [rmsnorm [attn] ffn] block rmsnorm head"
    ),
    "a repeat inside a branch": under(
        "mhc(expansion=2)",
        "inner = [rmsnorm ffn]\nmodel = embed [inner^2] block rmsnorm head",
    ),
    "a shared block": under(
        "gr(expansion=2, rank=8)",
        "shared twice = block\nmodel = embed twice^2 block rmsnorm head",
    ),
    "merge written": under("hc(expansion=2)", "model = embed block merge head"),
}

MTP = "shared out = head\n"
NEXTN = {
    "deepseek": BLOCK
    + MTP
    + "mtp = $h nextn block rmsnorm out\nmodel = embed block^2 @h rmsnorm out mtp",
    "shared, twice": BLOCK
    + MTP
    + "shared mtp = $h! nextn(norm=rmsnorm(eps=1e-5)) block rmsnorm @h out\n"
    "model = embed block^2 rmsnorm @h out mtp^2",
    "fresh, three times": BLOCK
    + MTP
    + "mtp = $h nextn(norm=layernorm) block rmsnorm out\n"
    "model = embed block rmsnorm @h out mtp^3",
    "tied, scaled": BLOCK + "mtp = $h nextn block rmsnorm head(tied=true)\n"
    "model = embed(scale_by_sqrt_dim=true) block @h rmsnorm head(tied=true) mtp",
    "on the streams": under(
        "gr(expansion=2, rank=8)",
        MTP + "mtp = $h nextn block out\nmodel = embed block^2 @h out mtp",
    ),
    "widening one stream": under(
        "mhc(expansion=2)",
        MTP + "model = embed block merge rmsnorm @h out $h nextn block rmsnorm out",
    ),
    "under msar": under(
        "msar(block_size=2)",
        MTP + "mtp = $h nextn block out\nmodel = embed block^2 @h out mtp",
    ),
    "single pass": under(
        "mhc(expansion=2, single_pass=true)",
        MTP + "mtp = $h nextn block out\nmodel = embed block^2 @h out mtp",
    ),
    "in the head's definition": BLOCK
    + "tail = rmsnorm head $h nextn block rmsnorm head\nmodel = embed block^2 @h tail",
    "after a recurrent trunk": BLOCK + MTP + "mix = gdn(num_heads=2, head_dim=8)\n"
    "model = embed [rmsnorm mix]^2 @h rmsnorm out $h nextn block rmsnorm out",
}


def engram(
    settings: str = "", model: str = "embed block mem block rmsnorm head"
) -> str:
    """Return a source with an engram layer `mem` of `settings`."""
    return BLOCK + f"mem = engram(head_dim=4, table_size=50{settings})\nmodel = {model}"


ENGRAM = {
    "engram": engram(),
    "no conv": engram(", conv_size=none"),
    "4-grams, a conv of 2": engram(", max_ngram_size=4, num_heads=2, conv_size=2"),
    "bigrams": engram(", max_ngram_size=2, num_heads=3, conv_size=3"),
    "classes": engram(", vocab_classes=40, norm=rmsnorm(zero_centered=true)"),
    "two engrams": engram(
        ", num_heads=2", "embed mem block engram(head_dim=8, table_size=50) head"
    ),
    "shared": BLOCK + "shared mem = engram(head_dim=4, table_size=60)\n"
    "model = embed mem block mem rmsnorm head",
    "on the streams": under(
        "mhc(expansion=2)",
        "mem = engram(head_dim=4, table_size=50)\n"
        "model = embed block mem block rmsnorm head",
    ),
    "under gr": under(
        "gr(expansion=4, rank=4)",
        "mem = engram(head_dim=4, table_size=50, conv_size=none)\n"
        "model = embed block mem block head",
    ),
    "no attention": engram(model="embed mem [rmsnorm ffn] mem rmsnorm head"),
}

GRAPH = {
    "shared repeat": BLOCK
    + "shared stack = block^2\nmodel = embed stack stack rmsnorm head",
    "shared taps": BLOCK
    + "shared skip = @s block ($s! + _)\nmodel = embed skip skip rmsnorm head",
    "constant gates": BLOCK
    + "model = embed [rmsnorm (attn + ffn * 0.5)] [rmsnorm (0.612 * attn)] "
    "(rmsnorm * 0.25) head",
    "gated product": BLOCK + "gate = linear(features=32) sigmoid\n"
    "model = embed [rmsnorm (ffn * gate)] block rmsnorm head",
    "negation": BLOCK
    + "unit = rmsnorm(affine=false)\nstep = scale(init=0.05, per_channel=true)\n"
    "model = embed unit [(attn unit - _) step] unit [(ffn unit - _) step] unit head",
}


@pytest.mark.parametrize("source", ALGEBRAS.values(), ids=ALGEBRAS.keys())
def test_residual_algebras_export(source: str) -> None:
    check.compare(compile(source)).check()


@pytest.mark.parametrize("source", NEXTN.values(), ids=NEXTN.keys())
def test_every_head_of_multi_token_prediction_exports(source: str) -> None:
    check.compare(compile(source)).check()


@pytest.mark.parametrize("source", ENGRAM.values(), ids=ENGRAM.keys())
def test_engram_exports(source: str) -> None:
    check.compare(compile(source), length=24, steps=8).check()


def test_engram_stops_at_the_boundary() -> None:
    """With few tokens, the boundary token comes up in the prompt and as it decodes."""
    source = engram(", boundary_token=3, max_ngram_size=4", "embed mem block head")
    source = source.replace("vocab_size = 256", "vocab_size = 8")
    check.compare(compile(source), length=16, steps=8).check()


@pytest.mark.parametrize("source", GRAPH.values(), ids=GRAPH.keys())
def test_graph_edge_cases_export(source: str) -> None:
    check.compare(compile(source)).check()


#: Zoo models that use these constructs, shrunk as tests/test_examples.py does: its
#: SHRUNK lines, or these. Other agents' layers that do not export yet (MoE, the
#: recurrent mixers) are swapped (`SWAPS`).
GQA = "gqa(num_heads=4, num_kv_heads=2, head_dim=8, rope=rope(fraction=0.5))"
ZOO = {
    "mhc.bux": [],
    "deepseek/deepseek_v4.1_flash.bux": [],
    "qwen/qwen3.8_flash_next.bux": [],
    "kimi/kimi_k3.bux": [],
    "muse/muse_spark_open_816b_a42b.bux": [],
    "glm/glm_5.3.bux": [],
    "deepseek/deepseek_v3.2.bux": [f"attn = {MLA}{YARN}{INDEXER})"],
    "nemotron/nemotron_3_super_120b_a12b.bux": [
        "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8)",
        "experts = mlp(intermediate_size=8, activation=relu2)",
    ],
    "mimo/mimo_v2.6_pro.bux": [
        (
            "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=12, v_head_dim=8,"
            " fused_qkv=true, rope=rope(base=10000000, fraction=0.334))"
        ),
        "local = attn(sliding_window_size=4, sinks=true, rope.base=10000)",
    ],
    "cohere/command_a_plus_05_2026.bux": [
        (
            "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8,"
            " rope=rope(layout=interleaved))"
        ),
        "local = attn(sliding_window_size=4)",
        "experts = dense * 0.5",
    ],
    "ngpt.bux": [
        (
            "attn = gqa(num_heads=4, num_kv_heads=4, qk_norm=rmsnorm,"
            " rope=rope(layout=interleaved))"
        ),
        "ffn = dense",
    ],
    "modded_nanogpt.bux": [
        (
            "attn = gqa(num_heads=2, num_kv_heads=2, head_dim=8, fused_qkv=true,"
            " qk_norm=norm, scale=0.12, rope=rope(base=1024, fraction=0.5))"
        ),
        "ffn = mlp(intermediate_size=8, activation=relu2)",
    ],
}
#: What the layers no export implements yet become, by the name they are defined as.
SWAPS = {"experts": "dense", "delta": GQA, "kda": GQA, "ssm": GQA}


def small(name: str) -> ir.Architecture:
    """Return the zoo model `name`, shrunk, its layers swapped as `SWAPS` says: of
    the lines that define a name, the last."""
    source = (ROOT / name).read_text()
    swaps = [f"{n} = {layer}" for n, layer in SWAPS.items() if f"\n{n} " in source]
    lines = [*SMALL.strip().splitlines(), *SHRUNK.get(name, []), *swaps, *ZOO[name]]
    overrides = {line.split("=", 1)[0].strip(): line for line in lines}
    return compile(source, overrides=list(overrides.values()))


@pytest.mark.parametrize("name", ZOO)
def test_zoo_models_export(name: str) -> None:
    check.compare(small(name), length=16, steps=6).check()


def test_generating_skips_multi_token_prediction() -> None:
    """The cache keeps the layers up to the first head's alone: the multi-token
    prediction after it runs only for every head's logits, without a cache."""
    p = plan(compile(NEXTN["shared, twice"]))
    assert p.stop() is not None
    attention = [c for c in p.calls if c.layer.config.type == "gqa"]
    assert len(attention) == 4
    assert [kind for kind, _ in cache_layers(p)] == ["full_attention"] * 2
    # A head in a definition stops in it: what comes after it there is skipped too.
    p = plan(compile(NEXTN["in the head's definition"]))
    assert p.stop() == (3, 1)
    assert [kind for kind, _ in cache_layers(p)] == ["full_attention"] * 2


def test_algebra_branches_hold_their_parameters() -> None:
    """Each branch of the algebra outside any other is a module named by it, and its
    body's layers still reach bux's modules through it."""
    arch = compile(ALGEBRAS["a repeat inside a branch"])
    p = plan(arch)
    assert [b.path for b in p.branches] == ["model.mhc", "model.mhc_1", "model.mhc_2"]
    model = Model(arch, rngs=nnx.Rngs(0))
    ffn = next(c for c in p.calls if c.path == "model.layers.1.ffn")
    assert isinstance(module_at(arch.graph, model.graph, ffn.trail), MLP)


def test_engram_hashes_as_bux_builds_it() -> None:
    """The plan's primes and multipliers are those of bux's modules."""
    arch = compile(ENGRAM["two engrams"])
    p = plan(arch)
    model = Model(arch, rngs=nnx.Rngs(0))
    calls = [c for c in p.calls if c.layer.config.type == "engram"]
    for call in calls:
        bux = module_at(arch.graph, model.graph, call.trail)
        assert isinstance(bux, Engram)
        mine = engram_hashes(p, call)
        assert np.array_equal(np.asarray(bux.primes[...]), mine["primes"])
        assert np.array_equal(np.asarray(bux.multipliers[...]), mine["multipliers"])


def test_nextn_pads_past_the_row_as_bux_does() -> None:
    """A row shorter than the tokens ahead embeds token 0 for all of them."""
    check.compare(compile(NEXTN["fresh, three times"]), length=2, steps=2).check()
