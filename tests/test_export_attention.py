"""Softmax attention exports compute what bux does (bux.export.hf.attention, mla,
sparse): every setting of gqa and mla, their ports, the sparse indexer, and the zoo's
models that use them, shrunk.

Each case decodes 6 tokens after a prompt of 6, so that windows of 3 to 5, convs of
3 taps and an indexer's top 4 all cross what one step sees."""

from pathlib import Path

import pytest

from bux.export import check
from bux.lang import compile

HEAD = "d_model = 64\nvocab_size = 256\n"
FFN = "ffn = gated_mlp(intermediate_size=64, activation=swish)\n"
GQA = "gqa(num_heads=4, num_kv_heads=2, rope=rope()"
MLA = (
    "mla(num_heads=4, q_lora_rank=16, kv_lora_rank=16, qk_nope_head_dim=8,"
    " qk_rope_head_dim=8, v_head_dim=8"
)
INDEXER = "indexer=indexer(num_heads=2, head_dim=8, top_k=4"


def stack(attn: str, depth: int = 2) -> str:
    """A pre-norm model of `depth` blocks of the attention `attn` and an MLP."""
    return (
        HEAD
        + FFN
        + f"attn = {attn}\n"
        + "block = [rmsnorm attn] [rmsnorm ffn]\n"
        + f"model = embed block^{depth} rmsnorm head"
    )


def wired(attn: str, *calls: str) -> str:
    """A model of a block for each attention call of `calls`, as ``attn(kv=@kv)``,
    which read and write ports."""
    blocks = " ".join(f"block({call})" for call in calls)
    return (
        HEAD
        + FFN
        + f"attn = {attn}\n"
        + "block(mixer) = [rmsnorm mixer] [rmsnorm ffn]\n"
        + f"model = embed {blocks} rmsnorm head"
    )


GQA_SETTINGS = {
    "plain": GQA + ")",
    "mha": "gqa(num_heads=4, num_kv_heads=4, rope=rope())",
    "mqa": "gqa(num_heads=4, num_kv_heads=1, rope=rope())",
    "no rope": "gqa(num_heads=4, num_kv_heads=2)",
    "window": GQA + ", sliding_window_size=4)",
    "sinks": GQA + ", sinks=true)",
    "sinks, window": GQA + ", sinks=true, sliding_window_size=3)",
    "softcap": GQA + ", softcap=2.0)",
    "gated": GQA + ", gated=true)",
    "gate per head": GQA
    + ", gated=true, gate_per_head=true, gate_activation=softplus)",
    "output norm": GQA + ", output_norm=rmsnorm(zero_centered=true))",
    "qk norm": GQA + ", qk_norm=layernorm(bias=true))",
    "fused qkv": GQA + ", fused_qkv=true, bias=true)",
    "biases": GQA + ", bias=true, output_bias=false)",
    "scale": GQA + ", scale=0.3)",
    "log scale": GQA + ", log_scale=0.5, log_scale_floor=2)",
    "relative bias": GQA + ", relative_bias=relbias(rank=4, extent=5))",
    "yarn": "gqa(num_heads=4, num_kv_heads=2, rope=rope(scaling=yarn(factor=4,"
    " original_max_len=8)))",
    "yarn on the softmax": "gqa(num_heads=4, num_kv_heads=2, rope=rope(scaling=yarn("
    "factor=4, original_max_len=8, beta_fast=8, attention_factor_on=softmax)))",
    "partial rope": "gqa(num_heads=4, num_kv_heads=2, rope=rope(fraction=0.5,"
    " layout=interleaved, base=500))",
    "head dims": GQA + ", head_dim=12, v_head_dim=8)",
    "kv conv": GQA + ", kv_conv_size=3)",
    "kv conv, window": GQA + ", kv_conv_size=3, sliding_window_size=4)",
    "dropouts": GQA + ", attention_dropout=0.1, output_dropout=0.1)",
    "everything at once": "gqa(num_heads=4, num_kv_heads=2, head_dim=8, v_head_dim=12,"
    " bias=true, sinks=true, softcap=3.0, gated=true, output_norm=rmsnorm,"
    " qk_norm=rmsnorm, log_scale=0.3, log_scale_floor=3, relative_bias=relbias(rank=2,"
    " extent=4), sliding_window_size=5, kv_conv_size=2, rope=rope(fraction=0.5))",
}


@pytest.mark.parametrize("attn", GQA_SETTINGS.values(), ids=GQA_SETTINGS.keys())
def test_gqa_settings(attn: str) -> None:
    check.compare(compile(stack(attn))).check()


WINDOWS = {
    "a window a layer": HEAD
    + FFN
    + "a = gqa(num_heads=4, num_kv_heads=2, rope=rope(), sliding_window_size=3)\n"
    + "b = a(sliding_window_size=5, sinks=true)\n"
    + "c = a(sliding_window_size=none)\n"
    + "model = embed [rmsnorm a] [rmsnorm b] [rmsnorm c] [rmsnorm a] rmsnorm head",
    "a shared layer": HEAD
    + FFN
    + "shared local = gqa(num_heads=4, num_kv_heads=2, rope=rope(), sliding_window_size=3)\n"
    + "model = embed [rmsnorm local] [rmsnorm ffn] [rmsnorm local] rmsnorm head",
}


@pytest.mark.parametrize("source", WINDOWS.values(), ids=WINDOWS.keys())
def test_windows_differ_by_layer(source: str) -> None:
    check.compare(compile(source)).check()


PORTS = {
    "value residual": wired(GQA + ")", "attn(v=@v1)", "attn(v=$v1)", "attn(v=$v1)"),
    "value residual, learned": wired(
        GQA + ", value_residual=learned)", "attn(v=@v1)", "attn(v=$v1)", "attn(v=$v1)"
    ),
    "value residual, convex": wired(
        GQA + ", value_residual=convex, kv_conv_size=2)", "attn(v=@v1)", "attn(v=$v1!)"
    ),
    "kv sharing": wired(
        GQA + ", qk_norm=rmsnorm)",
        "attn(kv=@kv)",
        "attn(kv=$kv)",
        "attn(kv=@kv)",
        "attn(kv=$kv!)",
    ),
    "kv sharing, windows": wired(
        GQA + ", sliding_window_size=4, fused_qkv=true)",
        "attn(kv=@kv)",
        "attn(kv=$kv, sliding_window_size=3, sinks=true)",
        "attn(kv=$kv!)",
    ),
    "kv sharing, a full writer": wired(
        GQA + ")",
        "attn(kv=@kv, kv_conv_size=3)",
        "attn(kv=$kv!, sliding_window_size=3)",
    ),
}


@pytest.mark.parametrize("source", PORTS.values(), ids=PORTS.keys())
def test_gqa_ports(source: str) -> None:
    check.compare(compile(source)).check()


def test_a_reader_of_a_shorter_window_than_its_own_says_so() -> None:
    source = wired(
        GQA + ", sliding_window_size=3)",
        "attn(kv=@kv)",
        "attn(kv=$kv!, sliding_window_size=none)",
    )
    with pytest.raises(NotImplementedError, match="caches 3"):
        check.compare(compile(source))


MLA_SETTINGS = {
    "plain": MLA + ")",
    "full-rank queries": "mla(num_heads=4, kv_lora_rank=16, qk_nope_head_dim=8,"
    " qk_rope_head_dim=4, v_head_dim=12)",
    "no rope": MLA + ", rope=none)",
    "yarn, interleaved": MLA + ", rope=rope(layout=interleaved, scaling=yarn(factor=40,"
    " original_max_len=8, attention_factor_on=softmax)))",
    "partial rope": MLA + ", rope=rope(fraction=0.5))",
    "qk norm, gated": MLA + ", qk_norm=rmsnorm, gated=true, gate_per_head=true)",
    "window, sinks": MLA + ", sliding_window_size=4, sinks=true)",
    "softcap, scale": MLA + ", softcap=2.0, scale=0.2)",
    "relative bias, log scale": MLA
    + ", relative_bias=relbias(rank=2, extent=4), log_scale=0.5, log_scale_floor=2)",
    "output norm": MLA + ", output_norm=rmsnorm)",
}


@pytest.mark.parametrize("attn", MLA_SETTINGS.values(), ids=MLA_SETTINGS.keys())
def test_mla_settings(attn: str) -> None:
    check.compare(compile(stack(attn))).check()


SPARSE = {
    "gqa": GQA + f", {INDEXER}))",
    "plain sum, norm": GQA + f", {INDEXER}, weighted=false, norm=rmsnorm))",
    "interleaved rope": "gqa(num_heads=4, num_kv_heads=2, rope=rope(fraction=0.5),"
    " indexer=indexer(num_heads=2, head_dim=4, top_k=4, rope_layout=interleaved))",
    "blocks": GQA + f", {INDEXER}, block_size=2))",
    "blocks, plain sum": GQA
    + f", {INDEXER}, block_size=3, weighted=false, norm=rmsnorm))",
    "window, sinks": GQA + f", sliding_window_size=5, sinks=true, {INDEXER}))",
    "relative bias": GQA + f", relative_bias=relbias(rank=2, extent=4), {INDEXER}))",
    "mla": MLA + f", {INDEXER}))",
    "mla, full-rank queries": "mla(num_heads=4, kv_lora_rank=16, qk_nope_head_dim=8,"
    " qk_rope_head_dim=8, v_head_dim=8, indexer=indexer(num_heads=2, head_dim=4, top_k=4))",
}


@pytest.mark.parametrize("attn", SPARSE.values(), ids=SPARSE.keys())
def test_sparse_attention(attn: str) -> None:
    check.compare(compile(stack(attn))).check()


INDEX_PORT = {
    "gqa": wired(
        GQA + f", {INDEXER}))",
        "attn(index=@topk)",
        "attn(index=$topk)",
        "attn(index=@topk)",
        "attn(index=$topk!)",
    ),
    "gqa, blocks, window": wired(
        GQA + f", sliding_window_size=5, {INDEXER}, block_size=2))",
        "attn(index=@topk)",
        "attn(index=$topk!, kv_conv_size=2)",
    ),
    "mla": wired(
        MLA + f", {INDEXER}))",
        "attn(index=@topk)",
        "attn(index=$topk)",
        "attn(index=$topk)",
    ),
}


@pytest.mark.parametrize("source", INDEX_PORT.values(), ids=INDEX_PORT.keys())
def test_index_port(source: str) -> None:
    check.compare(compile(source)).check()


def test_an_indexer_beside_a_conv_says_so() -> None:
    source = stack(GQA + f", kv_conv_size=2, {INDEXER}))")
    with pytest.raises(NotImplementedError, match="kv_conv_size"):
        check.compare(compile(source))


# The zoo's models that use these layers, shrunk as tests/test_examples.py shrinks
# them, their other layers made dense ones (MoE, Mamba, Gated DeltaNet), and their
# deepest repeats cut short; the multi-token heads are left out.
ROOT = Path(__file__).parent.parent / "examples" / "architectures"
SMALL = [
    "d_model = 32",
    "vocab_size = 64",
    "dense = gated_mlp(intermediate_size=8, activation=swish)",
]
SMALL_MLA = (
    "mla(num_heads=2, q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4,"
    " qk_rope_head_dim=4, v_head_dim=4"
)
YARN = (
    ", rope=rope(layout=interleaved, scaling=yarn(factor=40, original_max_len=4096,"
    " attention_factor_on=softmax))"
)
ZOO = {
    "gpt_oss_120b.bux": [
        (
            "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8, bias=true, sinks=true,"
            " rope=rope(base=150000, scaling=yarn(factor=32, original_max_len=4096)))"
        ),
        "local = attn(sliding_window_size=4)",
        "experts = dense",
        'model = embed "LG"^3 norm head',
    ],
    "gemma3_4b.bux": [
        (
            "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8, qk_norm=norm,"
            " rope=rope(base=10000))"
        ),
        "local = attn(sliding_window_size=4)",
        "global = attn(rope=rope(base=1000000, scaling=pi(factor=8)))",
        "ffn = dense",
        'model = embed(scale_by_sqrt_dim=true) "LLLLLG" "LL" norm head(tied=true)',
    ],
    "hymba.bux": [
        (
            "local = gqa(num_heads=4, num_kv_heads=2, v_head_dim=8, fused_qkv=true,"
            " sliding_window_size=4, rope=rope(base=74535))"
        ),
        "ssm = dense",
        "ffn = dense",
        'model = embed F "WR" F "WSR" "WR" F norm head(tied=true)',
    ],
    "resformer.bux": [
        "attn = gqa(num_heads=4, num_kv_heads=4, rope=rope())",
        "ffn = dense",
        "model = embed first rest^3 norm head",
    ],
    "deepseek_v3.bux": [
        f"attn = {SMALL_MLA}{YARN})",
        "experts = dense",
        "model = embed layer(dense) layer(experts)^2 norm head",
    ],
    "kimi/kimi_k2.5.bux": [
        (
            f"attn = {SMALL_MLA}, rope=rope(base=50000, layout=interleaved,"
            " scaling=yarn(factor=64, original_max_len=4096, attention_factor_on=softmax)))"
        ),
        "experts = dense",
        "model = embed layer(dense) layer(experts)^2 norm head",
    ],
    "kimi_linear.bux": [
        f"mla = {SMALL_MLA}, rope=none)",
        "kda = dense",
        "experts = dense",
        'model = embed "FKKA" "KKA" norm head',
    ],
    "deepseek/deepseek_v3.2_exp.bux": [
        f"attn = {SMALL_MLA}{YARN}, indexer=indexer(num_heads=2, head_dim=8, top_k=4))",
        "experts = dense",
        "model = embed layer(dense) layer(experts)^2 norm out",
    ],
    "glm/glm_5.3.bux": [
        (
            f"attn = {SMALL_MLA}, rope=rope(base=8000000, layout=interleaved),"
            " indexer=indexer(num_heads=2, head_dim=8, top_k=4, rope_layout=interleaved))"
        ),
        "experts = dense",
        'model = embed "AAA" "SSS" "FSSS" norm out',
    ],
    "qwen3_next.bux": [
        (
            "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8, gated=true, qk_norm=norm,"
            " rope=rope(base=10000000, fraction=0.25))"
        ),
        "delta = dense",
        "experts = dense",
        'model = embed "DDDA"^2 norm out',
    ],
    "inkling/inkling.bux": [
        (
            "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8, qk_norm=norm, scale=1 / 8,"
            " kv_conv_size=4, relative_bias=relbias(rank=2, extent=8), log_scale=0.1,"
            " log_scale_floor=4)"
        ),
        "local = attn(sliding_window_size=4, relative_bias.extent=4, log_scale=none)",
        "experts = dense",
        'model = embed "DD" "LLLG" norm logits',
    ],
}


@pytest.mark.parametrize("path", ZOO, ids=lambda p: Path(p).stem)
def test_zoo_models(path: str) -> None:
    arch = compile((ROOT / path).read_text(), overrides=[*SMALL, *ZOO[path]])
    check.compare(arch).check()
