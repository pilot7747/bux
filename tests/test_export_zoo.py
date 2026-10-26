"""Every model of the zoo, shrunk, exports to transformers and computes what bux does,
or says which layer or setting it lacks (bux.export). The table of what each lacks is
strict both ways: a model that comes to export must leave it."""

from pathlib import Path

import jax
import pytest
from flax import nnx
from test_examples import (
    EXAMPLES,
    EXPERTS,
    GROUPED,
    INDEXER,
    MLA,
    ROOT,
    SHRUNK,
    SMALL,
    YARN,
)

from bux.architecture import Model, num_params
from bux.export import check, codegen
from bux.export.huggingface import cache_layers
from bux.export.plan import plan
from bux.lang import compile, ir

#: The zoo's models that do not export yet, by what the export says they lack first:
#: a part of its NotImplementedError's message.
MISSING: dict[str, str] = {}

#: DeepSeek-V3's dense and MoE layers, a few of each.
DEEPSEEK = "layer(dense)^2 layer(experts)^3"
#: Qwen3-Next's token mixers, shrunk, and its experts.
QWEN_NEXT = [
    "delta = gdn(num_heads=2, num_v_heads=4, head_dim=8)",
    (
        "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8, gated=true, qk_norm=norm,"
        " rope=rope(base=10000000, fraction=0.25))"
    ),
]
QWEN_EXPERTS = (
    "experts = moe(num_experts=4, top_k=2, num_shared_experts=1,"
    " shared_expert_gate=true, expert=dense)"
)
#: The Nemotrons' mamba2, attention and LatentMoE, shrunk.
NEMOTRON = [
    "ssm = mamba2(state_size=8, head_dim=8, num_groups=2)",
    "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8)",
    (
        "experts = moe(num_experts=4, top_k=2, num_shared_experts=1, router=sigmoid,"
        " routed_scaling_factor=5.0, load_balancing=aux_free, latent_size=16,"
        " expert=mlp(intermediate_size=8, activation=relu2),"
        " shared_expert=mlp(intermediate_size=8, activation=relu2))"
    ),
]

#: Grouped-query attention, shrunk: a call that each entry ends with its own settings.
ATTN = "gqa(num_heads=4, num_kv_heads=2, head_dim=8"

#: The widths of the zoo's models that `SHRUNK` has no entry for, and `SMALL` leaves
#: wide: their heads, hidden units and experts, as `SHRUNK` shrinks the others'.
WIDTHS = {
    "arcee/trinity_mini.bux": [
        "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8, qk_norm=norm, gated=true)",
        (
            "experts = moe(num_experts=4, top_k=2, num_shared_experts=1, router=sigmoid,"
            " routed_scaling_factor=2.826, load_balancing=aux_free, center_bias=true,"
            " expert=dense)"
        ),
    ],
    "llama3_8b.bux": [f"attn = {ATTN}, rope=rope(base=500000))", "ffn = dense"],
    "llama3_70b.bux": [f"attn = {ATTN}, rope=rope(base=500000))", "ffn = dense"],
    "gemma3_4b.bux": [
        f"attn = {ATTN}, qk_norm=norm, rope=rope(base=10000))",
        "local = attn(sliding_window_size=4)",
        "ffn = gated_mlp(intermediate_size=8, activation=gelu_tanh)",
    ],
    "gpt_j.bux": [
        "attn = gqa(num_heads=4, num_kv_heads=4, rope=rope(fraction=0.25, layout=interleaved))",
        "ffn = mlp(intermediate_size=8, activation=gelu_tanh, bias=true)",
    ],
    "ngpt.bux": [
        "attn = gqa(num_heads=4, num_kv_heads=4, qk_norm=rmsnorm, rope=rope(layout=interleaved))",
        "ffn = dense",
    ],
    "hymba.bux": [
        (
            "local = gqa(num_heads=4, num_kv_heads=2, v_head_dim=8, fused_qkv=true,"
            " sliding_window_size=4, rope=rope(base=74535))"
        ),
        "ssm = mamba(state_size=4)",
        "ffn = dense",
    ],
    "cohere/command_a_plus_05_2026.bux": [
        f"attn = {ATTN}, rope=rope(base=50000, layout=interleaved))",
        "local = attn(sliding_window_size=4)",
        f"{EXPERTS} * 0.5",
    ],
    "cohere/north_mini_code_1.0.bux": [
        f"attn = {ATTN}, rope=rope(base=50000, layout=interleaved))",
        "local = attn(sliding_window_size=4)",
        (
            "experts = moe(num_experts=4, top_k=2, router=sigmoid,"
            " normalize_top_k=false, expert=dense)"
        ),
    ],
    "deepseek/deepseek_r1.bux": [f"attn = {MLA}{YARN})", GROUPED],
    "deepseek/deepseek_v3.2.bux": [f"attn = {MLA}{YARN}{INDEXER})", GROUPED],
    "gpt_oss_120b.bux": [
        (
            f"attn = {ATTN}, bias=true, sinks=true, rope=rope(base=150000,"
            " scaling=yarn(factor=32, original_max_len=4096)))"
        ),
        "local = attn(sliding_window_size=4)",
        (
            "experts = moe(num_experts=4, top_k=2, router_bias=true,"
            " expert=gated_mlp(intermediate_size=8, activation=quick_gelu, bias=true,"
            " limit=7.0, up_offset=1.0))"
        ),
    ],
    "kimi_linear.bux": [
        "kda = kda(num_heads=2, head_dim=8, output_gate_rank=4, norm_eps=1e-5)",
        f"mla = {MLA}, rope=none)",
        EXPERTS,
    ],
    "laguna/laguna_xs_2.1.bux": [
        (
            f"attn = {ATTN}, qk_norm=norm, gated=true, gate_per_head=true,"
            " gate_activation=softplus)"
        ),
        (
            "global = attn(num_heads=6, rope=rope(base=500000, fraction=0.5,"
            " scaling=yarn(factor=32, original_max_len=8192, beta_fast=64)))"
        ),
        "local = attn(sliding_window_size=4, rope=rope())",
        EXPERTS,
    ],
    "mellum/mellum2_12b_a2.5b_thinking.bux": [
        f"attn = {ATTN}, qk_norm=norm, rope=rope(base=500000))",
        "local = attn(sliding_window_size=4)",
        "experts = moe(num_experts=4, top_k=2, aux_loss_coef=1e-3, expert=dense)",
    ],
    "mimo/mimo_v2.6_pro.bux": [
        (
            "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=12, v_head_dim=8,"
            " fused_qkv=true, rope=rope(base=10000000, fraction=0.334))"
        ),
        "local = attn(sliding_window_size=4, sinks=true, rope.base=10000)",
        EXPERTS,
    ],
    "qwen/qwen3.8_2.4t_a95b.bux": [*QWEN_NEXT, QWEN_EXPERTS],
    "qwen/qwen3.8_27b.bux": [*QWEN_NEXT, "ffn = dense"],
    "qwen/qwen3_30b_a3b.bux": [
        f"attn = {ATTN}, qk_norm=norm, rope=rope(base=1000000))",
        "experts = moe(num_experts=4, top_k=2, aux_loss_coef=1e-3, expert=dense)",
    ],
    "qwen3_next.bux": [*QWEN_NEXT, QWEN_EXPERTS],
    "nemotron/nemotron_3.5_bolt_74b_a9b.bux": [
        *NEMOTRON,
        "E(n, width) = [norm experts]",
    ],
    "nemotron/nemotron_3_super_120b_a12b.bux": NEMOTRON,
}

#: The zoo's deep models, a few passes of each repeat. A repeat exports as one loop at
#: any depth, while dozens of passes of bux's eager forwards take a minute a model.
#: Deep stacks of delta-rule layers (gdn, kda) must be shallow too: float32 rounds its
#: way through the delta rule by about 5e-6 of the logits a layer, bux and the export
#: alike, which over the files' dozens of layers passes compare's 1e-4 where both are
#: right (docs/export/index.md, "Float32 and depth").
DEPTHS = {
    "deepseek_v3.bux": ["model = embed layer(dense)^2 layer(experts)^3 norm head"],
    "deepseek/deepseek_r1.bux": [f"model = embed {DEEPSEEK} norm @h out mtp"],
    "deepseek/deepseek_v3.2.bux": [f"model = embed {DEEPSEEK} norm @h out mtp"],
    "deepseek/deepseek_v3.2_exp.bux": [f"model = embed {DEEPSEEK} norm @h out mtp"],
    "deepseek/deepseek_v4.1_flash.bux": [
        'model = embed W memory W "Ppp" memory "Pp" "Ss" "Iss" norm head'
    ],
    "glm/glm_5.3.bux": ['model = embed "AAA" "SSS" "FSSS"^2 norm @h out mtp^3'],
    "inkling/inkling.bux": [
        (
            'model = embed(norm=norm) "DD" "LLLG" "LLLLLG"^2 norm @h logits'
            " mtp(local) mtp(attn) mtp(local)^2"
        )
    ],
    "kimi/kimi_k2.5.bux": ["model = embed layer(dense) layer(experts)^3 norm head"],
    "muse/muse_spark_open_816b_a42b.bux": [
        (
            'model = embed bare "LG" "LLLG"^2 (rmsnorm(eps=1e-5) * logit_scale)'
            " head(softcap=20)"
        )
    ],
    "nemotron/nemotron_3_ultra_550b_a55b.bux": [
        'model = embed ("MEMEMEMAE" "MEMEMAE")^2 "MEMEMEME" norm @h out mtp^2'
    ],
    "nemotron/nemotron_3_nano_30b_a3b.bux": [
        'model = embed "MEMEMA" "EMEMEMA"^2 "EMEMEMEME" norm head'
    ],
    "llama3_70b.bux": ["model = embed block^4 norm head"],
    "mhc.bux": ["model = embed block^4 norm head"],
    "gemma3_4b.bux": [
        'model = embed(scale_by_sqrt_dim=true) "LLLLLG"^2 "LLLL" norm head(tied=true)'
    ],
    "jamba.bux": ['model = embed "mMmM aMmM"^2 norm head'],
    "zamba2.bux": ["model = embed (m^2 h)^3 m^2 norm head(tied=true)"],
    "qwen/qwen3.8_flash_next.bux": ['model = embed D ple "DDA" "DDDA"^2 @h out mtp'],
    "qwen/qwen3.8_2.4t_a95b.bux": ['model = embed "DDDA"^2 norm @h out mtp'],
    "qwen/qwen3.8_27b.bux": ['model = embed "DDDA"^2 norm @h out mtp'],
    "qwen3_next.bux": ['model = embed "DDDA"^3 norm @h out mtp'],
    "kimi/kimi_k3.bux": ['model = embed "FKKA" "KKKA"^2 "A" norm head'],
    "kimi_linear.bux": ['model = embed "FKKA" "KKKA" "KKA" norm head'],
}

#: The most parameters a shrunk model is compared at; a larger one, which `SHRUNK`
#: does not shrink, is only checked for the layers it lacks.
LIMIT = 4_000_000


@pytest.fixture(autouse=True)
def _forget_compiled():
    """Drop what JAX compiled for a model after its test: the eager ops of forty
    models, each of its own shapes, would otherwise add up to gigabytes."""
    yield
    jax.clear_caches()


def name(path: Path) -> str:
    return str(path.relative_to(ROOT))


def shrunk(path: Path) -> ir.Architecture:
    """The model of `path`, shrunk as `tests/test_examples.py` shrinks it, and a few
    passes deep (`DEPTHS`)."""
    extra = SHRUNK.get(name(path), WIDTHS.get(name(path), []))
    extra = [*SMALL.strip().splitlines(), *extra, *DEPTHS.get(name(path), [])]
    return compile(path.read_text(), overrides=extra)


def generated(arch: ir.Architecture) -> None:
    """Generate the export's code, which looks up each layer's exported class."""
    p = plan(arch)
    cache_layers(p)
    codegen.modeling(p, [])


def test_the_tables_name_models_of_the_zoo():
    zoo = {name(p) for p in EXAMPLES}
    assert MISSING.keys() <= zoo and DEPTHS.keys() <= zoo
    assert WIDTHS.keys() <= zoo - SHRUNK.keys()


@pytest.mark.parametrize("path", EXAMPLES, ids=name)
def test_a_model_of_the_zoo_exports_or_says_what_it_lacks(path):
    arch = shrunk(path)
    n = num_params(nnx.eval_shape(lambda: Model(arch, rngs=nnx.Rngs(0))))
    report, said = None, None
    try:
        if n > LIMIT:
            generated(arch)  # too large to compare: its code, at least
        else:
            report = check.compare(arch)
    except NotImplementedError as err:
        said = str(err)
    if (missing := MISSING.get(name(path))) is not None:
        now = "generates its code" if report is None else "exports"
        assert said is not None, f"{name(path)} {now} now: take it out of MISSING"
        assert missing in said, f"{name(path)} lacks {said!r}, not {missing!r}"
    elif said is not None:
        pytest.fail(f"{name(path)}: {said}")
    elif report is None:
        pytest.skip(
            f"shrunk, it keeps {n / 1e6:.0f}M parameters: give it an entry in "
            "tests/test_examples.py's SHRUNK to compare it"
        )
    else:
        report.check()
