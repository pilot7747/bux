"""Recurrent token mixers export to transformers and compute what bux does
(bux.export): linear attention and state space models, their states kept in
transformers' cache while decoding, and left-padded batches as each row alone."""

import pytest
from test_examples import ROOT, SMALL, V

from bux.export import check
from bux.lang import compile

HEAD = "d_model = 64\nvocab_size = 256\n"
FFN = "ffn = gated_mlp(intermediate_size=128, activation=swish)\n"
ATTN = "attn = gqa(num_heads=4, num_kv_heads=2, rope=rope(fraction=0.5))\n"


def stack(mixer: str, blocks: int = 2) -> str:
    """A model of `blocks` pre-norm blocks of the token mixer `mixer` and an MLP."""
    return (
        HEAD
        + FFN
        + f"mixer = {mixer}\n"
        + f"model = embed ([rmsnorm mixer] [rmsnorm ffn])^{blocks} rmsnorm head"
    )


def hybrid(mixer: str) -> str:
    """The token mixer `mixer` with softmax attention between, so that the cache
    holds layers of both kinds."""
    return (
        HEAD
        + FFN
        + ATTN
        + f"mixer = {mixer}\n"
        + "model = embed [rmsnorm mixer] [rmsnorm ffn] [rmsnorm attn] [rmsnorm ffn]"
        + " [rmsnorm mixer] rmsnorm head"
    )


GDN = "gdn(num_heads=2, head_dim=16)"
KDA = "kda(num_heads=2, head_dim=16)"
GDN2 = "gdn2(num_heads=2, head_dim=16)"
DELTANET = "deltanet(num_heads=2, head_dim=16)"
GLA = "gla(num_heads=2, head_dim=16, gate_rank=8)"
LIGHTNING = "lightning(num_heads=2, head_dim=16)"
RWKV7 = "rwkv7(num_heads=2, head_dim=16, decay_rank=8, a_rank=8, v_rank=4, gate_rank=8)"
MAMBA = "mamba(state_size=4)"
MAMBA2 = "mamba2(state_size=8, head_dim=16)"
SHORT_CONV = "short_conv"

# Every layer, and every setting that changes what it computes, one at a time.
MIXERS = {
    "gdn": GDN,
    "gdn value heads": "gdn(num_heads=2, num_v_heads=4, head_dim=16, v_head_dim=8)",
    "gdn no conv": "gdn(num_heads=2, head_dim=16, conv_size=none)",
    "gdn conv 2": "gdn(num_heads=2, head_dim=16, conv_size=2)",
    "gdn sigmoid gate": "gdn(num_heads=2, head_dim=16, output_gate=sigmoid)",
    "gdn no gate": "gdn(num_heads=2, head_dim=16, output_gate=none)",
    "gdn norm eps": "gdn(num_heads=2, head_dim=16, norm_eps=0.5)",
    "kda": KDA,
    "kda value heads": "kda(num_heads=2, num_v_heads=4, head_dim=16, v_head_dim=8)",
    "kda gate rank": "kda(num_heads=2, head_dim=16, gate_rank=4)",
    "kda output gate rank": "kda(num_heads=2, head_dim=16, output_gate_rank=4)",
    "kda lower bound": "kda(num_heads=2, head_dim=16, gate_lower_bound=-5)",
    "kda swish gate": "kda(num_heads=2, head_dim=16, output_gate=swish)",
    "kda no conv": "kda(num_heads=2, head_dim=16, conv_size=none)",
    "gdn2": GDN2,
    "gdn2 value heads": "gdn2(num_heads=2, num_v_heads=4, head_dim=16, v_head_dim=8)",
    "gdn2 gate ranks": "gdn2(num_heads=2, head_dim=16, gate_rank=4, output_gate_rank=4)",
    "gdn2 lower bound": "gdn2(num_heads=2, head_dim=16, gate_lower_bound=-5)",
    "gdn2 negative eigenvalues": "gdn2(num_heads=2, head_dim=16, negative_eigenvalues=true)",
    "gdn2 sigmoid gate": "gdn2(num_heads=2, head_dim=16, output_gate=sigmoid)",
    "gdn2 no conv": "gdn2(num_heads=2, head_dim=16, conv_size=none)",
    "deltanet": DELTANET,
    "deltanet no gate": "deltanet(num_heads=2, head_dim=16, output_gate=none)",
    "deltanet value heads": "deltanet(num_heads=1, num_v_heads=2, head_dim=16)",
    "gla": GLA,
    "gla conv": "gla(num_heads=2, head_dim=16, gate_rank=8, conv_size=4)",
    "gla value heads": "gla(num_heads=2, num_v_heads=4, head_dim=16, v_head_dim=8)",
    "gla sigmoid gate": "gla(num_heads=2, head_dim=16, output_gate=sigmoid)",
    "lightning": LIGHTNING,
    "lightning conv": "lightning(num_heads=2, head_dim=16, conv_size=3)",
    "lightning value heads": "lightning(num_heads=2, num_v_heads=4, head_dim=16)",
    "lightning no gate": "lightning(num_heads=2, head_dim=16, output_gate=none)",
    "rwkv7": RWKV7,
    "rwkv7 value heads": RWKV7.replace("head_dim=16", "num_v_heads=4, head_dim=16"),
    "rwkv7 v head dim": RWKV7.replace("head_dim=16", "head_dim=16, v_head_dim=8"),
    "rwkv7 swish gate": RWKV7.replace(")", ", output_gate=swish)"),
    "rwkv7 no gate": RWKV7.replace(")", ", output_gate=none)"),
    "rwkv7 norm eps": RWKV7.replace(")", ", norm_eps=0.01)"),
    "mamba": MAMBA,
    "mamba dt rank": "mamba(state_size=4, dt_rank=2)",
    "mamba expand": "mamba(state_size=4, expand=1, conv_size=2)",
    "mamba2": MAMBA2,
    "mamba2 groups": "mamba2(state_size=8, head_dim=8, num_groups=2)",
    "mamba2 heads": "mamba2(state_size=8, head_dim=8, num_heads=6, num_groups=3)",
    "mamba2 expand": "mamba2(state_size=4, head_dim=16, expand=1, conv_size=2)",
    "short_conv": SHORT_CONV,
    "short_conv bias": "short_conv(bias=true)",
    "short_conv size": "short_conv(conv_size=5)",
}


@pytest.mark.parametrize("mixer", MIXERS.values(), ids=MIXERS.keys())
def test_a_recurrent_mixer_exports(mixer: str) -> None:
    check.compare(compile(stack(mixer))).check()


# Decoding from a prompt of 2 (beside one of 1) against a conv of 4, and of 80
# (beside 40) against prefill chunks of 64.
LENGTHS = {"shorter than the conv": (4, 6), "longer than a chunk": (160, 3)}
DECODED = {
    "gdn": GDN,
    "kda": KDA,
    "gdn2": GDN2,
    "deltanet": DELTANET,
    "gla": GLA,
    "lightning": LIGHTNING,
    "rwkv7": RWKV7,
    "mamba": MAMBA,
    "mamba2": MAMBA2,
    "short_conv": SHORT_CONV,
}


@pytest.mark.parametrize("length,steps", LENGTHS.values(), ids=LENGTHS.keys())
@pytest.mark.parametrize("mixer", DECODED.values(), ids=DECODED.keys())
def test_decoding_crosses_the_conv_and_a_chunk(
    mixer: str, length: int, steps: int
) -> None:
    check.compare(compile(stack(mixer)), length=length, steps=steps).check()


@pytest.mark.parametrize("mixer", DECODED.values(), ids=DECODED.keys())
def test_a_hybrid_with_attention_exports(mixer: str) -> None:
    check.compare(compile(hybrid(mixer)), steps=8).check()


def test_lightning_decays_by_its_depth_among_the_token_mixers() -> None:
    """Three lightning layers, one shared and a gqa between: their rates are their
    depths' among all five mixer calls."""
    source = (
        HEAD
        + FFN
        + ATTN
        + f"mix = {LIGHTNING}\nshared again = [rmsnorm mix]\n"
        + "model = embed again [rmsnorm mix] [rmsnorm attn] again [rmsnorm mix]"
        + " [rmsnorm ffn] rmsnorm head"
    )
    check.compare(compile(source)).check()


def test_rwkv7_value_residual_exports() -> None:
    """The first layer writes its values to a tap, the later ones mix them in, the
    last popping them."""
    source = (
        HEAD
        + FFN
        + f"mix = {RWKV7}\n"
        + "model = embed [rmsnorm mix(v=@v0)] [rmsnorm ffn] [rmsnorm mix(v=$v0)]"
        + " [rmsnorm ffn] [rmsnorm mix(v=$v0!)] rmsnorm head"
    )
    check.compare(compile(source)).check()


# Zoo models shrunk as tests/test_examples.py shrinks them, their MoE to 4 experts.
# Qwen3-Next makes 3 passes of its 12: through 36 gdn layers, float32's own rounding
# reaches 2e-4 of the logits, bux's and the export's alike (measured against the
# export in float64), past the check's 1e-4.
GQA = "gqa(num_heads=4, num_kv_heads=2, head_dim=8"
ZOO = {
    "qwen3_next.bux": [
        "delta = gdn(num_heads=2, num_v_heads=4, head_dim=8)",
        f"attn = {GQA}, gated=true, qk_norm=norm, rope=rope(base=10000000, fraction=0.25))",
        (
            "experts = moe(num_experts=4, top_k=2, num_shared_experts=1,"
            " shared_expert_gate=true, expert=dense)"
        ),
        'model = embed "DDDA"^3 norm @h out mtp',
    ],
    "kimi_linear.bux": [
        "kda = kda(num_heads=2, head_dim=8, output_gate_rank=8, norm_eps=1e-5)",
        (
            "mla = mla(num_heads=2, kv_lora_rank=8, qk_nope_head_dim=4,"
            " qk_rope_head_dim=4, v_head_dim=4, rope=none)"
        ),
        (
            "experts = moe(num_experts=4, top_k=2, num_shared_experts=1, router=sigmoid,"
            " routed_scaling_factor=2.446, load_balancing=aux_free, expert=dense)"
        ),
    ],
    "zamba2.bux": [
        "ssm = mamba2(state_size=8, head_dim=8)",
        "attn = gqa(num_heads=4, num_kv_heads=4, head_dim=8, scale=0.125, rope=rope())",
        "ffn = gated_mlp(intermediate_size=8, activation=gelu)",
    ],
    "nemotron/nemotron_3_nano_30b_a3b.bux": [
        "ssm = mamba2(state_size=8, head_dim=8, num_heads=6, num_groups=2)",
        f"attn = {GQA})",
        (
            "experts = moe(num_experts=4, top_k=2, num_shared_experts=1, router=sigmoid,"
            " load_balancing=aux_free, expert=mlp(intermediate_size=8, activation=relu2))"
        ),
    ],
    "jamba.bux": [
        f"attn = {GQA})",
        "ssm = mamba(state_size=4)",
        "experts = moe(num_experts=4, top_k=2, normalize_top_k=false, expert=dense)",
    ],
    "hymba.bux": [
        (
            f"local = {GQA}, v_head_dim=16, fused_qkv=true, sliding_window_size=4,"
            " rope=rope(base=74535))"
        ),
        "ssm = mamba(state_size=4)",
        "ffn = dense",
    ],
}


@pytest.mark.parametrize("path", ZOO, ids=ZOO.keys())
def test_zoo_models_export(path: str) -> None:
    overrides = [*SMALL.strip().splitlines(), *ZOO[path]]
    arch = compile((ROOT / path).read_text(), overrides=overrides)
    assert arch.vocab_size == V
    check.compare(arch).check()
