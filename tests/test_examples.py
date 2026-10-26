"""Every example architecture builds in the reference backend and runs: abstractly at
its full size, and for real shrunk to a few dims."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from bux.architecture import Model, num_params
from bux.lang import compile, compile_file
from bux.train.objective import prediction_offsets

ROOT = Path(__file__).parent.parent / "examples" / "architectures"
EXAMPLES = sorted(ROOT.rglob("*.bux"))
D, V = 32, 64


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_examples_build_and_run_at_full_size(path):
    """No parameter is allocated and nothing is computed, but every layer is built and
    traced with its shapes checked."""
    arch = compile_file(str(path))
    model = nnx.eval_shape(lambda: Model(arch, rngs=nnx.Rngs(0)))
    graphdef, state = nnx.split(model)
    tokens = jax.ShapeDtypeStruct((2, 16), jnp.int32)
    positions = jax.ShapeDtypeStruct((2, 16), jnp.int32)  # packed documents

    def outputs(state, tokens, positions):
        return nnx.merge(graphdef, state).outputs(tokens, positions)

    logits = jax.eval_shape(outputs, state, tokens, positions)
    heads = len(prediction_offsets(arch))
    assert [o.shape for o in logits] == [(2, 16, arch.vocab_size)] * heads
    assert num_params(model) > 0


# Example architectures shrunk to a few dims, a statement a line; the layers' wiring
# and their depth are the files'.
SMALL = f"""
d_model = {D}
vocab_size = {V}
dense = gated_mlp(intermediate_size=8, activation=swish)
"""
EXPERTS = (
    "experts = moe(num_experts=4, top_k=2, num_shared_experts=1, router=sigmoid,"
    " load_balancing=aux_free, expert=dense)"
)
GROUPED = (
    "experts = moe(num_experts=4, top_k=2, num_shared_experts=1, router=sigmoid,"
    " routed_scaling_factor=2.5, num_groups=2, top_k_groups=1,"
    " load_balancing=aux_free, expert=dense)"
)
MLA = (
    "mla(num_heads=2, q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4,"
    " qk_rope_head_dim=4, v_head_dim=4"
)
YARN = (
    ", rope=rope(layout=interleaved, scaling=yarn(factor=40, original_max_len=4096,"
    " attention_factor_on=softmax))"
)
INDEXER = ", indexer=indexer(num_heads=2, head_dim=8, top_k=4)"
SHRUNK = {
    "deepseek_v3.bux": [f"attn = {MLA}{YARN})", GROUPED],
    "deepseek/deepseek_v3.2_exp.bux": [f"attn = {MLA}{YARN}{INDEXER})", GROUPED],
    "glm/glm_5.3.bux": [
        (
            f"attn = {MLA}, indexer=indexer(num_heads=2, head_dim=8, top_k=4,"
            " rope_layout=interleaved))"
        ),
        EXPERTS,
    ],
    "kimi/kimi_k2.5.bux": [f"attn = {MLA})", EXPERTS],
    "deepseek/deepseek_v4.1_flash.bux": [
        "memory = engram(head_dim=4, table_size=50, conv_size=none, vocab_classes=40)",
        (
            "window = csa(num_heads=4, head_dim=16, q_lora_rank=8, o_lora_rank=4,"
            " o_groups=2, sliding_window_size=4, sinks=true,"
            " rope=rope(fraction=0.125, layout=interleaved))"
        ),
        (
            "pairs = window(compress_ratio=2, rope=far, indexer=indexer(num_heads=2,"
            " head_dim=8, top_k=3, candidate_block_size=2, candidate_top_k=2))"
        ),
        EXPERTS,
    ],
    "kimi/kimi_k3.bux": [
        "kda = kda(num_heads=2, head_dim=8, gate_lower_bound=-5)",
        f"mla = {MLA}, gated=true, rope=none)",
        (
            "experts = moe(num_experts=4, top_k=2, num_shared_experts=2, router=sigmoid,"
            " load_balancing=aux_free, latent_size=16, latent_output_norm=norm,"
            " expert=dense)"
        ),
    ],
    "qwen/qwen3.8_flash_next.bux": [
        "residual = gr(expansion=4, rank=4)",
        "delta = gdn(num_heads=2, num_v_heads=4, head_dim=8, output_gate=sigmoid)",
        (
            "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8, gated=true,"
            " qk_norm=norm, rope=rope(fraction=0.25), indexer=indexer(num_heads=2,"
            " head_dim=8, top_k=4, block_size=2, weighted=false, norm=norm))"
        ),
        (
            "experts = moe(num_experts=4, top_k=2, num_shared_experts=1,"
            " shared_expert_gate=true, expert=dense)"
        ),
        f"ple = engram(head_dim=4, table_size=50, norm=norm, boundary_token={V - 1})",
    ],
    "minimax/minimax_m3.bux": [
        (
            "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8, qk_norm=norm,"
            " rope=rope(fraction=0.5))"
        ),
        (
            "msa = attn(indexer=indexer(num_heads=2, head_dim=8, top_k=4, block_size=2,"
            " pool=max, per_group=true, relu=false, weighted=false, norm=norm))"
        ),
        (
            "experts = moe(num_experts=4, top_k=2, num_shared_experts=1, router=sigmoid,"
            " routed_scaling_factor=2.0, load_balancing=aux_free,"
            " expert=gated_mlp(intermediate_size=8, activation=quick_gelu, limit=7.0,"
            " up_offset=1.0))"
        ),
    ],
    "minimax/minimax_m3.1_flash.bux": [
        (
            "msa = gqa(num_heads=4, num_kv_heads=2, head_dim=8, qk_norm=norm,"
            " rope=rope(fraction=0.5), indexer=indexer(num_heads=2, head_dim=8,"
            " top_k=4, block_size=2, pool=max, per_group=true, relu=false,"
            " weighted=false, norm=norm))"
        ),
        (
            "experts = moe(num_experts=4, top_k=2, num_shared_experts=1, router=sigmoid,"
            " routed_scaling_factor=2.0, load_balancing=aux_free,"
            " expert=gated_mlp(intermediate_size=8, activation=quick_gelu, limit=7.0,"
            " up_offset=1.0))"
        ),
    ],
    "muse/muse_spark_open_816b_a42b.bux": [
        (
            "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8, qk_norm=bare,"
            " gated=true, output_norm=bare)"
        ),
        (
            "experts = moe(num_experts=4, top_k=2, load_balancing=aux_free,"
            " router=sigmoid, router_logit_scale=logit_scale, latent_size=16,"
            " latent_input_norm=norm, expert_norm=rescale, expert=dense)"
        ),
    ],
    "reflection/beam_1.0_501b_a23b.bux": [
        (
            "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8, qk_norm=norm,"
            " gated=true, gate_head_bias=true)"
        ),
        "local = attn(sliding_window_size=4, rope=rope(base=10000))",
        "global = attn(power_scale=0.1, power_scale_length=2)",
        (
            "experts = moe(num_experts=4, top_k=2, router=sigmoid,"
            " routed_scaling_factor=2.5, num_groups=2, top_k_groups=1,"
            " load_balancing=aux_free, expert=dense)"
        ),
    ],
    "inkling/inkling.bux": [
        (
            "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8, qk_norm=norm,"
            " kv_conv_size=4, relative_bias=relbias(rank=2, extent=8), log_scale=0.1,"
            " log_scale_floor=4)"
        ),
        "local = attn(sliding_window_size=4, relative_bias.extent=4, log_scale=none)",
        (
            "experts = moe(num_experts=4, top_k=2, num_shared_experts=2,"
            " shared_expert_sink=true, router=sigmoid, routed_scaling_factor=8,"
            " load_balancing=aux_free, expert=dense)"
        ),
    ],
    "arcee/trinity_large.bux": [
        "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8, qk_norm=norm, gated=true)",
        (
            "experts = moe(num_experts=4, top_k=2, num_shared_experts=1, router=sigmoid,"
            " routed_scaling_factor=2.448, load_balancing=smebu, aux_loss_coef=1e-4,"
            " aux_loss_scope=sequence, expert=dense)"
        ),
    ],
    "arcee/trinity_nano.bux": [
        "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8, qk_norm=norm, gated=true)",
        (
            "experts = moe(num_experts=4, top_k=2, num_shared_experts=1, router=sigmoid,"
            " load_balancing=aux_free, center_bias=true, expert=dense)"
        ),
    ],
    # minimax_m1 and minimax_m2.x only use these two; the full-size test builds them.
    "minimax/minimax_text_01.bux": [
        "lightning = lightning(num_heads=2, head_dim=8)",
        (
            "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8,"
            " rope=rope(base=10000000, fraction=0.5))"
        ),
        "experts = moe(num_experts=4, top_k=2, aux_loss_coef=1e-3, expert=dense)",
    ],
    "minimax/minimax_m2.bux": [
        (
            "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8, qk_norm=norm,"
            " qk_norm_all_heads=true, rope=rope(base=5000000, fraction=0.5))"
        ),
        (
            "experts = moe(num_experts=4, top_k=2, router=sigmoid,"
            " load_balancing=aux_free, aux_loss_coef=1e-3, expert=dense)"
        ),
    ],
    # The other OLMo files differ in sizes alone and SmolLM2 is a Llama; the full-size
    # test builds them.
    "olmo/olmo_2_1b.bux": [
        "attn.num_heads = 4",
        "attn.num_kv_heads = 2",
        "ffn = dense",
    ],
    "olmo/olmo_3_7b.bux": [
        "attn.num_heads = 4",
        "attn.num_kv_heads = 2",
        "local.sliding_window_size = 4",
        "ffn = dense",
    ],
    "smollm/smollm3_3b.bux": [
        "attn.num_heads = 4",
        "attn.num_kv_heads = 2",
        "ffn = dense",
    ],
    "mhc.bux": ["attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())", "ffn = dense"],
    "jamba.bux": [
        "attn = gqa(num_heads=4, num_kv_heads=2)",
        "ssm = mamba(state_size=4)",
        "experts = moe(num_experts=4, top_k=2, expert=dense)",
    ],
    "nemotron/nemotron_3_nano_30b_a3b.bux": [
        "ssm = mamba2(state_size=8, head_dim=8, num_heads=6, num_groups=2)",
        "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8)",
        (
            "experts = moe(num_experts=4, top_k=2, num_shared_experts=1, router=sigmoid,"
            " load_balancing=aux_free, expert=mlp(intermediate_size=8, activation=relu2))"
        ),
    ],
    "nemotron/nemotron_3_ultra_550b_a55b.bux": [
        "ssm = mamba2(state_size=8, head_dim=8, num_groups=2)",
        "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8)",
        (
            "experts = moe(num_experts=4, top_k=2, num_shared_experts=1, router=sigmoid,"
            " load_balancing=aux_free, latent_size=16,"
            " expert=mlp(intermediate_size=8, activation=relu2))"
        ),
    ],
}


@pytest.mark.parametrize("path", SHRUNK, ids=lambda p: Path(p).stem)
def test_examples_shrunk_run_and_are_causal(path):
    source = (ROOT / path).read_text()
    overrides = [*SMALL.strip().splitlines(), *SHRUNK[path]]
    m = Model(compile(source, overrides=overrides), rngs=nnx.Rngs(0))
    t = jax.random.randint(jax.random.key(0), (2, 6), 0, V - 1)
    logits = m.outputs(t)
    assert all(bool(jnp.isfinite(y).all()) for y in logits)
    changed = t.at[:, 3:].set((t[:, 3:] + 1) % (V - 1))
    assert jnp.allclose(m.outputs(changed)[0][:, :3], logits[0][:, :3], atol=1e-5)
