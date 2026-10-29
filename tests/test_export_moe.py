"""Mixtures of experts, the dense MLPs' every setting, the embedding's and dyt export
to transformers computing what bux does (bux.export): the same logits, the same
greedy tokens with and without the cache, and left-padded batches as each row alone.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from jaxtyping import Int

from bux.architecture import Model
from bux.architecture.experts.moe import MoE, _limit_groups
from bux.export import check
from bux.export.hf import moe as hf_moe
from bux.lang import compile, ir

HEAD = "d_model = 64\nvocab_size = 256\n"
ATTN = "attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())\n"


def two_ffns(a: str, b: str | None = None) -> str:
    """A model of an attention block and two FFN blocks, of `a` and `b` (or `a`
    twice)."""
    return (
        HEAD
        + ATTN
        + f"a = {a}\nb = {b or a}\n"
        + "model = embed [rmsnorm attn] [rmsnorm a] [rmsnorm b] rmsnorm head"
    )


# --- Dense MLPs ----------------------------------------------------------------------

ACTIVATIONS = [
    "relu",
    "relu2",
    "gelu",
    "gelu_tanh",
    "quick_gelu",
    "swish",
    "sigmoid",
    "tanh",
    "softplus",
    "situ",
]


@pytest.mark.parametrize("act", ACTIVATIONS)
def test_every_activation_exports(act: str) -> None:
    source = two_ffns(
        f"mlp(intermediate_size=96, activation={act})",
        f"gated_mlp(intermediate_size=96, activation={act})",
    )
    check.compare(compile(source)).check()


# The clip and cap act on values about 1 at these widths: 0.5 clips most of them.
MLPS = {
    "bias": "intermediate_size=96, activation=gelu, bias=true",
    "limit": "intermediate_size=96, activation=swish, limit=0.5",
    "up_softcap": "intermediate_size=96, activation=situ, up_softcap=1.0",
    "up_offset": "intermediate_size=96, activation=quick_gelu, up_offset=1.0",
    "gpt-oss": (
        "intermediate_size=96, activation=quick_gelu, bias=true, limit=0.5,"
        " up_offset=1.0"
    ),
    "every": (
        "intermediate_size=96, activation=swish, bias=true, limit=0.7,"
        " up_softcap=0.5, up_offset=-0.25"
    ),
}


@pytest.mark.parametrize("settings", MLPS.values(), ids=MLPS.keys())
def test_mlp_settings_export(settings: str) -> None:
    source = two_ffns(f"mlp({settings})", f"gated_mlp({settings})")
    check.compare(compile(source)).check()


# --- The embedding and dyt -----------------------------------------------------------

FFN = "ffn = gated_mlp(intermediate_size=128, activation=swish)\n"
EMBEDDINGS = {
    "norm": "embed(norm=rmsnorm)",
    "scale_by_sqrt_dim": "embed(scale_by_sqrt_dim=true)",
    "every": (
        "embed(scale_by_sqrt_dim=true, max_positions=32,"
        " norm=layernorm(bias=true, eps=1e-5))"
    ),
}


@pytest.mark.parametrize("embed", EMBEDDINGS.values(), ids=EMBEDDINGS.keys())
def test_embedding_settings_export(embed: str) -> None:
    source = HEAD + ATTN + FFN + f"model = {embed} [rmsnorm attn] [rmsnorm ffn] head"
    check.compare(compile(source)).check()


def test_dyt_exports() -> None:
    source = (
        HEAD
        + ATTN
        + FFN
        + "model = embed [dyt attn] [dyt(alpha_init=0.8) ffn] dyt head(tied=true)"
    )
    check.compare(compile(source)).check()


# --- Mixtures of experts ---------------------------------------------------------------

EXPERT = "gated_mlp(intermediate_size=32, activation=swish)"


def experts(settings: str = "", expert: str = EXPERT) -> str:
    """A model of two MoE blocks of 8 experts, 2 a token, with `settings`."""
    more = f", {settings}" if settings else ""
    return two_ffns(f"moe(num_experts=8, top_k=2{more}, expert={expert})")


MOES = {
    "softmax": "",
    "sigmoid": "router=sigmoid",
    "sqrtsoftplus": "router=sqrtsoftplus",
    "router_bias": "router_bias=true",
    "router_logit_scale": "router=sigmoid, router_logit_scale=3.0",
    "not normalized": "router=sigmoid, normalize_top_k=false",
    "routed_scaling_factor": "router=sigmoid, routed_scaling_factor=2.5",
    "groups": "router=sigmoid, num_groups=4, top_k_groups=2",
    "shared": "num_shared_experts=1",
    "shared of their own": (
        "num_shared_experts=2,"
        " shared_expert=mlp(intermediate_size=24, activation=relu2, bias=true)"
    ),
    "shared_expert_gate": "num_shared_experts=1, shared_expert_gate=true",
    "shared_expert_sink": (
        "num_shared_experts=2, shared_expert_sink=true, router=sigmoid"
    ),
    "sink, softmax, unnormalized": (
        "num_shared_experts=2, shared_expert_sink=true, normalize_top_k=false,"
        " routed_scaling_factor=1.5"
    ),
    "latent": "latent_size=32",
    "latent_input_norm": "latent_size=32, latent_input_norm=rmsnorm",
    "latent_output_norm": "latent_size=32, latent_output_norm=rmsnorm",
    "latent_output_norm, full width": "latent_output_norm=layernorm",
    "expert_norm": "expert_norm=rmsnorm",
    "expert_norm, Muse's": (
        "expert_norm=rmsnorm(eps=1e-8, zero_centered=true, scale_input=true)"
    ),
    "zero experts": "num_zero_experts=4",
    "zero experts, sigmoid": (
        "num_zero_experts=4, router=sigmoid, routed_scaling_factor=2.5, latent_size=32"
    ),
    "training settings": (
        "aux_loss_coef=0.1, aux_loss_scope=sequence, bias_update_rate=0.5,"
        " random_logits=true"
    ),
}


@pytest.mark.parametrize("settings", MOES.values(), ids=MOES.keys())
def test_moe_settings_export(settings: str) -> None:
    check.compare(compile(experts(settings)), experts="eager").check()


EXPERTS = {
    "mlp": "mlp(intermediate_size=32, activation=relu2)",
    "gated, biased": "gated_mlp(intermediate_size=32, activation=gelu, bias=true)",
    "mlp, biased": "mlp(intermediate_size=32, activation=relu2, bias=true)",
    "limit": "gated_mlp(intermediate_size=32, activation=swish, limit=0.5)",
    "up_softcap": "gated_mlp(intermediate_size=32, activation=situ, up_softcap=1.0)",
    "up_offset": "gated_mlp(intermediate_size=32, activation=quick_gelu, up_offset=1.0)",
    "mlp, limit and softcap": (
        "mlp(intermediate_size=32, activation=gelu, limit=0.5, up_softcap=0.4)"
    ),
}


@pytest.mark.parametrize("expert", EXPERTS.values(), ids=EXPERTS.keys())
def test_experts_of_every_kind_export(expert: str) -> None:
    source = experts("expert_norm=rmsnorm", expert=expert)
    check.compare(compile(source), experts="eager").check()


# Many settings at once, as the zoo combines them.
COMBINED = {
    "deepseek": (
        "num_shared_experts=1, router=sigmoid, routed_scaling_factor=2.5, num_groups=4,"
        " top_k_groups=2, load_balancing=aux_free"
    ),
    "gpt-oss": "router_bias=true",
    "kimi": (
        "num_shared_experts=2, router=sigmoid, load_balancing=aux_free, latent_size=32,"
        " latent_output_norm=rmsnorm"
    ),
    "muse": (
        "router=sigmoid, router_logit_scale=0.5, latent_size=32,"
        " latent_input_norm=rmsnorm(zero_centered=true), load_balancing=aux_free,"
        " expert_norm=rmsnorm(zero_centered=true, scale_input=true)"
    ),
    # LongCat-Flash's zero experts, on the timeline's experts.
    "zero experts": (
        "num_zero_experts=4, num_shared_experts=1, router=sigmoid,"
        " routed_scaling_factor=2.5, load_balancing=aux_free, aux_loss_coef=1e-4,"
        " aux_loss_scope=sequence"
    ),
    "inkling": (
        "num_shared_experts=2, shared_expert_sink=true, router=sigmoid,"
        " routed_scaling_factor=8, load_balancing=aux_free"
    ),
}
COMBINED_EXPERT = {
    "gpt-oss": (
        "gated_mlp(intermediate_size=32, activation=quick_gelu, bias=true, limit=0.5,"
        " up_offset=1.0)"
    ),
    "kimi": "gated_mlp(intermediate_size=32, activation=situ, up_softcap=1.0)",
}


def with_router_bias(arch: ir.Architecture, scale: float = 0.3) -> Model:
    """A model of `arch` of perturbed weights, whose MoEs' aux-free biases, which
    pick their experts but do not weight them, are drawn too (`perturbed` moves
    parameters alone)."""
    model = check.perturbed(Model(arch, rngs=nnx.Rngs(0)), 1)
    moes = [m for _, m in nnx.iter_modules(model) if isinstance(m, MoE)]
    for i, m in enumerate(moes):
        if m.router_bias is not None:
            key = jax.random.key(100 + i)
            m.router_bias[...] = scale * jax.random.normal(key, m.router_bias.shape)
    return model


def _routes(model: Model, arch: ir.Architecture) -> list[Int[np.ndarray, "b t k"]]:
    """Each MoE's top-k picks on random tokens' embeddings, a probe of the router."""
    assert arch.d_model is not None
    x = jax.random.normal(jax.random.key(7), (2, 12, arch.d_model))
    moes = [m for _, m in nnx.iter_modules(model) if isinstance(m, MoE)]
    return [np.asarray(m._route(x)[0]) for m in moes]


@pytest.mark.parametrize("name", COMBINED, ids=COMBINED.keys())
def test_moe_combinations_export(name: str) -> None:
    arch = compile(experts(COMBINED[name], COMBINED_EXPERT.get(name, EXPERT)))
    check.compare(arch, with_router_bias(arch), experts="eager").check()


# The load balancings that move a bias, which picks the experts (Trinity's smebu too).
BIASES = ["aux_free", "aux_free, center_bias=true", "smebu, bias_softclamp=3.0"]


@pytest.mark.parametrize("balancing", BIASES)
def test_the_load_balancing_bias_picks_the_experts(balancing: str) -> None:
    """The bias changes which experts bux picks, and the export picks the same."""
    arch = compile(experts(f"router=sigmoid, load_balancing={balancing}"))
    model = with_router_bias(arch)
    unbiased = check.perturbed(Model(arch, rngs=nnx.Rngs(0)), 1)
    changed = [
        not np.array_equal(a, b)
        for a, b in zip(_routes(model, arch), _routes(unbiased, arch), strict=True)
    ]
    assert all(changed)
    check.compare(arch, model, experts="eager").check()


def test_zero_experts_are_picked_beside_the_routed_ones() -> None:
    """The combination's routers pick zero experts and routed ones, which the
    export runs alike."""
    arch = compile(experts(COMBINED["zero experts"]))
    picks = np.concatenate(_routes(with_router_bias(arch), arch))
    assert (picks >= 8).any() and (picks < 8).any()


def test_equal_scores_pick_the_experts_bux_picks() -> None:
    """A router of zeros scores every expert alike: bux picks the first ones, groups
    first, and so does the export."""
    arch = compile(experts("router=sigmoid, num_groups=4, top_k_groups=2"))
    model = check.perturbed(Model(arch, rngs=nnx.Rngs(0)), 1)
    for _, m in nnx.iter_modules(model):
        if isinstance(m, MoE):
            m.router.kernel[...] = jnp.zeros_like(m.router.kernel[...])
    assert all((r == [0, 1]).all() for r in _routes(model, arch))
    check.compare(arch, model, experts="eager").check()


def test_top_k_breaks_ties_as_bux() -> None:
    """Equal scores, of experts and of groups, pick as jax.lax.top_k does: the lower
    index first."""
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 3, (64, 16)).astype(np.float32)  # many ties
    want = np.asarray(jax.lax.top_k(jnp.asarray(scores), 5)[1])
    got = hf_moe.top_k(torch.from_numpy(scores), 5).numpy()
    np.testing.assert_array_equal(got, want)
    limited = np.asarray(_limit_groups(jnp.asarray(scores)[None], 4, 2))[0]
    ours = hf_moe.limit_groups(torch.from_numpy(scores), 4, 2).numpy()
    np.testing.assert_array_equal(ours, limited)


IMPLEMENTATIONS = ["grouped_mm", "batched_mm"]


@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
@pytest.mark.parametrize("name", ["deepseek", "gpt-oss", "muse", "zero experts"])
def test_transformers_kernels_run_the_experts(impl: str, name: str) -> None:
    arch = compile(experts(COMBINED[name], COMBINED_EXPERT.get(name, EXPERT)))
    check.compare(arch, with_router_bias(arch), experts=impl).check()


@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_transformers_kernels_run_ungated_experts(impl: str) -> None:
    source = experts(
        "num_shared_experts=1, expert_norm=rmsnorm",
        "mlp(intermediate_size=32, activation=relu2, bias=true, limit=0.5)",
    )
    check.compare(compile(source), experts=impl).check()


# --- The zoo -------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent / "examples" / "architectures"
SMALL = [
    "d_model = 32",
    "vocab_size = 64",
    "dense = gated_mlp(intermediate_size=8, activation=swish)",
]
GQA = "gqa(num_heads=4, num_kv_heads=2, head_dim=8"
MLA = (
    "mla(num_heads=2, q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4,"
    " qk_rope_head_dim=4, v_head_dim=4"
)
YARN = "scaling=yarn(factor=40, original_max_len=4096, attention_factor_on=softmax)"
# Each zoo file shrunk, as tests/test_examples.py shrinks it: its experts as the file
# has them but for their numbers and sizes, its layers not exported yet (Mamba-2,
# KDA) plain attention, and its deepest repeats cut short.
ZOO = {
    "qwen/qwen3_30b_a3b.bux": [
        f"attn = {GQA}, qk_norm=norm, rope=rope(base=1000000))",
        "experts = moe(num_experts=8, top_k=2, expert=dense)",
        "model = embed block^3 norm head",
    ],
    "deepseek_v3.bux": [
        f"attn = {MLA}, rope=rope(layout=interleaved, {YARN}))",
        (
            "experts = moe(num_experts=8, top_k=2, num_shared_experts=1, router=sigmoid,"
            " routed_scaling_factor=2.5, num_groups=4, top_k_groups=2,"
            " load_balancing=aux_free, expert=dense)"
        ),
        "model = embed layer(dense) layer(experts)^2 norm head",
    ],
    "kimi/kimi_k2.5.bux": [
        f"attn = {MLA}, rope=rope(base=50000, layout=interleaved, {YARN}))",
        (
            "experts = moe(num_experts=8, top_k=2, num_shared_experts=1, router=sigmoid,"
            " routed_scaling_factor=2.827, load_balancing=aux_free, expert=dense)"
        ),
        "model = embed layer(dense) layer(experts)^2 norm head",
    ],
    "kimi/kimi_k3.bux": [
        f"kda = {GQA})",
        f"mla = {MLA}, gated=true, rope=none)",
        (
            "experts = moe(num_experts=8, top_k=2, num_shared_experts=2, router=sigmoid,"
            " load_balancing=aux_free, latent_size=16, latent_output_norm=norm,"
            " expert=gated_mlp(intermediate_size=8, activation=situ, up_softcap=25))"
        ),
        'model = embed "FKKA" "KKKA" "A" norm head',
    ],
    "gpt_oss_120b.bux": [
        (
            f"attn = {GQA}, bias=true, sinks=true,"
            " rope=rope(base=150000, scaling=yarn(factor=32, original_max_len=4096)))"
        ),
        "local = attn(sliding_window_size=4)",
        (
            "experts = moe(num_experts=8, top_k=2, router_bias=true,"
            " expert=gated_mlp(intermediate_size=8, activation=quick_gelu, bias=true,"
            " limit=7.0, up_offset=1.0))"
        ),
        'model = embed "LG"^2 norm head',
    ],
    "muse/muse_spark_open_816b_a42b.bux": [
        (
            f"attn = {GQA}, qk_norm=bare, scale=5.4730064863875 * 8^-0.5, gated=true,"
            " output_norm=bare)"
        ),
        "local = attn(sliding_window_size=4, rope=rope(base=500000))",
        (
            "experts = moe(num_experts=8, top_k=2, load_balancing=aux_free,"
            " router=sigmoid, router_logit_scale=logit_scale, latent_size=16,"
            " latent_input_norm=norm, expert_norm=rescale,"
            " expert=gated_mlp(intermediate_size=8, activation=swish))"
        ),
        'model = embed bare "LG" "LLLG" (rmsnorm(eps=1e-5) * logit_scale) head(softcap=20)',
    ],
    "inkling/inkling.bux": [
        (
            f"attn = {GQA}, qk_norm=norm, scale=1 / 8, kv_conv_size=4,"
            " relative_bias=relbias(rank=2, extent=8), log_scale=0.1, log_scale_floor=4)"
        ),
        "local = attn(sliding_window_size=4, relative_bias.extent=4, log_scale=none)",
        (
            "experts = moe(num_experts=8, top_k=2, num_shared_experts=2,"
            " shared_expert_sink=true, router=sigmoid, routed_scaling_factor=8,"
            " load_balancing=aux_free, expert=dense)"
        ),
        (
            'model = embed(norm=norm) "DD" "LLLG" "LG" norm @h logits mtp(local)'
            " mtp(attn)"
        ),
    ],
    "nemotron/nemotron_3_nano_30b_a3b.bux": [
        f"ssm = {GQA})",
        f"attn = {GQA})",
        (
            "experts = moe(num_experts=8, top_k=2, num_shared_experts=1, router=sigmoid,"
            " routed_scaling_factor=2.5, load_balancing=aux_free,"
            " expert=mlp(intermediate_size=8, activation=relu2),"
            " shared_expert=mlp(intermediate_size=16, activation=relu2))"
        ),
        'model = embed "MEMEMA" "EMEMEMA" norm head',
    ],
    "nemotron/nemotron_3_ultra_550b_a55b.bux": [
        f"ssm = {GQA})",
        f"attn = {GQA})",
        (
            "experts = moe(num_experts=8, top_k=2, num_shared_experts=1, router=sigmoid,"
            " routed_scaling_factor=5.0, load_balancing=aux_free, latent_size=16,"
            " expert=mlp(intermediate_size=8, activation=relu2),"
            " shared_expert=mlp(intermediate_size=16, activation=relu2))"
        ),
        'model = embed "MEMEMEMAE" "MEMEMAE" norm @h out mtp^2',
    ],
}


@pytest.mark.parametrize("path", ZOO, ids=lambda p: Path(p).stem)
def test_zoo_models_with_experts_export(path: str) -> None:
    arch = compile((ROOT / path).read_text(), overrides=[*SMALL, *ZOO[path]])
    check.compare(arch, with_router_bias(arch), experts="eager").check()
