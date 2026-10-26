"""bux's vLLM plugin (bux_vllm) builds what bux.export names: it places every
layer of every example architecture where bux.export.plan does, imports nothing of
bux, and reads an export's config.json as vLLM needs it. Whether vLLM then computes
what bux does takes a GPU: scripts/vllm_parity.py checks it."""

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import jax
import numpy as np
import pytest
import torch
from flax import nnx

from bux import config as cfg
from bux.architecture.attention import rope as bux_rope
from bux.architecture.experts.moe import MoE
from bux.export import plan as core
from bux.lang import compile, compile_file
from bux_vllm import common, config, plan, routing

ROOT = Path(__file__).parent.parent
EXAMPLES = sorted((ROOT / "examples" / "architectures").rglob("*.bux"))
FILES = sorted(Path(plan.__file__).parent.glob("*.py"))


def script(name: str) -> ModuleType:
    """Import scripts/`name`.py, which is no package's."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vllm_parity = script("vllm_parity")


def calls(arch) -> list[tuple[str, bool]]:
    """The plugin's calls of `arch`'s layers: each module's path, and whether it is
    the module's first call."""
    graph = arch.model_dump(mode="json")["graph"]
    return [(c.path, c.first) for c in plan.plan(graph).calls]


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_layers_are_where_bux_export_puts_them(path: Path) -> None:
    arch = compile_file(str(path))
    assert calls(arch) == [(c.path, c.first) for c in core.plan(arch).calls]


@pytest.mark.parametrize("name", vllm_parity.CASES)
def test_parity_cases_are_where_bux_export_puts_them(name: str) -> None:
    arch = compile(vllm_parity.CASES[name])
    assert calls(arch) == [(c.path, c.first) for c in core.plan(arch).calls]


def test_layers_after_the_first_head_are_not_live() -> None:
    source = vllm_parity.LLAMA.replace("block =", "shared block =")
    arch = compile(
        source + "shared out = head\nmodel = embed block @h out $h! block out"
    )
    live = plan.live(plan.plan(arch.model_dump(mode="json")["graph"]).calls)
    assert [c.path for c in live][-1] == "model.shared.out"
    assert sum(c.layer["config"]["type"] == "gqa" for c in live) == 1


def test_the_config_gives_vllm_its_heads_and_layer_kinds() -> None:
    arch = compile(vllm_parity.CASES["gdn"])
    bux = json.loads(arch.model_dump_json())
    c = config.BuxConfig(bux=bux, vocab_size=256, hidden_size=128)
    assert c.layer_types == ["linear_attention"] * 2 + ["full_attention"]
    assert (c.num_attention_heads, c.num_key_value_heads, c.head_dim) == (4, 2, 32)
    assert c.num_hidden_layers == 3


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_plugin_files_import_nothing_of_bux(path: Path) -> None:
    """The plugin installs alone, as bux_vllm, where vLLM runs: Python 3.10 on."""
    tree = ast.parse(path.read_text(), feature_version=(3, 10))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            assert not (node.module or "").startswith("bux"), node.module
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("bux") for a in node.names)


def test_parity_cases_export(tmp_path: Path) -> None:
    """Every case of scripts/vllm_parity.py exports, with its prompts."""
    for name in vllm_parity.CASES:
        vllm_parity.export(name, tmp_path / name, "float32")
        assert (tmp_path / name / "model.safetensors").exists()
        meta = json.loads((tmp_path / name / "parity.json").read_text())
        assert [len(p) for p in meta["prompts"]] == list(vllm_parity.LENGTHS)


ROUTERS = {
    "softmax": "moe(num_experts=8, top_k=2, expert=e)",
    "unnormalized": "moe(num_experts=8, top_k=3, normalize_top_k=false, expert=e)",
    "sigmoid groups": "moe(num_experts=8, top_k=2, router=sigmoid, num_groups=4,"
    " top_k_groups=2, routed_scaling_factor=2.5, load_balancing=aux_free, expert=e)",
    "sqrtsoftplus": "moe(num_experts=8, top_k=2, router=sqrtsoftplus,"
    " router_logit_scale=2, router_bias=true, expert=e)",
}


@pytest.mark.parametrize("source", ROUTERS.values(), ids=ROUTERS.keys())
def test_the_router_picks_and_weighs_as_bux(source: str) -> None:
    """The plugin's router, which vLLM's fused MoE takes the picks from, is bux's."""
    arch = compile(
        "d_model = 64\ne = gated_mlp(intermediate_size=16, activation=swish)\n"
        f"model = {source}"
    )
    c = next(arch.layers()).config
    assert isinstance(c, cfg.MoEConfig)
    moe = MoE(c, 64, rngs=nnx.Rngs(0))
    router = routing.Router(c.model_dump(mode="json"))
    if moe.router_bias is not None and router.bias is not None:
        bias = 0.1 * jax.random.normal(jax.random.key(1), (c.num_experts,))
        moe.router_bias[...] = bias
        router.bias.data = torch.from_numpy(np.array(bias))
    x = jax.random.normal(jax.random.key(2), (1, 32, 64))
    idx, weights, _ = moe._route(x)
    logits = torch.from_numpy(np.array(moe._logits(x))[0])
    got_w, got_idx = router(None, logits, c.top_k, c.normalize_top_k)
    assert got_idx.tolist() == np.asarray(idx)[0].tolist()
    np.testing.assert_allclose(got_w.numpy(), np.asarray(weights)[0], rtol=1e-6)


ROPES = {
    "plain": "rope()",
    "interleaved half": "rope(fraction=0.5, layout=interleaved)",
    "pi": "rope(scaling=pi(factor=4))",
    "llama3": "rope(base=500000, scaling=llama3(factor=8, original_max_len=16))",
    "yarn": "rope(scaling=yarn(factor=4, original_max_len=16))",
    "yarn softmax": "rope(scaling=yarn(factor=4, original_max_len=16,"
    " attention_factor_on=softmax))",
}


@pytest.mark.parametrize("rope", ROPES.values(), ids=ROPES.keys())
def test_rope_rotates_as_bux(rope: str) -> None:
    arch = compile(
        f"d_model = 64\nmodel = gqa(num_heads=2, num_kv_heads=2, rope={rope})"
    )
    c = next(arch.layers()).config
    assert isinstance(c, cfg.GQAConfig) and c.rope is not None
    x = jax.random.normal(jax.random.key(0), (1, 40, 2, 32))
    positions = jax.numpy.arange(40)[None] * 3
    want = bux_rope.rotate(x, c.rope, positions)[0]
    rope_json = c.rope.model_dump(mode="json")
    at = torch.from_numpy(np.array(positions[0]))
    got = common.Rope(rope_json, 32)(torch.from_numpy(np.array(x)[0]), at)
    np.testing.assert_allclose(got.numpy(), np.asarray(want), atol=2e-5)
    assert common.logit_gain(rope_json) == bux_rope.logit_gain(c.rope)


@pytest.mark.parametrize("gated", [True, False], ids=["gated", "plain"])
def test_experts_load_alike_from_bux_and_transformers(gated: bool) -> None:
    """bux's stacked experts, and transformers' (the gate first), give vLLM the same
    shards."""
    up = torch.randn(3, 8, 5, 2 if gated else 1)  # E, d, f, p
    down = torch.randn(3, 5, 8)
    parts = [up[..., i] for i in reversed(range(up.shape[-1]))]  # gate, then up
    name = "experts.gate_up_proj" if gated else "experts.up_proj"
    pairs = [
        (("up", up), (name, torch.cat(parts, -1).transpose(1, 2))),
        (("down", down), ("experts.down_proj", down.transpose(1, 2))),
    ]
    for mine, theirs in pairs:
        a = routing.expert_shards(*mine, gated)
        b = routing.expert_shards(*theirs, gated)
        assert a is not None and b is not None
        assert [s for s, _ in a] == [s for s, _ in b]
        for (_, x), (_, y) in zip(a, b, strict=True):
            torch.testing.assert_close(x, y)
