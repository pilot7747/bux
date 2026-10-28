"""bux's vLLM plugin (bux_vllm) builds what bux.export names: it places every
layer and residual algebra's branch of every example architecture where
bux.export.plan does, imports nothing of bux, reads an export's config.json as vLLM
needs it, and its plain PyTorch pieces (the router, RoPE, the residual algebras)
compute what the export does. Whether vLLM then computes what bux does takes a GPU:
scripts/vllm_parity.py checks it."""

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from flax import nnx

from bux import config as cfg
from bux.architecture.attention import gqa as bux_gqa
from bux.architecture.attention import rope as bux_rope
from bux.architecture.attention.gqa import Attention as BuxAttention
from bux.architecture.experts.moe import MoE
from bux.architecture.layers.conv import CausalConv
from bux.export import plan as core
from bux.export.hf import linear_attention as hf_linear
from bux.export.hf import residual as hf_residual
from bux.lang import compile, compile_file, compile_runs, ir
from bux_vllm import common, config, kv, linear, plan, residual, routing

ROOT = Path(__file__).parent.parent
EXAMPLES = sorted((ROOT / "examples" / "architectures").rglob("*.bux"))
TIMELINE = ROOT / "examples" / "experiments" / "timeline"
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
    tree = plan.plan(arch.model_dump(mode="json"), core.RESERVED)
    return [(c.path, c.first) for c in tree.calls]


def assert_placed_as_bux_export(arch) -> None:
    """Assert that the plugin places `arch`'s layers and residual algebra's
    branches, in the order they run, where bux.export.plan does."""
    tree = plan.plan(arch.model_dump(mode="json"), core.RESERVED)
    theirs = core.plan(arch)
    assert calls(arch) == [(c.path, c.first) for c in theirs.calls]
    assert [(b.path, b.first) for b in tree.branches] == [
        (b.path, b.first) for b in theirs.branches
    ]
    starts = [sum(c.trail < b.trail for c in theirs.calls) for b in theirs.branches]
    assert [b.calls for b in tree.branches] == starts


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_layers_are_where_bux_export_puts_them(path: Path) -> None:
    assert_placed_as_bux_export(compile_file(str(path)))


@pytest.mark.parametrize("name", vllm_parity.CASES)
def test_parity_cases_are_where_bux_export_puts_them(name: str) -> None:
    assert_placed_as_bux_export(compile(vllm_parity.CASES[name]))


def test_the_timeline_residual_arms_are_where_bux_export_puts_them() -> None:
    """Every residual algebra of the timeline's ablation, at its smallest rung."""
    path = TIMELINE / "residuals.bux"
    arms: dict = {}
    for run in compile_runs(path.read_text(), filename=str(path)):
        arms.setdefault(run.arch.residual and run.arch.residual.type, run.arch)
    assert set(arms) == {None, "hc", "mhc", "attnres", "gr", "vpgate"}
    for arch in arms.values():
        assert_placed_as_bux_export(arch)


def test_reserved_labels_are_named_as_bux_export_names_them() -> None:
    """The names bux.export reserves, which an export's config lists, rename the
    layers of the plugin's plan as they rename bux.export's."""
    arch = compile(
        vllm_parity.LLAMA.replace("attn", "forward").replace("ffn", "scope")
        + "model = embed block^2 rmsnorm head"
    )
    assert calls(arch) == [(c.path, c.first) for c in core.plan(arch).calls]
    assert ("model.layers.0.forward_1", True) in calls(arch)


def test_layers_after_the_first_head_are_not_live() -> None:
    source = vllm_parity.LLAMA.replace("block =", "shared block =")
    arch = compile(
        source + "shared out = head\nmodel = embed block @h out $h! block out"
    )
    live = plan.live(plan.plan(arch.model_dump(mode="json")).calls)
    assert [c.path for c in live][-1] == "model.shared.out"
    assert sum(c.layer["config"]["type"] == "gqa" for c in live) == 1


def test_a_sparse_layer_keeps_its_indexer_keys_beside_its_kv_cache() -> None:
    arch = compile(vllm_parity.CASES["dsa"])
    layer = next(n for n in arch.layers() if n.config.type == "gqa")
    assert config.states(layer.config.model_dump(mode="json")) == [
        "full_attention",
        "index",
    ]
    c = config.BuxConfig(bux=json.loads(arch.model_dump_json()))
    assert c.layer_types == ["full_attention"] * 2


def test_a_conv_on_keys_and_values_keeps_a_state_beside_the_kv_cache() -> None:
    arch = compile(vllm_parity.CASES["kv-conv"])
    c = config.BuxConfig(bux=json.loads(arch.model_dump_json()))
    assert c.layer_types == ["full_attention", "conv"] * 3


@pytest.mark.parametrize("name", ["gdn", "kda", "kda-kimi"])
def test_the_config_gives_vllm_its_heads_and_layer_kinds(name: str) -> None:
    """Linear attention beside softmax attention is a hybrid for vLLM, as Qwen3-Next
    and Kimi Linear are."""
    arch = compile(vllm_parity.CASES[name])
    bux = json.loads(arch.model_dump_json())
    c = config.BuxConfig(bux=bux, vocab_size=256, hidden_size=128)
    assert c.layer_types == ["linear_attention"] * 2 + ["full_attention"]
    assert (c.num_attention_heads, c.num_key_value_heads, c.head_dim) == (4, 2, 32)
    assert c.num_hidden_layers == 3


def bux_config(source: str) -> config.BuxConfig:
    """The plugin's config of the model `source`, as vLLM reads its export."""
    bux = json.loads(compile(source).model_dump_json())
    return config.BuxConfig(bux=bux, vocab_size=256, hidden_size=128)


def test_the_config_gives_vllm_the_latent_of_mla() -> None:
    """vLLM reads an MLA's dims from the config, DeepSeek's names, and runs it as
    MLA where they are set (bux_vllm.arch)."""
    c = bux_config(vllm_parity.CASES["mla-moe"])
    assert c.layer_types == ["full_attention"] * 2
    dims = (c.q_lora_rank, c.kv_lora_rank, c.qk_nope_head_dim, c.qk_rope_head_dim)
    assert (*dims, c.v_head_dim) == (None, 256, 64, 32, 64)
    assert (c.num_attention_heads, c.num_key_value_heads, c.head_dim) == (4, 4, 96)
    assert config.latent(c)
    assert config.latent(bux_config(vllm_parity.CASES["mla-deepseek"]))
    assert not config.latent(bux_config(vllm_parity.CASES["llama"]))


MIXED = {
    "gqa beside mla": "gqa(num_heads=4, num_kv_heads=2)",
    "other dims": "mla(num_heads=2, kv_lora_rank=128, rope=rope())",
    "other heads": "mla(num_heads=2, kv_lora_rank=256, qk_nope_head_dim=64,"
    " qk_rope_head_dim=32, v_head_dim=64)",
}


@pytest.mark.parametrize("other", MIXED.values(), ids=MIXED.keys())
def test_a_config_of_mla_and_other_attention_raises(other: str) -> None:
    """vLLM runs all of a model's attention as MLA of one set of dims, or none."""
    source = (
        vllm_parity.HEAD
        + vllm_parity.LATENT
        + f"other = {other}\n"
        + "model = embed [rmsnorm attn] [rmsnorm other] rmsnorm head"
    )
    with pytest.raises(NotImplementedError, match="mla"):
        bux_config(source)


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


HEAD = "d_model = 16\nvocab_size = 64\n"
RESIDUALS = {
    "hc": "hc(expansion=2)",
    "hc static": "hc(expansion=3, dynamic=false)",
    "hc single pass": "hc(expansion=2, single_pass=true, norm_eps=1e-3)",
    "attnres": "attnres()",
    "attnres blocks": "attnres(block_size=2)",
    "attnres block under way": "attnres(block_size=4, norm_eps=1e-5)",
    "mhc": "mhc(expansion=2)",
    "mhc static": "mhc(expansion=4, dynamic=false, sinkhorn_iters=3)",
    "mhc single pass": "mhc(expansion=4, single_pass=true, norm_eps=1e-20)",
    "gr": "gr(expansion=2, rank=8)",
    "gr eps": "gr(expansion=3, rank=4, norm_eps=1e-3)",
    "vpgate": "vpgate(temperature=0.5)",
    "vpgate clamped": "vpgate(eps=0.6)",
}


def algebra(setting: str) -> dict:
    """The residual algebra `setting`, as config.json holds it."""
    arch = compile(f"{HEAD}residual = {setting}\nmodel = embed [rmsnorm] head")
    assert arch.residual is not None
    return arch.residual.model_dump(mode="json")


def stand_in(**attributes: Any) -> Any:
    """A forward pass's context, or a layer's site, of `attributes` alone."""
    return SimpleNamespace(**attributes)


def twins(theirs: torch.nn.Module, mine: common.Layer) -> None:
    """Give the export's module `theirs` random parameters, and `mine` the same by
    their names, as the plugin loads the checkpoint."""
    for name, param in theirs.named_parameters():
        param.data = torch.randn_like(param) / 2
        mine.load(name, param.data)
    assert {n for n, _ in mine.named_parameters()} == {
        n for n, _ in theirs.named_parameters()
    }


def through(branches: list, x: Any, ctx: Any, body: torch.Tensor) -> tuple[list, Any]:
    """Run the stream `x` through `branches`, each body ``tanh(h body)``; return what
    each read, and the stream after them."""
    reads = []
    for branch in branches:
        h, kept = branch.read(x, ctx)
        reads.append(h)
        x = branch.write_back(kept, torch.tanh(h @ body))
    return reads, x


@pytest.mark.parametrize("setting", RESIDUALS.values(), ids=RESIDUALS.keys())
def test_residual_branches_compute_what_the_export_does(setting: str) -> None:
    """Six branches of the algebra, then a merge, of one row of tokens: the
    plugin's, on the tokens side by side, read what the export's do."""
    c, d = algebra(setting), 16
    torch.manual_seed(0)
    theirs = [hf_residual.ALGEBRAS[c["type"]](c, d) for _ in range(6)]
    mine = [residual.ALGEBRAS[c["type"]](c, d) for _ in range(6)]
    for a, b in zip(theirs, mine, strict=True):
        twins(a, b)
    tokens, body = torch.randn(5, d), torch.randn(d, d) / d**0.5
    copies = hf_residual.widening(c)
    x = tokens[None] if copies is None else hf_residual.widen(tokens[None], copies)
    ctx, ctx_mine = (
        SimpleNamespace(pending_read=None),
        SimpleNamespace(pending_read=None),
    )
    reads, x = through(theirs, x, ctx, body)
    got, y = through(mine, residual.widen(c, tokens), ctx_mine, body)
    for g, h in zip(got, reads, strict=True):
        torch.testing.assert_close(g, h[0])
    if copies is None:  # one stream, which nothing merges
        torch.testing.assert_close(y, x[0])
        return
    config = stand_in(bux={"residual": c, "d_model": d}, hidden_size=d)
    merge = hf_residual.Merge({}, "streams", stand_in(config=config))
    merge_mine = residual.Merge({}, "streams", stand_in(config=config))
    twins(merge, merge_mine)
    torch.testing.assert_close(merge_mine(y, ctx_mine), merge(x, ctx)[0])


def test_attnres_states_sum_and_scale_as_the_exports() -> None:
    """attnres's states, summed and scaled as a graph may, read as the export's
    stacked ones do."""
    c, d = algebra("attnres(block_size=2)"), 16
    torch.manual_seed(0)
    theirs, mine = hf_residual.AttentionResidual(c, d), residual.AttentionResidual(c, d)
    twins(theirs, mine)
    rows = torch.randn(2, 5, 4, d)  # 2 states of 5 tokens: the embedding, 3 outputs
    states = []
    for row in rows:
        state = residual.Sources((row[:, 0],))
        for i in (1, 2, 3):
            state = state.add(row[:, i], c["block_size"])
        states.append(state)
    ctx = stand_in(pending_read=None)
    want = theirs.read((rows[0] * 0.5 + -rows[1])[None], ctx)[0][0]
    torch.testing.assert_close(mine.read(states[0] * 0.5 + -states[1], ctx)[0], want)


LINEAR = {
    "kda": "kda(num_heads=2, num_v_heads=4, head_dim=16)",
    "kda low rank": "kda(num_heads=2, head_dim=16, gate_rank=4, output_gate_rank=4)",
    "gdn": "gdn(num_heads=2, num_v_heads=4, head_dim=16, v_head_dim=8)",
}


#: The export's linear-attention layers, by type.
EXPORTED = {
    kind: cls
    for cls in vars(hf_linear).values()
    if isinstance(cls, type)
    for kind in getattr(cls, "bux_types", ())
}


@pytest.mark.parametrize("mixer", LINEAR.values(), ids=LINEAR.keys())
def test_linear_attention_convolves_the_exports_channels(mixer: str) -> None:
    """The plugin's q, k and v, which its conv runs over, are the export's."""
    layer = next(compile(f"d_model = 64\nmodel = {mixer}").layers())
    c = layer.config.model_dump(mode="json")
    m = EXPORTED[c["type"]](c, 64)
    sizes = [m.q.weight.shape[0], m.k.weight.shape[0], m.v.weight.shape[0]]
    assert linear.conv_channels(c) == sizes
    assert m.conv is not None and m.conv.weight.shape == (c["conv_size"], sum(sizes))


def test_a_rank_convolves_its_share_of_q_k_and_v() -> None:
    """Under tensor parallelism a rank's conv taps are its q, k and v channels in
    turn, as its share of the merged input projection gives them."""
    sizes = [8, 8, 16]
    taps = torch.randn(4, sum(sizes))
    whole = linear.conv_shard(taps, sizes, 0, 1)
    torch.testing.assert_close(whole, linear.conv_weight(taps))
    q, k, v = whole.split(sizes)
    for rank in range(2):
        mine = [x.chunk(2)[rank] for x in (q, k, v)]
        torch.testing.assert_close(
            linear.conv_shard(taps, sizes, rank, 2), torch.cat(mine)
        )


def test_q_and_k_heads_serve_their_groups_of_value_heads() -> None:
    """Each q/k head, repeated for the kernels, serves the value heads next to it,
    as the export shares it out."""
    x = torch.randn(5, 2 * 3)  # T, heads * d
    got = linear.to_value_heads(x, 2, 2)
    assert got.shape == (1, 5, 4, 3)
    for v_head in range(4):
        torch.testing.assert_close(
            got[0, :, v_head], x.reshape(5, 2, 3)[:, v_head // 2]
        )


@pytest.mark.parametrize("kind", ["fixed", "learned", "convex"])
def test_the_value_residual_mixes_as_bux(kind: str) -> None:
    c = cfg.GQAConfig.model_validate(
        {"num_heads": 4, "num_kv_heads": 2, "value_residual": kind}
    )
    port = ir.Port(name="v", tap="v1", mode="read")
    attn = BuxAttention(c, 64, (port,), rngs=nnx.Rngs(0))
    w = None
    if attn.v_mix is not None:
        attn.v_mix[...] = jax.random.uniform(jax.random.key(1), attn.v_mix[...].shape)
        w = torch.from_numpy(np.array(attn.v_mix[...]))
    v, v1 = jax.random.normal(jax.random.key(2), (2, 1, 8, 2, 16))
    want = attn._mix_values(v, v1)[0].reshape(8, 32)
    flat = [torch.from_numpy(np.array(a)[0].reshape(8, 32)) for a in (v, v1)]
    got = kv.mix_values(kind, w, *flat)
    np.testing.assert_allclose(got.numpy(), np.asarray(want), rtol=1e-6)


def test_ports_are_read_and_written_as_bux_lists_them() -> None:
    ports = [["v", "v1", "pop"], ["v", "v2", "write"], ["v", "v3", "write"]]
    assert kv.reads(ports, "v") == ("v1", True)
    assert kv.reads([["v", "v1", "write"]], "v") is None
    assert kv.writes(ports, "v") == ["v2", "v3"]
    kv.check_ports(ports)
    with pytest.raises(NotImplementedError, match="kv port"):
        kv.check_ports([["kv", "kv", "write"]])


@pytest.mark.parametrize(
    ("kv_heads", "tp", "want"),
    [(8, 2, [(0, 4), (4, 4)]), (2, 4, [(0, 1), (0, 1), (1, 1), (1, 1)])],
)
def test_each_rank_keeps_its_kv_heads(
    kv_heads: int, tp: int, want: list[tuple[int, int]]
) -> None:
    assert [kv.kv_heads(kv_heads, tp, rank) for rank in range(tp)] == want


def test_the_kv_conv_taps_go_where_vllm_convolves_them() -> None:
    """bux's taps on 4 kv heads of 8, cut to rank 1 of 2's heads and laid out as
    vLLM's conv weight, (channels, taps), convolve as bux does: vLLM's kernels take
    the last tap on the token itself, as F.conv1d over the left-padded tokens."""
    conv = CausalConv(32, 3, False, nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (1, 10, 32))
    want = np.asarray(conv(x))[0, :, 16:]
    first, n = kv.kv_heads(4, 2, 1)
    w = kv.conv_rows(torch.from_numpy(np.array(conv.weight[...])), 8, first, n)
    xs = torch.from_numpy(np.array(x)[0, :, 16:]).T
    got = F.conv1d(F.pad(xs, (2, 0))[None], w[:, None], groups=16)[0].T
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-5, atol=1e-6)


def test_queries_scale_by_their_position_as_bux_scales_them() -> None:
    c = {"power_scale": 0.3, "power_scale_length": 8}
    c |= {"log_scale": 0.5, "log_scale_floor": 4}
    at = jax.numpy.arange(40)
    power = bux_gqa._power_scale(at, 0.3, 8)[:, 0, 0]
    length = bux_gqa._length_scale(at, 0.5, 4)[:, 0]
    got = kv.query_scale(c, torch.arange(40))
    assert got is not None
    np.testing.assert_allclose(got.numpy(), np.asarray(power * length), rtol=1e-6)
    none = {"power_scale": None, "log_scale": None}
    assert kv.query_scale(none, torch.arange(4)) is None


INDEXERS = {
    "dsa": {"num_heads": 2, "head_dim": 8, "top_k": 4},
    "plain sum, norm": {
        "num_heads": 2,
        "head_dim": 8,
        "top_k": 5,
        "weighted": False,
        "norm": {"type": "rmsnorm"},
    },
    "plain dots": {"num_heads": 2, "head_dim": 8, "top_k": 4, "relu": False},
    "mean blocks": {"num_heads": 2, "head_dim": 8, "top_k": 4, "block_size": 2},
    "mean blocks of 3": {
        "num_heads": 2,
        "head_dim": 8,
        "top_k": 6,
        "block_size": 3,
        "weighted": False,
        "norm": {"type": "rmsnorm"},
    },
    "max blocks": {
        "num_heads": 2,
        "head_dim": 8,
        "top_k": 4,
        "block_size": 2,
        "pool": "max",
    },
    "per group": {
        "num_heads": 2,
        "head_dim": 8,
        "top_k": 4,
        "per_group": True,
        "weighted": False,
    },
    "msa": {
        "num_heads": 2,
        "head_dim": 8,
        "top_k": 6,
        "block_size": 3,
        "pool": "max",
        "per_group": True,
        "relu": False,
        "weighted": False,
        "norm": {"type": "rmsnorm", "zero_centered": True},
    },
}


@pytest.mark.parametrize("window", [None, 5], ids=["causal", "window"])
@pytest.mark.parametrize("c", INDEXERS.values(), ids=INDEXERS.keys())
def test_sparse_attention_over_paged_caches_is_bux_s(c: dict, window: int | None):
    """The plugin's indexer and its attention over the picked keys of paged caches
    (bux_vllm.picks) give what the export's indexer and softmax give over all keys
    with its mask, for one request's queries together and one at a time."""
    from bux.export.hf import attention as hf_attention
    from bux.export.hf import sparse as hf_sparse
    from bux_vllm import indexer as plugin_indexer
    from bux_vllm import picks

    torch.manual_seed(0)
    ix = cfg.SparseIndexerConfig.model_validate(c).model_dump(mode="json")
    rope = cfg.RoPEConfig().model_dump(mode="json")
    width, s, h, g, dk = 16, 23, 4, 2, 8
    theirs = hf_sparse.Indexer(ix, width, width, rope, dk)
    for p in theirs.parameters():
        p.data.normal_()
    ours = plugin_indexer.Indexer(ix, width, rope, dk)
    ours.load_state_dict(theirs.state_dict())
    x, at = torch.randn(s, width), torch.arange(s)
    see = picks.seen(at, s, window)
    keys = theirs.keys(x[None], at[None])
    kept, _ = theirs(x[None], x[None], at[None], keys, see[None], at[None])
    q, k, v = (
        torch.randn(1, h, s, dk),
        torch.randn(1, g, s, dk),
        torch.randn(1, g, s, dk),
    )
    keep = see[None, None] & hf_attention.per_head(kept, h)
    sinks = torch.randn(h)
    want = hf_attention.softmax_attention(q, k, v, keep, 0.3, 4.0, sinks)[0]
    block, blocks = 4, 9
    table = torch.randperm(blocks)[None]
    kv_cache = torch.zeros(blocks, g, block, 2 * dk)
    index_cache = torch.zeros(blocks, 1, block, ix["head_dim"])
    slots = table[0, at // block] * block + at % block
    picks.write(kv_cache, slots, torch.cat([k[0], v[0]], -1).transpose(0, 1))
    picks.write(index_cache, slots, ours.keys(x, at)[:, None])
    spec = picks.Sparse(
        ix["top_k"],
        ix["block_size"],
        ix["block_size"] > 1 and ix["pool"] == "mean",
        ix["per_group"],
        ix["relu"],
        window,
        0.3,
        4.0,
        dk,
        (0, g),
        sinks,
        ours,
    )
    index = (ours.queries(x, at), ours.head_weights(x))
    qs = q[0].transpose(0, 1)
    caches = (kv_cache, index_cache)
    together = picks.attend_segment(spec, qs, index, at, caches, (table, table), s)
    rows = table.expand(s, -1)
    alone = picks.attend_segment(spec, qs, index, at, caches, (rows, rows), s)
    torch.testing.assert_close(together, want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(alone, want, rtol=1e-5, atol=1e-5)


def test_the_step_splits_into_single_queries_and_requests() -> None:
    from bux_vllm import picks

    parts = picks.segments([0, 1, 5, 6, 9], [7, 30, 2, 12], "cpu")
    assert [p.tokens.tolist() for p in parts] == [[0, 5], [1, 2, 3, 4], [6, 7, 8]]
    assert [p.requests.tolist() for p in parts] == [[0, 2], [1], [3]]
    assert [p.keys for p in parts] == [7, 30, 12]
