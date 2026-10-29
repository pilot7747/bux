"""bux's vLLM plugin (bux_vllm) builds what bux.export names: it places every
layer and residual algebra's branch of every example architecture where
bux.export.plan does, imports nothing of bux, reads an export's config.json as vLLM
needs it, and its plain PyTorch pieces (the router, RoPE, the residual algebras)
compute what the export does. Whether vLLM then computes what bux does takes a GPU:
scripts/vllm_parity.py checks it."""

import ast
import importlib.abc
import importlib.util
import itertools
import json
import math
import sys
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from flax import nnx

from bux import config as cfg
from bux.architecture import Model
from bux.architecture.attention import gqa as bux_gqa
from bux.architecture.attention import rope as bux_rope
from bux.architecture.attention.gqa import Attention as BuxAttention
from bux.architecture.experts import moe as bux_moe
from bux.architecture.experts.moe import MoE
from bux.architecture.layers.conv import CausalConv
from bux.architecture.registry import Taps
from bux.export import check
from bux.export import plan as core
from bux.export.hf import common as hf_common
from bux.export.hf import engram as hf_engram
from bux.export.hf import linear_attention as hf_linear
from bux.export.hf import moe as hf_moe
from bux.export.hf import residual as hf_residual
from bux.export.hf import rwkv7 as hf_rwkv7
from bux.export.huggingface import save
from bux.lang import compile, compile_file, ir
from bux_vllm import (
    common,
    compressed,
    config,
    delta,
    dplr,
    feeds,
    kv,
    latent,
    linear,
    ngram,
    outputs,
    plan,
    residual,
    routing,
    served,
)

ROOT = Path(__file__).parent.parent
EXAMPLES = sorted((ROOT / "examples" / "architectures").rglob("*.bux"))
FILES = sorted(Path(plan.__file__).parent.glob("*.py"))


def script(name: str) -> ModuleType:
    """Import scripts/`name`.py, which is no package's, as the module `name`, which
    the scripts imported after it import."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vllm_parity = script("vllm_parity")
vllm_fp8 = script("vllm_fp8")


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


@pytest.mark.parametrize("name", ["dsa", "dsa-share", "csa", "mla-dsa"])
def test_a_sparse_layer_keeps_its_indexer_keys_in_its_kv_cache(name: str) -> None:
    """One cache a call, whose blocks vLLM sizes as the other attention caches',
    so that prefix caching hits as often (a cache of indexer keys alone would get
    blocks of more tokens)."""
    arch = compile(vllm_parity.CASES[name])
    calls = plan.plan(arch.model_dump(mode="json")).calls
    layers = [c.layer for c in calls if config.sparse(c.layer["config"])]
    assert layers
    for n in layers:
        assert config.states(n["config"], plan.ports(n)) == ["full_attention"]


def test_a_sparse_cache_s_rows_hold_the_indexer_keys_beside_the_kv() -> None:
    """picks.Cache gives back the keys, values and indexer keys picks.merged
    wrote, a key's channels spread over the kv heads (padded where they do not
    divide), in pages padded apart as vLLM pads a smaller page to a larger one."""
    from bux_vllm import picks

    torch.manual_seed(0)
    g, c, i, block, blocks, s = 3, 10, 8, 4, 6, 17
    kv, index = torch.randn(s, g, c), torch.randn(s, i)
    rows = picks.merged(kv, index)
    assert rows.shape == (s, g, c + 3)
    page = g * block * (c + 3)
    raw = torch.zeros(blocks * (page + 5))
    pages = raw.as_strided(
        (blocks, g, block, c + 3), (page + 5, block * (c + 3), c + 3, 1)
    )
    table = torch.randperm(blocks - 1)[None] + 1
    at = torch.arange(s)
    picks.write(pages, table[0, at // block] * block + at % block, rows)
    cache = picks.Cache(pages, c, i)
    torch.testing.assert_close(cache.keys(table, at[None])[0], index)
    got = picks.gather(cache.kv, table, at[None])[0].transpose(0, 1)
    torch.testing.assert_close(got, kv)


def test_native_indexer_keys_are_in_deepgemm_s_paged_layout() -> None:
    """bux_vllm.dsa.Region writes FP8 indexer keys in the head slots after a cache's
    kv heads, as DeepGEMM's paged MQA logits read them: each block's keys, then
    their float32 scales; the kv heads keep their rows, and the views address the
    same bytes."""
    from bux_vllm import dsa

    torch.manual_seed(0)
    g, block, blocks, c, d, s = 3, 8, 5, 32, 64, 19
    slots = dsa.slots(d, c, torch.bfloat16.itemsize)
    assert slots == 2  # 68 bytes of a key and its scale, 64 a slot
    cache = torch.randn(blocks, g + slots, block, c, dtype=torch.bfloat16)
    kv = cache[:, :g].clone()
    table = (torch.randperm(blocks - 1) + 1)[None]
    at = torch.arange(s)
    keys = torch.randn(s, d) * torch.rand(s, 1) * 10
    region = dsa.Region(cache, g, d)
    region.write(table[0, at // block] * block + at % block, keys)
    assert torch.equal(cache[:, :g], kv)
    values, scales = dsa.quantized(keys)
    want = values.float() * scales[:, None]
    torch.testing.assert_close(region.dequantized(table, s), want, rtol=0, atol=0)
    near = 1e-3 * keys.abs().max().item()  # e4m3's rounding, and its subnormals'
    torch.testing.assert_close(want, keys, rtol=0.07, atol=near)
    raw = cache[:, g:].contiguous().view(torch.uint8).view(blocks, -1)
    b, o = table[0, 2 // block], 2 % block
    assert torch.equal(raw[b, o * d : (o + 1) * d], values[2].view(torch.uint8))
    sf = raw[b, block * d + 4 * o : block * d + 4 * o + 4].view(torch.float32)
    assert sf.item() == scales[2].item()
    paged = region.paged()
    assert paged.shape == (blocks, block, 1, d + 4)
    assert paged.stride(0) == region.raw.stride(0) == (g + slots) * block * c * 2


def test_native_indexer_queries_score_as_bux_s_indexer() -> None:
    """The FP8 queries and weights bux_vllm.dsa gives DeepGEMM, padded with heads of
    no weight, score FP8 keys as bux's indexer scores the keys it rounds,
    ``sum_j w_j ReLU(q_j . k)``, within FP8's rounding."""
    from bux_vllm import dsa, picks

    torch.manual_seed(0)
    q, w, keys = torch.randn(7, 4, 64), torch.randn(7, 4), torch.randn(30, 64)
    q8, weights = dsa.queries(q, w, 8)
    assert q8.dtype == dsa.FP8 and q8.shape == (7, 8, 64)
    assert torch.equal(weights[:, 4:], torch.zeros(7, 4))
    k8, sf = dsa.quantized(keys)
    dots = torch.einsum("qhd,sd->qhs", q8.float(), k8.float() * sf[:, None]).relu()
    got = torch.einsum("qhs,qh->qs", dots, weights)
    want = picks.scores(q, w, keys, relu=True, per_group=False)
    torch.testing.assert_close(got, want, rtol=0.1, atol=0.1 * want.abs().max().item())


def test_native_indexer_takes_deepseek_s_settings_alone() -> None:
    """vLLM's DSA kernels take single keys, ReLU'd dots summed over the heads, head
    dims of 32, 64 or 128, no window and a top-k of 2048 at most, in 16-bit
    models; not MLA's latents unless asked for."""
    from bux_vllm import dsa, picks

    def spec(**c: Any) -> picks.Sparse:
        settings: dict[str, Any] = {"top_k": 2048, "block_size": 1, "relu": True}
        settings |= c
        return picks.Sparse(
            settings["top_k"],
            settings["block_size"],
            False,
            bool(settings.get("per_group", False)),
            bool(settings["relu"]),
            settings.get("window"),
            1.0,
            None,
            64,
            (0, 1),
            latent=settings.get("latent", 0),
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dsa, "available", lambda: True)
        mp.setattr(dsa, "_head_sizes", lambda: (4, 8, 16, 32, 64, 128))
        assert dsa.takes(spec(), 64, 4, torch.bfloat16)
        assert dsa.takes(spec(), 128, 64, torch.float16)
        assert not dsa.takes(spec(), 64, 4, torch.float32)
        assert not dsa.takes(spec(), 16, 4, torch.bfloat16)
        assert not dsa.takes(spec(), 64, 200, torch.bfloat16)
        assert not dsa.takes(spec(top_k=4096), 64, 4, torch.bfloat16)
        assert not dsa.takes(spec(block_size=4), 64, 4, torch.bfloat16)
        assert not dsa.takes(spec(relu=False), 64, 4, torch.bfloat16)
        assert not dsa.takes(spec(window=16), 64, 4, torch.bfloat16)
        assert not dsa.takes(spec(per_group=True), 64, 4, torch.bfloat16)
        assert not dsa.takes(spec(latent=64), 64, 4, torch.bfloat16)
    assert dsa.padded_heads(3) >= 3


def test_a_step_s_decoded_tokens_are_one_segment() -> None:
    from bux_vllm import picks

    parts = picks.segments(3, [(3, 5, 3, 9)], 64)
    assert [(p.tokens, p.decoding) for p in parts] == [
        (slice(0, 3), True),
        (slice(3, 8), False),
    ]


def test_native_qsa_pools_its_blocks_as_bux_s_indexer() -> None:
    """For QSA on vLLM's DSA kernels, bux_vllm.dsa caches the keys as projected
    (Raw), and a token that completes a block writes the block's key (`pooled`) at
    the position of the block's index: what bux's indexer pools (block_keys), within
    FP8's rounding; `block_picks` adds the query's unfinished block as bux does."""
    from bux_vllm import dsa, picks
    from bux_vllm import indexer as plugin_indexer

    torch.manual_seed(0)
    ix = cfg.SparseIndexerConfig.model_validate(
        {"num_heads": 4, "head_dim": 64, "top_k": 8, "block_size": 4}
        | {"weighted": False, "norm": {"type": "rmsnorm"}}
    ).model_dump(mode="json")
    rope = cfg.RoPEConfig().model_dump(mode="json")
    ours = plugin_indexer.Indexer(ix, 32, rope, 16)
    for p in ours.parameters():
        p.data.normal_()
    g, block, blocks, c, s, size = 2, 8, 6, 32, 23, 4
    raw_slots = dsa.raw_slots(64, c * 4)
    cache = torch.zeros(blocks, g + raw_slots + dsa.slots(64, c, 4), block, c)
    table = (torch.randperm(blocks - 1) + 1)[None]
    at = torch.arange(s)
    slots = table[0, at // block] * block + at % block
    x = torch.randn(s, 32)
    raw = dsa.Raw(cache, g, 64)
    raw.write(slots, ours.keys(x, at))
    region = dsa.Region(cache, g + raw_slots, 64)
    rows = table.expand(s, -1)
    dsa.pooled(raw, region, ours, rows, at, size)
    keys = raw.at(rows[:1], at[None])[0]
    torch.testing.assert_close(keys, ours.keys(x, at).bfloat16().float())
    want = ours.block_keys(picks.pooled(keys, size), size)  # bux's, of 5 blocks
    got = region.dequantized(table, s // size)
    near = 1e-3 * want.abs().max().item()
    torch.testing.assert_close(got, want, rtol=0.07, atol=near)
    blocks_ = torch.tensor([[1, -1], [0, 1]], dtype=torch.int32)
    positions = torch.tensor([9, 11])
    keys_, valid = dsa.block_picks(blocks_, positions, size)
    see = picks.seen(positions, 12, None)
    theirs = picks.with_own_block(
        blocks_.long().clamp(min=0), blocks_ >= 0, see, positions, None, size
    )
    ours_ = torch.where(valid, keys_, -1).sort(-1).values
    want_ = torch.where(theirs[1], theirs[0], -1).sort(-1).values
    assert torch.equal(ours_, want_)


def test_vllm_s_sparse_gqa_attention_takes_plain_bfloat16_groups() -> None:
    """vLLM's own sparse GQA attention (Qwen4's QSA kernel) attends where it takes
    the layer: bfloat16 queries, keys and values of a power of two of dims, a
    bfloat16 or e4m3 cache, groups of 4 query heads or more, no sinks or softcap."""
    from bux_vllm import dsa, picks

    spec = picks.Sparse(16, 1, False, False, True, None, 0.125, None, 64, (0, 2))
    q = torch.zeros(3, 8, 64, dtype=torch.bfloat16)
    kv = torch.zeros(5, 2, 64, 128, dtype=torch.bfloat16)
    assert dsa.attends(spec, q, kv)
    assert dsa.attends(spec, q, kv.view(torch.uint8)[..., :128].view(dsa.FP8))
    assert not dsa.attends(
        spec, q, kv.view(torch.uint8)[..., :128].view(torch.float8_e5m2)
    )
    assert not dsa.attends(spec, q.float(), kv.float())
    assert not dsa.attends(spec, q, torch.zeros(5, 8, 64, 128, dtype=torch.bfloat16))
    assert not dsa.attends(spec, q[..., :48], kv[..., :96])
    sunk = picks.Sparse(
        16, 1, False, False, True, None, 0.1, None, 64, (0, 2), torch.ones(8)
    )
    assert not dsa.attends(sunk, q, kv)


def test_msa_takes_minimax_s_settings_alone() -> None:
    """vLLM's MiniMax-M3 kernels take MSA of blocks of 128, an unweighted index head
    a kv head, plain dots, heads of a power of two of dims, a power of two of kv
    heads on a rank and fewer than 64 blocks a query, in bfloat16; their picks of
    blocks are bux's keys of them."""
    from bux_vllm import msa, picks

    def spec(**c: Any) -> picks.Sparse:
        top_k, size = c.get("top_k", 2048), c.get("block_size", 128)
        relu, window = c.get("relu", False), c.get("window")
        groups = (0, c.get("kv_heads", 4))
        return picks.Sparse(
            top_k, size, False, True, relu, window, 1.0, None, 64, groups
        )

    ix = {"num_heads": 4, "head_dim": 128, "weighted": False}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(msa, "available", lambda: True)
        assert msa.takes(spec(), ix, 4, 128, torch.bfloat16)
        assert not msa.takes(spec(), ix, 4, 128, torch.float16)
        assert not msa.takes(spec(), ix, 8, 128, torch.bfloat16)
        assert not msa.takes(spec(), ix | {"weighted": True}, 4, 128, torch.bfloat16)
        assert not msa.takes(spec(), ix | {"head_dim": 96}, 4, 128, torch.bfloat16)
        assert not msa.takes(spec(block_size=64), ix, 4, 128, torch.bfloat16)
        assert not msa.takes(spec(relu=True), ix, 4, 128, torch.bfloat16)
        assert not msa.takes(spec(window=512), ix, 4, 128, torch.bfloat16)
        assert not msa.takes(spec(top_k=128 * 64), ix, 4, 128, torch.bfloat16)
        six = ix | {"num_heads": 6}
        assert not msa.takes(spec(kv_heads=6), six, 6, 128, torch.bfloat16)
    blocks = torch.tensor([[[2, -1]], [[0, 1]]], dtype=torch.int32)  # (g, Q, k)
    keys, valid = msa.as_picks(blocks, torch.tensor([300]))
    assert keys.shape == (1, 2, 256)
    assert keys[0, 0, :2].tolist() == [256, 257] and not valid[0, 0, 128:].any()
    assert valid[0, 0, :45].all() and not valid[0, 0, 45:].any()  # 256..300 seen
    assert valid[0, 1].all()


def test_native_picks_are_bux_s_picks() -> None:
    """bux_vllm.dsa.as_picks turns the kernels' picks, int32 positions with -1 for
    none, into bux_vllm.picks' positions and mask."""
    from bux_vllm import dsa

    keys, valid = dsa.as_picks(torch.tensor([[3, -1, 0]], dtype=torch.int32))
    assert keys.dtype == torch.long and keys.tolist() == [[3, 0, 0]]
    assert valid.tolist() == [[True, False, True]]


def test_a_conv_layer_keeps_a_conv_state() -> None:
    c = bux_config(vllm_parity.CASES["conv"])
    assert c.layer_types == ["full_attention", "conv", "conv", "full_attention", "conv"]
    c = bux_config(vllm_parity.CASES["short-conv"])
    assert c.layer_types == ["conv", "full_attention", "conv"]


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


@pytest.mark.parametrize("name", ["moe", "moe-latent", "llama"])
def test_the_config_gives_vllm_its_experts(name: str) -> None:
    """An MoE model gives vLLM its experts and picks (Mixtral's names), by which
    vLLM builds the expert-parallel group that FP8 experts reduce their scales
    over under tensor parallelism; a dense model none."""
    c = bux_config(vllm_parity.CASES[name])
    layers = compile(vllm_parity.CASES[name]).layers()
    moes = [n.config for n in layers if n.config.type == "moe"]
    experts = max((m.num_experts for m in moes), default=0)
    picks = max((m.top_k for m in moes), default=0)
    assert (c.num_experts, c.num_experts_per_tok) == (experts, picks)
    assert (experts > 0) == (name != "llama")


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


def test_a_model_of_sparse_mla_alone_is_no_mla_model_for_vllm() -> None:
    """Sparse MLA keeps its latents in caches of the plugin's own, one head each,
    so vLLM runs no MLA; beside MLA that vLLM runs, it raises as other attention."""
    c = bux_config(vllm_parity.CASES["mla-share"])
    assert not config.latent(c)
    assert (c.num_attention_heads, c.num_key_value_heads, c.head_dim) == (4, 1, 80)
    assert c.layer_types == ["full_attention"] * 3
    source = (
        vllm_parity.HEAD
        + vllm_parity.LATENT
        + "other = attn(indexer=indexer(num_heads=2, head_dim=16, top_k=8))\n"
        + "model = embed [rmsnorm attn] [rmsnorm other] rmsnorm head"
    )
    with pytest.raises(NotImplementedError, match="model-wide"):
        bux_config(source)


def naive_gather(cache, table, lens, starts) -> torch.Tensor:
    """Each request's cached context, token by token, from its blocks."""
    size, rows = cache.shape[1], []
    for r, (n, start) in enumerate(zip(lens, starts, strict=True)):
        for t in range(start, start + n):
            rows.append(cache[table[r, t // size], t % size])
    return torch.stack(rows)


@pytest.mark.parametrize("dtype", ["auto", "fp8", "fp8_e5m2"])
def test_the_latent_context_gathers_as_vllms_kernel(dtype: str) -> None:
    """bux_vllm.latent gathers a chunk's cached latents as vLLM's kernel does for
    the sizes it takes: from each request's blocks, from where the chunk starts in
    its context (here also off a block's start), from a cache of padded pages, an
    fp8 cache's bytes (uint8, as vLLM keeps them) read as e4m3 or e5m2, times its
    scale."""
    torch.manual_seed(0)
    padded = torch.randn(10, 4, 288 + 32)  # pages padded, as hybrids' are
    cache = values = padded[..., :288]
    if dtype != "auto":
        fp8 = padded.to(latent.FP8[dtype])
        cache, values = fp8.view(torch.uint8)[..., :288], fp8[..., :288]
    table = torch.tensor([[3, 7, 1, 0], [9, 2, 5, 8], [4, 6, 0, 0]], dtype=torch.int32)
    lens, starts = [6, 9, 2], [0, 5, 4]
    cu = torch.tensor([0, 6, 15, 17], dtype=torch.int32)
    to_seq = torch.repeat_interleave(
        torch.arange(3, dtype=torch.int32), torch.tensor(lens)
    )
    dst = torch.zeros(20, 288)
    scale = torch.tensor(0.5)
    seq_starts = torch.tensor(starts, dtype=torch.int32)
    latent.gather(cache, dst, table, cu, to_seq, 17, dtype, scale, seq_starts)
    want = naive_gather(values, table, lens, starts).float()
    want = want if dtype == "auto" else want * 0.5
    torch.testing.assert_close(dst[:17], want)
    assert not dst[17:].any()


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
    "shared expert sinks": "moe(num_experts=8, top_k=2, num_shared_experts=2,"
    " shared_expert_sink=true, router=sigmoid, routed_scaling_factor=2.5, expert=e)",
    "unnormalized sinks": "moe(num_experts=8, top_k=2, num_shared_experts=2,"
    " shared_expert_sink=true, normalize_top_k=false, expert=e)",
    "zero experts": "moe(num_experts=8, top_k=3, num_zero_experts=4, router=sigmoid,"
    " routed_scaling_factor=2.5, load_balancing=aux_free, expert=e)",
    "zero experts, longcat": "moe(num_experts=8, top_k=3, num_zero_experts=4,"
    " normalize_top_k=false, routed_scaling_factor=6, expert=e)",
    "zero experts, sinks": "moe(num_experts=8, top_k=3, num_zero_experts=4,"
    " num_shared_experts=2, shared_expert_sink=true, router=sigmoid, expert=e)",
}


@pytest.mark.parametrize("source", ROUTERS.values(), ids=ROUTERS.keys())
def test_the_router_picks_and_weighs_as_bux(source: str) -> None:
    """The plugin's router, which vLLM's fused MoE takes the picks from, is bux's,
    and so are the shared experts' weights where it scores them."""
    arch = compile(
        "d_model = 64\ne = gated_mlp(intermediate_size=16, activation=swish)\n"
        f"model = {source}"
    )
    c = next(arch.layers()).config
    assert isinstance(c, cfg.MoEConfig)
    moe = MoE(c, 64, rngs=nnx.Rngs(0))
    router = routing.Router(c.model_dump(mode="json"))
    if moe.router_bias is not None and router.bias is not None:
        bias = 0.1 * jax.random.normal(jax.random.key(1), moe.router_bias[...].shape)
        moe.router_bias[...] = bias
        router.bias.data = torch.from_numpy(np.array(bias))
    x = jax.random.normal(jax.random.key(2), (1, 32, 64))
    idx, weights, sinks = moe._route(x)
    logits = torch.from_numpy(np.array(moe._logits(x))[0])
    got_w, got_idx = router(None, logits, c.top_k, c.normalize_top_k)
    assert got_idx.tolist() == np.asarray(idx)[0].tolist()
    np.testing.assert_allclose(got_w.numpy(), np.asarray(weights)[0], rtol=1e-6)
    got_sinks = router.route(logits, c.top_k)[2].numpy()
    want = np.zeros((32, 0)) if sinks is None else np.asarray(sinks)[0]
    np.testing.assert_allclose(got_sinks, want, rtol=1e-6)


def test_zero_experts_split_as_bux_runs_them() -> None:
    """The zero experts' output (`zero_split`) is bux's, and the picks the fused MoE
    keeps of `given_logits` (`pick_given`), or `zero_split`'s for `apart`, weigh the
    routed experts as bux's do."""
    c = cfg.MoEConfig.model_validate(
        {
            "num_experts": 4,
            "top_k": 3,
            "num_zero_experts": 2,
            "router": "sigmoid",
            "expert": {
                "type": "gated_mlp",
                "intermediate_size": 8,
                "activation": "swish",
            },
        }
    )
    moe = MoE(c, 16, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(2), (1, 32, 16))
    idx, weights, _ = moe._route(x)
    assert (np.asarray(idx) >= 4).any()
    as_torch = (torch.from_numpy(np.array(a)[0]) for a in (x, weights, idx))
    zero, w, picks = routing.zero_split(*as_torch, 4)
    want = bux_moe.zero_experts(x, idx, weights, 4)[0]
    np.testing.assert_allclose(zero.numpy(), np.asarray(want), rtol=1e-6)
    dense = bux_moe.dense_weights(idx, weights, 4)[0]
    got = bux_moe.dense_weights(jnp.asarray(picks.numpy()), jnp.asarray(w.numpy()), 4)
    np.testing.assert_allclose(np.asarray(got), np.asarray(dense), rtol=1e-6)
    w, i = (torch.from_numpy(np.array(a)[0]) for a in (weights, idx))
    given_w, given_ids = routing.pick_given(
        torch.zeros(32, 16), routing.given_logits(w, i, 4), 3, True
    )
    assert given_ids.shape == (32, 3) and (given_ids < 4).all()
    given = bux_moe.dense_weights(jnp.asarray(given_ids), jnp.asarray(given_w), 4)
    np.testing.assert_allclose(np.asarray(given), np.asarray(dense), rtol=1e-6)


def test_zero_experts_route_in_one_graph() -> None:
    """The routing with zero experts traces whole, as torch.compile takes the
    model's forward, where vLLM's ZeroExpertRouter's assertion fails."""
    c = {
        "num_experts": 8,
        "num_zero_experts": 4,
        "top_k": 3,
        "router": "sigmoid",
        "router_logit_scale": 1.0,
        "num_groups": 1,
        "normalize_top_k": True,
        "routed_scaling_factor": 2.5,
        "load_balancing": "aux_free",
    }
    router = routing.Router(c)

    def routed(z: torch.Tensor, logits: torch.Tensor) -> tuple:
        weights, idx, _ = router.route(logits, 3)
        zero = routing.zero_split(z, weights, idx, 8)[0]
        return zero, routing.pick_given(
            z, routing.given_logits(weights, idx, 8), 3, True
        )

    z, logits = torch.randn(5, 16), torch.randn(5, 12)
    traced = torch.compile(routed, fullgraph=True, backend="eager")
    for a, b in zip(
        jax.tree.leaves(traced(z, logits)), jax.tree.leaves(routed(z, logits))
    ):
        assert torch.equal(a, b)


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


EXPERT_NORMS = {
    "rmsnorm": "rmsnorm",
    "rescale": "rmsnorm(eps=1e-8, zero_centered=true, scale_input=true)",
    "layernorm": "layernorm(bias=true)",
}


@pytest.mark.parametrize("norm", EXPERT_NORMS.values(), ids=EXPERT_NORMS.keys())
def test_experts_normalized_apart_compute_what_the_export_does(norm: str) -> None:
    """Under expert_norm each of a token's picks runs through its expert as a row
    of its own, as vLLM's fused MoE of one pick a row runs it (`pick_rows`), then
    the norm and the weights: the export's experts' output."""
    arch = compile(
        "d_model = 16\ne = gated_mlp(intermediate_size=8, activation=swish)\n"
        f"model = moe(num_experts=4, top_k=2, expert=e, expert_norm={norm})"
    )
    c = next(arch.layers()).config.model_dump(mode="json")
    torch.manual_seed(0)
    args = (SimpleNamespace(), c["expert"], 4, 16)
    theirs = hf_moe.Experts(*args, c["expert_norm"]).configure()
    bare = hf_moe.Experts(*args, None).configure()  # the experts, unnormalized
    for name in ("gate_up_proj", "down_proj"):
        param = getattr(theirs, name)
        param.data = torch.randn(param.shape) / 2
    bare.load_state_dict(theirs.state_dict(), strict=False)
    mine = common.Norm(c["expert_norm"], 16)
    assert theirs.post_expert_norm is not None
    twins(theirs.post_expert_norm, mine)
    z, weights = torch.randn(6, 16), torch.rand(6, 2)
    idx = torch.stack([torch.randperm(4)[:2] for _ in range(6)])

    def fused(rows: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        _, ids = routing.pick_rows(rows, logits, 1, False)
        return torch.cat(
            [bare.expert(r[None], int(e)) for r, e in zip(rows, ids[:, 0])]
        )

    got = routing.apart(fused, mine, z, weights, idx, 4)
    want = torch.zeros(6, 16)
    for t, j in itertools.product(range(6), range(2)):
        want[t] += weights[t, j] * theirs.expert(z[t : t + 1], int(idx[t, j]))[0]
    torch.testing.assert_close(got, want)


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


def test_lightning_decays_as_the_export_by_the_depth_the_export_gives() -> None:
    """The decay rates by the layer's depth among the token mixers, which the
    export's sites carry (bux_sites' extra)."""
    arch = compile(vllm_parity.CASES["lightning"])
    sites = core.plan(arch).sites()
    depths = [s["extra"] for s in sites.values() if s["extra"]]
    assert depths == [{"layer": 0, "layers": 3}, {"layer": 1, "layers": 3}]
    for depth in depths:
        want = hf_linear.slopes(2, depth["layer"], depth["layers"])
        torch.testing.assert_close(linear.slopes(2, *depth.values()), want)


def test_dyt_computes_what_the_export_does() -> None:
    from bux.export.hf import basic as hf_basic

    c = cfg.DyTConfig().model_dump(mode="json")
    theirs, mine = hf_basic.DyT(c, 16), common.DyT(c, 16)
    torch.manual_seed(0)
    twins(theirs, mine)
    x = torch.randn(5, 16) * 3
    torch.testing.assert_close(mine(x), theirs(x))


def through(branches: list, x: Any, ctx: Any, body: torch.Tensor) -> tuple[list, Any]:
    """Run the stream `x` through `branches`, each body ``tanh(h body)``; return what
    each read, and the stream after them."""
    reads = []
    for branch in branches:
        h, kept = branch.read(x, ctx)
        reads.append(h)
        x = branch.write_back(kept, torch.tanh(h @ body))
    return reads, x


def compare_branches(setting: str) -> None:
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
    ctx, ctx_mine = stand_in(pending_read=None), stand_in(pending_read=None)
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


@pytest.mark.parametrize("setting", RESIDUALS.values(), ids=RESIDUALS.keys())
def test_residual_branches_compute_what_the_export_does(setting: str) -> None:
    compare_branches(setting)


def hc_read(
    x: torch.Tensor,
    static_alpha: torch.Tensor,
    static_beta: torch.Tensor | None,
    wa: torch.Tensor | None,
    wb: torch.Tensor | None,
    scale_a: torch.Tensor | None,
    scale_b: torch.Tensor | None,
    pending: torch.Tensor | None,
    h: torch.Tensor,
    coef: torch.Tensor,
    eps: float,
    weighing: int,
) -> None:
    """What `bux_vllm.streams.hc_read` computes, in PyTorch: the layernorm's dot
    products as ``rstd (x W - mean sum(W))``, A's columns and then B's."""
    t, n, d = x.shape
    k, columns = static_alpha.shape[1], coef.shape[2]
    out = torch.zeros(t, n, columns)
    out[..., :k] = static_alpha.float()
    if static_beta is not None:
        out[..., k] = static_beta.float()
    if wa is not None and scale_a is not None:
        w, scale = torch.zeros(d, columns), torch.full((columns,), float(scale_a))
        w[:, :k] = wa.float()
        if wb is not None and scale_b is not None:
            w[:, k], scale[k] = wb.float()[:, 0], float(scale_b)
        xf = x.float()
        mean, square = xf.mean(-1, keepdim=True), xf.square().mean(-1, keepdim=True)
        rstd = torch.rsqrt((square - mean**2).clamp(min=0) + eps)
        out += scale * torch.tanh(rstd * (xf @ w - mean * w.sum(0)))
    coef.zero_()
    coef[:, :n] = out
    mix_into(h, x, out[..., 0], pending, weighing)


def mix_into(
    h: torch.Tensor, x: torch.Tensor, own: torch.Tensor, pending: Any, weighing: int
) -> None:
    """A fused read's ``sum_j r_j x_j`` into `h`, r as `weighing` says."""
    r = own if weighing == residual.OWN else pending
    if weighing == residual.FIRST:
        r = torch.zeros_like(own)
        r[:, 0] = 1
    h.copy_((r.float()[..., None] * x.float()).sum(1))


def mhc_read(
    x: torch.Tensor,
    proj: torch.Tensor | None,
    scale: torch.Tensor | None,
    bias: torch.Tensor,
    pending: torch.Tensor | None,
    h: torch.Tensor,
    logits: torch.Tensor,
    read: torch.Tensor,
    eps: float,
    weighing: int,
) -> None:
    """What `bux_vllm.streams.mhc_read` computes, in PyTorch."""
    t, n, d = x.shape
    k = bias.shape[0]
    out = bias.float().expand(t, k)
    if proj is not None and scale is not None:
        flat = x.float().reshape(t, n * d)
        rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
        widths = [n, n, n * n][: scale.shape[0]]
        a = torch.cat([s.float().expand(w) for s, w in zip(scale, widths)])
        out = flat @ proj.float() * rms * a[:k] + bias.float()
    logits.zero_()
    logits[:, :k] = out
    read.zero_()
    read[:, :n] = torch.sigmoid(out[:, :n]) + residual.EPS
    mix_into(h, x, read[:, :n], pending, weighing)


def mix_write(
    x: torch.Tensor,
    y: torch.Tensor,
    m: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    iters: int,
) -> None:
    """What `bux_vllm.streams.mix_write` computes, in PyTorch."""
    m, b = m.float(), b.float()
    if iters:
        m, b = residual.sinkhorn(m, iters), 2 * torch.sigmoid(b)
    mixed = (
        torch.einsum("tjk,tjd->tkd", m, x.float()) + b[..., None] * y.float()[:, None]
    )
    out.copy_(mixed)


def attnres_read(
    done: torch.Tensor,
    partial: torch.Tensor | None,
    gain: torch.Tensor,
    query: torch.Tensor,
    h: torch.Tensor,
    eps: float,
) -> None:
    """What `bux_vllm.streams.attnres_read` computes, in PyTorch."""
    v = done.float()
    if partial is not None:
        v = torch.cat([v, partial.to(done.dtype).float()[:, None]], 1)
    rstd = torch.rsqrt(v.square().mean(-1) + eps)
    scores = rstd * (v * (gain * query).float()).sum(-1)
    h.copy_((scores.softmax(-1)[..., None] * v).sum(1))


@cache
def kernels_in_pytorch() -> torch.library.Library:
    """Register the plugin's fused ops (`bux_vllm.streams`) as their computations in
    PyTorch, on the CPU, as vLLM does them with Triton on a GPU; return their
    library, which the cache keeps alive as long as the ops."""
    library = torch.library.Library("vllm", "FRAGMENT")
    ops = {
        "bux_hc_read": (hc_read, ["h", "coef"]),
        "bux_mhc_read": (mhc_read, ["h", "logits", "read"]),
        "bux_mix_write": (mix_write, ["out"]),
        "bux_attnres_read": (attnres_read, ["h"]),
    }
    for name, (fn, mutated) in ops.items():
        library.define(name + torch.library.infer_schema(fn, mutates_args=mutated))
        library.impl(name, fn, "CPU")
    return library


FUSED = {k: v for k, v in RESIDUALS.items() if k.split()[0] in ("hc", "mhc", "attnres")}


@pytest.mark.parametrize("setting", FUSED.values(), ids=FUSED.keys())
def test_fused_branches_compute_what_the_export_does(
    setting: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fused path, each kernel's computation in PyTorch: the coefficients'
    layout, the views the writes take of them, single-pass reads, attnres's
    stacked sources, and the merge, as the export computes them."""
    kernels_in_pytorch()
    monkeypatch.setattr(residual, "fused", lambda: True)
    compare_branches(setting)


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
        state = residual.widen(c, row[:, 0])
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
    for module in (hf_linear, hf_rwkv7)
    for cls in vars(module).values()
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


@pytest.mark.parametrize("learned", [False, True], ids=["fixed", "learned"])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_xsa_takes_out_what_bux_takes_out(learned: bool, groups: int) -> None:
    y = jax.random.normal(jax.random.key(1), (1, 8, 4, 16))
    v = jax.random.normal(jax.random.key(2), (1, 8, groups, 16))
    s = jax.random.normal(jax.random.key(3), (4,)) if learned else None
    want = bux_gqa.exclusive(y, v, None if s is None else jnp.tanh(s))[0]
    strength = None if s is None else torch.from_numpy(np.array(s)).tanh()
    got = outputs.exclusive(
        *(torch.from_numpy(np.array(a)[0]) for a in (y, v)), strength
    )
    np.testing.assert_allclose(got.numpy(), np.asarray(want), rtol=1e-5, atol=1e-6)


def test_differential_attention_pairs_heads_as_bux() -> None:
    c = cfg.GQAConfig(num_heads=4, num_kv_heads=2, differential=True)
    attn = BuxAttention(c, 64, (), rngs=nnx.Rngs(0))
    assert attn.lambda_proj is not None
    x = jax.random.normal(jax.random.key(1), (1, 8, 64))
    maps = jax.random.normal(jax.random.key(2), (1, 8, 8, 16))
    want = attn._difference(x, maps)[0]
    lam = torch.from_numpy(np.array(attn.lambda_proj(x)[0]))
    got = outputs.differential(torch.from_numpy(np.array(maps[0])), lam)
    np.testing.assert_allclose(got.numpy(), np.asarray(want), atol=1e-6)


def test_hyga_s_gate_is_bux_s() -> None:
    c = cfg.GQAConfig(
        num_heads=4,
        num_kv_heads=2,
        gated=True,
        gate_input_rank=3,
        gate_output_rank=5,
        gate_cross_head=True,
        gate_head_bias=True,
    )
    attn = BuxAttention(c, 64, (), rngs=nnx.Rngs(0))
    assert attn.gate is not None and attn.cross_gate is not None
    assert attn.gate_up is not None and attn.gate_bias is not None
    attn.gate_bias[...] = jax.random.normal(jax.random.key(3), (4,))
    assert attn.output_gate_down is not None and attn.output_gate_up is not None
    x = jax.random.normal(jax.random.key(1), (1, 8, 64))
    out = jax.random.normal(jax.random.key(2), (1, 8, 4, 16))
    want = attn._gate(x, out)[0]

    def t(a: Any) -> torch.Tensor:
        return torch.from_numpy(np.array(a))

    weights = (attn.gate_up, attn.output_gate_down, attn.output_gate_up)
    logits = t(attn.gate(x)[0]).view(8, 4, -1)
    logits = outputs.hybrid_logits(logits, t(out[0]), *(t(w[...]) for w in weights))
    gate = torch.sigmoid(logits + t(attn.gate_bias[...])[:, None])
    cross = torch.sigmoid(t(attn.cross_gate(out.reshape(1, 8, -1))[0]))[..., None]
    np.testing.assert_allclose((gate * cross).numpy(), np.asarray(want), atol=1e-6)


@pytest.mark.parametrize("gate", [6, None], ids=["gated", "plain"])
@pytest.mark.parametrize(("kv_heads", "tp"), [(2, 1), (2, 2), (1, 2)])
def test_value_embeddings_add_to_each_ranks_values_as_bux(
    gate: int | None, kv_heads: int, tp: int
) -> None:
    c = cfg.GQAConfig(num_heads=4, num_kv_heads=kv_heads, value_embed_gate=gate)
    port = ir.Port(name="ve", tap="ve1", mode="read")
    attn = BuxAttention(c, 64, (port,), rngs=nnx.Rngs(0))
    w = None
    if attn.ve_gate is not None:
        kernel = jax.random.normal(jax.random.key(1), attn.ve_gate.kernel[...].shape)
        attn.ve_gate.kernel[...] = kernel
        w = torch.from_numpy(np.array(kernel)).T.contiguous()
    x = jax.random.normal(jax.random.key(2), (1, 8, 64))
    v = jax.random.normal(jax.random.key(3), (1, 8, kv_heads, 16))
    e = jax.random.normal(jax.random.key(4), (1, 8, kv_heads * 16))
    want = np.asarray(attn._add_value_embedding(x, v, e)[0])
    xt, et = (torch.from_numpy(np.array(a)[0]) for a in (x, e))
    for rank in range(tp):
        first, g = kv.kv_heads(kv_heads, tp, rank)
        mine = torch.from_numpy(np.array(v)[0, :, first : first + g].reshape(8, -1))
        got = kv.add_value_embedding(xt, mine, et, w, (first, g))
        np.testing.assert_allclose(
            got.numpy(), want[:, first : first + g].reshape(8, -1), rtol=1e-5, atol=1e-6
        )


def test_values_from_the_keys_are_normalized_as_bux_and_convolved_alone() -> None:
    qk = cfg.NormalizationConfig(type="rmsnorm", eps=1e-5)
    c = cfg.GQAConfig(num_heads=4, num_kv_heads=2, qk_norm=qk, k_eq_v=True)
    attn = BuxAttention(c, 64, (), rngs=nnx.Rngs(0))
    k = jax.random.normal(jax.random.key(1), (1, 8, 32))
    _, want = attn._keys_as_values(k, None, Taps())
    norm = common.Norm(kv.value_norm(c.model_dump(mode="json")["qk_norm"]), 16)
    assert not list(norm.parameters())
    got = norm(torch.from_numpy(np.array(k)[0]).view(8, 2, 16))
    np.testing.assert_allclose(got.numpy(), np.asarray(want[0]), rtol=1e-5, atol=1e-6)
    conv = c.model_copy(update={"kv_conv_size": 4}).model_dump(mode="json")
    assert kv.conv_channels(conv, 64, 1) == 2 * 16
    assert kv.conv_channels(conv | {"k_eq_v": False}, 64, 2) == 16 + 16


KEYED = {
    "nope": ({}, (), True),
    "zero-centered gain": ({"qk_norm": {"zero_centered": True}}, (), True),
    "writes v": ({}, (["v", "v1", "write"],), True),
    "rope": ({"rope": cfg.RoPEConfig()}, (), False),
    "reads v": ({}, (["v", "v1", "read"],), False),
    "writes kv": ({}, (["kv", "kv", "write"],), False),
    "layernorm": ({"qk_norm": {"type": "layernorm"}}, (), False),
}


@pytest.mark.parametrize(("change", "ports", "keyed"), KEYED.values(), ids=KEYED)
def test_one_vector_a_token_is_cached_where_keys_are_values_times_a_gain(
    change: dict, ports: tuple, keyed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bux's keys are its values times the key norm's gain wherever the plugin
    caches the values alone, asked to, and puts the gain on the queries instead."""
    qk = {"type": "rmsnorm", "eps": 1e-5} | change.pop("qk_norm", {})
    c = cfg.GQAConfig.model_validate(
        {"num_heads": 4, "num_kv_heads": 2, "k_eq_v": True, "qk_norm": qk} | change
    )
    dumped = c.model_dump(mode="json")
    assert not kv.keys_from_values(dumped, list(ports))  # keys and values, unasked
    monkeypatch.setenv("BUX_VLLM_K_EQ_V_CACHE", "keys")
    assert kv.keys_from_values(dumped, list(ports)) == keyed
    if not keyed:
        return
    attn = BuxAttention(c, 64, (), rngs=nnx.Rngs(0))
    assert attn.k_norm is not None and attn.k_norm.scale is not None
    attn.k_norm.scale[...] = jax.random.normal(jax.random.key(2), (16,))
    k, v = attn._keys_as_values(
        jax.random.normal(jax.random.key(1), (1, 8, 32)), None, Taps()
    )
    gain = attn.k_norm.scale[...] + (1 if attn.k_norm.zero_centered else 0)
    np.testing.assert_allclose(
        np.asarray(k), np.asarray(v * gain), rtol=1e-5, atol=1e-6
    )


def test_ports_are_read_and_written_as_bux_lists_them() -> None:
    ports = [["v", "v1", "pop"], ["v", "v2", "write"], ["v", "v3", "write"]]
    assert kv.reads(ports, "v") == ("v1", True)
    assert kv.reads([["v", "v1", "write"]], "v") is None
    assert kv.writes(ports, "v") == ["v2", "v3"]
    gqa = {"indexer": None}
    kv.check_ports(gqa, ports)
    kv.check_ports(gqa, [["kv", "kv", "write"], ["v", "v1", "write"]])
    kv.check_ports(gqa, [["kv", "kv", "write"], ["index", "topk", "read"]])
    for other in (["v", "v1", "write"], ["index", "topk", "read"]):
        with pytest.raises(NotImplementedError, match="beside reading kv"):
            kv.check_ports(gqa, [["kv", "kv", "read"], other])


def test_kv_readers_attend_to_the_cache_of_the_call_their_tap_holds() -> None:
    """Hymba's wiring: each call that reads the kv port shares the KV cache of the
    call whose push its tap holds, the last one; ``$kv!`` takes it off, and a
    reader that writes the port pushes its writer's on."""
    arch = compile(vllm_parity.CASES["kv-share"])
    calls = plan.plan(arch.model_dump(mode="json")).calls
    gqa = [i for i, c in enumerate(calls) if c.layer["config"]["type"] == "gqa"]
    sources = kv.sources([plan.ports(c.layer) for c in calls])
    _, w1, s1, _, w2, _, g, r = gqa
    assert [sources[i] for i in gqa] == [None, None, w1, w1, None, w2, None, g]
    for reader, writer in [(s1, w1), (r, g)]:
        node, mine = calls[writer].layer, calls[reader].layer
        kv.check_shared(mine["config"], node["config"], node["width"])
        assert config.states(mine["config"], plan.ports(mine)) == ["full_attention"]


KV_REFUSED = {
    "a longer window": ("sliding_window_size=16", "window of 16 over 8"),
    "no window": ("sliding_window_size=none", "window of None over 8"),
    "other heads": ("num_kv_heads=4", "heads"),
}


@pytest.mark.parametrize(("setting", "match"), KV_REFUSED.values(), ids=KV_REFUSED)
def test_kv_readers_that_vllm_cannot_share_with_say_so(
    setting: str, match: str
) -> None:
    """A reader that attends past the writer's window, or reads other heads."""
    arch = compile(
        "d_model = 64\nlocal = gqa(num_heads=4, num_kv_heads=2, sliding_window_size=8)\n"
        f"model = local(kv=@kv) local({setting}, kv=$kv)"
    )
    writer, mine = (n.config.model_dump(mode="json") for n in arch.layers())
    with pytest.raises(NotImplementedError, match=match):
        kv.check_shared(mine, writer, 64)


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


def rule_inputs(
    n: int, floor: float | None, h: int = 2, dk: int = 8, dv: int = 4
) -> list[torch.Tensor]:
    """q, k, v, log decay, a and b of n tokens, as GDN-2 makes them: k
    L2-normalized, a decay per channel down to -floor (or unbounded, often near it),
    an erase gate in (0, 1)."""
    q, k = (delta.l2norm(torch.randn(n, h, dk)) for _ in range(2))
    rate = torch.randn(n, h, dk) * 3
    g = (
        -torch.nn.functional.softplus(rate)
        if floor is None
        else -floor * rate.sigmoid()
    )
    erase = torch.rand(n, h, dk)
    return [q, k, torch.randn(n, h, dv), g, -g.exp() * erase * k, k]


# GDN-2's gate bound at -5 (decays factored by blocks of 16), RWKV-7's bound
# (whole chunks), and an unbounded decay (chunks one after another); blocks of
# positions of 1, 2 or all chunks.
FLOORS = {"bound -5": 5.0, "rwkv7": math.exp(-0.5), "unbounded": None}
SPANS = {"one chunk a block": 384, "two": 768, "all": 2**16}


@pytest.mark.parametrize("span", SPANS.values(), ids=SPANS.keys())
@pytest.mark.parametrize("floor", FLOORS.values(), ids=FLOORS.keys())
def test_the_delta_rule_of_a_batch_is_the_exports_of_each_prompt(
    floor: float | None, span: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prompts of different lengths, padded side by side (across chunks), each
    give the export's outputs and last state of that prompt alone, from its own
    state; a decoding step of each is the export's step."""
    monkeypatch.setattr(delta, "BLOCK", span)
    torch.manual_seed(0)
    lengths = [3, 64, 300]
    starts = torch.tensor([0, 3, 67, 367])
    xs = rule_inputs(367, floor)
    before = torch.randn(3, 2, 8, 4)
    padded = [delta.padded(x, starts, max(lengths)) for x in xs]
    out, after = delta.rank_one(*padded, before, floor=floor)
    out = delta.unpadded(out, starts, 367)
    for i in range(3):
        mine = [x[None, starts[i] : starts[i + 1]] for x in xs]
        want, last = hf_linear.rank_one(*mine, state=before[i : i + 1])
        torch.testing.assert_close(out[starts[i] : starts[i + 1]], want[0])
        torch.testing.assert_close(after[i], last[0])
    one = rule_inputs(3, floor)
    got, state = delta.step(*one, before)
    q, k, v, g, a, b = (x[:, None] for x in one)
    want, last = hf_linear.recurrence(q, k, v, g, before, a, b)
    torch.testing.assert_close(got, want[:, 0])
    torch.testing.assert_close(state, last)


def test_a_token_shift_reads_the_token_before_in_its_sequence() -> None:
    x = torch.arange(6.0)[:, None]
    starts = torch.tensor([0, 2, 6])
    before = torch.tensor([[10.0], [20.0]])
    got = delta.previous(x, starts, before)
    assert got[:, 0].tolist() == [10, 0, 20, 2, 3, 4]


def exported(source: str) -> tuple[Any, dict]:
    """The export's layer of `source`, in a stream of 64, its weights drawn, and its
    config."""
    layer = next(compile(f"d_model = 64\nmodel = {source}").layers())
    c = layer.config.model_dump(mode="json")
    m = EXPORTED[c["type"]](c, 64, SimpleNamespace(ports=[], port=lambda *_: None))
    torch.manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.3)
    return m, c


def no_cache() -> Any:
    """The export's context of one whole sequence, without a cache."""
    none = torch.zeros(1, 0)  # neither tokens nor positions are read
    return hf_common.Ctx(none, none, None, None, torch.nn.ModuleDict(), lambda *_: None)


def by_chunks(
    xs: list[torch.Tensor], heads: tuple, floor: float | None
) -> torch.Tensor:
    """The plugin's rule over one prompt of tokens, from a zero state."""
    _, hv, dk, dv = heads
    q, k, v, g, a, b = (x[None] for x in xs)
    zero = torch.zeros(1, hv, dk, dv)
    return delta.rank_one(q, k, v, g, a, b, zero, floor=floor)[0][0]


GDN2 = {
    "gdn2": "gdn2(num_heads=2, num_v_heads=4, head_dim=16, gate_lower_bound=-5)",
    "gdn2 paper": "gdn2(num_heads=2, head_dim=16, v_head_dim=8, gate_rank=4,"
    " negative_eigenvalues=true, output_gate=sigmoid)",
}


@pytest.mark.parametrize("mixer", GDN2.values(), ids=GDN2.keys())
def test_gdn2_feeds_the_rule_what_the_export_does(mixer: str) -> None:
    """The plugin's GDN-2 between its projections and its output, on the export's
    weights and conv (vLLM's kernel convolves the same), is the export's layer."""
    m, c = exported(mixer)
    x = torch.randn(100, 64)
    qkv = torch.cat([m.q(x), m.k(x), m.v(x)], -1)
    qkv = torch.nn.functional.silu(m.conv(qkv[None], no_cache())[0])
    heads = linear.linear_dims(c)
    gates = (m.f(x), m.erase(x), m.write(x))
    xs = feeds.gdn2_rule(c, heads, qkv, gates, m.A_log, m.dt_bias)
    bound = c["gate_lower_bound"]
    o = by_chunks(xs, heads, None if bound is None else -bound)
    o = m.norm(o).flatten(1) * common.ACTIVATIONS[c["output_gate"]](m.gate(x))
    torch.testing.assert_close(m.o(o), m(x[None], no_cache())[0])


def test_deltanet_feeds_the_rule_what_the_export_does() -> None:
    """The plugin's DeltaNet between its projections and its output, on the
    export's weights and conv, is the export's layer."""
    m, c = exported("deltanet(num_heads=2, num_v_heads=4, head_dim=16)")
    x = torch.randn(100, 64)
    qkv = torch.cat([m.q(x), m.k(x), m.v(x)], -1)
    qkv = torch.nn.functional.silu(m.conv(qkv[None], no_cache())[0])
    heads = linear.linear_dims(c)
    xs = feeds.deltanet_rule(heads, qkv, m.b(x))
    o = by_chunks(xs, heads, 0.0)
    o = m.norm(o).flatten(1) * common.ACTIVATIONS[c["output_gate"]](m.gate(x))
    torch.testing.assert_close(m.o(o), m(x[None], no_cache())[0])


def projected(m: Any, x: torch.Tensor, silu: bool = True) -> torch.Tensor:
    """The export's q, k and v of x side by side, convolved and SiLU'd, or SiLU'd
    alone (if `silu`) without a conv, as the plugin's layers take them."""
    qkv = torch.cat([m.q(x), m.k(x), m.v(x)], -1)
    if m.conv is not None:
        return torch.nn.functional.silu(m.conv(qkv[None], no_cache())[0])
    return torch.nn.functional.silu(qkv) if silu else qkv


@pytest.mark.parametrize("conv", [4, None], ids=["conv", "no conv"])
def test_linear_layers_without_a_conv_feed_the_rule_what_the_export_does(
    conv: int | None,
) -> None:
    """DeltaNet and GDN-2 with or without a conv (q, k and v SiLU'd alone) are the
    export's layers."""
    size = "none" if conv is None else conv
    m, c = exported(f"deltanet(num_heads=2, head_dim=16, conv_size={size})")
    x = torch.randn(70, 64)
    heads = linear.linear_dims(c)
    o = by_chunks(feeds.deltanet_rule(heads, projected(m, x), m.b(x)), heads, 0.0)
    o = m.norm(o).flatten(1) * common.ACTIVATIONS[c["output_gate"]](m.gate(x))
    torch.testing.assert_close(m.o(o), m(x[None], no_cache())[0])
    m, c = exported(f"gdn2(num_heads=2, head_dim=16, conv_size={size})")
    gates = (m.f(x), m.erase(x), m.write(x))
    xs = feeds.gdn2_rule(c, heads, projected(m, x), gates, m.A_log, m.dt_bias)
    o = m.norm(by_chunks(xs, heads, None)).flatten(1)
    o = o * common.ACTIVATIONS[c["output_gate"]](m.gate(x))
    torch.testing.assert_close(m.o(o), m(x[None], no_cache())[0])


OWN_RULE = {
    "gdn": "gdn(num_heads=2, num_v_heads=4, head_dim=16, conv_size=none,"
    " output_gate=none)",
    "kda": "kda(num_heads=2, num_v_heads=4, head_dim=16, v_head_dim=8,"
    " conv_size=none, output_gate_rank=4, gate_lower_bound=-8)",
    "kda, conv": "kda(num_heads=2, head_dim=16, v_head_dim=8)",
}


@pytest.mark.parametrize("mixer", OWN_RULE.values(), ids=OWN_RULE.keys())
def test_gdn_and_kda_beyond_vllms_kernels_feed_the_rule_what_the_export_does(
    mixer: str,
) -> None:
    """GDN and KDA that vLLM's own kernels do not take (no conv or output gate,
    values of other widths than q and k's, a lower bound under -5) run bux's own
    rule (bux_vllm.deltanet): between their projections and their output, on the
    export's weights, the export's layers."""
    m, c = exported(mixer)
    x = torch.randn(70, 64)
    heads = linear.linear_dims(c)
    if c["type"] == "gdn":
        g = feeds.gdn_decay(m.a(x), m.A_log, m.dt_bias, heads[2])
        floor = None
    else:
        g = feeds.kda_decay(m.f(x), m.A_log, m.dt_bias, c)
        floor = -c["gate_lower_bound"] if c["gate_lower_bound"] else None
    xs = feeds.deltanet_rule(heads, projected(m, x), m.b(x), g)
    o = m.norm(by_chunks(xs, heads, floor)).flatten(1)
    if m.gate is not None:
        o = o * common.ACTIVATIONS[c["output_gate"]](m.gate(x))
    torch.testing.assert_close(m.o(o), m(x[None], no_cache())[0])


GLA = {
    "gla": "gla(num_heads=2, num_v_heads=4, head_dim=16, gate_rank=4)",
    "gla, conv": "gla(num_heads=2, head_dim=16, v_head_dim=8, conv_size=4,"
    " output_gate=sigmoid)",
}


LIGHTNING = {
    "lightning": "lightning(num_heads=2, head_dim=16)",
    "conv, value heads": "lightning(num_heads=2, num_v_heads=4, head_dim=16,"
    " v_head_dim=8, conv_size=4)",
}


@pytest.mark.parametrize("mixer", LIGHTNING.values(), ids=LIGHTNING.keys())
def test_lightning_feeds_the_rule_what_the_export_does(mixer: str) -> None:
    """The plugin's lightning attention between its projections and its output, on
    the export's weights, is the export's layer: q, k and v SiLU'd (convolved first
    with a conv), a fixed decay per value head by the layer's depth, the rule with
    no rank-one term, and the RMSNorm of all heads."""
    layer = next(compile(f"d_model = 64\nmodel = {mixer}").layers())
    c = layer.config.model_dump(mode="json")
    site = hf_common.Site(None, "mix", extra={"layer": 1, "layers": 3})
    m = hf_linear.LightningAttention(c, 64, site)
    torch.manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.3)
    x = torch.randn(70, 64)
    heads = linear.linear_dims(c)
    rates = linear.slopes(heads[1], 1, 3)
    xs = feeds.lightning_rule(heads, projected(m, x), rates)
    o = m.norm(by_chunks(xs, heads, None).flatten(1))
    assert m.gate is not None
    o = o * common.ACTIVATIONS[c["output_gate"]](m.gate(x))
    torch.testing.assert_close(m.o(o), m(x[None], no_cache())[0])


@pytest.mark.parametrize("mixer", GLA.values(), ids=GLA.keys())
def test_gla_feeds_the_rule_what_the_export_does(mixer: str) -> None:
    """The plugin's GLA between its projections and its output, on the export's
    weights (and conv), is the export's layer: q, k and v SiLU'd after a conv alone,
    and the rule with no rank-one term."""
    m, c = exported(mixer)
    x = torch.randn(100, 64)
    heads = linear.linear_dims(c)
    xs = feeds.gla_rule(heads, projected(m, x, silu=False), m.f(x))
    o = m.norm(by_chunks(xs, heads, None)).flatten(1)
    o = o * common.ACTIVATIONS[c["output_gate"]](m.gate(x))
    torch.testing.assert_close(m.o(o), m(x[None], no_cache())[0])


def test_rwkv7_feeds_the_rule_what_the_export_does() -> None:
    """The plugin's RWKV-7 between its projections and its output, on the export's
    weights, is the export's layer."""
    m, c = exported("rwkv7(num_heads=4, num_v_heads=8, head_dim=8, decay_rank=8)")
    x = torch.randn(100, 64)
    prev = delta.previous(x, torch.tensor([0, 100]), torch.zeros(1, 64))
    xr, xw, xk, xv, xa, xg = feeds.rwkv7_mixes(x, prev, m.shift)
    heads = linear.linear_dims(c)
    v = m.v(xv).unflatten(-1, heads[1::2])
    raw = (m.w(xw), m.a(xa))
    xs = feeds.rwkv7_rule(heads, m.r(xr), m.k(xk), v, raw, m.k_k, m.k_a)
    o = m.norm(by_chunks(xs, heads, math.exp(-0.5)))
    o = o + feeds.rwkv7_bonus(xs[0], xs[1], v, m.r_k)
    torch.testing.assert_close(m.o(o.flatten(1) * m.g(xg)), m(x[None], no_cache())[0])


@pytest.mark.parametrize("name", ["value-residual", "rwkv7-residual"])
def test_a_layers_ports_are_as_the_exports_sites_give_them(name: str) -> None:
    """The plugin's layers read their ports as [name, tap, mode], as the export's
    sites list them, from the IR's."""
    arch = compile(vllm_parity.CASES[name])
    sites = core.plan(arch).sites()
    tree = plan.plan(arch.model_dump(mode="json"), core.RESERVED)
    for call in (c for c in tree.calls if c.first):
        assert plan.ports(call.layer) == sites[call.path]["ports"]
    assert any(plan.ports(c.layer) for c in tree.calls)


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
    block, blocks, i = 4, 9, ix["head_dim"]
    table = torch.randperm(blocks)[None]
    pages = torch.zeros(blocks, g, block, 2 * dk + -(-i // g))
    slots = table[0, at // block] * block + at % block
    kv = torch.cat([k[0], v[0]], -1).transpose(0, 1)
    picks.write(pages, slots, picks.merged(kv, ours.keys(x, at)))
    cache = picks.Cache(pages, 2 * dk, i)
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
    together, picked = picks.attend_segment(spec, qs, index, at, cache, table, s)
    rows = table.expand(s, -1)
    alone, _ = picks.attend_segment(spec, qs, index, at, cache, rows, s)
    torch.testing.assert_close(together, want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(alone, want, rtol=1e-5, atol=1e-5)
    # A layer that reads these picks (the index port), with a window of its own.
    q2, k2, v2 = (torch.randn(1, n, s, dk) for n in (h, g, g))
    reader_see = picks.seen(at, s, 3)
    keep = reader_see[None, None] & hf_attention.per_head(kept, h)
    want = hf_attention.softmax_attention(q2, k2, v2, keep, 0.3, None, None)[0]
    kv_cache = torch.zeros(blocks, g, block, 2 * dk)
    picks.write(kv_cache, slots, torch.cat([k2[0], v2[0]], -1).transpose(0, 1))
    reader = picks.Sparse(0, 1, False, False, False, 3, 0.3, None, dk, (0, g))
    for r in (table, rows):
        got = picks.attend_picked(
            reader, q2[0].transpose(0, 1), picked, at, kv_cache, r, s
        )
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("window", [None, 5], ids=["causal", "window"])
@pytest.mark.parametrize(
    "name", [k for k, c in INDEXERS.items() if not c.get("per_group")]
)
def test_sparse_attention_in_the_latent_space_is_bux_s(name: str, window: int | None):
    """MLA's sparse attention as the plugin computes it, over paged latents, each
    head's queries absorbing its key projection (bux_vllm.picks.absorbed), gives
    what the export's indexer (its queries from the query latent) and softmax give
    over the keys and values projected up; and so does a layer that reads its picks
    and attends over latents of its own, by a window of its own."""
    from bux.export.hf import attention as hf_attention
    from bux.export.hf import sparse as hf_sparse
    from bux_vllm import indexer as plugin_indexer
    from bux_vllm import picks

    torch.manual_seed(0)
    ix = cfg.SparseIndexerConfig.model_validate(INDEXERS[name]).model_dump(mode="json")
    rope = cfg.RoPEConfig().model_dump(mode="json")
    width, latent, s, h, n, r, dv, rank = 16, 12, 23, 4, 8, 4, 8, 10
    theirs = hf_sparse.Indexer(ix, width, latent, rope, r)
    for p in theirs.parameters():
        p.data.normal_()
    ours = plugin_indexer.Indexer(ix, width, rope, r, latent)
    ours.load_state_dict(theirs.state_dict())
    x, query, at = torch.randn(s, width), torch.randn(s, latent), torch.arange(s)
    see = picks.seen(at, s, window)
    keys = theirs.keys(x[None], at[None])
    kept, _ = theirs(x[None], query[None], at[None], keys, see[None], at[None])
    kv_b = torch.randn(h * (n + dv), rank) / rank**0.5
    sinks = torch.randn(h)
    block, blocks = 4, 9
    table = torch.randperm(blocks)[None]
    slots = table[0, at // block] * block + at % block
    index_keys = ours.keys(x, at)

    def mla(
        window: int | None,
        softcap: float | None = None,
        sinks: Any = None,
        indexed: bool = False,
    ) -> tuple[Any, ...]:
        """Random latents in a paged cache (with the indexer's keys beside them if
        `indexed`), queries, and the export's output."""
        c, k_pe, q = torch.randn(s, rank), torch.randn(s, r), torch.randn(s, h, n + r)
        up = (c @ kv_b.T).view(s, h, n + dv)
        k = torch.cat([up[..., :n], k_pe[:, None].expand(s, h, r)], -1)
        keep = picks.seen(at, s, window)[None, None] & hf_attention.per_head(kept, h)
        qh, kh, vh = (t.transpose(0, 1)[None] for t in (q, k, up[..., n:]))
        want = hf_attention.softmax_attention(qh, kh, vh, keep, 0.3, softcap, sinks)
        want = want[0]
        rows = torch.cat([c, k_pe], -1)[:, None]
        rows = picks.merged(rows, index_keys) if indexed else rows
        pages = torch.zeros(blocks, 1, block, rows.shape[-1])
        picks.write(pages, slots, rows)
        cache = picks.Cache(pages, rank + r, index_keys.shape[-1] if indexed else 0)
        return picks.absorbed(q, kv_b, n), cache, want

    queries, cache, want = mla(window, 4.0, sinks, indexed=True)
    spec = picks.layer_spec(ix, ours, (window, 0.3, 4.0, sinks), rank + r, (0, 1), rank)
    index = (ours.queries(query, at), ours.head_weights(x))
    out, picked = picks.attend_segment(spec, queries, index, at, cache, table, s)
    torch.testing.assert_close(
        picks.projected(out, kv_b, n), want, rtol=1e-5, atol=1e-5
    )
    queries, cache, want = mla(3)
    reader = picks.layer_spec(None, None, (3, 0.3, None, None), rank + r, (0, 1), rank)
    out = picks.attend_picked(reader, queries, picked, at, cache.kv, table, s)
    torch.testing.assert_close(
        picks.projected(out, kv_b, n), want, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("window", [None, 6], ids=["causal", "window"])
@pytest.mark.parametrize("log_scale", [None, 0.5], ids=["plain", "log scale"])
def test_attention_with_a_relative_bias_over_paged_caches_is_bux_s(
    window: int | None, log_scale: float | None
) -> None:
    """Every key a query sees, read from paged caches, with the relative bias and
    the length scale on its logits (bux_vllm.picks.attend_all), gives what the
    export's softmax gives with its bias and gain, for one request's queries
    together and one at a time."""
    from bux.export.hf import attention as hf_attention
    from bux_vllm import picks

    torch.manual_seed(0)
    width, s, h, g, dk, rank, extent = 16, 23, 4, 2, 8, 3, 7
    x, at = torch.randn(1, s, width), torch.arange(s)
    proj = torch.nn.Linear(width, h * rank)
    table = torch.randn(rank, extent)
    layer = stand_in(rel_proj=proj, rel_table=table, h=h, log_scale=log_scale)
    layer.log_scale_floor = 4
    offsets = at[:, None] - at[None]
    bias = hf_attention.SoftmaxAttention._relative_bias(layer, x, offsets)
    gain = hf_attention.SoftmaxAttention._length_scale(
        layer, stand_in(positions=at[None])
    )
    q, k, v = (torch.randn(1, n, s, dk) for n in (h, g, g))
    keep = picks.seen(at, s, window)[None, None]
    sinks = torch.randn(h)
    want = hf_attention.softmax_attention(q, k, v, keep, 0.3, 4.0, sinks, bias, gain)[0]
    block, blocks = 4, 9
    table_rows = torch.randperm(blocks)[None]
    cache = torch.zeros(blocks, g, block, 2 * dk)
    slots = table_rows[0, at // block] * block + at % block
    picks.write(cache, slots, torch.cat([k[0], v[0]], -1).transpose(0, 1))
    spec = picks.layer_spec(None, None, (window, 0.3, 4.0, sinks), dk, (0, g))
    r = proj(x[0]).view(s, h, rank)
    lengths = None if gain is None else gain[0, 0, :, 0]
    qs = q[0].transpose(0, 1)
    for rows in (table_rows, table_rows.expand(s, -1)):
        got = picks.attend_all(spec, qs, (r, table, lengths), at, cache, rows, s)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_the_step_splits_into_its_decoded_tokens_and_prompt_chunks() -> None:
    """The decoded tokens, which vLLM puts first, attend together, as slices that a
    CUDA graph of the step fixes; each prompt chunk on its own."""
    from bux_vllm import picks

    parts = picks.segments(2, [(2, 4, 2, 30), (6, 3, 3, 12)], 64)
    assert [(p.tokens, p.requests, p.keys) for p in parts] == [
        (slice(0, 2), slice(0, 2), 64),
        (slice(2, 6), slice(2, 3), 30),
        (slice(6, 9), slice(3, 4), 12),
    ]
    assert picks.segments(0, [], 64) == []


ENGRAMS = {
    "engram": ("engram(head_dim=4, table_size=50)", None),
    "4-grams, boundary, classes": (
        (
            "engram(head_dim=4, table_size=50, max_ngram_size=4, num_heads=2,"
            " conv_size=2, boundary_token=5, vocab_classes=11)"
        ),
        None,
    ),
    "no conv": ("engram(head_dim=4, table_size=50, conv_size=none)", None),
    "on the streams": ("engram(head_dim=4, table_size=50)", "hc(expansion=2)"),
}

#: A step's tokens of each of two sequences, of 11 and 7 tokens: a prefill of each,
#: then the first's in a decoding step and a chunk, the second's in none.
STEPS = [(4, 7), (1, 0), (6, 0)]


class Stepped(ngram.EngramCore):
    """The plugin's engram, its conv's state kept here between steps, of R
    sequences, as vLLM's cache keeps it."""

    def __init__(self, c: dict, d: int, copies: int, hashes: dict, r: int):
        super().__init__(c, d, copies, hashes)
        self.conv = torch.nn.ParameterDict()
        if c["conv_size"]:
            channels, keep = copies * d, (c["conv_size"] - 1) * self.n
            self.conv["weight"] = torch.zeros(c["conv_size"], channels)
            self.past = torch.zeros(r, keep, channels)

    def convolve(self, x: torch.Tensor, ctx: Any) -> torch.Tensor:
        taps = self.conv["weight"]
        y, self.past = ngram.dilated_conv(
            x, taps, self.n, ctx.query_start_loc, self.past
        )
        return y


def engram_twins(setting: str, residual: str | None) -> tuple:
    """The export's engram and the plugin's, of the same random weights, and the
    model width."""
    head = HEAD + (f"residual = {residual}\n" if residual else "")
    arch = compile(f"{head}mem = {setting}\nmodel = embed mem head")
    tree = core.plan(arch)
    call = next(c for c in tree.calls if c.layer.config.type == "engram")
    extra = tree.sites()[call.path]["extra"]
    bux = arch.model_dump(mode="json")
    config = stand_in(bux=bux, hidden_size=arch.d_model)
    c, width = call.layer.config.model_dump(mode="json"), call.layer.width
    theirs = hf_engram.Engram(c, width, hf_common.Site(config, call.path, extra=extra))
    copies = 2 if residual else 1
    mine = Stepped(c, arch.d_model, copies, extra, len(STEPS[0]))
    twins(theirs, mine)
    return theirs, mine, arch.d_model, copies


@pytest.mark.parametrize(("setting", "residual"), ENGRAMS.values(), ids=ENGRAMS.keys())
def test_engram_computes_in_steps_what_the_export_does(
    setting: str, residual: str | None
) -> None:
    """Two sequences in steps of vLLM's kinds, the ids before each step given and
    the conv's state kept, add to the stream what the export's engram adds to each
    sequence whole."""
    torch.manual_seed(0)
    theirs, mine, d, copies = engram_twins(setting, residual)
    lengths = [sum(s[i] for s in STEPS) for i in range(2)]
    ids = [torch.randint(0, 12, (n,)) for n in lengths]
    ids[0][3] = 5  # a boundary token, where there is one
    xs = [torch.randn(n, copies, d) if residual else torch.randn(n, d) for n in lengths]
    want = []
    for tokens, x in zip(ids, xs, strict=True):
        at = torch.arange(len(tokens))[None]
        ctx = hf_common.Ctx(
            tokens[None], at, None, None, torch.nn.ModuleDict(), stand_in()
        )
        want.append(theirs(x[None], ctx)[0])
    got: list[list[torch.Tensor]] = [[], []]
    done = [0, 0]
    for step in STEPS:
        parts = [range(done[i], done[i] + step[i]) for i in range(2)]
        before = [
            [ids[i][j].item() if j >= 0 else -1 for j in range(done[i] - 4, done[i])]
            for i in range(2)
        ]
        ctx = stand_in(
            input_ids=torch.cat([ids[i][list(parts[i])] for i in range(2)]),
            query_start_loc=torch.tensor([0, step[0], step[0] + step[1]]),
            ngram_context=torch.tensor(before),
        )
        y = mine(torch.cat([xs[i][list(parts[i])] for i in range(2)]), ctx)
        got[0].append(y[: step[0]])
        got[1].append(y[step[0] :])
        done = [done[i] + step[i] for i in range(2)]
    for i in range(2):
        torch.testing.assert_close(torch.cat(got[i]), want[i])


def test_single_queries_of_several_requests_attend_to_their_own_keys() -> None:
    """A decoding step's queries, one a request, read their own requests' blocks."""
    from bux_vllm import picks

    torch.manual_seed(1)
    g, dk, i, block = 2, 8, 8, 4
    cache = picks.Cache(torch.randn(12, g, block, 2 * dk + i // g), 2 * dk, i)
    tables = torch.tensor([[3, 7, 1, 0, 0], [9, 2, 11, 5, 4]])
    at = torch.tensor([9, 16])  # the last positions of requests of 10 and 17 keys
    spec = picks.Sparse(4, 1, False, False, True, None, 0.3, None, dk, (0, g))
    q, iq, iw = torch.randn(2, 4, dk), torch.randn(2, 2, i), torch.randn(2, 2)
    both, _ = picks.attend_segment(spec, q, (iq, iw), at, cache, tables, 17)
    for r in range(2):
        alone, _ = picks.attend_segment(
            spec,
            q[r : r + 1],
            (iq[r : r + 1], iw[r : r + 1]),
            at[r : r + 1],
            cache,
            tables[r : r + 1],
            int(at[r]) + 1,
        )
        torch.testing.assert_close(both[r : r + 1], alone)


def fla_like(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fla's DPLR kernels' API, (B, T, H, D) and variable lengths by cu_seqlens, and
    its rule as fla's reference has it (``S = exp(gk) S + b (a^T S) + k v^T``, ``o =
    scale q^T S``), by the export's rule."""
    assert output_final_state and (lower_bound is None or gk.min() >= lower_bound)
    if cu_seqlens is None:
        rows = [(i, slice(None)) for i in range(q.shape[0])]
    else:
        ends = cu_seqlens.tolist()
        rows = [(0, slice(ends[i], ends[i + 1])) for i in range(len(ends) - 1)]
    outs, lasts = [], []
    for n, (row, span) in enumerate(rows):
        xs = [x[row : row + 1, span] for x in (q * scale, k, v, gk, a, b)]
        o, last = hf_linear.rank_one(*xs, state=initial_state[n : n + 1])
        outs.append(o)
        lasts.append(last)
    return torch.cat(outs, 1 if cu_seqlens is not None else 0), torch.cat(lasts)


def test_fla_takes_the_rule_as_its_kernels_take_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompts and decoding steps on fla's kernels (here as fla's API defines them)
    give what the plugin's PyTorch rule does: bux's a, b and log decay as fla's,
    a scale of 1, prompts by cu_seqlens."""
    monkeypatch.setattr(dplr, "kernels", lambda: (fla_like, fla_like))
    torch.manual_seed(0)
    starts = torch.tensor([0, 3, 67, 367])
    xs = rule_inputs(367, 5.0)
    before = torch.randn(3, 2, 8, 4)
    got, after = dplr.prompts(xs, starts, before, torch.float32, 5.0)
    padded = [delta.padded(x, starts, 300) for x in xs]
    want, last = delta.rank_one(*padded, before, floor=5.0)
    torch.testing.assert_close(got, delta.unpadded(want, starts, 367))
    torch.testing.assert_close(after, last)
    one = rule_inputs(3, 5.0)
    got, after = dplr.step(one, before, torch.float32)
    want, last = delta.step(*one, before)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(after, last)
    assert dplr.runs(torch.bfloat16) and not dplr.runs(torch.float32)


# --- compressed sparse attention (bux_vllm.compressed) ----------------------------------

#: csa settings the plugin runs, on 4 heads of 16 dims: each step below crosses
#: windows of 5 and 6, a compressed latent of 2 to 4 tokens and an indexer's top 3.
CSA = {
    "groups of 4": "csa(num_heads=4, head_dim=16, sliding_window_size=6,"
    " compress_ratio=4, sinks=true, rope=rope(fraction=0.5),"
    " indexer=indexer(num_heads=2, head_dim=8, top_k=3))",
    "window": "csa(num_heads=4, head_dim=16, sliding_window_size=5, sinks=true,"
    " rope=rope(fraction=0.25, layout=interleaved))",
    "every token, low-rank queries and output": "csa(num_heads=4, head_dim=16,"
    " q_lora_rank=8, o_lora_rank=4, o_groups=2, rope=rope(fraction=0.5))",
    "compressed by 3, no indexer, yarn, qk norm, softcap": "csa(num_heads=4,"
    " head_dim=16, sliding_window_size=5, compress_ratio=3, qk_norm=rmsnorm,"
    " softcap=3.0, rope=rope(fraction=0.5, scaling=yarn(factor=4,"
    " original_max_len=8)))",
    "single, plain unweighted dots, gated": "csa(num_heads=4, head_dim=16,"
    " sliding_window_size=6, compress_ratio=1, gated=true, gate_per_head=true,"
    " output_norm=rmsnorm, rope=rope(fraction=0.25, layout=interleaved),"
    " indexer=indexer(num_heads=2, head_dim=8, top_k=3, relu=false, weighted=false,"
    " norm=rmsnorm))",
    "compressed by 2, a latent query": "csa(num_heads=4, head_dim=16,"
    " sliding_window_size=6, compress_ratio=2, q_lora_rank=8, sinks=true,"
    " rope=rope(fraction=0.5), indexer=indexer(num_heads=2, head_dim=8, top_k=4))",
    "the parity case's": vllm_parity.SPARSE.removeprefix("attn = ").strip(),
    "a gate's head bias, a grouped output": "csa(num_heads=4, head_dim=16,"
    " sliding_window_size=5, compress_ratio=2, o_lora_rank=4, o_groups=2,"
    " gated=true, gate_head_bias=true, rope=rope(fraction=0.5))",
}

#: The tokens of a block of every csa cache: 12 hold whole groups of 1 to 4.
CSA_BLOCK = 12
#: The longest sequence the csa tests' caches take, as vLLM's max_model_len.
CSA_MAX_LEN = 48

#: Steps of two sequences, the second the first's first 25 tokens: each a list of
#: (sequence, first token, tokens), decoded tokens first, as vLLM orders them.
CSA_STEPS = [
    [(0, 0, 13), (1, 0, 7)],
    [(1, 7, 1), (0, 13, 17)],
    [(0, 30, 1), (1, 8, 17)],
    *([[(0, p, 1)] for p in range(31, 40)]),
]


class PagedCsa:
    """A csa call's paged caches, as vLLM keeps them for the sequences of
    `lengths`: blocks of `CSA_BLOCK` tokens, shuffled, block 0 (vLLM's null block)
    left out."""

    def __init__(self, c: dict, lengths: list[int], ports: list | None = None):
        self.paged: dict[str, compressed.Paged] = {}
        per_seq = -(-CSA_MAX_LEN // CSA_BLOCK)
        gen = torch.Generator().manual_seed(0)
        for name, (width, per, _) in compressed.cache_rows(c, ports).items():
            ids = torch.randperm(len(lengths) * per_seq, generator=gen) + 1
            cache = torch.zeros(len(ids) + 1, CSA_BLOCK // per, width)
            table = ids.view(len(lengths), per_seq)
            self.paged[name] = compressed.Paged(cache, table)

    def slots(self, name: str, seq: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """vLLM's slot mapping of the tokens at `pos` of sequences `seq`."""
        table = self.paged.get(name, self.paged["window"]).block_table
        return table[seq, pos // CSA_BLOCK] * CSA_BLOCK + pos % CSA_BLOCK

    def caches(self, steps: compressed.Steps, pad: int) -> compressed.Caches:
        """The caches of a batch of `steps`, its last `pad` tokens padding."""
        seq, pos = steps.seq, steps.positions
        state = self.paged.get("state")
        slots = {name: self.slots(name, seq, pos) for name in ("window", "state")}
        for name in ("window", "state") if pad else ():
            slots[name][-pad:] = -1
        return compressed.Caches(
            self.paged["window"],
            slots["window"],
            self.paged.get("compressed"),
            self.paged.get("index"),
            state,
            None if state is None else slots["state"],
        )


#: Padding after a step of decoded tokens alone, as a CUDA graph of a larger batch
#: replays it: each a (sequence, position) of stale metadata, which ends a group of
#: 1, 2 or 4, and writes nothing (slots of -1).
CSA_PADDING = [(0, 3), (1, 7)]


def csa_steps(step: list[tuple[int, int, int]]) -> compressed.Steps:
    """The batch of a step of `CSA_STEPS`, padded where it decodes alone."""
    padding = CSA_PADDING if all(n == 1 for *_, n in step) else []
    step = step + [(q, p, 1) for q, p in padding]
    positions = torch.cat([torch.arange(s, s + n) for _, s, n in step])
    seq = torch.cat([torch.full((n,), q) for q, _, n in step])
    starts = np.cumsum([0] + [n for *_, n in step]).tolist()
    lengths = [s + n for _, s, n in step]
    decodes = sum(n == 1 for *_, n in step)
    chunks = compressed.prompt_chunks(
        starts, lengths, decodes
    )  # by request: a sequence
    prompts = [(start, n, step[i][0], length) for start, n, i, length in chunks]
    # Decoded tokens alone read up to the longest a model takes, as in a CUDA graph.
    longest = max(lengths) if prompts else CSA_MAX_LEN
    return compressed.Steps(positions, seq, decodes, prompts, longest)


def csa_layer(
    m: Any, parts: Any, s: Any, caches: Any, steps: Any, x: Any, picks: Any = None
) -> Any:
    """A step of the plugin's csa layer (bux_vllm.csa) on x (T, width), with the
    export's module `m` for the projections it makes by vLLM's layers; its output,
    and the picks and candidate pool it makes (`compressed.mix`)."""
    t, pos = x.shape[0], steps.positions
    query = x if m.q_a is None else parts.q_a_norm(m.q_a(x))
    q = parts.queries(m.q(query).view(t, m.h, m.d), pos)
    latent = parts.latents(m.kv(x), pos)
    rows = index = None
    if m.compress_kv is not None:
        kv = m.compress_kv(x)
        if m.compress_gate is not None:
            both = torch.cat([kv, m.compress_gate(x)], -1)
            kv = compressed.pool(caches, steps, both, m.ratio)
        rows = parts.compressed(kv, pos)
    if parts.indexer is not None:
        w = m.indexer.weights
        index = parts.indexer.queries(
            m.indexer.q(query), None if w is None else w(x), pos
        )
    out, kept, pool = compressed.mix(
        caches, steps, s, q, latent, rows, index, m.sinks, picks
    )
    return m.output(x[None], parts.unrotate(out, pos)[None])[0], kept, pool


def csa_weights(c: dict, parts: Any, ports: list | None = None) -> set[str]:
    """The checkpoint's names of a csa's weights that the plugin's layer loads
    (bux_vllm.csa's targets): its GEMM's parts, its `parts`, and its own."""
    down, _ = compressed.down(c, ports)
    ix, rank = c["indexer"], c["o_lora_rank"]
    if compressed.reads(ports, "index"):
        ix = None
    own = {
        "q.weight": True,
        "indexer.q.weight": ix,
        "gate.weight": c["gated"],
        "gate_bias": c["gated"] and c["gate_head_bias"],
        "o_a": rank,
        "o.weight": True,
        "sinks": c["sinks"],
    }
    names = {f"{n}.weight" for n, _ in down} | {n for n, on in own.items() if on}
    return names | {n for n, _ in parts.named_parameters()}


@pytest.mark.parametrize("attn", CSA.values(), ids=CSA.keys())
def test_csa_computes_on_paged_caches_what_the_export_does(
    attn: str, tmp_path: Path
) -> None:
    """The plugin's csa, step by step over paged caches (prompts in chunks beside
    decoded tokens, two sequences in a batch), gives what the export's layer gives
    over each whole sequence."""
    arch = compile(
        f"d_model = 64\nvocab_size = 64\nmodel = embed [rmsnorm {attn}] head"
    )
    save(arch, check.perturbed(Model(arch, rngs=nnx.Rngs(0)), 1), tmp_path)
    hf = check._load(tmp_path, "eager")
    m = next(x for x in hf.modules() if type(x).__name__ == "CompressedSparseAttention")
    seen: list[torch.Tensor] = []
    m.register_forward_hook(lambda mod, args, out: seen.extend([args[0], out]))
    with torch.no_grad():
        hf.model(
            torch.randint(0, 64, (1, 40), generator=torch.Generator().manual_seed(1))
        )
    x, want = seen[0][0], seen[1][0]
    c = next(n.config for n in arch.layers() if n.config.type == "csa")
    c = c.model_dump(mode="json")
    parts = compressed.Parts(c)
    state = {k: v for k, v in m.state_dict().items() if k in parts.state_dict()}
    parts.load_state_dict(state)
    assert set(m.state_dict()) == csa_weights(c, parts)
    settings = compressed.Settings.of(c, m.scale)
    paged = PagedCsa(c, [40, 25])
    got = torch.zeros_like(want), torch.zeros_like(want[:25])
    with torch.no_grad():
        for step in CSA_STEPS:
            steps = csa_steps(step)
            pad = steps.positions.shape[0] - sum(n for *_, n in step)
            rows = torch.cat([x[s : s + n] for _, s, n in step] + [x[:pad]])
            out, _, _ = csa_layer(
                m, parts, settings, paged.caches(steps, pad), steps, rows
            )
            real = out[: out.shape[0] - pad].split([n for *_, n in step])
            for (q, start, n), y in zip(step, real):
                got[q][start : start + n] = y
    torch.testing.assert_close(got[0], want, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(got[1], want[:25], atol=2e-5, rtol=2e-5)


#: DeepSeek-V4.1's ports, small: a layer that compresses pairs, indexes them and
#: makes a candidate pool; one that reads its latents and picks; and one that reads
#: its latents and re-indexes them inside the pool.
CSA_PORTS = (
    "d_model = 64\nvocab_size = 64\n"
    "pairs = csa(num_heads=4, head_dim=16, sliding_window_size=6, compress_ratio=2,"
    " sinks=true, rope=rope(fraction=0.5), indexer=indexer(num_heads=2, head_dim=8,"
    " top_k=4, candidate_block_size=2, candidate_top_k=3))\n"
    "model = embed [rmsnorm pairs(kv=@kv, index=@topk, candidates=@pool)]"
    " [rmsnorm pairs(kv=$kv, index=$topk)]"
    " [rmsnorm pairs(kv=$kv!, index=@topk, candidates=$pool!)] head"
)


def test_csa_ports_compute_on_paged_caches_what_the_export_does(
    tmp_path: Path,
) -> None:
    """A csa writer and two readers of its kv port, step by step over paged caches:
    the second takes the first's picks (index), the third picks inside the first's
    candidate pool (candidates); each gives what the export's layer gives."""
    arch = compile(CSA_PORTS)
    save(arch, check.perturbed(Model(arch, rngs=nnx.Rngs(0)), 1), tmp_path)
    hf = check._load(tmp_path, "eager")
    mods = [x for x in hf.modules() if type(x).__name__ == "CompressedSparseAttention"]
    seen: dict[int, list[torch.Tensor]] = {}
    for i, m in enumerate(mods):
        m.register_forward_hook(
            lambda mod, args, out, i=i: seen.update({i: [args[0][0], out[0]]})
        )
    with torch.no_grad():
        hf.model(
            torch.randint(0, 64, (1, 40), generator=torch.Generator().manual_seed(1))
        )
    layers = [n for n in arch.layers() if n.config.type == "csa"]
    c = layers[0].config.model_dump(mode="json")
    calls = []
    for node, m in zip(layers, mods, strict=True):
        ports = [[p.name, p.tap, p.mode] for p in node.ports]
        parts = compressed.Parts(c, ports)
        state = {k: v for k, v in m.state_dict().items() if k in parts.state_dict()}
        parts.load_state_dict(state)
        assert set(m.state_dict()) == csa_weights(c, parts, ports)
        own = PagedCsa(c, [40, 25], ports)
        calls.append((m, parts, own))
    settings = compressed.Settings.of(c, mods[0].scale)
    got = [
        (torch.zeros_like(seen[i][1]), torch.zeros_like(seen[i][1][:25]))
        for i in range(3)
    ]
    with torch.no_grad():
        for step in CSA_STEPS:
            steps = csa_steps(step)
            pad = steps.positions.shape[0] - sum(n for *_, n in step)
            writer = calls[0][2].caches(steps, pad)
            kept = pool = None
            for i, (m, parts, own) in enumerate(calls):
                x = seen[i][0]
                rows = torch.cat([x[s : s + n] for _, s, n in step] + [x[:pad]])
                caches = own.caches(steps, pad)
                caches.compressed, caches.index = writer.compressed, writer.index
                picks = compressed.Picks(
                    kept if i == 1 else None, pool if i == 2 else None, i == 0
                )
                out, made, pooled = csa_layer(
                    m, parts, settings, caches, steps, rows, picks
                )
                if i == 0:
                    kept, pool = made, pooled
                real = out[: out.shape[0] - pad].split([n for *_, n in step])
                for (q, start, n), y in zip(step, real):
                    got[i][q][start : start + n] = y
    for i in range(3):
        torch.testing.assert_close(got[i][0], seen[i][1], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(got[i][1], seen[i][1][:25], atol=2e-5, rtol=2e-5)


def test_the_config_gives_vllm_the_heads_of_csa() -> None:
    """A csa's query heads and its one head of latents, as vLLM reads them."""
    c = bux_config(vllm_parity.CASES["csa"])
    assert c.layer_types == ["full_attention"] * 2
    assert (c.num_attention_heads, c.num_key_value_heads, c.head_dim) == (2, 1, 64)
    assert not config.latent(c)


@pytest.mark.parametrize("name", ["csa", "csa-latent"])
def test_csa_caches_a_window_and_compressed_latents(name: str) -> None:
    """Each csa call keeps a window of token latents, and a compressed latent with
    its indexer key for each compress_ratio tokens; and, for a sequence, the
    projections of the last compress_ratio tokens, which the next compressed latent
    pools. Each block of each is a power of two bytes, so that vLLM lays them side
    by side."""
    for layer in compile(vllm_parity.CASES[name]).layers():
        c = layer.config
        if c.type != "csa":
            continue
        rows = compressed.cache_rows(c.model_dump(mode="json"))
        for width, per, _ in rows.values():
            page = compressed.block_size(16, per) // per * width * 2  # in bfloat16
            assert page & (page - 1) == 0, (width, per)
        assert rows["window"] == (c.head_dim, 1, c.sliding_window_size)
        r = c.compress_ratio
        if r is None:
            assert set(rows) == {"window"}
            continue
        assert rows["compressed"] == (c.head_dim, r, None)
        if c.indexer is not None:
            assert rows["index"] == (c.indexer.head_dim, r, None)
        assert rows.get("state") == ((2 * c.head_dim, 1, r) if r > 1 else None)


def test_the_sparse_ops_read_only_what_a_paged_cache_has() -> None:
    """Every attribute bux_vllm.sparse reads of a cache, ``cache.<name>``, is one a
    PagedCache sets or defines: vLLM's compiler traces those reads."""
    tree = ast.parse((FILES[0].parent / "sparse.py").read_text())
    paged = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PagedCache"
    )
    has = {n.name for n in paged.body if isinstance(n, ast.FunctionDef)}
    for node in ast.walk(paged):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                elts = target.elts if isinstance(target, ast.Tuple) else [target]
                has |= {e.attr for e in elts if isinstance(e, ast.Attribute)}
    outside = [n for n in tree.body if n is not paged]  # its own reads a config
    reads = {
        n.attr
        for top in outside
        for n in ast.walk(top)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "cache"
    }
    assert reads <= has, reads - has


def test_each_layer_type_has_one_class() -> None:
    """Every bux layer type is named by one class's own ``bux_types``, which is
    all the registry reads: a subclass of a layer does not take over its types."""
    names: dict[str, list[str]] = {}
    for path in FILES:
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if (
                    isinstance(item, ast.Assign)
                    and any(getattr(t, "id", None) == "bux_types" for t in item.targets)
                    and isinstance(item.value, ast.Tuple)
                ):
                    for kind in item.value.elts:
                        names.setdefault(ast.literal_eval(kind), []).append(node.name)
    assert {k: v for k, v in names.items() if len(v) > 1} == {}
    assert 'vars(value).get("bux_types"' in (FILES[0].parent / "model.py").read_text()


def wide_mixer(kind: str, d_model: int) -> str:
    """A hybrid's delta-rule mixer `kind` at `d_model`: q/k heads of 128, as wide
    together as the stream, twice as many value heads, the gate bound at -5."""
    heads = d_model // 128
    return (
        f"{kind}(num_heads={heads}, num_v_heads={2 * heads}, head_dim=128,"
        " gate_lower_bound=-5)"
    )


# Hybrids at widths whose fused projections are no multiple of 64 (d_model 2816:
# KDA's projection 17068 wide unpadded), and shapes of odd widths.
FUSED = {
    **{
        f"{k} {d}": wide_mixer(k, d)
        for k in ("kda", "gdn2")
        for d in (2816, 3584, 4096)
    },
    "kda odd": "kda(num_heads=2, num_v_heads=6, head_dim=32)",
    "kda low rank": "kda(num_heads=4, num_v_heads=12, head_dim=16, output_gate_rank=8)",
    "gdn2 odd": "gdn2(num_heads=4, num_v_heads=12, head_dim=24, v_head_dim=8)",
}


@pytest.mark.parametrize("ranks", [1, 2, 4])
@pytest.mark.parametrize("mixer", FUSED.values(), ids=FUSED.keys())
def test_a_ranks_fused_projection_is_laid_out_as_vllm_lays_it(
    mixer: str, ranks: int
) -> None:
    """KDA's and GDN-2's fused input projection on each rank: its parts, which the
    layer splits its output by, are as wide as vLLM's merged layer makes the rank's
    output (KDA's decay down projection whole on every rank), a multiple of 64 so
    that inductor pads none of its matmuls, and the parts keep their widths."""
    c = next(compile(f"d_model = 64\nmodel = {mixer}").layers()).config
    c = c.model_dump(mode="json")
    if c["num_heads"] % ranks:
        pytest.skip("heads that the ranks do not split")
    h, hv, dk, dv = linear.linear_dims(c)
    want = [sum(linear.conv_channels(c)) // ranks]
    if c["type"] == "gdn2":
        p = linear.gdn2_projection(c, ranks)
        want += [h * dk // ranks, hv * dv // ranks]
    else:
        p = linear.kda_projection(c, ranks)
        gated = c["output_gate_rank"] is None
        want += [hv * dv // ranks] * gated + [c["gate_rank"] or dk, hv // ranks]
        assert p.whole is not None and p.sizes[p.whole] == (c["gate_rank"] or dk)
    assert p.split[: p.parts] == want
    assert p.width(ranks) == sum(p.split) and sum(p.split) % linear.ALIGN == 0
    if mixer == FUSED["kda 2816"] and ranks == 1:
        assert sum(want) == 17068 and sum(p.split) == 17088


def test_padding_columns_are_zero() -> None:
    layer = torch.nn.Linear(4, 10, bias=False)
    linear.zero_padding(layer, 3)
    assert not layer.weight[-3:].any() and layer.weight[:-3].any()


@pytest.mark.parametrize("before", ["none", "garbage"])
def test_engram_rows_stay_in_its_tables_on_any_inputs(before: str) -> None:
    """vLLM's dummy runs and its compiler's benchmarks run the model on ids, starts
    and ids before the step of no request (-1s, zeros or garbage): every gather of
    the engram stays in range, and so do its rows."""
    torch.manual_seed(0)
    _, mine, d, _ = engram_twins(ENGRAMS["4-grams, boundary, classes"][0], None)
    t, big = 9, 2**40
    context = torch.full((2, 4), -1)  # two sequences, as the conv's state holds
    if before == "garbage":
        context = torch.randint(-big, big, (2, 4))
    ctx = stand_in(
        input_ids=torch.randint(-big, big, (t,)),
        query_start_loc=torch.tensor([5, -7, 2**30]),
        ngram_context=context,
    )
    window = ngram.windows(ctx.input_ids, ctx.query_start_loc, context, mine.n)
    rows = ngram.ngram_rows(window, mine.hashes, mine.classes, mine.boundary)
    assert rows.min() >= 0 and rows.max() < mine.table.shape[0]
    assert mine(torch.randn(t, d), ctx).shape == (t, d)


# --- vLLM's online FP8 weights, on stand-ins of vLLM's layers ------------------------


def counted(param: torch.nn.Parameter) -> torch.nn.Parameter:
    """`param`, its weight loader counting the elements it loads, as vLLM's online
    quantization counts them before it fills and quantizes the weight."""
    param.loaded = 0  # ty: ignore[unresolved-attribute]

    def load(p: Any, tensor: torch.Tensor, *shard: Any) -> None:
        p.loaded += tensor.numel()

    param.weight_loader = load  # ty: ignore[unresolved-attribute]
    return param


def zeros(*shape: int) -> torch.nn.Parameter:
    """A parameter of zeros, its loader counting (`counted`)."""
    return counted(torch.nn.Parameter(torch.zeros(*shape), requires_grad=False))


class StubLinear(torch.nn.Module):
    """vLLM's linear layers on one rank: a weight (out, in) and a bias, their
    loaders counting, and the quantization config they were made with."""

    def __init__(self, input_size, output_size, bias=False, quant_config=None, **_):
        super().__init__()
        self.quant_config = quant_config
        self.weight = zeros(output_size, input_size)
        self.bias = zeros(output_size) if bias else None


class StubMerged(StubLinear):
    """vLLM's merged column-parallel layers, of their outputs side by side."""

    def __init__(self, input_size, output_sizes, *_, **kwargs):
        super().__init__(input_size, sum(output_sizes), **kwargs)
        self.output_sizes = list(output_sizes)


class StubQKV(StubLinear):
    """vLLM's ``QKVParallelLinear``: q, k and v's heads side by side."""

    def __init__(self, width, dk, heads, kv_heads, v_head_size=None, **kwargs):
        dv = v_head_size or dk
        super().__init__(width, heads * dk + kv_heads * (dk + dv), **kwargs)


class StubTable(torch.nn.Module):
    """vLLM's embedding and head, split by vocabulary, and their quantization
    config."""

    def __init__(self, vocab, width, bias=False, quant_config=None, **_):
        super().__init__()
        self.quant_config = quant_config
        self.weight = torch.nn.Parameter(torch.zeros(vocab, width), requires_grad=False)
        self.bias = zeros(vocab) if bias else None

    def tie_weights(self, embedding: torch.nn.Module) -> "StubTable":
        self.weight = embedding.weight
        return self


class StubExperts(torch.nn.Module):
    """vLLM's routed experts (`FusedMoEFactory`'s): their stacked weights, whose
    loaders count, and their quantization config."""

    def __init__(self, num_experts, hidden_size, intermediate_size, **kwargs):
        super().__init__()
        self.quant_config = kwargs.get("quant_config")
        shards = 2 if kwargs["ckpt_names"][2] is not None else 1
        self.w13_weight = zeros(num_experts, shards * intermediate_size, hidden_size)
        self.w2_weight = zeros(num_experts, hidden_size, intermediate_size)
        if kwargs["has_bias"]:
            self.w13_bias = zeros(num_experts, shards * intermediate_size)
            self.w2_bias = zeros(num_experts, hidden_size)

    def weight_loader(self, *args: Any) -> None:
        """The layer's own loader, which loads nothing here: vLLM's online
        quantization wraps the parameters' loaders, not it."""


class StubRunner(torch.nn.Module):
    """What vLLM's ``FusedMoEFactory`` makes: its routed experts, on one rank."""

    def __init__(self, **kwargs):
        super().__init__()
        self.routed_experts = StubExperts(**kwargs)
        rank = SimpleNamespace(tp_rank=0)
        self.moe_config = SimpleNamespace(moe_parallel_config=rank)


class StubModule(torch.nn.Module):
    """A vLLM module the test does not look into, such as its attention, and the
    arguments it was made with."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kwargs = kwargs


class StubNorm(torch.nn.Module):
    """vLLM's gated RMSNorm (KDA's)."""

    def __init__(self, width, **_):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width), requires_grad=False)


class Faked(type):
    """The type of `Fake`s: any name a fake is asked for (an enum's member, a
    static method) is a fake too, but not of the classes that subclass one."""

    def __getattr__(cls, name: str) -> Any:
        if name.startswith("__") or "fake" not in vars(cls):
            raise AttributeError(name)
        return fake(name)


class Fake(metaclass=Faked):
    """A name of vLLM's or Triton's that no layer here builds: a class that takes
    anything, and a decorator."""

    def __init__(self, *args, **kwargs):
        pass

    def __class_getitem__(cls, item):
        return cls

    def __call__(self, *args, **kwargs):
        return self


def fake(name: str) -> type:
    """A `Fake` class named `name`."""
    return Faked(name, (Fake,), {"fake": True})


#: The stand-ins of vLLM's names, by name.
STUBS: dict[str, Any] = {
    "ColumnParallelLinear": StubLinear,
    "RowParallelLinear": StubLinear,
    "ReplicatedLinear": StubLinear,
    "MergedColumnParallelLinear": StubMerged,
    "_KimiGDNMergedColumnParallelLinear": StubMerged,
    "QKVParallelLinear": StubQKV,
    "VocabParallelEmbedding": StubTable,
    "ParallelLMHead": StubTable,
    "FusedMoEFactory": StubRunner,
    "Attention": StubModule,
    "MLAAttention": StubModule,
    "FusedRMSNormGated": StubNorm,
    "get_tensor_model_parallel_world_size": lambda: 1,
    "get_tensor_model_parallel_rank": lambda: 0,
    "support_torch_compile": lambda **_: lambda cls: cls,
}


class Stubbed(ModuleType):
    """A module of vLLM's or Triton's, of stand-ins: `STUBS`, a `Fake` else."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        value = STUBS.get(name) or fake(name)
        setattr(self, name, value)
        return value


class StandIn(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Finds every module of vLLM and Triton, which the test's environment has not,
    as a `Stubbed` package."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] not in ("vllm", "triton"):
            return None
        return importlib.util.spec_from_loader(name, self, is_package=True)

    def create_module(self, spec):
        return Stubbed(spec.name)

    def exec_module(self, module):
        pass


@pytest.fixture
def stood_in(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """bux_vllm.model, importing stand-ins of vLLM and Triton (`StandIn`); the
    stand-ins and the plugin's modules that imported them are gone after the
    test."""
    before = set(sys.modules)
    monkeypatch.setattr(sys, "meta_path", [StandIn(), *sys.meta_path])
    yield importlib.import_module("bux_vllm.model")
    roots = ("bux_vllm", "vllm", "triton")
    for name in {n for n in sys.modules if n.split(".")[0] in roots} - before:
        module = sys.modules.pop(name)
        parent, _, child = name.rpartition(".")
        if getattr(sys.modules.get(parent), child, None) is module:
            delattr(sys.modules[parent], child)


def engine_config(where: Path, quant: Any, cache_dtype: str = "auto") -> Any:
    """vLLM's config of an engine serving the export at `where` under the
    quantization config `quant` and the KV cache dtype `cache_dtype`, as the
    plugin's layers read it."""
    hf = config.BuxConfig.from_dict(json.loads((where / "config.json").read_text()))
    model = {"dtype": torch.bfloat16, "max_model_len": 128}
    cache = {"cache_dtype": cache_dtype, "block_size": 16}
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, use_mla=config.latent(hf), **model),
        cache_config=SimpleNamespace(**cache),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        compilation_config=SimpleNamespace(static_forward_context={}, splitting_ops=[]),
        quant_config=quant,
        num_speculative_tokens=0,
    )


def built(
    source: str, where: Path, module: ModuleType, quant: Any, cache_dtype: str = "auto"
) -> Any:
    """The plugin's model (`module`'s, on stand-ins) of the model `source`, exported
    to `where`, its weights loaded, under the quantization config `quant` and the
    KV cache dtype `cache_dtype`."""
    from safetensors.torch import load_file

    arch = compile(source)
    with vllm_parity.placeholders(arch):
        save(arch, Model(arch, rngs=nnx.Rngs(0)), where)
    vllm_config = engine_config(where, quant, cache_dtype)
    model = module.BuxForCausalLM(vllm_config=vllm_config)
    model.load_weights(load_file(where / "model.safetensors").items())
    return model


def kept(model: Any, path: str) -> bool:
    """Whether the plugin keeps the vLLM linear layer at `path` in the model's
    dtype: an MoE's shared experts' gate of one output, as vLLM keeps Qwen's,
    HyGA's cross-head gate and differential attention's weights, of an output a
    head, and a sparse MLA's ``kv_b``, whose weight the sparse ops read."""
    owner, _, name = path.rpartition(".")
    if name in ("shared_gate", "cross_gate", "lambda_proj"):
        return True
    return name == "kv_b" and model.get_submodule(owner).spec is not None


def assert_quantized(model: Any, quant: Any) -> list:
    """Assert that every vLLM linear layer and routed experts of `model` take the
    quantization config `quant` (but those `kept`), and that their weights, padding
    included, load whole through their loaders; that the embeddings and heads, and
    the routers' weights, take none. Return the layers that take it."""
    layers = [
        (path, m)
        for path, m in model.named_modules()
        if isinstance(m, (StubLinear, StubExperts))
    ]
    for path, m in layers:
        assert m.quant_config is (None if kept(model, path) else quant), path
        for p in m.parameters(recurse=False):
            assert p.loaded == p.numel(), (path, p.shape, p.loaded)
    modules = list(model.modules())
    assert all(m.quant_config is None for m in modules if isinstance(m, StubTable))
    quantized = [m for path, m in layers if not kept(model, path)]
    held = {id(p) for m in quantized for p in m.parameters()}
    routers = [m.router_weight for m in modules if type(m).__name__ == "MoE"]
    assert not held & {id(p) for p in routers}
    return quantized


#: The kinds of vLLM layers that each milestone quantizes, and its MoE layers.
MILESTONES = {
    "gpt3": ({"StubQKV", "StubLinear"}, 0),
    "llama": ({"StubQKV", "StubLinear", "StubMerged"}, 0),
    "frontier": ({"StubQKV", "StubLinear", "StubMerged", "StubExperts"}, 3),
    "hybrid": ({"StubQKV", "StubLinear", "StubMerged", "StubExperts"}, 3),
}


@pytest.mark.parametrize("name", vllm_fp8.CASES)
def test_fp8_weights_quantize_every_projection_and_expert_alone(
    name: str, stood_in: ModuleType, tmp_path: Path
) -> None:
    """Under vLLM's online FP8 weights, every vLLM linear layer and routed experts
    of the timeline's milestones (scripts/vllm_fp8.py's, with 8 experts) take the
    engine's quantization config, and the embeddings, the heads and the routers
    none. Each such layer's weight, its padding included, loads whole through its
    loader, which vLLM's online quantization counts before it quantizes it."""
    source = vllm_fp8.CASES[name].replace("num_experts=256", "num_experts=8")
    quant = SimpleNamespace(name="fp8")
    model = built(source, tmp_path, stood_in, quant)
    kinds, moes = MILESTONES[name]
    assert {type(m).__name__ for m in assert_quantized(model, quant)} == kinds
    modules = list(model.modules())
    assert sum(isinstance(m, StubTable) for m in modules) == 2
    assert sum(type(m).__name__ == "MoE" for m in modules) == moes


#: The parity cases of vLLM's own recurrent layers, which have no stand-ins (vLLM's
#: Gated DeltaNet takes the quantization config itself), and gla without a conv,
#: which a 16-bit model refuses.
OWN_LAYERS = ("mamba", "mamba2", "gdn", "gla")


@pytest.mark.parametrize("name", [n for n in vllm_parity.CASES if n not in OWN_LAYERS])
def test_fp8_weights_quantize_every_layer_s_projections(
    name: str, stood_in: ModuleType, tmp_path: Path
) -> None:
    """Every family of layers the plugin runs gives its vLLM linear layers and its
    routed experts the quantization config, but those it keeps in the model's
    dtype (`kept`); their weights load whole through their loaders."""
    quant = SimpleNamespace(name="fp8")
    model = built(vllm_parity.CASES[name], tmp_path, stood_in, quant)
    assert assert_quantized(model, quant)


#: The parity cases whose attention keeps a cache of the plugin's own
#: (bux_vllm.sparse's, csa's), which takes no FP8 KV cache, by the layer it names:
#: on the CPU, where vLLM's DSA kernels pick none (see the test after).
OWN_CACHES = {
    **dict.fromkeys(("dsa", "dsa-conv", "msa", "qsa"), "gqa with an indexer"),
    **dict.fromkeys(("dsa-share", "msa-share"), "gqa with an indexer"),
    **dict.fromkeys(("dsa-native", "dsa-hybrid", "qsa-native"), "gqa with an indexer"),
    "msa-native": "gqa with an indexer",
    "relbias": "gqa with a relative bias",
    **dict.fromkeys(("mla-dsa", "mla-share"), "mla with an indexer"),
    "mla-dsa-native": "mla with an indexer",
    **dict.fromkeys(("csa", "csa-ports", "csa-latent"), "csa"),
}


@pytest.mark.parametrize("name", [n for n in vllm_parity.CASES if n not in OWN_LAYERS])
def test_an_fp8_kv_cache_is_vllm_s_or_refused(
    name: str, stood_in: ModuleType, tmp_path: Path
) -> None:
    """Under ``kv_cache_dtype="fp8"`` (beside FP8 weights) every attention layer on
    vLLM's own (gqa's, a dense mla's) gets the engine's cache config, from which
    vLLM keeps its cache in fp8; a layer whose cache is the plugin's own refuses as
    it is built, naming itself and the dtype, rather than keep bfloat16."""
    quant, source = SimpleNamespace(name="fp8"), vllm_parity.CASES[name]
    if name in OWN_CACHES:
        with pytest.raises(NotImplementedError, match="kv_cache_dtype=fp8") as e:
            built(source, tmp_path, stood_in, quant, "fp8")
        assert str(e.value).startswith(OWN_CACHES[name]), e.value
        return
    model = built(source, tmp_path, stood_in, quant, "fp8")
    caches = [m for m in model.modules() if isinstance(m, StubModule)]
    assert all(m.kwargs["cache_config"].cache_dtype == "fp8" for m in caches)


def test_native_dsa_takes_an_fp8_kv_cache(
    stood_in: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where vLLM's DSA kernels pick (as on a B300), a gqa layer's indexer keeps its
    FP8 keys in head slots after its kv heads (two, of 64 bytes, for 64 FP8 dims and
    their scale), and its cache, and that of a layer that reads its picks, take an
    FP8 KV cache; so do QSA's, its keys as projected in bfloat16 in slots of their
    own, and MSA's on vLLM's MiniMax-M3 kernels; sparse MLA's latents refuse it."""
    dsa = importlib.import_module("bux_vllm.dsa")
    monkeypatch.setattr(dsa, "available", lambda: True)
    msa = importlib.import_module("bux_vllm.msa")
    monkeypatch.setattr(msa, "available", lambda: True)
    model = built(vllm_parity.CASES["dsa-native"], tmp_path, stood_in, None, "fp8")
    caches = [m for m in model.modules() if type(m).__name__ == "PagedCache"]
    e4m3 = torch.float8_e4m3fn
    want = [(True, e4m3, 2), (True, e4m3, 2), (False, e4m3, 0)]
    assert [(c.native, c.fp8, c.slots) for c in caches] == want
    assert all(c.dtype is torch.uint8 and c.share == 0 for c in caches)
    model = built(
        vllm_parity.CASES["qsa-native"], tmp_path / "qsa", stood_in, None, "fp8"
    )
    caches = [m for m in model.modules() if type(m).__name__ == "PagedCache"]
    assert [(c.native, c.raw_slots, c.slots) for c in caches] == [(True, 2, 4)] * 2
    source = vllm_parity.CASES["msa-native"]
    model = built(source, tmp_path / "msa", stood_in, None, "fp8")
    caches = [m for m in model.modules() if type(m).__name__ == "PagedCache"]
    assert [(c.blocked, c.native, c.raw_slots, c.slots) for c in caches] == [
        (True, False, 1, 1)
    ] * 2
    with pytest.raises(NotImplementedError, match="mla with an indexer: kv_cache"):
        source = vllm_parity.CASES["mla-dsa-native"]
        built(source, tmp_path / "mla", stood_in, None, "fp8")


def test_the_plugin_s_own_caches_keep_the_model_s_dtype_alone() -> None:
    """A cache the plugin keeps itself takes ``kv_cache_dtype`` auto, or the
    model's own, and refuses any other: fp8, or float16 in a bfloat16 model."""

    def engine(cache_dtype: str) -> Any:
        return SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        )

    for name in ("auto", "bfloat16"):
        assert common.own_cache_dtype(engine(name), "csa") is torch.bfloat16
    for name in ("fp8", "fp8_e5m2", "float16"):
        with pytest.raises(NotImplementedError, match=f"csa: kv_cache_dtype={name} "):
            common.own_cache_dtype(engine(name), "csa")


def test_the_survey_counts_what_serves_each_kind_of_layer() -> None:
    """bux_vllm.served.survey, which vLLM's workers run on the model, counts each
    kind of layer's quantization method and FP8 GEMM, each attention layer's KV
    cache and backend, and the routed experts' backend."""
    gemm = type("CutlassFP8ScaledMMLinearKernel", (), {})()
    linear = type("Fp8PerTensorOnlineLinearMethod", (), {"fp8_linear": gemm})()
    experts = type(
        "Fp8PerTensorOnlineMoEMethod",
        (),
        {
            "fp8_backend": SimpleNamespace(value="TRITON"),
            "experts_cls": type("TritonExperts", (), {}),
        },
    )()
    backend = SimpleNamespace(get_name=lambda: "FLASHINFER")

    def layer(kind: str, **attributes: Any) -> torch.nn.Module:
        module = type(kind, (torch.nn.Module,), {})()
        vars(module).update(attributes)
        return module

    model = torch.nn.Module()
    model.qkv = layer("QKVParallelLinear", quant_method=linear)
    model.experts = layer("RoutedExperts", quant_method=experts)
    model.cache = layer("Attention", kv_cache_dtype="fp8", attn_backend=backend)
    assert served.survey(model) == {
        "methods": {
            "QKVParallelLinear qkv: Fp8PerTensorOnlineLinearMethod"
            " (CutlassFP8ScaledMMLinearKernel)": 1,
            "RoutedExperts experts: Fp8PerTensorOnlineMoEMethod": 1,
        },
        "caches": {"Attention: kv fp8 (FLASHINFER)": 1},
        "backends": {"TRITON (TritonExperts)": 1},
    }


def test_mla_of_dims_vllm_s_prefills_lack_takes_no_fp8_queries(
    stood_in: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where no MLA prefill backend of vLLM's takes the dims, the plugin's is
    FlashAttention's, and vLLM's check of whether the prefill takes FP8 queries,
    which an FP8 KV cache makes its metadata ask, answers no instead of raising."""
    mla = importlib.import_module("bux_vllm.mla")

    def no_backend(*args: Any) -> None:
        raise ValueError("No valid MLA prefill backend found")

    monkeypatch.setattr(mla, "get_mla_prefill_backend", no_backend)
    monkeypatch.setattr(
        mla.mla_attention, "backend_supports_prefill_query_quantization", no_backend
    )
    engine = SimpleNamespace(attention_config=SimpleNamespace(mla_prefill_backend=None))
    assert mla.prefill_backend(engine) is mla.FlashAttnPrefillBackend
    assert mla.mla_attention.backend_supports_prefill_query_quantization() is False


def test_the_fp8_emulation_rounds_what_vllm_quantizes(tmp_path: Path) -> None:
    """scripts/vllm_fp8.py's emulation of vLLM's FP8 rounds the weights of the
    linear layers and routed experts vLLM quantizes to FP8's grid, and keeps the
    head's, the embedding's and the router's as they are."""
    from transformers import AutoModelForCausalLM

    arch = compile(
        vllm_fp8.CASES["frontier"].replace("num_experts=256", "num_experts=8")
    )
    save(arch, Model(arch, rngs=nnx.Rngs(0)), tmp_path)
    load = AutoModelForCausalLM.from_pretrained
    model = load(tmp_path, trust_remote_code=True, dtype=torch.float32)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    vllm_fp8.emulate(model, kv=True)
    for name, p in model.named_parameters():
        kept = any(k in name for k in ("embed", "fr_out", "router", "norm", "conv"))
        if kept or p.dim() < 2:
            assert torch.equal(p, before[name]), name
            continue
        w = p.detach()
        scales = (
            [vllm_fp8._scale(x) for x in w] if w.dim() == 3 else [vllm_fp8._scale(w)]
        )
        again = [
            vllm_fp8._rounded(x, s) for x, s in zip(w if w.dim() == 3 else [w], scales)
        ]
        assert all(torch.equal(a, x) for a, x in zip(again, w if w.dim() == 3 else [w]))
        assert not torch.equal(p, before[name]), name
