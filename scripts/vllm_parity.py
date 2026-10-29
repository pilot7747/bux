"""Check that vLLM computes what bux does: models exported to transformers
(`bux.export.huggingface`) and served by bux's vLLM plugin (`bux_vllm`).

bux and vLLM do not share an environment (each pins its own PyTorch), so it runs in
three steps:

    # bux's environment: export each case and its prompts
    JAX_PLATFORMS=cpu python scripts/vllm_parity.py export DIR [--dtype bfloat16]
    # vLLM's, with the plugin (pip install src/bux_vllm): decode them
    python scripts/vllm_parity.py run DIR [--tp 2] [--eager] [--moe-backend triton]
        [--mamba-ssm-cache-dtype float32] [--attention-backend TRITON_MLA] [--tag TAG]
    # bux's again: compare
    JAX_PLATFORMS=cpu python scripts/vllm_parity.py compare DIR [--tag TAG]
        [--fp8-indexer]

`run` decodes every case's prompts greedily in one batch of different lengths and
keeps vLLM's log-probabilities of the whole vocabulary at every position: the
prompts' (prefill) and each generated token's (decoding, by the KV cache or the
recurrent state). `compare` runs bux on the same tokens, each prompt and vLLM's
continuation, and checks every position: the token vLLM picked is bux's greedy pick
(its log-probability within `--margin` of bux's best, 0 by default: the same token),
and the log-probabilities agree within `--tolerance`. A greedy decode that agrees at
every step is bux's own greedy decode. It prints a line a case, and exits with 1 if
one disagrees.

The cases are tiny models (d_model 128, a vocabulary of 256) of fresh weights,
perturbed (`bux.export.check.perturbed`), one per family of layers and settings the
plugin runs; `--only` picks some by name. In bfloat16 the export's weights are
bfloat16, and bux's reference computes in float32 from the same rounded weights, so
that the difference is vLLM's arithmetic alone; bux computes in bfloat16 too, and
vLLM may be off by up to twice as much as bux is from itself, and break near ties
that close either way. An indexer on vLLM's DSA kernels scores in FP8, which picks
other keys than bux's float32 scores do: ``--fp8-indexer`` has bux's reference
score in FP8 too (`fp8_indexer`), so that the rest is checked as for other cases.
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

HEAD = "d_model = 128\nvocab_size = 256\n"
FFN = "ffn = gated_mlp(intermediate_size=256, activation=swish)\n"
ATTN = "attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())\n"
BLOCK = "block = [rmsnorm attn] [rmsnorm ffn]\n"
LLAMA = HEAD + ATTN + FFN + BLOCK
EXPERT = "dense = gated_mlp(intermediate_size=64, activation=swish)\n"
MOE_BLOCK = "block = [rmsnorm attn] [rmsnorm experts]\n"
# Qwen3-Next's attention beside linear attention (`delta`), which the case defines.
GATED_HEAD = (
    HEAD
    + "attn = gqa(num_heads=4, num_kv_heads=2, qk_norm=rmsnorm(zero_centered=true),"
    " gated=true, rope=rope(fraction=0.25))\n" + FFN
)
LINEAR_HYBRID = (
    "model = embed [rmsnorm delta] [rmsnorm ffn] [rmsnorm delta] [rmsnorm ffn]"
    " [rmsnorm attn] [rmsnorm ffn] rmsnorm head"
)
#: MLA with a latent of 256 and 32 rotated dims, on 4 heads.
LATENT = (
    "attn = mla(num_heads=4, kv_lora_rank=256, qk_nope_head_dim=64,"
    " qk_rope_head_dim=32, v_head_dim=64, rope=rope())\n"
)
#: DeepSeek-V3's MLA, its dims and RoPE, on 4 heads and a query latent of 64.
DEEPSEEK = (
    "attn = mla(num_heads=4, q_lora_rank=64, kv_lora_rank=512, rope=rope("
    "layout=interleaved, scaling=yarn(factor=4, original_max_len=16,"
    " attention_factor_on=softmax)))\n"
)
#: MLA with DeepSeek-V3.2's indexer, on 4 heads, a query latent of 64 and a latent
#: of 64; and the same with a window of 16, sinks and a softcap (local).
SPARSE_MLA = (
    "attn = mla(num_heads=4, q_lora_rank=64, kv_lora_rank=64, qk_nope_head_dim=32,"
    " qk_rope_head_dim=16, v_head_dim=32, rope=rope(), indexer=indexer(num_heads=2,"
    " head_dim=16, top_k=8))\n"
    "local = attn(sliding_window_size=16, sinks=true, softcap=20)\n"
)
#: DSA's indexer as vLLM's DeepSeek-V3.2 kernels take it, the timeline's: 4 heads of
#: 64, its top 16 of the keys, so that the prompts and their decoding pick.
NATIVE_INDEXER = "indexer=indexer(num_heads=4, head_dim=64, top_k=16)"
NATIVE = f"attn = gqa(num_heads=4, num_kv_heads=2, rope=rope(), {NATIVE_INDEXER})\n"
#: csa compressing groups of 4 with half the dims rotated, at d_model 128, its
#: window and top-k small enough that the prompts and their decoding cross both.
SPARSE = (
    "attn = csa(num_heads=2, head_dim=64, compress_ratio=4, sliding_window_size=16,"
    " sinks=true, rope=rope(fraction=0.5), indexer=indexer(num_heads=4, head_dim=64,"
    " top_k=8))\n"
)
#: DeepSeek-V4.1's wiring of csa: a window-only layer, and one compressing pairs of
#: tokens, from a query latent, through a grouped low-rank output of 2 groups.
SPARSE_LATENT = (
    "window = csa(num_heads=4, head_dim=64, q_lora_rank=64, o_lora_rank=32,"
    " o_groups=2, sliding_window_size=8, sinks=true, rope=rope(fraction=0.25,"
    " layout=interleaved))\n"
    "pairs = window(compress_ratio=2, indexer=indexer(num_heads=2, head_dim=32,"
    " top_k=12))\n"
)

#: The hybrid's global layer, as the timeline's candidate has it, and a value
#: embedding of its 2 kv heads of 32.
GLOBAL = (
    "glob = gqa(num_heads=4, num_kv_heads=2, qk_norm=rmsnorm, gated=true,"
    " kv_conv_size=4)\nve = value_embed(features=64)\n"
)

#: Hymba's blocks: attention without a window (F) and with one, which writes its
#: keys and values to the tap kv (W), reads them (S) or takes them off (R); and one
#: without a window that writes them (G).
SHARING = (
    "local = gqa(num_heads=4, num_kv_heads=2, sliding_window_size=8, rope=rope())\n"
    "full = local(sliding_window_size=none)\n"
    + FFN
    + "layer(attn) = [rmsnorm attn] [rmsnorm ffn]\n"
    "F = layer(full)\nW = layer(local(kv=@kv))\nS = layer(local(kv=$kv))\n"
    "R = layer(local(kv=$kv!))\nG = layer(full(kv=@kv))\n"
)

#: The cases, by name: a model each, as bux source.
CASES = {
    "llama": LLAMA + "model = embed block^2 rmsnorm head",
    "qwen3": HEAD
    + "attn = gqa(num_heads=4, num_kv_heads=2, qk_norm=rmsnorm, rope=rope(base=1e6))\n"
    + FFN
    + BLOCK
    + "model = embed block^2 rmsnorm head(tied=true)",
    "gpt2": HEAD
    + "attn = gqa(num_heads=4, num_kv_heads=4, bias=true)\n"
    + "ffn = mlp(intermediate_size=256, activation=gelu, bias=true)\n"
    + "block = [layernorm(bias=true) attn] [layernorm(bias=true) ffn]\n"
    + "model = embed(max_positions=128) block^2 layernorm(bias=true) head(tied=true)",
    "rope": HEAD
    + FFN
    + "a = gqa(num_heads=4, num_kv_heads=1, rope=rope(fraction=0.5, layout=interleaved,"
    " scaling=pi(factor=4)))\n"
    + "b = gqa(num_heads=4, num_kv_heads=2, rope=rope(base=500000,"
    " scaling=llama3(factor=8, original_max_len=16)))\n"
    + "c = gqa(num_heads=4, num_kv_heads=2, rope=rope(scaling=yarn(factor=4,"
    " original_max_len=16)))\n"
    + "e = gqa(num_heads=4, num_kv_heads=2, rope=rope(scaling=yarn(factor=4,"
    " original_max_len=16, attention_factor_on=softmax)))\n"
    + "model = embed [rmsnorm a] [rmsnorm b] [rmsnorm ffn] [rmsnorm c] [rmsnorm e]"
    " rmsnorm head",
    "gemma": HEAD
    + "ffn = gated_mlp(intermediate_size=256, activation=gelu_tanh)\n"
    + "attn = gqa(num_heads=4, num_kv_heads=2, qk_norm=rmsnorm(zero_centered=true),"
    " softcap=50, rope=rope())\n"
    + "local = attn(sliding_window_size=8)\n"
    + "norm = rmsnorm(zero_centered=true)\n"
    + "block = [norm local norm] [norm ffn norm] [norm attn norm] [norm ffn norm]\n"
    + "model = embed(scale_by_sqrt_dim=true) block^2 norm head(tied=true, softcap=30)",
    "gpt-oss": HEAD
    + FFN
    + "attn = gqa(num_heads=4, num_kv_heads=2, sinks=true, bias=true,"
    " output_bias=false, rope=rope())\n"
    + "local = attn(sliding_window_size=8)\n"
    + "model = embed [rmsnorm local] [rmsnorm ffn] [rmsnorm attn] [rmsnorm ffn]"
    " rmsnorm head",
    "mqa": HEAD
    + "attn = gqa(num_heads=2, num_kv_heads=1, head_dim=64, rope=rope())\n"
    + FFN
    + BLOCK
    + "model = embed block^2 rmsnorm head(bias=true)",
    "graph": LLAMA.replace("block =", "shared block =")
    + "model = embed @x0 block^3 [rmsnorm (attn - 0.5 * ffn)]"
    " [$x0 scale(init=0.1, per_channel=true)] linear(features=128) gelu"
    " rmsnorm head",
    "extras": HEAD
    + "attn = gqa(num_heads=4, num_kv_heads=2, gated=true, fused_qkv=true,"
    " output_norm=rmsnorm, rope=rope())\n"
    + "laguna = gqa(num_heads=4, num_kv_heads=2, gated=true, gate_per_head=true,"
    " gate_activation=softplus, rope=rope())\n"
    + "ffn = gated_mlp(intermediate_size=256, activation=swish, limit=1, up_offset=1)\n"
    + "capped = mlp(intermediate_size=256, activation=relu2, up_softcap=2)\n"
    + "model = embed(norm=rmsnorm) [rmsnorm attn] [rmsnorm ffn] [rmsnorm laguna]"
    " [rmsnorm capped] rmsnorm head",
    # CPLM's copy pointer on the head: its mixture with the softmax is what vLLM
    # samples, the pointer's keys and tokens in a paged cache of the head's own.
    "cplm": LLAMA + "model = embed block^2 rmsnorm head(pointer=copy_pointer(dim=64))",
    "moe": HEAD
    + ATTN
    + EXPERT
    + "experts = moe(num_experts=8, top_k=2, num_shared_experts=1,"
    " shared_expert_gate=true, normalize_top_k=false, expert=dense)\n"
    + MOE_BLOCK
    + "model = embed block^2 rmsnorm head",
    "moe-grouped": HEAD
    + ATTN
    + EXPERT
    + "experts = moe(num_experts=8, top_k=2, num_shared_experts=1, router=sigmoid,"
    " routed_scaling_factor=2.5, num_groups=4, top_k_groups=2,"
    " load_balancing=aux_free, expert=dense)\n"
    + MOE_BLOCK
    + "model = embed block^2 rmsnorm head",
    "moe-relu2": HEAD
    + ATTN
    + "experts = moe(num_experts=8, top_k=2, router=sqrtsoftplus,"
    " expert=mlp(intermediate_size=64, activation=relu2))\n"
    + MOE_BLOCK
    + "model = embed block^2 rmsnorm head",
    "moe-clamped": HEAD
    + ATTN
    + "experts = moe(num_experts=8, top_k=2, router_bias=true, num_shared_experts=1,"
    " expert=gated_mlp(intermediate_size=64, activation=quick_gelu, bias=true,"
    " limit=1.0, up_offset=1.0))\n" + MOE_BLOCK + "model = embed block^2 rmsnorm head",
    "moe-latent": HEAD
    + ATTN
    + "experts = moe(num_experts=8, top_k=3, num_shared_experts=1, router=sigmoid,"
    " routed_scaling_factor=2.5, load_balancing=aux_free, latent_size=64,"
    " latent_input_norm=rmsnorm, latent_output_norm=rmsnorm,"
    " expert=mlp(intermediate_size=64, activation=gelu, bias=true))\n"
    + MOE_BLOCK
    + "model = embed block^2 rmsnorm head",
    "moe-sink": HEAD
    + ATTN
    + EXPERT
    + "experts = moe(num_experts=8, top_k=2, num_shared_experts=2,"
    " shared_expert_sink=true, router=sigmoid, routed_scaling_factor=2.5,"
    " load_balancing=aux_free, expert=dense)\n"
    + MOE_BLOCK
    + "model = embed block^2 rmsnorm head",
    # Kimi K3's SiTU-GLU experts, up soft-capped (low, so that the cap bites).
    "moe-situ": HEAD
    + ATTN
    + "experts = moe(num_experts=8, top_k=2, num_shared_experts=1, router=sigmoid,"
    " load_balancing=aux_free, expert=gated_mlp(intermediate_size=64,"
    " activation=situ, up_softcap=0.5))\n"
    + MOE_BLOCK
    + "model = embed block^2 rmsnorm head",
    # Muse Spark's routed experts: each one's output normalized, in a latent.
    "moe-norm": HEAD
    + ATTN
    + EXPERT
    + "experts = moe(num_experts=8, top_k=2, num_shared_experts=1, router=sigmoid,"
    " routed_scaling_factor=2.5, load_balancing=aux_free, latent_size=32,"
    " latent_input_norm=rmsnorm, expert_norm=rmsnorm(eps=1e-8, zero_centered=true,"
    " scale_input=true), expert=dense)\n"
    + MOE_BLOCK
    + "model = embed block^2 rmsnorm head",
    "engram": LLAMA
    + "memory = engram(head_dim=8, table_size=12288)\n"
    + "model = embed block memory block rmsnorm head(tied=true)",
    # MiniMax-01's lightning attention, its decay by depth, beside softmax attention.
    "lightning": HEAD
    + ATTN
    + FFN
    + "mix = lightning(num_heads=2, head_dim=64)\n"
    + "model = embed [rmsnorm mix] [rmsnorm ffn] [rmsnorm mix] [rmsnorm ffn]"
    " [rmsnorm attn] [rmsnorm ffn] rmsnorm head",
    # Lightning attention beyond MiniMax-01's: a conv, value heads of their own.
    "lightning-conv": HEAD
    + ATTN
    + FFN
    + "mix = lightning(num_heads=2, num_v_heads=4, head_dim=32, v_head_dim=16,"
    " conv_size=4)\n"
    + "model = embed [rmsnorm mix] [rmsnorm ffn] [rmsnorm mix] [rmsnorm ffn]"
    " [rmsnorm attn] [rmsnorm ffn] rmsnorm head",
    "mamba": HEAD
    + ATTN
    + FFN
    + "ssm = mamba(state_size=16)\n"
    + "model = embed [rmsnorm ssm] [rmsnorm ffn] [rmsnorm attn] [rmsnorm ffn]"
    " [rmsnorm ssm] [rmsnorm ffn] rmsnorm head",
    "mamba2": HEAD
    + ATTN
    + FFN
    + "ssm = mamba2(state_size=16, head_dim=16, num_groups=2)\n"
    + "model = embed [rmsnorm ssm] [rmsnorm ffn] [rmsnorm attn] [rmsnorm ffn]"
    " [rmsnorm ssm] [rmsnorm ffn] rmsnorm head",
    "gdn": GATED_HEAD
    + "delta = gdn(num_heads=2, num_v_heads=4, head_dim=32)\n"
    + LINEAR_HYBRID,
    "kda": GATED_HEAD
    + "delta = kda(num_heads=2, num_v_heads=4, head_dim=32, gate_lower_bound=-5)\n"
    + LINEAR_HYBRID,
    "kda-kimi": GATED_HEAD
    + "delta = kda(num_heads=4, head_dim=32, output_gate_rank=32, norm_eps=1e-5)\n"
    + LINEAR_HYBRID,
    # A hybrid at a width no multiple of 8: KDA's fused projection is
    # 550 wide (padded to 576), beside gated attention and a sigmoid MoE.
    "kda-odd": GATED_HEAD.replace("d_model = 128", "d_model = 136")
    + EXPERT
    + "experts = moe(num_experts=8, top_k=2, num_shared_experts=1, router=sigmoid,"
    " load_balancing=aux_free, expert=dense)\n"
    + "delta = kda(num_heads=2, num_v_heads=6, head_dim=32, gate_lower_bound=-5)\n"
    + "model = embed [rmsnorm delta] [rmsnorm experts] [rmsnorm delta] [rmsnorm ffn]"
    " [rmsnorm attn] [rmsnorm experts] rmsnorm head",
    "gdn2": GATED_HEAD
    + "delta = gdn2(num_heads=2, num_v_heads=4, head_dim=32, gate_lower_bound=-5)\n"
    + LINEAR_HYBRID,
    "gdn2-paper": GATED_HEAD
    + "delta = gdn2(num_heads=4, head_dim=32, gate_rank=8, output_gate_rank=16,"
    " negative_eigenvalues=true, output_gate=sigmoid)\n" + LINEAR_HYBRID,
    "deltanet": GATED_HEAD
    + "delta = deltanet(num_heads=2, num_v_heads=4, head_dim=32)\n"
    + LINEAR_HYBRID,
    # Gated linear attention, without a conv (its default) and with one.
    "gla": GATED_HEAD
    + "delta = gla(num_heads=2, num_v_heads=4, head_dim=32, gate_rank=8)\n"
    + LINEAR_HYBRID,
    "gla-conv": GATED_HEAD
    + "delta = gla(num_heads=2, head_dim=32, conv_size=4, output_gate=sigmoid)\n"
    + LINEAR_HYBRID,
    # GDN and KDA beyond vLLM's kernels, by bux's own rule: no conv or output gate,
    # values narrower than q and k, a lower bound under -5.
    "gdn-plain": GATED_HEAD
    + "delta = gdn(num_heads=2, num_v_heads=4, head_dim=32, conv_size=none,"
    " output_gate=none)\n" + LINEAR_HYBRID,
    "kda-plain": GATED_HEAD
    + "delta = kda(num_heads=2, num_v_heads=4, head_dim=32, v_head_dim=16,"
    " conv_size=none, output_gate_rank=16, gate_lower_bound=-8)\n" + LINEAR_HYBRID,
    # GDN-2 without a conv: q, k and v SiLU'd alone.
    "gdn2-plain": GATED_HEAD
    + "delta = gdn2(num_heads=2, num_v_heads=4, head_dim=32, conv_size=none)\n"
    + LINEAR_HYBRID,
    "rwkv7": GATED_HEAD + "delta = rwkv7(num_heads=2, head_dim=64)\n" + LINEAR_HYBRID,
    "rwkv7-residual": GATED_HEAD
    + "mix = rwkv7(num_heads=4, num_v_heads=8, head_dim=32, decay_rank=16, a_rank=16,"
    " v_rank=8, gate_rank=32)\n"
    + "model = embed [rmsnorm mix(v=@v0)] [rmsnorm ffn] [rmsnorm attn] [rmsnorm ffn]"
    " [rmsnorm mix(v=$v0)] [rmsnorm ffn] [rmsnorm mix(v=$v0!)] rmsnorm head",
    "hc": LLAMA + "residual = hc(expansion=4)\nmodel = embed block^2 rmsnorm head",
    "attnres": LLAMA
    + "residual = attnres(block_size=3)\nmodel = embed block^2 rmsnorm head",
    "mhc": LLAMA + "residual = mhc(expansion=4)\nmodel = embed block^2 rmsnorm head",
    "mhc-single": LLAMA + "residual = mhc(expansion=4, single_pass=true)\n"
    "model = embed block^2 rmsnorm head",
    "gr": HEAD
    + ATTN
    + FFN
    + "residual = gr(expansion=4, rank=16)\nmodel = embed ([attn] [ffn])^2 head(tied=true)",
    "vpgate": LLAMA
    + "residual = vpgate(temperature=0.3)\nmodel = embed block^2 rmsnorm head",
    "value-residual": HEAD
    + ATTN
    + FFN
    + "both = attn(value_residual=learned)\n"
    + "one = attn(value_residual=convex)\n"
    + "model = embed [rmsnorm attn(v=@v1)] [rmsnorm ffn] [rmsnorm attn(v=$v1)]"
    " [rmsnorm both(v=$v1)] [rmsnorm ffn] [rmsnorm one(v=$v1!)] rmsnorm head",
    # Hymba's cross-layer KV sharing: a windowed layer's keys and values read by
    # the next one or two, then a full layer's by one with a window of its own.
    "kv-share": HEAD + SHARING + "model = embed F W S R W R G R rmsnorm head",
    # Inkling's attention: a relative bias, a conv on keys and values, no RoPE, and
    # a length scale on the global layer's logits, its window-only layers biased
    # over their window.
    "relbias": HEAD
    + FFN
    + "attn = gqa(num_heads=4, num_kv_heads=2, qk_norm=rmsnorm, kv_conv_size=4,"
    " relative_bias=relbias(rank=4, extent=16), log_scale=0.5, log_scale_floor=8)\n"
    + "local = attn(sliding_window_size=8, relative_bias.extent=8, log_scale=none)\n"
    + "model = embed [rmsnorm local] [rmsnorm ffn] [rmsnorm attn] [rmsnorm ffn]"
    " rmsnorm head",
    # Inkling's shortcut: a conv on every branch's output, y + conv(y).
    "conv": HEAD
    + ATTN
    + FFN
    + "shortcut = conv(size=4, bias=true)\n"
    + "model = embed [rmsnorm attn [shortcut]] [rmsnorm ffn [shortcut]]"
    " [rmsnorm attn [shortcut]] [rmsnorm ffn] rmsnorm head",
    # LFM2's double-gated short conv beside attention.
    "short-conv": HEAD
    + ATTN
    + FFN
    + "mix = short_conv(conv_size=3, bias=true)\n"
    + "model = embed [rmsnorm mix] [rmsnorm ffn] [rmsnorm attn] [rmsnorm ffn]"
    " [rmsnorm mix] [rmsnorm ffn] rmsnorm head",
    "dyt": HEAD
    + ATTN
    + FFN
    + "model = embed [dyt attn] [dyt ffn] [dyt attn] [dyt ffn] dyt head",
    "mla": HEAD + LATENT + FFN + BLOCK + "model = embed block^2 rmsnorm head",
    "mla-moe": HEAD
    + LATENT
    + FFN
    + EXPERT
    + "experts = moe(num_experts=8, top_k=2, num_shared_experts=1, router=sigmoid,"
    " routed_scaling_factor=2.5, load_balancing=aux_free, expert=dense)\n"
    + "model = embed [rmsnorm attn] [rmsnorm ffn] [rmsnorm attn] [rmsnorm experts]"
    " rmsnorm head",
    "mla-chunked": HEAD + LATENT + FFN + BLOCK + "model = embed block^2 rmsnorm head",
    "mla-deepseek": HEAD
    + DEEPSEEK
    + "gated = attn(gated=true, gate_per_head=true, output_norm=rmsnorm)\n"
    + FFN
    + "model = embed [rmsnorm attn] [rmsnorm ffn] [rmsnorm gated] [rmsnorm ffn]"
    " rmsnorm head",
    # DeepSeek-V3.2's sparse MLA: the indexer's queries from the query latent.
    "mla-dsa": HEAD
    + SPARSE_MLA
    + FFN
    + "model = embed [rmsnorm attn] [rmsnorm ffn] [rmsnorm local] [rmsnorm ffn]"
    " rmsnorm head",
    # DeepSeek-V3.2's indexer on vLLM's kernels (under
    # BUX_VLLM_SPARSE_KERNELS=native), in two layers.
    "mla-dsa-native": HEAD
    + SPARSE_MLA.replace(
        "indexer=indexer(num_heads=2, head_dim=16, top_k=8)", NATIVE_INDEXER
    )
    + FFN
    + "model = embed [rmsnorm attn] [rmsnorm ffn] [rmsnorm attn] [rmsnorm ffn]"
    " rmsnorm head",
    # GLM-5.3's: one layer's picks shared with the next two.
    "mla-share": HEAD
    + SPARSE_MLA
    + FFN
    + "model = embed [rmsnorm attn(index=@topk)] [rmsnorm ffn]"
    " [rmsnorm attn(index=$topk)] [rmsnorm ffn] [rmsnorm local(index=$topk!)]"
    " [rmsnorm ffn] rmsnorm head",
    "query-scale": HEAD
    + FFN
    + "beam = gqa(num_heads=4, num_kv_heads=2, power_scale=0.5, power_scale_length=8,"
    " gated=true, gate_head_bias=true)\n"
    + "inkling = gqa(num_heads=4, num_kv_heads=2, log_scale=0.5, log_scale_floor=8,"
    " softcap=20, rope=rope())\n"
    + "model = embed [rmsnorm beam] [rmsnorm ffn] [rmsnorm inkling] [rmsnorm ffn]"
    " rmsnorm head",
    # XSA: each head's output less its part along its own value, all of it and a
    # learned share of it, with a window, sinks and the value residual.
    "xsa": HEAD
    + FFN
    + "attn = gqa(num_heads=4, num_kv_heads=2, exclusive=true, rope=rope())\n"
    + "share = attn(exclusive=learned, sliding_window_size=8, sinks=true,"
    " value_residual=learned)\n"
    + "model = embed [rmsnorm attn(v=@v1)] [rmsnorm ffn] [rmsnorm share(v=$v1)]"
    " [rmsnorm ffn] rmsnorm head",
    # The timeline's hybrid under hc, its global layers with learned XSA: no
    # positions, QK-norm, a gate, a conv on keys and values and the value residual,
    # 4 query heads to a kv head (each rank's copy of it under TP), beside KDA.
    "xsa-hybrid": HEAD
    + FFN
    + "attn = gqa(num_heads=4, num_kv_heads=1, qk_norm=rmsnorm, gated=true,"
    " kv_conv_size=4, exclusive=learned)\n"
    + "delta = kda(num_heads=2, num_v_heads=4, head_dim=32, gate_lower_bound=-5)\n"
    + "residual = hc(expansion=4)\n"
    + "model = embed [rmsnorm delta] [rmsnorm ffn] [rmsnorm attn(v=@v1)] [rmsnorm ffn]"
    " [rmsnorm delta] [rmsnorm ffn] [rmsnorm attn(v=$v1)] [rmsnorm ffn] rmsnorm head",
    # HyGA's gates: as the paper's main runs have them (the input's gate, the head's
    # own output's at its width, the cross-head gate, sinks), and low-rank with a
    # head bias.
    "hyga": HEAD
    + FFN
    + "attn = gqa(num_heads=4, num_kv_heads=2, gated=true, gate_output_rank=32,"
    " gate_cross_head=true, sinks=true, rope=rope())\n"
    + "low = attn(gate_input_rank=8, gate_output_rank=8, gate_head_bias=true,"
    " sinks=false)\n"
    + "model = embed [rmsnorm attn] [rmsnorm ffn] [rmsnorm low] [rmsnorm ffn]"
    " rmsnorm head",
    # xsa-hybrid with HyGA's gates on its global layers too.
    "hyga-hybrid": HEAD
    + FFN
    + "attn = gqa(num_heads=4, num_kv_heads=1, qk_norm=rmsnorm, gated=true,"
    " gate_output_rank=32, gate_cross_head=true, kv_conv_size=4, exclusive=learned)\n"
    + "delta = kda(num_heads=2, num_v_heads=4, head_dim=32, gate_lower_bound=-5)\n"
    + "residual = hc(expansion=4)\n"
    + "model = embed [rmsnorm delta] [rmsnorm ffn] [rmsnorm attn(v=@v1)] [rmsnorm ffn]"
    " [rmsnorm delta] [rmsnorm ffn] [rmsnorm attn(v=$v1)] [rmsnorm ffn] rmsnorm head",
    "dsa": HEAD + FFN + "attn = gqa(num_heads=4, num_kv_heads=2, rope=rope(),"
    " indexer=indexer(num_heads=2, head_dim=16, top_k=8))\n"
    + "local = attn(sliding_window_size=16, sinks=true, softcap=20)\n"
    + "model = embed [rmsnorm attn] [rmsnorm ffn] [rmsnorm local] [rmsnorm ffn]"
    " rmsnorm head",
    "dsa-conv": HEAD
    + FFN
    + "attn = gqa(num_heads=4, num_kv_heads=2, kv_conv_size=4, qk_norm=rmsnorm,"
    " gated=true, indexer=indexer(num_heads=2, head_dim=16, top_k=8))\n"
    + "model = embed [rmsnorm attn(v=@v1)] [rmsnorm ffn] [rmsnorm attn(v=$v1!)]"
    " [rmsnorm ffn] rmsnorm head",
    # Shared picks, as GLM-5.3's: one layer's indexer picks the keys the next two
    # attend over, the last with a window of its own.
    "dsa-share": HEAD + FFN + "attn = gqa(num_heads=4, num_kv_heads=2, rope=rope(),"
    " indexer=indexer(num_heads=2, head_dim=16, top_k=8))\n"
    + "model = embed [rmsnorm attn(index=@topk)] [rmsnorm ffn] [rmsnorm attn(index=$topk)]"
    " [rmsnorm ffn] [rmsnorm attn(sliding_window_size=16, index=$topk!)] [rmsnorm ffn]"
    " rmsnorm head",
    # DSA on vLLM's DeepSeek-V3.2 kernels (in 16-bit models): the timeline's
    # indexer of 4 heads of 64, a layer that reads its picks, one that indexes its own.
    "dsa-native": HEAD
    + FFN
    + NATIVE
    + "model = embed [rmsnorm attn(index=@topk)] [rmsnorm ffn] [rmsnorm attn]"
    " [rmsnorm ffn] [rmsnorm attn(index=$topk!)] [rmsnorm ffn] rmsnorm head",
    # The factorial's S arms: DSA on the hybrid's global layer (QK-norm, a gate, the
    # conv on keys and values) beside KDA, which takes blocks of many tokens.
    "dsa-hybrid": HEAD
    + FFN
    + NATIVE.replace("rope=rope()", "kv_conv_size=4, qk_norm=rmsnorm, gated=true")
    + "delta = kda(num_heads=2, num_v_heads=4, head_dim=32, gate_lower_bound=-5)\n"
    + LINEAR_HYBRID,
    # MSA's picks, of blocks and per kv head's group, shared likewise.
    "msa-share": HEAD
    + FFN
    + "attn = gqa(num_heads=4, num_kv_heads=2, rope=rope(), indexer=indexer("
    "num_heads=2, head_dim=16, top_k=16, block_size=4, pool=max, per_group=true,"
    " relu=false, weighted=false, norm=rmsnorm))\n"
    + "model = embed [rmsnorm attn(index=@topk)] [rmsnorm ffn]"
    " [rmsnorm attn(index=$topk!)] [rmsnorm ffn] rmsnorm head",
    "msa": HEAD
    + FFN
    + "attn = gqa(num_heads=4, num_kv_heads=2, rope=rope(), indexer=indexer("
    "num_heads=2, head_dim=16, top_k=16, block_size=4, pool=max, per_group=true,"
    " relu=false, weighted=false, norm=rmsnorm))\n"
    + BLOCK
    + "model = embed block^2 rmsnorm head",
    "qsa": HEAD
    + FFN
    + "attn = gqa(num_heads=4, num_kv_heads=2, rope=rope(fraction=0.5),"
    " indexer=indexer(num_heads=2, head_dim=16, top_k=16, block_size=4,"
    " weighted=false, norm=rmsnorm))\n" + BLOCK + "model = embed block^2 rmsnorm head",
    # MSA on vLLM's MiniMax-M3 kernels: blocks of 128, an index head a kv head.
    # The prompts fit in a block, so this checks their attention, not their picks
    # (scripts/vllm_native.py checks those).
    "msa-native": HEAD
    + FFN
    + "attn = gqa(num_heads=4, num_kv_heads=2, rope=rope(fraction=0.5), indexer=indexer("
    "num_heads=2, head_dim=32, top_k=256, block_size=128, pool=max, per_group=true,"
    " relu=false, weighted=false, norm=rmsnorm))\n"
    + BLOCK
    + "model = embed block^2 rmsnorm head",
    # QSA on vLLM's DSA kernels: blocks of 4, the timeline's 4 heads of 64.
    "qsa-native": HEAD
    + FFN
    + "attn = gqa(num_heads=4, num_kv_heads=2, rope=rope(fraction=0.5),"
    " indexer=indexer(num_heads=4, head_dim=64, top_k=16, block_size=4,"
    " weighted=false, norm=rmsnorm))\n" + BLOCK + "model = embed block^2 rmsnorm head",
    "qk-norm-all-heads": HEAD
    + FFN
    + "rms = gqa(num_heads=4, num_kv_heads=2, qk_norm=rmsnorm, qk_norm_all_heads=true,"
    " rope=rope())\n"
    + "ln = rms(qk_norm=layernorm(bias=true))\n"
    + "model = embed [rmsnorm rms] [rmsnorm ffn] [rmsnorm ln] [rmsnorm ffn]"
    " rmsnorm head",
    # Value embeddings on the hybrid's global layer (QK-norm, a gate, the conv on
    # keys and values, no positions): a table a layer, gated, beside the value
    # residual; then one table two layers read, ungated, under hc.
    "value-embed": HEAD
    + FFN
    + GLOBAL
    + "model = embed ve(ve=@t) [rmsnorm glob(v=@v1, ve=$t!)] [rmsnorm ffn]"
    " [rmsnorm ve(ve=@u) glob(v=$v1!, ve=$u!)] [rmsnorm ffn] rmsnorm head",
    "value-embed-shared": HEAD
    + "residual = hc(expansion=2)\n"
    + FFN
    + GLOBAL
    + "plain = glob(value_embed_gate=none)\n"
    + "model = embed ve(ve=@t) [rmsnorm plain(ve=$t)] [rmsnorm ffn]"
    " [rmsnorm plain(ve=$t!)] [rmsnorm ffn] rmsnorm head",
    # Gemma 4's values from the keys: with RoPE, a window and a fused projection,
    # beside the value residual; then on the hybrid's global layer (QK-norm, a gate,
    # the conv on keys and values, no positions).
    "k-eq-v": HEAD
    + FFN
    + "attn = gqa(num_heads=4, num_kv_heads=2, k_eq_v=true, qk_norm=rmsnorm,"
    " rope=rope())\n"
    + "local = attn(sliding_window_size=8, fused_qkv=true)\n"
    + "model = embed [rmsnorm attn(v=@v1)] [rmsnorm ffn] [rmsnorm local]"
    " [rmsnorm ffn] [rmsnorm attn(v=$v1!)] rmsnorm head",
    "k-eq-v-nope": HEAD
    + FFN
    + "glob = gqa(num_heads=4, num_kv_heads=2, k_eq_v=true, qk_norm=rmsnorm,"
    " gated=true, kv_conv_size=4)\n"
    + "model = embed [rmsnorm glob] [rmsnorm ffn] [rmsnorm glob] [rmsnorm ffn]"
    " rmsnorm head",
    "kv-conv": HEAD
    + FFN
    + "attn = gqa(num_heads=4, num_kv_heads=2, kv_conv_size=4, rope=rope())\n"
    + "local = attn(sliding_window_size=8, qk_norm=rmsnorm)\n"
    + "model = embed [rmsnorm attn(v=@v1)] [rmsnorm ffn] [rmsnorm local]"
    " [rmsnorm ffn] [rmsnorm attn(v=$v1!)] rmsnorm head",
    "csa": HEAD + SPARSE + FFN + BLOCK + "model = embed block^2 rmsnorm head",
    # DeepSeek-V4.1's ports: a layer that compresses pairs, indexes them and makes a
    # candidate pool; one that reads its latents and picks; one that re-indexes
    # them inside the pool.
    "csa-ports": HEAD
    + "pairs = csa(num_heads=2, head_dim=64, compress_ratio=2, sliding_window_size=16,"
    " sinks=true, rope=rope(fraction=0.5), indexer=indexer(num_heads=4, head_dim=64,"
    " top_k=8, candidate_block_size=2, candidate_top_k=4))\n"
    + FFN
    + "model = embed [rmsnorm pairs(kv=@kv, index=@topk, candidates=@pool)]"
    " [rmsnorm ffn] [rmsnorm pairs(kv=$kv, index=$topk)] [rmsnorm ffn]"
    " [rmsnorm pairs(kv=$kv!, index=@topk, candidates=$pool!)] [rmsnorm ffn]"
    " rmsnorm head",
    "csa-latent": HEAD
    + SPARSE_LATENT
    + FFN
    + "model = embed [rmsnorm window] [rmsnorm ffn] [rmsnorm pairs] [rmsnorm ffn]"
    " rmsnorm head",
}

#: The engine settings of a case of its own: mla-chunked prefills 16 tokens a step,
#: so that each later chunk of a prompt attends to the latents its earlier ones
#: cached (vLLM's chunked-context prefill, as a cached prefix does).
ENGINES = {"mla-chunked": {"max_num_batched_tokens": 16, "max_num_seqs": 4}}

LENGTHS = (3, 17, 40, 64)  # the prompts', in one batch: within and past a window
STEPS = 16  # tokens decoded after each
MAX_LEN = 128


def main() -> None:
    """Parse the command line and run its step."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="step", required=True)
    for name in ("export", "run", "compare", "run-one"):
        s = sub.add_parser(name)
        s.add_argument("dir", type=Path)
        s.add_argument("--only", nargs="*", default=None)
        s.add_argument("--tag", default="vllm")
    sub.choices["export"].add_argument("--dtype", default="float32")
    for name in ("run", "run-one"):
        sub.choices[name].add_argument("--tp", type=int, default=1)
        sub.choices[name].add_argument("--eager", action="store_true")
        sub.choices[name].add_argument("--moe-backend", default="auto")
        sub.choices[name].add_argument("--mamba-ssm-cache-dtype", default="auto")
        sub.choices[name].add_argument("--attention-backend", default=None)
    sub.choices["compare"].add_argument("--tolerance", type=float, default=1e-3)
    sub.choices["compare"].add_argument("--margin", type=float, default=0.0)
    sub.choices["compare"].add_argument("--fp8-indexer", action="store_true")
    args = p.parse_args()
    if getattr(args, "fp8_indexer", False):
        fp8_indexer()
    cases = [c for c in CASES if args.only is None or c in args.only]
    match args.step:
        case "export":
            for name in cases:
                export(name, args.dir / name, args.dtype)
        case "run":
            for name in cases:
                run_alone(args, name)
        case "run-one":
            run(args.dir / cases[0], args.tag, engine(args))
        case "compare":
            bad = [n for n in cases if not compare(n, args)]
            sys.exit(1 if bad else 0)


# --- export, in bux's environment ---------------------------------------------------


def export(name: str, out: Path, dtype: str, cases: dict[str, str] = CASES) -> None:
    """Export case `name`'s model, from `cases`, to `out` in `dtype`, with its
    prompts."""
    import torch

    from bux.export.huggingface import save
    from bux.lang import compile

    arch = compile(cases[name])
    model = reference_model(arch, dtype)
    with placeholders(arch):
        save(arch, model, out, dtype=getattr(torch, dtype))
    _add_router_biases(model, arch, out)
    rng = np.random.default_rng(0)
    prompts = [rng.integers(0, arch.vocab_size, n).tolist() for n in LENGTHS]
    meta = {"name": name, "dtype": dtype, "prompts": prompts, "steps": STEPS}
    (out / "parity.json").write_text(json.dumps(meta))
    print(f"exported {name} to {out}")


def reference_model(arch: Any, dtype: str) -> Any:
    """Return bux's model of `arch`: fresh weights, perturbed, with the aux-free
    biases drawn too, all rounded to `dtype` (but kept in float32)."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from bux.architecture import Model
    from bux.export.check import perturbed

    model = perturbed(Model(arch, rngs=nnx.Rngs(0)), 1)
    model.eval()  # inference: no dropout
    for path, var in nnx.iter_graph(model):
        if path and path[-1] == "router_bias" and isinstance(var, nnx.Variable):
            key = jax.random.key(len(path))
            var[...] = 0.1 * jax.random.normal(key, var[...].shape)
    state = nnx.state(model, nnx.Variable)
    rounded = jax.tree.map(
        lambda x: (
            x.astype(getattr(jnp, dtype)).astype(x.dtype)
            if jnp.issubdtype(x.dtype, jnp.floating)
            else x
        ),
        state,
    )
    nnx.update(model, rounded)
    return model


class placeholders:
    """Stand in for the transformers classes of layers that only the vLLM plugin
    exports so far, so that `save` writes their weights by the generic rule: their
    ``modeling_bux.py`` does not run, but vLLM does not read it."""

    STATES: ClassVar = {"mamba2": "mamba", "gdn": "linear_attention"}

    def __init__(self, arch: Any):
        from bux.export import codegen

        self.classes = codegen.layer_classes()
        kinds = {layer.config.type for layer in arch.layers()}
        self.added = [k for k in kinds if k not in self.classes]

    def __enter__(self) -> None:
        for kind in self.added:
            attrs: dict[str, Any] = {"__module__": "placeholder.vllm_only"}
            if kind in self.STATES:
                attrs["cache_type"] = staticmethod(_kind_of(self.STATES[kind]))
            self.classes[kind] = type(kind.title().replace("_", ""), (), attrs)

    def __exit__(self, *exc: object) -> None:
        for kind in self.added:
            del self.classes[kind]


def _kind_of(state: str) -> Any:
    """A placeholder's ``cache_type``: the cache layer `state` of every call."""

    def cache_type(c: dict, ports: list) -> str:
        return state

    return cache_type


def _add_router_biases(model: Any, arch: Any, out: Path) -> None:
    """Add the aux-free router biases to the export's weights if `save` left them
    out (they are not parameters), as ``<moe>.router_bias``."""
    import torch
    from safetensors.torch import load_file, save_file

    from bux.export.plan import plan
    from bux.export.weights import module_at

    file = out / "model.safetensors"
    weights = load_file(file)
    dtype = next(iter(weights.values())).dtype
    added = {}
    for call in (c for c in plan(arch).calls if c.first):
        module = module_at(arch.graph, model.graph, call.trail)
        bias = getattr(module, "router_bias", None)
        name = f"{call.path}.router_bias"
        if bias is not None and name not in weights:
            added[name] = torch.from_numpy(np.array(bias[...])).to(dtype)
    if added:
        save_file(weights | added, file, metadata={"format": "pt"})


# --- run, in vLLM's environment ------------------------------------------------------


def run_alone(args: argparse.Namespace, name: str) -> None:
    """Run case `name` in a process of its own, as each vLLM engine wants one."""
    cmd = [sys.executable, __file__, "run-one", str(args.dir), "--only", name]
    cmd += ["--tag", args.tag, "--tp", str(args.tp), "--moe-backend", args.moe_backend]
    cmd += ["--mamba-ssm-cache-dtype", args.mamba_ssm_cache_dtype]
    if args.attention_backend is not None:
        cmd += ["--attention-backend", args.attention_backend]
    subprocess.run(cmd + ["--eager"] * args.eager, check=False)


def engine(args: argparse.Namespace) -> dict:
    """vLLM's engine arguments the command line sets."""
    return {
        "tensor_parallel_size": args.tp,
        "enforce_eager": args.eager,
        "moe_backend": args.moe_backend,
        "mamba_ssm_cache_dtype": args.mamba_ssm_cache_dtype,
        "attention_backend": args.attention_backend,
    }


def run(where: Path, tag: str, engine: dict) -> None:
    """Decode the prompts of the export at `where` greedily with vLLM, its engine
    made with the arguments `engine` (`serve`); keep the tokens and every
    position's log-probabilities in ``<tag>.npz``."""
    decode(serve(where, engine), where, tag)


def serve(where: Path, engine: dict) -> Any:
    """vLLM's engine of the export at `where`, made with the arguments `engine` and
    the case's own (`ENGINES`)."""
    from vllm import LLM

    meta = json.loads((where / "parity.json").read_text())
    vocab = json.loads((where / "config.json").read_text())["vocab_size"]
    return LLM(
        model=str(where),
        dtype=meta["dtype"],
        skip_tokenizer_init=True,
        max_model_len=MAX_LEN,
        max_logprobs=vocab,
        gpu_memory_utilization=0.5,
        seed=0,
        **engine | ENGINES.get(meta["name"], {}),
    )


def decode(llm: Any, where: Path, tag: str) -> None:
    """Decode the prompts of the export at `where` greedily with `llm`; keep the
    tokens and every position's log-probabilities in ``<tag>.npz``."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    meta = json.loads((where / "parity.json").read_text())
    vocab = json.loads((where / "config.json").read_text())["vocab_size"]
    params = SamplingParams(
        temperature=0.0,
        max_tokens=meta["steps"],
        logprobs=vocab,
        prompt_logprobs=vocab,
        ignore_eos=True,
        detokenize=False,
    )
    prompts = [TokensPrompt(prompt_token_ids=p) for p in meta["prompts"]]
    outs = llm.generate(prompts, params)
    arrays: dict[str, Any] = {}
    for i, out in enumerate(outs):
        gen = out.outputs[0]
        rows = [*out.prompt_logprobs[1:], *gen.logprobs]
        arrays[f"tokens{i}"] = np.array(gen.token_ids)
        arrays[f"logprobs{i}"] = _dense(rows, vocab)
    np.savez(where / f"{tag}.npz", **arrays)
    print(f"ran {meta['name']}: {tag}, {len(outs)} prompts")


def _dense(rows: list[dict], vocab: int) -> np.ndarray:
    """vLLM's log-probabilities of each position, by token, as an array."""
    out = np.full((len(rows), vocab), -np.inf, np.float32)
    for i, row in enumerate(rows):
        for token, lp in row.items():
            out[i, token] = lp.logprob
    return out


# --- compare, in bux's environment ---------------------------------------------------


def fp8_indexer() -> None:
    """Have bux's indexer score as vLLM's DSA kernels do (``compare
    --fp8-indexer``, for the cases they run): each query head and each key it
    ranks by (QSA's pooled block keys among them) rounded to FP8 e4m3 at a scale of
    its largest magnitude over 448, as DeepSeek-V3.2's indexer quantizes them
    (`bux_vllm.dsa`)."""
    import jax.numpy as jnp

    from bux.architecture.attention import indexer

    queries, ranking = indexer.Indexer._queries, indexer.Indexer._ranking

    def rounded(x: Any) -> Any:
        scale = jnp.maximum(jnp.abs(x).max(-1, keepdims=True), 1e-4) / 448
        return (x / scale).astype(jnp.float8_e4m3fn).astype(x.dtype) * scale

    def fp8_queries(self: Any, query: Any, pos: Any) -> Any:
        return rounded(queries(self, query, pos))

    def fp8_ranking(
        self: Any, x: Any, query: Any, pos: Any, keys: Any, end: Any
    ) -> Any:
        return ranking(self, x, query, pos, rounded(keys), end)

    indexer.Indexer._queries = fp8_queries
    indexer.Indexer._ranking = fp8_ranking


def compare(name: str, args: argparse.Namespace) -> bool:
    """Compare case `name`'s vLLM run with bux; print a line, return whether they
    agree.

    vLLM agrees where its log-probabilities are within `--tolerance` of bux's in
    float32, and each token it picked within `--margin` of bux's best. In a lower
    dtype bux runs in it too (``bux.train.trainer.cast``), on the same tokens, and
    vLLM may be off by up to twice as much as bux is from itself in float32; and it
    may pick a token whose log-probability is that far, bux's own error, from the
    best, as a near tie two such runs may break either way.
    """
    from bux.lang import compile
    from bux.train.trainer import cast

    where = args.dir / name
    if not (where / f"{args.tag}.npz").exists():
        print(f"{name:12s} {args.tag:16s} no run")
        return False
    meta = json.loads((where / "parity.json").read_text())
    run = np.load(where / f"{args.tag}.npz")
    model = reference_model(compile(CASES[name]), meta["dtype"])
    low = None if meta["dtype"] == "float32" else cast(model, meta["dtype"])
    vllm, own = Agreement(), Agreement()
    for i, prompt in enumerate(meta["prompts"]):
        tokens, starts = run[f"tokens{i}"], len(prompt) - 1
        sequence = [*prompt, *tokens.tolist()]
        want = _logprobs(model, sequence)[:-1]
        vllm.add(run[f"logprobs{i}"], want, starts, tokens)
        if low is not None:
            got = _logprobs(low, sequence)[:-1]
            own.add(got, want, starts, got[starts:].argmax(-1))
    error, gap = max(args.tolerance, 2 * own.error), max(args.margin, own.error)
    ok = vllm.error <= error and vllm.gap <= gap
    print(
        f"{name:12s} {args.tag:16s} {meta['dtype']:9s} max |dlogprob| "
        f"{vllm.error:.2e}{own.note(own.error)}  greedy "
        f"{'same' if vllm.same else 'differs'} (worst gap {vllm.gap:.2e}"
        f"{own.note(own.gap)})  {'ok' if ok else 'DIFFERS'}"
    )
    return ok


@dataclass
class Agreement:
    """How a run's log-probabilities and picks agree with bux's in float32.

    Attributes:
        error: The largest difference of a log-probability.
        gap: The largest gap between bux's best log-probability at a step and that
            of the token picked.
        same: Whether every pick is bux's.
        seen: Whether it measured anything.
    """

    error: float = 0.0
    gap: float = 0.0
    same: bool = True
    seen: bool = False

    def add(
        self, got: np.ndarray, want: np.ndarray, starts: int, picks: np.ndarray
    ) -> None:
        """Measure a prompt's positions: the log-probabilities `got` against `want`,
        and the `picks` of its steps, from position `starts` on."""
        steps = want[starts:]
        gap = steps.max(-1) - steps[np.arange(len(picks)), picks]
        off = np.where(got == want, 0, np.abs(got - want))  # -inf at <copy>, both
        self.error = max(self.error, float(off.max()))
        self.gap = max(self.gap, float(gap.max()))
        self.same &= bool((steps.argmax(-1) == picks).all())
        self.seen = True

    def note(self, value: float) -> str:
        """`value` as bux's own in the run's dtype, if measured."""
        return f" (bux's own {value:.2e})" if self.seen else ""


def _logprobs(model: Any, tokens: list[int]) -> np.ndarray:
    """bux's log-probabilities of every position of `tokens`, in float32: of its copy
    pointer's mixture, for a first head with one (`bux.architecture.copy`)."""
    import jax
    import jax.numpy as jnp

    from bux.architecture import copy

    [first, *_] = model.heads(jnp.asarray([tokens]))
    if first.pointer is not None:
        return np.asarray(copy.log_probs(first)[0])
    logits = model(jnp.asarray([tokens]))[0].astype(jnp.float32)
    return np.asarray(jax.nn.log_softmax(logits, -1))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
