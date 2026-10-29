(export-vllm)=
# Serving in vLLM

bux's vLLM plugin serves an export as a native [vLLM](https://docs.vllm.ai) model,
with vLLM's paged attention, fused MoE kernels, Mamba cache, CUDA graphs and tensor
parallelism. vLLM does not run a checkpoint's own code, so the plugin brings a model,
`BuxForCausalLM`, that builds any exported architecture from the export's
`config.json`, out of vLLM's own layers. It reads the directory
[`bux export`](index.md) writes; nothing is exported twice.

```sh
pip install vllm ./src/bux_vllm
vllm serve exported/ --dtype bfloat16
```

## Installing

The plugin is a separate package, `bux_vllm`, in `src/bux_vllm/`. It runs in vLLM's
environment, which doesn't need bux, JAX or Python 3.14 (vLLM and bux each pin
their own PyTorch). Its files import nothing from bux, only PyTorch, transformers and
vLLM, so it installs on its own:

```sh
pip install vllm "bux-vllm @ git+https://github.com/pilot7747/bux#subdirectory=src/bux_vllm"
# or from a checkout
pip install vllm ./src/bux_vllm
```

With uv, install it from outside the checkout:
`uv pip install --python VENV vllm /path/to/bux/src/bux_vllm`. Inside the checkout,
`uv pip` reads bux's own `[tool.uv]` settings, which are made for bux's Python 3.14
environment rather than vLLM's.

vLLM loads the plugin through its entry point (`vllm.general_plugins`) in every
process it starts. The plugin registers two things: the config for `model_type`
`bux`, so that no remote code runs, and the model class `BuxForCausalLM`.

:::{warning}
On a B300 (SM 10.3), vLLM 0.32 picks FlashInfer's CUTLASS kernels for an MoE and
compiles them the first time it runs. That build fails ("ninja: build stopped", in
TensorRT-LLM's `MoeGemmRunner`; 2026-10-26). Use vLLM's Triton kernels instead:
`--moe-backend triton`, or `moe_backend="triton"` among the engine's arguments.
:::

## Serving

The export of a trained run holds its tokenizer, if the run had one, and its
`seq_len` as the longest sequence to serve
([Exporting a run](index.md#exporting-a-run)). So vLLM needs only the dtype to run in:

```sh
vllm serve exported/ --dtype bfloat16 --tensor-parallel-size 2
```

Otherwise, tell vLLM what the export lacks:

- `--tokenizer`: the tokenizer the model was trained with. A run that trained
  without one, a `.bux` file and {py:func}`bux.export.huggingface.save` export none.
- `--max-model-len`: the longest sequence to serve. A `.bux` file's export, and one
  that `save` writes without `max_positions`, don't record it. vLLM then assumes 2048
  tokens, unless the model has learned positions, the only thing that bounds a bux
  model's context.

```sh
vllm serve exported/ --tokenizer meta-llama/Llama-3.1-8B --dtype bfloat16 \
  --max-model-len 8192 --tensor-parallel-size 2
```

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="exported/",
    tokenizer="meta-llama/Llama-3.1-8B",  # only for an export without one
    dtype="bfloat16",
    max_model_len=8192,
)
print(llm.generate("The capital of France is", SamplingParams(max_tokens=8)))
```

An export of a `.bux` file has no weights
([Exporting an architecture](index.md#exporting-an-architecture)). Serve it on random
ones, `--load-format dummy`, to time it.

- **Dtype.** vLLM runs a model in a single dtype, float32 or bfloat16. A model with
  `gdn` or `mla` layers runs in bfloat16 only, because vLLM's Gated DeltaNet and MLA
  kernels take nothing else (MLA also takes float16).
- **Recurrent state** is kept in the model's dtype, unless
  `--mamba-ssm-cache-dtype float32` keeps it in float32, as bux does. The state of
  `kda`, `gdn2` and `rwkv7` layers is float32 either way, as vLLM keeps Kimi
  Linear's.
- **Heads.** The model's output is its first head. Layers after it, such as a
  multi-token prediction module, are not built, and vLLM skips their weights. (vLLM
  drafts tokens by MTP only for the model types it knows.)
- **Unsupported layers.** A layer or setting the plugin does not run raises
  `NotImplementedError` as the model is built, naming it ([What runs](#what-runs)).

## FP8 weights

Serve a bfloat16 export with FP8 weights by vLLM's online quantization. The
checkpoint stays as it is: vLLM quantizes each weight as it loads it.

```sh
vllm serve exported/ --dtype bfloat16 --quantization fp8 --moe-backend triton
```

```python
llm = LLM(model="exported/", dtype="bfloat16", quantization="fp8", moe_backend="triton")
```

vLLM 0.32 takes `fp8`, on a checkpoint without FP8 weights, as its `fp8_per_tensor`:
each weight in FP8 (e4m3) with one scale per tensor, and each GEMM's input
quantized to FP8 as it runs, with a scale per token (per tensor for the routed
experts). Its other online schemes reach the same layers: `fp8_per_channel`, a
weight's scale per output channel (llm-compressor's FP8_DYNAMIC), and
`fp8_per_block`, per block of 128 by 128 with inputs per group of 128 (DeepSeek's
FP8). Emulated on the `gpt3`, `llama` and `frontier` models of
[Checking FP8](#checking-fp8), they come within 0.002 of cosine of `fp8`: the
rounding of the inputs, which all three scale as they run, is most of what FP8
costs. On a B300, serve with `fp8` ([Schemes on a B300](#schemes-on-a-b300)).

- **Quantized:** every projection the plugin runs on vLLM's linear layers, and the
  routed and shared experts. That is attention's q, k, v, gate and output
  projections (`gqa`, `mla`, `csa`), a `gqa` indexer's projections of its queries,
  keys and head weights (one GEMM, one scale, where its head dim is a multiple of
  16), the MLPs and `linear`, LatentMoE's
  projections, and the projections of `kda`, `gdn2`, `deltanet`, `gla`,
  `lightning`, `rwkv7` and `short_conv`, their low-rank ones included. vLLM's own
  `gdn` layer quantizes its projections too, and so does its Mamba-2, where tensor
  parallelism splits its groups.
- **Kept in the model's dtype:** the embeddings (`value_embed`'s too) and the heads, the routers (which
  the plugin computes in float32), the shared experts' gate, the norms and convs,
  the residual algebras, the indexers of `mla` and `csa`, `engram`, `mamba` (vLLM's Mamba layer takes
  no quantization), and the KV cache (`--kv-cache-dtype` sets that). So are the
  weights the plugin reads itself: a sparse `mla`'s `kv_b`, which its sparse
  attention reads, `csa`'s grouped low-rank output, and HyGA's matrices of each
  head and its cross-head gate (a `gqa` gate's projection of the input, low-rank
  or not, is quantized), and differential attention's projection to a weight a
  head.
- **Dtype.** vLLM's FP8 takes bfloat16 and float16 models only.
- **Random weights.** `--load-format dummy` quantizes the random weights, so a
  `.bux` file's export times in FP8 too.
- **Tensor parallelism.** The layers split over ranks as they do in bfloat16, and
  each weight's scale is the largest over its ranks' shares. For the routed
  experts' scales vLLM reduces over its expert-parallel group, which it builds for
  a model whose config names experts: the plugin's names them as Mixtral's does
  (`num_experts`, `num_experts_per_tok`).

### Schemes on a B300

Serve FP8 weights on a B300 with `--quantization fp8`, one scale per tensor. It is
the fastest of vLLM's online schemes there, and as accurate as the others on the
models of [Checking FP8](#checking-fp8). `fp8_per_channel` runs as it is, and is
slower on MoE models. `fp8_per_block` runs only on a fallback GEMM. vLLM 0.32 picks
DeepGEMM's for it, and the DeepGEMM in vLLM's wheel fails to load its kernel on
SM 10.3 (CUDA error 719). Name CUTLASS's instead:

```python
llm = LLM(
    model="exported/",
    dtype="bfloat16",
    quantization="fp8_per_block",
    kernel_config={"linear_backend_per_quant": {"fp8_block_w8a8": "cutlass"}},
    moe_backend="triton",
)
```

Blocks of 128 take MLPs, and experts, whose hidden units are a multiple of 128.

The table gives the sessions per hour per GPU of an agentic load, the most that
keep the median time per output token within 30 ms. Each session shares a prompt
of 2k tokens with the others, adds 14k of its own, and runs 4 turns, each 512
tokens out and 2k of tool output back. Measured on one B300 (TP=1) with vLLM
0.32.0 and an FP8 KV cache, on 2026-10-28. In parentheses are the sessions at
once:

| Model | `fp8` | `fp8_per_channel` | `fp8_per_block`, CUTLASS | `fp8_per_block`, Triton |
|-------|-------|-------------------|--------------------------|-------------------------|
| dense, 6.7B class (`d_model` 4096) | 1924 | 1925 | 1757 (28) | 1212 (20) |
| hybrid of 256 experts (`d_model` 2816) | 3683 | 3301 | 3069 (64) | 2376 (51) |
| the same with `hc` (`d_model` 3584) | 1737 | 1515 | 1389 (32) | 928 (20) |

Per channel serves as many sessions as per tensor on the dense model, and 10 to
13 % fewer on the hybrids. Per block serves 9 to 20 % fewer on CUTLASS's GEMM, and
35 to 47 % fewer on Triton's.

## FP8 KV cache

Keep the KV cache in FP8 too, with FP8 weights or without:

```sh
vllm serve exported/ --dtype bfloat16 --quantization fp8 --kv-cache-dtype fp8 \
  --moe-backend triton
```

The layers that attend by vLLM's own attention keep their cache in FP8 (e4m3), by
vLLM's backends: it quantizes the keys and values as it caches them. Each layer has
a scale for its keys and one for its values, which vLLM reads from a checkpoint. An
export has none, so both are 1.0: the keys and values are cached as they are, to 3
bits of mantissa, and clip past ±448, FP8's largest value. A model whose keys or
values reach further needs scales of its own, which the export does not write yet.
Most layers whose cache is the plugin's own refuse an FP8 cache as the model is
built (`NotImplementedError`, naming the layer and `kv_cache_dtype`), rather than
keep it in bfloat16. A sparse `gqa` layer whose indexer runs on vLLM's DSA kernels,
and one that reads its `index` port, keep theirs in FP8 at the same scale of 1, as
the plugin's attention kernel reads e4m3 and e5m2 ([Sparse attention](#sparse-attention)).

| Layers | FP8 KV cache |
|--------|--------------|
| `gqa` | yes, by vLLM's attention: sliding windows, sinks, soft-capping, gates, QK-norm, XSA (it reads the step's own values, before they are cached), `kv_conv_size` (its conv state stays in the model's dtype), `k_eq_v` (its one vector a token too, [Values from the keys](#values-from-the-keys)), the `ve` and `v` ports, and the `kv` port (a reader attends to its writer's FP8 cache). On SM 10.x, see the note on FlashAttention below |
| `gqa` with an indexer on vLLM's DSA or MSA kernels (DSA's, QSA's, MSA's), or reading the `index` port | yes, in a 16-bit model: its keys and values in FP8 beside the indexer's keys ([DSA on vLLM's kernels](#dsa-on-vllms-kernels), [MSA on vLLM's kernels](#msa-on-vllms-kernels)) |
| `gqa` with another indexer (one with a window, say) or a relative bias | refused: its cache is the plugin's ([Sparse attention](#sparse-attention)) |
| `mla` | yes, by vLLM's MLA backends that take FP8: Triton's (a latent of 256 and 32 rotated dims), FlashInfer's on SM 10.x (DeepSeek's dims). A prompt's cached context is dequantized as it is gathered, by vLLM's kernel or, for the latent sizes it lacks, the plugin's. Where no prefill backend of vLLM's takes the dims, the prefill is FlashAttention's, in the model's dtype |
| `mla` with an indexer or the `index` port | refused: its cache is the plugin's |
| `csa` | refused: its caches are the plugin's ([Compressed sparse attention](#compressed-sparse-attention)) |
| `mamba`, `mamba2`, `gdn`, `kda`, `gdn2`, `rwkv7`, `deltanet`, `gla`, `lightning`, `conv`, `short_conv`, `engram` | no KV cache: their states follow `--mamba-cache-dtype` and `--mamba-ssm-cache-dtype` |

:::{warning}
On a B300 (SM 10.x), vLLM 0.32 attends with FlashAttention (its FA4) where
FlashInfer does not take a layer, such as at heads of 32 dims. FA4 with an FP8 KV
cache fails where a rank keeps one KV head ("Expected strides[leading_dim] == 1"):
under tensor parallelism, or with one KV head to begin with. It also served the
`kv-conv` case wrong, which crashed on a device-side assert. Pass
`--attention-backend TRITON_ATTN`, under which both run.
:::

## Checking FP8

`scripts/vllm_fp8.py` checks FP8 weights, and an FP8 KV cache, on a GPU. It serves
the same export, with weights, in bfloat16 and in FP8, for a llama and for the
timeline's milestones at a width of 256: GPT-3's, and 2026's two, a sliding-window
stack and its KDA hybrid, both with gates and an MoE of 256 experts. `--only` takes
the cases of `scripts/vllm_parity.py` too. It prints the quantization method vLLM
picked for each kind of layer, each attention layer's KV cache dtype and backend,
the routed experts' backend, and how close the FP8 logits are to bfloat16's:

```sh
JAX_PLATFORMS=cpu python scripts/vllm_fp8.py export runs/fp8   # bux's environment
python scripts/vllm_fp8.py run runs/fp8 --moe-backend triton  # vLLM's: weights
python scripts/vllm_fp8.py run runs/fp8 --moe-backend triton --kv-cache-dtype fp8
```

What FP8 costs is the model's own, so the check holds vLLM to an emulation of it.
`export` runs each case's transformers model in float32 with FP8's rounding
emulated, as vLLM rounds it: each linear layer that vLLM quantizes, its weight per
tensor and its input per token; the routed experts' weights per expert and their
input per tensor; and, for an FP8 KV cache, the attention's queries, keys and values
at a scale of 1. A case passes if the cosine of vLLM's FP8 logits with its bfloat16
logits loses no more than twice the emulation's loss, plus 0.002: their mean at the
prompts' positions, and their median at the decoding steps, few, of which one may
pick an expert or a sparse layer's key differently and fall far. That is the rule
`scripts/vllm_parity.py` holds bfloat16 to against bux. vLLM compares with
bfloat16 rather than float32, and the emulation leaves the experts' hidden units
unrounded, so the two differ a little. A weight loaded in the wrong place, or a
scale set by garbage, costs many times more. A case of layers that only the plugin
exports has no transformers model to emulate, and its accuracy goes unchecked.

These tiny models of random weights lose far more to FP8 than a trained model
does. The mean cosine at the prompts' positions, emulated on the CPU and measured
on one B300 with vLLM 0.32.0 on 2026-10-28, with Triton's MoE kernels:

| Case | Emulated: weights | vLLM: weights | Emulated: weights and KV cache | vLLM: weights and KV cache |
|------|-------------------|---------------|--------------------------------|----------------------------|
| `gpt3` | 0.9915 | 0.9919 | 0.9897 | 0.9895 |
| `llama` | 0.9797 | 0.9803 | 0.9749 | 0.9746 |
| `frontier` | 0.9629 | 0.9631 | 0.9584 | 0.9599 |
| `hybrid` | 0.9709 | 0.9714 | 0.9706 | 0.9715 |

Every case passed that day:
- **The four milestones:** in FP8 at TP=1 and TP=2, with an FP8 KV cache at TP=1,
  and for `frontier` and `hybrid` at TP=2 too, and under `fp8_per_channel`.
- **`kv-conv-64`**, with an FP8 KV cache.
- **The MLA cases**, with an FP8 KV cache: `mla`, `mla-chunked`, `mla-moe` and
  `mla-deepseek`.
- **On Triton's attention, with an FP8 KV cache:** `kv-conv`, and `kv-share` and
  `gpt-oss` at TP=2.

At the prompts' positions, vLLM's mean cosine was within 0.006 of the emulation's
in every run.

## How it works

- **The module tree.** `bux_vllm.plan` places every layer where
  {py:mod}`bux.export.plan` does, from the IR in `config.json` (its `bux` field) and
  the names no module takes (`bux_reserved`), so a weight's name is its module's
  path, as in `model.layers.3.attn.q.weight`. A test checks that both agree on every
  example architecture.
- **The graph.** The model runs the IR as the generated `modeling_bux.py` does, over
  the batch's tokens side by side, as vLLM schedules them: residual branches, sums,
  products, repeats, taps and shared definitions. vLLM compiles it with
  `torch.compile` and captures it in CUDA graphs.
- **Residual algebras.** Under a residual algebra, each outermost residual branch
  is a module named by the algebra, `model.layers.3.hc`, which holds its parameters,
  as in the export. It reads the stream for the branch's body and writes the body's
  output back, per token: the n copies of hc, mhc and gr are a (tokens, n, d)
  tensor. attnres keeps the embedding and the sum of each finished block of branch
  outputs side by side, and the sum of the block under way, which is what bux's
  softmax over depth reads; it does not keep every output. Nothing of it is cached
  between steps. See [Residual algebras](#residual-algebras) for how they run.
- **The layers** are vLLM's: column- and row-parallel projections, the vocabulary
  split over ranks, vLLM's attention with its KV cache, its MLA with the latent
  cache, `FusedMoEFactory` for the routed experts, vLLM's Mamba-2 and Gated
  DeltaNet layers, and its KDA kernels. `csa` computes bux's arithmetic on vLLM's
  paged caches ([Compressed sparse attention](#compressed-sparse-attention)). Each
  layer loads its own weights from bux's names (q, k and v into one projection, an
  MoE's stacked experts into vLLM's layout), and every weight in the checkpoint
  must find a parameter.
- **State.** Each call of a layer that keeps state is a separate vLLM layer, named
  `caches.<i>.<path>` by its place among them. A shared attention layer called three
  times keeps three KV caches over one set of weights. A model with recurrent layers
  is a hybrid for vLLM, as the `layer_types` in its config say.
- **Convs on keys and values.** A `gqa` layer with `kv_conv_size` keeps a conv
  state beside its KV cache: each sequence's last `kv_conv_size - 1` keys and
  values, as projected, in vLLM's Mamba cache. It convolves them with vLLM's
  short-conv kernels, so the model is a hybrid for vLLM. vLLM pads each state to a
  page of the attention's KV cache, a block of tokens (16 by default) per sequence,
  and with prefix caching keeps the states at block boundaries (its `align` mode).
- **Engram.** An `engram` layer hashes each token's n-grams from the step's token
  ids and, for those that reach back before the step, each sequence's last
  `max_ngram_size - 1` ids, which vLLM's model runner reads from the tokens it holds
  and passes to the model, as it does for Qwen4Exp's n-gram embeddings. Nothing of
  them is cached, and vLLM's V2 model runner (its default) is needed. Its conv
  keeps each sequence's last `(conv_size - 1) max_ngram_size` inputs in vLLM's Mamba
  cache, like a conv on keys and values. The whole layer, its tables included, is
  on every rank.
- **Ports.** A layer's ports go through the forward pass's taps, as in bux: a `gqa`
  layer that reads the `v` port mixes those values into its own before they enter
  its KV cache, and one that writes it pushes its values, mixed or not.
- **Shared KV caches.** A `gqa` layer that reads the `kv` port keeps no KV cache:
  vLLM shares its writer's with it (its cross-layer KV sharing, as Gemma 3n's), and
  it projects its queries alone. The plugin finds each reader's writer before the
  model runs, by the taps' pushes and pops in the order the layers run, so a model
  that shares its caches serves more tokens in the same memory.

## What runs

| Layers | In vLLM |
|--------|---------|
| `embed`, `value_embed`, `head` | the vocabulary split over ranks; learned positions, `scale_by_sqrt_dim`, `norm`; tied or untied, a bias, soft-capping. `value_embed` looks up the step's token ids, and a reader takes its kv heads' columns of the rows |
| `rmsnorm`, `layernorm`, `dyt`, `mlp`, `gated_mlp`, `linear`, `act`, `scale`, `dropout` | yes, with the MLPs' `limit`, `up_softcap` and `up_offset` |
| `conv` | by vLLM's short-conv kernels, its last inputs in vLLM's Mamba cache, as the conv on keys and values; the whole stream on every rank |
| `short_conv` | LFM2's, `C * conv(B u)`, by the same kernels and state, its channels split over ranks, with or without biases |
| `gqa` | RoPE (`pi`, `llama3`, `yarn`), QK-norm, GQA and MQA, biases, `fused_qkv`, sliding windows, soft-capping, sinks, its scale, `gated` (a gate per channel or per head, with `gate_head_bias`, and HyGA's input, output and cross-head gates, the last's logits summed over the ranks by an all-reduce of their own), `output_norm`, `power_scale` and `log_scale` (on the queries, which scales the logits as bux does), the conv on keys and values (`kv_conv_size`), XSA (`exclusive`, fixed or learned), differential attention (`differential`: its query heads in one call of vLLM's attention over the same KV cache, each pair's subtraction fused by torch.compile, pairs kept on one rank), values from the keys (`k_eq_v`, [Values from the keys](#values-from-the-keys)), an indexer ([Sparse attention](#sparse-attention); DeepSeek-V3.2's on vLLM's DSA kernels, [DSA on vLLM's kernels](#dsa-on-vllms-kernels)), a relative bias (Inkling's, by the plugin's own op, as an indexer), the `ve` port: value embeddings, gated or not, added to the values before the value residual, the `v` port: the value residual, fixed, learned or convex, the `kv` port: cross-layer KV sharing, by vLLM's own, between layers of the same kv heads, the reader's window no longer than the writer's, and the `index` port: one layer's picks shared with others ([Sparse attention](#sparse-attention)). QK-norm runs across all heads too (`qk_norm_all_heads`), its statistics summed over the ranks, where each rank keeps kv heads of its own. Not yet: the `kv` port of a sparse layer, a relative bias beside an indexer or ports, `differential` reading the `index` port, or paired heads (`paired`), which need an attention over two heads' keys in one softmax: vLLM's attention takes a query head's keys from one kv head, and returns no log-sum-exp to merge two of its calls |
| `mla` | by vLLM's MLA and its latent cache ([Multi-head latent attention](#multi-head-latent-attention)): queries at full rank or from a latent, RoPE (`pi`, `llama3`, `yarn`; partial, interleaved) or none, its scale, `gated` (a gate per channel or per head), `output_norm`; in bfloat16 or float16. With an indexer (DeepSeek-V3.2's, its picks by vLLM's DSA kernels where they take it) or the `index` port's picks, sparse attention in the latent space by the plugin's ops ([Sparse attention](#sparse-attention)), with a window, soft-capping and sinks too, in any dtype. Not yet: QK-norm, a window, soft-capping or sinks without an indexer, a relative bias, `log_scale`, `gate_head_bias`, HyGA's gates, the `kv` port, an indexer that picks per group, `mla` that vLLM runs as MLA beside other attention, or such `mla` layers of different dims or heads |
| `csa` | bux's arithmetic on vLLM's paged caches ([Compressed sparse attention](#compressed-sparse-attention)): a sliding window or every token, compressed latents of any `compress_ratio`, the indexer's top-k (ReLU'd or plain dots, weighted or summed, its norm), queries at full rank or from a latent, QK-norm, RoPE (`pi`, `llama3`, `yarn`; partial, interleaved), sinks, soft-capping, its scale, `gated` (a gate per channel or per head, `gate_head_bias`), `output_norm`, a dense or grouped low-rank output, and its `kv`, `index` and `candidates` ports; in float32 or bfloat16. Not yet: an indexer of blocks, a relative bias, `log_scale`, HyGA's gates, speculative decoding, decode context parallelism, or an FP8 KV cache |
| `moe` | softmax, sigmoid and `sqrtsoftplus` routers, `router_logit_scale`, groups, the aux-free bias, a router bias, shared experts and their gate, gated and plain experts (`swish`, `gelu`, `gelu_tanh`, `relu2`) with biases, a `limit` and `up_offset` on `swish` and `quick_gelu` gates (gpt-oss's), SiTU-GLU experts (`situ` with an `up_softcap`, Kimi K3's, by vLLM's own), LatentMoE with its norms (vLLM's fused experts at the latent's width), shared experts the router scores (`shared_expert_sink`), and each routed expert's output normalized before its weight (`expert_norm`; each of a token's picks a row of its own in vLLM's fused MoE, so that their outputs come back apart, summed over the ranks, for the norm), and zero experts (`num_zero_experts`, LongCat-Flash's, [Zero experts](#zero-experts)). Not yet: an `up_softcap` on experts other than `situ` ones, or zero experts under expert parallelism |
| `mamba`, `mamba2`, `gdn` | by vLLM's own layers (`mamba` as Jamba's, without speculative decoding), with one type and shape of recurrent layer per model; `gdn` in bfloat16 only, and without a short conv or an output gate by bux's own rule ([DPLR delta rules](#dplr-delta-rules)). Under tensor parallelism, vLLM 0.32's Mamba-2 failed with one group (its fused all-reduce), and in bfloat16 where a rank's share of the input projection is not a multiple of 8 wide |
| `kda` | by vLLM's KDA kernels, Kimi Linear's (by chunks for a prompt, the recurrence for each decoded token) and its causal conv, with one type and shape of recurrent layer per model: value heads (`num_v_heads`), the decay's rank, `gate_lower_bound` of -5 and above, and a dense or low-rank output gate; by bux's own rule ([DPLR delta rules](#dplr-delta-rules)) without a conv or an output gate, with a `v_head_dim` other than `head_dim`, or a lower bound under -5. Not yet: speculative decoding |
| `gdn2`, `rwkv7`, `deltanet`, `gla`, `lightning` | the delta rule on fla's kernels where they are installed, else in PyTorch ([DPLR delta rules](#dplr-delta-rules)), the rest by vLLM's layers and causal conv, with one type and shape of recurrent layer per model: every setting, with a conv or none, `lightning`'s decay by the layer's depth among the token mixers and its RMSNorm of all heads, its statistics summed over the ranks, and `rwkv7`'s `v` port (the value residual). Not yet: `gla` without a conv in a 16-bit model, or speculative decoding |
| `doc_copy` | every setting and its `copy` port, on a single or a widened stream: the candidates found before the model runs, from vLLM's tokens of each request ([The document-local copy feature](#the-document-local-copy-feature)). Not yet: speculative decoding, more than 16 lengths, layers of different lengths or counts, or vLLM's V1 model runner |
| `engram` | every setting, on a single or a widened stream: the n-gram sizes and hash heads, the norms, the dilated conv or none, `vocab_classes` and `boundary_token`. Not yet: speculative decoding, a conv of one tap, or vLLM's V1 model runner |
| residual algebras, `merge` | `hc`, `mhc`, `gr`, `attnres`, `msar`, `vpgate`, with every setting, single-pass included; `msar`'s convs keep each sequence's last values in vLLM's Mamba cache ([Multi-stream attention residuals](#multi-stream-attention-residuals)) |
| the graph | everything; a shared MoE or recurrent layer may run only once per forward pass |

Every other layer type is not in vLLM yet. Serve those models with transformers
([Exporting models](index.md)).

vLLM's own KDA layer gives every value head a q and a k head of its own. bux's
`kda` shares one among a group of value heads, so the plugin runs vLLM's kernels in
a layer of its own: q and k are projected and convolved at their own heads, as bux
computes them, and repeated out to the value heads only for the kernels. Its weights
and its state per sequence are bux's.

## Values from the keys

A `gqa` layer with `k_eq_v` (Gemma 4's global layers) has no value projection.
vLLM's Gemma 4 loads the key projection into both the k and the v shard of its
`QKVParallelLinear`, and so does the plugin: the v shard's output goes unused, and
its FP8 scale is the keys'. The values are each head's key, after the conv on the
keys alone, RMS-normalized without a gain.

- **Keys and values.** By default the layer's KV cache holds both, by vLLM's
  attention as for any `gqa` layer, on the backend vLLM picks for it.
- **One vector a token.** Under `BUX_VLLM_K_EQ_V_CACHE=keys`, where each key is its
  value times the key norm's gain, the cache holds the values alone, half the bytes
  of keys and values, and the gain goes on the queries. That holds for a layer with
  no RoPE, a QK RMSNorm of each head without a bias, and no port that mixes other
  values in (a `v` or `ve` read) or shares its cache (`kv`). vLLM's Triton attention
  runs it, with the one paged slab as both its key and its value cache, in FP8 too
  under `--kv-cache-dtype fp8`. vLLM matches pages rather than tokens, so beside
  layers that keep both, or a recurrent state's page, its block holds twice the
  tokens, and prefix caching hits at twice the granularity.

The second serves slower. On one B300 (2026-10-29, vLLM 0.32.0), the hybrid at a
width of 3328 with FP8 weights and an FP8 KV cache, under the agentic load of
[Schemes on a B300](#schemes-on-a-b300), served 1990 sessions an hour per GPU. With
`k_eq_v` on every global layer, no value residual, and their caches one vector a
token, it served 638,
at a TPOT of 19.3 ms at 8 sessions at once against 14.8. Those layers attend by
Triton's kernel where the others attend by FlashInfer's, and their blocks hold 4192
tokens where the others' hold 2096.

A layer that reads the `v` port (a value residual) mixes the first layer's values
into its own, which its keys do not share, so it caches both.

## The document-local copy feature

`doc_copy`'s candidates depend on the step's tokens and their requests' earlier
tokens alone, so vLLM's model runner (V2, its default) finds them before the model
runs, in the plugin's runner state, as it gives an engram its ids. A request is one
document.

- **The scan.** A Triton kernel takes a program per token of the step. It scans the
  token's request, in vLLM's token ids of its requests, a block of earlier positions
  at a time: each one's common suffix with the token's, up to the longest length.
  For each length it keeps the latest that matches and how many do, and picks the
  longest. Prompts, cached prefixes and decoded tokens read the same tokens, with no
  host sync and no loop over requests.
- **The vector.** The model reads each token's candidate and bucket from a buffer
  of a fixed shape, which CUDA graphs replay. The candidates' rows come from the
  first embedding, split by vocabulary, and the bucket tables stay in the model's
  dtype, unquantized.
- **Cost.** A decoded token reads its request's tokens once: about 4 bytes a token
  of context, from vLLM's token ids in host memory, which the GPU reads in place.

On one B300 (2026-10-29, vLLM 0.32.0), the hybrid at a width of 3328 with the
feature, FP8 weights and an FP8 KV cache served 1948 sessions an hour under the
agentic load of [Schemes on a B300](#schemes-on-a-b300), against 1985 without it:
1.9% fewer.

## Zero experts

An `moe` with zero experts (`num_zero_experts`) routes once, by bux's rule, in the
compiled graph beside vLLM's fused MoE:

- **Routing:** the router's scores of the FFN and zero experts in float32, the
  aux-free bias on the scores that pick, the top-k, and the weights normalized and
  scaled, as bux computes them, which torch.compile fuses.
- **Zero experts:** each token times its zero picks' weights, added to the fused
  MoE's output after its sum over the ranks.
- **FFN experts:** vLLM's fused MoE takes the picks as given: logits of the picks'
  weights and $-\infty$ elsewhere, whose top-k it keeps.
- **Zero picks skip the GEMMs:** a zero expert's pick reaches the fused MoE as a
  phantom expert after the routed ones, which an expert map sends to -1, as expert
  parallelism sends another rank's experts. vLLM's Triton kernels then multiply
  none of its rows: `moe_align_block_size` drops them, or, where an FP8 GEMM's input
  takes one scale for all rows, gives them blocks that only write zeros, and
  `moe_sum` leaves them out. So a token costs the GEMMs of its FFN experts alone,
  and FP8 experts and CUDA graphs run as without zero experts. The phantom's
  weights are zeros, one expert's more memory.
- **Limits:** expert parallelism, which keeps a map of its own, is not taken yet,
  and a kernel without an expert map (not Triton's, `--moe-backend triton`) runs
  the phantom's rows, at a weight of 0.

vLLM serves LongCat-Flash's zero experts by its `ZeroExpertRouter`, which keeps
their output aside inside the fused MoE's op. torch.compile's trace never runs that
op, so vLLM 0.32 fails to compile such a model ("assert zero_expert_output is not
None"); the plugin does not use it.

## DPLR delta rules

vLLM has no kernel for the delta rules of `gdn2` and `rwkv7`. Their erase term is
a rank-one map of two different vectors, which makes them DPLR rules, not KDA's.
`deltanet`'s rule, with no decay, runs the same way, and so do `gla`'s and
`lightning`'s, with no rank-one term, and those of `gdn` and `kda` that vLLM's
kernels do not take. Each value head keeps a state of `head_dim` by `v_head_dim` in
vLLM's Mamba cache, in float32, beside the conv's last inputs (one position that
nothing reads without a conv) or the token before (`rwkv7`, for its token shift).
The projections, convs and norms run on vLLM's layers and kernels, as for the
other layers.

**On fla's kernels.** Install
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) into
vLLM's environment, and a model in bfloat16 or float16 runs the rule on its DPLR
kernels: by chunks for the prompts, by its recurrent kernel for decoding.

```sh
pip install flash-linear-attention==0.6.0
```

fla's rule is bux's, with bux's vectors and decay as fla's `a`, `b` and `gk`, and a
scale of 1. Its kernels take the rule's inputs in the model's dtype, where bux keeps
some of them in float32 (`gdn2`'s gated values and erase term), so they round as a
bfloat16 run of bux does. The decay and the state stay in float32. A bound on the
decay (`gate_lower_bound`, and `rwkv7`'s own) lets fla run its tensor-core scheme.

**In PyTorch.** Without fla, or in float32, the plugin runs the rule in PyTorch, in
float32, as bux does:

- **A prompt** runs by chunks of 64 tokens, with matmuls within a chunk and the
  state carried across, as fla's chunked kernels compute it. The batch's prompts run
  side by side, each padded to the longest, and padding leaves a state as it was.
- **Chunks side by side.** Where the decay is bounded below (`rwkv7`'s always,
  `gdn2`'s with `gate_lower_bound`), the decay between two tokens factors into
  matmuls, and every chunk of a block of tokens runs at once: only the state each
  chunk starts from runs one chunk after another. `rwkv7` factors whole chunks,
  `gdn2` at -5 blocks of 16 rows. An unbounded decay runs one chunk after another.
- **Decoding** takes one step of every sequence at once, which CUDA graphs capture.

This rule computes in float32 on CUDA cores, so a prompt costs several times what a
fused kernel would. Tests check both paths against the export's layers on the CPU,
the PyTorch rule as computed and fla's as its API defines it.

## Residual algebras

A residual algebra widens the stream: hc, mhc and gr into n copies of it, attnres
into the embedding and the sums of blocks of branch outputs. Each outermost branch
reads it for its body and writes the body's output back into it, and the widened
stream is n times (or as many sources as) the plain one, so what a branch costs is
how often it moves it.

- **Kernels.** In a 16-bit model, hc's, mhc's and attnres's reads and writes run
  fused Triton kernels (`bux_vllm.streams`), in float32 from the stream's dtype. A
  branch reads its streams in one kernel and writes them in another, so it moves
  the widened stream once into the read, and once in and once out of the write.
  Each token's program takes its whole row at once, up to 4096 channels of each
  stream, so that a decoding step's few tokens each make one round of loads:
  - hc's read computes the token's coefficients, A and B, static plus the dynamic
    term ``s tanh(norm(x_j) W)``, the norm's statistics and its dot products with
    W in one pass (``norm(x) W`` as ``rstd (x W - mean sum(W))``), then the read.
  - mhc's read computes its logits ``a r W + b`` of the flattened streams, and the
    read; its write computes Sinkhorn's matrix of them, for each token.
  - The write is ``x'_k = sum_j M_jk x_j + b_k y`` for a token's n streams.
  - attnres's read computes the softmax of every source's ``rmsnorm(v) . w`` and
    their mix; a finished block's sum joins its sources, the block under way's
    sum stays in float32.
- **PyTorch.** A float32 model, and any under `BUX_VLLM_RESIDUAL_KERNELS=torch`,
  takes the PyTorch path, which computes what the export does, term by term: a
  branch moves the widened stream about ten times (its norms, the projections, the
  read and the mix as batched matrix products of n rows, the add), and mhc runs
  Sinkhorn's rounds as two reductions each. `BUX_VLLM_RESIDUAL_KERNELS=triton`
  runs the kernels in float32 too. `gr` and `vpgate` always take PyTorch's path,
  as torch.compile compiles it.
- **Checks.** A CPU test runs the fused path with each kernel's computation in
  PyTorch against the export, and `scripts/vllm_residual.py` compares the kernels
  with the PyTorch path on a GPU and times a branch both ways.

### Multi-stream attention residuals

`msar` keeps attnres's sources, each finished one as its streams: itself and its
causal convs over the tokens. A branch expands the newest source, which its convs
read the last values of from before the step: each sequence's last
`max(kernel_sizes) - 1` values of the source, which vLLM keeps in its Mamba cache, as
a conv's state.

- **Packs.** vLLM gives every state of its Mamba cache a page as large as the
  model's largest. So the plugin packs the branches' states, d channels each, into
  as few states as fill that page: a recurrent layer's, such as KDA's in the
  hybrid. They come after every other state in `layer_types`. In the hybrid at a
  width of 3328, the 104 branches before the head pack 46 to a state, in 3 pages per
  sequence (about 10 MB), where a state each would take 104.
- **Kernels.** In a 16-bit model, two Triton kernels convolve a step's tokens and
  keep each sequence's last values, by vLLM's slots: decoded tokens one a sequence,
  a prompt's chunk after its kept values. The op runs between the CUDA graph's
  pieces, as a short conv does, and a decoding step's graph captures it. A float32
  model, and any under `BUX_VLLM_RESIDUAL_KERNELS=torch`, takes the PyTorch path.
  The read's softmax over depth and its mixer run in PyTorch, which torch.compile
  fuses.
- **Checks.** CPU tests run the plugin's branches against the export's, and its
  steps' convs (prompts in chunks, then decoded tokens) against each whole sequence;
  `scripts/vllm_residual.py` compares the kernels with the PyTorch path on a GPU.

The other algebras' kernels are custom ops that stay inside vLLM's CUDA graphs: the
graphs split at attention and the recurrent and MoE layers' ops, not at these. On an NVIDIA B300
(2026-10-28, vLLM 0.32.0, a step of 8 tokens in a CUDA graph), an `hc` branch's read
and write take 0.013 ms fused against 0.094 ms in PyTorch.

## Sparse attention

A `gqa` or `mla` layer with an indexer attends to the keys its indexer picks, as
bux's does: DSA's top-k keys, QSA's mean-pooled blocks (`block_size` with
`pool=mean`), MSA's max-pooled blocks and picks per group (`pool=max`,
`per_group=true`, on `gqa`), with plain or ReLU'd dot products, weighted or summed
heads, and the indexer's own norm. The layer's other settings apply as in a dense
layer: a window, sinks, a softcap, and a `gqa` layer's conv on keys and values and
`v` port.

- **Caches.** Each call keeps its keys and values in vLLM's paged KV cache, as a
  dense layer does, and its indexer's keys in the same rows: one vector of the
  indexer's `head_dim` per token, normalized and rotated (QSA's as projected,
  pooled again at each step), split over the kv heads. vLLM allocates the cache
  as for full attention, so it grows with the sequence, in blocks of as many
  tokens as the model's other attention caches, and prefix caching hits as often
  as in a dense model.
- **Picks.** Each query scores every key it sees by the indexer, in float32, and
  keeps the best as bux ranks them: by the floats' bits, the first of equal ones.
  It attends to those keys alone, read from the paged cache by the block table,
  with the layer's scale, softcap and sinks, in float32.
- **Kernels.** In a 16-bit model, Triton kernels read the paged caches in place,
  for DSA's keys, MSA's blocks with picks per group or not, and QSA's blocks, heads
  of up to 256 dims. One attends over each query's picks. One scores every key of
  a decoded token's request. One scores a prompt's keys for a tile of its queries
  at a time, each head's dots on the tensor cores and their sum in float32. Both
  rank DSA's keys, or find each of MSA's blocks' best score, in place. One pools a
  decoded token's QSA blocks, normalizes and rotates their keys and scores them; a
  prompt's are pooled once by PyTorch. The picks are ranked as bux ranks them.
  MLA's latents, and a float32 model unless `BUX_VLLM_SPARSE_KERNELS=triton`, take
  the PyTorch path, which gathers each query's scores, keys and values into memory
  first and so moves several times the bytes (`BUX_VLLM_SPARSE_KERNELS=torch` asks
  for it). `scripts/vllm_kernels.py` compares the two on a GPU. Either way the work
  runs in ops that vLLM's piecewise CUDA graphs run between their pieces, and a
  step of decoded tokens alone is one CUDA graph: as captured, its tokens read up
  to `max_model_len` keys, which every replay of it may have.
- **Latents.** A sparse `mla` layer caches what vLLM's MLA does: each token's
  normalized latent and its rotated key dims, one head of `kv_lora_rank +
  qk_rope_head_dim`. Each head's queries take in its key projection, attend to the
  picked latents, and its value projection takes the latents they weigh, in
  float32, as DeepSeek's MLA decodes. Its indexer's queries come from the query
  latent. A model whose `mla` layers are all sparse is no MLA model for vLLM.
- **Shared picks.** The `index` port passes a layer's picks on to later layers.
  A layer that reads it keeps its own keys and values but no indexer keys,
  runs no indexer, and attends to the keys the writer picked that its own window
  sees, in a second op beside the first.
- **Tensor parallelism.** Every rank computes the whole indexer and picks the same
  keys. With picks per group, a rank keeps its own kv heads' picks.

The KV cache must be unquantized (`--kv-cache-dtype auto`), but where DSA runs on
vLLM's kernels.

### DSA on vLLM's kernels

In a 16-bit model, DSA's indexer runs on the kernels vLLM runs DeepSeek-V3.2's on,
where they take its settings: single keys, ReLU'd dots summed over the heads (no
`per_group`), weighted or summed heads, an indexer head of 32, 64 or 128 dims, 128
heads at most, no window, and a `top_k` of 2048 at most. It takes DeepGEMM, which
vLLM installs, on an SM 9.0 or 10.x GPU. QSA's indexer (mean-pooled blocks of a
size that divides 64) runs on them as DSA's over its blocks, with `top_k /
block_size` blocks. MSA's (plain dots, picks per group) runs on vLLM's MiniMax-M3
kernels ([MSA on vLLM's kernels](#msa-on-vllms-kernels)), and the rest keep the
plugin's own kernels above. So does a sparse `mla` layer's indexer, unless
`BUX_VLLM_SPARSE_KERNELS=native`: its FP8 keys would take a head slot as wide as a
latent, which doubles the layer's cache.

- **Projections.** A `gqa` layer's indexer projects its queries, keys and head
  weights from the layer's input by one GEMM, which takes `--quantization` as the
  layer's other projections do.
- **Keys.** Each token's indexer key is cached in FP8 (e4m3) with a float32 scale,
  in head slots of the call's cache after its kv heads: each block's keys, then
  their scales, as DeepGEMM's paged kernel reads them. They page with the keys and
  values, one cache a call, so prefix caching hits as in a dense model. The cache
  takes blocks of 64 tokens, which vLLM sets (or splits larger blocks into). One
  Triton kernel writes a step's keys, values and indexer keys into the cache, and
  another quantizes the indexer's queries, each in one launch and bit for bit as
  PyTorch would.
- **Scores.** A prompt chunk's queries score the keys of its request, as vLLM
  gathers them, by DeepGEMM's MQA logits (`fp8_fp4_mqa_logits`); decoded tokens
  score theirs in place, by its paged MQA logits. Queries are FP8 too, with a scale
  for each head that joins the head's weight; heads are padded to a count DeepGEMM
  takes (4 or more, a power of two, on SM 10.x), with weights of 0.
- **QSA's blocks.** Its keys as projected are cached in the model's dtype, in slots of their
  own. The token that completes a block writes the block's key, the mean of its
  keys normalized and rotated to its first position, as bux pools it, into the FP8
  keys at the position of the block's index, so the kernels score a request's
  complete blocks as DSA's keys. A query also keeps its own block while that is
  unfinished.
- **Picks.** vLLM's top-k kernels keep each query's best `top_k` of the keys (or
  blocks) it sees, as DeepSeek-V3.2's do. A prompt chunk's scores are float32, as
  DeepGEMM's MQA logits give them: vLLM 0.32 has no bfloat16 scores or fused
  scores and top-k for a prompt's dense indexer.
- **Attention.** vLLM's own sparse GQA attention, the kernel it runs Qwen4's QSA
  on, attends over the picks, reading the paged keys and values in place, into
  the op's output: in a bfloat16 model, at heads of a power of two of dims, in
  groups of 4 query heads or more, without sinks or a softcap. Other layers attend
  by the plugin's Triton kernel, and MLA's latents by PyTorch.
- **FP8 KV cache.** Under `--kv-cache-dtype fp8`, such a call keeps its keys and
  values in FP8 (e4m3, at a scale of 1, as vLLM's attention does), and so does a
  layer that reads its `index` port. The attention kernel reads them as they are,
  its values through bfloat16 into its dots.

The indexer scores in FP8, as DeepSeek's does, where bux's scores in float32, so a
query picks another key than bux's where two scores are within FP8's rounding.
`BUX_VLLM_SPARSE_KERNELS=triton` keeps the plugin's own kernels (and its float32
scores), `torch` the PyTorch path, and `native` runs vLLM's kernels in a float32
model too. `scripts/vllm_native.py` checks the kernels on a GPU against PyTorch, on
the keys as the plugin caches them, and times them beside the plugin's own.

On an NVIDIA B300 (2026-10-28, vLLM 0.32.0), for an indexer of 4 heads of 64 and a
top 2048, a 16k-token prompt's scores and picks take 0.73 ms a layer on vLLM's
kernels against 6.8 ms on the plugin's, and a decoding step of 64 requests of 20k
keys 0.07 ms against 0.82 ms. Attending over the picks at a width of 2816 (44 query
heads of 64 over 11 kv heads), vLLM's kernel takes 13 ms for the 16k prompt and
0.13 ms for the step, the plugin's 39 ms and 0.35 ms. Under an agentic coding load
(prompts of 16k tokens, four turns), the hybrid with DSA global layers at that
width serves 2793 sessions an hour on one B300 within a TPOT of 30 ms in bfloat16,
and 2952 with FP8 weights and an FP8 KV cache, against 1692 on the plugin's own
kernels, at a TTFT of 0.46 s against 0.74 s.

At long context DSA decodes faster than dense attention. On one B300 (2026-10-29,
vLLM 0.32.0, FP8 weights and an FP8 KV cache), a hybrid of 56 layers, KDA and
MoE with DSA in its 14 global attention layers, at a `d_model` of 3584 (56 query
heads of 64 over 14 kv heads, an indexer of 4 heads of 64 keeping 2048 keys),
decodes one sequence at 174.8k tokens in 8.36 ms a token, and the same hybrid
with dense attention in 8.83 ms. A decoding step's GPU time is 8.37 ms against 8.98 ms. The
DSA layers' scores take 0.09 ms of it, their picks 0.22 ms and their attention
0.10 ms, where the dense layers' attention takes 0.76 ms. Writing their caches and
preparing their queries and picks take 0.09 ms, where writing the dense layers'
caches takes 0.07 ms.

### MSA on vLLM's kernels

In a bfloat16 model, MSA's indexer and attention run on the Triton kernels vLLM
runs MiniMax-M3's sparse layers on, where they take its settings: blocks of 128
keys (`block_size=128`, `pool=max`, `per_group=true`), plain dots, unweighted
heads, an index head for each kv head, index and attention heads of a power of two
of dims (keys and values alike), a power of two of kv heads on a GPU, fewer than 64
blocks a query, and no window, sinks or softcap. MiniMax-M3's own settings are
these.

- **Keys.** Each token's index key, normalized and rotated, is cached in bfloat16
  in head slots after the call's kv heads. The cache takes blocks of 128 tokens, a
  block of MSA's to a page, as the kernels read them.
- **Picks.** Each index head scores the blocks its query sees by their best key,
  and keeps its group's best `top_k / 128`, the query's own block among them.
- **Attention.** Each kv head's group of query heads attends to its group's
  blocks, reading the paged keys and values in place, in bfloat16 or FP8 (e4m3 or
  e5m2, as under `--kv-cache-dtype fp8`).

The picks are bux's, but for ties, which the kernels break their own way.

A `gqa` layer with a relative bias (Inkling's) runs the same way, as no attention
kernel of vLLM's adds a bias that depends on how far behind its query a key is: its
keys and values in a paged cache of the plugin's own, every key a query sees read
from it, the bias on the logits, then the length scale (`log_scale`), in float32.

## Multi-head latent attention

The plugin runs `mla` by vLLM's own MLA, as vLLM runs DeepSeek-V3. vLLM caches each
token's normalized key-value latent and its rotated shared key dims:
`kv_lora_rank + qk_rope_head_dim` values per token and layer, as one head. It
attends to a prompt with the keys and values projected up, and decodes in the
latent space, with the up projection absorbed into the queries and the output. The
rest computes as bux does: the latents' norms and RoPE in float32, by bux's
frequencies and scale.

vLLM reads the MLA's dims from the export's config, by DeepSeek's names
(`kv_lora_rank`, `qk_nope_head_dim`, ...), so a model's `mla` layers share their
dims and heads, and all its softmax attention is `mla`. vLLM picks an MLA backend
by those dims and the GPU. On a B300 (SM 10.x):

- **Decoding.** FlashInfer's MLA (`FLASHINFER_MLA`) takes DeepSeek's dims, a latent
  of 512 and 64 rotated dims. A latent of 256 and 32 rotated dims runs on Triton's
  (`TRITON_MLA`), the one MLA backend that takes any dims. vLLM's MLA metadata
  takes only a latent and rotated dims of 320, 512 or 576 in all, whatever the
  backend; for other sizes the plugin checks them by the backend's own sizes.
- **Prefill.** vLLM's MLA prefill backends take a few sets of dims (DeepSeek's, among
  them), and vLLM picks among them. For any other dims the plugin uses
  FlashAttention's MLA prefill, which on a B300 (FlashAttention 4) takes query, key
  and value heads of 8 to 128 dims.
- **Cached context.** A prompt's chunk attends to the latents its earlier chunks
  cached, or a cached prefix's, by gathering them first. vLLM's gather kernel takes
  320, 512 or 576 values per token; the plugin gathers other sizes in PyTorch.

To pick the decoding backend yourself, name it, as for any model:

```sh
vllm serve exported/ --dtype bfloat16 --attention-backend TRITON_MLA
```

The prefill backend is `--attention-config '{"mla_prefill_backend": "FLASH_ATTN"}'`
(or `FLASHINFER`, `TRTLLM_RAGGED`), for the dims it takes.

An `mla` layer with an indexer, DeepSeek-V3.2's, runs on none of vLLM's MLA
backends: the plugin's sparse attention attends to its picks in the latent space
([Sparse attention](#sparse-attention)), over caches of the same latents.

## Compressed sparse attention

The plugin runs `csa` with bux's arithmetic on vLLM's paged caches.
Each call keeps, for each sequence:

- **Its window:** the latents of its last `sliding_window_size` tokens, `head_dim`
  values each (every token's, without a window).
- **Compressed latents:** one for every `compress_ratio` tokens, `head_dim` values,
  and its indexer key, of the indexer's `head_dim`.
- **What the next one pools:** the projections of the last `compress_ratio`
  tokens, `2 * head_dim` values each.

Each is a cache of vLLM's own kinds, as vLLM keeps DeepSeek-V4's, so vLLM pages,
frees and prefix-caches them. vLLM lays the caches' blocks side by side when their
sizes divide one another, as they do where the dims are powers of two.

A step writes its tokens' rows, then reads every key from the caches. The indexer
scores, in float32, the compressed latents a query sees, and keeps the top-k, the
first of equal scores as bux does; while there are no more than `top_k`, it keeps
them all. One softmax in float32 covers the window, the kept latents and the
sinks. In a 16-bit model both run by Triton kernels that read the caches in place,
as for the other sparse layers (`BUX_VLLM_SPARSE_KERNELS`), but for picks inside a
candidate pool; else in PyTorch, a prompt chunk's keys gathered once, in chunks of
bounded size, with no loop over tokens.

vLLM's own DeepSeek-V4 layers compute something else. At a ratio of 4 their
compressor pools overlapping windows of 8 tokens, with a learned position bias.
Their indexer has a compressor of its own and scores FP8 or MXFP4 keys. Their
kernels take heads of 512, the last 64 dims rotated. bux pools each group alone,
projects its indexer keys from the compressed latents, and takes any head width.

Its ports share the compressed latents with later layers, as DeepSeek-V4.1's do. A
layer that reads `kv` keeps its window of token latents alone and attends to the
compressed latents and indexer keys of the call that wrote them, whose caches the
plugin finds before the model runs. One that reads `index` takes that call's picks;
one that reads `candidates` re-indexes inside the candidate pool another call made:
its `candidate_top_k` blocks of `candidate_block_size` latents, each scored by its
best, and the block of the newest latent a query sees.

What reads or writes the caches runs outside the compiled graph, in two ops of the
plugin's, as vLLM's attention does. A step of decoded tokens alone is one CUDA graph:
its shapes are those of `max_model_len`, not of its longest sequence. A step with
a prompt runs the ops between the graph's pieces.

## How it was checked

`scripts/vllm_parity.py` exports a tiny model for each family of layers and settings,
with perturbed weights. It decodes a batch of prompts of 3 to 64 tokens greedily in
vLLM, and compares every position with bux: the token vLLM picks, and its
log-probabilities over the whole vocabulary, for the prompts and for every decoding
step. bux and vLLM live in separate environments, so the script runs in three steps:

```sh
JAX_PLATFORMS=cpu python scripts/vllm_parity.py export runs/parity/float32   # bux's
python scripts/vllm_parity.py run runs/parity/float32 --tp 2                 # vLLM's
JAX_PLATFORMS=cpu python scripts/vllm_parity.py compare runs/parity/float32  # bux's
```

- **In float32**, the log-probabilities must agree within 1e-3, and every pick must
  be bux's.
- **In bfloat16**, bux's reference still computes in float32, from the same rounded
  weights. bux also runs in bfloat16 (`bux.train.trainer.cast`) on the same tokens,
  to measure how far bfloat16 moves bux itself. vLLM passes if it is off by no more
  than twice that, and if it picks a token other than bux's only within that
  distance of the best: a near tie that two bfloat16 runs may break either way.
- **On vLLM's DSA kernels**, the indexer scores in FP8 and picks other keys than
  bux's float32 scores do, so those cases are compared with bux's indexer scoring
  in FP8 too (`compare --fp8-indexer`), and the rest checked as above.

Measured on 2026-10-28 on one and two (TP=2) NVIDIA B300 GPUs, with vLLM 0.32.0,
PyTorch 2.13.0 (CUDA 13.0), flash-linear-attention 0.6.0 and Python 3.12; bux's
reference ran on the CPU in float32. vLLM ran with CUDA graphs and `torch.compile` (its default), its own pick of attention backend, Triton's MoE kernels (`--moe-backend triton`), and, in float32, Triton's dots in IEEE float32 (`TRITON_F32_DEFAULT=ieee`).
The rows of XSA, HyGA, value embeddings, values from the keys, differential
attention and multi-stream attention residuals (`xsa`, `hyga`, `diff`, `msar`,
their `-hybrid` cases, `value-embed`, `value-embed-shared`, `k-eq-v` and
`k-eq-v-nope`) and of zero experts (`moe-zero`, `moe-zero-longcat`,
`moe-zero-norm`) were measured the same way on 2026-10-29, and so were the sparse
`gqa` and `mla` rows (all but `csa`'s), with each `gqa` indexer's projections in
one GEMM.

The table gives the largest difference of a log-probability, over every position of
4 prompts with 16 decoding steps each, across the whole vocabulary of 256. In
parentheses is bux's own bfloat16 error, against the same reference:

| Case | What it covers | float32 | bfloat16 (bux's own) | bfloat16, TP=2 |
|------|----------------|---------|----------------------|----------------|
| `llama` | GQA, RoPE, SwiGLU, an untied head | 6.7e-6 | 0.065 (0.10) | 0.076 |
| `qwen3` | QK-norm, a tied head | 6.2e-6 | 0.071 (0.091) | - |
| `gpt2` | learned positions, layernorms and biases, a GELU MLP | 3.8e-6 | 0.039 (0.057) | - |
| `rope` | partial and interleaved RoPE, `pi`, `llama3`, `yarn` on cos and sin and on the softmax, MQA beside GQA | 7.6e-6 | 0.061 (0.074) | - |
| `gemma` | a window of 8 keys, soft-capped attention and logits, zero-centered norms, a scaled embedding | 2.0e-5 | 0.057 (0.098) | - |
| `gpt-oss` | sinks, a window, attention biases | 5.7e-6 | 0.091 (0.098) | - |
| `mqa` | one KV head (each rank's copy under TP), head dim 64, a head bias | 7.6e-6 | 0.062 (0.10) | - |
| `graph` | a shared block run three times (three KV caches), taps, a sum, a scale, a linear | 1.1e-5 | 0.10 (0.10) | - |
| `extras` | gates on attention (per channel, per head), an output norm, fused qkv, an embedding norm, a clamped SwiGLU, a capped relu2 MLP | 3.8e-6 | 0.043 (0.065) | - |
| `dyt` | Dynamic Tanh in place of every norm | 9.5e-7 | 0.0030 (0.0035) | - |
| `value-residual` | the `v` port: fixed, learned and convex value residuals | 7.6e-6 | 0.088 (0.11) | 0.089 |
| `value-embed` | value embeddings on the hybrid's global layer (QK-norm, a gate, the conv on keys and values), a table a layer, gated, beside the value residual | 7.6e-6 | 0.075 (0.094) | 0.093 |
| `value-embed-shared` | one table two layers read, ungated, under `hc` | 6.7e-6 | 0.085 (0.11) | 0.092 |
| `query-scale` | `power_scale` and `log_scale` on the queries, a gate's bias per head | 1.2e-5 | 0.13 (0.15) | - |
| `qk-norm-all-heads` | QK-norm across all heads | 5.7e-6 | 0.057 (0.075) | - |
| `kv-conv` | the conv on keys and values | 8.6e-6 | 0.12 (0.13) | 0.10 |
| `xsa` | XSA (`exclusive`), all of each head's part along its value and a learned share of it, a window, sinks, the `v` port | 2.2e-5 | 0.10 (0.096) | 0.083 |
| `xsa-hybrid` | learned XSA on the hybrid's global layers under `hc` (no positions, QK-norm, a gate, the conv on keys and values, the `v` port, one kv head), beside KDA | - | 0.23 (0.82) | 0.46 |
| `hyga` | HyGA's gates: the head's own output's at its width, the cross-head gate and sinks; and low-rank with a head bias | 5.7e-6 | 0.082 (0.11) | 0.071 |
| `hyga-hybrid` | `xsa-hybrid` with HyGA's output and cross-head gates on its global layers | - | 0.34 (0.30) | 0.22 |
| `k-eq-v` | Gemma 4's values from the keys, with RoPE, a window, a fused projection and the value residual | 5.7e-6 | 0.061 (0.099) | 0.076 |
| `k-eq-v-nope` | the same on the hybrid's global layer (QK-norm, a gate, the conv on keys and values, no positions), its cache one vector a token | 6.2e-6 | 0.079 (0.093) | 0.071 |
| `diff` | differential attention (DIFF V2): pairs of query heads in a kv head's group, sinks for each query head, a window; a gate, QK-norm of all heads, fused projections | 1.1e-5 | 0.11 (0.15) | 0.13 |
| `diff-hybrid` | `xsa-hybrid`'s global layers differential, without XSA | - | 0.33 (0.30) | 0.32 |
| `conv` | Inkling's shortcut, a conv on branch outputs, `y + conv(y)` | 5.7e-6 | 0.055 (0.085) | - |
| `short-conv` | LFM2's double-gated short conv beside attention | 6.7e-6 | 0.11 (0.13) | - |
| `kv-share` | Hymba's cross-layer KV sharing: a windowed layer's keys and values read by the next one or two, a full layer's by one with a window of its own | 1.1e-5 | 0.13 (0.19) | 0.20 |
| `relbias` | Inkling's attention: a relative bias, a conv on keys and values, a length scale, window-only layers biased over their window | 6.2e-6 | 0.080 (0.10) | 0.075 |
| `dsa` | DSA's top 8 of the keys, a window, sinks and a softcap | 7.6e-6 | 2.4 (2.0) | 2.4 |
| `dsa-conv` | an indexer beside the conv on keys and values, QK-norm, a gate, the `v` port | 6.2e-6 | 3.3 (3.1) | 4.1 |
| `dsa-share` | DSA's picks shared through the `index` port with two layers, the last with a window | 6.7e-6 | 3.9 (4.2) | 3.9 |
| `msa` | MSA's max-pooled blocks of 4, picks per group, plain dots, summed heads, the indexer's norm | 7.2e-6 | 3.5 (3.3) | 3.6 |
| `msa-share` | MSA's picks shared through the `index` port | 6.7e-6 | 4.7 (4.7) | 4.7 |
| `qsa` | QSA's mean-pooled blocks of 4, partial RoPE | 7.6e-6 | 0.63 (3.1) | - |
| `csa` | compressed sparse attention: groups of 4, a window of 16, the indexer's top 8, sinks | 6.7e-6 | 0.42 (2.7) | 2.0 |
| `dsa-native` | DSA on vLLM's DSA kernels: an indexer of 4 heads of 64, its top 16, one layer that reads its picks | - | 3.8 (4.9) | 3.8 |
| `dsa-hybrid` | the same on the hybrid's global layer (QK-norm, a gate, the conv on keys and values), beside KDA | - | 0.66 (0.53) | 0.41 |
| `qsa-native` | QSA on vLLM's DSA kernels: blocks of 4, 4 heads of 64, partial RoPE | - | 3.1 (5.1) | 3.1 |
| `msa-native` | MSA on vLLM's MiniMax-M3 kernels: blocks of 128, an index head a kv head | - | 0.088 (0.094) | 0.067 |
| `mla-dsa-native` | sparse MLA with its indexer on vLLM's DSA kernels (`BUX_VLLM_SPARSE_KERNELS=native`) | - | 3.1 (3.5) | - |
| `csa-latent` | DeepSeek-V4.1's wiring of csa: a window-only layer and one compressing pairs, a query latent, a grouped low-rank output | 6.2e-6 | 0.31 (0.88) | - |
| `csa-ports` | DeepSeek-V4.1's ports: a layer that compresses pairs, indexes them and makes a candidate pool, one that reads its latents and picks, one that re-indexes inside the pool | 8.1e-6 | 0.87 (1.4) | - |
| `mla` | MLA with a latent of 256 and 32 rotated dims (Triton's MLA) | - | 0.10 (0.10) | 0.097 |
| `mla-moe` | the same MLA beside a sigmoid MoE | - | 1.6 (0.093), differs | 1.4 |
| `mla-chunked` | the same MLA prefilled 16 tokens per step (`max_num_batched_tokens=16`), each later chunk attending to the latents the earlier ones cached | - | 0.078 (0.10) | 0.085 |
| `mla-deepseek` | MLA at DeepSeek-V3's dims (FlashInfer's), a query latent, interleaved YaRN, a gate per head, an output norm | - | 0.089 (0.14) | - |
| `mla-dsa` | sparse MLA, DeepSeek-V3.2's: an indexer, a window, sinks and a softcap | 7.6e-6 | 2.6 (2.7) | - |
| `mla-share` | sparse MLA, GLM-5.3's: picks shared through the `index` port | 8.6e-6 | 3.3 (2.8) | - |
| `moe` | a softmax router, unnormalized top-2 of 8, a gated shared expert | 6.7e-6 | 0.68 (1.2) | - |
| `moe-grouped` | a sigmoid router, 4 groups, the aux-free bias, a scaling of 2.5 | 8.6e-6 | 2.1 (3.8) | 1.8 |
| `moe-relu2` | a `sqrtsoftplus` router, plain relu2 experts | 5.7e-6 | 1.8 (1.4) | - |
| `moe-clamped` | gpt-oss's experts (a `quick_gelu` gate, a limit, an offset, biases), a router bias | 5.7e-6 | 0.81 (0.72) | 0.83 |
| `moe-situ` | Kimi K3's SiTU-GLU experts, their up projection soft-capped | 5.3e-6 | 0.70 (1.7) | - |
| `moe-latent` | LatentMoE with its norms, a sigmoid router | 3.8e-6 | 1.9 (1.9) | 0.054 |
| `moe-sink` | shared experts the router scores (`shared_expert_sink`) | 8.6e-6 | 2.3 (1.7) | 3.2, differs |
| `moe-norm` | Muse Spark's experts: each one's output normalized before its weight, in a latent of 32 | 9.1e-6 | 1.7 (2.1) | 2.0 |
| `moe-zero` | zero experts: 4 beside 8 FFN experts, a top 3, a shared expert, a sigmoid router, the aux-free bias | 4.8e-6 | 1.1 (1.5) | 1.1 |
| `moe-zero-longcat` | LongCat-Flash's zero experts: a softmax router, unnormalized, scaled by 6, no bias | 7.2e-6 | 2.7 (1.3), differs | 2.7, differs |
| `moe-zero-norm` | zero experts beside an expert norm, routed by bux's rule | 4.3e-6 | 1.3 (2.0) | 1.4 |
| `mamba` | Jamba's Mamba beside attention | 9.5e-6 | 0.11 (0.14) | 0.096 |
| `mamba2` | Mamba-2 of 16 heads in 2 groups, beside attention | 7.6e-6 | 0.078 (0.13) | 0.076 |
| `gdn` | Gated DeltaNet of 2 q/k heads and 4 value heads, beside gated attention | - | 0.27 (0.23) | 0.28 |
| `gdn-plain` | GDN without a conv or an output gate, by bux's own rule | 2.2e-5 | 0.39 (0.41) | - |
| `kda` | KDA of 2 q/k heads and 4 value heads, its gate's lower bound of -5 | - | 0.13 (0.24) | 0.19 |
| `kda-odd` | KDA of 2 q/k heads and 6 value heads at a width of 136, its fused projection no multiple of 64 wide, beside an MoE | - | 2.1 (2.5) | 2.1 |
| `kda-kimi` | Kimi Linear's KDA: 4 heads, a low-rank output gate | - | 0.13 (0.43) | - |
| `kda-plain` | KDA without a conv, its values narrower than q and k, a low-rank gate and a lower bound of -8, by bux's own rule | 1.1e-5 | 0.14 (0.17) | - |
| `gdn2` | Gated DeltaNet-2 of 2 q/k heads and 4 value heads | 1.1e-5 | 0.15 (0.23) | 0.14 |
| `gdn2-paper` | the paper's gdn2: low-rank gates, negative eigenvalues, a sigmoid output gate | 8.6e-6 | 0.12 (0.20) | - |
| `gdn2-plain` | GDN-2 without a conv, its q, k and v SiLU'd alone | 9.1e-6 | 0.24 (0.23) | - |
| `deltanet` | DeltaNet of 2 q/k heads and 4 value heads, beside gated attention | 1.6e-5 | 0.18 (0.22) | - |
| `gla` | gated linear attention of 2 q/k heads and 4 value heads, without a conv | 8.1e-6 | - | - |
| `gla-conv` | gated linear attention with a conv and a sigmoid gate | 8.1e-6 | 0.14 (0.18) | - |
| `lightning` | MiniMax-01's lightning attention, two layers of their own depths | 6.7e-6 | 0.075 (0.17) | 0.11 |
| `lightning-conv` | lightning with a conv and value heads of their own | 1.0e-5 | 0.20 (0.18) | - |
| `rwkv7` | RWKV-7 of 2 heads of 64 | 3.4e-5 | 0.17 (1.0) | 0.70 |
| `rwkv7-residual` | RWKV-7's value residual through its `v` port, low ranks | 1.3e-5 | 0.11 (0.25) | - |
| `hc` | hyper-connections, 4 streams | 8.1e-6 | 0.079 (0.10) | 0.086 |
| `attnres` | attention residuals in blocks of 3 | 8.6e-6 | 0.098 (0.17) | 0.14 |
| `msar` | multi-stream attention residuals in blocks of 3, convs of 4, 8 and 12 taps over the tokens | 1.2e-5 | 0.12 (0.20) | 0.12 |
| `msar-hybrid` | `msar` in blocks of 2 on the hybrid's shape: KDA, gated global layers without positions, the conv on keys and values, the `v` port | - | 0.35 (0.73) | 0.49 |
| `mhc` | manifold-constrained hyper-connections | 8.1e-6 | 0.097 (0.11) | 0.11 |
| `mhc-single` | mhc, single-pass | 6.7e-6 | 0.11 (0.10) | - |
| `gr` | `gr`'s residual algebra | 1.9e-6 | 0.018 (0.025) | - |
| `vpgate` | `vpgate`'s residual algebra | 9.5e-6 | 0.072 (0.13) | - |
| `engram` | Engram's memory between two blocks, a tied head | 1.4e-5 | 0.097 (0.11) | 0.074 |
| `doc-copy` | the document-local copy feature at its two sites, over 16 tokens | - | - | - |
| `doc-copy-hybrid` | the same on the hybrid's shape under `hc`, contexts of up to 3 tokens | - | - | - |

Every float32 run decoded bux's greedy tokens exactly. In bfloat16 an MoE's and a
sparse layer's errors are large, and so are bux's own: where two experts' or two
keys' scores are within bfloat16's rounding of each other, a token's picks change,
and so does its output. Two cases differ: `moe-sink` at TP=2 picks a token 1.8
below the best, at bux's own 1.8; and `mla-moe` on one GPU is off by 1.6, where
bux's own bfloat16 run is off by 0.093 (at TP=2, by 1.4 against bux's own 1.1).
`moe-zero-longcat` differs too, off by 2.7 against bux's own 1.3 on one GPU and at
TP=2, though it decodes bux's greedy tokens: LongCat-Flash's unnormalized softmax
router ties as the others do, and in float32 it agrees within 7.2e-6.
The `-native` cases run vLLM's sparse kernels, in bfloat16 alone. Those on its DSA
kernels are compared with bux's indexer scoring in FP8 too (`--fp8-indexer`). In
float32, with the kernels forced (`BUX_VLLM_SPARSE_KERNELS=native`), they differ
from that reference by up to 3.4: FP8 scores often tie exactly, and vLLM's top-k
breaks a tie its own way, where bux keeps the first key.
`msa-native`'s prompts fit in one block of 128, so it checks the kernels' attention
and plumbing; `scripts/vllm_native.py` checks their picks, which agree with bux's.
`gdn`, `kda`, `kda-odd`, `kda-kimi` and the `mla` cases run in bfloat16 only, and
`gla` without a conv in float32 alone. A `-` marks a case not measured that way.

On the CPU, `tests/test_export_vllm.py` runs the plugin's sparse and compressed
attention on paged caches of its own, prompts in chunks beside decoded tokens,
against the export's layers.
