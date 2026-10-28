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
costs.

- **Quantized:** every projection the plugin runs on vLLM's linear layers, and the
  routed and shared experts. That is attention's q, k, v, gate and output
  projections (`gqa`, `mla`, `csa`), the MLPs and `linear`, LatentMoE's
  projections, and the projections of `kda`, `gdn2`, `deltanet`, `gla`,
  `lightning`, `rwkv7` and `short_conv`, their low-rank ones included. vLLM's own
  `gdn` layer quantizes its projections too, and so does its Mamba-2, where tensor
  parallelism splits its groups.
- **Kept in the model's dtype:** the embeddings and the heads, the routers (which
  the plugin computes in float32), the shared experts' gate, the norms and convs,
  the residual algebras, the indexers, `engram`, `mamba` (vLLM's Mamba layer takes
  no quantization), and the KV cache (`--kv-cache-dtype` sets that). So are the
  weights the plugin reads itself: a sparse `mla`'s `kv_b`, which its sparse
  attention reads, and `csa`'s grouped low-rank output.
- **Dtype.** vLLM's FP8 takes bfloat16 and float16 models only.
- **Random weights.** `--load-format dummy` quantizes the random weights, so a
  `.bux` file's export times in FP8 too.
- **Tensor parallelism.** The layers split over ranks as they do in bfloat16, and
  each weight's scale is the largest over its ranks' shares. For the routed
  experts' scales vLLM reduces over its expert-parallel group, which it builds for
  a model whose config names experts: the plugin's names them as Mixtral's does
  (`num_experts`, `num_experts_per_tok`).

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
The layers whose cache is the plugin's own refuse an FP8 cache as the model is built
(`NotImplementedError`, naming the layer and `kv_cache_dtype`), rather than keep it
in bfloat16.

| Layers | FP8 KV cache |
|--------|--------------|
| `gqa` | yes, by vLLM's attention: sliding windows, sinks, soft-capping, gates, QK-norm, `kv_conv_size` (its conv state stays in the model's dtype), the `v` port, and the `kv` port (a reader attends to its writer's FP8 cache) |
| `gqa` with an indexer, the `index` port or a relative bias | refused: its cache is the plugin's ([Sparse attention](#sparse-attention)) |
| `mla` | yes, by vLLM's MLA backends that take FP8: Triton's, FlashInfer's on SM 10.x. A prompt's cached context is dequantized as it is gathered, by vLLM's kernel or, for the latent sizes it lacks, the plugin's |
| `mla` with an indexer or the `index` port | refused: its cache is the plugin's |
| `csa` | refused: its caches are the plugin's ([Compressed sparse attention](#compressed-sparse-attention)) |
| `mamba`, `mamba2`, `gdn`, `kda`, `gdn2`, `rwkv7`, `deltanet`, `gla`, `lightning`, `conv`, `short_conv`, `engram` | no KV cache: their states follow `--mamba-cache-dtype` and `--mamba-ssm-cache-dtype` |

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
at a scale of 1. A case passes if the mean cosine of vLLM's FP8 logits with its
bfloat16 logits, at the prompts' positions and at the decoding steps, loses no more
than twice the emulation's loss, plus 0.002. That is the rule
`scripts/vllm_parity.py` holds bfloat16 to against bux. vLLM compares with
bfloat16 rather than float32, and the emulation leaves the experts' hidden units
unrounded, so the two differ a little. A weight loaded in the wrong place, or a
scale set by garbage, costs many times more. A case of layers that only the plugin
exports has no transformers model to emulate, and its accuracy goes unchecked.

These tiny models of random weights lose far more to FP8 than a trained model
does. The mean cosine at the prompts' positions, emulated on the CPU and measured
on one B300 with vLLM 0.32.0 on 2026-10-28:

| Case | Emulated: weights | vLLM: weights | Emulated: weights and KV cache | vLLM: weights and KV cache |
|------|-------------------|---------------|--------------------------------|----------------------------|
| `gpt3` | 0.9915 | 0.9919 | 0.9897 | 0.9895 |
| `llama` | 0.9797 | 0.9803 | 0.9749 | 0.9746 |
| `frontier` | 0.9629 | 0.9631 | 0.9584 | 0.9599 |
| `hybrid` | 0.9709 | 0.9714 | 0.9706 | 0.9715 |

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
| `embed`, `head` | the vocabulary split over ranks; learned positions, `scale_by_sqrt_dim`, `norm`; tied or untied, a bias, soft-capping |
| `rmsnorm`, `layernorm`, `dyt`, `mlp`, `gated_mlp`, `linear`, `act`, `scale`, `dropout` | yes, with the MLPs' `limit`, `up_softcap` and `up_offset` |
| `conv` | by vLLM's short-conv kernels, its last inputs in vLLM's Mamba cache, as the conv on keys and values; the whole stream on every rank |
| `short_conv` | LFM2's, `C * conv(B u)`, by the same kernels and state, its channels split over ranks, with or without biases |
| `gqa` | RoPE (`pi`, `llama3`, `yarn`), QK-norm, GQA and MQA, biases, `fused_qkv`, sliding windows, soft-capping, sinks, its scale, `gated` (a gate per channel or per head, with `gate_head_bias`), `output_norm`, `power_scale` and `log_scale` (on the queries, which scales the logits as bux does), the conv on keys and values (`kv_conv_size`), an indexer ([Sparse attention](#sparse-attention)), a relative bias (Inkling's, by the plugin's own op, as an indexer), the `v` port: the value residual, fixed, learned or convex, the `kv` port: cross-layer KV sharing, by vLLM's own, between layers of the same kv heads, the reader's window no longer than the writer's, and the `index` port: one layer's picks shared with others ([Sparse attention](#sparse-attention)). QK-norm runs across all heads too (`qk_norm_all_heads`), its statistics summed over the ranks, where each rank keeps kv heads of its own. Not yet: the `kv` port of a sparse layer, or a relative bias beside an indexer or ports |
| `mla` | by vLLM's MLA and its latent cache ([Multi-head latent attention](#multi-head-latent-attention)): queries at full rank or from a latent, RoPE (`pi`, `llama3`, `yarn`; partial, interleaved) or none, its scale, `gated` (a gate per channel or per head), `output_norm`; in bfloat16 or float16. With an indexer (DeepSeek-V3.2's) or the `index` port's picks, sparse attention in the latent space by the plugin's ops ([Sparse attention](#sparse-attention)), with a window, soft-capping and sinks too, in any dtype. Not yet: QK-norm, a window, soft-capping or sinks without an indexer, a relative bias, `log_scale`, `gate_head_bias`, the `kv` port, an indexer that picks per group, `mla` that vLLM runs as MLA beside other attention, or such `mla` layers of different dims or heads |
| `csa` | bux's arithmetic on vLLM's paged caches ([Compressed sparse attention](#compressed-sparse-attention)): a sliding window or every token, compressed latents of any `compress_ratio`, the indexer's top-k (ReLU'd or plain dots, weighted or summed, its norm), queries at full rank or from a latent, QK-norm, RoPE (`pi`, `llama3`, `yarn`; partial, interleaved), sinks, soft-capping, its scale, `gated` (a gate per channel or per head, `gate_head_bias`), `output_norm`, a dense or grouped low-rank output, and its `kv`, `index` and `candidates` ports; in float32 or bfloat16. Not yet: an indexer of blocks, a relative bias, `log_scale`, speculative decoding, decode context parallelism, or an FP8 KV cache |
| `moe` | softmax, sigmoid and `sqrtsoftplus` routers, `router_logit_scale`, groups, the aux-free bias, a router bias, shared experts and their gate, gated and plain experts (`swish`, `gelu`, `gelu_tanh`, `relu2`) with biases, a `limit` and `up_offset` on `swish` and `quick_gelu` gates (gpt-oss's), SiTU-GLU experts (`situ` with an `up_softcap`, Kimi K3's, by vLLM's own), LatentMoE with its norms (vLLM's fused experts at the latent's width), shared experts the router scores (`shared_expert_sink`), and each routed expert's output normalized before its weight (`expert_norm`; each of a token's picks a row of its own in vLLM's fused MoE, so that their outputs come back apart, summed over the ranks, for the norm). Not yet: an `up_softcap` on experts other than `situ` ones |
| `mamba`, `mamba2`, `gdn` | by vLLM's own layers (`mamba` as Jamba's, without speculative decoding), with one type and shape of recurrent layer per model; `gdn` in bfloat16 only, and without a short conv or an output gate by bux's own rule ([DPLR delta rules](#dplr-delta-rules)). Under tensor parallelism, vLLM 0.32's Mamba-2 failed with one group (its fused all-reduce), and in bfloat16 where a rank's share of the input projection is not a multiple of 8 wide |
| `kda` | by vLLM's KDA kernels, Kimi Linear's (by chunks for a prompt, the recurrence for each decoded token) and its causal conv, with one type and shape of recurrent layer per model: value heads (`num_v_heads`), the decay's rank, `gate_lower_bound` of -5 and above, and a dense or low-rank output gate; by bux's own rule ([DPLR delta rules](#dplr-delta-rules)) without a conv or an output gate, with a `v_head_dim` other than `head_dim`, or a lower bound under -5. Not yet: speculative decoding |
| `gdn2`, `rwkv7`, `deltanet`, `gla`, `lightning` | the delta rule on fla's kernels where they are installed, else in PyTorch ([DPLR delta rules](#dplr-delta-rules)), the rest by vLLM's layers and causal conv, with one type and shape of recurrent layer per model: every setting, with a conv or none, `lightning`'s decay by the layer's depth among the token mixers and its RMSNorm of all heads, its statistics summed over the ranks, and `rwkv7`'s `v` port (the value residual). Not yet: `gla` without a conv in a 16-bit model, or speculative decoding |
| `engram` | every setting, on a single or a widened stream: the n-gram sizes and hash heads, the norms, the dilated conv or none, `vocab_classes` and `boundary_token`. Not yet: speculative decoding, a conv of one tap, or vLLM's V1 model runner |
| residual algebras, `merge` | `hc`, `mhc`, `gr`, `attnres`, `vpgate`, with every setting, single-pass included |
| the graph | everything; a shared MoE or recurrent layer may run only once per forward pass |

Every other layer type is not in vLLM yet. Serve those models with transformers
([Exporting models](index.md)).

vLLM's own KDA layer gives every value head a q and a k head of its own. bux's
`kda` shares one among a group of value heads, so the plugin runs vLLM's kernels in
a layer of its own: q and k are projected and convolved at their own heads, as bux
computes them, and repeated out to the value heads only for the kernels. Its weights
and its state per sequence are bux's.

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

The kernels are custom ops that stay inside vLLM's CUDA graphs: the graphs split at
attention and the recurrent and MoE layers' ops, not at these. On an NVIDIA B300
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

The KV cache must be unquantized (`--kv-cache-dtype auto`).

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

Measured on 2026-10-28 on one and two (TP=2) NVIDIA B300 GPUs, with vLLM 0.32.0,
PyTorch 2.13.0 (CUDA 13.0), flash-linear-attention 0.6.0 and Python 3.12; bux's
reference ran on the CPU in float32. vLLM ran with CUDA graphs and `torch.compile` (its default), its own pick of attention backend, Triton's MoE kernels (`--moe-backend triton`), and, in float32, Triton's dots in IEEE float32 (`TRITON_F32_DEFAULT=ieee`).

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
| `query-scale` | `power_scale` and `log_scale` on the queries, a gate's bias per head | 1.2e-5 | 0.13 (0.15) | - |
| `qk-norm-all-heads` | QK-norm across all heads | 5.7e-6 | 0.057 (0.075) | - |
| `kv-conv` | the conv on keys and values | 8.6e-6 | 0.12 (0.13) | 0.10 |
| `conv` | Inkling's shortcut, a conv on branch outputs, `y + conv(y)` | 5.7e-6 | 0.055 (0.085) | - |
| `short-conv` | LFM2's double-gated short conv beside attention | 6.7e-6 | 0.11 (0.13) | - |
| `kv-share` | Hymba's cross-layer KV sharing: a windowed layer's keys and values read by the next one or two, a full layer's by one with a window of its own | 1.1e-5 | 0.13 (0.19) | 0.20 |
| `relbias` | Inkling's attention: a relative bias, a conv on keys and values, a length scale, window-only layers biased over their window | 6.2e-6 | 0.080 (0.10) | 0.075 |
| `dsa` | DSA's top 8 of the keys, a window, sinks and a softcap | 7.6e-6 | 2.4 (2.0) | 2.4 |
| `dsa-conv` | an indexer beside the conv on keys and values, QK-norm, a gate, the `v` port | 6.2e-6 | 3.3 (3.1) | 3.3 |
| `dsa-share` | DSA's picks shared through the `index` port with two layers, the last with a window | 7.2e-6 | 3.9 (4.2) | 3.9 |
| `msa` | MSA's max-pooled blocks of 4, picks per group, plain dots, summed heads, the indexer's norm | 5.7e-6 | 3.5 (3.3) | 3.6 |
| `msa-share` | MSA's picks shared through the `index` port | 8.1e-6 | 4.7 (4.7) | 4.7 |
| `qsa` | QSA's mean-pooled blocks of 4, partial RoPE | 8.1e-6 | 0.63 (3.1) | - |
| `csa` | compressed sparse attention: groups of 4, a window of 16, the indexer's top 8, sinks | 6.7e-6 | 0.42 (2.7) | 0.62 |
| `csa-latent` | DeepSeek-V4.1's wiring of csa: a window-only layer and one compressing pairs, a query latent, a grouped low-rank output | 6.2e-6 | 0.31 (0.88) | - |
| `csa-ports` | DeepSeek-V4.1's ports: a layer that compresses pairs, indexes them and makes a candidate pool, one that reads its latents and picks, one that re-indexes inside the pool | 8.1e-6 | 0.87 (1.4) | - |
| `mla` | MLA with a latent of 256 and 32 rotated dims (Triton's MLA) | - | 0.10 (0.10) | 0.097 |
| `mla-moe` | the same MLA beside a sigmoid MoE | - | 1.6 (0.093), differs | 1.4 |
| `mla-chunked` | the same MLA prefilled 16 tokens per step (`max_num_batched_tokens=16`), each later chunk attending to the latents the earlier ones cached | - | 0.078 (0.10) | 0.085 |
| `mla-deepseek` | MLA at DeepSeek-V3's dims (FlashInfer's), a query latent, interleaved YaRN, a gate per head, an output norm | - | 0.089 (0.14) | - |
| `mla-dsa` | sparse MLA, DeepSeek-V3.2's: an indexer, a window, sinks and a softcap | 6.7e-6 | 2.6 (2.7) | - |
| `mla-share` | sparse MLA, GLM-5.3's: picks shared through the `index` port | 8.1e-6 | 3.3 (2.8) | - |
| `moe` | a softmax router, unnormalized top-2 of 8, a gated shared expert | 6.7e-6 | 0.68 (1.2) | - |
| `moe-grouped` | a sigmoid router, 4 groups, the aux-free bias, a scaling of 2.5 | 8.6e-6 | 2.1 (3.8) | 1.8 |
| `moe-relu2` | a `sqrtsoftplus` router, plain relu2 experts | 5.7e-6 | 1.8 (1.4) | - |
| `moe-clamped` | gpt-oss's experts (a `quick_gelu` gate, a limit, an offset, biases), a router bias | 5.7e-6 | 0.81 (0.72) | 0.83 |
| `moe-situ` | Kimi K3's SiTU-GLU experts, their up projection soft-capped | 5.3e-6 | 0.70 (1.7) | - |
| `moe-latent` | LatentMoE with its norms, a sigmoid router | 3.8e-6 | 1.9 (1.9) | 0.054 |
| `moe-sink` | shared experts the router scores (`shared_expert_sink`) | 8.6e-6 | 2.3 (1.7) | 3.2, differs |
| `moe-norm` | Muse Spark's experts: each one's output normalized before its weight, in a latent of 32 | 9.1e-6 | 1.7 (2.1) | 2.0 |
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
| `mhc` | manifold-constrained hyper-connections | 8.1e-6 | 0.097 (0.11) | 0.11 |
| `mhc-single` | mhc, single-pass | 6.7e-6 | 0.11 (0.10) | - |
| `gr` | `gr`'s residual algebra | 1.9e-6 | 0.018 (0.025) | - |
| `vpgate` | `vpgate`'s residual algebra | 9.5e-6 | 0.072 (0.13) | - |
| `engram` | Engram's memory between two blocks, a tied head | 1.4e-5 | 0.097 (0.11) | 0.074 |

Every float32 run decoded bux's greedy tokens exactly. In bfloat16 an MoE's and a
sparse layer's errors are large, and so are bux's own: where two experts' or two
keys' scores are within bfloat16's rounding of each other, a token's picks change,
and so does its output. Two cases differ: `moe-sink` at TP=2 picks a token 1.8
below the best, at bux's own 1.8; and `mla-moe` on one GPU is off by 1.6, where
bux's own bfloat16 run is off by 0.093 (at TP=2, by 1.4 against bux's own 1.1).
`gdn`, `kda`, `kda-odd`, `kda-kimi` and the `mla` cases run in bfloat16 only, and
`gla` without a conv in float32 alone. A `-` marks a case not measured that way.

On the CPU, `tests/test_export_vllm.py` runs the plugin's sparse and compressed
attention on paged caches of its own, prompts in chunks beside decoded tokens,
against the export's layers.
