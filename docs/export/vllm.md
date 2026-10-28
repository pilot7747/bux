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
  tensor. attnres keeps the embedding, the sum of each finished block of branch
  outputs and the sum of the block under way, which is what bux's softmax over depth
  reads; it does not keep every output. Nothing of it is cached between steps.
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
| `rmsnorm`, `layernorm`, `mlp`, `gated_mlp`, `linear`, `act`, `scale`, `dropout` | yes, with the MLPs' `limit`, `up_softcap` and `up_offset` |
| `gqa` | RoPE (`pi`, `llama3`, `yarn`), QK-norm, GQA and MQA, biases, `fused_qkv`, sliding windows, soft-capping, sinks, its scale, `gated` (a gate per channel or per head, with `gate_head_bias`), `output_norm`, `power_scale` and `log_scale` (on the queries, which scales the logits as bux does), the conv on keys and values (`kv_conv_size`), an indexer ([Sparse attention](#sparse-attention)), the `v` port: the value residual, fixed, learned or convex, the `kv` port: cross-layer KV sharing, by vLLM's own, between layers of the same kv heads, the reader's window no longer than the writer's, and the `index` port: one layer's picks shared with others ([Sparse attention](#sparse-attention)). QK-norm runs across all heads too (`qk_norm_all_heads`), its statistics summed over the ranks, where each rank keeps kv heads of its own. Not yet: a relative bias, or the `kv` port of a sparse layer |
| `mla` | by vLLM's MLA and its latent cache ([Multi-head latent attention](#multi-head-latent-attention)): queries at full rank or from a latent, RoPE (`pi`, `llama3`, `yarn`; partial, interleaved) or none, its scale, `gated` (a gate per channel or per head), `output_norm`; in bfloat16 or float16. With an indexer (DeepSeek-V3.2's) or the `index` port's picks, sparse attention in the latent space by the plugin's ops ([Sparse attention](#sparse-attention)), with a window, soft-capping and sinks too, in any dtype. Not yet: QK-norm, a window, soft-capping or sinks without an indexer, a relative bias, `log_scale`, `gate_head_bias`, the `kv` port, an indexer that picks per group, `mla` that vLLM runs as MLA beside other attention, or such `mla` layers of different dims or heads |
| `csa` | bux's arithmetic on vLLM's paged caches ([Compressed sparse attention](#compressed-sparse-attention)): a sliding window or every token, compressed latents of any `compress_ratio`, the indexer's top-k (ReLU'd or plain dots, weighted or summed, its norm), queries at full rank or from a latent, QK-norm, RoPE (`pi`, `llama3`, `yarn`; partial, interleaved), sinks, soft-capping, its scale, `gated` (a gate per channel or per head, `gate_head_bias`), `output_norm`, and a dense or grouped low-rank output; in float32 or bfloat16. Not yet: the `kv`, `index` and `candidates` ports, an indexer of blocks, a relative bias, `log_scale`, speculative decoding, decode context parallelism, or an FP8 KV cache |
| `moe` | softmax, sigmoid and `sqrtsoftplus` routers, `router_logit_scale`, groups, the aux-free bias, a router bias, shared experts and their gate, gated and plain experts (`swish`, `gelu`, `gelu_tanh`, `relu2`) with biases, a `limit` and `up_offset` on `swish` and `quick_gelu` gates (gpt-oss's), LatentMoE with its norms (vLLM's fused experts at the latent's width), shared experts the router scores (`shared_expert_sink`), and each routed expert's output normalized before its weight (`expert_norm`; each of a token's picks a row of its own in vLLM's fused MoE, so that their outputs come back apart, summed over the ranks, for the norm). Not yet: the experts' `up_softcap` |
| `mamba2`, `gdn` | by vLLM's own layers, with one type and shape of recurrent layer per model; `gdn` in bfloat16 only, and with its short conv and output gate. Under tensor parallelism, vLLM 0.32's Mamba-2 failed with one group (its fused all-reduce), and in bfloat16 where a rank's share of the input projection is not a multiple of 8 wide |
| `kda` | by vLLM's KDA kernels, Kimi Linear's (by chunks for a prompt, the recurrence for each decoded token) and its causal conv, with one type and shape of recurrent layer per model: value heads (`num_v_heads`), the decay's rank, `gate_lower_bound` of -5 and above, and a dense or low-rank output gate. Not yet: no conv, no output gate, a `v_head_dim` other than `head_dim`, or speculative decoding |
| `gdn2`, `rwkv7` | the delta rule on fla's kernels where they are installed, else in PyTorch ([DPLR delta rules](#dplr-delta-rules)), the rest by vLLM's layers and causal conv, with one type and shape of recurrent layer per model: every setting, and `rwkv7`'s `v` port (the value residual). Not yet: `gdn2` without its conv, or speculative decoding |
| `engram` | every setting, on a single or a widened stream: the n-gram sizes and hash heads, the norms, the dilated conv or none, `vocab_classes` and `boundary_token`. Not yet: speculative decoding, a conv of one tap, or vLLM's V1 model runner |
| residual algebras, `merge` | `hc`, `mhc`, `gr`, `attnres`, `vpgate`, with every setting, single-pass included |
| the graph | everything; a shared MoE or recurrent layer may run only once per forward pass |

Every other layer type is not in vLLM yet: among them `dyt`, `conv`,
`short_conv`, `mamba`, and the linear attentions `deltanet`, `gla` and
`lightning`. Serve those models with transformers ([Exporting models](index.md)).

vLLM's own KDA layer gives every value head a q and a k head of its own. bux's
`kda` shares one among a group of value heads, so the plugin runs vLLM's kernels in
a layer of its own: q and k are projected and convolved at their own heads, as bux
computes them, and repeated out to the value heads only for the kernels. Its weights
and its state per sequence are bux's.

## DPLR delta rules

vLLM has no kernel for the delta rules of `gdn2` and `rwkv7`. Their erase term is
a rank-one map of two different vectors, which makes them DPLR rules, not KDA's.
Each value head keeps a state of `head_dim` by `v_head_dim` in vLLM's Mamba cache,
in float32, beside the conv's last inputs (`gdn2`) or the token before (`rwkv7`,
for its token shift). The projections, convs and norms run on vLLM's layers and
kernels, as for the other layers.

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
  rank DSA's keys, or find each of MSA's blocks' best score, in place. QSA pools
  and scores its blocks by PyTorch, and the picks are ranked as bux ranks them.
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

The plugin runs `csa` with bux's arithmetic, in PyTorch, on vLLM's paged caches.
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
scores, in float32, the compressed latents a query sees, its sequence's gathered
once for a prompt's chunk, and keeps the top-k, the first of equal scores as bux
does; while there are no more than `top_k`, it keeps them all. One softmax in
float32 covers the window, the kept latents and the sinks: in a 16-bit model by a
Triton kernel that reads the latents in place, as for the other sparse layers
(`BUX_VLLM_SPARSE_KERNELS`), else in PyTorch, in chunks of bounded size, with no
loop over tokens.

vLLM's own DeepSeek-V4 layers compute something else. At a ratio of 4 their
compressor pools overlapping windows of 8 tokens, with a learned position bias.
Their indexer has a compressor of its own and scores FP8 or MXFP4 keys. Their
kernels take heads of 512, the last 64 dims rotated. bux pools each group alone,
projects its indexer keys from the compressed latents, and takes any head width.

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

Measured on 2026-10-26 on one and two (TP=2) NVIDIA B300 GPUs, with vLLM 0.32.0,
PyTorch 2.13.0 (CUDA 13.0), transformers 5.19.0 and Python 3.12; bux's reference ran
on the CPU in float32. vLLM used Triton's attention in float32 and FlashAttention in
bfloat16 (FlashInfer for MQA's head dim of 64); its Triton MoE kernels
(`--moe-backend triton`; FlashInfer's, which vLLM picks on Blackwell where it can
build them, were not checked); CUDA graphs and `torch.compile` (vLLM's default; eager
gave the same results); and Triton's dots in IEEE float32
(`TRITON_F32_DEFAULT=ieee`).

The table gives the largest difference of a log-probability, over every position of
4 prompts with 16 decoding steps each, across the whole vocabulary of 256. In
parentheses is bux's own bfloat16 error, against the same reference:

| Case | What it covers | float32 | float32, TP=2 | bfloat16 (bux's own) | bfloat16, TP=2 |
|------|----------------|---------|---------------|----------------------|----------------|
| `llama` | GQA, RoPE, SwiGLU, an untied head | 1.1e-5 | 1.2e-5 | 0.15 (0.16) | 0.13 |
| `qwen3` | QK-norm, a tied head | 9.5e-6 | 8.6e-6 | 0.09 (0.13) | - |
| `gpt2` | learned positions, layernorms and biases, a GELU MLP | 8.3e-6 | 6.7e-6 | 0.07 (0.12) | - |
| `rope` | partial and interleaved RoPE, `pi`, `llama3`, `yarn` on cos and sin and on the softmax, MQA beside GQA | 2.7e-5 | 2.7e-5 | 0.20 (0.31) | - |
| `gemma` | a window of 8 keys, soft-capped attention and logits, zero-centered norms, a scaled embedding | 2.2e-5 | 2.1e-5 | 0.13 (0.17) | - |
| `gpt-oss` | sinks, a window, attention biases | 1.1e-5 | 1.1e-5 | 0.12 (0.14) | - |
| `mqa` | one KV head (each rank's copy under TP), head dim 64, a head bias | 1.7e-5 | 1.3e-5 | 0.19 (0.20) | - |
| `graph` | a shared block run three times (three KV caches), taps, a sum, a scale, a linear | 1.7e-5 | 1.6e-5 | 0.19 (0.31) | - |
| `extras` | gates on attention (per channel, per head), an output norm, fused qkv, an embedding norm, a clamped SwiGLU, a capped relu2 MLP | 1.1e-5 | 1.6e-5 | 0.15 (0.26) | - |
| `moe` | a softmax router, unnormalized top-2 of 8, a gated shared expert | 1.4e-5 | 1.5e-5 | 1.7 (1.3) | - |
| `moe-grouped` | a sigmoid router, 4 groups, the aux-free bias, a scaling of 2.5 | 1.9e-5 | 2.1e-5 | 3.3 (2.8) | 3.3 |
| `moe-relu2` | a `sqrtsoftplus` router, plain relu2 experts | 1.1e-5 | 1.1e-5 | 3.9 (4.0) | - |
| `moe-clamped` | gpt-oss's experts (a `quick_gelu` gate, a limit, an offset, biases), a router bias | 1.1e-5 | 1.2e-5 | 1.7 (1.1) | 1.1 |
| `mamba2` | Mamba-2 of 16 heads in 2 groups, beside attention | 2.1e-5 | 2.0e-5 | 0.19 (0.20); SSM32 0.19 | 0.15 |
| `gdn` | Gated DeltaNet of 2 q/k heads and 4 value heads, beside gated attention | - | - | 0.57 (0.56); SSM32 0.48 | 0.45 |

Every float32 run decoded bux's greedy tokens exactly. In bfloat16 an MoE's error is
large, and so is bux's own: where two experts' scores are within bfloat16's rounding
of each other, a token's top-k changes, and so does its output. `gdn` runs in
bfloat16 only. With `--mamba-ssm-cache-dtype float32` (SSM32) the recurrent state
stays in float32, as in bux: Gated DeltaNet's error falls, and Mamba-2's stays where
it was.

The script's `mla` cases check multi-head latent attention, in bfloat16 only:
`mla` with a latent of 256 and 32 rotated dims (Triton's MLA), `mla-moe` the same
beside a sigmoid MoE, `mla-chunked` the same prefilled 16 tokens per step
(`max_num_batched_tokens=16`), so that each later chunk attends to the cached
latents, and `mla-deepseek` at DeepSeek-V3's dims (FlashInfer's), with a query
latent, interleaved YaRN, a gate per head and an output norm. The table above does
not cover them.

The script's `csa` cases check compressed sparse attention, in float32 and
bfloat16: `csa` the timeline's layer on 2 heads of 64, its window of 16 and its
top 8 crossed by the prompts, and `csa-latent` DeepSeek-V4.1's wiring, a
window-only layer and one compressing pairs of tokens, from a query latent, through
a grouped low-rank output of 2 groups. The table above does not cover them either.
On the CPU, `tests/test_export_vllm.py` runs the plugin's arithmetic on paged caches
of its own, prompts in chunks beside decoded tokens, against the export's layer.

The script's `moe-norm` case checks Muse Spark's routed experts, each one's output
normalized before its weight, in a latent of 32; `kv-share` Hymba's cross-layer KV
sharing: a windowed layer's keys and values read by the next layer or two, and a
full layer's by one with a window of its own; `dsa-share` and `msa-share` picks
shared through the `index` port, DSA's by two layers, the last with a window, and
MSA's blocks per group; and `mla-dsa` and `mla-share` sparse MLA, DeepSeek-V3.2's
and GLM-5.3's, one layer with a window, sinks and a softcap. The table above does
not cover them.
