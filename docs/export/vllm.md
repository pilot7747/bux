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
  `--mamba-ssm-cache-dtype float32` keeps it in float32, as bux does. A `kda`
  layer's state is float32 either way, as vLLM keeps Kimi Linear's.
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
  DeltaNet layers, and its KDA kernels. Each layer loads its own weights from
  bux's names (q, k and v into one projection, an MoE's stacked experts into
  vLLM's layout), and every weight in the checkpoint must find a parameter.
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
- **Ports.** A layer's ports go through the forward pass's taps, as in bux: a `gqa`
  layer that reads the `v` port mixes those values into its own before they enter
  its KV cache, and one that writes it pushes its values, mixed or not.

## What runs

| Layers | In vLLM |
|--------|---------|
| `embed`, `head` | the vocabulary split over ranks; learned positions, `scale_by_sqrt_dim`, `norm`; tied or untied, a bias, soft-capping |
| `rmsnorm`, `layernorm`, `mlp`, `gated_mlp`, `linear`, `act`, `scale`, `dropout` | yes, with the MLPs' `limit`, `up_softcap` and `up_offset` |
| `gqa` | RoPE (`pi`, `llama3`, `yarn`), QK-norm, GQA and MQA, biases, `fused_qkv`, sliding windows, soft-capping, sinks, its scale, `gated` (a gate per channel or per head), `output_norm`, the conv on keys and values (`kv_conv_size`), and the `v` port: the value residual, fixed, learned or convex. Not yet: an indexer, a relative bias, `log_scale`, `power_scale`, `gate_head_bias`, QK-norm across all heads (`qk_norm_all_heads`), or the `kv` and `index` ports |
| `mla` | by vLLM's MLA and its latent cache ([Multi-head latent attention](#multi-head-latent-attention)): queries at full rank or from a latent, RoPE (`pi`, `llama3`, `yarn`; partial, interleaved) or none, its scale, `gated` (a gate per channel or per head), `output_norm`; in bfloat16 or float16. Not yet: QK-norm, sliding windows, soft-capping, sinks, an indexer, a relative bias, `log_scale`, `gate_head_bias`, ports, `mla` beside `gqa`, or `mla` layers of different dims or heads |
| `moe` | softmax, sigmoid and `sqrtsoftplus` routers, `router_logit_scale`, groups, the aux-free bias, a router bias, shared experts and their gate, gated and plain experts (`swish`, `gelu`, `gelu_tanh`, `relu2`) with biases, a `limit` and `up_offset` on `swish` and `quick_gelu` gates (gpt-oss's), LatentMoE with its norms (vLLM's fused experts at the latent's width), and shared experts the router scores (`shared_expert_sink`). Not yet: `expert_norm`, or the experts' `up_softcap` |
| `mamba2`, `gdn` | by vLLM's own layers, with one type and shape of recurrent layer per model; `gdn` in bfloat16 only, and with its short conv and output gate. Under tensor parallelism, vLLM 0.32's Mamba-2 failed with one group (its fused all-reduce), and in bfloat16 where a rank's share of the input projection is not a multiple of 8 wide |
| `kda` | by vLLM's KDA kernels, Kimi Linear's (by chunks for a prompt, the recurrence for each decoded token) and its causal conv, with one type and shape of recurrent layer per model: value heads (`num_v_heads`), the decay's rank, `gate_lower_bound` of -5 and above, and a dense or low-rank output gate. Not yet: no conv, no output gate, a `v_head_dim` other than `head_dim`, or speculative decoding |
| residual algebras, `merge` | `hc`, `mhc`, `gr`, `attnres`, `vpgate`, with every setting, single-pass included |
| the graph | everything; a shared MoE or recurrent layer may run only once per forward pass |

Every other layer type is not in vLLM yet: among them `csa`, `dyt`, `conv`,
`short_conv`, `mamba`, `rwkv7`, `engram`, and the linear attentions other than `gdn`
and `kda` (`gdn2`, `deltanet`, `gla`, `lightning`). Serve those models with
transformers ([Exporting models](index.md)).

vLLM's own KDA layer gives every value head a q and a k head of its own. bux's
`kda` shares one among a group of value heads, so the plugin runs vLLM's kernels in
a layer of its own: q and k are projected and convolved at their own heads, as bux
computes them, and repeated out to the value heads only for the kernels. Its weights
and its state per sequence are bux's.

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
  (`TRITON_MLA`), the one MLA backend that takes any dims.
- **Prefill.** vLLM's MLA prefill backends take a few sets of dims (DeepSeek's, among
  them), and vLLM picks among them. For any other dims the plugin uses
  FlashAttention's MLA prefill.

To pick the decoding backend yourself, name it, as for any model:

```sh
vllm serve exported/ --dtype bfloat16 --attention-backend TRITON_MLA
```

The prefill backend is `--attention-config '{"mla_prefill_backend": "FLASH_ATTN"}'`
(or `FLASHINFER`, `TRTLLM_RAGGED`), for the dims it takes.

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
beside a sigmoid MoE, and `mla-deepseek` at DeepSeek-V3's dims (FlashInfer's), with
a query latent, interleaved YaRN, a gate per head and an output norm. The table
above does not cover them.
