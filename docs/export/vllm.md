(export-vllm)=
# Serving in vLLM

An export runs in [vLLM](https://docs.vllm.ai) as a native model, with its paged
attention, fused MoE kernels, Mamba cache, CUDA graphs and tensor parallelism. vLLM
does not run a checkpoint's own code, so bux brings a plugin for it: a model,
`BuxForCausalLM`, that builds any exported architecture from the export's
`config.json`, out of vLLM's own layers. The checkpoint is the one
{py:func}`bux.export.huggingface.save` writes; nothing is exported twice.

## Installing

The plugin is a package of its own, `bux_vllm` in `src/bux_vllm/`, beside bux rather
than in it: it runs in vLLM's environment, which need not have bux, JAX or Python
3.14 (vLLM and bux each pin their own PyTorch), so its files import nothing of bux,
only PyTorch, transformers and vLLM, as the files an export copies for transformers
do. It installs alone:

```bash
pip install vllm "bux-vllm @ git+https://github.com/pilot7747/bux#subdirectory=src/bux_vllm"
# or from a checkout
pip install vllm ./src/bux_vllm
```

vLLM loads it by its entry point (`vllm.general_plugins`), in every process it
starts: it registers the config of `model_type` `bux`, so that no remote code runs,
and the model class `BuxForCausalLM`.

## Serving

An export has no tokenizer: give vLLM the one the model was trained with, the dtype
to run in, and the longest sequence to serve. Only learned positions bound a bux
model's context, so vLLM assumes 2048 tokens unless told otherwise.

```bash
vllm serve exported/ --tokenizer meta-llama/Llama-3.1-8B --dtype bfloat16 \
  --max-model-len 8192 --tensor-parallel-size 2
```

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="exported/",
    tokenizer="meta-llama/Llama-3.1-8B",
    dtype="bfloat16",
    max_model_len=8192,
)
print(llm.generate("The capital of France is", SamplingParams(max_tokens=8)))
```

vLLM runs a model in one dtype, float32 as well as bfloat16, but for its Gated
DeltaNet kernels, which take bfloat16 alone. A recurrent layer's state is in the
model's dtype, unless `--mamba-ssm-cache-dtype float32` keeps it in float32, as bux
does.

The model's output is its first head: layers after it, a multi-token prediction
module's, are not built, and vLLM skips their weights (it drafts tokens by MTP only
for the model types it knows). A layer or setting the plugin does not run raises
`NotImplementedError` as the model is built, naming it.

## How it works

- **The module tree.** `bux_vllm.plan` places every layer where
  {py:mod}`bux.export.plan` does, from the IR in `config.json` (the `bux` field), so
  a weight's name is its module's path, `model.layers.3.attn.q.weight`. A test checks
  both on every example architecture.
- **The graph.** The model runs the IR as the generated `modeling_bux.py` does, over
  the batch's tokens side by side as vLLM schedules them: residual branches, sums,
  products, repeats, taps and shared definitions. vLLM compiles it with
  `torch.compile` and captures it in CUDA graphs.
- **The layers** are vLLM's: column- and row-parallel projections, the vocabulary
  split over ranks, vLLM's attention with its KV cache, `FusedMoEFactory` for the
  routed experts and vLLM's Mamba-2 and Gated DeltaNet layers. Each layer loads its
  own weights from bux's names (q, k and v into one projection, an MoE's stacked
  experts into vLLM's), and every weight of the checkpoint must find a parameter.
- **State.** Each call of a layer that keeps state is a layer of vLLM's, named
  `caches.<i>.<path>` by its place among them: a shared attention layer called three
  times keeps three KV caches, over one set of weights. A model with recurrent layers
  is a hybrid for vLLM, its `layer_types` in the config say so.

## What runs

| Layers | In vLLM |
|--------|---------|
| `embed`, `head` | yes, the vocabulary split over ranks: learned positions, `scale_by_sqrt_dim`, `norm`; tied or not, a bias, soft-capping |
| `rmsnorm`, `layernorm`, `mlp`, `gated_mlp`, `linear`, `act`, `scale`, `dropout` | yes, with the MLPs' `limit`, `up_softcap` and `up_offset` |
| `gqa` | RoPE (`pi`, `llama3`, `yarn`), QK-norm, GQA and MQA, biases, `fused_qkv`, sliding windows, soft-capping, sinks, its scale, `gated` (a gate a channel or a head), `output_norm`; not yet `kv_conv_size`, an indexer, a relative bias, `log_scale` or ports |
| `moe` | softmax, sigmoid and `sqrtsoftplus` routers, `router_logit_scale`, groups, the aux-free bias, a router bias, shared experts and their gate, gated and plain experts (`swish`, `gelu`, `gelu_tanh`, `relu2`), with biases, and a `limit` and an `up_offset` of `swish` and `quick_gelu` gates (gpt-oss's); not yet LatentMoE, `expert_norm`, `shared_expert_sink` or the experts' `up_softcap` |
| `mamba2`, `gdn` | yes, by vLLM's layers, one type and shape of them a model; `gdn` in bfloat16 alone. Under TP, vLLM 0.32's Mamba-2 failed with one group (its fused all-reduce) and, in bfloat16, where a rank's share of the input projection is not a multiple of 8 wide |
| the graph | everything but residual algebras; a shared MoE or recurrent layer runs once a pass |

## How it was checked

`scripts/vllm_parity.py` exports a model for each family of layers and settings
(tiny ones, of perturbed weights), decodes a batch of prompts of 3 to 64 tokens
greedily in vLLM, and compares every position with bux: the token vLLM picks, and
its log-probabilities of the whole vocabulary, of the prompts and of every decoding
step. bux and vLLM have environments of their own, so it runs in three steps:

```bash
JAX_PLATFORMS=cpu python scripts/vllm_parity.py export runs/parity/float32   # bux's
python scripts/vllm_parity.py run runs/parity/float32 --tp 2                 # vLLM's
JAX_PLATFORMS=cpu python scripts/vllm_parity.py compare runs/parity/float32  # bux's
```

In float32 the log-probabilities must agree within 1e-3 and every pick be bux's. In
bfloat16, bux's reference still computes in float32, from the same rounded weights,
and bux runs in bfloat16 too (`bux.train.trainer.cast`) on the same tokens: vLLM
passes where it is off by no more than twice as much as bux is from itself, and picks
a token other than bux's only within that far of the best, a near tie that two
bfloat16 runs may break either way.

Measured on 2026-10-26 on NVIDIA B300 GPUs, one and two (TP=2), with vLLM 0.32.0,
PyTorch 2.13.0 (CUDA 13.0), transformers 5.19.0 and Python 3.12, bux's reference on
the CPU in float32. vLLM picked Triton's attention in float32 and FlashAttention (or
FlashInfer for MQA's head dim of 64) in bfloat16, its Triton MoE kernels
(`--moe-backend triton`; FlashInfer's, which vLLM picks on Blackwell where it can build them, were not checked), CUDA graphs and `torch.compile` (vLLM's default; eager gave
the same), and Triton's dots in IEEE float32 (`TRITON_F32_DEFAULT=ieee`). The largest
difference of a log-probability, over every position of 4 prompts and 16 steps each
and the whole vocabulary of 256:

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
| `extras` | gates on attention (a channel's, a head's), an output norm, fused qkv, an embedding norm, a clamped SwiGLU, a capped relu2 MLP | 1.1e-5 | 1.6e-5 | 0.15 (0.26) | - |
| `moe` | a softmax router, unnormalized top-2 of 8, a gated shared expert | 1.4e-5 | 1.5e-5 | 1.7 (1.3) | - |
| `moe-grouped` | a sigmoid router, 4 groups, the aux-free bias, a scaling of 2.5 | 1.9e-5 | 2.1e-5 | 3.3 (2.8) | 3.3 |
| `moe-relu2` | a `sqrtsoftplus` router, plain relu2 experts | 1.1e-5 | 1.1e-5 | 3.9 (4.0) | - |
| `moe-clamped` | gpt-oss's experts (a `quick_gelu` gate, a limit, an offset, biases), a router bias | 1.1e-5 | 1.2e-5 | 1.7 (1.1) | 1.1 |
| `mamba2` | Mamba-2 of 16 heads in 2 groups, beside attention | 2.1e-5 | 2.0e-5 | 0.19 (0.20); SSM32 0.19 | 0.15 |
| `gdn` | Gated DeltaNet of 2 q/k heads and 4 value heads, beside gated attention | - | - | 0.57 (0.56); SSM32 0.48 | 0.45 |

Every float32 run decoded bux's greedy tokens exactly. In bfloat16 an MoE's error is
large, as bux's own is: where two experts' scores are within bfloat16's rounding, a
token's top-k changes, and so does its output. `gdn` runs in bfloat16 alone. With
`--mamba-ssm-cache-dtype float32` (SSM32) the recurrent state stays in float32, as
bux's does: Gated DeltaNet's error falls, Mamba-2's stays where it was.
