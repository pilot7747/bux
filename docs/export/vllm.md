(export-vllm)=
# Serving in vLLM

An export runs in [vLLM](https://docs.vllm.ai) as a native model, with its paged
attention, fused MoE kernels, Mamba cache, CUDA graphs and tensor parallelism. vLLM
does not run a checkpoint's own code, so bux brings a plugin for it: a model,
`BuxForCausalLM`, that builds any exported architecture from the export's
`config.json`, out of vLLM's own layers. The checkpoint is the one
{py:func}`bux.export.huggingface.save` writes; nothing is exported twice.

## Installing

The plugin is `bux.export.vllm`, a folder of files that import nothing of bux, only
PyTorch, transformers and vLLM. It installs alone, as the package `bux_vllm`, into
vLLM's environment, which need not have bux, JAX or Python 3.14:

```bash
pip install vllm "bux-vllm @ git+https://github.com/pilot7747/bux#subdirectory=src/bux/export/vllm"
# or from a checkout
pip install vllm ./src/bux/export/vllm
```

vLLM loads it by its entry point (`vllm.general_plugins`), in every process it
starts: it registers the config of `model_type` `bux`, so that no remote code runs,
and the model class `BuxForCausalLM`.

## Serving

An export has no tokenizer: give vLLM the one the model was trained with, and the
dtype to run in.

```bash
vllm serve exported/ --tokenizer meta-llama/Llama-3.1-8B --dtype bfloat16 \
  --max-model-len 8192 --tensor-parallel-size 2
```

```python
from vllm import LLM, SamplingParams

llm = LLM(model="exported/", tokenizer="meta-llama/Llama-3.1-8B", dtype="bfloat16")
print(llm.generate("The capital of France is", SamplingParams(max_tokens=8)))
```

The model's output is its first head: layers after it, a multi-token prediction
module's, are not built, and vLLM skips their weights (it drafts tokens by MTP only
for the model types it knows). A layer or setting the plugin does not run raises
`NotImplementedError` as the model is built, naming it.

## How it works

- **The module tree.** `bux.export.vllm.plan` places every layer where
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
| `embed`, `head` | yes, the vocabulary split over ranks; tied or not, a bias, soft-capping; not `embed(norm=...)` |
| `rmsnorm`, `layernorm`, `mlp`, `gated_mlp`, `linear`, `act`, `scale`, `dropout` | yes, but the MLPs' `limit`, `up_softcap` and `up_offset` |
| `gqa` | RoPE (`pi`, `llama3`, `yarn`), QK-norm, GQA and MQA, biases, sliding windows, soft-capping, sinks, its scale; not yet `gated`, `fused_qkv`, `output_norm` or ports |
| `moe` | softmax, sigmoid and `sqrtsoftplus` routers, groups, the aux-free bias, shared experts and their gate, gated and plain experts (`swish`, `gelu`, `gelu_tanh`, `relu2`); not LatentMoE, `expert_norm` or `shared_expert_sink` |
| `mamba2`, `gdn` | yes, by vLLM's layers, one type and shape of them a model |
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
