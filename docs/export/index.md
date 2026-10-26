# Exporting models

A bux model exports to [transformers](https://huggingface.co/docs/transformers) as a
model of its own: bux compiles the architecture to a `modeling_bux.py`, not to an
existing model class, so a novel architecture exports as readily as Llama does. The
weights go with it, as safetensors, and the export loads as remote code:

```python
from flax import nnx
from transformers import AutoModelForCausalLM

from bux.architecture import Model
from bux.export.huggingface import save
from bux.lang import compile

arch = compile("""
d_model    = 64
vocab_size = 256

attn  = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn   = gated_mlp(intermediate_size=128, activation=swish)
block = [rmsnorm attn] [rmsnorm ffn]

model = embed block^2 rmsnorm head
""")
save(arch, Model(arch, rngs=nnx.Rngs(0)), "exported/")
model = AutoModelForCausalLM.from_pretrained("exported/", trust_remote_code=True)
```

`bux export` needs PyTorch, transformers and safetensors: the `export` extra,
`pip install "bux[export]"` (a checkout of bux has them already).

## What an export holds

| File | What it is |
|------|------------|
| `config.json` | the IR, each layer's site, the cache layers' kinds, and the `auto_map` that points transformers at the code |
| `modeling_bux.py` | the architecture's graph, generated: a class per repeated block, a line per node |
| `common.py`, `attention.py`, ... | bux's layers in PyTorch, the same files for every model (`bux.export.hf`) |
| `configuration_bux.py` | the config's class |
| `model.safetensors` | the weights, by the paths of the generated modules |

The generated code reads like a hand-written model. Llama 3's block, shrunk:

```python
class Block0(nn.Module):
    """bux: block."""

    def __init__(self, config: BuxConfig, prefix: str):
        super().__init__()
        self.rmsnorm = Norm(RMSNORM, 64, site(config, prefix + "rmsnorm"))
        self.attn = Attention(ATTN, 64, site(config, prefix + "attn"))
        self.rmsnorm_1 = Norm(RMSNORM, 64, site(config, prefix + "rmsnorm_1"))
        self.ffn = MLP(FFN, 64, site(config, prefix + "ffn"))

    def forward(self, x: Any, ctx: Ctx) -> Any:
        h1 = self.rmsnorm(x, ctx)
        h2 = self.attn(h1, ctx)
        h3 = x + h2
        h4 = self.rmsnorm_1(h3, ctx)
        h5 = self.ffn(h4, ctx)
        h6 = h3 + h5
        return h6
```

## How it works

- **The plan** (`bux.export.plan`) places every layer in a tree of named modules: the
  model's, each repeated block's (the longest repeat is `model.layers`), and each
  shared definition's, once, at `model.shared.<key>`. A layer's path is the prefix of
  its weights, `model.layers.3.attn.q.weight`. Under a residual algebra, each
  residual branch outside any other is a module too, named by the algebra, which
  holds the algebra's parameters: `model.layers.3.mhc.bias`. The plan imports
  neither JAX nor PyTorch, so that every exporter places the layers alike.
- **The code** (`bux.export.codegen`) turns each scope of the plan into a class.
  `[f]` is `x + f(x)`, `f + g` and `f * g` add and multiply their branches, `f^n`
  loops over a `ModuleList`, and `@name` and `$name` push and read the forward pass's
  taps, as in bux. Under a residual algebra, a branch outside any other is
  `h, kept = self.mhc.read(x, ctx)`, the body on `h`, then
  `self.mhc.write_back(kept, y)`, and the embedding's output widens, as bux's does.
- **The heads.** The model's forward gives every head's logits with
  `model.model(ids, all_heads=True)`, on whole sequences without a cache. Otherwise
  it stops after the first head, as it generates: multi-token prediction after it
  (`nextn` reads tokens ahead) costs nothing and keeps no cache.
- **The layers** (`bux.export.hf`) are bux's, in PyTorch: one class per family of
  layer types, built from the layer's config as JSON. Their attributes mirror the
  NNX modules', so weights convert by one rule (`bux.export.weights`): a linear's
  kernel, transposed, an embedding's table and a norm's scale become `weight`.
- **The cache.** Each call of a layer that keeps state takes the next of
  transformers' cache layers, in the order the calls run, as `config.layer_types`
  lists them; a shared layer called three times keeps three, and the layers
  after the first head keep none. What one cache layer needs of its own, a
  sliding window or a number of conv states, is in `config.per_layer_config`,
  so that two layers keep windows of their own.

## What exports

A layer type exports once, as a class in `bux.export.hf`, and every architecture made
of exported layers exports; one that is not raises `NotImplementedError` naming the
layer or the setting.

| Layers | Exported |
|--------|----------|
| `embed`, `head`, `mlp`, `gated_mlp`, `linear`, `act`, `scale`, `dropout`, `layernorm`, `rmsnorm`, `dyt` | yes, every setting and activation |
| `moe` | yes, every setting inference reads: the routers, group-limited routing, the load-balancing bias (`aux_free`, `smebu`), shared experts, LatentMoE, the expert norm, experts of either MLP type; the routed experts by transformers' experts implementations |
| `gqa` | every setting and port: RoPE (`pi`, `llama3`, `yarn`; partial, interleaved), QK-norm of each head or of all heads together, biases, fused or not, head dims, scale, sliding windows, sinks, softcap, gates, output norm, length scale, relative bias, the conv on keys and values, the indexer; `v`, `kv` and `index` ports. Dropout is training's alone |
| `indexer` | every setting inference reads: DSA's top-k keys, QSA's mean-pooled blocks, MSA's max-pooled blocks and picks for each kv head's group, plain dot products or ReLU'd, weighted or summed; its loss is training's alone |
| `mla` | every setting, the indexer and the `index` port; keys and values cached as projected up from the latents |
| `csa` | every setting, the indexer (but max-pooled blocks, which bux does not build on compressed latents either) and its candidate pools, and the `kv`, `index` and `candidates` ports; every token's latents cached, the window a mask over them |
| `conv` | yes, its inputs' tail kept in the cache as it decodes |
| `gdn`, `kda`, `deltanet`, `gla`, `lightning` | yes, every setting: value heads, the short conv, the output gate and its rank, KDA's decay rank and lower bound, lightning's rates by its depth among the token mixers |
| `rwkv7` | yes, every setting, and its `v` port (the value residual) |
| `mamba`, `mamba2`, `short_conv` | yes, every setting; `dt_min` and `dt_max` only start training |
| `engram` | yes, on one stream or a widened one; its last tokens and its conv's inputs' tail kept in the cache as it decodes. A token's class is its id mod `vocab_classes`, as bux makes them |
| `nextn` | yes, on one stream or a widened one; every head's logits on whole sequences (`all_heads=True`), and skipped as the model generates |
| `merge` | yes |
| residual algebras | `hc`, `mhc`, `gr`, `attnres`, `vpgate`, with every setting, single-pass included |
| the graph | sequences, residual branches, sums, products, constants, negations, repeats, taps, shared definitions |

## Checking an export

`bux.export.check.compare` exports a model, loads it back with transformers, and
compares it with bux, in float32: every head's logits, greedy decoding with the cache,
without it and by bux's full forwards, and a left-padded batch against each of its
prompts alone. Each exported layer is tested so (`tests/test_export*.py`).

```python
from bux.export.check import compare

compare(arch).check()  # fresh weights, perturbed; raises unless all agree
```

### Float32 and depth

The check holds logits to 1e-4 of the largest, in float32. Float32's own rounding
grows with depth, and through the delta rule faster than through attention. Against
the export run in float64 (eager attention; the weights `compare` makes, seed 0,
perturbed by 0.1; rows of 2 by 12 tokens; `d_model = 32`, a vocabulary of 64; on an
Apple-silicon Mac's CPU, 2026-10-26), the largest relative error of any head's logits:

| Model | bux in float32 | the export in float32 |
|-------|----------------|-----------------------|
| `gdn` blocks, 1 / 2 / 8 / 12 | 1.4e-6 / 2.0e-6 / 1.0e-5 / 1.5e-5 | 8.4e-7 / 2.2e-6 / 1.5e-5 / 1.8e-5 |
| `gdn` blocks, 48 | 2.0e-4 | 1.6e-4 |
| Qwen3-Next shrunk, 48 blocks (36 `gdn`) | 1.8e-4 | 2.5e-4 |
| `gqa` blocks, 48 | 6.4e-6 | 4.1e-6 |

That is about 5e-6 a `gdn` layer, on both sides alike: a deep recurrent stack passes
1e-4 by float32's rounding alone. The tests shorten such stacks (Qwen3-Next's to 12
blocks) rather than loosen the check.

## Exporting a run

`bux export` writes a trained run's model, from its latest checkpoint, as a directory
that transformers loads as it is:

```console
$ bux export runs/smoke-f114ac17ae exported/ --check
smoke: step 50 exported to exported, its weights in float32
eager: logits within 5.1e-07 of bux's largest (tolerance 1e-04); greedy decoding with cache as bux, no cache as bux; a left-padded batch as its prompts alone
sdpa: logits within 5.0e-07 of bux's largest (tolerance 1e-04); greedy decoding with cache as bux, no cache as bux; a left-padded batch as its prompts alone
```

It reads the run's `run.json` and restores a checkpoint (`--step N`, or the latest)
on the CPU of one process, whatever mesh, layout and methods trained it: FSDP's and
tensor parallelism's shards restore as whole arrays, and a scanned repeat's passes,
stacked along a leading dimension or spread over a pipeline's stages, come apart
into a module each, as `methods.repeat = unroll` builds them. The rest of the
model's state comes with them, an MoE's router bias too. Besides the model, the
directory holds:

| File | What it is |
|------|------------|
| `tokenizer.json`, `tokenizer_config.json` | the run's tokenizer, a Hub repo's or its `tokenizer.json`, with its `bos`, `eos` and `added` tokens, at the ids bux gave them: `AutoTokenizer.from_pretrained(OUT)` loads it |
| `generation_config.json` | the tokenizer's bos, eos and pad tokens for `generate`; the pad is the eos where the tokenizer has none |
| `README.md` | a model card: the run's name, id and step, how to load it, and its `run.bux` |

Its `config.json` gives the run's `seq_len` as `max_position_embeddings`, the
longest sequence the model is known to take, which vLLM serves up to unless
`--max-model-len` says otherwise.

A run that trains without a tokenizer (on `synthetic` data, or on ids it is given)
exports without one. The weights are in the parameters' dtype, float32 as bux keeps
them whatever a run computes in, or in `--dtype` (`bfloat16`, `float16`). In
Python:

```python
from bux.export.runs import export, restore

done = export("runs/smoke-f114ac17ae", "exported/", dtype="bfloat16", check=True)
done.report.check()  # raises unless the export computes what bux does

restored = restore("runs/smoke-f114ac17ae", step=40)  # the run, and its bux model
```

### What the check checks

`--check` runs `compare` ([Checking an export](#checking-an-export)) on the run's
own weights, not perturbed ones, prints a line for each attention implementation,
and fails the command if any check fails: every head's logits within 1e-4 of bux's
largest, greedy tokens with the cache and without equal to bux's, and a left-padded
batch decoding as each of its prompts alone.

Both sides compute in float32, on the weights the directory holds: with `--dtype
bfloat16` bux's are rounded to bfloat16 too, so that the two compute with the same
numbers. A run that trained in bfloat16 is checked so as well. In bfloat16, two
correct implementations would differ by bfloat16's own rounding, near 1e-2 of the
logits, which would hide a wrong weight. In float32 only the order of operations
differs: the smoke run's export above is within 5e-7 of bux, and its bfloat16 export
within 6e-7 (on a Mac's CPU, 2026-10-26). 1e-4 leaves room for depth, and a misplaced
or mistransposed weight moves the logits by far more; a deep stack of delta-rule
layers rounds faster ([Float32 and depth](#float32-and-depth)). Greedy tokens must be
equal: only a tie between the two top logits closer than float32's error could flip
one.

## Softmax attention

`gqa` and `mla` attend by transformers' attention interface (`eager`, `sdpa`,
`flash_attention_2`, `flex_attention`) and keep their keys and values in its cache.
Where a setting needs the logits (sinks, a softcap, a relative bias, a length scale,
an indexer's kept keys), the layer computes the softmax itself, in float32 as bux
does, whichever implementation the model loads with: sdpa and flash attention would
drop the sinks, and sdpa has no softcap. `csa` always computes its own, over its
window and its compressed latents at once.

- **Windows.** A layer with `sliding_window_size` keeps that many keys in a
  `sliding_attention` cache layer, its window its own, and reads the mask of its
  window.
- **Shared keys and values.** A layer that reads a `kv` port attends to the keys and
  values the writer's cache layer returns, all of them while decoding, and keeps
  none of its own. Its window may be no longer than the writer's.
- **Sparse attention.** The indexer's keys are cached beside the layer's, in an
  `indexed_attention` cache layer, so that each decoding step ranks every key before
  it. What a layer keeps is a mask over its cached keys, which the `index` port
  passes on; a sparse layer keeps every key, which the mask indexes, and applies its
  window to them. Ties rank as in bux: by the floats' bits, the first of equal ones.
- **Blocks of keys.** With `block_size > 1` the indexer keeps whole blocks, counted
  from a document's first key. Mean-pooled (`pool=mean`, Qwen3.8 QSA), the cache
  keeps the keys as projected, and each step pools them again; a query also keeps
  its own unfinished block. Max-pooled (`pool=max`, MiniMax MSA), the cache keeps
  each key normalized and rotated, as without blocks, and each step scores a block
  by the best of its keys the query sees, its own block always among the
  `top_k // block_size`.
- **Picks for each group.** With `per_group=true` each indexer head picks for its kv
  head's group of query heads (MSA): the mask is b t g s, and each query head
  attends to its group's keys, through the `index` port too.
- **Compressed sparse attention.** A `csa` layer's cache layer keeps every token's
  latent as its keys and, as its values, the compressed latent the token ends, with
  the indexer's key of it; its conv state keeps the last tokens that the next
  compressed latent pools (a `hybrid` layer).

The cache is transformers' `DynamicCache`; a static one
(`cache_implementation="static"`) is not exported yet.

## Mixtures of experts

An exported `moe` routes as bux does: the scores and the top-k choice in float32,
equal scores picking the lower expert, and the load-balancing bias of `aux_free`
or `smebu`, which the checkpoint holds as `router_bias`, in the choice alone. Its routed experts are stacked as
transformers' MoE models keep theirs, `gate_up_proj` (E, 2f, d), the gate's rows
first, and `down_proj` (E, d, f), so that transformers' experts implementations run
them:

```python
model = AutoModelForCausalLM.from_pretrained(
    "exported/", trust_remote_code=True, experts_implementation="grouped_mm"
)
```

`grouped_mm`, transformers' default, and `batched_mm` run every setting of the
experts; `eager` is a loop over them, the reference. A kernel that fuses the
activation itself (`sonicmoe`) fits SwiGLU experts alone. The settings only training
reads are left out: the balance loss, how the bias moves (its rate, `center_bias`,
SMEBU's soft clamp and momentum), and random routing.

## Recurrent layers

Linear attention and the state space models keep their state in transformers' cache,
a `linear_attention` layer each (`conv` for `short_conv`), with the last inputs of
their short convs. A prefill runs the recurrence by chunks of 64 positions, with
matmuls within a chunk and a loop across them, as fla's chunked kernels do; a
decoding step runs one position from the state the cache holds. Mamba's scan, one
decay per channel and state dim, runs one position at a time, as bux runs it. All of
it is PyTorch, in float32 whatever the model's dtype (float64 in a float64 model),
with no kernels of fla, mamba_ssm or causal_conv1d yet.

A padded token of a left-padded batch leaves the state as it was: its input is zeroed
and its decay is 1. A model of recurrent layers alone has no attention layer to count
the tokens its cache holds, so its first forward makes the cache, not `generate`.

## Serving in vLLM

The same checkpoint runs in vLLM as a native model, by bux's plugin, with vLLM's
paged attention, fused MoE kernels, Mamba cache and tensor parallelism:
[Serving in vLLM](vllm.md).

```{toctree}
:hidden:

vllm
```
