# Exporting models

Take a trained model out of bux to evaluate or serve it elsewhere. `bux export`
writes a run as a [transformers](https://huggingface.co/docs/transformers) model
directory, with its weights and tokenizer, and can check that the export computes
what bux computes. The same directory serves in [vLLM](vllm.md). bux generates the
model's code from its architecture instead of mapping it onto an existing model
class, so a novel architecture exports as readily as Llama does.

## Quick start

Export a run's latest checkpoint, and check it against bux:

```console
$ bux export runs/smoke-5a4f0f7d85 exported/ --check
smoke: step 50 exported to exported, its weights in float32
eager: logits within 2.9e-07 of bux's largest (tolerance 1e-04); greedy decoding with cache as bux, no cache as bux; a left-padded batch as its prompts alone
sdpa: logits within 2.9e-07 of bux's largest (tolerance 1e-04); greedy decoding with cache as bux, no cache as bux; a left-padded batch as its prompts alone
```

Load it in transformers as remote code. A run that trained with a tokenizer exports
it alongside the model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("exported/", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("exported/")
```

Tools that load transformers models, such as evaluation harnesses, take the
directory the same way, with `trust_remote_code=True`. To serve it in vLLM, install
bux's plugin into vLLM's environment and point vLLM at the directory
([Serving in vLLM](vllm.md)):

```sh
pip install vllm ./src/bux_vllm     # from a checkout of bux
vllm serve exported/                # add --tokenizer NAME if the run had none
```

`bux export` needs PyTorch, transformers and safetensors. A checkout of bux has them
already; an installed bux needs the `export` extra, `pip install "bux[export]"`.

## Exporting a run

`bux export RUN_DIR OUT` writes the model of a trained run as a directory that
transformers loads as it is. It reads the run's `run.json` and restores the latest
checkpoint, or the one `--step N` names, on one CPU. The mesh, layout and methods
that trained the run don't matter:

- shards from FSDP and tensor parallelism restore as whole arrays;
- a scanned repeat's passes, stacked along a leading dimension or spread over
  pipeline stages, come apart into one module per pass, as `methods.repeat = unroll`
  builds them;
- the rest of the model's state comes along, such as an MoE's router bias.

Besides the model ([What an export holds](#what-an-export-holds)), the directory
holds:

| File | What it is |
|------|------------|
| `tokenizer.json`, `tokenizer_config.json` | the run's tokenizer (a Hub repo's, or its `tokenizer.json`) with its `bos`, `eos` and `added` tokens at the ids bux gave them; `AutoTokenizer.from_pretrained(OUT)` loads it |
| `generation_config.json` | the tokenizer's bos, eos and pad tokens, for `generate`; the pad token is the eos token if the tokenizer has none |
| `README.md` | a model card: the run's name, id and step, how to load it, and its `run.bux` |

A run that trains without a tokenizer, on `synthetic` data or on ids it is given,
exports without one. The export's `config.json` sets `max_position_embeddings` to the
run's `seq_len`, the longest sequence the model is known to handle. vLLM serves up to
that length unless `--max-model-len` says otherwise.

The weights keep the parameters' dtype, which is float32 whatever precision the run
computed in, unless `--dtype` picks `bfloat16` or `float16`. In Python:

```python
from bux.export.runs import export, restore

done = export("runs/smoke-5a4f0f7d85", "exported/", dtype="bfloat16", check=True)
done.report.check()  # raises unless the export computes what bux does

restored = restore("runs/smoke-5a4f0f7d85", step=40)  # the run, and its bux model
```

### What the check checks

`--check` compares the export with bux on the run's own weights
([Checking an export](#checking-an-export)). It prints one line per attention
implementation, `eager` and `sdpa`, and exits with 1 unless all of these hold:

- every head's logits are within 1e-4 of bux's largest logit;
- greedy tokens, with the cache and without it, equal bux's;
- a left-padded batch decodes exactly as each of its prompts does alone.

Both sides compute in float32, on the weights the directory holds. With
`--dtype bfloat16`, bux's weights are rounded to bfloat16 as well, so both sides
compute with the same numbers. A run that trained in bfloat16 is checked the same
way.

Float32 is what makes the check sharp. In bfloat16, two correct implementations
differ by bfloat16's own rounding, near 1e-2 of the logits, which would hide a wrong
weight. In float32 only the order of operations differs: the smoke run's export
above is within 3e-7 of bux, and so is its bfloat16 export (Mac CPU, normal init,
2026-10-26). A misplaced or mistransposed weight moves the logits by far more than
1e-4, and the margin leaves room for depth: deep stacks of delta-rule layers round
faster ([Float32 and depth](#float32-and-depth)). Greedy tokens must match exactly;
only a tie between the two top logits closer than float32's error could flip one.

## Exporting an architecture

Given a `.bux` file instead of a run, `bux export` writes the file's model without
weights: its code and `config.json`, but no `model.safetensors` (it removes one an
earlier export left in `OUT`). vLLM serves it on random weights
(`--load-format dummy`), so you can measure an architecture's speed
at any size before training anything. Settings after `OUT` redefine the file's, as
they do for `bux train`:

```console
$ bux export examples/experiments/old_vs_new/new_2026.bux exported/new-3072 'd_model = 3072'
examples/experiments/old_vs_new/new_2026.bux: 91.1B parameters exported to exported/new-3072, without weights
```

The config's `dtype` is bfloat16 unless `--dtype` says otherwise, and `--entry`
exports another definition, as for `bux check`. bux never builds the model, only its
shapes, so a model of a hundred billion parameters exports in seconds on a laptop.
The tutorial [Old against new at the same serving cost](../tutorials/old-vs-new.md)
times a family of widths this way.

## Exporting from Python

{py:func}`bux.export.huggingface.save` exports any bux model, trained or not:

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

### Checking an export

{py:func}`bux.export.check.compare` exports a model, loads it back with
transformers, and compares it with bux in float32. It compares every head's logits,
greedy decoding (with the cache, without it, and by bux's full forward passes), and
a left-padded batch against each of its prompts alone. The test suite checks every
exported layer this way (`tests/test_export*.py`).

```python
from bux.export.check import compare

compare(arch).check()  # fresh weights, perturbed; raises unless all agree
```

## What exports

Each layer type exports once, as a class in `bux.export.hf`, and every architecture
built from exported layers exports. Anything else raises `NotImplementedError`,
naming the layer or the setting.

| Layers | What exports |
|--------|--------------|
| `embed`, `head`, `mlp`, `gated_mlp`, `linear`, `act`, `scale`, `dropout`, `layernorm`, `rmsnorm`, `dyt` | every setting and activation |
| `moe` | every setting inference reads: the routers, group-limited routing, the load-balancing bias (`aux_free`, `smebu`), shared experts, LatentMoE, the expert norm, and experts of either MLP type. The routed experts run on transformers' experts implementations |
| `gqa` | every setting and port: RoPE (`pi`, `llama3`, `yarn`; partial, interleaved), QK-norm per head or across all heads, biases, fused or separate projections, head dims, scale, sliding windows, sinks, softcap, gates, output norm, length scale, relative bias, the conv on keys and values, and the indexer; the `v`, `kv` and `index` ports. Dropout applies only in training |
| `indexer` | every setting inference reads: DSA's top-k keys, QSA's mean-pooled blocks, MSA's max-pooled blocks and per-group picks, plain or ReLU'd dot products, weighted or summed. Its loss applies only in training |
| `mla` | every setting, the indexer and the `index` port; keys and values are cached as projected up from the latents |
| `csa` | every setting, its candidate pools and the `kv`, `index` and `candidates` ports; the indexer, except max-pooled blocks, which bux does not build on compressed latents either. Every token's latents are cached, and the window is a mask over them |
| `conv` | every setting; the tail of its inputs stays in the cache while decoding |
| `gdn`, `kda`, `gdn2`, `deltanet`, `gla`, `lightning` | every setting: value heads, the short conv, the output gate and its rank, the decay rank and lower bound of KDA and GDN-2, GDN-2's erase and write gates and negative eigenvalues, and lightning's rates by its depth among the token mixers |
| `rwkv7` | every setting, and its `v` port (the value residual) |
| `mamba`, `mamba2`, `short_conv` | every setting; `dt_min` and `dt_max` matter only at initialization |
| `engram` | on a single or a widened stream; its last tokens and its conv's input tail stay in the cache while decoding. A token's class is its id mod `vocab_classes`, as in bux |
| `nextn` | on a single or a widened stream; every head's logits on whole sequences (`all_heads=True`), skipped during generation |
| `merge` | yes |
| residual algebras | `hc`, `mhc`, `gr`, `attnres`, `vpgate`, with every setting, single-pass included |
| the graph | sequences, residual branches, sums, products, constants, negations, repeats, taps, shared definitions |

A few combinations of exported settings raise: a layer that reads a `kv` port and
also has a `v` port or an indexer of its own, and candidate pools of a `csa` indexer
with `block_size > 1`.

## What an export holds

| File | What it is |
|------|------------|
| `config.json` | the IR, each layer's site, the kinds of cache layers, the names no module takes (`bux_reserved`), and the `auto_map` that points transformers at the code |
| `modeling_bux.py` | the architecture's graph, generated: a class per repeated block, a line per node |
| `common.py`, `attention.py`, ... | bux's layers in PyTorch, the same files for every model (`bux.export.hf`) |
| `configuration_bux.py` | the config class |
| `model.safetensors` | the weights, named by the paths of the generated modules |

The generated code reads like a hand-written model. Here is Llama 3's block, shrunk:

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

- **The plan** (`bux.export.plan`) places every layer in a tree of named modules:
  the model, each repeated block (the longest repeat is `model.layers`), and each
  shared definition, once, at `model.shared.<key>`. A layer's path is the prefix of
  its weights, as in `model.layers.3.attn.q.weight`. Under a residual algebra, each
  outermost residual branch is a module too, named by the algebra and holding its
  parameters: `model.layers.3.mhc.bias`. A label that torch's or transformers'
  module classes have as an attribute (`forward`, `config`, `training`) takes a
  suffix, `forward_1`, from the names {py:data}`bux.export.plan.RESERVED` lists. The
  plan imports neither JAX nor PyTorch, so every exporter places the layers the same
  way.
- **The code** (`bux.export.codegen`) turns each scope of the plan into a class.
  `[f]` is `x + f(x)`, `f + g` and `f * g` add and multiply their branches, `f^n`
  loops over a `ModuleList`, and `@name` and `$name` push and read the forward pass's
  taps, as in bux. Under a residual algebra, an outermost branch becomes
  `h, kept = self.mhc.read(x, ctx)`, then the body on `h`, then
  `self.mhc.write_back(kept, y)`; the embedding's output widens, as in bux.
- **The heads.** `model.model(ids, all_heads=True)` returns every head's logits on
  whole sequences, without a cache. Otherwise the forward pass stops after the first
  head, as it does in generation, so a multi-token prediction module after it
  (`nextn` reads tokens ahead) costs nothing and keeps no cache.
- **The layers** (`bux.export.hf`) are bux's, in PyTorch: one class per family of
  layer types, built from the layer's config as JSON. Their attributes mirror the
  NNX modules', so weights convert by one rule (`bux.export.weights`): a linear's
  kernel, transposed, an embedding's table and a norm's scale each become `weight`.
- **The cache.** Each call of a layer that keeps state takes the next of
  transformers' cache layers, in the order the calls run, as `config.layer_types`
  lists them. A shared layer called three times keeps three; layers after the first
  head keep none. What a single cache layer needs, such as a sliding window or a
  number of conv states, is in `config.per_layer_config`, so two layers can keep
  windows of different sizes.

## Softmax attention

`gqa` and `mla` attend through transformers' attention interface (`eager`, `sdpa`,
`flash_attention_2`, `flex_attention`) and keep their keys and values in its cache.
Some settings need the attention logits: sinks, a softcap, a relative bias, a length
scale, or an indexer's selected keys. With any of these, the layer computes the
softmax itself, in float32 as bux does, whichever implementation the model loads
with. sdpa and flash attention would drop the sinks, and sdpa has no softcap. `csa`
always computes its own softmax, over its window and its compressed latents at once.

- **Windows.** A layer with `sliding_window_size` keeps that many keys in its own
  `sliding_attention` cache layer, and reads the mask of its window.
- **Shared keys and values.** A layer that reads a `kv` port attends to the keys and
  values that the writer's cache layer returns (all of them, while decoding) and
  keeps none of its own. Its window may be no longer than the writer's.
- **Sparse attention.** The indexer's keys are cached beside the layer's, in an
  `indexed_attention` cache layer, so each decoding step can rank every earlier key.
  With a conv on the keys and values too, the cache layer also keeps the conv's
  states (`indexed_hybrid`, a kind of cache layer of the export's own).
  A sparse layer caches every key and selects with a mask over them, which the
  `index` port passes on; its window applies on top. Ties rank as in bux: by the
  floats' bits, the first of equal ones winning.
- **Blocks of keys.** With `block_size > 1` the indexer keeps whole blocks, counted
  from a document's first key. Mean-pooled (`pool=mean`, Qwen3.8 QSA), the cache
  keeps the projected keys, and each step pools them again; a query also keeps its
  own unfinished block. Max-pooled (`pool=max`, MiniMax MSA), the cache keeps each
  key normalized and rotated, as without blocks. Each step scores a block by the
  best of its keys that the query sees, and the query's own block is always among
  the `top_k // block_size`.
- **Picks per group.** With `per_group=true`, each indexer head picks keys for its kv
  head's group of query heads (MSA): the mask is b t g s, and each query head attends
  to its group's keys, through the `index` port too.
- **Compressed sparse attention.** A `csa` layer's cache layer keeps every token's
  latent as its keys, and as its values the compressed latent that the token
  completes, along with the indexer's key for it. Its conv state keeps the last
  tokens that the next compressed latent pools (a `hybrid` cache layer).

The cache is transformers' `DynamicCache`. A static cache
(`cache_implementation="static"`) is not exported yet.

## Mixtures of experts

An exported `moe` routes as bux does. The scores and the top-k choice are in float32,
and equal scores pick the lower expert. The load-balancing bias of `aux_free` or
`smebu`, which the checkpoint holds as `router_bias`, affects only the choice. The
routed experts are stacked the way transformers' MoE models keep theirs, so
transformers' experts implementations run them: `gate_up_proj` is (E, 2f, d), the
gate's rows first, and `down_proj` is (E, d, f).

```python
model = AutoModelForCausalLM.from_pretrained(
    "exported/", trust_remote_code=True, experts_implementation="grouped_mm"
)
```

`grouped_mm`, transformers' default, and `batched_mm` run every setting of the
experts. `eager` loops over them and is the reference. A kernel that fuses the
activation itself (`sonicmoe`) fits only SwiGLU experts. Settings that only training
reads are left out: the balance loss, how the bias moves (its rate, `center_bias`,
SMEBU's soft clamp and momentum), and random routing.

## Recurrent layers

Linear attention and the state space models keep their state in transformers'
cache, one `linear_attention` layer each (`conv` for `short_conv`), along with the
last inputs of their short convs. A prefill runs the recurrence in chunks of 64
positions, with matmuls within a chunk and a loop across chunks, as fla's chunked
kernels do. A decoding step advances one position from the cached state. Mamba's
scan, with one decay per channel and state dim, runs one position at a time, as in
bux. All of it is plain PyTorch, in float32 whatever the model's dtype (float64 in a
float64 model). It uses no kernels from fla, mamba_ssm or causal_conv1d yet.

A padded token in a left-padded batch leaves the state unchanged: its input is
zeroed and its decay is 1. In a model with no attention layers, no cache layer
counts the tokens it holds, so the model's first forward pass creates the cache
rather than `generate`.

## Float32 and depth

Float32 rounding grows with depth, and faster through the delta rule than through
attention. The table gives the largest relative error of any head's logits against
the export run in float64. Setup: eager attention; the weights `compare` makes
(seed 0, perturbed by 0.1); batches of 2 rows by 12 tokens; `d_model = 32` and a
vocabulary of 64; on an Apple-silicon Mac's CPU, 2026-10-26.

| Model | bux in float32 | the export in float32 |
|-------|----------------|-----------------------|
| `gdn` blocks, 1 / 2 / 8 / 12 | 1.4e-6 / 2.0e-6 / 1.0e-5 / 1.5e-5 | 8.4e-7 / 2.2e-6 / 1.5e-5 / 1.8e-5 |
| `gdn` blocks, 48 | 2.0e-4 | 1.6e-4 |
| Qwen3-Next shrunk, 48 blocks (36 `gdn`) | 1.8e-4 | 2.5e-4 |
| `gqa` blocks, 48 | 6.4e-6 | 4.1e-6 |

That is about 5e-6 per `gdn` layer, on both sides alike, so a deep recurrent stack
exceeds 1e-4 from float32 rounding alone. The tests shorten such stacks (Qwen3-Next's
to 12 blocks) rather than loosen the check.

Rounding also differs between CPUs. In Kimi Linear's shrunk zoo model (27 token
mixers), bux and the export are each within 5e-5 of the float64 export on the Mac,
and 3.1e-5 apart; on an x86 CI runner they were 1.8e-4 apart (2026-10-26). So the
tests of whole zoo models hold logits to 1e-3, and each layer's own cases to 1e-4.
Greedy tokens and padded batches must match bux exactly in both.

```{toctree}
:hidden:

vllm
```
