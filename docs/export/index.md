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
  its weights, `model.layers.3.attn.q.weight`. The plan imports neither JAX nor
  PyTorch, so that every exporter places the layers alike.
- **The code** (`bux.export.codegen`) turns each scope of the plan into a class.
  `[f]` is `x + f(x)`, `f + g` and `f * g` add and multiply their branches, `f^n`
  loops over a `ModuleList`, and `@name` and `$name` push and read the forward pass's
  taps, as in bux.
- **The layers** (`bux.export.hf`) are bux's, in PyTorch: one class per family of
  layer types, built from the layer's config as JSON. Their attributes mirror the
  NNX modules', so weights convert by one rule (`bux.export.weights`): a linear's
  kernel, transposed, an embedding's table and a norm's scale become `weight`.
- **The cache.** Each call of a layer that keeps state takes the next of
  transformers' cache layers, in the order the calls run, as `config.layer_types`
  lists them; a shared layer called three times keeps three. What one cache layer
  needs of its own, a sliding window or a number of conv states, is in
  `config.per_layer_config`, so that two layers keep windows of their own.

## Checking an export

`bux.export.check.compare` exports a model, loads it back with transformers, and
compares it with bux, in float32: every head's logits, greedy decoding with the cache,
without it and by bux's full forwards, and a left-padded batch against each of its
prompts alone. Each exported layer is tested so (`tests/test_export*.py`).

```python
from bux.export.check import compare

compare(arch).check()  # fresh weights, perturbed; raises unless all agree
```

## What exports

A layer type exports once, as a class in `bux.export.hf`, and every architecture made
of exported layers exports; one that is not raises `NotImplementedError` naming the
layer or the setting.

| Layers | Exported |
|--------|----------|
| `embed`, `head`, `mlp`, `gated_mlp`, `linear`, `act`, `scale`, `dropout`, `layernorm`, `rmsnorm`, `dyt` | yes, every setting and activation |
| `moe` | yes, every setting inference reads: the routers, group-limited routing, the aux-free bias, shared experts, LatentMoE, the expert norm, experts of either MLP type; the routed experts by transformers' experts implementations |
| `gqa` | every setting and port: RoPE (`pi`, `llama3`, `yarn`; partial, interleaved), QK-norm, biases, fused or not, head dims, scale, sliding windows, sinks, softcap, gates, output norm, length scale, relative bias, the conv on keys and values, the indexer; `v`, `kv` and `index` ports. Dropout is training's alone |
| `mla` | every setting, the indexer and the `index` port; keys and values cached as projected up from the latents |
| `csa` | every setting, the indexer and its candidate pools, and the `kv`, `index` and `candidates` ports; every token's latents cached, the window a mask over them |
| `conv` | yes, its inputs' tail kept in the cache as it decodes |
| the graph | sequences, residual branches, sums, products, constants, repeats, taps, shared definitions; not yet residual algebras |

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
  window to them.
- **Compressed sparse attention.** A `csa` layer's cache layer keeps every token's
  latent as its keys and, as its values, the compressed latent the token ends, with
  the indexer's key of it; its conv state keeps the last tokens that the next
  compressed latent pools (a `hybrid` layer).

The cache is transformers' `DynamicCache`; a static one
(`cache_implementation="static"`) is not exported yet.

## Mixtures of experts

An exported `moe` routes as bux does: the scores and the top-k choice in float32,
equal scores picking the lower expert, and the aux-free bias, which the checkpoint
holds as `router_bias`, in the choice alone. Its routed experts are stacked as
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
reads are left out: the balance loss, how far the aux-free bias moves, and random
routing.
