# Model zoo

The model zoo is `examples/architectures/`: 40 `.bux` files, each a model you can
`use` by name, shrink, sweep or train. This page says what the files are, how to use
one, and how to add your own, then lists every model with its description, its
layer counts and its source.

```bux
use qwen3_30b_a3b                 # the whole of Qwen3-30B-A3B's architecture

d_model = 1024                    # then change what you like
model   = embed block^12 norm head
```

## What the files are

Each file transcribes a public model's configuration:

- The files at the top of the folder are classics and primitives: GPT-2, GPT-J,
  Llama 3, Gemma 3, DeepSeek-V3, Jamba, Qwen3-Next, and wiring studies such as
  ResFormer, nGPT, mHC and modded-nanogpt. They transcribe public configs from
  memory. Ones marked "-style" or "approximate" simplify details, so verify the
  numbers before relying on them.
- The folders, one per model family (`cohere/`, `deepseek/`, `glm/`, `inkling/`,
  `kimi/`, `laguna/`, `mellum/`, `mimo/`, `muse/`, `nemotron/`, `qwen/`,
  `reflection/`), transcribe recent releases from their published `config.json` and
  modeling code.

A file's header comment says what the model is and what makes it different, and
names what it leaves out (vision and audio encoders, speculative drafters) or
approximates. Read it before you compare against the original.

:::{note}
A file describes an architecture, not a checkpoint: a run of it starts from fresh
parameters.
:::

## Using a model

### By name

`use name` brings in every definition of `name.bux`. A module is found by its name,
never by a path: in the folder of the file that uses it, then under the folders that
`[tool.bux] path` lists in the nearest `pyproject.toml` (bux's lists `examples`),
then under those in `BUX_PATH` ([Modules](../language/modules.md)). A name that is not
an identifier is a string:

```bux
use "qwen3.8_27b"
```

Definitions after the `use` replace the module's everywhere, including inside its
own definitions. So changing `attn` changes the attention of the module's `block`,
and an amendment changes one setting:

```bux
use llama3_8b

attn.num_kv_heads = 1             # MQA in every block
```

### Shrunk for a laptop

To train a zoo model on a laptop, shrink its width and depth and keep its wiring.
This file keeps Qwen3-30B-A3B's block, a GQA with QK-norm and an MoE, at 1.05M
parameters:

```bux
use qwen3_30b_a3b

d_model    = 128
vocab_size = 512
attn.num_heads    = 4
attn.num_kv_heads = 2
attn.head_dim     = 32
experts.num_experts = 8
experts.top_k       = 2
experts.expert.intermediate_size = 64
model = embed block^4 norm head(tied=true)

data      = synthetic
optimizer = adamw(lr=3e-3)
seq_len   = 64
batch     = 8
steps     = 50
```

```console
$ bux check tiny_qwen.bux
d_model=128 vocab_size=512 layers=19
rmsnorm×9  gqa×4  moe×4  embed×1  head×1
50 steps × 8 × 64 tokens = 25.6k tokens
optimizer=adamw(lr=0.003) schedule=constant
data=synthetic()
$ JAX_PLATFORMS=cpu bux train tiny_qwen.bux
...
tiny_qwen step 10/50  loss 6.5421  lr 3.00e-03  grad 1.387  ...
tiny_qwen step 50/50  loss 6.3270  lr 3.00e-03  grad 1.711  ...
```

Outside bux's own folder, point `BUX_PATH` at bux's `examples` folder for `use` to
find the zoo. On random tokens (`synthetic`) the loss stays near $\ln 512 = 6.24$:
use real data to see a model learn ([Data](../data/index.md)).

:::{tip}
`bux check` runs no JAX and counts every layer, so it is the quickest way to see a
transcription's shape. `bux render file.bux` draws its blocks.
:::

## Adding a model

1. **Write the file** under `examples/architectures/`, in the folder of its family
   (a new folder makes a new section on this page). Its name must be unique: modules
   are found by name.
2. **Start it with a header comment**: the model's name, its maker and year, what
   makes it different, and what the file leaves out or approximates. Mark a
   simplification "-style" or say "Simplified:". This page shows the header as the
   model's description:

   ```bux
   # Zamba2-1.2B-style (Zyphra, 2024): a Mamba-2 backbone that calls one shared transformer
   # block before every sixth Mamba-2 layer. ...
   #
   # Simplified: Zamba2 feeds the block the stream concatenated with the embedding, and
   # gives each call LoRA adapters on q, k, v and the MLP; here it reads the stream alone.
   ```

3. **Check it**: `bux check` counts its layers, which you can compare with the
   `config.json`'s, and `bux render` draws it.
4. **Test it** (`tests/test_examples.py`). Every file builds and runs abstractly at
   its full size, with no parameter allocated, by
   `test_examples_build_and_run_at_full_size`, with nothing to add. To also run it for
   real, add an entry to `SHRUNK`: the statements that shrink its layers to a few
   dims, after `SMALL` sets `d_model = 32`, `vocab_size = 64` and a small `dense`
   MLP. `test_examples_shrunk_run_and_are_causal` then runs it on CPU and checks that
   its outputs are finite and causal:

   ```python
   SHRUNK = {
       "jamba.bux": [
           "attn = gqa(num_heads=4, num_kv_heads=2)",
           "ssm = mamba(state_size=4)",
           "experts = moe(num_experts=4, top_k=2, expert=dense)",
       ],
   }
   ```

   Run one with `uv run pytest tests/test_examples.py -k jamba`.

## The models

Every file, by family. The counts are what `bux check` prints of the file.

```{include} ../_generated/models.md
```
