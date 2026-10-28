# Starting from released weights

A run can start from a released checkpoint instead of fresh parameters: write the
model as the zoo describes it, name the checkpoint in `weights`, and train. Use it to
fine-tune a released model, or to change part of one and train the rest from where
its makers left it.

```bux
use "qwen3.5_0.8b"

weights = hf(name="Qwen/Qwen3.5-0.8B")
```

`name` is a Hugging Face Hub repo, which downloads to the Hugging Face cache, or a
folder with the checkpoint's `config.json` and safetensors files. The weights load
right after the model is built. A run that resumes from its own checkpoint
([Checkpoints](running.md#checkpoints)) loads that one instead, and does not read
the released weights again.

## How a checkpoint loads

A checkpoint loads in the order the model's layers run. Its family's importer reads it
as a list of layers: a type, as `gdn`, and the layer's tensors. The model's layers that
hold parameters take them one for one. The load stops at the first difference and
names it:

- a layer of another type: `model.layers.0.attn is a 'gqa' layer, but the checkpoint's
  is a 'gdn'`;
- a tensor of another shape; or
- another number of layers.

So the `.bux` file can name its definitions as it likes, and only the order, the
types and the shapes have to be the released model's. Settings that no shape shows,
such as a norm's epsilon or RoPE's base, come from the zoo's file, which transcribes
them from the released config.

### Optional parts

Some parts of a checkpoint are optional groups: a model may leave them out, and the
rest loads. Qwen3.5 has two: its vision encoder (`vision`) and its multi-token
prediction module (`mtp`). The zoo's file has both. This model of text, which leaves
out both, loads too:

```bux
use "qwen3.5_0.8b"

model = embed "DDDA"^6 norm out

weights = hf(name="Qwen/Qwen3.5-0.8B")
```

## Changing the architecture

`skip` lists the model's layers that keep their fresh parameters, by type or by the
name of their definition. The other layers load as before. This model swaps
Qwen3.5's attention for gated linear attention and loads everything else:

```bux
use "qwen3.5_0.8b"

attn = gla(num_heads=8, head_dim=128)

weights = hf(name="Qwen/Qwen3.5-0.8B", skip=[gla])
```

A skipped layer takes the place of the checkpoint's layer where it is, whatever that
layer's type. If the layers line up only without the skipped ones, the skipped ones
are new instead, as a `[scale(per_channel=true)]` added after every MLP. A model that
both replaces and adds skipped layers does not load.

A definition's name picks out one layer where its type would pick out several. This
encoder ends with a new layer, `pool`, that joins each 2 x 2 of the released
merger's tokens into one; the merger's own `linear` layers still load:

```bux
pool   = linear(features=d_model, bias=true)
vision = (patches(size=16, features=768, grid=48) ([vnorm vattn] [vnorm vffn])^12
          vnorm merge_patches(size=2) linear(features=3072, bias=true) gelu
          linear(features=d_model, bias=true) merge_patches(size=2) pool)

weights = hf(name="Qwen/Qwen3.5-0.8B", skip=[pool])
```

To load a model in Python, read the checkpoint and load it into a built model:

```python
from flax import nnx

from bux import pretrained
from bux.architecture import Model
from bux.lang import compile_file

arch = compile_file("examples/architectures/qwen/qwen3.5_0.8b.bux")
model = Model(arch, rngs=nnx.Rngs(0))
pretrained.load(model, arch, pretrained.read("Qwen/Qwen3.5-0.8B"))
```

## Training part of a model

`frozen` names definitions whose parameters do not train. Their updates are zero,
the optimizer keeps no state for them, and their gradients count toward no clipping
norm. This run trains Qwen3.5-0.8B's decoder on OCR and keeps its vision encoder as
released:

```bux
use "qwen3.5_0.8b"

weights = hf(name="Qwen/Qwen3.5-0.8B")
frozen  = [vision]
```

A name is any definition of the model: a block, as `vision` or `A`, or a layer, as
`attn`, which freezes every use of it. A sweep compares what trains, as
`examples/experiments/ocr.bux` does, which fine-tunes Qwen3.5-0.8B on pages of text:

```bux
frozen = all: [] | encoder: [vision]
```

The frozen parameters' gradients are still computed: the backward pass costs as much
as without `frozen`.

## Saving back

A run that started from released weights exports in their format, for transformers'
and vLLM's own code of the family:
`bux export RUN_DIR OUT --released` ([As the released checkpoint](../export/index.md#as-the-released-checkpoint)).
In Python, `pretrained.save(model, arch, "Qwen/Qwen3.5-0.8B", out)` does the same
for any model of the released architecture. The family's table runs the other way:
what bux splits, it fuses back. What bux cannot tell apart keeps its released split:
Qwen3.5's patch embedding sums two frames of a video, and a change to the sum is
shared equally by the frames. Qwen3.5-0.8B, loaded and saved, gives back each of its
488 tensors bit for bit.

## What loads

| Checkpoint `model_type` | Models | Zoo files | Checked against |
|-------------------------|--------|-----------|-----------------|
| `qwen3_5` | Qwen3.5's dense models: the vision encoder, the text decoder and its MTP module | `qwen3.5_0.8b`, `qwen3.5_9b` | transformers' logits, on a shrunk model in the tests and on Qwen3.5-0.8B, with an image and without |

On Qwen3.5-0.8B in float32 on an Apple M5 Pro CPU, bux's logits are within 6.3e-5 of
transformers 5.20.2's over an 18-token sentence, whose largest logit is 22.0, and
every next-token prediction is the same (October 2026).

Weights load into a model whose repeats are unrolled, `methods(repeat=unroll)`, the
default.

## Adding a family

A family is a table registered for a checkpoint's `model_type`: its layers, in the
order the model runs them. Each layer is a bux type and its parts. A part names some
of the layer's tensors, as the export names a bux layer's weights
([Export](../export/index.md)): `q.weight`, `norm.weight`, a linear layer's weight as
`(out, in)`. It also names the checkpoint's tensors they are made of, with a function
each way. `same` is a tensor kept as it is; a part with functions splits what the
checkpoint fuses, as Qwen3.5's q, k and v, and fuses it back:

```python
import numpy as np

from bux import pretrained
from bux.pretrained import Part, Spec, same


@pretrained.register("my_model")
def specs(config: dict) -> list[Spec]:
    qkv = Part(
        ("q.weight", "k.weight", "v.weight"),
        ("layers.0.attn.qkv.weight",),
        lambda w: tuple(np.split(w, 3)),
        lambda ys, _: (np.concatenate(ys),),
    )
    return [Spec("embed", (same("tokens.weight", "embed.weight"),)), ...]
```

`bux.pretrained.qwen3_5` is a complete one. Test a new importer as
`tests/test_pretrained.py` tests Qwen3.5's: build a shrunk model in transformers with
random weights, load it into bux, and compare the logits.
