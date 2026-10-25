# Language reference

Every name a `.bux` file can call, and every setting each one takes. A layer is a
function of the residual stream (`gqa`, `rmsnorm`, `moe`); a setting is a value other
configs or the run take (`rope`, `indexer`, `adamw`, `mesh`). Both are called the same
way, `name(key=value, ...)`, and checked against the same schema.

These pages are generated from `bux/config.py` and `bux/train/config.py`: an entry's
text is its config's docstring, and each setting's description is the comment above
or beside its field. Link to an entry from any page with ``{bux}`gqa` ``.

| Page | What it lists |
|------|---------------|
| [Top-level settings](top-level.md) | `d_model`, `model`, and every training setting of a run |
| [Layers and model settings](layers.md) | norms, positional encodings, MLPs and MoE, attention, linear attention and SSMs, elementwise layers, residual algebras |
| [Training settings](training.md) | data sources, packings, optimizers, schedules, loss, eval, checkpoints, precision, mesh, layout, methods, compiler |
| [Activations](activations.md) | the activation names, as layers and as values |
| [Registries](registries.md) | the names each registry knows: methods, optimizers, schedules, data sources, matmul formats, presets and passes |

```{toctree}
:hidden:

top-level
layers
training
activations
registries
```
