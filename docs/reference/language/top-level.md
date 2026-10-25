# Top-level settings

A `.bux` file's top-level definitions with reserved names configure the architecture
and, when a file sets any training setting, make it a run.

## The architecture

| Setting | Type | Default | Description |
|---|---|---|---|
| `model` | a block | required | the model: `embed ... head` |
| `d_model` | `int` | required | the width of the residual stream |
| `vocab_size` | `int` | required with `embed` or `head` | the vocabulary `embed` and `head` take |
| `residual` | `hc \| mhc \| gr \| attnres \| vpgate \| none` | `none` | the residual algebra: what `[f]` means ([Residual algebras](../../language/residual-algebras.md)) |

## The run

A file is a run once it sets any of these; then it needs every required one. The
types link to the [training settings](training.md).

```{include} ../../_generated/settings.md
```
