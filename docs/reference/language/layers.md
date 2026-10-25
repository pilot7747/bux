# Layers and model settings

Everything a `.bux` file can call to describe a model, in the order of
`bux/config.py`. A *layer* is a function of the stream and can appear in a block
(`[norm attn]`); a *setting* is a value that a layer or the architecture takes
(`rope=rope(base=1e6)`, `residual = mhc(expansion=4)`).

Write a layer's settings as keyword arguments, `gqa(num_heads=32, num_kv_heads=8)`,
and reach into nested settings with dots, `attn(rope.base=1e6)`. See
[Settings and values](../../language/values.md) for the values each type takes.

```{include} ../../_generated/layers.md
```
