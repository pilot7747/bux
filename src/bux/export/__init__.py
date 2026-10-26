"""Export bux models to other frameworks, with their weights, and check the exports.

An architecture compiles to its own model code, not to an existing model class:

- `bux.export.plan` places every layer in a tree of named modules, which every
  exporter shares, so that one checkpoint loads into each;
- `bux.export.weights` names a bux model's parameters by that tree;
- `bux.export.huggingface` writes a transformers model directory: the generated
  ``modeling_bux.py`` (`bux.export.codegen`), bux's layers in PyTorch
  (`bux.export.hf`), the config and the weights;
- `bux.export.check` exports a model, loads it back, and compares its logits and
  greedy decoding with bux's.

A layer type is exported once, as a class in `bux.export.hf`, and every architecture
that uses it exports; one that is not raises ``NotImplementedError`` naming it.
"""
