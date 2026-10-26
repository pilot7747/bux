"""bux's layers in PyTorch, for transformers: the files every exported model's repo
copies beside its generated ``modeling_bux.py`` (`bux.export.huggingface`).

Each module holds layers of one family, classes with ``bux_types``, and imports
nothing of bux (`bux.export.hf.common` says how a layer is built and called). This
``__init__`` is the only file not copied.
"""
