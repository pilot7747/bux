"""Softmax attention: its layers, sparse attention's indexer and kernels, and RoPE.

- `~bux.architecture.attention.gqa`: grouped-query attention (``gqa``), and what MLA
  and CSA share with it.
- `~bux.architecture.attention.mla`: multi-head latent attention (``mla``).
- `~bux.architecture.attention.csa`: compressed sparse attention (``csa``).
- `~bux.architecture.attention.indexer`: sparse attention's lightning indexer, which
  picks the keys each query attends to.
- `~bux.architecture.attention.scoring`: the kernels of the indexer's scores and of its
  loss's target.
- `~bux.architecture.attention.sparse`: the attention over the keys the indexer keeps.
- `~bux.architecture.attention.rope`: rotary position embeddings.
- `~bux.architecture.attention.norm_rope`: QK-norm and RoPE in one kernel.
"""
