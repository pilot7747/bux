"""Mixture of experts (``moe``), and the grouped matmuls its experts run: by
`jax.lax.ragged_dot` and kernels (``grouped``), and block-scaled
(``grouped_blockscaled``), by Triton's kernels on a GPU (``grouped_triton``)."""
