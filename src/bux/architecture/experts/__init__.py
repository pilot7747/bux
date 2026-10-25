"""Mixture of experts (``moe``), and the grouped matmuls its experts run: by
`jax.lax.ragged_dot` and kernels (``grouped``), and block-scaled
(``grouped_blockscaled``), by Mosaic GPU's kernels on a Blackwell GPU
(``grouped_mxfp8``) after Triton's quantizer (``grouped_triton``); and a SwiGLU FFN's
whole, fused, by cuDNN's kernels there (``grouped_glu`` over ``grouped_cudnn``)."""
