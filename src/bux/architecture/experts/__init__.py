"""Mixture of experts, and the grouped matmuls its experts run.

- `~bux.architecture.experts.moe`: the layer ``moe``, its router, and a token's path
  through its experts.
- `~bux.architecture.experts.grouped`: grouped matmuls by `jax.lax.ragged_dot`, and by
  kernels on a GPU.
- `~bux.architecture.experts.grouped_blockscaled`: the same, block-scaled (MXFP8).
- `~bux.architecture.experts.grouped_mxfp8`: their MXFP8 GEMMs by Mosaic GPU's kernels
  on a Blackwell GPU.
- `~bux.architecture.experts.grouped_triton`: Triton's MXFP8 quantizer, into those
  kernels' layout.
- `~bux.architecture.experts.grouped_glu`: a SwiGLU FFN's whole, fused, by cuDNN's
  kernels on a Blackwell GPU.
- `~bux.architecture.experts.grouped_cudnn`: those cuDNN kernels.
"""
