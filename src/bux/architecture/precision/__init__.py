"""The formats the projections multiply in.

- `~bux.architecture.precision.matmuls`: the registry of formats, and the one in
  force for each projection: the dtype the run computes in, or a block-scaled one.
- `~bux.architecture.precision.blockscaled`: block-scaled GEMMs, and MXFP8's
  quantizers.
- `~bux.architecture.precision.nvfp4`: NVFP4's quantizers.
"""
