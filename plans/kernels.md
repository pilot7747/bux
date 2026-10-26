# Kernels: what cuDNN, TransformerEngine, DeepSeek and FlashAttention had that bux lacked

A survey of NVIDIA's, DeepSeek's, Moonshot's and the community's kernel repositories
(cudnn-frontend, TransformerEngine, DeepGEMM, FlashMLA, TileKernels, flash-attention,
fla, quack, tokamax) for kernels bux obviously lacked and that would likely speed it
up, then one agent a kernel. A kernel landed only if it was exact against bux's XLA
path to rounding, forward and backward, on a B300; faster on a real workload, before
and after on the same node; and fell back to XLA by itself where it can't run.

The rules were plans/performance.md's: readable first, measured on B300s against the
tree before, the DSL as the interface, tests on 8 fake CPU devices.

## As built

| what | by | on B300s, before → after |
|------|----|--------------------------|
| packed documents, windows, MLA and soft caps in softmax attention | FlashAttention-4's sm100 kernels (`parallel/fa4.py`, CuTe DSL by `cutlass.jax`), first in `auto` for rows of documents | 1.3B, rows of 4096: +5.1%; of 16384: +21%; MLA +4.7%; gpt-oss's attention +26% |
| the DSA indexer's top-k and its loss | cuDNN's DSA kernels (`attention/indexer_cudnn.py`): a radix top-k, bitwise XLA's; the picks' scores recomputed, their backward by its GEMM | the DSA 1B 1.46-2.09x, 4k to 64k |
| CSA's attention (DeepSeek-V4) | cuDNN's DSA sparse attention, forward and backward, and the target at the picks (`attention/sparse_cudnn.py`) | V4.1-Flash shrunk to a GPU 3.70k → 31.7k tok/s at 4k; 16k and 32k fit |
| GLA and Lightning | cuDNN's Frost GDN-2 at a step of 0 (`fused_delta`), cudnn-frontend's develop build | GLA hybrid +53% (a node +39%), Lightning +5% |
| RWKV-7 | its chunks take a pair's decay through the chunk's middle, its decay being bounded (`recurrence(floor=...)`) | +94%, a node +95% |
| MoE experts' ReLU², clamped GeGLU, SiTU-GLU in MXFP8 | cuDNN's CuTe DSL grouped GEMMs with the activation in the epilogue, as SwiGLU was (`grouped_glu`) | Nemotron 3 Nano +5.1% in 14 GiB less; expert layers 1.04-1.31x |
| a short conv's backward | a Pallas kernel for q, k and v (`layers/pallas_conv.py`), XLA from dy shifted ahead elsewhere | Mamba-2 hybrid +4.3%, GDN +2.7%, KDA +1.5-2.1% |
| NVFP4's quantizers | TransformerEngine's fused amax and cast, the Hadamard GEMM fused with it (`precision/nvfp4_te.py`), from 8192 rows a device | 1.3B in NVFP4 +2.2-3.0% |

Also: mHC's single-pass read kept the stream in float32 (a float32 one-hot), so in
DeepSeek-V4.1-Flash every projection multiplied in float32 and no 16-bit kernel ran;
fixed.

Dependencies: cudnn-frontend's develop build 1.32.0.dev76547535 (GDN-2, the
deterministic top-k), pinned by URL to its GitHub release; flash-attn-4 4.0.0b36; and
PyTorch's CPU build, which FlashAttention-4's and cuDNN's DSA modules import. With
torch imported, cudnn-frontend's API classes ask torch whether CUDA is there, which its
CPU build denies: build their kernels directly, as `grouped_cudnn` does.

The one-GPU suite on the merged tree, against main on the same node, each arm the mean
of two runs with the node's halves swapped:

| arm | main | merged |
|-----|------|--------|
| dense | 153.8k | 153.8k |
| gdn | 132.6k | 136.6k (+3.0%) |
| kda | 98.6k | 100.2k (+1.6%) |
| kdab | 132.4k | 135.0k (+2.0%) |
| moe | 113.1k | 112.9k |
| mamba2 | 96.6k | 100.7k (+4.3%) |
| mxfp8 | 185.5k | 185.8k |
| nvfp4 | 203.4k | 208.9k (+2.7%) |

Peak memory is the same in every arm. Qwen3-30B-A3B at Megatron-Bridge's batch on a
node (`megatron.bux`, `sorted`) trains at 41.2k tokens a second a GPU before and after,
in 214.5 GiB.

## Measured, not landed

- **Mamba-2 by cuDNN's Frost kernels**: 1.22x on one GPU but 1.00x on 8 and on
  Nemotron 3 Nano (each call's Python callback runs one GPU at a time), and the time
  steps' gradients 2-70x XLA's rounding, worse as decays steepen.
- **KDA with unbounded gates by GDN-2**: trained gates soon steepen past its fast path,
  and its exact path relaunches the whole call: 54.6k tok/s against bux's own kernels'
  99.3k.
- **A dense SwiGLU fused with MXFP8's quantizer**: 0.63 ms against XLA's fusion and two
  quantizer passes' 0.62.
- **TransformerEngine's NVFP4 GEMM**: the same kernels as cuDNN's, at the same speed.

## Left open

- cudnn-frontend's develop tags are deleted after 14 newer ones: move the pin to 1.32.0
  when it is released, or mirror the wheel.
- An RWKV-7 kernel (fla's takes 6.7 ms a layer against the chunks' 15.5) and
  Lightning's (2.1 against GDN-2's 3.5).
- A Python-free launcher for cuDNN's graph kernels (GDN, GDN-2, Mamba-2), whose
  callbacks run one GPU at a time on a node.
- FA4's pack-GQA backward (newer than 4.0.0b36) for rows of one document, where cuDNN
  still wins; its scheduler metadata in place of a fixed room of documents; ring and
  gathered context parallelism by FA4.
- The fused indexer scores and top-k; 128-head indexers' loss; gpt-oss's biased
  experts in the fused path.
- The surveyed kernels not tried: fused mHC (TileKernels, TransformerEngine's main),
  absorbed MLA for DSA.
