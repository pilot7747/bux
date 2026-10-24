# Polish: clean up what landed, then MoE

The wave after `sota` (plans/performance.md) had three aims, in this order:

1. **Clean up.** Ten agents wrote about 10k lines in parallel: passes were counted in
   three places, helpers were duplicated, and four modules reached into JAX's
   internals. Several defaults were behind what had been measured.
2. **MoE at scale**, bux's biggest gap: MXFP8 experts were slower than bf16, and EP
   ran at 0.67x of `sorted` on a node and couldn't cross nodes well.
3. **Hybrids and long context**: KDA without a gate bound, Mamba-2's memory, sparse
   attention at 32k, context parallelism at 128k, and 4 to 8 nodes.

The rules were plans/performance.md's: readable first, measured on B300s against the
tree before, the DSL as the interface, tests on 8 fake CPU devices.

## As built

### Cleanup

- One way to count passes: `architecture/repeats/tracing.py` says which branch and
  which pass a forward is tracing, and cuts a repeat's passes into runs that trace
  alike. Precision's `first`/`last`, remat's `first` and a pipeline's stages ask it.
- `bux/jax_internals.py` holds every private or unstable JAX, tokamax and jax-triton
  API bux uses, each with what it is for; `tests/test_jax_internals.py` pins the
  versions they were checked on and fails loudly on any other.
- Readability passes over the schedules, the split backward, TP, the experts, MXFP8
  and NVFP4, and a last one by a reader who wrote none of it; duplicated helpers
  folded into one home each (`parallel/layout.py`, `parallel/pipeline.py`,
  `bux/kernels.py`, `precision/blockscaled.py`, `tests/conftest.py`).
- `scripts/benchmark.py` with suites in `examples/benchmarks/` (one GPU, a node, two
  and four), and its table of two trees: every later change is measured the same way.
- Resume checked on GPUs, one node to four, onto other meshes. Fixed: an uneven
  pipeline resumed onto the wrong layers; a folder the nodes don't share hung the
  first save; a half-written save hung the next launch; metrics repeated steps.

### Defaults

| setting | now | was |
|---------|-----|-----|
| `methods.pipeline` | `1f1b`, stages balanced by FLOPs | `gpipe`, even |
| `methods.remat` | `auto`: the first of none, dots_attention, dots, save_attention, full that fits | `none` |
| routed experts under `matmuls=mxfp8` | MXFP8, by `grouped_mxfp8` | bf16 |

`repeat=scan` with `compiler(preset=maxtext)` stays opt-in: it wins only at about one
row a GPU under FSDP.

### New settings

- `methods(moe=nccl_ep)`: TransformerEngine's NCCL EP, which the CUDA images now have
  (2.20.2). `methods(moe_chunks=n)` and `all_to_all` by sorted buffers.
- `methods(states=recompute)` for Mamba-2 and Mamba.

### Measured on B300s

Against `main`, side by side by `scripts/benchmark.py`: no run of the one-GPU, one-node
or two-node suites is slower past noise (3%). On packed documents, with autotuning off,
the losses are main's bit for bit at every log in bfloat16, float32 and MXFP8. With
nothing but `matmuls=mxfp8` set, the MoE trains 14% faster under FSDP and 19% by EP; with
nothing set, a pipeline 8% faster in half the memory.

The MXFP8 experts' rows ran with GEMM autotuning off and deterministic ops, so they
read lower than the others; compare them within a row.

| run | before | after |
|-----|--------|-------|
| MoE, 1 GPU, MXFP8 experts | 74.2k tok/s (slower than bf16's 86.2k) | 95.0k (bf16 + 10%) |
| MoE, fsdp=8 sorted, MXFP8 experts | 580k | 743k (bf16 + 11%) |
| MoE, EP=8 ragged, MXFP8 experts | 417k | 539k (bf16 + 15%) |
| MoE, EP=8, best bf16 exchange | 549k (ragged) | 615k (`nccl_ep`), 586k (`all_to_all`, 2 chunks) |
| MoE, EP=16 over two nodes | 324k (ragged) | 773k (`nccl_ep`) |
| PP stage=4, 1.3B | gpipe 491k | 1f1b balanced [5,5,5,1] 598k |
| KDA hybrid, unbounded gates, 1 GPU | 60.5k, 122 GiB | 96.5k, 88 GiB |
| Mamba-2 hybrid, 1 GPU | 16 rows fit | 32 rows fit (`states=recompute`, -4 to -9% tok/s) |
| DSA 1B at 32k, 1 GPU | 7.1k (5.8k packed) | 9.9k (11.7k packed) |
| 1.3B at 128k, context=8, packed documents | ring 144k | ring 346k |
| HSDP, 8 nodes (64 GPUs) | - | 96% a GPU of one node's |

## Left open

- **EP still loses to `sorted`** on one node (0.74x at best): the exchange's volume
  bounds it, and NCCL EP's kernels move 330-370 GB/s there. Next: DualPipe-style
  overlap of the exchange with attention, NCCL EP's SM count as a setting.
- **MXFP8 experts:** quantize in the GEMM's epilogue, with the GLU (about 0.45 ms a layer).
- **KDA unbounded** is 96k against the bounded kernel's 131k: its pairs kernels spend
  their time in 16x16 dots; Mosaic GPU's tcgen05 would be next.
- **`attention=auto`** picks ulysses, where ring is 1.5x faster on long packed rows.
- **FSDP over all GPUs** falls to 87% a GPU on 8 nodes: HSDP is the way there.
- A `synthetic(documents=...)` source, so that the benchmark measures packed rows.
