# Megatron: Qwen3-30B-A3B on a node, at Megatron-Bridge's speed

Megatron-Bridge's performance summary (26.06) trains Qwen3-30B-A3B on 8 B300s at
**40,769 tokens a second a GPU** (938 model TFLOP/s): MXFP8, EP=8, 8 rows of 4096 a
GPU, GBS 512, no recompute, the routing forced even. Its recipe also runs attention in
FP8 (cuDNN), dispatches by HybridEP on 32 SMs beside the other microbatch's compute,
and captures the whole iteration in a CUDA graph.

bux's `examples/benchmarks/megatron.bux` is the same model and batch (23.1 GFLOP a
token, as Megatron counts it). Before this wave, on 8 B300s:

| run | tok/s a GPU | of Megatron's | peak | remat=auto took |
|-----|-------------|---------------|------|-----------------|
| `sorted`, `fsdp=8` | 26.4k | 65% | 179 GiB | `dots` |
| the same, the pool at 95% of a GPU | 28.0k | 69% | 200 GiB | `dots_attention` |
| the same, the pool at 97% | 30.9k | 76% | 250 GiB | `none` |
| `nccl_ep`, `expert=8`, capacity 1.25 | 21.7k | 53% | 99.5 GiB | `save_attention` |
| `ragged_all_to_all`, `expert=8` | out of memory | | | |

## Where a step goes

`sorted`, profiled on GPU 0 (1309 ms a step of 32k tokens; Megatron takes ~800):

| what | ms | why |
|------|----|-----|
| attention, tokamax | 294 | the forward twice (remat); the backward at 0.69 PFLOP/s |
| experts' MXFP8 quantizer | 157 | 10 calls a layer, 4 of them the remat's; 2.3 TB/s; every GPU quantizes all 128 experts' weights |
| experts' GEMMs | 169 | 2.1 PFLOP/s: fine |
| transposes | ~96 | FSDP splits the expert stacks along `embed`, not `experts`; up and gate concatenated every step |
| the picks' rows | 67 | 262144 x 2048 rows gathered in bfloat16 before the quantizer, and back |
| the router's weights | ~45 | they multiply each pick's 2048 outputs, which the backward then keeps: 1 GiB a layer |
| FSDP's collectives | 373 | 303 hidden, 69 exposed, and 72 idle |

`nccl_ep` (1545 ms): its dispatch and combine run on the compute stream, 532 ms of
them and 42 hidden; `none` takes 266 GiB, so `save_attention` runs the dispatch and
the experts' forward twice. The synthetic batch's routing drifts uneven (a balance
loss of 1.6-1.8 a layer), so EP drops 10-17% of the picks at 1.25.

## Work packages

Each owns its files; the integrator merges into `megatron` and measures the whole.

- **X, the MoE layer** (`experts/moe.py`, `MoEConfig`, the weights in
  `parallel/experts.py` and `nccl_ep.py`): the router's weights on each pick's hidden
  units, before `down` (as Megatron does), so the combine is a plain sum and nothing
  d wide is kept; up and gate held as one stack; routing by random logits for
  benchmarks (Megatron's `moe_router_force_load_balancing`).
- **L, layouts and memory** (`parallel/layout.py`, `architecture/logits.py`,
  `runtime.py`, `train/fitting.py`): FSDP splits expert stacks by expert; the head
  cast once a step, not once a chunk; the GPU's pool; fewer compiles in `remat=auto`.
- **Q, the experts' quantizer** (`experts/grouped_*.py`): at these shapes, near the
  bandwidth it reaches elsewhere; the picks' rows gathered inside it; each weight
  shard quantized once, before FSDP gathers it.
- **A, attention** (`parallel/context.py`, `parallel/flash.py`, cuDNN in
  `jax_internals.py`): the faster kernel at this shape, and FP8 attention by cuDNN as
  a precision, opt-in.
- **E, EP beside compute** (`parallel/nccl_ep.py`'s exchange, `experts.in_chunks`):
  NCCL EP's kernels on a stream of their own, so a chunk's exchange runs beside
  another's experts.

**Done when** `megatron.bux` trains past 34k tokens a second a GPU (85%) on 8 B300s
in bfloat16 attention, and each change is measured against the tree before it.

## Rules

plans/performance.md's: readable first, measured on B300s against the tree before,
the DSL as the interface, tests on 8 fake CPU devices.
