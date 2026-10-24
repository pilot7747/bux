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
| experts' MXFP8 quantizer | 157 | 10 calls a layer, 4 of them the remat's (each at 5.1-5.8 TB/s, as Q measured: not slow); every GPU quantizes all 128 experts' weights |
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

## As built

- **X:** `experts()` takes each row's weight and multiplies the hidden units by it
  before `down` (after `expert_norm`, where one is set), recomputing them in the
  backward from up and gate's output; every EP method sends its rows' weights. Up and
  gate are one stack, (E, d, f, 2): an (E, d, 2, f) one was gathered whole across
  `tensor` when reshaped for the GEMM. `moe(random_logits=true)` routes a training
  forward by N(0, 1) logits, the router's gradient straight through.
- **Q:** the experts' quantizer reads x's rows from the tokens by index
  (`grouped_blockscaled.Picked`), and the down GEMM sums its rows into their tokens
  (`Summed`), its backward reading dy the same way: no 262144 x 2048 copy either way.
  The quantizer itself was already at 5.1-5.8 TB/s at every shape.
- **L:** FSDP splits a parameter's first whole dimension, so shards are whole rows and
  collectives need no transposes. XLA's pool takes 95% of a GPU by default (beside it
  B300s held at most 9.8 GiB, with NCCL EP over two nodes). `remat=auto` compiles
  `none` alone, then the other candidates side by side. The head's "cast once a
  chunk" was each chunk's softmax gradient, not the head: nothing to change. Two fixes
  the full suite found: FSDP keeps off a dimension its own axes cannot divide (CSA's
  2 groups under `tensor=4`, which XLA rematerialized in full), and Muon's `owner`
  takes the optimizer state's real sharding (`states`, a new argument of every
  orthogonalize method), where it guessed it from the old rule.
- **E:** NCCL EP takes chunks and microbatches (every dispatch into the rows it was
  started with: fewer failed with "invalid argument"); `methods(moe_stream=true)` runs
  its kernels on a stream of their own, opt-in.
- **A:** rows of one document attend by cuDNN's causal kernels inside bux's VJP
  (`flash.CUDNN`: cuDNN's forward with its logsumexp, its backward from it), chosen by
  a `lax.cond`, as every run's rows have ids; tokamax's backward where XLA's
  deterministic ops leave cuDNN's without a graph. Softmax attention stays in compute:
  cuDNN's FP8 forward with a bfloat16 backward diverged in all 11 runs of the 1.3B,
  jax 0.12 passes the FP8 backward's scales in the wrong order, and that backward
  gives E4M3 gradients only, which need delayed scaling; at rows of 4096 it was no
  faster (33.9k against 34.2k).

Measured on 8 B300s, `megatron.bux` at its defaults (random logits, the pool at 95%,
`remat=auto`):

| run | before | E, X, Q, L | and A | of Megatron's |
|-----|--------|------------|-------|---------------|
| `sorted`, `fsdp=8` | 26.4k, `dots`, 179 GiB | 35.8k, `none`, 215 GiB | **36.9k** | 90% |
| `nccl_ep`, `expert=8` | 21.7k, `save_attention`, 99.5 GiB | 29.2k, `none`, 211 GiB | 30.8k | 76% |
| the same, `moe_stream=true` | | 29.6k, 211 GiB | | |

`remat=auto` now compiles once for both (130-150 s, from 644 and 950). 3000 steps of
FineWeb-Edu on the benchmarks' MoE end within 0.11% of the tree before at three seeds,
which differ by 1.7%; the 1.3B by cuDNN's attention at tokamax's loss. `sorted`'s step
on GPU 0 is 884 ms, from 1309 (Megatron's ~800): the experts' GEMMs 179, attention
138, their quantizer 100, FSDP's collectives 281 of which 62 exposed, and 52 idle.

The suites before against after, each tree at its own defaults, side by side: on a
node FSDP +5.9%, at 8 microbatches +4.8%, the MoE sorted +9.1% and by EP +6.7% (12
GiB less each), TP +2.3%, the pipeline +0.6% (1.2 GiB more); on one GPU dense +5.8%,
MoE +6.2% (16 GiB less), MXFP8 +3.5%, NVFP4 +8.3%, the hybrids within 2.1%.

## Left open

- The exposed collectives and idle time (114 ms), and the experts' quantizer (100).
- NCCL EP's exchange beside the experts: XLA's schedule puts each call's done right
  after its start, and a scheduling group a chunk fails in the backward, whose chunks
  share the forward's ids.
- Each weight shard quantized once, before FSDP gathers it (23 ms a step).
- The MXFP8 experts' Mosaic kernels are lowered again at every call: about a minute of
  each compile of this model.

## Rules

plans/performance.md's: readable first, measured on B300s against the tree before,
the DSL as the interface, tests on 8 fake CPU devices.
