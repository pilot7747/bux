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

## Round 2: Megatron's own workload

Megatron-Bridge's config for that row (r0.5.0) trains a global batch of 512 rows of
4096: 8 microbatches of 8 rows a GPU, EP=8, no recompute. It hides 7 of every 8 token
exchanges by running a microbatch's forward beside the previous one's backward
(combined 1F1B), delays the weight gradients' GEMMs under the exchanges, captures the
whole iteration in one CUDA graph, runs the experts by cuDNN-frontend's CuTe DSL MXFP8
grouped GEMMs with the SwiGLU in their epilogue, and attention in MXFP8. `megatron.bux`
trains that workload now; `sorted` gathers a layer at a time, as a whole copy of the
model does not fit beside a microbatch.

At it, main trained `sorted` at 37.5k tokens a second a GPU (229.6 GiB), and `nccl_ep`
not at all: its buffers were sized for a step. At one microbatch, `sorted`'s step took
898 ms (768 computing, 61 of collectives alone, 69 idle), `nccl_ep`'s 1093 (688, 326,
78).

- **O, the reductions.** XLA's scheduler put every weight gradient's GEMM and
  reduce-scatter after the backward, where they ran beside nothing, ~95 ms a
  microbatch. `bux.train.reorder`, a pass over XLA's schedule, starts each beside the
  next layer's backward (`compiler(reductions=early)`, the default); `bux.passes` is the
  registry of such passes, and keys JAX's cache by them. The log reads a step's
  metrics after the next is launched. No XLA flag or NCCL setting helped: PGLE, two
  collectives in flight, fewer NCCL channels; symmetric buffers fail to compile beside
  cuDNN's MXFP8 GEMMs.
- **K, the experts.** Their SwiGLU MLP by cuDNN-frontend's CuTe DSL MXFP8 kernels
  (`grouped_glu`, `grouped_cudnn`): the SwiGLU, the row's weight and h's quantization in
  up's epilogue, dSwiGLU in down's dgrad. Under FSDP each GPU quantizes its own shard
  of the experts' kernels, and gathers them in MXFP8.
- **A, the rest of a layer.** QK-norm and RoPE in one pass each way, the dense MXFP8
  quantizer at 4.3-4.7 TB/s, the loss's target logits gathered, the router's GEMM from
  bfloat16. cuDNN's MXFP8 attention is 1.35x bfloat16's forward, but its backward is no
  faster at rows of 4096: not used.
- **E, NCCL EP.** Buffers sized by a microbatch; each backward call prepares its own
  route, so any order of the calls computes the same; `moe_stream`'s calls start early
  and finish late (`parts_off_the_stream` strips the stream mark the SPMD partitioner
  copies onto their results). In turn, nothing independent sits beside an exchange,
  and the arm gains nothing from it.
- **S, microbatches side by side.** `methods(interleave=true)` runs each microbatch's
  forward beside the backward of the one before, a piece of the model at a time:
  `nccl_ep` on 12 layers +4%. Off by default: its residuals cross the scan's
  iterations (25-30 GiB more), so 48 layers fit only with recompute.

- **D, the host.** The log's learning rate queued behind the step just launched, and
  nnx.jit walked the model's and the optimizer's graph at every call: the GPU waited
  for the host between steps (58 ms a step of the 1.3B's 257, logging every step). A
  step is a jax.jit over their variables' values, split once, and the log reads its
  learning rate behind its own step: the 1.3B's GPU idles 2.2 ms a step.

On 8 B300s, `megatron.bux` at its defaults, one node, main against the merged tree:

| run | main | round 2 | of Megatron's 40,769 |
|-----|------|---------|----------------------|
| `sorted`, `fsdp=8` | 37,436, 229.6 GiB | **42,391**, 214.5 GiB | 104% |
| `nccl_ep`, `expert=8` | fails | 34,873, 236.1 GiB | 86% |

O's pass gave 5%, K's MLP 4.5%, A's kernels 2.6%, D's 0.6%. 3000 steps of FineWeb-Edu
on the baseline's MoE with experts of 768 and QK-norm end within 0.12% of main at each
of three seeds, which differ by 1.6% (mean -0.01%). The suites, main against the merged
tree before D: on a node FSDP +0.9%, at 8 microbatches +6.3%, the MoE sorted +0.1% and
by EP +1.2%, TP +1.9%, the pipeline -0.0%; on one GPU dense +1.5%, MoE +1.5%, NVFP4
+2.1%, the others within 1% (the GDN hybrid, -4.5% side by side on other GPUs, trains
at 128.3k on one GPU in turn in both trees, pinned to the same loss). D on its own: on
a node FSDP +1.7%, at 8 microbatches +4.2%; on one GPU +1.2% to +7.2%.

## Left open

- `nccl_ep`'s exchange: hidden only beside another microbatch (`interleave`), which
  needs a carry XLA does not copy to fit 48 layers without recompute.
- An eval or a checkpoint still makes the host wait for the GPU before the next step,
  and XLA's launch of the 30B's step waits on CUDA's full queue of kernels.
- ~40 ms of reductions a microbatch still alone at the end of its backward (the last
  layers', the embedding's, the head's in float32).
- The MXFP8 experts' Mosaic kernels are lowered again at every call, and CuTe DSL
  compiles its kernels in every process: 20-110 s of a compile.

## Rules

plans/performance.md's: readable first, measured on B300s against the tree before,
the DSL as the interface, tests on 8 fake CPU devices.
