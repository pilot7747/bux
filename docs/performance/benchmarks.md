# Benchmarks

bux measures every change to its speed the same way: a suite of runs in
`examples/benchmarks/`, trained by `scripts/benchmark.py`, against the tree before the
change on the same node. This page describes the suites, shows how to run and compare
them, and collects the results: one GPU, one node, two, four and eight nodes, and
Qwen3-30B-A3B against Megatron-Bridge at its own workload.

Use it to check that a change of yours is not slower, or to see how fast bux trains a
model like yours before you start.

## The suites

Every suite but `megatron.bux` uses `baseline.bux`: a Llama-style 1.3B (`d_model`
2048, 16 layers, a vocabulary of 128256, GQA of 16 heads and 8 kv heads, a gated MLP of
5632), its hybrids with Gated DeltaNet, KDA and Mamba-2, and its MoE (64 experts of
704, top 8), on random tokens in rows of 4096, 8 rows a GPU, in bfloat16, for 30 steps
logged every 5.

| suite | GPUs | its runs |
|-------|------|----------|
| `one_gpu.bux` | a GPU a run | `dense`; the hybrids `gdn` (`"DDDA"^4`), `kda` (`"KKKA"^4`), `kdab` (KDA with `gate_lower_bound=-5`), `mamba2` (`"MMMA"^4`); `moe` (`E^16`); the dense model in `mxfp8` and `nvfp4` |
| `one_node.bux` | 8 | `fsdp` (`fsdp=8`); `fsdp_m8`, the same in 8 microbatches; `moe`, the experts sorted under `fsdp=8`; `moe_ep`, by a ragged all-to-all over `expert=8` at a capacity of 1.25; `tp`, 4 of Llama 3 70B's layers (`d_model` 8192) by `tensor=8` with sequence parallelism, scanned, `remat=full`, `tensor=chunked`, a row a GPU; `pp`, two stages of 10 and 6 layers on `stage=2, fsdp=4`, 1f1b in 8 microbatches |
| `two_nodes.bux` | 16 | `fsdp` (`fsdp=16`); `hsdp` (`data=2, fsdp=8`); `gdn`, the GDN hybrid under HSDP; `moe`, sorted under HSDP; `moe_ep`, by EP within a node (`data=2, expert=8`) |
| `four_nodes.bux` | 32 | `fsdp` (`fsdp=32`); `hsdp` (`data=4, fsdp=8`); `moe` and `moe_ep` as on two nodes; `pp`, a stage a node (`stage=4, fsdp=8`) in 16 microbatches, 1f1b on the stages it balances (`[5, 5, 5, 1]`); `pp_even` on `[4, 4, 4, 4]`; `gpipe` |
| `megatron.bux` | 8 | Qwen3-30B-A3B at Megatron-Bridge's workload, MXFP8, routed by random logits: `sorted` (`fsdp=8`, 8 microbatches, `gather=layer`) and `nccl_ep` (`expert=8`, a capacity of 1.25, 8 microbatches) |

`bux plan` shows each run's size and split before it trains:

```console
$ bux plan examples/benchmarks/one_node.bux
6 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
bec9fee8c8  1.28B   7.86M   0.0        6.91B      8     611MiB      1.19GiB    0%      one_node[fsdp]
bec9fee8c8  1.28B   7.86M   0.0        6.91B      8     611MiB      1.19GiB    0%      one_node[fsdp_m8]
47227be352  5.16B   7.86M   0.0        6.92B      8     2.4GiB      4.8GiB     0%      one_node[moe]
47227be352  5.16B   7.86M   0.0        6.92B      8     2.4GiB      4.8GiB     0%      one_node[moe_ep]
4788ab297e  5.52B   983k    0.0        27.6B      8     2.57GiB     5.15GiB    0%      one_node[tp]
bec9fee8c8  1.47B   7.86M   0.0        6.91B      8     951MiB      1.86GiB    11%     one_node[pp]
$ bux plan examples/benchmarks/megatron.bux
2 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
d8e4bfa9e3  30.5B   25.2M   0.0        23.1B      8     14.2GiB     28.4GiB    0%      megatron[sorted]
d8e4bfa9e3  30.5B   25.2M   0.0        23.1B      8     14.2GiB     28.4GiB    0%      megatron[nccl_ep]
```

Runs that split the same model another way share an id, as `mesh`, `layout` and
`methods` are not in it.

## Running a suite

```sh
python scripts/benchmark.py run examples/benchmarks/one_gpu.bux --out new.json
python scripts/benchmark.py compare old.json new.json
```

| command | arguments |
|---------|-----------|
| `run SUITE [SETTING ...]` | `--out FILE`, the results (a `.json`, required); `--tree PATH`, train with the bux of another checkout; `--only TEXT`, only the runs whose name has TEXT. Settings after the suite go to every run: `'steps = 60'` |
| `compare OLD NEW` | `--noise`, of tokens a second and memory (`0.03`); `--loss`, of the last loss (`0.05`) |

`run` trains each run in a worker of its own, on the GPUs its mesh needs, as many side
by side as fit, as a sweep's launch does: the runs of a one-GPU suite train a GPU each,
all at once. Of each run it keeps, in the JSON file:

- the median `tokens_per_s` and `mfu` of the logs after the first, whose time includes
  compiling;
- the peak memory of the busiest GPU, of all hosts;
- the loss at every log, the seconds the run took, the device and the bux it ran.

The runs' directories and each worker's output (`<run>.txt`) go in the folder beside
the JSON file (`new/` beside `new.json`). A run that fails says why, and where its
output is.

On many hosts each runs the same command, with `JAX_NUM_PROCESSES` (or Slurm's or Open
MPI's variables) as `bux launch` takes them ([Launching](../training/launching.md)).
The runs then train one at a time on all of them, each run's coordinator at
`JAX_COORDINATOR_ADDRESS`'s port plus its index, and the first host writes the
results.

## Comparing two trees

Nodes differ by a few percent, so measure a change against the tree before it, on the
same node. `--tree PATH` trains the suite with another checkout's bux, `PATH/src`, in
this one's environment. A one-GPU suite then runs both side by side on half a node
each, and a node's suite runs one tree after the other:

```sh
git worktree add ../before main
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/benchmark.py run examples/benchmarks/one_gpu.bux \
    --tree ../before --out old.json &
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/benchmark.py run examples/benchmarks/one_gpu.bux \
    --out new.json
wait
python scripts/benchmark.py compare old.json new.json
```

`compare` prints a line a run. Here, a tree against the one before it trained KDA by
bux's own kernels (`pallas_delta`), side by side on a node of 8 B300s:

```text
run              old tok/s  new tok/s    diff  new MFU  old GiB  new GiB  loss diff
one_gpu[dense]     145,744    142,645   -2.1%    43.8%     51.1     51.1    +0.0004
one_gpu[gdn]       123,164    120,909   -1.8%    38.0%     70.6     70.6    -0.0007
one_gpu[kda]        59,425     96,273  +62.0%    30.4%    122.0     88.0    -0.0057  faster, less memory
one_gpu[kdab]      127,287    127,309   +0.0%    40.2%     91.4     91.4    -0.0012
one_gpu[moe]       106,900    105,503   -1.3%    32.5%    134.0    134.0    +0.0125
one_gpu[mamba2]     95,117     95,220   +0.1%    32.4%    122.6    122.6    -0.0004
one_gpu[mxfp8]     165,818    164,798   -0.6%    50.6%     46.9     46.9    -0.0001
one_gpu[nvfp4]     181,715    181,760   +0.0%    55.8%     43.8     43.8    -0.0001
```

The rest moved by noise: with the halves of the node swapped, the dense model moved by
+2.4% where it had by -2.1%.

`diff` is the change of the tokens a second, and `loss diff` the change of the loss at
the last log. The flags at the end of a line:

| flag | when |
|------|------|
| `slower`, `faster` | its tokens a second move by more than `--noise` (3%) |
| `more memory`, `less memory` | its peak moves by more than `--noise` |
| `loss` | its last loss moves by more than `--loss` (0.05) |
| `failed` | the new run failed |

`compare` exits with 1 if any run is flagged but `faster` or `less memory`, so it can
gate a change in a script.

## Noise and determinism

A one-GPU suite side by side on two halves of a node moves by about 2% from one half
to the other (above). The node's suite, one tree after the other, moved by 0.3% at
most, and the two nodes' by 1.7%.

XLA's GEMM autotuner picks kernels by timing them, and some kernels add in no fixed
order, so two runs of one tree round differently: on these random tokens the MoE,
whose top-k picks amplify it, ended up to 0.054 apart, every other run 0.011 at most.
Pin both before calling a smaller change of a loss a regression:

```sh
XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true" \
    python scripts/benchmark.py run examples/benchmarks/one_gpu.bux --out new.json
```

on both trees. Under XLA's deterministic ops cuDNN's attention backward runs as
tokamax's, and the runs read slower than the defaults: compare pinned runs only with
pinned runs.

## Results

Every number here is from B300s (268.6 GiB, 148 SMs), in nodes of 8, with InfiniBand
between nodes, in October 2026. Each table says when it was measured, and the merge
into `main` that it measured or that followed it.

(against-megatron-bridge)=

### Against Megatron-Bridge

Megatron-Bridge's performance summary (26.06) trains Qwen3-30B-A3B on 8 B300s at
**40,769 tokens a second a GPU** (938 model TFLOP/s): MXFP8, EP=8, 8 rows of 4096 a GPU,
no recompute, the routing forced even. Its config for that row (r0.5.0) trains a
global batch of 512 rows of 4096: 8 microbatches of 8 rows a GPU. Its recipe also:

- runs attention in FP8 (cuDNN), and in MXFP8 on B300s;
- dispatches by HybridEP on 32 SMs beside the other microbatch's compute, hiding 7 of
  every 8 token exchanges by running a microbatch's forward beside the previous one's
  backward (combined 1F1B);
- delays the weight gradients' GEMMs under the exchanges;
- captures the whole iteration in one CUDA graph;
- runs the experts by cuDNN-frontend's CuTe DSL MXFP8 grouped GEMMs with the SwiGLU in
  their epilogue.

`examples/benchmarks/megatron.bux` is the same model and workload, 23.1 GFLOP a token
as Megatron counts it (`bux plan` above), routed by random logits as Megatron's
`moe_router_force_load_balancing` routes. Its `sorted` arm splits the experts' kernels
under FSDP and gathers a layer at a time, as a whole copy of the model does not fit
beside a microbatch; its `nccl_ep` arm splits the experts over the GPUs by NCCL EP.
bux keeps softmax attention in bfloat16
([Precision](../training/precision.md) says why).

**On 8 B300s, `megatron.bux` at its defaults, tokens a second a GPU:**

| tree | workload | `sorted`, `fsdp=8` | `nccl_ep`, `expert=8` | `sorted`, of Megatron's |
|------|----------|--------------------|-----------------------|-------------------------|
| main before the first wave (2026-10-24) | 8 rows a GPU, one microbatch | 26.4k, 179 GiB (`remat=auto` took `dots`) | 21.7k, 99.5 GiB (`save_attention`) | 65% |
| the first wave, merged at `496fe21` (2026-10-24) | the same | **36.9k** (`remat=auto` takes `none`) | 30.8k (`none`) | 90% |
| main before the second round | Megatron's: 512 rows in 8 microbatches | 37,436, 229.6 GiB | fails | |
| the second round, merged at `0f0cfe2` (2026-10-25) | the same | **42,391**, 214.5 GiB | 34,873, 236.1 GiB | **104%** |

`nccl_ep` reaches 86% of Megatron's at its workload.

::::{dropdown} What the first wave changed (26.4k to 36.9k)

Before the wave, the pool at JAX's 75% of a GPU: `sorted` 26.4k (65%), at 95% of a GPU
28.0k (69%, `dots_attention`, 200 GiB), at 97% 30.9k (76%, `none`, 250 GiB);
`nccl_ep` at a capacity of 1.25 21.7k (53%); `ragged_all_to_all` out of memory.

- **The MoE layer.** The router's weights multiply each pick's hidden units before
  `down`, as Megatron does, so the combine is a plain sum and nothing d wide is kept;
  up and gate are one stack; `moe(random_logits=true)` routes as Megatron's benchmarks
  do.
- **The experts' quantizer** reads the picks' rows from their tokens by index, and the
  down GEMM sums its rows into their tokens: no 262144 × 2048 copy either way.
- **Layouts and memory.** FSDP splits a parameter's first whole dimension, so shards
  are whole rows and collectives need no transposes; XLA's pool takes 95% of a GPU;
  `remat=auto` compiles `none` alone, then the rest side by side.
- **Attention.** Rows of one document attend by cuDNN's causal kernels: `sorted`
  +8.8%.
- **NCCL EP** takes chunks and microbatches, and runs on a stream of its own with
  `moe_stream=true`, opt-in.

| run | before | the MoE, quantizer, layouts and NCCL EP | and attention |
|-----|--------|-----------------------------------------|---------------|
| `sorted`, `fsdp=8` | 26.4k, `dots`, 179 GiB | 35.8k, `none`, 215 GiB | **36.9k** |
| `nccl_ep`, `expert=8` | 21.7k, `save_attention`, 99.5 GiB | 29.2k, `none`, 211 GiB | 30.8k |
| the same, `moe_stream=true` | | 29.6k, 211 GiB | |

`remat=auto` compiles once for both now (130-150 s, from 644 and 950). `sorted`'s
step on GPU 0 is 884 ms, from 1309 (Megatron's ~800): the experts' GEMMs 179,
attention 138, their quantizer 100, FSDP's collectives 281 of which 62 exposed, and 52
idle ([Profiling](profiling.md#an-example-qwen3-30b-a3b-on-8-b300s)). 3000 steps of
FineWeb-Edu on the benchmarks' MoE end within 0.11% of the tree before at three seeds,
which differ by 1.7%.
::::

::::{dropdown} What the second round changed (37.4k to 42.4k)

At Megatron's batch, main trained `sorted` at 37.4k and `nccl_ep` not at all: NCCL
EP's buffers were sized for a step. At one microbatch `sorted`'s step took 898 ms
(768 computing, 61 of collectives alone, 69 idle), `nccl_ep`'s 1093 (688, 326, 78).

| change | `sorted` |
|--------|----------|
| each weight gradient's reduce-scatter starts beside the next layer's backward (`compiler(reductions=early)`, {ref}`XLA's options <compiler-reductions>`) | +5% |
| the experts' SwiGLU MLP by cuDNN-frontend's CuTe DSL MXFP8 kernels, each GPU's shard quantized before FSDP gathers it ([Kernels](kernels.md)) | +4.5%, 15 GiB less |
| QK-norm and RoPE in one pass each way, the dense MXFP8 quantizer at 4.3-4.7 TB/s, the loss's target logits gathered, the router's GEMM from bfloat16 | +2.6% |
| the GPU no longer waits for the host between steps ([Tuning](tuning.md#the-host-between-steps)) | +0.6% |

NCCL EP's buffers are sized by a microbatch now, so `nccl_ep` trains at Megatron's
batch; each backward call prepares its own route, so any order of the calls computes
the same. `methods(interleave=true)` runs each microbatch's forward beside the
backward of the one before: `nccl_ep` on 12 layers +4%, off by default, as 48 layers
then fit only with recompute ([Overlap](tuning.md#overlap-of-communication)).

3000 steps of FineWeb-Edu on the baseline's MoE with experts of 768 and QK-norm end
within 0.12% of main at each of three seeds, which differ by 1.6% (mean -0.01%).

Left open: `nccl_ep`'s exchange is hidden only beside another microbatch
(`interleave`), which needs a carry XLA does not copy to fit 48 layers without
recompute; an eval or a checkpoint still makes the host wait for the GPU before the
next step; ~40 ms of reductions a microbatch still run alone at the end of its
backward (the last layers', the embedding's, the head's in float32); and the MXFP8
experts' Mosaic kernels are lowered again at every call, and CuTe DSL compiles its
kernels in every process: 20-110 s of a compile.
::::

### One GPU

The one-GPU suite, a B300 each, side by side on a node, as `compare` printed it above
(the polish wave, 2026-10-24, merged at `22a9ee3`):

| run | tok/s | MFU | peak |
|-----|-------|-----|------|
| `dense` | 142,645 | 43.8% | 51.1 GiB |
| `gdn` | 120,909 | 38.0% | 70.6 GiB |
| `kda` | 96,273 | 30.4% | 88.0 GiB |
| `kdab` | 127,309 | 40.2% | 91.4 GiB |
| `moe` | 105,503 | 32.5% | 134.0 GiB |
| `mamba2` | 95,220 | 32.4% | 122.6 GiB |
| `mxfp8` | 164,798 | 50.6% | 46.9 GiB |
| `nvfp4` | 181,760 | 55.8% | 43.8 GiB |

Since then, each tree against the one before, side by side:

| merged | change of the one-GPU suite |
|--------|-----------------------------|
| `496fe21` (2026-10-24), the first Megatron wave | dense +5.8%, MoE +6.2% (16 GiB less), MXFP8 +3.5%, NVFP4 +8.3%, the hybrids within 2.1% |
| `0f0cfe2` (2026-10-25), the second round, but the host fix | dense +1.5%, MoE +1.5%, NVFP4 +2.1%, the others within 1% (the GDN hybrid trains at 128.3k on one GPU in turn in both trees) |
| `cc9e425` (2026-10-25), the host fix | +1.2% to +7.2% |

Tokens a second by model, with their MFU and on `fsdp=8`, are in
[Kernels](kernels.md#tokens-a-second-by-model).

### One node

The one-node suite, one tree after the other on 8 B300s (the polish wave, 2026-10-24,
merged at `22a9ee3`):

| run | tok/s |
|-----|-------|
| `fsdp` | 1.13M |
| `fsdp_m8` | 923k |
| `moe` | 832k |
| `moe_ep` | 540k |
| `tp` | 227k |
| `pp` | 683k |

Since then:

| merged | change of the one-node suite |
|--------|------------------------------|
| `496fe21` (2026-10-24) | `fsdp` +5.9%, `fsdp_m8` +4.8%, `moe` +9.1% and `moe_ep` +6.7% (12 GiB less each), `tp` +2.3%, `pp` +0.6% (1.2 GiB more) |
| `0f0cfe2` (2026-10-25), but the host fix | `fsdp` +0.9%, `fsdp_m8` +6.3%, `moe` +0.1%, `moe_ep` +1.2%, `tp` +1.9%, `pp` -0.0% |
| `cc9e425` (2026-10-25), the host fix | `fsdp` 1.224M to **1.245M**, `fsdp_m8` 1.012M to **1.054M** |

FSDP over 8 GPUs gets 97% of 8 times one GPU's tokens. Each way of splitting the
baseline on a node, with its measurements, is in [Parallelism](../parallelism/index.md).

### Two nodes

The two-node suite on 16 B300s (the polish wave, 2026-10-24): FSDP and HSDP 2.21M, the
MoE sorted 1.58M and by EP 1.04M, the GDN hybrid 1.60M.

- **The dense 1.3B** (2026-10-23), 8 rows of 4096 a GPU: FSDP over both nodes,
  `fsdp=16`, keeps 98.1% of one node's rate a GPU, and HSDP, `data=2, fsdp=8`, 98.9%.
  With the compile cache on a filesystem both nodes mount, a launch reaches its fifth
  step in 42 s warm, against 78-80 s cold
  ([On B300s](../parallelism/multi-node.md#on-b300s)).
- **The MoE** at a capacity of 1.25, by the suites' `moe` and `moe_ep` (2026-10-24, the
  polish wave): `sorted` under HSDP 1,586k. EP in a node and data across
  (`data=2, expert=8`): 1,016k by `ragged_all_to_all` and 1,192k by `nccl_ep`. EP over
  both nodes (`expert=16`): 324k and 773k
  ([One node and two](../parallelism/experts.md#one-node-and-two)).

### Four and eight nodes

`four_nodes.bux` on 32 GPUs, and the same runs on 64 GPUs and 512 rows, 8 rows of 4096
a GPU (2026-10-24, the polish wave). A GPU's rate is against one node's, the mean of
four nodes alone (they spread by 4.5%):

| run | mesh | GPUs | a GPU | of one node's |
|-----|------|------|-------|---------------|
| dense, FSDP | `fsdp=32` | 32 | 132.3k | 94.3% |
| | `fsdp=64` | 64 | 122.2k | 87.0% |
| dense, HSDP | `data=4, fsdp=8` | 32 | 136.6k | 97.3% |
| | `data=8, fsdp=8` | 64 | 135.4k | 96.4% |
| MoE, sorted under HSDP | `data=4, fsdp=8` | 32 | 95.2k | 92.7% |
| | `data=8, fsdp=8` | 64 | 92.8k | 90.4% |
| MoE, EP within a node | `data=4, expert=8` | 32 | 62.2k | 91.4% |
| | `data=8, expert=8` | 64 | 61.4k | 90.3% |
| dense, 1f1b, a stage a node | `stage=4, fsdp=8` | 32 | 76.2k | 54.3% |
| | `stage=4, data=2, fsdp=8` | 64 | 76.4k | 54.4% |

HSDP keeps 96-97% of a node's rate a GPU on four nodes and on eight, where FSDP over
all the GPUs falls to 87% on eight. The pipeline's rate is against one node's FSDP:
what it loses is its bubble and its microbatches' small GEMMs, not the sends between
nodes. Tokens a second, launch times and the reasons are in
[Multi-node](../parallelism/multi-node.md).

### Kernel speedups

On a B300, bux's kernels run 1.4x (QK-norm then RoPE, forward) to 27x (the MoE's
experts, against `jax.lax.ragged_dot`) as fast as XLA's own program for the same
computation: softmax attention 7x, Mamba-2 by chunks 25x a scan over positions, KDA
bounded at -5 by cuDNN's kernel 10x the chunks. [The kernels](kernels.md#the-kernels)
lists every kernel with its speedup, and when it falls back.

### Earlier waves

::::{dropdown} The performance wave: `sota` against `main`, merged at `cfc8561` (2026-10-23)

On B300s, `sota` against `main` (`83dc97f`), each with its own defaults:

| run | main | sota |
|-----|------|------|
| dense, 1 GPU | 143.8k tok/s, 70.6 GiB | 144.0k, 51.1 GiB |
| GDN hybrid, 1 GPU | 93.5k, 140.4 GiB | 122.1k, 70.6 GiB |
| KDA bounded, 1 GPU | 113.5k, 176.2 GiB | 128.0k, 91.4 GiB |
| MXFP8, 1 GPU | 168.2k | 171.0k |
| NVFP4 everywhere, 1 GPU | 0.3x bf16 | 178.3k |
| PP stage=4, 8 GPUs | gpipe 483k, 65.0 GiB | 1f1b, uneven stages, 595k, 52.7 GiB |
| TP=8 + SP in MXFP8 | 245.1k | `tensor=chunked` 260.9k |
| EP=8, 32k tokens a GPU | out of memory | 549k (`sorted` without EP: 820k) |

Two nodes, 16 GPUs, a dense 1.3B: FSDP and HSDP train at 97% of one node's rate a GPU;
at 8 microbatches, reducing once makes HSDP 100% of it, against 90% reduced each
microbatch. A second launch with JAX's cache on a shared filesystem reaches step 5 in
42 s instead of 75.
::::

::::{dropdown} The polish wave, merged at `22a9ee3` (2026-10-24)

Against `main`, side by side by `scripts/benchmark.py`: no run of the one-GPU, one-node
or two-node suites is slower past noise (3%). On packed documents, with autotuning off,
the losses are main's bit for bit at every log in bfloat16, float32 and MXFP8. With
nothing but `matmuls=mxfp8` set, the MoE trains 14% faster under FSDP and 19% by EP;
with nothing set, a pipeline 8% faster in half the memory.

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
::::
