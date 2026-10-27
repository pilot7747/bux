# Benchmarks

Every change to bux's speed is measured the same way: a suite of runs from
`examples/benchmarks/`, trained by `scripts/benchmark.py`, against the tree before the
change on the same node. This page describes the suites, shows how to run and compare
them, and collects the results: from one GPU to eight nodes, and against
Megatron-Bridge at its own workloads.

Use it to check that your change is not slower, or to see how fast bux trains a model
like yours before you start.

## The suites

Every suite except the `megatron*.bux` ones builds on `baseline.bux`: a Llama-style
1.3B (`d_model` 2048, 16 layers, a vocabulary of 128,256, GQA with 16 heads and 8 kv
heads, a gated MLP of 5632), its hybrids with Gated DeltaNet, KDA and Mamba-2, and its
MoE (64 experts of 704, top 8). They train on random tokens in rows of 4096, 8 rows per
GPU, in bfloat16, for 30 steps logged every 5.

| Suite | GPUs | Runs |
|-------|------|------|
| `one_gpu.bux` | 1 per run | `dense`; the hybrids `gdn` (`"DDDA"^4`), `kda` (`"KKKA"^4`), `kdab` (KDA with `gate_lower_bound=-5`) and `mamba2` (`"MMMA"^4`); `moe` (`E^16`); the dense model in `mxfp8` and `nvfp4` |
| `one_node.bux` | 8 | `fsdp` (`fsdp=8`); `fsdp_m8`, the same in 8 microbatches; `moe`, the experts sorted under `fsdp=8`; `moe_ep`, a ragged all-to-all over `expert=8` at a capacity of 1.25; `tp`, 4 of Llama 3 70B's layers (`d_model` 8192) on `tensor=8` with sequence parallelism, scanned, `remat=full`, `tensor=chunked`, 1 row per GPU; `pp`, two stages of 10 and 6 layers on `stage=2, fsdp=4`, 1f1b in 8 microbatches |
| `two_nodes.bux` | 16 | `fsdp` (`fsdp=16`); `hsdp` (`data=2, fsdp=8`); `gdn`, the GDN hybrid under HSDP; `moe`, sorted under HSDP; `moe_ep`, EP within a node (`data=2, expert=8`) |
| `four_nodes.bux` | 32 | `fsdp` (`fsdp=32`); `hsdp` (`data=4, fsdp=8`); `moe` and `moe_ep` as on two nodes; `pp`, one stage per node (`stage=4, fsdp=8`), 1f1b in 16 microbatches on stages balanced by FLOPs (`[5, 5, 5, 1]`); `pp_even` on `[4, 4, 4, 4]`; `gpipe` |
| `megatron.bux` | 8 | Qwen3-30B-A3B at Megatron-Bridge's workload: `sorted` (`fsdp=8`, 8 microbatches, `gather=layer`) and `nccl_ep` (`expert=8`, a capacity of 1.25, 8 microbatches) |
| `megatron_nano.bux` | 8 | Nemotron 3 Nano 30B-A3B (512 rows of 8192, 4 rows per GPU per microbatch): `sorted` (`fsdp=8`, `gather=layer`) and `nccl_ep` (`expert=8`, a capacity of 1.25) |
| `megatron_llama3_8b.bux` | 8 | Llama 3 8B at Megatron-Bridge's B300 recipe (128 rows of 8192, 4 rows per GPU per microbatch): `fsdp`, and `zero1` (only the optimizer state split, as Megatron's distributed optimizer splits it) |
| `megatron_gpt_oss.bux` | 64 | gpt-oss-120b (1280 rows of 4096, 4 rows per GPU per microbatch), the experts over a node's GPUs and their shards over the nodes (`fsdp=8, expert=8`): `nccl_ep` and `all_to_all`, at a capacity of 1.25 |
| `megatron_super.bux` | 64 | Nemotron 3 Super 120B-A12B (512 rows of 8192, 1 row per GPU per microbatch), `fsdp=8, expert=8` by NCCL EP, its MTP module in bfloat16: `mxfp8`, and `nvfp4` as Megatron's recipe quantizes it |

The `megatron*.bux` suites multiply in MXFP8 and route their MoEs by random logits, as
Megatron-Bridge's benchmarks do ([Against Megatron-Bridge](#against-megatron-bridge)).

`bux plan` shows each run's size and split before it trains:

```console
$ bux plan examples/benchmarks/one_node.bux
6 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
2b8064152c  1.28B   7.86M   0.0        6.91B      8     611MiB      1.19GiB    0%      one_node[fsdp]
2b8064152c  1.28B   7.86M   0.0        6.91B      8     611MiB      1.19GiB    0%      one_node[fsdp_m8]
fdebd6e1ac  5.16B   7.86M   0.0        6.92B      8     2.4GiB      4.8GiB     0%      one_node[moe]
fdebd6e1ac  5.16B   7.86M   0.0        6.92B      8     2.4GiB      4.8GiB     0%      one_node[moe_ep]
0e1a7bf3da  5.52B   983k    0.0        27.6B      8     2.57GiB     5.15GiB    0%      one_node[tp]
2b8064152c  1.47B   7.86M   0.0        6.91B      8     951MiB      1.86GiB    11%     one_node[pp]
$ bux plan examples/benchmarks/megatron.bux
2 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
e10546bbb2  30.5B   25.2M   0.0        23.1B      8     14.2GiB     28.4GiB    0%      megatron[sorted]
e10546bbb2  30.5B   25.2M   0.0        23.1B      8     14.2GiB     28.4GiB    0%      megatron[nccl_ep]
```

Runs that split the same model differently share an id, because `mesh`, `layout` and
`methods` are not part of it.

## Running a suite

```sh
python scripts/benchmark.py run examples/benchmarks/one_gpu.bux --out new.json
python scripts/benchmark.py compare old.json new.json
```

| Command | Options |
|---------|---------|
| `run SUITE [SETTING ...]` | `--out FILE`: where the results go, a `.json` (required). `--tree PATH`: train with another checkout's bux. `--only TEXT`: only the runs whose name contains TEXT. Settings after the suite apply to every run: `'steps = 60'` |
| `compare OLD NEW` | `--noise`: the change in tokens/s and memory that counts (`0.03`). `--loss`: the change in the last loss that counts (`0.05`) |

`run` trains each run in its own worker, on the GPUs its mesh needs, as many side by
side as fit, like a sweep's launch: the runs of a one-GPU suite take a GPU each and
train at once. For each run, the JSON file records:

- the median `tokens_per_s` and `mfu` of the logs after the first, since the first
  includes compiling;
- the peak memory of the busiest GPU across all hosts;
- the loss at every log, the run's wall time, the device, and the bux it ran.

The run directories and each worker's output (`<run>.txt`) go in a folder beside the
JSON file (`new/` beside `new.json`). A run that fails says why, and where its output
is.

On many hosts, each host runs the same command, with `JAX_NUM_PROCESSES` (or Slurm's
or Open MPI's variables) set as for `bux launch` ([Launching](../training/launching.md)).
The runs then train one at a time across all hosts, each run's coordinator at the port
of `JAX_COORDINATOR_ADDRESS` plus the run's index, and the first host writes the
results.

## Comparing two trees

Nodes differ by a few percent, so measure a change against the tree before it, on the
same node. `--tree PATH` trains the suite with another checkout's bux (`PATH/src`) in
this checkout's environment. A one-GPU suite can run both trees side by side, on half
a node each; a node's suite runs one tree after the other:

```sh
git worktree add ../before main
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/benchmark.py run examples/benchmarks/one_gpu.bux \
    --tree ../before --out old.json &
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/benchmark.py run examples/benchmarks/one_gpu.bux \
    --out new.json
wait
python scripts/benchmark.py compare old.json new.json
```

`compare` prints one line per run. Here a change that moved KDA onto bux's own kernels
(`pallas_delta`), side by side on a node of 8 B300s:

```text
run              old tok/s  new tok/s    diff  new MFU  old GiB  new GiB  loss diff
one_gpu[dense]     145,744    142,645   -2.1%    43.8%     51.1     51.1    +0.0004
one_gpu[gdn]       123,164    120,909   -1.8%    38.0%     70.6     70.6    -0.0007
one_gpu[kda]        59,425     96,273  +62.0%    30.4%    122.0     88.0    -0.0057  faster, less memory
one_gpu[kdab]      127,287    127,309   +0.0%    40.2%     91.4     91.4    -0.0012
...
```

`diff` is the change in tokens/s, and `loss diff` the change in the loss at the last
log. A flag at the end of a line marks a change past the thresholds:

| Flag | When |
|------|------|
| `slower`, `faster` | tokens/s moved by more than `--noise` (3%) |
| `more memory`, `less memory` | peak memory moved by more than `--noise` |
| `loss` | the last loss moved by more than `--loss` (0.05) |
| `failed` | the new run failed |

`compare` exits with 1 if any run has a flag other than `faster` or `less memory`, so
a script can use it to gate a change.

## Noise and determinism

Side by side on two halves of a node, a one-GPU suite moves by about 2%: in the
comparison above, the dense model's -2.1% became +2.4% with the halves swapped. A
node's suite, one tree after the other, moved by 0.3% at most, and the two-node suite
by 1.7%.

Losses move too. XLA's GEMM autotuner picks kernels by timing them, and some kernels
add in no fixed order, so two runs of one tree round differently. On random tokens the
MoE, whose top-k picks amplify the difference, ended up to 0.054 apart, and every other
run at most 0.011. Before you call a smaller change in a loss a regression, pin both on
both trees:

```sh
XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true" \
    python scripts/benchmark.py run examples/benchmarks/one_gpu.bux --out new.json
```

:::{note}
Under XLA's deterministic ops, cuDNN's attention backward runs as tokamax's, so pinned
runs are slower than the defaults. Compare pinned runs only with pinned runs.
:::

## Results

Every number here is from B300s (268.6 GiB, 148 SMs) in nodes of 8, with InfiniBand
between nodes, in October 2026; each section gives its date. `tokens_per_s` is the
median of the logs after the first, as `scripts/benchmark.py` reads it.

(against-megatron-bridge)=

### Against Megatron-Bridge

Megatron-Bridge's performance summary (26.06) trains Qwen3-30B-A3B on 8 B300s at
**40,769 tok/s/GPU** (938 model TFLOP/s): MXFP8, EP=8, 8 rows of 4096 per GPU, no
recompute, routing forced even. Its config for that row (r0.5.0) trains a global batch
of 512 rows of 4096 in 8 microbatches of 8 rows per GPU. Its recipe also:

- runs attention in FP8 by cuDNN (MXFP8 on B300s);
- dispatches by HybridEP on 32 SMs, and hides 7 of every 8 token exchanges by running
  one microbatch's forward beside the previous one's backward (combined 1F1B);
- delays the weight gradients' GEMMs to run under the exchanges;
- captures the whole iteration in one CUDA graph;
- runs the experts by cuDNN-frontend's CuTe DSL MXFP8 grouped GEMMs, with the SwiGLU
  in their epilogue.

`examples/benchmarks/megatron.bux` trains the same model at the same workload: 23.1
GFLOP per token as Megatron counts it (`bux plan` above), routed by random logits as
Megatron's `moe_router_force_load_balancing` routes. Its `sorted` run splits the
experts' weights under FSDP and gathers one layer at a time, because a whole copy of
the model does not fit beside a microbatch. Its `nccl_ep` run splits the experts over
the GPUs by NCCL EP. bux keeps softmax attention in bfloat16
([Precision](../training/precision.md) says why).

On 8 B300s, on 2026-10-25:

| Run | Mesh | tok/s/GPU | Peak memory | Of Megatron-Bridge |
|-----|------|-----------|-------------|--------------------|
| `sorted` | `fsdp=8` | **42,391** | 214.5 GiB | **104%** |
| `nccl_ep` | `expert=8` | 34,873 | 236.1 GiB | 86% |

Known gaps:

- **NCCL EP's exchange hides only beside another microbatch.**
  `methods(interleave=true)` runs it there, and gains 4% on 12 layers, but 48 layers
  then fit only with recompute, so it is off by default
  ([Overlap](tuning.md#overlap-of-communication)).
- **About 40 ms of reductions per microbatch run alone** at the end of its backward:
  the last layers', the embedding's, and the head's in float32.
- **An eval or a checkpoint makes the host wait for the GPU** before the next step
  ([The host between steps](tuning.md#the-host-between-steps)).
- **Compiles are long.** The MXFP8 experts' Mosaic kernels are lowered again at every
  call, and CuTe DSL compiles its kernels in every process: 20 to 110 s of a compile.

(megatron-other-workloads)=

### Megatron-Bridge's other workloads

The same summary has four more B300 rows that fit on 8 or 64 GPUs.
`examples/benchmarks/megatron_*.bux` train them as Megatron-Bridge's configs do: its
global batch, microbatch and row length, EP=8, every layer in MXFP8 (its performance
script turns off the bfloat16 first and last layers), and routing forced even by random
logits. On B300s, on 2026-10-25:

| Model | GPUs | bux run | bux, tok/s/GPU | Megatron-Bridge | Of Megatron-Bridge | Peak | `remat=auto` took |
|-------|------|---------|----------------|-----------------|--------------------|------|-------------------|
| Nemotron 3 Nano 30B-A3B | 8 | `sorted`, `fsdp=8` | **31,912** | 37,316 | **86%** | 192 GiB | `dots` |
| | | `nccl_ep`, `expert=8` | out of memory at the first step | | | | |
| gpt-oss-120b | 64 | `nccl_ep`, `fsdp=8, expert=8` | **16,581** | 18,534 | **89%** | 151 GiB | `none` |
| | | `all_to_all` | 16,042 | | 87% | 161 GiB | `none` |
| Nemotron 3 Super 120B-A12B | 64 | `mxfp8`, NCCL EP | **5,317** | 7,867 (MXFP8) | **68%** | 240 GiB | `dots` |
| | | `nvfp4` | 5,049 | 8,131 (NVFP4) | 62% | 240 GiB | `dots` |
| Llama 3 8B | 8 | `fsdp=8` | **32,749** | 29,521 (8 B200s) | **111%** | 174 GiB | |
| | | `zero1` | out of memory | | | | |

- **Llama 3 8B.** Megatron-Bridge publishes no B300 number for it. Its B200 number
  (25.11, MXFP8) stands in, since a B200 has the same dense FP8 peak as a B300. In FP8
  with current scaling it trains at 30,624 there.
- **gpt-oss-120b** was measured before bux gave its routed experts their biases:
  without them, and with its activation approximated by a SwiGLU clamped at 7.
  `gpt_oss_120b.bux` now has both ([its experts](../architectures/experts.md#what-it-computes)).
- **Nemotron 3 Super** runs its two MTP depths, which share one module, in bfloat16. In
  NVFP4, as Megatron's quantization config has it, attention, the latent projections
  and the last 14 layers run in bfloat16 too, and Mamba's output projection in MXFP8.
- **The models match in size.** FLOPs per token, bux's against what Megatron's numbers
  imply: gpt-oss-120b 32.7G against 32.6G, Super 85.9G against 84.8G, Llama 3 8B 51.5G
  against 51.45G, Nano 20.9G against 22.3G (the two count Mamba differently).
- **Vocabularies** are those of the published configs: gpt-oss's is 201,088.
- **Read the median.** In these runs the last log's `tokens_per_s` read too high
  (Nano: 376k against a median of 255k), because the host no longer waits for the GPU
  between steps.

Known gaps:

- **An MoE across nodes reduces its experts' gradients every microbatch.** A router
  learns from the whole batch, so `methods.accumulate` is `reduced` there, and each
  microbatch reduce-scatters the experts' gradients between nodes, where Megatron does
  it once per step ([Gathered once, reduced
  once](../parallelism/microbatches.md#gathered-once-reduced-once-methodsgather-methodsaccumulate)).
  That is likely much of Super's gap.
- **Nano's `nccl_ep` runs out of memory**: `remat=auto` estimates its step without NCCL
  EP's own buffers, which at rows of 8192 do not fit beside `none`.

### One GPU

The one-GPU suite on 2026-10-24, one run per B300, side by side on a node (the `new`
column of the comparison above):

| Run | tok/s | MFU | Peak memory |
|-----|-------|-----|-------------|
| `dense` | 142,645 | 43.8% | 51.1 GiB |
| `gdn` | 120,909 | 38.0% | 70.6 GiB |
| `kda` | 96,273 | 30.4% | 88.0 GiB |
| `kdab` | 127,309 | 40.2% | 91.4 GiB |
| `moe` | 105,503 | 32.5% | 134.0 GiB |
| `mamba2` | 95,220 | 32.4% | 122.6 GiB |
| `mxfp8` | 164,798 | 50.6% | 46.9 GiB |
| `nvfp4` | 181,760 | 55.8% | 43.8 GiB |

Changes merged by 2026-10-25 made the suite faster still, each measured against the
tree before it: the dense model by 5.8% and then 1.5%, the MoE by 6.2% (in 16 GiB less)
and 1.5%, MXFP8 by 3.5%, NVFP4 by 8.3% and 2.1%, and the hybrids by up to 2.1%. Every
run then gained another 1.2% to 7.2% once the GPU stopped waiting for the host between
steps.

### One node

The one-node suite on 2026-10-24, on 8 B300s, one run after the other:

| Run | tok/s | tok/s/GPU |
|-----|-------|-----------|
| `fsdp` | 1.13M | 141k |
| `fsdp_m8` | 923k | 115k |
| `moe` | 832k | 104k |
| `moe_ep` | 540k | 67.5k |
| `tp` | 227k | 28.4k |
| `pp` | 683k | 85.4k |

By 2026-10-25, `fsdp` trained at **1.245M** and `fsdp_m8` at **1.054M**, and the other
runs had gained from 0.6% (`pp`) to 9.2% (`moe`, in 12 GiB less).

On a node, FSDP keeps 97% of a GPU's rate alone: 1.12M tokens/s on `mesh(fsdp=8)`
against 144k on one GPU, measured together in October 2026. In the same measurement,
the suite's other models under `mesh(fsdp=8)` trained at 818k (the GDN hybrid), 741k
(the Mamba-2 hybrid), 926k (bounded KDA) and 834k (the MoE). Each way of splitting the
baseline on a node, with its measurements, is in [Parallelism](../parallelism/index.md).

### Two nodes

The two-node suite on 16 B300s, on 2026-10-24:

| Run | Mesh | tok/s | tok/s/GPU |
|-----|------|-------|-----------|
| dense, FSDP | `fsdp=16` | 2.21M | 138k |
| dense, HSDP | `data=2, fsdp=8` | 2.21M | 138k |
| GDN hybrid, HSDP | `data=2, fsdp=8` | 1.60M | 100k |
| MoE, sorted under HSDP | `data=2, fsdp=8` | 1.59M | 99.1k |
| MoE, EP within a node, `nccl_ep` | `data=2, expert=8` | 1.19M | 74.5k |
| MoE, EP within a node, `ragged_all_to_all` | `data=2, expert=8` | 1.02M | 63.5k |
| MoE, EP over both nodes, `nccl_ep` | `expert=16` | 773k | 48.3k |
| MoE, EP over both nodes, `ragged_all_to_all` | `expert=16` | 324k | 20.3k |

The MoE runs use a capacity of 1.25. On 2026-10-23, the dense 1.3B kept 98.1% of one
node's rate per GPU under FSDP (`fsdp=16`) and 98.9% under HSDP. With the compile cache
on a filesystem both nodes mount, a launch reaches its fifth step in 42 s warm, against
78 to 80 s cold ([Many nodes](../parallelism/multi-node.md#two-nodes)). Every MoE method
over one node and two is in
[Experts](../parallelism/experts.md#one-node-and-two).

### Four and eight nodes

`four_nodes.bux` on 32 GPUs, and the same runs on 64 GPUs with a batch of 512, 8 rows
of 4096 per GPU, on 2026-10-24. Each rate is against one node's, the mean of four nodes
alone (which spread by 4.5%):

| Run | Mesh | GPUs | tok/s/GPU | Of one node's |
|-----|------|------|-----------|---------------|
| dense, FSDP | `fsdp=32` | 32 | 132.3k | 94.3% |
| | `fsdp=64` | 64 | 122.2k | 87.0% |
| dense, HSDP | `data=4, fsdp=8` | 32 | 136.6k | 97.3% |
| | `data=8, fsdp=8` | 64 | 135.4k | 96.4% |
| MoE, sorted under HSDP | `data=4, fsdp=8` | 32 | 95.2k | 92.7% |
| | `data=8, fsdp=8` | 64 | 92.8k | 90.4% |
| MoE, EP within a node | `data=4, expert=8` | 32 | 62.2k | 91.4% |
| | `data=8, expert=8` | 64 | 61.4k | 90.3% |
| dense, 1f1b, one stage per node | `stage=4, fsdp=8` | 32 | 76.2k | 54.3% |
| | `stage=4, data=2, fsdp=8` | 64 | 76.4k | 54.4% |

HSDP keeps 96-97% of a node's rate per GPU on four nodes and on eight, while FSDP over
all the GPUs falls to 87% on eight. The pipeline's rate is against one node's FSDP:
what it loses is its bubble and its microbatches' small GEMMs, not the sends between
nodes. Totals, launch times and the reasons are in
[Multi-node](../parallelism/multi-node.md).

### Kernel speedups

On a B300, bux's kernels run 1.4x (QK-norm then RoPE, forward) to 27x (the MoE's
experts, against `jax.lax.ragged_dot`) as fast as XLA's own program for the same
computation: softmax attention 7x, Mamba-2 by chunks 25x a scan over positions, and
KDA bounded at -5 by cuDNN's kernel 10x its chunks. [The kernels](kernels.md#the-kernels)
lists every kernel with its speedup and when it falls back.
