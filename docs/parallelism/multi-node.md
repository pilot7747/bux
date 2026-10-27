(parallelism-multi-node)=
# Many nodes

A run trains on many hosts with the same `.bux` file and the same command on every
host, one process each. The mesh then spans every process's devices: each node holds
a block of the innermost axes (`tensor` first), and the outer axes span the nodes.
This page covers how the mesh lays out over nodes, the compile cache every process
must share, the network NCCL needs, and how FSDP, HSDP, EP and pipelines scale on two,
four and eight nodes of 8 B300s.

:::{tip}
Across nodes, prefer HSDP, `mesh(data=<nodes>, fsdp=8)`. It keeps 96-97% of one
node's per-GPU rate on four nodes and on eight, where FSDP over every GPU falls to 87%
on eight ([Four nodes and eight](#four-nodes-and-eight)).
:::

## Quick start

FSDP over two nodes of 8 GPUs:

```bux
mesh = mesh(fsdp=16)
```

Or sharded within a node and copied across nodes (HSDP):

```bux
mesh = mesh(data=2, fsdp=8)
```

Each host runs one process, with the same command and three variables:
`JAX_COORDINATOR_ADDRESS` (the first host's `ip:port`), `JAX_NUM_PROCESSES` and
`JAX_PROCESS_ID`:

```sh
JAX_COORDINATOR_ADDRESS=10.0.0.1:12355 JAX_NUM_PROCESSES=2 JAX_PROCESS_ID=0 \
JAX_COMPILATION_CACHE_DIR=/shared/bux-cache \
    bux launch bench.bux 'mesh = mesh(data=2, fsdp=8)' --out /shared/runs
```

Slurm and Open MPI set the processes up the same way
([Launching](../training/launching.md)). `JAX_COMPILATION_CACHE_DIR` must be one
folder that every host mounts ([below](#the-compile-cache)).

`examples/benchmarks/two_nodes.bux` and `four_nodes.bux` are the suites measured
below. `examples/experiments/parallel/tp_70b.bux`, `pp_70b.bux` and `ep_moe.bux` are
runs across nodes ([Examples](index.md#examples)).

## How a mesh spans nodes

XLA puts the GPUs of a node, or of an NVLink domain that spans nodes, in one slice
(`slice_index`), and `jax.make_mesh` refuses devices from more than one slice. So
`parallel.mesh` lays them out itself (`mesh.grid`), as MaxText's hybrid mesh does:

1. Each node holds a block of the innermost axes, filled from `tensor` outward
   (`mesh.split`).
2. The axes outside that block span nodes.

On nodes of 8, `mesh(data=2, fsdp=8, tensor=2)` puts `tensor=2 × fsdp=4` in each node,
and `data=2 × fsdp=2` across them.

These are errors:

- A mesh that does not fill nodes this way, such as `tensor=3` on nodes of 8. The
  error names the axis.
- Devices spread unevenly over nodes: a mesh across nodes takes as many on each.

Where devices have no slice, as CPUs do, each process is a node.

## Each process reads its own rows

Each process reads only the rows of a batch that its devices hold, and leaves the
others 0 (`parallel.local_rows`). No process reads the whole batch, and no source
makes the permutation of a whole corpus ([Reading](../data/pipeline.md)).
`tests/test_nodes.py` checks that processes reading their own rows, from every kind of
source, make the same batch one process does.

## The compile cache

The processes of a job must either all find a program in JAX's cache, or all compile
it. On GPUs, XLA's autotuner splits its work among the processes that compile a
program, and each waits for the others' shares (`xla_gpu_shard_autotuning`). A
process that compiles a program while another loads it from the cache waits a day.

Only the first process writes the cache, so the processes agree only when it is one
folder for all of them:

- Set `JAX_COMPILATION_CACHE_DIR` to a folder on a filesystem every host mounts. The
  default, `~/.cache/bux/jax`, is each host's own. A URL such as `gs://...` counts as
  one folder for all.
- When the processes join, `runtime.agree_on_cache` checks it: the first process
  makes a file there, and the others must see it.
- Where they do not, bux turns the cache off and warns. Every launch then compiles
  everything, in every process together.

:::{warning}
JAX's `jax_share_binary_between_hosts` hangs the same way: the first process
compiles alone and waits for the others' autotuning shares. bux leaves it off.
:::

(multi-node-network)=
## The network

NCCL reaches InfiniBand through libibverbs and its mlx5 provider, which the Docker
image installs. Without them, NCCL talks between nodes over TCP sockets. On two B300
nodes, an all-reduce of 64 MiB per GPU across the nodes then ran at a bus bandwidth of
1 GB/s, against 25 GB/s over InfiniBand (4 and 42 GB/s over all 16 GPUs).

To check, run with `NCCL_DEBUG=INFO`: it should say `Using network IB`, not `Socket`.

A step of microbatches whose rows' axes span hosts adds up its gradients unreduced and
reduces them once
([Gathered once, reduced once](microbatches.md#gathered-once-reduced-once-methodsgather-methodsaccumulate)).
EP across nodes is on [the experts' page](experts.md#transformerengines-nccl-ep).

## Measurements

(multi-node-on-b300s)=
### Two nodes

Two nodes of 8 B300s, 2026-10-23. The 1.3B dense model, 8 rows of 4096 tokens per
GPU, bfloat16, with the cache on a filesystem both nodes mount. The launch time runs
from the start to the fifth step.

| Mesh | GPUs | tok/s | Per GPU | vs. one node | Launch, cold / warm |
|---|---|---|---|---|---|
| `fsdp=8` | 8 | 1,133k | 141.4k | | |
| `fsdp=16` | 16 | 2,220k | 138.7k | 98.1% | 78 s / 42 s |
| `data=2, fsdp=8` (HSDP) | 16 | 2,238k | 139.9k | 98.9% | 80 s / 42 s |

Warm, both processes load the train step from the cache in 0.3 s, where compiling it
takes 16 s. With a cache on each node, which bux turns off, every launch is cold
(76 s).

### Four nodes and eight

2026-10-24. `examples/benchmarks/four_nodes.bux` trains the baseline models on 32 GPUs
of four nodes, 8 rows of 4096 tokens per GPU; on eight nodes the same runs took 64
GPUs and 512 rows. "vs. one node" compares the per-GPU rate with the one-node suite's,
averaged over four of the nodes alone (they spread by 4.5%). The launch runs from the
start to the fifth step, with the cache on a filesystem every node mounts, empty and
then full:

| Run | Mesh | GPUs | tok/s | Per GPU | vs. one node | Launch, cold / warm |
|---|---|---|---|---|---|---|
| dense, FSDP | `fsdp=8` | 8 | 1,123k | 140.4k | | 75 s |
| | `fsdp=32` | 32 | 4,235k | 132.3k | 94.3% | 74 s / 41 s |
| | `fsdp=64` | 64 | 7,820k | 122.2k | 87.0% | 73 s |
| dense, HSDP | `data=4, fsdp=8` | 32 | 4,371k | 136.6k | 97.3% | 74 s / 40 s |
| | `data=8, fsdp=8` | 64 | 8,664k | 135.4k | 96.4% | 77 s |
| MoE, sorted under HSDP | `fsdp=8` | 8 | 822k | 102.7k | | 98 s |
| | `data=4, fsdp=8` | 32 | 3,046k | 95.2k | 92.7% | 105 s / 57 s |
| | `data=8, fsdp=8` | 64 | 5,942k | 92.8k | 90.4% | 112 s |
| MoE, EP within a node | `expert=8` | 8 | 544k | 68.0k | | 108 s |
| | `data=4, expert=8` | 32 | 1,989k | 62.2k | 91.4% | 117 s / 65 s |
| | `data=8, expert=8` | 64 | 3,932k | 61.4k | 90.3% | 116 s |
| dense, 1f1b | `stage=4, fsdp=8` | 32 | 2,438k | 76.2k | 54.3% | 68 s / 36 s |
| | `stage=4, data=2, fsdp=8` | 64 | 4,892k | 76.4k | 54.4% | 73 s |

What the table says:

- **HSDP** gathers and reduces a parameter within a node, and adds up its gradient
  across nodes once a step. It keeps 96-97% of a node's rate on four nodes and on
  eight.
- **FSDP over all the GPUs** gathers every layer across the nodes, forward and
  backward, and loses more as they grow: 98% of a node's rate on two, 94% on four,
  87% on eight.
- **The MoE** keeps 90-93% either way.
- **A pipeline with one stage per node** runs at about 54% of one node's FSDP rate,
  but keeps that rate per GPU on eight nodes, as two copies of it
  ([Stages](pipelines.md#stages-methodsstages)).
- **Warm**, every process loads every program from the cache, and the launch takes
  half as long.
