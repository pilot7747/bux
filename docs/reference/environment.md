# Environment variables

This page lists every environment variable that bux reads or sets, and the ones the
guides tell you to set for JAX, XLA, NCCL and launching on many hosts. Each table
gives what the variable does, its default, and where it is read. You need this page
when a run behaves differently on one machine than on another, when you launch on
many hosts, or when you want a run that is reproducible to the bit.

```sh
# Train on 8 CPU devices of a laptop, as the parallelism examples do
XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
    bux train examples/experiments/parallel/laptop.bux

# Debug a kernel: every bux kernel falls back to XLA
BUX_KERNELS=off bux train examples/experiments/smoke.bux

# Keep the compile cache somewhere every host of a job mounts
JAX_COMPILATION_CACHE_DIR=/shared/bux-cache bux launch speedrun.bux
```

:::{note}
`bux train` and `bux launch` (and each worker of a sweep) call
{py:func}`bux.runtime.configure` before anything starts a backend. It sets the
variables marked *set by bux* below, each only where the environment does not set it
already, so a value you export always wins. A script that trains calls
`configure()` itself; `bux check`, `plan`, `prepare`, `render`, `fmt` and `json`
do not.
:::

## bux

| Variable | What it does | Default | Read in |
|----------|--------------|---------|---------|
| `BUX_PATH` | More folders to search for the modules that `use` names, after the `path` list under `[tool.bux]` in `pyproject.toml`. Separated as in `PATH` (`:`, or `;` on Windows); subfolders are searched too. See [Modules](../language/modules.md). | unset | {py:mod}`bux.lang.modules` |
| `BUX_KERNELS` | `off` makes every bux kernel fall back to XLA for the whole process, as {py:func}`bux.kernels.off` does for a block: to debug a kernel, or to count the FLOPs one hides. Read once, when `bux.kernels` is imported. See [Kernels](../performance/kernels.md). | unset: kernels run on GPUs | {py:mod}`bux.kernels` |
| `BUX_PROFILER` | The tool a process runs under already, `nsys` or `ncu`: `bux launch` then runs it under no tool of its own, and {py:func}`~bux.runtime.configure` turns off XLA's command buffers and, under ncu, its pool's preallocation ({py:data}`bux.train.profilers.XLA_FLAGS`, {py:data}`~bux.train.profilers.VARIABLES`). Set it yourself to run under a command of your own, as `BUX_PROFILER=nsys nsys profile --capture-range=cudaProfilerApi ... bux launch run.bux`. See [Profiling](../performance/profiling.md#nsight-systems-and-nsight-compute). | *set by bux* in a process it runs under a `profile` setting's `nsys` or `ncu` | {py:mod}`bux.train.profilers`, {py:mod}`bux.runtime` |
| `BUX_SHARD` | `i/n` keeps the tests whose id hashes to shard `i` of `n`, so that `n` runners share the suite. CI runs four, `0/4` to `3/4`: `BUX_SHARD=1/4 make test` runs what its second runner does. | unset: every test | `tests/conftest.py` |

## JAX and XLA

| Variable | What it does | Default | Read in |
|----------|--------------|---------|---------|
| `XLA_FLAGS` | XLA's flags. bux puts its own first ({py:data}`bux.runtime.XLA_FLAGS`), and a flag you set keeps your value. With `--xla_gpu_deterministic_ops=true`, attention's backward runs by tokamax's kernels where it would run by cuDNN's, and FlashAttention-4's orders its adds. A run's `compiler` setting goes over these flags for its training step alone. | *set by bux*: `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false` before yours, and under nsys or ncu `--xla_gpu_enable_command_buffer=` | {py:mod}`bux.runtime`, {py:mod}`bux.parallel.flash`, {py:mod}`bux.parallel.fa4` |
| `JAX_PLATFORMS` | The backend JAX uses. `cpu` trains on the CPU of a machine with GPUs, runs the bux image under Docker on a Mac, and is what CI and the tests use. | JAX's: a GPU if there is one | JAX |
| `JAX_COMPILATION_CACHE_DIR` | Where JAX's persistent cache keeps compiled programs, and on GPUs the picks of XLA's GEMM autotuner, so a launch after the first compiles little. Set it empty to turn the cache off. In a job of many processes, put it on a filesystem every host mounts: bux turns off a cache that is each host's own. | *set by bux*: `$XDG_CACHE_HOME/bux/jax` | {py:mod}`bux.runtime` |
| `XDG_CACHE_HOME` | The base of the default compile cache. | `~/.cache`, so the cache is `~/.cache/bux/jax` | {py:func}`bux.runtime.cache_dir` |
| `XLA_CLIENT_MEM_FRACTION` | The share of a GPU's memory that XLA's pool takes. JAX's default is 0.75; bux sets 0.95, which leaves room for the CUDA context and NCCL's buffers beside the pool. | *set by bux*: `0.95`, unless it or `XLA_PYTHON_CLIENT_MEM_FRACTION` is set | {py:mod}`bux.runtime` |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` makes XLA's pool take memory as it needs it, not its share of the GPU at once. ncu saves and restores a kernel's memory around each pass of its replay: with the whole pool taken, a kernel of the smoke run took 3 minutes. | *set by bux*: `false` under `profile(tool=ncu)` | {py:mod}`bux.runtime` |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | The older name of `XLA_CLIENT_MEM_FRACTION`. If you set it, bux sets neither. | unset | {py:mod}`bux.runtime` |

The guides use a few values of `XLA_FLAGS` on purpose:

| Value | What it is for |
|-------|----------------|
| `--xla_force_host_platform_device_count=8` | 8 CPU devices on one machine, to try a mesh on a laptop ([Parallelism](../parallelism/index.md)); the tests use it too |
| `--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true` | Runs whose losses agree to the bit: the autotuner picks kernels by timing them, and some kernels add in no fixed order. Pin both before you call a small change of a loss a regression ([Benchmarks](../performance/benchmarks.md)). |
| `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false` | Without it, expert parallelism's ragged all-to-all hangs on B300s on jax 0.12. bux adds it, so you need it only in a script that does not call {py:func}`~bux.runtime.configure`. |

:::{tip}
`bux launch` and a sweep's next run load a step from the cache instead of compiling
it: 0.5 s instead of 29 s for a MoE on 8 B300s, and 17 s for a run that differs in a
flag, which still finds the autotuner's picks.
:::

## NCCL and cuDNN

| Variable | What it does | Default | Read in |
|----------|--------------|---------|---------|
| `NCCL_GIN_TYPE` | How NCCL EP (`methods(moe=nccl_ep)`) reaches other nodes. bux sets `2`, GIN by a CPU proxy: on two B300 nodes, GDAKI, NCCL's default, took 3x as long for an EP layer's exchanges. | *set by bux*: `2` | {py:mod}`bux.runtime` |
| `NCCL_EP_TOKENS_PER_CHUNK` | The tokens NCCL EP sends a chunk between nodes. Chunks of 64 took 1.3x as long as chunks of 256. | *set by bux*: `256` | {py:mod}`bux.runtime` |
| `NCCL_DEBUG` | `INFO` makes NCCL log which network it uses: look for `Using network IB`, not `Socket`. On two B300 nodes, an all-reduce of 64 MiB a GPU across the nodes ran at a bus bandwidth of 1 GB/s over TCP sockets, and 25 GB/s over InfiniBand. | unset | NCCL |
| `CUDNNFE_CLUSTER_OVERLAP_MARGIN` | Pairs of SMs that the experts' cuDNN kernels (`grouped_cudnn`) leave to kernels that run beside them, such as FSDP's collectives, as cuDNN's fused kernels do. In Qwen3-30B-A3B's step at Megatron-Bridge's batch, on one node of 8 B300s, 8 gained 0.3% and 16 gained 1.1%, and 32 lost 2%, so bux leaves it at 0. | `0` | {py:mod}`bux.architecture.experts.grouped_cudnn` |

## Launching on many hosts

A run alone trains on many hosts when the environment says how many processes there
are: under Slurm's `srun`, under Open MPI's `mpirun`, or with the three `JAX_`
variables below set on every host. Every process runs the same command and they join
through `jax.distributed` ({py:func}`bux.launch.join`). See
[Multi-node](../parallelism/multi-node.md) and [Launching](../training/launching.md).

```sh
# On each of two hosts, with JAX_PROCESS_ID=1 on the second
JAX_COORDINATOR_ADDRESS=10.0.0.1:12355 JAX_NUM_PROCESSES=2 JAX_PROCESS_ID=0 \
JAX_COMPILATION_CACHE_DIR=/shared/bux-cache \
    bux launch bench.bux 'mesh = mesh(data=2, fsdp=8)' --out /shared/runs
```

| Variable | What it does | Default | Read in |
|----------|--------------|---------|---------|
| `JAX_COORDINATOR_ADDRESS` | The first process's `host:port`, where the others join. `scripts/benchmark.py` adds a run's index to the port, so that no host joins the last run's coordinator while it closes. | unset | JAX; `scripts/benchmark.py` |
| `JAX_NUM_PROCESSES` | How many processes train one run. With it set, bux joins with it and `JAX_PROCESS_ID`, and detects no cluster. It wins over Slurm's and Open MPI's counts. | unset: one process | {py:func}`bux.launch.processes`, {py:func}`bux.launch.join` |
| `JAX_PROCESS_ID` | This process's index, from 0; needed with `JAX_NUM_PROCESSES`. Process 0 alone writes the console, `status.json` and the run's files. | unset | {py:func}`bux.launch.join`, `scripts/benchmark.py` |
| `SLURM_STEP_NUM_TASKS` | Set by `srun`: the processes of a run, when `JAX_NUM_PROCESSES` is not set. JAX then reads the rest of Slurm's environment, and gives each process the GPU of its local rank, so start a task a GPU. | unset | {py:func}`bux.launch.processes`, JAX |
| `OMPI_COMM_WORLD_SIZE` | Set by `mpirun`: the same, for Open MPI. | unset | {py:func}`bux.launch.processes`, JAX |
| `SLURM_PROCID`, `OMPI_COMM_WORLD_RANK` | A host's rank, which `scripts/benchmark.py` names its log by when `JAX_PROCESS_ID` is not set. | unset | `scripts/benchmark.py` |
| `CUDA_VISIBLE_DEVICES` | The GPUs a process sees. If it is set, `bux launch` shares out the GPUs it lists, and otherwise every GPU `nvidia-smi -L` finds. It gives each worker of a sweep its group of GPUs through it, and a run alone the first GPUs its mesh needs. `scripts/benchmark.py` does the same for its workers. | unset: every GPU | {py:mod}`bux.launch` |

:::{warning}
In a job of many processes, all of them must find a program in JAX's cache, or all
compile it: XLA's autotuner splits its work among the processes that compile, and one
that compiles while another loads waits a day. Only the first process writes the
cache, so bux keeps it only where it is one folder for all of them
({py:func}`bux.runtime.agree_on_cache`), and says so when it turns it off. For the
same reason bux leaves JAX's `jax_share_binary_between_hosts` option off.
:::

### Array jobs

bux reads no array-job variable itself. Pass the task's index to `--index`, which
trains the run at that index, from 0, in the order `bux plan` lists them:

| Variable | Set by | Use |
|----------|--------|-----|
| `SLURM_ARRAY_TASK_ID` | a Slurm array job | `bux launch sweep.bux --index $SLURM_ARRAY_TASK_ID` |
| `JOB_COMPLETION_INDEX` | an indexed Kubernetes Job | `bux launch sweep.bux --index $(JOB_COMPLETION_INDEX)` |

Tasks that read the same documents each prepare them, which is safe but wasted work:
run `bux prepare sweep.bux` once beforehand.

## Scripts and tests

| Variable | Where | What it does |
|----------|-------|--------------|
| `PYTHONPATH` | `scripts/benchmark.py` | `--tree PATH` sets it to `PATH/src` for the workers, so they train with another checkout's bux in this one's environment. |
| `JAX_PLATFORMS` | `tests/conftest.py` | `cpu`, unless set. |
| `JAX_COMPILATION_CACHE_DIR` | `tests/conftest.py` | Empty, so the tests use no compile cache, unless set. |
| `XLA_FLAGS` | `tests/conftest.py` | bux's flags and `--xla_force_host_platform_device_count=8`, added to yours. |

The Docker image sets uv's variables for its own layout: `UV_PROJECT_ENVIRONMENT` and
`VIRTUAL_ENV` (`/opt/venv`), `UV_PYTHON_INSTALL_DIR` (`/opt/python`),
`UV_COMPILE_BYTECODE=1`, `UV_LINK_MODE=copy`, and `UV_NO_SYNC=1`, without which
`uv run` would sync back to the default dependencies and drop the `cuda` extra.
