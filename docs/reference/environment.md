# Environment variables

Every environment variable bux reads or sets, plus the JAX, XLA and NCCL variables
the guides tell you to set. Each table gives what a variable does, its default, and
where it is read. Come here when a run behaves differently on two machines, when you
launch on many hosts, or when you need a run that is reproducible to the bit.

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
{py:func}`bux.runtime.configure` before any backend starts. It sets the variables
marked *set by bux* below, each only if the environment doesn't set it already, so a
value you export always wins. A training script should call `configure()` itself;
`bux check`, `plan`, `prepare`, `render`, `fmt` and `json` don't call it.
:::

## bux

| Variable | What it does | Default | Read in |
|----------|--------------|---------|---------|
| `BUX_PATH` | More folders to search for the modules that `use` names, after the `path` list under `[tool.bux]` in `pyproject.toml`. Separated as in `PATH` (`:`, or `;` on Windows); subfolders are searched too. See [Modules](../language/modules.md). | unset | {py:mod}`bux.lang.modules` |
| `BUX_KERNELS` | `off` makes every bux kernel fall back to XLA for the whole process, as {py:func}`bux.kernels.off` does for a block: use it to debug a kernel, or to count the FLOPs a kernel hides from XLA. Read once, when `bux.kernels` is imported. See [Kernels](../performance/kernels.md). | unset: kernels run on GPUs | {py:mod}`bux.kernels` |
| `BUX_PROFILER` | The profiler a process already runs under, `nsys` or `ncu`. `bux launch` then starts no profiler of its own, and {py:func}`~bux.runtime.configure` turns off XLA's command buffers and, under ncu, the pool's preallocation ({py:data}`bux.train.profilers.XLA_FLAGS`, {py:data}`~bux.train.profilers.VARIABLES`). Set it yourself to profile with your own command line, as in `BUX_PROFILER=nsys nsys profile --capture-range=cudaProfilerApi ... bux launch run.bux`. See [Profiling](../performance/profiling.md#nsight-systems-and-nsight-compute). | *set by bux* when it runs a process under `profile(tool=nsys)` or `profile(tool=ncu)` | {py:mod}`bux.train.profilers`, {py:mod}`bux.runtime` |
| `BUX_SHARD` | `i/n` keeps the tests whose id hashes to shard `i` of `n`, so that `n` runners split the suite. CI runs six, `0/6` to `5/6`: `BUX_SHARD=1/6 make test` runs the second runner's share. | unset: every test | `tests/conftest.py` |
| `BUX_VLLM_SPARSE_KERNELS` | Which path the vLLM plugin's sparse attention takes: `triton`, Triton kernels that read vLLM's paged caches in place (for the settings they take: DSA's keys, MSA's and QSA's blocks, `csa`'s attention, heads of up to 256 dims, not MLA's latents), or `torch`, the PyTorch path that gathers the keys first. `scripts/vllm_kernels.py` compares the two on a GPU. Read once, when the plugin loads. See [Sparse attention](../export/vllm.md#sparse-attention). | unset: the kernels in a 16-bit model, PyTorch in float32 | `bux_vllm.kernels` |

## JAX and XLA

| Variable | What it does | Default | Read in |
|----------|--------------|---------|---------|
| `XLA_FLAGS` | XLA's flags. bux puts its own first ({py:data}`bux.runtime.XLA_FLAGS`), and a flag you set keeps your value. With `--xla_gpu_deterministic_ops=true`, attention's backward runs on tokamax's kernels instead of cuDNN's, and FlashAttention-4 fixes the order of its adds. A run's `compiler` setting goes over these flags for its training step alone. | *set by bux*: `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false` before yours, and under nsys or ncu `--xla_gpu_enable_command_buffer=` | {py:mod}`bux.runtime`, {py:mod}`bux.parallel.flash`, {py:mod}`bux.parallel.fa4` |
| `JAX_PLATFORMS` | The backend JAX uses. `cpu` trains on the CPU of a machine with GPUs, runs the bux image under Docker on a Mac, and is what CI and the tests use. | JAX's: a GPU if there is one | JAX |
| `JAX_COMPILATION_CACHE_DIR` | Where JAX's persistent cache keeps compiled programs (and, on GPUs, the picks of XLA's GEMM autotuner), so later launches compile little. Set it empty to turn the cache off. In a job of many processes, put it on a filesystem every host mounts: bux turns off a cache that is local to each host. | *set by bux*: `$XDG_CACHE_HOME/bux/jax` | {py:mod}`bux.runtime` |
| `XDG_CACHE_HOME` | The base of the default compile cache. | `~/.cache`, so the cache is `~/.cache/bux/jax` | {py:func}`bux.runtime.cache_dir` |
| `XLA_CLIENT_MEM_FRACTION` | The share of a GPU's memory that XLA's pool takes. JAX's default is 0.75; bux sets 0.95, which still leaves room for the CUDA context and NCCL's buffers. | *set by bux*: `0.95`, unless it or `XLA_PYTHON_CLIENT_MEM_FRACTION` is set | {py:mod}`bux.runtime` |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` makes XLA's pool allocate memory as needed instead of taking its whole share up front. ncu saves and restores a kernel's memory around each replay pass: with the whole pool allocated, one kernel of the smoke run took 3 minutes to profile. | *set by bux*: `false` under `profile(tool=ncu)` | {py:mod}`bux.runtime` |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | The older name of `XLA_CLIENT_MEM_FRACTION`. If you set it, bux sets neither. | unset | {py:mod}`bux.runtime` |

The guides use a few values of `XLA_FLAGS` on purpose:

| Value | What it is for |
|-------|----------------|
| `--xla_force_host_platform_device_count=8` | 8 CPU devices on one machine, to try a mesh on a laptop ([Parallelism](../parallelism/index.md)); the tests use it too |
| `--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true` | Runs whose losses agree to the bit: the autotuner picks kernels by timing them, and some kernels add in no fixed order. Pin both before you call a small change in loss a regression ([Benchmarks](../performance/benchmarks.md)). |
| `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false` | Without it, expert parallelism's ragged all-to-all hangs on B300s on jax 0.12. bux adds it, so you need it only in a script that does not call {py:func}`~bux.runtime.configure`. |

:::{tip}
A relaunch, or a sweep's next run, loads the step from the cache instead of compiling
it: 0.5 s instead of 29 s for an MoE on 8 B300s, and 17 s for a run that differs by a
flag (it still reuses the autotuner's picks).
:::

## NCCL and cuDNN

| Variable | What it does | Default | Read in |
|----------|--------------|---------|---------|
| `NCCL_GIN_TYPE` | How NCCL EP (`methods(moe=nccl_ep)`) reaches other nodes. bux sets `2`, GIN through a CPU proxy: on two B300 nodes, NCCL's default (GDAKI) took 3x as long for an EP layer's exchanges. | *set by bux*: `2` | {py:mod}`bux.runtime` |
| `NCCL_EP_TOKENS_PER_CHUNK` | Tokens per chunk that NCCL EP sends between nodes. Chunks of 64 took 1.3x as long as chunks of 256. | *set by bux*: `256` | {py:mod}`bux.runtime` |
| `NCCL_DEBUG` | `INFO` makes NCCL log which network it uses: look for `Using network IB`, not `Socket`. On two B300 nodes, a cross-node all-reduce of 64 MiB per GPU reached a bus bandwidth of 1 GB/s over TCP sockets and 25 GB/s over InfiniBand. | unset | NCCL |
| `CUDNNFE_CLUSTER_OVERLAP_MARGIN` | Pairs of SMs that the experts' cuDNN kernels (`grouped_cudnn`) leave free for kernels running beside them, such as FSDP's collectives. On Qwen3-30B-A3B at Megatron-Bridge's batch on 8 B300s, 8 gained 0.3%, 16 gained 1.1% and 32 lost 2%, so bux leaves it at 0. | `0` | {py:mod}`bux.architecture.experts.grouped_cudnn` |

## Launching on many hosts

A run trains on many hosts when the environment says how many processes there are:
under Slurm's `srun`, under Open MPI's `mpirun`, or with the three `JAX_` variables
below set on every host. Every process runs the same command, and they join through
`jax.distributed` ({py:func}`bux.launch.join`). See
[Multi-node](../parallelism/multi-node.md) and [Launching](../training/launching.md).

```sh
# On each of two hosts, with JAX_PROCESS_ID=1 on the second
JAX_COORDINATOR_ADDRESS=10.0.0.1:12355 JAX_NUM_PROCESSES=2 JAX_PROCESS_ID=0 \
JAX_COMPILATION_CACHE_DIR=/shared/bux-cache \
    bux launch bench.bux 'mesh = mesh(data=2, fsdp=8)' --out /shared/runs
```

| Variable | What it does | Default | Read in |
|----------|--------------|---------|---------|
| `JAX_COORDINATOR_ADDRESS` | The first process's `host:port`, where the others join. `scripts/benchmark.py` adds a run's index to the port, so that no host joins the previous run's coordinator while it shuts down. | unset | JAX; `scripts/benchmark.py` |
| `JAX_NUM_PROCESSES` | How many processes train one run. When it is set, bux joins with it and `JAX_PROCESS_ID` and skips cluster detection; it takes precedence over Slurm's and Open MPI's counts. | unset: one process | {py:func}`bux.launch.processes`, {py:func}`bux.launch.join` |
| `JAX_PROCESS_ID` | This process's index, from 0; needed with `JAX_NUM_PROCESSES`. Only process 0 writes the console, `status.json` and the run's files. | unset | {py:func}`bux.launch.join`, `scripts/benchmark.py` |
| `SLURM_STEP_NUM_TASKS` | Set by `srun`: the number of processes of a run, when `JAX_NUM_PROCESSES` is not set. JAX then reads the rest of Slurm's environment and gives each process the GPU of its local rank, so start one task per GPU. | unset | {py:func}`bux.launch.processes`, JAX |
| `OMPI_COMM_WORLD_SIZE` | Set by `mpirun`: the same, for Open MPI. | unset | {py:func}`bux.launch.processes`, JAX |
| `SLURM_PROCID`, `OMPI_COMM_WORLD_RANK` | A host's rank; `scripts/benchmark.py` names its log after it when `JAX_PROCESS_ID` is not set. | unset | `scripts/benchmark.py` |
| `CUDA_VISIBLE_DEVICES` | The GPUs a process sees. `bux launch` shares out the GPUs it lists, or, if it is unset, every GPU `nvidia-smi -L` finds. Through it, the launcher gives each worker of a sweep its own group of GPUs, and a single run the first GPUs its mesh needs. `scripts/benchmark.py` does the same for its workers. | unset: every GPU | {py:mod}`bux.launch` |

:::{warning}
In a job of many processes, either all of them find a program in JAX's cache or all
of them compile it. XLA's autotuner splits its work among the compiling processes, so
one that compiles while another loads from the cache can stall for a day. Only the
first process writes the cache, so bux keeps it only when every process sees the same
folder ({py:func}`bux.runtime.agree_on_cache`), and says so when it turns the cache
off. For the same reason, bux leaves JAX's `jax_share_binary_between_hosts` off.
:::

### Array jobs

bux reads no array-job variable itself. Pass the task's index to `--index`, which
trains the run at that 0-based index, in the order `bux plan` lists them:

| Variable | Set by | Use |
|----------|--------|-----|
| `SLURM_ARRAY_TASK_ID` | a Slurm array job | `bux launch sweep.bux --index $SLURM_ARRAY_TASK_ID` |
| `JOB_COMPLETION_INDEX` | an indexed Kubernetes Job | `bux launch sweep.bux --index $(JOB_COMPLETION_INDEX)` |

Tasks that read the same documents each prepare them, which is safe but wasteful:
run `bux prepare sweep.bux` once beforehand.

## Scripts and tests

| Variable | Where | What it does |
|----------|-------|--------------|
| `PYTHONPATH` | `scripts/benchmark.py` | `--tree PATH` sets it to `PATH/src` for the workers, so they train with another checkout's bux in this environment. |
| `JAX_PLATFORMS` | `tests/conftest.py` | `cpu`, unless set. |
| `JAX_COMPILATION_CACHE_DIR` | `tests/conftest.py` | Empty, so the tests use no compile cache, unless set. |
| `XLA_FLAGS` | `tests/conftest.py` | bux's flags and `--xla_force_host_platform_device_count=8`, added to yours. |

The Docker image sets uv's variables for its own layout: `UV_PROJECT_ENVIRONMENT` and
`VIRTUAL_ENV` (`/opt/venv`), `UV_PYTHON_INSTALL_DIR` (`/opt/python`),
`UV_COMPILE_BYTECODE=1`, `UV_LINK_MODE=copy`, and `UV_NO_SYNC=1` (without it,
`uv run` would sync back to the default dependencies and drop the `cuda` extra).
