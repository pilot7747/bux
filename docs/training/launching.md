# Launching

`bux launch` trains a file's runs to their end, however often you stop it and start
it again. It is the command for a cluster, and the Docker image's entrypoint. This
page covers how it gives runs GPUs, what it records, how it stops, array jobs, many
hosts, and resuming. For a laptop or a debugger, `bux train` ([Running](running.md))
trains in its own process.

```sh
bux launch examples/experiments/attention_ablation.bux             # all 6 runs, one GPU each
bux launch examples/experiments/attention_ablation.bux --index 3   # the fourth, in an array job
```

`bux launch` takes the same arguments as [`bux train`](running.md#bux-train), plus
`--workers N`, the number of processes that prepare data (one per core by default).

| | `bux train` | `bux launch` |
|---|---|---|
| Runs of a sweep | one after another, in this process | side by side, one worker process per run, each on its own GPUs |
| A run that is done | trained again | skipped |
| Its status | none | `status.json` in its directory |
| SIGTERM, Ctrl-C | stop at once, losing the step | stop after the step, saving it with a `checkpoint` setting |
| A failed run | stops the command | leaves the others training |
| Many hosts | no | one run, under `srun`, `mpirun` or `JAX_*` variables |

## What it does

1. It skips the runs whose `status.json` says `done`.
2. It checks that each run's mesh fits the GPUs it has, and that the tool its
   `profile` names is installed. A run that fails either is recorded as failed, at
   once.
3. It prepares the datasets the runs read, once each ([Data](../data/index.md)).
4. A run alone trains in the launcher's own process, on the first GPUs it needs. A
   sweep trains in workers, one process per run. A run whose `profile` names nsys or
   ncu trains under that tool: the launcher restarts itself under the tool first for
   a run alone, and starts a sweep's worker under it
   ([Profiling](../performance/profiling.md#nsight-systems-and-nsight-compute)).
5. A sweep ends with a table of its runs.

## GPUs and workers

A run needs the GPUs of its `mesh`: the product of its axes' sizes, where an unset
`data` means every visible GPU for a run alone, and one GPU in a sweep. `bux plan`
lists them. A run that needs more GPUs than there are fails before training. On a
node of 8 GPUs:

```console
$ bux launch examples/experiments/smoke.bux 'mesh = mesh(fsdp=16)'
smoke: failed before training: mesh: needs 16 GPUs, but there are 8
```

The visible GPUs are those `CUDA_VISIBLE_DEVICES` lists, or else every GPU that
`nvidia-smi -L` finds. A sweep trains in workers, one process per run, each on a group
of free GPUs that it gets in `CUDA_VISIBLE_DEVICES`. The first pending run that fits
starts first, and a group frees up when its run ends. Without GPUs, as on a laptop,
the runs train one at a time on the default device.

A worker's output goes to the console and to its run's `train.log`. A run that fails
or runs out of memory leaves the others training, and the launcher points at its log:
`<run>: failed (exit 1); its output is in <its directory>/train.log`.

A worker trains its run from the `run.json` the launcher wrote: the run as the
launcher compiled it at the start. Editing the file while a sweep trains changes none
of its runs.

## Status and the summary

Each run's directory keeps a `status.json`: whether the run is `running`, `done`,
`failed` or `stopped`, on which host and GPUs, since when, at what step, and its
latest metrics (all except the per-layer ones, which stay in `metrics.jsonl`):

```json
{
  "state": "done",
  "host": "laptop",
  "pid": 6963,
  "gpus": null,
  "started": "2026-10-26T22:05:19+00:00",
  "ended": "2026-10-26T22:05:30+00:00",
  "error": null,
  "step": 3000,
  "metrics": {
    "grad_norm": 0.03431686386466026,
    "loss": 6.238302230834961,
    "lr": 0.0,
    "tokens": 1536000,
    "tokens_per_s": 78700.21272668002,
    "time/step_ms": 6.505700331181288,
    "time/data_ms": 0.009188344702124596,
    "time/wait_ms": 0.4499139990657568,
    "time/hooks_ms": 0.35634555481374264,
    "time/launch_ms": 5.287754703313112,
    "time/compile_s": 0.5438455422408879
  }
}
```

A run that is done is not trained again, and one that is not resumes from its latest
checkpoint. So launching an experiment again, after a preemption, a crash or a fix,
trains what is left of it. A sweep ends with a table of its runs; here, a sweep of the
smoke run on a laptop:

```console
$ bux launch examples/experiments/smoke.bux 'optimizer.lr = 1e-3 | 3e-3'
training 2 runs, one at a time
smoke[optimizer.lr=1e-3]: training on cpu of laptop with jax 0.12.0
smoke[optimizer.lr=1e-3]: runs/smoke_optimizer.lr=1e-3-bcdf2abce8, on 1 × cpu
...
smoke[optimizer.lr=1e-3] step 10/50  loss 6.2462  lr 1.00e-03  grad 0.623  8,526 tok/s  ...
...
smoke[optimizer.lr=3e-3] step 50/50  loss 6.2633  lr 0.00e+00  grad 0.418  76,501 tok/s  ...
smoke[optimizer.lr=3e-3]: done
2 runs: 2 done
state  step   loss    eval loss  run
done   50/50  6.2594  -          smoke[optimizer.lr=1e-3]
done   50/50  6.2633  -          smoke[optimizer.lr=3e-3]
$ bux launch examples/experiments/smoke.bux 'optimizer.lr = 1e-3 | 3e-3'
2 of 2 runs done already
```

A sweep stopped partway shows each run where it stopped:

```text
3 runs: 2 done, 1 stopped
state    step       loss    eval loss  run
done     3814/3814  3.2141  3.2577     width_sweep[s]
done     3814/3814  3.0836  3.1302     width_sweep[m]
stopped  2731/3814  3.0412  3.0815     width_sweep[l]
```

`bux results FILE` prints the table again at any time, without training. A sweep that
sets an `objective` gets a column for it ([The objective](searching.md#the-objective)).

## Signals

SIGTERM or Ctrl-C (SIGINT) stops training after the current step, and saves that step
if the run has a `checkpoint` setting. A second signal, a second or more after the
first, stops it at once. One that comes sooner is taken as the first delivered twice,
to the launcher and to a parent that passes it on (such as `uv run` or `srun`), and is
ignored. A signal before training starts stops at once. A sweep's launcher passes
signals on to its workers, and starts no new run after one.

The launcher exits with:

| Code | When |
|------|------|
| `0` | every run is done |
| `1` | a run failed, before or during training; many processes were given a sweep; or none of a search's trials has an objective that is a number |
| `128 +` the signal's number | a signal stopped it: 143 for SIGTERM, 130 for SIGINT |

A run stopped without a `checkpoint` setting starts over when launched again, and the
launcher says so.

## The compile cache

`bux train`, `bux launch` and a sweep's workers keep what they compile in JAX's
persistent cache, in `$XDG_CACHE_HOME/bux/jax` (`~/.cache/bux/jax`), and on GPUs the
picks of XLA's GEMM autotuner with it. A run launched again, or the next run of a
sweep with the same program, loads its step instead of compiling it: 0.5 s instead of
29 s for an MoE on 8 B300s. A run that differs in a flag still finds the autotuner's
picks (17 s).

- `JAX_COMPILATION_CACHE_DIR` puts the cache elsewhere, and set empty turns it off.
- bux puts its own XLA flags before `XLA_FLAGS`, and a flag that `XLA_FLAGS` already
  sets keeps its value ([Environment variables](../reference/environment.md)).
- In a job of many hosts, only the first process writes the cache, so put it on a
  filesystem they all mount: bux turns off a cache that is each host's own, and says
  so ([The compile cache](../parallelism/multi-node.md#the-compile-cache)).

## Searches

A file with dimensions is a search for each of its runs
([Searching hyperparameters](searching.md)). `bux launch` trains its trials as it
trains a sweep, one worker per trial on the GPUs its mesh needs, and proposes each
trial as GPUs free up, from the trials done and those still training. A search
launched again goes on where it stopped: its trials are in its own directory, and a
trial that is done is read, not trained again. A search trains many runs, so it
launches on one host; in an array job, `--index` picks one of a sweep's searches.

## Array jobs

`--index N` trains only the run at N, from 0, in the order `bux plan` lists them. So
an array job trains a sweep with one run per task:

::::{tab-set}

:::{tab-item} Slurm
```sh
#SBATCH --array=0-5
bux launch examples/experiments/attention_ablation.bux --index $SLURM_ARRAY_TASK_ID
```
:::

:::{tab-item} Kubernetes
An indexed Job gives each pod its index in `JOB_COMPLETION_INDEX`, which the
container's arguments expand as `$(JOB_COMPLETION_INDEX)`:

```yaml
spec:
  completionMode: Indexed
  completions: 6
  template:
    spec:
      containers:
        - name: bux
          args: [bux, launch, examples/experiments/attention_ablation.bux,
                 --index, $(JOB_COMPLETION_INDEX)]
```
:::

::::

`--index` past the last run is an error: `--index 6, but the file makes 6 runs`.

:::{tip}
Tasks that read the same documents each prepare them. That is safe but wasted work:
the first to finish keeps its dataset, and the others use it. Run `bux prepare` once
beforehand.
:::

## Many hosts

A run alone trains on many hosts when the environment says how many processes there
are:

- under Slurm's `srun`, or Open MPI's `mpirun`;
- or with `JAX_COORDINATOR_ADDRESS` (process 0's `host:port`), `JAX_NUM_PROCESSES`
  and `JAX_PROCESS_ID` set on each process.

Every process runs the same command, and they join through `jax.distributed`. With
`JAX_NUM_PROCESSES` set, those three variables say everything, and no cluster is
detected; otherwise JAX reads Slurm's or Open MPI's environment. Under Slurm and Open
MPI, JAX gives each process the GPU of its local rank, so start one task per GPU:

```sh
srun --nodes=4 --ntasks-per-node=8 bux launch speedrun.bux   # 32 GPUs, one mesh
```

- Only the first process writes the console, `metrics.jsonl`, `status.json`, the
  trackers and the run's files. Every process writes its part of a checkpoint.
- Each process reads only the rows of a batch that its devices hold
  ([Reading prepared data](../data/pipeline.md#reading-prepared-data)).
- A signal to any process stops them all after the same step.
- A sweep does not train on many hosts: give each run its own job, with `--index`.
  Launched with several runs, the launcher says
  `8 processes train one run, not 6: use --index` and exits with 1. A search says
  `a search trains many runs, each on one host: launch it on one`.

How the mesh lays out over nodes is in [Many nodes](../parallelism/multi-node.md).

## Shared filesystems

In a job of many processes, three folders must be the same folder on every host:

| Folder | Why | What checks it |
|--------|-----|----------------|
| `--out`, the run directories | every process saves its part of a checkpoint into the run's directory | the checkpoint hook, before the first step: it fails at once if one process does not see the files the others write there |
| `--cache`, the prepared datasets | every process reads the rows it holds from them | |
| `JAX_COMPILATION_CACHE_DIR` | the processes must all find a program, or all compile it | `bux launch`, as they join: it turns the cache off where it is each host's own |

So `--out` must be a folder every node sees, not a node's local disk or `/dev/shm`.
The checkpoint hook's error says so:

```text
ValueError: the 16 processes of this job do not share <out>/<run>/checkpoints: one sees
8 of the files they wrote there. Each saves its part of a checkpoint into it, so put the
run's directory (--out) where every node sees it
```

Without this check, orbax waits at the first save for the parts it cannot see: on
two nodes with separate folders, it hung for twenty minutes before it timed out.

## Resuming

A run launched again loads its latest checkpoint and trains on as if it had never
stopped. Here the smoke run, stretched to 3000 steps with a checkpoint every 500, is
stopped by SIGTERM just after step 1500 and launched again. It saves the step it stops
at:

```console
$ bux launch examples/experiments/smoke.bux steps=3000 'log = log(every=250)' 'checkpoint = checkpoint(every=500)'
smoke: training on cpu of laptop with jax 0.12.0
...
smoke step  250/3000  loss 6.2383  lr 3.00e-03  grad 0.154  57,512 tok/s  ...
...
smoke step 1500/3000  loss 6.2417  lr 3.00e-03  grad 0.053  76,424 tok/s  ...
smoke: stopped at step 1528 by SIGTERM; launching it again resumes it from its latest checkpoint
$ bux launch examples/experiments/smoke.bux steps=3000 'log = log(every=250)' 'checkpoint = checkpoint(every=500)'
smoke: training on cpu of laptop with jax 0.12.0
smoke: restored in 0.1s, resuming at step 1528
...
smoke step 1750/3000  loss 6.2386  lr 3.00e-03  grad 0.020  56,132 tok/s  ...
...
smoke step 3000/3000  loss 6.2383  lr 0.00e+00  grad 0.034  78,700 tok/s  ...
smoke: done
```

### What makes a resume exact

- The checkpoint holds the model's state and the optimizer's, including random keys
  and an MoE's router bias.
- Batch `i` depends on the seed and `i` alone, so a run that resumes at step `i` reads
  what it would have read ([Mixtures](../data/sources.md#mixtures)).
- `metrics.jsonl` drops the lines of any steps the run takes again, such as those a
  crash after the checkpoint left behind.
- A run's mesh, layout and methods are not in its id, so it also resumes on another
  mesh or another number of nodes, with orbax reading each array onto the shardings
  the new mesh gives it.

### Verified on B300s

On one node of 8 B300s, on two nodes and on four (2026-10-24), with GEMM autotuning
off and deterministic ops (`--xla_gpu_autotune_level=0
--xla_gpu_deterministic_ops=true`): a 0.2B Llama of 8 layers and a 0.4B MoE (16
experts, top-2) on packed documents, in bfloat16 with donation, 8 rows of 2048 tokens
per GPU, 30 steps with a checkpoint every 10. Each was stopped through `bux launch` by
SIGTERM once step 15 was logged, or by SIGKILL as a crash would, and launched again.
Compared with the run trained in one go on its mesh:

| Saved on | Resumed on | At | Restore | Losses after it |
|----------|------------|----|---------|-----------------|
| 1 GPU: dense, MXFP8, NVFP4 | the same | 16-17 | 1.4-1.9s | equal, bit for bit |
| 1 GPU, stopped twice | the same | 9, 23 | 1.2-1.4s | equal |
| 1 GPU, SIGKILL after step 15 | the same | 10 | 1.5s | equal |
| fsdp=8: AdamW, Muon, MXFP8 | the same | 16 | 1.7-2.1s | equal |
| fsdp=8 | data=2, fsdp=4 | 16 | 1.9s | within 1e-4 (4e-3) |
| fsdp=8 | tensor=2, fsdp=4 | 16 | 1.9s | within 6e-4 (3e-3) |
| MoE, expert=8, `ragged_all_to_all` | the same | 16 | 2.3s | equal |
| MoE, expert=8, `ragged_all_to_all` | fsdp=8, `sorted` | 16 | 2.5s | within 7e-2 (1.1, as it spikes) |
| 1f1b, stage=2 x fsdp=4, even | the same; one stage | 16 | 1.9s | equal; within 5e-4 (2e-2) |
| 1f1b, stage=2 x fsdp=4, `stages=[5, 3]` | the same; one stage | 16 | 2.0s | equal; fails at once |
| two nodes, fsdp=16 | the same | 21 | 1.7s | equal |
| two nodes, fsdp=16 | one node, fsdp=8 | 21 | 2.1s | within 2e-5 (1e-3) |
| one node, fsdp=8 | two nodes, fsdp=16 | 16 | 2.0s | within 7e-5 (1e-3) |
| four nodes, fsdp=32, SIGKILL twice | the same, twice | 10, 30 | 1.9s | equal |
| four nodes, fsdp=32 | two nodes, fsdp=16 | 10 | 2.1s | within 7e-4 (1e-3) |

In brackets: how far apart the losses of runs trained in one go on the two meshes
are.

- On two nodes, the SIGTERM went to the second node's process alone, and both stopped
  at the same step.
- When a process dies (SIGKILL), the others exit by themselves once the coordination
  service misses its heartbeats (`Task 2 heartbeat timeout`): 70 s later on four
  nodes, or 9-10 s later when the dead process is the first, which runs that service.
  Launched again, the run resumed from its latest checkpoint each time.

### When a resume cannot work

**A save that a crash left half written** (`20.orbax-checkpoint-tmp`) is deleted when
the run starts again, and the run saves again only past that step: saving the same
step into a folder of the same name hung on four nodes, where the other nodes still
saw the deleted one.

**A checkpoint that does not hold the run's state** fails before it is read, and the
error says what changed: the arrays' paths, dtypes and shapes, and the order of a
pipeline's layers in them. Three methods change them, so a run must resume with the
values it was saved with: `methods.repeat` (`scan` stacks a repeat's passes into one
array), a circular pipeline's `rounds` (one stack per round), and an uneven
pipeline's `stages` (each stage's passes, then copies of its first):

```text
ValueError: the checkpoint of step 16 does not hold this run's state: it was saved
with methods.stages = [5, 3], and this run has none. A run resumes with the methods it
was saved with where they lay out its state: repeat (unroll or scan), a circular
pipeline's rounds and an uneven one's stages
```

An even `gpipe` or `1f1b` pipeline keeps its layers in the order of `repeat=scan`, so
it resumes on any number of stages, or on none
([Resuming on another pipeline](../parallelism/pipelines.md#resuming-on-another-pipeline)).

**A run directory that each node sees alone** fails at once, before the first save
([Shared filesystems](#shared-filesystems)).
