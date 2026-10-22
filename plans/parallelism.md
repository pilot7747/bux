# Parallelism: the implementation plan

FSDP, TP (with sequence parallelism), EP, CP and PP in bux. A `.bux` file sets the
device grid and sweeps it like any other setting, and every method that communicates
is a function registered by name, so another one is a function too. Correctness
against one device comes first, and speed second.

## How to run it

Work packages (WPs) run in waves. The WPs of a wave own disjoint files, so each runs
as its own agent in its own worktree, and they merge in the order listed. After a
wave, remove the worktrees and run `make ci` on main.

| Wave | WPs | Needs |
|------|-----|-------|
| 0 | WP0 JAX 0.12 | |
| 1 | WP1 the core: mesh, layout, methods, FSDP | WP0 |
| 2 | WP2 attention TP, WP3 the other layers' TP, WP4 dropless MoE, WP6 launcher and multi-host, WP9 distributed Muon | WP1 |
| 3 | WP5 precision and microbatches; WP7 EP; WP8 CP for attention; WP11 scan over repeats and remat | WP4 for WP7, WP2 for WP8 |
| 4 | WP10 CP for recurrent mixers, WP12 PP | WP3 + WP8, WP11 |

Commit this file before spawning agents: worktrees branch from HEAD. Every WP adds a
section to `docs/parallelism.md`, the one file merged by hand. The brief for each
agent is at the end.

## What we know

Probes on 8 fake CPU devices, with jax 0.11.2 and again with 0.12.0, flax 0.12.10:

| Probe | Result |
|-------|--------|
| FSDP 2 × EP 2 × TP 2 on the unmodified `Model`, from 7 regex rules over parameter paths, AdamW and Muon | Trains; losses match one device to 2e-6; shardings stay put across steps |
| The same, with one `with_sharding_constraint` on the stream in `ResidualBranch` | "Involuntary full rematerialization" warnings 34 → 4, all-gathers 74 → 33 |
| Explicit axis types, same code | Fails at the first op (`ShardingTypeError`) |
| Sharded init: `nnx.eval_shape`, rules, `jax.jit(out_shardings=...)` | The same values as eager init; no parameter is ever whole on one device |
| `nnx.Optimizer` built eagerly from sharded parameters | Its state takes their shardings, for AdamW and for Muon |
| The `Checkpoint` hook: save on 2×2×2, restore onto fsdp=8 | Works unchanged; parameters identical |
| The sequence split 4 ways, `gqa`, `gdn`, `mamba2` | All correct. Attention all-gathers K/V by itself (all-gather CP); convs and shifts exchange halos; scans gather the whole sequence onto every device |
| Muon vs AdamW under FSDP | 227 all-reduces a step vs 30: Newton-Schulz runs on sharded matrices |
| Flax names on parameters (`nnx.with_partitioning`) with its default eager sharding | Without a mesh, creating the parameter fails; with `nnx.use_eager_sharding(False)` names are metadata only |
| Collectives on XLA:CPU | `all_to_all`, `ppermute`, `psum_scatter`, `all_gather` work; `ragged_all_to_all` is unimplemented |
| The test suite on jax 0.12.0 + orbax 0.12.8 | 466 passed, 1 skipped |

The probe scripts are in `/tmp/bux_probe/` on the machine that wrote this plan, if it
still has them. WP1's tests make the same checks.

## Decisions

1. **Auto axes.** Layers stay unaware of sharding: the compiler (Shardy) propagates
   it from parameters and a few constraints. Explicit mode would put `out_sharding=`
   on nearly every einsum of every layer, and MaxText measured no speed difference. A
   method that wants exact control opts in locally (`jax.sharding.explicit_axes`, or a
   `jax.shard_map`). `jax.make_mesh` defaults to Explicit: always pass Auto.
2. **One mesh, six axes, always all of them**, size 1 where unused, in the order
   `data, stage, fsdp, context, expert, tensor`: the slowest links first, so `tensor`
   stays inside a node.
3. **Three settings, none in the run's id.** `mesh` (axis sizes), `layout` (which
   named dimensions split over which axes) and `methods` (implementations by registry
   name) change how a run computes, not what, so `Run.id` leaves them out, as it does
   `log`. A run resumes on another mesh, and a sweep over meshes is still several runs,
   as their names differ. `precision` (WP5) changes the math, and is in the id.
4. **Names on parameters, resolved by bux.** Layers name the dimensions of their
   parameters where they make them (`kernel_init=parallel.named("embed", "heads")`, a
   wrapper around `nnx.with_partitioning`). bux turns off Flax's eager sharding and
   resolves names itself, so names never fail without a mesh and the rules are ours
   to change. The names are a closed set, the fields of `LayoutConfig`.
5. **FSDP is not a name.** After the names resolve, every parameter's largest
   dimension left whole, if the fsdp size divides it, splits over `fsdp`. A layer with
   no names at all trains under FSDP; names only matter for TP and EP.
6. **Activations: one constraint, and a few more.** The stream is constrained at every
   residual branch of the top level; layers add `shard(x, ...)` where they know what a
   dimension means. `shard` does nothing outside `parallel.use`, so one device runs
   exactly as now.
7. **Manual communication only inside methods**: a `jax.shard_map` with
   `axis_names={its axis}`, the other axes staying Auto. What reads the whole batch
   (MoE routing, balance losses, the router bias) stays outside, in the global view,
   so its semantics do not change.
8. **Tests on 8 fake CPU devices**, against one device, and with no involuntary
   rematerialization where the layout is complete. What CPU cannot run
   (`ragged_all_to_all`, tokamax's GPU kernels) has a CPU-testable sibling method, and
   is tested on GPU by a script.

## The contract

WP1 builds this; every other WP codes against it.

### Settings

```
mesh    = mesh(fsdp=8, tensor=2)                  # data takes the devices the others leave
layout  = layout(stream=[context, tensor])        # sequence parallelism
methods = methods(attention=ring, moe=all_to_all, remat=dots)

mesh        = fsdp: mesh(fsdp=16) | tp: mesh(fsdp=8, tensor=2)   # the grid, as a sweep
layout.seq  = context | none
```

In `bux/train/config.py`, each a field of `TrainConfig` with a default and a member of
`Setting`:

```python
Axes = str | tuple[str, ...] | None  # the mesh axes a named dimension splits over


class MeshConfig(Config):
    type: Literal["mesh"] = "mesh"
    data: Positive | None = None  # None: the devices the other axes leave
    stage: Positive = 1
    fsdp: Positive = 1
    context: Positive = 1
    expert: Positive = 1
    tensor: Positive = 1


class LayoutConfig(Config):
    type: Literal["layout"] = "layout"
    batch: Axes = ("data", "fsdp", "expert")  # rows
    seq: Axes = "context"  # positions, inside a layer
    stream: Axes = "context"  # positions of the residual stream, between branches
    embed: Axes = None  # the model's width
    vocab: Axes = "tensor"
    heads: Axes = "tensor"  # query heads, or a mixer's heads
    kv_heads: Axes = "tensor"
    mlp: Axes = "tensor"  # hidden units
    experts: Axes = "expert"
    latent: Axes = None  # low-rank bottlenecks: MLA, LatentMoE
    zero: Literal[1, 3] = 3  # 1: FSDP splits the optimizer state alone


class MethodsConfig(Config):
    type: Literal["methods"] = "methods"
    # Keys of bux.parallel's registries: a new method is a function, not a field.
    attention: str = "xla"
    moe: str = "dense"
    recurrence: str = "scan"
    repeat: str = "unroll"
    remat: str = "none"
    pipeline: str = "gpipe"
    orthogonalize: str = "gspmd"  # how Muon reaches whole matrices
    microbatches: Positive = 1
```

A later WP may add a field it needs, as WP12 does for circular pipelines.

### Python

```python
# bux/parallel/__init__.py, re-exporting its modules
AXES = ("data", "stage", "fsdp", "context", "expert", "tensor")


def mesh(c: MeshConfig) -> Mesh: ...  # Auto axes, all six
def use(
    mesh: MeshConfig | Mesh,
    layout: LayoutConfig = LayoutConfig(),
    methods: MethodsConfig = MethodsConfig(),
) -> ContextManager[Mesh]: ...  # jax.set_mesh, and what spec and method read
def spec(names: Sequence[str | None], shape: tuple[int, ...]) -> P: ...
def shard(x: Shaped[Array, "*s"], *names: str | None) -> Shaped[Array, "*s"]: ...
def stream(x: Shaped[Array, "b t *r"]) -> Shaped[Array, "b t *r"]: ...
def named(*names: str | None, init: Initializer = lecun_normal()) -> Initializer: ...
def shardings(state: nnx.State) -> nnx.State: ...  # a NamedSharding a leaf
def build[M: nnx.Module](make: Callable[[], M]) -> M: ...  # born sharded
def place(*nodes: nnx.Module) -> None: ...  # resharded to `shardings`
def rows[T](batch: T) -> T: ...  # every (b, t) array over batch and seq
def method(kind: str) -> Callable[..., Any]: ...  # the run's, from `kind`'s registry
```

`rows` takes any pytree of `(b, t)` arrays, so that `bux.parallel` never imports
`bux.train.data`. The registries are `bux.train.registry.Registry`s, one per field of
`MethodsConfig`, in `bux/parallel/methods.py`; the WP that wires a call site fixes
its signature and registers the default where the layer lives (`"dense"` in
`moe.py`). Outside `use`, `method` returns the default.

### How names resolve

1. Dimension by dimension, each name's axes, less the axes of size 1 and those an
   earlier dimension took. If their sizes do not divide the dimension, it stays
   whole: GQA's 2 kv heads under tensor=4 are replicated.
2. Then FSDP, as in decision 5: for parameters, and for the optimizer's state, or for
   the optimizer's state alone with `zero=1`.
3. An optimizer leaf takes the spec of the parameter whose path ends its own and
   whose shape it has; any other leaf is replicated.
4. A name that is not a field of `LayoutConfig` is an error that names the parameter.

## As built: what WP1 changed in the contract

WP0 and WP1 are done (`src/bux/parallel/`, `docs/parallelism.md`). Where they differ
from the contract above, the code wins:

- **Names are bux's metadata, not Flax's.** `parallel.named(*names, init=LECUN)` is
  `nnx.with_metadata(init, names=names)`, and a raw parameter is
  `nnx.Param(value, names=("heads",))`. Flax's `out_sharding` would make
  `nnx.eval_shape` under a mesh resolve the names as mesh axes and fail, so bux never
  uses it, and nothing turns Flax's eager sharding off. Pass the layer's current
  init as `init=`, so that values do not change.
- **The optimizer's state inherits the names**: Flax's `to_opt_state` copies a
  parameter's metadata to its moments. Rule 3 (matching paths) is gone; FSDP applies
  to `nnx.Param` (unless `zero=1`) and to `nnx.OptState`, names to every variable.
- `use(grid, layout, methods)` validates that every non-default method is
  registered; a default not registered yet is a call site not wired yet.
  `parallel.check(batch, seq_len)` raises the batch and seq_len errors, and `train()`
  calls it.
- The trainer calls `parallel.place(model, optimizer)` when it is made, and inside
  every jitted step, so the state never drifts from its shardings (zero=1 needs it).
- `shard(x, *names)` takes a name, or None, for every dimension. None is whole.
- `CausalConv(..., channel="kv_heads")` names a conv's channels.
- `Registry.find(tag)` returns a builder by name; `parallel.method(kind)` uses it.
  A method registers with `@bux.parallel.methods.<kind>.register("name")`.
- Fixtures: `losses(source, steps=3, **settings)`, each setting its source
  (`mesh="mesh(fsdp=8)"`, `optimizer="muon(lr=0.02)"`), with synthetic data,
  `adamw(lr=1e-2)`, `seq_len=16` and `batch=8` unless given; `involuntary()` counts
  XLA's full-rematerialization warnings since its last call.
- Measured with no layer named: every mesh of WP1's tests matches one device to 3e-6
  with AdamW and Muon. The MoE under `fsdp=2 × expert=2 × tensor=2` rematerializes
  twice; nothing else does.
- The suite takes about 6 minutes on 8 fake devices, idle.

## As built: what the later waves changed

All twelve WPs are merged; `docs/parallelism.md` is the reference. Where they left
the plan:

- **Decision 2:** the axes are `stage, data, fsdp, context, expert, tensor`. XLA aborts
  on a `shard_map` nested in the pipeline's over an axis earlier than `stage` (WP12).
- **Decision 7:** a method that sorts, routes or exchanges tokens (the MoE's, `ulysses`)
  is manual over every axis that splits them, not over its own alone: a partly manual
  `shard_map` makes XLA gather what the other axes split (WP4, WP7, WP8). `pass_state`
  runs its combine in the global view instead (WP10).
- **Counted names:** `("kv_heads", 2)` names a flattened `heads × head_dim` dimension,
  which then splits only between heads (WP2). Fused projections stay unnamed (WP3).
- **Each mixer's input** is constrained to `(batch, seq, None)`, which sequence
  parallelism needs (WP2, WP3).
- **Muon's `gather`** is a `shard_map`; constraints alone left Newton-Schulz on shards
  (WP9).
- **Microbatches** and the pipeline share `methods.microbatches`: under `stage > 1`
  the trainer does not accumulate. The router bias is deferred in a training step and
  moves once, from the whole batch's picks, under both (WP5, WP12).
- **`precision`** at its default is left out of the run's id, so ids made before it
  resume.
- **`methods.repeat`** changes the checkpoint's structure: a restore across `unroll`
  and `scan` is an error that says so (WP11). gpipe keeps `scan`'s layout.
- **Run on GPUs since (8 B300s):** every `methods.attention` and `methods.moe`, with
  tokamax's kernels (`scripts/gpu_smoke/`); `ragged_all_to_all` needs NCCL's barrier
  off on jax 0.12 (docs/parallelism.md). **Not yet:** bfloat16 pipelines on more than
  one device, which XLA:CPU cannot compile.

## WP0: JAX 0.12

**Owns:** `pyproject.toml`, `uv.lock`.

- `jax>=0.12.0`, `jax[cuda13]>=0.12.0` in the `cuda` extra, `orbax-checkpoint>=0.12.8`,
  `tokamax[cuda]>=0.0.17`; then `uv lock --upgrade-package` each of them, and jaxlib.
- The test suite already passes on the new versions; `ty` has not been run on them.
- Look through the 0.12 changelog for deprecations that bux's code hits, and fix them.
- **Done when** `make ci` passes.

## WP1: the core

**Owns:**
- New: `src/bux/parallel/` (`__init__.py`, `mesh.py`, `layout.py`, `place.py`,
  `methods.py`), `tests/test_parallel.py`, `docs/parallelism.md`.
- New, and empty: `experts.py`, `context.py`, `recurrence.py` and `pipeline.py` in
  `src/bux/parallel/`, already imported by its `__init__.py`, so that the WPs that
  fill them never edit that file.
- Edits:
  - `src/bux/train/config.py`
  - `src/bux/lang/ir.py`, where `Run.id` excludes the three settings
  - `src/bux/train/trainer.py`
  - `src/bux/architecture/graph.py` and `src/bux/architecture/residual.py`, where
    `stream(...)` goes on what every branch returns
  - `tests/conftest.py`
  - `docs/training.md`, `AGENTS.md` and `CLAUDE.md`: settings table rows and a link to
    the new guide

**Steps:**
1. The three configs, and the contract above. `bux/parallel/__init__.py` turns Flax's
   eager sharding off (`nnx.use_eager_sharding(False)`): every module that names a
   parameter imports it first. Check that the compiler, the printer
   (`run_to_source`) and the LSP take the new settings as they are; extend them if
   they don't.
2. `train()` runs inside `parallel.use(t.mesh, t.layout, t.methods)`. It builds the
   model with `parallel.build`, and the optimizer follows as it is built today. With
   `zero=1` it then calls `parallel.place(trainer.model, trainer.optimizer)`.
   `Trainer.update` and `Trainer.evaluate` put batches with `parallel.rows` instead of
   `jax.device_put`.
3. Errors at `use`, each naming the setting:
   - The mesh does not fit the devices.
   - `batch` does not divide over the batch axes.
   - `seq_len` does not divide over `context`.
4. Ahead of jax, `tests/conftest.py`:
   - sets `XLA_FLAGS=--xla_force_host_platform_device_count=8`;
   - adds `bux.parallel` to the jaxtyping hook;
   - gets two fixtures: `losses(source, steps=3, **settings)`, which trains and
     returns the losses, and `involuntary(capfd)`, which counts the warnings.
5. `docs/parallelism.md`: the three settings, the names, the rules, and how to add a
   method. Later WPs add a section each.

**Tests**, in `tests/test_parallel.py`:
- Every probe above that is about the core: one device vs data=8, fsdp=8, data=2 ×
  fsdp=4, and fsdp=2 × expert=2 × tensor=2. Run each with AdamW and Muon, on a Llama
  block, a dense MoE, hc, attnres and nextn heads, with losses within 1e-5 over 3
  steps.
- FSDP alone compiles with no involuntary rematerialization.
- A parameter is never whole on one device after `build`; `build` gives the values
  eager init does.
- A name whose axes do not divide the dimension leaves it whole; an unknown name is
  an error.
- `zero=1` keeps parameters whole and splits the optimizer state.
- A checkpoint restores onto another mesh.
- `Run.id` ignores the three settings, while a sweep over meshes is two runs.
- `shard` does nothing without a mesh.

**Note:** until WP6, `bux launch` still gives a single run one GPU, while `bux train`
uses every visible device.

## WP2: TP for softmax attention

**Owns:** `attention.py`, `mla.py`, `csa.py`, `indexer.py` (all under
`src/bux/architecture/`), `tests/test_attention.py`.

**Parameter names:**

| Layer | Parameter | Names |
|-------|-----------|-------|
| GQA | `q`, `gate`, `rel_proj` | `(embed, heads)` |
| GQA | `k`, `v` | `(embed, kv_heads)` |
| GQA | `o` | `(heads, embed)` |
| GQA | `sinks` | `(heads,)` |
| GQA | the k/v convs | their channels `kv_heads` |
| GQA | fused `qkv` | `(embed, None)`: its split points do not fall on shard boundaries |
| MLA | down projections | `(embed, latent)` |
| MLA | up projections | `(latent, heads)` |
| MLA | `o` | `(heads, embed)` |
| Indexer | | its heads `heads` |

**Activation constraints:**
- q: `shard(q, "batch", "seq", "heads", None)`; k and v the same with `kv_heads`.
- The head outputs before `o`.
- The logits `(b, h, t, s)` as `("batch", "heads", "seq", None)`, the keys whole:
  that is all-gather CP.

**Tests:**
- tensor=2 and 4: gqa, mqa, fused qkv, gated, sinks, mla, csa, and the indexer with its
  loss.
- Sequence parallelism: `layout(stream=tensor)`.
- All of them match one device; unfused gqa and mla compile with no involuntary
  rematerialization.

## WP3: TP for the other layers

**Owns:**
- Layers: `mlp.py`, `embedding.py`, `elementwise.py`, `linear_attention.py`,
  `ssm.py`, `rwkv7.py`, `conv.py`, `engram.py`, `residual.py`.
- Tests: `tests/test_linear_attention.py`, `tests/test_ssm.py`,
  `tests/test_residual.py`, and the layer tests in `tests/test_backend.py`.

**Parameter names:**
- MLP: `up` and `gate` `(embed, mlp)`, `down` `(mlp, embed)`; constrain the hidden
  units to `mlp`.
- Embedding and head, so that the logits split over the vocabulary and the loss is
  vocab-parallel:
  - token table `(vocab, embed)`, learned positions `(None, embed)`;
  - head `(embed, vocab)`, its bias `(vocab,)`;
  - logits constrained to `("batch", "seq", "vocab")`, tied heads included.
- Linear attention, SSMs and RWKV: projections `(embed, heads)` in and `(heads, embed)`
  out, per-head parameters on `heads`. Mamba's channels are `mlp`. The recurrence then
  runs on its own heads, with no communication.
- Engram tables: `(vocab, None)`.
- Residual algebras: their parameters are small, so leave them unnamed. Check that
  every widened stream passes `stream()` under tensor and fsdp meshes.

**Tests:**
- Each family under tensor=2 and 4, against one device.
- The loss, z-loss and mtp heads with the vocabulary split.

## WP4: a dropless MoE on one device

**Owns:** `src/bux/architecture/moe.py`, `tests/test_moe.py`; the `moe` default in
`MethodsConfig`.

- `_route` returns the top-k indices and weights (and the sinks). The dense path makes
  its one-hot itself. Balance losses and the bias stay as they are.
- A pure function, `experts(x, group_sizes, up, gate, down, ...)`, runs the expert FFN
  on rows sorted by expert with `jax.lax.ragged_dot`, or `tokamax.ragged_dot` where
  the import works. It applies the activation, `limit`, `up_softcap` and
  `expert_norm`. WP7 calls it inside `shard_map` with local weights.
- The method signature is `(moe, z, idx, weights) -> routed sum`, in latent space,
  before `latent_output_norm`. Register `"dense"` (today's code) and `"sorted"` (sort,
  `experts`, unsort). `"sorted"` becomes the default once it matches.
- Names: router `(embed, None)`; `up` and `gate` `(experts, embed, mlp)`; `down`
  `(experts, mlp, embed)`; `to_latent` `(embed, latent)`; `from_latent`
  `(latent, embed)`.
- **Tests:** sorted against dense to 1e-5, forward and gradients, for every MoE option
  the example architectures use: gated, latent, expert norm, limit, softcap, shared
  experts, sinks, groups.

## WP5: precision and microbatches

**Owns:**
- `src/bux/train/trainer.py`, `src/bux/train/config.py` (the `precision` setting),
  `src/bux/architecture/norm.py`.
- Lines in `moe.py`, `linear_attention.py`, `ssm.py` and `rwkv7.py`, for the float32
  islands. Attention's island is WP8's, which owns `_attend` in this wave.

**Precision:**
- `precision = precision(compute=bfloat16)`. Parameters stay float32. The train and
  eval steps cast the model's state to `compute` inside the loss, so gradients reach
  the float32 parameters.
- Float32 islands:
  - norm statistics;
  - router scores and top-k;
  - recurrent states;
  - the loss (already).

**Microbatches:** `methods.microbatches` is gradient accumulation, a scan over
microbatches with one reduction of the gradients after it. WP12 reuses the
microbatches.

**Tests:**
- bf16 losses near float32 ones; gradients and parameters float32.
- Microbatches give the same gradients as the whole batch.

## WP6: the launcher and multi-host

**Owns:** `src/bux/launch.py`, `src/bux/__main__.py`, `src/bux/train/sinks.py`,
`src/bux/train/hooks.py`, `tests/test_launch.py`; `parallel.rows` for many processes.

**GPUs per run:**
- A run needs the product of its mesh's sizes; `data=None` counts as 1 in a sweep,
  and as every visible GPU for a run alone.
- `supervise` gives each worker a group of free GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3`,
  first fit), and `status.json` lists them.
- A run that needs more GPUs than there are fails at once.

**Multi-host:**
- When the environment says so, `train_here` calls `jax.distributed.initialize()`
  before touching a device. This covers Slurm and Open MPI, and
  `JAX_COORDINATOR_ADDRESS`, `JAX_NUM_PROCESSES` and `JAX_PROCESS_ID` otherwise. Only
  a run alone trains on many hosts, not a sweep.
- `rows` makes the global batch with `jax.make_array_from_callback`. Every process can
  make any rows, as batch i depends only on the seed and i; read only the addressable
  ones.
- Only process 0 writes the console, metrics.jsonl, status.json and the run's files.
  Checkpoints are every process's: orbax handles that.
- A signal on any process stops all of them at the same step, so check `stopping`
  across processes when there are several.

**Plan output:** `bux plan` shows each run's GPUs, and its parameter and optimizer
bytes per device: `parallel.shardings` on the abstract state over a
`jax.sharding.AbstractMesh`.

**Tests:**
- GPU groups as a pure function.
- Two CPU processes of 4 devices each, through `jax.distributed` on localhost, training
  the smoke run with fsdp=8 as one process does.

## WP7: EP

**Owns:** `src/bux/parallel/experts.py`, `tests/test_experts.py`.

**Methods,** each a `shard_map` over `expert`:
- `all_gather`: gather the tokens, indices and weights over `expert`; every shard runs
  its own experts on the tokens they got (WP4's `experts`); `psum_scatter` back. The
  simple baseline.
- `all_to_all`: buffers of a fixed size per peer, through `jax.lax.all_to_all`.
  Dropless by default: a buffer holds the local tokens × top_k.
- `ragged_all_to_all`: as Grug and MaxText do it. Not on XLA:CPU, so its tests skip
  there, and `scripts/gpu_smoke.py` runs it.

**Semantics:**
- Routing, the balance loss and the bias update run before the `shard_map`, in the
  global view.
- The batch splits over `expert` too (`layout.batch`); FSDP dims of the expert
  weights stay Auto.

**GPU caveats** (reported, unverified):
- On jax 0.11, NCCL's ragged all-to-all hangs unless
  `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`.
- Its cost grows with devices even when sizes are 0 (openxla/xla#46982).

**References:** `lib/levanter/src/levanter/grug/_moe/ep_*.py` in
marin-community/marin; MaxText `RoutedMoE.sparse_matmul`.

**Tests:** each CPU method against `sorted` on expert=2 and 4, with data and fsdp,
forward and gradients; the router bias moves the same.

## WP8: CP for attention

**Owns:** `src/bux/parallel/context.py`, `tests/test_context.py`; the call site in
`attention.py`'s `_attend`.

**The call site:** route the core of `_attend` (logits, probabilities, values) through
`method("attention")`, with a signature `(q, k, v, visibility) -> out`. The visibility
carries:
- the documents' ids, the window, the scale, the softcap and the sinks;
- not `keep` masks, relative bias, length scale or attention dropout: those stay
  `xla`'s alone.

A method that lacks a feature raises `NotImplementedError` naming it.

**Methods:**
- `xla`: today's code. With `seq` over `context` it is all-gather CP already.
- Every method takes its softmax in float32: WP5's island for attention.
- `ring`: a `shard_map` over `context`. K/V blocks go round by `ppermute`, with an
  online softmax, skipping blocks ahead of the queries, and masks for documents and
  windows. Differentiate through the scan first, and write a custom VJP later. Port
  MaxText's `jax_ring_attention.py`, which is pure JAX.
- `ulysses`: `all_to_all` from positions to heads over `context`, local `xla`
  attention, and back. It needs `context` to divide the heads.
- `tokamax`: `tokamax.dot_product_attention(..., q_sharding=...)`, all-gather CP in a
  kernel. Test it on CPU through its `xla` implementation if that works, and on GPU
  through the smoke script.

**Load balancing** of causal attention (zigzag) is later. Note that `Segments` reads
documents from the order of positions, so compute it before any permutation.

**Tests:** every method against `xla` on context=2 and 4, with packed documents,
windows and GQA, forward and gradients.

## WP9: distributed Muon (optional)

**Owns:** `src/bux/train/optim.py`, the Muon tests in `tests/test_train.py`.

- `methods.orthogonalize="gather"`: constrain each update matrix whole (stacked
  experts: whole per expert) before Newton-Schulz, and back after.
- Measure collectives against `gspmd`. Owner-per-matrix schemes (Moonlight, Dion) are
  the next method.
- **Tests:** the same updates as one device; fewer all-reduces.

## WP10: CP for recurrent mixers

**Owns:** `linear_attention.py`, `ssm.py`, `rwkv7.py`; `src/bux/parallel/recurrence.py`.

- `methods.recurrence="pass_state"`, a `shard_map` over `context`:
  1. Each shard runs its chunk from a zero state and keeps the chunk's transition,
     the affine map S ↦ S·A + B per head, with A = 0 across a document start.
  2. The maps combine across shards by an all-gather and a local prefix (LASP-2), or
     by a `ppermute` chain.
  3. Each shard corrects its outputs with the state that comes in.
- A is a matrix per head for the delta rule and RWKV-7, and a diagonal or a scalar for
  Mamba-2, GLA and lightning.
- Refactor `recurrence` and `selective_scan` to return outputs, final state and
  transition, by chunk.
- **Tests:** `pass_state` against `scan` on context=2 and 4, with documents that cross
  shard boundaries.

## WP11: scan over repeats, and remat

**Owns:** `src/bux/architecture/graph.py`; `decayed`, `undecayed` and `matrices` in
`optim.py`.

**Remat:** `methods.remat` wraps each top-level branch's body in `nnx.remat`, and a
scanned layer as a whole:
- `none`;
- `full`: `nothing_saveable`;
- `dots`: `dots_saveable`;
- `dots_no_batch`: `dots_with_no_batch_dims_saveable`.

They are registered, so another policy is a function.

**What scans:**
- `methods.repeat="scan"`: build the `times` bodies as today, so that layer-dependent
  inits stay the same. If their graphdefs are identical, stack their states and run
  them by `jax.lax.scan`.
- Unroll instead when:
  - the graphdefs differ;
  - the body leaves taps unbalanced;
  - the stream grows (attnres);
  - the body holds shared modules.
- Stacked parameters get a leading `layers` dimension: None, or `stage` for PP.

**Gotchas:**
- `decayed` decays what has ndim ≥ 2, which stacked norm scales are: count from the
  stacked axis.
- Muon needs the stacked axis as a batch dimension of `MuonDimensionNumbers`.

**Tests:**
- Scan against unroll for Llama, `"DDDA"^k` and hc; compile time falls for 24 layers.
- Remat gives the same losses, and less temp memory in `compiled.memory_analysis()`.

## WP12: PP

**Owns:** `src/bux/parallel/pipeline.py`, `tests/test_pipeline.py`; the call site for
scanned repeats in `graph.py`; a field in `MethodsConfig` for circular repeats.

**How it pipelines:**
- With `mesh.stage > 1`, pipeline the longest scanned repeat. Its stacked parameters
  reshape to `[stage, layers / stage, ...]` and split over `stage`; the rest of the
  model does not.
- `methods.pipeline` is `gpipe` or `circular`, over `methods.microbatches`. This is
  MaxText's design:
  - a vmap over stages with `spmd_axis_name="stage"`;
  - activations rotating by a `shard_map`'d `ppermute`;
  - a `lax.scan` over microbatches + stages − 1 iterations.

**Taps:** a tap read across the pipelined repeat (`$v1`) is an error that names it.
Sending taps along is later, and a cost of the architecture that `bux plan` could
show.

**Plan output:** `bux plan` shows the bubble, (S − 1) / (M + S − 1).

**References:** MaxText `src/maxtext/layers/pipeline.py` (`NNXPipeline`); the UvA DL
notebooks on pipeline parallelism (about 100 lines).

**Tests:** stage=2 and 4 × 4 and 8 microbatches against one device; the error on taps;
the bubble.

## Not in this plan

- FP8.
- Overlapping communication with compute (async collectives, collective matmuls).
- Zigzag CP.
- MoE with a capacity that drops tokens.
- DeepEP.
- Explicit mode.
- Elastic training.

## The brief for an agent

> You implement WP*n* of `plans/parallelism.md` in bux. Read `AGENTS.md`, then the
> plan's sections "Decisions" and "The contract", then WP*n*. Edit only the files your
> WP owns; if you need a change elsewhere, make the smallest one and say so in your
> summary. Match the code around you: short functions, jaxtyping shapes on every
> array, tests named as sentences (`test_a_run_trains_the_same_sharded`). Compare
> against one device on the 8 fake CPU devices the tests have. Run
> `uv run ruff check . && uv run ruff format --check . && uv run ty check src tests &&
> uv run pytest -q` (not `make ci`: worktrees break a bare `ty check`). Add your
> section to `docs/parallelism.md`, and run `ruff format` after editing docs. You are
> done when your WP's tests and the checks pass; report what you did, what you
> skipped, and anything the plan got wrong.
