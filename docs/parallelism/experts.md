(parallelism-experts)=
# Mixture of experts and expert parallelism

This page is about running an MoE layer on many devices: which method of
`methods.moe` runs the experts the router picks, and how expert parallelism (EP)
splits the experts themselves over the `expert` axis. The layer itself, with its
routing, shared experts and load balancing, is on
[Mixture of experts](../architectures/experts.md).

The default method, `sorted`, runs every expert on every device, each on that
device's own picks, with the experts' kernels gathered by FSDP. `expert` is then one
more batch axis. An EP method instead gives each device its own experts and sends them
the picks of every device in its group.

Which to choose:

- **On one node**, keep the default, `sorted` under FSDP. At the baseline MoE's shape
  it is the fastest method bux has: 831k tokens/s on 8 B300s, against 615k for
  `nccl_ep`, the fastest EP. Its FSDP gathers move less than EP's exchanges.
- **Across nodes**, `sorted` under HSDP is still the fastest. If you want EP, keep
  `expert` inside a node and `data` across nodes, and use `nccl_ep` where
  TransformerEngine is installed.
- **On B300s**, multiply the experts in MXFP8 (`precision(matmuls=mxfp8)`). They train
  10 to 15% faster than in bfloat16.

The numbers behind these are in [Measurements](#measurements).

## Quick start

`examples/experiments/parallel/ep_moe.bux` trains a 16B MoE in DeepSeek-V3's shape on
two nodes of 8 GPUs. Its experts split 8 ways inside each node, and FSDP spans the
nodes:

```bux
use deepseek_v3
use fineweb_wsd

d_model = 2048
attn.num_heads          = 16
dense.intermediate_size = 10944
experts.num_experts        = 64
experts.top_k              = 6
experts.num_shared_experts = 2
experts.num_groups         = 1
experts.top_k_groups       = 1
experts.expert             = gated_mlp(intermediate_size=1408, activation=swish)
model = embed layer(dense) layer(experts)^26 norm head

mesh      = mesh(fsdp=2, expert=8)
precision = precision(compute=bfloat16)
methods   = methods(moe=all_gather, repeat=scan, remat=dots)
```

The benchmark suites run EP on one node of 8 GPUs in two ways. The `moe_ep` arm of
`examples/benchmarks/one_node.bux` sends a ragged all-to-all at a capacity of 1.25:

```bux
mesh    = mesh(expert=8)
methods = methods(moe=ragged_all_to_all, capacity=1.25)
```

The `nccl_ep` arm of `examples/benchmarks/megatron.bux` sends by NCCL EP in 8
microbatches, with the matmuls in MXFP8:

```bux
mesh      = mesh(expert=8)
methods   = methods(moe=nccl_ep, capacity=1.25, microbatches=8)
precision = precision(compute=bfloat16, matmuls=mxfp8)
```

## Options

| Setting | Values | Default | What it does |
|---|---|---|---|
| `mesh.expert` | a size | `1` | the devices the experts split over ({bux}`mesh`) |
| `layout.experts` | axes | `expert` | the mesh axes the experts split over ({bux}`layout`, [Names](layout.md#names)) |
| `methods.moe` | a method below, or your own | `sorted` | how the picked experts run ({bux}`methods`) |
| `methods.capacity` | a number above 0, or `none` | `none` | the picks a device takes, as a multiple of its own; the rest drop ([A capacity](#a-capacity)) |
| `methods.moe_chunks` | a count | `1` | the slices of each device's tokens that EP sends one after another ([In chunks](#in-chunks-methodsmoe_chunks)) |
| `methods.moe_stream` | `true`, `false` | `false` | runs NCCL EP's kernels on their own GPU stream ([On a stream of its own](#on-a-stream-of-its-own-methodsmoe_stream)) |
| `precision.roles.experts` | `compute`, `mxfp8`, `mxfp8_2d`, `none` | `none` | the format the routed experts multiply in; `none` follows `matmuls` where a grouped GEMM supports it ({bux}`roles`, [The experts in MXFP8](#the-experts-in-mxfp8)) |
| `precision.dispatch` | `compute`, `mxfp8` | `compute` | the format EP sends tokens to their experts in ({bux}`precision`, [FP8 dispatch](#fp8-dispatch-precisiondispatch)) |

The methods of `methods.moe`:

| Method | Splits the experts | What it does | Runs on |
|---|---|---|---|
| `sorted` | no | The default. Sorts the picks by expert, runs each expert on its own rows, and unsorts. Drops no token. | any device |
| `dense` | no | Runs every expert on every token, weighted by the router: the reference. | any device |
| `all_gather` | yes | Each device gathers its group's tokens and runs its own experts on them. A `psum_scatter` brings each token its sum back. | any device |
| `all_to_all` | yes | Sends each pick to its expert's device and back by `jax.lax.all_to_all`, in a buffer of one size per peer, sorted by expert. A device runs its experts on each peer's buffer as it arrives, one grouped GEMM per peer. | any device |
| `ragged_all_to_all` | yes | Sends each pick to its expert's device and back by `jax.lax.ragged_all_to_all`, one slice per expert per peer. The rows arrive sorted by expert, and the experts run on them in place. | GPUs: XLA:CPU has no ragged all-to-all |
| `ragged_tokens` | yes | Like `ragged_all_to_all`, but sends each token once to each device that holds any of its experts, with its picks there, and gets back their weighted sum: one row per device, not per pick. | GPUs |
| `nccl_ep` | yes | Sends each token once to each device that holds any of its experts, and once to each other node, by NCCL EP's kernels through TransformerEngine. It arrives as one row per pick, grouped by expert. | GPUs with TransformerEngine ([NCCL EP](#transformerengines-nccl-ep)) |

## Constraints and errors

- Where no axis splits the experts (`expert=1`, or a number of experts that `expert`
  does not divide), every EP method runs as `sorted`.
- Experts split over an axis that leaves the tokens whole, as `layout(experts=tensor)`
  does, split the tokens over it too, inside the method. Rows that do not divide over
  that axis are an error.
- `roles(experts=nvfp4)` is an error: no grouped GEMM multiplies in NVFP4.
- A capacity that drops picks changes what a run computes, but it stays out of the
  run id, like the rest of `methods`. `precision.dispatch = mxfp8` changes what a run
  computes too, and is in the run id.
- `nccl_ep` needs a GPU and TransformerEngine's JAX bindings, which the Docker image
  has. It takes one MoE shape per run, the experts over one axis, and tokens in
  bfloat16: `precision(compute=bfloat16, dispatch=compute)`. Anything else is an error.
- On a CPU, `sorted` and the EP methods make a bfloat16 `psum` over more than one
  device, which XLA:CPU cannot compile. Such runs set `precision(compute=float32)`, or
  use `methods(moe=dense)` ([bfloat16 psums on a CPU](precision.md#bfloat16-psums-on-a-cpu)).
- XLA's ragged all-to-all leaves an allocation inside the memory pool after the first
  step, so a step whose temporaries take over half the pool runs once, then runs out
  of memory. bux's pool of 95% of a GPU, against JAX's 75%, leaves it more room.

:::{warning}
Keep `expert` inside a node. Across hosts, XLA splits a ragged all-to-all into an
all-to-all inside each host and an all-gather of the tokens across hosts. On two
nodes, `ragged_all_to_all` on `expert=16` trains at a third of the speed of
`expert=8` with `data=2`.
:::

:::{warning}
On jax 0.12, NCCL's ragged all-to-all hangs unless `XLA_FLAGS` has
`--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`. `bux train`
and `bux launch` add it. In your own script, call `bux.runtime.configure()` before
anything touches a device.
:::

## Expert parallelism

Five methods of `methods.moe` split the experts over the axes `layout.experts` gives
them, `expert` by default. Each device holds its own experts' kernels and runs them on
the picks of every device in its group.

### How the EP methods run

- Routing, the balance loss and the router bias run first, in the global view, as
  under `sorted`.
- The method runs in a `jax.shard_map` that is manual over the axes that split the
  tokens (those of `batch` and `seq`), not over `expert` alone: sorting tokens split
  over an Auto axis would gather them all onto every device. `tensor` stays Auto.
- The kernels enter split over the experts' axes and whole over the others, so FSDP
  gathers only a device's own experts.
- Each pick's weight travels to its expert's device with it. The experts weight the
  rows they make, so what comes back is summed as it is, and nothing the backward
  keeps for the weights is `d_model` wide.
- The ragged methods send each exchange in at least 2048 pieces (`experts.PIECES`).
  XLA's one-shot ragged all-to-all copies each slice with one block of 128 threads:
  on 8 B300s, one slice per expert per peer moved 90 GB/s, and the pieces move 600.

### A capacity

```bux
methods = methods(moe=ragged_all_to_all, capacity=1.25)
```

Without a capacity, a device of `all_gather` or of the ragged methods takes every pick
its peers could send: `expert` times its own picks. Its experts then run on all those
rows. `methods.capacity` bounds them:

- A device takes at most `capacity` times its own picks (its tokens times `top_k`),
  about its average share. It drops the rest: its last experts' picks first, and of
  each expert's, the last peers'.
- `ragged_tokens` runs that many picks, and takes as many tokens, or every token that
  could come if that is fewer.
- `all_to_all` bounds each of a device's buffers, one per peer, at `capacity` times
  that peer's even share of its picks (its picks over `expert`). It drops what does
  not fit, that peer's last experts' picks first. Without a capacity every buffer
  holds every pick, so a device sends `expert` times the rows it has.
- A dropped pick adds nothing to its token.
- A training step reports the share of picks dropped as `dropped`, summed over the MoE
  layers, as `balance_loss` is.

On random tokens at the start of a run, a capacity of 1.25 drops 4-10% of the picks
per layer, and 1.5 drops 3-4% (at 16k tokens per GPU). A larger capacity does not
always buy fewer drops: on `megatron.bux`'s synthetic batch under `nccl_ep`, 1.5 took
2.4 GiB more at 5% fewer tokens/s, and from step 15 on dropped as much as 1.25 (8% of
the picks per layer). On random tokens the router's routing drifts uneven, which is
why the megatron benchmarks route by random logits: then nothing drops
([Routing by random logits](#routing-by-random-logits)).

### In chunks: `methods.moe_chunks`

```bux
methods = methods(moe=all_to_all, capacity=1.25, moe_chunks=2)
```

An EP layer waits for four exchanges: the picks there and back, and the backward's
two. `methods.moe_chunks` cuts each device's tokens into that many slices, which the
method runs one after another (`experts.in_chunks`). The compiler can then send one
slice's picks while another slice's experts run.

- Where the count does not divide a device's tokens, it takes the largest count below
  it that does.
- Each slice's experts add their part of the kernels' gradients, which add up in
  float32, as one call's would.
- Each chunk adds a grouped GEMM per peer per layer to compile: for the baseline MoE
  on 8 B300s, `moe_chunks=2` compiles in 5 minutes, and 4 in more than 8.
- Only `all_to_all` gains, 3% on one node. XLA's ragged all-to-all takes every SM, so
  it overlaps nothing, and in chunks a step is slower.

### FP8 dispatch: `precision.dispatch`

```bux
precision = precision(dispatch=mxfp8)
```

The exchanges send the tokens to their experts in MXFP8, as DeepSeek-V3 does: E4M3
values and a power-of-two scale per 32, `d + d/32` bytes per row. At `d = 2048` that
is 52% of bfloat16's bytes. The tokens are dequantized to the compute dtype where they
arrive.

- Each token quantizes once, before its k picks' rows are gathered, so the quantizer
  reads a k-th of the rows and the gather writes half the bytes.
- What comes back, and the backward's two exchanges, go in `compute`.
- The gradient is the exchange's, as if the tokens went unquantized
  (`experts._picked_in_mxfp8` for the ragged methods, `experts._swapped_in_mxfp8` for
  `all_to_all`).
- Rows that are not whole blocks go as they are.
- It changes what a run computes, so it is in the run id. `nccl_ep` does not take it.

The exchanges name what they bring `experts.exchanged`, under `experts`, so that
`remat=dots`, which keeps `experts`, does not exchange them again in the backward
([Repeats and remat](remat.md)). On one node, `ragged_all_to_all` trains at 556k
tokens/s with it, against 539k without.

## TransformerEngine's NCCL EP

```bux
mesh    = mesh(expert=16)                        # over two nodes
methods = methods(moe=nccl_ep, capacity=1.25)
```

TransformerEngine (2.18 and later, `transformer_engine.jax.ep`) dispatches and
combines by NCCL EP, a DeepEP built on NCCL's device API. It moves tokens by loads and
stores over NVLink inside a node, and by RDMA that the GPU starts (NCCL's GIN) between
nodes, with no NVSHMEM.

- A token goes once to each GPU that holds any of its experts, as in `ragged_tokens`,
  and once to each other node, whose NVLink takes it on.
- It arrives as one row per pick there, grouped by expert, each expert's rows padded
  to 128.
- Since TE 2.20, one process drives all its node's GPUs, on a communicator TE borrows
  from XLA.
- The Docker image has TE 2.20.2 (`--build-arg TE_VERSION` picks another).
- On one node it is the fastest EP (615k tokens/s for the benchmarks' MoE, against
  831k for `sorted`). Over two nodes it is 2.4x `ragged_all_to_all`
  ([One node and two](#one-node-and-two)).

bux sets `NCCL_GIN_TYPE=2` (GIN by a CPU proxy) and `NCCL_EP_TOKENS_PER_CHUNK=256`
wherever the environment does not set them (`runtime.configure`). On two B300 nodes,
GDAKI, NCCL's default, took 3x as long for an EP layer's exchanges, and chunks of 64
tokens 1.3x ([Environment variables](../reference/environment.md)).

:::{tip}
Give `nccl_ep` microbatches. NCCL EP sizes its buffers for the most tokens a device
sends at once. Sized for a whole step, `megatron.bux`'s batch of 512 rows asked for a
29 GiB staging buffer per GPU and did not start. In 8 microbatches it trains at 32.2k
tokens/s per GPU on 8 B300s (`remat=none`, 235 GiB).
:::

### How `nccl_ep` runs

`nccl_ep` runs TE's one-device primitives in the same `shard_map` as the other EP
methods, with a `jax.custom_vjp` each way (`parallel/nccl_ep.py`). The experts take
the rows as they arrive, in their blocks padded to 128. TE's global-view ops
(`ep_dispatch`, `ep_combine`) hang across nodes, so bux does not use them.

- **Buffers, sized once.** NCCL EP keeps its buffers outside the step. They are sized
  for a chunk of a device's share of a microbatch (`methods.moe_chunks`,
  `methods.microbatches`), and for the rows `methods.capacity` lets it take. So
  `methods.prepare` starts NCCL EP once per process, before the step traces
  (`nccl_ep.start`): 25 to 60 s between two nodes.
- **Every dispatch brings the rows it was started for.** NCCL EP fails on fewer. An
  eval step, which takes a step's rows whole, sends them one after another in pieces
  of at most a training chunk's tokens. It is an error if those pieces do not divide
  its tokens evenly.
- **Under a capacity**, the picks that do not fit drop before NCCL EP sends them, as
  under `ragged_all_to_all` (`nccl_ep._kept`). With NCCL EP's own drops, EP=16's
  balance loss rose to 35 in 25 steps.

::::{dropdown} Calling TE's EP primitives yourself
- **No handle outlives its pass.** TE keys a route's state by the address of the
  handle its prepare writes, which a copy changes. So the combine takes its own
  dispatch's handle, and each backward call prepares the route again from the picks
  its forward kept (1 MiB per layer), at no measurable cost.
- **Padding.** An expert's rows past its picks are 0, but their weights are garbage,
  as are the rows past the last expert's. Each expert's count of the picks kept for it
  says which rows hold picks.
- **Import order.** TE's core library loads the NCCL that jax's GPU backend loads, so
  it imports after `jax.devices()`.
::::

`tests/test_nccl_ep.py` checks a token's path on CPU, through an emulation of TE's
primitives. On two nodes of 8 GPUs (EP=16), a layer's output and gradients come within
bfloat16's rounding (4e-3 to 8e-3 of the largest) of `sorted` without a capacity, and
of `ragged_all_to_all` at a capacity of 1.

### On a stream of its own: `methods.moe_stream`

```bux
methods = methods(moe=nccl_ep, capacity=1.25, moe_stream=true)
```

By default NCCL EP's kernels run on the compute stream, where nothing runs beside
them. With `moe_stream` they run on their own stream (`nccl_ep.STREAM`), each of
TE's calls an async call there (`jax_internals.on_stream`). The compute stream waits
for a call's results only where it uses them.

:::{note}
`moe_stream` is off by default: a step whose microbatches run one after another gains
nothing from it, and loses 3% in 2 chunks ([Measurements](#moe_stream-at-megatron-bridges-batch)).
It is there for schedules of overlapped microbatches, as `methods(interleave=true)`
makes ([Side by side](microbatches.md#side-by-side-methodsinterleave)).
:::

::::{dropdown} How `moe_stream` overlaps
- **Scheduler hints.** XLA's scheduler starts each call as soon as its inputs are
  ready and waits for it as late as possible (`force_delay_async`). Of calls ready at
  once, those that send rows to the experts start first
  (`scheduler_delay_priority`), so the next chunk's dispatch does not queue behind
  this chunk's combine.
- **A pass over XLA's HLO.** Inside a `shard_map`, XLA would also put the parts of each
  call's result on the stream, where they queue behind later calls.
  `jax_internals.parts_off_the_stream`, registered as the pass
  `nccl_ep_parts_off_the_stream` ([bux's passes](../performance/compiler.md#buxs-passes-buxpasses)),
  takes them off. Once `moe_stream` is set, `nccl_ep.start` has XLA run this pass on every GPU
  program the process compiles from then on, and JAX's cache keys those programs by
  it.
::::

## The experts in MXFP8

Three ways to set the experts' format, one line each:

```bux
precision = precision(matmuls=mxfp8)                          # the experts too
precision = precision(roles=roles(experts=mxfp8))             # the experts alone
precision = precision(matmuls=mxfp8, roles=roles(experts=compute))
```

The routed experts' role, `experts`, takes `compute`, `mxfp8` or `mxfp8_2d`
([Precision](../training/precision.md)). Left out, it follows `matmuls` where a
grouped GEMM multiplies in that format (`mxfp8`, `mxfp8_2d`), and is `compute` under
any other, such as `nvfp4` (`PrecisionConfig.experts`). On B300s, MXFP8 experts train
10 to 15% faster than bfloat16 ones ([MXFP8 experts](#mxfp8-experts-on-b300s)).

### How the experts multiply in a format

- Each GEMM runs by `grouped_blockscaled.ragged_dot`. Every operand of the forward's
  GEMM and of the backward's two is quantized along what that GEMM contracts, as a
  projection's are ([Matmul formats](precision.md#matmul-formats)), and each expert's
  W on its own. Up and gate's one GEMM quantizes x once.
- One pass of a Triton kernel (`grouped_triton`) quantizes x and x^T, zeroes the rows
  past the experts' (which an exchange's buffer leaves as they were), and lays both
  out as the kernels take them.
- Under `sorted`, the quantizer reads each of x's rows from its token by index
  (`grouped_blockscaled.Picked`), so the copy that `moe.token_rows` would gather is
  never written: 1 GiB per layer at 32k tokens per GPU and 8 picks per token. A GEMM
  whose rows sum into their tokens (`Summed`) returns the tokens' sums, and its
  backward reads dy from the tokens the same way.
- On a Blackwell GPU, the Mosaic GPU kernels of `grouped_mxfp8` multiply by tcgen05's
  block-scaled MMA. Elsewhere XLA dequantizes and multiplies, which computes the same
  values, so the tests run it on CPU.
- Only the GEMMs of `precision.passes` multiply in the format; the others run by
  `grouped`'s kernels.

:::{note}
The experts' dW quantizes a device's rows of an expert in blocks of 32 as they lie, so
a block that two experts share has one scale. Which rows share a block depends on how
the mesh splits the tokens. In float32, 3 steps on `fsdp=2, expert=4` come within 1e-4
of one device's losses, not bit for bit (`tests/test_precision.py`).
:::

::::{dropdown} Inside the `grouped_mxfp8` kernels
`rows` computes x W and dy W^T, and `contract` computes dW.

- A pair of SMs multiplies each tile of 256 rows by 128 columns, by tcgen05's MMA of
  two SMs.
- Each pair takes tiles one after another, in bands of 8 tiles of rows, down a band's
  rows first, so the tiles in flight share few rows and columns, which stay in L2.
  Taking a whole row of tiles at a time was 1.7x slower.
- One warp loads each step's tiles and their scales by TMA into a ring of 7 buffers,
  the scales laid out as tensor memory takes them (128 rows by 4 scales per 512-byte
  tile). A second warp copies the scales into tensor memory (`tcgen05.cp`) and issues
  the MMAs into one of two accumulators there. A warpgroup stores the other
  accumulator, so a tile's store overlaps the next tile's MMAs.
- `rows` runs a tile that two experts share once for each, and each run stores only
  its own rows. It stores the rows past the experts as 0.
- `contract` takes an expert's rows in tiles of 128 along what dW contracts. The first
  and last tiles of x^T come from copies with the neighbours' rows as 0.
::::

## Activations fused, by cuDNN

On a Blackwell GPU with cudnn-frontend (the `cuda` extra), cuDNN's CuTe DSL kernels
run the routed experts (`grouped_glu`, over `grouped_cudnn`, by `cudnn.jax.call` on
XLA's stream) wherever all of these hold:

- their activation is one of cuDNN's epilogues (`grouped_glu.epilogue`):

  | The experts | cuDNN's epilogue |
  |---|---|
  | `mlp` of `relu2` (Nemotron 3) | SReLU, relu(u)^2 |
  | `gated_mlp` of `swish` (Qwen3, DeepSeek-V3) | SwiGLU |
  | `gated_mlp` of `swish` or `quick_gelu` with a `limit` or an `up_offset` (DeepSeek-V4, gpt-oss) | GeGLU, the gate times 1 or 1.702 in its sigmoid, clamped and offset as `limit` and `up_offset` say |
  | `gated_mlp` of `situ` with an `up_softcap` (Kimi K3) | SiTU-GLU |

- they have no `bias` (so gpt-oss's own experts run by bux's kernels) and no
  `expert_norm`;
- they multiply in `mxfp8`, every pass;
- their width is a multiple of 64 (Nemotron 3 Nano's 2688 and gpt-oss's 2880 are),
  and their hidden units a multiple of 32.

Hidden units that are not a multiple of 128 run padded with units of 0, whose
activation is 0 and whose gradients the padding drops: Nano's 1856 run as 1920.

The kernels fuse the activation, its gradient, the routing weights and the
quantizations into the GEMMs' epilogues. bux's own kernels leave those to XLA.

- Under FSDP, where a layer gathers its parameters as it runs (one microbatch, or
  `methods(gather=layer)`:
  [Gathered once, reduced once](microbatches.md#gathered-once-reduced-once-methodsgather-methodsaccumulate)),
  each GPU quantizes its own shard of the experts' kernels, and FSDP gathers them in
  MXFP8. That is an eighth of the quantizer's work on `fsdp=8`, and no bfloat16 copy
  of all the experts.
- CuTe DSL compiles each kernel in every process, which adds 20 to 110 s to a run's
  compile.

At Megatron-Bridge's batch, the fused kernels train Qwen3-30B-A3B's `sorted` arm 4.5%
faster in 15 GiB less, and Nemotron 3 Nano's 5.1% faster in 14 GiB less
([The fused activations](#the-fused-activations)).

::::{dropdown} How the fused kernels work
What each GEMM's epilogue does:

| GEMM | Kernel | In its epilogue |
|---|---|---|
| x W, up (and gate) | `activated`: cuDNN's SReLU or GLU kernel | the activation, times each row's weight, quantized both ways for down and its dW; the product kept for the backward |
| h W, down | `rows` | |
| dy W^T, down | `activation_grad`: cuDNN's dSReLU, dSwiGLU or dGLU kernel | the activation's gradient at the kept product, times the row's weight, quantized both ways; the weight's gradient |
| dp W^T, up (and gate) | `rows` | |
| dW, both | `contract`: cuDNN's block-scaled wgrad | |

- Their tiles are 256 by 256 on a pair of SMs, where `grouped_mxfp8`'s are 256 by 128.
  Tiles of 256 by 128 were slower at every model's shape measured below.
- They take each expert's rows from a multiple of 256. The quantizer reads each row
  from its token and writes the padding as 0, and the tokens' sums read the rows back.
- An exchange's rows come out as they lay, by `grouped_mxfp8.rows`, which reads each
  expert's rows from its own start and packs its output, saving a gather each way.
- Their GLU takes each gate 32 columns before its up, where up and gate's stack holds
  them side by side. The quantizer reads W_up's pairs that way, and the transpose of
  the product's gradient for up's dW writes them back.
- Where the hidden units are padded, the quantizer pads the weights, and the wgrads
  write only the real units.
- Clamped, the kernels pass the gradient of a product exactly at the limit whole,
  where bux's `jnp.clip` splits it.
- `CUDNNFE_CLUSTER_OVERLAP_MARGIN` leaves pairs of SMs to the collectives beside them;
  bux leaves it at 0 ([Environment variables](../reference/environment.md)).
::::

## Writing your own method

The router runs first, in the global view: it picks each token's top-k experts,
reports the balance loss and moves the router bias. A method of `methods.moe` then
runs the picked experts, as a function `(moe, z, idx, weights, taps) -> y`:

| Argument | What it is |
|---|---|
| `z` | the tokens `(b, t, d)`, in the experts' width: the latent one in a LatentMoE |
| `idx`, `weights` | each token's picks and their weights, `(b, t, k)` |
| `taps` | the forward's taps, through which a method reports what it measures, as the EP methods report `dropped` ([A capacity](#a-capacity)) |
| `y` | the picks' weighted sum `(b, t, d)`, before `latent_output_norm` |

Register it by name:

```python
from bux.architecture.experts.moe import routed
from bux.parallel import methods


@methods.moe.register("mine")
def _(moe, z, idx, weights, taps=None):
    return routed(z, idx, weights, *moe.kernels(), moe.config.expert, moe.expert_norm)
```

`methods = methods(moe=mine)` then picks it.

### An EP method

An EP method is a body that runs on one device's tokens, named as the method is in
`bux.parallel.experts`. `experts.expert_parallel` makes it a method of `methods.moe`:

```python
from bux.parallel import experts, methods


def mine(z, idx, weights, kernels, *, axes, ffn, norm):
    # This device's tokens and picks, and its experts' kernels: experts
    # [i * n, (i + 1) * n) for n = len(kernels[0]), i = jax.lax.axis_index(axes).
    # Return the tokens' sums, and the share of the group's picks it dropped.
    ...


methods.moe.register("mine")(experts.expert_parallel(mine))
```

`expert_parallel` runs the body in the `shard_map` of
[How the EP methods run](#how-the-ep-methods-run), in `methods.moe_chunks` slices, and
runs `sorted` where no axis splits the experts.

- The receiving buffers can be `jax.lax.empty`: the experts make 0 of the rows past
  theirs, and the rows that do not come back are masked.
- `experts.ragged_all_to_all` and `experts.ragged_tokens` take the ragged exchange as
  `exchange=`, so another collective is a function. The tests run them on CPU through
  one built from `all_to_all`, and `scripts/gpu_smoke/experts.py` runs the real one,
  and every other method, on GPUs.

### The pieces: `experts` and `routed`

A token's path is made of functions of arrays alone, in
`bux.architecture.experts.moe`, so a method can run them on one device's experts
inside a `shard_map`.

`experts(x, sizes, weights, up, down, ffn, norm)` runs the expert FFN on rows sorted by
expert:

- The first `sizes[0]` rows are expert 0's, the next `sizes[1]` expert 1's, and so
  on. Each row's output is scaled by its weight.
- The products run by `grouped.ragged_dot`: on a GPU, tokamax's Mosaic kernel, and its
  Triton one for the kernels' gradient (27x XLA's `jax.lax.ragged_dot` on a B300);
  elsewhere XLA's.
- Up and gate are one GEMM.
- The weights scale each row's hidden units, before `down`, as Megatron-Core does.
  `down` is linear, so the sum is the same, and the weights' gradient reads a row's f
  hidden units rather than its d outputs, which saves 1 GiB per layer at
  Qwen3-30B-A3B's shape. The backward recomputes the hidden units, elementwise, from
  up and gate's GEMM, which it keeps anyway (`weighted_units`).
- With a `norm`, the weights scale its output instead.
- Rows past `sizes.sum()` come out 0, before `norm`.

`routed(z, idx, weights, up, down, ffn, norm)` is `sort_by_expert`, then `experts`,
then each token's sum:

- A pick outside the experts of `up` adds nothing, so a device that holds experts
  `[i, i + n)` passes `idx - i` and its own kernels.
- The sort is `token_rows`, which takes only the first m picks. Its gradient gathers by
  the inverse permutation, where XLA's would scatter-add.
- `token_sums`, its transpose, sums each token's k picks' rows.
- `picked_weights` sorts the weights with the picks.

Both take optional arguments past these, by which the MXFP8 kernels read the rows in
place ([The experts in MXFP8](#the-experts-in-mxfp8)).

### Parameter names

| Parameter | Names |
|---|---|
| the router | `(embed, None)` |
| `up`, with a gated MLP's gate | `(experts, embed, mlp, None)` |
| `down` | `(experts, mlp, embed)` |
| `to_latent` (LatentMoE) | `(embed, latent)` |
| `from_latent` (LatentMoE) | `(latent, embed)` |

In a LatentMoE, `latent` takes the place of `embed` in `up` and `down`.

`up` holds a gated MLP's gate too, each hidden unit's up and gate columns side by
side, `(E, d, f, 2)`:

- It reshapes for free to the `(E, d, 2f)` of one GEMM.
- Where `tensor` splits the hidden units, each device keeps an up column with its
  gate's, which `(E, d, 2, f)` would not.
- Muon orthogonalizes up and gate separately, as two matrices.

`dense` constrains its hidden units to `(batch, seq, experts, mlp)`, so that under
FSDP the weights move to the tokens rather than the tokens to the weights.

## Measurements

All on B300s, 2026-10-23 to 2026-10-26, except where a table says CPU. The benchmarks'
MoE is the baseline's: `d_model` 2048, 16 layers, 64 experts of 704, top 8, bfloat16,
32k tokens per GPU, and a capacity of 1.25 for EP
([Benchmarks](../performance/benchmarks.md)). On 8 B300s every method computes what
one GPU does, to 7e-7 in float32 (`scripts/gpu_smoke/experts.py`). So do gpt-oss's
experts with their biases, which every method adds, `nccl_ep` too: the layer within
7.7e-7, and 12 steps' losses within 1.2e-4.

### One node and two

The baseline MoE by `scripts/benchmark.py` (`one_node.bux` and `two_nodes.bux`, their
`moe` and `moe_ep` arms, the method and mesh as below), 2026-10-24. Tokens/s, peak
memory, and the loss at step 30 of 30 (at step 5, 12.233 on one node and 12.222 on
two, for every method):

| GPUs | Method | Mesh | tok/s | Peak | Loss |
|---|---|---|---|---|---|
| 8 | `sorted` | `fsdp=8` | 831k | 83.8 GiB | 11.843 |
| | `ragged_all_to_all` | `expert=8` | 544k | 88.4 GiB | 11.842 |
| | `nccl_ep` | `expert=8` | 615k | 89.3 GiB | 11.867 |
| 16 | `sorted` | `data=2, fsdp=8` | 1,586k | 83.8 GiB | 11.820 |
| | `ragged_all_to_all` | `data=2, expert=8` | 1,016k | 87.0 GiB | 11.825 |
| | `nccl_ep` | `data=2, expert=8` | 1,192k | 89.3 GiB | 11.822 |
| | `ragged_all_to_all` | `expert=16` | 324k | 84.4 GiB | 11.836 |
| | `nccl_ep` | `expert=16` | 773k | 85.0 GiB | 11.844 |

- `sorted` is the fastest of all at this shape.
- EP inside a node with `data` across nodes is the faster EP: `nccl_ep` is 17% faster
  than `ragged_all_to_all` side by side (1,158k and 1,023k in another pair of runs).
- EP over both nodes is 2.4x faster by `nccl_ep` than by `ragged_all_to_all`, but at
  63% of a node's tokens/s per GPU.

Every method on one node, 2026-10-23, under `remat=none` and `remat=dots`:

| Method | Mesh | `remat=none` | `remat=dots` |
|---|---|---|---|
| `sorted` | `fsdp=8` | 816k, 84 GiB | 737k, 52 GiB |
| `all_gather` | `expert=8` | 612k, 89 GiB | 515k, 65 GiB |
| `ragged_all_to_all` | `expert=8` | 550k, 88 GiB | 515k, 78 GiB |
| `ragged_tokens` | `expert=8` | 547k, 90 GiB | 479k, 74 GiB |
| `all_to_all` | `expert=8` | 559k, 89 GiB | |
| `all_to_all`, `moe_chunks=2` | `expert=8` | 574k, 89 GiB | |

`ragged_tokens` moves about 30% fewer rows than `ragged_all_to_all` when the routing
is random, and half as many or fewer where a group limit keeps each token on half the
devices (`num_groups` the size of `expert`, `top_k_groups` half of it). It also sends
the picks and their weights, so here it gains nothing.

### What bounds EP on one node

Each of the baseline MoE's EP layers makes four exchanges of 1.1 to 1.3 GB per GPU:
150 ms of a step at NCCL's 500 GB/s, nearly all of it exposed. `sorted`'s FSDP gathers
0.5 GB of experts per layer and reduces their gradients, 66 ms that hide behind
compute. Chunks hide a third of the exchange, but a GEMM beside NCCL's kernels runs a
third slower and gives most of it back. Side by side on one node, `sorted` trains at
835k tokens/s, `ragged_all_to_all` at 539k, and `all_to_all` in 2 chunks at 586k.
EP would beat `sorted` here only with an exchange near NVLink's 900 GB/s, on a few
SMs, without the capacity's padding, and hidden behind attention as well as the
experts.

What does not help, on 8 B300s with jax 0.12:

- DeepSeek's two overlapped microbatches (unrolled by hand): no better than chunks.
- Fewer NCCL channels (`NCCL_MAX_P2P_NCHANNELS=8`): 349k. A high-priority stream for
  collectives: no change.
- A collective-permute per peer instead of the all-to-all: 394k.
- `--xla_gpu_experimental_ragged_all_to_all_use_device_kernel`,
  `--xla_gpu_ragged_all_to_all_mode=peer`: no change; `=symmetric` fails in NCCL.
- `compiler(preset=lhs)` ([XLA's options](../performance/compiler.md)): out of memory
  at 32k tokens per GPU.
- `jax.lax.ragged_dot` for the experts: `sorted` 20x slower; with
  `--xla_gpu_experimental_use_ragged_dot_fusion` it fails to compile.

::::{dropdown} Where a step's time goes
Profiled, in ms of a step per GPU, 2026-10-23. "Exposed" is the log's
`exposed_comm_ms`, the collectives with no compute beside them.

| Method | tok/s | Collectives | Exposed | Experts' GEMMs |
|---|---|---|---|---|
| `sorted`, `fsdp=8` | 830k | 66 | 7 | 110 |
| `ragged_all_to_all` | 546k | 167 | 152 | 116 |
| `all_to_all` | 559k | 165 | 143 | 117 |
| `all_to_all`, `moe_chunks=2` | 574k | 208 | 90 | 153 |
| `all_to_all`, 2 microbatches, unrolled | 562k | 219 | 98 | 155 |

A collective-permute per peer (in turn i, each device sends the i-th after it its
buffer and runs its experts on the one it gets) hides half its time, but seven of them
take twice as long as one all-to-all, by NCCL or by copy engines
(`--xla_gpu_use_memcpy_local_p2p`).
::::

### One layer's exchange

One layer's dispatch and combine at the baseline MoE's shapes, its experts the
identity, on B300s of 148 SMs, 2026-10-24. "Sent" is what an exchange sends off a GPU
(on two nodes, off its node):

| GPUs | Exchange | SMs | Forward | Forward and backward | Sent |
|---|---|---|---|---|---|
| 8, a node | `ragged_all_to_all` | all | 3.95 ms | 8.20 ms | 0.94 GB, 476 GB/s |
| | `ragged_tokens` | all | 4.73 | 9.55 | 0.64 GB, 270 GB/s |
| | NCCL EP | 32 | 3.85 | 7.50 | 0.64 GB, 332 GB/s |
| | NCCL EP | 64 | 3.51 | 6.80 | 0.64 GB, 364 GB/s |
| 16, two nodes | `ragged_all_to_all` | all | 38.0 | 76.8 | 0.54 GB |
| | `ragged_tokens` | all | 39.4 | 79.6 | 0.45 GB |
| | NCCL EP, GIN by GDAKI | 32 | 25.6 | 80.4 | 0.13 GB |
| | NCCL EP, GIN by a CPU proxy | 32 | 16.0 | 27.0 | 0.13 GB |
| | the same, 256 tokens per chunk | 32 | 13.0 | 21.2 | 0.13 GB |

Inside a node, NCCL EP wins by sending less, not faster: its kernels move 330 to
370 GB/s, far from NVLink's 900. Between nodes it sends a quarter as much over
InfiniBand, and with `NCCL_GIN_TYPE=2 NCCL_EP_TOKENS_PER_CHUNK=256`, bux's defaults,
its exchanges take 28% as long as with NCCL's.

### MXFP8 experts on B300s

`scripts/benchmark.py` on the arms of `examples/benchmarks`, against bfloat16 in the
same job, GEMM autotuning off and deterministic ops on, 2026-10-24. Tokens/s and peak
memory:

| Run | bfloat16 | MXFP8 |
|---|---|---|
| the baseline MoE, one GPU, 8 × 4096 tokens (`one_gpu.bux`'s `moe`) | 86.2k, 133.9 GiB | 95.0k (1.10x), 119.5 GiB |
| experts of 2048, 8 layers, one GPU | 90.4k, 143.2 GiB | 101.7k (1.12x)†, 129.6 GiB |
| `ragged_all_to_all`, `expert=8` (`one_node.bux`'s `moe_ep`) | 468k, 88.5 GiB | 539k (1.15x), 72.3 GiB |
| `all_to_all`, `expert=8` | 492k, 91.4 GiB | 539k (1.10x)†, 78.4 GiB |
| `sorted`, `fsdp=8` (`one_node.bux`'s `moe`) | 671k, 83.9 GiB | 743k (1.11x), 69.4 GiB |

† Measured with an earlier quantizer, under which a layer's experts took 6.1 ms
rather than 5.0.

- An MoE layer's experts, forward and backward, take 5.0 ms against 7.3 in bfloat16
  (1.45x), and 11.8 against 19.7 with experts of 2048 (1.67x;
  `scripts/gpu_smoke/grouped_mxfp8.py`). The GEMMs take half of bfloat16's time,
  3.0 ms per layer against 5.8, and the quantizer gives 1.5 ms of that back.
- Both MXFP8 paths compute the same values: the runs' losses agree to the last digit
  logged, and with bfloat16's to the rounding between them.
- The experts keep x^T in MXFP8 for dW rather than x in bfloat16, so a run takes
  14 GiB less.
- It trains at bfloat16's loss: 98M tokens of FineWeb-Edu end 0.13% below it, and
  0.05% below with FP8 dispatch as well ([Precision](../training/precision.md)).

::::{dropdown} Each GEMM and the quantizer on one B300
Each GEMM of the baseline MoE's experts (64 experts, `d_model` 2048, intermediate
704, 262144 picks), in TFLOP/s of the GEMM's own kernel by CUPTI:

| GEMM | bfloat16, tokamax | MXFP8, `grouped_mxfp8` |
|---|---|---|
| x W, up and gate | 1269 | 2405 (1.9x) |
| h W, down | 1069 | 1596 (1.5x) |
| dy W^T, down | 1213 | 2230 (1.8x) |
| dh W^T, up and gate | 1266 | 2244 (1.8x) |
| dW, up and gate | 1093 | 2573 (2.4x) |
| dW, down | 1034 | 2367 (2.3x) |

- With experts of 2048: 1.7x to 2.3x bfloat16, 2423 to 2616 TFLOP/s against 1139 to
  1414.
- At `all_to_all`'s GEMMs of a peer's buffer under `expert=8` (40960 rows, 32768 of
  them its 8 experts'): 1.3x to 2.7x, 1271 (h W) to 2490 (dW, up and gate) against
  904 to 1213.
- h W contracts 704, which TMA takes in tiles of 128, so `rows` pads it to 768 and
  does 9% more work.

The quantizer, at what one GPU of `megatron.bux`'s `sorted` arm quantizes per layer
(128 experts of 768, 262144 picks), by CUPTI:

| Operand | Shape | TB/s |
|---|---|---|
| x, read by index from 32768 tokens | 262144 × 2048 | 5.8 |
| dy | 262144 × 2048 | 5.2 |
| h | 262144 × 768 | 5.1 |
| dh | 262144 × 1536 | 5.2 |
| each expert's W of up and gate | 128 × 2048 × 1536 | 5.2 |
| each expert's W of down | 128 × 768 × 2048 | 5.1 |

- It reads each operand once and writes it twice, k padded to whole tiles of 128 and
  the scales in tensor memory's tiles, so no XLA pass lays them out after it. x^T's
  blocks run down x's columns, so only its E4M3 bytes transpose.
- A copy that casts x and x^T to E4M3 unscaled runs at 6.5 TB/s (XLA's `x * 2` at
  6.9): the scales and the blocks' largest magnitudes take the rest.
::::

### The fused activations

One MoE layer's experts, forward and backward on a B300, by CUPTI (2026-10-26,
cudnn-frontend 1.31.0). `sorted` takes all the experts and a microbatch's picks, their
kernels quantized in the layer; EP takes one GPU's experts of 8, at a capacity of
1.25:

| The experts | `sorted`: bux's kernels | fused | EP: bux's kernels | fused |
|---|---|---|---|---|
| Nemotron 3 Nano: relu2, 2688 × 1856 (run as 1920); 128, 196608 picks; EP 16 | 9.27 ms | 7.57 (1.22x) | 7.57 | 6.61 (1.15x) |
| Nemotron 3 Super: relu2, 1024 × 2688; 512, 180224 picks; EP 64 | 9.79 | 8.08 (1.21x) | 5.52 | 4.20 (1.31x) |
| gpt-oss without its biases: GeGLU, 2880 × 2880 (as 2944); 128, 65536 picks; EP 16 | 10.97 | 10.56 (1.04x) | 6.38 | 5.55 (1.15x) |
| DeepSeek-V4-Flash: clamped SwiGLU, 5120 × 2304; 384, 98304 picks; EP 48 | 38.47 | 35.71 (1.08x) | 13.50 | 12.38 (1.09x) |
| Qwen3-30B-A3B: SwiGLU, 2048 × 768; 128, 262144 picks; EP 16 | 6.36 | 5.55 (1.15x) | 5.90 | 5.01 (1.18x) |

- The values are within MXFP8's rounding of bux's kernels': 0.7-2.2% apart in the
  output and every gradient (relative Frobenius norm), and as near bfloat16's.
- With few picks per expert, as gpt-oss and DeepSeek-V4-Flash have, quantizing every
  expert's kernels takes much of a `sorted` layer. Under FSDP each GPU quantizes only
  its shard.

In training on one node of 8 B300s, `remat=none` unless noted, bux's kernels and the
fused ones on the same node:

| Run | bux's kernels | fused |
|---|---|---|
| Qwen3-30B-A3B, `sorted`, Megatron-Bridge's batch (512 rows in 8 microbatches, `gather=layer`) | 36.1k tok/s/GPU, 229.6 GiB | 37.7k (+4.5%), 214.6 GiB |
| Qwen3-30B-A3B, `sorted`, one microbatch of 8 rows per GPU | 37.7k, 215.1 GiB | 38.5k (+2.1%), 200.3 GiB |
| Qwen3-30B-A3B, `nccl_ep`, one microbatch of 8 rows per GPU | 31.4k, 211.1 GiB | 32.4k (+3.1%), 212.7 GiB |
| Nemotron 3 Nano, `sorted`, 512 rows of 8192 in 16 microbatches, `gather=layer, remat=dots` | 32.2k, 192.6 GiB | 33.8k (+5.1%), 178.4 GiB |
| Nemotron 3 Super, 26 of its 88 layers, NCCL EP on `expert=8`, a row of 8192 per GPU in each of 8 microbatches | 24.0k, 184.1 GiB | 24.9k (+3.9%), 186.1 GiB |

The Qwen3 rows are from 2026-10-24 (`megatron.bux`), the Nemotron rows from
2026-10-26 (`megatron_nano.bux`'s `sorted` arm; `megatron_super.bux`'s `mxfp8` arm
shortened to fit a node). Nano's `nccl_ep` arm runs out of memory either way. Their
losses agree to four digits at each logged step.

Quality: 3000 steps of FineWeb-Edu on the baseline MoE, one B300 per run, autotuning
pinned and XLA's deterministic ops on, end within 0.08% of bux's kernels (SwiGLU
experts of 768) and 0.11% (ReLU^2 experts of 704, run as 768) at each of three seeds,
which differ by 1.6%. With gpt-oss's clamped GeGLU at one seed: 4.0802 against 4.0772.

::::{dropdown} Per-kernel detail, Qwen3-30B-A3B's experts
- At these shapes cuDNN's GEMM of h W takes 0.44 ms against `grouped_mxfp8`'s 0.53,
  dp W^T 0.65 against 0.77, and its wgrads 0.30 and 0.60 ms against 0.39 and 0.76.
- On GPU 0 at one microbatch, the experts' kernels and XLA's passes over h and dp take
  263 ms of a step against 319 under `sorted`, and 234 against 280 under `nccl_ep`.
- Beside FSDP's collectives cuDNN's kernels run slower than alone: `dswiglu` takes
  1.13 ms against 0.62 (0.59 under `nccl_ep`, which has none). Its tiles are
  statically scheduled, where `grouped_mxfp8`'s take the SMs as they free.
- `CUDNNFE_CLUSTER_OVERLAP_MARGIN` of 8 gains 0.3% at Megatron-Bridge's batch, 16
  gains 1.1%, and 32 loses 2%.
- 3000 steps on FineWeb-Edu at three seeds: eval loss 4.0708, 4.0127 and 4.0346 fused,
  against 4.0742, 4.0099 and 4.0363, at 104k tokens/s against 100k on one B300.
::::

### FP8 dispatch on 8 B300s

The baseline MoE (the one-node suite's `moe_ep`), side by side on one node,
2026-10-24, in tokens/s:

| Method | `dispatch=compute` | `dispatch=mxfp8` |
|---|---|---|
| `ragged_all_to_all` | 539k | 556k |
| `all_to_all`, 2 chunks | 586k | 589k |

A ragged dispatch takes 17 ms less per step, and quantizing a token rather than each
of its picks costs little. `all_to_all`'s dispatch already hides behind the experts.

### Routing by random logits

Qwen3-30B-A3B on random tokens, on 8 B300s (2026-10-24). The router's own routing
drifts uneven there, to a balance loss of 1.4 to 1.8 per layer, and `nccl_ep` at a
capacity of 1.25 drops 10 to 15% of the picks. Routed by random logits
([Random logits for benchmarks](../architectures/experts.md#random-logits-for-benchmarks)), the
balance loss is 1.00 and no pick drops (`dropped` 0.0 every step), with or without
`moe_stream`:

| Method | the router's routing | random logits |
|---|---|---|
| `nccl_ep`, capacity 1.25 | 27.7k tok/s/GPU | 29.6k (+7%) |
| `sorted` | 31.4k | 31.8k (+1.4%) |

### `moe_stream` at Megatron-Bridge's batch

`megatron.bux`'s `nccl_ep` arm (512 rows of 4096, 8 microbatches one after another,
`remat=none`), side by side, 2026-10-24:

| NCCL EP's kernels | tok/s/GPU | Peak |
|---|---|---|
| on the compute stream | 34.1k | 236.1 GiB |
| on their own, `moe_stream=true` | 34.1k | 234.2 GiB |
| the same, `moe_chunks=2` | 33.2k | 239.1 GiB |

- What overlaps in one layer alone does not carry over to a step of 48 layers. Why is
  not known yet: jax's profiler loses most of a step's events at this batch.
- XLA's own all-to-all is no alternative here: `moe=all_to_all` at a capacity of 1.25,
  the tokens sent in MXFP8, trains the arm at 26.6k tokens/s per GPU, and 27.4k in 2
  chunks (244-245 GiB), where NCCL EP took 32.2k on the same code.

::::{dropdown} One MoE layer, with and without `moe_stream`
At `megatron.bux`'s shapes (EP=8, 32k tokens per GPU, a capacity of 1.25, the experts
by cuDNN's MXFP8 kernels), in ms:

| | Compute stream | `moe_stream` | Hints only | The pass only |
|---|---|---|---|---|
| an exchange beside an independent grouped GEMM (2.2 ms) | 5.18 | 4.30 | 4.38 | 4.64 |
| forward | 5.37 | 5.35 | 5.36 | 5.35 |
| forward, 2 chunks | 5.71 | 4.97 | 5.32 | 5.71 |
| forward and backward | 9.39 | 8.07 | 8.90 | 8.18 |
| forward and backward, 2 chunks | 10.06 | 9.55 | 9.35 | 9.83 |
| forward and backward beside 12.6 ms of independent GEMMs | 21.66 | 20.62 | 21.52 | 20.87 |

- The forward of one chunk has nothing to put beside its exchange. The backward does:
  its prepares wait only for the picks, and run beside the forward's experts.
- More chunks cost more than they hide: 4 chunks take 11.29 ms forward and backward.
- The last row is what a schedule of two microbatches, one's backward beside the
  other's forward (`methods(interleave=true)`), gives the exchange: about 1 ms of a
  layer's ~6 ms of exchange hides.
- XLA's cost model prices the experts' and attention's custom calls at 10 µs and an
  async call at 5 ms, so the hints put everything ready beside the last call before
  it. The interleaved schedule's barriers bound what that is.
::::

::::{dropdown} Collectives on 8 CPU devices
The collectives in the compiled train step of the two-MoE model of
`tests/test_experts.py`: their number, and the KiB their results take on one device.

| Mesh | Method | all-gather | all-to-all | reduce-scatter | all-reduce |
|---|---|---|---|---|---|
| `expert=8` | `sorted` | 48, 688 | | | 3, 678 |
| | `all_gather` | 50, 372 | | 6, 8 | 3, 294 |
| | `all_to_all` | 42, 304 | 10, 256 | | 3, 294 |
| `data=2, fsdp=2, expert=2` | `sorted` | 54, 880 | | | 7, 736 |
| | `all_gather` | 56, 513 | | 6, 8 | 8, 544 |
| | `all_to_all` | 48, 496 | 10, 64 | | 8, 544 |
| `fsdp=2, expert=4` | `sorted` | 54, 784 | | | 5, 678 |
| | `all_gather` | 56, 434 | | 6, 8 | 6, 390 |
| | `all_to_all` | 48, 400 | 10, 128 | | 6, 390 |

- `sorted` gathers every expert's kernels and all-reduces their gradients; the others
  move tokens instead.
- Without a capacity, `all_to_all`'s fixed-size buffers move `top_k` times the rows
  `all_gather` does, since every peer's buffer holds every local pick. Under a
  capacity it moves `capacity` times the picks, where `ragged_all_to_all` moves only
  the picks.
- No method rematerializes anything on these meshes, nor on
  `fsdp=2, expert=2, tensor=2`.
::::
