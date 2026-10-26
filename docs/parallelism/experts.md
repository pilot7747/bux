(parallelism-experts)=
# Mixture of experts and expert parallelism

An MoE layer's router picks each token's top-k experts, and a method of
`methods.moe` runs them. By default, `sorted` runs each expert on its own picks on
every device, with the experts' kernels gathered by FSDP, so `expert` acts as one
more batch axis. Expert parallelism (EP) splits the experts themselves over the
`expert` axis instead: each device holds its own experts and runs them on the picks
that every device of its group sends it.

Use this page to choose how an MoE trains on many GPUs:

- **On one node**, keep the default: at the baseline MoE's shape `sorted` under FSDP
  is the fastest method bux has (831k tokens a second on 8 B300s, against 615k by
  `nccl_ep`, the fastest EP), as its FSDP gathers move less than EP's exchanges.
- **Across nodes**, `sorted` under HSDP stays the fastest; for EP, keep `expert`
  inside a node and `data` across them, and send by `nccl_ep` where
  TransformerEngine is installed.
- **On B300s**, multiply the experts in MXFP8 (`precision(matmuls=mxfp8)`): they
  train 10 to 15% faster than in bfloat16.

## Turn it on

`examples/experiments/parallel/ep_moe.bux` trains a 16B MoE in DeepSeek-V3's shape on
2 nodes of 8 GPUs, its experts split 8 ways inside a node and FSDP across the nodes:

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

The benchmark suites run EP on one node of 8 GPUs as follows: a ragged all-to-all at a
capacity of 1.25 (`examples/benchmarks/one_node.bux`'s `moe_ep`), and NCCL EP over
microbatches with the matmuls in MXFP8 (`examples/benchmarks/megatron.bux`'s
`nccl_ep`):

```bux
mesh    = mesh(expert=8)
methods = methods(moe=ragged_all_to_all, capacity=1.25)
```

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
| `methods.capacity` | a number above 0, or `none` | `none` | the picks a device takes, as a multiple of its own; what does not fit drops ([A capacity](#a-capacity)) |
| `methods.moe_chunks` | a count | `1` | the slices of each device's tokens EP sends one after another ([In chunks](#in-chunks-methodsmoe_chunks)) |
| `methods.moe_stream` | `true`, `false` | `false` | NCCL EP's kernels on a GPU stream of their own ([On a stream of its own](#on-a-stream-of-its-own-methodsmoe_stream)) |
| `precision.roles.experts` | `compute`, `mxfp8`, `mxfp8_2d`, `none` | `none` | the format the routed experts multiply in; `none` is `matmuls` where a grouped GEMM takes it ({bux}`roles`, [The experts in MXFP8](#the-experts-in-mxfp8)) |
| `precision.dispatch` | `compute`, `mxfp8` | `compute` | what the exchanges send tokens to their experts in ({bux}`precision`, [FP8 dispatch](#fp8-dispatch-precisiondispatch)) |

The methods of `methods.moe`:

| Method | Splits the experts | What it does | Runs on |
|---|---|---|---|
| `sorted` | no | The default. Sorts the picks by expert, runs each expert on its rows alone, and unsorts; drops no token. | any device |
| `dense` | no | Every expert on every token, weighted by the router: the reference. | any device |
| `all_gather` | yes | Each device gathers its group's tokens, runs its own experts on them, and a `psum_scatter` brings each token its sum back. | any device |
| `all_to_all` | yes | Each pick goes to its expert's device and back by `jax.lax.all_to_all`, in a buffer of one size for each peer, sorted by expert. | any device |
| `ragged_all_to_all` | yes | Each pick goes to its expert's device and back by `jax.lax.ragged_all_to_all`, a slice for each expert of each peer. | GPUs: XLA:CPU lacks the ragged all-to-all |
| `ragged_tokens` | yes | As `ragged_all_to_all`, but each token goes once to each device that holds any of its experts, and comes back as their weighted sum. | GPUs, as `ragged_all_to_all` |
| `nccl_ep` | yes | Each token goes once to each device that holds any of its experts, and once to each other node, by NCCL EP's kernels through TransformerEngine. | GPUs, with TransformerEngine ([NCCL EP](#transformerengines-nccl-ep)) |

## Constraints and errors

- Where no axis splits the experts (`expert=1`, or a number of experts that `expert`
  does not divide), every EP method runs as `sorted`.
- Experts over an axis that leaves the tokens whole, as `layout(experts=tensor)` does,
  split the tokens over it too, inside the method. Rows that do not split over it are
  an error.
- `roles(experts=nvfp4)` is an error: no grouped GEMM takes NVFP4.
- A capacity that drops picks changes what a run computes, unlike the rest of
  `methods`, though it stays out of the run's id. `precision.dispatch = mxfp8` changes
  what a run computes too, and is in the run's id.
- `nccl_ep` runs on GPUs only, and needs TransformerEngine's JAX bindings, which the
  Docker image has. It takes one MoE shape a run, the experts over one axis, and
  tokens in bfloat16: `precision(compute=bfloat16, dispatch=compute)`. MXFP8 dispatch
  is an error, and so is a run without TransformerEngine.
- On a CPU, the MoE's `sorted` and EP methods make a bfloat16 `psum` on more than one
  device, which XLA:CPU cannot compile: such runs set `precision(compute=float32)`, or
  split an MoE by `methods(moe=dense)` ([Precision](precision.md#precision)).

:::{warning}
Keep `expert` inside a node for now: across hosts, XLA splits a ragged all-to-all
into an intra-host one and an all-gather of the tokens across hosts.
:::

:::{warning}
On jax 0.12, NCCL's ragged all-to-all hangs unless `XLA_FLAGS` has
`--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`. `bux train`
and `bux launch` add it, and so does `bux.runtime.configure()` in a script that calls
it before it touches a device.
:::

## How an MoE layer runs

### Routing, then a method

The router runs in the global view. It picks each token's top-k experts, reports the
balance loss and moves the router bias. A method of `methods.moe` then runs the picked
experts:

```text
(moe, z, idx, weights, taps) -> y
```

| Argument | What it is |
|---|---|
| `z` | the tokens `(b, t, d)`, in the experts' width: the latent one in a LatentMoE |
| `idx`, `weights` | each token's picks and their weights, `(b, t, k)` |
| `taps` | the forward's taps, through which a method reports what it measures, as the ragged methods report `dropped` ([A capacity](#a-capacity)) |
| `y` | the picks' weighted sum `(b, t, d)`, before `latent_output_norm` |

### `sorted` and `dense` on a mesh

On a mesh, `sorted` sorts each device's own tokens, in a `shard_map` over the axes
that split rows and positions, with every expert's kernels gathered onto every
device. `expert` then acts as a batch axis, as it does under `dense`. Splitting the
experts themselves is expert parallelism, a method of its own
([below](#expert-parallelism)).

### The pieces: `experts` and `routed`

The pieces of a token's path are functions of arrays alone, in
`bux.architecture.experts.moe`, so that a method can run them on one device's experts
inside a `shard_map`.

`experts(x, sizes, weights, up, down, ffn, norm)` runs the expert FFN on rows sorted
by expert:

- `sizes[i]` of the rows are expert i's, and each row's output is times its weight.
- The products run by `grouped.ragged_dot`: on a GPU tokamax's Mosaic kernel, and its
  Triton one for the kernels' gradient (27 times XLA's `jax.lax.ragged_dot` on a
  B300); elsewhere XLA's.
- Up and gate are one GEMM.
- The weights multiply each row's hidden units, before `down`, as Megatron-Core does.
  `down` is linear, so that is the same sum. The weights' gradient then takes a row's
  f hidden units, where after `down` it took its d outputs: 1 GiB a layer of
  Qwen3-30B-A3B's. The backward computes the hidden units again, elementwise, from up
  and gate's GEMM, which it keeps for the activation's gradient anyway
  (`weighted_units`; [measured](#the-routers-weights-before-down)).
- After `norm`, where there is one, the weights multiply its output instead.
- Rows past `sizes.sum()` come out 0, before `norm`.

`routed(z, idx, weights, up, down, ffn, norm)` is `sort_by_expert`, `experts`, and
each token's sum:

- A pick outside the experts of `up` adds nothing, so a device that holds experts
  `[i, i + n)` passes `idx - i` and its own kernels.
- The sort is `token_rows`, which takes the first m picks alone. Its gradient gathers
  by the inverse permutation, where XLA's would scatter-add.
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
  gate's, as `(E, d, 2, f)` would not.
- Muon orthogonalizes up and gate apart, as two matrices.

`dense` constrains its hidden units to `(batch, seq, experts, mlp)`, so that under
FSDP the weights move to the tokens rather than the tokens to the weights.

### Writing your own method

A method of `methods.moe` is a function of the layer, its tokens and their picks:

```python
from bux.architecture.experts.moe import routed
from bux.parallel import methods


@methods.moe.register("mine")
def _(moe, z, idx, weights, taps=None):
    return routed(z, idx, weights, *moe.kernels(), moe.config.expert, moe.expert_norm)
```

`methods = methods(moe=mine)` then picks it. An EP method is a body of one device's
tokens instead ([below](#writing-your-own-ep-method)).

## The experts in MXFP8

Three ways to set the experts' format, one line each:

```bux
precision = precision(matmuls=mxfp8)                          # the experts too
precision = precision(roles=roles(experts=mxfp8))             # the experts alone
precision = precision(matmuls=mxfp8, roles=roles(experts=compute))
```

The routed experts' role, `experts`, takes `compute`, `mxfp8` or `mxfp8_2d`
([Precision](../training/precision.md)); `nvfp4` is an error. Left out, it is
`matmuls` where a grouped GEMM multiplies in it, `mxfp8` and `mxfp8_2d`, and `compute`
under any other format, as under `nvfp4`: no grouped GEMM takes NVFP4
(`PrecisionConfig.experts`). On B300s MXFP8 experts train 10 to 15% faster than
bfloat16 ones ([below](#in-training-on-b300s)).

:::{note}
Before this, an MoE run that left its experts to `matmuls=mxfp8` multiplied them in
bfloat16. So such a run now takes the id of a run that names `mxfp8` for them, and a
launch trains it again. A run that names their format keeps its id, and so does a
model without routed experts.
:::

### How the experts multiply in a format

- Up and gate's one GEMM quantizes x once.
- Each GEMM runs by `grouped_blockscaled.ragged_dot`. Every operand of the forward's
  GEMM and of the backward's two is quantized along what that GEMM contracts, as a
  projection's are ([Matmul formats](precision.md#matmul-formats)), and each expert's
  W on its own.
- One pass of a Triton kernel (`grouped_triton`) quantizes x and x^T, zeroes the rows
  past the experts' (which an exchange's buffer leaves as they were), and writes both
  as the kernels below take them.
- Under `sorted`, x's rows are the picks' token rows. Given where they were picked
  (`grouped_blockscaled.Picked`), the kernel reads each from the tokens by index, so
  that the copy of them `moe.token_rows` gathers is never written (XLA drops it): 1
  GiB a layer at 32k tokens a GPU and 8 picks a token.
- A GEMM whose rows sum into their tokens (`Summed`) returns the tokens' sums, and its
  backward reads each row's dy from its token's the same way.
- Off a Blackwell GPU, XLA dequantizes and multiplies, which computes the same values,
  so the tests run it on CPU.
- Only the GEMMs of `precision.passes` multiply in the format; the others run by
  `grouped`'s kernels, as they are.

### The kernels: `grouped_mxfp8`

On a Blackwell GPU, tcgen05's block-scaled MMA multiplies them, in the Mosaic GPU
kernels of `grouped_mxfp8`: `rows` for x W and dy W^T, and `contract` for dW.

- A pair of SMs multiplies each tile, 256 rows by 128 columns, by tcgen05's MMA of two
  SMs.
- Each pair takes tiles one after another, in bands of 8 tiles of rows, down a band's
  rows first. The tiles in flight then share few rows of one operand and columns of
  the other, which stay in L2.

Three roles share a tile's steps along k:

| Role | Does |
|---|---|
| a warp | loads each step's tiles of both operands and their scales by TMA into a ring of 7 buffers, the scales laid out as tensor memory takes them, 128 rows by 4 scales a 512-byte tile |
| a warp | copies the step's scales on into tensor memory (`tcgen05.cp`) and issues its MMAs into one of two accumulators there |
| a warpgroup | stores the other accumulator in the output's dtype, so that a tile's store overlaps the next one's MMAs |

- `rows` takes the 256 rows an expert has rows in, so a tile two experts share runs
  once for each, and each run stores only its own rows. It stores the rows past the
  experts as 0.
- `contract` takes an expert's rows in tiles of 128 along what dW contracts. The first
  and last tiles of x^T, which it may share with its neighbours, come from copies with
  the neighbours' rows as 0, so that dy's add nothing there.

### Blocks that two experts share

Unlike a projection's, the experts' dW quantizes the rows a device has of an expert in
blocks of 32 as they lie: a block that two experts share has one scale, and each
counts the other's rows as 0. Which rows share a block, and so dW, depend on how the
mesh splits the tokens. In float32, 3 steps on `fsdp=2, expert=4` lose within 1e-4 of
one device's (`tests/test_precision.py`), not the same.

## SwiGLU fused, by cuDNN

On a Blackwell GPU with cudnn-frontend (the `cuda` extra), cuDNN's CuTe DSL kernels
run the routed experts, through `cudnn.jax.call` on XLA's stream (`grouped_glu`, over
`grouped_cudnn`), where all of these hold:

- the experts are a SwiGLU FFN: `gated_mlp` of `swish`, with no `limit`,
  `up_softcap`, `up_offset`, `bias` or `expert_norm`;
- they multiply in `mxfp8`, every pass in it;
- their widths are whole tiles of 256: Qwen3-30B-A3B's 2048 and 768, but not the
  baseline's 704.

They fuse what bux's kernels left to XLA:

| GEMM | Kernel | In its epilogue |
|---|---|---|
| x W, up and gate | `swiglu` | SwiGLU, times each row's weight, quantized along h's rows for down and down its columns for down's dW; the product kept for the backward |
| h W, down | `rows` | |
| dy W^T, down | `dswiglu` | SwiGLU's gradient at the kept product, times the row's weight, quantized both ways; the weight's gradient |
| dp W^T, up and gate | `rows` | |
| dW, both | `contract`, cuDNN's block-scaled wgrad | |

### How the fused kernels take their rows

- Their tiles are 256 by 256 on a pair of SMs, where `grouped_mxfp8`'s are 256 by 128.
- They take each expert's rows from a multiple of 256. The quantizer reads each row
  from its token where it lies and writes the padding as 0, and the tokens' sums read
  the rows back.
- An exchange's rows come out as they lay, by `grouped_mxfp8.rows`, which reads each
  expert's rows from its own start and packs its output: otherwise it would take a
  gather each way.
- Their GLU takes each gate 32 columns before its up, where up and gate's stack holds
  them side by side. The quantizer reads W_up's pairs so, and the transpose of the
  product's gradient for up's dW writes them back.
- Under FSDP, where a layer gathers its parameters as it runs (one microbatch, or
  `methods(gather=layer)`: [Gathered once, reduced once](microbatches.md#gathered-once-reduced-once-methodsgather-methodsaccumulate)),
  each GPU quantizes its own shard of the experts' kernels, and they are gathered in
  MXFP8: an eighth of the quantizer's work on `fsdp=8`, at bfloat16's bytes, and no
  bfloat16 copy of all 128 experts.

At Megatron's batch they train Qwen3-30B-A3B's `sorted` arm 4.5% faster in 15 GiB
less ([The fused SwiGLU](#the-fused-swiglu)).

## Expert parallelism

Five methods of `methods.moe` split the experts over the axes `layout.experts` gives
them, `expert` by default. Each device holds its own experts' kernels, and runs them on
the picks of every device of its group.

```bux
mesh    = mesh(fsdp=2, expert=4)
methods = methods(moe=ragged_all_to_all, capacity=1.25)
```

| Method | What it does |
|---|---|
| `all_gather` | Each device gathers its group's tokens, runs its own experts on them, and a `psum_scatter` brings each token its sum back. |
| `all_to_all` | Each pick goes to its expert's device and back by `jax.lax.all_to_all`, in a buffer of one size for each peer, sorted by expert. A device runs its experts on each peer's buffer as it arrives, a grouped GEMM a peer. |
| `ragged_all_to_all` | Each pick goes to its expert's device and back by `jax.lax.ragged_all_to_all`, a slice for each expert of each peer. A device's rows arrive sorted by its experts, which run on them as they are. XLA:CPU lacks it. |
| `ragged_tokens` | The same, but each token goes once to each device that holds any of its experts, with its picks there, and comes back as their weighted sum: a row per device, not per pick. |
| `nccl_ep` | Each token goes once to each device that holds any of its experts, and once to each other node, by NCCL EP's kernels through TransformerEngine's JAX bindings. It arrives as a row for each of its picks there, grouped by expert. GPUs only ([below](#transformerengines-nccl-ep)). |

How fast each trains, and what bounds them on one node, is in
[The EP methods](#the-ep-methods).

### How the EP methods run

- Each runs in a `jax.shard_map` manual over the axes that split the tokens (those of
  `batch` and `seq`), not over `expert` alone: a sort of tokens split over an Auto
  axis gathers them all onto every device.
- The kernels enter split over the experts' axes and whole over the others, so FSDP
  gathers a device's own experts alone. `tensor` stays Auto.
- Routing, the balance loss and the router bias run before, in the global view, and do
  not change.
- Each pick's weight goes with it to its expert's device: gathered with the tokens
  (`all_gather`), beside its token (`all_to_all`, `ragged_all_to_all`), with the
  token's picks there (`ragged_tokens`), or in NCCL EP's dispatch (`nccl_ep`).
  `experts` weights the rows it makes, so what comes back is summed as it is, and
  nothing the backward keeps for the weights is d wide.

### A capacity

```bux
methods = methods(moe=ragged_all_to_all, capacity=1.25)
```

Dropless, a device of `all_gather` or of the ragged methods takes what every peer
could send it: `expert` times the picks it has, and its experts run on all those rows.
`methods.capacity` bounds them:

- A device takes at most `capacity` times its own picks (its tokens times `top_k`),
  about what it takes on average, and drops what does not fit: its last experts' picks
  first, and of an expert's, the last peers'.
- `ragged_tokens` takes as many tokens as picks, or as many as could come, a token
  from each device.
- `all_to_all` bounds each of a device's buffers, one a peer, at `capacity` times a
  peer's even share of its picks (its picks over `expert`), and drops what does not
  fit there, that peer's last experts' picks first. Dropless, a buffer holds every
  pick, so a device sends `expert` times the rows it has.
- A dropped pick adds nothing to its token.
- A training step reports the share of the picks dropped as `dropped`, summed over the
  MoE layers, as `balance_loss` is.

:::{note}
A capacity that drops anything changes what a run computes, unlike the rest of
`methods`, though it stays out of the run's id.
:::

The buffers that receive are `jax.lax.empty`: the experts make 0 of the rows past
theirs, and the rows that do not come back are masked. XLA's one-shot ragged
all-to-all, on one node, copies each slice with one block of 128 threads, so each
exchange goes in 2048 pieces at least (`experts.PIECES`). On 8 B300s a slice for each
expert of each peer moved 90 GB/s, and the pieces 600 GB/s.

### Writing your own EP method

An EP method is a body that runs on one device's tokens, named as the method is in
`bux.parallel.experts`, and `experts.expert_parallel` makes it one of `methods.moe`:

```python
from bux.parallel import experts, methods


def mine(z, idx, weights, kernels, *, axes, ffn, norm):
    # This device's tokens and picks, and its experts' kernels: experts
    # [i * n, (i + 1) * n) for n = len(kernels[0]), i = jax.lax.axis_index(axes).
    # Return the tokens' sums, and the share of the group's picks it dropped.
    ...


methods.moe.register("mine")(experts.expert_parallel(mine))
```

`expert_parallel` runs the body in the `shard_map` above, in `methods.moe_chunks`
slices, and runs `sorted` where no axis splits the experts.

`experts.ragged_all_to_all` and `experts.ragged_tokens` take the ragged exchange as
`exchange=`, so another collective is a function. The tests run them on CPU through
one built from `all_to_all`, and `scripts/gpu_smoke/experts.py` runs the real one, and
every other method, on GPUs.

## In chunks: `methods.moe_chunks`

```bux
methods = methods(moe=all_to_all, capacity=1.25, moe_chunks=2)
```

An EP layer waits for four exchanges: the picks there and back, and the backward's
two. `methods.moe_chunks` cuts each device's tokens into that many slices, which a
method runs one after another, each on its own (`experts.in_chunks`), so that the
compiler can send one slice's picks while another slice's experts run.

- Where it does not divide a device's tokens, it takes as many slices as do.
- A slice's experts, and each peer's under `all_to_all`, add their part of the
  kernels' gradients, which add up in float32, as one call's would.
- Each chunk is a grouped GEMM a peer a layer more to compile: for the baseline's MoE
  on 8 B300s, `moe_chunks=2` compiles in 5 minutes, and 4 in more than 8.
- Only `all_to_all` gains, 3% on one node
  ([Why only `all_to_all` gains in chunks](#why-only-all_to_all-gains-in-chunks)).

## FP8 dispatch: `precision.dispatch`

```bux
precision = precision(dispatch=mxfp8)
```

The exchanges send the tokens to their experts in MXFP8, as DeepSeek-V3 does: E4M3
values and a power-of-two scale per 32, d + d/32 bytes a row, 52% of bfloat16's at
d = 2048, dequantized to the compute dtype where they arrive.

- Each token quantizes once, before the rows of its k picks are picked, so the
  quantizer reads a k-th of the rows and the gather writes half the bytes.
- What comes back, and the backward's two exchanges, go in `compute`.
- The gradient is the exchange's, as if the tokens went unquantized
  (`experts._picked_in_mxfp8` for the ragged methods, `experts._swapped_in_mxfp8` for
  `all_to_all`).
- Rows that are not whole blocks go as they are.
- It changes what a run computes, so it is in the run's id. `nccl_ep` does not take
  it ([Constraints](#constraints-and-errors)).

The exchanges name what they bring `experts.exchanged`, under `experts`, so that
`remat=dots`, which keeps `experts`, does not exchange them again in the backward
([Repeats and remat](remat.md)).

On one node `ragged_all_to_all` trains at 556k tokens a second by it, against 539k
([FP8 dispatch on 8 B300s](#fp8-dispatch-on-8-b300s)).

## TransformerEngine's NCCL EP

```bux
mesh    = mesh(expert=16)                        # over two nodes
methods = methods(moe=nccl_ep, capacity=1.25)
```

TransformerEngine (2.18 on, `transformer_engine.jax.ep`) dispatches and combines by
NCCL EP, a DeepEP of NCCL's device API: loads and stores over NVLink in a node, RDMA
that the GPU starts (NCCL's GIN) between nodes, and no NVSHMEM.

- A token goes once to each GPU that holds any of its experts, as in `ragged_tokens`,
  and once to each other node, whose NVLink takes it on.
- It arrives as a row for each of its picks there, grouped by expert, each expert's
  rows padded to 128.
- Since 2.20 one process drives all its node's GPUs, on a communicator TE borrows from
  XLA.
- The Docker image has it (2.20.2, by `--build-arg TE_VERSION`).
- On one node it is the fastest EP (615k tokens a second for the benchmarks' MoE,
  against `sorted`'s 831k), and over two nodes 2.4x `ragged_all_to_all`
  ([NCCL EP](#nccl-ep)).

### How `nccl_ep` runs

`nccl_ep` runs TE's one-device primitives in the `shard_map` the other methods run in,
a `jax.custom_vjp` each way (`parallel/nccl_ep.py`). The experts take the rows as they
arrive, their blocks padded to 128 as they are.

**Buffers, sized once.** NCCL EP keeps its buffers outside the step. They are sized
for the most tokens a device sends at once, a chunk of its share of a microbatch's
(`methods.moe_chunks`, `methods.microbatches`), and for the rows `methods.capacity`
lets it take. So `methods.prepare` starts it once a process, before the step traces
(`nccl_ep.start`): 25 to 60 s between two nodes.

- An eval step, which takes a step's rows whole, sends them in pieces of that many,
  one after another.
- Sized by the step's tokens, `megatron.bux`'s batch of 512 rows asked for a 29 GiB
  staging buffer a GPU and did not start. Sized by a microbatch's, it trains at 32.2k
  tokens a second a GPU on 8 B300s (`remat=none`, 235 GiB).
- Every dispatch brings the rows it was started for: NCCL EP takes no fewer, and fails
  with "invalid argument" where they are fewer.

**Routes and handles.** TE keeps its own state of a route by the address of the handle
its prepare writes the route into, which a copy changes. So no handle outlives the
pass it is prepared in: the combine takes its dispatch's, and each call of a backward
prepares the route again from the picks its forward kept (1 MiB a layer, where the
handle is 37).

- A backward on its forward's handle gave wrong gradients, with no error, where the
  residuals were copied, as `methods(interleave=true)` keeps them
  ([Side by side](microbatches.md#side-by-side-methodsinterleave)): on 8 B300s, a
  gap of 1.0 against each microbatch's forward and backward in turn. Now they agree
  within float error, in any order of two microbatches' forwards and backwards.
- On the compute stream the backward's prepares cost nothing measurable (32.6k tokens
  a second a GPU before and after).

**Under a capacity**, the picks that do not fit drop before NCCL EP sends them, as
`ragged_all_to_all` drops them (`nccl_ep._kept`). With NCCL EP's own drops, EP=16's
balance loss rose to 35 in 25 steps and its loss came off `sorted`'s. The weights of
the padding are garbage, so each expert's count of the picks kept for it says which
rows hold picks.

**Settings.** bux sets `NCCL_GIN_TYPE=2` (GIN by a CPU proxy) and
`NCCL_EP_TOKENS_PER_CHUNK=256` where the environment does not (`runtime.configure`).
On two B300 nodes, GDAKI, NCCL's default, took 3x as long for an EP layer's
exchanges, and chunks of 64 tokens 1.3x.

**Limits.** It takes one MoE shape a run, the experts over one axis, and bfloat16:
MXFP8 dispatch is an error, as is a run without TE.

**Tests.** `tests/test_nccl_ep.py` runs a token's path on CPU through an emulation of
TE's primitives, against `sorted`, and under a capacity against `ragged_all_to_all`.
On 16 GPUs of two nodes, EP=16, a layer's output and gradients come within bfloat16's
rounding (4e-3 to 8e-3 of the largest) of `sorted`'s dropless, and of
`ragged_all_to_all`'s at a capacity of 1.

### What NCCL EP takes from JAX

- Across nodes only its one-device primitives ran, as `nccl_ep` runs them: its
  global-view ops (`ep_dispatch`, `ep_combine`) hung at their first dispatch, on
  XLA's collective stream or off it.
- Those global-view ops run on XLA's collective stream, where on jax 0.12 a
  resharding next to them fails to compile ("No ExecutionStreamId found"): what they
  take and give must stay split over the experts' axis.
- Its core library loads the NCCL that jax's GPU backend loads, so it imports after
  `jax.devices()`.
- An expert's rows past its picks are 0, up to its 128, but their weights are garbage,
  as are the rows past the last expert's. It sends bfloat16 only.

### On a stream of its own: `methods.moe_stream`

```bux
methods = methods(moe=nccl_ep, capacity=1.25, moe_stream=true)
```

NCCL EP's kernels go on the compute stream, where nothing runs beside them. With
`moe_stream` they run on one of their own (`nccl_ep.STREAM`). Each of TE's calls is
an async call there (`jax_internals.on_stream`;
`--xla_gpu_experimental_stream_annotation`, on by default): prepare and dispatch as
one, the combine, and each backward call after a prepare of its own. The compute
stream waits for a call's results where it uses them.

:::{note}
It stays off by default: a step of microbatches one after another gains nothing from
it ([below](#moe_stream-in-megatronbuxs-nccl_ep-arm)). It is there for a
schedule of overlapped microbatches, as `methods(interleave=true)` makes
([Side by side](microbatches.md#side-by-side-methodsinterleave)).
:::

#### How it overlaps

Two things make it overlap:

- **The scheduler's hints.** XLA's scheduler starts each call as soon as its inputs
  are ready and waits for it as late as its results are needed (the call's hint
  `force_delay_async`). Of calls ready at once, those that send rows to the experts
  (the dispatch, the combine's gradient, and a backward's prepares) start before those
  that send them back (`scheduler_delay_priority`). So in chunks the next chunk's
  dispatch is not queued behind this chunk's combine on NCCL EP's stream.
- **A pass over XLA's HLO.** In a `shard_map`, XLA's SPMD partitioner copies a call's
  frontend attributes onto the get-tuple-elements of its tuple of results, and XLA
  then makes each of them an async call of its own on the stream
  (`ExplicitStreamAnnotationAsyncWrapper` takes any instruction so marked). Queued
  there behind the calls issued after it, the compute stream waited for the next
  chunk's dispatch before it took this chunk's rows, and the scheduler put every
  call's done right after its start. `jax_internals.parts_off_the_stream`, one of
  bux's passes over XLA's HLO (`bux.passes`) before its scheduler, takes that mark off
  them. Where `moe_stream` is set, `nccl_ep.start` has XLA run it on every program the
  process compiles for GPUs from then on, and JAX's cache keys those programs by it.

## Measurements

All on B300s, in October 2026, but the collectives counted on 8 CPU devices. The
benchmarks' MoE is the baseline's: `d_model` 2048, 64 experts of 704, top 8
([Benchmarks](../performance/benchmarks.md)).

### The router's weights before `down`

`examples/benchmarks/megatron.bux` (Qwen3-30B-A3B, 32k tokens a GPU, MXFP8, routed by
the router, XLA's pool at 97% of a GPU), before the weights moved before `down` and up
and gate became one stack, and after. Tokens a second a GPU and peak memory:

| Run | Before | After |
|---|---|---|
| `sorted`, `fsdp=8`, `remat=none` | 31.1k, 250.4 GiB | 31.9k, 215.0 GiB |
| `nccl_ep`, `expert=8`, capacity 1.25, `remat=save_attention` | 21.3k, 99.6 GiB | 22.8k, 98.3 GiB |
| the same, `remat=none` | needs 266 GiB | 27.7k, 210.7 GiB |

- `sorted`'s step on GPU 0 took 1075 ms against 1098. Gone: the weights' d-wide
  multiply, its transpose and reduce, and the concatenations of up and gate and of
  their gradient.
- With GEMM autotuning off, its losses agree to 5e-5 for three steps, then drift apart
  as two runs of random tokens do.
- Under `remat=auto` at JAX's default pool ([remat=auto](remat.md#rematauto-the-default)),
  `sorted` now takes `dots_attention` (158.5 GiB) where it took `dots` (179 GiB), and
  `nccl_ep` takes `dots` (187.4 GiB, 24.5k) where it took `save_attention` (21.7k).

### MXFP8 experts

#### Each GEMM on one B300

Each GEMM of the baseline MoE's experts (64 of them, `d_model` 2048, intermediate 704,
262144 picks), in TFLOP/s of the GEMM's own kernel by CUPTI:

| GEMM | bfloat16, tokamax | MXFP8, Triton's `tl.dot_scaled` (before) | MXFP8, `grouped_mxfp8` |
|---|---|---|---|
| x W, up and gate | 1269 | 964 | 2405 (1.9x) |
| h W, down | 1069 | 680 | 1596 (1.5x) |
| dy W^T, down | 1213 | 908 | 2230 (1.8x) |
| dh W^T, up and gate | 1266 | 901 | 2244 (1.8x) |
| dW, up and gate | 1093 | 644 | 2573 (2.4x) |
| dW, down | 1034 | 616 | 2367 (2.3x) |

- With experts of 2048: 1.7 to 2.3 times bfloat16's, 2423 to 2616 TFLOP/s against
  1139 to 1414.
- At `all_to_all`'s GEMMs of a peer's buffer under `expert=8` (40960 rows, of which
  32768 are its 8 experts'): 1.3 to 2.7 times, 1271 (h W) to 2490 (dW, up and gate)
  against 904 to 1213.
- h W contracts 704, which TMA takes in tiles of 128, so `rows` pads it to 768 and
  does 9% more.
- Taking the tiles a row of them at a time, all its columns before the next row, was
  1.7x slower (1623 against 2725 TFLOP/s for dh W^T at k 4096): the rows and columns in
  flight then fell out of L2.
- Triton's kernels took the scales, rows of 4 bytes too short for TMA, through
  registers into tensor memory at every step of k.

#### In training on B300s

Side by side, GEMM autotuning off and deterministic ops: the experts in bfloat16, in
MXFP8 by Triton's kernels as before, by `grouped_mxfp8`, and by it with the quantizer
writing its layout ([below](#the-quantizer-on-one-b300)). By
`scripts/benchmark.py` on `examples/benchmarks`' arms, against bfloat16 in the same
job; tokens a second and peak memory:

| Run | bfloat16 | MXFP8, Triton | MXFP8, `grouped_mxfp8` | quantized into its layout |
|---|---|---|---|---|
| the baseline MoE, one GPU, 8 × 4096 tokens: `one_gpu.bux`'s `moe` | 86.2k, 133.9 GiB | 74.2k | 91.5k (1.06x), 119.5 GiB | 95.0k (1.10x) |
| experts of 2048, 8 layers, one GPU | 90.4k, 143.2 GiB | 75.6k | 101.7k (1.12x), 129.6 GiB | |
| `ragged_all_to_all`, `capacity=1.25`, `expert=8`, 32k tokens a GPU: `one_node.bux`'s `moe_ep` | 468k, 88.5 GiB | 417k | 517k (1.10x), 72.3 GiB | 539k (1.15x) |
| the same, `dispatch=mxfp8` | | | 515k | |
| `all_to_all`, `capacity=1.25`, `expert=8`, 32k tokens a GPU | 492k, 91.4 GiB | 449k | 539k (1.10x), 78.4 GiB | |
| `sorted`, `fsdp=8`, 32k tokens a GPU: `one_node.bux`'s `moe` | 671k, 83.9 GiB | 580k | 712k (1.06x), 69.4 GiB | 743k (1.11x) |

- Both MXFP8 kernels compute the same values: the runs' losses agree to the last digit
  logged, and with bfloat16's to the rounding between them. The two quantizers write
  the same bytes, and the memory is the same.
- The experts' GEMMs take half of bfloat16's time, 3.0 ms a layer against 5.8, and the
  quantizer takes 1.5 ms of what they save. An MoE layer's experts, forward and
  backward, take 5.0 ms against 7.3 in bfloat16 (1.45x), and 11.8 against 19.7 with
  experts of 2048 (1.67x; `scripts/gpu_smoke/grouped_mxfp8.py`).
- The quantizer reads each operand once and writes it twice at 5.1 TB/s, k padded to
  whole tiles of 128 and the scales in tensor memory's tiles, so no pass of XLA lays
  them out after it. x^T's blocks run down x's columns, so that only its E4M3 bytes
  transpose, and it scales in x's dtype, by a power of two, exact. It took 2.2 ms a
  layer before, in float32, and XLA's pads and layouts 90 to 160 µs a call more: 6.1
  ms a layer against 7.3.
- The experts keep x^T in MXFP8 for dW rather than x in bfloat16, so a run takes 14
  GiB less.
- It trains at bfloat16's loss: 98M tokens of FineWeb-Edu end 0.13% and 0.05% below
  it, with FP8 dispatch ([Precision](../training/precision.md)).

#### The quantizer on one B300

At what one GPU of `megatron.bux`'s `sorted` arm quantizes a layer (128 experts of
768, 262144 picks), on one B300 by CUPTI, no shape is slow:

| Operand | Shape | TB/s |
|---|---|---|
| x, read by index from 32768 tokens | 262144 × 2048 | 5.8 |
| dy | 262144 × 2048 | 5.2 |
| h | 262144 × 768 | 5.1 |
| dh | 262144 × 1536 | 5.2 |
| each expert's W of up and gate | 128 × 2048 × 1536 | 5.2 |
| each expert's W of down | 128 × 768 × 2048 | 5.1 |

- x read by index takes 377 µs, where gathering it in bfloat16 took 378 and
  quantizing the copy 420.
- A copy that casts x and x^T to E4M3 unscaled runs at 6.5 TB/s (XLA's `x * 2` at
  6.9), and the quantizer without its scales at 5.8. The scales' bytes take the rest
  (4 a row of x, but one of each 4 of x^T's in a program of 32 rows), and so do the
  blocks' largest magnitudes.
- No faster: programs of 64 or 128 rows, or of 128 rows 32 at a time with each tile of
  scales in one store; more warps; other grid orders; the scaling in float32.

#### x and dy read in place, on 8 B300s

`megatron.bux`'s `sorted` arm at `remat=none` with the pool at 97%, 40 steps:

- With the router's weights on the hidden units, so that down's rows sum plainly into
  their tokens, reading x and dy in place took the arm from 260.3k to 269.4k tokens a
  second (32.5k to 33.7k a GPU) at the same peak, 215.0 GiB. On GPU 0 a step fell from
  1067 to 1029 ms: both gathers' 39 ms went, and the quantizer's 95 ms became 88.
- With the weights after down, x in place alone went from 254.2k to 255.0k: the
  gather's 25 ms went, but the FSDP collectives the step overlapped came out from
  under it.
- Each GPU quantized all 128 experts' W, 23 ms of a step. Under the fused kernels
  ([SwiGLU fused, by cuDNN](#swiglu-fused-by-cudnn)), each quantizes its own shard before FSDP gathers
  it.

### The fused SwiGLU

#### One layer on a B300

One layer of Qwen3-30B-A3B's experts (128 of 768, 262144 picks; by EP 16 of them,
327680 rows of which 262144 are filled), forward and backward on a B300, by CUPTI:

| | bux's kernels | fused | |
|---|---|---|---|
| `sorted` | 6.30 ms | 5.51 ms | the GEMMs with their epilogues 3.20 against 3.43 (2.3 against 2.2 PFLOP/s); none of XLA's 0.73 ms over h and dp, nor the quantizer's of them |
| by EP | 5.83 ms | 4.97 ms | |

- At these shapes cuDNN's GEMM of h W takes 0.44 ms against `grouped_mxfp8`'s 0.53,
  and dp W^T 0.65 against 0.77; its wgrad 0.30 and 0.60 ms against 0.39 and 0.76.
- The values are within MXFP8's rounding of bux's kernels': 2% apart in the output and
  every gradient (relative Frobenius norm), each 6.6% from bfloat16's as before.

#### `megatron.bux` on 8 B300s

`remat=none`, the tree before and after on the same node:

| Run | bux's kernels | fused |
|---|---|---|
| `sorted`, Megatron's batch (512 rows in 8 microbatches, `gather=layer`) | 36.1k tok/s a GPU, 229.6 GiB | 37.7k (+4.5%), 214.6 GiB |
| `sorted`, one microbatch of 8 rows a GPU | 37.7k, 215.1 GiB | 38.5k (+2.1%), 200.3 GiB |
| `nccl_ep`, one microbatch of 8 rows a GPU | 31.4k, 211.1 GiB | 32.4k (+3.1%), 212.7 GiB |

- On GPU 0, at one microbatch (traced without XLA's command buffers, whose kernels the
  trace then loses), the experts' kernels and XLA's passes over h and dp take 263 ms a
  step against 319 under `sorted`, and 234 against 280 under `nccl_ep`.
- Beside FSDP's collectives cuDNN's kernels run slower than alone: `dswiglu` takes
  1.13 ms against 0.62 (0.59 under `nccl_ep`, which has none). Its tiles are
  statically scheduled, where `grouped_mxfp8`'s take the SMs as they free.
- Leaving pairs of SMs to the collectives (`CUDNNFE_CLUSTER_OVERLAP_MARGIN`, which all
  of `grouped_cudnn`'s kernels heed) gains 0.3% at Megatron's batch with 8 (two runs
  each, on one node) and 1.1% with 16 (one run), and loses 2% with 32. It is left at
  0.
- `sorted` keeps no bfloat16 copy of all 128 experts' kernels now, so it takes 15 GiB
  less.
- Its compiles take 20 to 110 s more: CuTe DSL compiles each kernel in every process.
- 3000 steps of FineWeb-Edu on the baseline's MoE with experts of 768 (the fused
  kernels' shapes, one B300 a run) end within 0.08% of bux's kernels at each of three
  seeds, which differ by 1.6%: eval loss 4.0708, 4.0127 and 4.0346 against 4.0742,
  4.0099 and 4.0363, at 104k tokens a second against 100k.

### The EP methods

#### Collectives on 8 CPU devices

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
- Dropless with buffers of a fixed size, `all_to_all` moves `top_k` times the rows
  `all_gather` does, as every peer's buffer holds every local pick. Under a capacity
  it moves `capacity` times the picks, where `ragged_all_to_all` moves the picks
  alone.
- No method rematerializes anything on these meshes, nor on
  `fsdp=2, expert=2, tensor=2`.

#### The methods on 8 B300s

On 8 B300s every method computes what one GPU does (`scripts/gpu_smoke/experts.py`, in
float32: 7e-7 apart at most). For the baseline's MoE (`d_model` 2048, 64 experts,
`top_k` 8, 16 layers, `bfloat16`) on 32k tokens a GPU, tokens a second and peak
memory:

| Method | Mesh | remat `none` | remat `dots` |
|---|---|---|---|
| `sorted` | `fsdp=8` | 816k, 84 GiB | 737k, 52 GiB |
| `all_gather`, `capacity=1.25` | `expert=8` | 612k, 89 GiB | 515k, 65 GiB |
| `ragged_all_to_all`, `capacity=1.25` | `expert=8` | 550k, 88 GiB | 515k, 78 GiB |
| `ragged_tokens`, `capacity=1.25` | `expert=8` | 547k, 90 GiB | 479k, 74 GiB |
| `all_to_all`, `capacity=1.25` | `expert=8` | 559k, 89 GiB | |
| the same, `moe_chunks=2` | `expert=8` | 574k, 89 GiB | |

:::{tip}
On one node at this shape, `sorted` is still the fastest: its FSDP gathers 0.5 GB of
experts a layer, where each of an EP layer's four exchanges moves 1.1 to 1.3 GB of
picks a GPU ([What bounds EP on one node](#what-bounds-ep-on-one-node)).
:::

- Before the by-expert slices, the capacity and the pieces, `all_to_all` and
  `ragged_all_to_all` ran out of memory at 16k tokens a GPU, and under `remat=full`
  at 32k they made 72k and 25k tokens a second (now `ragged_all_to_all` dropless:
  263k).
- Under `dots` the exchanges ran again in the backward, since their results are no
  products. Named for it ([FP8 dispatch](#fp8-dispatch-precisiondispatch)),
  `ragged_all_to_all` trains at 515k tokens a second in 77.5 GiB, from 452k in 72.6.
- XLA's ragged all-to-all leaves an allocation inside the memory pool after the first
  step, so a step whose temporaries take over half of it fits only once. bux's pool of
  95% of a GPU, against JAX's 75%, gives it more room.
- `ragged_tokens` moves about 30% fewer rows when the routing is random, and half as
  few or fewer where a group limit keeps each token on half the devices (`num_groups`
  the size of `expert`, `top_k_groups` half of it). But it sends the picks and their
  weights besides, and here gains nothing.
- At the start of a run on random tokens, a capacity of 1.25 drops 4-10% of the picks
  a layer, and 1.5 at 16k tokens 3-4%.
- Neither `--xla_gpu_experimental_ragged_all_to_all_use_device_kernel` nor
  `--xla_gpu_ragged_all_to_all_mode=peer` changes the speed; `=symmetric` fails in
  NCCL ("Window address must be suitably aligned").
- Nor does XLA's grouped matmul catch up with tokamax's: `jax.lax.ragged_dot` trains
  `sorted` 20 times slower, and with `--xla_gpu_experimental_use_ragged_dot_fusion`
  (cuDNN) it fails to compile.

#### Where a step's time goes

For the baseline's MoE at 32k tokens a GPU and a capacity of 1.25, profiled: tokens a
second, and ms of a step of a GPU for all its collectives, for those with no compute
beside them (the log's `exposed_comm_ms`), and for the experts' grouped GEMMs:

| Method | Tokens a second | Collectives | Exposed | Experts' GEMMs |
|---|---|---|---|---|
| `sorted`, `fsdp=8` | 830k | 66 | 7 | 110 |
| `ragged_all_to_all` | 546k | 167 | 152 | 116 |
| `all_to_all` | 559k | 165 | 143 | 117 |
| `all_to_all`, `moe_chunks=2` | 574k | 208 | 90 | 153 |
| `all_to_all`, 2 microbatches, unrolled | 562k | 219 | 98 | 155 |

The table's runs come from several nodes. Side by side on one node
(`scripts/benchmark.py`, the one-node suite's `moe` and `moe_ep`, the latter as
`methods` says), `sorted` trains at 835k tokens a second, `ragged_all_to_all` at 539k,
and `all_to_all` in 2 chunks at 586k, 0.70 of `sorted`.

#### What bounds EP on one node

What bounds EP here is its exchange: four of 1.1 to 1.3 GB a layer a GPU, 150 ms of a
step at NCCL's 500 GB/s. `sorted`'s FSDP gathers 0.5 GB of experts a layer and reduces
their gradients, 66 ms that hide behind compute.

- Chunks hide a third of the exchange, but a GEMM beside NCCL's kernels runs a third
  slower and gives most of it back: the experts' GEMMs take 153 ms rather than 117,
  and the step gains 3%.
- DeepSeek's two microbatches, each one's exchanges beside the other's compute (the
  trainer's microbatches unrolled rather than scanned, by hand), do no better.
- Fewer NCCL channels, to leave the GEMMs more SMs, slow the exchange more
  (`NCCL_MAX_P2P_NCHANNELS=8`: 349k), and a high-priority stream for collectives
  changes nothing.

EP would beat `sorted` on one node at this shape with an exchange near NVLink's 900
GB/s, on a few SMs and without the capacity's padding, as DeepEP's and
TransformerEngine's kernels send it, hidden behind the attention as well as the
experts.

#### Why only `all_to_all` gains in chunks

On 8 B300s with jax 0.12:

- NCCL's all-to-all, which `all_to_all` sends by, runs on its own stream on a few of
  the SMs: beside an independent grouped GEMM the two take a fifth longer than the
  all-to-all alone.
- XLA's ragged all-to-all overlaps nothing. Its one-shot kernel copies in 2048 blocks
  of threads, which take every SM, so beside a GEMM the two take as long as one after
  the other, and in chunks a step is slower. Nor does XLA schedule for it: its latency
  hiding scheduler, which moves collectives beside compute, runs on Hopper and
  Blackwell only where its cost model knows every collective of a step
  (`SolLatencyEstimator::IsSupportedForModule`), and ragged-all-to-all is none of
  them. `compiler(preset=lhs)` ([XLA's options](../performance/compiler.md)) runs it
  anyway, and at 32k tokens a GPU the step then runs out of memory, in the hole the
  ragged exchange leaves in the pool.
- A collective-permute a peer (in turn i each device sends the i-th after it its
  buffer and runs its experts on the one it gets) hides half its time, but seven of
  them take twice as long as one all-to-all, by NCCL or by copy engines
  (`--xla_gpu_use_memcpy_local_p2p`): 394k tokens a second.
- `nccl_ep`'s kernels run beside the experts on a stream of their own
  ([On a stream of its own](#on-a-stream-of-its-own-methodsmoe_stream)). In 2
  chunks the next chunk's dispatch runs beside this one's experts, and a layer's
  forward takes 4.97 ms against 5.35 whole, but its backward gives more back (forward
  and backward 9.55 ms against 8.07 whole).

#### FP8 dispatch on 8 B300s

For the baseline's MoE at 32k tokens a GPU and a capacity of 1.25 (the one-node
suite's `moe_ep`, side by side on one node):

| Method | `dispatch=compute` | `dispatch=mxfp8` |
|---|---|---|
| `ragged_all_to_all` | 539k | 556k |
| `all_to_all`, 2 chunks | 586k | 589k |

A ragged dispatch takes 17 ms a step less, and quantizing a token rather than each of
its picks costs little; `all_to_all`'s dispatch hides behind the experts already.

### NCCL EP

#### One node and two

The baseline's MoE at 32k tokens a GPU and a capacity of 1.25, by
`scripts/benchmark.py` (`one_node.bux` and `two_nodes.bux`, their `moe` and `moe_ep`,
the rest set as below). Tokens a second, peak memory, and the loss of the 30th step of
30 (at the 5th, 12.233 on one node and 12.222 on two, for all):

| GPUs | Method | Mesh | tok/s | Peak | Loss |
|---|---|---|---|---|---|
| 8 | `sorted` | `fsdp=8` | 831k | 83.8 GiB | 11.843 |
| | `ragged_all_to_all` | `expert=8` | 544k | 88.4 GiB | 11.842 |
| | `nccl_ep` | `expert=8` | 615k | 89.3 GiB | 11.867 |
| 16 | `sorted` | `data=2, fsdp=8` | 1,586k | 83.8 GiB | 11.820 |
| | `ragged_all_to_all` | `data=2, expert=8` | 1,016k | 87.0 GiB | 11.825 |
| | `nccl_ep` | `data=2, expert=8` | 1,192k | 89.3 GiB | 11.822 |
| | `ragged_all_to_all` | `data=1, expert=16` | 324k | 84.4 GiB | 11.836 |
| | `nccl_ep` | `data=1, expert=16` | 773k | 85.0 GiB | 11.844 |

- EP over two nodes is 2.4x as fast by `nccl_ep` as by `ragged_all_to_all`, but at 63%
  of a node's tokens a GPU.
- EP in a node and data across them stays the faster EP, 17% past
  `ragged_all_to_all` side by side (another pair of runs: 1,158k and 1,023k).
- `sorted` is the fastest of all at this shape.

#### One layer's exchange

One layer's dispatch and combine at the baseline MoE's shapes (d 2048, 64 experts, top
8, 32k tokens a GPU, a capacity of 1.25), its experts the identity, on B300s of 148
SMs, and what an exchange sends off a GPU (on two nodes, off its node):

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
| | the same, 256 tokens a chunk | 32 | 13.0 | 21.2 | 0.13 GB |

In a node NCCL EP wins by sending less, not faster: its kernels move 330 to 370 GB/s,
far from NVLink's 900. Between nodes it sends a quarter as much over InfiniBand, and
with `NCCL_GIN_TYPE=2 NCCL_EP_TOKENS_PER_CHUNK=256` its exchanges take 28% as long.

#### What NCCL EP's step holds

Under `remat=none`, XLA's analysis says the step holds 266 GiB:

- 42.7 GiB are the parameters and the optimizer's state, and 223.2 one buffer of
  temporaries, of which 221.8 are live at once at the end of the forward.
- 148 GiB of those are the experts': four arrays over each layer's 329,728 rows (at a
  capacity of 1.25) kept for the backward. They are the `down` GEMM's bfloat16 output
  (1.26 GiB a layer), `up` and `gate`'s (0.94), and the MXFP8 copies of the rows and
  of the activations that the weights' gradients take (0.63 and 0.24).
- Attention keeps 24 GiB, and the dense MXFP8 matmuls 21.

A capacity of 1.5 took 2.4 GiB more under `save_attention`, at 5% fewer tokens a
second. On `megatron.bux`'s synthetic batch it dropped as much as 1.25 from its 15th
step on (8% of the picks a layer, its balance loss rising to 83 summed over the
layers).

#### `moe_stream`: one MoE layer

One MoE layer at `megatron.bux`'s shapes (EP=8, 32k tokens a GPU, a capacity of 1.25,
the experts by cuDNN's MXFP8 kernels), in ms:

| | Compute stream | `moe_stream` | Hints only | The pass only |
|---|---|---|---|---|
| an exchange beside an independent grouped GEMM (2.2 ms) | 5.18 | 4.30 | 4.38 | 4.64 |
| forward | 5.37 | 5.35 | 5.36 | 5.35 |
| forward, 2 chunks | 5.71 | 4.97 | 5.32 | 5.71 |
| forward and backward | 9.39 | 8.07 | 8.90 | 8.18 |
| forward and backward, 2 chunks | 10.06 | 9.55 | 9.35 | 9.83 |
| forward and backward beside 12.6 ms of independent GEMMs | 21.66 | 20.62 | 21.52 | 20.87 |

- The forward of one chunk has nothing to put beside its exchange. The backward does:
  its prepares wait for nothing but the picks, and run beside the forward's experts.
- More chunks cost more than they hide here: 4 chunks take 11.29 ms forward and
  backward.
- The last row is what a schedule of two microbatches, one's backward beside the
  other's forward (`methods(interleave=true)`), gives the exchange: about 1 ms of a
  layer's ~6 ms of exchange hides.
- XLA's cost model prices the experts' and attention's custom calls at 10 us and an
  async call at 5 ms, and the hints put all that is ready beside the last call before
  it in the schedule. The interleaved schedule's barriers, each a tick of a piece's
  forward and a piece's backward, bound what that is.

#### `moe_stream` in `megatron.bux`'s `nccl_ep` arm

At Megatron's batch (512 rows of 4096, 8 microbatches one after another,
`remat=none`), side by side:

| NCCL EP's kernels | tok/s a GPU | Peak |
|---|---|---|
| on the compute stream, the tree before | 34.3k | 236.1 GiB |
| on the compute stream | 34.1k | 236.1 GiB |
| on their own, `moe_stream=true` | 34.1k | 234.2 GiB |
| the same, `moe_chunks=2` | 33.2k | 239.1 GiB |

- A step of microbatches one after another gains nothing from it, and in 2 chunks it
  loses 3%. What overlaps in one layer on its own does not carry over to a step of
  48 layers, and why is not known yet, as jax's profiler loses most of a step's events
  at this batch.
- XLA's own all-to-all is no alternative here: `moe=all_to_all` at a capacity of 1.25,
  the tokens sent in MXFP8, trains the arm at 26.6k tokens a second a GPU, and 27.4k
  in 2 chunks (244-245 GiB), where NCCL EP took 32.2k on the same tree.
- With the routing's logits drawn at random, a capacity of 1.25 drops no pick
  (`dropped` 0.0 every step), at either.
