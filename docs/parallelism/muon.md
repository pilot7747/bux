(parallelism-muon)=
# Muon: `methods.orthogonalize`

Muon orthogonalizes each update matrix by Newton-Schulz iterations: fifteen products
of the matrix a step, at the default 5 iterations. On a matrix split over devices,
most of those products need an all-reduce, so how Muon reaches whole matrices is a
method of {bux}`methods`, `methods.orthogonalize`. The default, `owner`, gives
each matrix to one device, which runs it whole; you rarely need to change it. The
optimizer itself is on the [optimizers page](../training/optimizers.md).

## Turn it on

Muon is the optimizer; the method only says where its Newton-Schulz runs. A 1.3B
Llama under FSDP over 8 GPUs, with each update gathered whole on every device that
holds it:

```bux
use llama3_8b
use fineweb_wsd

d_model = 2048
attn.num_heads        = 16
ffn.intermediate_size = 5632
model = embed block^16 norm head

mesh      = mesh(fsdp=8)
precision = precision(compute=bfloat16)
optimizer = muon(lr=0.02, ns_dtype=float32)
methods   = methods(orthogonalize=gather)
```

Leave `orthogonalize` out to get `owner`.

## Options

| Method | What it does |
|---|---|
| `gspmd` | Newton-Schulz on whatever shards the compiler gives each update, where a product over a split dimension is an all-reduce. |
| `gather` | Each update whole in a `jax.shard_map` for Newton-Schulz, and back on its parameter's shards after: one gather per matrix instead of the all-reduces, the work repeated on each device that holds the matrix. |
| `owner` | The default. Each matrix whole on one device, its owner, which runs its matrices of a shape as one batch. |

`muon`'s own `ns_dtype` (`bfloat16` by default, or `float32`) is the dtype
Newton-Schulz runs in and the updates move in, under every method. Its
`ns_algorithm=gram` ([Gram Newton-Schulz](../training/optimizers.md#gram-newton-schulz))
works under every method too: it changes how a whole matrix is orthogonalized, not
how matrices move. Under `gspmd` it all-reduces less: at 5 iterations it takes four
products by the matrix, where the standard iteration takes ten.

## How it works

### `gather`

A stack of matrices (the experts) is whole matrix by matrix, and its stacked
dimension splits over the axes that frees, where they divide it: each device
orthogonalizes its own experts. A lone matrix is whole on every device, and every
device runs its Newton-Schulz.

### `owner`

`owner` spreads that work, as Moonlight's and Dion's distributed Muon do:

1. Every matrix, a stack's one by one, has one owner.
2. The matrices of a shape and split, from every update of the step, are dealt out
   evenly over the devices, padded with zero matrices to a multiple of them, so that
   each device does the same work. The work of a shape (m·n·min(m, n) a matrix) is
   split evenly.
3. One all-to-all takes each device's shards of them to their owners, which
   orthogonalize their matrices of a shape as one batched chain of products.
4. Another all-to-all brings the results back to the parameters' shards, and an
   all-gather copies them over the axes that do not split the parameter (`data`, say).

An update comes split as the optimizer's state is: under `layout(zero=1)`, by FSDP
too. Its matrices go back whole along what FSDP split by an all-gather, as the
parameter is. Where FSDP splits a side, or a stack's experts, each device
orthogonalizes the matrices where they are, with no all-to-all.

## Writing your own method

A method gets every update of a step at once, so that it can choose where each runs:

```python
from bux.parallel import methods


@methods.orthogonalize.register("mine")
def _(updates, dims, shardings, states, newton_schulz):
    # updates: an array per Muon parameter, in muon's ns_dtype; dims: their
    # MuonDimensionNumbers; shardings and states: their parameters' NamedShardings
    # and their optimizer states', where the updates come from, None outside
    # parallel.use; newton_schulz(x, dims): muon's ns_algorithm, in x's dtype.
    return [newton_schulz(x, d) for x, d in zip(updates, dims)]
```

`methods = methods(orthogonalize=mine)` then picks it.

## Measurements

### Collectives on 8 CPU devices

The collectives in the compiled train step (all-reduces / all-gathers / all-to-alls)
for the 2-block Llama and the MoE of `tests/test_parallel.py`, Newton-Schulz in
float32:

| Model, mesh | AdamW | Muon, `gspmd` | Muon, `gather` | Muon, `owner` |
|---|---|---|---|---|
| Llama, `fsdp=8` | 22 / 26 / 0 | 73 / 86 / 0 | 22 / 40 / 0 | 22 / 26 / 6 |
| Llama, `fsdp=4, tensor=2` | 34 / 33 / 0 | 133 / 213 / 0 | 34 / 61 / 0 | 34 / 33 / 16 |
| MoE, `fsdp=8` | 24 / 31 / 0 | 65 / 181 / 0 | 24 / 46 / 6 | 24 / 31 / 8 |
| MoE, `fsdp=4, tensor=2` | 36 / 35 / 0 | 164 / 250 / 0 | 36 / 73 / 0 | 36 / 36 / 18 |
| Llama, `fsdp=8`, `zero=1` | 22 / 21 / 0 | 73 / 81 / 0 | 22 / 21 / 0 | 22 / 10 / 3 |
| Llama, `fsdp=4, tensor=2`, `zero=1` | 34 / 21 / 0 | 133 / 201 / 0 | 34 / 35 / 0 | 34 / 11 / 12 |

- **With `gather`**, Muon all-reduces as often as AdamW, and adds one all-gather per
  matrix and two all-to-alls per expert stack; the optimizer's update alone does no
  all-reduce.
- **With `owner`**, Muon all-reduces and all-gathers as AdamW does, and adds one
  all-to-all each way for every shape and split of matrix, two where a matrix splits
  on both sides (`tensor`).
- **Under `zero=1`**, `owner`'s few all-gathers of whole matrices replace the one each
  parameter takes otherwise.

Step times on fake CPU devices, which share one CPU, say nothing.

### Throughput on 8 B300s

`mesh(fsdp=8)`, 64 rows of 4096 tokens a step computed in bfloat16. Thousands of
tokens a second for a 1.3B dense Llama (16 layers, `d_model=2048`, 16 query heads and
8 kv heads, a gated MLP of 5632, a vocabulary of 128k) and a 5.2B MoE of its shape (64
experts of 704, top 8), by `ns_dtype`:

| Model | AdamW | `gspmd` | `gather` | `owner` |
|---|---|---|---|---|
| dense, float32 | 1,147 | 856 | 849 | 1,073 |
| dense, bfloat16 | | 923 | 950 | 1,087 |
| dense, `repeat=scan` | | 863 | 938 | 935 |
| MoE, float32 | | 531 | 712 | 754 |
| MoE, bfloat16 | | | | 786 |

The dense model's optimizer update alone takes 74 ms with `gspmd`, 73 with `gather`
and 12 with `owner` in float32; 48, 42 and 12 in bfloat16. With `repeat=scan`
(float32 here) the stacked `layers` already split over `fsdp` under `gather`, as under
`owner`.

### Gram Newton-Schulz on B300s

`ns_algorithm=gram` against `standard`, 5 iterations, `ns_dtype=bfloat16`, on one node
of 8 B300 SXM6s, jax 0.12.0, 2026-10-26.

**Newton-Schulz alone, one GPU.** Milliseconds a call (the median of 20) for a stack
of the matrices of the 1.3B Llama and the MoE above, and of Kimi K2 (the 216 experts
of 2048×7168 dealt over 8 devices, and its largest matrix), by Keller Jordan's
quintic:

| Matrices | Stack × m × n | `standard` | `gram` | |
|---|---|---|---|---|
| Llama's q, o | 32 × 2048 × 2048 | 7.54 | 7.53 | the standard: square |
| Llama's k, v | 32 × 2048 × 1024 | 2.56 | 2.90 | 0.88× |
| Llama's up, gate | 32 × 2048 × 5632 | 15.03 | 13.33 | 1.13× |
| Llama's down | 16 × 5632 × 2048 | 7.66 | 6.81 | 1.12× |
| MoE's experts | 1024 × 704 × 2048 | 30.74 | 29.18 | 1.05× |
| Kimi K2's experts | 27 × 2048 × 7168 | 15.42 | 12.33 | 1.25× |
| Kimi K2's largest | 1 × 7168 × 18432 | 16.69 | 13.90 | 1.20× |
| eight times as wide | 16 × 1024 × 8192 | 3.55 | 2.62 | 1.36× |

Polar Express's quintics take about the same, but for k and v, 2.65 → 2.64. The
speedups are below what the products saved predict (a third fewer at n = 4m, 46% at
8m). XLA computes every product whole, with no symmetric GEMM. And the three products
an iteration that add `a` in float32 take a pass of their own over a float32 result.
On the up and gate projections, Gram's products alone take 8.2 ms. Summed in 16 bits,
as a GEMM's epilogue would, they take 12.5 ms in all, but that blows up on CPU, where
each operation rounds. Its authors' fused symmetric kernels make it up to twice as
fast.

**Accuracy.** Iterating in float16, `gram`'s result is within 0.4-0.6% of float32's
standard iteration (relative Frobenius norm, Gaussian matrices), the standard in
bfloat16 within 3-5%.

**Muon's optimizer update alone**, `mesh(fsdp=8)`, ms a step: dense 57.4 → 58.0 by
`gspmd`, 42.0 → 42.0 by `gather`, 13.6 → 13.4 by `owner`; MoE 37.2 → 37.2, 30.9 →
32.1, 20.1 → 19.3. Under `owner` each device orthogonalizes an eighth of the
matrices, and Newton-Schulz is a few milliseconds of the update.

**End to end**, as the throughput table above: dense 1,135,636 → 1,139,894 tokens a
second, MoE 823,556 → 828,208, within the runs' noise.

**Loss.** The dense model on packed documents (the docs.bux of the bfloat16
comparison above), one GPU each, `muon(lr=0.02)`, `wsd(warmup=30)`, 200 steps: the
loss at the last step is 3.9290 by `standard`, 3.9292 by `gram`, 3.9291 by
`standard` in float32, and 3.9472 by `standard` from another seed. With 8 iterations
of Polar Express, two restarts, `standard` reaches 3.9303 and `gram` 3.9309.
