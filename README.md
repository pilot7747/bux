# bux

**Architecture and data ablations in a few lines, trained at Megatron-class speed.**

bux is a JAX framework for comparing language models. You write a model as an algebra
over the residual stream in a short `.bux` file, say in the same file how to train it,
and turn any line into a sweep with `|`. bux checks every setting before anything
runs, then trains with Flax NNX on anything from a laptop CPU to multi-node GPU
clusters.

```bux
use llama3_8b                      # an architecture, found by name
use fineweb_wsd                    # a recipe: data mix, optimizer, schedule

d_model = 768
model   = embed block^12 norm head(tied=true)

attn.num_heads    = 12                          # llama3_8b's block picks these up
attn.num_kv_heads = mha: 12 | gqa: 4 | mqa: 1
optimizer.lr      = 1e-3 | 3e-3                 # 3 × 2 = 6 runs
```

```sh
bux plan   examples/experiments/attention_ablation.bux   # the 6 runs, their sizes and budgets
bux launch examples/experiments/attention_ablation.bux   # train them, one GPU each
```

## Why bux

- **A model is an expression.** `[f]` means `x + f(x)`, so `[norm attn] [norm ffn]`
  is a pre-norm Transformer block and `block^32` is 32 of them. Hybrids,
  hyper-connections, parallel blocks and shared layers are one line each.
- **One file is the whole experiment.** Model, data, optimizer, schedule, precision
  and parallelism all live in the same file, and `|` sweeps any of them, including
  the architecture. An ablation is the few lines that differ, not a fork of a
  training script.
- **Errors before GPUs.** Every setting is type-checked and cross-checked at the line
  that wrote it, with a suggestion, before JAX is even imported.
- **Fast at scale.** FSDP, tensor, expert, context and pipeline parallelism, and any
  mix of them, from one GPU to eight nodes. On Megatron-Bridge's own workloads on
  B300s, bux reaches 104% of its throughput on Qwen3-30B-A3B and 86-89% on
  Nemotron 3 Nano and gpt-oss-120b.
- **A model zoo to start from.** 62 transcribed architectures, from GPT-2 to 2026
  releases (DeepSeek, Qwen, Kimi, GLM, MiniMax, Nemotron, OLMo and more), each usable
  by name.
- **Out to the ecosystem.** Trained models export to transformers, checked
  numerically against bux, and serve in vLLM.

## Install

bux needs Python 3.14+ and [uv](https://docs.astral.sh/uv/), which fetches Python
for you if needed.

```sh
git clone https://github.com/pilot7747/bux.git
cd bux
uv sync                      # add --extra cuda on Linux with NVIDIA GPUs
uv run bux check examples/architectures/gpt2.bux
```

## Quickstart

Train a tiny Llama-style model on your laptop in seconds:

```console
$ uv run bux train examples/experiments/smoke.bux
smoke: runs/smoke-5a4f0f7d85, on 1 × cpu
smoke: 156k parameters, 50 steps × 8 × 64 tokens = 25.6k tokens, adamw(lr=0.003), wsd(warmup=10)
smoke: log every 10
remat=auto: none in 1s (none 0.0 GiB in 0s)
smoke step 10/50  loss 6.2429  lr 3.00e-03  grad 0.607  8,097 tok/s  ...
...
```

Draw it, change it from the command line, and sweep it:

```sh
uv run bux render examples/experiments/smoke.bux
uv run bux train  examples/experiments/smoke.bux 'optimizer = muon(lr=0.02)' steps=100
uv run bux plan   examples/experiments/smoke.bux 'optimizer.lr = 1e-3 | 3e-3 | 1e-2'
```

The [quickstart](docs/get-started/quickstart.md) walks through each step and its
output.

## What's inside

**Training.** `bux launch` trains each run of a sweep on its own GPUs, records how
every run went, checkpoints and exits cleanly on SIGTERM, and, launched again, trains
only what is left. Every run directory keeps the run as one self-contained `run.bux`.
See [Training](docs/training/index.md).

**Hyperparameter search.** Write a setting as a distribution and the sweep becomes a
search: Gaussian process, TPE or quasi-random proposals, ASHA for early stopping, and
an objective of your own ([Searching](docs/training/searching.md)).

```bux
optimizer.lr = log_uniform(1e-4, 1e-2)
objective    = 0.8 * web_bpb + 0.2 * code_bpb
search       = search(method=gp, trials=27, halving=halving(eta=3, rungs=3))
```

**Observability.** Each head's loss, throughput, MFU, step-time breakdown and memory
go to the console, `metrics.jsonl`, W&B or TensorBoard. `observe(every=100)` adds
per-layer statistics: the residual stream, attention logits, MoE routing, and each
layer's gradients, weights and updates ([Observing a run](docs/training/observing.md)).

**Data.** Training data is Parquet: plain text, or chat in the OpenAI format.
`bux prepare` applies the tokenizer's chat template, tokenizes with
[gigatoken](https://github.com/marcelroed/gigatoken), shuffles, and packs documents
into rows (dense, whole-document, or best-fit), on your machine's cores or on Spark.
Packed documents attend only to themselves ([Data](docs/data/index.md)).

```bux
tokenizer = tokenizer(name="gpt2")
fineweb   = text(path="data/fineweb/sample/10BT/*.parquet", bos=true, eos=false)
packing   = best_fit
data      = fineweb(split=[0, 0.99])                 # hold out 1%, by hash
eval      = eval(data=fineweb(split=[0.99, 1]))
```

**Parallelism.** Three settings split a run over devices: `mesh` sizes the axes,
`layout` maps named dimensions onto them, and `methods` picks how each part that
communicates runs. None of them changes what a run computes
([Parallelism](docs/parallelism/index.md)).

```bux
mesh    = mesh(data=2, fsdp=8)              # HSDP: sharded in a node, copied across two
methods = methods(microbatches=4)            # gradient accumulation
```

**Performance.** cuDNN, FlashAttention-4, tokamax and bux's own Pallas kernels;
MXFP8 and NVFP4 matmuls; automatic remat. The [tuning guide](docs/performance/tuning.md)
and [benchmarks](docs/performance/benchmarks.md) show where the time goes and how to
compare two versions of bux on the same node.

**Export.** `bux export` writes a trained run as a transformers model directory, and
`--check` verifies it numerically against bux; a vLLM plugin serves it ([Exporting](docs/export/index.md)).

**Tooling.** `bux-lsp` gives editors diagnostics, completion and hover docs for
`.bux` files, and `bux studio` opens a visual editor that draws a file's model and
sweep and edits them in place ([Editor support](docs/language/tooling.md#editor-support),
[bux studio](docs/language/studio.md)).

## Documentation

Start with the [quickstart](docs/get-started/quickstart.md) and the
[concepts](docs/get-started/concepts.md), then follow the
[tutorials](docs/tutorials/index.md): from your first architecture to a full study
that pits a 2022 dense 7B against a 2026 hybrid MoE at the same serving cost. The
guides cover the [language](docs/language/index.md),
[architectures](docs/architectures/index.md), [training](docs/training/index.md),
[data](docs/data/index.md), [parallelism](docs/parallelism/index.md) and
[performance](docs/performance/index.md); the [reference](docs/reference/index.md),
generated from the code, lists every layer, setting, command and Python API.

`make docs` builds the site into `docs/_build/html`, as Read the Docs does.

## Examples

- [`examples/architectures/`](examples/architectures/): the model zoo.
- [`examples/experiments/`](examples/experiments/): runs, sweeps, searches and
  recipes, including a modded-nanogpt-style speedrun.
- [`examples/experiments/parallel/`](examples/experiments/parallel/): one run per
  kind of parallelism, one of them on a laptop's CPU.
- [`examples/benchmarks/`](examples/benchmarks/): benchmark suites from one GPU to
  eight nodes, run by `scripts/benchmark.py`.

## Docker

The image runs JAX on CUDA 13 for Hopper and Blackwell GPUs, with tokamax kernels and
TransformerEngine's JAX bindings, as the non-root user `bux` (UID 1000). It needs an
NVIDIA driver 580+ and the NVIDIA Container Toolkit. Its entrypoint is `bux launch`:

```sh
docker build -t bux .
docker run --rm --gpus all bux                     # trains the smoke run
docker run --rm --gpus all -v "$PWD/data:/opt/bux/data" -v "$PWD/runs:/opt/bux/runs" \
  bux examples/experiments/speedrun.bux
docker run --rm -it --entrypoint bash bux          # a shell
```

Pass `-e WANDB_API_KEY` to log to W&B. A prebuilt image is published as
`ghcr.io/pilot7747/bux`; [Installation](docs/get-started/installation.md#docker) has
the build arguments and how to publish it.

## Development

`make` runs what CI runs: ruff, ty, the tests and the docs; `make lint`,
`make typecheck`, `make test` and `make docs` run one each. Arrays carry their dtype
and shape (`Float[Array, "b t d"]`, via [jaxtyping](https://github.com/patrick-kidger/jaxtyping)),
and the tests check every call against them. Extending bux means registering a
function: see the [development guide](docs/development/index.md).

## License

[Apache License 2.0](LICENSE). Copyright 2026 Nikita Pavlichenko.
