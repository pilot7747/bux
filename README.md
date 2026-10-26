# bux

A moddable training framework for pre-training ablations — and a fun pet project.

Describe model architectures in a compact `.bux` language, swap building blocks and
residual wiring, and train them with a JAX/Flax NNX backend. The same files say how
to train: data, optimizer, schedule, and sweeps over any of it, architecture included.
Early and experimental.

## Quick start

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run bux check examples/architectures/gpt2.bux
uv run bux render examples/architectures/gpt2.bux
```

## Documentation

The [documentation](docs/index.md) has a [quickstart](docs/get-started/quickstart.md),
[tutorials](docs/tutorials/index.md), guides to the [language](docs/language/index.md),
[architectures](docs/architectures/index.md), [training](docs/training/index.md),
[data](docs/data/index.md) and [parallelism](docs/parallelism/index.md), a
[performance tuning guide](docs/performance/tuning.md) with
[benchmarks](docs/performance/benchmarks.md), and a reference of every layer and
setting, the command line and the Python API, generated from the code. `make docs`
builds it as Read the Docs does (`.readthedocs.yaml`), into `docs/_build/html`.

## Runs and sweeps

A run is a `.bux` file that also says how to train its model. It can build on other
files by name, with no paths, and `|` makes it a sweep:

```
use llama3_8b                      # an architecture, found by name
use fineweb_wsd                    # a recipe: data mix, optimizer, schedule

d_model = 768
model   = embed block^12 norm head(tied=true)

attn.num_heads    = 12                          # llama3_8b's block picks these up
attn.num_kv_heads = mha: 12 | gqa: 4 | mqa: 1
optimizer.lr      = 1e-3 | 3e-3                 # 3 × 2 = 6 runs
```

```sh
uv run bux plan examples/experiments/attention_ablation.bux   # the 6 runs, their sizes and budgets
uv run bux train examples/experiments/smoke.bux               # a few seconds on a laptop
uv run bux train examples/experiments/smoke.bux 'optimizer = muon(lr=0.02)' steps=100
uv run bux launch examples/experiments/attention_ablation.bux # the 6 runs, one GPU each
```

Every setting is checked before anything runs, at the line that wrote it, and every
run directory keeps the run as one self-contained `run.bux`. `bux launch` is the
entrypoint for production: it trains each run of a sweep on its own GPUs, records how
every run went, stops at SIGTERM with a checkpoint, and, launched again, trains only
what is left. See the [training guide](docs/training/index.md).

A setting written as a distribution makes a hyperparameter search: each run of the
sweep is a search of its own, whose trials a Gaussian process, TPE or quasi-random
search proposes, short runs first with ASHA, ranked by an objective of your own
([Searching hyperparameters](docs/training/searching.md)):

```
optimizer.lr = log_uniform(1e-4, 1e-2)
objective    = 0.8 * web_bpb + 0.2 * code_bpb
search       = search(method=gp, trials=27, halving=halving(eta=3, rungs=3))
```

A run logs each head's loss, its throughput and MFU, where a step's time goes and the
device's memory, to the console (plain lines that `kubectl logs` reads), to
`metrics.jsonl`, and to W&B or TensorBoard. `observe = observe(every=100)` adds the
model by layer: the residual stream, attention logits, MoE routing, and each layer's
gradient, weights and update, on those steps alone
([Observing a run](docs/training/observing.md)):

```
log     = log(every=10, trackers=[wandb(project="ablations"), tensorboard])
observe = observe(every=100, histograms=true)
```

## Data

Documents, raw text or chat conversations in the OpenAI format, are Parquet files.
A MapReduce job renders conversations with the tokenizer's own chat template,
tokenizes them with [gigatoken](https://github.com/marcelroed/gigatoken) (Hugging
Face's tokenizers for one gigatoken cannot run), shuffles them, and packs them into
rows of `input_ids`, `position_ids` and `loss_mask`, densely (`concat`), whole
(`greedy`), or best-fit (`best_fit`), on this machine's cores or on Spark. Documents
packed into a row see only themselves.

```
tokenizer = tokenizer(name="gpt2")
fineweb   = text(path="data/fineweb/sample/10BT/*.parquet", bos=true, eos=false)
packing   = best_fit
data      = fineweb(split=[0, 0.99])                 # a hundredth held out, by hash
eval      = eval(data=fineweb(split=[0.99, 1]))
```

```sh
uv run bux prepare examples/experiments/speedrun.bux          # the speedrun's FineWeb, once
```

See the [data guide](docs/data/index.md) for conversations, packings, and the pipeline.

Explore the [example architectures](examples/architectures/), from GPT-2 to recent
releases grouped by model family (Arcee, Cohere, DeepSeek, GLM, Inkling, Kimi, Laguna,
Mellum, MiMo, MiniMax, Muse, Nemotron, OLMo, Qwen, Reflection, SmolLM), the
[example experiments](examples/experiments/) that train them, and the
[language guide](docs/language/index.md) for composition, residual connections,
parameter sharing, and editor support.

A run splits over devices by three settings, `mesh`, `layout` and `methods`: FSDP,
tensor, expert, context and pipeline parallelism, and any mix of them, from one GPU
to many hosts. [examples/experiments/parallel/](examples/experiments/parallel/) has a
run for each, one of them on a laptop's CPU, and the
[parallelism guide](docs/parallelism/index.md) explains them. The suites in
[examples/benchmarks/](examples/benchmarks/) measure them on one GPU, a node and two:
`scripts/benchmark.py` trains a suite and prints how a change moved each run's speed,
memory and loss ([Benchmarks](docs/performance/benchmarks.md)).

Run the CI checks (ruff, ty, and the tests) with `make`, or one of them with `make lint`,
`make typecheck`, or `make test`. Arrays say their dtype and shape with
[jaxtyping](https://github.com/patrick-kidger/jaxtyping), as `Float[Array, "b t d"]`,
and the tests check every call against them.

## Docker

The image runs JAX on CUDA 13 for Blackwell and Hopper GPUs, with
[tokamax](https://github.com/openxla/tokamax) kernels, as the non-root user `bux` (UID 1000).
It needs an NVIDIA driver 580 or newer and the NVIDIA Container Toolkit.

Its entrypoint is `bux launch`, which takes a `.bux` file and trains it; with none, it
trains the smoke run. Mount `data/` and `runs/` to keep what it prepares and trains:

```sh
docker build -t bux .
docker run --rm --gpus all bux
docker run --rm --gpus all -v "$PWD/data:/opt/bux/data" -v "$PWD/runs:/opt/bux/runs" \
  bux examples/experiments/speedrun.bux
docker run --rm -it --entrypoint bash bux    # a shell, or --entrypoint bux for the CLI
```

The image has TransformerEngine's JAX bindings, for its expert-parallel dispatch and
combine ([NCCL EP](docs/parallelism/experts.md#transformerengines-nccl-ep)); `--build-arg TE_VERSION=...`
picks another release. It has W&B and TensorBoard for `log.trackers`: pass
`-e WANDB_API_KEY` to log to W&B.

To publish it to `ghcr.io/pilot7747/bux` for amd64 and arm64, run the Docker workflow from
the Actions tab or with `gh workflow run docker.yml` (add `-f tag=v0.1.0` for an extra tag).

Outside Docker, `uv sync --extra cuda` installs the same GPU stack on Linux.

## License

[Apache License 2.0](LICENSE). Copyright 2026 Nikita Pavlichenko.
