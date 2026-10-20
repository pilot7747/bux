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
uv run bux check examples/gpt2.bux
uv run bux render examples/gpt2.bux
```

## Runs and sweeps

A run is a `.bux` file that also says how to train its model. It can build on other
files by name, with no paths, and `|` makes it a sweep:

```
use llama3_8b                      # an architecture, found by name
use fineweb_wsd                    # a recipe: data mix, optimizer, schedule

d_model = 768
model   = embed block^12 norm head(tied=true)

attn.num_kv_heads = mha: 12 | gqa: 4 | mqa: 1   # llama3_8b's block picks this up
optimizer.lr      = 1e-3 | 3e-3                 # 3 × 2 = 6 runs
```

```sh
uv run bux plan experiments/attention_ablation.bux   # the 6 runs, their sizes and budgets
uv run bux train experiments/smoke.bux               # a few seconds on a laptop
uv run bux train experiments/smoke.bux 'optimizer = muon(lr=0.02)' steps=100
```

Every setting is checked before anything runs, at the line that wrote it, and every
run directory keeps the run as one self-contained `run.bux`. See the
[training guide](docs/training.md).

## Data

Documents, raw text or chat conversations in the OpenAI format, are Parquet files.
A MapReduce job tokenizes them with [gigatoken](https://github.com/marcelroed/gigatoken),
shuffles them, and packs them into rows of `input_ids`, `position_ids` and
`loss_mask`, densely (`concat`), whole (`greedy`), or best-fit (`best_fit`), on this
machine's cores or on Spark. Documents packed into a row see only themselves.

```
tokenizer = tokenizer(name="gpt2")
fineweb   = text(path="data/fineweb/sample/10BT/*.parquet", bos=true, eos=false)
packing   = best_fit
data      = fineweb(split=[0, 0.99])                 # a hundredth held out, by hash
eval      = eval(data=fineweb(split=[0.99, 1]))
```

```sh
uv run bux prepare experiments/speedrun.bux          # the speedrun's FineWeb, once
```

See the [data guide](docs/data.md) for conversations, packings, and the pipeline.

Explore the [example architectures](examples/), from GPT-2 to recent releases
grouped by model family (Cohere, DeepSeek, GLM, Inkling, Kimi, Laguna, Mellum, MiMo,
Muse, Nemotron, Qwen), and the [language guide](docs/language.md) for composition, residual
connections, parameter sharing, and editor support.

Run the CI checks (ruff, ty, and the tests) with `make`, or one of them with `make lint`,
`make typecheck`, or `make test`. Arrays say their dtype and shape with
[jaxtyping](https://github.com/patrick-kidger/jaxtyping), as `Float[Array, "b t d"]`,
and the tests check every call against them.

## Docker

The image runs JAX on CUDA 13 for Blackwell and Hopper GPUs, with
[tokamax](https://github.com/openxla/tokamax) kernels, as the non-root user `bux` (UID 1000).
It needs an NVIDIA driver 580 or newer and the NVIDIA Container Toolkit.

```sh
docker build -t bux .
docker run --rm --gpus all bux python main.py
```

To publish it to `ghcr.io/pilot7747/bux` for amd64 and arm64, run the Docker workflow from
the Actions tab or with `gh workflow run docker.yml` (add `-f tag=v0.1.0` for an extra tag).

Outside Docker, `uv sync --extra cuda` installs the same GPU stack on Linux.

## License

[Apache License 2.0](LICENSE). Copyright 2026 Nikita Pavlichenko.
