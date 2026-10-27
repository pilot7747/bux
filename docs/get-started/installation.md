# Installation

Install bux from source with uv, add what GPUs, exports, the editor and Spark need,
or run it in Docker. It takes a minute; then head to the [quickstart](quickstart.md).

## Quick install

bux needs **Python 3.14 or newer** and [uv](https://docs.astral.sh/uv/). uv
downloads Python 3.14 itself if your system lacks it.

```sh
git clone https://github.com/pilot7747/bux.git
cd bux
uv sync
uv run bux check examples/architectures/gpt2.bux
```

The last command compiles GPT-2 small. It doesn't import JAX, so it answers in about
a second:

```console
$ uv run bux check examples/architectures/gpt2.bux
d_model=768 vocab_size=50257 layers=51
layernorm×25  gqa×12  mlp×12  embed×1  head×1
```

`uv sync` creates a virtual environment in `.venv/` with bux installed in editable
mode, so changes to `src/bux/` take effect immediately. The docs write commands as
`bux ...`; run them as `uv run bux ...`, or activate the environment once:

```sh
source .venv/bin/activate
```

## What gets installed

`uv sync` installs bux's dependencies (JAX, Flax, optax, orbax, pydantic, pyarrow,
gigatoken, Hugging Face's tokenizers, ...) and the three dependency groups that
`pyproject.toml` makes default:

| group    | what it has                                                       | skip it with  |
|----------|-------------------------------------------------------------------|---------------|
| `dev`    | pytest, ruff, ty, beartype, JupyterLab and pygls: what `make` runs | `--no-dev`    |
| `docs`   | Sphinx, MyST, the PyData theme, sphinx-design and sphinx-copybutton: what `make docs` builds | `--no-group docs` |
| `export` | PyTorch (CPU), transformers and safetensors: what `bux export` and its tests need | `--no-group export` |

`uv sync --no-default-groups` installs bux alone.

### Optional extras

Extras add what only some machines need. Add one with `--extra`, and several by
repeating it:

| extra         | what it adds                                                   | where         |
|---------------|----------------------------------------------------------------|---------------|
| `cuda`        | JAX on CUDA 13 (`jax[cuda13]`), [tokamax](https://github.com/openxla/tokamax)'s GPU kernels and FlashAttention-4's, cudnn-frontend 1.32, and the CPU build of PyTorch, which FlashAttention-4 and cuDNN's DSA kernels import | Linux only    |
| `export`      | PyTorch, transformers and safetensors, for `bux export` outside a checkout ([Exporting](../export/index.md)) | anywhere |
| `wandb`       | the W&B client, for `log(trackers=[wandb(...)])` ([Observing](../training/observing.md)) | anywhere |
| `tensorboard` | tensorboardX, for `log(trackers=[tensorboard])`                | anywhere      |
| `lsp`         | pygls, which the language server `bux-lsp` runs on             | anywhere      |
| `spark`       | PySpark and pandas, to prepare data on a Spark cluster         | anywhere      |

```sh
uv sync --extra cuda                 # a Linux machine with NVIDIA GPUs
uv sync --extra cuda --extra spark   # and prepare data on Spark
```

:::{note}
The `cuda` extra brings CUDA, cuDNN and NCCL as pip wheels, pinned in `uv.lock`, so
the machine needs only an NVIDIA driver: 580 or newer for CUDA 13. On macOS and on
Linux without the extra, JAX runs on the CPU.
:::

## GPUs

On a Linux machine with NVIDIA GPUs, install the `cuda` extra and check that JAX sees
them:

```sh
uv sync --extra cuda
uv run python -c "import jax; print(jax.devices())"
```

It should list one device per GPU. bux runs its GPU kernels where they apply (cuDNN
and FlashAttention-4 attention, tokamax's flash attention and grouped matmuls, and
its own Pallas kernels) and XLA's code everywhere else; see
[GPU kernels](../performance/kernels.md).

## CPU only, and macOS

bux trains on a CPU with nothing to configure: the smoke run takes a few seconds on a
laptop. On a machine with GPUs, `JAX_PLATFORMS=cpu` keeps JAX on the CPU:

```sh
JAX_PLATFORMS=cpu uv run bux train examples/experiments/smoke.bux
```

```console
$ uv run python -c "import jax; print(jax.devices())"
[CpuDevice(id=0)]
```

A CPU can also stand in for several devices, so you can try every kind of parallelism
on a laptop: `XLA_FLAGS=--xla_force_host_platform_device_count=8` gives JAX 8 CPU
devices ([Scale out](../tutorials/scale-out.md)).

:::{tip}
XLA:CPU cannot compile some bfloat16 collectives. On more than one CPU device, the
MoE methods (all but `dense`), pipelines, and sparse attention split over `context`
need them. bux stops with an error before XLA would crash; set
`precision = precision(compute=float32)` for those runs on a CPU, as
`examples/experiments/parallel/laptop.bux` does.
:::

## Docker

The image in the repository's `Dockerfile` runs JAX on CUDA 13 for Blackwell and
Hopper GPUs, with tokamax's kernels and TransformerEngine's JAX bindings, as the
non-root user `bux` (UID 1000). It needs an NVIDIA driver **580 or newer** and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

Its entrypoint is `bux launch`, which takes a `.bux` file and trains what is left of
its runs; with no file, it trains the smoke run. The image works in `/opt/bux`: mount
`data/` and `runs/` there to keep the data it prepares and the runs it trains:

```sh
docker build -t bux .
docker run --rm --gpus all bux                     # the smoke run
docker run --rm --gpus all \
  -v "$PWD/data:/opt/bux/data" -v "$PWD/runs:/opt/bux/runs" \
  bux examples/experiments/speedrun.bux
docker run --rm -it --entrypoint bash bux          # a shell
docker run --rm --entrypoint bux bux check examples/architectures/gpt2.bux  # the CLI
```

The launcher is the container's first process, so the SIGTERM that stops a container
reaches it: it finishes the current step, saves a checkpoint if the run has a
`checkpoint` setting, and exits ([Signals](../training/launching.md#signals)).

| build argument | default  | what it picks                                        |
|----------------|----------|------------------------------------------------------|
| `TE_VERSION`   | `2.20.2` | the TransformerEngine release, for its NCCL EP dispatch and combine in expert parallelism ([Experts](../parallelism/experts.md)) |
| `CUDA_VERSION` | `13.4.3` | the `nvidia/cuda` base image                         |

```sh
docker build --build-arg TE_VERSION=2.20.2 -t bux .
```

The image also has:

- libibverbs and its mlx5 provider, so NCCL can use InfiniBand between nodes
  ([Many nodes](../parallelism/multi-node.md));
- NVIDIA's profilers, Nsight Systems (`nsys`) and Nsight Compute (`ncu`), for
  `profile(tool=nsys)` and `profile(tool=ncu)`
  ([Profiling](../performance/profiling.md#nsight-systems-and-nsight-compute));
- the W&B and TensorBoard clients: pass `-e WANDB_API_KEY` to log to W&B.

### The published image

`ghcr.io/pilot7747/bux` is built for amd64 and arm64 by the repository's Docker
workflow. Run it from the Actions tab, or:

```sh
gh workflow run docker.yml                 # tags sha-<commit>, the branch, latest on main
gh workflow run docker.yml -f tag=v0.1.0   # and an extra tag
```

```sh
docker run --rm --gpus all ghcr.io/pilot7747/bux:main
```

On a machine without NVIDIA GPUs, such as a Mac, the image runs on the CPU: leave out
`--gpus all` and pass `-e JAX_PLATFORMS=cpu`.

## Check the installation

From the repository's root, compile the smoke run (no JAX needed), then train it: a
tiny model on random tokens, in a few seconds.

```sh
uv run bux check examples/experiments/smoke.bux
uv run bux train examples/experiments/smoke.bux
```

```console
$ uv run bux train examples/experiments/smoke.bux
smoke: runs/smoke-5a4f0f7d85, on 1 × cpu
smoke: 156k parameters, 50 steps × 8 × 64 tokens = 25.6k tokens, adamw(lr=0.003), wsd(warmup=10)
smoke: log every 10
remat=auto: none in 1s (none 0.0 GiB in 0s)
smoke step 10/50  loss 6.2429  lr 3.00e-03  grad 0.607  8,097 tok/s  ...
smoke step 20/50  loss 6.2618  lr 3.00e-03  grad 0.642  77,084 tok/s  ...
smoke step 30/50  loss 6.2731  lr 3.00e-03  grad 0.516  75,977 tok/s  ...
smoke step 40/50  loss 6.2478  lr 3.00e-03  grad 0.440  76,711 tok/s  ...
smoke step 50/50  loss 6.2633  lr 0.00e+00  grad 0.418  75,557 tok/s  ...
```

If you see the loss lines, you're set. To run the checks CI runs (ruff, ty, the tests
and the docs), use `make`, or one of `make lint`, `make typecheck`, `make test` and
`make docs` ([Development](../development/index.md)).

## Editor support

`bux-lsp` is a language server for `.bux` files: errors as you type, completion of
every layer's settings, and their documentation on hover. Install the `lsp` extra and
point your editor at `.venv/bin/bux-lsp`; [Editor support](../language/tooling.md#editor-support) has
the steps for PyCharm and Neovim.

```sh
uv sync --extra lsp
```
