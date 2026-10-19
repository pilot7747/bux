# bux

A moddable training framework for pre-training ablations — and a fun pet project.

Describe model architectures in a compact `.bux` language, swap building blocks and
residual wiring, and experiment with a JAX/Flax NNX backend. Early and experimental:
the architecture tooling and model backend are in place; training workflows are
still taking shape.

## Quick start

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run bux check examples/gpt2.bux
uv run bux render examples/gpt2.bux
```

Explore the [example architectures](examples/), from GPT-2 to recent releases
grouped by model family (Cohere, DeepSeek, GLM, Inkling, Kimi, Laguna, Mellum, MiMo,
Muse, Nemotron, Qwen), and the [language guide](docs/language.md) for composition, residual
connections, parameter sharing, and editor support.

Run the CI checks (ruff, ty, and the tests) with `make`, or one of them with `make lint`,
`make typecheck`, or `make test`.

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
