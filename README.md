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
grouped by model family (DeepSeek, GLM, Inkling, Kimi, Laguna, Mellum, MiMo, Muse,
Nemotron, Qwen), and the [language guide](docs/language.md) for composition, residual
connections, parameter sharing, and editor support.

Run the tests with `uv run pytest`.

## License

[Apache License 2.0](LICENSE). Copyright 2026 Nikita Pavlichenko.
