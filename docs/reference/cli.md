# Command line

`bux COMMAND FILE [SETTING=VALUE ...] [options]` compiles a `.bux` file and does one
thing with it. Every command takes statements after the file, which redefine the
file's as one more layer over it would: `optimizer.lr=1e-3`, `'use fineweb_wsd'`,
`'optimizer.lr = 1e-3 | 3e-3'`. `bux export` takes a trained run's directory, or a file,
and `bux studio` takes the file alone.

| Command | What it does |
|---------|--------------|
| `check` | Compile the file; print its layers by type, its training budget and its sweep. |
| `plan` | Print every run of a sweep: its id, parameters, tokens, FLOPs a token, GPUs, and bytes a GPU. |
| `prepare` | Tokenize and pack the documents the runs read, once ([Data](../data/index.md)). |
| `train` | Train every run, one after another, in this process; a search's trials too. |
| `launch` | Train what is left of every run, each on its own GPUs, and resume after a stop ([Launching](../training/launching.md)). |
| `results` | Print every run's state and objective, or a search's trials, the best first ([Searching](../training/searching.md)). |
| `render` | Draw the model as a circuit diagram. |
| `fmt` | Print the file in canonical form: every run as a file of its own. |
| `json` | Print the IR of the model, or of every run. |
| `export` | Write a trained run's model as a transformers model directory, and check it against bux ([Exporting a run](../export/index.md#exporting-a-run)); or a file's model without weights, to time in vLLM ([Exporting an architecture](../export/index.md#exporting-an-architecture)). |
| `studio` | Serve a page that draws the file's model, training and sweep, and edits them in the file ([bux studio](../language/studio.md)). |

`bux-lsp` is the language server ([Editor support](../language/tooling.md)).

## `bux export`

```sh
bux export RUN_DIR OUT [--step N] [--dtype DTYPE] [--check]
bux export FILE.bux OUT [SETTING=VALUE ...] [--entry NAME] [--dtype DTYPE]
```

| Option | Default | What it does |
|--------|---------|--------------|
| `RUN_DIR` | | the run's directory, `runs/<name>-<id>`, with its `run.json` and `checkpoints/` |
| `FILE.bux` | | instead, a file whose model to write without weights, its code and config alone |
| `OUT` | | the directory to write, made if missing |
| `SETTING=VALUE` | | a file's: statements that redefine its own, as for `bux check` |
| `--entry NAME` | `model` | a file's: the definition to export |
| `--step N` | the latest | the checkpoint to export |
| `--dtype DTYPE` | the parameters', `float32`; a file's `bfloat16` | the weights' dtype: `float32`, `bfloat16` or `float16` |
| `--check` | | compare the export with bux on the run's weights, print what it measured, and exit 1 if they disagree ([What the check checks](../export/index.md#what-the-check-checks)) |

It needs the `export` extra (PyTorch, transformers, safetensors). A model with a
layer that is not exported yet fails with the layer's name, exit 1.

## `bux studio`

```sh
bux studio FILE [--port PORT] [--host HOST] [--no-browser]
```

| Option | Default | What it does |
|--------|---------|--------------|
| `FILE` | | the `.bux` file to show and edit; if it does not exist, the page starts from a template and the first save creates it |
| `--port PORT` | `8765` | the port to try first; if it is taken, the next few are tried |
| `--host HOST` | `127.0.0.1` | the address to listen on |
| `--no-browser` | | print the page's URL, with its token, and open no browser; over SSH no browser opens anyway, and the `ssh -L` command to forward the port is printed |

It runs until interrupted (Ctrl-C), with JAX on the CPU unless `JAX_PLATFORMS` says
otherwise.

## `bux --help`

```{include} ../_generated/cli.md
```
