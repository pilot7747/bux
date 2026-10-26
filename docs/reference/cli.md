# Command line

`bux COMMAND FILE [SETTING=VALUE ...] [options]` compiles a `.bux` file and does one
thing with it. Every command takes statements after the file, which redefine the
file's as one more layer over it would: `optimizer.lr=1e-3`, `'use fineweb_wsd'`,
`'optimizer.lr = 1e-3 | 3e-3'`. `bux export` takes a trained run's directory instead.

| Command | What it does |
|---------|--------------|
| `check` | Compile the file; print its layers by type, its training budget and its sweep. |
| `plan` | Print every run of a sweep: its id, parameters, tokens, FLOPs a token, GPUs, and bytes a GPU. |
| `prepare` | Tokenize and pack the documents the runs read, once ([Data](../data/index.md)). |
| `train` | Train every run, one after another, in this process. |
| `launch` | Train what is left of every run, each on its own GPUs, and resume after a stop ([Launching](../training/launching.md)). |
| `render` | Draw the model as a circuit diagram. |
| `fmt` | Print the file in canonical form: every run as a file of its own. |
| `json` | Print the IR of the model, or of every run. |
| `export` | Write a trained run's model as a transformers model directory, and check it against bux ([Exporting a run](../export/index.md#exporting-a-run)). |

`bux-lsp` is the language server ([Editor support](../language/tooling.md)).

## `bux export`

```sh
bux export RUN_DIR OUT [--step N] [--dtype DTYPE] [--check]
```

| Option | Default | What it does |
|--------|---------|--------------|
| `RUN_DIR` | | the run's directory, `runs/<name>-<id>`, with its `run.json` and `checkpoints/` |
| `OUT` | | the directory to write, made if missing |
| `--step N` | the latest | the checkpoint to export |
| `--dtype DTYPE` | the parameters', `float32` | the weights' dtype: `float32`, `bfloat16` or `float16` |
| `--check` | | compare the export with bux on the run's weights, print what it measured, and exit 1 if they disagree ([What the check checks](../export/index.md#what-the-check-checks)) |

It needs the `export` extra (PyTorch, transformers, safetensors). A model with a
layer that is not exported yet fails with the layer's name, exit 1.

## `bux --help`

```{include} ../_generated/cli.md
```
