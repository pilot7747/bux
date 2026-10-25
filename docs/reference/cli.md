# Command line

`bux COMMAND FILE [SETTING=VALUE ...] [options]` compiles a `.bux` file and does one
thing with it. Every command takes statements after the file, which redefine the
file's as one more layer over it would: `optimizer.lr=1e-3`, `'use fineweb_wsd'`,
`'optimizer.lr = 1e-3 | 3e-3'`.

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

`bux-lsp` is the language server ([Editor support](../language/tooling.md)).

## `bux --help`

```{include} ../_generated/cli.md
```
