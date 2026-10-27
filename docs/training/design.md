# Design notes

Why a run is a `.bux` file, and what was considered instead. You need none of this
to train: it explains the choices behind [Settings](settings.md) and the
[language's modules and sweeps](../language/modules.md).

## Why extend the language

The alternative was a YAML or TOML file that points at the `.bux` architecture.
Putting training in the language instead took almost nothing new:

- Layer settings were already typed keyword configs, checked against pydantic with
  errors at the source and suggestions.
- They already had arithmetic, `name.field` reads and nested overrides.
- The editor already completed and documented them from the schema.

A training setting is one more such config, so all of that applies to it unchanged,
and a run is one file you can read top to bottom.

The new syntax is the three things layer configs lacked:

- a way to build on another file without a path, `use`;
- a way to change one setting of it, `name.key =`;
- a way to say "each of these", `|`.

Each is also useful for architectures alone.

## Late binding

A definition replaces the one it shadows everywhere, and that is what makes `use`
worth having: an ablation is the lines that differ. It came with the language, since
definitions were already looked up by name, in any order. Nix overlays, Jsonnet and
Pkl's `amends` work the same way.

## Alternatives considered

- **Config groups selected by path** (Hydra) tie experiments to the layout of a
  folder.
- **Configs written in Python** (Fiddle, gin) can do anything, and so can only be
  checked by running them.
- **CUE and Pkl** type and merge configs well, but they would be a second language
  next to `.bux`, with the architecture still out of reach.
- **Sweep files that list values per key** (W&B sweeps) cannot vary the wiring, and
  they zip or cross axes by convention instead of by name.

## Other front ends

The resolved run is plain data ({py:class}`bux.lang.ir.Run`), so other front ends
can make one: a `Run` built in Python trains the same way.

Possible extensions:

- per-parameter-group optimizer settings, selected by definition label;
- learning-rate schedules composed in sequence with the same algebra;
- your own hooks, sinks and builders, named in a `.bux` file as plugins found on the
  module path.
