"""Training: what a .bux file says about how to train its model, and a trainer that does it.

`bux.train.config` is plain data and needs no JAX, so the compiler and the editor
can import it. The rest trains a compiled `bux.lang.ir.Run`, out of blocks that each
do one thing and can each be replaced:

- `bux.train.data`: batches of rows from the run's sources; a source per config type.
- `bux.train.optim`: learning-rate schedules and optimizers; one per config type.
- `bux.train.objective`: the loss, a weighted sum of named terms; a term is a
  function.
- `bux.train.trainer`: `~bux.train.trainer.Trainer` compiles a step and runs the
  loop; a subclass overrides ``train_step``.
- `bux.train.hooks`: what the loop does every so many steps: log, eval, checkpoint.
- `bux.train.sinks`: where what the hooks report goes: the console, metrics.jsonl.

`bux.train.trainer.train` puts the run's blocks together. Config types find their
builders in registries (`bux.train.registry`), as layers do theirs.

Example::

    from bux.lang import compile_runs
    from bux.train.trainer import train

    source = open("examples/experiments/smoke.bux").read()
    for run in compile_runs(source, filename="smoke.bux"):
        metrics = train(run)  # into runs/<name>-<id>/
        print(run.name, metrics["loss"])
"""
