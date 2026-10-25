"""bux: architecture and data ablations in JAX, written in the .bux language.

A ``.bux`` file describes a model as an algebra over the residual stream and says how
to train it; ``bux train FILE`` trains it. The packages, in the order a run uses them:

- `bux.lang`: the language, compiled to a typed IR (no JAX).
- `bux.config`: the layers' configs.
- `bux.architecture`: the Flax NNX backend that builds a model from the IR.
- `bux.data`: documents tokenized and packed into rows.
- `bux.train`: the trainer, its settings, optimizers and hooks.
- `bux.parallel`: the mesh, and every kind of parallelism.
- `bux.launch`: sweeps over GPUs, many hosts, status and signals.
- `bux.runtime`, `bux.kernels`, `bux.passes`, `bux.jax_internals`: JAX and XLA.
"""
