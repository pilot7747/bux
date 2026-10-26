# Python API

bux is a library as much as a command line: `bux.lang` compiles `.bux` source to a
typed IR, `bux.architecture` builds a Flax NNX model from it, `bux.train` trains it,
`bux.parallel` splits it over devices, and `bux.data` prepares its documents. Every
piece is plain Python with a registry to plug into, so a script or a notebook can use
any of them alone.

```python
from flax import nnx

from bux.architecture import Model
from bux.lang import compile_runs
from bux.train.trainer import train

source = open("examples/experiments/smoke.bux").read()
for run in compile_runs(source, filename="smoke.bux"):
    print(run.name, run.id)
    model = Model(run.arch, rngs=nnx.Rngs(0))  # the model alone
    train(run)  # or the whole run, into runs/<name>-<id>/
```

Arrays say their dtype and shape with [jaxtyping](https://docs.kidger.site/jaxtyping/),
as `Float[Array, "b t d"]`: `b` rows of `t` positions, `d` wide. The tests check every
call against them.

| Package | What it is |
|---------|------------|
| {py:mod}`bux.lang` | The `.bux` language: parser, compiler, IR, printer, renderer, modules, editor support |
| {py:mod}`bux.config` | The layers' configs, as plain pydantic data |
| {py:mod}`bux.architecture` | The reference Flax NNX backend: layers, kernels, residual algebras, repeats |
| {py:mod}`bux.train` | Training configs, the trainer, optimizers, objectives, hooks, sinks |
| {py:mod}`bux.parallel` | The mesh, the layout, and the methods of each kind of parallelism |
| {py:mod}`bux.data` | Tokenizing, shuffling and packing documents into rows |
| {py:mod}`bux.launch` | The launcher: sweeps over GPUs, status, signals, many hosts |
| {py:mod}`bux.search` | Hyperparameter search: dimensions' coordinates, GP, TPE and quasi-random methods, halving, a search's trials |
| {py:mod}`bux.softmax` | A softmax and log-softmax whose gradients hold where a float rounds a probability to 1 |
| {py:mod}`bux.runtime` | The process's JAX and XLA settings, and the compile cache |
| {py:mod}`bux.kernels` | Turning GPU kernels off, to debug one |
| {py:mod}`bux.passes` | bux's passes over XLA's HLO |
| {py:mod}`bux.jax_internals` | The few places bux reaches into JAX's and XLA's internals |

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   bux.lang
   bux.config
   bux.architecture
   bux.train
   bux.parallel
   bux.data
   bux.launch
   bux.search
   bux.softmax
   bux.runtime
   bux.kernels
   bux.passes
   bux.jax_internals
```
