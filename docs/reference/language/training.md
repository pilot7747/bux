# Training settings

The configs a run's top-level settings take, in the order of `bux/train/config.py`:
data sources and packings, optimizers and schedules, the loss, eval, checkpoints and
logs, precision, and how a run splits over devices. Each is called by name, as layers
are: `optimizer = muon(lr=0.02)`, `mesh = mesh(fsdp=8)`.

```{include} ../../_generated/training.md
```
