# Registries

Most settings that name a method do so by a name in a registry: `methods(attention=ring)`
calls the function registered as `ring` in `bux.parallel.methods.attention`. Registering
a function under a new name makes a new method; registering it under a taken name
replaces bux's own. These tables list what each registry holds once every module of
bux is imported.

```python
from bux.parallel import methods


@methods.attention.register("mine")
def _(q, k, v, args): ...
```

```{include} ../../_generated/registries.md
```
