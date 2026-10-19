"""Dense channel mixers: mlp and gated_mlp."""

import jax.numpy as jnp
from flax import nnx

from bux import config as cfg
from bux.architecture.functional import activation, bound
from bux.architecture.registry import Context, Taps, register


class MLP(nnx.Module):
    def __init__(self, c: cfg.MLPConfig, width: int, rngs: nnx.Rngs, size: int = 0):
        size = size or c.intermediate_size
        self.up = nnx.Linear(width, size, use_bias=c.bias, rngs=rngs)
        self.gate = (
            nnx.Linear(width, size, use_bias=c.bias, rngs=rngs)
            if c.type == "gated_mlp"
            else None
        )
        self.down = nnx.Linear(size, width, use_bias=c.bias, rngs=rngs)
        self.activation = c.activation
        self.limit, self.up_softcap = c.limit, c.up_softcap
        self.dropout = nnx.Dropout(c.dropout, rngs=rngs)

    def __call__(self, x, taps: Taps | None = None):
        up = bound(self.up(x), self.limit, self.up_softcap)
        if self.gate is not None:
            gate = self.gate(x)
            if self.limit is not None:
                gate = jnp.minimum(gate, self.limit)
            h = activation(self.activation, gate) * up
        else:
            h = activation(self.activation, up)
        return self.dropout(self.down(h))


@register("mlp", "gated_mlp")
def _(c: cfg.MLPConfig, width, ports, ctx: Context):
    return MLP(c, width, ctx.rngs)
