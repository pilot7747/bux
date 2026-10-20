"""Dense channel mixers: mlp and gated_mlp."""

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux.architecture.functional import activation, bound
from bux.architecture.registry import Context, Taps, register
from bux.lang import ir


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

    def hidden(self, x: Float[Array, "*lead d"]) -> Float[Array, "*lead f"]:
        """What the down projection takes."""
        up = bound(self.up(x), self.limit, self.up_softcap)
        if self.gate is None:
            return activation(self.activation, up)
        gate = self.gate(x)
        if self.limit is not None:
            gate = jnp.minimum(gate, self.limit)
        return activation(self.activation, gate) * up

    def __call__(
        self, x: Float[Array, "*lead d"], taps: Taps | None = None
    ) -> Float[Array, "*lead d"]:
        return self.dropout(self.down(self.hidden(x)))


@register("mlp", "gated_mlp")
def _(c: cfg.MLPConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context) -> MLP:
    return MLP(c, width, ctx.rngs)
