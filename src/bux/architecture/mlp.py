"""Dense channel mixers: mlp and gated_mlp."""

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux import parallel
from bux.architecture.functional import activation, bound
from bux.architecture.registry import Context, Taps, register
from bux.lang import ir


def _linear(
    n_in: int,
    n_out: int,
    names: tuple[parallel.Name, parallel.Name],
    bias: bool,
    rngs: nnx.Rngs,
) -> nnx.Linear:
    """A dense layer whose kernel's dimensions are `names`, and its bias's the last."""
    return nnx.Linear(
        n_in,
        n_out,
        use_bias=bias,
        kernel_init=parallel.named(*names),
        bias_init=parallel.named(names[1], init=nnx.initializers.zeros),
        rngs=rngs,
    )


def _hidden_units(h: Float[Array, "*lead f"]) -> Float[Array, "*lead f"]:
    """`h` over `mlp`, and its rows and positions, if it has them, split as inside a
    layer (`seq`, not `stream`)."""
    lead = ("batch", "seq", *(None,) * h.ndim)[: h.ndim - 1]
    return parallel.shard(h, *lead, "mlp")


class MLP(nnx.Module):
    def __init__(self, c: cfg.MLPConfig, width: int, rngs: nnx.Rngs, size: int = 0):
        size = size or c.intermediate_size
        self.up = _linear(width, size, ("embed", "mlp"), c.bias, rngs)
        self.gate = (
            _linear(width, size, ("embed", "mlp"), c.bias, rngs)
            if c.type == "gated_mlp"
            else None
        )
        self.down = _linear(size, width, ("mlp", "embed"), c.bias, rngs)
        self.activation = c.activation
        self.limit, self.up_softcap = c.limit, c.up_softcap
        self.dropout = nnx.Dropout(c.dropout, rngs=rngs)

    def hidden(self, x: Float[Array, "*lead d"]) -> Float[Array, "*lead f"]:
        """What the down projection takes: its hidden units, over `mlp`."""
        up = bound(self.up(x), self.limit, self.up_softcap)
        if self.gate is None:
            return _hidden_units(activation(self.activation, up))
        gate = self.gate(x)
        if self.limit is not None:
            gate = jnp.minimum(gate, self.limit)
        return _hidden_units(activation(self.activation, gate) * up)

    def __call__(
        self, x: Float[Array, "*lead d"], taps: Taps | None = None
    ) -> Float[Array, "*lead d"]:
        return self.dropout(self.down(self.hidden(x)))


@register("mlp", "gated_mlp")
def _(c: cfg.MLPConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context) -> MLP:
    return MLP(c, width, ctx.rngs)
