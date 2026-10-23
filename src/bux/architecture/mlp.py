"""Dense channel mixers: mlp and gated_mlp."""

import jax.numpy as jnp
from flax import nnx
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float

from bux import config as cfg
from bux import parallel
from bux.architecture.functional import activation, softcap
from bux.architecture.registry import Context, Taps, register
from bux.architecture.sharding import linear
from bux.lang import ir


def activate(
    config: cfg.MLPConfig,
    up: Float[Array, "*lead f"],
    gate: Float[Array, "*lead f"] | None,
) -> Float[Array, "*lead f"]:
    """Return act(up), or act(gate) * up when gated; `limit` clips both (gate from
    above only), and `up_softcap` caps up."""
    if config.limit is not None:
        up = jnp.clip(up, -config.limit, config.limit)
    up = softcap(up, config.up_softcap)
    if gate is None:
        return activation(config.activation, up)
    if config.limit is not None:
        gate = jnp.minimum(gate, config.limit)
    return activation(config.activation, gate) * up


def shard_hidden(h: Float[Array, "*lead f"]) -> Float[Array, "*lead f"]:
    """Shard the hidden units over `mlp`, and their rows and positions, if any, as
    inside a layer."""
    lead = ("batch", "seq", *(None,) * h.ndim)[: h.ndim - 1]
    return parallel.shard(h, *lead, "mlp")


class MLP(nnx.Module):
    def __init__(
        self, config: cfg.MLPConfig, width: int, rngs: nnx.Rngs, size: int = 0
    ):
        size = size or config.intermediate_size
        gated = config.type == "gated_mlp"
        into, out_of = ("embed", "mlp"), ("mlp", "embed")
        self.up = linear(width, size, into, config.bias, rngs, role="mlp")
        self.gate = (
            linear(width, size, into, config.bias, rngs, role="mlp") if gated else None
        )
        self.down = linear(size, width, out_of, config.bias, rngs, role="mlp")
        self.config = config
        self.dropout = nnx.Dropout(config.dropout, rngs=rngs)

    def hidden(self, x: Float[Array, "*lead d"]) -> Float[Array, "*lead f"]:
        """Return the hidden units the down projection takes, from the up and gate
        projections, named `mlp` (``remat``)."""
        up = checkpoint_name(self.up(x), "mlp")
        gate = None if self.gate is None else checkpoint_name(self.gate(x), "mlp")
        return shard_hidden(activate(self.config, up, gate))

    def __call__(
        self, x: Float[Array, "*lead d"], taps: Taps | None = None
    ) -> Float[Array, "*lead d"]:
        return self.dropout(self.down(self.hidden(x)))


@register("mlp", "gated_mlp")
def _(c: cfg.MLPConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context) -> MLP:
    return MLP(c, width, ctx.rngs)
