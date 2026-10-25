"""Dense channel mixers: mlp and gated_mlp.

Under tensor parallelism (`bux.parallel`) the hidden units split over ``mlp``: up and
gate by columns, down by rows, as the run's ``methods.tensor`` multiplies them.
"""

import jax.numpy as jnp
from flax import nnx
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float

from bux import config as cfg
from bux import parallel
from bux.architecture.common.functional import activation, softcap
from bux.architecture.common.sharding import linear, project_input, project_output
from bux.architecture.registry import Context, Taps, register
from bux.lang import ir


def activate(
    config: cfg.MLPConfig,
    up: Float[Array, "*lead f"],
    gate: Float[Array, "*lead f"] | None,
) -> Float[Array, "*lead f"]:
    """Return ``act(up)``, or ``act(gate) * up`` when gated; `limit` clips both (gate
    from above only), and `up_softcap` caps up. The experts of an MoE activate by it
    too."""
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
    """A dense MLP, ``mlp``: ``down(act(up(x)))``; or a gated one, ``gated_mlp``:
    ``down(act(gate(x)) * up(x))``, SwiGLU with ``activation=swish`` (Shazeer, 2020).

    The up and gate projections are named ``mlp`` for a remat policy to keep, and
    every projection multiplies in the format the run's precision gives the ``mlp``
    role. Dropout, if any, is of the output.

    Args:
        config: The MLP's settings (`bux.config.MLPConfig`).
        width: The width of the stream, d.
        rngs: The random streams the projections start from.
        size: The hidden units, if not the config's `intermediate_size`: an MoE's
            shared experts run as one MLP of all their units.
    """

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

    def hidden(self, x: Float[Array, "b t d"]) -> Float[Array, "b t f"]:
        """Return the hidden units the down projection takes, from the up and gate
        projections, named `mlp` (``remat``)."""
        projections = (self.up,) if self.gate is None else (self.up, self.gate)
        _, (up, *gate) = project_input(x, *projections)
        up = checkpoint_name(up, "mlp")
        gate = checkpoint_name(gate[0], "mlp") if gate else None
        return shard_hidden(activate(self.config, up, gate))

    def __call__(
        self, x: Float[Array, "b t d"], taps: Taps | None = None
    ) -> Float[Array, "b t d"]:
        """Return the MLP of every position of x, ``b t d``, in the stream's width."""
        return self.dropout(project_output(self.hidden(x), self.down))


@register("mlp", "gated_mlp")
def _(c: cfg.MLPConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context) -> MLP:
    return MLP(c, width, ctx.rngs)
