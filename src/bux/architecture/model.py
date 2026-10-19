"""The whole model: an Architecture built into one module from tokens to logits."""

import jax
from flax import nnx

from bux.architecture.graph import build_node
from bux.architecture.registry import OUTPUTS, POSITIONS, TOKENS, Context, Taps
from bux.lang import ir


class Model(nnx.Module):
    """tokens (batch, seq) -> logits (batch, seq, vocab).

    Every head is an output. ``outputs`` returns them all in execution order; head j
    predicts the token 1 + k positions ahead, where k counts the nextn layers run before it.
    Rows can hold several documents: given positions, (batch, seq) offsets of every
    token in its document, each one is attended to, convolved and embedded on its own.
    """

    def __init__(self, arch: ir.Architecture, *, rngs: nnx.Rngs):
        if arch.residual is not None:
            raise NotImplementedError(
                "hyper-connections are not in the reference backend yet"
            )
        self.graph = build_node(arch.graph, Context(arch, rngs))

    def outputs(
        self, tokens: jax.Array, positions: jax.Array | None = None
    ) -> list[jax.Array]:
        taps: Taps = {TOKENS: [tokens]}
        if positions is not None:
            taps[POSITIONS] = [positions]
        last = self.graph(tokens, taps)
        return taps.get(OUTPUTS) or [last]

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """The first head's logits: the next-token prediction."""
        return self.outputs(tokens)[0]


def num_params(module: nnx.Module) -> int:
    return sum(x.size for x in jax.tree.leaves(nnx.state(module, nnx.Param)))
