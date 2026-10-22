"""The whole model: an Architecture built into one module from tokens to logits."""

import jax
from flax import nnx
from jaxtyping import Array, Float, Int

from bux.architecture.graph import build_node
from bux.architecture.registry import Context, Loss, Taps
from bux.architecture.repeat import pipe_longest
from bux.architecture.segments import Segments
from bux.lang import ir


class Model(nnx.Module):
    """Tokens to logits.

    Every head is an output. ``outputs`` returns them all in execution order; head j
    predicts the token 1 + k positions ahead, where k counts the nextn layers run before it.
    Rows can hold several documents: given positions, the offsets of every token in its
    document, a document starts wherever they stop increasing, and each one is attended
    to, convolved and embedded on its own (``Segments``).
    """

    def __init__(self, arch: ir.Architecture, *, rngs: nnx.Rngs):
        self.graph = pipe_longest(build_node(arch.graph, Context(arch, rngs)))

    def outputs(
        self, tokens: Int[Array, "b t"], positions: Int[Array, "b t"] | None = None
    ) -> list[Float[Array, "b t vocab"]]:
        logits, _ = self._forward(tokens, positions, training=False)
        return logits

    def training_outputs(
        self, tokens: Int[Array, "b t"], positions: Int[Array, "b t"] | None = None
    ) -> tuple[list[Float[Array, "b t vocab"]], list[Loss]]:
        """Return the outputs, and the losses the layers report (``registry.report``):
        a training step's forward, in which layers also update state of their own, as
        an MoE does its router bias."""
        return self._forward(tokens, positions, training=True)

    def _forward(
        self,
        tokens: Int[Array, "b t"],
        positions: Int[Array, "b t"] | None,
        training: bool,
    ) -> tuple[list[Float[Array, "b t vocab"]], list[Loss]]:
        """Run the graph; return every head's logits (the graph's output if it has no
        head), and the losses the layers report."""
        docs = None if positions is None else Segments.of(positions)
        taps = Taps(tokens=tokens, segments=docs, training=training)
        y = self.graph(tokens, taps)
        return taps.logits or [y], taps.losses

    def __call__(self, tokens: Int[Array, "b t"]) -> Float[Array, "b t vocab"]:
        """Return the first head's logits: the next-token prediction."""
        return self.outputs(tokens)[0]


def num_params(module: nnx.Module) -> int:
    return sum(x.size for x in jax.tree.leaves(nnx.state(module, nnx.Param)))
