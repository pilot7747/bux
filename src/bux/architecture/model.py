"""The whole model: an Architecture built into one module from tokens to logits."""

import jax
from flax import nnx
from jaxtyping import Array, Float, Int

from bux.architecture.common.segments import Segments
from bux.architecture.graph import build_node
from bux.architecture.logits import HeadOutput
from bux.architecture.precision import matmuls
from bux.architecture.registry import Context, Loss, Stream, Taps
from bux.architecture.repeats.repeat import pipe_longest
from bux.lang import ir


class Model(nnx.Module):
    """Tokens to logits.

    Every head is an output. ``heads`` returns them all in execution order, each as
    the stream it reads and its projection (``HeadOutput``), and ``outputs`` their
    logits; head j predicts the token 1 + k positions ahead, where k counts the nextn
    layers run before it. Rows can hold several documents: given positions, the
    offsets of every token in its document, a document starts wherever they stop
    increasing, and each one is attended to, convolved and embedded on its own
    (``Segments``).
    """

    def __init__(self, arch: ir.Architecture, *, rngs: nnx.Rngs):
        self.graph = pipe_longest(build_node(arch.graph, Context(arch, rngs)))
        self.main_branches = matmuls.main_branches(arch.graph)  # precision.last

    def outputs(
        self, tokens: Int[Array, "b t"], positions: Int[Array, "b t"] | None = None
    ) -> list[Float[Array, "b t vocab"]]:
        """Return every head's logits, all at once; the graph's output if it has no
        head."""
        y, taps = self._forward(tokens, positions, training=False)
        return [head.logits() for head in taps.heads] or [y]

    def heads(
        self, tokens: Int[Array, "b t"], positions: Int[Array, "b t"] | None = None
    ) -> list[HeadOutput]:
        """Return every head's output, from which the objective makes its logits a
        chunk at a time."""
        return self._forward(tokens, positions, training=False)[1].heads

    def training_heads(
        self, tokens: Int[Array, "b t"], positions: Int[Array, "b t"] | None = None
    ) -> tuple[list[HeadOutput], list[Loss]]:
        """Return the heads' outputs, and the losses the layers report
        (``registry.report``): a training step's forward, in which layers also update
        state of their own, as an MoE does its router bias."""
        taps = self._forward(tokens, positions, training=True)[1]
        return taps.heads, taps.losses

    def _forward(
        self,
        tokens: Int[Array, "b t"],
        positions: Int[Array, "b t"] | None,
        training: bool,
    ) -> tuple[Stream, Taps]:
        """Run the graph; return its output, and the taps it leaves."""
        docs = None if positions is None else Segments.of(positions)
        taps = Taps(tokens=tokens, segments=docs, training=training)
        with matmuls.forward(self.main_branches):
            return self.graph(tokens, taps), taps

    def __call__(self, tokens: Int[Array, "b t"]) -> Float[Array, "b t vocab"]:
        """Return the first head's logits: the next-token prediction."""
        return self.outputs(tokens)[0]


def num_params(module: nnx.Module) -> int:
    return sum(x.size for x in jax.tree.leaves(nnx.state(module, nnx.Param)))
