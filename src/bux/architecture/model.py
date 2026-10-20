"""The whole model: an Architecture built into one module from tokens to logits."""

import jax
from flax import nnx
from jaxtyping import Array, Float, Int

from bux.architecture.graph import build_node
from bux.architecture.registry import (
    LOSSES,
    OUTPUTS,
    SEGMENTS,
    TOKENS,
    Context,
    Loss,
    Taps,
)
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
        self.graph = build_node(arch.graph, Context(arch, rngs))

    def outputs(
        self, tokens: Int[Array, "b t"], positions: Int[Array, "b t"] | None = None
    ) -> list[Float[Array, "b t vocab"]]:
        return self._run(tokens, _taps(tokens, positions))

    def training_outputs(
        self, tokens: Int[Array, "b t"], positions: Int[Array, "b t"] | None = None
    ) -> tuple[list[Float[Array, "b t vocab"]], list[Loss]]:
        """The outputs, and the losses the layers report (``registry.report``): a
        training step's forward, the one in which layers update state of their own,
        as an MoE does its router bias."""
        taps = _taps(tokens, positions) | {LOSSES: []}
        return self._run(tokens, taps), taps[LOSSES]

    def _run(
        self, tokens: Int[Array, "b t"], taps: Taps
    ) -> list[Float[Array, "b t vocab"]]:
        last = self.graph(tokens, taps)
        return taps.get(OUTPUTS) or [last]

    def __call__(self, tokens: Int[Array, "b t"]) -> Float[Array, "b t vocab"]:
        """The first head's logits: the next-token prediction."""
        return self.outputs(tokens)[0]


def _taps(tokens: Int[Array, "b t"], positions: Int[Array, "b t"] | None) -> Taps:
    taps: Taps = {TOKENS: [tokens]}
    if positions is not None:
        taps[SEGMENTS] = [Segments.of(positions)]
    return taps


def num_params(module: nnx.Module) -> int:
    return sum(x.size for x in jax.tree.leaves(nnx.state(module, nnx.Param)))
