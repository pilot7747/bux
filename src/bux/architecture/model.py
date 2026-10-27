"""The whole model: an Architecture built into one module from tokens to logits."""

import jax
from flax import nnx
from jaxtyping import Array, Float, Int

from bux import initializers
from bux.architecture.common.segments import Segments
from bux.architecture.experts.moe import MoE
from bux.architecture.graph import build_node
from bux.architecture.logits import HeadOutput
from bux.architecture.registry import Context, Loss, Stream, Taps
from bux.architecture.repeats import tracing
from bux.architecture.repeats.repeat import pipe_longest
from bux.lang import ir


class Model(nnx.Module):
    """Tokens to logits: an architecture built into one module.

    Every head is an output. `heads` returns them all in execution order, each as
    the stream it reads and its projection (`~bux.architecture.logits.HeadOutput`),
    and `outputs` their logits; head j predicts the token 1 + k positions ahead,
    where k counts the nextn layers run before it. Rows can hold several documents:
    given positions, the offsets of every token in its document, a document starts
    wherever they stop increasing, and each one is attended to, convolved and
    embedded on its own (`~bux.architecture.common.segments.Segments`).

    Under a mesh with more than one pipeline stage (`bux.parallel`), building the
    model pipelines its longest scanned repeat over the stages. Every weight matrix
    starts as the architecture's ``init`` says (`bux.initializers`).

    Args:
        arch: The architecture to build, as `bux.lang.compile` makes it.
        rngs: The random streams the parameters start from, and dropout draws from.

    Example:
        ::

            model = Model(bux.lang.compile(source), rngs=nnx.Rngs(0))
            logits = model(tokens)  # (b, t, vocab): the next token's
            logits, *mtp = model.outputs(tokens, positions)  # every head's

    Attributes:
        graph: The module of the architecture's graph.
        main_branches: How many residual branches outside any other run before the
            first head: the model's, which precision's ``first`` and ``last`` count,
            and not its multi-token prediction modules'.
    """

    def __init__(self, arch: ir.Architecture, *, rngs: nnx.Rngs):
        with initializers.initializing(arch.init):
            self.graph = pipe_longest(build_node(arch.graph, Context(arch, rngs)))
        self.main_branches = tracing.main_branches(arch.graph)  # precision.last

    def outputs(
        self, tokens: Int[Array, "b t"], positions: Int[Array, "b t"] | None = None
    ) -> list[Float[Array, "b t vocab"]]:
        """Return every head's logits, all at once; the graph's output if it has no
        head.

        Args:
            tokens: The token ids, b rows of t.
            positions: Each token's position in its document, from which the
                documents packed into each row follow; None when each row is one
                document.

        Returns:
            The logits of every head, in the order the heads run, in the dtype the
            stream has; or, for a graph without a head, its output.
        """
        y, taps = self._forward(tokens, positions, training=False)
        return [head.logits() for head in taps.heads] or [y]

    def heads(
        self, tokens: Int[Array, "b t"], positions: Int[Array, "b t"] | None = None
    ) -> list[HeadOutput]:
        """Return every head's output, from which the objective makes its logits a
        chunk at a time.

        Args:
            tokens: The token ids, b rows of t.
            positions: Each token's position in its document; None when each row is
                one document.

        Returns:
            Each head's input and projection, in the order the heads run.
        """
        return self._forward(tokens, positions, training=False)[1].heads

    def training_heads(
        self, tokens: Int[Array, "b t"], positions: Int[Array, "b t"] | None = None
    ) -> tuple[list[HeadOutput], list[Loss]]:
        """Return the heads' outputs, and the losses the layers report: a training
        step's forward.

        In it, layers add terms to the loss (`~bux.architecture.registry.report`)
        and update state of their own, as an MoE does its router bias; dropout is
        as the module's mode says (``model.train()``, ``model.eval()``).

        Args:
            tokens: The token ids, b rows of t.
            positions: Each token's position in its document; None when each row is
                one document.

        Returns:
            Each head's output, in the order the heads run, and the losses the
            layers reported, in the order they ran.
        """
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
        with tracing.forward(self.main_branches):
            return self.graph(tokens, taps), taps

    def __call__(self, tokens: Int[Array, "b t"]) -> Float[Array, "b t vocab"]:
        """Return the first head's logits: the next-token prediction.

        Args:
            tokens: The token ids, b rows of t, each row one document.

        Returns:
            The logits over the vocabulary at every position.
        """
        return self.outputs(tokens)[0]


def num_params(module: nnx.Module) -> int:
    """Return how many parameters `module` holds: the sizes of its `nnx.Param`
    arrays, summed.

    Example:
        ``num_params(Model(arch, rngs=nnx.Rngs(0)))`` is 106816 for the model of
        two blocks in `bux.architecture`'s example.
    """
    return sum(x.size for x in jax.tree.leaves(nnx.state(module, nnx.Param)))


def active_params(module: nnx.Module) -> int:
    """Return how many parameters a token goes through: `num_params`, less the
    routed experts that each MoE (`bux.architecture.experts.moe.MoE`) skips,
    ``num_experts - top_k`` of every ``num_experts``.

    Example:
        A model whose MoE takes 2 of 8 experts counts its router, its shared
        experts and a quarter of its routed experts.
    """
    idle = 0.0
    for _, node in nnx.iter_graph(module):
        if isinstance(node, MoE):
            c = node.config
            stacks = (getattr(node, name) for name in _ROUTED)
            routed = sum(s.get_value().size for s in stacks if s is not None)
            idle += routed * (c.num_experts - c.top_k) / c.num_experts
    return num_params(module) - round(idle)


_ROUTED = ("up", "down", "up_bias", "down_bias")  # an MoE's stacks of routed experts
