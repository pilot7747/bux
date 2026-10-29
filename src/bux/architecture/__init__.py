"""The reference Flax NNX backend: an architecture, compiled from ``.bux``, built into
a runnable `flax.nnx.Module`.

`bux.lang` compiles a ``.bux`` file to a typed IR, an `~bux.lang.ir.Architecture`, and
this package builds it into a model: `~bux.architecture.model.Model`, from tokens to
logits, made of one small module for each node of the IR's graph. The trainer
(`bux.train`) builds its models here, and so can a script or a notebook. It is also
where a new layer goes: a module, and a builder registered for its config's type.

Example:
    Build a model of two pre-norm blocks and run it on a batch of tokens::

        import jax.numpy as jnp
        from flax import nnx

        from bux.architecture import Model, num_params
        from bux.lang import compile

        arch = compile('''
        d_model    = 64
        vocab_size = 256
        attn  = gqa(num_heads=4, num_kv_heads=2, rope=rope())
        ffn   = gated_mlp(intermediate_size=128, activation=swish)
        model = embed ([rmsnorm attn] [rmsnorm ffn])^2 rmsnorm head
        ''')
        model = Model(arch, rngs=nnx.Rngs(0))
        logits = model(jnp.zeros((2, 16), jnp.int32))  # (2, 16, 256)
        num_params(model)  # 106816

.. rubric:: How a model is built

- `~bux.architecture.graph.build_node` builds the IR's graph, one module per node
  kind (`~bux.architecture.graph`). Nodes with the same share key (``shared`` in
  ``.bux``) are built once and reused.
- A layer is built by the builder registered for its config's ``type``
  (`~bux.architecture.registry`).
- A residual branch ``[f]`` outside any other is the residual algebra's, if the
  architecture sets one (`~bux.architecture.residual`); otherwise, and inside
  another branch, it is ``x + f(x)``.
- A repeat ``f^n`` is built by the run's ``methods.repeat``: unrolled, or stacked
  and scanned, and pipelined over the mesh's stages; each residual branch outside
  any other is rematerialized as ``methods.remat`` says (`~bux.architecture.repeats`).

.. rubric:: The layers

Each family of layers is a package, made of `~bux.architecture.common`'s pieces, and
its projections multiply in the formats of `~bux.architecture.precision`:

- `~bux.architecture.attention`: softmax attention, ``gqa``, ``mla`` and ``csa``, and
  the sparse attention indexer;
- `~bux.architecture.recurrent`: linear attention (``gla``, ``deltanet``, ``gdn``,
  ``kda``, ``gdn2``, ``lightning``, ``rwkv7``) and state space models (``mamba``,
  ``mamba2``, ``short_conv``);
- `~bux.architecture.experts`: the mixture of experts, ``moe``, and its grouped
  matmuls;
- `~bux.architecture.layers`: the rest, norms, MLPs, the causal conv, the embedding,
  the head, ``nextn``, ``engram``, and ``linear``, ``act``, ``scale`` and
  ``dropout``.

The language reference documents every layer's settings
(:doc:`/reference/language/layers`), and each layer's class what it computes.

.. rubric:: Calling a module

Every module is called as ``module(x, taps)``: `x` is the stream (the token ids, for
the embedding) and `taps` the `~bux.architecture.registry.Taps` of the forward pass.
Taps hold the stacks that ``@name`` pushes to and ``$name`` reads, and as attributes
the pass's input tokens, the documents packed into its rows
(`~bux.architecture.registry.segments`, `~bux.architecture.registry.positions`), the
output of every head run so far, and, in a training step
(`~bux.architecture.registry.training`), the losses layers add to the objective
(`~bux.architecture.registry.report`) and what they observe on the steps a run
observes (`bux.architecture.observe`).

.. rubric:: Adding a layer

A builder registered for a config's ``type`` makes its module, and replaces the one
registered before it, so another implementation of a layer, a fused kernel say,
plugs in the same way as a new one::

    from bux.architecture import Context, register


    @register("gdn")
    def _(config: GatedDeltaNetConfig, width: int, ports: tuple[Port, ...], ctx: Context):
        return MyGatedDeltaNet(config, width, rngs=ctx.rngs)

See `~bux.architecture.registry.register` and :doc:`/architectures/adding-a-layer`.

.. rubric:: Shapes

Arrays say their dtype and shape with jaxtyping, as ``x: Float[Array, "b t d"]`` for a
stream of b rows of t positions, d wide, and the tests check them on every call. A
stream widened by a residual algebra is ``b t n d``: n copies of it.
"""

# Importing the layer modules registers their layers.
from bux.architecture.attention.csa import CSA
from bux.architecture.attention.gqa import Attention, SoftmaxAttention
from bux.architecture.attention.indexer import Indexer
from bux.architecture.attention.mla import MLA
from bux.architecture.common.segments import Segments
from bux.architecture.experts.moe import MoE
from bux.architecture.graph import build_node
from bux.architecture.layers.conv import CausalConv
from bux.architecture.layers.doc_copy import DocCopy
from bux.architecture.layers.elementwise import Activation, Dropout, Linear, Scale
from bux.architecture.layers.embedding import Embedding, Head, NextN, ValueEmbedding
from bux.architecture.layers.engram import Engram
from bux.architecture.layers.mlp import MLP
from bux.architecture.layers.norm import DyT, Norm
from bux.architecture.model import Model, num_params
from bux.architecture.recurrent.linear_attention import (
    DeltaNet,
    GatedDeltaNet,
    GatedLinearAttention,
    KimiDeltaAttention,
    LightningAttention,
)
from bux.architecture.recurrent.rwkv7 import RWKV7
from bux.architecture.recurrent.ssm import Mamba, Mamba2, ShortConv
from bux.architecture.registry import (
    Builder,
    Built,
    Context,
    Loss,
    Taps,
    positions,
    register,
    report,
    segments,
    training,
)
from bux.architecture.repeats.repeat import Scanned
from bux.architecture.residual import (
    AttentionResidual,
    GatedResidual,
    HyperConnections,
    ManifoldHyperConnections,
    Merge,
    ResidualGate,
)

__all__ = [
    "CSA",
    "MLA",
    "MLP",
    "RWKV7",
    "Activation",
    "Attention",
    "AttentionResidual",
    "Builder",
    "Built",
    "CausalConv",
    "Context",
    "DeltaNet",
    "DocCopy",
    "Dropout",
    "DyT",
    "Embedding",
    "Engram",
    "GatedDeltaNet",
    "GatedLinearAttention",
    "GatedResidual",
    "Head",
    "HyperConnections",
    "Indexer",
    "KimiDeltaAttention",
    "LightningAttention",
    "Linear",
    "Loss",
    "Mamba",
    "Mamba2",
    "ManifoldHyperConnections",
    "Merge",
    "MoE",
    "Model",
    "NextN",
    "Norm",
    "ResidualGate",
    "Scale",
    "Scanned",
    "Segments",
    "ShortConv",
    "SoftmaxAttention",
    "Taps",
    "ValueEmbedding",
    "build_node",
    "num_params",
    "positions",
    "register",
    "report",
    "segments",
    "training",
]
