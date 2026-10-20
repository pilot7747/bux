"""Reference Flax NNX backend: turns an Architecture into a runnable nnx.Module.

Every IR node kind maps to one small module (``graph``), so the graph side is complete,
and a residual branch outside any other is the residual algebra's (``residual``).
Leaves come from a registry keyed by config type (``registry``); the other modules here
each register one family of layers. A builder registered for a type replaces the one
before it, so another implementation plugs in the same way, a fused kernel say:

    @register("gdn")
    def _(config: GatedDeltaNetConfig, width: int, ports: tuple[Port, ...], ctx: Context):
        return MyGatedDeltaNet(config, width, rngs=ctx.rngs)

Modules are called as ``module(x, taps)``: ``taps`` holds the tap stacks for one
forward pass, and under reserved keys the input tokens, the documents rows are packed
with (``segments(taps)``, None for a document a row) and the logits of every head run
so far. Repeats are unrolled; nodes with a share key are built once and reused.

Arrays say their dtype and shape with jaxtyping, as ``x: Float[Array, "b t d"]`` for a
stream of b rows of t positions, d wide, and the tests check them on every call.
"""

# Importing the layer modules registers their layers.
from bux.architecture.attention import Attention, SoftmaxAttention
from bux.architecture.conv import CausalConv
from bux.architecture.csa import CSA
from bux.architecture.elementwise import Activation, Dropout, Linear, Scale
from bux.architecture.embedding import Embedding, Head, NextN
from bux.architecture.engram import Engram
from bux.architecture.graph import build_node
from bux.architecture.indexer import Indexer
from bux.architecture.linear_attention import (
    DeltaNet,
    GatedDeltaNet,
    GatedLinearAttention,
    KimiDeltaAttention,
    LightningAttention,
)
from bux.architecture.mla import MLA
from bux.architecture.mlp import MLP
from bux.architecture.model import Model, num_params
from bux.architecture.moe import MoE
from bux.architecture.norm import DyT, Norm
from bux.architecture.registry import (
    NEXTN,
    OUTPUTS,
    SEGMENTS,
    TOKENS,
    Context,
    Taps,
    register,
    segments,
)
from bux.architecture.residual import (
    AttentionResidual,
    GatedResidual,
    HyperConnections,
    ManifoldHyperConnections,
    Merge,
    ResidualGate,
)
from bux.architecture.rwkv7 import RWKV7
from bux.architecture.segments import Segments
from bux.architecture.ssm import Mamba, Mamba2, ShortConv

__all__ = [
    "CSA",
    "MLA",
    "MLP",
    "NEXTN",
    "OUTPUTS",
    "RWKV7",
    "SEGMENTS",
    "TOKENS",
    "Activation",
    "Attention",
    "AttentionResidual",
    "CausalConv",
    "Context",
    "DeltaNet",
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
    "Segments",
    "ShortConv",
    "SoftmaxAttention",
    "Taps",
    "build_node",
    "num_params",
    "register",
    "segments",
]
