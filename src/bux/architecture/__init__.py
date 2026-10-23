"""Reference Flax NNX backend: turns an Architecture into a runnable nnx.Module.

Every IR node kind maps to one small module (``graph``), and a residual branch outside
any other is the residual algebra's (``residual``). Repeats are unrolled, or stacked
and scanned, and pipelined over the mesh's stages (``repeats``). Layers come from a
registry keyed by config type (``registry``), and each family of them is a package:
``attention``, ``recurrent`` (linear attention and SSMs), ``experts`` (MoE) and
``layers``, the rest. They are made of ``common``'s pieces, and multiply in the
formats of ``precision``. A builder registered for a type replaces the one before it,
so another implementation, a fused kernel say, plugs in the same way:

    @register("gdn")
    def _(config: GatedDeltaNetConfig, width: int, ports: tuple[Port, ...], ctx: Context):
        return MyGatedDeltaNet(config, width, rngs=ctx.rngs)

Modules are called as ``module(x, taps)``. ``Taps`` holds one forward pass's tap
stacks, and as attributes its input tokens and their documents (``segments(taps)``),
the output of every head run so far, and in a training step the losses layers add
(``report``). Nodes with a share key are built once and reused.

Arrays say their dtype and shape with jaxtyping, as ``x: Float[Array, "b t d"]`` for a
stream of b rows of t positions, d wide, and the tests check them on every call.
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
from bux.architecture.layers.elementwise import Activation, Dropout, Linear, Scale
from bux.architecture.layers.embedding import Embedding, Head, NextN
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
    "build_node",
    "num_params",
    "positions",
    "register",
    "report",
    "segments",
    "training",
]
