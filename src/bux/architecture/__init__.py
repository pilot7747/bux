"""Reference Flax NNX backend: turns an Architecture into a runnable nnx.Module.

Every IR node kind maps to one small module (``graph``), so the graph side is complete.
Leaves come from a registry keyed by config type (``registry``); the other modules here
each register one family of layers. Other mixers plug in the same way:

    @register("gdn")
    def _(config: GatedDeltaNetConfig, width: int, ports: tuple[Port, ...], ctx: Context):
        return MyGatedDeltaNet(config, width, rngs=ctx.rngs)

Modules are called as ``module(x, taps)``: ``taps`` holds the tap stacks for one
forward pass, and under reserved keys the input tokens and the logits of every head
run so far. Repeats are unrolled; nodes with a share key are built once and reused.
"""

# Importing the layer modules registers their layers.
from bux.architecture.attention import Attention
from bux.architecture.conv import CausalConv
from bux.architecture.elementwise import Activation, Dropout, Linear, Scale
from bux.architecture.embedding import Embedding, Head, NextN
from bux.architecture.engram import Engram
from bux.architecture.graph import build_node
from bux.architecture.mlp import MLP
from bux.architecture.model import Model, num_params
from bux.architecture.moe import MoE
from bux.architecture.norm import DyT, Norm
from bux.architecture.registry import NEXTN, OUTPUTS, TOKENS, Context, Taps, register

__all__ = [
    "MLP",
    "NEXTN",
    "OUTPUTS",
    "TOKENS",
    "Activation",
    "Attention",
    "CausalConv",
    "Context",
    "Dropout",
    "DyT",
    "Embedding",
    "Engram",
    "Head",
    "Linear",
    "MoE",
    "Model",
    "NextN",
    "Norm",
    "Scale",
    "Taps",
    "build_node",
    "num_params",
    "register",
]
