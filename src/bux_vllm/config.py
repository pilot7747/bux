"""The config of an exported bux model, as vLLM reads ``config.json``: the IR, and
what vLLM asks of every model, derived from it.

The plugin registers it for ``model_type`` ``bux``, so vLLM loads an export without
its remote code. vLLM sizes the KV cache and checks tensor parallelism by the model's
attention heads (``num_attention_heads``, ``num_key_value_heads``, ``head_dim``):
they are its first attention layer's (a ``csa`` layer's latents are one key-value
head, which its own caches keep, `bux_vllm.csa`). ``layer_types`` lists the kinds
of state of each call that keeps any, in the order they run, which tells vLLM
whether the model is a hybrid of attention and recurrent layers (or convs, which
keep their last inputs).

A model whose attention is ``mla`` also carries the latent's dims, as DeepSeek's
configs do (`LATENT`): vLLM reads them to size the latent KV cache, pick its MLA
backends and check them (`bux_vllm.arch` tells vLLM the model is MLA). They are one
set for the model, so its ``mla`` layers must agree on them, and vLLM runs all of a
model's softmax attention as MLA or none of it, so ``mla`` mixes with no other. An
``mla`` layer that attends to picked keys (`sparse`) keeps its latents in caches of
the plugin's own, so a model of those alone is no MLA model for vLLM.
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PreTrainedConfig

from .kv import reads
from .plan import live, plan, ports

#: The kind of state each stateful layer type keeps, as vLLM's ``layer_types``.
STATES = {
    "gqa": "full_attention",
    "mla": "full_attention",
    "csa": "full_attention",
    "mamba": "mamba",
    "mamba2": "mamba",
    "gdn": "linear_attention",
    "kda": "linear_attention",
    "gdn2": "linear_attention",
    "rwkv7": "linear_attention",
}

#: The softmax attention layer types, whose first sets the heads vLLM reads.
ATTENTION = ("gqa", "mla", "csa")

#: The dims of an ``mla`` layer that vLLM reads from the config, as DeepSeek's name
#: them: the query and key-value latents' widths, and each head's dims.
LATENT = (
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
)


def states(c: dict, layer_ports: list | None = None) -> list[str]:
    """The kinds of state a call of the layer `c`, with the ports `layer_ports`,
    keeps, in order: its type's (`STATES`), then a ``gqa`` layer's conv state under
    ``kv_conv_size``, or a ``conv`` layer's or an ``engram``'s conv's; none for a
    layer that keeps none. An indexer's keys are in its attention's cache
    (`bux_vllm.sparse`). A ``gqa`` layer that reads its ``kv`` port names an
    attention layer of its own, over its writer's cache, and keeps nothing else."""
    if c["type"] == "engram":
        return ["conv"] if c["conv_size"] else []
    if c["type"] in ("conv", "short_conv"):
        return ["conv"]
    if c["type"] not in STATES:
        return []
    if c["type"] == "gqa" and reads(layer_ports or [], "kv"):
        return [STATES["gqa"]]
    conv = ["conv"] if c.get("kv_conv_size") else []
    return [STATES[c["type"]], *conv]


def sparse(c: dict, layer_ports: list | None = None) -> bool:
    """Whether the softmax attention layer `c`, with the ports `layer_ports`,
    attends to picked keys alone: it has an indexer, or reads another layer's picks
    from its ``index`` port."""
    return bool(c.get("indexer")) or reads(layer_ports or [], "index") is not None


#: What ``config.json`` says of transformers' cache layers, which vLLM reads
#: otherwise (a model-wide ``sliding_window`` would window every layer): the layers'
#: own settings are in the IR.
TRANSFORMERS_CACHE = (
    "layer_types",
    "num_hidden_layers",
    "per_layer_config",
    "sliding_window",
    "number_of_conv_states",
)


class BuxConfig(PreTrainedConfig):
    """A bux architecture as vLLM loads it: ``bux`` is its IR as JSON
    (``bux json``), ``bux_sites`` each layer's place (`bux.export.plan`), and
    ``bux_reserved`` the names no module took, which the plan names them by too."""

    model_type = "bux"

    def __init__(
        self,
        bux: dict | None = None,
        bux_sites: dict | None = None,
        bux_reserved: list | None = None,
        vocab_size: int = 0,
        hidden_size: int = 0,
        **kwargs: Any,
    ):
        for name in TRANSFORMERS_CACHE:  # its cache layers'; vLLM's are set below
            kwargs.pop(name, None)
        super().__init__(**kwargs)
        self.bux, self.bux_sites = bux, bux_sites
        self.bux_reserved = bux_reserved or []
        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        calls = live(plan(bux, self.bux_reserved).calls) if bux else []
        self.layer_types = [
            k for c in calls for k in states(c.layer["config"], ports(c.layer))
        ]
        self.num_hidden_layers = len(self.layer_types)
        layers = [c.layer for c in calls]
        self._set_heads(layers)
        embed = next((n for n in layers if n["config"]["type"] == "embed"), None)
        if embed is not None and embed["config"]["max_positions"]:
            self.max_position_embeddings = embed["config"]["max_positions"]

    def _set_heads(self, layers: list[dict]) -> None:
        """Set the attention heads vLLM sizes the KV cache by, from the first
        attention layer (none for a model without one), and an MLA's latent dims:
        a sparse ``mla`` layer's latents are one head of its own caches.

        Raises:
            NotImplementedError: The model mixes ``mla`` that vLLM runs as MLA with
                other softmax attention, or its ``mla`` layers differ in their heads
                or dims.
        """
        attn = [n for n in layers if n["config"]["type"] in ATTENTION]
        dense = [n["config"] for n in attn if _dense_mla(n)]
        if dense and len(dense) < len(attn):
            raise NotImplementedError("mla beside other attention: MLA is model-wide")
        if not attn:
            self.num_attention_heads = self.num_key_value_heads = self.head_dim = 0
            return
        if dense:
            self._set_latent(dense)
            return
        c = attn[0]["config"]
        if c["type"] == "mla":
            self.num_attention_heads, self.num_key_value_heads = c["num_heads"], 1
            self.head_dim = c["kv_lora_rank"] + c["qk_rope_head_dim"]
            return
        heads, kv = c["num_heads"], c.get("num_kv_heads", 1)  # csa: one latent head
        self.num_attention_heads, self.num_key_value_heads = heads, kv
        self.head_dim = c["head_dim"] or attn[0]["width"] // heads

    def _set_latent(self, attn: list[dict]) -> None:
        """Set the heads and `LATENT` dims of the ``mla`` layers `attn`: every head
        has keys and values of its own, as in DeepSeek's configs.

        Raises:
            NotImplementedError: They differ in their heads or dims.
        """
        dims = {tuple(c[n] for n in ("num_heads", *LATENT)) for c in attn}
        if len(dims) > 1:
            raise NotImplementedError("mla layers of different dims not in vLLM yet")
        c = attn[0]
        for name in LATENT:
            setattr(self, name, c[name])
        self.num_attention_heads = self.num_key_value_heads = c["num_heads"]
        self.head_dim = c["qk_nope_head_dim"] + c["qk_rope_head_dim"]


def _dense_mla(layer: dict) -> bool:
    """Whether `layer` is an ``mla`` layer that vLLM runs as MLA: not `sparse`."""
    c = layer["config"]
    return c["type"] == "mla" and not sparse(c, ports(layer))


def latent(config: Any) -> bool:
    """Whether vLLM runs the model of `config` as multi-head latent attention: its
    attention is ``mla``, whose latent's dims `BuxConfig` sets."""
    return getattr(config, "kv_lora_rank", None) is not None
