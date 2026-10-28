"""The config of an exported bux model, as vLLM reads ``config.json``: the IR, and
what vLLM asks of every model, derived from it.

The plugin registers it for ``model_type`` ``bux``, so vLLM loads an export without
its remote code. vLLM sizes the KV cache and checks tensor parallelism by the model's
attention heads (``num_attention_heads``, ``num_key_value_heads``, ``head_dim``):
they are its first attention layer's. ``layer_types`` lists the kinds of state of
each call that keeps any, in the order they run, which tells vLLM whether the model
is a hybrid of attention and recurrent layers (or convs, which keep their last
inputs).
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PreTrainedConfig

from .plan import live, plan

#: The kind of state each stateful layer type keeps, as vLLM's ``layer_types``.
STATES = {"gqa": "full_attention", "mamba2": "mamba", "gdn": "linear_attention"}


def states(c: dict) -> list[str]:
    """The kinds of state a call of the layer `c` keeps, in order: its type's
    (`STATES`), and a conv state after a ``gqa`` layer's KV cache under
    ``kv_conv_size``; none for a layer that keeps none."""
    if c["type"] not in STATES:
        return []
    return [STATES[c["type"]], *(["conv"] if c.get("kv_conv_size") else [])]


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
        self.layer_types = [kind for c in calls for kind in states(c.layer["config"])]
        self.num_hidden_layers = len(self.layer_types)
        layers = [c.layer for c in calls]
        self._set_heads(layers)
        embed = next((n for n in layers if n["config"]["type"] == "embed"), None)
        if embed is not None and embed["config"]["max_positions"]:
            self.max_position_embeddings = embed["config"]["max_positions"]

    def _set_heads(self, layers: list[dict]) -> None:
        """Set the attention heads vLLM sizes the KV cache by, from the first
        attention layer: none for a model without one."""
        attn = next((n for n in layers if n["config"]["type"] == "gqa"), None)
        if attn is None:
            self.num_attention_heads = self.num_key_value_heads = self.head_dim = 0
            return
        c = attn["config"]
        self.num_attention_heads, self.num_key_value_heads = (
            c["num_heads"],
            c["num_kv_heads"],
        )
        self.head_dim = c["head_dim"] or attn["width"] // c["num_heads"]
