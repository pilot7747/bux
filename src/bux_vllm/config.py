"""The config of an exported bux model, as vLLM reads ``config.json``: the IR, and
what vLLM asks of every model, derived from it.

The plugin registers it for ``model_type`` ``bux``, so vLLM loads an export without
its remote code. vLLM sizes the KV cache and checks tensor parallelism by the model's
attention heads (``num_attention_heads``, ``num_key_value_heads``, ``head_dim``):
they are its first attention layer's. ``layer_types`` lists the kind of state of each
call that keeps one, in the order they run, which tells vLLM whether the model is a
hybrid of attention and recurrent layers.
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PreTrainedConfig

from .plan import live, plan

#: The kind of state each stateful layer type keeps, as vLLM's ``layer_types``.
STATES = {"gqa": "full_attention", "mamba2": "mamba", "gdn": "linear_attention"}

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
    (``bux json``), ``bux_sites`` each layer's place (`bux.export.plan`)."""

    model_type = "bux"

    def __init__(
        self,
        bux: dict | None = None,
        bux_sites: dict | None = None,
        vocab_size: int = 0,
        hidden_size: int = 0,
        **kwargs: Any,
    ):
        for name in TRANSFORMERS_CACHE:  # its cache layers'; vLLM's are set below
            kwargs.pop(name, None)
        super().__init__(**kwargs)
        self.bux, self.bux_sites = bux, bux_sites
        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        calls = live(plan(bux).calls) if bux else []
        self.layer_types = [
            STATES[c.layer["config"]["type"]]
            for c in calls
            if c.layer["config"]["type"] in STATES
        ]
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
