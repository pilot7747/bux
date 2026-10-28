"""The config of an exported bux model: the IR it was compiled from, and what
transformers asks of every config."""

from __future__ import annotations

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import ALLOWED_ATTN_LAYER_TYPES, PreTrainedConfig

#: The export's own kinds of cache layer, beside transformers': a sparse layer's
#: with a conv on its keys and values (``attention.IndexedHybridLayer``).
OWN_LAYER_TYPES = ("indexed_hybrid",)


@strict
class BuxConfig(PreTrainedConfig):
    """A bux architecture, as transformers loads it.

    `bux` is the architecture's IR, as ``bux json`` prints it; `bux_sites` holds each
    layer's place in the model (``common.Site``), by its module path; `bux_reserved`
    the names no module took, which bux_vllm names its modules by too. The generated
    ``modeling_bux.py`` builds the model they describe.
    """

    model_type = "bux"
    bux: dict | None = None
    bux_sites: dict | None = None
    bux_reserved: list | None = None  # the names no module of the export took
    vocab_size: int = 0
    hidden_size: int = 0
    num_hidden_layers: int = 0  # the cache layers: one per call that keeps state
    layer_types: list | None = None  # each cache layer's kind, in the order they run

    def validate_layer_type(self) -> None:
        """Check that each cache layer is of a kind transformers or the export keeps,
        one for each of the ``num_hidden_layers``.

        Raises:
            ValueError: A kind is neither's, or the counts differ.
        """
        kinds = self.layer_types or []
        unknown = sorted(set(kinds) - {*ALLOWED_ATTN_LAYER_TYPES, *OWN_LAYER_TYPES})
        if unknown:
            raise ValueError(f"unknown kinds of cache layer: {unknown}")
        if self.layer_types is not None and len(kinds) != self.num_hidden_layers:
            raise ValueError(
                f"{len(kinds)} layer_types for {self.num_hidden_layers} cache layers"
            )
