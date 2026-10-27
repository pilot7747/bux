"""The config of an exported bux model: the IR it was compiled from, and what
transformers asks of every config."""

from __future__ import annotations

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig


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
