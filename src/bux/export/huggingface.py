"""Export a bux model to a transformers model directory, loaded as remote code.

The directory holds:

- ``config.json``: the IR, each layer's site, the cache layers' kinds, the names no
  module takes (`bux.export.plan.RESERVED`), and the ``auto_map`` that points
  transformers at the code;
- ``configuration_bux.py`` and the exported layers' files (`bux.export.hf`), as they
  are in bux;
- ``modeling_bux.py``, generated from the architecture (`bux.export.codegen`);
- ``model.safetensors``, the weights by the generated modules' paths.

::

    from transformers import AutoModelForCausalLM

    save(arch, model, "out/")
    hf_model = AutoModelForCausalLM.from_pretrained("out/", trust_remote_code=True)
"""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from bux import config as cfg
from bux.architecture import Model
from bux.export import codegen, hf
from bux.export.hf.common import KEEP_FLOAT32
from bux.export.plan import RESERVED, Plan, plan
from bux.export.weights import Convert, Tensors, tensors
from bux.lang import ir

#: The files of `bux.export.hf` a repo does not take.
_NOT_COPIED = {"__init__.py"}


def _check_pointer(arch: ir.Architecture) -> None:
    """Raise for a head's copy pointer, before anything is written.

    Raises:
        NotImplementedError: A head has a copy pointer, whose mixture generation
            would sample from by a cache of the pointer's keys, not exported yet.
    """
    for layer in arch.layers():
        c = layer.config
        if isinstance(c, cfg.LMHeadConfig) and c.pointer is not None:
            raise NotImplementedError(
                "head: pointer not exported yet (a copy pointer's mixture, which "
                "generation would sample by a cache of the pointer's keys)"
            )


def save(
    arch: ir.Architecture,
    model: Model | None,
    out: Path | str,
    *,
    dtype: torch.dtype = torch.float32,
    max_positions: int | None = None,
) -> Plan:
    """Write `model`, built from `arch`, to the directory `out` as a transformers model.

    Args:
        arch: The architecture.
        model: Its bux model, whose weights to export, built with ``methods.repeat``
            ``unroll``; None writes no weights, and removes those an earlier export
            left in `out`, for vLLM to time the architecture on random ones
            (``load_format="dummy"``).
        out: The directory, made if missing; files in it of the same names are
            replaced.
        dtype: The weights' dtype in the checkpoint.
        max_positions: The longest sequence the model is known to take, as
            ``max_position_embeddings``, which vLLM serves up to by default; None
            says nothing of it.

    Returns:
        The exported model's plan.

    Raises:
        NotImplementedError: A layer, or a setting of one, is not exported yet.
    """
    _check_pointer(arch)
    p = plan(arch)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    layers = cache_layers(p)
    kinds = [kind for kind, _ in layers]
    # The masks of the layers it runs as it generates; those after them run without
    # a cache, and make theirs as they need them.
    masks = sorted({m for call in p.generation() for m in _masks(call.layer)}, key=repr)
    (out / "modeling_bux.py").write_text(codegen.modeling(p, masks))
    for f in Path(hf.__path__[0]).glob("*.py"):
        if f.name not in _NOT_COPIED:
            shutil.copy(f, out / f.name)
    config = {
        "architectures": ["BuxForCausalLM"],
        "model_type": "bux",
        "auto_map": {
            "AutoConfig": "configuration_bux.BuxConfig",
            "AutoModel": "modeling_bux.BuxModel",
            "AutoModelForCausalLM": "modeling_bux.BuxForCausalLM",
        },
        "vocab_size": arch.vocab_size,
        "hidden_size": arch.d_model,
        "num_hidden_layers": len(kinds),
        "layer_types": kinds,
        **_per_layer(layers),
        "tie_word_embeddings": False,
        "dtype": str(dtype).removeprefix("torch."),  # what dtype="auto" loads in
        **({} if max_positions is None else {"max_position_embeddings": max_positions}),
        "bux": arch.model_dump(mode="json"),
        "bux_sites": p.sites(),
        "bux_reserved": sorted(RESERVED),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    checkpoint = out / "model.safetensors"
    if model is None:
        checkpoint.unlink(missing_ok=True)  # an earlier export's, of another model
        return p
    weights = tensors(p, model, _converter)
    save_file(_torch(weights, dtype), checkpoint, metadata={"format": "pt"})
    return p


#: transformers' defaults of what a cache layer's config may set for it alone.
_CACHE_DEFAULTS = {"sliding_window": None, "number_of_conv_states": 1}


def cache_layers(p: Plan) -> list[tuple[str, dict]]:
    """Return the kind of each cache layer, in the order the calls that keep state run
    as the model generates (`Plan.generation`), and what its config sets for it alone
    (`bux.export.codegen.cache_layer`), with settings of `_CACHE_DEFAULTS` (a sliding
    window, conv states)."""
    layers = (codegen.cache_layer(call.layer) for call in p.generation())
    return [layer for layer in layers if layer is not None]


def _per_layer(layers: list[tuple[str, dict]]) -> dict:
    """Return the config's settings of single cache layers: ``per_layer_config``, by
    the layer's index, and transformers' default of each for the others."""
    overrides = {i: settings for i, (_, settings) in enumerate(layers) if settings}
    if not overrides:
        return {}
    names = {name for settings in overrides.values() for name in settings}
    return {n: _CACHE_DEFAULTS[n] for n in sorted(names)} | {
        "per_layer_config": overrides
    }


def _masks(layer: ir.Layer) -> list[tuple[str, object]]:
    """Return the masks `layer`'s exported class reads."""
    fn = getattr(codegen.layer_class(layer), "masks", None)
    return [tuple(m) for m in fn(layer.config.model_dump(mode="json"))] if fn else []


def _converter(kind: str) -> Convert | None:
    """Return the conversion of `kind`'s tensors its exported class declares, if any."""
    cls = codegen.layer_classes().get(kind)
    return getattr(cls, "bux_weights", None)


def _torch(weights: Tensors, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Return the arrays as contiguous torch tensors, floats in `dtype` but the state
    bux keeps in float32 (``KEEP_FLOAT32``)."""
    out = {}
    for name, array in weights.items():
        t = torch.from_numpy(np.array(array, order="C"))
        kept = name.rsplit(".", 1)[-1] in KEEP_FLOAT32
        out[name] = t.to(dtype) if t.is_floating_point() and not kept else t
    return out
