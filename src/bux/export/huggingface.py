"""Export a bux model to a transformers model directory, loaded as remote code.

The directory holds:

- ``config.json``: the IR, each layer's site, the cache layers' kinds, and the
  ``auto_map`` that points transformers at the code;
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

from bux.architecture import Model
from bux.export import codegen, hf
from bux.export.plan import Plan, plan
from bux.export.weights import Convert, Tensors, tensors
from bux.lang import ir

#: The files of `bux.export.hf` a repo does not take.
_NOT_COPIED = {"__init__.py"}


def save(
    arch: ir.Architecture,
    model: Model,
    out: Path | str,
    *,
    dtype: torch.dtype = torch.float32,
) -> Plan:
    """Write `model`, built from `arch`, to the directory `out` as a transformers model.

    Args:
        arch: The architecture.
        model: Its bux model, whose weights to export, built with ``methods.repeat``
            ``unroll``.
        out: The directory, made if missing; files in it of the same names are
            replaced.
        dtype: The weights' dtype in the checkpoint.

    Returns:
        The exported model's plan.

    Raises:
        NotImplementedError: A layer, or a setting of one, is not exported yet.
    """
    p = plan(arch)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    layers = cache_layers(p)
    kinds = [kind for kind, _ in layers]
    masks = sorted({m for call in p.calls for m in _masks(call.layer)}, key=repr)
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
        "bux": arch.model_dump(mode="json"),
        "bux_sites": p.sites(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    weights = tensors(p, model, _converter)
    save_file(
        _torch(weights, dtype), out / "model.safetensors", metadata={"format": "pt"}
    )
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
    """Return the arrays as contiguous torch tensors, floats in `dtype`."""
    out = {}
    for name, array in weights.items():
        t = torch.from_numpy(np.array(array, order="C"))
        out[name] = t.to(dtype) if t.is_floating_point() else t
    return out
