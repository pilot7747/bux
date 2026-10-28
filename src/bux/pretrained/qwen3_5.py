"""Qwen3.5's checkpoints (``model_type`` ``qwen3_5``): its vision encoder and its
text decoder, layer by layer, as `bux.pretrained` reads and writes them.

The vision encoder, after the embedding, is the optional group ``vision``: the patch
embedding, each block's LayerNorm, attention, LayerNorm and MLP, and the merger's
LayerNorm and two linear layers. The checkpoint's patch embedding sees two frames of
a video; bux's sees one image, so its weights are the sum over the two, which a still
image's identical frames make the same. Written back, a change to the sum is shared
equally by the two frames.

Each decoder layer is four of bux's: an RMSNorm, the token mixer (``gdn`` or a gated
``gqa``), an RMSNorm and a ``gated_mlp``. The checkpoint fuses what bux keeps apart:
Gated DeltaNet's q, k and v in one projection, and the attention's queries and output
gate, interleaved by head, in another. The multi-token prediction module, after the
head, is the optional group ``mtp``: its nextn layer, its decoder layers and its norm.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Shaped

from bux import pretrained
from bux.pretrained import Part, Spec, same

#: An array of a checkpoint or of bux.
type Tensor = Shaped[np.ndarray, "..."]


@pretrained.register("qwen3_5")
def specs(config: dict) -> list[Spec]:
    """Return the checkpoint's layers: the embedding, the vision encoder, every
    decoder layer's four, the final norm, the head unless it is tied, and the MTP
    module."""
    text = config["text_config"]
    lm = "model.language_model."
    out = [Spec("embed", (same("tokens.weight", f"{lm}embed_tokens.weight"),))]
    if "vision_config" in config:
        vision = _vision(config["vision_config"], "model.visual.")
        out += [x._replace(group="vision") for x in vision]
    for i, kind in enumerate(text["layer_types"]):
        out += _decoder_layer(text, f"{lm}layers.{i}.", kind)
    out.append(_norm(f"{lm}norm.weight"))
    if not config.get("tie_word_embeddings", text.get("tie_word_embeddings", False)):
        out.append(Spec("head", (same("proj.weight", "lm_head.weight"),)))
    if n := text.get("mtp_num_hidden_layers", 0):
        nextn = (
            same("hidden_norm.weight", "mtp.pre_fc_norm_hidden.weight"),
            same("embed_norm.weight", "mtp.pre_fc_norm_embedding.weight"),
            same("proj.weight", "mtp.fc.weight"),
        )
        mtp = [Spec("nextn", nextn)]
        for i in range(n):
            mtp += _decoder_layer(text, f"mtp.layers.{i}.", "full_attention")
        mtp.append(_norm("mtp.norm.weight"))
        out += [x._replace(group="mtp") for x in mtp]
    return out


def _norm(key: str) -> Spec:
    return Spec("rmsnorm", (same("weight", key),))


def _decoder_layer(text: dict, at: str, kind: str) -> list[Spec]:
    """Return a decoder layer's four: norm, token mixer, norm and MLP."""
    mixer = (
        _gdn(text, f"{at}linear_attn.")
        if kind == "linear_attention"
        else _gqa(text, f"{at}self_attn.")
    )
    mlp = tuple(
        same(f"{k}.weight", f"{at}mlp.{k}_proj.weight") for k in ("up", "gate", "down")
    )
    return [
        _norm(f"{at}input_layernorm.weight"),
        mixer,
        _norm(f"{at}post_attention_layernorm.weight"),
        Spec("gated_mlp", mlp),
    ]


def _fused(names: tuple[str, ...], key: str, cuts: list[int]) -> Part:
    """`names`, the rows of `key` cut at `cuts`, one after another."""
    return Part(
        names,
        (key,),
        lambda x: tuple(np.split(x, cuts)),
        lambda ys, _: (np.concatenate(ys),),
    )


def _gdn(text: dict, at: str) -> Spec:
    """Gated DeltaNet: q, k and v apart, the conv as ``(size, channels)``."""
    dk = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    conv = Part(
        ("conv.weight",),
        (f"{at}conv1d.weight",),
        lambda w: (w[:, 0].T,),
        lambda ys, _: (ys[0].T[:, None],),
    )
    return Spec(
        "gdn",
        (
            _fused(
                ("q.weight", "k.weight", "v.weight"),
                f"{at}in_proj_qkv.weight",
                [dk, 2 * dk],
            ),
            conv,
            same("gate.weight", f"{at}in_proj_z.weight"),
            same("norm.weight", f"{at}norm.weight"),
            same("o.weight", f"{at}out_proj.weight"),
            same("b.weight", f"{at}in_proj_b.weight"),
            same("a.weight", f"{at}in_proj_a.weight"),
            same("A_log", f"{at}A_log"),
            same("dt_bias", f"{at}dt_bias"),
        ),
    )


def _gqa(text: dict, at: str) -> Spec:
    """Gated attention: the queries and the output gate apart, each head's."""
    heads, dh = text["num_attention_heads"], text["head_dim"]

    def read(w: Tensor) -> tuple[Tensor, Tensor]:
        qg = w.reshape(heads, 2, dh, -1)
        return qg[:, 0].reshape(heads * dh, -1), qg[:, 1].reshape(heads * dh, -1)

    def write(ys: tuple[Tensor, ...], _: tuple) -> tuple[Tensor]:
        q, gate = (y.reshape(heads, 1, dh, -1) for y in ys)
        return (np.concatenate([q, gate], 1).reshape(2 * heads * dh, -1),)

    return Spec(
        "gqa",
        (
            Part(("q.weight", "gate.weight"), (f"{at}q_proj.weight",), read, write),
            same("k.weight", f"{at}k_proj.weight"),
            same("v.weight", f"{at}v_proj.weight"),
            same("o.weight", f"{at}o_proj.weight"),
            same("q_norm.weight", f"{at}q_norm.weight"),
            same("k_norm.weight", f"{at}k_norm.weight"),
        ),
    )


def _patches(key: str) -> Part:
    """The patch embedding's projection: the checkpoint's, of two frames ``(d,
    channels, frames, size, size)``, summed over them; written back, a change to the
    sum is shared equally by the frames."""

    def read(w: Tensor) -> tuple[Tensor]:
        return (w.astype(np.float32).sum(2).reshape(w.shape[0], -1),)

    def write(ys: tuple[Tensor, ...], old: tuple[Tensor, ...]) -> tuple[Tensor]:
        before = old[0].astype(np.float32)
        frames = before.shape[2]
        change = ys[0].reshape(before.sum(2).shape) - before.sum(2)
        return (before + change[:, :, None] / frames,)

    return Part(("proj.weight",), (key,), read, write)


def _vision(vision: dict, at: str) -> list[Spec]:
    """Return the vision encoder's layers: the patch embedding, every block's four,
    and the merger's norm and two linear layers."""

    def layernorm(key: str) -> Spec:
        return Spec(
            "layernorm", (same("weight", f"{key}.weight"), same("bias", f"{key}.bias"))
        )

    def linear(to: str, key: str) -> tuple[Part, Part]:
        return same(f"{to}.weight", f"{key}.weight"), same(f"{to}.bias", f"{key}.bias")

    patches = (
        _patches(f"{at}patch_embed.proj.weight"),
        same("proj.bias", f"{at}patch_embed.proj.bias"),
        same("positions.weight", f"{at}pos_embed.weight"),
    )
    out = [Spec("patches", patches)]
    d = vision["hidden_size"]
    for i in range(vision["depth"]):
        block = f"{at}blocks.{i}."
        attn = (
            _fused(
                ("q.weight", "k.weight", "v.weight"),
                f"{block}attn.qkv.weight",
                [d, 2 * d],
            ),
            _fused(("q.bias", "k.bias", "v.bias"), f"{block}attn.qkv.bias", [d, 2 * d]),
            *linear("o", f"{block}attn.proj"),
        )
        mlp = (
            *linear("up", f"{block}mlp.linear_fc1"),
            *linear("down", f"{block}mlp.linear_fc2"),
        )
        out += [
            layernorm(f"{block}norm1"),
            Spec("gqa", attn),
            layernorm(f"{block}norm2"),
            Spec("mlp", mlp),
        ]
    return [
        *out,
        layernorm(f"{at}merger.norm"),
        Spec("linear", linear("proj", f"{at}merger.linear_fc1")),
        Spec("linear", linear("proj", f"{at}merger.linear_fc2")),
    ]
