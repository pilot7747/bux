"""Released checkpoints load into bux models that compute what transformers does
(bux.pretrained): Qwen3.5's, shrunk, with random weights."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from bux import pretrained
from bux.architecture import Model
from bux.export import weights
from bux.export.plan import plan
from bux.lang import compile

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

#: Qwen3.5's text decoder, shrunk: three Gated DeltaNet layers and one of gated
#: attention.
QWEN3_5 = """
d_model    = 64
vocab_size = 128

norm  = rmsnorm(zero_centered=true)
delta = gdn(num_heads=2, num_v_heads=4, head_dim=16)
attn  = gqa(num_heads=4, num_kv_heads=2, head_dim=32, gated=true, qk_norm=norm,
            rope=rope(base=10000000, fraction=0.25))
ffn   = gated_mlp(intermediate_size=96, activation=swish)

layer(mixer) = [norm mixer] [norm ffn]
D = layer(delta)
A = layer(attn)

model = embed "DDDA" norm head(tied={tied})
"""


#: The tokens an image is written with in the shrunk Qwen3.5.
IMAGE = 5
#: Its vision encoder, shrunk, before its decoder.
VISION = """
vnorm  = layernorm(bias=true)
vattn  = gqa(num_heads=2, num_kv_heads=2, bias=true, causal=false, rope=rope(axial=2))
vffn   = mlp(intermediate_size=64, activation=gelu_tanh, bias=true)
vision = (patches(size=16, features=32, grid=48) ([vnorm vattn] [vnorm vffn])^2
          vnorm merge_patches(size=2) linear(features=128, bias=true) gelu
          linear(features=d_model, bias=true))
"""


def qwen3_5(tied: bool, depth: int = 1):
    """A shrunk Qwen3.5, as transformers builds it, with random weights: a vision
    encoder of `depth` blocks, and image tokens 5 between 3 and 4."""
    from transformers.models.qwen3_5 import configuration_qwen3_5 as c

    text = c.Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        tie_word_embeddings=tied,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    )
    vision = c.Qwen3_5VisionConfig(
        depth=depth,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=64,
    )
    config = c.Qwen3_5Config(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        tie_word_embeddings=tied,
        image_token_id=IMAGE,
        video_token_id=6,
        vision_start_token_id=3,
        vision_end_token_id=4,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3_5ForConditionalGeneration(config).eval()
    with torch.no_grad():  # nonzero norm gains and dt biases, so a swap shows
        for name, p in model.named_parameters():
            if "norm" in name or "dt_bias" in name:
                p.add_(0.1 * torch.randn_like(p))
    return model


@pytest.mark.parametrize("tied", [True, False])
def test_qwen3_5_loads_and_computes_transformers_logits(tmp_path, tied):
    hf = qwen3_5(tied)
    hf.save_pretrained(tmp_path)
    arch = compile(QWEN3_5.replace("{tied}", str(tied).lower()))
    model = Model(arch, rngs=nnx.Rngs(0))
    pretrained.load(model, arch, pretrained.read(tmp_path))

    tokens = np.random.default_rng(0).integers(0, 128, (2, 24))
    with torch.no_grad():
        want = hf(input_ids=torch.tensor(tokens)).logits.float().numpy()
    with jax.default_matmul_precision("highest"):
        got = model(jnp.asarray(tokens))
    np.testing.assert_allclose(np.asarray(got), want, atol=2e-4, rtol=2e-4)


def test_a_checkpoint_of_other_layers_names_the_first_difference(tmp_path):
    qwen3_5(True).save_pretrained(tmp_path)
    arch = compile(QWEN3_5.replace("{tied}", "true").replace('"DDDA"', '"DDAA"'))
    model = Model(arch, rngs=nnx.Rngs(0))
    with pytest.raises(
        ValueError, match="'gqa' layer, but the checkpoint's is a 'gdn'"
    ):
        pretrained.load(model, arch, pretrained.read(tmp_path))


def test_skipped_layers_keep_their_fresh_parameters(tmp_path):
    qwen3_5(True).save_pretrained(tmp_path)
    arch = compile(QWEN3_5.replace("{tied}", "true"))
    model = Model(arch, rngs=nnx.Rngs(0))
    fresh = Model(arch, rngs=nnx.Rngs(0))
    pretrained.load(model, arch, pretrained.read(tmp_path), skip=("gqa",))
    got, want = (weights.tensors(plan(arch), m) for m in (model, fresh))
    for name in got:
        same = np.array_equal(got[name], want[name])
        assert same == (".attn." in name), name


def test_a_run_starts_from_the_weights_it_names(tmp_path, losses):
    qwen3_5(True).save_pretrained(tmp_path / "ckpt")
    source = QWEN3_5.replace("{tied}", "true")
    fresh = losses(source, steps=1)
    loaded = losses(source, steps=1, weights=f'hf(name="{tmp_path / "ckpt"}")')
    again = losses(source, steps=1, weights=f'hf(name="{tmp_path / "ckpt"}")')
    assert loaded == again != fresh


#: The end of the shrunk Qwen3.5's model with its MTP module, its head untied.
MTP_TAIL = (
    "norm @h out mtp\nshared out = head(tied=false)\n"
    "mtp = $h nextn(norm=norm) A norm out"
)


def with_mtp(folder):
    """Add an MTP module of random tensors to the checkpoint in `folder`, as the
    released ones have, and return them."""
    import json

    from safetensors.numpy import save_file

    rng = np.random.default_rng(1)
    at, d = "mtp.layers.0.", 64
    shapes = {
        "mtp.fc.weight": (d, 2 * d),
        "mtp.pre_fc_norm_hidden.weight": (d,),
        "mtp.pre_fc_norm_embedding.weight": (d,),
        "mtp.norm.weight": (d,),
        f"{at}input_layernorm.weight": (d,),
        f"{at}post_attention_layernorm.weight": (d,),
        f"{at}self_attn.q_proj.weight": (2 * 4 * 32, d),
        f"{at}self_attn.k_proj.weight": (2 * 32, d),
        f"{at}self_attn.v_proj.weight": (2 * 32, d),
        f"{at}self_attn.o_proj.weight": (d, 4 * 32),
        f"{at}self_attn.q_norm.weight": (32,),
        f"{at}self_attn.k_norm.weight": (32,),
        f"{at}mlp.up_proj.weight": (96, d),
        f"{at}mlp.gate_proj.weight": (96, d),
        f"{at}mlp.down_proj.weight": (d, 96),
    }
    tensors = {k: rng.standard_normal(s).astype(np.float32) for k, s in shapes.items()}
    save_file(tensors, folder / "mtp.safetensors")
    config = json.loads((folder / "config.json").read_text())
    config["text_config"]["mtp_num_hidden_layers"] = 1
    (folder / "config.json").write_text(json.dumps(config))
    return tensors


@pytest.mark.parametrize("mtp", [True, False])
def test_a_model_loads_the_mtp_module_or_leaves_it_out(tmp_path, mtp):
    qwen3_5(True).save_pretrained(tmp_path)
    tensors = with_mtp(tmp_path)
    tail = "@h out mtp" if mtp else "out"
    source = QWEN3_5.replace("{tied}", "true").replace(
        "norm head(tied=true)",
        f"norm {tail}\nshared out = head(tied=true)\nmtp = $h nextn(norm=norm) A norm out",
    )
    arch = compile(source)
    model = Model(arch, rngs=nnx.Rngs(0))
    pretrained.load(model, arch, pretrained.read(tmp_path))
    got = weights.tensors(plan(arch), model)
    proj = [v for k, v in got.items() if k.endswith("nextn.proj.weight")]
    assert len(proj) == mtp
    if mtp:
        np.testing.assert_array_equal(proj[0], tensors["mtp.fc.weight"])


def test_a_skipped_layer_takes_the_place_of_another_type_or_is_new(tmp_path):
    qwen3_5(True).save_pretrained(tmp_path)
    layers = pretrained.read(tmp_path)
    swapped = QWEN3_5.replace("{tied}", "true").replace(
        "A = layer(attn)", "A = layer(gla(num_heads=4, head_dim=16))"
    )
    added = QWEN3_5.replace("{tied}", "true").replace(
        "[norm ffn]", "[norm ffn] [scale(per_channel=true)]"
    )
    for source, skip in [(swapped, "gla"), (added, "scale")]:
        arch = compile(source)
        pretrained.load(Model(arch, rngs=nnx.Rngs(0)), arch, layers, skip=(skip,))


def page(rng: np.random.Generator, h: int, w: int):
    """An image of h x w random pixels: its patches and grid, as bux reads them."""
    from bux.data import images

    image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return images.patches(image, size=16, merge=2)


def vision_language(tied: bool = False) -> str:
    """The source of the shrunk Qwen3.5 with its vision encoder and M-RoPE."""
    source = QWEN3_5.replace("{tied}", str(tied).lower()).replace(
        "rope=rope(base=10000000, fraction=0.25)",
        "rope=rope(base=10000000, fraction=0.25, sections=[2, 1, 1],"
        " sections_layout=interleaved)",
    )
    return source.replace('"DDDA"', f'image(vision, token={IMAGE}) "DDDA"').replace(
        "model =", VISION + "\nmodel ="
    )


def row_with_images():
    """A row of text and two images, as transformers takes it and as bux does: its
    tokens, transformers' keyword arguments, and bux's `Images`."""
    from bux.architecture import Images
    from bux.data import images

    rng = np.random.default_rng(0)
    (a, grid_a), (b, grid_b) = page(rng, 64, 96), page(rng, 32, 32)
    text = list(rng.integers(7, 128, 5))
    tokens = np.array([text[:2] + [3] + [IMAGE] * 6 + [4] + text[2:] + [3, IMAGE, 4]])
    is_image = tokens[0] == IMAGE
    pixels = np.concatenate([a, b])
    frames = (pixels / 255 - 0.5) / 0.5  # two of a still image
    frames = np.repeat(frames.reshape(-1, 3, 1, 256), 2, 2).reshape(len(pixels), -1)
    theirs = {
        "input_ids": torch.tensor(tokens),
        "pixel_values": torch.tensor(frames, dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 4, 6], [1, 2, 2]]),
        "mm_token_type_ids": torch.tensor(is_image[None].astype(np.int64)),
    }
    rope = images.mrope(is_image, [(2, 3), (1, 1)])
    pad = 8  # patches of no image, after the row's
    patches = np.concatenate([pixels, np.zeros((pad, 768), np.uint8)])
    positions = np.concatenate([np.arange(24), np.arange(4), np.arange(pad)])
    grid = np.concatenate([grid_a, grid_b, np.zeros((pad, 2), np.int32)])
    sizes = np.array([[4, 6]] * 24 + [[2, 2]] * 4 + [[1, 1]] * pad)
    ours = Images(
        *(jnp.asarray(x[None]) for x in (patches, positions, grid, sizes, rope))
    )
    return tokens, theirs, ours


def bux_logits(model: Model, tokens, images) -> np.ndarray:
    with jax.default_matmul_precision("highest"):
        return np.asarray(model.outputs(jnp.asarray(tokens), None, images)[0])


def test_qwen3_5_with_images_computes_transformers_logits(tmp_path):
    hf = qwen3_5(False, depth=2)
    hf.save_pretrained(tmp_path)
    arch = compile(vision_language())
    model = Model(arch, rngs=nnx.Rngs(0))
    pretrained.load(model, arch, pretrained.read(tmp_path))
    tokens, theirs, ours = row_with_images()
    with torch.no_grad():
        want = hf(**theirs).logits.numpy()
        hf_rope, _ = hf.model.get_rope_index(
            theirs["input_ids"],
            theirs["mm_token_type_ids"],
            image_grid_thw=theirs["image_grid_thw"],
        )
    np.testing.assert_array_equal(ours.rope[0], hf_rope[:, 0].numpy().T)
    np.testing.assert_allclose(
        bux_logits(model, tokens, ours), want, atol=2e-4, rtol=2e-4
    )


# --- Saving ------------------------------------------------------------------------


def test_a_model_saves_as_the_checkpoint_it_loaded(tmp_path):
    from safetensors.numpy import load_file

    qwen3_5(False, depth=2).save_pretrained(tmp_path / "released")
    with_mtp(tmp_path / "released")
    arch = compile(vision_language().replace("norm head(tied=false)", MTP_TAIL))
    model = Model(arch, rngs=nnx.Rngs(0))
    pretrained.load(model, arch, pretrained.read(tmp_path / "released"))
    assert pretrained.save(model, arch, tmp_path / "released", tmp_path / "saved") == []
    released = {
        k: v
        for f in (tmp_path / "released").glob("*.safetensors")
        for k, v in load_file(f).items()
    }
    saved = {
        k: v
        for f in (tmp_path / "saved").glob("*.safetensors")
        for k, v in load_file(f).items()
    }
    assert saved.keys() == released.keys()
    for k, v in released.items():
        np.testing.assert_allclose(saved[k], v, rtol=1e-6, atol=1e-6, err_msg=k)
    assert (tmp_path / "saved" / "config.json").read_text() == (
        tmp_path / "released" / "config.json"
    ).read_text()


def test_a_trained_model_saved_computes_in_transformers_what_it_does_in_bux(tmp_path):
    qwen3_5(False, depth=2).save_pretrained(tmp_path / "released")
    arch = compile(vision_language())
    model = Model(arch, rngs=nnx.Rngs(0))
    pretrained.load(model, arch, pretrained.read(tmp_path / "released"))
    rng = np.random.default_rng(1)  # as training would, every parameter moves
    for _, p in nnx.iter_graph(model):
        if isinstance(p, nnx.Param):
            p[...] = p[...] + 0.02 * rng.standard_normal(p[...].shape).astype(
                np.float32
            )
    pretrained.save(model, arch, tmp_path / "released", tmp_path / "saved")
    hf = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        tmp_path / "saved"
    ).eval()
    tokens, theirs, ours = row_with_images()
    with torch.no_grad():
        want = hf(**theirs).logits.numpy()
    np.testing.assert_allclose(
        bux_logits(model, tokens, ours), want, atol=2e-4, rtol=2e-4
    )


def test_a_model_without_mtp_saves_the_released_mtp(tmp_path):
    qwen3_5(True).save_pretrained(tmp_path / "released")
    tensors = with_mtp(tmp_path / "released")
    arch = compile(QWEN3_5.replace("{tied}", "true"))
    model = Model(arch, rngs=nnx.Rngs(0))
    pretrained.load(model, arch, pretrained.read(tmp_path / "released"))
    left = pretrained.save(model, arch, tmp_path / "released", tmp_path / "saved")
    assert left == ["mtp", "vision"]
    saved = pretrained.Checkpoint(tmp_path / "saved")
    np.testing.assert_array_equal(saved["mtp.fc.weight"], tensors["mtp.fc.weight"])


def test_a_model_of_another_architecture_does_not_save_as_the_released_one(tmp_path):
    qwen3_5(True).save_pretrained(tmp_path / "released")
    swapped = QWEN3_5.replace("{tied}", "true").replace(
        "A = layer(attn)", "A = layer(gla(num_heads=4, head_dim=16))"
    )
    arch = compile(swapped)
    model = Model(arch, rngs=nnx.Rngs(0))
    with pytest.raises(
        ValueError, match="'gla' layer, but the checkpoint's is a 'gqa'"
    ):
        pretrained.save(model, arch, tmp_path / "released", tmp_path / "saved")


def test_a_model_of_text_loads_a_checkpoint_with_a_vision_encoder(tmp_path):
    qwen3_5(True).save_pretrained(tmp_path)
    layers = pretrained.read(tmp_path)
    assert {x.group for x in layers} == {None, "vision"}
    arch = compile(QWEN3_5.replace("{tied}", "true"))
    pretrained.load(Model(arch, rngs=nnx.Rngs(0)), arch, layers)


def test_a_trained_run_exports_as_the_released_checkpoint(tmp_path):
    from conftest import RUN

    from bux.export.runs import export_released
    from bux.lang import compile_runs
    from bux.train.trainer import train

    qwen3_5(True).save_pretrained(tmp_path / "released")
    settings = RUN | {
        "steps": "2",
        "weights": f'hf(name="{tmp_path / "released"}")',
        "checkpoint": "checkpoint(every=2)",
    }
    source = QWEN3_5.replace("{tied}", "true")
    [run] = compile_runs(source + "".join(f"\n{k} = {v}" for k, v in settings.items()))
    folder = tmp_path / "runs"
    train(run, folder, echo=lambda _: None)
    [done] = folder.iterdir()
    restored, left = export_released(done, tmp_path / "out")
    assert left == ["vision"]

    hf = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(tmp_path / "out")
    tokens = np.random.default_rng(0).integers(0, 128, (2, 24))
    with torch.no_grad():
        want = hf.eval()(input_ids=torch.tensor(tokens)).logits.numpy()
    got = bux_logits(restored.model, tokens, None)
    np.testing.assert_allclose(got, want, atol=2e-4, rtol=2e-4)
    key = "model.language_model.embed_tokens.weight"
    before = pretrained.Checkpoint(tmp_path / "released")[key]
    assert not np.allclose(pretrained.Checkpoint(tmp_path / "out")[key], before)


def test_a_run_that_resumes_reads_its_own_checkpoint_not_the_released_one(tmp_path):
    import shutil

    from conftest import RUN

    from bux.lang import compile_runs
    from bux.train.trainer import train

    qwen3_5(True).save_pretrained(tmp_path / "released")
    settings = RUN | {
        "weights": f'hf(name="{tmp_path / "released"}")',
        "checkpoint": "checkpoint(every=2)",
    }
    source = QWEN3_5.replace("{tied}", "true")
    text = source + "".join(f"\n{k} = {v}" for k, v in settings.items())
    [run] = compile_runs(text + "\nsteps = 2")
    train(run, tmp_path / "runs", echo=lambda _: None)
    shutil.rmtree(tmp_path / "released")  # gone: a resume must not read it
    said: list[str] = []
    train(run, tmp_path / "runs", echo=said.append)
    assert any("resuming at step 2" in line for line in said)
    assert not any("loaded" in line for line in said)


def test_a_layer_skipped_by_its_definition_is_new_and_the_rest_load(tmp_path):
    qwen3_5(False, depth=2).save_pretrained(tmp_path)
    source = (
        vision_language()
        .replace(
            "linear(features=d_model, bias=true))",
            "linear(features=d_model, bias=true) merge_patches(size=2) pool)",
        )
        .replace("model =", "pool = linear(features=d_model, bias=true)\nmodel =")
    )
    arch = compile(source)
    model, fresh = (Model(arch, rngs=nnx.Rngs(0)) for _ in range(2))
    pretrained.load(model, arch, pretrained.read(tmp_path), skip=["pool"])
    got, want = (weights.tensors(plan(arch), m) for m in (model, fresh))
    released = pretrained.Checkpoint(tmp_path)
    pooled = [k for k in got if ".pool." in k]
    assert pooled and all(np.array_equal(got[k], want[k]) for k in pooled)
    for key in ("linear_fc1", "linear_fc2"):  # the released merger, loaded
        merger = released[f"model.visual.merger.{key}.weight"]
        assert any(np.array_equal(v, merger) for v in got.values()), key
