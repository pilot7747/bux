"""Models that read images export to transformers and compute what bux does
(bux.export.hf.vision): their logits on a row of text and images, and their greedy
generation from it, with a cache."""

import importlib

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from test_pretrained import IMAGE, bux_logits, row_with_images, vision_language

from bux.architecture import Images, Model
from bux.export import huggingface
from bux.export.hf import pixels
from bux.lang import compile

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """A shrunk Qwen3.5 of random weights, with its vision encoder, in bux and as
    exported to transformers, and the export's vision module."""
    arch = compile(vision_language())
    model = Model(arch, rngs=nnx.Rngs(0))
    out = tmp_path_factory.mktemp("export")
    images = {"size": 16, "merge": 2, "min_pixels": 1024, "max_pixels": 8192}
    huggingface.save(arch, model, out, images=images)
    hf = transformers.AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True)
    vision = importlib.import_module(type(hf).__module__.rsplit(".", 1)[0] + ".vision")
    return model, hf.eval(), vision


def torch_images(vision, ours: Images):
    return vision.Images(*(torch.tensor(np.array(x)) for x in ours))


def test_the_export_computes_bux_logits_with_images(exported):
    model, hf, vision = exported
    tokens, _, ours = row_with_images()
    with torch.no_grad():
        want = hf(input_ids=torch.tensor(tokens), images=torch_images(vision, ours))
    np.testing.assert_allclose(
        bux_logits(model, tokens, ours), want.logits.numpy(), atol=1e-4, rtol=1e-4
    )


def test_the_export_generates_from_images_as_bux_does(exported):
    model, hf, vision = exported
    tokens, _, ours = row_with_images()
    n, new = tokens.shape[1], 6
    with torch.no_grad():
        out = hf.generate(
            input_ids=torch.tensor(tokens),
            images=torch_images(vision, ours),
            max_new_tokens=new,
            do_sample=False,
        )
    # bux, the whole row each step: a token after the prompt is at its index plus
    # the prompt's offset on every axis.
    rope = np.asarray(ours.rope[0])
    offset = rope.max() + 1 - n
    row = list(tokens[0])
    for _ in range(new):
        later = np.arange(n, len(row))[:, None] + offset
        at = np.concatenate([rope, np.repeat(later, 3, 1)])[None]
        step = ours._replace(rope=jnp.asarray(at))
        row.append(int(bux_logits(model, np.array([row]), step)[0, -1].argmax()))
    assert out[0].tolist() == row


def test_inputs_cut_images_as_bux_data_does():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (70, 100, 3), dtype=np.uint8)
    cut = {"size": 16, "merge": 2, "min_pixels": 1024, "max_pixels": 8192}
    vision = importlib.import_module("bux.export.hf.vision")
    ids, got = vision.inputs([7, IMAGE, 8], [a], IMAGE, cut)
    h, w = pixels.resized(70, 100, size=16, merge=2, min_pixels=1024, max_pixels=8192)
    want, grid = pixels.patches(pixels.resize(a, h, w), size=16, merge=2)
    blocks = h // 32 * (w // 32)
    assert ids[0].tolist() == [7, *[IMAGE] * blocks, 8]
    np.testing.assert_array_equal(got.pixels[0].numpy(), want)
    np.testing.assert_array_equal(got.grid[0].numpy(), grid)
    is_image = ids[0].numpy() == IMAGE
    rope = pixels.mrope(is_image, [(h // 32, w // 32)])
    np.testing.assert_array_equal(got.rope[0].numpy(), rope)
