"""Vision encoders and images in the stream: multi-axis RoPE, the patch layers, and
image(...), each against what transformers' Qwen3.5 computes."""

import jax.numpy as jnp
import numpy as np
import pytest

from bux import config as cfg
from bux.architecture.attention.rope import rotate

torch = pytest.importorskip("torch")
qwen = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")


def as_torch(x) -> "torch.Tensor":
    return torch.tensor(np.asarray(x))


def test_axial_rope_is_qwen3_5s_vision_rope():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    config = Qwen3_5VisionConfig(hidden_size=72, num_heads=1)
    hf = qwen.Qwen3_5VisionRotaryEmbedding(config)
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, 40, (2, 12))
    x = rng.standard_normal((1, 12, 1, 72)).astype(np.float32)

    cos, sin = hf(as_torch(x), torch.tensor(np.stack([rows, cols], -1)))
    q = as_torch(x)[0]
    want, _ = qwen.apply_rotary_pos_emb_vision(q, q, cos, sin)
    rope = cfg.RoPEConfig(base=10000, axial=2)
    got = rotate(jnp.asarray(x), rope, jnp.asarray(np.stack([rows, cols], -1))[None])
    np.testing.assert_allclose(np.asarray(got[0]), want.numpy(), atol=1e-5)


@pytest.mark.parametrize("layout", ["interleaved", "blocks"])
def test_mrope_is_qwen3_5s_text_rope(layout):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    sections = [11, 11, 10]
    config = Qwen3_5TextConfig(
        head_dim=256,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": sections,
            "mrope_interleaved": True,
        },
    )
    hf = qwen.Qwen3_5TextRotaryEmbedding(config)
    rng = np.random.default_rng(0)
    pos = rng.integers(0, 500, (3, 1, 10))  # time, height, width
    x = rng.standard_normal((1, 10, 1, 256)).astype(np.float32)
    rope = cfg.RoPEConfig(
        base=10000000,
        fraction=0.25,
        sections=tuple(sections),
        sections_layout=layout,
    )
    got = rotate(jnp.asarray(x), rope, jnp.asarray(pos.transpose(1, 2, 0)))
    if layout == "blocks":  # Qwen2-VL's runs: a different rotation
        assert not np.allclose(np.asarray(got), x, atol=1e-3)
        return
    cos, sin = hf(as_torch(x), torch.tensor(pos))
    q = as_torch(x).transpose(1, 2)
    want, _ = qwen.apply_rotary_pos_emb(q, q, cos, sin)
    np.testing.assert_allclose(np.asarray(got), want.transpose(1, 2).numpy(), atol=1e-4)


def test_positions_of_one_axis_turn_every_axis_alike():
    rope = cfg.RoPEConfig(sections=(2, 1, 1), sections_layout="interleaved")
    plain = cfg.RoPEConfig()
    x = jnp.asarray(np.random.default_rng(0).standard_normal((2, 6, 1, 8)), jnp.float32)
    pos = jnp.broadcast_to(jnp.arange(6), (2, 6))
    np.testing.assert_allclose(rotate(x, rope, pos), rotate(x, plain, pos), atol=1e-6)
    same = jnp.repeat(pos[..., None], 3, -1)
    np.testing.assert_allclose(rotate(x, rope, same), rotate(x, plain, pos), atol=1e-6)


def test_sections_that_miss_the_pairs_say_so():
    rope = cfg.RoPEConfig(sections=(2, 2, 2))
    x = jnp.zeros((1, 2, 1, 8))
    with pytest.raises(ValueError, match="deals out 6 pairs, but 4 rotate"):
        rotate(x, rope, jnp.zeros((1, 2, 3), jnp.int32))


def test_bidirectional_attention_sees_its_whole_document_and_no_other():
    from flax import nnx

    from bux.architecture import Segments, Taps
    from bux.architecture.attention.gqa import Attention

    config = cfg.GQAConfig(num_heads=2, num_kv_heads=2, causal=False)
    attn = Attention(config, 16, (), nnx.Rngs(0))
    x = jnp.asarray(np.random.default_rng(0).standard_normal((1, 6, 16)), jnp.float32)
    docs = Segments.of(jnp.asarray([[0, 1, 2, 0, 1, 2]]))
    y = attn(x, Taps(segments=docs))
    later = x.at[:, 2].add(1.0)  # the first document's last token
    z = attn(later, Taps(segments=docs))
    assert not np.allclose(y[:, 0], z[:, 0])  # its first token sees it
    np.testing.assert_allclose(y[:, 3:], z[:, 3:], atol=1e-6)  # the next does not


def test_causal_false_takes_no_window():
    with pytest.raises(ValueError, match="takes no window"):
        cfg.GQAConfig(num_heads=2, num_kv_heads=2, causal=False, sliding_window_size=4)


TINY = """
d_model = 32
vocab_size = 64
vision = patches(size=4, features=16) linear(features=d_model)
"""


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ("embed image(vision) head", "image needs an encoder and token=..."),
        ("embed image(vision, token=-1) head", "token is a token id"),
        (
            "embed image(rmsnorm, token=3) head",
            "rmsnorm needs hidden states, but receives pixels",
        ),
        (
            "embed image(patches(size=4, features=16), token=3) head",
            "an encoder ends d_model = 32 wide",
        ),
        ("patches(size=4, features=16) head", "patches embeds an image's pixels"),
        (
            "embed linear(features=8) image(vision, token=3) head",
            "the stream here is 8",
        ),
    ],
)
def test_image_errors(model, message):
    from bux.lang import compile
    from bux.lang.syntax import BuxError

    with pytest.raises(BuxError, match=message):
        compile(TINY + f"model = {model}\n")


def png(image) -> bytes:
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(image).save(out, "PNG")
    return out.getvalue()


def test_a_rows_images_come_from_the_store_in_order_with_their_positions(tmp_path):
    from bux.data import images

    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    part = tmp_path / "map-0"
    part.mkdir()
    images.write_part(
        part, [(7, png(a), (4, 6)), (9, png(b), (2, 2)), (7, png(a), (4, 6))]
    )
    images.write_store(tmp_path, [part])
    store = images.Store(tmp_path)
    assert store.keys.tolist() == [7, 9]

    # Two documents: text, image a, text; then image b and text.
    keys = np.array([0, 7, 7, 7, 7, 7, 7, 0, 9, 0], np.int64)
    positions = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1], np.int32)
    pixels, index, grid, sizes, rope = images.row(
        keys, positions, store, size=16, merge=2, patches_per_row=32
    )
    want_a, grid_a = images.patches(a, size=16, merge=2)
    want_b, grid_b = images.patches(b, size=16, merge=2)
    np.testing.assert_array_equal(pixels[:28], np.concatenate([want_a, want_b]))
    assert not pixels[28:].any() and index.tolist() == [
        *range(24),
        *range(4),
        *range(4),
    ]
    np.testing.assert_array_equal(grid[:28], np.concatenate([grid_a, grid_b]))
    assert sizes[0].tolist() == [4, 6] and sizes[24].tolist() == [2, 2]
    first = images.mrope(keys[:8] != 0, [(2, 3)])
    second = images.mrope(keys[8:] != 0, [(1, 1)])
    np.testing.assert_array_equal(rope, np.concatenate([first, second]))


def test_a_row_with_more_patches_than_room_says_so(tmp_path):
    from bux.data import images

    image = np.zeros((32, 32, 3), np.uint8)
    part = tmp_path / "map-0"
    part.mkdir()
    images.write_part(part, [(5, png(image), (2, 2))])
    images.write_store(tmp_path, [part])
    keys, positions = np.array([5], np.int64), np.array([0], np.int32)
    with pytest.raises(ValueError, match="have 4 patches, but it has room for 2"):
        images.row(
            keys, positions, images.Store(tmp_path), size=16, merge=2, patches_per_row=2
        )


#: The byte tokenizer's special tokens, then those that write an image.
IMAGE_SPECIALS = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]


@pytest.fixture
def image_tokenizer(tmp_path) -> str:
    """The tests' byte-level tokenizer, with Qwen's image tokens after its own."""
    import json

    from conftest import CHATML, SPECIALS, byte_spec

    spec = byte_spec()
    first = 256 + len(SPECIALS)
    spec["added_tokens"] += [
        spec["added_tokens"][0] | {"id": first + i, "content": s}
        for i, s in enumerate(IMAGE_SPECIALS)
    ]
    every = [*SPECIALS, *IMAGE_SPECIALS]
    added = {256 + i: {"content": s, "special": True} for i, s in enumerate(every)}
    folder = tmp_path / "tokenizer"
    folder.mkdir()
    (folder / "tokenizer.json").write_text(json.dumps(spec))
    config = {"chat_template": CHATML, "added_tokens_decoder": added}
    (folder / "tokenizer_config.json").write_text(json.dumps(config))
    return str(folder / "tokenizer.json")


def conversations(folder, n: int = 12) -> str:
    """Write n conversations, each about an image of its own, to a Parquet file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        h, w = rng.integers(16, 80, 2)
        image = png(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        user = [
            {"type": "image", "image": image, "text": None},
            {"type": "text", "image": None, "text": "Read this."},
        ]
        reply = [{"type": "text", "image": None, "text": f"page {i}"}]
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": reply},
                ]
            }
        )
    path = folder / "pages.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return str(path)


TINY_VL = """
d_model    = 32
vocab_size = 264

vattn  = gqa(num_heads=2, num_kv_heads=2, causal=false, rope=rope(axial=2))
vision = (patches(size=16, features=16, grid=4) [layernorm vattn]
          merge_patches(size=2) linear(features=d_model))
attn   = gqa(num_heads=2, num_kv_heads=2, rope=rope(sections=[4, 2, 2],
             sections_layout=interleaved))
model  = embed image(vision, token=260) [rmsnorm attn] rmsnorm head
"""


def image_run(tmp_path, tokenizer: str, steps: int = 3):
    from bux.lang import compile_runs

    data = conversations(tmp_path)
    settings = f"""
tokenizer = tokenizer(name="{tokenizer}")
data      = chat(path="{data}", images=images(min_pixels=1024, max_pixels=8192))
patches_per_row = 64
seq_len   = 64
batch     = 8
steps     = {steps}
optimizer = adamw(lr=1e-2)
log       = log(every=1)
precision = precision(compute=float32)
"""
    [run] = compile_runs(TINY_VL + settings)
    return run


def test_a_run_trains_on_conversations_about_images(tmp_path, image_tokenizer):
    from conftest import Losses

    from bux.train.data import training_batches
    from bux.train.trainer import train

    run = image_run(tmp_path, image_tokenizer)
    sink = Losses()
    train(
        run,
        tmp_path / "runs",
        cache=tmp_path / "cache",
        echo=lambda _: None,
        sinks=[sink],
    )
    assert len(sink.losses) == 3 and all(np.isfinite(sink.losses))

    batch = next(training_batches(run.train, 264, cache=tmp_path / "cache"))
    assert batch.images is not None and batch.images.pixels.shape == (8, 64, 768)
    tokens = (batch.input_ids == 260).sum(1)  # each stands for a block of 4 patches
    real = (batch.images.sizes.prod(-1) > 1).sum(1)
    np.testing.assert_array_equal(real, 4 * tokens)
    assert (tokens > 0).all()


def test_a_run_whose_data_has_images_needs_room_and_an_encoder(
    tmp_path, image_tokenizer
):
    from bux.lang import compile_runs
    from bux.lang.syntax import BuxError

    data = conversations(tmp_path)
    base = f"""
tokenizer = tokenizer(name="{image_tokenizer}")
data = chat(path="{data}", images=images())
seq_len = 64
batch = 4
steps = 1
optimizer = adamw(lr=1e-2)
"""
    with pytest.raises(BuxError, match="needs room for their patches"):
        compile_runs(TINY_VL + base)
    with pytest.raises(BuxError, match="the model reads none"):
        compile_runs(
            TINY_VL.replace("image(vision, token=260) ", "")
            + base
            + "patches_per_row = 64\n"
        )
    with pytest.raises(BuxError, match="cut size = 14"):
        compile_runs(
            TINY_VL
            + base.replace("images()", "images(size=14)")
            + "patches_per_row = 64\n"
        )
    unmerged = TINY_VL.replace("merge_patches(size=2) ", "")
    with pytest.raises(
        BuxError, match="cut merge = 2, but the model's encoder merge = 1"
    ):
        compile_runs(unmerged + base + "patches_per_row = 64\n")
    with pytest.raises(BuxError, match="is not whole blocks of 2 x 2"):
        compile_runs(TINY_VL + base + "patches_per_row = 63\n")
    text = base.replace(", images=images()", "")
    with pytest.raises(BuxError, match="but the data has none"):
        compile_runs(TINY_VL + text + "patches_per_row = 64\n")
    compile_runs(TINY_VL + text)  # a model of images reads text alone


def test_causal_false_takes_no_indexer():
    indexer = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=4)
    with pytest.raises(ValueError, match="takes no indexer"):
        cfg.GQAConfig(num_heads=2, num_kv_heads=2, causal=False, indexer=indexer)


def test_frozen_definitions_take_no_update():
    import jax
    from flax import nnx

    from bux.architecture import Model
    from bux.lang import compile
    from bux.train import config as tcfg
    from bux.train.optim import frozen_paths, optimizer

    arch = compile(TINY_VL)
    model = Model(arch, rngs=nnx.Rngs(0))
    frozen = frozen_paths(model, arch, ["vision"])
    assert frozen and all(p[:3] == ("graph", "items", 1) for p in frozen)
    params = nnx.state(model, nnx.Param)
    tx = optimizer(
        tcfg.AdamWConfig(lr=1e-2), lambda _: jnp.ones(()), model, 1.0, frozen=frozen
    )
    state = tx.init(params)
    grads = jax.tree.map(jnp.ones_like, params)
    updates, _ = tx.update(grads, state, params)
    moved = {
        jax.tree_util.keystr(path): bool(jnp.any(u != 0))
        for path, u in jax.tree_util.tree_leaves_with_path(updates)
    }
    assert moved and all(
        still != name.startswith("['graph']['items'][1]")
        for name, still in moved.items()
    ), moved


def test_two_merges_of_two_merge_the_blocks_of_four():
    from bux.export.hf import pixels

    image = np.zeros((128, 192, 3), np.uint8)
    _, grid = pixels.patches(image, size=16, merge=4)
    once = grid.reshape(-1, 16, 2)  # merge_patches(size=4)'s blocks
    twice = grid.reshape(-1, 4, 4, 2)  # (size=2), then (size=2) again
    for block in once:
        assert np.ptp(block[:, 0]) == 3 and np.ptp(block[:, 1]) == 3
        assert block[0, 0] % 4 == 0 and block[0, 1] % 4 == 0
    for inner in twice.reshape(-1, 4, 2):
        assert np.ptp(inner[:, 0]) == 1 and np.ptp(inner[:, 1]) == 1
    _, two = pixels.patches(np.zeros((64, 64, 3), np.uint8), size=16, merge=2)
    assert two[:4].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]  # as Qwen's


def test_merge_patches_by_mean_averages_a_block():
    from bux.architecture import Taps
    from bux.architecture.layers.vision import MergePatches

    x = jnp.arange(2 * 8 * 3, dtype=jnp.float32).reshape(2, 8, 3)
    y = MergePatches(cfg.MergePatchesConfig(size=2, pool="mean"))(x, Taps())
    np.testing.assert_allclose(y, x.reshape(2, 2, 4, 3).mean(2))
