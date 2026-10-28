# Vision encoders

A model can read images as well as text. The images go through a vision encoder,
which is a graph like any other, and `image(...)` writes the encoder's outputs into the
text stream at the image tokens. Use it to fine-tune a vision-language model, such as
Qwen3.5 for OCR, or to ablate its encoder: its depth, its RoPE, how many patches make
a token.

```bux
vnorm  = layernorm(bias=true)
vattn  = gqa(num_heads=12, num_kv_heads=12, bias=true, causal=false, rope=rope(axial=2))
vffn   = mlp(intermediate_size=3072, activation=gelu_tanh, bias=true)
vision = (patches(size=16, features=768, grid=48) ([vnorm vattn] [vnorm vffn])^12
          vnorm merge_patches(size=2) linear(features=3072, bias=true) gelu
          linear(features=d_model, bias=true))

model = embed image(vision, token=248056) "DDDA"^6 norm head(tied=true)
```

This is Qwen3.5-0.8B's encoder (`examples/architectures/qwen/qwen3.5_0.8b.bux`). It
cuts each image into patches of 16 x 16 pixels, embeds them 768 wide, and runs 12
pre-norm blocks over them. It then joins each 2 x 2 block of patches into one vector
and projects it to `d_model`. `bux render` draws the encoder inside its `image` frame:

```console
$ bux render examples/architectures/qwen/qwen3.5_0.8b.bux
model
         ╭─ image ──╮  ╭─ ×6 ───────────────╮
──embed──┤─┤vision├─├──┤─┤D├──┤D├──┤D├──┤A├─├──norm──@h──out──┤mtp├──
         ╰──────────╯  ╰────────────────────╯
```

## `image(...)`

`image(encoder, token=id)` passes the stream on, with each token whose id is `token`
replaced by the encoder's next output. The encoder reads the row's images, not the
stream, and returns one vector per image token, in order. So a row that holds an image
of 4 x 6 blocks has 24 image tokens where the image goes, and those tokens take the
encoder's 24 outputs. A row without images passes through unchanged.

The compiler checks the shapes:

- `image(...)` goes where the stream is `d_model` wide, as right after `embed`;
- the encoder starts with `patches` and ends `d_model` wide; and
- `patches` starts an encoder and goes nowhere else.

`image(...)` does not run under a residual algebra (`residual = hc(...)` and the
others).

## The encoder's layers

| Layer | What it does |
|-------|--------------|
| {bux}`patches` | each patch's pixels, normalized per channel, projected to `features` channels, plus a learned position of its row and column |
| {bux}`merge_patches` | each `size` x `size` block of patches joined into one vector, `size * size` times as wide; or, with `pool=mean`, their mean, as wide as one |

`patches(grid=48)` learns a 48 x 48 table of positions and resizes it, bilinearly, to
each image's grid of patches, corners on corners. `mean` and `std` normalize each
channel of the pixels, which come in from 0 to 255 and are divided by 255 first. Their
default, 0.5 and 0.5, is Qwen's, and maps pixels to [-1, 1].

A block's patches come in Z-order: each 2 x 2 of them in a row, then the next. So
`merge_patches(size=2)` twice merges the blocks of 4 x 4 that `merge_patches(size=4)`
does, and the merges can sit anywhere in the encoder. The data's `images(merge=...)`
is their product. This encoder keeps Qwen3.5's merger and joins each 2 x 2 of its
tokens again, by their mean, for a quarter of the image tokens
(`sky/ocr_merge.bux`):

```bux
vision = (patches(size=16, features=768, grid=48) ([vnorm vattn] [vnorm vffn])^12
          vnorm merge_patches(size=2) linear(features=3072, bias=true) gelu
          linear(features=d_model, bias=true) merge_patches(size=2, pool=mean))
cut    = images(merge=4)
```

The rest of the encoder is the layers you know:

- **Attention to the whole image.** `gqa(causal=false)` lets every patch see every
  other patch of its image, ahead of it too, and none of another image. It takes no
  window. It runs by tokamax's flash kernel on a GPU, else by XLA, whatever
  `methods.attention` says, and not under context parallelism.
- **Axial RoPE.** `rope(axial=2)` splits the rotated pairs into two runs. The first
  turns by the patch's row and the second by its column, each as a RoPE of its own
  over its half of the dims.

## M-RoPE in the text

A vision-language model gives each token of the text three positions: time, height
and width. A token of text has one position, the same on all three. An image's tokens
are all at the image's start in time, and at their row and column past it in height
and width. The token after an image is one past the image's longer side.
`rope(sections=...)` says how many of the rotated pairs turn by each axis:

```bux
attn = gqa(num_heads=8, num_kv_heads=2, head_dim=256, gated=true, qk_norm=norm,
           rope=rope(base=10000000, fraction=0.25, sections=[11, 11, 10],
                     sections_layout=interleaved))
```

`sections_layout=blocks` gives each axis a run of pairs, as Qwen2-VL does.
`interleaved` gives pair i to axis i mod 3 while that axis has pairs left, as Qwen3-VL
and Qwen3.5 do. On rows of text alone, every axis is the token's position, so M-RoPE
turns as a RoPE of one axis does.

## Images in the forward pass

{py:class}`bux.architecture.Model` takes the images beside the tokens, as an
{py:class}`~bux.architecture.registry.Images`:

```python
logits = model.outputs(tokens, positions, images)[0]
```

`Images` holds the patches of each row's images, one image after another, then
padding up to a fixed number of patches per row:

| Field | Shape | What it is |
|-------|-------|------------|
| `pixels` | `b n p`, bytes | each patch's pixels, channels first |
| `positions` | `b n` | each patch's index in its image; an image starts where they stop increasing |
| `grid` | `b n 2` | each patch's row and column |
| `sizes` | `b n 2` | the rows and columns of the patch's image |
| `rope` | `b t 3` | each token's M-RoPE position |

A run reads them from chat data with images ([Images](../data/sources.md#images)).
In Python, {py:mod}`bux.data.images` makes them from an image: `resized` picks the size, a
multiple of `size * merge` on each side, `patches` cuts the image in the order
`merge_patches` reads it, and `mrope` gives a document's M-RoPE positions. They follow
transformers' `Qwen2VLImageProcessor` and its `get_rope_index`.

## Exporting

A model with `image(...)` exports to transformers with its encoder, and generates
from images there ([Models that read images](../export/index.md#models-that-read-images)).

## Checked against transformers

`tests/test_pretrained.py` loads a shrunk Qwen3.5 with random weights into bux and
compares the logits with transformers' on a row of text and two images. On the
released Qwen3.5-0.8B, an image of 320 x 160 pixels of text in a row of 89 tokens
gives logits within 3.2e-3 of transformers 5.20.2's, whose largest is 38.0, and the
same next-token prediction at every position (float32, Apple M5 Pro CPU, October
2026).
