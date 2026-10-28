"""Compare Qwen3.5's logits in bux and in transformers, on a page of text as an image.

python sky/vision_parity.py Qwen/Qwen3.5-0.8B examples/architectures/qwen/qwen3.5_0.8b.bux
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch
import transformers
from flax import nnx
from PIL import Image, ImageDraw

from bux import pretrained
from bux.architecture import Images, Model
from bux.export.hf import pixels
from bux.lang import compile_file

repo, path = sys.argv[1], sys.argv[2]
img = Image.new("RGB", (320, 160), "white")
d = ImageDraw.Draw(img)
d.text((10, 10), "Invoice 4821\nTotal: $1,337.00\nDue: 2026-11-30", fill="black")
x = np.asarray(img)
h, w = pixels.resized(
    *x.shape[:2], size=16, merge=2, min_pixels=65536, max_pixels=1 << 24
)
x = pixels.resize(x, h, w)
cut, grid = pixels.patches(x, size=16, merge=2)
gh, gw = h // 16, w // 16
n = gh * gw // 4
tok = transformers.AutoTokenizer.from_pretrained(repo)
assert tok is not None
pre = tok("<|im_start|>user\n", add_special_tokens=False).input_ids
post = tok(
    "Read the text in the image.<|im_end|>\n<|im_start|>assistant\n",
    add_special_tokens=False,
).input_ids
ids = np.array([pre + [248053] + [248056] * n + [248054] + post])
is_image = ids[0] == 248056
frames = np.repeat(((cut / 255 - 0.5) / 0.5).reshape(-1, 3, 1, 256), 2, 2).reshape(
    len(cut), -1
)
hf = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
    repo, dtype=torch.float32
).eval()
with torch.no_grad():
    want = hf(
        input_ids=torch.tensor(ids),
        pixel_values=torch.tensor(frames, dtype=torch.float32),
        image_grid_thw=torch.tensor([[1, gh, gw]]),
        mm_token_type_ids=torch.tensor(is_image[None].astype(np.int64)),
    ).logits.numpy()
arch = compile_file(path)
model = Model(arch, rngs=nnx.Rngs(0))
pretrained.load(model, arch, pretrained.read(repo))
rope = pixels.mrope(is_image, [(gh // 2, gw // 2)])
inp = Images(
    *(
        jnp.asarray(a[None])
        for a in (
            cut,
            np.arange(len(cut)),
            grid,
            np.tile([gh, gw], (len(cut), 1)),
            rope,
        )
    )
)
with jax.default_matmul_precision("highest"):
    got = np.asarray(model.outputs(jnp.asarray(ids), None, inp)[0])
print("tokens", ids.shape[1], "image tokens", n, "patches", len(cut))
print("max abs diff", np.abs(got - want).max(), "max |logit|", np.abs(want).max())
print("argmax agree", (got.argmax(-1) == want.argmax(-1)).mean())
print(
    "next token",
    repr(tok.decode(want[0, -1].argmax())),
    repr(tok.decode(got[0, -1].argmax())),
)
