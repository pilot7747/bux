"""``image(encoder, token=...)``: images in the stream.

The encoder is a graph like any other, but it reads the forward pass's images
(`~bux.architecture.registry.Images`), not the stream: their patches, with the
documents of its attention the images and its RoPE positions their rows and columns.
It returns a vector per block of patches, ``d_model`` wide. Each token of the row whose
id is `token` then takes the next of them, in order, in place of its embedding, as
Qwen2-VL writes its images into the text.
"""

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux.architecture.common.segments import Segments
from bux.architecture.registry import Taps
from bux.parallel import shard_stream


class ImageSplice(nnx.Module):
    """The stream, with each image token replaced by the next output of the encoder
    on the row's images; the stream as it is without images.

    Args:
        body: The encoder: its first layer reads the images' patches.
        token: The id of the tokens an image is written with.
    """

    def __init__(self, body: nnx.Module, token: int):
        self.body = body
        self.token = token

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        """Return x with its image tokens replaced by the encoder's outputs."""
        images = taps.images
        if images is None:
            return x
        assert taps.tokens is not None
        docs = Segments.of(images.positions)._replace(axes=images.grid)
        inner = Taps(segments=docs, training=taps.training, images=images)
        y = self.body(images.pixels, inner)
        taps.losses += inner.losses
        is_image = taps.tokens == self.token
        k = jnp.maximum(jnp.cumsum(is_image, 1) - 1, 0)  # the image token's index
        out = jnp.take_along_axis(y, k[..., None], 1).astype(x.dtype)
        return shard_stream(jnp.where(is_image[..., None], out, x))
