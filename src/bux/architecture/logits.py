"""A head's logits, made a chunk of positions at a time.

At a vocabulary of 128K, the float32 logits of 8 rows of 4096 tokens take 16 GB, and
their gradient as much again. So a head leaves the objective its input and its
projection (``HeadOutput``), and ``HeadOutput.scores`` takes them through the matmul,
the logsumexp and the target's logit a chunk of positions at a time, as Liger's and
Apple's Cut Cross-Entropy do: the backward makes each chunk's logits again.

Each device takes its own positions in chunks of at most `CHUNK_BYTES` of logits,
which split over `vocab` as whole logits would, and adds up its share of the head's
gradient in float32, which the mesh reduces once, after the last chunk.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Shaped

from bux import parallel

CHUNK_BYTES = 2**31  # of float32 logits that a device makes at once


class Projection(NamedTuple):
    """A head's map from the stream to logits, softcap(x W + b): float32, from a
    matmul in the stream's dtype. A tied head's W is the embedding's table,
    transposed. The softcap is an array, so that jax's transforms pass it through."""

    kernel: Float[Array, "d vocab"]
    bias: Float[Array, " vocab"] | None
    softcap: Float[Array, ""] | None

    def __call__(self, x: Float[Array, "*n d"]) -> Float[Array, "*n vocab"]:
        bias = None if self.bias is None else self.bias.astype(jnp.float32)
        return self.finish(self.matmul(x), bias)

    def matmul(self, x: Float[Array, "*n d"]) -> Float[Array, "*n vocab"]:
        """Return x W, accumulated and given in float32."""
        return jnp.einsum(
            "...d,dv->...v", x, self.kernel, preferred_element_type=jnp.float32
        )

    def finish(
        self, raw: Float[Array, "*n vocab"], bias: Float[Array, " vocab"] | None
    ) -> Float[Array, "*n vocab"]:
        """Return the logits of x W: plus the bias, in float32, softcapped. The
        backward of ``scores`` differentiates it by jax, so another needs none."""
        logits = raw if bias is None else raw + bias
        if self.softcap is None:
            return logits
        return self.softcap * jnp.tanh(logits / self.softcap)


class Scores(NamedTuple):
    """What the cross-entropy and the z-loss take of the logits at each position:
    log Z, the log of the softmax's normalizer, and the target's logit."""

    lse: Float[Array, "b t"]
    target: Float[Array, "b t"]

    @classmethod
    def of(
        cls, logits: Float[Array, "b t vocab"], targets: Int[Array, "b t"]
    ) -> "Scores":
        """Return the scores of logits given whole."""
        logits = logits.astype(jnp.float32)
        return cls(jax.nn.logsumexp(logits, -1), _pick(logits, targets))


class HeadOutput(NamedTuple):
    """What a head leaves the objective: the stream it reads, and its projection."""

    hidden: Float[Array, "b t d"]
    projection: Projection

    def logits(self) -> Float[Array, "b t vocab"]:
        """Return the logits, all at once, in float32."""
        return parallel.shard(self.projection(self.hidden), "batch", "seq", "vocab")

    def scores(self, targets: Int[Array, "b t"]) -> Scores:
        """Return the scores of `targets` at each position, a chunk at a time."""
        return Scores(*_scores(self.projection, self.hidden, targets))


def _pick(
    logits: Float[Array, "*n vocab"], targets: Int[Array, "*n"]
) -> Float[Array, "*n"]:
    """Return each target's logit: a sum over the vocabulary, as it splits."""
    return jnp.sum(jnp.where(_is_target(logits, targets), logits, 0), -1)


def _is_target(
    logits: Float[Array, "*n vocab"], targets: Int[Array, "*n"]
) -> Bool[Array, "*n vocab"]:
    return jnp.arange(logits.shape[-1]) == targets[..., None]


class Chunks(NamedTuple):
    """How a (b, t) of positions splits: `rows` and `seq` ways over the mesh, as
    `batch` and `seq` split them, then each device's into `count` chunks."""

    rows: int
    seq: int
    count: int

    @classmethod
    def of(cls, b: int, t: int, vocab: int) -> "Chunks":
        """Return the fewest chunks of at most `CHUNK_BYTES` of logits each."""
        rows, seq = parallel.splits(("batch", "seq"), (b, t))
        local = b // rows * (t // seq)
        least = max(1, math.ceil(local * vocab * 4 / CHUNK_BYTES))
        count = next(n for n in range(least, local + 1) if local % n == 0)
        return cls(rows, seq, count)

    def split(self, a: Shaped[Array, "b t *r"]) -> Shaped[Array, "n B C c *r"]:
        """Return the chunks of `a`, each on every device: its rows and positions
        split `rows` by `seq` ways, then each device's c of them."""
        b, t, *r = a.shape
        a = a.reshape(self.rows, b // self.rows, self.seq, t // self.seq, *r)
        a = a.swapaxes(1, 2).reshape(self.rows, self.seq, self.count, -1, *r)
        return _shard(jnp.moveaxis(a, 2, 0), None, *(None,) * len(r))

    def join(
        self, a: Shaped[Array, "n B C c *r"], b: int, t: int
    ) -> Shaped[Array, "b t *r"]:
        """Undo `split`."""
        r = a.shape[4:]
        a = jnp.moveaxis(a, 0, 2).reshape(self.rows, self.seq, b // self.rows, -1, *r)
        return a.swapaxes(1, 2).reshape(b, t, *r)


def _shard(a: Shaped[Array, "*n"], *names: parallel.Name) -> Shaped[Array, "*n"]:
    """Split a chunked array over `batch` and `seq` by its shards' dimensions."""
    lead = a.ndim - 2 - len(names)
    return parallel.shard(a, *(None,) * lead, "batch", "seq", *names)


def _gathered(p: Projection) -> Projection:
    """Return `p` split as its logits are, over `vocab`: gathered over `fsdp` once,
    not for every chunk."""
    bias = None if p.bias is None else parallel.shard(p.bias, "vocab")
    return p._replace(kernel=parallel.shard(p.kernel, "embed", "vocab"), bias=bias)


@jax.custom_vjp
def _scores(
    p: Projection, x: Float[Array, "b t d"], targets: Int[Array, "b t"]
) -> tuple[Float[Array, "b t"], Float[Array, "b t"]]:
    """Return log Z and the target's logit at each position, a chunk at a time."""
    return _scores_fwd(p, x, targets)[0]


def _scores_fwd(
    p: Projection, x: Float[Array, "b t d"], targets: Int[Array, "b t"]
) -> tuple[tuple[Float[Array, "b t"], Float[Array, "b t"]], tuple]:
    b, t, _ = x.shape
    chunks, whole = Chunks.of(b, t, p.kernel.shape[1]), _gathered(p)

    def step(
        _: None, xs: tuple[Float[Array, "B C c d"], Int[Array, "B C c"]]
    ) -> tuple[None, tuple[Float[Array, "B C c"], Float[Array, "B C c"]]]:
        x, targets = xs
        logits = _shard(whole(x), None, "vocab")
        return None, (jax.nn.logsumexp(logits, -1), _pick(logits, targets))

    xs = (chunks.split(x), chunks.split(targets))
    _, (lse, target) = jax.lax.scan(step, None, xs)
    lse, target = chunks.join(lse, b, t), chunks.join(target, b, t)
    return (lse, target), (p, x, targets, lse)


def _scores_bwd(
    saved: tuple, cotangents: tuple[Float[Array, "b t"], Float[Array, "b t"]]
) -> tuple[Projection, Float[Array, "b t d"], None]:
    """Return the gradients of `_scores`, a chunk at a time (``_chunk_grads``)."""
    p, x, targets, lse = saved
    b, t, d = x.shape
    chunks, whole = Chunks.of(b, t, p.kernel.shape[1]), _gathered(p)
    bias = None if whole.bias is None else whole.bias.astype(jnp.float32)

    def step(sums: tuple, xs: tuple) -> tuple[tuple, Float[Array, "B C c d"]]:
        d_kernel, d_bias, dx = _chunk_grads(whole, bias, *xs)
        return jax.tree.map(jnp.add, sums, (d_kernel, d_bias)), dx

    vocab = p.kernel.shape[1]
    zeros = _shard(jnp.zeros((chunks.rows, chunks.seq, d, vocab)), None, "vocab")
    start = (zeros, None if bias is None else jnp.zeros_like(bias))
    xs = tuple(chunks.split(a) for a in (x, targets, lse, *cotangents))
    (d_kernel, d_bias), dx = jax.lax.scan(step, start, xs)
    d_kernel = d_kernel.sum((0, 1)).astype(p.kernel.dtype)
    d_bias = jax.tree.map(lambda g, b: g.astype(b.dtype), d_bias, p.bias)
    return Projection(d_kernel, d_bias, None), chunks.join(dx, b, t), None


def _chunk_grads(
    p: Projection,
    bias: Float[Array, " vocab"] | None,
    x: Float[Array, "B C c d"],
    targets: Int[Array, "B C c"],
    lse: Float[Array, "B C c"],
    d_lse: Float[Array, "B C c"],
    d_target: Float[Array, "B C c"],
) -> tuple[
    Float[Array, "B C d vocab"], Float[Array, " vocab"] | None, Float[Array, "B C c d"]
]:
    """Return a chunk's gradients of the kernel, each device's share apart, of the
    bias and of x, from its logits made again."""
    raw = _shard(p.matmul(x), None, "vocab")
    logits, finish = jax.vjp(p.finish, raw, bias)
    probs = jnp.exp(logits - lse[..., None])
    hit = _is_target(logits, targets)
    d_raw, d_bias = finish(d_lse[..., None] * probs + d_target[..., None] * hit)
    d_raw = d_raw.astype(x.dtype)
    f32 = jnp.float32
    d_kernel = jnp.einsum("BCcd,BCcv->BCdv", x, d_raw, preferred_element_type=f32)
    dx = jnp.einsum("BCcv,dv->BCcd", d_raw, p.kernel, preferred_element_type=f32)
    return d_kernel, d_bias, dx.astype(x.dtype)


_scores.defvjp(_scores_fwd, _scores_bwd)
