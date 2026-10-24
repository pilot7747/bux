"""A head's output, and the two ways the loss takes its logits: `methods.loss`.

A head leaves the objective its input and its projection (``HeadOutput``) rather
than its logits, and the loss takes what it needs of them (``HeadOutput.sums``): the
sums over positions of the cross-entropy, log Z minus the target's logit, and of log
Z squared, each position weighted, by the run's method:

    chunks   the matmul, the logsumexp and the target's logit a chunk of positions at
             a time, in float32, as Liger's and Apple's Cut Cross-Entropy do, and the
             chunk's gradients with them, so that the backward only scales those: the
             default
    logits   one matmul for all positions, the logits whole in the stream's dtype

At a vocabulary of 128K, the float32 logits of 8 rows of 4096 tokens take 16 GB, and
their gradient as much again; by chunks, each device takes its own positions in
chunks of at most `CHUNK_BYTES` of logits, which split over `vocab` as whole logits
would, and adds up its share of the head's gradient in float32, which the mesh
reduces once, after the last chunk. A sum is linear in the positions' terms, so its
gradient is known before the backward, but for one factor, the sum's cotangent:
each sum keeps its own gradient, and `z` costs a second one. Another method is a
function of the head's output, the targets, their weights and whether to sum the
z-loss, that returns their `Sums`:

    @methods.loss.register("mine")
    def _(head: HeadOutput, targets: Int[Array, "b t"], weights, z: bool) -> Sums: ...
"""

import math
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Int, Shaped

from bux import parallel
from bux.parallel import methods

CHUNK_BYTES = 2**31  # of float32 logits that a device makes at once


class Projection(NamedTuple):
    """A head's map from the stream to logits, softcap(x W + b), from a matmul in the
    stream's dtype accumulated in float32. A tied head's W is the embedding's table,
    transposed. The softcap is an array, so that jax's transforms pass it through."""

    kernel: Float[Array, "d vocab"]
    bias: Float[Array, " vocab"] | None
    softcap: Float[Array, ""] | None

    def __call__(
        self, x: Float[Array, "*n d"], dtype: DTypeLike = jnp.float32
    ) -> Float[Array, "*n vocab"]:
        """Return the logits of `x`, in `dtype`."""
        return self.finish(self.matmul(x, dtype), self.bias)

    def matmul(
        self, x: Float[Array, "*n d"], dtype: DTypeLike = jnp.float32
    ) -> Float[Array, "*n vocab"]:
        """Return x W, accumulated in float32 and given in `dtype`."""
        dims = (((x.ndim - 1,), (0,)), ((), ()))
        return jax.lax.dot_general(x, self.kernel, dims, preferred_element_type=dtype)

    def finish(
        self, raw: Float[Array, "*n vocab"], bias: Float[Array, " vocab"] | None
    ) -> Float[Array, "*n vocab"]:
        """Return the logits of x W: plus the bias, softcapped, in x W's dtype.
        `chunks` differentiates it by jax as it goes, so another needs none."""
        logits = raw if bias is None else raw + bias.astype(raw.dtype)
        if self.softcap is None:
            return logits
        cap = self.softcap.astype(raw.dtype)
        return cap * jnp.tanh(logits / cap)


class Sums(NamedTuple):
    """A head's sums over positions, each weighted: of the cross-entropy, log Z minus
    the target's logit, and of log Z squared, the z-loss; 0 where not asked for."""

    ce: Float[Array, ""]
    z: Float[Array, ""]


class Scores(NamedTuple):
    """What the cross-entropy and the z-loss take of the logits at each position:
    log Z, the log of the softmax's normalizer, and the target's logit."""

    lse: Float[Array, "*n"]
    target: Float[Array, "*n"]

    @classmethod
    def of(
        cls, logits: Float[Array, "*n vocab"], targets: Int[Array, "*n"]
    ) -> "Scores":
        """Return the scores of logits given whole, in float32."""
        logits = logits.astype(jnp.float32)
        return cls(jax.nn.logsumexp(logits, -1), _pick(logits, targets))

    @classmethod
    def of_chunk(
        cls, logits: Float[Array, "*n vocab"], targets: Int[Array, "*n"]
    ) -> "Scores":
        """Return the scores of a chunk's logits, which `chunks` differentiates by
        hand: each target's logit gathered where a device holds whole rows of them,
        where a sum over them would take a pass of its own."""
        if parallel.splits(("vocab",), logits.shape[-1:])[0] > 1:
            return cls.of(logits, targets)
        logits = logits.astype(jnp.float32)
        target = jnp.take_along_axis(logits, targets[..., None], -1, mode="clip")
        return cls(jax.nn.logsumexp(logits, -1), target[..., 0])

    def sums(self, weights: Float[Array, "*n"], z: bool) -> Sums:
        """Return their sums, each position by its weight; the z-loss's if `z`."""
        ce = jnp.sum(weights * (self.lse - self.target))
        return Sums(ce, jnp.sum(weights * self.lse**2) if z else jnp.zeros(()))


class HeadOutput(NamedTuple):
    """What a head leaves the objective: the stream it reads, and its projection."""

    hidden: Float[Array, "b t d"]
    projection: Projection

    def logits(self) -> Float[Array, "b t vocab"]:
        """Return the logits, all at once, in the stream's dtype."""
        logits = self.projection(self.hidden, self.hidden.dtype)
        return parallel.shard(logits, "batch", "seq", "vocab")

    def sums(
        self, targets: Int[Array, "b t"], weights: Float[Array, "b t"], z: bool
    ) -> Sums:
        """Return the sums of `targets`, each position by its weight, and the
        z-loss's if `z`, by `methods.loss`."""
        return parallel.method("loss")(self, targets, weights, z)


@methods.loss.register("logits")
def by_logits(
    head: HeadOutput, targets: Int[Array, "b t"], weights: Float[Array, "b t"], z: bool
) -> Sums:
    """Sum the scores of the head's logits made whole, as a term that reads them
    makes them."""
    return Scores.of(head.logits(), targets).sums(weights, z)


@methods.loss.register("chunks")
def by_chunks(
    head: HeadOutput, targets: Int[Array, "b t"], weights: Float[Array, "b t"], z: bool
) -> Sums:
    """Sum the scores of a chunk of each device's positions at a time."""
    return _sums(head.projection, head.hidden, targets, weights, z)


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


class Grads(NamedTuple):
    """A sum's gradients: of the kernel, each device's share apart until the last
    chunk, of the bias, and of x."""

    kernel: Float[Array, "*B d vocab"]
    bias: Float[Array, " vocab"] | None
    x: Float[Array, "*B c d"]


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _sums(
    p: Projection,
    x: Float[Array, "b t d"],
    targets: Int[Array, "b t"],
    weights: Float[Array, "b t"],
    z: bool,
) -> Sums:
    """Return the sums, a chunk at a time, without their gradients."""
    b, t, _ = x.shape
    chunks, whole = Chunks.of(b, t, p.kernel.shape[1]), _gathered(p)

    def step(sums: Sums, xs: tuple) -> tuple[Sums, None]:
        x, targets, weights = xs
        logits = _shard(whole(x), None, "vocab")
        chunk = Scores.of_chunk(logits, targets).sums(weights, z)
        return jax.tree.map(jnp.add, sums, chunk), None

    start = Sums(jnp.zeros(()), jnp.zeros(()))
    xs = tuple(chunks.split(a) for a in (x, targets, weights))
    return jax.lax.scan(step, start, xs)[0]


def _sums_fwd(
    p: Projection,
    x: Float[Array, "b t d"],
    targets: Int[Array, "b t"],
    weights: Float[Array, "b t"],
    z: bool,
) -> tuple[Sums, tuple[Grads, Grads | None]]:
    """Return the sums, and the gradients of each, a chunk at a time
    (``_chunk_grads``)."""
    b, t, d = x.shape
    vocab = p.kernel.shape[1]
    chunks, whole = Chunks.of(b, t, vocab), _gathered(p)
    bias = None if whole.bias is None else whole.bias.astype(jnp.float32)

    def step(carry: tuple, xs: tuple) -> tuple[tuple, list[Float[Array, "B C c d"]]]:
        sums, kernels = carry
        chunk, grads = _chunk_grads(whole, bias, *xs, z)
        sums = jax.tree.map(jnp.add, sums, chunk)
        kernels = jax.tree.map(jnp.add, kernels, [(g.kernel, g.bias) for g in grads])
        return (sums, kernels), [g.x for g in grads]

    zeros = _shard(jnp.zeros((chunks.rows, chunks.seq, d, vocab)), None, "vocab")
    kernel = (zeros, None if bias is None else jnp.zeros_like(bias))
    # Each sum's gradients of the kernel and the bias so far: the cross-entropy's, and
    # the z-loss's if `z`.
    start = (Sums(jnp.zeros(()), jnp.zeros(())), [kernel] * (1 + z))
    xs = tuple(chunks.split(a) for a in (x, targets, weights))
    (sums, kernels), dxs = jax.lax.scan(step, start, xs)
    grads = [_whole(p, *k, chunks.join(dx, b, t)) for k, dx in zip(kernels, dxs)]
    return sums, (grads[0], grads[1] if z else None)


def _whole(
    p: Projection,
    kernel: Float[Array, "B C d vocab"],
    bias: Float[Array, " vocab"] | None,
    x: Float[Array, "b t d"],
) -> Grads:
    """Return gradients summed over the devices' shares, in their arrays' dtypes."""
    kernel = kernel.sum((0, 1)).astype(p.kernel.dtype)
    bias = jax.tree.map(lambda g, b: g.astype(b.dtype), bias, p.bias)
    return Grads(kernel, bias, x)


def _sums_bwd(
    z: bool, saved: tuple[Grads, Grads | None], cotangent: Sums
) -> tuple[Projection, Float[Array, "b t d"], None, None]:
    """Return the gradients of `_sums`: each sum's, times its cotangent."""
    ce, z_grads = saved
    grads = _scaled(ce, cotangent.ce)
    if z_grads is not None:
        grads = jax.tree.map(jnp.add, grads, _scaled(z_grads, cotangent.z))
    return Projection(grads.kernel, grads.bias, None), grads.x, None, None


def _scaled(grads: Grads, factor: Float[Array, ""]) -> Grads:
    return jax.tree.map(
        lambda g: (factor * g.astype(jnp.float32)).astype(g.dtype), grads
    )


def _chunk_grads(
    p: Projection,
    bias: Float[Array, " vocab"] | None,
    x: Float[Array, "B C c d"],
    targets: Int[Array, "B C c"],
    weights: Float[Array, "B C c"],
    z: bool,
) -> tuple[Sums, list[Grads]]:
    """Return a chunk's sums, and their gradients from its logits: the
    cross-entropy's, and the z-loss's if `z`."""
    raw = _shard(p.matmul(x), None, "vocab")
    logits, finish = jax.vjp(p.finish, raw, bias)
    scores = Scores.of_chunk(logits, targets)
    probs = jnp.exp(logits - scores.lse[..., None])
    hit = _is_target(logits, targets)
    d_ce = weights[..., None] * (probs - hit)
    d_logits = [d_ce]
    if z:
        d_logits.append((2 * weights * scores.lse)[..., None] * probs)
    grads = [_chunk_grad(p, finish, x, d) for d in d_logits]
    return scores.sums(weights, z), grads


def _chunk_grad(
    p: Projection,
    finish: Callable,
    x: Float[Array, "B C c d"],
    d_logits: Float[Array, "B C c vocab"],
) -> Grads:
    """Return the gradients of a sum whose gradient by the chunk's logits is
    `d_logits`: of the kernel, each device's share apart, of the bias, and of x."""
    d_raw, d_bias = finish(d_logits)
    d_raw = d_raw.astype(x.dtype)
    f32 = jnp.float32
    d_kernel = jnp.einsum("BCcd,BCcv->BCdv", x, d_raw, preferred_element_type=f32)
    dx = jnp.einsum("BCcv,dv->BCcd", d_raw, p.kernel, preferred_element_type=f32)
    return Grads(d_kernel, d_bias, dx.astype(x.dtype))


_sums.defvjp(_sums_fwd, _sums_bwd)
