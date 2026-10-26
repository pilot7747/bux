"""CSA's attention by cuDNN's DSA kernels (bux.architecture.attention.sparse_cudnn) off
a GPU: where the kernels are taken, and all that runs around them, with the kernels
replaced by XLA's computation of what they compute. Each query's list of slots, the
flat latents, the heads in groups of 64, the sinks and the gradients back to each part
give `sparse.attend`'s values, on one device and split over 8; a CSA layer trains as
by XLA, its indexer's loss against the target at its picks.
scripts/gpu_smoke/sparse.py checks the kernels themselves on a B300."""

import collections

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from bux import parallel
from bux.architecture import Model, Segments
from bux.architecture.attention import sparse, sparse_cudnn
from bux.architecture.attention.indexer import _top
from bux.lang import compile
from bux.train import config as tcfg

B, T, D = 2, 16, sparse_cudnn.DIMS
# Three documents in the first row, cut mid-way, and one in the second.
POSITIONS = jnp.asarray([[*range(3, 8), *range(7), *range(4)], [*range(T)]])
ROWS = Segments.of(POSITIONS)
SCALE = D**-0.5
WINDOW = 3
F32 = jnp.float32


# --- the kernels, by XLA ------------------------------------------------------------


def forward(q, latents, lists, sinks, scale):
    """cuDNN's forward of 64 heads: the output, and the logsumexp of the slots alone,
    +inf for none."""
    keys = latents[jnp.maximum(lists, 0)]
    logits = jnp.einsum("nhd,nkd->nhk", q, keys, preferred_element_type=F32) * scale
    logits = jnp.where(lists[:, None] >= 0, logits, -jnp.inf)
    lse = jax.nn.logsumexp(logits, -1)
    total = jnp.logaddexp(lse, sinks)
    p = jnp.exp(logits - jnp.where(jnp.isfinite(total), total, 0)[..., None])
    out = jnp.einsum("nhk,nkd->nhd", p, keys.astype(F32))
    return out.astype(q.dtype), jnp.where(jnp.isfinite(lse), lse, jnp.inf)


def backward(q, latents, out, dout, lse, sinks, lists, scale):
    """cuDNN's backward of 64 heads: the gradients of `forward`."""

    def attend(q, latents, sinks):
        return forward(q, latents, lists, sinks, scale)[0]

    return jax.vjp(attend, q, latents, sinks)[1](dout)


def score(q, latents, lse, picks, scale):
    """cuDNN's target at the picks: summed over heads, normalized over the picks."""
    keys = jax.vmap(lambda k, i: k[jnp.maximum(i, 0)])(latents, picks)
    logits = jnp.einsum("bthd,btkd->bthk", q, keys, preferred_element_type=F32)
    p = jnp.exp(logits * scale - lse[..., None]).sum(2)
    p = jnp.where(picks >= 0, p, 0)
    total = p.sum(-1, keepdims=True)
    return jnp.where(total > 1e-10, p / jnp.where(total > 1e-10, total, 1), 0)


def emulate(patch: pytest.MonkeyPatch) -> collections.Counter:
    """Run the kernels by XLA, taken wherever they would be on a B300, in float32 too;
    return how many times each one traces."""
    traced = collections.Counter()

    def counted(f):
        def call(*args, **kwargs):
            traced[f.__name__] += 1
            return f(*args, **kwargs)

        return call

    patch.setattr(sparse_cudnn, "runs", lambda: True)
    patch.setattr(sparse_cudnn, "DTYPES", (F32,))
    patch.setattr(sparse_cudnn, "TARGET_DTYPES", (F32,))
    calls = {"_forward_call": forward, "_backward_call": backward, "_score_call": score}
    for name, f in calls.items():
        patch.setattr(sparse_cudnn, name, counted(f))
    return traced


@pytest.fixture
def emulated(monkeypatch) -> collections.Counter:
    return emulate(monkeypatch)


# --- a layer's parts ------------------------------------------------------------------


def case(heads: int, window_only: bool = False):
    """CSA's parts: a window of the tokens' latents, then latents every other slot,
    each query's top 3 of them (some outside its span); queries of `heads` heads, and
    sinks."""
    key = jax.random.split(jax.random.key(0), 5)
    kv = jax.random.normal(key[0], (B, T, 1, D))
    parts = [sparse.Part(kv, kv, sparse.token_span(ROWS, WINDOW))]
    if not window_only:
        tokens = sparse.token_span(ROWS, None)
        latents = jax.random.normal(key[1], (B, T, 1, D))
        exists = jnp.broadcast_to(jnp.arange(T) % 2 == 1, (B, T))
        scores = jax.random.uniform(key[2], (B, T, T))
        kept = sparse.Kept(_top(jnp.where(sparse.mask(tokens, T), scores, -1), 3))
        parts.append(sparse.Part(latents, latents, tokens, exists, kept))
    q = jax.random.normal(key[3], (B, T, heads, D))
    return q, parts, jax.random.normal(key[4], (heads,))


def windows(parts) -> tuple[int | None, ...]:
    return (WINDOW, None)[: len(parts)]


def by_kernels(q, parts, sinks):
    return sparse_cudnn.attend(q, parts, sinks, SCALE, windows(parts))


def by_xla(q, parts, sinks):
    return sparse.attend(q, parts, sinks, SCALE, None)


def run(f, q, parts, sinks):
    """f's output and logsumexp, and the gradients of q, each part's latents (its
    keys and values) and the sinks."""

    def loss(q, latents, sinks):
        given = [p._replace(k=k, v=k) for p, k in zip(parts, latents)]
        out, lse = f(q, given, sinks)
        cotangent = jax.random.normal(jax.random.key(9), out.shape)
        return (out * cotangent).sum(), (out, lse)

    latents = [p.k for p in parts]
    return jax.jit(jax.grad(loss, argnums=(0, 1, 2), has_aux=True))(q, latents, sinks)


def close(a, b, atol=2e-5) -> None:
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, atol=atol, rtol=1e-5)


# --- the tests ------------------------------------------------------------------------


@pytest.mark.parametrize("heads", [4, 64, 70], ids=["padded", "one_group", "two"])
@pytest.mark.parametrize("window_only", [False, True], ids=["csa", "window"])
@pytest.mark.parametrize("has_sinks", [True, False], ids=["sinks", "none"])
def test_the_kernels_attend_as_xla(emulated, heads, window_only, has_sinks):
    q, parts, sinks = case(heads, window_only)
    sinks = sinks if has_sinks else None
    close(run(by_kernels, q, parts, sinks), run(by_xla, q, parts, sinks))


MESHES = {
    "context_tensor": tcfg.MeshConfig(context=2, tensor=2),
    "data_tensor": tcfg.MeshConfig(data=2, tensor=4),
}


@pytest.mark.parametrize("mesh", MESHES.values(), ids=MESHES.keys())
def test_split_over_devices_they_attend_as_on_one(emulated, mesh):
    """A device has its rows, queries and heads, and every slot."""
    q, parts, sinks = case(8)
    want = run(by_kernels, q, parts, sinks)
    with parallel.use(mesh):
        close(run(by_kernels, q, parts, sinks), want)


def test_a_query_s_list_holds_its_slots_by_their_place_in_the_flat_latents():
    _, parts, _ = case(4)
    latents, lists = sparse_cudnn._lists(parts, windows(parts))
    assert latents.shape == (2 * B * T, D)
    assert lists.shape == (B * T, sparse_cudnn.SLOTS)  # 3 + 3 slots, a tile
    seen = [sparse.visible(p) for p in parts]  # (b, t, s), each part's
    for row in range(B):
        for t in range(T):
            got = set(np.asarray(lists[row * T + t]).tolist()) - {-1}
            want = {
                part * B * T + row * T + slot
                for part in range(2)
                for slot in np.flatnonzero(seen[part][row, t])
            }
            assert got == want


def test_the_kernels_take_csa_s_latents_alone(monkeypatch):
    q, parts, _ = case(4)
    half = q.astype(jnp.bfloat16)
    latents = [p._replace(k=p.k.astype(jnp.bfloat16)) for p in parts]
    assert not sparse_cudnn.takes(half, latents, windows(parts), None)  # no GPU
    monkeypatch.setattr(sparse_cudnn, "runs", lambda: True)
    assert sparse_cudnn.takes(half, latents, windows(parts), None)
    assert not sparse_cudnn.takes(half, latents, windows(parts), 30.0)  # a soft cap
    assert not sparse_cudnn.takes(half, parts, windows(parts), None)  # mixed dtypes
    assert not sparse_cudnn.takes(q, parts, windows(parts), None)  # float32
    assert not sparse_cudnn.takes(half, latents, (None, None), None)  # no window
    narrow = [p._replace(k=p.k[..., :256]) for p in latents]
    assert not sparse_cudnn.takes(half[..., :256], narrow, windows(parts), None)
    block = jnp.zeros((B, T), jnp.int32)
    blocks = [
        latents[0],
        latents[1]._replace(kept=sparse.Kept(parts[1].kept.index, block)),
    ]
    assert not sparse_cudnn.takes(half, blocks, windows(parts), None)


def test_the_target_at_the_picks_is_the_attention_to_them(emulated):
    """It sums the heads' attention to each pick, and normalizes it over them: 0 for
    a query with none."""
    q, parts, sinks = case(4)
    _, lse = by_xla(q, parts, sinks)
    got = sparse_cudnn.probabilities(q, parts[1], lse, SCALE)
    every = sparse.probabilities(q, parts[1], lse, SCALE, None)
    index = parts[1].kept.index
    kept = jnp.where(
        index >= 0, jnp.take_along_axis(every, jnp.maximum(index, 0), -1), 0
    )
    total = kept.sum(-1, keepdims=True)
    np.testing.assert_allclose(got, kept / jnp.where(total > 0, total, 1), atol=1e-6)


PRELUDE = """
d_model = 32
vocab_size = 64
norm = rmsnorm
ffn = gated_mlp(intermediate_size=48, activation=swish)
small = indexer(num_heads=2, head_dim=8, top_k=3)
window = csa(num_heads=4, head_dim=512, q_lora_rank=12, o_lora_rank=6, o_groups=2,
             sliding_window_size=3, sinks=true, rope=rope(fraction=0.25, layout=interleaved))
model = embed [norm window] [norm window(compress_ratio=2, indexer=small)] [norm ffn]
        norm head
"""


def trained() -> tuple[float, float, nnx.State]:
    """The language-modeling and indexer losses of a training forward on packed rows,
    and the gradient of their sum."""
    m = Model(compile(PRELUDE), rngs=nnx.Rngs(0))
    tokens = jax.random.randint(jax.random.key(1), (B, T), 0, 64)

    def loss(m):
        [head], reported = m.training_heads(tokens, POSITIONS)
        lm = -jax.nn.log_softmax(head.logits())[..., 0].mean()
        indexer = sum(r.value for r in reported)
        return lm + indexer, (lm, indexer)

    (_, (lm, indexer)), grads = nnx.value_and_grad(loss, has_aux=True)(m)
    return float(lm), float(indexer), nnx.state(grads)


def test_a_csa_layer_trains_by_the_kernels_as_by_xla(monkeypatch):
    """Its window-only and compressed layers, the indexer's loss and every gradient."""
    lm, indexer, grads = trained()
    traced = emulate(monkeypatch)
    got_lm, got_indexer, got_grads = trained()
    assert set(traced) == {"forward", "backward", "score"}
    assert indexer > 0 and np.isclose(got_indexer, indexer, atol=1e-6)
    assert np.isclose(got_lm, lm, atol=1e-6)
    close(got_grads, grads, atol=1e-5)
