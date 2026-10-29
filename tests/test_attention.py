"""Softmax attention layers: MLA, CSA, the sparse attention indexer and its loss, and
what they share with GQA (tests/test_backend.py has GQA's own); and all of them split
by head over the 8 CPU devices the tests have, against one device."""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from bux import config as cfg
from bux import initializers, parallel
from bux.architecture import Attention, Indexer, Model, MoE, Segments, Taps, num_params
from bux.architecture.attention import sparse
from bux.architecture.attention.csa import CSA
from bux.architecture.attention.gqa import build_gqa
from bux.architecture.attention.indexer import _top
from bux.architecture.attention.mla import MLA, build_mla
from bux.architecture.attention.rope import rotate
from bux.architecture.layers.embedding import ValueEmbedding
from bux.lang import compile, compile_runs, ir
from bux.train import config as tcfg
from bux.train.data import Batch
from bux.train.objective import objective
from bux.train.trainer import Trainer, train

D, V = 32, 64

PRELUDE = f"""
d_model = {D}
vocab_size = {V}
norm = rmsnorm
ffn = gated_mlp(intermediate_size=48, activation=swish)
small = indexer(num_heads=2, head_dim=8, top_k=3)
latent = mla(num_heads=4, q_lora_rank=12, kv_lora_rank=8, qk_nope_head_dim=6,
             qk_rope_head_dim=4, v_head_dim=5)
window = csa(num_heads=4, head_dim=16, q_lora_rank=12, o_lora_rank=6, o_groups=2,
             sliding_window_size=3, sinks=true, rope=rope(fraction=0.25, layout=interleaved))
experts = moe(num_experts=4, top_k=2, expert=ffn)
"""

BODIES = {
    "mla": "model = embed [norm latent] [norm ffn] norm head",
    "mla_nope_gated_full_rank": (
        "a = latent(q_lora_rank=none, rope=none, gated=true)\n"
        "model = embed [norm a] [norm ffn] norm head"
    ),
    "mla_indexer": "model = embed [norm latent(indexer=small)] [norm ffn] norm head",
    # what every softmax attention layer has, through MLA and CSA
    "mla_options": (
        "a = latent(qk_norm=norm, output_norm=norm, softcap=20, sinks=true,"
        " gated=true, gate_per_head=true, relative_bias=relbias(rank=2, extent=4),"
        " log_scale=0.1, log_scale_floor=2, sliding_window_size=5)\n"
        "model = embed [norm a] [norm ffn] norm head"
    ),
    "csa_options": (
        "a = window(compress_ratio=2, qk_norm=norm, output_norm=norm, softcap=20,"
        " gated=true, relative_bias=relbias(rank=2, extent=4), log_scale=0.1,"
        " log_scale_floor=2)\n"
        "model = embed [norm a] [norm ffn] norm head"
    ),
    "gqa_indexer": (
        "a = gqa(num_heads=4, num_kv_heads=2, rope=rope(), indexer=small)\n"
        "model = embed [norm a] [norm ffn] norm head"
    ),
    "gqa_block_indexer": (
        "a = gqa(num_heads=4, num_kv_heads=2, rope=rope(fraction=0.5), sliding_window_size=6,"
        " indexer=small(top_k=4, block_size=2, weighted=false))\n"
        "model = embed [norm a] [norm ffn] norm head"
    ),
    # MiniMax MSA: max-pooled blocks, a pick for each kv head's group, plain dots
    "gqa_msa": (
        "a = gqa(num_heads=4, num_kv_heads=2, rope=rope(fraction=0.5), sliding_window_size=7,"
        " indexer=small(top_k=4, block_size=2, pool=max, per_group=true, relu=false,"
        " weighted=false))\n"
        "model = embed [norm a] [norm ffn] norm head"
    ),
    "csa_window": "model = embed [norm window] [norm ffn] norm head",
    "csa_compressed": (
        "model = embed [norm window(compress_ratio=2, indexer=small(top_k=2))]"
        " [norm ffn] norm head"
    ),
    "csa_heavily_compressed": (
        "model = embed [norm window(compress_ratio=3, o_lora_rank=none)] [norm ffn] norm head"
    ),
    # DeepSeek-V4.1: compressed latents and top-k shared, and a re-index in a pool
    "csa_ports": (
        "pairs = window(compress_ratio=2, indexer=small)\n"
        "single = pairs(compress_ratio=1, indexer=small(candidate_block_size=2,"
        " candidate_top_k=2))\n"
        "model = embed [norm window] [norm pairs(kv=@kv, index=@topk)]"
        " [norm pairs(kv=$kv, index=$topk)]"
        " [norm single(kv=@kv, index=@topk, candidates=@pool)]"
        " [norm single(kv=$kv, index=@topk, candidates=$pool)]"
        " [norm single(kv=$kv, index=$topk)] norm head"
    ),
    # GLM-5's IndexCache, with a reader that has no indexer of its own at all
    "mla_index_ports": (
        "model = embed [norm latent(indexer=small, index=@k)]"
        " [norm latent(indexer=small, index=$k)] [norm latent(index=$k)]"
        " [norm gqa(num_heads=4, num_kv_heads=4, index=$k)] norm head"
    ),
}


def model(body: str) -> Model:
    return Model(compile(PRELUDE + body), rngs=nnx.Rngs(0))


def tokens(seed: int = 0, length: int = 12):
    return jax.random.randint(jax.random.key(seed), (2, length), 0, V)


def inputs(seed: int = 1, length: int = 12):
    return jax.random.normal(jax.random.key(seed), (2, length, D))


def causal(t: int, window: int | None = None):
    i = jnp.arange(t)
    mask = i[:, None] >= i[None, :]
    if window is not None:
        mask &= i[:, None] - i[None, :] < window
    return jnp.broadcast_to(mask, (2, t, t))


def keeps(kept, visible):
    """Which keys `kept` keeps, of those `visible` says each query sees: (b, t, s), or
    (b, t, g, s) for kept keys of each of g groups."""
    b, t, s = visible.shape
    every = sparse.Span(jnp.zeros((b, t), jnp.int32), jnp.full((b, t), s, jnp.int32))
    picked = sparse.mask(every, s, None, kept)
    return picked & (visible if picked.ndim == 3 else visible[:, :, None])


def per_head(groups, h):
    """Each of h query heads' mask, its group's of (b, t, g, s), as (b, h, t, s)."""
    return jnp.repeat(jnp.moveaxis(groups, 2, 1), h // groups.shape[2], axis=1)


def attention_probs(q, k, scale, mask, sinks=None):
    """softmax(q . k * scale) over the keys `mask` keeps, next to the heads' sinks;
    `mask` is every head's, (b, t, s), or each head's, (b, h, t, s)."""
    logits = jnp.einsum("bthd,bshd->bhts", q, k) * scale
    logits = jnp.where(mask[:, None] if mask.ndim == 3 else mask, logits, -jnp.inf)
    if sinks is not None:
        sink = jnp.broadcast_to(sinks[None, :, None, None], (*logits.shape[:3], 1))
        logits = jnp.concatenate([logits, sink], -1)
    return jnp.nan_to_num(jax.nn.softmax(logits)[..., : k.shape[1]])


def masked_attention(q, k, v, scale, mask, sinks=None):
    """softmax(q . k * scale) v over the keys `mask` keeps, next to the heads' sinks."""
    probs = attention_probs(q, k, scale, mask, sinks)
    return jnp.einsum("bhts,bshd->bthd", probs, v)


@pytest.mark.parametrize("body", BODIES.values(), ids=BODIES.keys())
def test_layers_run_and_are_causal(body):
    m = model(body)
    t = tokens()
    logits = nnx.jit(lambda m, t: m(t))(m, t)
    assert logits.shape == (2, 12, V) and bool(jnp.isfinite(logits).all())
    changed = t.at[:, 7:].set((t[:, 7:] + 1) % V)
    assert jnp.allclose(m(changed)[:, :7], logits[:, :7], atol=1e-5)
    assert not jnp.allclose(m(changed)[:, 7:], logits[:, 7:])
    grads = nnx.grad(lambda m: m(t).mean())(m)
    assert all(bool(jnp.isfinite(g).all()) for g in jax.tree.leaves(grads))


@pytest.mark.parametrize(
    "body",
    ["mla_indexer", "gqa_block_indexer", "gqa_msa", "csa_compressed", "csa_ports"],
)
def test_documents_packed_in_a_row_see_only_themselves(body):
    """With an odd first document, a compressed group or an indexer's block that ran on
    across documents would show."""
    m = model(BODIES[body])
    a, b = tokens(0, 5), tokens(1, 7)
    positions = jnp.concatenate([jnp.arange(5), jnp.arange(7)])[None].repeat(2, 0)
    [packed] = m.outputs(jnp.concatenate([a, b], axis=1), positions)
    assert jnp.allclose(packed[:, :5], m(a), atol=1e-5)
    assert jnp.allclose(packed[:, 5:], m(b), atol=1e-5)
    later = jnp.concatenate([jnp.arange(5), jnp.arange(7) + 3])[None].repeat(2, 0)
    [packed] = m.outputs(jnp.concatenate([a, b], axis=1), later)
    assert jnp.allclose(packed[:, 5:], m.outputs(b, later[:, 5:])[0], atol=1e-5)


def test_mla_parameter_count():
    h, n, r, dv, kv = 4, 6, 4, 5, 8

    def params(q_lora_rank):
        c = cfg.MLAConfig(
            num_heads=h,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv,
            qk_nope_head_dim=n,
            qk_rope_head_dim=r,
            v_head_dim=dv,
        )
        return num_params(MLA(c, D, (), nnx.Rngs(0)))

    keys_values = D * (kv + r) + kv + kv * h * (n + dv)  # with the latent's norm
    assert params(None) == D * h * (n + r) + keys_values + h * dv * D
    low = D * 12 + 12 + 12 * h * (n + r)  # with the latent's norm
    assert params(12) == low + keys_values + h * dv * D


def test_mla_is_deepseeks_attention():
    """The rotated and the other dims scored apart, as DeepSeek's inference code does,
    against latents projected up into keys and values."""
    yarn = cfg.YaRNConfig(
        factor=40, original_max_len=4096, attention_factor_on="softmax"
    )
    c = cfg.MLAConfig(
        num_heads=4,
        q_lora_rank=12,
        kv_lora_rank=8,
        qk_nope_head_dim=6,
        qk_rope_head_dim=4,
        v_head_dim=5,
        rope=cfg.RoPEConfig(layout="interleaved", scaling=yarn),
    )
    mla = MLA(c, D, (), nnx.Rngs(0))
    assert math.isclose(mla.scale, 10**-0.5 * (0.1 * math.log(40) + 1) ** 2)
    assert mla.rope is not None and mla.q_a is not None and mla.q_a_norm is not None
    x = inputs()
    b, t, _ = x.shape
    q = mla.q(mla.q_a_norm(mla.q_a(x))).reshape(b, t, 4, 10)
    latent, k_rope = mla.kv_a(x)[..., :8], mla.kv_a(x)[..., 8:]
    kv = mla.kv_b(mla.kv_a_norm(latent)).reshape(b, t, 4, 11)
    q_rope, k_rope = rotate(q[..., 6:], mla.rope), rotate(k_rope[:, :, None], mla.rope)
    logits = jnp.einsum("bthd,bshd->bhts", q[..., :6], kv[..., :6])
    logits += jnp.einsum("bthr,bsr->bhts", q_rope, k_rope[:, :, 0])
    logits = jnp.where(causal(t)[:, None], logits * mla.scale, -jnp.inf)
    out = jnp.einsum("bhts,bshv->bthv", jax.nn.softmax(logits), kv[..., 6:])
    assert jnp.allclose(mla(x, Taps()), mla.o(out.reshape(b, t, -1)), atol=1e-5)


def test_mla_without_rope_sees_no_positions():
    c = cfg.MLAConfig(num_heads=2, kv_lora_rank=8, rope=None)
    mla = MLA(c, D, (), nnx.Rngs(0))
    x = inputs()
    shifted = Taps(segments=Segments.of(jnp.arange(12)[None].repeat(2, 0) + 100))
    assert jnp.allclose(mla(x, shifted), mla(x, Taps()), atol=1e-5)


def gqa(ports=(), **kw):
    c = cfg.GQAConfig(num_heads=4, num_kv_heads=2, rope=cfg.RoPEConfig(), **kw)
    return build_gqa(c, D, ports, nnx.Rngs(0))


def gqa_heads(attn, x):
    """The queries, keys and values of every head."""
    b, t, _ = x.shape
    h, g, d = attn.heads, attn.kv_heads, attn.head_dim
    assert attn.q is not None and attn.k is not None and attn.v is not None
    assert attn.rope is not None
    q = rotate(attn.q(x).reshape(b, t, h, d), attn.rope)
    k = rotate(attn.k(x).reshape(b, t, g, d), attn.rope)
    v = attn.v(x).reshape(b, t, g, d)
    return q, *(jnp.repeat(a, h // g, axis=2) for a in (k, v))


def gqa_reference(attn, x, mask):
    b, t, _ = x.shape
    q, k, v = gqa_heads(attn, x)
    return attn.o(masked_attention(q, k, v, attn.scale, mask).reshape(b, t, -1))


def index_port(mode):
    return (ir.Port(name="index", tap="topk", mode=mode),)


def mla(indexer=None):
    c = cfg.MLAConfig(
        num_heads=4,
        q_lora_rank=12,
        kv_lora_rank=8,
        qk_nope_head_dim=6,
        qk_rope_head_dim=4,
        v_head_dim=5,
        indexer=indexer,
    )
    return build_mla(c, D, (), nnx.Rngs(0))


MSA = {"block_size": 3, "pool": "max", "per_group": True, "relu": False}
INDEXERS = {
    "gqa": lambda indexer: gqa(indexer=indexer),
    "gqa_blocks": lambda indexer: gqa(
        indexer=indexer and indexer.model_copy(update={"block_size": 3})
    ),
    "gqa_msa": lambda indexer: gqa(
        indexer=indexer and indexer.model_copy(update=MSA | {"weighted": False})
    ),
    "mla": mla,
    "mla_max_blocks": lambda indexer: mla(
        indexer and indexer.model_copy(update={"block_size": 3, "pool": "max"})
    ),
    "csa": lambda indexer: csa(
        compress_ratio=2, sliding_window_size=2, indexer=indexer
    ),
}


@pytest.mark.parametrize("make", INDEXERS.values(), ids=INDEXERS.keys())
def test_an_indexer_that_keeps_every_key_is_dense_attention(make):
    """The indexer's parameters come last, so with the same seed the layer has the same
    others as without it."""
    every = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=12)
    x = inputs()
    assert jnp.allclose(make(every)(x, Taps()), make(None)(x, Taps()), atol=1e-5)
    few = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=3)
    assert not jnp.allclose(make(few)(x, Taps()), make(None)(x, Taps()), atol=1e-3)


@pytest.mark.parametrize("weighted", [True, False])
def test_a_small_top_k_attends_to_the_best_scored_keys_only(weighted):
    c = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=3, weighted=weighted)
    attn = gqa(index_port("write"), indexer=c)
    assert attn.indexer is not None
    x, taps = inputs(), Taps()
    y = attn(x, taps)
    selected = keeps(taps["topk"][0], causal(12))
    # The best 3 of the keys each query sees, by the indexer's scores
    scores = attn.indexer.scores(x, x, None, attn.indexer.keys(x, None))
    scores = jnp.where(causal(12), scores, -jnp.inf)
    best = jnp.argsort(-scores, axis=-1)[..., :3]
    expected = jax.nn.one_hot(best, 12).sum(-2).astype(bool) & causal(12)
    assert (selected == expected).all()
    assert (selected.sum(-1) == jnp.minimum(jnp.arange(12) + 1, 3)).all()
    assert jnp.allclose(y, gqa_reference(attn, x, selected), atol=1e-5)


def test_blocks_are_kept_whole_with_the_querys_unfinished_one():
    c = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=4, block_size=2)
    attn = gqa(index_port("write"), indexer=c)
    x, taps = inputs(), Taps()
    y = attn(x, taps)
    selected = keeps(taps["topk"][0], causal(12))
    pairs = selected.reshape(2, 12, 6, 2)
    i = jnp.arange(12)
    # 2 of the blocks each query sees all of, its own among them once it is complete,
    # and the query itself while its block is unfinished.
    assert (pairs.all(-1).sum(-1) == jnp.minimum((i + 1) // 2, 2)).all()
    unfinished = (i % 2 == 0)[:, None] & (jnp.arange(6) == i[:, None] // 2)
    assert (pairs.any(-1) & ~pairs.all(-1) == unfinished).all()
    assert jnp.allclose(y, gqa_reference(attn, x, selected), atol=1e-5)


def test_qsa_keeps_the_blocks_qwen_does():
    """Qwen3.8 QSA's selection as transformers' Qwen4ExpTextQSAIndexer makes it, query
    by query: each complete block of 4 of the keys a query sees is the mean of its raw
    keys, normalized, then turned to its first position; the best 2 are kept, and the
    keys after them."""
    norm = cfg.NormalizationConfig(type="rmsnorm", zero_centered=True)
    c = cfg.SparseIndexerConfig(
        num_heads=2, head_dim=16, top_k=8, block_size=4, weighted=False, norm=norm
    )
    attn = gqa(index_port("write"), indexer=c)
    ix = attn.indexer
    assert ix is not None and ix.k is not None and ix.rope is not None
    assert ix.q_norm is not None and ix.k_norm is not None
    ix.q_norm.scale[...] = jax.random.normal(jax.random.key(3), (16,))
    ix.k_norm.scale[...] = jax.random.normal(jax.random.key(4), (16,))
    pos = jnp.concatenate([jnp.arange(9), jnp.arange(23) + 5])[None].repeat(2, 0)
    x, taps = inputs(length=32), Taps(segments=Segments.of(pos))
    attn(x, taps)
    [selected] = taps["topk"]

    def turn(a, at):  # the first 8 dims, which the layer's RoPE turns in its heads
        return jnp.concatenate([rotate(a[..., :8], ix.rope, at), a[..., 8:]], -1)

    q = turn(ix.q_norm(ix.q(x).reshape(2, 32, 2, 16)), pos)
    raw, doc = ix.k(x), jnp.arange(32) >= 9
    visible = causal(32) & (doc[:, None] == doc[None])
    expected = jnp.zeros((2, 32, 32), bool)
    for row in range(2):
        for i in range(32):
            seen = jnp.flatnonzero(visible[row, i])
            n = len(seen) // 4
            expected = expected.at[row, i, seen[4 * n :]].set(True)
            if n == 0:
                continue
            blocks = seen[: 4 * n].reshape(n, 4)
            k = ix.k_norm(raw[row, blocks].mean(1))
            k = turn(k[None, :, None], pos[row, blocks[:, 0]][None])[0, :, 0]
            scores = jax.nn.relu(q[row, i] @ k.T).sum(0)
            expected = expected.at[row, i, blocks[jnp.argsort(-scores)[:2]]].set(True)
    assert (keeps(selected, visible) == expected).all()


ZERO_CENTERED = cfg.NormalizationConfig(type="rmsnorm", zero_centered=True)
# MiniMax MSA's: one indexer head for each kv head, plain dot products, and a block's
# best score
GROUPS = {"per_group": True, "relu": False, "weighted": False}
MAX_BLOCKS = {"block_size": 4, "pool": "max"}


def msa(ports=(), window=None, **kw):
    """MiniMax MSA's attention: gqa whose indexer has a head for each of its 2 kv
    heads, of 16 dims, a pick of the best 2 blocks of 4 keys for each group."""
    small = {"num_heads": 2, "head_dim": 16, "top_k": 8, "norm": ZERO_CENTERED}
    settings = small | GROUPS | MAX_BLOCKS | kw
    c = cfg.SparseIndexerConfig.model_validate(settings)
    return gqa(ports, indexer=c, sliding_window_size=window)


def msa_scores(ix, x, pos=None):
    """MSA's scores, (b, t, g, s): each group's head's plain dot products with the keys
    over sqrt(16), its queries and keys normalized, then their first 8 dims (the
    layer's head dims) turned."""
    assert ix.k is not None and ix.k_norm is not None and ix.q_norm is not None
    b, t, _ = x.shape

    def turn(a):
        return jnp.concatenate([rotate(a[..., :8], ix.rope, pos), a[..., 8:]], -1)

    q = turn(ix.q_norm(ix.q(x).reshape(b, t, 2, 16)))
    k = turn(ix.k_norm(ix.k(x))[:, :, None])[:, :, 0]
    return jnp.einsum("btgi,bsi->btgs", q, k) / 4


@pytest.mark.parametrize("window", [None, 7])
def test_msa_keeps_the_blocks_minimax_does(window):
    """MiniMax MSA's selection as transformers' MiniMaxM3VLIndexer makes it, query by
    query and group by group: a block of 4 keys, counted from its document's start,
    scores the best of its keys the query sees; the query keeps its own block and the
    best other one, 8 keys' worth, for each group apart."""
    attn = msa(index_port("write"), window=window)
    ix = attn.indexer
    assert ix is not None and ix.q_norm is not None and ix.k_norm is not None
    ix.q_norm.scale[...] = jax.random.normal(jax.random.key(3), (16,))
    ix.k_norm.scale[...] = jax.random.normal(jax.random.key(4), (16,))
    pos = jnp.concatenate([jnp.arange(9), jnp.arange(23) + 5])[None].repeat(2, 0)
    x, taps = inputs(length=32), Taps(segments=Segments.of(pos))
    attn(x, taps)
    [selected] = taps["topk"]
    scores = np.asarray(msa_scores(ix, x, pos))
    doc = np.arange(32) >= 9
    visible = np.asarray(causal(32, window)) & (doc[:, None] == doc[None])
    block = np.where(doc, 100 + (np.arange(32) - 9) // 4, np.arange(32) // 4)
    expected = np.zeros((2, 32, 2, 32), bool)
    for row, i, r in np.ndindex(2, 32, 2):
        seen = visible[row, i]
        best = {m: scores[row, i, r][seen & (block == m)].max() for m in block[seen]}
        others = sorted(set(best) - {block[i]}, key=lambda m: (-best[m], m))
        for m in [block[i], *others[:1]]:
            expected[row, i, r] |= seen & (block == m)
    got = np.asarray(keeps(selected, jnp.asarray(visible)))
    assert (got == expected).all()
    assert (expected[:, :, 0] != expected[:, :, 1]).any()  # the groups pick apart


def test_msa_attends_each_group_to_its_own_blocks():
    """Each group of query heads attends to its own picks; with every block kept, the
    layer is dense attention."""
    attn = msa(index_port("write"), top_k=4, block_size=2)
    x, taps = inputs(), Taps()
    y = attn(x, taps)
    selected = keeps(taps["topk"][0], causal(12))
    i = jnp.arange(12)
    own, other = i % 2 + 1, jnp.where(i >= 2, 2, 0)  # its block's keys, and another's
    assert (selected.sum(-1) == (own + other)[:, None]).all()
    assert jnp.allclose(y, gqa_reference(attn, x, per_head(selected, 4)), atol=1e-5)
    every = msa(top_k=12, block_size=2)
    assert jnp.allclose(every(x, Taps()), gqa()(x, Taps()), atol=1e-5)


def test_per_group_without_blocks_keeps_each_groups_best_keys():
    """Each group's head keeps the 3 best keys the query sees by its own scores, the
    group attends to them, and the loss is MSA's over them."""
    attn = msa(index_port("write"), top_k=3, block_size=1)
    assert attn.indexer is not None
    x, taps = inputs(), Taps(training=True)
    y = attn(x, taps)
    [kept], [loss] = taps["topk"], taps.losses
    selected, seen = keeps(kept, causal(12)), causal(12)[:, :, None]
    scores = msa_scores(attn.indexer, x)
    best = jnp.argsort(-jnp.where(seen, scores, -jnp.inf), axis=-1)[..., :3]
    assert (selected == jax.nn.one_hot(best, 12).sum(-2).astype(bool) & seen).all()
    assert jnp.allclose(y, gqa_reference(attn, x, per_head(selected, 4)), atol=1e-5)
    q, k, _ = gqa_heads(attn, x)
    p = attention_probs(q, k, attn.scale, per_head(selected, 4))
    p = jnp.moveaxis(p.reshape(2, 2, 2, 12, 12).mean(2), 1, 2)
    assert jnp.allclose(loss.value, kl(p, scores, selected).mean(), atol=1e-6)


def test_a_layer_reading_an_msa_index_attends_by_its_groups():
    """A per-group index is read by a layer with as many kv heads alone."""
    writer = msa(index_port("write"), top_k=4, block_size=2)
    reader = gqa(index_port("read"))
    x, taps = inputs(), Taps()
    writer(x, taps)
    selected = keeps(taps["topk"][0], causal(12))
    h = inputs(2)
    expected = gqa_reference(reader, h, per_head(selected, 4))
    assert jnp.allclose(reader(h, taps), expected, atol=1e-5)
    mha = build_gqa(
        cfg.GQAConfig(num_heads=4, num_kv_heads=4), D, index_port("read"), nnx.Rngs(0)
    )
    with pytest.raises(ValueError, match="4 kv heads cannot read an index"):
        mha(h, taps)


def test_msa_settings_that_do_not_go_together_say_so():
    small = {"num_heads": 2, "head_dim": 8}

    def indexer(**kw):
        return cfg.SparseIndexerConfig.model_validate(small | GROUPS | kw)

    with pytest.raises(ValueError, match="set weighted=false"):
        indexer(weighted=True)
    with pytest.raises(ValueError, match="takes pool=max"):
        indexer(block_size=4)
    with pytest.raises(ValueError, match="less than one block"):
        indexer(**MAX_BLOCKS, top_k=2)
    with pytest.raises(ValueError, match="num_heads=2 is not 4"):
        cfg.GQAConfig(num_heads=8, num_kv_heads=4, indexer=indexer(**MAX_BLOCKS))
    with pytest.raises(ValueError, match="which gqa has alone"):
        cfg.MLAConfig(num_heads=4, indexer=indexer(**MAX_BLOCKS))
    with pytest.raises(NotImplementedError, match="max-pooled blocks"):
        csa(compress_ratio=2, indexer=small | MAX_BLOCKS)


def test_glms_indexer_turns_its_rope_dims_in_interleaved_pairs():
    """GLM-5.3's scores as transformers' GlmMoeDsaIndexer makes them: it turns pairs of
    neighbouring dims, and lays them out evens first, which the scores do not see.
    DeepSeek-V3.2's half layout scores differently."""
    c = cfg.SparseIndexerConfig(
        num_heads=2, head_dim=8, top_k=3, rope_layout="interleaved"
    )
    attn = mla(c)
    ix = attn.indexer
    assert ix is not None and ix.k is not None and ix.k_norm is not None
    assert ix.weights is not None and attn.q_a is not None and attn.q_a_norm is not None
    x = inputs()
    query = attn.q_a_norm(attn.q_a(x))
    angles = jnp.arange(12)[:, None, None] * 10000.0 ** -(jnp.arange(0, 4, 2) / 4)
    cos, sin = jnp.cos(angles), jnp.sin(angles)

    def turn(a):  # the first 4 dims, which the layer's RoPE turns
        a1, a2 = a[..., 0:4:2], a[..., 1:4:2]
        return jnp.concatenate(
            [a1 * cos - a2 * sin, a2 * cos + a1 * sin, a[..., 4:]], -1
        )

    q = turn(ix.q(query).reshape(2, 12, 2, 8))
    k = turn(ix.k_norm(ix.k(x))[:, :, None])[:, :, 0]
    dots = jax.nn.relu(jnp.einsum("bthi,bsi->bths", q, k)) * 8**-0.5
    expected = jnp.einsum("bths,bth->bts", dots, ix.weights(x) * 2**-0.5)
    assert jnp.allclose(
        ix.scores(x, query, None, ix.keys(x, None)), expected, atol=1e-5
    )
    half = mla(c.model_copy(update={"rope_layout": "half"})).indexer
    assert half is not None
    scores = half.scores(x, query, None, half.keys(x, None))
    assert not jnp.allclose(scores, expected, atol=1e-3)


def test_a_layer_reading_index_attends_to_the_writers_selection():
    c = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=2)
    writer = gqa(index_port("write"), indexer=c)
    reader = gqa(index_port("read"))
    configured = gqa(index_port("read"), indexer=c)  # reading it, it builds none
    assert reader.indexer is None and configured.indexer is None
    assert num_params(configured) == num_params(reader) == num_params(gqa())
    x, taps = inputs(), Taps()
    writer(x, taps)
    selected = keeps(taps["topk"][0], causal(12))
    h = inputs(2)
    assert jnp.allclose(reader(h, taps), gqa_reference(reader, h, selected), atol=1e-5)
    dense = gqa_reference(reader, h, causal(12))
    assert not jnp.allclose(reader(h, taps), dense, atol=1e-3)


def test_ports_that_need_an_indexer_say_so():
    with pytest.raises(ValueError, match="'index' port needs an indexer"):
        gqa(index_port("write"))
    pool = (ir.Port(name="candidates", tap="pool", mode="write"),)
    c = cfg.SparseIndexerConfig(num_heads=2, head_dim=8)
    with pytest.raises(ValueError, match="'candidates' port needs an indexer"):
        gqa(index_port("read") + pool, indexer=c)
    with pytest.raises(NotImplementedError, match="block_size"):
        gqa(pool, indexer=c.model_copy(update={"block_size": 4}))
    with pytest.raises(ValueError, match="set compress_ratio"):
        CSA(cfg.CSAConfig(num_heads=2, head_dim=8), D, index_port("read"), nnx.Rngs(0))


def csa(ports=(), **kw):
    c = cfg.CSAConfig(
        num_heads=4,
        head_dim=16,
        q_lora_rank=12,
        o_lora_rank=6,
        o_groups=2,
        sinks=True,
        rope=cfg.RoPEConfig(fraction=0.25, layout="interleaved"),
    )
    return CSA(cfg.CSAConfig.model_validate(c.model_dump() | kw), D, ports, nnx.Rngs(0))


def csa_reference(attn, x, latents=None, visible=None, window=3):
    """Attention to a window of latents, then to `latents` where `visible` says, with
    the head outputs turned back by their query's position and grouped down."""
    b, t, _ = x.shape
    assert attn.q_a is not None and attn.q_a_norm is not None
    assert attn.rope is not None and attn.sinks is not None and attn.o_a is not None
    q = attn.q(attn.q_a_norm(attn.q_a(x))).reshape(b, t, 4, 16)
    q = rotate(q, attn.rope)
    kv = rotate(attn.kv_norm(attn.kv(x))[:, :, None], attn.rope)
    mask = causal(t, window)
    if latents is not None and visible is not None:
        kv = jnp.concatenate([kv, latents[:, :, None]], 1)
        mask = jnp.concatenate([mask, visible], -1)
    kv = jnp.repeat(kv, 4, axis=2)
    out = masked_attention(q, kv, kv, attn.scale, mask, attn.sinks[...])
    out = rotate(out, attn.rope, -jnp.arange(t)[None].repeat(b, 0))
    low = jnp.einsum("btgi,gir->btgr", out.reshape(b, t, 2, -1), attn.o_a[...])
    return attn.o(low.reshape(b, t, -1))


def test_csa_without_compression_is_window_attention():
    attn = csa(sliding_window_size=3)
    x = inputs()
    assert jnp.allclose(attn(x, Taps()), csa_reference(attn, x), atol=1e-5)


def test_the_latent_norm_is_every_latents_and_the_indexer_keys():
    """DeepSeek-V4.1 normalizes them all with eps 1e-20."""
    tiny = cfg.NormalizationConfig(type="rmsnorm", eps=1e-20)
    c = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=2)
    attn = csa(compress_ratio=2, indexer=c, latent_norm=tiny)
    assert attn.indexer is not None
    norms = (attn.q_a_norm, attn.kv_norm, attn.compress_norm, attn.indexer.k_norm)
    assert [n.eps if n else None for n in norms] == [1e-20] * 4


def test_compressed_latents_pool_their_groups():
    """Each 3 tokens pooled by a softmax over them per channel, normalized and rotated
    to their first token's position; a query sees the group from its last token on."""
    attn = csa(sliding_window_size=3, compress_ratio=3)
    assert attn.compress_kv is not None and attn.compress_gate is not None
    assert attn.compress_norm is not None and attn.rope is not None
    x = inputs()
    b, t, _ = x.shape
    kv = attn.compress_kv(x).reshape(b, 4, 3, 16)
    gate = jax.nn.softmax(attn.compress_gate(x).reshape(b, 4, 3, 16), axis=2)
    latents = attn.compress_norm((kv * gate).sum(2))
    first = jnp.arange(0, t, 3)[None].repeat(b, 0)
    latents = rotate(latents[:, :, None], attn.rope, first)[:, :, 0]
    visible = jnp.arange(t)[:, None] >= jnp.arange(2, t, 3)[None]
    visible = jnp.broadcast_to(visible, (b, t, 4))
    expected = csa_reference(attn, x, latents, visible)
    assert jnp.allclose(attn(x, Taps()), expected, atol=1e-5)


@pytest.mark.parametrize("ratio", [3, 1])
def test_compressed_latents_are_hidden_until_complete(ratio):
    """With a window of one token, a query only sees an earlier one through the latent
    of its group, which it sees once the group is complete."""
    c = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=2)
    attn = csa(sliding_window_size=1, compress_ratio=ratio, indexer=c)
    x = inputs()
    changed = attn(x.at[:, 3].multiply(-2.0), Taps())
    differs = (jnp.abs(changed - attn(x, Taps())) > 1e-6).any(-1).any(0)
    complete = 3 + ratio - 1 - 3 % ratio  # the last token of position 3's group
    assert differs[:3].tolist() == [False] * 3 and differs[3]
    assert not differs[4:complete].any() and differs[complete]


def test_reading_kv_reuses_the_latents_and_their_indexer_keys():
    c = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=2)
    write, read = (ir.Port(name="kv", tap="kv", mode=m) for m in ("write", "read"))
    writer = csa(compress_ratio=2, indexer=c, ports=(write,))
    reader = csa(compress_ratio=2, indexer=c, ports=(read,))
    assert reader.compress_kv is None and reader.compress_gate is None
    assert reader.compress_norm is None
    assert writer.indexer is not None and reader.indexer is not None
    assert reader.indexer.k is None  # its keys come with the latents
    # With the writer's parameters, which its compressor shifted, it is the writer.
    reader.o_a, reader.o = writer.o_a, writer.o
    reader.indexer.q, reader.indexer.weights = writer.indexer.q, writer.indexer.weights
    x, taps = inputs(), Taps()
    y = writer(x, taps)
    assert jnp.allclose(reader(x, taps), y, atol=1e-5)


def test_an_indexer_reading_candidates_selects_inside_them():
    pool = cfg.SparseIndexerConfig(
        num_heads=2, head_dim=8, top_k=4, candidate_block_size=2, candidate_top_k=2
    )
    kv = ir.Port(name="kv", tap="kv", mode="write")
    candidates = ir.Port(name="candidates", tap="pool", mode="write")
    source = csa(compress_ratio=1, indexer=pool, ports=(kv, candidates))
    reads = {"mode": "read"}
    second = csa(
        compress_ratio=1,
        indexer=pool,
        ports=(kv.model_copy(update=reads), candidates.model_copy(update=reads),
               ir.Port(name="index", tap="topk", mode="write")),
    )  # fmt: skip
    x, taps = inputs(), Taps()
    source(x, taps)
    second(inputs(2), taps)
    inside, selected = (keeps(taps[n][0], causal(12)) for n in ("pool", "topk"))
    blocks = inside.reshape(2, 12, 6, 2).any(-1)
    own = jnp.arange(12) // 2
    assert (blocks[:, jnp.arange(12), own]).all()  # the newest latent's block
    assert (blocks.sum(-1) == jnp.minimum(own + 1, 2)).all()
    assert (inside <= causal(12)).all() and (selected <= inside).all()
    assert (selected.sum(-1) == jnp.minimum(inside.sum(-1), 4)).all()


def kl(p, scores, over):
    """Every query's KL(p || softmax(scores)) over the keys `over` says, p normalized
    over them; 0 for a query with none."""
    p = jnp.where(over, p, 0)
    p = p / p.sum(-1, keepdims=True)
    log_q = jax.nn.log_softmax(jnp.where(over, scores, -jnp.inf))
    terms = jnp.where(over & (p > 0), p * (jnp.log(p) - log_q), 0)
    return jnp.where(over.any(-1), terms.sum(-1), 0)


@pytest.mark.parametrize("top_k", [3, 12])
def test_the_indexer_loss_is_the_kl_from_the_attention_to_its_top_k(top_k):
    """DeepSeek-V3.2's, eq. 4: KL(p || softmax(I)) over the keys kept, p the attention
    summed over heads and normalized over them, averaged over the queries. With every
    key kept, the attention is dense, and this is the warm-up's loss, eq. 3."""
    c = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=top_k, loss_weight=0.5)
    attn = gqa(index_port("write"), indexer=c)
    assert attn.indexer is not None
    x, taps = inputs(), Taps(training=True)
    attn(x, taps)
    [kept], [loss] = taps["topk"], taps.losses
    selected = keeps(kept, causal(12))
    assert (loss.name, loss.weight) == ("indexer_loss", 0.5)
    assert top_k < 12 or (selected == causal(12)).all()
    q, k, _ = gqa_heads(attn, x)
    p = attention_probs(q, k, attn.scale, selected).sum(1)
    scores = attn.indexer.scores(x, x, None, attn.indexer.keys(x, None))
    assert jnp.allclose(loss.value, kl(p, scores, selected).mean(), atol=1e-6)


def test_qsas_loss_is_over_the_blocks_kept_each_by_its_most_attended_key():
    """Qwen3.8 QSA's, eqs. 17 and 20: the attention summed over heads and max-pooled in
    each block, over the complete blocks kept but not the unfinished one."""
    c = cfg.SparseIndexerConfig(
        num_heads=2, head_dim=8, top_k=4, block_size=2, weighted=False
    )
    attn = gqa(index_port("write"), indexer=c)
    ix = attn.indexer
    assert ix is not None and ix.k is not None and ix.k_norm is not None
    x, taps = inputs(), Taps(training=True)
    attn(x, taps)
    [kept], [loss] = taps["topk"], taps.losses
    selected = keeps(kept, causal(12))
    q, k, _ = gqa_heads(attn, x)
    p = attention_probs(q, k, attn.scale, selected).sum(1)
    blocks = p.reshape(2, 12, 6, 2).max(-1)
    kept = selected.reshape(2, 12, 6, 2).all(-1)
    means = ix.k_norm(ix.k(x).reshape(2, 6, 2, 8).mean(2))
    first = jnp.arange(0, 12, 2)[None].repeat(2, 0)
    scores = ix.scores(x, x, None, ix._rotate(means[:, :, None], first)[:, :, 0])
    assert jnp.allclose(loss.value, kl(blocks, scores, kept).mean(), atol=1e-6)


@pytest.mark.parametrize("top_k", [4, 12])
def test_msas_loss_is_the_kl_of_each_group_over_the_keys_of_its_blocks(top_k):
    """MiniMax MSA's: for each query and group, KL(p || softmax(I)) over the keys the
    query sees of the blocks the group kept, p the mean of the group's heads'
    attention and I its head's dot products over sqrt(16); averaged over the queries
    and the groups. With every block kept, it is the dense warm-up's, over every key
    the query sees."""
    attn = msa(index_port("write"), top_k=top_k, block_size=2)
    assert attn.indexer is not None
    x, taps = inputs(), Taps(training=True)
    attn(x, taps)
    [kept], [loss] = taps["topk"], taps.losses
    selected = keeps(kept, causal(12))
    assert top_k < 12 or (selected == causal(12)[:, :, None]).all()
    q, k, _ = gqa_heads(attn, x)
    p = attention_probs(q, k, attn.scale, per_head(selected, 4))
    p = jnp.moveaxis(p.reshape(2, 2, 2, 12, 12).mean(2), 1, 2)
    scores = msa_scores(attn.indexer, x)
    assert jnp.allclose(loss.value, kl(p, scores, selected).mean(), atol=1e-6)


def test_the_language_modeling_loss_does_not_train_an_msa_indexer():
    """Its picks pass no gradient: only its own loss trains it."""
    m, t = model(BODIES["gqa_msa"]), tokens()

    def lm(m):
        [head], _ = m.training_heads(t)
        return head.logits().mean()

    grads = nnx.grad(lm)(m)
    for path, g in jax.tree_util.tree_leaves_with_path(grads):
        name = jax.tree_util.keystr(path)
        assert "['indexer']" not in name or not g.any(), name
    assert any(bool(g.any()) for g in jax.tree.leaves(grads))


def test_csas_loss_is_against_the_attention_its_window_and_sinks_share():
    """DeepSeek-V4's: the latents' part of the attention, whose softmax the window and
    the sinks are in too, and not an attention to the latents alone."""
    c = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=2)
    ports = tuple(ir.Port(name=n, tap=n, mode="write") for n in ("kv", "index"))
    # LeCun's weights, of attention far enough from uniform that the two part.
    with initializers.initializing(cfg.FanInitConfig(type="fan_in")):
        attn = csa(ports, compress_ratio=2, sliding_window_size=3, indexer=c)
    ix = attn.indexer
    assert ix is not None and attn.q_a is not None and attn.q_a_norm is not None
    assert attn.rope is not None and attn.sinks is not None
    x, taps = inputs(), Taps(training=True)
    attn(x, taps)
    [latents], [kept], [loss] = taps["kv"], taps["index"], taps.losses
    selected = keeps(kept, causal(12) & (latents.order >= 0)[:, None])
    query = attn.q_a_norm(attn.q_a(x))
    q = rotate(attn.q(query).reshape(2, 12, 4, 16), attn.rope)
    window = rotate(attn.kv_norm(attn.kv(x))[:, :, None], attn.rope)
    kv = jnp.repeat(jnp.concatenate([window, latents.kv[:, :, None]], 1), 4, axis=2)
    mask = jnp.concatenate([causal(12, 3), selected], -1)
    p = attention_probs(q, kv, attn.scale, mask, attn.sinks[...])[..., 12:].sum(1)
    scores = ix.scores(x, query, jnp.arange(12)[None].repeat(2, 0), latents.index)
    assert jnp.allclose(loss.value, kl(p, scores, selected).mean(), atol=1e-6)
    alone = attention_probs(q, kv[:, 12:], attn.scale, selected).sum(1)
    assert not jnp.allclose(loss.value, kl(alone, scores, selected).mean(), atol=1e-4)


@pytest.mark.parametrize("make", INDEXERS.values(), ids=INDEXERS.keys())
def test_the_indexer_loss_is_reported_in_training_forwards_only(make):
    c = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=6)
    layer, x, taps = make(c), inputs(), Taps()
    y = layer(x, taps)
    assert taps.losses == []
    taps = Taps(training=True)
    assert jnp.array_equal(layer(x, taps), y)
    [loss] = taps.losses
    assert loss.name == "indexer_loss" and loss.weight == 1.0
    assert bool(jnp.isfinite(loss.value)) and loss.value > 0
    taps = Taps(training=True)
    make(c.model_copy(update={"loss_weight": 0.0}))(x, taps)
    assert taps.losses == []


@pytest.mark.parametrize(
    "body",
    [
        "mla",
        "csa_window",
        "mla_indexer",
        "csa_compressed",
        "csa_ports",
        "mla_index_ports",
    ],
)
def test_the_layers_that_run_an_indexer_report_its_loss_and_no_others(body):
    """A layer that reads the index port runs none, and a model without an indexer
    reports nothing."""
    m, t = model(BODIES[body]), tokens()
    heads, reported = m.training_heads(t)
    indexers = sum(isinstance(i, Indexer) for _, i in nnx.iter_modules(m))
    assert [r.name for r in reported] == ["indexer_loss"] * indexers
    assert jnp.array_equal(heads[0].logits(), m(t))


def indexer_loss(m, t, positions=None):
    """The indexer losses of a training forward of `m`, summed."""
    _, reported = m.training_heads(t, positions)
    return sum(r.value for r in reported if r.name == "indexer_loss")


@pytest.mark.parametrize(
    "body",
    ["mla_indexer", "gqa_block_indexer", "gqa_msa", "csa_ports", "mla_index_ports"],
)
def test_the_indexer_loss_trains_the_indexers_and_nothing_else(body):
    """Their inputs are detached, and so is the attention they learn from: none of its
    gradient reaches the attention's projections, the MoE, the norms or the embedding.
    An indexer that scores another's keys (DeepSeek-V4.1's Reindex) trains them too."""
    body = BODIES[body]
    m, t = model(body.replace("[norm ffn]", "[norm experts]")), tokens()
    moe = any(isinstance(e, MoE) for _, e in nnx.iter_modules(m))
    assert moe == ("[norm ffn]" in body)
    grads = nnx.grad(lambda m: indexer_loss(m, t))(m)
    for path, g in jax.tree_util.tree_leaves_with_path(grads):
        name = jax.tree_util.keystr(path)
        trained = "['indexer']" in name
        assert bool(g.any()) == trained and bool(jnp.isfinite(g).all()), name


def test_training_on_the_indexer_loss_alone_makes_it_keep_what_attention_attends_to():
    """How many of the 4 keys that the attention attends to most it scores best."""
    c = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=4)
    attn, x = gqa(indexer=c), inputs(length=32)
    ix, mask = attn.indexer, causal(32)
    assert ix is not None

    def agreement():
        q, k, _ = gqa_heads(attn, x)
        p = attention_probs(q, k, attn.scale, mask).sum(1)
        scores = jnp.where(mask, ix.scores(x, x, None, ix.keys(x, None)), -jnp.inf)
        best = [jax.nn.one_hot(jax.lax.top_k(a, 4)[1], 32).sum(-2) for a in (p, scores)]
        return float((best[0] * best[1]).sum(-1)[:, 4:].mean() / 4)

    def loss(attn):
        taps = Taps(training=True)
        attn(x, taps)
        return taps.losses[0].value

    @nnx.jit
    def step(attn, optimizer):
        value, grads = nnx.value_and_grad(loss)(attn)
        optimizer.update(attn, grads)
        return value

    optimizer = nnx.Optimizer(attn, optax.adam(1e-2), wrt=nnx.Param)
    before, first = agreement(), step(attn, optimizer)
    for _ in range(100):
        last = step(attn, optimizer)
    assert last < first / 2 and agreement() > before + 0.1


M3 = "examples/architectures/minimax/minimax_m3.bux"
M3_SHRUNK = [
    f"d_model = {D}",
    f"vocab_size = {V}",
    (
        "attn = gqa(num_heads=4, num_kv_heads=2, head_dim=8, qk_norm=norm,"
        " rope=rope(fraction=0.5))"
    ),
    (
        "msa = attn(indexer=indexer(num_heads=2, head_dim=8, top_k=TOP_K, block_size=2,"
        " pool=max, per_group=true, relu=false, weighted=false, norm=norm))"
    ),
    (
        "dense = gated_mlp(intermediate_size=8, activation=quick_gelu, limit=7.0,"
        " up_offset=1.0)"
    ),
    (
        "experts = moe(num_experts=4, top_k=2, num_shared_experts=1, router=sigmoid,"
        " routed_scaling_factor=2.0, load_balancing=aux_free, expert=dense)"
    ),
]


def test_minimax_m3_trains_through_its_dense_warm_up_and_then_sparse():
    """MiniMax-M3, shrunk, in its two stages: its MSA layers keep every block of the
    12 keys a row has, then 2 blocks of 2 keys, from the warm-up's parameters. The
    stages differ in top_k alone, which no parameter depends on: the indexer learns
    in the warm-up, and the sparse stage attends otherwise and trains on."""
    path = Path(__file__).parents[1] / M3

    def stage(top_k: int, state: nnx.State | None = None):
        overrides = [s.replace("TOP_K", str(top_k)) for s in M3_SHRUNK]
        arch = compile(path.read_text(), overrides=overrides)
        m = Model(arch, rngs=nnx.Rngs(0))
        if state is not None:
            nnx.update(m, state)
        return m, Trainer(m, optax.adam(1e-2), objective(tcfg.LossConfig(), arch))

    t = tokens()
    batch = Batch(np.asarray(t), np.tile(np.arange(12), (2, 1)), np.ones((2, 12), bool))
    _, trainer = stage(12)
    first = trainer.train_on_batch(batch)
    for _ in range(8):
        last = trainer.train_on_batch(batch)
    assert last["indexer_loss"] < first["indexer_loss"]
    state = nnx.state(trainer.model)
    sparse_m, trainer = stage(4, state)
    assert jax.tree.structure(nnx.state(sparse_m)) == jax.tree.structure(state)
    assert not jnp.allclose(sparse_m(t), stage(12, state)[0](t), atol=1e-4)
    metrics = [trainer.train_on_batch(batch) for _ in range(3)]
    assert all(
        np.isfinite(m["indexer_loss"]) and m["indexer_loss"] > 0 for m in metrics
    )
    assert all(np.isfinite(m["grad_norm"]) for m in metrics)


@pytest.mark.parametrize(
    "body",
    ["mla_indexer", "gqa_block_indexer", "gqa_msa", "csa_compressed", "csa_ports"],
)
def test_the_indexer_loss_of_documents_packed_in_a_row_is_theirs(body):
    """Each document's queries' share of the mean, also where a query sees nothing the
    indexer scores: before CSA's first latent, or QSA's first complete block. A train
    step logs it, with a finite gradient."""
    m = model(BODIES[body])
    a, b = tokens(0, 5), tokens(1, 7)
    t = jnp.concatenate([a, b], axis=1)
    positions = jnp.concatenate([jnp.arange(5), jnp.arange(7)])[None].repeat(2, 0)
    packed = indexer_loss(m, t, positions)
    alone = (5 * indexer_loss(m, a) + 7 * indexer_loss(m, b)) / 12
    assert bool(jnp.isfinite(packed)) and jnp.isclose(packed, alone, atol=1e-6)
    arch = compile(PRELUDE + BODIES[body])
    trainer = Trainer(m, optax.adamw(1e-3), objective(tcfg.LossConfig(), arch))
    batch = Batch(np.asarray(t), np.asarray(positions), np.ones((2, 12), bool))
    metrics = trainer.train_on_batch(batch)
    assert np.isclose(metrics["indexer_loss"], packed, atol=1e-5)
    assert np.isfinite(metrics["grad_norm"])


def test_unrotate_inverts_rotate_with_a_gain_on_cos_and_sin():
    yarn = cfg.YaRNConfig(factor=16, original_max_len=4, attention_factor=1.5)
    attn = csa(rope=cfg.RoPEConfig(fraction=0.5, scaling=yarn))
    x = jax.random.normal(jax.random.key(0), (2, 12, 4, 16))
    pos = jnp.arange(12)[None].repeat(2, 0) * 7
    assert jnp.allclose(attn._unrotate(attn._rotate(x, pos), pos), x, atol=1e-5)


def test_dropout_on_the_probabilities_and_the_output():
    dropped = gqa(attention_dropout=0.5, output_dropout=0.5)
    x = inputs()
    assert not jnp.allclose(dropped(x, Taps()), dropped(x, Taps()))
    dropped.eval()
    assert jnp.allclose(dropped(x, Taps()), gqa()(x, Taps()), atol=1e-6)


@pytest.mark.parametrize(
    ("kind", "learned", "weights"),
    [
        ("fixed", None, (0.5, 0.5)),
        ("learned", [0.3, 2.0], (0.3, 2.0)),
        ("convex", 0.3, (0.3, 0.7)),
    ],
)
def test_a_value_residual_mixes_the_taps_values_into_the_layers(kind, learned, weights):
    """a v + b v1: half and half (ResFormer's code), both weights learned from 0.5
    (Learnable-ResFormer), or b = 1 - a with a learned."""
    v1 = ir.Port(name="v", tap="v1", mode="read")
    kv = ir.Port(name="kv", tap="kv", mode="write")
    attn = gqa((v1, kv), value_residual=kind)
    assert attn.v is not None
    if learned is None:
        assert attn.v_mix is None
    else:
        assert attn.v_mix is not None and (attn.v_mix[...] == 0.5).all()
        attn.v_mix[...] = jnp.asarray(learned)
    x = inputs()
    first = jax.random.normal(jax.random.key(2), (2, 12, 2, 8))
    taps = Taps({"v1": [first]})
    attn(x, taps)
    [(_, v)] = taps["kv"]
    a, b = weights
    own = attn.v(x).reshape(2, 12, 2, 8)
    assert jnp.allclose(v, a * own + b * first, atol=1e-6)


VE_READ = ir.Port(name="ve", tap="ve1", mode="read")
KV_WRITE = ir.Port(name="kv", tap="kv", mode="write")


@pytest.mark.parametrize("gate", [6, None], ids=["gated", "plain"])
def test_value_embeddings_add_to_the_values_by_a_gate_that_starts_at_one(gate):
    """v + 2 sigmoid(W [x[:6], e[:6]]) e, W at 0 (modded-nanogpt), or v + e."""
    attn = gqa((VE_READ, KV_WRITE), value_embed_gate=gate)
    assert attn.v is not None
    x, e = inputs(), jax.random.normal(jax.random.key(2), (2, 12, 16))
    taps = Taps({"ve1": [e]})
    attn(x, taps)
    [(_, v)] = taps["kv"]
    own = attn.v(x).reshape(2, 12, 2, 8)
    assert jnp.allclose(v, own + e.reshape(2, 12, 2, 8), atol=1e-6)
    if gate is None:
        assert attn.ve_gate is None
        return
    assert attn.ve_gate is not None and attn.ve_gate.kernel.shape == (12, 2)
    w = jax.random.normal(jax.random.key(3), (12, 2))
    attn.ve_gate.kernel[...] = w
    taps = Taps({"ve1": [e]})
    attn(x, taps)
    [(_, v)] = taps["kv"]
    g = 2 * jax.nn.sigmoid(jnp.concatenate([x[..., :6], e[..., :6]], -1) @ w)
    want = own + g[..., None] * e.reshape(2, 12, 2, 8)
    assert jnp.allclose(v, want, atol=1e-5)


def test_value_embeddings_go_in_before_the_value_residual():
    v1 = ir.Port(name="v", tap="v1", mode="read")
    attn = gqa((VE_READ, v1, KV_WRITE), value_embed_gate=None)
    x, e = inputs(), jax.random.normal(jax.random.key(2), (2, 12, 16))
    first = jax.random.normal(jax.random.key(4), (2, 12, 2, 8))
    taps = Taps({"ve1": [e], "v1": [first]})
    attn(x, taps)
    [(_, v)] = taps["kv"]
    assert attn.v is not None
    own = attn.v(x).reshape(2, 12, 2, 8) + e.reshape(2, 12, 2, 8)
    assert jnp.allclose(v, 0.5 * own + 0.5 * first, atol=1e-6)


def test_a_value_embedding_pushes_the_rows_of_the_input_tokens():
    c = cfg.ValueEmbeddingConfig(features=16)
    write = ir.Port(name="ve", tap="t", mode="write")
    table = ValueEmbedding(c, V, (write,), nnx.Rngs(0))
    rows = table.table.embedding[...]
    assert rows.shape == (V, 16)
    assert jnp.allclose(rows.std(), 0.01, rtol=0.1)  # modded-nanogpt's init
    ids, x = tokens(), inputs()
    taps = Taps(tokens=ids)
    assert table(x, taps) is x
    assert jnp.array_equal(taps["t"][0], rows[ids])
    m = model(
        "ve = value_embed(features=16)\n"
        "a = gqa(num_heads=4, num_kv_heads=2)\n"
        "model = embed ve(ve=@t) [norm a(ve=$t)] [norm ve(ve=@u) a(ve=$u!)] norm head"
    )
    assert m(ids).shape == (2, 12, V)


def test_value_embeddings_must_match_the_kv_heads_and_have_values_to_add_to():
    narrow = "ve = value_embed(features=8)\na = gqa(num_heads=4, num_kv_heads=2)\n"
    m = model(narrow + "model = embed ve(ve=@t) [norm a(ve=$t)] norm head")
    with pytest.raises(ValueError, match="features=16"):
        m(tokens())
    with pytest.raises(ValueError, match="reads kv"):
        gqa((VE_READ, ir.Port(name="kv", tap="kv", mode="read")))


def test_value_embeddings_pass_a_widened_stream_through():
    arch = compile(
        PRELUDE + "residual = hc(expansion=2)\n"
        "ve = value_embed(features=16)\n"
        "a = gqa(num_heads=4, num_kv_heads=2)\n"
        "model = embed ve(ve=@t) [norm a(ve=$t)] [norm ffn] norm head"
    )
    first = list(ir.unrolled_layers(arch.graph))[1]
    assert first.config.type == "value_embed" and first.width == "streams"
    assert Model(arch, rngs=nnx.Rngs(0))(tokens()).shape == (2, 12, V)


KV_WRITE = ir.Port(name="kv", tap="kv", mode="write")


def rms(x, eps):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps)


@pytest.mark.parametrize("conv", [None, 3], ids=["plain", "conv"])
@pytest.mark.parametrize("fused", [False, True], ids=["split", "fused"])
def test_values_from_the_keys_are_each_heads_key_rms_normalized(conv, fused):
    """Gemma 4's K=V: v = rmsnorm(W_k x) per head, with no gain and no RoPE, at the
    key norm's eps; the keys go on through their norm and RoPE."""
    kw = {
        "qk_norm": cfg.NormalizationConfig(type="rmsnorm", eps=1e-5),
        "kv_conv_size": conv,
    }
    attn = gqa((KV_WRITE,), k_eq_v=True, fused_qkv=fused, **kw)
    assert attn.v is None and attn.v_conv is None and attn.v_norm is not None
    x = inputs()
    taps = Taps()
    attn(x, taps)
    [(k, v)] = taps["kv"]
    raw = attn.qkv(x)[..., 32:] if attn.qkv is not None else attn.k(x)
    assert raw.shape == (2, 12, 16)
    if attn.k_conv is not None:
        raw = raw + attn.k_conv(raw, Taps())
    heads = raw.reshape(2, 12, 2, 8)
    assert jnp.allclose(v, rms(heads, 1e-5), atol=1e-5)
    assert attn.k_norm is not None and attn.rope is not None
    assert jnp.allclose(k, rotate(attn.k_norm(heads), attn.rope), atol=1e-5)


def test_values_from_the_keys_take_no_parameters_and_default_off_changes_nothing():
    plain, same = gqa(), gqa(k_eq_v=False)
    shared = gqa(k_eq_v=True)
    params = lambda m: jax.tree.leaves(nnx.state(m, nnx.Param))
    assert all(jnp.array_equal(a, b) for a, b in zip(params(plain), params(same)))
    assert num_params(plain) - num_params(shared) == 32 * 16  # no v projection
    x = inputs()
    assert jnp.array_equal(plain(x, Taps()), same(x, Taps()))
    with pytest.raises(ValueError, match="v_head_dim"):
        cfg.GQAConfig(
            num_heads=4, num_kv_heads=2, head_dim=8, v_head_dim=4, k_eq_v=True
        )


# --- Split over devices (bux.parallel) -------------------------------------------------

ONE = "mesh(data=1)"
SPLIT = {
    "gqa": "gqa(num_heads=4, num_kv_heads=2, rope=rope())",
    "mqa": "gqa(num_heads=4, num_kv_heads=1, rope=rope())",
    "fused_qkv": "gqa(num_heads=4, num_kv_heads=2, rope=rope(), fused_qkv=true, bias=true)",
    "gated_sinks": (
        "gqa(num_heads=4, num_kv_heads=2, rope=rope(), gated=true, sinks=true,"
        " kv_conv_size=3, relative_bias=relbias(rank=2, extent=4))"
    ),
    # Beam's global layers: queries scaled by their position, a gate bias per head
    "power_scale": (
        "gqa(num_heads=4, num_kv_heads=2, gated=true, gate_head_bias=true,"
        " power_scale=0.5, power_scale_length=2)"
    ),
    # XSA, learned: each query head takes its group's value, which on tensor=4 stays
    # whole
    "xsa": "gqa(num_heads=4, num_kv_heads=2, gated=true, exclusive=learned)",
    # HyGA: per-head matrices of the gate, and a gate of each head from all of them
    "hyga": (
        "gqa(num_heads=4, num_kv_heads=2, gated=true, gate_input_rank=4,"
        " gate_output_rank=4, gate_cross_head=true)"
    ),
    # MiniMax-M2's QK-norm of all heads together, whose mean square spans the devices
    "qk_norm_all_heads": (
        "gqa(num_heads=4, num_kv_heads=2, qk_norm=norm, qk_norm_all_heads=true,"
        " rope=rope(fraction=0.5))"
    ),
    "mla": "latent",
    "mla_indexer": "latent(indexer=small)",
    # On tensor=4 MSA's 2 groups stay whole, and its picks are each query head's
    "msa": (
        "gqa(num_heads=4, num_kv_heads=2, rope=rope(), indexer=small(top_k=4,"
        " block_size=2, pool=max, per_group=true, relu=false, weighted=false))"
    ),
    "csa": "window(compress_ratio=2, indexer=small(top_k=2))",
}
# A mesh and a layout each. On tensor=4, GQA's 2 kv heads, MQA's one, the indexer's 2
# heads and CSA's 2 output groups do not split, and stay whole.
BY_HEAD = [
    ("mesh(fsdp=4, tensor=2)", "layout()"),
    ("mesh(fsdp=2, tensor=4)", "layout()"),
]
BY_POSITION = [
    ("mesh(fsdp=2, tensor=4)", "layout(stream=tensor)"),  # sequence parallelism
    ("mesh(context=2, tensor=2)", "layout()"),  # all-gather context parallelism
]


def split(layer: str) -> str:
    return PRELUDE + f"a = {layer}\nmodel = embed [norm a] [norm ffn] norm head"


def trains_the_same(losses, involuntary, layer: str, splits: list) -> None:
    """Assert that `layer` trains as on one device, and that XLA rematerializes no
    array in full, split as each of `splits` says."""
    one = losses(split(layer), mesh=ONE)
    for mesh, layout in splits:
        involuntary()
        split_losses = losses(split(layer), mesh=mesh, layout=layout)
        assert np.allclose(split_losses, one, atol=1e-5), (mesh, layout)
        assert involuntary() == 0, (mesh, layout)


@pytest.mark.parametrize("layer", SPLIT.values(), ids=SPLIT.keys())
def test_attention_split_by_head_trains_the_same_and_rematerializes_nothing(
    losses, involuntary, layer
):
    trains_the_same(losses, involuntary, layer, BY_HEAD)


@pytest.mark.parametrize(
    "layer", ["gqa", "power_scale", "xsa", "hyga", "mla", "msa", "csa"]
)
def test_attention_trains_the_same_with_the_positions_split(losses, involuntary, layer):
    trains_the_same(losses, involuntary, SPLIT[layer], BY_POSITION)


def test_the_indexers_loss_is_the_same_split_by_head(tmp_path):
    def indexer_loss(mesh: str) -> float:
        settings = "data = synthetic\noptimizer = adamw(lr=1e-2)\nsteps = 3\n"
        settings += "seq_len = 16\nbatch = 8\n"
        settings += "precision = precision(compute=float32)\n"  # splits to 1e-5
        [run] = compile_runs(f"{split(SPLIT['mla_indexer'])}\n{settings}mesh = {mesh}")
        return train(run, tmp_path / mesh, echo=lambda _: None)["indexer_loss"]

    one = indexer_loss(ONE)
    assert indexer_loss("mesh(fsdp=4, tensor=2)") == pytest.approx(one, abs=1e-5)


@pytest.mark.parametrize(("tensor", "kv"), [(2, P(None, "tensor")), (4, P(None, None))])
def test_kv_heads_split_only_between_heads(tensor, kv):
    """2 kv heads of 8 dims are 16 columns, which split 4 ways; the heads do not."""
    c = cfg.GQAConfig(num_heads=4, num_kv_heads=2)
    with parallel.use(tcfg.MeshConfig(tensor=tensor)):
        state = nnx.state(parallel.build(lambda: Attention(c, D, (), nnx.Rngs(0))))
        specs = parallel.shardings(state)
    assert specs["q"]["kernel"].spec == P(None, "tensor")
    assert specs["o"]["kernel"].spec == P("tensor", None)
    assert specs["k"]["kernel"].spec == specs["v"]["kernel"].spec == kv


def test_the_indexers_top_k_keeps_the_first_of_equal_scores_and_splits_by_row():
    scores = jnp.asarray([[[1.0, 2.0, 2.0, 2.0, -jnp.inf]]])
    assert _top(scores, 2).tolist() == [[[1, 2]]]
    assert _top(scores[..., -2:], 2).tolist() == [[[0, -1]]]
    with parallel.use(tcfg.MeshConfig(data=8)) as mesh:
        rows = jax.device_put(jnp.ones((8, 4, 6)), NamedSharding(mesh, P("data")))
        compiled = jax.jit(lambda s: _top(s, 3)).lower(rows).compile().as_text()
    assert compiled is not None and "all-gather" not in compiled
