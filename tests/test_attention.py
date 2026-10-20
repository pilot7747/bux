"""Softmax attention layers: MLA, CSA, the sparse attention indexer, and what they share
with GQA (tests/test_backend.py has GQA's own)."""

import math

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from bux import config as cfg
from bux.architecture import SEGMENTS, Attention, Model, Segments, num_params
from bux.architecture.csa import CSA
from bux.architecture.mla import MLA
from bux.architecture.rope import rotate
from bux.lang import compile, ir

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


def masked_attention(q, k, v, scale, mask, sinks=None):
    """softmax(q . k * scale) v over the keys `mask` keeps, next to the heads' sinks."""
    logits = jnp.einsum("bthd,bshd->bhts", q, k) * scale
    logits = jnp.where(mask[:, None], logits, -jnp.inf)
    if sinks is not None:
        sink = jnp.broadcast_to(sinks[None, :, None, None], (*logits.shape[:3], 1))
        logits = jnp.concatenate([logits, sink], -1)
    probs = jax.nn.softmax(logits)[..., : k.shape[1]]
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
    ["mla_indexer", "gqa_block_indexer", "csa_compressed", "csa_ports"],
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
    assert jnp.allclose(mla(x, {}), mla.o(out.reshape(b, t, -1)), atol=1e-5)


def test_mla_without_rope_sees_no_positions():
    c = cfg.MLAConfig(num_heads=2, kv_lora_rank=8, rope=None)
    mla = MLA(c, D, (), nnx.Rngs(0))
    x = inputs()
    shifted = {SEGMENTS: [Segments.of(jnp.arange(12)[None].repeat(2, 0) + 100)]}
    assert jnp.allclose(mla(x, shifted), mla(x, {}), atol=1e-5)


def gqa(ports=(), **kw):
    c = cfg.GQAConfig(num_heads=4, num_kv_heads=2, rope=cfg.RoPEConfig(), **kw)
    return Attention(c, D, ports, nnx.Rngs(0))


def gqa_reference(attn, x, mask):
    b, t, _ = x.shape
    h, g, d = attn.heads, attn.kv_heads, attn.head_dim
    assert attn.q is not None and attn.k is not None and attn.v is not None
    assert attn.rope is not None
    q = rotate(attn.q(x).reshape(b, t, h, d), attn.rope)
    k = rotate(attn.k(x).reshape(b, t, g, d), attn.rope)
    v = attn.v(x).reshape(b, t, g, d)
    k, v = (jnp.repeat(a, h // g, axis=2) for a in (k, v))
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
    return MLA(c, D, (), nnx.Rngs(0))


INDEXERS = {
    "gqa": lambda indexer: gqa(indexer=indexer),
    "gqa_blocks": lambda indexer: gqa(
        indexer=indexer and indexer.model_copy(update={"block_size": 3})
    ),
    "mla": mla,
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
    assert jnp.allclose(make(every)(x, {}), make(None)(x, {}), atol=1e-5)
    few = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=3)
    assert not jnp.allclose(make(few)(x, {}), make(None)(x, {}), atol=1e-3)


@pytest.mark.parametrize("weighted", [True, False])
def test_a_small_top_k_attends_to_the_best_scored_keys_only(weighted):
    c = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=3, weighted=weighted)
    attn = gqa(index_port("write"), indexer=c)
    assert attn.indexer is not None
    x, taps = inputs(), {}
    y = attn(x, taps)
    [selected] = taps["topk"]
    # The best 3 of the keys each query sees, by the indexer's scores
    scores = attn.indexer.scores(x, x, None, attn.indexer.keys(x, None))
    scores = jnp.where(causal(12), scores, -jnp.inf)
    best = jnp.argsort(-scores, axis=-1)[..., :3]
    expected = jax.nn.one_hot(best, 12).sum(-2).astype(bool) & causal(12)
    assert (selected == expected).all()
    assert (selected.sum(-1) == jnp.minimum(jnp.arange(12) + 1, 3)).all()
    assert jnp.allclose(y, gqa_reference(attn, x, selected), atol=1e-5)


def test_blocks_are_kept_whole_with_the_querys_own():
    c = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=4, block_size=2)
    attn = gqa(index_port("write"), indexer=c)
    x, taps = inputs(), {}
    y = attn(x, taps)
    [selected] = taps["topk"]
    pairs = selected.reshape(2, 12, 6, 2)
    own = jnp.arange(12) // 2
    # Each query keeps its own block, up to itself, and 2 whole earlier ones.
    assert pairs[:, jnp.arange(12), own].sum(-1).tolist() == [[1, 2] * 6] * 2
    earlier = pairs.all(-1) & (jnp.arange(6) < own[:, None])
    assert (earlier.sum(-1) == jnp.minimum(own, 2)).all()
    assert (pairs.any(-1) == pairs.all(-1) | (jnp.arange(6) == own[:, None])).all()
    assert jnp.allclose(y, gqa_reference(attn, x, selected), atol=1e-5)


def test_a_layer_reading_index_attends_to_the_writers_selection():
    c = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=2)
    writer = gqa(index_port("write"), indexer=c)
    reader = gqa(index_port("read"))
    configured = gqa(index_port("read"), indexer=c)  # reading it, it builds none
    assert reader.indexer is None and configured.indexer is None
    assert num_params(configured) == num_params(reader) == num_params(gqa())
    x, taps = inputs(), {}
    writer(x, taps)
    [selected] = taps["topk"]
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
    assert jnp.allclose(attn(x, {}), csa_reference(attn, x), atol=1e-5)


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
    assert jnp.allclose(attn(x, {}), expected, atol=1e-5)


@pytest.mark.parametrize("ratio", [3, 1])
def test_compressed_latents_are_hidden_until_complete(ratio):
    """With a window of one token, a query only sees an earlier one through the latent
    of its group, which it sees once the group is complete."""
    c = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=2)
    attn = csa(sliding_window_size=1, compress_ratio=ratio, indexer=c)
    x = inputs()
    changed = attn(x.at[:, 3].multiply(-2.0), {})
    differs = (jnp.abs(changed - attn(x, {})) > 1e-6).any(-1).any(0)
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
    x, taps = inputs(), {}
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
    x, taps = inputs(), {}
    source(x, taps)
    second(inputs(2), taps)
    [inside], [selected] = taps["pool"], taps["topk"]
    blocks = inside.reshape(2, 12, 6, 2).any(-1)
    own = jnp.arange(12) // 2
    assert (blocks[:, jnp.arange(12), own]).all()  # the newest latent's block
    assert (blocks.sum(-1) == jnp.minimum(own + 1, 2)).all()
    assert (inside <= causal(12)).all() and (selected <= inside).all()
    assert (selected.sum(-1) == jnp.minimum(inside.sum(-1), 4)).all()


def test_unrotate_inverts_rotate_with_a_gain_on_cos_and_sin():
    yarn = cfg.YaRNConfig(factor=16, original_max_len=4, attention_factor=1.5)
    attn = csa(rope=cfg.RoPEConfig(fraction=0.5, scaling=yarn))
    x = jax.random.normal(jax.random.key(0), (2, 12, 4, 16))
    pos = jnp.arange(12)[None].repeat(2, 0) * 7
    assert jnp.allclose(attn._unrotate(attn._rotate(x, pos), pos), x, atol=1e-5)


def test_dropout_on_the_probabilities_and_the_output():
    dropped = gqa(attention_dropout=0.5, output_dropout=0.5)
    x = inputs()
    assert not jnp.allclose(dropped(x, {}), dropped(x, {}))
    dropped.eval()
    assert jnp.allclose(dropped(x, {}), gqa()(x, {}), atol=1e-6)
