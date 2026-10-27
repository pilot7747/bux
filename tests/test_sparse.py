"""Attention over kept keys (bux.architecture.attention.sparse) by XLA, a chunk of
queries at a time, against all t by t logits: values, gradients and the indexer's loss,
with packed documents, on one device and split over the 8 CPU devices the tests have;
and memory that grows linearly with the queries; the indexer's routes to cuDNN's
kernels, which lax and XLA stand in for here. scripts/gpu_smoke/sparse.py checks
tokamax's kernel and cuDNN's against it on a GPU."""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux import kernels, parallel
from bux.architecture import Model, Segments, Taps
from bux.architecture.attention import indexer_cudnn, scoring, sparse, sparse_cudnn
from bux.architecture.attention.gqa import Sparse, build_gqa
from bux.architecture.attention.indexer import _first, _top
from bux.lang import compile
from bux.parallel.context import masked_softmax, scaled_logits, softcap, weighted_values
from bux.train import config as tcfg
from bux.train import flops

B, T, H, D = 2, 16, 4, 8
# Three documents in the first row, cut mid-way, and one in the second.
POSITIONS = jnp.asarray([[*range(3, 8), *range(7), *range(4)], [*range(T)]])
ROWS = Segments.of(POSITIONS)
TINY = 1 << 9  # sparse.ELEMENTS for chunks of 2 to 8 queries


def kept(
    seed: int, span: sparse.Span, top_k: int, size: int = 1, groups: int | None = None
) -> sparse.Kept:
    """Each query's top_k random slots of its span, or blocks of `size` slots; for
    each of `groups` groups apart, if given."""
    shape = (B, T) if groups is None else (B, T, groups)
    scores = jax.random.uniform(jax.random.key(seed), (*shape, T // size))
    if size == 1:
        return sparse.Kept(_top(jnp.where(sparse.mask(span, T), scores, -1), top_k))
    block = jnp.broadcast_to(jnp.arange(T) // size, (B, T))
    return sparse.Kept(_top(scores, top_k), block)


def heads(
    seed: int, g: int, dv: int = D
) -> tuple[Float[Array, "b s g dk"], Float[Array, "b s g dv"]]:
    """Random keys and values of g heads."""
    key = jax.random.split(jax.random.key(seed))
    return (
        jax.random.normal(key[0], (B, T, g, D)),
        jax.random.normal(key[1], (B, T, g, dv)),
    )


def case(name: str) -> tuple[list[sparse.Part], bool, float | None]:
    """The parts of a case, whether it has sinks, and its soft cap."""
    tokens, window = sparse.token_span(ROWS, None), sparse.token_span(ROWS, 6)
    if name == "csa":  # MQA: a window, then latents every other slot, top 3 of them
        k, v = heads(1, 1)
        latents = heads(2, 1)[0]
        exists = jnp.broadcast_to(jnp.arange(T) % 2 == 1, (B, T))
        some = sparse.Part(latents, latents, tokens, exists, kept(3, tokens, 3))
        return [sparse.Part(k, v, sparse.token_span(ROWS, 3)), some], True, 5.0
    g, span = (2, window) if name == "window" else (2, tokens)
    k, v = heads(1, g, D + 2)
    chosen = {
        "all": None,
        "tokens": kept(3, span, 5),
        "window": kept(3, span, 4),
        "blocks": kept(3, span, 2, size=3),
        "groups": kept(3, span, 2, size=4, groups=g),  # MiniMax MSA's
    }[name]
    return [sparse.Part(k, v, span, None, chosen)], name == "window", None


CASES = ["all", "tokens", "window", "blocks", "groups", "csa"]
SCALE = D**-0.5


def reference(q, parts, sinks, cap):
    """All t by t logits, as the `xla` method takes them."""
    k, v = (jnp.concatenate(kv, axis=1) for kv in zip(*((p.k, p.v) for p in parts)))
    mask = jnp.concatenate([sparse.visible(p) for p in parts], -1)
    if mask.ndim == 4:  # each group's, for its query heads
        mask = jnp.repeat(jnp.moveaxis(mask, 2, 1), H // mask.shape[2], axis=1)
    else:
        mask = mask[:, None]
    logits = softcap(scaled_logits(q, k, SCALE), cap)
    return weighted_values(masked_softmax(logits, mask, sinks), v)


def run(f, name: str):
    """f's output on a case, and its gradients for q, every part's k and v, and the
    sinks."""
    parts, has_sinks, cap = case(name)
    key = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(key[0], (B, T, H, D))
    sinks = jax.random.normal(key[1], (H,)) if has_sinks else None
    keys = [(p.k, p.v) for p in parts]

    def loss(q, keys, sinks):
        given = [p._replace(k=k, v=v) for p, (k, v) in zip(parts, keys)]
        out = f(q, given, sinks, cap)
        cotangent = jax.random.normal(key[2], out.shape)
        return (out * cotangent).sum(), out

    return jax.jit(jax.grad(loss, argnums=(0, 1, 2), has_aux=True))(q, keys, sinks)


def kernel(q, parts, sinks, cap):
    return sparse.attend(q, parts, sinks, SCALE, cap)[0]


def close(a, b) -> None:
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("elements", [sparse.ELEMENTS, TINY], ids=["whole", "chunks"])
@pytest.mark.parametrize("name", CASES)
def test_attention_over_kept_keys_is_all_logits_attention(monkeypatch, name, elements):
    monkeypatch.setattr(sparse, "ELEMENTS", elements)
    grads, out = run(kernel, name)
    want_grads, want = run(reference, name)
    close(out, want)
    close(grads, want_grads)


def test_tokamaxs_run_a_group_at_a_time_attends_as_one_run(monkeypatch):
    """On a GPU, picks for each group run tokamax's kernels a group at a time, so
    that each mask is a group's; here XLA's runs stand in for the kernels."""
    monkeypatch.setattr(sparse, "ELEMENTS", TINY)
    want_grads, want = run(kernel, "groups")
    monkeypatch.setattr(sparse, "_flash", lambda q: True)
    monkeypatch.setattr(sparse, "_flash_fwd", sparse._xla_fwd)
    monkeypatch.setattr(sparse, "_flash_bwd", sparse._xla_bwd)
    grads, out = run(kernel, "groups")
    close(out, want)
    close(grads, want_grads)


@pytest.mark.parametrize("dims", [(128, 128), (192, 128), (256, 256)])
def test_tokamax_takes_the_heads_dense_attention_gives_it(monkeypatch, dims):
    """On a GPU, a part runs by tokamax's kernel in heads dense attention gives it, up
    to `flash.TOKAMAX_HEAD_DIM` (its backward finds no tiling for heads of 256 on a
    B300); wider ones (MLA's 192, Qwen3.8's 256) run by XLA."""
    monkeypatch.setattr(kernels, "gpu", lambda: True)
    monkeypatch.setattr(sparse, "MOSAIC", object())
    monkeypatch.setattr(parallel.flash, "MOSAIC", object())
    dk, dv = dims
    q = jnp.zeros((1, T, H, dk), jnp.bfloat16)
    k = jnp.zeros((1, T, 2, dk), jnp.bfloat16)
    v = k[..., :dv]
    part = sparse.Part(k, v, sparse.token_span(Segments.of(POSITIONS[:1]), None))
    forward, _ = sparse._runs(part, sparse._Static(SCALE, None, sparse._flash(q), True))
    dense = parallel.flash.tokamax_takes(q) and parallel.flash.tokamax_takes(v)
    assert (forward is sparse._flash_fwd) == dense
    assert (forward is sparse._xla_fwd) == (not dense)


def test_the_logsumexp_is_the_softmaxs_of_every_part_and_the_sinks(monkeypatch):
    monkeypatch.setattr(sparse, "ELEMENTS", TINY)
    parts, _, cap = case("csa")
    q, sinks = jax.random.normal(jax.random.key(0), (B, T, H, D)), jnp.arange(H) - 1.0
    _, lse = sparse.attend(q, parts, sinks, SCALE, cap)
    k = jnp.repeat(jnp.concatenate([p.k for p in parts], 1), H, axis=2)
    logits = softcap(jnp.einsum("bthd,bshd->bths", q, k) * SCALE, cap)
    seen = jnp.concatenate([sparse.visible(p) for p in parts], -1)[:, :, None]
    sink = jnp.broadcast_to(sinks[:, None], (B, T, H, 1))
    every = jnp.concatenate([jnp.where(seen, logits, -jnp.inf), sink], -1)
    np.testing.assert_allclose(lse, jax.nn.logsumexp(every, -1), atol=1e-5)


MESHES = {
    "context_tensor": tcfg.MeshConfig(context=2, tensor=2),
    "context_fsdp": tcfg.MeshConfig(context=4, fsdp=2),
    "data_tensor": tcfg.MeshConfig(data=2, tensor=4),
}


@pytest.mark.parametrize("mesh", MESHES.values(), ids=MESHES.keys())
@pytest.mark.parametrize("name", ["window", "blocks", "groups", "csa"])
def test_split_over_devices_it_attends_as_on_one(monkeypatch, name, mesh):
    """A device has its rows, queries and heads, and every key."""
    monkeypatch.setattr(sparse, "ELEMENTS", TINY)
    grads, out = run(kernel, name)
    with parallel.use(mesh):
        split_grads, split_out = run(kernel, name)
    close(split_out, out)
    close(split_grads, grads)


# --- In a layer -----------------------------------------------------------------------

PRELUDE = """
d_model = 32
vocab_size = 64
norm = rmsnorm
ffn = gated_mlp(intermediate_size=48, activation=swish)
small = indexer(num_heads=2, head_dim=8, top_k=3)
latent = mla(num_heads=4, q_lora_rank=12, kv_lora_rank=8, qk_nope_head_dim=6,
             qk_rope_head_dim=4, v_head_dim=5)
window = csa(num_heads=4, head_dim=16, q_lora_rank=12, o_lora_rank=6, o_groups=2,
             sliding_window_size=3, sinks=true, softcap=20,
             rope=rope(fraction=0.25, layout=interleaved))
"""
LAYERS = {
    "gqa": "[norm gqa(num_heads=4, num_kv_heads=2, rope=rope(), sliding_window_size=9,"
    " indexer=small)]",
    "qsa": "[norm gqa(num_heads=4, num_kv_heads=2, rope=rope(fraction=0.5),"
    " indexer=small(top_k=4, block_size=2, weighted=false))]",
    "msa": "[norm gqa(num_heads=4, num_kv_heads=2, rope=rope(fraction=0.5),"
    " indexer=small(top_k=4, block_size=2, pool=max, per_group=true, relu=false,"
    " weighted=false))]",
    "mla_ports": "[norm latent(indexer=small, index=@k)] [norm latent(index=$k)]",
    "csa_ports": "[norm window(compress_ratio=2, indexer=small(candidate_block_size=2,"
    " candidate_top_k=2), kv=@kv, candidates=@pool)]"
    " [norm window(compress_ratio=2, indexer=small, kv=$kv, candidates=$pool)]",
}


def trained(layers: str, logits: bool) -> tuple[float, float, nnx.State]:
    """The language-modeling and indexer losses of a training forward on packed rows,
    and the gradient of their sum; by all t by t logits if `logits`."""
    m = Model(
        compile(PRELUDE + f"model = embed {layers} [norm ffn] norm head"),
        rngs=nnx.Rngs(0),
    )
    tokens = jax.random.randint(jax.random.key(1), (B, T), 0, 64)

    def loss(m):
        [head], reported = m.training_heads(tokens, POSITIONS)
        lm = -jax.nn.log_softmax(head.logits())[..., 0].mean()
        indexer = sum(r.value for r in reported)
        return lm + indexer, (lm, indexer)

    with pytest.MonkeyPatch.context() as patch:
        if logits:  # as a layer with one of FullLogits' features attends
            patch.setattr(Sparse, "_feature", lambda self: "all t by t logits")
        (_, (lm, indexer)), grads = nnx.value_and_grad(loss, has_aux=True)(m)
    return float(lm), float(indexer), nnx.state(grads)


@pytest.mark.parametrize("layers", LAYERS.values(), ids=LAYERS.keys())
def test_a_layer_trains_as_by_all_t_by_t_logits(monkeypatch, layers):
    """The values, the indexer's loss and every gradient, of the indexer's too."""
    monkeypatch.setattr(sparse, "ELEMENTS", TINY)
    lm, indexer, grads = trained(layers, logits=False)
    want_lm, want_indexer, want_grads = trained(layers, logits=True)
    assert indexer > 0 and np.isclose(indexer, want_indexer, atol=1e-6)
    assert np.isclose(lm, want_lm, atol=1e-6)
    close(grads, want_grads)


def picked_log_softmax(q, w, keys, index):
    """cuDNN's log-softmax of each query's picks' scores, by XLA: 0 for a query with
    no pick, and clipped at log(tiny), as the kernels give it."""
    rows = jax.vmap(lambda k, i: k[i])(keys, jnp.maximum(index, 0))
    dots = jax.nn.relu(jnp.einsum("bthi,btki->bthk", q, rows))
    scores = jnp.where(index >= 0, jnp.einsum("bthk,bth->btk", dots, w), -1e30)
    p = jnp.where(index >= 0, jax.nn.softmax(scores), 0)
    return jnp.log(jnp.maximum(p, indexer_cudnn.TINY))


@pytest.mark.parametrize("layers", ["gqa", "mla_ports"])
def test_the_loss_by_cudnns_kernels_trains_as_by_xla(monkeypatch, layers):
    """With the kernels (by XLA here), 4 queries' picks at a time: the indexer's loss
    and every gradient."""
    monkeypatch.setattr(sparse, "ELEMENTS", TINY)
    want_lm, want_indexer, want_grads = trained(LAYERS[layers], logits=False)
    monkeypatch.setattr("bux.architecture.attention.indexer.LOSS_PICKS", B * 4 * 3)
    monkeypatch.setattr(indexer_cudnn, "takes", lambda *a: True)
    monkeypatch.setattr(indexer_cudnn, "log_softmax", picked_log_softmax)
    lm, indexer, grads = trained(LAYERS[layers], logits=False)
    assert indexer > 0 and np.isclose(indexer, want_indexer, atol=1e-6)
    assert np.isclose(lm, want_lm, atol=1e-6)
    close(grads, want_grads)


def step_memory(t: int, logits: bool = False) -> int:
    """The temporary bytes of a sparse GQA layer's training forward and backward on t
    tokens, its indexer's loss with it."""
    indexer = cfg.SparseIndexerConfig(num_heads=2, head_dim=8, top_k=8)
    c = cfg.GQAConfig(num_heads=4, num_kv_heads=2, indexer=indexer)
    graphdef, state = nnx.split(build_gqa(c, 32, (), nnx.Rngs(0)))

    def loss(state, x):
        taps = Taps(training=True)
        y = nnx.merge(graphdef, state)(x, taps)
        return y.sum() + taps.losses[0].value

    with pytest.MonkeyPatch.context() as patch:
        if logits:
            patch.setattr(Sparse, "_feature", lambda self: "all t by t logits")
        compiled = jax.jit(jax.grad(loss)).lower(state, jnp.ones((1, t, 32))).compile()
    stats = compiled.memory_analysis()
    assert stats is not None
    return stats.temp_size_in_bytes


def test_memory_grows_linearly_with_the_queries(monkeypatch):
    """Twice the tokens, twice the memory; by all t by t logits, four times."""
    monkeypatch.setattr(sparse, "ELEMENTS", 1 << 14)
    assert step_memory(2048) < 2.3 * step_memory(1024)
    assert step_memory(2048, logits=True) > 3.5 * step_memory(1024, logits=True)


def test_mfu_counts_the_keys_a_query_keeps_and_the_indexer_s_ranking():
    """The attention, the indexer's loss and its target count top_k keys a query
    (`flops.Mask`), and the indexer's ranking every key a query may pick; each product
    once, so not the logits that the attention's backward pass computes again."""
    top_k, heads, dims = 3, 2, 8
    indexer = cfg.SparseIndexerConfig(num_heads=heads, head_dim=dims, top_k=top_k)
    c = cfg.GQAConfig(num_heads=H, num_kv_heads=2, indexer=indexer)
    graphdef, state = nnx.split(build_gqa(c, 32, (), nnx.Rngs(0)))

    def loss(state, x):
        taps = Taps(training=True)
        y = nnx.merge(graphdef, state)(x, taps)
        return y.sum() + taps.losses[0].value

    with kernels.off():
        work = flops.count(jax.grad(loss), state, jnp.ones((B, T, 32)))
    attention = 2 * 2 * B * H * T * T * D  # the logits and the weighted values
    target = 2 * B * H * T * T * D  # the logits again, for the indexer's loss
    scores = 2 * B * T * heads * (dims + 1)  # a key's dot products, weighted
    assert work.attention == {
        flops.Mask(): T * scores,  # the ranking: no gradient
        flops.Mask(top_k=top_k): 3 * attention + target,
        flops.Mask(top_k=top_k, picks=True): 3 * top_k * scores,  # the indexer's loss
    }


# --- The indexer's selection ----------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_the_indexers_top_k_keeps_what_lax_top_k_does(seed):
    """Of equal scores the first, none at -inf; many ties at 0, as ReLU makes."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(2, 9, 200)).astype(np.float32)
    scores[rng.random(scores.shape) < 0.3] = 0.0
    scores[rng.random(scores.shape) < 0.2] = -np.inf
    for k in (1, 7, 64, 200):
        values, index = jax.lax.top_k(jnp.asarray(scores), k)
        want = jnp.where(values > -jnp.inf, index, -1)
        got = _top(jnp.asarray(scores), k)
        assert np.array_equal(np.sort(got, -1), np.sort(want, -1)), k


def radix(rows, lengths, k):
    """cuDNN's top-k kernel by lax: each row's k highest of its first `lengths`
    slots, the first of equal ones, in order of index; -1 and -inf past a short row."""
    inside = jnp.arange(rows.shape[-1]) < lengths[:, None]
    values, picks = jax.lax.top_k(jnp.where(inside, rows, -jnp.inf), k)
    order = jnp.argsort(jnp.where(values > -jnp.inf, picks, rows.shape[-1]), -1)
    picks, values = (jnp.take_along_axis(a, order, -1) for a in (picks, values))
    return jnp.where(values > -jnp.inf, picks, -1), values


@pytest.mark.parametrize("rows", [None, 5], ids=["all", "5 at a time"])
@pytest.mark.parametrize("mesh", [None, tcfg.MeshConfig(data=2, context=4)])
def test_the_kernels_top_k_keeps_what_xla_does(monkeypatch, mesh, rows):
    """With cuDNN's kernel (by lax here): unseen keys never kept, though fewer than
    k are seen; ties at 0 and at the k-th; spans of k slots or fewer, and empty ones,
    by no kernel; a query's end before the last slot; rows a few at a time, as past
    what the kernel's scratch indexes."""
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(2, 8, 96)).astype(np.float32)
    scores[rng.random(scores.shape) < 0.3] = 0.0
    scores[rng.random(scores.shape) < 0.1] = -np.inf  # keys a query does not see
    start = np.broadcast_to(np.arange(8) % 2 * 50, (2, 8)).astype(np.int32)
    end = np.broadcast_to(np.arange(8) * 12 + 12, (2, 8)).astype(np.int32)
    slot = np.arange(96)
    scores[(slot < start[..., None]) | (slot >= end[..., None])] = -np.inf
    scores, span = (
        jnp.asarray(scores),
        sparse.Span(jnp.asarray(start), jnp.asarray(end)),
    )
    want = {k: _top(scores, k) for k in (1, 7, 30, 64, 96)}
    monkeypatch.setattr(indexer_cudnn, "runs", lambda k: True)
    monkeypatch.setattr(indexer_cudnn, "_radix_rows", radix)
    if rows:
        monkeypatch.setattr(indexer_cudnn, "SCRATCH", 2**31 // (96 * rows))
    with parallel.use(mesh) if mesh else contextlib.nullcontext():
        for k, top in want.items():
            got = jax.jit(_top, static_argnums=1)(scores, k, span)
            np.testing.assert_array_equal(got, top)


def test_the_kernels_unseen_keys_are_the_lowest_floats_and_none_tie():
    unseen = np.asarray(indexer_cudnn._unseen(1000))
    assert np.isfinite(unseen).all() and (np.diff(unseen) > 0).all()
    assert unseen[0] == np.finfo(np.float32).min and unseen[-1] < -3.3e38


def test_cudnns_indexer_kernels_do_not_run_off_a_gpu():
    assert not indexer_cudnn.runs(2048)
    assert not indexer_cudnn.takes(64, 128, 2048, jnp.bfloat16)


class Taken(Exception):
    """A cuDNN DSA kernel was called."""


def taken(*_):
    raise Taken


# MiniMax MSA's indexer settings, together and each alone
MSA_SETTINGS = {
    "msa": {"block_size": 2, "pool": "max", "per_group": True, "relu": False},
    "max_blocks": {"block_size": 2, "pool": "max"},
    "per_group": {"per_group": True, "relu": False},
    "plain_dots": {"relu": False},
}


def indexed(settings: dict, x: Float[Array, "b t d"]) -> tuple:
    """A training forward of a sparse GQA layer whose indexer has `settings`: its
    loss and output, and the loss's gradient."""
    weighted = not settings.get("per_group", False)
    indexer = cfg.SparseIndexerConfig(
        num_heads=2, head_dim=8, top_k=4, weighted=weighted, **settings
    )
    c = cfg.GQAConfig(
        num_heads=H, num_kv_heads=2, rope=cfg.RoPEConfig(), indexer=indexer
    )
    attn = build_gqa(c, 32, (), nnx.Rngs(0))

    def loss(attn):
        taps = Taps(training=True)
        y = attn(x, taps)
        return taps.losses[0].value, y

    return nnx.value_and_grad(loss, has_aux=True)(attn)


@pytest.mark.parametrize("settings", MSA_SETTINGS.values(), ids=MSA_SETTINGS.keys())
def test_cudnns_indexer_kernels_decline_msas_settings(monkeypatch, settings):
    """With cuDNN's top-k and loss kernels said to run, an indexer with any of MSA's
    settings picks and learns by XLA, as without them: their predicates decline it
    (`Indexer._by_cudnn`), as they compute DSA's sums of ReLU'd heads. DSA's own
    indexer goes to them."""
    x = jax.random.normal(jax.random.key(0), (B, T, 32))
    want = indexed(settings, x)
    monkeypatch.setattr(indexer_cudnn, "runs", lambda k: True)
    monkeypatch.setattr(indexer_cudnn, "takes", lambda *a: True)
    monkeypatch.setattr(indexer_cudnn, "top_k", taken)
    monkeypatch.setattr(indexer_cudnn, "log_softmax", taken)
    close(indexed(settings, x), want)
    with pytest.raises(Taken):
        indexed({}, x)


def test_cudnns_csa_kernels_decline_picks_for_each_group(monkeypatch):
    """cuDNN's sparse attention takes latents kept by index for all heads, and not
    MSA's picks for each group, which only gqa makes."""
    monkeypatch.setattr(sparse_cudnn, "runs", lambda: True)
    q = jnp.zeros((1, 8, 64, 512), jnp.bfloat16)
    latents = jnp.zeros((1, 8, 1, 512), jnp.bfloat16)
    span = sparse.token_span(Segments.of(jnp.arange(8)[None]), None)
    index = jnp.zeros((1, 8, 4), jnp.int32)
    one = sparse.Part(latents, latents, span, None, sparse.Kept(index))
    each = one._replace(kept=sparse.Kept(index[:, :, None]))
    assert sparse_cudnn.takes(q, [one], [None], None)
    assert not sparse_cudnn.takes(q, [each], [None], None)


def test_the_kept_ones_go_first():
    index = jnp.asarray([[[3, -1, 5, -1, 7], [-1, -1, -1, -1, 2]]])
    assert _first(index).tolist() == [[[3, 5, 7, -1, -1], [2, -1, -1, -1, -1]]]


def test_the_scoring_kernel_scores_as_xla_does():
    """Interpreted, as off a GPU; a block of keys after every query's end scores 0."""
    key = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(key[0], (2, 32, 3, 16))
    w = jax.random.normal(key[1], (2, 32, 3))
    keys = jax.random.normal(key[2], (2, 256, 16))
    end = jnp.broadcast_to(jnp.arange(1, 33) * 4, (2, 32))  # up to 128: a block
    want = jnp.einsum("bthi,bsi->bths", q, keys)
    want = jnp.einsum("bths,bth->bts", jax.nn.relu(want), w)
    got = scoring._kernel(q, w, keys, end)
    np.testing.assert_allclose(got[..., :128], want[..., :128], atol=1e-4)
    assert not got[..., 128:].any()
    passes = jax.grad(lambda q: scoring.scores(q, w, keys, end).sum())(q)
    assert not passes.any()  # no gradient, the kernel's or its inputs'


def test_the_picked_kernels_score_and_differentiate_as_xla_does():
    """Interpreted, as off a GPU: each query's picks, distinct and -1 past them, the
    scores and the gradients of q, the weights and the keys (atomic adds)."""
    t, s, h, i, k = 8, 160, 16, 16, 128  # two blocks of picks
    key = jax.random.split(jax.random.key(0), 5)
    q = jax.random.normal(key[0], (B, t, h, i))
    w = jax.random.normal(key[1], (B, t, h))
    keys = jax.random.normal(key[2], (B, s, i))
    order = jax.random.uniform(key[3], (B, t, s)).argsort(-1)[..., :k]
    index = jnp.where(jnp.arange(k) < 4 + 17 * jnp.arange(t)[:, None], order, -1)
    cotangent = jax.random.normal(key[4], (B, t, k))

    def xla(q, w, keys):
        picked = jax.vmap(lambda keys, i: keys[i])(keys, jnp.maximum(index, 0))
        dots = jnp.einsum("bthi,btki->bthk", q, picked)
        scores = jnp.einsum("bthk,bth->btk", jax.nn.relu(dots), w)
        return jnp.where(index >= 0, scores, 0)

    def kernel(q, w, keys):
        return scoring.picked(q, w, keys, index)

    want, pull_want = jax.vjp(xla, q, w, keys)
    got, pull = jax.vjp(kernel, q, w, keys)
    np.testing.assert_allclose(got, want, atol=1e-4)
    close(pull(cotangent), pull_want(cotangent))


@pytest.mark.parametrize("cap", [None, 5.0])
def test_the_probabilities_kernel_sums_the_heads_as_xla_does(cap):
    """Interpreted, as off a GPU: GQA's heads, packed documents and a window; 0
    outside each query's span."""
    t = 64
    rows = Segments.of(jnp.asarray([[*range(20), *range(44)], [*range(t)]]))
    span = sparse.token_span(rows, 30)
    key = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(key[0], (B, t, H, 16))
    k = jax.random.normal(key[1], (B, t, 2, 16))
    part = sparse.Part(k, k, span)
    _, lse = sparse.attend(q, [part], None, SCALE, cap)
    want = sparse.probabilities(q, part, lse, SCALE, cap)  # by XLA, off a GPU
    got = scoring.probabilities(q, k, lse, *span, SCALE, cap)
    np.testing.assert_allclose(got, want, atol=1e-5)
    np.testing.assert_allclose(want.sum(-1), H, rtol=1e-5)  # a softmax a head
