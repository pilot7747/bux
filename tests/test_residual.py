import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux import kernels
from bux.architecture import Built, Context, Embedding, Model, NextN, Taps, num_params
from bux.architecture.graph import Sequence
from bux.architecture.layers.norm import Norm
from bux.architecture.residual import (
    AttentionResidual,
    GatedResidual,
    HyperConnections,
    ManifoldHyperConnections,
    Merge,
    ResidualGate,
    sinkhorn,
    sources,
    widen,
)
from bux.lang import compile
from bux.train.trainer import cast

D, V = 32, 64
GPU = pytest.mark.skipif(not kernels.gpu(), reason="needs a GPU")

PRELUDE = f"""
d_model = {D}
vocab_size = {V}
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=48, activation=swish)
mem = engram(head_dim=4, table_size=50)
shared out = head
"""

WIDENING = [
    "hc(expansion=3)",
    "hc(expansion=2, dynamic=false)",
    "hc(expansion=3, single_pass=true)",
    "mhc(expansion=4)",
    "mhc(expansion=2, dynamic=false)",
    "mhc(expansion=4, single_pass=true)",
    "gr(expansion=2, rank=4)",
    "attnres(block_size=2)",
    "attnres",
]
ALGEBRAS = [*WIDENING, "vpgate"]


def source(residual: str | None, body: str) -> str:
    return PRELUDE + (f"residual = {residual}\n" if residual else "") + body


def model(residual: str | None, body: str, seed: int = 0) -> Model:
    return Model(compile(source(residual, body)), rngs=nnx.Rngs(seed))


def top(m: Model) -> list[nnx.Module]:
    assert isinstance(m.graph, Sequence)
    return list(m.graph.items)


def tokens(seed: int = 0, length: int = 8) -> Int[Array, "b t"]:
    return jax.random.randint(jax.random.key(seed), (2, length), 0, V)


class Zero(nnx.Module):
    def __call__(self, x, taps):
        return jnp.zeros_like(x)


class Tanh(nnx.Module):
    def __init__(self, seed: int):
        self.w = jax.random.normal(jax.random.key(seed), (D, D)) / D**0.5

    def __call__(self, x, taps):
        return jnp.tanh(x @ self.w)


def randomize(module: nnx.Module, seed: int = 1, scale: float = 1.0) -> None:
    """Every parameter of `module` at random, so that nothing is at its init."""
    state = nnx.state(module, nnx.Param)
    leaves, tree = jax.tree.flatten(state)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    new = [scale * jax.random.normal(k, x.shape) for k, x in zip(keys, leaves)]
    nnx.update(module, jax.tree.unflatten(tree, new))


@pytest.mark.parametrize("residual", ALGEBRAS)
def test_every_algebra_runs_and_is_causal(residual):
    m = model(residual, "model = embed [norm attn] [norm ffn] norm head")
    t = tokens()
    logits = nnx.jit(lambda m, t: m(t))(m, t)
    assert logits.shape == (2, 8, V) and bool(jnp.isfinite(logits).all())
    randomize(m, scale=0.3)  # away from init, where the streams are all alike
    changed = t.at[:, 5:].set((t[:, 5:] + 1) % V)
    assert jnp.allclose(m(t)[:, :5], m(changed)[:, :5], atol=1e-5)
    assert not jnp.allclose(m(t)[:, 5:], m(changed)[:, 5:])


@pytest.mark.parametrize("residual", ALGEBRAS)
@pytest.mark.parametrize(
    "body",
    [
        # MTP from the main stream after the read (merge @h out $h nextn ...)
        (
            "mtp = $h! nextn [norm attn] [norm ffn] norm @h out\n"
            "model = embed [norm attn] [norm ffn [ffn]] norm @h out mtp^2"
        ),
        # a tap that holds the widened stream, and engram (on every copy, under hc,
        # mhc and gr; on the stream read back into one, under attnres)
        "model = embed [norm attn] @x [norm ffn [ffn]] $x! [norm ffn] mem norm head",
    ],
    ids=["mtp", "engram_taps"],
)
def test_mtp_engram_and_taps_run(residual, body):
    m = model(residual, body)
    t = tokens()
    outputs = m.outputs(t)
    assert all(o.shape == (2, 8, V) and bool(jnp.isfinite(o).all()) for o in outputs)
    if "mtp" in body:
        # The head after k nextn layers sees k tokens past its position.
        assert len(outputs) == 3
        changed = m.outputs(t.at[:, 5].set((t[:, 5] + 1) % V))
        for k, (a, b) in enumerate(zip(outputs, changed)):
            assert jnp.allclose(a[:, : 5 - k], b[:, : 5 - k], atol=1e-5)
            assert not jnp.allclose(a[:, 5 - k], b[:, 5 - k])


@pytest.mark.parametrize(
    "residual",
    [
        "hc(expansion=4)",
        "hc(dynamic=false)",
        "hc(single_pass=true)",
        "mhc(post_bias=0)",
        "mhc(dynamic=false, post_bias=0)",
        "mhc(single_pass=true, post_bias=0)",
        "attnres(block_size=3)",
        "vpgate",
    ],
)
def test_at_init_an_algebra_is_the_pre_norm_residual(residual):
    """hc as the paper initializes it, mhc writing to every stream alike, and attnres
    and vpgate as they are here, start as the pre-norm residual with the same weights
    (they take no random init).
    Up to the norms' eps, which weighs on the copies of a widened stream apart: it
    shows on embeddings as small as the default init's."""
    body = "norm.eps = 1e-12\nmodel = embed ([norm attn] [norm ffn])^3 norm head"
    t = tokens()
    assert jnp.allclose(model(residual, body)(t), model(None, body)(t), atol=1e-4)


def test_hc_reads_stream_k_mod_n_and_keeps_and_writes_all_of_them():
    blocks = "[norm attn] [norm ffn] [norm attn] [norm ffn]"
    m = model("hc(expansion=3)", f"model = embed {blocks} norm head")
    _, *branches, merge, _, _ = top(m)
    for k, branch in enumerate(branches):
        assert isinstance(branch, HyperConnections)
        alpha, beta = branch.static_alpha[...], branch.static_beta
        assert (alpha[:, 0] == jax.nn.one_hot(k % 3, 3)).all()
        assert (alpha[:, 1:] == jnp.eye(3)).all()
        assert beta is not None and (beta[...] == 1).all()
    assert isinstance(merge, Merge) and isinstance(merge.reader, HyperConnections)
    assert (merge.reader.static_alpha[...] == 1).all()  # the paper's sum of the streams


def test_mhc_starts_the_k_th_branch_on_stream_k_mod_n():
    blocks = "[norm attn] [norm ffn] [norm attn] [norm ffn]"
    m = model("mhc(expansion=3, res_bias=-6)", f"model = embed {blocks} norm head")
    _, *branches, merge, _, _ = top(m)
    for k, branch in enumerate(branches):
        assert isinstance(branch, ManifoldHyperConnections)
        favored = 2 * jax.nn.one_hot(k % 3, 3) - 1  # mHC-lite's +-1
        bias = branch.bias[...]
        assert (bias[:3] == favored).all() and (bias[3:6] == favored).all()
        assert (bias[6:].reshape(3, 3) == -6 * (1 - jnp.eye(3))).all()
    assert isinstance(merge, Merge)
    assert isinstance(merge.reader, ManifoldHyperConnections)
    assert jnp.allclose(jax.nn.sigmoid(merge.reader.bias[...]), 1 / 3)  # the mean


def next_token_loss(m: Model, t: Int[Array, "b t"]) -> Float[Array, ""]:
    logp = jax.nn.log_softmax(m(t)[:, :-1])
    return -jnp.take_along_axis(logp, t[:, 1:, None], -1).mean()


def bias_grads(m: Model) -> list[Float[Array, " k"]]:
    """The next-token loss's gradients by each mhc branch's static logits b."""
    grads = nnx.grad(next_token_loss)(m, tokens())
    graphdef, _, rest = nnx.split(m, nnx.Param, ...)
    tree = top(nnx.merge(graphdef, grads, rest))
    kept = [b for b in tree if isinstance(b, ManifoldHyperConnections) and b.body]
    return [b.bias[...] for b in kept]


@pytest.mark.parametrize(
    ("residual", "apart"),
    [
        ("mhc", True),
        ("mhc(post_bias=0)", True),
        ("mhc(pre_bias=0, post_bias=0)", False),
    ],
)
def test_mhc_streams_part_unless_their_biases_start_alike(residual, apart):
    """Streams apart from the start, or set apart by the first gradients as under
    post_bias=0, train apart: the writes' gradients differ across them. Streams that
    start alike stay so for good: every gradient treats them alike, and the mixing
    gets none, since it sums copies of one stream to that stream."""
    m = model(residual, "model = embed [norm attn] [norm ffn] [norm attn] norm head")
    first, second, _ = bias_grads(m)
    post = first[4:8]
    spread = jnp.abs(post - post.mean()).max() / jnp.abs(post).max()
    assert bool(spread > 1e-2) == apart
    mixing = jnp.linalg.norm(second[8:]) / jnp.linalg.norm(second[4:8])
    assert bool(mixing > 1e-4) == (residual == "mhc")  # the streams differ at once
    assert apart or bool(mixing < 1e-6)


def test_mhc_mixes_the_streams_through_doubly_stochastic_matrices():
    logits = jax.random.normal(jax.random.key(0), (5, 4, 4))
    res = sinkhorn(logits, 20)
    assert (res > 0).all()
    assert jnp.allclose(res.sum(-2), 1, atol=1e-5)  # what every stream receives
    assert jnp.allclose(res.sum(-1), 1, atol=1e-3)  # what every stream gives
    # So the mixing keeps the sum of the streams, whatever its coefficients are.
    arch = compile(PRELUDE + "residual = mhc(expansion=4)\nmodel = embed [attn] head")
    c = arch.residual
    assert isinstance(c, cfg.HyperConnectionsConfig)
    branch = ManifoldHyperConnections(c, D, 0, Zero())
    randomize(branch, scale=0.1)
    assert branch.scale is not None
    branch.scale[...] = jnp.ones(3)  # logits of about unit size
    x = jax.random.normal(jax.random.key(2), (2, 8, 4, D))
    y = branch(x, Taps())
    assert not jnp.allclose(y, x, atol=1e-1)
    assert jnp.allclose(y.sum(2), x.sum(2), atol=2e-3)


def hc_split_sinkhorn(mixes, scale, base, n=4, eps=1e-6):
    """DeepSeek-V4.1's kernel.py, with the Sinkhorn above: pre, post and comb[j, k]."""
    pre = jax.nn.sigmoid(mixes[..., :n] * scale[0] + base[:n]) + eps
    post = 2 * jax.nn.sigmoid(mixes[..., n : 2 * n] * scale[1] + base[n : 2 * n])
    comb = mixes[..., 2 * n :] * scale[2] + base[2 * n :]
    return pre, post, sinkhorn(comb.reshape(*comb.shape[:-1], n, n), 20)


def test_single_pass_mhc_is_deepseek_v4_1s():
    """As V4.1's inference/model.py runs it: a branch computes pre, post and comb from
    the streams it gets, but reads them with the pre of the branch before (the first,
    with stream 0 alone), and the head reads with the last pre."""
    c = cfg.HyperConnectionsConfig(type="mhc", single_pass=True, norm_eps=1e-20)
    bodies = [Tanh(k) for k in range(3)]
    branches = [ManifoldHyperConnections(c, D, k, b) for k, b in enumerate(bodies)]
    for k, branch in enumerate(branches):
        randomize(branch, seed=k, scale=0.3)
    x = 1e-3 * jax.random.normal(jax.random.key(9), (2, 8, 4, D))  # small: eps shows
    taps, y = Taps(), x
    for branch in branches:
        y = branch(y, taps)
    ours = Merge(None)(y, taps)

    h, pre_mix = x, jax.nn.one_hot(jnp.zeros((2, 8), int), 4)  # make_identity_pre_mix
    for branch, body in zip(branches, bodies):  # Block.forward
        assert branch.proj is not None and branch.scale is not None
        flat = h.reshape(2, 8, 4 * D)  # hc_mixes
        rsqrt = jax.lax.rsqrt(jnp.square(flat).mean(-1, keepdims=True) + 1e-20)
        mixes = (flat @ branch.proj[...]) * rsqrt
        pre, post, comb = hc_split_sinkhorn(mixes, branch.scale[...], branch.bias[...])
        out = body(jnp.sum(pre_mix[..., None] * h, 2), Taps())  # hc_pre
        kept = jnp.sum(comb[..., None] * h[:, :, :, None], 2)  # hc_post
        h, pre_mix = post[..., None] * out[:, :, None] + kept, pre
    assert jnp.allclose(ours, jnp.sum(pre_mix[..., None] * h, 2), rtol=1e-4, atol=1e-9)
    # The head's read has no parameters of its own.
    m = model("mhc(single_pass=true)", "model = embed [norm attn] [norm ffn] norm head")
    merge = top(m)[3]
    assert isinstance(merge, Merge) and merge.reader is None and num_params(merge) == 0


def test_a_single_pass_stream_stays_in_its_dtype():
    """The first branch reads stream 0 alone in the stream's dtype: a float32 read
    would turn every branch after it to float32 under bfloat16 parameters."""
    c = cfg.HyperConnectionsConfig(type="mhc", single_pass=True)
    branches = [ManifoldHyperConnections(c, D, k, Zero()) for k in range(2)]
    y, taps = jnp.ones((2, 8, 4, D), jnp.bfloat16), Taps()
    for branch in branches:
        y = cast(branch, jnp.bfloat16)(y, taps)  # as the trainer casts a model
    assert y.dtype == jnp.bfloat16


def test_mhc_without_dynamic_has_static_coefficients():
    c = cfg.HyperConnectionsConfig(type="mhc", dynamic=False)
    branch = ManifoldHyperConnections(c, D, 0, Zero())
    assert branch.proj is None and branch.scale is None
    randomize(branch)
    res = sinkhorn(branch.bias[...][8:].reshape(4, 4), 20)
    x = jax.random.normal(jax.random.key(0), (2, 8, 4, D))
    assert jnp.allclose(
        branch(x, Taps()), jnp.einsum("jk,btjd->btkd", res, x), atol=1e-5
    )


def test_norm_eps_reaches_the_norms_of_every_algebra():
    hc_config = cfg.HyperConnectionsConfig(type="hc", norm_eps=1e-5)
    hc = HyperConnections(hc_config, D, 0, Zero())
    gr = GatedResidual(cfg.GatedResidualConfig(rank=4, norm_eps=1e-5), D, nnx.Rngs(0))
    attnres = AttentionResidual(cfg.AttentionResidualConfig(norm_eps=1e-5), D)
    assert hc.dynamic_alpha is not None
    assert hc.dynamic_alpha.norm.eps == gr.norm.eps == attnres.key_norm.eps == 1e-5


def test_attnres_sources_are_the_embedding_and_blocks_of_outputs():
    state = jax.random.normal(jax.random.key(0), (2, 3, 6, D))  # embedding, 5 outputs
    e, o = state[:, :, 0], state[:, :, 1:]
    expected = jnp.stack(
        [e, o[:, :, 0] + o[:, :, 1], o[:, :, 2] + o[:, :, 3], o[:, :, 4]], 2
    )
    assert jnp.allclose(sources(state, 2), expected, atol=1e-6)
    # The block under way is a source once it has an output.
    assert sources(state[:, :, :5], 2).shape[2] == 3
    assert sources(state[:, :, :1], 2).shape[2] == 1
    assert (sources(state, None) == state).all()
    assert jnp.allclose(sources(state, 8)[:, :, 1], o.sum(2), atol=1e-6)
    # With its query at 0, a read averages its sources.
    for size in (None, 2, 3):
        read = AttentionResidual(cfg.AttentionResidualConfig(block_size=size), D)
        expected = sources(state, size).mean(2)
        assert jnp.allclose(read.read(state), expected, atol=1e-5)


def test_attnres_branches_add_their_outputs_to_the_state():
    m = model("attnres(block_size=2)", "model = embed [norm attn] [norm ffn] norm head")
    embed, first, second, merge, *_ = top(m)
    taps = Taps(tokens=tokens())
    x = embed(tokens(), taps)
    assert x.shape == (2, 8, 1, D)
    assert first(x, taps).shape == (2, 8, 2, D)
    assert second(first(x, taps), taps).shape == (2, 8, 3, D)
    assert isinstance(first, AttentionResidual) and isinstance(merge, Merge)


def test_vpgate_keeps_the_variance_of_every_channel():
    m = model(
        "vpgate(temperature=0.5)", "model = embed [norm attn] [norm ffn] norm head"
    )
    _, first, second, *_ = top(m)
    assert isinstance(first, ResidualGate) and isinstance(second, ResidualGate)
    # The k-th branch starts at b = 1 / sqrt(k + 2).
    assert jnp.allclose(first.weights()[1], 2**-0.5)
    assert jnp.allclose(second.weights()[1], 3**-0.5)
    first.gate[...] = jnp.linspace(-3, 3, D)
    a, b = first.weights()
    assert jnp.allclose(a**2 + b**2, 1, atol=1e-5)
    first.gate[...] = jnp.full(D, 10.0)  # b near 1: a stays at sqrt(eps)
    a, b = first.weights()
    assert jnp.allclose(a**2, first.eps) and jnp.allclose(b, 1, atol=1e-6)


@pytest.mark.parametrize("residual", WIDENING)
def test_merge_reads_the_stream_back_where_the_compiler_puts_it(residual):
    m = model(residual, "model = embed [norm attn] [norm ffn] norm head")
    merge = top(m)[3]
    assert isinstance(merge, Merge) and isinstance(top(m)[4], Norm)
    taps = Taps(tokens=tokens())
    x = top(m)[2](top(m)[1](top(m)[0](tokens(), taps), taps), taps)
    assert merge(x, taps).shape == (2, 8, D)
    # The merge has parameters of its own, apart from the branches'.
    assert all(merge.reader is not branch for branch in top(m)[1:3])


def test_gr_reads_normalized_streams_and_its_read_replaces_the_final_norm():
    m = model("gr(expansion=4, rank=8)", "model = embed [attn] [ffn] head")
    *_, merge, _ = top(m)
    assert isinstance(merge, Merge) and isinstance(merge.reader, GatedResidual)
    x = jax.random.normal(jax.random.key(0), (2, 8, 4, D))
    scaled = x * jnp.asarray([3.0, 0.5, 1.0, 2.0])[:, None]  # each stream its own size
    assert jnp.allclose(merge(x, Taps()), merge(scaled, Taps()), atol=1e-5)
    branch = top(m)[1]
    assert isinstance(branch, GatedResidual)
    y = branch(x, Taps(tokens=tokens()))
    assert y.shape == x.shape and not jnp.allclose(y, x)
    logits = m(tokens())
    assert logits.shape == (2, 8, V) and bool(jnp.isfinite(logits).all())


class Affine(nnx.Module):
    def __init__(self, d: int):
        self.lin = nnx.Linear(d, d, rngs=nnx.Rngs(1))

    def __call__(self, x, taps):
        return self.lin(x)


@GPU
def test_gr_computes_on_a_gpu_as_op_by_op():
    """A gr branch on a widened stream, its value and gradients jitted, as it runs op
    by op: before the barriers around f, XLA halved the value on a B300."""
    n, d = 4, 512
    gr = GatedResidual(cfg.GatedResidualConfig(rank=64), d, nnx.Rngs(0), Affine(d))
    graphdef, state = nnx.split(gr)
    keys = jax.random.split(jax.random.key(0))
    e = jax.random.normal(keys[0], (8, 128, d))
    c = jax.random.normal(keys[1], (8, 128, n, d))

    def loss(state, e):
        return jnp.sum(nnx.merge(graphdef, state)(widen(e, n), Taps()) * c)

    step = jax.value_and_grad(loss, argnums=(0, 1))
    with jax.default_matmul_precision("highest"):
        jitted = jax.jit(step)(state, e)
        with jax.disable_jit():
            eager = step(state, e)
    for a, b in zip(jax.tree.leaves(jitted), jax.tree.leaves(eager), strict=True):
        assert jnp.allclose(a, b, rtol=1e-4, atol=1e-4 * float(jnp.abs(b).max()))


@pytest.mark.parametrize("residual", WIDENING)
def test_a_shared_block_runs_at_every_depth(residual):
    fresh = model(
        residual, "blk = [norm attn] [norm ffn]\nmodel = embed blk^3 norm head"
    )
    shared = model(
        residual, "shared blk = [norm attn] [norm ffn]\nmodel = embed blk^3 norm head"
    )
    one_block = (num_params(fresh) - num_params(shared)) // 2
    assert num_params(fresh) - num_params(shared) == 2 * one_block > 0
    logits = shared(tokens())
    assert logits.shape == (2, 8, V) and bool(jnp.isfinite(logits).all())


@pytest.mark.parametrize(
    "residual",
    [
        "gr(expansion=4, rank=8)",
        "mhc(expansion=2)",
        "mhc(expansion=2, single_pass=true)",
    ],
)
def test_mtp_goes_on_from_the_widened_stream(residual):
    """Qwen3.8-Flash-Next: the MTP module starts from the copies before the main read,
    mixes the next token's embedding into each one, and reads them back for its head."""
    m = model(
        residual,
        "D = [norm attn] [ffn]\n"
        "mtp = $h nextn(norm=norm) D out\n"
        "model = embed D mem D @h out mtp",
    )
    nextn = next(x for _, x in nnx.iter_graph(m) if isinstance(x, NextN))
    n = 4 if "gr" in residual else 2
    assert nextn.copies is None and nextn.hidden_norm.scale is not None
    assert nextn.hidden_norm.scale.shape == (n * D,)
    t = tokens()
    outputs = m.outputs(t)
    assert len(outputs) == 2
    assert all(bool(jnp.isfinite(o).all()) for o in outputs)
    changed = m.outputs(t.at[:, 5].set((t[:, 5] + 1) % V))
    assert jnp.allclose(outputs[1][:, :4], changed[1][:, :4], atol=1e-5)
    assert not jnp.allclose(outputs[1][:, 4], changed[1][:, 4])


def test_nextn_mixes_the_embedding_into_every_copy_normalized_together():
    source = PRELUDE + "residual = gr(rank=8)\nmodel = embed [attn] head"
    embed = top(Model(compile(source), rngs=nnx.Rngs(0)))[0]
    assert isinstance(embed, Embedding)
    ctx = Context(compile(source), nnx.Rngs(0), built=Built(embedding=embed))
    nextn = NextN(cfg.NextNConfig(), "streams", ctx)
    t, x = tokens(), jax.random.normal(jax.random.key(1), (2, 8, 4, D))
    y = nextn(x, Taps(tokens=t))
    assert y.shape == x.shape
    ahead = jnp.pad(t[:, 1:], ((0, 0), (0, 1)))
    e = nextn.embed_norm(embed.embed_tokens(ahead))
    h = nextn.hidden_norm(x.reshape(2, 8, 4 * D)).reshape(x.shape)
    for j in range(4):
        expected = nextn.proj(jnp.concatenate([e, h[:, :, j]], -1))
        assert jnp.allclose(y[:, :, j], expected, atol=1e-5)
    # Together: the copies keep their sizes relative to each other.
    assert jnp.allclose(nextn(3 * x, Taps(tokens=t)), y, atol=1e-4)
    assert not jnp.allclose(
        nextn(x.at[:, :, 0].multiply(3.0), Taps(tokens=t))[:, :, 1], y[:, :, 1]
    )
    # On one stream under an algebra that widens it, nextn widens its output.
    one = NextN(cfg.NextNConfig(), D, ctx)
    assert one(x[:, :, 0], Taps(tokens=t)).shape == (2, 8, 4, D)


# --- Split over devices ----------------------------------------------------------------

# Tensor parallelism 4 ways with FSDP, and 2 ways with sequence parallelism.
SPLITS = [
    {"mesh": "mesh(fsdp=2, tensor=4)"},
    {"mesh": "mesh(fsdp=4, tensor=2)", "layout": "layout(stream=tensor)"},
]
# At 1e-2, these small runs are chaotic enough that rounding grows past 1e-5.
STABLE = "adamw(lr=1e-3)"


@pytest.mark.parametrize("residual", ALGEBRAS)
def test_every_algebra_trains_as_on_one_device_split(losses, involuntary, residual):
    """The widened stream, from the embedding, every branch, engram and nextn, is
    the stream wherever it is: split by rows, and by positions over `stream`."""
    body = (
        "mtp = $h! nextn [norm ffn] norm @h out\n"
        "model = embed [norm conv] [norm ffn] mem norm @h out mtp"
    )
    one = losses(source(residual, body), mesh="mesh(data=1)", optimizer=STABLE)
    involuntary()
    for split in SPLITS:
        split_losses = losses(source(residual, body), optimizer=STABLE, **split)
        assert np.allclose(split_losses, one, atol=1e-5)
    assert involuntary() == 0
