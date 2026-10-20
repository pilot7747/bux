import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jaxtyping import Array, Int

from bux import config as cfg
from bux.architecture import Model
from bux.architecture.graph import ResidualBranch, Sequence
from bux.architecture.linear_attention import (
    GatedDeltaNet,
    KimiDeltaAttention,
    LightningAttention,
    delta_rule,
    l2norm,
    recurrence,
    slopes,
)
from bux.architecture.rwkv7 import RWKV7
from bux.lang import compile

D, V, F = 32, 64, 48

PRELUDE = f"""
d_model = {D}
vocab_size = {V}
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2)
ffn = gated_mlp(intermediate_size={F}, activation=swish)
"""

MIXERS = {
    "gdn": "gdn(num_heads=2, head_dim=8)",
    "gdn_gva": (
        "gdn(num_heads=2, num_v_heads=4, head_dim=8, v_head_dim=4, output_gate=sigmoid)"
    ),
    "gdn_plain": "gdn(num_heads=2, head_dim=8, conv_size=none, output_gate=none)",
    "kda": "kda(num_heads=2, head_dim=8, gate_rank=4, output_gate_rank=4)",
    "kda_bounded": "kda(num_heads=2, num_v_heads=4, head_dim=8, gate_lower_bound=-5)",
    "deltanet": "deltanet(num_heads=2, head_dim=8)",
    "gla": "gla(num_heads=2, head_dim=8, gate_rank=4)",
    "lightning": "lightning(num_heads=2, head_dim=8, v_head_dim=4)",
    "rwkv7": (
        "rwkv7(num_heads=2, head_dim=8, decay_rank=4, a_rank=4, v_rank=4, gate_rank=4)"
    ),
}


def model(name: str, *, draw: bool = True) -> Model:
    """Two blocks of the mixer; rwkv7's second mixes in the first's values, and with
    `draw` its output projections, which start at zero, are drawn at random."""
    first, rest = ("mix(v=@v0)", "mix(v=$v0)") if name == "rwkv7" else ("mix", "mix")
    body = f"model = embed [norm {first}] [norm ffn] [norm {rest}] [norm ffn] norm head"
    m = Model(compile(PRELUDE + f"mix = {MIXERS[name]}\n" + body), rngs=nnx.Rngs(0))
    for i, (_, node) in enumerate(nnx.iter_graph(m)):
        if draw and isinstance(node, RWKV7):
            shape = node.o.kernel.shape
            node.o.kernel[...] = jax.random.normal(jax.random.key(i), shape) / 4
    return m


def mixer(branch: nnx.Module) -> nnx.Module:
    """The mixer of a branch [norm mixer]."""
    assert isinstance(branch, ResidualBranch) and isinstance(branch.body, Sequence)
    return branch.body.items[1]


def tokens(seed: int = 0, length: int = 8) -> Int[Array, "b t"]:
    return jax.random.randint(jax.random.key(seed), (2, length), 0, V)


@pytest.mark.parametrize("name", MIXERS)
def test_runs_jitted_and_differentiates(name):
    m = model(name)
    logits = nnx.jit(lambda m, t: m(t))(m, tokens())
    assert logits.shape == (2, 8, V) and bool(jnp.isfinite(logits).all())
    grads = nnx.grad(lambda m: jnp.square(m(tokens())).mean())(m)
    assert all(bool(jnp.isfinite(g).all()) for g in jax.tree.leaves(grads))


@pytest.mark.parametrize("name", MIXERS)
def test_is_causal(name):
    m = model(name)
    t = tokens()
    changed = t.at[:, 5:].set((t[:, 5:] + 1) % V)
    assert jnp.allclose(m(t)[:, :5], m(changed)[:, :5], atol=1e-5)
    assert not jnp.allclose(m(t)[:, 5:], m(changed)[:, 5:])


@pytest.mark.parametrize("name", MIXERS)
def test_documents_packed_in_a_row_see_only_themselves(name):
    """The state, the short conv and the token shift all start again at a document."""
    m = model(name)
    a, b = tokens(0, 5), tokens(1, 7)
    positions = jnp.concatenate([jnp.arange(5), jnp.arange(7)])[None].repeat(2, 0)
    [packed] = m.outputs(jnp.concatenate([a, b], axis=1), positions)
    assert jnp.allclose(packed[:, :5], m(a), atol=1e-5)
    assert jnp.allclose(packed[:, 5:], m(b), atol=1e-5)
    assert not jnp.allclose(m(jnp.concatenate([a, b], axis=1))[:, 5:], m(b), atol=1e-3)


def _inputs(seed: int, b=2, t=6, h=3, dk=4, dv=5):
    keys = jax.random.split(jax.random.key(seed), 5)
    q, k = (jax.random.normal(key, (b, t, h, dk)) for key in keys[:2])
    v = jax.random.normal(keys[2], (b, t, h, dv))
    log_decay = -jax.random.uniform(keys[3], (b, t, h, dk), maxval=0.7)
    return q, k, v, log_decay, keys[4]


def test_the_delta_rule_decays_then_moves_the_state_toward_v_at_k():
    q, k, v, g, key = _inputs(0)
    q, k = l2norm(q), l2norm(k)
    beta = jax.nn.sigmoid(jax.random.normal(key, g.shape[:3]))
    o = delta_rule(q, k, v, g, beta, None)
    state = jnp.zeros((2, 3, 4, 5))  # b h k v
    for i in range(q.shape[1]):
        state = jnp.exp(g[:, i])[..., None] * state
        held = jnp.einsum("bhk,bhkv->bhv", k[:, i], state)
        step = beta[:, i, :, None] * (v[:, i] - held)
        state = state + k[:, i, ..., None] * step[:, :, None, :]
        expected = jnp.einsum("bhk,bhkv->bhv", q[:, i], state)
        assert jnp.allclose(o[:, i], expected, atol=1e-5)


def test_decayed_linear_attention_is_the_parallel_masked_decay_form():
    """GLA and Lightning: o_t is the sum over s <= t of (q_t exp(G_t - G_s) . k_s) v_s,
    with G the running sum of the log decay."""
    q, k, v, g, _ = _inputs(1)
    G = jnp.cumsum(g, axis=1)
    decay = jnp.exp(G[:, :, None] - G[:, None, :])  # b t s h k
    causal = jnp.tril(jnp.ones((6, 6)))[None, :, :, None, None]
    expected = jnp.einsum("bthk,btshk,bshk,bshv->bthv", q, decay * causal, k, v)
    assert jnp.allclose(recurrence(q, k, v, g, None), expected, atol=1e-4)


def test_rwkv7_is_its_kernels_recurrence():
    """RWKV-LM's kernel keeps the state as v by k: sa = S a, S <- S diag(w) + sa b^T +
    v k^T, and y = S r."""
    r, k, v, log_w, key = _inputs(2)
    a, b = (0.5 * jax.random.normal(key_, k.shape) for key_ in jax.random.split(key))
    o = recurrence(r, k, v, log_w, None, a=a, b=b)
    state = jnp.zeros((2, 3, 5, 4))  # b h v k
    for i in range(r.shape[1]):
        sa = jnp.einsum("bhvk,bhk->bhv", state, a[:, i])
        state = (
            state * jnp.exp(log_w[:, i])[:, :, None, :]
            + sa[..., None] * b[:, i, :, None, :]
            + v[:, i, ..., None] * k[:, i, :, None, :]
        )
        expected = jnp.einsum("bhvk,bhk->bhv", state, r[:, i])
        assert jnp.allclose(o[:, i], expected, atol=1e-5)


def test_grouped_value_heads_share_each_qk_head():
    """With twice the value heads of q/k heads, Gated DeltaNet is the one with a q/k
    head for every value head, each a copy of its group's."""

    def gdn(heads: int, seed: int) -> GatedDeltaNet:
        c = cfg.GatedDeltaNetConfig(
            num_heads=heads, num_v_heads=4, head_dim=8, v_head_dim=4
        )
        return GatedDeltaNet(c, D, nnx.Rngs(seed))

    gva, mha = gdn(2, 0), gdn(4, 1)

    def per_value_head(w):  # q/k channels, head by head, each twice
        heads = w.reshape(*w.shape[:-1], 2, 8)
        return jnp.repeat(heads, 2, axis=-2).reshape(*w.shape[:-1], -1)

    mha.q.kernel[...] = per_value_head(gva.q.kernel[...])
    mha.k.kernel[...] = per_value_head(gva.k.kernel[...])
    assert gva.conv is not None and mha.conv is not None
    q, k, v = jnp.split(gva.conv.weight[...], [16, 32], -1)
    mha.conv.weight[...] = jnp.concatenate(
        [per_value_head(q), per_value_head(k), v], -1
    )
    for name in ("v", "a", "b", "gate", "norm", "o", "A_log", "dt_bias"):
        setattr(mha, name, getattr(gva, name))
    x = jax.random.normal(jax.random.key(0), (2, 6, D))
    assert jnp.allclose(mha(x, {}), gva(x, {}), atol=1e-5)


def test_kda_decay_is_bounded_below_with_a_lower_bound():
    c = cfg.KDAConfig(num_heads=2, head_dim=8, gate_lower_bound=-5)
    kda = KimiDeltaAttention(c, D, nnx.Rngs(0))
    g = kda._log_decay(3 * jax.random.normal(jax.random.key(0), (2, 6, D)))
    assert g.shape == (2, 6, 2, 8)
    assert bool(((g >= -5) & (g <= 0)).all())
    assert float(g.min()) < -4.5  # it reaches toward the bound
    assert (kda.A_log[...] == 0).all()


def test_the_output_norm_eps_is_a_setting():
    """Kimi Linear normalizes KDA's heads with eps 1e-5, Qwen3-Next its GDN's with 1e-6,
    and RWKV-7's group norm has its eps times the head dim."""
    kda = cfg.KDAConfig(num_heads=2, head_dim=8, norm_eps=1e-5)
    gdn = cfg.GatedDeltaNetConfig(num_heads=2, head_dim=8)
    rwkv = cfg.RWKV7Config(num_heads=2, head_dim=8, v_head_dim=4)
    norms = [
        KimiDeltaAttention(kda, D, nnx.Rngs(0)).norm,
        GatedDeltaNet(gdn, D, nnx.Rngs(0)).norm,
        RWKV7(rwkv, D, (), nnx.Rngs(0)).norm,
    ]
    assert [n.eps for n in norms] == pytest.approx([1e-5, 1e-6, 4e-5])


def test_lightning_decays_by_minimax_slopes_weakening_with_depth():
    # MiniMax's rates for 8 heads, and the factors of layers 0, 1 and 2 of 3
    base = [2.0**-i for i in range(1, 9)]
    for layer in range(3):
        factor = 1 - layer / (2 + 1e-5) + 1e-5
        rates = [r * factor for r in base]
        assert jnp.allclose(slopes(8, layer, 3), jnp.asarray(rates), rtol=1e-6)
    # In a model, the index counts every token mixer before it, softmax ones too.
    m = Model(
        compile(
            PRELUDE + "mix = lightning(num_heads=2, head_dim=8)\n"
            "model = embed [norm mix] [norm attn] ([norm mix] [norm ffn])^2 norm head"
        ),
        rngs=nnx.Rngs(0),
    )
    assert isinstance(m.graph, Sequence)
    repeat = m.graph.items[3]
    assert isinstance(repeat, Sequence)
    blocks = [block for block in repeat.items if isinstance(block, Sequence)]
    branches = [m.graph.items[1], *(block.items[0] for block in blocks)]
    for layer, branch in zip([0, 2, 3], branches, strict=True):
        found = mixer(branch)
        assert isinstance(found, LightningAttention)
        assert jnp.allclose(found.slopes[...], slopes(2, layer, 4))


def test_rwkv7_starts_by_depth_and_reads_values_only_through_its_port():
    m = model("rwkv7", draw=False)
    assert isinstance(m.graph, Sequence)
    first, later = (mixer(m.graph.items[i]) for i in (1, 3))
    assert isinstance(first, RWKV7) and isinstance(later, RWKV7)
    assert first.v_mix is None and later.v_mix is not None
    # Layer 0 of 2 shifts its first channel fully onto the previous token; deeper,
    # every channel moves less (RWKV-LM's x_r, ..., x_g).
    assert (first.shift[...][:, 0] == 1).all()
    assert (later.shift[...] <= first.shift[...] + 1e-6).all()
    assert (first.o.kernel[...] == 0).all()  # the output projection starts at zero
    x = jax.random.normal(jax.random.key(0), (2, 6, D))
    assert jnp.allclose(first(x, {}), 0)
