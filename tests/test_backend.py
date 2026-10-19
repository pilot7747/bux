import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from bux.backend import Model, num_params
from bux.lang import compile

D, V, H, F = 32, 64, 4, 48

PRELUDE = f"""
d_model = {D}
vocab_size = {V}
norm = rmsnorm
attn = gqa(num_heads={H}, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size={F}, activation=swish)
"""


def model(body: str) -> Model:
    return Model(compile(PRELUDE + body), rngs=nnx.Rngs(0))


def tokens(seed: int = 0, length: int = 8) -> jax.Array:
    return jax.random.randint(jax.random.key(seed), (2, length), 0, V)


def test_llama_forward_and_parameter_count():
    m = model("block = [norm attn] [norm ffn]\nmodel = embed block^2 norm head")
    logits = nnx.jit(lambda m, t: m(t))(m, tokens())
    assert logits.shape == (2, 8, V) and bool(jnp.isfinite(logits).all())
    hd = D // H
    attention = D * H * hd + 2 * D * 2 * hd + H * hd * D
    block = D + attention + D + 3 * D * F
    assert num_params(m) == V * D + 2 * block + D + D * V


def test_attention_is_causal():
    m = model("model = embed [norm attn] [norm ffn] norm head")
    t = tokens()
    changed = t.at[:, 5:].set((t[:, 5:] + 1) % V)
    assert jnp.allclose(m(t)[:, :5], m(changed)[:, :5], atol=1e-5)
    assert not jnp.allclose(m(t)[:, 5:], m(changed)[:, 5:])


def test_sharing_and_tying_reduce_parameters():
    fresh = model("blk = [norm attn] [norm ffn]\nmodel = embed blk^4 norm head")
    shared = model("shared blk = [norm attn] [norm ffn]\nmodel = embed blk^4 norm head")
    tied = model(
        "blk = [norm attn] [norm ffn]\nmodel = embed blk^4 norm head(tied=true)"
    )
    one_block = (num_params(fresh) - V * D - D - D * V) // 4
    assert num_params(fresh) - num_params(shared) == 3 * one_block
    assert num_params(fresh) - num_params(tied) == D * V


@pytest.mark.parametrize(
    "body",
    [
        # U-net skips and x0 mixing
        (
            "blk = (scale + $x0 scale) [norm attn] [norm ffn]\n"
            "model = embed @x0 (blk @skip)^2 ([$skip! scale] blk)^2 norm head"
        ),
        # value residual and cross-layer KV sharing through ports
        (
            "model = embed [norm attn(v=@v1)] [norm attn(v=$v1)]"
            " [norm attn(kv=@kv)] [norm attn(kv=$kv)] norm head"
        ),
        # parallel block, gating, MoE, nGPT-style subtraction
        (
            "e = moe(num_experts=4, top_k=2, num_shared_experts=1, expert=ffn)\n"
            "model = embed [norm (attn + e)] [norm attn * (linear(features=32) sigmoid)]"
            " [(ffn norm - _) scale(per_channel=true)] norm head"
        ),
    ],
    ids=["unet", "ports", "parallel"],
)
def test_wiring_runs(body):
    logits = model(body)(tokens())
    assert logits.shape == (2, 8, V) and bool(jnp.isfinite(logits).all())


def test_unregistered_layers_say_how_to_add_them():
    with pytest.raises(NotImplementedError, match=r"register\('gdn'\)"):
        model("model = embed [norm gdn(num_heads=2, head_dim=8)] head")
