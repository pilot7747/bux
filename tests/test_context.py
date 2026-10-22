"""Context parallelism for attention (bux.parallel.context): every method of
`methods.attention` against `xla` on one device, with the positions split over
`context` on the 8 CPU devices the tests have."""

import importlib.util
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.extend.core import Jaxpr
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from bux import parallel
from bux.architecture import Model
from bux.lang import compile
from bux.parallel import methods
from bux.parallel.context import AttentionArgs
from bux.train import config as tcfg
from bux.train.data import Batch
from bux.train.objective import objective
from bux.train.trainer import Trainer

TOKAMAX = pytest.mark.skipif(
    importlib.util.find_spec("tokamax") is None,
    reason="tokamax is the cuda extra's (Linux); scripts/gpu_smoke/attention.py",
)
METHODS = ["ring", "ulysses", pytest.param("tokamax", marks=TOKAMAX)]
# Each context size with another axis: heads over tensor, or rows over fsdp.
MESHES = {
    2: tcfg.MeshConfig(context=2, tensor=2),
    4: tcfg.MeshConfig(context=4, fsdp=2),
}
B, T, D = 2, 16, 8
# Three documents in the first row, cut mid-way, and one in the second.
POSITIONS = np.asarray([[*range(3, 8), *range(7), *range(4)], [*range(T)]])


def documents(positions: Int[np.ndarray, "b t"]) -> Int[Array, "b t"]:
    """The documents' ids of rows at `positions`, from 1, as Segments makes them."""
    start = np.concatenate(
        [np.ones((len(positions), 1), bool), np.diff(positions) <= 0], 1
    )
    return jnp.asarray(np.cumsum(start, 1), jnp.int32)


IDS = documents(POSITIONS)


class Case(NamedTuple):
    heads: int
    kv_heads: int
    ids: Int[Array, "b t"] | None = None
    window: int | None = None
    softcap: float | None = None
    sinks: bool = False

    def visibility(self, sinks: Float[Array, " h"]) -> AttentionArgs:
        return AttentionArgs(
            self.ids, self.window, D**-0.5, self.softcap, sinks if self.sinks else None
        )


# Documents, windows smaller and larger than a block of 4 positions, soft-capped
# logits and sinks, with GQA, MHA and MQA.
CASES = {
    "gqa_documents": Case(4, 2, IDS),
    "mha_window": Case(4, 4, IDS, window=3),
    "mqa_softcap_window": Case(8, 1, window=6, softcap=2.0),
    "gqa_sinks": Case(4, 2, IDS, window=5, sinks=True),
}


def attention(case: Case) -> tuple[Float[Array, "..."], ...]:
    """Queries, keys, values, sinks, and a cotangent for the output."""
    h, g = case.heads, case.kv_heads
    key = jax.random.split(jax.random.key(0), 5)
    return (
        jax.random.normal(key[0], (B, T, h, D)),
        jax.random.normal(key[1], (B, T, g, D)),
        jax.random.normal(key[2], (B, T, g, D + 2)),
        jax.random.normal(key[3], (h,)),
        jax.random.normal(key[4], (B, T, h, D + 2)),
    )


def attend(
    method: str, case: Case
) -> tuple[Float[Array, "b t h v"], tuple[Float[Array, "..."], ...]]:
    """The output of `method` on `case`, and its gradients for q, k, v and sinks."""
    q, k, v, sinks, cotangent = attention(case)
    f = methods.attention.find(method)

    def loss(q, k, v, sinks):
        out = f(q, k, v, case.visibility(sinks))
        return (out * cotangent).sum(), out

    grad = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3), has_aux=True))
    grads, out = grad(q, k, v, sinks)
    return out, grads


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
@pytest.mark.parametrize("context", MESHES)
@pytest.mark.parametrize("method", METHODS)
def test_every_method_attends_and_differentiates_as_xla_does(method, context, case):
    if method == "tokamax" and case.sinks:
        pytest.skip("tokamax has no sinks")
    out, grads = attend("xla", case)
    with parallel.use(MESHES[context]):
        split_out, split_grads = attend(method, case)
    np.testing.assert_allclose(split_out, out, atol=1e-5)
    for split, whole in zip(split_grads, grads):
        np.testing.assert_allclose(split, whole, atol=1e-5, rtol=1e-5)


def exponents(jaxpr: Jaxpr) -> list[str]:
    """The dtype of every exp in `jaxpr` and the jaxprs inside it: float32, bfloat16."""
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "exp":
            found.append(str(eqn.invars[0].aval).split("[")[0])
        for param in eqn.params.values():
            for inner in param if isinstance(param, tuple | list) else [param]:
                inner = getattr(inner, "jaxpr", inner)
                if isinstance(inner, Jaxpr):
                    found += exponents(inner)
    return found


@pytest.mark.parametrize("method", ["xla", *METHODS])
def test_every_method_takes_its_softmax_in_float32(method):
    """Forward and backward, on bfloat16 queries, keys and values."""
    case = CASES["gqa_documents"]._replace(softcap=5.0)
    q, k, v, sinks, cotangent = (x.astype(jnp.bfloat16) for x in attention(case))
    f = methods.attention.find(method)

    def loss(q, k, v):
        return (f(q, k, v, case.visibility(sinks)) * cotangent).sum()

    with parallel.use(MESHES[2]):
        jaxpr = jax.make_jaxpr(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v).jaxpr
    found = exponents(jaxpr)
    assert found and set(found) == {"float32"}


def test_ring_sends_keys_round_and_gathers_none():
    """Only the documents' ids are gathered; the keys and values go round."""
    case = CASES["gqa_documents"]
    q, k, v, sinks, _ = attention(case)
    seen = case.visibility(sinks)
    with parallel.use(tcfg.MeshConfig(context=4, fsdp=2)) as mesh:
        put = NamedSharding(mesh, P("fsdp", "context"))
        q, k, v = (jax.device_put(x, put) for x in (q, k, v))
        f = jax.jit(lambda q, k, v: methods.attention.find("ring")(q, k, v, seen))
        text = f.lower(q, k, v).compile().as_text()
    assert text is not None and "collective-permute" in text
    gathered = [line for line in text.splitlines() if " all-gather(" in line]
    assert gathered and all(" s32[" in line.split("=")[1] for line in gathered)


def test_names_inside_a_shard_map_leave_out_the_axes_it_maps():
    names = ("batch", "seq", "heads")

    def inside(x: Float[Array, "b t h"]) -> Float[Array, "b t h"]:
        assert parallel.spec(names, x.shape) == P("data", None, "tensor")
        return parallel.shard(x, *names)

    with parallel.use(tcfg.MeshConfig(context=2, tensor=2)):
        x = jnp.ones((2, 8, 4))
        assert parallel.spec(names, x.shape) == P("data", "context", "tensor")
        positions = P(None, "context")
        f = jax.shard_map(
            inside, in_specs=positions, out_specs=positions, axis_names={"context"}
        )
        jax.jit(f)(x)


def test_ulysses_needs_context_to_divide_the_heads_of_a_device():
    q, k, v, *_ = attention(Case(4, 2))
    seen = Case(4, 2).visibility(jnp.zeros(4))
    f = jax.jit(lambda q, k, v: methods.attention.find("ulysses")(q, k, v, seen))
    error = pytest.raises(ValueError, match="context = 4 does not divide the 2 heads")
    with parallel.use(tcfg.MeshConfig(context=4, tensor=2)), error:
        f(q, k, v)


# --- In a model --------------------------------------------------------------------

PRELUDE = """
d_model = 32
vocab_size = 64
norm = rmsnorm
ffn = gated_mlp(intermediate_size=48, activation=swish)
"""
# Sinks and a soft cap, but tokamax has no sinks.
LAYER = {
    "ring": "gqa(num_heads=4, num_kv_heads=2, rope=rope(), sliding_window_size=6, "
    "sinks=true, softcap=20)",
    "ulysses": "gqa(num_heads=4, num_kv_heads=2, rope=rope(), sinks=true, softcap=20)",
    "tokamax": "gqa(num_heads=4, num_kv_heads=2, rope=rope(), sliding_window_size=6)",
}


def source(layer: str) -> str:
    return (
        PRELUDE + f"a = {layer}\nmodel = embed [norm a] [norm ffn] [norm a] norm head"
    )


def packed(step: int) -> Batch:
    """8 rows of 16 tokens, each of documents of random lengths, cut mid-way."""
    rng = np.random.default_rng(step)
    positions = []
    for _ in range(8):
        lengths = rng.integers(1, 9, 16)
        row = np.concatenate(
            [
                np.arange(int(n)) + (int(rng.integers(5)) if i == 0 else 0)
                for i, n in enumerate(lengths)
            ]
        )
        positions.append(row[:16])
    ids = rng.integers(0, 64, (8, 16), np.int32)
    return Batch(ids, np.asarray(positions, np.int32), np.ones((8, 16), bool))


def packed_losses(layer: str, mesh: tcfg.MeshConfig, method: str) -> list[float]:
    """Three steps of training on packed rows: the losses."""
    arch = compile(source(layer))
    with parallel.use(mesh, methods=tcfg.MethodsConfig(attention=method)):
        model = parallel.build(lambda: Model(arch, rngs=nnx.Rngs(0)))
        trainer = Trainer(model, optax.adamw(1e-2), objective(tcfg.LossConfig(), arch))
        return [
            float(trainer.train_on_batch(packed(step))["loss"]) for step in range(3)
        ]


def trains_the_same(losses, involuntary, layer: str, method: str) -> None:
    """Assert that `layer` trains as on one device by `method`, and that XLA
    rematerializes no array in full, on context=2 and 4."""
    one = losses(source(layer), mesh="mesh(data=1)")
    for mesh in ("mesh(context=2, tensor=2)", "mesh(context=4, fsdp=2)"):
        involuntary()
        split = losses(source(layer), mesh=mesh, methods=f"methods(attention={method})")
        assert np.allclose(split, one, atol=1e-5), mesh
        assert involuntary() == 0, mesh


@pytest.mark.parametrize("method", METHODS)
def test_a_run_trains_the_same_by_each_method_and_rematerializes_nothing(
    losses, involuntary, method
):
    trains_the_same(losses, involuntary, LAYER[method], method)


# MLA, and CSA's window alone: one kv head that is its keys and its values, and sinks.
OTHERS = {
    "mla": "mla(num_heads=4, q_lora_rank=12, kv_lora_rank=8, qk_nope_head_dim=6, "
    "qk_rope_head_dim=4, v_head_dim=5)",
    "csa": "csa(num_heads=4, head_dim=16, q_lora_rank=12, o_lora_rank=6, o_groups=2, "
    "sliding_window_size=3, sinks=true, rope=rope(fraction=0.25, layout=interleaved))",
}


@pytest.mark.parametrize("layer", OTHERS.values(), ids=OTHERS.keys())
@pytest.mark.parametrize("method", ["ring", "ulysses"])
def test_mla_and_csa_train_the_same_by_ring_and_ulysses(
    losses, involuntary, method, layer
):
    trains_the_same(losses, involuntary, layer, method)


@pytest.mark.parametrize("method", METHODS)
def test_packed_documents_train_the_same_by_each_method(method):
    one = packed_losses(LAYER[method], tcfg.MeshConfig(data=1), "xla")
    for mesh in MESHES.values():
        assert np.allclose(packed_losses(LAYER[method], mesh, method), one, atol=1e-5)


# What only xla has, each with the name its error gives it.
ONLY_XLA = {
    "relative_bias": "gqa(num_heads=4, num_kv_heads=2, relative_bias=relbias(rank=2, extent=4))",
    "log_scale": "gqa(num_heads=4, num_kv_heads=2, log_scale=0.1)",
    "attention_dropout": "gqa(num_heads=4, num_kv_heads=2, attention_dropout=0.1)",
    "top-k": (
        "gqa(num_heads=4, num_kv_heads=2, indexer=indexer(num_heads=2, head_dim=8, top_k=3))"
    ),
    "CSA": "csa(num_heads=4, head_dim=8, q_lora_rank=12, compress_ratio=2)",
}


@pytest.mark.parametrize(("feature", "layer"), ONLY_XLA.items(), ids=ONLY_XLA.keys())
def test_what_only_xla_has_is_an_error_under_another_method(losses, feature, layer):
    with pytest.raises(NotImplementedError, match=f"ring has no .*{feature}"):
        losses(source(layer), mesh="mesh(context=2)", methods="methods(attention=ring)")


def test_tokamax_has_no_sinks():
    q, k, v, sinks, _ = attention(Case(4, 2))
    seen = Case(4, 2, sinks=True).visibility(sinks)
    match = "tokamax has no sinks|tokamax needs tokamax"
    with pytest.raises(NotImplementedError, match=match):
        methods.attention.find("tokamax")(q, k, v, seen)
