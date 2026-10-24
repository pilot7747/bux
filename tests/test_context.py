"""Context parallelism for attention (bux.parallel.context and ring): every method of
`methods.attention` against `xla` on one device, with the positions split over
`context` on the 8 CPU devices the tests have. A CPU runs neither cuDNN's kernel nor
tokamax's, so tokamax's is jnp's here (bux.parallel.flash.JNP), which has its
interface; scripts/gpu_smoke/attention.py runs the GPU kernels."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.extend.core import Jaxpr, Literal
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from bux import parallel
from bux.architecture import Model
from bux.lang import compile
from bux.parallel import flash, methods
from bux.parallel.context import CUMMAX_ROWS, AttentionArgs, document_starts
from bux.train import config as tcfg
from bux.train.data import Batch
from bux.train.objective import objective
from bux.train.trainer import Trainer

METHODS = ["ring", "ulysses", "tokamax", "auto"]
# Each context size with another axis, heads over tensor or rows over fsdp; and
# context=1 for the kernels on whole rows.
CONTEXTS = {
    2: tcfg.MeshConfig(context=2, tensor=2),
    4: tcfg.MeshConfig(context=4, fsdp=2),
}
MESHES = CONTEXTS | {
    8: tcfg.MeshConfig(context=8),
    1: tcfg.MeshConfig(fsdp=2, tensor=2),
}
B, T, D = 2, 16, 8
# Three documents in the first row, cut mid-way, and one in the second.
POSITIONS = np.asarray([[*range(3, 8), *range(7), *range(4)], [*range(T)]])


@pytest.fixture
def kernels(monkeypatch):
    """tokamax's kernel on a CPU: jnp's, which takes float32 too."""
    monkeypatch.setattr(flash, "TOKAMAX", flash.JNP)
    monkeypatch.setattr(flash, "tokamax_runs", lambda: True)
    monkeypatch.setattr(flash, "tokamax_takes", lambda x: True)


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
    "gqa_causal_sinks_softcap": Case(4, 2, softcap=3.0, sinks=True),
    "gqa_one_document": Case(4, 2, documents(np.tile(np.arange(T), (B, 1)))),
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
def test_every_method_attends_and_differentiates_as_xla_does(
    kernels, method, context, case
):
    if method in ("ring", "ulysses") and context == 1:
        pytest.skip("whole rows: auto's and tokamax's")
    if method == "ulysses" and case.heads % context:
        pytest.skip("context must divide the heads")
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
def test_every_method_takes_its_softmax_in_float32(kernels, method):
    """Forward and backward, on bfloat16 queries, keys and values."""
    case = CASES["gqa_sinks"]._replace(softcap=5.0)
    q, k, v, sinks, cotangent = (x.astype(jnp.bfloat16) for x in attention(case))
    f = methods.attention.find(method)

    def loss(q, k, v):
        return (f(q, k, v, case.visibility(sinks)) * cotangent).sum()

    with parallel.use(CONTEXTS[2]):
        jaxpr = jax.make_jaxpr(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v).jaxpr
    found = exponents(jaxpr)
    assert found and set(found) == {"float32"}


@pytest.mark.parametrize(("ids", "sends"), [(IDS, False), (None, True)])
def test_ring_gathers_the_keys_and_sends_only_zigzag_queries(ids, sends):
    """The keys and values are gathered, and nothing goes round; without documents or
    a window the queries go zigzag and back by collective-permutes. (The documents'
    starts, a prefix scan of their ids, send int32s round.)"""
    q, k, v, sinks, _ = attention(Case(4, 2))
    seen = Case(4, 2, ids).visibility(sinks)
    with parallel.use(tcfg.MeshConfig(context=4, fsdp=2)) as mesh:
        put = NamedSharding(mesh, P("fsdp", "context"))
        q, k, v = (jax.device_put(x, put) for x in (q, k, v))
        f = jax.jit(lambda q, k, v: methods.attention.find("ring")(q, k, v, seen))
        text = f.lower(q, k, v).compile().as_text()
    assert text is not None
    lines = text.splitlines()
    permuted = [line for line in lines if " collective-permute(" in line]
    assert any(" f32[" in line.split("=")[1] for line in permuted) == sends
    gathered = [line for line in lines if " all-gather(" in line]
    assert any(" f32[" in line.split("=")[1] for line in gathered)


def ungathered_products(jaxpr: Jaxpr, gathered: frozenset = frozenset()) -> int:
    """Count the matrix products in `jaxpr` and the jaxprs inside it that read no
    all-gather's result, `gathered` the variables that do as it starts."""
    tainted, count = set(gathered), 0

    def read(v: object) -> bool:
        return not isinstance(v, Literal) and v in tainted

    for eqn in jaxpr.eqns:
        reads = any(map(read, eqn.invars))
        count += eqn.primitive.name == "dot_general" and not reads
        for param in eqn.params.values():
            inner = getattr(param, "jaxpr", param)
            if isinstance(inner, Jaxpr) and len(inner.invars) == len(eqn.invars):
                inputs = zip(inner.invars, eqn.invars, strict=True)
                count += ungathered_products(
                    inner, frozenset(i for i, o in inputs if read(o))
                )
        if reads or eqn.primitive.name == "all_gather":
            tainted.update(eqn.outvars)
    return count


@pytest.mark.parametrize(("ids", "own"), [(IDS, 2), (None, 0)])
def test_ring_attends_to_its_own_block_while_the_others_are_gathered(ids, own):
    """Its logits and weighted values; zigzag, its own block is a gathered one."""
    q, k, v, sinks, _ = attention(Case(4, 2))
    seen = Case(4, 2, ids).visibility(sinks)
    with parallel.use(tcfg.MeshConfig(context=4, fsdp=2)):
        f = lambda q, k, v: methods.attention.find("ring")(q, k, v, seen)
        jaxpr = jax.make_jaxpr(f)(q, k, v).jaxpr
    assert ungathered_products(jaxpr) == own


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


def test_cudnns_packed_layout_has_where_each_document_starts_and_its_length():
    """Starts only, then -1 (an end there gives wrong results), as cuDNN takes them."""
    lengths, starts = flash.packed_documents(IDS, 4)
    assert lengths.tolist() == [[5, 7, 4, -1], [16, -1, -1, -1]]
    assert starts.tolist() == [[0, 5, 12, -1, -1], [0, -1, -1, -1, -1]]
    assert int(flash.documents_in(IDS)) == 3


@pytest.mark.parametrize("t", [64, CUMMAX_ROWS + 64])
def test_documents_start_where_their_ids_change_by_cummax_or_a_prefix_scan(t):
    """XLA's cummax in rows of up to CUMMAX_ROWS, a prefix scan in longer ones."""
    lengths = np.random.default_rng(t).integers(1, 50, t)
    ids = np.repeat(np.arange(t), lengths)[None, :t].astype(np.int32)  # 1-49 tokens
    new = np.ones(ids.shape, bool)
    new[:, 1:] = ids[:, 1:] != ids[:, :-1]
    want = np.maximum.accumulate(np.where(new, np.arange(t), 0), axis=1)
    assert (np.asarray(document_starts(jnp.asarray(ids))) == want).all()
    jaxpr = jax.make_jaxpr(document_starts)(ids)
    assert any(e.primitive.name == "cummax" for e in jaxpr.eqns) == (t <= CUMMAX_ROWS)


def test_the_gpu_kernels_say_what_they_need_on_a_cpu():
    q, k, v, sinks, _ = attention(Case(4, 2))
    for method, need in (("tokamax", "a GPU"), ("cudnn", "cuDNN on a GPU")):
        with pytest.raises(NotImplementedError, match=f"{method} needs.*{need}"):
            methods.attention.find(method)(q, k, v, Case(4, 2).visibility(sinks))
    seen = Case(4, 2, softcap=1.0).visibility(sinks)
    with pytest.raises(NotImplementedError, match="cudnn has no softcap"):
        methods.attention.find("cudnn")(q, k, v, seen)


# --- In a model --------------------------------------------------------------------

PRELUDE = """
d_model = 32
vocab_size = 64
norm = rmsnorm
ffn = gated_mlp(intermediate_size=48, activation=swish)
"""
LAYER = (
    "gqa(num_heads=4, num_kv_heads=2, rope=rope(), sliding_window_size=6, "
    "sinks=true, softcap=20)"
)


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
    kernels, losses, involuntary, method
):
    trains_the_same(losses, involuntary, LAYER, method)


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
def test_packed_documents_train_the_same_by_each_method(kernels, method):
    one = packed_losses(LAYER, tcfg.MeshConfig(data=1), "xla")
    for mesh in CONTEXTS.values():
        assert np.allclose(packed_losses(LAYER, mesh, method), one, atol=1e-5)


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
