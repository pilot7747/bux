"""Tensor parallelism's projections by `methods.tensor`: `chunked` against `gspmd` on
the same mesh, and against one device, on the 8 CPU devices the tests have."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jaxtyping import Array, Float, Int

from bux import parallel
from bux.architecture import Model
from bux.architecture.precision import matmuls
from bux.lang import compile
from bux.train import config as tcfg

PRELUDE = """
d_model = 64
vocab_size = 128
norm = rmsnorm
ffn = gated_mlp(intermediate_size=256, activation=swish)
"""
# Heads of 32, so that a device's share of each projection is whole blocks of MXFP8.
LLAMA = (
    PRELUDE
    + "attn = gqa(num_heads=8, num_kv_heads=2, head_dim=32, rope=rope(), bias=true)\n"
    + "block = [norm attn] [norm ffn]\nmodel = embed block^2 norm head"
)
# Its width whole blocks on every device too, so that a GEMM keeps it split.
WIDE = LLAMA.replace("d_model = 64", "d_model = 256")
MOE = (
    PRELUDE
    + "attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())\n"
    + "experts = moe(num_experts=8, top_k=2, num_shared_experts=1, "
    + "expert=gated_mlp(intermediate_size=64, activation=swish))\n"
    + "model = embed [norm attn] [norm experts] [norm attn] [norm ffn] norm head"
)
# Mesh sizes and layout. On tensor=8 the 2 kv heads are every device's, and the 8
# heads' output projection splits by heads of 8 columns.
MESHES = {
    "fsdp=2, tensor=4, sequence parallel": ({"fsdp": 2, "tensor": 4}, "tensor"),
    "tensor=8, sequence parallel": ({"tensor": 8}, "tensor"),
    "context=2, tensor=2, the stream over both": (
        {"fsdp": 2, "context": 2, "tensor": 2},
        ["context", "tensor"],
    ),
    "data=4, tensor=2": ({"data": 4, "tensor": 2}, "context"),
}
ONE = {"one device": ({"data": 1}, "context")}
MX = tcfg.PrecisionConfig(matmuls="mxfp8")
# Precisions; the branches of LLAMA (of 4) that chunked splits into chunks, as a
# projection whose share would multiply otherwise than the whole stays gspmd's; and
# how near its gradients come. Where NVFP4 takes a gradient that a chunked branch
# summed otherwise in its last bits, its 2-bit mantissa moves some values a step.
PRECISIONS = {
    "compute": (tcfg.PrecisionConfig(), 4, 1e-5),
    "mxfp8": (MX, 4, 1e-5),
    "mxfp8_2d": (tcfg.PrecisionConfig(matmuls="mxfp8_2d"), 4, 1e-5),
    "nvfp4": (tcfg.PrecisionConfig(matmuls="nvfp4"), 0, 1e-5),
    "mxfp8, the MLPs nvfp4": (
        MX.model_copy(update={"roles": tcfg.RolesConfig(mlp="nvfp4")}),
        2,
        1e-3,
    ),
    "nvfp4, compute first and last": (
        tcfg.PrecisionConfig(matmuls="nvfp4", first=1, last=1),
        2,
        1e-3,
    ),
}
COLLECTIVES = ("all-gather", "reduce-scatter", "all-reduce")


def _tokens() -> Int[np.ndarray, "b t"]:
    return np.random.default_rng(0).integers(0, 128, (8, 32))


def _loss(model: Model, tokens: Int[Array, "b t"]) -> Float[Array, ""]:
    logits = model(tokens).astype(jnp.float32)
    return jnp.mean(logits**2) + jnp.mean(logits[..., :3])


@functools.cache
def _outputs(
    method: str,
    mesh: str,
    source: str = LLAMA,
    precision: str = "compute",
    chunks: int = 2,
):
    """The logits and every parameter's gradient of a loss by `method` on `mesh`, in
    one of the PRECISIONS, in `chunks` chunks under `chunked`, on the host; and the
    collectives the gradient makes."""
    sizes, stream = (MESHES | ONE)[mesh]
    with (
        parallel.use(
            tcfg.MeshConfig.model_validate(sizes),
            tcfg.LayoutConfig.model_validate({"stream": stream}),
            tcfg.MethodsConfig(tensor=method, tensor_chunks=chunks),
        ),
        matmuls.use(PRECISIONS[precision][0]),
    ):
        model = parallel.build(lambda: Model(compile(source), rngs=nnx.Rngs(0)))
        tokens = parallel.shard_batch(_tokens())
        logits = nnx.jit(lambda m, t: m(t))(model, tokens)
        grad = nnx.jit(nnx.grad(_loss))
        grads = grad(model, tokens)
        hlo = grad.lower(model, tokens).compile().as_text() or ""
    counts = {
        op: hlo.count(f" {op}(") + hlo.count(f" {op}-start(") for op in COLLECTIVES
    }
    return jax.device_get((logits, nnx.to_pure_dict(grads))), counts


def assert_close(a, b, tolerance: float = 1e-5) -> None:
    def close(x: Float[np.ndarray, "*s"], y: Float[np.ndarray, "*s"]) -> bool:
        return np.allclose(x, y, atol=tolerance * max(1.0, np.abs(y).max()))

    assert jax.tree.all(jax.tree.map(close, a, b))


@pytest.mark.parametrize("chunks", [1, 2, 4])
@pytest.mark.parametrize("mesh", MESHES)
def test_chunked_projections_compute_what_gspmd_s_do_with_the_same_gradients(
    mesh, chunks
):
    chunked, _ = _outputs("chunked", mesh, chunks=chunks)
    assert_close(chunked, _outputs("gspmd", mesh)[0])
    assert_close(chunked, _outputs("gspmd", "one device")[0])


# Chunks of a device's 4 rows: 8 make 4, and 3 one, as they must divide them.
@pytest.mark.parametrize(("chunks", "made"), [(1, 1), (2, 2), (3, 1), (4, 4), (8, 4)])
def test_chunked_reduces_each_projection_a_chunk_of_rows_at_a_time(chunks, made):
    """2 blocks, each with an attention and an MLP, each reduce-scattered forward and
    its gathered input's gradient backward, a chunk at a time."""
    mesh = "fsdp=2, tensor=4, sequence parallel"
    _, gspmd = _outputs("gspmd", mesh)
    _, chunked = _outputs("chunked", mesh, chunks=chunks)
    assert gspmd["reduce-scatter"] == 0  # XLA:CPU's are all-reduces
    assert chunked["reduce-scatter"] == 2 * 2 * 2 * made


@pytest.mark.parametrize("precision", PRECISIONS)
def test_each_projection_by_chunks_multiplies_as_it_does_whole_in_its_format(
    precision,
):
    """MXFP8's blocks are a device's share's, chunk by chunk, MXFP8 2D's tiles too;
    NVFP4 scales by the whole operand's largest value, so its projections stay
    gspmd's, by the format of their role and their branch."""
    mesh = "fsdp=2, tensor=4, sequence parallel"
    _, branches, tolerance = PRECISIONS[precision]
    (logits, grads), counts = _outputs("chunked", mesh, WIDE, precision)
    (whole_logits, whole_grads), _ = _outputs("gspmd", mesh, WIDE, precision)
    assert_close(logits, whole_logits)
    assert_close(grads, whole_grads, tolerance)
    # Each chunked branch reduce-scatters its output forward and its input's gradient
    # backward, in 2 chunks.
    assert counts["reduce-scatter"] == branches * 2 * 2


# Scanned and rematerialized, as a large run is; the MoE in one microbatch, as more
# would change its balance loss over the batch.
RUNS = {
    "llama": (LLAMA, "repeat=scan, remat=full, microbatches=2"),
    "moe": (MOE, "remat=full"),
}


@pytest.mark.parametrize("run", RUNS)
def test_a_run_trains_the_same_by_chunked_projections(losses, involuntary, run):
    source, how = RUNS[run]
    one = losses(source, mesh="mesh(data=1)", methods=f"methods({how})")
    involuntary()
    chunked = losses(
        source,
        mesh="mesh(fsdp=2, tensor=4)",
        layout="layout(stream=tensor)",
        methods=f"methods(tensor=chunked, {how})",
    )
    assert involuntary() == 0
    assert np.allclose(chunked, one, atol=1e-5)
