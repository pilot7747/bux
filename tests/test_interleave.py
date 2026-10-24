"""Microbatches that overlap (`methods.interleave`): each one's forward beside the
backward of the one before, a piece of the model at a time, against the same
microbatches one after another, on one device and split over the 8 CPU devices.

XLA:CPU computes a microbatch inside a loop to other bits than outside one, so the
gradients and metrics agree to float32's rounding; the state the forwards leave (the
router's bias and loads, RNG counts) agrees bit for bit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from bux import parallel
from bux.architecture import Model
from bux.lang import compile_runs
from bux.train import config as tcfg
from bux.train import interleave
from bux.train.data import Batch
from bux.train.objective import objective
from bux.train.optim import lr_multiplier, optimizer
from bux.train.trainer import Trainer

PRELUDE = """
d_model = 32
vocab_size = 128
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=64, activation=swish)
"""
EXPERTS = (
    "num_experts=8, top_k=2, expert=gated_mlp(intermediate_size=32, activation=swish)"
)
BLOCKS = "block = [norm attn] [norm x]\nmodel = embed block^3 norm head"
MODELS = {
    # Residual branches as the graph's items, and an unrolled repeat's passes.
    "llama": "model = embed [norm attn] [norm ffn] [norm attn] [norm ffn] norm head",
    "blocks": BLOCKS.replace("x]", "ffn]"),
    # The router's bias moves once a step, from every microbatch's loads.
    "moe": f"x = moe({EXPERTS}, load_balancing=aux_free)\n{BLOCKS}",
    # Each layer reports a balance loss, which the backward takes by its weight.
    "moe_sequence_loss": f"x = moe({EXPERTS}, aux_loss_scope=sequence)\n{BLOCKS}",
    # The head reads the embedding's table: two pieces share a parameter.
    "tied": BLOCKS.replace("x]", "ffn]").replace("norm head", "norm head(tied=true)"),
    # Dropout draws each microbatch's masks from the RNG streams in turn.
    "dropout": BLOCKS.replace("x]", "ffn]").replace("swish", "swish, dropout=0.1"),
}
TRAIN = (
    "data = synthetic\noptimizer = adamw(lr=1e-2)\nseq_len = 16\nbatch = 16\nsteps = 3"
)
ONE, FSDP = {"data": 1}, {"fsdp": 8}
TP = {"fsdp": 2, "expert": 2, "tensor": 2}
MOES = ("moe", "moe_sequence_loss")
# A model, a mesh, the microbatches its batch axes leave rows for, the MoE's method.
CASES = [
    *[(arch, ONE, 4, "sorted") for arch in MODELS],
    *[(arch, FSDP, 2, "sorted") for arch in MODELS],
    *[(arch, TP, 4, "sorted") for arch in ("llama", *MOES)],
    *[(arch, {"expert": 8}, 2, "all_to_all") for arch in MOES],
]


@pytest.mark.parametrize(("arch", "mesh", "n", "moe"), CASES)
def test_interleaved_microbatches_give_the_loop_s_gradients_and_state(
    arch, mesh, n, moe
):
    methods = tcfg.MethodsConfig(moe=moe)
    with parallel.use(tcfg.MeshConfig.model_validate(mesh), methods=methods):
        loop, both = _trainer(arch, n), _trainer(arch, n, interleave=True)
        batch = parallel.shard_batch(_documents())
        g1, m1 = nnx.jit(loop.gradients)(loop.model, batch)
        g2, m2 = nnx.jit(both.gradients)(both.model, batch)
    close = jax.tree.map(lambda a, b: jnp.allclose(a, b, atol=1e-6), g1, g2)
    assert all(jax.tree.leaves(close))
    assert m1.keys() == m2.keys()
    assert all(np.isclose(m1[k], m2[k], atol=1e-6) for k in m1)
    for a, b in zip(_others(loop.model), _others(both.model), strict=True):
        assert jnp.array_equal(a, b)


@pytest.mark.parametrize(
    "methods",
    [
        # MXFP8 but the first and the last branch, which the pieces number as the
        # whole forward does: a branch in the wrong format differs by its rounding.
        tcfg.MethodsConfig(),
        # Each branch rematerialized inside its piece, by the run's policy.
        tcfg.MethodsConfig(remat="dots"),
    ],
)
def test_interleaved_pieces_multiply_and_rematerialize_as_the_whole_model(methods):
    formats = tcfg.PrecisionConfig(matmuls="mxfp8", first=1, last=1)
    with parallel.use(tcfg.MeshConfig(data=1), methods=methods):
        loop = _trainer("blocks", 2, matmuls=formats)
        both = _trainer("blocks", 2, interleave=True, matmuls=formats)
        g1, _ = nnx.jit(loop.gradients)(loop.model, _documents())
        g2, _ = nnx.jit(both.gradients)(both.model, _documents())
    close = jax.tree.map(lambda a, b: jnp.allclose(a, b, atol=1e-6), g1, g2)
    assert all(jax.tree.leaves(close))


@pytest.mark.parametrize("arch", ["llama", "moe"])
def test_a_run_interleaved_trains_as_its_microbatches_in_a_row(losses, arch):
    source = PRELUDE + MODELS[arch]
    for mesh in ("mesh(data=1)", "mesh(fsdp=8)"):
        given = {"mesh": mesh, "batch": "16"}
        loop = losses(source, methods="methods(microbatches=2)", **given)
        both = losses(
            source, methods="methods(microbatches=2, interleave=true)", **given
        )
        assert np.allclose(both, loop, atol=1e-5)


def test_a_repeat_of_passes_alike_holds_one_microbatch_of_residuals():
    """The embedding and four passes alike: a forward's residuals go to the slot the
    tick before read, and the passes take four slots, as one microbatch."""
    embed = (((16, 32), jnp.dtype("int32")),)
    layer = (((16, 64), jnp.dtype("float32")),)
    slots = interleave.Slots([embed, *[layer] * 4], 3)
    assert slots.sizes == [2, 4]
    for before, after in zip(slots.table, slots.table[1:], strict=False):
        assert after[1:].tolist() == before[1:][::-1].tolist()
        assert after[0] != before[0]


def test_a_model_whose_pieces_pass_more_than_the_stream_is_an_error():
    source = "model = embed @h [norm attn] [$h scale] norm head"
    trainer = _trainer(None, 2, interleave=True, source=PRELUDE + source)
    with pytest.raises(ValueError, match=r"pass each other the stream alone"):
        nnx.jit(trainer.gradients)(trainer.model, _documents())


def test_a_pipeline_schedules_its_own_microbatches():
    mesh = tcfg.MeshConfig(stage=2, fsdp=4)
    methods = tcfg.MethodsConfig(repeat="scan", microbatches=2)
    source = PRELUDE + BLOCKS.replace("x]", "ffn]").replace("^3", "^4")
    with parallel.use(mesh, methods=methods), pytest.raises(ValueError, match="stage"):
        _trainer(None, 1, interleave=True, source=source)


# --- Helpers ---------------------------------------------------------------------------


def _trainer(
    arch: str | None,
    n: int,
    *,
    source: str | None = None,
    interleave: bool = False,
    matmuls: tcfg.PrecisionConfig | str = "compute",
) -> Trainer:
    text = source if source is not None else PRELUDE + MODELS[arch or "llama"]
    [run] = compile_runs(text + "\n" + TRAIN)
    t = run.train
    model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(0)))
    tx = optimizer(t.optimizer, lr_multiplier(t.schedule, t.steps), model, t.clip)
    loss = objective(t.loss, run.arch)
    return Trainer(
        model, tx, loss, matmuls=matmuls, microbatches=n, interleave=interleave
    )


def _documents() -> Batch:
    """16 rows of documents, row r's first r tokens not targets: microbatches of
    unequal numbers of targets."""
    ids = np.random.default_rng(1).integers(0, 128, (16, 16)).astype(np.int32)
    positions = np.tile(np.r_[0:5, 0:11].astype(np.int32), (16, 1))
    mask = np.arange(16)[None, :] >= np.arange(16)[:, None]
    return Batch(ids, positions, mask)


def _others(model: Model) -> list[jax.Array]:
    """The model's state but its parameters: buffers, the router's, RNG counts."""
    return jax.tree.leaves(nnx.state(model, nnx.Not(nnx.Param)))
