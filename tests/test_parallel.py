"""bux.parallel's core: the mesh, names and their layout, FSDP, and runs split over the
8 CPU devices the tests have, against one device."""

import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from bux import parallel
from bux.architecture import MLP, Model
from bux.lang import compile, compile_runs
from bux.lang.printer import run_to_source
from bux.train import config as tcfg
from bux.train.data import plain
from bux.train.objective import objective
from bux.train.optim import lr_multiplier, optimizer
from bux.train.trainer import Trainer, directory, train

ROOT = Path(__file__).parent.parent
PRELUDE = """
d_model = 64
vocab_size = 128
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=256, activation=swish)
"""
LLAMA = PRELUDE + "block = [norm attn] [norm ffn]\nmodel = embed block^2 norm head"
ARCHITECTURES = {
    "llama": LLAMA,
    "moe": PRELUDE
    + "experts = moe(num_experts=8, top_k=2, num_shared_experts=1, "
    + "expert=gated_mlp(intermediate_size=64, activation=swish))\n"
    + "model = embed [norm attn] [norm experts] [norm attn] [norm ffn] norm head",
    "hc": PRELUDE
    + "residual = hc(expansion=2, dynamic=true)\n"
    + "model = embed [norm attn] [norm ffn] merge norm head",
    "attnres": PRELUDE
    + "residual = attnres(block_size=2)\n"
    + "model = embed [norm attn] [norm ffn] [norm attn] merge norm head",
    "nextn": PRELUDE
    + "mtp = $h! nextn [norm ffn] norm @h head\n"
    + "model = embed [norm attn] [norm ffn] norm @h head mtp^2",
}
MESHES = [
    "mesh(data=8)",
    "mesh(fsdp=8)",
    "mesh(data=2, fsdp=4)",
    "mesh(fsdp=2, expert=2, tensor=2)",
]
ONE = "mesh(data=1)"


def use(**sizes: int):
    return parallel.use(tcfg.MeshConfig.model_validate(sizes))


# --- Runs split over devices -----------------------------------------------------------


@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_a_run_trains_the_same_split_over_devices(losses, arch, mesh):
    source = ARCHITECTURES[arch]
    assert np.allclose(losses(source, mesh=mesh), losses(source, mesh=ONE), atol=1e-5)


@pytest.mark.parametrize("mesh", MESHES)
def test_muon_trains_the_same_split_over_devices(losses, mesh):
    muon = "muon(lr=0.02, ns_dtype=float32)"  # bfloat16 rounds a split apart
    one = losses(ARCHITECTURES["moe"], mesh=ONE, optimizer=muon)
    split = losses(ARCHITECTURES["moe"], mesh=mesh, optimizer=muon)
    assert np.allclose(split, one, atol=1e-5)


def test_fsdp_alone_needs_no_full_rematerialization(losses, involuntary):
    involuntary()
    losses(LLAMA, steps=1, mesh="mesh(fsdp=8)")
    assert involuntary() == 0


def test_a_step_keeps_the_state_where_it_was():
    run = _run(LLAMA, "mesh = mesh(fsdp=4, tensor=2)", "layout = layout(zero=1)")
    t = run.train
    with parallel.use(t.mesh, t.layout, t.methods):
        trainer = _trainer(run)
        before = jax.tree.leaves(trainer.state())
        trainer.train_on_batch(_batch())
        after = jax.tree.leaves(trainer.state())
    assert all(
        a.sharding.is_equivalent_to(b.sharding, a.ndim) for a, b in zip(after, before)
    )


# --- Parameters ------------------------------------------------------------------------


def test_build_makes_what_eager_init_does_and_no_parameter_whole():
    arch = compile(ARCHITECTURES["moe"])
    eager = nnx.state(Model(arch, rngs=nnx.Rngs(0)))
    with use(fsdp=8):
        model = parallel.build(lambda: Model(arch, rngs=nnx.Rngs(0)))
    built = nnx.state(model)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, eager, built))
    for x in jax.tree.leaves(nnx.state(model, nnx.Param)):
        assert x.addressable_shards[0].data.size < x.size


class Toy(nnx.Module):
    def __init__(self, heads: int, rngs: nnx.Rngs):
        self.up = nnx.Linear(
            16, 64, kernel_init=parallel.named("embed", "mlp"), rngs=rngs
        )
        self.sinks = nnx.Param(jnp.zeros(heads), names=("heads",))


def _toy_specs(heads: int, **sizes: int) -> dict:
    with use(**sizes):
        state = nnx.state(parallel.build(lambda: Toy(heads, nnx.Rngs(0))))
        found = parallel.shardings(state)
    return {
        "up.kernel": found["up"]["kernel"].spec,
        "up.bias": found["up"]["bias"].spec,
        "sinks": found["sinks"].spec,
    }


def test_names_split_over_their_axes_and_fsdp_the_largest_dimension_left():
    specs = _toy_specs(heads=4, fsdp=2, tensor=4)
    assert specs["up.kernel"] == P("fsdp", "tensor")
    assert specs["up.bias"] == P("fsdp")  # unnamed: FSDP alone
    assert specs["sinks"] == P("tensor")


def test_a_dimension_its_axes_do_not_divide_stays_whole():
    assert _toy_specs(heads=6, tensor=4)["sinks"] == P(None)
    assert _toy_specs(heads=6, fsdp=2, tensor=4)["sinks"] == P("fsdp")


def test_an_unknown_name_is_an_error_that_names_the_parameter():
    with pytest.raises(ValueError, match="'hidden' is not a dimension"):
        parallel.named("embed", "hidden")
    bad = {"w": nnx.Param(jnp.zeros(4), names=("hidden",))}
    with use(tensor=2), pytest.raises(ValueError, match=r"w.*'hidden'"):
        parallel.shardings(nnx.state(nnx.Dict(bad)))


def test_zero_1_keeps_parameters_whole_and_splits_the_optimizer_state():
    run = _run(LLAMA, "mesh = mesh(fsdp=8)", "layout = layout(zero=1)")
    t = run.train
    with parallel.use(t.mesh, t.layout, t.methods):
        trainer = _trainer(run)
    mlp = next(x for _, x in nnx.iter_graph(trainer.model) if isinstance(x, MLP))
    kernel = mlp.up.kernel
    assert kernel[...].sharding.spec == P(None, None)
    moments = [
        x.sharding.spec
        for x in jax.tree.leaves(nnx.state(trainer.optimizer, nnx.OptState))
        if x.shape == kernel.shape
    ]
    assert moments and all(spec == P(None, "fsdp") for spec in moments)


def test_shard_called_eagerly_moves_an_array_onto_the_mesh():
    with use(data=2, tensor=4) as mesh:
        x = parallel.shard(jnp.ones((8, 4)), "batch", "mlp")
    assert x.sharding.is_equivalent_to(NamedSharding(mesh, P("data", "tensor")), 2)


def test_shard_does_nothing_without_a_mesh():
    x = jnp.ones((8, 4))
    assert parallel.shard(x, "batch", "embed") is x
    assert parallel.spec(("batch", None), x.shape) == P()


# --- Settings --------------------------------------------------------------------------


def test_the_mesh_takes_the_devices_the_other_axes_leave():
    assert parallel.sizes(tcfg.MeshConfig(fsdp=2, tensor=2), 8)["data"] == 2
    with pytest.raises(ValueError, match="take 3 devices, which do not divide"):
        parallel.sizes(tcfg.MeshConfig(tensor=3), 8)
    with pytest.raises(ValueError, match="needs 16 devices, but there are 8"):
        parallel.sizes(tcfg.MeshConfig(data=2, fsdp=8), 8)


def test_a_batch_that_does_not_split_is_an_error_that_names_the_setting(tmp_path):
    with pytest.raises(ValueError, match=r"batch = 4 does not split.*data=8"):
        train(_run(LLAMA, "batch = 4"), tmp_path, echo=lambda _: None)
    run = _run(LLAMA, "mesh = mesh(context=8)", "seq_len = 12")
    with pytest.raises(ValueError, match=r"seq_len = 12 does not split.*context=8"):
        train(run, tmp_path, echo=lambda _: None)


def test_a_method_not_registered_is_an_error():
    methods = tcfg.MethodsConfig(attention="mine")
    hint = r"no 'mine' attention method yet; add one with @bux\.parallel\.methods"
    with pytest.raises(NotImplementedError, match=hint):
        parallel.use(tcfg.MeshConfig(), methods=methods).__enter__()
    assert parallel.active() is None


def test_how_a_run_splits_is_not_in_its_id_but_a_sweep_over_meshes_is_two_runs():
    plain = _run(LLAMA)
    split = _run(LLAMA, "mesh = mesh(fsdp=8)", "layout = layout(seq=none)")
    assert split.id == plain.id
    assert _run(LLAMA, "precision = precision(compute=bfloat16)").id != plain.id
    a, b = compile_runs(_source(LLAMA, "mesh = dp: mesh(data=8) | fsdp: mesh(fsdp=8)"))
    assert a.id == b.id and a.name != b.name
    [again] = compile_runs(run_to_source(split))
    assert again.train.mesh.fsdp == 8 and again.train.layout.seq is None


def test_the_laptop_example_trains_every_split_to_the_same_loss(tmp_path):
    path = ROOT / "examples" / "experiments" / "parallel" / "laptop.bux"
    few = ["steps = 3", "log = log(every=1)"]
    runs = compile_runs(path.read_text(), filename=str(path), overrides=few)
    assert len(runs) == 6 and len({run.id for run in runs}) == 1
    last = [train(run, tmp_path, echo=lambda _: None)["loss"] for run in runs]
    assert np.allclose(last, last[0], atol=1e-4)


def test_a_checkpoint_restores_onto_another_mesh(tmp_path):
    def run(mesh: str):
        checkpoint = "checkpoint = checkpoint(every=2)"
        return _run(LLAMA, f"mesh = {mesh}", "steps = 4", checkpoint)

    whole = train(
        run("mesh(fsdp=2, expert=2, tensor=2)"), tmp_path, echo=lambda _: None
    )
    shutil.rmtree(directory(run(ONE), tmp_path) / "checkpoints" / "4")
    said: list[str] = []
    resumed = train(run("mesh(fsdp=8)"), tmp_path, echo=said.append)
    assert said[0].endswith("resuming at step 2")
    assert resumed["loss"] == pytest.approx(whole["loss"], abs=1e-5)


# --- Helpers ---------------------------------------------------------------------------

TRAIN = [
    "data = synthetic",
    "optimizer = adamw(lr=1e-2)",
    "seq_len = 16",
    "batch = 8",
    "steps = 3",
]


def _source(arch: str, *settings: str) -> str:
    given = {s.split("=")[0].strip(): s for s in [*TRAIN, *settings]}
    return arch + "\n" + "\n".join(given.values())


def _run(arch: str, *settings: str):
    [run] = compile_runs(_source(arch, *settings))
    return run


def _trainer(run):
    t = run.train
    model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(0)))
    tx = optimizer(t.optimizer, lr_multiplier(t.schedule, t.steps), model, t.clip)
    return Trainer(model, tx, objective(t.loss, run.arch))


def _batch():
    return plain(np.random.default_rng(0).integers(0, 128, (8, 16)))
