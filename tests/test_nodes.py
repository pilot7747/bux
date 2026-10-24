"""Runs across nodes: the mesh over several of them, each process's rows of a batch,
and FSDP over several axes, on the 8 CPU devices the tests have, against one."""

import math
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from conftest import corpus
from flax import nnx
from jax.sharding import AbstractMesh, AxisType

from bux import parallel
from bux.architecture import Model
from bux.data.pipeline import jobs, prepare
from bux.lang import compile
from bux.parallel.layout import as_split, fsdp
from bux.parallel.mesh import grid, split
from bux.train import config as tcfg
from bux.train.data import batches, permuted

PRELUDE = """
d_model = 64
vocab_size = 128
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=256, activation=swish)
"""
DENSE = PRELUDE + "block = [norm attn] [norm ffn]\nmodel = embed block^2 norm head"
MOE = (
    PRELUDE
    + "experts = moe(num_experts=8, top_k=2, num_shared_experts=1, "
    + "expert=gated_mlp(intermediate_size=64, activation=swish))\n"
    + "model = embed [norm attn] [norm experts] [norm attn] [norm ffn] norm head"
)


def _sizes(per_node: int = 8, nodes: int = 2, **axes: int) -> dict[str, int]:
    return parallel.sizes(tcfg.MeshConfig.model_validate(axes), per_node * nodes)


def _split(**axes: int) -> list[dict[str, int]]:
    """What a node of 8 holds of `axes`, and what spans nodes: the axes above 1."""
    halves = split(_sizes(**axes), per_node=8)
    return [{a: n for a, n in half.items() if n > 1} for half in halves]


def _gpus(nodes: int, per_node: int) -> list[SimpleNamespace]:
    """GPUs as XLA lists them on `nodes` hosts, a slice each, out of order."""
    ids = np.random.default_rng(0).permutation(nodes * per_node)
    return [
        SimpleNamespace(
            id=int(i),
            slice_index=int(i) // per_node,
            process_index=int(i) // per_node,
            platform="gpu",
            device_kind="NVIDIA B300",
        )
        for i in ids
    ]


# --- The mesh --------------------------------------------------------------------------


def test_a_node_fills_from_tensor_outward_and_the_outer_axes_span_nodes():
    assert _split(fsdp=16) == [{"fsdp": 8}, {"fsdp": 2}]
    assert _split(data=2, fsdp=8) == [{"fsdp": 8}, {"data": 2}]
    assert _split(data=2, fsdp=4, tensor=2) == [{"fsdp": 4, "tensor": 2}, {"data": 2}]
    assert _split(stage=2, fsdp=2, tensor=4) == [{"fsdp": 2, "tensor": 4}, {"stage": 2}]
    assert _split(fsdp=4, expert=4) == [{"fsdp": 2, "expert": 4}, {"fsdp": 2}]


def test_axes_that_do_not_fill_nodes_from_tensor_outward_are_an_error():
    with pytest.raises(ValueError, match=r"nodes of 8 devices.*tensor=3"):
        split({**_sizes(), "data": 1, "fsdp": 4, "tensor": 3}, per_node=8)
    with pytest.raises(ValueError, match=r"nodes of 8 devices.*expert=3"):
        split({**_sizes(), "data": 1, "fsdp": 4, "expert": 3, "tensor": 4}, per_node=8)


def test_each_node_holds_a_block_of_the_inner_axes():
    shape = _sizes(per_node=4, fsdp=4, tensor=2)
    devices = grid(shape, _gpus(nodes=2, per_node=4)).squeeze()  # (fsdp, tensor)
    nodes = np.vectorize(lambda d: d.slice_index)(devices)
    assert (nodes == np.arange(4)[:, None] // 2).all()
    stages = grid(_sizes(per_node=4, stage=2, fsdp=4), _gpus(2, 4)).squeeze()
    assert (
        np.vectorize(lambda d: d.slice_index)(stages) == np.arange(2)[:, None]
    ).all()


def test_a_mesh_over_nodes_of_unequal_devices_is_an_error():
    devices = _gpus(nodes=2, per_node=4)
    with pytest.raises(ValueError, match="nodes of 4, 2"):
        grid(_sizes(per_node=3, fsdp=6), [d for d in devices if d.id < 6])


# --- Each process's rows ---------------------------------------------------------------


def _sources(folder) -> dict[str, tcfg.DataConfig]:
    for i in range(2):
        tokens = np.arange(1000, dtype=np.uint16) + 1000 * i
        (folder / f"s{i}.bin").write_bytes(tokens.tobytes())
    shards = tcfg.TokensConfig(path=str(folder / "s*.bin"))
    synthetic = tcfg.SyntheticConfig()
    mix = (tcfg.Weighted(weight=w, source=s) for w, s in ((3, shards), (1, synthetic)))
    return {
        "synthetic": synthetic,
        "tokens": shards,
        "text": tcfg.TextConfig(path=corpus(folder / "corpus")),
        "mix": tcfg.MixtureConfig(sources=tuple(mix)),
    }


@pytest.mark.parametrize("source", ["synthetic", "tokens", "text", "mix"])
def test_processes_that_read_their_own_rows_make_the_batch_one_process_does(
    tmp_path, byte_tokenizer, source
):
    data = _sources(tmp_path)[source]
    t = tcfg.TrainConfig(
        data=data,
        tokenizer=byte_tokenizer,
        packing=tcfg.BestFitPacking(chunk=2000),
        optimizer=tcfg.AdamWConfig(lr=1e-3),
        seq_len=9,
        batch=8,
        steps=1,
    )
    cache = tmp_path / "cache"
    for job in jobs(t):
        prepare(job, cache, workers=0, echo=lambda _: None)

    def read(rows: list[int] | None):
        return next(batches(data, t, 2048, cache=cache, start=3, rows=rows))

    whole = read(None)
    for rows in ([0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 5], [1, 3, 4, 6, 7]):
        part, others = read(rows), np.setdiff1d(np.arange(8), rows)
        for a, b in zip(part, whole, strict=True):
            assert (a[rows] == b[rows]).all() and not a[others].any()


def test_windows_come_in_a_permutation_of_each_epoch_drawn_from_the_seed():
    for n in (1, 7, 64, 1000, 4097):
        assert sorted(permuted(np.arange(n), n, (0, 1)).tolist()) == list(range(n))
    a, b = (permuted(np.arange(1000), 1000, (0, epoch)) for epoch in (1, 2))
    assert (a != b).mean() > 0.99
    assert (permuted(np.arange(500, 600), 1000, (0, 1)) == a[500:600]).all()
    # As a shuffle should, a window lands anywhere, about evenly.
    landed = np.stack([permuted(np.arange(8), 8, (s,)) for s in range(4000)])
    assert np.abs(np.bincount(landed[:, 0], minlength=8) / 4000 - 1 / 8).max() < 0.02


# --- The compile cache -----------------------------------------------------------------

# A process of a job of two, which says where its cache is once they have joined.
JOINED = """
import jax

from bux.launch import join
from bux.runtime import configure

configure()
join()
print(jax.config.jax_compilation_cache_dir)
"""


def _joined(folders: list[Path], **more: str) -> list[str]:
    """What two processes, whose caches are `folders`, print once joined, with `more`
    in their environment."""
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
    env = os.environ | more
    env |= {"JAX_COORDINATOR_ADDRESS": f"localhost:{port}", "JAX_NUM_PROCESSES": "2"}
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", JOINED],
            env=env | {"JAX_PROCESS_ID": str(i), "JAX_COMPILATION_CACHE_DIR": str(f)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for i, f in enumerate(folders)
    ]
    return [w.communicate(timeout=120)[0].strip() for w in workers]


def _kept(folders: list[Path]) -> list[str]:
    """Where two processes, whose caches are `folders`, keep them once joined."""
    return [printed.split("\n")[-1] for printed in _joined(folders)]


def test_processes_keep_their_cache_only_where_it_is_one_folder_for_all(tmp_path):
    one = tmp_path / "one"
    assert _kept([one, one]) == [str(one)] * 2
    assert not list(one.iterdir())  # the marker is gone
    assert _kept([tmp_path / "a", tmp_path / "b"]) == ["None", "None"]


def test_processes_the_variables_join_ask_no_cluster_how_to(tmp_path):
    # As on a Kubernetes pod, where jax's detection would warn of a missing client.
    printed = _joined([tmp_path] * 2, KUBERNETES_SERVICE_HOST="10.0.0.1")
    assert not any("Kubernetes" in p for p in printed)


# --- FSDP over several axes ------------------------------------------------------------


def _splits(source: str, layout: tcfg.LayoutConfig, **mesh: int) -> list[tuple]:
    """Each parameter's names, and the mesh axes each of its dimensions splits over."""
    arch = compile(source)
    with parallel.use(tcfg.MeshConfig.model_validate(mesh), layout):
        params = nnx.state(parallel.build(lambda: Model(arch, rngs=nnx.Rngs(0))))
        found = parallel.shardings(nnx.state(params, nnx.Param))
    variables = jax.tree.leaves(nnx.state(params, nnx.Param), is_leaf=_variable)
    return [
        (x.get_metadata().get(parallel.NAMES), [set(as_split(e)) for e in s.spec])
        for x, s in zip(variables, jax.tree.leaves(found), strict=True)
    ]


def _variable(x: object) -> bool:
    return isinstance(x, nnx.Variable)


def test_fsdp_splits_dense_parameters_over_context_and_expert_too_and_experts_keep_theirs():
    mesh = {"fsdp": 2, "context": 2, "expert": 2}
    for names, axes in _splits(MOE, tcfg.LayoutConfig(), **mesh):
        if names and names[0] == "experts":
            assert axes[0] == {"expert"} and {"fsdp", "context"} in axes[1:]
        elif axes:
            assert {"fsdp", "context", "expert"} in axes
    alone = tcfg.LayoutConfig(fsdp="fsdp")
    assert all(
        set().union(*axes) <= {"fsdp", "expert"}
        for _, axes in _splits(MOE, alone, **mesh)
    )


def _fsdp(shape: tuple[int, ...], **mesh: int) -> list[tuple[str, ...]]:
    """How FSDP splits a parameter of `shape` without names on an abstract `mesh`."""
    sizes = _sizes(per_node=math.prod(mesh.values()), nodes=1, **mesh)
    auto = (AxisType.Auto,) * len(parallel.AXES)
    with parallel.use(
        AbstractMesh(tuple(sizes.values()), parallel.AXES, axis_types=auto)
    ):
        on = parallel.active()
        assert on is not None
        return fsdp([()] * len(shape), shape, on, (None,) * len(shape))


def test_a_dimension_that_fsdps_axes_do_not_all_divide_splits_over_the_first_few():
    assert _fsdp((12, 4), fsdp=2, context=2, expert=3) == [
        ("fsdp", "context", "expert"),
        (),
    ]
    assert _fsdp((12, 6), fsdp=2, context=2, expert=2) == [("fsdp", "context"), ()]
    assert _fsdp((6, 10), fsdp=2, context=2, expert=2) == [(), ("fsdp",)]
    assert _fsdp((3, 5), fsdp=2, context=2, expert=2) == [(), ()]


@pytest.mark.parametrize(
    "mesh", ["mesh(fsdp=2, context=2, expert=2)", "mesh(context=2, expert=4)"]
)
@pytest.mark.parametrize("source", [DENSE, MOE], ids=["dense", "moe"])
def test_a_run_whose_fsdp_spans_context_and_expert_trains_as_one_device_does(
    losses, involuntary, source, mesh
):
    one = losses(source, mesh="mesh(data=1)")
    involuntary()
    assert np.allclose(losses(source, mesh=mesh), one, atol=1e-5)
    assert involuntary() == 0
