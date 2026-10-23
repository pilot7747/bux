"""Runs across nodes: the mesh over several of them, each process's rows of a batch,
and FSDP over several axes, on the 8 CPU devices the tests have, against one."""

import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import corpus

from bux import parallel
from bux.data.pipeline import jobs, prepare
from bux.parallel.mesh import grid, split
from bux.train import config as tcfg
from bux.train.data import batches, permuted


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

# A process of a job of two on CPUs, which asks whether a folder is one for both. Gloo
# carries their collectives, its peers on localhost, which a laptop resolves.
SHARED = """
import sys
from pathlib import Path

import jax.extend.backend
from jax._src import distributed, xla_bridge
from jax._src.lib import _jax


def cpu():
    client = distributed.global_state.client
    gloo = _jax.make_gloo_tcp_collectives(client, hostname="localhost")
    return xla_bridge.make_cpu_client(collectives=gloo)


jax.extend.backend.register_backend_factory("cpu", cpu, fail_quietly=False)
from bux.launch import join
from bux.runtime import shared

join()
print(shared(Path(sys.argv[1])))
"""


def _asked(folders: list[Path]) -> list[str]:
    """What two processes say of `folders`, one each."""
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
    env = os.environ | {
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        "JAX_COORDINATOR_ADDRESS": f"localhost:{port}",
        "JAX_NUM_PROCESSES": "2",
    }
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", SHARED, str(folder)],
            env=env | {"JAX_PROCESS_ID": str(i)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for i, folder in enumerate(folders)
    ]
    return [w.communicate(timeout=120)[0].strip().split("\n")[-1] for w in workers]


def test_processes_find_out_whether_their_cache_is_one_folder(tmp_path):
    assert _asked([tmp_path / "one"] * 2) == ["True", "True"]
    assert not list((tmp_path / "one").iterdir())  # the marker is gone
    assert _asked([tmp_path / "a", tmp_path / "b"]) == ["False", "False"]
