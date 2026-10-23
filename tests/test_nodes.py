"""Runs across nodes: the mesh over several of them, each process's rows of a batch,
and FSDP over several axes, on the 8 CPU devices the tests have, against one."""

from types import SimpleNamespace

import numpy as np
import pytest

from bux import parallel
from bux.parallel.mesh import grid, split
from bux.train import config as tcfg


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
