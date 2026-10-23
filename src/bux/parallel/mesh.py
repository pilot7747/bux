"""The device grid: one mesh of all six axes, and the settings `use` puts in force."""

import contextvars
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import AbstractMesh, AxisType, Mesh
from jaxtyping import Shaped

from bux.train.config import LayoutConfig, MeshConfig, MethodsConfig

# The slowest links first, so that `tensor` stays inside a node. `stage` comes before
# `data`: a pipeline's sends need the least bandwidth, and XLA nests a shard_map in a
# pipeline's only over the axes after `stage` (bux.parallel.pipeline).
AXES = ("stage", "data", "fsdp", "context", "expert", "tensor")
# What tells a device's node: XLA puts the GPUs of a host, or of an NVLink domain
# that spans hosts, in one slice.
NODE = "slice_index"


def sizes(config: MeshConfig, devices: int) -> dict[str, int]:
    """Return each axis's size on `devices` devices; `data` gets what the rest leave."""
    rest = {a: getattr(config, a) for a in AXES if a != "data"}
    used = math.prod(rest.values())
    if config.data is None and devices % used:
        raise ValueError(
            f"mesh: the axes other than data take {used} devices, which do not "
            f"divide the {devices} there are; set mesh.data"
        )
    data = devices // used if config.data is None else config.data
    if data * used > devices:
        raise ValueError(
            f"mesh: {_shape(_ordered(data, rest))} needs {data * used} devices, "
            f"but there are {devices}"
        )
    return _ordered(data, rest)


def _ordered(data: int, rest: dict[str, int]) -> dict[str, int]:
    return {a: data if a == "data" else rest[a] for a in AXES}


def _shape(axis_sizes: dict[str, int]) -> str:
    """Format the axes of more than one device: data=2 × tensor=4."""
    split = [f"{a}={n}" for a, n in axis_sizes.items() if n > 1]
    return " × ".join(split) or "one device"


def mesh(config: MeshConfig) -> Mesh:
    """Return the mesh `config` sets, on the first devices it needs, every axis Auto:
    the compiler propagates shardings from the parameters and a few constraints."""
    devices = jax.devices()
    shape = sizes(config, len(devices))
    used = devices[: math.prod(shape.values())]
    return Mesh(grid(shape, used), AXES, axis_types=(AxisType.Auto,) * len(AXES))


def grid(shape: dict[str, int], devices: Sequence[Any]) -> Shaped[np.ndarray, "..."]:
    """Lay `devices` out on the axes of `shape`. On several nodes, each holds a block
    of the innermost axes (`split`), so that the outer ones span nodes; jax.make_mesh
    refuses devices of more than one node."""
    nodes = _nodes(devices)
    if len(nodes) == 1:
        return mesh_utils.create_device_mesh(tuple(shape.values()), devices)
    within, across = split(shape, len(nodes[0]))
    return mesh_utils.create_hybrid_device_mesh(
        tuple(within.values()),
        tuple(across.values()),
        devices,
        process_is_granule=not hasattr(devices[0], NODE),
    )


def _nodes(devices: Sequence[Any]) -> list[list[Any]]:
    """Group `devices` by node: by slice (a GPU's NVLink domain), or by process where
    devices have no slice, as CPUs do."""
    key = NODE if hasattr(devices[0], NODE) else "process_index"
    nodes: dict[int, list[Any]] = {}
    for d in devices:
        nodes.setdefault(getattr(d, key), []).append(d)
    grouped = [nodes[k] for k in sorted(nodes)]
    if len({len(n) for n in grouped}) > 1:
        counts = ", ".join(str(len(n)) for n in grouped)
        raise ValueError(
            f"mesh: its {len(devices)} devices are on nodes of {counts}; a mesh "
            "across nodes takes as many on each"
        )
    return grouped


def split(
    shape: dict[str, int], per_node: int
) -> tuple[dict[str, int], dict[str, int]]:
    """Split each axis's size into its devices within a node and its nodes, filling a
    node from the innermost axis (`tensor`) outward: data=2 × fsdp=8 on nodes of 4 is
    fsdp=4 within each, and data=2 × fsdp=2 across."""
    within, left = {}, per_node
    for axis in reversed(AXES):
        inside = min(shape[axis], left)
        if shape[axis] % inside or left % inside:
            raise ValueError(
                f"mesh: {_shape(shape)} does not fill nodes of {per_node} devices "
                f"from tensor outward: {axis}={shape[axis]} does not split into the "
                f"{left} a node has left for it, nor they into it"
            )
        within[axis], left = inside, left // inside
    within = {a: within[a] for a in AXES}
    return within, {a: shape[a] // within[a] for a in AXES}


@dataclass(frozen=True)
class Active:
    """What `use` puts in force."""

    mesh: Mesh | AbstractMesh
    layout: LayoutConfig
    methods: MethodsConfig

    def size(self, *axes: str) -> int:
        return math.prod(self.mesh.shape[a] for a in axes)


ACTIVE = contextvars.ContextVar[Active | None]("bux.parallel", default=None)


def active() -> Active | None:
    """Return what `use` put in force, or None outside it."""
    return ACTIVE.get()
