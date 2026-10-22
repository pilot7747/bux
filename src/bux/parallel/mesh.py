"""The device grid: one mesh of all six axes, and the settings `use` puts in force."""

import contextvars
import math
from dataclasses import dataclass

import jax
from jax.sharding import AbstractMesh, AxisType, Mesh

from bux.train.config import LayoutConfig, MeshConfig, MethodsConfig

# The slowest links first, so that `tensor` stays inside a node. `stage` comes before
# `data`: a pipeline's sends need the least bandwidth, and XLA nests a shard_map in a
# pipeline's only over the axes after `stage` (bux.parallel.pipeline).
AXES = ("stage", "data", "fsdp", "context", "expert", "tensor")


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
    return jax.make_mesh(
        tuple(shape.values()),
        AXES,
        axis_types=(AxisType.Auto,) * len(AXES),
        devices=devices[: math.prod(shape.values())],
    )


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
