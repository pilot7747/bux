"""`use`, which puts a run's mesh, layout and methods in force, and `check`, which
says whether its batches split over the mesh."""

import contextlib
from collections.abc import Generator

import jax
from jax.sharding import AbstractMesh, Mesh, use_abstract_mesh

from bux.parallel import methods as registries
from bux.parallel.layout import axes
from bux.parallel.mesh import ACTIVE, AXES, Active, active, mesh
from bux.train.config import LayoutConfig, MeshConfig, MethodsConfig

DEFAULT_LAYOUT, DEFAULT_METHODS = LayoutConfig(), MethodsConfig()


@contextlib.contextmanager
def use(
    grid: MeshConfig | Mesh | AbstractMesh,
    layout: LayoutConfig = DEFAULT_LAYOUT,
    methods: MethodsConfig = DEFAULT_METHODS,
) -> Generator[Mesh | AbstractMesh]:
    """Train on `grid` with `layout` and `methods`: `jax.set_mesh`, and what `spec`,
    `shard` and `method` read. On an AbstractMesh, of no devices, `shardings` says
    where state would live."""
    m = mesh(grid) if isinstance(grid, MeshConfig) else grid
    if tuple(m.axis_names) != AXES:
        raise ValueError(f"a bux mesh has the axes {AXES}, not {m.axis_names}")
    registries.validate(methods)
    token = ACTIVE.set(Active(m, layout, methods))
    try:
        with jax.set_mesh(m) if isinstance(m, Mesh) else use_abstract_mesh(m):
            yield m
    finally:
        ACTIVE.reset(token)


def _row_axes(name: str, on: Active) -> tuple[str, ...]:
    return tuple(a for a in axes(on.layout, name) if on.size(a) > 1)


def check(batch: int, seq_len: int) -> None:
    """Raise unless `batch` rows of `seq_len` tokens split evenly by rows over the
    batch axes and by positions over the seq and stream axes."""
    on = active()
    if on is None:
        return
    for setting, n, name in (
        ("batch", batch, "batch"),
        ("seq_len", seq_len, "seq"),
        ("seq_len", seq_len, "stream"),
    ):
        over = _row_axes(name, on)
        if n % on.size(*over):
            split = " × ".join(f"{a}={on.size(a)}" for a in over)
            raise ValueError(
                f"{setting} = {n} does not split over layout.{name}, {split}: "
                f"make it a multiple of {on.size(*over)}, or change mesh or layout"
            )
