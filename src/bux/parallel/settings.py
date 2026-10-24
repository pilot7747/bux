"""`use`, which puts a run's mesh, layout and methods in force, and `check`, which
raises unless its batches split into its microbatches and over the mesh."""

import contextlib
from collections.abc import Generator

import jax
from jax.sharding import AbstractMesh, Mesh, use_abstract_mesh

from bux.parallel import methods as registries
from bux.parallel.layout import split_axes
from bux.parallel.mesh import ACTIVE, AXES, Active, active, mesh
from bux.train.config import LayoutConfig, MeshConfig, MethodsConfig

DEFAULT_LAYOUT, DEFAULT_METHODS = LayoutConfig(), MethodsConfig()


@contextlib.contextmanager
def use(
    grid: MeshConfig | Mesh | AbstractMesh,
    layout: LayoutConfig = DEFAULT_LAYOUT,
    methods: MethodsConfig = DEFAULT_METHODS,
) -> Generator[Mesh | AbstractMesh]:
    """Train on `grid` with `layout` and `methods`.

    Set the mesh for jax, and the settings that `spec`, `shard` and `method` read. On
    an AbstractMesh, `shardings` says where state would live.
    """
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


def check(batch: int, seq_len: int) -> None:
    """Raise unless `batch` rows of `seq_len` tokens split evenly: the rows into
    `methods.microbatches`, then over the batch axes, and the positions over the seq
    and stream axes."""
    on = active()
    if on is None:
        return
    n = on.methods.microbatches
    if batch % n:
        raise ValueError(
            f"batch = {batch} does not split into methods.microbatches = {n}"
        )
    per_microbatch = f", {n} microbatches of {batch // n} rows" if n > 1 else ""
    for setting, size, name, parts, detail in (
        ("batch", batch, "batch", n, per_microbatch),
        ("seq_len", seq_len, "seq", 1, ""),
        ("seq_len", seq_len, "stream", 1, ""),
    ):
        over = split_axes(on, name)
        if (size // parts) % on.size(*over):
            split = " × ".join(f"{a}={on.size(a)}" for a in over)
            raise ValueError(
                f"{setting} = {size} does not split over layout.{name}, {split}"
                f"{detail}: make it a multiple of {on.size(*over) * parts}, or change "
                "mesh or layout"
            )
