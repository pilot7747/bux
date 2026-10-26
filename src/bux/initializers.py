"""How weight matrices start: what the model's ``init`` draws each one from, by the
role it plays.

A model is built inside `initializing`, with its architecture's init
(`bux.config.Init`), ``normal(std=0.02)`` unless it says otherwise. A layer starts a
weight matrix by `weight`, or by `bux.parallel.named`, which takes the matrix's role
from the names of its dimensions (`role_of`). Parameters that start in a way of their
own, such as norms' gains, biases, short convolutions and Mamba's decays, do not ask.

An init is a config in `bux.config`, added to `~bux.config.Init`, and a builder
registered for its tag in `inits`, which returns the `flax.nnx` initializer that a
weight matrix is drawn by::

    @initializers.inits.register("orthogonal")
    def _(c: OrthogonalInitConfig, matrix: Matrix) -> nnx.Initializer:
        return nnx.initializers.orthogonal(c.gain)
"""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from flax import nnx

from bux import config as cfg
from bux.train.registry import Registry

#: A weight matrix's role in the model: an embedding table, an untied head, a
#: projection back to the stream's width, or any other.
Role = Literal["embed", "head", "out", "hidden"]


@dataclass(frozen=True)
class Matrix:
    """A weight matrix, as an init sees it.

    Attributes:
        role: Its role in the model.
        batch_axis: The axes of its array that stack matrices, each one its own, as
            an MoE's experts.
    """

    role: Role = "hidden"
    batch_axis: tuple[int, ...] = ()


# The distributions of fan inits, by the names jax gives them.
_DISTRIBUTIONS: dict[str, Literal["truncated_normal", "normal", "uniform"]] = {
    "truncated": "truncated_normal",
    "gaussian": "normal",
    "uniform": "uniform",
}
#: The builder of each init, by its tag: the initializer it draws a matrix by.
inits = Registry[nnx.Initializer]("init", "bux.initializers.inits")
# The init of the model being built; a config is frozen.
_MODEL: ContextVar[cfg.Init] = ContextVar("init", default=cfg.NormalInitConfig())  # noqa: B039


@contextmanager
def initializing(init: cfg.Init) -> Generator[None]:
    """Start every weight matrix made inside by `init`: a model is built in its
    architecture's. Outside any, matrices start by the default, ``normal``."""
    token = _MODEL.set(init)
    try:
        yield
    finally:
        _MODEL.reset(token)


def weight(role: Role = "hidden", batch_axis: tuple[int, ...] = ()) -> nnx.Initializer:
    """Return how a weight matrix of `role` starts in the model being built: by the
    init its ``init`` gives the role, if any, else by the init itself.

    Args:
        role: The matrix's role in the model.
        batch_axis: The axes of its array that stack matrices, as an MoE's experts.

    Example::

        table = weight("embed")(rngs.params(), (vocab, d))
    """
    init = _MODEL.get()
    own = None if role == "hidden" else getattr(init, role)
    return inits.build(own or init, Matrix(role, batch_axis))


def role_of(names: Sequence[str | tuple[str, int] | None]) -> Role:
    """Return the role of a matrix whose dimensions are `names` (`bux.parallel.Name`):
    ``out`` for one whose output is the stream's width, ``embed``, as attention's
    output and an MLP's down; ``hidden`` for any other."""
    return "out" if names and names[-1] == "embed" else "hidden"


@inits.register("normal")
def _(c: cfg.NormalInitConfig, matrix: Matrix) -> nnx.Initializer:
    """Draw from a normal of ``std``, truncated at ``truncate`` standard deviations if
    it says so."""
    if c.truncate is None:
        return nnx.initializers.normal(c.std)
    return nnx.initializers.truncated_normal(c.std, lower=-c.truncate, upper=c.truncate)


@inits.register("fan_in", "fan_out", "fan_avg")
def _(c: cfg.FanInitConfig, matrix: Matrix) -> nnx.Initializer:
    """Draw at a variance of ``scale`` over the matrix's fan in, out, or their mean
    (LeCun, He, Glorot); an embedding table's fans are its width."""
    # A table (rows, d) has fans of d this way, as nnx.Embed's.
    out_axis = 0 if matrix.role == "embed" else -1
    distribution = _DISTRIBUTIONS[c.distribution]
    return nnx.initializers.variance_scaling(
        c.scale, c.type, distribution, out_axis=out_axis, batch_axis=matrix.batch_axis
    )


@inits.register("zeros")
def _(c: cfg.ZerosInitConfig, matrix: Matrix) -> nnx.Initializer:
    """Start at 0."""
    return nnx.initializers.zeros
