"""How weight matrices start: what the model's ``init`` draws each one from, by the
role it plays.

A model is built inside `initializing`, with its architecture's init
(`bux.config.Init`). A layer starts a weight matrix by `weight`, or by
`bux.parallel.named`, which takes the matrix's role from the names of its dimensions
(`role_of`). A model that sets no init starts every matrix from `DEFAULT`, but those
whose layer starts them its own way (`weight`'s ``own``, as Gated DeltaNet-2's
Xavier); one that sets an init starts them all by it. Parameters that start in a way
of their own whatever the init, such as norms' gains, biases, short convolutions and
Mamba's decays, do not ask.

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

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Key

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
#: How a model that sets no init starts its weight matrices.
DEFAULT = cfg.NormalInitConfig()
# The init of the model being built; None for none set.
_MODEL: ContextVar[cfg.Init | None] = ContextVar("init", default=None)


@contextmanager
def initializing(init: cfg.Init | None) -> Generator[None]:
    """Start every weight matrix made inside by `init`: a model is built in its
    architecture's. None, as outside any, starts them as a model that sets none."""
    token = _MODEL.set(init)
    try:
        yield
    finally:
        _MODEL.reset(token)


def weight(
    role: Role = "hidden",
    batch_axis: tuple[int, ...] = (),
    own: nnx.Initializer | None = None,
) -> nnx.Initializer:
    """Return how a weight matrix of `role` starts in the model being built: by the
    init its ``init`` gives the role, if any, else by the init itself.

    Args:
        role: The matrix's role in the model.
        batch_axis: The axes of its array that stack matrices, as an MoE's experts.
        own: How its layer starts it, where the model sets no init; None: as
            `DEFAULT` does.

    Example::

        table = weight("embed")(rngs.params(), (vocab, d))
    """
    init = _MODEL.get()
    if init is None and own is not None:
        return own
    init = init or DEFAULT
    by_role = None if role == "hidden" else getattr(init, role)
    return inits.build(by_role or init, Matrix(role, batch_axis))


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


def mimetic_vo(
    width: int,
    heads: int,
    kv_heads: int,
    v_head_dim: int,
    beta: float,
    noise: float,
    key: Key[Array, ""],
) -> tuple[Float[Array, "d gv"], Float[Array, "hv d"]]:
    """Return the value and output kernels of a mimetic init (modded-nanogpt's
    mimetic V-O init, after Trockman & Kolter, 2023).

    Value head j reads the stream's channels ``j dv`` to ``(j + 1) dv``, times
    ``sqrt(beta)``, and each query head of its group writes its output back onto
    them, times ``-sqrt(beta) / group``, a group being the ``heads / kv_heads`` query
    heads that share it. A head that attends to itself alone then subtracts ``beta /
    group`` of those channels, and the layer ``beta`` of each: with as many kv heads
    as heads tiling the width, its value-output map is ``-beta I``, the source's.
    Both kernels then take noise of std `noise`, in place of their own draws.

    Args:
        width: The stream's width d, which v reads and o writes.
        heads: The query heads h.
        kv_heads: The value heads g, each read by h / g query heads.
        v_head_dim: The width of a value head dv.
        beta: The scale of the value-output map.
        noise: The std of the noise on both kernels.
        key: Draws the noise.

    Returns:
        v's kernel, (d, g dv), and o's, (h dv, d), as gqa's projections hold them.

    Raises:
        ValueError: If the value heads together are wider than the stream.
    """
    if kv_heads * v_head_dim > width:
        raise ValueError(
            f"mimetic_vo maps {kv_heads} value heads of {v_head_dim} onto the stream's "
            f"channels: {kv_heads * v_head_dim} of them do not fit in a width of {width}"
        )
    values, group = kv_heads * v_head_dim, heads // kv_heads
    root = beta**0.5
    v = root * jnp.eye(width, values)
    # Query head i's dim r goes to the channel of its value head's dim r.
    read = jnp.arange(heads * v_head_dim)
    channel = (read // v_head_dim) // group * v_head_dim + read % v_head_dim
    o = -root / group * jax.nn.one_hot(channel, width)
    a, b = jax.random.split(key)
    return (
        v + noise * jax.random.normal(a, v.shape),
        o + noise * jax.random.normal(b, o.shape),
    )
