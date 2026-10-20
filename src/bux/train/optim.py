"""Learning-rate schedules and optimizers, built from their configs with optax.

Each is found by its config's type: a schedule in ``schedules``, an optimizer in
``optimizers``. A schedule's builder gets the config and the run's steps, and returns
its curve after the warmup; an optimizer's gets the config, the schedule's multiplier
and the model, and returns an optax transformation.

Muon orthogonalizes its updates by the run's ``orthogonalize`` method
(``bux.parallel.methods``): ``gspmd`` on whatever shards the compiler gives them, or
``gather``, each matrix whole.
"""

import math
import typing
from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import AbstractMesh, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, ArrayLike, Float, PyTree, Real
from optax.contrib import MuonDimensionNumbers
from optax.contrib._muon import orthogonalize_via_newton_schulz, scale_by_shape

from bux import parallel
from bux.architecture import DyT, Embedding, Engram, Head, MoE, Norm
from bux.architecture.graph import Scanned
from bux.parallel import methods
from bux.parallel.layout import Split, partition
from bux.train import config as tcfg
from bux.train.registry import Registry

Schedule = Callable[[Real[ArrayLike, ""]], Float[Array, ""]]  # of the step
# A matrix, or a stack of them as its dimension numbers say, made orthogonal.
NewtonSchulz = Callable[[Float[Array, "*m"], MuonDimensionNumbers], Float[Array, "*m"]]

schedules = Registry[Schedule]("schedule", "bux.train.optim.schedules")
optimizers = Registry[optax.GradientTransformation](
    "optimizer", "bux.train.optim.optimizers"
)


# --- Schedules --------------------------------------------------------------------------


def schedule(c: tcfg.ScheduleConfig, steps: int) -> Schedule:
    """The multiplier of the peak learning rate at each step of a run of `steps`: a
    linear warmup, times the schedule's own curve."""
    curve = schedules.build(c, steps)

    def multiplier(step: Real[ArrayLike, ""]) -> Float[Array, ""]:
        step = jnp.asarray(step, jnp.float32)
        warm = jnp.minimum((step + 1) / c.warmup, 1.0) if c.warmup else 1.0
        return jnp.asarray(warm * curve(step))

    return multiplier


def peak(lr: float, multiplier: Schedule) -> Schedule:
    """The learning rate at each step: the peak `lr`, scaled by `multiplier`."""
    return lambda step: lr * multiplier(step)


@schedules.register("constant")
def _(c: tcfg.ConstantSchedule, steps: int) -> Schedule:
    return lambda step: jnp.asarray(1.0)


@schedules.register("cosine")
def _(c: tcfg.CosineSchedule, steps: int) -> Schedule:
    def curve(step: Real[ArrayLike, ""]) -> Float[Array, ""]:
        t = jnp.clip((step + 1 - c.warmup) / max(steps - c.warmup, 1), 0, 1)
        return c.final + (1 - c.final) * 0.5 * (1 + jnp.cos(math.pi * t))

    return curve


COOLDOWNS: dict[str, Callable[[Float[Array, ""]], Float[Array, ""]]] = {
    "linear": lambda t: 1 - t,
    "sqrt": lambda t: 1 - jnp.sqrt(t),
    "cosine": lambda t: 0.5 * (1 + jnp.cos(math.pi * t)),
}


@schedules.register("wsd")
def _(c: tcfg.WSDSchedule, steps: int) -> Schedule:
    cooldown = c.cooldown(steps)

    def curve(step: Real[ArrayLike, ""]) -> Float[Array, ""]:
        t = jnp.clip((step + 1 - (steps - cooldown)) / max(cooldown, 1), 0, 1)
        return c.final + (1 - c.final) * COOLDOWNS[c.shape](t)

    return curve


# --- Optimizers -------------------------------------------------------------------------


def optimizer(
    c: tcfg.OptimizerConfig,
    multiplier: Schedule,
    model: nnx.Module,
    clip: float | None = None,
) -> optax.GradientTransformation:
    """The update rule of `c`, its learning rate scaled by `multiplier`, after clipping
    gradients to the global norm `clip`."""
    tx = optimizers.build(c, multiplier, model)
    if clip is None:
        return tx
    return optax.chain(optax.clip_by_global_norm(clip), tx)


def _keys(path: tuple) -> tuple:
    """A pytree path as plain keys, without the Variable's trailing `value`."""
    keys = tuple(
        next(getattr(k, a) for a in ("key", "idx", "name") if hasattr(k, a))
        for k in path
    )
    return keys[:-1] if keys and keys[-1] == "value" else keys


def stacked(model: nnx.Module) -> Callable[[tuple], int]:
    """How many stacked dimensions lead the parameter at a path: one for each scanned
    repeat it is in, whose passes it stacks."""
    scans = [path for path, node in nnx.iter_graph(model) if isinstance(node, Scanned)]
    return lambda keys: sum(keys[: len(s)] == s for s in scans)


def undecayed(model: nnx.Module) -> list[tuple]:
    """The paths that weight decay leaves alone, whatever their shape: norms' and
    Engram's n-gram tables, stacked or not."""
    found: list[tuple] = []
    for path, node in nnx.iter_graph(model):
        if isinstance(node, Norm | DyT):
            found.append(path)
        elif isinstance(node, Engram):
            found.append((*path, "table"))
    return found


def decayed(model: nnx.Module) -> Callable[[PyTree], PyTree[bool]]:
    """The weight decay mask: matrices and stacks of them, outside `undecayed(model)`.
    A scanned repeat's stacked dimension does not count: its norm scales are vectors."""
    skip, lead = undecayed(model), stacked(model)

    def decays(path: tuple, x: Float[Array, "*_"]) -> bool:
        keys = _keys(path)
        return x.ndim - lead(keys) >= 2 and not any(keys[: len(s)] == s for s in skip)

    return lambda params: jax.tree_util.tree_map_with_path(decays, params)


@optimizers.register("adamw")
def adamw(
    c: tcfg.AdamWConfig, multiplier: Schedule, model: nnx.Module
) -> optax.GradientTransformation:
    return optax.adamw(
        peak(c.lr, multiplier),
        b1=c.b1,
        b2=c.b2,
        eps=c.eps,
        weight_decay=c.weight_decay,
        mask=decayed(model),
    )


def matrices(model: nnx.Module) -> dict[tuple, MuonDimensionNumbers]:
    """The parameters Muon trains, by path: the kernels of linear layers in the blocks,
    and stacked expert weights, one expert at a time. The embedding and the LM head
    are left out. A scanned repeat's stacked dimension leads, a batch dimension."""
    outside = [
        path
        for path, node in nnx.iter_graph(model)
        if isinstance(node, Embedding | Head)
    ]
    lead = stacked(model)
    found: dict[tuple, MuonDimensionNumbers] = {}
    for path, node in nnx.iter_graph(model):
        if any(path[: len(o)] == o for o in outside):
            continue
        n = lead(path)
        if isinstance(node, nnx.Linear):
            found[(*path, "kernel")] = MuonDimensionNumbers(n, n + 1)
        elif isinstance(node, MoE):
            for name in ("up", "gate", "down"):
                if getattr(node, name, None) is not None:
                    found[(*path, name)] = MuonDimensionNumbers(n + 1, n + 2)
    return found


@optimizers.register("muon")
def muon(
    c: tcfg.MuonConfig, multiplier: Schedule, model: nnx.Module
) -> optax.GradientTransformation:
    """Muon for `matrices(model)`, and AdamW for the rest."""
    dims = matrices(model)

    def labels(params: PyTree[Float[Array, "*_"]]) -> PyTree[str]:
        return jax.tree_util.tree_map_with_path(
            lambda p, _: "muon" if _keys(p) in dims else "adam", params
        )

    # Muon sees only its own parameters; the others are masked out of its tree.
    def shapes(updates: PyTree[Float[Array, "*_"]]) -> PyTree[MuonDimensionNumbers]:
        return jax.tree_util.tree_map_with_path(lambda p, _: dims[_keys(p)], updates)

    rule = optax.chain(
        momentum(c.momentum, c.nesterov),
        orthogonalized(c.ns_steps, dims, model),
        scale_by_shape(
            weight_dimension_numbers=shapes, consistent_rms=c.consistent_rms
        ),
        optax.add_decayed_weights(c.weight_decay, mask=decayed(model)),
        optax.scale_by_learning_rate(peak(c.lr, multiplier)),
    )
    adam = c.adam or tcfg.AdamWConfig(lr=c.lr)
    return optax.partition(
        {"muon": rule, "adam": adamw(adam, multiplier, model)}, labels
    )


# --- Muon's parts: optax's scale_by_muon, with Newton-Schulz by method ----------------

NS_COEFFS = (3.4445, -4.7750, 2.0315)  # Keller Jordan's quintic, optax's default


def momentum(beta: float, nesterov: bool) -> optax.GradientTransformation:
    """Muon's momentum, as optax's `scale_by_muon` keeps it: the gradients' average,
    bias-corrected, and with Nesterov's look-ahead if `nesterov`."""

    def init(params: PyTree[Float[Array, "*_"]]) -> optax.EmaState:
        return optax.EmaState(jnp.zeros([], jnp.int32), optax.tree.zeros_like(params))

    def update(
        updates: PyTree[Float[Array, "*_"]],
        state: optax.EmaState,
        params: PyTree[Float[Array, "*_"]] | None = None,
    ) -> tuple[PyTree[Float[Array, "*_"]], optax.EmaState]:
        mu = optax.tree.update_moment(updates, state.ema, beta, 1)
        t = optax.safe_increment(state.count)
        if not nesterov:
            return optax.tree.bias_correction(mu, beta, t), optax.EmaState(t, mu)
        ahead = optax.tree.bias_correction(mu, beta, optax.safe_increment(t))
        now = optax.tree.bias_correction(updates, beta, t)
        mixed = jax.tree.map(lambda m, g: beta * m + (1 - beta) * g, ahead, now)
        return mixed, optax.EmaState(t, mu)

    # optax's state is any tree of arrays; this one's is always an EmaState.
    return optax.GradientTransformation(
        init, typing.cast(optax.TransformUpdateFn, update)
    )


def orthogonalized(
    steps: int, dims: dict[tuple, MuonDimensionNumbers], model: nnx.Module
) -> optax.GradientTransformation:
    """Each update, a matrix or a stack as `dims` says by its path, orthogonalized by
    `steps` Newton-Schulz iterations, where the run's `orthogonalize` method says:
    a method gets every update of a step, so that it can choose where each runs."""
    variables = nnx.state(model, nnx.Param)  # whose shardings the updates take

    def update(
        updates: PyTree[Float[Array, "*_"]],
        params: PyTree[Float[Array, "*_"]] | None = None,
    ) -> PyTree[Float[Array, "*_"]]:
        leaves, tree = jax.tree_util.tree_flatten_with_path(updates)
        keys = [_keys(path) for path, _ in leaves]
        placed = placements(variables)
        done = parallel.method("orthogonalize")(
            [x for _, x in leaves],
            [dims[k] for k in keys],
            None if placed is None else [placed[k] for k in keys],
            newton_schulz(steps),
        )
        return jax.tree.unflatten(tree, done)

    return optax.stateless(update)


def newton_schulz(steps: int) -> NewtonSchulz:
    """optax's Newton-Schulz, `steps` iterations of it."""

    def orthogonal(
        x: Float[Array, "*m"], dims: MuonDimensionNumbers
    ) -> Float[Array, "*m"]:
        coeffs = jnp.asarray(NS_COEFFS)
        return orthogonalize_via_newton_schulz(x, coeffs, steps, dimension_numbers=dims)

    return orthogonal


def placements(params: nnx.State) -> dict[tuple, NamedSharding] | None:
    """The sharding of each of `params`, by path, under ``parallel.use``; None
    outside it."""
    if parallel.active() is None:
        return None
    found = jax.tree_util.tree_leaves_with_path(parallel.shardings(params))
    return {_keys(path): s for path, s in found}


# --- How Muon reaches whole matrices: methods.orthogonalize ----------------------------
# A method takes every update Muon orthogonalizes in a step, the dimension numbers of
# each, the sharding of each one's parameter (None outside parallel.use), and the
# Newton-Schulz to run; it returns the updates orthogonalized.


@methods.orthogonalize.register("gspmd")
def gspmd(
    updates: list[Float[Array, "*_"]],
    dims: list[MuonDimensionNumbers],
    shardings: list[NamedSharding] | None,
    newton_schulz: NewtonSchulz,
) -> list[Float[Array, "*_"]]:
    """Newton-Schulz on whatever shards the compiler gives each update: a product
    over a split dimension is an all-reduce."""
    return [newton_schulz(x, d) for x, d in zip(updates, dims, strict=True)]


@methods.orthogonalize.register("gather")
def gather(
    updates: list[Float[Array, "*_"]],
    dims: list[MuonDimensionNumbers],
    shardings: list[NamedSharding] | None,
    newton_schulz: NewtonSchulz,
) -> list[Float[Array, "*_"]]:
    """Each update whole, a stack whole matrix by matrix (`whole`), for Newton-Schulz,
    and back on its parameter's shards after: a gather of each matrix in place of the
    all-reduces, with the work repeated on every device that holds it whole."""
    if shardings is None:
        return gspmd(updates, dims, shardings, newton_schulz)
    return [
        _gathered(x, d, s, newton_schulz)
        for x, d, s in zip(updates, dims, shardings, strict=True)
    ]


def _gathered(
    x: Float[Array, "*m"],
    dims: MuonDimensionNumbers,
    sharding: NamedSharding,
    newton_schulz: NewtonSchulz,
) -> Float[Array, "*m"]:
    split = whole(sharding.spec, dims, x.shape, sharding.mesh)
    local = jax.shard_map(
        lambda x: newton_schulz(x, dims),
        mesh=sharding.mesh,
        in_specs=split,
        out_specs=split,
    )
    # On the parameter's shards on the way in too: else the compiler moves the
    # momentum and the gradient to `split` to mix them there, two moves for one.
    on = jax.lax.with_sharding_constraint
    return on(local(on(x, sharding)), sharding)


def whole(
    spec: P,
    dims: MuonDimensionNumbers,
    shape: tuple[int, ...],
    mesh: Mesh | AbstractMesh,
) -> P:
    """`spec` with each matrix that `dims` finds in the array whole, and the axes that
    frees on the first stacked dimension they divide, so that each device
    orthogonalizes whole matrices of a stack, its own."""
    matrix = _matrix_axes(dims, len(shape))
    splits = [_split(spec[i] if i < len(spec) else None) for i in range(len(shape))]
    freed = tuple(a for i in sorted(matrix) for a in splits[i])
    kept = [() if i in matrix else s for i, s in enumerate(splits)]
    for i, n in enumerate(shape):
        fits = n % math.prod(mesh.shape[a] for a in (*kept[i], *freed)) == 0
        if freed and i not in matrix and fits:
            kept[i] = (*kept[i], *freed)
            break
    return partition(kept)


def _matrix_axes(dims: MuonDimensionNumbers, ndim: int) -> set[int]:
    """The dimensions of each matrix: `dims`'s reduction and output axes."""
    given = (dims.reduction_axis, dims.output_axis)
    return {a % ndim for g in given for a in ((g,) if isinstance(g, int) else g)}


def _split(entry: str | tuple[str, ...] | None) -> Split:
    return () if entry is None else (entry,) if isinstance(entry, str) else entry
