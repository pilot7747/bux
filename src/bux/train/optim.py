"""Learning-rate schedules and optimizers, built from their configs with optax.

Each is found by its config's type: a schedule in ``schedules``, an optimizer in
``optimizers``. A schedule's builder gets the config and the run's steps, and returns
its curve after the warmup; an optimizer's gets the config, the schedule's multiplier
and the model, and returns an optax transformation.

Modules say how the optimizers treat their parameters with class attributes:
`no_weight_decay`, `no_muon` and `muon_axes`.

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
from bux.architecture import Scanned
from bux.parallel import methods
from bux.parallel.layout import Split, partition
from bux.train import config as tcfg
from bux.train.registry import Registry

Schedule = Callable[[Real[ArrayLike, ""]], Float[Array, ""]]  # step -> value
# A matrix, or a stack of them as its dimension numbers say, made orthogonal.
NewtonSchulz = Callable[[Float[Array, "*m"], MuonDimensionNumbers], Float[Array, "*m"]]

schedules = Registry[Schedule]("schedule", "bux.train.optim.schedules")
optimizers = Registry[optax.GradientTransformation](
    "optimizer", "bux.train.optim.optimizers"
)


# --- Schedules --------------------------------------------------------------------------


def lr_multiplier(config: tcfg.ScheduleConfig, steps: int) -> Schedule:
    """Return the multiplier of the peak learning rate, from 0 to 1, at each step of a
    run of `steps`: a linear warmup, times the schedule's own curve."""
    curve = schedules.build(config, steps)

    def multiplier(step: Real[ArrayLike, ""]) -> Float[Array, ""]:
        step = jnp.asarray(step, jnp.float32)
        warm = jnp.minimum((step + 1) / config.warmup, 1.0) if config.warmup else 1.0
        return jnp.asarray(warm * curve(step))

    return multiplier


def lr_schedule(lr: float, multiplier: Schedule) -> Schedule:
    """Return the learning rate at each step: the peak `lr`, scaled by `multiplier`."""
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
    config: tcfg.OptimizerConfig,
    multiplier: Schedule,
    model: nnx.Module,
    clip: float | None = None,
) -> optax.GradientTransformation:
    """Return the update rule of `config`, its learning rate scaled by `multiplier`,
    after clipping gradients to the global norm `clip`."""
    tx = optimizers.build(config, multiplier, model)
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


def stacked_dims(model: nnx.Module) -> Callable[[tuple], int]:
    """Return how many stacked dimensions lead the parameter at a path: one for each
    scanned repeat it is in, whose passes it stacks."""
    scans = [path for path, node in nnx.iter_graph(model) if isinstance(node, Scanned)]
    return lambda keys: sum(keys[: len(s)] == s for s in scans)


def no_decay_paths(model: nnx.Module) -> list[tuple]:
    """Return the paths of the parameters that modules name in `no_weight_decay`
    (norms' gains, Engram's tables), which weight decay skips whatever their shape."""
    found: list[tuple] = []
    for path, node in nnx.iter_graph(model):
        found += [(*path, name) for name in getattr(type(node), "no_weight_decay", ())]
    return found


def weight_decay_mask(model: nnx.Module) -> Callable[[PyTree], PyTree[bool]]:
    """Return the weight decay mask: matrices and stacks of them, outside
    `no_decay_paths(model)`. A scanned repeat's stacked dimension does not count: its
    norm scales are vectors."""
    skip, lead = no_decay_paths(model), stacked_dims(model)

    def decays(path: tuple, x: Float[Array, "*_"]) -> bool:
        keys = _keys(path)
        return x.ndim - lead(keys) >= 2 and not any(keys[: len(s)] == s for s in skip)

    return lambda params: jax.tree_util.tree_map_with_path(decays, params)


@optimizers.register("adamw")
def adamw(
    c: tcfg.AdamWConfig, multiplier: Schedule, model: nnx.Module
) -> optax.GradientTransformation:
    return optax.adamw(
        lr_schedule(c.lr, multiplier),
        b1=c.b1,
        b2=c.b2,
        eps=c.eps,
        weight_decay=c.weight_decay,
        mask=weight_decay_mask(model),
    )


def muon_matrices(model: nnx.Module) -> dict[tuple, MuonDimensionNumbers]:
    """Return the parameters Muon trains, by path, with their matrices' axes: linear
    layers' kernels and what modules name in `muon_axes`, outside `no_muon` modules
    (the embedding, the LM head). A scanned repeat's stacked dimension leads."""
    outside = [
        path
        for path, node in nnx.iter_graph(model)
        if getattr(type(node), "no_muon", False)
    ]
    lead = stacked_dims(model)
    found: dict[tuple, MuonDimensionNumbers] = {}
    for path, node in nnx.iter_graph(model):
        if any(path[: len(o)] == o for o in outside):
            continue
        n = lead(path)
        if isinstance(node, nnx.Linear):
            found[(*path, "kernel")] = MuonDimensionNumbers(n, n + 1)
        for name, (reduction, output) in getattr(type(node), "muon_axes", {}).items():
            if getattr(node, name, None) is not None:
                found[(*path, name)] = MuonDimensionNumbers(n + reduction, n + output)
    return found


@optimizers.register("muon")
def muon(
    config: tcfg.MuonConfig, multiplier: Schedule, model: nnx.Module
) -> optax.GradientTransformation:
    """Muon for `muon_matrices(model)`, and AdamW for the rest."""
    dims = muon_matrices(model)

    def labels(params: PyTree[Float[Array, "*_"]]) -> PyTree[str]:
        return jax.tree_util.tree_map_with_path(
            lambda p, _: "muon" if _keys(p) in dims else "adam", params
        )

    # Muon sees only its own parameters; the others are masked out of its tree.
    def shapes(updates: PyTree[Float[Array, "*_"]]) -> PyTree[MuonDimensionNumbers]:
        return jax.tree_util.tree_map_with_path(lambda p, _: dims[_keys(p)], updates)

    rule = optax.chain(
        momentum(config.momentum, config.nesterov),
        orthogonalized(config.ns_steps, dims, model),
        scale_by_shape(
            weight_dimension_numbers=shapes, consistent_rms=config.consistent_rms
        ),
        optax.add_decayed_weights(config.weight_decay, mask=weight_decay_mask(model)),
        optax.scale_by_learning_rate(lr_schedule(config.lr, multiplier)),
    )
    adam = config.adam or tcfg.AdamWConfig(lr=config.lr)
    return optax.partition(
        {"muon": rule, "adam": adamw(adam, multiplier, model)}, labels
    )


# --- Muon's parts: optax's scale_by_muon, with Newton-Schulz by method ----------------

NS_COEFFS = (3.4445, -4.7750, 2.0315)  # Keller Jordan's quintic, optax's default


def momentum(beta: float, nesterov: bool) -> optax.GradientTransformation:
    """Return Muon's momentum as optax's `scale_by_muon` keeps it: the gradients'
    bias-corrected average, with Nesterov's look-ahead if `nesterov`."""

    def init(params: PyTree[Float[Array, "*_"]]) -> optax.EmaState:
        return optax.EmaState(jnp.zeros([], jnp.int32), optax.tree.zeros_like(params))

    def update(
        updates: PyTree[Float[Array, "*_"]],
        state: optax.EmaState,
        params: PyTree[Float[Array, "*_"]] | None = None,
    ) -> tuple[PyTree[Float[Array, "*_"]], optax.EmaState]:
        mu = optax.tree.update_moment(updates, state.ema, beta, 1)
        count = optax.safe_increment(state.count)
        if not nesterov:
            mu_hat = optax.tree.bias_correction(mu, beta, count)
            return mu_hat, optax.EmaState(count, mu)
        mu_next = optax.tree.bias_correction(mu, beta, optax.safe_increment(count))
        g_hat = optax.tree.bias_correction(updates, beta, count)
        mu_hat = jax.tree.map(lambda m, g: beta * m + (1 - beta) * g, mu_next, g_hat)
        return mu_hat, optax.EmaState(count, mu)

    # optax's state is any tree of arrays; this one's is always an EmaState.
    return optax.GradientTransformation(
        init, typing.cast(optax.TransformUpdateFn, update)
    )


def orthogonalized(
    steps: int, dims: dict[tuple, MuonDimensionNumbers], model: nnx.Module
) -> optax.GradientTransformation:
    """Orthogonalize each update, a matrix or a stack of them as `dims` says by path,
    with `steps` Newton-Schulz iterations, by the run's `orthogonalize` method. The
    method gets all of a step's updates at once, so it can choose where each runs."""
    model_params = nnx.state(model, nnx.Param)  # whose shardings the updates take

    def update(
        updates: PyTree[Float[Array, "*_"]],
        params: PyTree[Float[Array, "*_"]] | None = None,
    ) -> PyTree[Float[Array, "*_"]]:
        leaves, tree = jax.tree_util.tree_flatten_with_path(updates)
        keys = [_keys(path) for path, _ in leaves]
        shardings = shardings_by_path(model_params)
        orthogonal = parallel.method("orthogonalize")(
            [x for _, x in leaves],
            [dims[k] for k in keys],
            None if shardings is None else [shardings[k] for k in keys],
            newton_schulz(steps),
        )
        return jax.tree.unflatten(tree, orthogonal)

    return optax.stateless(update)


def newton_schulz(steps: int) -> NewtonSchulz:
    """Return optax's Newton-Schulz, with `steps` iterations."""

    def orthogonal(
        x: Float[Array, "*m"], dims: MuonDimensionNumbers
    ) -> Float[Array, "*m"]:
        coeffs = jnp.asarray(NS_COEFFS)
        return orthogonalize_via_newton_schulz(x, coeffs, steps, dimension_numbers=dims)

    return orthogonal


def shardings_by_path(params: nnx.State) -> dict[tuple, NamedSharding] | None:
    """Return the sharding of each of `params` by path under ``parallel.use``, or None
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
    """Run Newton-Schulz on whatever shards the compiler gives each update, where a
    product over a split dimension is an all-reduce."""
    return [newton_schulz(x, d) for x, d in zip(updates, dims, strict=True)]


@methods.orthogonalize.register("gather")
def gather(
    updates: list[Float[Array, "*_"]],
    dims: list[MuonDimensionNumbers],
    shardings: list[NamedSharding] | None,
    newton_schulz: NewtonSchulz,
) -> list[Float[Array, "*_"]]:
    """Gather each update whole for Newton-Schulz, a stack matrix by matrix
    (`whole_matrix_spec`), then put it back on its parameter's shards: one gather per
    matrix instead of the all-reduces, with the work repeated on each device that
    holds the matrix."""
    if shardings is None:
        return gspmd(updates, dims, shardings, newton_schulz)
    return [
        _orthogonalize_gathered(x, d, s, newton_schulz)
        for x, d, s in zip(updates, dims, shardings, strict=True)
    ]


def _orthogonalize_gathered(
    x: Float[Array, "*m"],
    dims: MuonDimensionNumbers,
    sharding: NamedSharding,
    newton_schulz: NewtonSchulz,
) -> Float[Array, "*m"]:
    gathered = whole_matrix_spec(sharding.spec, dims, x.shape, sharding.mesh)
    per_device = jax.shard_map(
        lambda x: newton_schulz(x, dims),
        mesh=sharding.mesh,
        in_specs=gathered,
        out_specs=gathered,
    )
    # On the parameter's shards on the way in too: else the compiler moves the
    # momentum and the gradient to `gathered` to mix them there, two moves for one.
    shard = jax.lax.with_sharding_constraint
    return shard(per_device(shard(x, sharding)), sharding)


def whole_matrix_spec(
    spec: P,
    dims: MuonDimensionNumbers,
    shape: tuple[int, ...],
    mesh: Mesh | AbstractMesh,
) -> P:
    """Return `spec` with each matrix that `dims` finds in the array whole, and the
    mesh axes that frees moved onto the first stacked dimension they divide, so that
    each device orthogonalizes whole matrices, its own part of the stack."""
    matrix_dims = _matrix_axes(dims, len(shape))
    axes = [_mesh_axes(spec[i] if i < len(spec) else None) for i in range(len(shape))]
    freed = tuple(a for i in sorted(matrix_dims) for a in axes[i])
    new = [() if i in matrix_dims else a for i, a in enumerate(axes)]
    for i, n in enumerate(shape):
        divides = n % math.prod(mesh.shape[a] for a in (*new[i], *freed)) == 0
        if freed and i not in matrix_dims and divides:
            new[i] = (*new[i], *freed)
            break
    return partition(new)


def _matrix_axes(dims: MuonDimensionNumbers, ndim: int) -> set[int]:
    """Return the dimensions of each matrix: `dims`'s reduction and output axes."""
    given = (dims.reduction_axis, dims.output_axis)
    return {a % ndim for g in given for a in ((g,) if isinstance(g, int) else g)}


def _mesh_axes(entry: str | tuple[str, ...] | None) -> Split:
    return () if entry is None else (entry,) if isinstance(entry, str) else entry
