"""Learning-rate schedules and optimizers, built from their configs with optax.

Each is found by its config's type: a schedule in `schedules`, an optimizer in
`optimizers`. A schedule's builder gets the config and the run's steps, and returns
its curve after the warmup; an optimizer's gets the config, the schedule's multiplier
and the model, and returns an optax transformation. A builder registered under a
taken name replaces bux's own::

    @optimizers.register("adamw")
    def _(c: tcfg.AdamWConfig, multiplier: Schedule, model: nnx.Module):
        return optax.adamw(lr_schedule(c.lr, multiplier), b1=c.b1, b2=c.b2)

Modules say how the optimizers treat their parameters with class attributes:
``no_weight_decay`` (names of parameters weight decay skips), ``no_muon`` (a module
whose parameters train with AdamW under Muon) and ``muon_axes`` (each parameter's
matrix axes, for Muon).

Muon orthogonalizes its updates by the run's ``orthogonalize`` method
(`bux.parallel.methods`): ``gspmd`` on whatever shards the compiler gives them,
``gather``, each matrix whole on every device that holds it, or ``owner``, each
matrix whole on one device. Its Newton-Schulz's coefficients are found by name in
`ns_coefficients`, a builder that gets the number of iterations, and how it iterates
in `ns_algorithms`: the standard iteration, or Gram's.
"""

import itertools
import math
import typing
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import AbstractMesh, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, ArrayLike, Float, PyTree, Real, Shaped
from optax.contrib import MuonDimensionNumbers
from optax.contrib._muon import (
    _compute_muon_reshape,
    orthogonalize_via_newton_schulz,
    scale_by_shape,
)

from bux import parallel
from bux.architecture import Scanned
from bux.parallel import methods
from bux.parallel.layout import Split, as_split, partition
from bux.train import config as tcfg
from bux.train.registry import Registry

#: A value at each step, counted from 0, as a learning rate or its multiplier.
Schedule = Callable[[Real[ArrayLike, ""]], Float[Array, ""]]
# A matrix, or a stack of them as its dimension numbers say, made orthogonal.
NewtonSchulz = Callable[[Float[Array, "*m"], MuonDimensionNumbers], Float[Array, "*m"]]

#: The schedules, by the ``type`` of their config. A builder gets the config and the
#: run's steps, and returns the multiplier's curve after the warmup.
schedules = Registry[Schedule]("schedule", "bux.train.optim.schedules")
#: The optimizers, by the ``type`` of their config. A builder gets the config, the
#: schedule's multiplier and the model, and returns an optax transformation.
optimizers = Registry[optax.GradientTransformation](
    "optimizer", "bux.train.optim.optimizers"
)
#: Newton-Schulz's quintics ``a x + b x³ + c x⁵`` for a number of iterations, by
#: name: one (a, b, c) for every iteration, or one for each. A builder gets the
#: number of iterations.
ns_coefficients = Registry[Float[Array, "*steps 3"]](
    "set of Newton-Schulz coefficients", "bux.train.optim.ns_coefficients"
)
#: Newton-Schulz's ways to iterate, by name. A builder gets the number of
#: iterations, their quintics (from `ns_coefficients`) and the iterations after
#: which Gram's restarts, and returns the `NewtonSchulz`.
ns_algorithms = Registry[NewtonSchulz](
    "Newton-Schulz algorithm", "bux.train.optim.ns_algorithms"
)


# --- Schedules --------------------------------------------------------------------------


def lr_multiplier(config: tcfg.ScheduleConfig, steps: int) -> Schedule:
    """Return the multiplier of the peak learning rate, from 0 to 1, at each step of a
    run of `steps`: a linear warmup, times the schedule's own curve.

    The warmup rises from ``1 / warmup`` at step 0 to 1 at step ``warmup - 1``.

    Args:
        config: The run's ``schedule`` setting.
        steps: The run's steps.

    Returns:
        The multiplier at each step, counted from 0.

    Raises:
        NotImplementedError: If no schedule is registered under the config's type.

    Example::

        >>> m = lr_multiplier(WSDSchedule(warmup=10, decay=0.2), steps=100)
        >>> [round(float(m(s)), 2) for s in (0, 9, 50, 89, 99)]
        [0.1, 1.0, 1.0, 0.5, 0.0]
    """
    curve = schedules.build(config, steps)

    def multiplier(step: Real[ArrayLike, ""]) -> Float[Array, ""]:
        step = jnp.asarray(step, jnp.float32)
        warm = jnp.minimum((step + 1) / config.warmup, 1.0) if config.warmup else 1.0
        return jnp.asarray(warm * curve(step))

    return multiplier


def lr_schedule(lr: float, multiplier: Schedule) -> Schedule:
    """Return the learning rate at each step: the peak `lr`, scaled by `multiplier`
    (`lr_multiplier`)."""
    return lambda step: lr * multiplier(step)


@schedules.register("constant")
def _(c: tcfg.ConstantSchedule, steps: int) -> Schedule:
    """The peak learning rate at every step after the warmup."""
    return lambda step: jnp.asarray(1.0)


@schedules.register("cosine")
def _(c: tcfg.CosineSchedule, steps: int) -> Schedule:
    """A cosine decay from the peak after the warmup to `final` times it at the last
    step."""

    def curve(step: Real[ArrayLike, ""]) -> Float[Array, ""]:
        t = jnp.clip((step + 1 - c.warmup) / max(steps - c.warmup, 1), 0, 1)
        return c.final + (1 - c.final) * 0.5 * (1 + jnp.cos(math.pi * t))

    return curve


#: The curves of a ``wsd`` cooldown by its ``shape``, from 1 at t = 0 to 0 at t = 1.
COOLDOWNS: dict[str, Callable[[Float[Array, ""]], Float[Array, ""]]] = {
    "linear": lambda t: 1 - t,
    "sqrt": lambda t: 1 - jnp.sqrt(t),
    "cosine": lambda t: 0.5 * (1 + jnp.cos(math.pi * t)),
}


@schedules.register("wsd")
def _(c: tcfg.WSDSchedule, steps: int) -> Schedule:
    """Warmup-stable-decay: the peak until the last `decay` of the steps, then a
    cooldown of `shape` to `final` times the peak at the last step."""
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
    moments: DTypeLike = jnp.float32,
) -> optax.GradientTransformation:
    """Return the update rule of `config`, its learning rate scaled by `multiplier`,
    after clipping gradients to the global norm `clip`, its moments kept in
    `moments`.

    Args:
        config: The run's ``optimizer`` setting, built by its builder in `optimizers`.
        multiplier: The schedule's multiplier of the peak learning rate
            (`lr_multiplier`).
        model: The model it trains, whose modules say which parameters decay and
            which Muon takes.
        clip: The global norm gradients are clipped to; None does not clip.
        moments: The dtype the optimizer's state is kept in between steps
            (`kept_in`).

    Returns:
        An optax transformation of the model's parameters' gradients.

    Raises:
        NotImplementedError: If no optimizer is registered under the config's type.

    Example::

        multiplier = lr_multiplier(run.train.schedule, run.train.steps)
        tx = optimizer(run.train.optimizer, multiplier, model, clip=1.0)
        trainer = Trainer(model, tx, objective(run.train.loss, run.arch))
    """
    tx = kept_in(optimizers.build(config, multiplier, model), moments)
    if clip is None:
        return tx
    return optax.chain(optax.clip_by_global_norm(clip), tx)


def kept_in(
    tx: optax.GradientTransformation, dtype: DTypeLike
) -> optax.GradientTransformation:
    """Return `tx` with its state's floating-point arrays, its moments, kept in
    `dtype` between steps and updated in float32: DeepSeek-V3's AdamW in bfloat16.
    Rounded to nearest, bfloat16 drops a change below 2^-9 of the moment, and a moment
    of decay b changes by 1 - b of the difference: at b2 = 0.95 a square gradient 4%
    off the second moment moves it, at 0.999 only one 3x off."""
    if jnp.dtype(dtype) == jnp.float32:
        return tx

    def init(params: PyTree[Float[Array, "*_"]]) -> optax.OptState:
        return _floats_in(tx.init(params), dtype)

    def update(
        updates: PyTree[Float[Array, "*_"]],
        state: optax.OptState,
        params: PyTree[Float[Array, "*_"]] | None = None,
    ) -> tuple[PyTree[Float[Array, "*_"]], optax.OptState]:
        updates, state = tx.update(updates, _floats_in(state, jnp.float32), params)
        return updates, _floats_in(state, dtype)

    return optax.GradientTransformation(init, update)


def _floats_in(tree: PyTree, dtype: DTypeLike) -> PyTree:
    def cast(x: Shaped[Array, "*s"]) -> Shaped[Array, "*s"]:
        return x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x

    return jax.tree.map(cast, tree)


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
    """AdamW by optax, its weight decay on the matrices outside the paths modules
    name in ``no_weight_decay`` (`weight_decay_mask`)."""
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
    """Muon for the matrices inside the blocks (`muon_matrices`), and AdamW (its
    ``adam``) for the rest: Nesterov momentum, orthogonalized by Newton-Schulz by
    the run's ``methods.orthogonalize``, scaled to ``consistent_rms``, with decoupled
    weight decay."""
    dims = muon_matrices(model)

    def labels(params: PyTree[Float[Array, "*_"]]) -> PyTree[str]:
        return jax.tree_util.tree_map_with_path(
            lambda p, _: "muon" if _keys(p) in dims else "adam", params
        )

    # Muon sees only its own parameters; the others are masked out of its tree.
    def shapes(updates: PyTree[Float[Array, "*_"]]) -> PyTree[MuonDimensionNumbers]:
        return jax.tree_util.tree_map_with_path(lambda p, _: dims[_keys(p)], updates)

    ns = newton_schulz(
        config.ns_steps,
        config.ns_coefficients,
        config.ns_algorithm,
        config.ns_restarts,
    )
    rule = optax.chain(
        momentum(config.momentum, config.nesterov),
        orthogonalized(ns, dims, model, dtype=config.ns_dtype),
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
    newton_schulz: NewtonSchulz,
    dims: dict[tuple, MuonDimensionNumbers],
    model: nnx.Module,
    dtype: DTypeLike = jnp.float32,
) -> optax.GradientTransformation:
    """Orthogonalize each update, a matrix or a stack of them as `dims` says by path,
    by `newton_schulz` (`newton_schulz`), in `dtype`, by the run's `orthogonalize`
    method. The method gets all of a step's updates at once, so it can choose where
    each runs, and gets them in `dtype`, so that what it moves is in it too."""
    model_params = nnx.state(model, nnx.Param)  # whose shardings the updates take

    def update(
        updates: PyTree[Float[Array, "*_"]],
        params: PyTree[Float[Array, "*_"]] | None = None,
    ) -> PyTree[Float[Array, "*_"]]:
        leaves, tree = jax.tree_util.tree_flatten_with_path(updates)
        keys = [_keys(path) for path, _ in leaves]
        shardings = shardings_by_path(model_params)
        states = shardings_by_path(model_params, as_state=True)
        orthogonal = parallel.method("orthogonalize")(
            [x.astype(dtype) for _, x in leaves],
            [dims[k] for k in keys],
            None if shardings is None else [shardings[k] for k in keys],
            None if states is None else [states[k] for k in keys],
            newton_schulz,
        )
        cast = [y.astype(x.dtype) for y, (_, x) in zip(orthogonal, leaves, strict=True)]
        return jax.tree.unflatten(tree, cast)

    return optax.stateless(update)


def newton_schulz(
    steps: int,
    coefficients: str = "jordan",
    algorithm: str = "standard",
    restarts: Sequence[int] | None = None,
) -> NewtonSchulz:
    """Return the Newton-Schulz that `algorithm` names in `ns_algorithms`, with
    `steps` iterations of the quintics `coefficients` names in `ns_coefficients`,
    for a matrix in the dtype it returns it in.

    Args:
        steps: The number of iterations.
        coefficients: The name of their quintics.
        algorithm: How it iterates: ``standard`` or ``gram``.
        restarts: The iterations after which ``gram`` restarts; None, its own.

    Returns:
        A function of a matrix, or a stack of them as its dimension numbers say,
        that orthogonalizes each.

    Raises:
        NotImplementedError: If either name is not registered.
    """
    coeffs = ns_coefficients.find(coefficients)(steps)
    return ns_algorithms.find(algorithm)(steps, coeffs, restarts)


@ns_algorithms.register("standard")
def standard(
    steps: int,
    coefficients: Float[Array, "*steps 3"],
    restarts: Sequence[int] | None = None,
) -> NewtonSchulz:
    """optax's iteration of the quintics on the matrix itself: three products an
    iteration, two of them by the matrix. It has no restarts."""

    def orthogonal(
        x: Float[Array, "*m"], dims: MuonDimensionNumbers
    ) -> Float[Array, "*m"]:
        return orthogonalize_via_newton_schulz(
            x, coefficients, steps, dimension_numbers=dims
        )

    return orthogonal


@ns_algorithms.register("gram")
def gram(
    steps: int,
    coefficients: Float[Array, "*steps 3"],
    restarts: Sequence[int] | None = None,
) -> NewtonSchulz:
    """Gram Newton-Schulz (Zhang, Amsel, Chen and Dao, 2026): the same quintics,
    iterated on the Gram matrix of the matrix's shorter side, with products by the
    matrix only to start, at each restart and at the end. It restarts from the
    matrix so far after each iteration in `restarts`, before the Gram matrix's
    rounding grows: by default after the second and every third after it (2, 5, 8,
    ...), which keeps 4 to 12 iterations of either quintic about as near orthogonal
    as the standard's. On a matrix too near square for Gram's to save products, it
    runs the standard iteration.

    It normalizes in float32 and iterates a matrix of 16 bits in float16, as its
    authors do: bfloat16 rounds the Gram matrix too coarsely."""
    quintics = jnp.broadcast_to(coefficients, (steps, 3))
    restarts = range(2, steps, 3) if restarts is None else restarts
    ends = [0, *sorted({i for i in restarts if 0 < i < steps}), steps]
    runs = [quintics[i:j] for i, j in itertools.pairwise(ends)]
    fallback = standard(steps, coefficients)

    def orthogonal(
        x: Float[Array, "*m"], dims: MuonDimensionNumbers
    ) -> Float[Array, "*m"]:
        flat, back = _compute_muon_reshape(x, dims)
        matrices = flat(x)
        wide = matrices.shape[-2] <= matrices.shape[-1]
        matrices = matrices if wide else matrices.mT
        if not _gram_saves(*matrices.shape[-2:], steps, len(runs)):
            return fallback(x, dims)
        y = _normalized(matrices)
        for run in runs:
            y = _gram_run(y @ y.mT, run) @ y
        y = y.astype(x.dtype)
        return back(y if wide else y.mT)

    return orthogonal


def _gram_saves(m: int, n: int, steps: int, runs: int) -> bool:
    """Whether `runs` of Gram's iterations of an m×n matrix, m ≤ n, multiply less
    than the standard iteration's ``steps (m³ + 2 m²n)``, every product computed
    whole: each run takes ``4 length - 3`` products of m×m matrices, and two by the
    matrix."""
    return (4 * steps - 3 * runs) * m + 2 * runs * n < steps * (m + 2 * n)


def _normalized(x: Float[Array, "b m n"]) -> Float[Array, "b m n"]:
    """Each matrix of `x` over its Frobenius norm, in float32, then in what Gram's
    iteration runs in: float16 for 16 bits."""
    dtype = jnp.float16 if jnp.finfo(x.dtype).bits == 16 else x.dtype
    x = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    return (x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + 1e-8)).astype(dtype)


def _gram_run(
    r: Float[Array, "b m m"], quintics: Float[Array, "steps 3"]
) -> Float[Array, "b m m"]:
    """Return the Q of a run of Gram iterations from the Gram matrix `r`, from Q = I:
    with ``Z = b R + c R²``, ``Q ← Q Z + a Q`` and ``R ← (Z + a I) R (Z + a I)``,
    Zhang et al.'s stabilized iteration. It adds each ``a`` in float32 before the
    product rounds: Z rounded with ``a I`` would lose it, and a product rounded
    before ``a`` is added would lose what cancels."""
    (a, b, c), *rest = quintics
    z = (r @ r) * c.astype(r.dtype) + r * b.astype(r.dtype)
    q = (z + a * jnp.eye(r.shape[-1], dtype=jnp.float32)).astype(r.dtype)
    for next_a, b, c in rest:
        rz = _plus(r, z, r, a)
        r = _plus(z, rz, rz, a)
        a, z = next_a, (r @ r) * c.astype(r.dtype) + r * b.astype(r.dtype)
        q = _plus(q, z, q, a)
    return q


def _plus(
    x: Float[Array, "*b m k"],
    y: Float[Array, "*b k n"],
    plus: Float[Array, "*b m n"],
    by: Float[ArrayLike, ""],
) -> Float[Array, "*b m n"]:
    """``x y + by plus``, summed in float32 and rounded to x's dtype once, as a GEMM's
    epilogue adds."""
    xy = jnp.matmul(x, y, preferred_element_type=jnp.float32)
    return (xy + plus.astype(jnp.float32) * by).astype(x.dtype)


@ns_coefficients.register("jordan")
def _(steps: int) -> Float[Array, "3"]:
    """Keller Jordan's quintic, (3.4445, -4.7750, 2.0315), the same at every
    iteration: optax's default."""
    return jnp.asarray((3.4445, -4.7750, 2.0315))  # Keller Jordan's, optax's default


# Amsel et al. (2025), for singular values in [0.001, 1]; the last one repeats.
POLAR_EXPRESS = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
)


@ns_coefficients.register("polar_express")
def _(steps: int, safety: float = 1.01) -> Float[Array, "steps 3"]:
    """Polar Express (Amsel et al., 2025): a quintic for each iteration, closer to
    orthogonal in as many; past the eighth, the last repeats.

    As its authors run it: each quintic but the last for singular values up to
    `safety`, and the first for a matrix divided by `safety` times its norm, where
    optax's divides by its norm."""
    found = [POLAR_EXPRESS[min(i, len(POLAR_EXPRESS) - 1)] for i in range(steps)]
    last = len(POLAR_EXPRESS) - 1
    scale = [safety ** (i < last) * safety ** (i == 0) for i in range(steps)]
    return jnp.asarray(
        [(a / s, b / s**3, c / s**5) for (a, b, c), s in zip(found, scale)]
    )


def shardings_by_path(
    params: nnx.State, as_state: bool = False
) -> dict[tuple, NamedSharding] | None:
    """Return the sharding of each of `params` by path under ``parallel.use``, or None
    outside it; with `as_state`, that of the optimizer's state of each, which FSDP
    splits under ``layout(zero=1)`` too."""
    if parallel.active() is None:
        return None
    if as_state:
        params = jax.tree.map(_as_state, params, is_leaf=_is_variable)
    found = jax.tree_util.tree_leaves_with_path(parallel.shardings(params))
    return {_keys(path): s for path, s in found}


def _as_state(param: nnx.Variable) -> nnx.OptState:
    """The optimizer's state of `param`, which takes its names."""
    return nnx.OptState(param.get_value(), **param.get_metadata())


def _is_variable(x: object) -> bool:
    return isinstance(x, nnx.Variable)


# --- How Muon reaches whole matrices: methods.orthogonalize ----------------------------
# A method takes every update Muon orthogonalizes in a step, the dimension numbers of
# each, the sharding of each one's parameter and of its optimizer state, where the
# update comes from (None outside parallel.use), and the Newton-Schulz to run; it
# returns the updates orthogonalized.


@methods.orthogonalize.register("gspmd")
def gspmd(
    updates: list[Float[Array, "*_"]],
    dims: list[MuonDimensionNumbers],
    shardings: list[NamedSharding] | None,
    states: list[NamedSharding] | None,
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
    states: list[NamedSharding] | None,
    newton_schulz: NewtonSchulz,
) -> list[Float[Array, "*_"]]:
    """Gather each update whole for Newton-Schulz, a stack matrix by matrix
    (`whole_matrix_spec`), then put it back on its parameter's shards: one gather per
    matrix instead of the all-reduces, with the work repeated on each device that
    holds the matrix."""
    if shardings is None:
        return gspmd(updates, dims, shardings, states, newton_schulz)
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
    axes = _splits(spec, len(shape))
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


def _splits(spec: P, ndim: int) -> list[Split]:
    """Return the mesh axes that `spec` splits each of `ndim` dimensions over."""
    return [as_split(spec[i] if i < len(spec) else None) for i in range(ndim)]


@methods.orthogonalize.register("owner")
def owner(
    updates: list[Float[Array, "*_"]],
    dims: list[MuonDimensionNumbers],
    shardings: list[NamedSharding] | None,
    states: list[NamedSharding] | None,
    newton_schulz: NewtonSchulz,
) -> list[Float[Array, "*_"]]:
    """Orthogonalize each matrix, a stack's one by one, whole on one device: its owner.
    The matrices of a shape and split are dealt out evenly over the devices, reach
    their owners by an all-to-all, run as one batch there, and go back to their
    parameters' shards by another all-to-all, and an all-gather over the axes that
    copy them.

    An update comes split as the optimizer's state is (`states`), under
    ``layout(zero=1)`` by FSDP too, and its matrices go back whole along what FSDP
    split: a side, or a stack's experts, each device's of which it orthogonalizes as
    they are, whole."""
    if shardings is None or states is None:
        return gspmd(updates, dims, shardings, states, newton_schulz)
    mesh, outs = shardings[0].mesh, [s.spec for s in shardings]
    ins = [s.spec for s in states]
    plans = [
        _plan(x.shape, d, i, o, mesh) for x, d, i, o in zip(updates, dims, ins, outs)
    ]
    per_device = jax.shard_map(
        lambda *xs: tuple(_orthogonalize_owned(list(xs), plans, newton_schulz)),
        mesh=mesh,
        in_specs=tuple(ins),
        out_specs=tuple(outs),
    )
    shard = jax.lax.with_sharding_constraint
    return list(
        per_device(*(shard(x, NamedSharding(mesh, s)) for x, s in zip(updates, ins)))
    )


class Owned(typing.NamedTuple):
    """How a device's shard of an update is a stack of matrix shards: its dimensions
    in `order` (the stacked ones, then each matrix's shorter side and its longer), the
    mesh axes that split each side, those the update is copied over, and those that
    the update splits but its parameter does not (FSDP's under zero=1); and the
    stacked dimensions those split, which the results gather whole."""

    order: tuple[int, ...]
    shape: tuple[int, int]  # a matrix's, whole
    rows: Split
    cols: Split
    copies: Split
    state_only: Split
    back_shape: tuple[int, ...]  # the shard the results come back as
    gathers: tuple[tuple[int, Split], ...]  # a stacked dimension, the axes to gather


def _plan(
    shape: tuple[int, ...],
    dims: MuonDimensionNumbers,
    spec: P,
    out: P,
    mesh: Mesh | AbstractMesh,
) -> Owned:
    """Return how an update of `shape`, split by `spec`, stacks into matrix shards,
    and how its parameter's shards, by `out`, are."""
    r, o = (_side(a, len(shape)) for a in (dims.reduction_axis, dims.output_axis))
    if shape[r] > shape[o]:  # optax's Newton-Schulz transposes it so too
        r, o = o, r
    splits, kept = _splits(spec, len(shape)), _splits(out, len(shape))
    used, sizes = {a for s in splits for a in s}, mesh.shape
    stacked = [i for i in range(len(shape)) if i not in (r, o)]
    back = [splits[i] if i in stacked else kept[i] for i in range(len(shape))]
    return Owned(
        order=(*stacked, r, o),
        shape=(shape[r], shape[o]),
        rows=splits[r],
        cols=splits[o],
        copies=tuple(a for a, n in sizes.items() if n > 1 and a not in used),
        state_only=tuple(a for a in sizes if a in used - {a for k in kept for a in k}),
        back_shape=tuple(
            n // math.prod(sizes[a] for a in k) for n, k in zip(shape, back)
        ),
        gathers=tuple((i, splits[i]) for i in stacked if splits[i] and not kept[i]),
    )


def _side(axis: int | Sequence[int], ndim: int) -> int:
    """The dimension of one side of a matrix: `owner` takes no side of several."""
    [a] = (axis,) if isinstance(axis, int) else axis
    return a % ndim


def _orthogonalize_owned(
    shards: list[Float[Array, "*_"]], plans: list[Owned], newton_schulz: NewtonSchulz
) -> list[Float[Array, "*_"]]:
    """On one device: send each stack of matrix shards to the matrices' owners, run
    Newton-Schulz on what this device owns, and bring the results back."""
    stacks = [_as_stack(x, p) for x, p in zip(shards, plans, strict=True)]
    groups: dict[tuple, list[int]] = {}
    for i, (x, p) in enumerate(zip(stacks, plans)):
        key = (x.dtype, *p.shape, p.rows, p.cols, p.copies, p.state_only)
        groups.setdefault(key, []).append(i)
    owned = {
        key: _to_owners(jnp.concatenate([stacks[i] for i in group]), plans[group[0]])
        for key, group in groups.items()
    }
    orthogonal = _batched(owned, newton_schulz)
    out: dict[int, Float[Array, "*_"]] = {}
    for key, group in groups.items():
        back = _from_owners(orthogonal[key], plans[group[0]])
        ends = np.cumsum([len(stacks[i]) for i in group])
        for i, x in zip(group, jnp.split(back[: ends[-1]], ends[:-1]), strict=True):
            out[i] = _from_stack(x, plans[i])
    return [out[i] for i in range(len(shards))]


def _as_stack(x: Float[Array, "*s"], p: Owned) -> Float[Array, "b m n"]:
    x = x.transpose(p.order)
    return x.reshape(-1, *x.shape[-2:])


def _from_stack(x: Float[Array, "b m n"], p: Owned) -> Float[Array, "*s"]:
    """`_as_stack` undone, onto the parameter's shard: each stacked dimension that only
    the optimizer's state splits gathered whole."""
    x = x.reshape(tuple(p.back_shape[i] for i in p.order))
    x = x.transpose(np.argsort(p.order))
    for dim, split in p.gathers:
        x = jax.lax.all_gather(x, split, axis=dim, tiled=True, to="invarying")
    return x


def _to_owners(x: Float[Array, "b m n"], p: Owned) -> Float[Array, "c M N"]:
    """Move a stack of matrix shards to the matrices' owners: padded to deal out
    evenly, this device's share among its copies, each side whole from the devices
    that split it."""
    devices = _size(p.rows + p.cols + p.copies)
    x = jnp.pad(x, ((0, -len(x) % devices), (0, 0), (0, 0)))
    if p.copies:
        share = len(x) // _size(p.copies)
        start = jax.lax.axis_index(p.copies) * share
        x = jax.lax.dynamic_slice_in_dim(x, start, share)
    if p.rows:
        x = jax.lax.all_to_all(x, p.rows, 0, 1, tiled=True)
    if p.cols:
        x = jax.lax.all_to_all(x, p.cols, 0, 2, tiled=True)
    return x


def _size(axes: Split) -> int:
    return math.prod(jax.lax.axis_size(a) for a in axes)


def _from_owners(x: Float[Array, "c M N"], p: Owned) -> Float[Array, "b m n"]:
    """`_to_owners` undone, but for the padding, onto the parameter's shards: every
    copy's, and a side that only the optimizer's state splits whole."""
    for side, split in ((2, p.cols), (1, p.rows)):
        if split and split == p.state_only:
            x = jax.lax.all_gather(x, split, axis=0, tiled=True, to="invarying")
        elif split:
            x = jax.lax.all_to_all(x, split, side, 0, tiled=True)
    if p.copies:
        x = jax.lax.all_gather(x, p.copies, axis=0, tiled=True, to="invarying")
    return x


def _batched(
    stacks: dict[tuple, Float[Array, "c m n"]], newton_schulz: NewtonSchulz
) -> dict[tuple, Float[Array, "c m n"]]:
    """Orthogonalize the stacks of whole matrices of each dtype and shape as one."""
    by_shape: dict[tuple, list[tuple]] = {}
    for key in stacks:
        by_shape.setdefault(key[:3], []).append(key)
    out = {}
    for keys in by_shape.values():
        joined = jnp.concatenate([stacks[k] for k in keys])
        done = newton_schulz(joined, MuonDimensionNumbers(1, 2))
        ends = np.cumsum([len(stacks[k]) for k in keys])
        out.update(zip(keys, jnp.split(done, ends[:-1]), strict=True))
    return out
