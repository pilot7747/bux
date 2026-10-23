"""Mixture of experts, with shared experts and LatentMoE projections.

The router picks each token's top-k experts (``MoE._route``), and a method of
``methods.moe`` runs them (docs/parallelism.md): "dense" runs every expert on every
token, the reference; "sorted", the default, runs each expert on its own picks alone
(``routed``), dropping none. A token's path through ``routed``, which expert
parallelism (``bux.parallel.experts``) takes apart:

    sort_by_expert  the picks in order of their experts, and how many each one has
    token_rows      each pick's token row, in that order
    experts         each row through its expert's FFN, by a grouped GEMM
    permuted        each pick's output back in token order
    einsum          each token's sum of its k outputs, weighted by the router

Shapes: b rows of t tokens of width d; E experts, k picks a token, f hidden units an
expert.
"""

from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from bux import config as cfg
from bux import parallel
from bux.architecture.common.sharding import linear
from bux.architecture.experts import grouped_blockscaled
from bux.architecture.experts.grouped import ragged_dot
from bux.architecture.layers.mlp import MLP, activate
from bux.architecture.layers.norm import Norm
from bux.architecture.precision import blockscaled, matmuls
from bux.architecture.registry import Context, Taps, register, report, training
from bux.lang import ir
from bux.parallel import methods

RowNorm = Callable[[Float[Array, "n d"]], Float[Array, "n d"]]
# The experts' kernels, stacked: up, gate (None for an mlp) and down.
Kernels = tuple[
    Float[Array, "E d f"], Float[Array, "E d f"] | None, Float[Array, "E f d"]
]


class Sorted(NamedTuple):
    """The M flat picks sorted by expert: their `order`, its `inverse` (where each
    pick went), and how many picks each expert has."""

    order: Int[Array, " M"]
    inverse: Int[Array, " M"]
    sizes: Int[Array, " E"]


class MoE(nnx.Module):
    """Token-choice top-k experts, run by the run's method of ``methods.moe``."""

    muon_axes: ClassVar = {"up": (1, 2), "gate": (1, 2), "down": (1, 2)}

    def __init__(self, config: cfg.MoEConfig, width: int, rngs: nnx.Rngs):
        _check_groups(config)
        n, f = config.num_experts, config.expert.intermediate_size
        self.config = config
        # With shared_expert_sink, the router scores the shared experts too, last.
        sinks = config.num_shared_experts if config.shared_expert_sink else 0
        self.router = linear(
            width, n + sinks, ("embed", None), False, rngs, in_compute=True
        )
        aux_free = config.load_balancing == "aux_free"
        self.router_bias = nnx.Variable(jnp.zeros(n)) if aux_free else None
        # The picks of each expert since the bias last moved; with `update_at_step_end`
        # set (by the trainer), the bias moves in `step_end_update`, once per step.
        self.router_load = nnx.Variable(jnp.zeros(n)) if aux_free else None
        self.update_at_step_end = False

        # LatentMoE: the routed experts work between these two projections.
        latent = config.latent_size or width
        inner = "latent" if config.latent_size else "embed"  # the width the experts see
        self.to_latent = (
            linear(width, latent, ("embed", "latent"), False, rngs, role="latent")
            if config.latent_size
            else None
        )
        self.from_latent = (
            linear(latent, width, ("latent", "embed"), False, rngs, role="latent")
            if config.latent_size
            else None
        )
        self.latent_input_norm = _norm(config.latent_input_norm, latent)
        self.latent_output_norm = _norm(config.latent_output_norm, latent)

        self.up = _stacked((n, latent, f), ("experts", inner, "mlp"), rngs)
        self.gate = (
            _stacked((n, latent, f), ("experts", inner, "mlp"), rngs)
            if config.expert.type == "gated_mlp"
            else None
        )
        self.down = _stacked((n, f, latent), ("experts", "mlp", inner), rngs)
        self.expert_norm = _norm(config.expert_norm, latent)
        self.shared = _shared_experts(config, width, rngs)
        self.shared_gate = (
            linear(width, 1, ("embed", None), False, rngs, in_compute=True)
            if config.shared_expert_gate
            else None
        )

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        idx, weights, sinks = self._route(x, taps)
        y = parallel.method("moe")(self, self._latent(x), idx, weights, taps)
        if self.latent_output_norm is not None:
            y = self.latent_output_norm(y)
        if self.from_latent is not None:
            y = self.from_latent(y)
        return y if self.shared is None else y + self._shared(x, sinks)

    def _route(
        self, x: Float[Array, "b t d"], taps: Taps | None = None
    ) -> tuple[
        Int[Array, "b t k"], Float[Array, "b t k"], Float[Array, "b t s"] | None
    ]:
        """Return each token's top-k experts and their weights, and the shared
        experts' weights if the router scores them too. A training forward also
        balances the experts. Scores and top-k are in float32, the weights in x's
        dtype."""
        config, n = self.config, self.config.num_experts
        logits = self.router(x.astype(jnp.float32)) * config.router_logit_scale
        match config.router:
            case "softmax":
                scores = jax.nn.softmax(logits)
            case "sigmoid":
                scores = jax.nn.sigmoid(logits)
            case "sqrtsoftplus":
                scores = jnp.sqrt(jax.nn.softplus(logits))
        scores, sinks = scores[..., :n], scores[..., n:]
        choice = scores if self.router_bias is None else scores + self.router_bias[...]
        if config.num_groups > 1:
            choice = _limit_groups(choice, config.num_groups, config.top_k_groups)
        _, idx = jax.lax.top_k(choice, config.top_k)
        if training(taps):
            self._balance(scores, idx, taps)
        weights = jnp.take_along_axis(scores, idx, axis=-1)
        if config.normalize_top_k:
            total = weights.sum(-1, keepdims=True) + sinks.sum(-1, keepdims=True)
            weights, sinks = weights / total, sinks / total
        scale = config.routed_scaling_factor
        weights, sinks = (w.astype(x.dtype) * scale for w in (weights, sinks))
        return idx, weights, sinks if sinks.shape[-1] else None

    def _balance(
        self, scores: Float[Array, "b t E"], idx: Int[Array, "b t k"], taps: Taps
    ) -> None:
        """Report the balance loss, and count the picks for the router bias, which
        moves now unless `update_at_step_end`."""
        config = self.config
        picks = jax.nn.one_hot(idx, scores.shape[-1]).sum(-2)
        if config.aux_loss_weight:
            per_sequence = config.aux_loss_scope == "sequence"
            loss = balance_loss(scores, picks, config.top_k, per_sequence)
            report(taps, "balance_loss", config.aux_loss_weight, loss)
        if self.router_load is not None:
            self.router_load[...] = self.router_load[...] + picks.sum((0, 1))
            if not self.update_at_step_end:
                self.step_end_update()

    def step_end_update(self) -> None:
        """Move the router bias toward the experts picked less than the mean since it
        last moved (DeepSeek-V3), and count again from zero."""
        if self.router_bias is None or self.router_load is None:
            return
        load = self.router_load[...]
        step = self.config.bias_update_rate * jnp.sign(load.mean() - load)
        self.router_bias[...] = self.router_bias[...] + step
        self.router_load[...] = jnp.zeros_like(load)

    def kernels(self) -> Kernels:
        """Return the routed experts' kernels as arrays."""
        gate = None if self.gate is None else self.gate[...]
        return self.up[...], gate, self.down[...]

    def _latent(self, x: Float[Array, "b t d"]) -> Float[Array, "b t l"]:
        """Return the tokens as the routed experts take them: in LatentMoE, projected
        to the latent width and normalized."""
        z = x if self.to_latent is None else self.to_latent(x)
        return z if self.latent_input_norm is None else self.latent_input_norm(z)

    def _shared(
        self, x: Float[Array, "b t d"], sinks: Float[Array, "b t s"] | None
    ) -> Float[Array, "b t d"]:
        """Return the shared experts' output. If the router scores them, each one's
        share of the hidden units is weighted by its `sinks`."""
        assert self.shared is not None
        if sinks is None:
            y = self.shared(x)
        else:
            h = self.shared.hidden(x)
            h = h * jnp.repeat(sinks, h.shape[-1] // sinks.shape[-1], axis=-1)
            y = self.shared.dropout(self.shared.down(h))
        if self.shared_gate is not None:
            y = y * jax.nn.sigmoid(self.shared_gate(x))
        return y


def _limit_groups(
    choice: Float[Array, "b t E"], groups: int, keep: int
) -> Float[Array, "b t E"]:
    """Return the scores at -inf outside the `keep` best of `groups` groups of
    experts, each group scored by the sum of its best two (DeepSeek-V3)."""
    b, t, n = choice.shape
    grouped = choice.reshape(b, t, groups, n // groups)
    best = jax.lax.top_k(grouped, min(2, n // groups))[0].sum(-1)
    kept = jax.nn.one_hot(jax.lax.top_k(best, keep)[1], groups).sum(-2) > 0
    return jnp.where(kept[..., None], grouped, -jnp.inf).reshape(b, t, n)


def balance_loss(
    scores: Float[Array, "b t E"], picks: Float[Array, "b t E"], k: int, sequence: bool
) -> Float[Array, ""]:
    """Return E * sum_e f_e P_e, with f_e the fraction of the top-k `picks` that are
    expert e and P_e its mean score normalized over the experts; over the batch, or
    per sequence and then averaged (DeepSeek-V3, eqs. 17-20). It is 1 when both are
    uniform."""
    over = 1 if sequence else (0, 1)
    f = picks.mean(over) / k
    p = (scores / scores.sum(-1, keepdims=True)).mean(over)
    return scores.shape[-1] * (f * p).sum(-1).mean()


def _check_groups(config: cfg.MoEConfig) -> None:
    n, groups, picked = config.num_experts, config.num_groups, config.top_k_groups
    if n % groups or config.top_k > picked * (n // groups):
        raise ValueError(
            f"{groups} groups of {n} experts, {picked} of them picked, cannot hold "
            f"the top {config.top_k}"
        )


def _norm(config: cfg.NormalizationConfig | None, width: int) -> Norm | None:
    return None if config is None else Norm(config, width)


def _stacked(
    shape: tuple[int, int, int], names: tuple[str, str, str], rngs: nnx.Rngs
) -> nnx.Param:
    init = nnx.initializers.lecun_normal()
    return nnx.Param(init(rngs.params(), shape), names=names)


def _shared_experts(config: cfg.MoEConfig, width: int, rngs: nnx.Rngs) -> MLP | None:
    if not config.num_shared_experts:
        return None
    shared = config.shared_expert or config.expert
    size = shared.intermediate_size * config.num_shared_experts
    return MLP(shared, width, rngs, size=size)


# --- Methods: (moe, z, idx, weights, taps) -> the routed sum, in the experts' width -


@methods.moe.register("dense")
def run_dense(
    moe: MoE,
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    taps: Taps | None = None,
) -> Float[Array, "b t d"]:
    """Run every expert on every token and weight the outputs by the router: the
    reference. Each device runs every expert on its own tokens."""
    up, gate, down = moe.kernels()
    dense = dense_weights(idx, weights, moe.config.num_experts)

    def project(w: Float[Array, "E d f"]) -> Float[Array, "b t E f"]:
        h = jnp.einsum("btd,edf->btef", z, w)
        return parallel.shard(h, "batch", "seq", "experts", "mlp")

    h = activate(
        moe.config.expert, project(up), None if gate is None else project(gate)
    )
    if moe.expert_norm is None:
        return jnp.einsum("btef,efd,bte->btd", h, down, dense)
    per_expert = moe.expert_norm(jnp.einsum("btef,efd->bted", h, down))
    return jnp.einsum("bted,bte->btd", per_expert, dense)


def dense_weights(
    idx: Int[Array, "*n k"], weights: Float[Array, "*n k"], num_experts: int
) -> Float[Array, "*n E"]:
    """Return every expert's weight for each token, 0 outside its top-k picks `idx`."""
    one_hot = jax.nn.one_hot(idx, num_experts, dtype=weights.dtype)
    return (one_hot * weights[..., None]).sum(-2)


@methods.moe.register("sorted")
def run_sorted(
    moe: MoE,
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    taps: Taps | None = None,
) -> Float[Array, "b t d"]:
    """Run each expert on its own rows (``routed``), each device on its own tokens
    with every expert's kernels, so that no device sorts another's tokens."""

    def local(
        z: Float[Array, "b t d"],
        idx: Int[Array, "b t k"],
        weights: Float[Array, "b t k"],
        kernels: Kernels,
    ) -> Float[Array, "b t d"]:
        return routed(z, idx, weights, *kernels, moe.config.expert, moe.expert_norm)

    tokens = parallel.spec(("batch", "seq", None), z.shape)
    if not axes_of(tokens):
        return local(z, idx, weights, moe.kernels())
    return on_each_device(local, tokens, P(), tokens)(z, idx, weights, moe.kernels())


def on_each_device(
    local: Callable[..., Any], tokens: P, kernels: P, out: Any
) -> Callable[..., Any]:
    """Return `local` in a shard_map manual over the axes `tokens` splits: each device
    takes its own tokens, picks and weights, split as `tokens`, and the experts'
    kernels, split as `kernels`; `out` splits what it returns."""
    # Untyped by what varies: tokamax's kernel takes jnp.searchsorted, which fails in
    # a shard_map on jax 0.12 where it is typed.
    return jax.shard_map(
        local,
        in_specs=(tokens, tokens, tokens, kernels),
        out_specs=out,
        axis_names=set(axes_of(tokens)),
        check_vma=False,
    )


def axes_of(spec: P) -> tuple[str, ...]:
    """Return the mesh axes `spec` splits over, in order."""
    return tuple(a for s in spec if s for a in ((s,) if isinstance(s, str) else s))


# --- A token's path: sort by expert, run the experts, sort back --------------------


def routed(
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    up: Float[Array, "E d f"],
    gate: Float[Array, "E d f"] | None,
    down: Float[Array, "E f d"],
    ffn: cfg.MLPConfig,
    norm: RowNorm | None = None,
) -> Float[Array, "b t d"]:
    """Return each token's sum of its experts' outputs, weighted by `weights`. A pick
    outside the experts of `up` adds nothing, so that one device's experts can take
    every pick."""
    b, t, d = z.shape
    k, n = idx.shape[-1], up.shape[0]
    order, inverse, sizes = sort_by_expert(idx, n)
    rows = token_rows(z.reshape(b * t, d), order, inverse, k)
    out = experts(rows, sizes, up, gate, down, ffn, norm)
    per_pick = permuted(out, inverse, order).reshape(b, t, k, d)
    kept = jnp.where(in_range(idx, n), weights, 0)
    return jnp.einsum("btkd,btk->btd", per_pick, kept)


def sort_by_expert(idx: Int[Array, "*n k"], num_experts: int) -> Sorted:
    """Return the flat picks `idx` sorted by expert, each expert's in token order.
    Picks outside [0, num_experts) sort last and count for none."""
    flat = idx.reshape(-1)
    flat = jnp.where(in_range(flat, num_experts), flat, num_experts)
    order = jnp.argsort(flat, stable=True)
    sizes = jnp.bincount(flat, length=num_experts + 1)[:num_experts]
    return Sorted(order, jnp.argsort(order), sizes)


def in_range(idx: Int[Array, "*s"], num_experts: int) -> Bool[Array, "*s"]:
    """Return which picks are of the experts [0, num_experts)."""
    return (idx >= 0) & (idx < num_experts)


def experts(
    x: Float[Array, "n d"],
    sizes: Int[Array, " E"],
    up: Float[Array, "E d f"],
    gate: Float[Array, "E d f"] | None,
    down: Float[Array, "E f d"],
    ffn: cfg.MLPConfig,
    norm: RowNorm | None = None,
) -> Float[Array, "n d"]:
    """Run each row of `x` through its expert's FFN, then `norm`. The rows are sorted
    by expert: the first sizes[0] are expert 0's, the next sizes[1] expert 1's, and so
    on; rows past them come out 0 before `norm`. It depends on its arguments alone, so
    it runs on one device's experts in a shard_map as it does on all of them."""
    recipe = grouped_blockscaled.RECIPES.get(experts_format())

    def times(x: Float[Array, "n e"], w: Float[Array, "E e f"]) -> Float[Array, "n f"]:
        return grouped_product(x, w, sizes, recipe)

    if gate is not None and recipe is not None and grouped_blockscaled.fits(x, up):
        # One GEMM for both: x quantizes once.
        u, g = jnp.split(times(x, jnp.concatenate([up, gate], -1)), 2, axis=-1)
    else:
        u, g = times(x, up), None if gate is None else times(x, gate)
    y = times(activate(ffn, u, g), down)
    return y if norm is None else norm(y)


def grouped_product(
    x: Float[Array, "n e"],
    w: Float[Array, "E e f"],
    sizes: Int[Array, " E"],
    recipe: blockscaled.Recipe | None,
) -> Float[Array, "n f"]:
    """Return each expert's rows of x times its matrix of w, named `experts` for
    ``remat``: in `recipe`'s format where x and w are whole blocks of it
    (``grouped_blockscaled``), else in compute (``grouped``)."""
    if recipe is None or not grouped_blockscaled.fits(x, w):
        return checkpoint_name(ragged_dot(x, w, sizes), "experts")
    precision, seed = matmuls.in_force(), matmuls.seed()
    y = grouped_blockscaled.ragged_dot(x, w, sizes, recipe, seed, precision.passes)
    return checkpoint_name(y, "experts")


def experts_format() -> str:
    """Return the format the routed experts multiply in, in the branch that is
    tracing: their role's (``matmuls.format_of``), or compute where `roles.experts` is
    none. Unlike the other roles, they take no format from `matmuls`: on B300s their
    grouped MXFP8 GEMM trains the baseline's MoE slower than bfloat16 does."""
    if matmuls.in_force().roles.experts is None:
        return "compute"
    return matmuls.format_of("experts")


# The gathers of a token's path, whose gradients gather too, by the inverse
# permutation, where XLA's would scatter-add, 4 times slower on a B300.


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def token_rows(
    z: Float[Array, "n d"], order: Int[Array, " m"], inverse: Int[Array, " M"], k: int
) -> Float[Array, "m d"]:
    """Return z[order // k], the token row of each pick in `order`: the first m of
    the M = n k picks sorted, k a row of z, which `inverse` unsorts. Its gradient
    sums each token's picks'."""
    return z[order // k]


def _token_rows_forward(
    z: Float[Array, "n d"], order: Int[Array, " m"], inverse: Int[Array, " M"], k: int
) -> tuple[Float[Array, "m d"], Int[Array, " M"]]:
    return z[order // k], inverse


def _token_rows_backward(
    k: int, inverse: Int[Array, " M"], g: Float[Array, "m d"]
) -> tuple[Float[Array, "n d"], None, None]:
    return _sum_by_token(g, inverse, k), None, None


token_rows.defvjp(_token_rows_forward, _token_rows_backward)


@jax.custom_vjp
def permuted(
    x: Float[Array, "m d"], order: Int[Array, " m"], inverse: Int[Array, " m"]
) -> Float[Array, "m d"]:
    """Return x[order], for a permutation `order` that `inverse` undoes. Its gradient
    gathers by `inverse`."""
    return x[order]


def _permuted_forward(
    x: Float[Array, "m d"], order: Int[Array, " m"], inverse: Int[Array, " m"]
) -> tuple[Float[Array, "m d"], Int[Array, " m"]]:
    return x[order], inverse


def _permuted_backward(
    inverse: Int[Array, " m"], g: Float[Array, "m d"]
) -> tuple[Float[Array, "m d"], None, None]:
    return g[inverse], None, None


permuted.defvjp(_permuted_forward, _permuted_backward)


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def token_sums(
    x: Float[Array, "m d"], inverse: Int[Array, " M"], order: Int[Array, " m"], k: int
) -> Float[Array, "n d"]:
    """Return each token's sum of its k picks' rows of x, which holds the first m of
    the picks `order` sorts; a pick past them adds 0. The transpose of
    ``token_rows``: its gradient gathers by `order`."""
    return _sum_by_token(x, inverse, k)


def _token_sums_forward(
    x: Float[Array, "m d"], inverse: Int[Array, " M"], order: Int[Array, " m"], k: int
) -> tuple[Float[Array, "n d"], Int[Array, " m"]]:
    return _sum_by_token(x, inverse, k), order


def _token_sums_backward(
    k: int, order: Int[Array, " m"], g: Float[Array, "n d"]
) -> tuple[Float[Array, "m d"], None, None]:
    return g[order // k], None, None


token_sums.defvjp(_token_sums_forward, _token_sums_backward)


def _sum_by_token(
    x: Float[Array, "m d"], inverse: Int[Array, " M"], k: int
) -> Float[Array, "n d"]:
    """Return each token's sum of its k picks' rows, in float32: x[inverse], 0 where
    `inverse` is past the rows of x."""
    picks = jnp.take(x, inverse, axis=0, mode="fill", fill_value=0)
    summed = jnp.sum(picks.reshape(-1, k, x.shape[-1]), 1, dtype=jnp.float32)
    return summed.astype(x.dtype)


@register("moe")
def _(c: cfg.MoEConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context) -> MoE:
    return MoE(c, width, ctx.rngs)
