"""Mixture of experts, with shared experts and LatentMoE projections.

The router picks each token's top-k experts, and a method of ``methods.moe`` runs
them (docs/parallelism.md): "dense" runs every expert on every token, the
reference; "sorted", the default, sorts the picks by expert and runs each expert on
its own rows (``experts``), dropping none.

Shapes: E experts, k picks per token, f hidden units per expert.
"""

from collections.abc import Callable
from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
from flax import nnx
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux import parallel
from bux.architecture.grouped import ragged_dot
from bux.architecture.mlp import MLP, activate
from bux.architecture.norm import Norm
from bux.architecture.registry import Context, Taps, register, report, training
from bux.architecture.sharding import linear
from bux.lang import ir
from bux.parallel import methods

Rows = Callable[[Float[Array, "n d"]], Float[Array, "n d"]]  # a norm of rows
# The experts' kernels, stacked: up, gate (None for an mlp) and down.
Kernels = tuple[
    Float[Array, "E d f"], Float[Array, "E d f"] | None, Float[Array, "E f d"]
]


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


def dense_weights(
    idx: Int[Array, "*n k"], weights: Float[Array, "*n k"], num_experts: int
) -> Float[Array, "*n E"]:
    """Return every expert's weight for each token, 0 outside its top-k picks `idx`."""
    one_hot = jax.nn.one_hot(idx, num_experts, dtype=weights.dtype)
    return (one_hot * weights[..., None]).sum(-2)


def experts(
    x: Float[Array, "n d"],
    sizes: Int[Array, " E"],
    up: Float[Array, "E d f"],
    gate: Float[Array, "E d f"] | None,
    down: Float[Array, "E f d"],
    ffn: cfg.MLPConfig,
    norm: Rows | None = None,
) -> Float[Array, "n d"]:
    """Run each row of `x` through its expert's FFN, then `norm`. The rows are sorted
    by expert: the first sizes[0] are expert 0's, the next sizes[1] expert 1's, and so
    on; rows past them come out 0 before `norm`. It depends on its arguments alone, so
    it runs on one device's experts in a shard_map as it does on all of them. Its
    products are named `experts` (``remat``)."""

    def product(
        x: Float[Array, "n e"], w: Float[Array, "E e f"]
    ) -> Float[Array, "n f"]:
        return checkpoint_name(ragged_dot(x, w, sizes), "experts")

    g = None if gate is None else product(x, gate)
    y = product(activate(ffn, product(x, up), g), down)
    return y if norm is None else norm(y)


def sort_by_expert(
    idx: Int[Array, "*n k"], num_experts: int
) -> tuple[Int[Array, " m"], Int[Array, " E"]]:
    """Return the order that sorts the flattened picks `idx` by expert (each expert's
    in token order), and how many picks each expert has. Picks outside
    [0, num_experts) sort last and count for none."""
    flat = idx.reshape(-1)
    flat = jnp.where((flat >= 0) & (flat < num_experts), flat, num_experts)
    order = jnp.argsort(flat, stable=True)
    return order, jnp.bincount(flat, length=num_experts + 1)[:num_experts]


def _take(x: Float[Array, "m d"], at: Int[Array, " M"]) -> Float[Array, "M d"]:
    """Return x[at], 0 where `at` is past the rows of x."""
    return jnp.take(x, at, axis=0, mode="fill", fill_value=0)


def _summed(
    x: Float[Array, "m d"], back: Int[Array, " M"], k: int
) -> Float[Array, "n d"]:
    """Return each row's sum of the rows of its k picks, in x where `back` says."""
    picks = _take(x, back).reshape(-1, k, x.shape[-1])
    return jnp.sum(picks, 1, dtype=jnp.float32).astype(x.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def pick_rows(
    z: Float[Array, "n d"], order: Int[Array, " m"], back: Int[Array, " M"], k: int
) -> Float[Array, "m d"]:
    """Return z[order // k], the rows of the picks in `order`: the first m of a
    permutation of the M = n k picks, k per row of z, which `back` undoes. Its
    gradient gathers by `back` and sums each row's k picks, where XLA's would
    scatter-add, 4 times slower on a B300."""
    return z[order // k]


def _pick_rows_forward(
    z: Float[Array, "n d"], order: Int[Array, " m"], back: Int[Array, " M"], k: int
) -> tuple[Float[Array, "m d"], Int[Array, " M"]]:
    return z[order // k], back


def _pick_rows_backward(
    k: int, back: Int[Array, " M"], g: Float[Array, "m d"]
) -> tuple[Float[Array, "n d"], None, None]:
    return _summed(g, back, k), None, None


pick_rows.defvjp(_pick_rows_forward, _pick_rows_backward)


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def sum_rows(
    x: Float[Array, "m d"], back: Int[Array, " M"], order: Int[Array, " m"], k: int
) -> Float[Array, "n d"]:
    """Return each row's sum of its k picks' rows of x, which holds the first m picks
    in `order`: the transpose of ``pick_rows``. Its gradient gathers by `order`."""
    return _summed(x, back, k)


def _sum_rows_forward(
    x: Float[Array, "m d"], back: Int[Array, " M"], order: Int[Array, " m"], k: int
) -> tuple[Float[Array, "n d"], Int[Array, " m"]]:
    return _summed(x, back, k), order


def _sum_rows_backward(
    k: int, order: Int[Array, " m"], g: Float[Array, "n d"]
) -> tuple[Float[Array, "m d"], None, None]:
    return g[order // k], None, None


sum_rows.defvjp(_sum_rows_forward, _sum_rows_backward)


@jax.custom_vjp
def permuted(
    x: Float[Array, "m d"], order: Int[Array, " m"], back: Int[Array, " m"]
) -> Float[Array, "m d"]:
    """Return x[order], for a permutation `order` that `back` undoes. Its gradient
    gathers by `back`, where XLA's would scatter-add."""
    return x[order]


def _permuted_forward(
    x: Float[Array, "m d"], order: Int[Array, " m"], back: Int[Array, " m"]
) -> tuple[Float[Array, "m d"], Int[Array, " m"]]:
    return x[order], back


def _permuted_backward(
    back: Int[Array, " m"], g: Float[Array, "m d"]
) -> tuple[Float[Array, "m d"], None, None]:
    return g[back], None, None


permuted.defvjp(_permuted_forward, _permuted_backward)


def routed(
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    up: Float[Array, "E d f"],
    gate: Float[Array, "E d f"] | None,
    down: Float[Array, "E f d"],
    ffn: cfg.MLPConfig,
    norm: Rows | None = None,
) -> Float[Array, "b t d"]:
    """Return each token's sum of its experts' outputs, weighted by `weights`: sort
    the picks `idx` by expert, run them through ``experts``, and sort them back. A pick
    outside the experts of `up` adds nothing, so that one device's experts can take
    every pick."""
    b, t, d = z.shape
    k, n = idx.shape[-1], up.shape[0]
    order, sizes = sort_by_expert(idx, n)
    back = jnp.argsort(order)
    rows = pick_rows(z.reshape(b * t, d), order, back, k)
    out = experts(rows, sizes, up, gate, down, ffn, norm)
    per_pick = permuted(out, back, order).reshape(b, t, k, d)
    kept = jnp.where((idx >= 0) & (idx < n), weights, 0)
    return jnp.einsum("btkd,btk->btd", per_pick, kept)


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


@methods.moe.register("sorted")
def run_sorted(
    moe: MoE,
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    taps: Taps | None = None,
) -> Float[Array, "b t d"]:
    """Run each expert on its own rows (``routed``), each device on its own tokens
    with every expert's kernels: a shard_map over the axes that split the tokens, so
    that no device sorts another's."""

    def local(
        z: Float[Array, "b t d"],
        idx: Int[Array, "b t k"],
        weights: Float[Array, "b t k"],
        kernels: Kernels,
    ) -> Float[Array, "b t d"]:
        return routed(z, idx, weights, *kernels, moe.config.expert, moe.expert_norm)

    tokens = parallel.spec(("batch", "seq", None), z.shape)
    if not _axes(tokens):
        return local(z, idx, weights, moe.kernels())
    # Untyped by what varies: tokamax's kernel takes jnp.searchsorted, which fails in
    # a shard_map on jax 0.12 where it is typed; nothing here communicates.
    return jax.shard_map(
        local,
        in_specs=(tokens, tokens, tokens, P()),
        out_specs=tokens,
        axis_names=_axes(tokens),
        check_vma=False,
    )(z, idx, weights, moe.kernels())


def _axes(spec: P) -> set[str]:
    """Return the mesh axes `spec` splits over."""
    return {a for s in spec if s for a in ((s,) if isinstance(s, str) else s)}


@register("moe")
def _(c: cfg.MoEConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context) -> MoE:
    return MoE(c, width, ctx.rngs)
