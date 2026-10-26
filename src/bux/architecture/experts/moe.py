"""Mixture of experts, with shared experts and LatentMoE projections.

The router picks each token's top-k experts (``MoE._route``), and a method of
``methods.moe`` runs them (docs/parallelism.md): "dense" runs every expert on every
token, the reference; "sorted", the default, runs each expert on its own picks alone
(``routed``), dropping none. A token's path through ``routed``, which expert
parallelism (``bux.parallel.experts``) takes apart:

    sort_by_expert  the picks in order of their experts, and how many each one has
    token_rows      each pick's token row, in that order
    experts         each row through its expert's FFN, by grouped GEMMs, times its
                    pick's weight
    token_sums      each token's sum of its k picks' rows

Shapes: b rows of t tokens of width d; E experts, k picks a token, f hidden units an
expert, and p projections into them: 2 for a gated MLP (up and gate), else 1.
"""

from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int, Key

from bux import config as cfg
from bux import parallel
from bux.architecture.common.sharding import linear
from bux.architecture.experts import grouped_blockscaled, grouped_glu
from bux.architecture.experts.grouped import ragged_dot
from bux.architecture.layers.mlp import MLP, activate
from bux.architecture.layers.norm import Norm
from bux.architecture.precision import blockscaled, matmuls
from bux.architecture.registry import Context, Taps, register, report, training
from bux.lang import ir
from bux.parallel import methods

RowNorm = Callable[[Float[Array, "n d"]], Float[Array, "n d"]]
LECUN = nnx.initializers.lecun_normal()


class Kernels(NamedTuple):
    """The experts' kernels, stacked: up, with the gate beside it in a gated MLP, and
    down; and their biases, where the experts have them."""

    up: Float[Array, "E d f p"]
    down: Float[Array, "E f d"]
    up_bias: Float[Array, "E f p"] | None = None
    down_bias: Float[Array, "E d"] | None = None


class Sorted(NamedTuple):
    """The M flat picks sorted by expert: their `order`, its `inverse` (where each
    pick went), and how many picks each expert has."""

    order: Int[Array, " M"]
    inverse: Int[Array, " M"]
    sizes: Int[Array, " E"]


class MoE(nnx.Module):
    """Token-choice top-k experts, run by the run's method of ``methods.moe``."""

    # Up's last dimension, of up and gate, is a batch one: Muon orthogonalizes them apart.
    muon_axes: ClassVar = {"up": (1, 2), "down": (1, 2)}
    no_weight_decay: ClassVar = ("up_bias", "down_bias")  # stacks, but of vectors

    def __init__(self, config: cfg.MoEConfig, width: int, rngs: nnx.Rngs):
        _check_groups(config)
        n, f = config.num_experts, config.expert.intermediate_size
        self.config = config
        # With shared_expert_sink, the router scores the shared experts too, last.
        sinks = config.num_shared_experts if config.shared_expert_sink else 0
        self.router = linear(
            width, n + sinks, ("embed", None), config.router_bias, rngs, in_compute=True
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

        # A gated MLP's up and gate, each hidden unit's two columns side by side,
        # (E, d, f, 2): one GEMM makes both, and where tensor parallelism splits the
        # hidden units, each up column stays with its gate's.
        p = 2 if config.expert.type == "gated_mlp" else 1
        self.up = _side_by_side(p, (n, latent, f), ("experts", inner, "mlp"), rngs)
        self.down = _stacked((n, f, latent), ("experts", "mlp", inner), rngs)
        biased = config.expert.bias
        self.up_bias = _zeros((n, f, p), ("experts", "mlp", None)) if biased else None
        self.down_bias = _zeros((n, latent), ("experts", inner)) if biased else None
        self.expert_norm = _norm(config.expert_norm, latent)
        self.shared = _shared_experts(config, width, rngs)
        self.shared_gate = (
            linear(width, 1, ("embed", None), False, rngs, in_compute=True)
            if config.shared_expert_gate
            else None
        )
        # The random logits' stream, as dropout keeps its own.
        self.routing = rngs["routing"].fork() if config.random_logits else None

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
        balances the experts, and routes by random logits under `random_logits`.
        Scores and top-k are in float32, the weights in x's dtype."""
        config, n = self.config, self.config.num_experts
        logits = self._logits(x) * config.router_logit_scale
        if self.routing is not None and training(taps):
            logits = random_logits(logits, self.routing())
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

    def _logits(self, x: Float[Array, "b t d"]) -> Float[Array, "b t s"]:
        """Return the router's logits in float32: x times its kernel, both in x's
        dtype, whose products float32 holds exactly. In bfloat16 the GEMM is one of
        cuBLAS's: on a B300 0.03 ms for Qwen3-30B-A3B's 32k tokens, where float32
        operands take 0.11. The backward multiplies in float32."""
        kernel = self.router.kernel[...].astype(x.dtype)
        logits = jnp.dot(x, kernel, preferred_element_type=jnp.float32)
        bias = self.router.bias
        return logits if bias is None else logits + bias[...].astype(jnp.float32)

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
        """Return the routed experts' kernels, and their biases, as arrays."""
        biases = (None if b is None else b[...] for b in (self.up_bias, self.down_bias))
        return Kernels(self.up[...], self.down[...], *biases)

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


def random_logits(
    logits: Float[Array, "b t e"], key: Key[Array, ""]
) -> Float[Array, "b t e"]:
    """Return logits drawn from N(0, 1) in place of `logits`, whose gradient goes to
    `logits` as if they were those (a straight-through estimator): the router runs
    and learns, but every expert is as likely a pick as any other."""
    drawn = jax.random.normal(key, logits.shape, logits.dtype)
    return drawn + (logits - jax.lax.stop_gradient(logits))  # drawn, exactly


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
    return nnx.Param(LECUN(rngs.params(), shape), names=names)


def _zeros(shape: tuple[int, ...], names: tuple[str | None, ...]) -> nnx.Param:
    return nnx.Param(jnp.zeros(shape), names=names)


def _side_by_side(
    p: int, shape: tuple[int, int, int], names: tuple[str, str, str], rngs: nnx.Rngs
) -> nnx.Param:
    """Return p stacks of matrices of `shape` (E, d, f) side by side, (E, d, f, p),
    each drawn as ``_stacked`` draws one: LECUN takes an array's last two dimensions
    for its matrices'."""
    stacks = [LECUN(rngs.params(), shape) for _ in range(p)]
    return nnx.Param(jnp.stack(stacks, axis=-1), names=(*names, None))


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
    up, down, up_bias, down_bias = moe.kernels()
    dense = dense_weights(idx, weights, moe.config.num_experts)
    projected = jnp.einsum("btd,edfp->btefp", z, up)
    if up_bias is not None:
        projected = projected + up_bias
    projected = parallel.shard(projected, "batch", "seq", "experts", "mlp", None)
    h = hidden_units(moe.config.expert, projected)
    if moe.expert_norm is None and down_bias is None:
        return jnp.einsum("btef,efd,bte->btd", h, down, dense)
    per_expert = jnp.einsum("btef,efd->bted", h, down)
    if down_bias is not None:
        per_expert = per_expert + down_bias
    if moe.expert_norm is not None:
        per_expert = moe.expert_norm(per_expert)
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
    with every expert's kernels, so that no device sorts another's tokens. Where the
    fused kernels run them and FSDP splits them by expert as the layer runs, each
    device quantizes its own shard of them before they are gathered."""
    ffn, norm = moe.config.expert, moe.expert_norm
    tokens = parallel.spec(("batch", "seq", None), z.shape)
    if not parallel.axes_of(tokens):
        return routed(z, idx, weights, moe.kernels(), ffn, norm)
    split = _split_by_expert(moe, tokens) if fused(ffn, norm, moe.up[...]) else ()

    def local(
        z: Float[Array, "b t d"],
        idx: Int[Array, "b t k"],
        weights: Float[Array, "b t k"],
        kernels: Kernels,
    ) -> Float[Array, "b t d"]:
        if not split:
            return routed(z, idx, weights, kernels, ffn, norm)
        up, down, quantized = grouped_glu.gathered(kernels.up, kernels.down, split)
        return routed(z, idx, weights, Kernels(up, down), ffn, norm, quantized)

    kernels = P(split) if split else P()
    return in_shard_map(local, tokens, kernels, tokens)(z, idx, weights, moe.kernels())


def _split_by_expert(moe: MoE, tokens: P) -> tuple[str, ...]:
    """Return the axes that FSDP splits the experts' kernels over by expert as a layer
    runs them, if the tokens split over each: with one microbatch, or a layer's
    parameters gathered as it runs (`methods.gather`)."""
    on = parallel.active()
    assert on is not None
    once = on.methods.microbatches > 1 and on.methods.gather == "step"
    if once or parallel.pipeline.pipelined():
        return ()
    shardings = parallel.shardings(nnx.state(moe, nnx.Param))
    up, down = (_splits(shardings[k], ndim) for k, ndim in (("up", 4), ("down", 3)))
    first = up[0]
    by_expert = down[0] == first and not any(up[1:] + down[1:])
    return first if by_expert and set(first) <= set(parallel.axes_of(tokens)) else ()


def _splits(sharding: object, ndim: int) -> list[tuple[str, ...]]:
    """Return the axes each of the `ndim` dimensions of an array with `sharding`
    splits over."""
    assert isinstance(sharding, NamedSharding)
    entries = [*sharding.spec, *(None,) * (ndim - len(sharding.spec))]
    return [parallel.layout.as_split(entry) for entry in entries]


def in_shard_map(
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
        axis_names=set(parallel.axes_of(tokens)),
        check_vma=False,
    )


# --- A token's path: sort by expert, run the experts, sort back --------------------


def routed(
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    kernels: Kernels,
    ffn: cfg.MLPConfig,
    norm: RowNorm | None = None,
    quantized: grouped_glu.Weights | None = None,
) -> Float[Array, "b t d"]:
    """Return each token's sum of its experts' outputs, weighted by `weights`. A pick
    outside the experts of `kernels` adds nothing, so that one device's experts can
    take every pick. Given the kernels `quantized`, the fused kernels take them so."""
    b, t, d = z.shape
    k, n = idx.shape[-1], kernels.up.shape[0]
    order, inverse, sizes = sort_by_expert(idx, n)
    tokens = z.reshape(b * t, d)
    rows = token_rows(tokens, order, inverse, k)
    kept = jnp.where(in_range(idx, n), weights, 0)
    row_weights = picked_weights(kept, order, inverse)
    picked = grouped_blockscaled.Picked(tokens, order // k)
    summed = grouped_blockscaled.Summed(order, inverse.reshape(b * t, k))
    out = experts(
        rows, sizes, row_weights, kernels, ffn, norm, picked, summed, quantized
    )
    return out.reshape(b, t, d)


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
    weights: Float[Array, " n"],
    kernels: Kernels,
    ffn: cfg.MLPConfig,
    norm: RowNorm | None = None,
    picked: grouped_blockscaled.Picked | None = None,
    summed: grouped_blockscaled.Summed | None = None,
    quantized: grouped_glu.Weights | None = None,
) -> Float[Array, "r d"]:
    """Run each row of `x` through its expert's FFN, then `norm`, times the row's
    weight in `weights`. The rows are sorted by expert: the first sizes[0] are expert
    0's, the next sizes[1] expert 1's, and so on; rows past them come out 0 before
    `norm`, with no bias. It depends on its arguments alone, so it runs on one
    device's experts in a shard_map as it does on all of them. The kernels and biases
    multiply in x's dtype, so that wider ones add up the gradients of several calls in
    their own. Given where x was `picked`, a format's quantizer reads it there; given
    the tokens its rows are `summed` into, it returns their sums. In MXFP8 on a
    Blackwell GPU, cuDNN's fused kernels run a SwiGLU FFN with no norm or bias
    (``grouped_glu``), its kernels as they are `quantized`, if they are."""
    up, down = kernels.up, kernels.down
    if fused(ffn, norm, up):
        y = grouped_glu.mlp(x, sizes, weights, up, down, picked, summed, quantized)
        return checkpoint_name(y, "experts")
    recipe = grouped_blockscaled.RECIPES.get(experts_format())
    n, (e, d, f, p) = x.shape[0], up.shape
    up, down = up.reshape(e, d, f * p).astype(x.dtype), down.astype(x.dtype)

    def bias(b: Float[Array, "E ..."] | None) -> Float[Array, "n _"] | None:
        return None if b is None else by_expert(b.reshape(e, -1), sizes, n, x.dtype)

    def times(
        x: Float[Array, "n e"],
        w: Float[Array, "E e f"],
        picked: grouped_blockscaled.Picked | None = None,
        summed: grouped_blockscaled.Summed | None = None,
    ) -> Float[Array, "r f"]:
        return grouped_product(x, w, sizes, recipe, picked, summed)

    projected = times(x, up, picked)  # up and gate by one GEMM
    if (up_bias := bias(kernels.up_bias)) is not None:
        projected = projected + up_bias
    projected, down_bias = projected.reshape(n, f, p), bias(kernels.down_bias)
    if norm is None:
        # Weighted before `down`, which is linear: the weights' gradient then needs
        # each row's f hidden units, not its d outputs. Its bias adds, weighted, after.
        y = times(weighted_units(ffn, projected, weights), down, summed=summed)
        if down_bias is None:
            return y
        b = weighted(down_bias, weights)
        return y + (b if summed is None else sums_of_tokens(b, summed))
    y = times(hidden_units(ffn, projected), down)
    y = weighted(norm(y if down_bias is None else y + down_bias), weights)
    return y if summed is None else sums_of_tokens(y, summed)


def by_expert(
    b: Float[Array, "E c"], sizes: Int[Array, " E"], rows: int, dtype: Any
) -> Float[Array, "n c"]:
    """Return each row's expert's row of b, in `dtype`: the first sizes[0] rows b[0],
    the next sizes[1] b[1], and so on, and 0 past them. By a one-hot matmul, whose
    gradient sums each expert's rows by another, where a gather's would scatter-add."""
    at, ends = jnp.arange(rows)[:, None], jnp.cumsum(sizes)
    one_hot = ((at >= ends - sizes) & (at < ends)).astype(dtype)
    product = jnp.dot(one_hot, b.astype(dtype), preferred_element_type=jnp.float32)
    return product.astype(dtype)


def fused(
    ffn: cfg.MLPConfig, norm: RowNorm | None, up: Float[Array, "E d f p"]
) -> bool:
    """Return whether cuDNN's fused kernels run the experts here
    (``grouped_glu.runs``)."""
    recipe = grouped_blockscaled.RECIPES.get(experts_format())
    passes = matmuls.in_force().passes
    return norm is None and grouped_glu.runs(recipe, passes, ffn, up)


def weighted_units(
    ffn: cfg.MLPConfig, projected: Float[Array, "n f p"], weights: Float[Array, " n"]
) -> Float[Array, "n f"]:
    """Return each row's hidden units times its weight. The backward computes them
    again, elementwise, from `projected`, which the activation's gradient keeps
    anyway: else it would keep them, for the weights' gradient, and their parts."""

    def units(
        projected: Float[Array, "n f p"], weights: Float[Array, " n"]
    ) -> Float[Array, "n f"]:
        return weighted(hidden_units(ffn, projected), weights)

    return jax.checkpoint(units)(projected, weights)


def hidden_units(
    ffn: cfg.MLPConfig, projected: Float[Array, "*n f p"]
) -> Float[Array, "*n f"]:
    """Return the hidden units of an FFN from its p projections into each: up, and
    for a gated MLP the gate (``activate``)."""
    gate = projected[..., 1] if projected.shape[-1] == 2 else None
    return activate(ffn, projected[..., 0], gate)


def weighted(
    x: Float[Array, "n e"], weights: Float[Array, " n"]
) -> Float[Array, "n e"]:
    """Return each row of x times its weight, in x's dtype."""
    return x * weights[:, None].astype(x.dtype)


def grouped_product(
    x: Float[Array, "n e"],
    w: Float[Array, "E e f"],
    sizes: Int[Array, " E"],
    recipe: blockscaled.Recipe | None,
    picked: grouped_blockscaled.Picked | None = None,
    summed: grouped_blockscaled.Summed | None = None,
) -> Float[Array, "r f"]:
    """Return each expert's rows of x times its matrix of w, named `experts` for
    ``remat``, or the sums of the tokens they are `summed` into: in `recipe`'s format
    where x and w are whole blocks of it (``grouped_blockscaled``, x read from where
    it was `picked`), else in compute (``grouped``)."""
    if recipe is None or not grouped_blockscaled.fits(x, w):
        y = checkpoint_name(ragged_dot(x, w, sizes), "experts")
        return y if summed is None else sums_of_tokens(y, summed)
    precision, seed = matmuls.in_force(), matmuls.seed()
    passes = precision.passes
    y = grouped_blockscaled.ragged_dot(
        x, w, sizes, recipe, seed, passes, picked, summed
    )
    return checkpoint_name(y, "experts")


def sums_of_tokens(
    y: Float[Array, "m d"], summed: grouped_blockscaled.Summed
) -> Float[Array, "t d"]:
    """Return each token's sum of its rows of y, as `summed` says (``token_sums``)."""
    tokens, k = summed.inverse.shape
    return token_sums(y, summed.inverse.reshape(tokens * k), summed.order, k)


def experts_format() -> str:
    """Return the format the routed experts multiply in, in the branch that is
    tracing: their role's (``matmuls.format_of``), or compute where the precision
    gives them none (``PrecisionConfig.experts``): unlike the other roles, they take
    `matmuls` only where a grouped GEMM multiplies in it."""
    if matmuls.in_force().experts() is None:
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


def picked_weights(
    weights: Float[Array, "*n k"], order: Int[Array, " m"], inverse: Int[Array, " M"]
) -> Float[Array, " m"]:
    """Return the weight of each pick in `order`, the first m of the M picks sorted,
    which `inverse` unsorts: ``token_rows`` of a pick a row."""
    return token_rows(weights.reshape(-1, 1), order, inverse, 1)[:, 0]


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
