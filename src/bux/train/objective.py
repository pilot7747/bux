"""The objective: the loss a run minimizes, as a weighted sum of named terms.

A term is a function of every head's logits and the batch, and also a metric under
its name, so another term is a name, a weight and a function:

    obj = objective(run.train.loss, run.arch)
    obj.terms["logit_l2"] = Term(1e-4, lambda logits, batch: jnp.mean(logits[0] ** 2))

The losses that layers report from a training forward (``report`` in
``bux.architecture.registry``) are terms too, weighted by each layer.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float

from bux.architecture.registry import Loss
from bux.architecture.segments import Segments
from bux.lang import ir
from bux.train import config as tcfg
from bux.train.data import Batch

Metrics = dict[str, Float[Array, ""]]  # what a step measured, still on the device
Logits = Sequence[Float[Array, "b t vocab"]]  # every head's, in execution order


class Term(NamedTuple):
    weight: float
    fn: Callable[[Logits, Batch], Float[Array, ""]]


class Part(NamedTuple):
    """A part of the objective: its values' weighted sum, and each value by name."""

    loss: Float[Array, ""]
    metrics: Metrics


@dataclass
class Objective:
    """The sum of weight * term over `terms`, and over the losses layers report."""

    terms: dict[str, Term]

    def __call__(
        self, logits: Logits, batch: Batch, aux_losses: Sequence[Loss] = ()
    ) -> tuple[Part, Part]:
        """Return the objective's two parts: its terms, means over target tokens, and
        the losses layers report, means over rows, each name's values summed."""
        metrics = {name: term.fn(logits, batch) for name, term in self.terms.items()}
        loss = jnp.zeros((), jnp.float32)
        for name, term in self.terms.items():
            loss = loss + term.weight * metrics[name]
        aux_loss, aux_metrics = jnp.zeros((), jnp.float32), {}
        for name, weight, value in aux_losses:
            if name in self.terms:
                raise ValueError(f"a layer reports {name!r}, a term of the objective")
            aux_metrics[name] = aux_metrics.get(name, 0) + value
            aux_loss = aux_loss + weight * value
        return Part(loss, metrics), Part(aux_loss, aux_metrics)


def objective(config: tcfg.LossConfig, arch: ir.Architecture) -> Objective:
    """Return the loss of `config` for a model of `arch`: the first head's
    cross-entropy as `loss`, its `z_loss`, and as `mtp` the mean loss of the heads
    that predict further ahead."""
    offsets = prediction_offsets(arch)
    first = Term(1.0, lambda logits, b: cross_entropy(logits[0], b, offsets[0]))
    terms = {"loss": first}
    if config.z:
        z = Term(config.z, lambda logits, b: z_loss(logits[0], target_mask(b)))
        terms["z_loss"] = z
    if len(offsets) > 1:

        def mtp(logits: Logits, b: Batch) -> Float[Array, ""]:
            further = zip(logits[1:], offsets[1:], strict=True)
            losses = [cross_entropy(head, b, offset) for head, offset in further]
            return jnp.mean(jnp.stack(losses))

        terms["mtp"] = Term(config.mtp, mtp)
    return Objective(terms)


def prediction_offsets(arch: ir.Architecture) -> list[int]:
    """Return how many tokens ahead each head predicts, in execution order: 1 + the
    nextn layers before it."""
    offsets, nextn = [], 0
    for layer in arch.layers():
        if layer.config.type == "nextn":
            nextn += 1
        elif layer.config.type == "head":
            offsets.append(1 + nextn)
    return offsets or [1]


def target_mask(batch: Batch, offset: int = 1) -> Bool[Array, "b _"]:
    """Return whether each position t predicts a target: token t + offset is in the
    loss mask and in t's document. The mask leaves out each row's last `offset`."""
    within = Segments.of(jnp.asarray(batch.position_ids)).within
    return jnp.asarray(batch.loss_mask[:, offset:]) & (within[:, offset:] >= offset)


def _masked_mean(
    values: Float[Array, "b t"], mask: Bool[Array, "b t"]
) -> Float[Array, ""]:
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1)


def cross_entropy(
    logits: Float[Array, "b t vocab"], batch: Batch, offset: int = 1
) -> Float[Array, ""]:
    """Return the mean cross-entropy of predicting token t + offset from each position
    t that has a target (`target_mask`). The last `offset` positions are masked out
    rather than sliced off the logits, whose gradient would then pad a copy of them."""
    pad = ((0, 0), (0, offset))
    mask = jnp.pad(target_mask(batch, offset), pad)
    targets = jnp.pad(jnp.asarray(batch.input_ids)[:, offset:], pad)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), targets
    )
    return _masked_mean(loss, mask)


def z_loss(
    logits: Float[Array, "b t vocab"], mask: Bool[Array, "b _"] | None = None
) -> Float[Array, ""]:
    """Return the mean of log(Z)^2, which keeps the softmax's normalizer near 1 (PaLM),
    over the first positions of each row that `mask` covers, or over all of them."""
    z = jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1) ** 2
    return jnp.mean(z) if mask is None else _masked_mean(z[:, : mask.shape[1]], mask)
