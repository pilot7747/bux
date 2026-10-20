"""The objective: the loss a run minimizes, as a weighted sum of named terms.

A term is a function of every head's logits and the batch, and also a metric under
its name, so another term is a name, a weight and a function:

    obj = objective(run.train.loss, run.arch)
    obj.terms["logit_l2"] = Term(1e-4, lambda logits, batch: jnp.mean(logits[0] ** 2))
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float

from bux.architecture.segments import Segments
from bux.lang import ir
from bux.train import config as tcfg
from bux.train.data import Batch

Logits = Sequence[Float[Array, "b t vocab"]]  # every head's, in execution order


class Term(NamedTuple):
    weight: float
    fn: Callable[[Logits, Batch], Float[Array, ""]]


@dataclass
class Objective:
    """The sum of weight * term over `terms`."""

    terms: dict[str, Term]

    def __call__(
        self, logits: Logits, batch: Batch
    ) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
        """The loss to minimize, and its terms."""
        metrics = {name: term.fn(logits, batch) for name, term in self.terms.items()}
        total = jnp.zeros((), jnp.float32)
        for name, term in self.terms.items():
            total = total + term.weight * metrics[name]
        return total, metrics


def objective(c: tcfg.LossConfig, arch: ir.Architecture) -> Objective:
    """The loss of `c` for a model of `arch`: the cross-entropy of its first head, as
    `loss`, its `z_loss`, and as `mtp` the mean loss of the heads that predict further
    ahead."""
    ahead = lookahead(arch)
    terms = {"loss": Term(1.0, lambda logits, b: cross_entropy(logits[0], b, ahead[0]))}
    if c.z:
        terms["z_loss"] = Term(c.z, lambda logits, b: z_loss(logits[0], targets(b)))
    if len(ahead) > 1:

        def mtp(logits: Logits, b: Batch) -> Float[Array, ""]:
            further = zip(logits[1:], ahead[1:], strict=True)
            return jnp.mean(jnp.stack([cross_entropy(o, b, a) for o, a in further]))

        terms["mtp"] = Term(c.mtp, mtp)
    return Objective(terms)


def lookahead(arch: ir.Architecture) -> list[int]:
    """How far ahead each head predicts, in execution order: 1 + the nextn layers before it."""
    ahead, nextn = [], 0
    for layer in arch.layers():
        if layer.config.type == "nextn":
            nextn += 1
        elif layer.config.type == "head":
            ahead.append(1 + nextn)
    return ahead or [1]


def targets(batch: Batch, ahead: int = 1) -> Bool[Array, "b _"]:
    """Whether each position t but the row's last `ahead` predicts a target: token
    t + ahead is one (by loss_mask), and of t's own document."""
    within = Segments.of(jnp.asarray(batch.position_ids)).within
    return jnp.asarray(batch.loss_mask[:, ahead:]) & (within[:, ahead:] >= ahead)


def _mean(values: Float[Array, "b t"], weight: Bool[Array, "b _"]) -> Float[Array, ""]:
    """The mean of the first values of each row, where `weight` is true."""
    values = values[:, : weight.shape[1]]
    return jnp.sum(values * weight) / jnp.maximum(jnp.sum(weight), 1)


def cross_entropy(
    logits: Float[Array, "b t vocab"], batch: Batch, ahead: int = 1
) -> Float[Array, ""]:
    """The mean loss of logits over positions t that predict a target, token t + ahead.
    A position with nothing that far ahead in its row, or in its document, is left
    out."""
    weight = targets(batch, ahead)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits[:, : weight.shape[1]].astype(jnp.float32), batch.input_ids[:, ahead:]
    )
    return _mean(loss, weight)


def z_loss(
    logits: Float[Array, "b t vocab"], weight: Bool[Array, "b _"] | None = None
) -> Float[Array, ""]:
    """The mean of log(Z)^2, which keeps the softmax's normalizer near 1 (PaLM), over
    the positions `weight` says, as `targets` does, or all of them."""
    z = jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1) ** 2
    return jnp.mean(z) if weight is None else _mean(z, weight)
