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

from bux.lang import ir
from bux.train import config as tcfg
from bux.train.data import Batch

Logits = Sequence[jax.Array]  # every head's, (batch, seq, vocab), in execution order


class Term(NamedTuple):
    weight: float
    fn: Callable[[Logits, Batch], jax.Array]  # a scalar


@dataclass
class Objective:
    """The sum of weight * term over `terms`."""

    terms: dict[str, Term]

    def __call__(
        self, logits: Logits, batch: Batch
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
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
        terms["z_loss"] = Term(c.z, lambda logits, b: z_loss(logits[0]))
    if len(ahead) > 1:

        def mtp(logits: Logits, b: Batch) -> jax.Array:
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


def cross_entropy(logits: jax.Array, batch: Batch, ahead: int = 1) -> jax.Array:
    """The mean loss of logits over positions t that predict a target, token t + ahead,
    of their own document: at least `ahead` tokens into it. A position with nothing
    that far ahead in its row, or in its document, is left out."""
    targets = batch.input_ids[:, ahead:]
    weight = batch.loss_mask[:, ahead:] & (batch.position_ids[:, ahead:] >= ahead)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits[:, : targets.shape[1]].astype(jnp.float32), targets
    )
    return jnp.sum(loss * weight) / jnp.maximum(jnp.sum(weight), 1)


def z_loss(logits: jax.Array) -> jax.Array:
    """The mean of log(Z)^2, which keeps the softmax's normalizer near 1 (PaLM)."""
    return jnp.mean(jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1) ** 2)
