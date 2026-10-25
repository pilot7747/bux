"""The objective: the loss a run minimizes, as a weighted sum of named terms.

A term is a function of every head's logits and the batch, and also a metric under
its name, so another term is a name, a weight and a function::

    obj = objective(run.train.loss, run.arch)
    obj.terms["logit_l2"] = Term(1e-4, lambda logits, batch: jnp.mean(logits[0] ** 2))

A head's logits are made only when a term reads them (`Heads`). The terms here take
only their sums over positions (`Heads.sums`), which ``methods.loss = chunks`` takes a
chunk of positions at a time, and so never makes them all (`bux.architecture.logits`).
The losses that layers report from a training forward (``report`` in
`bux.architecture.registry`) are terms too, weighted by each layer.

Position t of a row predicts token t + offset, a target where it is in the loss mask
and in t's document (`target_mask`); a term is a mean over the batch's targets.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple, overload

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float, Int

from bux.architecture.common.segments import Segments
from bux.architecture.logits import HeadOutput, Scores, Sums
from bux.architecture.registry import Loss
from bux.lang import ir
from bux.train import config as tcfg
from bux.train.data import Batch

#: What a step measured, by name, as scalars still on the device.
Metrics = dict[str, Float[Array, ""]]
Output = HeadOutput | Float[Array, "b t vocab"]  # a head's output, or its logits


class Heads(Sequence):  # not Sequence[Array]: a type checker would read an item
    """Every head's logits on `batch`, in execution order, each made once a term
    reads it. `sums` gives a head's cross-entropy and z-loss, without them.

    ``heads[i]`` is head i's logits, ``b t vocab``: rows, positions, the vocabulary.

    Args:
        outputs: Each head's output (`~bux.architecture.logits.HeadOutput`), or its
            logits.
        batch: The rows the heads ran on.
    """

    def __init__(self, outputs: Sequence[Output], batch: Batch):
        self.outputs, self.batch = list(outputs), batch
        self._logits: dict[int, Float[Array, "b t vocab"]] = {}
        self._sums: dict[tuple[int, int, bool], Sums] = {}

    def __len__(self) -> int:
        return len(self.outputs)

    @overload
    def __getitem__(self, i: int) -> Float[Array, "b t vocab"]: ...
    @overload
    def __getitem__(self, i: slice) -> list[Float[Array, "b t vocab"]]: ...
    def __getitem__(
        self, i: int | slice
    ) -> Float[Array, "b t vocab"] | list[Float[Array, "b t vocab"]]:
        if isinstance(i, slice):
            return [self[j] for j in range(len(self))[i]]
        if i not in self._logits:
            out = self.outputs[i]
            self._logits[i] = out.logits() if isinstance(out, HeadOutput) else out
        return self._logits[i]

    def sums(self, i: int, offset: int, z: bool = False) -> Sums:
        """Return head i's mean cross-entropy of predicting token t + offset at each
        position t with a target (`target_mask`), and its mean log Z squared there if
        `z`, by the run's `methods.loss` (``HeadOutput.sums``). Ask for both at once:
        each call takes the head's logits again."""
        if (i, offset, z) not in self._sums:
            out, targets = self.outputs[i], targets_ahead(self.batch, offset)
            weights = mean_weights(self.batch, offset)
            self._sums[i, offset, z] = (
                out.sums(targets, weights, z)
                if isinstance(out, HeadOutput)
                else Scores.of(out, targets).sums(weights, z)
            )
        return self._sums[i, offset, z]


class Term(NamedTuple):
    """A term of the objective: its weight, and the function that computes it from
    every head's logits (`Heads`) and the batch, a mean over the batch's targets.
    The trainer scales it by each microbatch's share of the targets."""

    weight: float  #: What the objective multiplies it by; 0 still logs it.
    #: The term of every head's logits and the batch, a scalar.
    fn: Callable[[Heads, Batch], Float[Array, ""]]


class Part(NamedTuple):
    """A part of the objective: its values' weighted sum, and each value by name."""

    loss: Float[Array, ""]  #: The weighted sum of its values.
    metrics: Metrics  #: Each value by name, unweighted.


@dataclass
class Objective:
    """The sum of ``weight * term`` over `terms`, and over the losses layers report.

    Attributes:
        terms: Each term by name, which is also its metric's name.
    """

    terms: dict[str, Term]

    def __call__(
        self, outputs: Sequence[Output], batch: Batch, aux_losses: Sequence[Loss] = ()
    ) -> tuple[Part, Part]:
        """Return the objective's two parts: its terms, means over target tokens, and
        the losses layers report, means over rows, each name's values summed.

        Args:
            outputs: Each head's output, or its logits, in execution order.
            batch: The rows the heads ran on.
            aux_losses: What the layers reported from a training forward.

        Returns:
            The terms' part and the reported losses' part, each its weighted sum and
            its values by name. The trainer weighs the first by a microbatch's share of
            the targets and the second by its share of the rows.

        Raises:
            ValueError: If a layer reports a loss under a term's name.
        """
        heads = Heads(outputs, batch)
        metrics = {name: term.fn(heads, batch) for name, term in self.terms.items()}
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
    that predict further ahead.

    `z_loss` is a term only where ``config.z`` is not 0, and `mtp` only where the
    model has more than one head.

    Args:
        config: The run's ``loss`` setting.
        arch: The model, whose ``nextn`` layers say how far ahead each head predicts.

    Returns:
        The objective, whose terms a script may add to or replace.

    Example::

        obj = objective(run.train.loss, run.arch)
        per_token, per_row = obj(model.heads(tokens, positions), batch)
        per_token.metrics["loss"]  # the first head's mean cross-entropy
    """
    offsets, z = prediction_offsets(arch), bool(config.z)

    def first(heads: Heads) -> Sums:
        return heads.sums(0, offsets[0], z)

    terms = {"loss": Term(1.0, lambda heads, _: first(heads).ce)}
    if z:
        terms["z_loss"] = Term(config.z, lambda heads, _: first(heads).z)
    if len(offsets) > 1:

        def mtp(heads: Heads, _: Batch) -> Float[Array, ""]:
            further = enumerate(offsets[1:], 1)
            losses = [heads.sums(i, offset).ce for i, offset in further]
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


def targets_ahead(batch: Batch, offset: int = 1) -> Int[Array, "b t"]:
    """Return token t + offset at each position t, and 0 at the last `offset`."""
    return jnp.pad(jnp.asarray(batch.input_ids)[:, offset:], ((0, 0), (0, offset)))


def _padded_mask(batch: Batch, offset: int) -> Bool[Array, "b t"]:
    """Return `target_mask`, False at the last `offset` positions."""
    return jnp.pad(target_mask(batch, offset), ((0, 0), (0, offset)))


def mean_weights(batch: Batch, offset: int = 1) -> Float[Array, "b t"]:
    """Return each position's weight in a mean over the positions with a target
    `offset` ahead: 1 / their count there, 0 elsewhere."""
    mask = _padded_mask(batch, offset)
    return mask / jnp.maximum(jnp.sum(mask), 1)


def _masked_mean(
    values: Float[Array, "b t"], mask: Bool[Array, "b t"]
) -> Float[Array, ""]:
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1)


def cross_entropy(
    logits: Float[Array, "b t vocab"], batch: Batch, offset: int = 1
) -> Float[Array, ""]:
    """Return the mean cross-entropy of predicting token t + offset from each position
    t that has a target (`target_mask`), from logits given whole. The last `offset`
    positions are masked out rather than sliced off the logits, whose gradient would
    then pad a copy of them."""
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), targets_ahead(batch, offset)
    )
    return _masked_mean(loss, _padded_mask(batch, offset))


def z_loss(
    logits: Float[Array, "b t vocab"], mask: Bool[Array, "b _"] | None = None
) -> Float[Array, ""]:
    """Return the mean of log(Z)^2, which keeps the softmax's normalizer near 1 (PaLM),
    over the first positions of each row that `mask` covers, or over all of them."""
    z = jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1) ** 2
    return jnp.mean(z) if mask is None else _masked_mean(z[:, : mask.shape[1]], mask)
