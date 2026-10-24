"""The documents packed into rows, from the positions given with them.

A document starts at a row's first token and wherever positions stop increasing, as
at the 0 after a document. So a row cut out of the middle of one starts mid-way
through it, and positions that skip ahead stay in their document.

``Segments`` says where the documents are in the forms the layers need, made once per
forward pass and none of them t by t: `ids`, the segment ids flash attention kernels
take (splash attention's, Pallas's) to skip the blocks no query sees, and `within`,
where causal convolutions, n-grams and targets stop looking back. A kernel that
takes the first key each query sees instead (tokamax's Mask, as k_start) takes the
token's index less `within`, and varlen kernels' cu_seqlens are where `ids` change.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int


class Segments(NamedTuple):
    positions: Int[Array, "b t"]  # as given: what rotary and learned positions use
    ids: Int[Array, "b t"]  # which document of its row each token is in, from 1
    within: Int[Array, "b t"]  # how many tokens of its document come before it

    @classmethod
    def of(cls, positions: Int[Array, "b t"]) -> "Segments":
        """Return the documents of rows at `positions`."""
        b, t = positions.shape
        index = jnp.arange(t)
        start = jnp.concatenate(
            [jnp.ones((b, 1), bool), positions[:, 1:] <= positions[:, :-1]], axis=1
        )
        # Prefix scans of log2(t) steps: XLA's cummax and cumsum of a row of 128k
        # positions split over `context` take 2 and 25 ms on a B300.
        first = jax.lax.associative_scan(
            jnp.maximum, jnp.where(start, index, 0), axis=1
        )
        ids = jax.lax.associative_scan(jnp.add, start.astype(jnp.int32), axis=1)
        return cls(positions, ids, (index - first).astype(jnp.int32))
