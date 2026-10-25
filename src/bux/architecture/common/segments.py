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

import jax.numpy as jnp
from jaxtyping import Array, Int

from bux.parallel.context import running_max, running_sum


class Segments(NamedTuple):
    """Where the documents packed into the rows are, in the forms layers need: one
    array of b rows of t per form, each made once per forward pass.

    Example:
        Two documents in a row of 6, of positions ``[0, 1, 2, 0, 1, 2]``, have ids
        ``[1, 1, 1, 2, 2, 2]`` and within ``[0, 1, 2, 0, 1, 2]``; positions that skip
        ahead, ``[5, 6, 9]``, stay one document.
    """

    #: The positions as given: what rotary and learned positions use.
    positions: Int[Array, "b t"]
    #: Which document of its row each token is in, from 1.
    ids: Int[Array, "b t"]
    #: How many tokens of its document come before each token.
    within: Int[Array, "b t"]

    @classmethod
    def of(cls, positions: Int[Array, "b t"]) -> "Segments":
        """Return the documents of rows at `positions`."""
        b, t = positions.shape
        index = jnp.arange(t)
        start = jnp.concatenate(
            [jnp.ones((b, 1), bool), positions[:, 1:] <= positions[:, :-1]], axis=1
        )
        first = running_max(jnp.where(start, index, 0))
        ids = running_sum(start.astype(jnp.int32))
        return cls(positions, ids, (index - first).astype(jnp.int32))
