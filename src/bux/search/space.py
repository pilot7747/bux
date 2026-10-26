"""A search's space as its methods see it: a coordinate for each dimension.

A dimension that is a number, `uniform`, `log_uniform` or `int_uniform`, has a
coordinate in [0, 1], the fraction of the way along its scale; a choice has the index
of its option. A method proposes coordinates, which `Space.decode` turns into the
values a trial takes, to `bux.train.config.DIGITS` significant digits and integers
where they must be, and `Space.encode` back, so that what a method learns from is
the point that was trained.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from bux.lang.compiler import Dimension
from bux.train import config as tcfg


@dataclass(frozen=True)
class Space:
    """The dimensions of a search, in order, by name.

    Attributes:
        names: Each dimension's name, as `bux.lang.compiler.Space` names them.
        dims: Each one's distribution.
    """

    names: tuple[str, ...]
    dims: tuple[tcfg.Distribution, ...]

    @classmethod
    def of(cls, dimensions: Mapping[str, Dimension]) -> "Space":
        """Return the space of a compiled run's `dimensions`."""
        return cls(tuple(dimensions), tuple(d.space for d in dimensions.values()))

    @property
    def sizes(self) -> tuple[int, ...]:
        """Return the options of each choice, and 0 for each number."""
        return tuple(
            len(d.options) if isinstance(d, tcfg.ChoiceConfig) else 0 for d in self.dims
        )

    def from_unit(self, u: Float[np.ndarray, "*n d"]) -> Float[np.ndarray, "*n d"]:
        """Return the coordinates at `u` in [0, 1] in every dimension: a choice's
        option a fraction `u` of the way along its options, a number's `u`."""
        x = np.array(u, dtype=np.float64)
        for i, k in enumerate(self.sizes):
            if k:
                x[..., i] = np.minimum(np.floor(x[..., i] * k), k - 1)
        return x

    def decode(self, x: Float[np.ndarray, " d"]) -> dict[str, float]:
        """Return the value each dimension takes at the coordinates `x`, by name: a
        number's rounded as a trial takes it, a choice's option's index."""
        values: dict[str, float] = {}
        for name, dim, xi in zip(self.names, self.dims, x.tolist(), strict=True):
            if isinstance(dim, tcfg.ChoiceConfig):
                values[name] = int(min(max(round(xi), 0), len(dim.options) - 1))
            else:
                values[name] = dim.value(min(max(xi, 0.0), 1.0))
        return values

    def encode(self, values: Mapping[str, float]) -> Float[np.ndarray, " d"]:
        """Return the coordinates of a trial's `values`, by name: `decode`'s inverse
        on the values it returns."""
        return np.array(
            [
                float(values[name])
                if isinstance(dim, tcfg.ChoiceConfig)
                else dim.unit(values[name])
                for name, dim in zip(self.names, self.dims, strict=True)
            ]
        )
