"""Whether bux's kernels run: on a GPU, unless they are ``off``.

Every kernel (``affine_scan``, ``grouped``, ``fused_delta``, ``blockscaled``'s and
``nvfp4``'s quantizers, ``scoring``, and the flash attention of tokamax and cuDNN,
dense and ``sparse``) asks `gpu` and falls back to XLA where it is false. So inside
``off``, or with BUX_KERNELS=off, everything traces by XLA: to count the FLOPs a
kernel hides, or to debug one. A Pallas kernel called off a GPU all the same, as the
tests call them, runs interpreted (`interpreted`).
"""

import contextlib
import contextvars
import os
from collections.abc import Generator

import jax

_OFF = contextvars.ContextVar("off", default=os.environ.get("BUX_KERNELS") == "off")


def gpu() -> bool:
    """Return whether kernels run here: on a GPU, outside `off`."""
    return jax.default_backend() == "gpu" and not _OFF.get()


def interpreted() -> bool:
    """Return whether a kernel called here runs interpreted, by XLA: off a GPU, as
    the tests call them."""
    return jax.default_backend() != "gpu"


@contextlib.contextmanager
def off() -> Generator[None]:
    """Run no kernel while the block traces."""
    token = _OFF.set(True)
    try:
        yield
    finally:
        _OFF.reset(token)
