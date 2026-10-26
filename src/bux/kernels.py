"""Whether bux's kernels run: on a GPU, unless they are ``off``.

Every kernel (``affine_scan``, ``grouped``, ``fused_delta``, ``pallas_delta``,
``blockscaled``'s and ``nvfp4``'s quantizers, ``scoring``, and the flash attention of
tokamax, cuDNN and FlashAttention-4, dense and ``sparse``) asks `gpu` and falls back
to XLA where it is false. So inside `off`, or with ``BUX_KERNELS=off``, everything
traces by XLA: to count the FLOPs a kernel hides, or to debug one. A Pallas kernel
called off a GPU all the same, as the tests call them, runs interpreted
(`interpreted`). On a mesh, which XLA cannot split a kernel over, `per_device` runs
one on each device's part.

None of this changes what a run computes, so none of it is in a run's id.
"""

import contextlib
import contextvars
import os
from collections.abc import Callable, Generator
from typing import Any

import jax

_OFF = contextvars.ContextVar("off", default=os.environ.get("BUX_KERNELS") == "off")


def gpu() -> bool:
    """Return whether kernels run here: on a GPU, outside `off`."""
    return jax.default_backend() == "gpu" and not _OFF.get()


def interpreted() -> bool:
    """Return whether a kernel called here runs interpreted, by XLA: off a GPU, as
    the tests call them."""
    return jax.default_backend() != "gpu"


def per_device(kernel: Callable[..., Any], ins: Any, outs: Any) -> Callable[..., Any]:
    """Return `kernel` in a shard_map that gives it each device's part of its inputs,
    split as the specs `ins` say, and joins its outputs as `outs` say. Nothing in it
    communicates."""
    # Untyped by what varies, which a kernel's outputs do not say.
    return jax.shard_map(kernel, in_specs=ins, out_specs=outs, check_vma=False)


@contextlib.contextmanager
def off() -> Generator[None]:
    """Run no kernel while the block traces: every one falls back to XLA.

    It acts on tracing, so wrap what traces the step, such as its first call or a
    ``jax.jit(...).lower(...)``; a program compiled before keeps its kernels.
    ``BUX_KERNELS=off`` in the environment turns them off for the whole process.

    Example::

        from bux import kernels

        with kernels.off():
            loss = step(state, batch)  # attention, scans and experts by XLA
    """
    token = _OFF.set(True)
    try:
        yield
    finally:
        _OFF.reset(token)
