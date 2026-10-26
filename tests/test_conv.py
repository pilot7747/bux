"""The depthwise causal convolution's own backward pass on a GPU
(bux.architecture.layers.conv), by XLA and by its kernel (pallas_conv), interpreted on
the CPU: their gradients against autodiff's, dilated, in packed documents, with and
without a bias, and on a mesh; and where each runs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import count, equations, outputs_and_grads
from flax import nnx
from jax.sharding import PartitionSpec as P

from bux import parallel
from bux.architecture import Segments, Taps
from bux.architecture.layers import conv, pallas_conv
from bux.train import config as tcfg

B, T, C = 8, 48, 32


def _inputs(size: int, bias: bool):
    """x, the weights and the bias, if any, of a conv of `size` taps."""
    keys = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(keys[0], (B, T, C))
    weight = jax.random.normal(keys[1], (size, C))
    return x, weight, jax.random.normal(keys[2], (C,)) if bias else None


def _positions(every: int):
    """Each token's position in documents of `every` tokens."""
    return jnp.broadcast_to(jnp.arange(T) % every, (B, T)).astype(jnp.int32)


def _within(every: int | None):
    """How many tokens of its document come before each token, in documents of
    `every` tokens; None for one document a row."""
    return None if every is None else Segments.of(_positions(every)).within


def _grads(by, size, dilation, every, bias):
    """The conv's outputs and gradients, by autodiff (`by` None), the kernel or XLA."""
    x, weight, b = _inputs(size, bias)
    within = _within(every)

    def f(x, weight, *b):
        xs = (x, weight, b[0] if b else None, dilation, within)
        if by is None:
            return conv._shifted_sum(*xs)
        return conv._with_own_backward(*xs, by == "kernel")

    return outputs_and_grads(f, x, weight, *([] if b is None else [b]))


def _assert_close(got, want):
    for a, b in zip(got, want, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


# Taps 1, 2 and 9 apart; documents of the whole row, of 7 tokens, and of one.
@pytest.mark.parametrize("by", ["kernel", "xla"])
@pytest.mark.parametrize(("size", "dilation"), [(4, 1), (3, 2), (2, 9)])
@pytest.mark.parametrize("every", [None, 7, 1])
@pytest.mark.parametrize("bias", [True, False])
def test_the_backward_gives_the_gradients_autodiff_does(
    by, size, dilation, every, bias
):
    want = _grads(None, size, dilation, every, bias)
    _assert_close(_grads(by, size, dilation, every, bias), want)


def test_on_a_mesh_each_device_runs_the_kernel_on_its_rows():
    want = _grads("kernel", 4, 1, 7, True)
    with parallel.use(tcfg.MeshConfig(data=2, fsdp=2)):
        assert pallas_conv._rows((B, T, C)) == P(("data", "fsdp"), None, None)
        got = jax.jit(lambda: _grads("kernel", 4, 1, 7, True))()
    _assert_close(got, want)


@pytest.mark.parametrize(
    ("shape", "mesh", "runs"),
    [
        ((B, T, C), tcfg.MeshConfig(data=1), True),
        ((B, T, C), tcfg.MeshConfig(fsdp=8), True),
        ((B, T, C), tcfg.MeshConfig(data=2, tensor=2), False),
        ((B, T, C), tcfg.MeshConfig(data=2, context=2), False),
        ((B, 47, C), tcfg.MeshConfig(data=1), False),
        ((B, T, 24), tcfg.MeshConfig(data=1), False),
        ((B, 40, C), tcfg.MeshConfig(data=1), True),
        ((B, 40, C), tcfg.MeshConfig(fsdp=8), False),
    ],
    ids=[
        "one-device",
        "fsdp",
        "tensor",
        "context",
        "odd-positions",
        "odd-channels",
        "rows-of-40",
        "a-device-a-row-of-40",
    ],
)
def test_the_kernel_runs_on_a_gpu_where_the_mesh_splits_only_rows(
    monkeypatch, shape, mesh, runs
):
    monkeypatch.setattr(pallas_conv.kernels, "gpu", lambda: True)
    with parallel.use(mesh):
        assert pallas_conv.runs(shape) is runs


def _jaxprs(layer, xs, taps):
    """The jaxprs of a conv `layer` of the inputs `xs`, side by side if more than
    one, and of the gradients of its outputs."""

    def outputs(*xs):
        return layer.side_by_side(xs, taps) if len(xs) > 1 else [layer(xs[0], taps)]

    def loss(*xs):
        return sum(y.sum() for y in outputs(*xs))

    grads = jax.grad(loss, tuple(range(len(xs))))
    return jax.make_jaxpr(outputs)(*xs).jaxpr, jax.make_jaxpr(grads)(*xs).jaxpr


@pytest.mark.parametrize(
    ("gpu", "parts", "docs", "kernels", "own"),
    [
        (False, (32,), True, 0, False),
        (False, (32, 32, 64), True, 0, False),
        (True, (32,), True, 0, True),
        (True, (32,), False, 0, False),
        (True, (32, 32, 64), True, 3, True),
        (True, (32, 32, 64), False, 3, True),
    ],
    ids=[
        "cpu",
        "cpu-side-by-side",
        "gpu",
        "gpu-no-documents",
        "gpu-side-by-side",
        "gpu-side-by-side-no-documents",
    ],
)
def test_on_a_gpu_side_by_side_parts_take_the_kernel_and_the_rest_xla(
    monkeypatch, gpu, parts, docs, kernels, own
):
    """q, k and v side by side, laid out channels last, by the kernel; a conv of
    its own over documents by XLA's form of the backward; else by autodiff."""
    monkeypatch.setattr(conv.kernels, "gpu", lambda: gpu)
    layer = conv.CausalConv(sum(parts), 4, True, nnx.Rngs(0))
    xs = [jax.random.normal(jax.random.key(i), (B, T, c)) for i, c in enumerate(parts)]
    taps = Taps(segments=Segments.of(_positions(7))) if docs else None
    forward, backward = _jaxprs(layer, xs, taps)
    names = {e.primitive.name for e, _ in equations(forward)}
    assert any(n.startswith("custom_vjp") for n in names) is own
    assert count(backward, "pallas_call") == kernels
