"""bux.jax_internals on the versions it was verified on, and each internal by what bux
relies on: on a CPU, and on a GPU those that need one (skipped elsewhere), with

    JAX_PLATFORMS=cuda uv run --extra cuda pytest tests/test_jax_internals.py
"""

import importlib.metadata
import inspect
import os
import socket
import subprocess
import sys
import textwrap
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import equations
from jax.ad_checkpoint import checkpoint_name, name_p
from jax.extend.core import primitives
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Shaped

from bux import jax_internals, kernels
from bux.architecture.repeats import remat
from bux.jax_internals import core
from bux.parallel import flash

GPU = pytest.mark.skipif(not kernels.gpu(), reason="needs a GPU")
F8, E8 = jnp.float8_e4m3fn, jnp.float8_e8m0fnu


def _installed(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _entries() -> list[str]:
    """Each function of bux.jax_internals, with the first sentence of its docstring."""
    found = inspect.getmembers(jax_internals, inspect.isfunction)
    return [
        f"  {name}: {' '.join((f.__doc__ or '').split()).split('. ')[0]}"
        for name, f in found
        if f.__module__ == jax_internals.__name__ and not name.startswith("_")
    ]


HOW = """How: run this file here and on a GPU node (JAX_PLATFORMS=cuda uv run --extra
cuda pytest tests/test_jax_internals.py), then each of scripts/gpu_smoke/*.py on
GPUs; fix in src/bux/jax_internals.py what fails, and set its VERIFIED_ON to the
versions here."""


def _recheck(found: dict[str, str]) -> str:
    """What to recheck, and how, on the packages' versions `found`."""
    pins = ", ".join(f"{p} {v}" for p, v in jax_internals.VERIFIED_ON.items())
    here = ", ".join(f"{p} {v}" for p, v in found.items())
    return "\n".join(
        [
            f"bux reaches into JAX's internals, verified on {pins}; here: {here}.",
            "Recheck each entry of src/bux/jax_internals.py:",
            *_entries(),
            "  CALLS, VJP, Primitive, core: the names and types they import.",
            HOW,
        ]
    )


def test_the_packages_are_the_versions_bux_s_internals_were_verified_on():
    found = {p: _installed(p) for p in jax_internals.VERIFIED_ON}
    pinned = jax_internals.VERIFIED_ON
    changed = {p: v for p, v in found.items() if v and v != pinned[p]}
    assert not changed, _recheck(changed)


# --- Jaxprs ---------------------------------------------------------------------------


def _calls(x: Shaped[Array, "..."]) -> Shaped[Array, "..."]:
    y = jax.jit(jnp.sin)(x)
    return jax.checkpoint(jnp.cos)(y)


def test_jit_and_checkpoint_call_jaxprs_by_the_names_split_inlines():
    x = jnp.arange(3.0)
    for f in (_calls, jax.grad(lambda x: _calls(x).sum())):
        eqns = jax.make_jaxpr(f)(x).jaxpr.eqns
        assert {e.primitive.name for e in eqns} >= jax_internals.CALLS
        for e in filter(jax_internals.calls_a_jaxpr, eqns):
            assert isinstance(jax_internals.called_jaxpr(e), core.ClosedJaxpr)


def _eval(
    closed: core.ClosedJaxpr, *args: Shaped[Array, "..."]
) -> list[Shaped[Array, "..."]]:
    """Run `closed` on `args` one equation at a time, by `jax_internals.bind`."""
    env = dict(zip(closed.jaxpr.constvars, closed.consts, strict=True))
    env |= dict(zip(closed.jaxpr.invars, args, strict=True))

    def read(v: Any) -> Shaped[Array, "..."]:
        return v.val if isinstance(v, core.Literal) else env[v]

    for e in closed.jaxpr.eqns:
        outs = jax_internals.bind(e, [read(v) for v in e.invars])
        env |= dict(zip(e.outvars, outs, strict=True))
    return [read(v) for v in closed.jaxpr.outvars]


def test_bind_runs_each_equation_as_the_function_does():
    def f(x: Shaped[Array, "..."]) -> tuple[Shaped[Array, "..."], Shaped[Array, "..."]]:
        values, order = jax.lax.sort((x * 2.0 + 1.0, jnp.arange(3)), num_keys=1)
        return values, order  # sort has two results

    x = jnp.array([3.0, 1.0, 2.0])
    np.testing.assert_array_equal(_eval(jax.make_jaxpr(f)(x), x), f(x))


def test_a_scan_s_body_counts_and_options_run_it_again():
    w = jnp.arange(4.0)

    def f(
        c: Shaped[Array, "..."], xs: Shaped[Array, "..."]
    ) -> tuple[Shaped[Array, "..."], Shaped[Array, "..."]]:
        step = lambda c, x: (c * x + w, (c * w).sum())
        return jax.lax.scan(step, c, xs, reverse=True, unroll=2)

    c, xs = jnp.ones(4), jnp.linspace(0, 1, 5)
    [eqn] = [e for e in jax.make_jaxpr(f)(c, xs).jaxpr.eqns if jax_internals.is_scan(e)]
    assert jax_internals.scan_counts(eqn) == (1, 1)  # w, then c
    assert jax_internals.scan_options(eqn) == {
        "length": 5,
        "reverse": True,
        "unroll": 2,
    }
    body = jax_internals.scan_body(eqn)

    def step(
        c: Shaped[Array, "..."], x: Shaped[Array, "..."]
    ) -> tuple[Shaped[Array, "..."], Shaped[Array, "..."]]:
        new, y = _eval(body, w, c, x)
        return new, y

    again = jax.lax.scan(step, c, xs, **jax_internals.scan_options(eqn))
    _assert_close(again, f(c, xs))


def _assert_close(x: object, y: object) -> None:
    """Assert that pytrees x and y are close, leaf by leaf."""
    for a, b in zip(jax.tree.leaves(x), jax.tree.leaves(y), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6)


def test_a_cond_s_branches_are_its_functions_by_index():
    fs = [jnp.sin, jnp.cos, jnp.tanh]
    closed = jax.make_jaxpr(lambda i, x: jax.lax.switch(i, fs, x))(0, 1.0)
    [eqn] = [e for e in closed.jaxpr.eqns if jax_internals.is_cond(e)]
    branches = jax_internals.cond_branches(eqn)
    for f, b in zip(fs, branches, strict=True):
        np.testing.assert_allclose(_eval(b, jnp.float32(0.5))[0], f(0.5), rtol=1e-6)


def _names(f, *args) -> set[str]:
    """The primitives' names in f's jaxpr, inside its calls too."""
    return {e.primitive.name for e, _ in equations(jax.make_jaxpr(f)(*args).jaxpr)}


def test_the_primitives_bux_matches_by_name_are_jax_s():
    a, scales = jnp.ones((1, 64, 64), F8), jnp.ones((1, 64, 2), E8)
    x, w, sizes = jnp.ones((4, 8)), jnp.ones((2, 8, 3)), jnp.array([1, 3])
    known = {p.name for p in vars(primitives).values() if isinstance(p, core.Primitive)}
    known |= _names(jax.lax.ragged_dot, x, w, sizes)
    known |= _names(jax.nn.scaled_matmul, a, a, scales, scales)
    scaled = jax_internals.scaled_matmul_unpartitioned
    known |= _names(lambda *s: scaled(*s, jnp.bfloat16), a, a, scales, scales)
    products = {"dot_general", "ragged_dot_general", "conv_general_dilated"}
    flops = {*products, "scan", "cond"}  # what flops.py counts
    matched = remat.CHEAP | remat.GEMMS | {"sharding_constraint"} | flops
    assert not matched - known, "JAX renamed or dropped these primitives"


def test_the_params_remat_and_flops_read_are_lax_s_arguments():
    x, w = jnp.ones((2, 4, 8)), jnp.ones((2, 8, 3))
    sizes, image = jnp.array([1, 3]), jnp.ones((1, 2, 5))

    def f(x, w, sizes, image):
        dot = jax.lax.dot_general(x, w, (((2,), (1,)), ((0,), (0,))))
        ragged = jax.lax.ragged_dot(x[0], w, sizes)
        conv = jax.lax.conv_general_dilated(image, jnp.ones((3, 2, 2)), (1,), "VALID")
        return checkpoint_name(dot, "qkv"), ragged, conv

    eqns = {e.primitive.name: e for e in jax.make_jaxpr(f)(x, w, sizes, image).eqns}
    (_, contracting), (_, batch) = eqns["dot_general"].params["dimension_numbers"]
    assert (contracting, batch) == ((1,), (0,))
    numbers = eqns["ragged_dot_general"].params["ragged_dot_dimension_numbers"]
    assert numbers.dot_dimension_numbers == (((1,), (1,)), ((), ()))
    assert numbers.rhs_group_dimensions == (0,)
    assert eqns["conv_general_dilated"].params["dimension_numbers"].rhs_spec[0] == 0
    assert eqns["name"].primitive is name_p
    assert eqns["name"].params["name"] == "qkv"


def test_a_sharding_constraint_says_its_devices():
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    shard = jax.sharding.NamedSharding(mesh, P("data"))
    f = lambda x: jax.lax.with_sharding_constraint(x, shard)
    [eqn] = jax.make_jaxpr(f)(jnp.ones(jax.device_count())).eqns
    assert eqn.primitive.name == "sharding_constraint"
    assert eqn.params["sharding"].num_devices == jax.device_count()


# --- A vjp from its residuals ---------------------------------------------------------


def test_a_vjp_is_rebuilt_from_its_residuals_and_its_unsaved_primal():
    def f(params: dict, x: Shaped[Array, "..."]) -> Shaped[Array, "..."]:
        return jnp.tanh(x @ params["w"]) * params["b"]

    params = {"w": jnp.linspace(-1, 1, 12).reshape(4, 3), "b": jnp.arange(3.0)}
    x, dy = jnp.linspace(0, 1, 8).reshape(2, 4), jnp.ones((2, 3))
    _, vjp = jax.vjp(f, params, x, saveable_args=(False, True))
    residuals = jax.tree.leaves(vjp)
    saved = jax.tree.leaves(jax.vjp(f, params, x)[1])
    assert len(residuals) < len(saved), "it saved the parameters"

    @jax.jit
    def rebuilt(residuals: list, params: dict) -> tuple:
        return jax_internals.rebuild_vjp(vjp, residuals, params)(dy)

    _assert_close(rebuilt(residuals, params), jax.vjp(f, params, x)[1](dy))


# --- Kernels --------------------------------------------------------------------------


def test_the_unpartitioned_scaled_matmul_is_jax_nn_s_in_a_shard_map():
    a = jnp.linspace(-2, 2, 64 * 64).reshape(1, 64, 64).astype(F8)
    scales = jnp.full((1, 64, 2), 0.5).astype(E8)
    args = (a, a, scales, scales)
    expected = jax.nn.scaled_matmul(*args, preferred_element_type=jnp.bfloat16)

    one = jax.sharding.Mesh(jax.devices()[:1], ("x",))

    @jax.jit
    @jax.shard_map(mesh=one, in_specs=P(), out_specs=P())
    def manual(*args: Shaped[Array, "..."]) -> Shaped[Array, "..."]:
        return jax_internals.scaled_matmul_unpartitioned(*args, jnp.bfloat16)

    np.testing.assert_array_equal(manual(*args), expected)


def test_cudnn_s_attention_takes_the_arguments_bux_passes():
    takes = inspect.signature(jax_internals.cudnn_fused.dot_product_attention)
    asked = ("q_seqlen", "kv_seqlen", "q_offsets", "kv_offsets", "scale")
    assert set(takes.parameters) >= {*asked, "mask_type", "sliding_window_length"}
    assert {"CAUSAL", "PADDING_CAUSAL"} <= set(
        jax_internals.cudnn_fused.MaskType.__members__
    )
    assert jax_internals.cudnn_attention_here() == (jax.default_backend() == "gpu")


CPU = pytest.mark.skipif(jax.default_backend() != "cpu", reason="numpy reads a CPU's")
RESULTS = [jax.ShapeDtypeStruct((3,), jnp.float32)]


@CPU
def test_a_buffer_callback_writes_its_results_into_xla_s_buffers():
    def run(context: Any, outs: tuple, *ins: Any) -> None:
        np.multiply(np.from_dlpack(ins[0]), 2, out=np.from_dlpack(outs[0]))

    (y,) = jax.jit(jax_internals.buffer_callback(run, RESULTS))(jnp.arange(3.0))
    np.testing.assert_array_equal(y, [0.0, 2.0, 4.0])


@GPU
def test_a_buffer_callback_hands_its_run_xla_s_stream_and_device():
    seen = []

    def run(context: Any, outs: tuple, *ins: Any) -> None:
        seen.append((context.stream, ins[0].__dlpack_device__()))

    jax.jit(jax_internals.buffer_callback(run, RESULTS))(jnp.arange(3.0))
    [(stream, (kind, device))] = seen
    assert isinstance(stream, int) and kind == 2 and device == 0  # kDLCUDA, GPU 0


@GPU
def test_cudnn_s_attention_attends_causally_and_by_documents():
    q, k, v = _qkv(jax.random.key(0), (2, 256, 4, 64))
    ids = jnp.repeat(jnp.array([[1, 2, 3, 4], [1, 1, 2, 2]]), 64, axis=1)
    lengths, starts = flash.packed_documents(ids, 4)
    same = ids[:, None, :, None] == ids[:, None, None, :]
    for mask, documents in ((None, (None, None)), (same, (lengths, starts))):
        out = jax_internals.cudnn_attention(q, k, v, *documents, 0.125, None)
        assert _off(out, _attention(q, k, v, mask)) < 2e-2


def _qkv(
    key: Shaped[Array, "..."], shape: tuple[int, ...]
) -> list[Shaped[Array, "..."]]:
    return [jax.random.normal(k, shape, jnp.bfloat16) for k in jax.random.split(key, 3)]


def _attention(q, k, v, mask=None) -> Shaped[Array, "..."]:
    """XLA's causal attention, within `mask` if any, at a scale of 1/8."""
    return jax.nn.dot_product_attention(
        q, k, v, mask=mask, scale=0.125, is_causal=True, implementation="xla"
    )


def _off(x: Shaped[Array, "..."], y: Shaped[Array, "..."]) -> float:
    """How far x is from y at worst, relative to y's largest magnitude."""
    a, b = np.asarray(x, np.float32), np.asarray(y, np.float32)
    return float(np.abs(a - b).max() / np.abs(b).max())


def test_tokamax_s_flash_attention_takes_bux_s_mask_and_has_a_vjp():
    pytest.importorskip("tokamax", reason="tokamax comes with the cuda extra (Linux)")
    for name in ("mosaic_gpu", "triton"):
        op = jax_internals.tokamax_attention(name)
        assert callable(op.bind) and callable(op.vjp), name
    starts = jnp.zeros((1, 1, 8), jnp.int32)
    jax_internals.tokamax_mask(k_start=starts, k_end=None, is_causal=True)
    kept = jnp.ones((1, 1, 8, 8), bool)
    jax_internals.tokamax_mask(kept, k_start=starts, k_end=starts + 8, is_causal=False)


@GPU
def test_tokamax_s_kernels_attend_and_take_the_gradients_back():
    pytest.importorskip("tokamax", reason="tokamax comes with the cuda extra (Linux)")
    op = jax_internals.tokamax_attention("mosaic_gpu")
    q, k, v = _qkv(jax.random.key(1), (1, 256, 2, 64))
    dout = jax.random.normal(jax.random.key(2), q.shape, q.dtype)
    args = {
        "mask": jax_internals.tokamax_mask(is_causal=True),
        "logits_scale": 0.125,
        "logits_soft_cap": None,
        "logits_dtype": jnp.float32,
    }
    out, (m, total) = op(q, k, v, return_residuals=True, **args)
    grads = jax_internals.tokamax_vjp(op, m + jnp.log(total), out, dout, q, k, v, args)
    expected, vjp = jax.vjp(_attention, q, k, v)
    assert _off(out, expected) < 2e-2
    for g, e in zip(grads, vjp(dout), strict=True):
        assert _off(g, e) < 2e-2


@GPU
def test_jax_triton_s_calls_survive_dead_code_elimination():
    jax_triton = pytest.importorskip("jax_triton", reason="the cuda extra (Linux)")
    from jax._src.interpreters import partial_eval

    from bux.architecture.experts import grouped_triton  # installs the rule

    def values(
        x: Shaped[Array, "..."],
    ) -> Shaped[Array, "..."]:  # one of the quantizer's 4 results
        return grouped_triton.quantized_twice(x, jnp.int32(64))[0].values

    jaxpr = jax.make_jaxpr(values)(jnp.ones((64, 128), jnp.bfloat16)).jaxpr
    kept, _ = partial_eval.dce_jaxpr(jaxpr, [True])
    call = jax_triton.triton_lib.triton_kernel_call_p
    assert any(e.primitive is call for e in kept.eqns)


# --- jax.distributed ------------------------------------------------------------------


def test_processes_and_their_store_are_jax_distributed_s():
    assert jax_internals.processes() == (0, 1)
    assert jax_internals.key_value_store() is None
    joined = textwrap.dedent(
        f"""
        import jax
        from bux import jax_internals

        jax.distributed.initialize("localhost:{_free_port()}", 1, 0)
        store = jax_internals.key_value_store()
        store.key_value_set("bux/test", "seen")
        assert jax_internals.processes() == (0, 1)
        assert store.blocking_key_value_get("bux/test", 10_000) == "seen"
        """
    )
    env = os.environ | {"JAX_PLATFORMS": "cpu", "XLA_FLAGS": ""}
    command = [sys.executable, "-c", joined]
    done = subprocess.run(command, env=env, capture_output=True, check=False)
    assert done.returncode == 0, done.stderr.decode()[-2000:]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]
