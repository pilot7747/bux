"""FlashAttention-4's kernels (bux.parallel.fa4) on a CPU, which runs none of them:
jnp's attention stands in for them on the layouts they take (whole rows, or tokens
end to end in the documents of cu_seqlens), so that the documents, the windows, the
logsumexp's layout, the room for documents and the routing are checked against xla.
scripts/gpu_smoke/attention.py runs the kernels."""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import equations
from jaxtyping import Array, Int
from test_context import CASES, IDS, MESHES, Case, attend, attention

from bux import parallel
from bux.parallel import context, fa4, flash, methods
from bux.parallel.flash import Keys


def token_keys(cu: Int[Array, " n1"], n: int, window: int | None) -> Keys:
    """Each token's first key, of n tokens end to end in the documents of cu."""
    index = jnp.arange(n)
    document = (index[:, None] >= cu[None, 1:]).sum(1)  # the ends before each token
    start = cu[document]
    if window is not None:
        start = jnp.maximum(start, index - window + 1)
    return Keys(True, start[None])


def forward(window, q, k, v, cu, scale, cap, t):
    """FA4's forward call by jnp: rows (b, t, h, d), or tokens (n, h, d) and cu."""
    if cu is None:
        return flash.JNP.forward(q, k, v, Keys(True), scale, cap)
    keys = token_keys(cu, q.shape[0], window)
    out, lse = flash.JNP.forward(q[None], k[None], v[None], keys, scale, cap)
    return out[0], lse[0]


def backward(window, q, k, v, out, dout, lse, cu, scale, cap, t):
    """FA4's backward call by jnp, laid out as `forward`."""
    if cu is None:
        return flash.JNP.backward(q, k, v, out, dout, lse, Keys(True), scale, cap)
    keys = token_keys(cu, q.shape[0], window)
    rows = (x[None] for x in (q, k, v, out, dout, lse))
    return tuple(g[0] for g in flash.JNP.backward(*rows, keys, scale, cap))


@pytest.fixture
def kernels(monkeypatch):
    """FA4's calls by jnp, with room for every document of the tests' rows; and
    tokamax's kernel jnp's, which a batch of more documents falls back to."""
    monkeypatch.setattr(fa4, "_forward_call", forward)
    monkeypatch.setattr(fa4, "_backward_call", backward)
    monkeypatch.setattr(fa4, "takes", lambda q, k, v: True)
    monkeypatch.setattr(fa4, "DOCS", 4096)
    monkeypatch.setattr(flash, "TOKAMAX", flash.JNP)
    monkeypatch.setattr(flash, "tokamax_runs", lambda: True)
    monkeypatch.setattr(flash, "tokamax_takes", lambda x: True)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
@pytest.mark.parametrize("mesh", [None, 1])
@pytest.mark.parametrize("method", ["fa4", "auto"])
def test_fa4_attends_and_differentiates_as_xla_does(kernels, method, mesh, case):
    out, grads = attend("xla", case)
    with parallel.use(MESHES[mesh]) if mesh else contextlib.nullcontext():
        fa4_out, fa4_grads = attend(method, case)
    np.testing.assert_allclose(fa4_out, out, atol=1e-5)
    for got, want in zip(fa4_grads, grads):
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)


def test_cu_seqlens_start_each_document_of_the_rows_end_to_end(monkeypatch):
    """Three documents in the first row of 16, one in the second; the rest empty."""
    monkeypatch.setattr(fa4, "DOCS", 1024)  # room for 4 a row of 16
    start = context.document_starts(IDS)
    assert fa4.cu_seqlens(start).tolist() == [0, 5, 12, 16, 32, 32, 32, 32, 32]
    assert bool(fa4.fits(start))
    monkeypatch.setattr(fa4, "DOCS", 256)  # room for 1 a row
    assert not bool(fa4.fits(start))


def test_a_window_s_first_keys_start_no_document():
    """But the documents' own, so a window's rows hold no more documents."""
    seen = Case(4, 2, IDS, window=3).visibility(jnp.zeros(4))
    start = context.first_keys(seen, *IDS.shape)
    assert start is not None
    assert (fa4.starts(start) == fa4.starts(context.document_starts(IDS))).all()


def test_a_batch_of_more_documents_than_fa4_has_room_for_goes_to_tokamax(
    kernels, monkeypatch
):
    """By a lax.cond, tokamax's kernel jnp's here."""
    monkeypatch.setattr(fa4, "DOCS", 256)
    case = CASES["gqa_documents"]
    out, grads = attend("xla", case)
    fa4_out, fa4_grads = attend("fa4", case)
    np.testing.assert_allclose(fa4_out, out, atol=1e-5)
    for got, want in zip(fa4_grads, grads):
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)
    q, k, v, sinks, _ = attention(case)
    f = methods.attention.find("fa4")
    jaxpr = jax.make_jaxpr(lambda *x: f(*x, case.visibility(sinks)))(q, k, v)
    assert any(e.primitive.name == "cond" for e, _ in equations(jaxpr.jaxpr))


def test_rows_of_one_document_take_fa4_where_cudnn_does_not(monkeypatch):
    """MLA's head dims, a soft cap: whole rows, no cu_seqlens."""
    monkeypatch.setattr(fa4, "takes", lambda q, k, v: True)
    q, k, v, *_ = attention(Case(4, 4))
    args = Case(4, 4, softcap=2.0).visibility(jnp.zeros(4))
    assert context.causal_kernel(q, k, v, args) is fa4.kernel(None)
    assert context.causal_kernel(q, k, v, args._replace(window=4)) is None


def test_fa4_says_what_it_needs_on_a_cpu():
    q, k, v, sinks, _ = attention(Case(4, 2))
    seen = Case(4, 2).visibility(sinks)
    with pytest.raises(NotImplementedError, match="fa4 needs a Blackwell GPU"):
        methods.attention.find("fa4")(q, k, v, seen)
    assert not fa4.takes(q, k, v)


@pytest.mark.parametrize("dims", [(128, 128), (64, 64), (192, 128), (256, 256)])
def test_fa4_takes_its_blackwell_kernels_head_dims(monkeypatch, dims):
    monkeypatch.setattr(fa4, "runs", lambda: True)
    dk, dv = dims
    q = k = jnp.zeros((1, 8, 2, dk), jnp.bfloat16)
    v = jnp.zeros((1, 8, 2, dv), jnp.bfloat16)
    assert fa4.takes(q, k, v) == (dims != (256, 256))
    assert not fa4.takes(q.astype(jnp.float32), k, v)


def test_the_kernels_take_causal_keys_with_no_last_key():
    q, k, v, *_ = attention(Case(2, 2))
    end = jnp.ones(q.shape[:2], jnp.int32)
    with pytest.raises(ValueError, match="causal keys with no last key"):
        fa4.kernel(None).forward(q, k, v, Keys(False, None, end), 1.0, None)
