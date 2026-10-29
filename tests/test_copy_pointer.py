"""CPLM's copy pointer: the mixture of the head's softmax and the pointer, the
pointer's attention over the earlier tokens of each document, the head that leaves
it to the loss, and runs without it as before."""

import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jaxtyping import Array, Float, Int

from bux import parallel
from bux.architecture import Model
from bux.architecture.copy import Pointer, _mixture_sums, log_probs, mixture_nll
from bux.architecture.layers.embedding import Head
from bux.architecture.logits import HeadOutput, Projection, Scores
from bux.export.hf import basic as hf_basic
from bux.export.huggingface import save
from bux.lang import compile, compile_runs, ir
from bux.train import config as tcfg
from bux.train.data import plain
from bux.train.objective import held_out, objective
from bux.train.sinks import Scalars, Sink
from bux.train.trainer import forward, train

SMOKE = Path(__file__).parent.parent / "examples" / "experiments" / "smoke.bux"
TINY = """
d_model = 32
vocab_size = 64
norm = rmsnorm
attn = gqa(num_heads=2, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=64, activation=swish)
block = [norm attn] [norm ffn]
"""


def test_without_the_gate_the_mixture_is_the_softmax_s_cross_entropy_to_the_bit():
    """a = 0, the <copy> row's logit at -inf: the cross-entropy of the softmax over
    the other rows, as Scores takes it, whatever the pointer says."""
    logits = jax.random.normal(jax.random.key(0), (3, 7, 10)) * 3
    targets = jax.random.randint(jax.random.key(1), (3, 7), 0, 9)
    closed = logits.at[..., 9].set(-jnp.inf)
    log_c = jnp.log(jax.random.uniform(jax.random.key(2), (3, 7)))
    log_s = jnp.log(jax.random.uniform(jax.random.key(3), (3, 7)))
    nll, _ = mixture_nll(closed, targets, jnp.int32(9), log_c, log_s)
    assert jnp.array_equal(nll, Scores.of(closed, targets).ce)
    nothing, _ = mixture_nll(closed, targets, jnp.int32(9), log_c - jnp.inf, log_s)
    assert jnp.array_equal(nothing, nll)


def test_the_mixture_is_its_formula_in_probability_space():
    """p = p_lm (1 - a (1 - a_sink)) + a p_copy, p_lm the softmax without <copy>."""
    logits = np.array([[1.0, 2.0, 0.5, -1.0, 1.5]])  # <copy> is row 4
    target, a_sink, p_copy = 1, 0.3, 0.6
    e = np.exp(logits[0])
    a = e[4] / e.sum()
    p_lm = e[target] / e[:4].sum()
    want = -math.log(p_lm * (1 - a * (1 - a_sink)) + a * p_copy)
    got, lse = mixture_nll(
        jnp.asarray(logits),
        jnp.array([target]),
        jnp.int32(4),
        jnp.log(jnp.array([p_copy])),
        jnp.log(jnp.array([a_sink])),
    )
    assert float(got[0]) == pytest.approx(want, rel=1e-6)
    assert float(lse[0]) == pytest.approx(math.log(e.sum()), rel=1e-6)
    # The mixture over the vocabulary sums to 1.
    total = sum(
        math.exp(
            -float(
                mixture_nll(
                    jnp.asarray(logits),
                    jnp.array([y]),
                    jnp.int32(4),
                    jnp.log(jnp.array([p_copy if y == 1 else 0.0])),
                    jnp.log(jnp.array([a_sink])),
                )[0][0]
            )
        )
        for y in range(4)
    )
    assert total == pytest.approx(1 - a * (1 - a_sink) + a * p_copy, rel=1e-6)


@pytest.mark.parametrize("z", [False, True])
def test_the_chunked_mixture_s_gradients_are_its_formula_s(z):
    """The head's gradients made with the chunks' logits are those of the mixture
    of the logits made whole, its pointer's logs' too."""
    keys = jax.random.split(jax.random.key(0), 7)
    b, t, d, vocab = 2, 6, 8, 12
    p = Projection(
        jax.random.normal(keys[0], (d, vocab)),
        jax.random.normal(keys[1], (vocab,)),
        jnp.asarray(3.0),
    )
    x = jax.random.normal(keys[2], (b, t, d))
    targets = jax.random.randint(keys[3], (b, t), 0, vocab - 1)
    weights = jax.random.uniform(keys[4], (b, t))
    log_c = jnp.where(
        jax.random.uniform(keys[5], (b, t)) < 0.5,
        -jnp.inf,
        jnp.log(jax.random.uniform(keys[5], (b, t))),
    )
    log_s = jnp.log(jax.random.uniform(keys[6], (b, t)))
    copy = jnp.int32(vocab - 1)

    def ours(kernel, bias, x, log_c, log_s):
        sums = _mixture_sums(
            p._replace(kernel=kernel, bias=bias),
            x,
            targets,
            weights,
            log_c,
            log_s,
            copy,
            z,
        )
        return sums.ce + 0.5 * sums.z

    def whole(kernel, bias, x, log_c, log_s):
        logits = p._replace(kernel=kernel, bias=bias)(x)
        nll, lse = mixture_nll(logits, targets, copy, log_c, log_s)
        return jnp.sum(weights * nll) + 0.5 * z * jnp.sum(weights * lse**2)

    args = (p.kernel, p.bias, x, log_c, log_s)
    got = jax.value_and_grad(ours, argnums=range(5))(*args)
    want = jax.value_and_grad(whole, argnums=range(5))(*args)
    assert float(got[0]) == pytest.approx(float(want[0]), rel=1e-5)
    for g, w in zip(got[1], want[1], strict=True):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-4, atol=1e-5)


def brute(
    q: Float[np.ndarray, "t c"],
    k: Float[np.ndarray, "t c"],
    sink: Float[np.ndarray, " t"],
    tokens: Int[np.ndarray, " t"],
    docs: Int[np.ndarray, " t"],
    targets: Int[np.ndarray, " t"],
    window: int,
) -> tuple[Float[np.ndarray, " t"], Float[np.ndarray, " t"]]:
    """The pointer by its definition, one query at a time."""
    t, c = q.shape
    log_c, log_s = np.full(t, -np.inf), np.zeros(t)
    for i in range(t):
        seen = [j for j in range(t) if j < i and docs[j] == docs[i] and i - j <= window]
        scores = np.array([q[i] @ k[j] / math.sqrt(c) for j in seen] + [sink[i]])
        p = np.exp(scores - scores.max())
        p /= p.sum()
        hits = [n for n, j in enumerate(seen) if tokens[j] == targets[i]]
        if hits:
            log_c[i] = math.log(p[hits].sum())
        log_s[i] = math.log(p[-1])
    return log_c, log_s


@pytest.mark.parametrize("window", [None, 2])
def test_the_pointer_attends_to_the_earlier_tokens_of_its_document(window):
    t, c = 9, 4
    q, k = (np.asarray(jax.random.normal(jax.random.key(i), (1, t, c))) for i in (0, 1))
    sink = np.asarray(jax.random.normal(jax.random.key(2), (1, t)))
    tokens = np.array([[3, 1, 3, 1, 5, 3, 5, 5, 1]])
    docs = np.array([[1, 1, 1, 1, 1, 2, 2, 2, 2]])
    targets = np.array([[1, 3, 1, 5, 3, 5, 5, 1, 2]])
    span = 2**30 if window is None else window
    pointer = Pointer(
        *(jnp.asarray(a) for a in (q, k, sink, tokens, docs)),
        jnp.int32(63),
        jnp.int32(span),
    )
    log_c, log_s = pointer.logs(jnp.asarray(targets))
    want_c, want_s = brute(q[0], k[0], sink[0], tokens[0], docs[0], targets[0], span)
    np.testing.assert_allclose(np.asarray(log_s[0]), want_s, rtol=1e-5, atol=1e-6)
    finite = np.isfinite(want_c)
    assert (np.isfinite(np.asarray(log_c[0])) == finite).all()
    np.testing.assert_allclose(np.asarray(log_c[0])[finite], want_c[finite], rtol=1e-5)
    # Its gradient stays finite where a target has no earlier source.
    grads = jax.grad(
        lambda q: jnp.sum(
            jnp.exp(pointer._replace(query=q).logs(jnp.asarray(targets))[0])
        )
    )(pointer.query)
    assert jnp.isfinite(grads).all() and (grads != 0).any()


def pointed_arch(**settings: str) -> ir.Architecture:
    """A model of two blocks, its head shared with an MTP module, the head's pointer
    of 16 with `settings`."""
    given = ", ".join(f"{k}={v}" for k, v in settings.items())
    source = (
        TINY
        + f"shared out = head(pointer=copy_pointer(dim=16{', ' if given else ''}{given}))\n"
        + "model = embed block^2 norm @h out $h nextn(norm=norm) [norm attn] norm out"
    )
    return compile(source)


def pointed(**settings: str) -> Model:
    return Model(pointed_arch(**settings), rngs=nnx.Rngs(0))


def test_the_first_head_leaves_its_pointer_and_the_mtp_head_its_softmax():
    model = pointed()
    tokens = jax.random.randint(jax.random.key(0), (2, 16), 0, 63)
    heads = forward(model, plain(np.asarray(tokens)))
    assert heads[0].pointer is not None and heads[1].pointer is None
    pointer = heads[0].pointer
    assert int(pointer.copy) == 63 and pointer.query.shape == (2, 16, 16)


def test_a_head_s_loss_and_eval_are_its_mixture_s():
    model = pointed()
    tokens = np.asarray(jax.random.randint(jax.random.key(0), (2, 16), 0, 63))
    batch = plain(tokens)
    heads = forward(model, batch)
    terms, _ = objective(tcfg.LossConfig(mtp=0.3), pointed_arch())(heads, batch)
    head, pointer = heads[0], heads[0].pointer
    assert pointer is not None
    targets = jnp.pad(jnp.asarray(tokens)[:, 1:], ((0, 0), (0, 1)))
    log_c, log_s = pointer.logs(targets)
    logits = head.projection(head.hidden)
    nll, _ = mixture_nll(logits, targets, pointer.copy, log_c, log_s)
    want = jnp.mean(nll[:, :-1])
    assert float(terms.metrics["loss"]) == pytest.approx(float(want), rel=1e-5)
    evaluated = held_out(head, batch)
    assert float(evaluated["loss"][0] / evaluated["loss"][1]) == pytest.approx(
        float(want), rel=1e-5
    )


def test_the_loss_trains_the_pointer_and_the_gate():
    model = pointed()
    tokens = np.asarray(jax.random.randint(jax.random.key(0), (2, 16), 0, 8))
    batch = plain(tokens)
    loss_of = objective(tcfg.LossConfig(), pointed_arch())

    def loss(model: Model) -> Float[Array, ""]:
        return loss_of(forward(model, batch), batch)[0].loss

    found = {
        jax.tree_util.keystr(path): g
        for path, g in jax.tree_util.tree_leaves_with_path(nnx.grad(loss)(model))
    }
    for name in ("'q'", "'k'", "'gain'", "'sink'"):
        moved = [g for path, g in found.items() if "pointer" in path and name in path]
        assert moved and all(jnp.isfinite(g).all() for g in moved), name
        assert any((g != 0).any() for g in moved), name


def test_the_pointer_s_window_and_token_are_settings():
    with pytest.raises(ValueError, match="no row of 64"):
        pointed(token="64")
    pointed(window="4")


def test_an_export_holds_the_pointer_which_transformers_refuses(tmp_path):
    """vLLM's plugin serves the pointer from the export's weights; transformers'
    modeling, whose generation would need the pointer's key cache, refuses it."""
    from safetensors import safe_open

    save(pointed_arch(), pointed(), tmp_path)
    with safe_open(tmp_path / "model.safetensors", "pt") as f:
        saved = f.keys()  # a safetensors file, no dict
        names = [k for k in saved if ".pointer." in k]
    assert sorted(n.split(".pointer.")[1] for n in names) == [
        "gain",
        "k.weight",
        "q.weight",
        "sink",
    ]
    c = next(layer for layer in pointed_arch().layers() if layer.config.type == "head")
    config = SimpleNamespace(vocab_size=64)
    with pytest.raises(NotImplementedError, match="head: pointer not exported yet"):
        hf_basic.Head(
            c.config.model_dump(mode="json"),
            32,
            cast(Any, SimpleNamespace(config=config)),
        )


def test_dense_log_probs_are_the_mixture_the_loss_scores():
    """`log_probs`, what a server samples, gives each target the mixture's
    log-likelihood that the loss takes, and sums to 1 over the vocabulary."""
    model = pointed()
    tokens = jax.random.randint(jax.random.key(0), (1, 16), 0, 8)
    [head, *_] = model.heads(tokens)
    assert head.pointer is not None
    dense = log_probs(head)
    targets = jnp.pad(tokens[:, 1:], ((0, 0), (0, 1)))
    log_c, log_s = head.pointer.logs(targets)
    nll, _ = mixture_nll(
        head.projection(head.hidden), targets, head.pointer.copy, log_c, log_s
    )
    picked = jnp.take_along_axis(dense, targets[..., None], -1)[..., 0]
    np.testing.assert_allclose(
        np.asarray(-picked[0, :-1]), np.asarray(nll[0, :-1]), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(np.asarray(jnp.exp(dense).sum(-1)), 1.0, rtol=1e-5)
    assert jnp.isneginf(dense[..., 63]).all()


def test_a_run_without_a_pointer_trains_as_before(monkeypatch):
    """smoke.bux keeps its id, and a pointer gives it another; and on this machine,
    without a pointer it trains to the bit as with the head and its loss as they
    were before the pointer (04f1159), both runs in one process, so that the
    comparison holds on any CPU. (A closed gate is no stand-in here: smoke.bux's
    tied head has no row whose logit can be -inf.)"""
    sets = [
        "schedule.warmup = 1",
        "steps = 4",
        "log = log(every=1)",
        "eval = eval(data=synthetic, every=2, batch=4, batches=2)",
        "mesh = mesh(data=1)",
    ]
    [run] = compile_runs(SMOKE.read_text(), filename=str(SMOKE), overrides=sets)
    assert run.id == "0e4953952b"
    on = "model = embed block^2 norm head(tied=true, pointer=copy_pointer())"
    [pointed_run] = compile_runs(
        SMOKE.read_text(), filename=str(SMOKE), overrides=[*sets, on]
    )
    assert pointed_run.id != run.id

    def losses() -> list[tuple[int, str, float]]:
        seen: list[tuple[int, str, float]] = []

        class Seen(Sink):
            def write(self, step: int, scalars: Scalars) -> None:
                kept = [(k, float(v)) for k, v in scalars.items() if "loss" in k]
                seen.extend((step, k, v) for k, v in kept)

        train(run, tempfile.mkdtemp(), echo=lambda _: None, sinks=[Seen()])
        return seen

    now = losses()

    def head_before(self: Head, x: Any, taps: Any) -> Any:
        taps.heads.append(HeadOutput(x, self.projection()))
        return x

    def sums_before(self: HeadOutput, targets: Any, weights: Any, z: bool) -> Any:
        return parallel.method("loss")(self, targets, weights, z)

    monkeypatch.setattr(Head, "__call__", head_before)
    monkeypatch.setattr(HeadOutput, "sums", sums_before)
    assert losses() == now
    assert [(s, k) for s, k, _ in now] == [
        (1, "loss"),
        (2, "loss"),
        (2, "eval_loss"),
        (3, "loss"),
        (4, "loss"),
        (4, "eval_loss"),
    ]
