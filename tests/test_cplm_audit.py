"""An independent audit of CPLM's copy pointer (bux.architecture.copy) for leakage:
causality in the training loss and the eval, document boundaries, a negative control
on data without repetition, and the eval's bits per byte recomputed in numpy from
the model's raw outputs. It reads the pointer's code, not its tests."""

import inspect
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jaxtyping import Float, Int

from bux.architecture import Model
from bux.architecture.copy import log_probs
from bux.architecture.layers.embedding import Head
from bux.export.check import perturbed
from bux.lang import compile
from bux.train import config as tcfg
from bux.train.data import Batch
from bux.train.objective import held_out, objective, targets_ahead
from bux.train.trainer import Trainer, forward

V = 64  # the vocabulary; its last row, 63, is <copy>
COPY = V - 1
T = 24
SOURCE = f"""
d_model = 32
vocab_size = {V}
norm = rmsnorm
attn = gqa(num_heads=2, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=64, activation=swish)
block = [norm attn] [norm ffn]
model = embed block^2 norm head(pointer=copy_pointer(dim=16))
"""
# held_out takes buckets of positions on the timeline's tree, not on nl-cplm's: this
# file runs on both, and calls them through `bucketed` and `buckets` there alone.
BUCKETS = "positions" in inspect.signature(held_out).parameters
bucketed: Any = held_out
buckets: Any = getattr(tcfg, "buckets", None)


def the_head(model: Model) -> Head:
    return next(m for _, m in model.iter_modules() if isinstance(m, Head))


def gated_model(seed: int = 0) -> Model:
    """A tiny model with the pointer, its weights perturbed, its <copy> logit set so
    that the gate a is well inside (0, 1), and its pointer's attention sharpened."""
    model = perturbed(Model(compile(SOURCE), rngs=nnx.Rngs(seed)), seed=seed + 1)
    head = the_head(model)
    assert head.proj is not None and head.pointer is not None
    ids = np.random.default_rng(seed).integers(0, 8, (2, T))
    x = np.asarray(forward(model, rows(ids))[0].hidden, np.float64)
    mean = x.reshape(-1, x.shape[-1]).mean(0)
    kernel = head.proj.kernel[...]
    head.proj.kernel[...] = kernel.at[:, COPY].set(4.5 * mean / (mean @ mean))
    head.pointer.gain[...] = jnp.asarray(3.0)
    return model


def rows(ids, positions=None, mask=None) -> Batch:
    ids = np.asarray(ids, np.int32)
    if positions is None:
        positions = np.broadcast_to(np.arange(ids.shape[1], dtype=np.int32), ids.shape)
    if mask is None:
        mask = np.ones(ids.shape, bool)
    return Batch(ids, np.asarray(positions, np.int32), np.asarray(mask, bool))


def tokens(seed: int, alphabet: int = 8) -> Int[np.ndarray, "b t"]:
    """Rows of a small alphabet, so that targets repeat earlier tokens."""
    return np.random.default_rng(seed).integers(0, alphabet, (2, T))


def gate(model: Model, batch: Batch) -> Float[np.ndarray, "*s"]:
    """The softmax's mass on <copy> at each position."""
    logits = np.asarray(forward(model, batch)[0].logits(), np.float64)
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return e[..., COPY] / e.sum(-1)


def per_position_nll(model: Model, batch: Batch) -> Float[np.ndarray, "*s"]:
    """Each position's term of the training loss, by the loss's own path
    (HeadOutput.sums, the pointer's chunked mixture), one position at a time."""
    out = forward(model, batch)[0]
    targets = targets_ahead(batch, 1)
    b, t = targets.shape
    eye = jnp.eye(b * t, dtype=jnp.float32).reshape(b * t, b, t)
    sums = jax.vmap(lambda w: out.sums(targets, w, False).ce)(eye)
    return np.asarray(sums).reshape(b, t)


# --- 1. causality ---------------------------------------------------------------------


@pytest.mark.parametrize("t0", [3, 11, 20])
def test_1_the_mixture_at_t_reads_nothing_after_t(t0):
    """The whole distribution at positions <= t0, p_copy over every token included,
    and each loss term whose target is <= t0, are bitwise unchanged by tokens after
    t0; in the training loss's path and the eval's."""
    model = gated_model()
    ids = tokens(1)
    later = ids.copy()
    later[:, t0 + 1 :] = tokens(2, alphabet=COPY)[:, t0 + 1 :]
    a, b = rows(ids), rows(later)
    g = gate(model, a)[:, : t0 + 1]
    assert ((g > 0.05) & (g < 0.95)).mean() > 0.5, g  # the gate is open
    dist = [np.asarray(log_probs(forward(model, r)[0])) for r in (a, b)]
    assert np.array_equal(dist[0][:, : t0 + 1], dist[1][:, : t0 + 1])
    nll = [per_position_nll(model, r) for r in (a, b)]
    assert np.array_equal(nll[0][:, :t0], nll[1][:, :t0])  # targets <= t0
    assert not np.array_equal(nll[0][:, t0 + 1 :], nll[1][:, t0 + 1 :])  # it moves


def every_token_nll(out, each) -> Float[np.ndarray, "*s"]:
    """The loss path's negative log-likelihood of every real token y, as the
    target of each position that `each` weighs alone."""

    def nll(y, w):
        return out.sums(jnp.full((2, T), y, jnp.int32), w, False).ce

    return np.asarray(
        jax.vmap(lambda y: jax.vmap(lambda w: nll(y, w))(each))(jnp.arange(COPY))
    )


@pytest.mark.parametrize("t0", [3, 11, 20])
def test_1_the_loss_s_mixture_of_every_token_at_t_reads_nothing_after_t(t0):
    """By the loss's and the eval's own path (HeadOutput.sums, which held_out calls),
    the mixture's probability of every token y at positions <= t0, its next token's
    position t0 + 1 included, is bitwise unchanged by the tokens after t0: no
    pointer source at or past the target."""
    model = gated_model()
    ids = tokens(1)
    later = ids.copy()
    later[:, t0 + 1 :] = tokens(2, alphabet=COPY)[:, t0 + 1 :]
    weights = jnp.zeros((2, T)).at[:, : t0 + 1].set(1.0)
    each = jnp.eye(2 * T).reshape(2 * T, 2, T) * weights
    probs = [every_token_nll(forward(model, rows(r))[0], each) for r in (ids, later)]
    assert np.array_equal(probs[0], probs[1])


@pytest.mark.parametrize("t0", [5, 15])
def test_1_the_training_objective_and_its_gradient_read_nothing_after_t(t0):
    """The objective over targets <= t0 (the loss mask ends there), and its gradient
    of every parameter, by the training forward, are bitwise those of rows whose
    later tokens differ."""
    model = gated_model()
    obj = objective(tcfg.LossConfig(mtp=0.0), compile(SOURCE))
    ids = tokens(3)
    later = ids.copy()
    later[:, t0 + 1 :] = tokens(4, alphabet=COPY)[:, t0 + 1 :]
    mask = np.zeros(ids.shape, bool)
    mask[:, 1 : t0 + 1] = True  # the targets of positions 0 .. t0 - 1
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def loss(params, batch):
        m = nnx.merge(graphdef, params, rest)
        heads, _ = m.training_heads(jnp.asarray(batch.input_ids))
        part, _ = obj(heads, batch)
        return part.loss

    got = [jax.value_and_grad(loss)(params, rows(r, mask=mask)) for r in (ids, later)]
    assert np.array_equal(np.asarray(got[0][0]), np.asarray(got[1][0]))
    for g0, g1 in zip(jax.tree.leaves(got[0][1]), jax.tree.leaves(got[1][1])):
        assert np.array_equal(np.asarray(g0), np.asarray(g1))


def test_1_the_trainer_s_eval_reads_nothing_after_t():
    """held_out, as the trainer's eval step calls it, on targets <= t0, with bits per
    byte (and by position, where the tree has buckets)."""
    model = gated_model()
    trainer = Trainer(
        model, optax.adamw(1e-3), objective(tcfg.LossConfig(), compile(SOURCE))
    )
    t0 = 13
    ids = tokens(5)
    later = ids.copy()
    later[:, t0 + 1 :] = tokens(6, alphabet=COPY)[:, t0 + 1 :]
    mask = np.zeros(ids.shape, bool)
    mask[:, 1 : t0 + 1] = True
    nbytes = np.random.default_rng(7).integers(1, 4, ids.shape)
    out = []
    for r in (ids, later):
        batch = rows(r, mask=mask)
        if BUCKETS:
            heads = forward(model, batch)
            m = bucketed(heads[0], batch, jnp.asarray(nbytes), 1, positions=(4, 8))
        else:
            m = trainer.eval_step(model, batch, jnp.asarray(nbytes))
        out.append({k: (float(s), float(n)) for k, (s, n) in m.items()})
    assert out[0] == out[1]
    assert out[0]["loss"][1] == 2 * t0


def test_1_p_copy_is_over_the_tokens_before_the_query_alone():
    """At query t, p_copy(y) is the pointer's attention on the positions j < t whose
    token is y: every token with mass appeared before t, and the target at t + 1
    gets mass only if it did."""
    model = gated_model()
    ids = np.array([[5, 6, 7, 5, 6, 7, 1, 2, 3, 9, 9, 4] * 2])
    batch = rows(ids)
    out = forward(model, batch)[0]
    assert out.pointer is not None
    targets = targets_ahead(batch, 1)
    log_copy, _ = out.pointer.logs(targets)
    log_copy = np.asarray(log_copy)[0]
    seen = [set(ids[0, :t].tolist()) for t in range(ids.shape[1])]
    for t in range(ids.shape[1] - 1):
        target = int(ids[0, t + 1])
        if target not in seen[t]:
            assert log_copy[t] == -np.inf, (t, target)
        else:
            assert np.isfinite(log_copy[t]), (t, target)
    # Over the whole vocabulary: only tokens before t carry pointer mass. Every
    # target at t = 0 (no earlier token) has none.
    full = np.stack(
        [
            np.asarray(out.pointer.logs(jnp.full_like(targets, y))[0])[0]
            for y in range(V)
        ]
    )
    for t in range(ids.shape[1]):
        carried = {y for y in range(V) if np.isfinite(full[y, t])}
        assert carried <= seen[t], (t, carried - seen[t])


# --- 2. documents -----------------------------------------------------------------------


def packed() -> tuple[Int[np.ndarray, "b t"], Int[np.ndarray, "b t"]]:
    """A row of three documents, as best_fit packs them: positions start again at 0."""
    lengths = (7, 10, 7)
    positions = np.concatenate([np.arange(n) for n in lengths])[None].repeat(2, 0)
    return tokens(8)[:, : sum(lengths)], positions


def test_2_the_pointer_never_crosses_a_document():
    """Each document's distribution, loss terms and eval sums are bitwise unchanged
    by another document's tokens; a target that only another document holds gets no
    pointer mass."""
    model = gated_model()
    ids, positions = packed()
    other = ids.copy()
    other[:, :7] = tokens(9, alphabet=COPY)[:, :7]  # the first document
    other[:, 17:] = tokens(10, alphabet=COPY)[:, 17:]  # the third
    a, b = rows(ids, positions), rows(other, positions)
    dist = [np.asarray(log_probs(forward(model, r)[0])) for r in (a, b)]
    assert np.array_equal(dist[0][:, 7:17], dist[1][:, 7:17])
    nll = [per_position_nll(model, r) for r in (a, b)]
    # The second document's targets (positions 7 .. 15 predict 8 .. 16).
    assert np.array_equal(nll[0][:, 7:16], nll[1][:, 7:16])
    mask = np.zeros(ids.shape, bool)
    mask[:, 8:17] = True
    nbytes = jnp.ones(ids.shape, jnp.int32)
    sums = [
        {
            k: (float(s), float(n))
            for k, (s, n) in held_out(
                forward(model, rows(r, positions, mask))[0],
                rows(r, positions, mask),
                nbytes,
            ).items()
        }
        for r in (ids, other)
    ]
    assert sums[0] == sums[1]

    # A target the first document alone holds: no pointer mass in the second.
    row = np.array([[11, 12, 13, 14, 15, 16, 17, 1, 2, 3, 4, 5, 6, 0, 1, 11, 12]])
    pos = np.concatenate([np.arange(7), np.arange(10)])[None]
    batch = rows(row, pos)
    out = forward(model, batch)[0]
    assert out.pointer is not None
    log_copy, _ = out.pointer.logs(targets_ahead(batch, 1))
    log_copy = np.asarray(log_copy)[0]
    # positions 14 and 15 predict 11 and 12, which only the first document has
    assert log_copy[14] == -np.inf and log_copy[15] == -np.inf
    assert np.isfinite(log_copy[13])  # 1 is in the second document, at 8


# --- 4. accounting ----------------------------------------------------------------------


def reference_nll(
    out, ids: Int[np.ndarray, "b t"], positions: Int[np.ndarray, "b t"]
) -> Float[np.ndarray, "*s"]:
    """-log(p_lm (1 - a (1 - a_sink)) + a p_copy) of each true next token, from the
    head's raw outputs, in numpy float64: its logits, the pointer's queries, keys and
    sink logits, and the row's tokens and documents."""
    p = out.projection
    x = np.asarray(out.hidden, np.float64)
    logits = x @ np.asarray(p.kernel, np.float64)
    if p.bias is not None:
        logits += np.asarray(p.bias, np.float64)
    if p.softcap is not None:
        cap = float(p.softcap)
        logits = cap * np.tanh(logits / cap)
    pt = out.pointer
    q, k = np.asarray(pt.query, np.float64), np.asarray(pt.key, np.float64)
    sink = np.asarray(pt.sink, np.float64)
    b, t, c = q.shape
    docs = np.cumsum(
        np.concatenate([np.ones((b, 1)), positions[:, 1:] <= positions[:, :-1]], 1), 1
    )
    nll = np.zeros((b, t - 1))
    for r in range(b):
        for i in range(t - 1):
            z = logits[r, i]
            e = np.exp(z - z.max())
            a = e[COPY] / e.sum()
            lm = np.delete(e, COPY)
            lm = lm / lm.sum()  # p_lm: the other rows, renormalized
            keys = [j for j in range(i) if docs[r, j] == docs[r, i]]
            scores = np.array(
                [q[r, i] @ k[r, j] / np.sqrt(c) for j in keys] + [sink[r, i]]
            )
            w = np.exp(scores - scores.max())
            w = w / w.sum()
            a_sink = w[-1]
            y = int(ids[r, i + 1])
            p_copy = sum(w[n] for n, j in enumerate(keys) if ids[r, j] == y)
            p_y = lm[y if y < COPY else y - 1] * (1 - a * (1 - a_sink)) + a * p_copy
            nll[r, i] = -np.log(p_y)
    return nll


def test_4_bits_per_byte_are_the_mixture_s_recomputed_in_numpy():
    model = gated_model()
    ids, positions = packed()
    batch = rows(ids, positions)
    out = forward(model, batch)[0]
    nll = reference_nll(out, ids, positions)
    nbytes = np.random.default_rng(11).integers(0, 4, ids.shape)  # some tokens 0 bytes
    mask = np.zeros(ids.shape, bool)
    mask[:, 1:] = True
    mask[:, [7, 17]] = False  # no target across a document's start
    m = held_out(out, rows(ids, positions, mask), jnp.asarray(nbytes))
    targets = mask[:, 1:]
    want_loss = nll[targets].sum()
    assert float(m["loss"][0]) == pytest.approx(want_loss, rel=2e-5)
    assert float(m["loss"][1]) == targets.sum()
    counted = np.where(targets, nbytes[:, 1:], 0)
    want_bits = nll[counted > 0].sum() / np.log(2)
    assert float(m["bpb"][0]) == pytest.approx(want_bits, rel=2e-5)
    assert float(m["bpb"][1]) == counted.sum()
    # Per position, by the loss's own path.
    per = per_position_nll(model, batch)[:, :-1]
    np.testing.assert_allclose(per[targets], nll[targets], rtol=2e-5)
    if not BUCKETS:
        return
    # By the targets' positions in their documents, as the timeline's repo_bpb_8k.
    edges = (4, 8)
    by = bucketed(out, rows(ids, positions, mask), jnp.asarray(nbytes), 1, edges)
    bucket = np.searchsorted(np.asarray(edges), positions[:, 1:], "right")
    for i, name in enumerate(buckets(edges)):
        here = targets & (bucket == i)
        assert float(by[f"loss_{name}"][0]) == pytest.approx(nll[here].sum(), rel=2e-5)
        bits = nll[here & (counted > 0)].sum() / np.log(2)
        assert float(by[f"bpb_{name}"][0]) == pytest.approx(bits, rel=2e-5)
        assert float(by[f"bpb_{name}"][1]) == counted[here].sum()


def test_4_copy_carries_no_mass_and_the_mixture_sums_to_one():
    """<copy> is -inf in the distribution served, which sums to 1 over the real
    tokens; a real token is never <copy> in the timeline's vocabulary."""
    model = gated_model()
    out = forward(model, rows(tokens(12)))[0]
    lp = np.asarray(log_probs(out), np.float64)
    assert (lp[..., COPY] == -np.inf).all()
    np.testing.assert_allclose(np.exp(lp).sum(-1), 1.0, rtol=1e-5)
    from bux.data import tokenizer

    tok = tokenizer.load(tcfg.TokenizerConfig(name="EleutherAI/gpt-neox-20b"))
    assert tok.vocab_size <= 50304 - 1  # the timeline's vocab_size; <copy> is 50303


def dense_nll(out, ids, positions) -> Float[jax.Array, "*s"]:
    """The mixture's negative log-likelihood of each true next token, in plain jax
    from the head's raw outputs (logits, the pointer's queries, keys and sink), all
    t by t at once, for autodiff: the reference the custom gradient must match."""
    p, pt = out.projection, out.pointer
    logits = p(out.hidden).astype(jnp.float32)
    b, t, c = pt.query.shape
    scores = jnp.einsum("bic,bjc->bij", pt.query, pt.key) * c**-0.5
    docs = jnp.cumsum(
        jnp.concatenate(
            [
                jnp.ones((b, 1)),
                (positions[:, 1:] <= positions[:, :-1]).astype(jnp.float32),
            ],
            1,
        ),
        1,
    )
    valid = (jnp.arange(t)[:, None] > jnp.arange(t)[None]) & (
        docs[:, :, None] == docs[:, None, :]
    )
    full = jnp.concatenate([jnp.where(valid, scores, -jnp.inf), pt.sink[..., None]], -1)
    attn = jax.nn.softmax(full, -1)
    target = jnp.asarray(ids)[:, 1:]
    hit = jnp.asarray(ids)[:, None, :] == jnp.pad(target, ((0, 0), (0, 1)))[:, :, None]
    p_copy = jnp.sum(jnp.where(hit, attn[..., :t], 0), -1)[:, :-1]
    a_sink = attn[..., t][:, :-1]
    probs = jax.nn.softmax(logits, -1)[:, :-1]
    a = probs[..., COPY]
    lm = jax.nn.softmax(logits.at[..., COPY].set(-jnp.inf), -1)[:, :-1]
    p_lm = jnp.take_along_axis(lm, target[..., None], -1)[..., 0]
    return -jnp.log(p_lm * (1 - a * (1 - a_sink)) + a * p_copy)


def test_4_the_loss_s_custom_gradient_is_the_formula_s():
    """The gradient of the loss's chunked mixture (its custom VJP) of every
    parameter is the autodiff gradient of the dense formula above: a pointer that
    learns as the mixture says, not some other way."""
    model = gated_model()
    ids, positions = packed()
    batch = rows(ids, positions)
    targets = targets_ahead(batch, 1)
    mask = np.zeros(ids.shape, np.float32)
    mask[:, :-1] = 1
    mask[:, [6, 16]] = 0  # positions whose target starts a document
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def ours(params):
        out = forward(nnx.merge(graphdef, params, rest), batch)[0]
        return out.sums(targets, jnp.asarray(mask), False).ce

    def reference(params):
        out = forward(nnx.merge(graphdef, params, rest), batch)[0]
        return jnp.sum(dense_nll(out, ids, jnp.asarray(positions)) * mask[:, :-1])

    v1, g1 = jax.value_and_grad(ours)(params)
    v2, g2 = jax.value_and_grad(reference)(params)
    assert float(v1) == pytest.approx(float(v2), rel=1e-5)
    for a, b in zip(jax.tree.leaves(g1), jax.tree.leaves(g2), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=2e-3, atol=2e-5)


# --- 3. negative control ----------------------------------------------------------------

NEG_V = 128  # 127 real tokens, so that a row of 64 has no token twice
NEG = """
d_model = 32
vocab_size = 128
norm = rmsnorm
attn = gqa(num_heads=2, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=64, activation=swish)
block = [norm attn] [norm ffn]
model = embed block^2 norm {head}
"""


def unique_rows(rng, b: int, t: int) -> Int[np.ndarray, "b t"]:
    """Each row a document of distinct tokens: nothing to copy."""
    return np.stack([rng.permutation(NEG_V - 1)[:t] for _ in range(b)])


def repeated_rows(rng, b: int, t: int) -> Int[np.ndarray, "b t"]:
    """Each row a random sequence of 16 tokens, four times over."""
    return np.stack(
        [np.tile(rng.integers(0, NEG_V - 1, 16), t // 16) for _ in range(b)]
    )


def trained_eval(head: str, make, steps: int = 300) -> float:
    arch = compile(NEG.format(head=head))
    model = Model(arch, rngs=nnx.Rngs(0))
    trainer = Trainer(model, optax.adamw(3e-3), objective(tcfg.LossConfig(), arch))
    rng = np.random.default_rng(0)
    for _ in range(steps):
        trainer.train_on_batch(rows(make(rng, 8, 64)))
    held = np.random.default_rng(99)
    return trainer.evaluate([rows(make(held, 8, 64)) for _ in range(4)])["loss"]


def test_3_without_repetition_the_pointer_gains_nothing(capsys):
    """On documents of distinct tokens the target never appeared before, so an honest
    pointer has nothing to copy: no gain beyond noise. A pointer that saw its target
    would drive the loss toward 0 here."""
    off = trained_eval("head", unique_rows)
    on = trained_eval("head(pointer=copy_pointer(dim=16))", unique_rows)
    with capsys.disabled():
        print(
            f"\n[audit] unique: eval loss off {off:.4f}, on {on:.4f} ({on - off:+.4f})"
        )
    assert on >= off - 0.01


def test_3_with_repetition_the_pointer_copies_the_repeats(capsys):
    """On rows of a sequence repeated four times the gate opens where the next token
    repeats an earlier one, far less where it is new, and the pointer's mass on the
    target grows as it trains. (Its loss stays above the plain head's at these
    budgets: two layers learn induction themselves, faster than the pointer's gain
    grows.)"""
    off = trained_eval("head", repeated_rows)
    arch = compile(NEG.format(head="head(pointer=copy_pointer(dim=16))"))
    model = Model(arch, rngs=nnx.Rngs(0))
    trainer = Trainer(model, optax.adamw(3e-3), objective(tcfg.LossConfig(), arch))
    rng = np.random.default_rng(0)
    for _ in range(300):
        trainer.train_on_batch(rows(repeated_rows(rng, 8, 64)))
    held = np.random.default_rng(99)
    on = trainer.evaluate([rows(repeated_rows(held, 8, 64)) for _ in range(4)])["loss"]
    batch = rows(repeated_rows(np.random.default_rng(5), 4, 64))
    out = forward(trainer.model, batch)[0]
    assert out.pointer is not None
    logits = np.asarray(out.logits(), np.float64)
    e = np.exp(logits - logits.max(-1, keepdims=True))
    a = e[..., NEG_V - 1] / e.sum(-1)
    log_copy, _ = out.pointer.logs(targets_ahead(batch, 1))
    p_copy = np.exp(np.asarray(log_copy))
    repeats, first = a[:, 16:63].mean(), a[:, :15].mean()
    with capsys.disabled():
        print(
            f"\n[audit] repeated: eval loss off {off:.4f}, on {on:.4f} ({on - off:+.4f});"
            f" gate {repeats:.3f} at repeats, {first:.3f} at new tokens;"
            f" p_copy(target) {p_copy[:, 16:63].mean():.3f} at repeats"
        )
    assert repeats > 0.9 and first < 0.5
    assert p_copy[:, 16:63].mean() > 1 / 16  # sharper than copying any earlier token
