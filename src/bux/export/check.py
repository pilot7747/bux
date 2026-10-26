"""Check an export against bux: the same logits, and the same greedy tokens.

`compare` exports a bux model, loads it back with transformers, and measures:

- every head's logits on a batch, against bux's;
- greedy decoding with transformers' cache, without it, and by bux's full forwards:
  the cache must change nothing;
- left-padded batches: each prompt decoded beside a shorter one, as alone.

::

    report = compare(arch)  # a model of fresh weights, perturbed
    report.check()  # raises unless everything agrees
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jaxtyping import Float, Int

from bux.architecture import Model
from bux.export.huggingface import save
from bux.lang import ir


@dataclass
class Result:
    """How one attention implementation's run of an export compares with bux.

    Attributes:
        implementation: transformers' attention implementation: ``eager``, ``sdpa``.
        error: The largest difference of any head's logits from bux's, over the
            largest of bux's.
        greedy: The greedy tokens with the cache, without it, and bux's, one row
            each, prompt included.
        padded: Whether greedy tokens of a left-padded batch equal each prompt's
            decoded alone.
    """

    implementation: str
    error: float
    greedy: dict[str, list[int]]
    padded: bool


@dataclass
class Report:
    """What `compare` measured, by attention implementation."""

    results: list[Result] = field(default_factory=list)
    tolerance: float = 1e-4

    def check(self) -> None:
        """Raise unless the logits agree within the tolerance and every greedy
        decode, padded or not, gives bux's tokens.

        Raises:
            AssertionError: Saying what disagrees.
        """
        for r in self.results:
            assert r.error <= self.tolerance, (
                f"{r.implementation}: logits off by {r.error:.2e}"
            )
            want = r.greedy["bux"]
            for how in ("cache", "no cache"):
                got = r.greedy[how]
                assert got == want, (
                    f"{r.implementation} greedy, {how}: {got}, bux {want}"
                )
            assert r.padded, (
                f"{r.implementation}: a left-padded batch decodes otherwise"
            )


def perturbed(model: Model, seed: int = 1, scale: float = 0.1) -> Model:
    """Add noise of `scale` to every parameter of `model`, in place, so that the ones
    and zeros parameters start at hide no mistake of an export."""
    state = nnx.state(model, nnx.Param)
    leaves, tree = jax.tree.flatten(state)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    noisy = [
        x + scale * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, keys)
    ]
    nnx.update(model, jax.tree.unflatten(tree, noisy))
    return model


def compare(
    arch: ir.Architecture,
    model: Model | None = None,
    *,
    implementations: tuple[str, ...] = ("eager", "sdpa"),
    experts: str | None = None,
    batch: int = 2,
    length: int = 12,
    steps: int = 6,
    tolerance: float = 1e-4,
    out: Path | None = None,
    seed: int = 0,
) -> Report:
    """Export `model` and compare transformers' run of it with bux's, in float32.

    Args:
        arch: The architecture.
        model: Its bux model; None for fresh weights, perturbed (`perturbed`).
        implementations: transformers' attention implementations to run it by.
        experts: transformers' experts implementation to run an MoE's routed experts
            by (``eager``, ``grouped_mm``, ...); None for its default.
        batch: Rows of the batch the logits are compared on.
        length: Tokens in each row.
        steps: Tokens to decode greedily after a prompt of half the length.
        tolerance: The largest relative error of logits `Report.check` allows.
        out: Where to export it; None for a temporary directory.
        seed: Draws the tokens, and the weights when `model` is None.

    Returns:
        The report, which `Report.check` asserts on.
    """
    if model is None:
        model = perturbed(Model(arch, rngs=nnx.Rngs(seed)), seed + 1)
    assert arch.vocab_size is not None, "an export needs a vocabulary"
    tokens = jax.random.randint(
        jax.random.key(seed), (batch, length), 0, arch.vocab_size
    )
    want = [np.asarray(y, np.float32) for y in model.outputs(tokens)]
    prompt = tokens[:1, : length // 2]
    greedy = np.asarray(_greedy(model, prompt, steps))[0].tolist()
    with tempfile.TemporaryDirectory() as tmp:
        where = Path(out or tmp)
        save(arch, model, where)
        report = Report(tolerance=tolerance)
        for impl in implementations:
            hf = _load(where, impl, experts)
            got = _heads(hf, np.asarray(tokens))
            error = max(_relative(g, w) for g, w in zip(got, want, strict=True))
            runs = {
                "cache": _generate(hf, prompt, steps, True),
                "no cache": _generate(hf, prompt, steps, False),
                "bux": greedy,
            }
            padded = _padded(hf, np.asarray(tokens), length // 2, steps)
            report.results.append(Result(impl, error, runs, padded))
    return report


def _greedy(
    model: Model, prompt: Int[jax.Array, "1 t"], steps: int
) -> Int[jax.Array, "1 n"]:
    """Decode greedily by full forwards of bux's model."""
    for _ in range(steps):
        nxt = jnp.argmax(model(prompt)[:, -1], -1)
        prompt = jnp.concatenate([prompt, nxt[:, None]], 1)
    return prompt


def _load(where: Path, impl: str, experts: str | None = None) -> Any:
    """Load the export at `where` with transformers, in float32, by the attention
    implementation `impl` and the experts implementation `experts`.

    Raises:
        AssertionError: A weight of the checkpoint or of the model has no match;
            transformers would only warn, and draw the missing ones at random.
    """
    from transformers import AutoModelForCausalLM  # slow to import

    chosen = {} if experts is None else {"experts_implementation": experts}
    loaded: Any = AutoModelForCausalLM.from_pretrained(
        where,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation=impl,
        output_loading_info=True,
        **chosen,
    )
    model, info = loaded
    unmatched = {k: v for k, v in info.items() if k != "error_msgs" and v}
    assert not unmatched, f"weights without a match: {unmatched}"
    return model.eval()


def _heads(hf: Any, tokens: Int[np.ndarray, "b t"]) -> list[np.ndarray]:
    """Return every head's logits of transformers' run, as numpy."""
    with torch.no_grad():
        heads, _ = hf.model(torch.from_numpy(tokens).long())
    return [h.float().numpy() for h in heads]


def _relative(got: Float[np.ndarray, "..."], want: Float[np.ndarray, "..."]) -> float:
    """The largest difference of `got` from `want`, over the largest of `want`."""
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-6))


def _generate(
    hf: Any, prompt: Int[jax.Array, "1 t"], steps: int, cache: bool
) -> list[int]:
    """Decode greedily with transformers' generate."""
    with torch.no_grad():
        ids = torch.from_numpy(np.asarray(prompt)).long()
        out = hf.generate(
            ids,
            max_new_tokens=steps,
            do_sample=False,
            use_cache=cache,
            eos_token_id=None,
            pad_token_id=0,
        )
    return out[0].tolist()


def _padded(hf: Any, tokens: Int[np.ndarray, "b t"], length: int, steps: int) -> bool:
    """Whether a batch of a prompt and a shorter one, left-padded, decodes as each
    prompt alone."""
    short = max(length // 2, 1)
    rows = [tokens[0, :length], tokens[1 % len(tokens), :short]]
    ids = np.zeros((2, length), np.int64)
    mask = np.zeros((2, length), np.int64)
    for i, row in enumerate(rows):
        ids[i, length - len(row) :], mask[i, length - len(row) :] = row, 1
    with torch.no_grad():
        both = hf.generate(
            torch.from_numpy(ids),
            attention_mask=torch.from_numpy(mask),
            max_new_tokens=steps,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )
        for i, row in enumerate(rows):
            alone = hf.generate(
                torch.from_numpy(row[None].astype(np.int64)),
                max_new_tokens=steps,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
            )
            if both[i, length:].tolist() != alone[0, len(row) :].tolist():
                return False
    return True
