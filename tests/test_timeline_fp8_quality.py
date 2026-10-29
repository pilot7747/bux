"""scripts/timeline/fp8_quality.py: its eval sets are the documents bux's own eval
reads, so that bits per byte from their tokens' log-probabilities, each document
scored alone as vLLM serves it, are what the run logged."""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import corpus
from jaxtyping import Float32

from bux.export import runs
from bux.lang import compile_runs
from bux.train import config as tcfg
from bux.train.data import Batch
from bux.train.trainer import directory, forward, train

ROOT = Path(__file__).parent.parent


def script() -> ModuleType:
    """Import scripts/timeline/fp8_quality.py, which is no package's."""
    path = ROOT / "scripts" / "timeline" / "fp8_quality.py"
    spec = importlib.util.spec_from_file_location("fp8_quality", path)
    assert spec is not None and spec.loader is not None
    module = sys.modules["fp8_quality"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fq = script()

#: A tiny hybrid of the timeline's layers, each of which keeps to its document: KDA,
#: gated attention with a conv on its keys and values, an MoE of aux-free routing,
#: hyper-connections. The data are relative to the run's working directory, as the
#: timeline's are, and its evals are named and bucketed as its recipe's.
SOURCE = """
d_model = 128
vocab_size = 264
delta = kda(num_heads=1, num_v_heads=2, head_dim=128, gate_lower_bound=-5)
attn = gqa(num_heads=2, num_kv_heads=1, kv_conv_size=4, qk_norm=rmsnorm, gated=true,
           rope=rope())
experts = moe(num_experts=4, top_k=2, num_shared_experts=1, router=sigmoid,
              load_balancing=aux_free,
              expert=gated_mlp(intermediate_size=32, activation=swish))
layer(mixer) = [rmsnorm mixer] [rmsnorm experts]
model = embed layer(delta) layer(attn) rmsnorm head
residual = hc(expansion=2)

tokenizer = tokenizer(name="{tokenizer}")
code = text(path="data/code/*.parquet")
repos = text(path="data/repos/*.parquet")
data = code(split=[0, 0.9])
seq_len = 64
batch = 4
steps = 2
optimizer = adamw(lr=1e-2)
precision = precision(compute=float32)
mesh = mesh(data=2)
checkpoint = checkpoint(every=2, keep=1)
eval = [eval(name="code", data=code(split=[0.9, 1]), every=2, batches=2),
        eval(name="repo", data=repos, every=2, seq_len=256, batch=2, batches=2,
             positions=[16, 64])]
"""


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, byte_tokenizer) -> tuple[Path, Path]:
    """A run trained in a directory of its own data, and what ``prepare`` wrote of it:
    the run's directory and the output's."""
    root = tmp_path_factory.mktemp("root")
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        (root / "data").mkdir()
        corpus(Path("data/code"))
        corpus(Path("data/repos"), n=200)
        [run] = compile_runs(SOURCE.format(tokenizer=byte_tokenizer.name))
        train(run, root / "runs", echo=lambda _: None)
    folder, out = directory(run, root / "runs"), root / "out"
    args = argparse.Namespace(step=None, workers=0, pad=128)
    fq.prepare(folder, out, root, [root / "data" / "packed"], args)
    return folder, out


def bux_scores(model, s, seq_len: int) -> Float32[np.ndarray, " tokens"]:
    """Each token's log-probability by bux's `model`, each document of `s` alone in a
    row from position 0, as vLLM serves it; NaN at each document's first."""
    spans = s.spans
    ids = np.zeros((len(spans), seq_len), np.int32)
    for i, (a, b) in enumerate(spans):
        ids[i, : b - a] = s.tokens[a:b]
    positions = np.broadcast_to(np.arange(seq_len, dtype=np.int32), ids.shape)
    rows = Batch(ids, positions, np.ones(ids.shape, bool))
    logits = forward(model, rows)[0].logits()
    lp = np.asarray(jax.nn.log_softmax(logits.astype(jnp.float32), -1))
    out = np.full(len(s.tokens), np.nan, np.float32)
    for i, (a, b) in enumerate(spans):
        out[a + 1 : b] = lp[i, np.arange(b - a - 1), ids[i, 1 : b - a]]
    return out


def save(out: Path, tag: str, scores: dict[str, Float32[np.ndarray, " _n"]]) -> None:
    """Keep `scores` as ``run`` keeps a configuration's, under `tag`."""
    info = {"engine": {}, "built": {}, "seconds": {}}
    fq.save_scores(out, tag, scores, info)


@pytest.fixture(scope="module")
def scored(prepared) -> dict[str, Float32[np.ndarray, " _n"]]:
    """Each set's log-probabilities by the restored run's own model, kept as ``bux``."""
    folder, out = prepared
    model = runs.restore(folder).model
    meta = json.loads((out / "prepared.json").read_text())
    scores = {
        n: bux_scores(model, fq.EvalSet.load(out, n), info["seq_len"])
        for n, info in meta["sets"].items()
    }
    save(out, "bux", scores)
    return scores


def test_bits_per_byte_of_the_documents_alone_are_the_runs_logged_eval(
    prepared, scored
):
    _, out = prepared
    one = fq.summary(out, "bux")
    repo = [f"repo_{k}_{b}" for k in ("bpb", "loss") for b in ("0", "16", "64")]
    expected = {"code_bpb", "code_loss", "repo_bpb", "repo_loss", *repo}
    assert set(one["check"]) == expected
    assert one["logged_step"] == one["step"] == 2
    for name, c in one["check"].items():
        assert c["served"] == pytest.approx(c["logged"], rel=1e-5), name
    meta = json.loads((out / "prepared.json").read_text())
    assert meta["sets"]["repo"]["buckets"] == ["0", "16", "64"]
    assert meta["sets"]["code"]["seq_len"] == 64
    for info in meta["sets"].values():  # rows of several documents
        assert info["documents"] > info["batch"] * info["batches"]
    assert np.diff(fq.EvalSet.load(out, "repo").offsets).max() > 64  # past a bucket
    config = json.loads((out / "model" / "config.json").read_text())
    assert config["dtype"] == "bfloat16"


def widths(x) -> set[int]:
    """Every MLP's ``intermediate_size`` in the JSON data `x`."""
    if isinstance(x, list):
        return set().union(*map(widths, x))
    if not isinstance(x, dict):
        return set()
    own = {x["intermediate_size"]} if "intermediate_size" in x else set()
    return own.union(*map(widths, x.values()))


def test_an_export_padded_for_fp8_blocks_computes_what_the_run_does(prepared):
    folder, out = prepared
    restored = runs.restore(folder)
    arch, changed = fq.padded(restored.arch, 128)
    assert changed == {32: 128}  # the experts', routed and shared
    model = fq.padded_model(arch, restored.model)
    ids = np.random.default_rng(0).integers(0, 259, (2, 48)).astype(np.int32)
    rows = Batch(
        ids, np.broadcast_to(np.arange(48), ids.shape), np.ones(ids.shape, bool)
    )
    want, got = (forward(m, rows)[0].logits() for m in (restored.model, model))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    config = json.loads((out / "model" / "config.json").read_text())
    assert widths(config["bux"]) == {128}
    meta = json.loads((out / "prepared.json").read_text())
    assert meta["padded"] == {"32": 128}


def test_a_change_is_measured_on_the_same_documents(prepared, scored):
    _, out = prepared
    save(out, "worse", {n: lp.astype(np.float64) - 0.01 for n, lp in scored.items()})
    change = fq.summary(out, "bux")["tags"]["worse"]["change"]
    s, token_bytes = fq.EvalSet.load(out, "code"), np.load(out / "token_bytes.npy")
    nbytes = np.where(s.targets, token_bytes[s.tokens], 0)
    bits = 0.01 / math.log(2) * (nbytes > 0).sum()
    assert change["code_bpb"]["abs"] == pytest.approx(bits / nbytes.sum())
    assert change["code_loss"]["abs"] == pytest.approx(0.01)
    assert change["code_bpb"]["pct_se"] < abs(change["code_bpb"]["pct"])
    (out / "runs" / "worse.json").unlink()


def test_a_dataset_prepared_again_holds_the_same_documents(prepared, tmp_path):
    folder, out = prepared
    run, root = runs.read_run(folder), out.parent
    [e] = [e for e in tcfg.evals(run.train) if e.name == "repo"]
    fq.write_set(run, e, tmp_path, root, [tmp_path / "none"], workers=0)
    again, first = (np.load(d / "sets" / "repo.npz") for d in (tmp_path, out))
    for k in ("tokens", "offsets", "targets"):
        np.testing.assert_array_equal(again[k], first[k])
    with pytest.raises(SystemExit, match="no file matches"):
        fq.write_set(run, e, tmp_path, tmp_path, [tmp_path / "none"], workers=0)


def test_a_run_says_what_vllm_built_that_was_not_asked():
    online = "Fp8OnlineLinearMethod (Fp8LinearOp)"
    built = {
        "methods": {
            f"QKVParallelLinear qkv: {online}": 2,
            "ParallelLMHead head: UnquantizedEmbeddingMethod": 1,
        },
        "caches": {"Attention: kv fp8_e4m3 (FLASHINFER)": 1},
    }
    assert fq.as_asked(built, quantized=True, kv=None)
    assert not fq.as_asked(built, quantized=False, kv=None)
    assert not fq.as_asked(built, quantized=True, kv="fp8")
    built["caches"] = {"Attention: kv fp8 (FLASHINFER)": 1}
    assert fq.as_asked(built, quantized=True, kv="fp8")
    asked = [
        ("none", None, None),
        ("fp8", "fp8", None),
        ("fp8_per_block", "fp8", "cutlass"),
    ]
    tags = [
        fq.tag_of(
            argparse.Namespace(
                tag=None, quantization=q, kv_cache_dtype=kv, linear_backend=b
            )
        )
        for q, kv, b in asked
    ]
    assert tags == ["bf16", "fp8+kv", "fp8_per_block-cutlass+kv"]


def test_a_linear_backend_picks_the_fp8_matrix_products_kernels_alone(tmp_path):
    def engine(*flags: str) -> dict:
        return fq.engine_args(fq.parser().parse_args(["run", str(tmp_path), *flags]), 9)

    blocks = engine("--quantization", "fp8_per_block", "--linear-backend", "cutlass")
    per_quant = {"linear_backend_per_quant": {"fp8_block_w8a8": "cutlass"}}
    assert blocks["kernel_config"] == per_quant
    tensors = engine("--quantization", "fp8", "--linear-backend", "triton")
    assert tensors["kernel_config"]["linear_backend_per_quant"] == {
        "fp8_w8a8": "triton"
    }
    assert "kernel_config" not in engine("--quantization", "fp8")


def test_wait_returns_once_the_run_is_done_or_failed(prepared, tmp_path):
    folder, _ = prepared
    assert fq.finished(folder)[0] == "done"
    assert fq.wait(folder, every=0, timeout=0) == 0
    copy = tmp_path / folder.name
    copy.mkdir()
    (copy / "run.json").write_text((folder / "run.json").read_text())
    lines = (folder / "metrics.jsonl").read_text().splitlines(keepends=True)
    (copy / "metrics.jsonl").write_text(
        "".join(lines[:-1])
    )  # the repo eval yet to come
    assert fq.finished(copy)[0] is None
    assert fq.wait(copy, every=0, timeout=0) == 2
    (copy / "status.json").write_text(json.dumps({"state": "failed", "error": "OOM"}))
    assert fq.wait(copy, every=0, timeout=0) == 1
    (copy / "status.json").write_text(json.dumps({"state": "stopped"}))
    assert fq.finished(copy)[0] is None  # a job may resume it


class FakeLLM:
    """vLLM's LLM as `run` uses it: each prompt token's log-probability at position
    j is -j / 100, and the model it serves has no layers."""

    engine: ClassVar[dict] = {}

    def __init__(self, model: str, **engine):
        FakeLLM.engine = engine

    def generate(self, prompts, params, use_tqdm=True):
        assert (params.max_tokens, params.prompt_logprobs) == (1, 0)
        return [self._scored(p["prompt_token_ids"]) for p in prompts]

    def _scored(self, ids: list[int]) -> SimpleNamespace:
        rows = [{t: SimpleNamespace(logprob=-j / 100)} for j, t in enumerate(ids)]
        return SimpleNamespace(prompt_logprobs=[None, *rows[1:]])

    def apply_model(self, f):
        return [f(SimpleNamespace(named_modules=list))]


def test_a_run_keeps_each_tokens_log_probability_from_vllm(prepared, monkeypatch):
    _, out = prepared
    vllm = ModuleType("vllm")
    vllm.__dict__.update(LLM=FakeLLM, SamplingParams=SimpleNamespace)
    vllm.__dict__.update(__version__="fake")
    inputs = ModuleType("vllm.inputs")
    inputs.__dict__.update(TokensPrompt=dict)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    monkeypatch.setattr(fq, "CHUNK", 100)  # several calls of generate a set
    args = fq.parser().parse_args(["run", str(out), "--tag", "fake"])
    assert fq.run(args)
    repo = fq.EvalSet.load(out, "repo")
    assert FakeLLM.engine["max_model_len"] == np.diff(repo.offsets).max() + 1
    assert FakeLLM.engine["enable_prefix_caching"] is False
    got = np.load(out / "runs" / "fake.npz")["repo"]
    within = np.concatenate([np.arange(n) for n in np.diff(repo.offsets)])
    np.testing.assert_allclose(got[within > 0], -within[within > 0] / 100)
    assert np.isnan(got[within == 0]).all()
    (out / "runs" / "fake.json").unlink()


def test_each_step_runs_by_its_own_name() -> None:
    """The step a command line names is the one main runs: prepare's own --step (the
    checkpoint's) does not take its place."""
    args = fq.parser().parse_args(["prepare", "run", "out"])
    assert (args.command, args.step) == ("prepare", None)
