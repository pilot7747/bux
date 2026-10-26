"""Trained runs export (bux.export.runs): a checkpoint restores on one CPU as the model
that trained, whatever mesh, layout and methods trained it, and its export computes
what that model does, with the run's tokenizer beside it."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from conftest import SPECIALS, _byte_alphabet
from flax import nnx

from bux.__main__ import main
from bux.data.tokenizer import load
from bux.export import runs
from bux.lang import compile_runs
from bux.train.data import plain
from bux.train.hooks import Hook
from bux.train.objective import objective
from bux.train.trainer import cast, directory, forward, train

ARCH = """
d_model = 32
vocab_size = 264
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope(), qk_norm=rmsnorm)
ffn = gated_mlp(intermediate_size=64, activation=swish)
block = [rmsnorm attn] [rmsnorm ffn]
model = embed block^4 rmsnorm head
"""
STEPS = 3
RUN = {
    "data": "synthetic",
    "optimizer": "adamw(lr=1e-2)",
    "seq_len": "16",
    "batch": "8",
    "steps": str(STEPS),
    "log": f"log(every={STEPS})",
    "checkpoint": "checkpoint(every=2, keep=2)",
    # XLA:CPU cannot compile a pipeline's bfloat16 psums on more than one device.
    "precision": "precision(compute=float32)",
}
SCAN = "methods(repeat=scan, microbatches=2"
# How each run trains: its layout of the state, which a restore takes apart.
LAYOUTS = {
    "unrolled": {},
    "scanned": {"methods": "methods(repeat=scan)"},
    "sharded": {"mesh": "mesh(fsdp=4, tensor=2)", "methods": "methods(repeat=scan)"},
    "pipelined": {"mesh": "mesh(stage=2)", "methods": f"{SCAN})"},
    "uneven stages": {"mesh": "mesh(stage=2)", "methods": f"{SCAN}, stages=[3, 1])"},
    "circular": {
        "mesh": "mesh(stage=2)",
        "methods": f"{SCAN}, pipeline=circular, rounds=2)",
    },
    "bfloat16": {"precision": "precision(compute=bfloat16)"},
}
MOE = ARCH.replace(
    "block = [rmsnorm attn] [rmsnorm ffn]",
    "experts = moe(num_experts=4, top_k=2, router=sigmoid, load_balancing=aux_free, "
    "expert=gated_mlp(intermediate_size=16, activation=swish))\n"
    "block = [rmsnorm attn] [rmsnorm experts]",
)
BATCH = plain(np.random.default_rng(0).integers(0, 256, (8, 16)))


class LastLoss(Hook):
    """The trainer's loss on `BATCH` after the last step, as its evaluation computes it."""

    every = STEPS

    def __init__(self):
        self.loss: float | None = None

    def __call__(self, trainer, metrics):
        self.loss = trainer.evaluate([BATCH])


def loss(restored: runs.Restored) -> float:
    """The restored model's loss on `BATCH`, as the run's evaluation computes it."""
    run = restored.run
    model = cast(restored.model, run.train.precision.compute)
    per_token, _ = objective(run.train.loss, run.arch)(forward(model, BATCH), BATCH)
    return float(per_token.metrics["loss"])


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train a run of `source` with `settings` over `RUN`, once a module: its
    directory, and its loss after the last step."""
    done: dict[str, tuple[Path, float]] = {}

    def run_of(source: str, **settings: str) -> tuple[Path, float]:
        given = RUN | settings
        text = source + "".join(f"\n{k} = {v}" for k, v in given.items())
        if text not in done:
            [run] = compile_runs(text)
            out, hook = tmp_path_factory.mktemp("runs"), LastLoss()
            train(run, out, echo=lambda _: None, hooks=[hook])
            assert hook.loss is not None
            done[text] = directory(run, out), hook.loss
        return done[text]

    return run_of


@pytest.mark.parametrize("settings", LAYOUTS.values(), ids=LAYOUTS.keys())
def test_a_run_restores_as_it_trained_and_exports_as_it_computes(
    trained, tmp_path, settings
):
    folder, trained_loss = trained(ARCH, **settings)
    done = runs.export(folder, tmp_path / "out", check=True)
    restored = done.restored
    bf16 = "bfloat16" in settings.get("precision", "")
    assert loss(restored) == pytest.approx(trained_loss, rel=1e-2 if bf16 else 1e-5)
    assert restored.step == STEPS and done.dtype == "float32"
    assert done.report is not None
    done.report.check()


def test_an_moe_exports_with_its_router_bias_in_float32(trained, tmp_path):
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    folder, trained_loss = trained(MOE, methods="methods(repeat=scan)")
    done = runs.export(folder, tmp_path / "out", dtype="bfloat16", check=True)
    biases = [
        np.asarray(m.router_bias[...])
        for _, m in nnx.iter_graph(done.restored.model)
        if isinstance(getattr(m, "router_bias", None), nnx.Variable)
    ]
    assert len(biases) == 4 and all(np.abs(b).max() > 0 for b in biases)
    assert loss(done.restored) == pytest.approx(trained_loss, rel=1e-5)
    assert done.report is not None
    done.report.check()
    # A bias rounded to bfloat16 would flip near-tied picks of experts.
    with safe_open(done.out / "model.safetensors", "pt") as f:
        dtypes = {k: f.get_slice(k).get_dtype() for k in f.keys()}  # noqa: SIM118
    assert {d for k, d in dtypes.items() if k.endswith("router_bias")} == {"F32"}
    assert {d for k, d in dtypes.items() if k.endswith("router.weight")} == {"BF16"}
    model = AutoModelForCausalLM.from_pretrained(done.out, trust_remote_code=True)
    kept = {
        b.dtype for name, b in model.named_buffers() if name.endswith("router_bias")
    }
    assert kept == {torch.float32} and model.dtype == torch.bfloat16


def test_an_earlier_step_restores_and_a_missing_one_is_named(trained):
    folder, _ = trained(ARCH)
    assert runs.restore(folder, step=2).step == 2
    with pytest.raises(FileNotFoundError, match="no checkpoint of step 1: 2, 3"):
        runs.restore(folder, step=1)


def test_bux_export_writes_a_directory_that_loads_and_checks(trained, tmp_path, capsys):
    folder, _ = trained(ARCH, methods="methods(repeat=scan)")
    out = tmp_path / "out"
    assert (
        main(["export", str(folder), str(out), "--dtype", "bfloat16", "--check"]) == 0
    )
    said = capsys.readouterr().out
    assert "step 3 exported" in said and "eager: logits within" in said
    config = json.loads((out / "config.json").read_text())
    assert config["dtype"] == "bfloat16" and config["auto_map"]
    assert config["max_position_embeddings"] == 16  # the run's seq_len
    card = (out / "README.md").read_text()
    run_id = folder.name.rsplit("-", 1)[1]
    assert f"`{run_id}`" in card and "trust_remote_code=True" in card
    assert "block = [rmsnorm attn] [rmsnorm ffn]" in card
    assert main(["export", str(folder), str(out), "--step", "1"]) == 1
    assert "no checkpoint of step 1" in capsys.readouterr().err


@pytest.fixture(scope="module")
def tokenizer_json(tmp_path_factory) -> str:
    """conftest's byte-level BPE, saved by tokenizers, which transformers reads."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

    vocab = {c: b for b, c in _byte_alphabet().items()}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([AddedToken(s, special=True) for s in SPECIALS])
    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    tok.save(str(path))
    return str(path)


def test_the_tokenizer_loads_from_the_export_as_bux_encodes(
    trained, tmp_path, tokenizer_json
):
    from transformers import AutoTokenizer, GenerationConfig

    setting = (
        f'tokenizer(name="{tokenizer_json}", bos="<|im_start|>", '
        'eos="<|im_end|>", added=["<|tool|>", "<|call|>"])'
    )
    folder, _ = trained(ARCH, tokenizer=setting)
    out = runs.export(folder, tmp_path / "out").out
    theirs: Any = AutoTokenizer.from_pretrained(out)
    config = runs.read_run(folder).train.tokenizer
    assert config is not None
    ours = load(config)
    texts = ["Hello, world", "données 123", ""]
    ids, offsets = ours.encode(pa.array(texts))
    for i, text in enumerate(texts):
        want = ids[offsets[i] : offsets[i + 1]].tolist()
        assert theirs(text, add_special_tokens=False).input_ids == want
    # Special tokens in text are text to bux's encoder, and markup to both.
    template = "<|im_start|>user\nhi<|im_end|><|tool|><|call|>"
    want = ours.markup(template).tolist()
    assert theirs(template, add_special_tokens=False).input_ids == want
    assert want[0] == 257 and want[-3:] == [258, 259, 260]
    generation = GenerationConfig.from_pretrained(out)
    assert (generation.bos_token_id, generation.eos_token_id) == (257, 258)
    assert generation.pad_token_id == theirs.pad_token_id == 258
