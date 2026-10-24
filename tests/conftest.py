"""Fixtures: a tokenizer and a corpus that need no network, runs trained on 8
devices, and the equations of a jaxpr.

First, before anything imports bux, jaxtyping instruments the packages that handle
arrays, so that every call the tests make checks the dtype and shape of every array
into and out of a function, with beartype taking an int where a float goes, as a static
type checker does. (Its --jaxtyping-packages flag takes one package with that checker.)
Then, before anything starts JAX, it gets 8 CPU devices, so that the tests split runs
over them and compare with one, and bux's XLA flags, as `bux train` sets them: the
tests that need a GPU run with them too.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")  # the command line's, off

import jaxtyping

jaxtyping.install_import_hook(
    ["bux.architecture", "bux.data", "bux.parallel", "bux.train"],
    "beartype.beartype(conf=beartype.BeartypeConf(is_pep484_tower=True))",
)

from bux import runtime

os.environ["XLA_FLAGS"] = runtime.xla_flags(
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
)

import itertools
import json
from collections.abc import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jax.extend.core import Jaxpr, JaxprEqn, jaxprs_in_params

from bux.lang import compile_runs
from bux.train import config as tcfg
from bux.train.sinks import Sink
from bux.train.trainer import train

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def _byte_alphabet() -> dict[int, str]:
    """GPT-2's byte-level alphabet: a printable character for every byte."""
    printable = [*range(33, 127), *range(161, 173), *range(174, 256)]
    rest = [b for b in range(256) if b not in printable]
    return {
        b: chr(c)
        for b, c in zip(
            printable + rest, printable + [256 + i for i in range(len(rest))]
        )
    }


@pytest.fixture(scope="session")
def byte_tokenizer(tmp_path_factory) -> tcfg.TokenizerConfig:
    """A byte-level BPE without merges: every byte is the token of its value, and the
    special tokens come after them."""
    spec = {
        "version": "1.0",
        "added_tokens": [
            {"id": 256 + i, "content": s, "special": True, "normalized": False}
            | {"single_word": False, "lstrip": False, "rstrip": False}
            for i, s in enumerate(SPECIALS)
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "use_regex": True,
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "vocab": {c: b for b, c in _byte_alphabet().items()},
            "merges": [],
        },
    }
    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    path.write_text(json.dumps(spec))
    return tcfg.TokenizerConfig(name=str(path))


def corpus(folder, n=600) -> str:
    """A glob of two Parquet files of `n` short texts, in row groups of 50."""
    rng = np.random.default_rng(0)
    words = ["data", "pack", "row", "token", "fits", "best", "chunk", "shuffle"]
    texts = [" ".join(rng.choice(words, rng.integers(1, 30))) for _ in range(n)]
    folder.mkdir()
    pq.write_table(
        pa.table({"text": texts[: n // 2]}), folder / "a.parquet", row_group_size=50
    )
    pq.write_table(
        pa.table({"text": texts[n // 2 :]}), folder / "b.parquet", row_group_size=50
    )
    return str(folder / "*.parquet")


def equations(jaxpr: Jaxpr, times: int = 1) -> Iterator[tuple[JaxprEqn, int]]:
    """Every equation of `jaxpr` and of the jaxprs inside it, with how many times it
    runs: a scan's body as many times as the scan's length."""
    for e in jaxpr.eqns:
        yield e, times
        inner = times * e.params["length"] if e.primitive.name == "scan" else times
        for sub in jaxprs_in_params(e.params):
            yield from equations(sub, inner)


def count(jaxpr: Jaxpr, primitive: str) -> int:
    """How many equations of `primitive` `jaxpr` has, in the jaxprs inside it too."""
    return sum(e.primitive.name == primitive for e, _ in equations(jaxpr))


# What `losses` trains with, unless the source or its settings say otherwise.
RUN = {
    "data": "synthetic",
    "optimizer": "adamw(lr=1e-2)",
    "seq_len": "16",
    "batch": "8",
    "log": "log(every=1)",
    # Splits are compared to 1e-5, which bfloat16 rounds past, and XLA:CPU cannot
    # compile the bfloat16 psums of an MoE or a pipeline on more than one device.
    "precision": "precision(compute=float32)",
}


class Losses(Sink):
    def __init__(self):
        self.losses: list[float] = []

    def write(self, step, scalars):
        if "loss" in scalars:
            self.losses.append(scalars["loss"])


@pytest.fixture
def losses(tmp_path):
    """Train the model of `source` for `steps` steps with `settings`, each the source
    of a setting (mesh="mesh(fsdp=8)"), and with `RUN` where they are silent: the
    loss at every step."""
    folders = itertools.count()

    def train_losses(source: str, steps: int = 3, **settings: str) -> list[float]:
        given = RUN | {"steps": str(steps)} | settings
        [run] = compile_runs(source + "".join(f"\n{k} = {v}" for k, v in given.items()))
        sink = Losses()
        train(run, tmp_path / str(next(folders)), echo=lambda _: None, sinks=[sink])
        return sink.losses

    return train_losses


@pytest.fixture
def involuntary(capfd):
    """How many times XLA said it rematerialized an array in full since the last
    call: what a sharding it cannot reach efficiently costs."""

    def count() -> int:
        return capfd.readouterr().err.count("Involuntary full rematerialization")

    return count
