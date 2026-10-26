"""Fixtures: a tokenizer and a corpus that need no network, runs trained on 8
devices, the equations of a jaxpr, a function's outputs and gradients, and what its
backward pass keeps.

First, before anything imports bux, jaxtyping instruments the packages that handle
arrays, so that every call the tests make checks the dtype and shape of every array
into and out of a function, with beartype taking an int where a float goes, as a static
type checker does. (Its --jaxtyping-packages flag takes one package with that checker.)
Then, before anything starts JAX, it gets 8 CPU devices, so that the tests split runs
over them and compare with one, and bux's XLA flags, as `bux train` sets them: the
tests that need a GPU run with them too.
"""

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")  # the command line's, off

import jaxtyping

# Every module of bux.data but spark, whose functions Spark pickles for its workers:
# a checked function's wrapper neither pickles nor shows Spark its arguments.
DATA = [
    f"bux.data.{p.stem}"
    for p in Path(__file__).parents[1].joinpath("src", "bux", "data").glob("*.py")
    if p.stem not in ("__init__", "spark")
]
jaxtyping.install_import_hook(
    ["bux.architecture", *DATA, "bux.parallel", "bux.search", "bux.train"],
    "beartype.beartype(conf=beartype.BeartypeConf(is_pep484_tower=True))",
)

from bux import runtime

os.environ["XLA_FLAGS"] = runtime.xla_flags(
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
)

import itertools
import json
import zlib
from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from flax import nnx
from jax.extend.core import Jaxpr, JaxprEqn, jaxprs_in_params
from jaxtyping import Array, Float, Shaped

from bux import parallel
from bux.architecture.recurrent.linear_attention import States
from bux.lang import compile_runs
from bux.train import config as tcfg
from bux.train.sinks import Sink
from bux.train.trainer import train

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def pytest_collection_modifyitems(config, items) -> None:
    """With BUX_SHARD=i/n, as CI sets it, keep the tests whose id hashes to shard i of n,
    so that n runners share the suite."""
    shard = os.environ.get("BUX_SHARD")
    if not shard:
        return
    i, n = map(int, shard.split("/"))
    mine = [t for t in items if zlib.crc32(t.nodeid.encode()) % n == i]
    config.hook.pytest_deselected(items=[t for t in items if t not in mine])
    items[:] = mine


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


#: SmolLM-135M-Instruct's chat template: ChatML, a message at a time.
CHATML = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + "
    "message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def byte_spec(missing: int | None = None) -> dict:
    """The tokenizer.json of a byte-level BPE without merges: every byte is the token
    of its value, and the special tokens come after them. A `missing` byte has no
    token, its id another's ("aa"), as SmolLM2's vocabulary lacks the byte 0x04."""
    vocab = {c: b for b, c in _byte_alphabet().items()}
    if missing is not None:
        del vocab[_byte_alphabet()[missing]]
        vocab["aa"] = missing
    byte_level = {"add_prefix_space": False, "trim_offsets": True, "use_regex": True}
    bpe = {"dropout": None, "unk_token": None, "fuse_unk": False}
    bpe |= {"continuing_subword_prefix": None, "end_of_word_suffix": None}
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 256 + i, "content": s, "special": True, "normalized": False}
            | {"single_word": False, "lstrip": False, "rstrip": False}
            for i, s in enumerate(SPECIALS)
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel"} | byte_level,
        "post_processor": None,
        "decoder": {"type": "ByteLevel"} | byte_level,
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [],
        }
        | bpe,
    }


@pytest.fixture(scope="session")
def byte_tokenizer(tmp_path_factory) -> tcfg.TokenizerConfig:
    """The byte-level BPE of `byte_spec`, beside a tokenizer_config.json whose chat
    template is ChatML, as a Hub repo has them."""
    folder = tmp_path_factory.mktemp("tokenizer")
    (folder / "tokenizer.json").write_text(json.dumps(byte_spec()))
    added = {256 + i: {"content": s, "special": True} for i, s in enumerate(SPECIALS)}
    config = {"chat_template": CHATML, "added_tokens_decoder": added}
    (folder / "tokenizer_config.json").write_text(json.dumps(config))
    return tcfg.TokenizerConfig(name=str(folder / "tokenizer.json"))


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


def outputs_and_grads(
    f: Callable[..., Float[Array, "..."]], *args: Float[Array, "..."]
) -> list[Float[Array, "..."]]:
    """f's outputs, then the gradients for each of `args` of a weighted sum of them,
    each output weighted by the cosine of its index."""

    def loss(
        *args: Float[Array, "..."],
    ) -> tuple[Float[Array, ""], Float[Array, "..."]]:
        out = f(*args)
        return (out * jnp.cos(jnp.arange(out.size)).reshape(out.shape)).sum(), out

    grads, out = jax.grad(loss, tuple(range(len(args))), has_aux=True)(*args)
    return [out, *grads]


def random_biases[M: nnx.Module](module: M) -> M:
    """Return `module` with every bias at random, small: at their init, 0, they would
    show nothing a test could check."""
    keys = iter(jax.random.split(jax.random.key(7), 64))
    for path, param in nnx.iter_graph(module):
        if isinstance(param, nnx.Param) and path and "bias" in str(path[-1]):
            param[...] = 0.1 * jax.random.normal(next(keys), param[...].shape)
    return module


def kept_bytes(
    f: Callable[..., Float[Array, "..."]], *args: Float[Array, "..."], states: States
) -> int:
    """The bytes f's backward pass keeps, on one device, by `methods(states=...)`."""

    def kept(*args: Float[Array, "..."]) -> list[Shaped[Array, "..."]]:
        return jax.tree.leaves(jax.vjp(f, *args)[1])

    methods = tcfg.MethodsConfig(states=states)
    with parallel.use(tcfg.MeshConfig(data=1), methods=methods):
        shapes = jax.eval_shape(kept, *args)
    return sum(x.size * x.dtype.itemsize for x in shapes)


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
