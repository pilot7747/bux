"""Measure what serving a trained timeline run in FP8 costs in quality: its bits per
byte on the timeline's ``code`` and ``repo`` evals (repo by position too), served by
vLLM in bfloat16, with FP8 weights, and with an FP8 KV cache besides, each against
bfloat16 and against what the run itself logged.

It runs in three steps once the run has finished training (``wait``), as
`scripts/vllm_fp8.py` does (bux and vLLM do not share an environment):

    # either: wait until the run has finished training
    python scripts/timeline/fp8_quality.py wait RUN_DIR [--every 120] [--timeout S]
    # bux's environment, on the CPU: the run's export in bfloat16, and its eval sets
    JAX_PLATFORMS=cpu python scripts/timeline/fp8_quality.py prepare RUN_DIR OUT \\
        --data-root ROOT [--cache DIR ...] [--step N] [--pad 128]
    # vLLM's, with the plugin (pip install src/bux_vllm): a run per configuration
    python scripts/timeline/fp8_quality.py run OUT
    python scripts/timeline/fp8_quality.py run OUT --quantization fp8_per_block \\
        --linear-backend cutlass [--kv-cache-dtype fp8]
    # either: the table, of one run or several
    python scripts/timeline/fp8_quality.py report OUT [OUT ...] [--json FILE]

``prepare`` reads the eval sets as bux's own eval does (`bux.train.hooks.Eval`): the
first ``batches`` batches of the eval's held-out stream, of its ``batch`` rows of
its ``seq_len``, from the prepared dataset the run read (`bux.train.data.batches`).
It looks for that dataset in each ``--cache`` and in ``ROOT/data/packed``, and
prepares it again from the Parquet files under ``ROOT`` if none has it. A row packs
documents and pieces of documents, each attending only to itself from position 0,
so each is a prompt of its own: ``run`` asks vLLM for every token's log-probability
given the tokens of its document before it (``prompt_logprobs``), with prefix
caching off. ``report`` turns them into bits per byte as bux does
(`bux.train.objective.held_out`): the bits of the targets that stand for text over
their bytes, each id's UTF-8 bytes (`bux.data.tokenizer.Tokenizer.token_bytes`); a
target is any token of a document but its first, and the end-of-text token stands
for no text. The repo eval's buckets go by a target's position in its document.

``--pad 128`` pads every MLP's hidden units with zeros to a multiple of 128
(`padded`), which changes no output: vLLM's ``fp8_per_block`` quantizes a matrix
product's input in groups of 128 of its inner dimension, and refuses the timeline's
SwiGLU widths otherwise. ``--linear-backend`` picks the kernels of the scheme's FP8
matrix products (vLLM's ``kernel_config.linear_backend_per_quant``): on a B300,
``fp8_per_block``'s default fails as the engine starts, and ``cutlass`` or
``triton`` serve it.

``run``'s tag defaults to its configuration: ``bf16``, ``fp8``, ``fp8+kv``,
``fp8_per_block-cutlass``, ``fp8_per_block-cutlass+kv``. ``report`` compares every tag of an ``OUT``
with ``bf16``: the change of bits per byte, relative and absolute, with twice its
standard error over the documents; and bfloat16's against the run's logged eval at
the checkpoint's step, which agree to about 0.1%.
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SETS = ("code", "repo")  # the evals measured, by name
BASELINE = "bf16"
SCHEMES = ("none", "fp8", "fp8_per_tensor", "fp8_per_channel", "fp8_per_block")
#: The kind of FP8 matrix product each scheme runs, as vLLM keys its kernels' backends
#: (``kernel_config.linear_backend_per_quant``): blocks of 128, or not.
GEMMS = {"fp8_per_block": "fp8_block_w8a8"} | dict.fromkeys(SCHEMES[1:4], "fp8_w8a8")
CHUNK = 1 << 20  # the prompts' tokens in one call of vLLM's generate
FLAG = 0.3  # a change of bits per byte, in %, that needs a quality correction
CHECK = 0.1  # how far, in %, bfloat16 may be from the run's logged eval


def main() -> None:
    """Parse the command line and run its step."""
    args = parser().parse_args()
    match args.command:
        case "prepare":
            caches = [*args.cache, args.data_root / "data" / "packed"]
            prepare(args.run_dir, args.out, args.data_root, caches, args)
        case "run":
            sys.exit(0 if run(args) else 1)
        case "report":
            sys.exit(0 if report(args.outs, args.baseline, args.json) else 1)
        case "wait":
            sys.exit(wait(args.run_dir, args.every, args.timeout))


def parser() -> argparse.ArgumentParser:
    """The command line's parser: a subcommand a step."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("prepare", help="export the run, write its eval sets")
    s.add_argument("run_dir", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--data-root", type=Path, default=Path("."))
    s.add_argument("--cache", type=Path, action="append", default=[])
    s.add_argument("--step", type=int, default=None)
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--pad", type=int, default=0)
    s = sub.add_parser("run", help="score the eval sets in vLLM")
    s.add_argument("out", type=Path)
    s.add_argument("--quantization", choices=SCHEMES, default="none")
    s.add_argument("--kv-cache-dtype", default=None)
    s.add_argument("--linear-backend", default=None)
    s.add_argument("--tag", default=None)
    s.add_argument("--tp", type=int, default=1)
    s.add_argument("--eager", action="store_true")
    s.add_argument("--moe-backend", default="triton")
    s.add_argument("--mamba-ssm-cache-dtype", default="auto")
    s.add_argument("--attention-backend", default=None)
    s.add_argument("--max-num-batched-tokens", type=int, default=16384)
    s.add_argument("--max-model-len", type=int, default=None)
    s.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    s = sub.add_parser("report", help="bits per byte, and their changes")
    s.add_argument("outs", type=Path, nargs="+")
    s.add_argument("--baseline", default=BASELINE)
    s.add_argument("--json", type=Path, default=None)
    s = sub.add_parser("wait", help="wait until a run has finished training")
    s.add_argument("run_dir", type=Path)
    s.add_argument("--every", type=float, default=120.0)
    s.add_argument("--timeout", type=float, default=None)
    return p


# --- prepare, in bux's environment ----------------------------------------------------


def prepare(
    folder: Path, out: Path, root: Path, caches: list[Path], args: argparse.Namespace
) -> None:
    """Export the run in `folder` in bfloat16 to ``out/model``, and write its eval sets
    (`write_set`), its tokens' bytes and what it logged to `out`."""
    from bux.export.runs import read_run
    from bux.train import config as tcfg
    from bux.train.trainer import _token_bytes

    run = read_run(folder)
    evals = {e.name: e for e in tcfg.evals(run.train) if e.name in SETS}
    if missing := [n for n in SETS if n not in evals]:
        raise SystemExit(f"{folder}: the run has no eval named {', '.join(missing)}")
    out.mkdir(parents=True, exist_ok=True)
    sets = {
        n: write_set(run, e, out, root, caches, args.workers) for n, e in evals.items()
    }
    token_bytes = _token_bytes(run.train, run.arch.vocab_size)
    assert token_bytes is not None, "a run of text has a tokenizer"
    np.save(out / "token_bytes.npy", token_bytes)
    start = time.perf_counter()
    step, widths = export_model(folder, out / "model", args.step, args.pad)
    print(f"exported step {step} in {time.perf_counter() - start:.0f}s")
    meta = {
        "run": run.name,
        "id": run.id,
        "seed": run.train.seed,
        "folder": str(folder.resolve()),
        "step": step,
        "steps": run.train.steps,
        "padded": {str(n): m for n, m in widths.items()},
        "sets": sets,
        "logged": logged(folder, step, evals.values()),
    }
    (out / "prepared.json").write_text(json.dumps(meta, indent=1))
    print(f"prepared {run.name} at step {meta['step']} in {out}")


def export_model(
    folder: Path, out: Path, step: int | None, pad: int
) -> tuple[int, dict[int, int]]:
    """Export the run in `folder` from its checkpoint of `step` to `out` in bfloat16,
    as ``bux export`` does, every MLP's hidden units padded with zeros to a multiple
    of `pad` (`padded`) unless it is 0; return the step and the widths padded."""
    import torch

    from bux.export import huggingface, tokenizer
    from bux.export.runs import _longest, export, restore

    if not pad:
        return export(folder, out, step=step, dtype="bfloat16").restored.step, {}
    restored = restore(folder, step)
    run, (arch, widths) = restored.run, padded(restored.arch, pad)
    model = padded_model(arch, restored.model)
    length = _longest(run.train, restored.step)
    huggingface.save(arch, model, out, dtype=torch.bfloat16, max_positions=length)
    if run.train.tokenizer is not None:
        assert run.arch.vocab_size is not None, "a run has a vocabulary"
        tokenizer.save(run.train.tokenizer, out, run.arch.vocab_size)
    return restored.step, widths


def padded(arch: Any, pad: int) -> tuple[Any, dict[int, int]]:
    """Return `arch` with each MLP's hidden units (``intermediate_size``, an
    expert's too) rounded up to a multiple of `pad`, and the widths it changed.

    vLLM's ``fp8_per_block`` quantizes a matrix product's input in groups of 128 of
    its inner dimension, which must divide by 128; it pads the routed experts so,
    and not the dense layers or the shared experts. A hidden unit of zero weights
    adds nothing, so the padded model computes what `arch`'s does."""
    widths: dict[int, int] = {}

    def walk(x: Any) -> Any:
        if isinstance(x, list):
            return [walk(v) for v in x]
        if not isinstance(x, dict):
            return x
        n = x.get("intermediate_size")
        if x.get("type") in ("mlp", "gated_mlp") and isinstance(n, int):
            widths[n] = -(-n // pad) * pad
            x = x | {"intermediate_size": widths[n]}
        return {k: walk(v) for k, v in x.items()}

    out = type(arch).model_validate(walk(arch.model_dump(mode="json")))
    return out, {n: m for n, m in widths.items() if n != m}


def padded_model(arch: Any, model: Any) -> Any:
    """Return the model of `arch` (`padded`) with `model`'s state, each variable
    padded with zeros at the end of each dimension `arch` widens."""
    import jax.numpy as jnp
    from flax import nnx

    from bux.architecture import Model

    new = nnx.eval_shape(lambda: Model(arch, rngs=nnx.Rngs(0)))
    old = dict(nnx.iter_graph(model))
    for path, var in nnx.iter_graph(new):
        if not isinstance(var, nnx.Variable):
            continue
        value = old[path].get_value()
        if value.shape != var.shape:
            widen = [(0, n - o) for o, n in zip(value.shape, var.shape, strict=True)]
            value = jnp.pad(value, widen)
        var.set_value(value)
    new.eval()
    return new


def write_set(
    run: Any, e: Any, out: Path, root: Path, caches: list[Path], workers: int | None
) -> dict:
    """Write eval `e`'s documents as bux's eval reads them to ``out/sets/<name>.npz``:
    their tokens end to end, where each starts, and which tokens are targets; return
    what it is."""
    from bux.train import config as tcfg

    shape = tcfg.evaluated(run.train, e)
    data, cache, where = located(e.data, shape, root, caches, out / "packed", workers)
    rows = eval_batches(data, shape, run.arch.vocab_size, cache, e.batches)
    parts = [documents(b) for b in rows]
    tokens = np.concatenate([p[0] for p in parts])
    lengths = np.concatenate([p[1] for p in parts])
    targets = np.concatenate([p[2] for p in parts])
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    (out / "sets").mkdir(exist_ok=True)
    np.savez(
        out / "sets" / f"{e.name}.npz", tokens=tokens, offsets=offsets, targets=targets
    )
    print(
        f"{e.name}: {len(lengths):,} documents, {int(targets.sum()):,} targets; {where}"
    )
    return {
        "seq_len": shape.seq_len,
        "batch": shape.batch,
        "batches": e.batches,
        "positions": list(e.positions),
        "buckets": tcfg.buckets(e.positions) if e.positions else [],
        "documents": len(lengths),
        "targets": int(targets.sum()),
        "read": where,
    }


def located(
    data: Any,
    shape: Any,
    root: Path,
    caches: list[Path],
    fresh: Path,
    workers: int | None,
) -> tuple[Any, Path, str]:
    """Return the eval's data, the cache that holds it prepared, and where it came
    from: the first of `caches` with the run's own prepared datasets, or `fresh`,
    where it prepares them from the files under `root`.

    Raises:
        SystemExit: No cache has them, and no file under `root` matches a source.
    """
    from bux.data.pipeline import Job
    from bux.data.pipeline import prepare as prepare_job

    names = [Job.of(s, shape).name for s in documents_of(data)]
    for cache in caches:
        if all((cache / n / "dataset.json").is_file() for n in names):
            return data, cache, f"read {', '.join(names)} in {cache}, as the run did"
    rooted = under(data, root)
    for s in documents_of(rooted):
        if not glob.glob(s.path):
            raise SystemExit(
                f"no cache of {[str(c) for c in caches]} has {', '.join(names)}, "
                f"and no file matches {s.path}: give --cache or --data-root"
            )
    for s in documents_of(rooted):
        prepare_job(Job.of(s, shape), fresh, workers=workers)
    return rooted, fresh, f"prepared again from {root.resolve()} into {fresh}"


def documents_of(data: Any) -> list:
    """The sources of `data`, each of documents.

    Raises:
        SystemExit: A source is of tokens or synthetic, with no documents to serve.
    """
    from bux.train import config as tcfg

    found = [s for s in tcfg.leaves(data) if isinstance(s, tcfg.DOCUMENTS)]
    if len(found) < len(tcfg.leaves(data)):
        raise SystemExit("an eval of tokens or synthetic data has no documents")
    return found


def under(data: Any, root: Path) -> Any:
    """Return the data config `data` with each source's relative path under `root`."""
    from bux.train import config as tcfg

    if isinstance(data, tcfg.MixtureConfig):
        parts = tuple(
            w.model_copy(update={"source": under(w.source, root)}) for w in data.sources
        )
        return data.model_copy(update={"sources": parts})
    if not isinstance(data, tcfg.DOCUMENTS):
        raise SystemExit(f"an eval of {data.type} data has no documents to serve")
    path = Path(data.path)
    return data.model_copy(
        update={"path": str(path if path.is_absolute() else root / path)}
    )


def eval_batches(data: Any, shape: Any, vocab: int, cache: Path, n: int) -> list:
    """The first `n` held-out batches of `data` in rows of `shape`, every row of each:
    what bux's eval reads."""
    import itertools

    from bux.train.data import batches

    rows = np.arange(shape.batch)
    stream = batches(data, shape, vocab, cache=cache, held_out=True, rows=rows)
    return list(itertools.islice(stream, n))


def documents(batch: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The documents of `batch` that have a target, as bux finds them
    (`bux.architecture.common.Segments`): one starts at each row's first token and
    wherever the positions stop increasing. Return their tokens end to end, their
    lengths, and whether each token is a target (`bux.train.objective.target_mask`):
    in the loss mask and not its document's first.

    Raises:
        ValueError: A document's positions are not 0, 1, 2, ..., which serving it
            alone would assume.
    """
    ids, pos, mask = (np.asarray(x) for x in batch)
    b, t = ids.shape
    start = np.ones((b, t), bool)
    start[:, 1:] = pos[:, 1:] <= pos[:, :-1]
    index = np.broadcast_to(np.arange(t), (b, t))
    within = index - np.maximum.accumulate(np.where(start, index, 0), axis=1)
    target = (mask & (within >= 1)).ravel()
    doc = np.cumsum(start.ravel()) - 1
    has = np.bincount(doc, weights=target, minlength=doc[-1] + 1) > 0
    keep = has[doc]
    if (pos.ravel()[keep] != within.ravel()[keep]).any():
        raise ValueError("a document's positions do not count from 0 by 1")
    lengths = np.bincount(doc, minlength=doc[-1] + 1)[has]
    return ids.ravel()[keep].astype(np.int32), lengths, target[keep]


def logged(folder: Path, step: int, evals: Any) -> dict:
    """What the run logged of the evals at `step`, from its metrics.jsonl: each
    eval's loss and bits per byte, and their buckets; or at the last step before it
    that has them, with its step."""
    from bux.train import config as tcfg

    path = folder / "metrics.jsonl"
    if not path.is_file():
        return {"step": None, "metrics": {}, "note": f"no {path}"}
    names = {m for e in evals for m in tcfg.metrics(e, True)}
    found: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["step"] <= step and (
            got := {k: v for k, v in row.items() if k in names}
        ):
            found.setdefault(row["step"], {}).update(got)
    if not found:
        return {"step": None, "metrics": {}, "note": f"no eval in {path}"}
    last = max(found)
    return {"step": last, "metrics": found[last]}


# --- run, in vLLM's environment ---------------------------------------------------------


@dataclass
class EvalSet:
    """An eval's documents: their tokens end to end, where each starts (and the
    last ends), and whether each token is a target."""

    tokens: np.ndarray
    offsets: np.ndarray
    targets: np.ndarray

    @classmethod
    def load(cls, out: Path, name: str) -> "EvalSet":
        """Read the set `name` that ``prepare`` wrote to `out`."""
        f = np.load(out / "sets" / f"{name}.npz")
        return cls(f["tokens"], f["offsets"], f["targets"])

    @property
    def spans(self) -> list[tuple[int, int]]:
        """Each document's first token and the one past its last."""
        return list(zip(self.offsets[:-1].tolist(), self.offsets[1:].tolist()))

    def prompts(self) -> list[list[int]]:
        """Each document's tokens."""
        ids = self.tokens.tolist()
        return [ids[a:b] for a, b in self.spans]


def tag_of(args: argparse.Namespace) -> str:
    """The run's tag: ``--tag``, or its configuration's name, such as
    ``fp8_per_block-cutlass+kv``."""
    if args.tag:
        return args.tag
    base = BASELINE if args.quantization == "none" else args.quantization
    if args.linear_backend:
        base += f"-{args.linear_backend}"
    return base + ("+kv" if args.kv_cache_dtype else "")


def run(args: argparse.Namespace) -> bool:
    """Serve ``OUT/model`` as the command line says, score every set's documents,
    and keep their log-probabilities and what vLLM built in ``OUT/runs/<tag>``;
    return whether vLLM built what was asked."""
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # apply_model's
    # A document as long as the export's longest rows leaves no room for the token
    # generate makes, which no forward pass reads: a longer max_model_len makes it.
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    import vllm

    from bux_vllm.served import survey

    if args.linear_backend and args.quantization == "none":
        raise SystemExit("--linear-backend picks FP8 kernels: give --quantization")
    tag, sets = tag_of(args), {n: EvalSet.load(args.out, n) for n in SETS}
    longest = max(int(np.diff(s.offsets).max()) for s in sets.values())
    engine = engine_args(args, args.max_model_len or longest + 1)
    start = time.perf_counter()
    llm = vllm.LLM(model=str(args.out / "model"), **engine)
    scores, seconds = {}, {"engine": time.perf_counter() - start}
    for name, s in sets.items():
        start = time.perf_counter()
        scores[name] = score(llm, s)
        seconds[name] = time.perf_counter() - start
        print(f"{tag} {name}: {len(s.tokens):,} tokens in {seconds[name]:.0f}s")
    built = llm.apply_model(survey)[0] | {"kv_cache_dtype": args.kv_cache_dtype}
    info = {"tag": tag, "engine": engine, "vllm": vllm.__version__}
    info |= {"built": built, "seconds": seconds}
    save_scores(args.out, tag, scores, info)
    return as_asked(built, args.quantization != "none", args.kv_cache_dtype)


def engine_args(args: argparse.Namespace, max_len: int) -> dict:
    """vLLM's engine arguments: the command line's, prefix caching off.
    ``--linear-backend`` picks the backend of the scheme's FP8 matrix products alone
    (`GEMMS`), not of those kept in bfloat16."""
    kernels = {}
    if args.linear_backend:
        gemm = GEMMS[args.quantization]
        kernels["kernel_config"] = {
            "linear_backend_per_quant": {gemm: args.linear_backend}
        }
    return kernels | {
        "dtype": "bfloat16",
        "quantization": None if args.quantization == "none" else args.quantization,
        "kv_cache_dtype": args.kv_cache_dtype or "auto",
        "tensor_parallel_size": args.tp,
        "enforce_eager": args.eager,
        "moe_backend": args.moe_backend,
        "mamba_ssm_cache_dtype": args.mamba_ssm_cache_dtype,
        "attention_backend": args.attention_backend,
        "max_model_len": max_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "skip_tokenizer_init": True,
        "seed": 0,
    }


def score(llm: Any, s: EvalSet) -> np.ndarray:
    """Each token's log-probability given the tokens of its document before it, by
    `llm`; NaN at each document's first."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(
        max_tokens=1, temperature=0.0, prompt_logprobs=0, detokenize=False
    )
    out = np.full(len(s.tokens), np.nan, np.float32)
    prompts, spans = s.prompts(), s.spans
    for group in chunks(spans):
        asked = [TokensPrompt(prompt_token_ids=prompts[i]) for i in group]
        for i, got in zip(group, llm.generate(asked, params, use_tqdm=False)):
            a, b = spans[i]
            rows, ids = got.prompt_logprobs, prompts[i]
            out[a + 1 : b] = [rows[j][ids[j]].logprob for j in range(1, b - a)]
    return out


def chunks(spans: list[tuple[int, int]]) -> list[list[int]]:
    """The documents of `spans` in groups of about `CHUNK` tokens, in order."""
    groups, size = [[]], 0
    for i, (a, b) in enumerate(spans):
        if groups[-1] and size + b - a > CHUNK:
            groups.append([])
            size = 0
        groups[-1].append(i)
        size += b - a
    return groups


def save_scores(out: Path, tag: str, scores: dict[str, Any], info: dict) -> None:
    """Keep a run's log-probabilities by set in ``out/runs/<tag>.npz``, and `info`,
    what served them, in ``<tag>.json``."""
    (out / "runs").mkdir(exist_ok=True)
    np.savez(out / "runs" / f"{tag}.npz", **scores)
    (out / "runs" / f"{tag}.json").write_text(json.dumps(info, indent=1))


def as_asked(built: dict, quantized: bool, kv: str | None) -> bool:
    """Whether vLLM built what was asked (`bux_vllm.served.survey`'s `built`): some
    layers quantized and no head or embedding if `quantized`, else none, and an
    attention layer's KV cache in `kv`; say what it lacks."""
    some = [k for k in built["methods"] if "Unquantized" not in k.split(": ")[-1]]
    heads = [k for k in some if "Embedding" in k or "LMHead" in k]
    ok = True
    if quantized != bool(some) or heads:
        print(f"quantized: {some or 'nothing'}; heads among them: {heads}")
        ok = False
    if kv is not None and not any(f": kv {kv} " in k for k in built["caches"]):
        print(f"no attention layer kept a {kv} KV cache: {built['caches']}")
        ok = False
    return ok


# --- wait, in either -----------------------------------------------------------------


def wait(folder: Path, every: float, timeout: float | None) -> int:
    """Wait until the run in `folder` has finished training (`finished`), looking
    every `every` seconds; print a line, and return 0, 1 if it failed, or 2 if it is
    not done after `timeout` seconds."""
    start = time.monotonic()
    while (state := finished(folder))[0] is None:
        if timeout is not None and time.monotonic() - start >= timeout:
            print(f"{folder.name}: not done after {timeout:g}s: {state[1]}")
            return 2
        time.sleep(every)
    print(state[1])
    return 0 if state[0] == "done" else 1


def finished(folder: Path) -> tuple[str | None, str]:
    """Whether the run in `folder` has finished training: ``done`` once its
    status.json says so (`bux.launch.status`), or once its last step's evals are in
    metrics.jsonl and its checkpoint of that step is committed, as `bux train`
    leaves it; ``failed`` where status.json says so; None, and where it is, until
    then. A run stopped by a signal is not done: its job may resume it."""
    if not (folder / "run.json").is_file():
        return None, "no run.json yet"
    train = json.loads((folder / "run.json").read_text(encoding="utf-8"))["train"]
    steps, name = train["steps"], folder.name
    status = folder / "status.json"
    state = json.loads(status.read_text(encoding="utf-8")) if status.is_file() else {}
    if state.get("state") == "failed":
        return (
            "failed",
            f"{name}: failed at step {state.get('step')}: {state.get('error')}",
        )
    evals = _final_evals(folder / "metrics.jsonl", steps, _eval_names(train["eval"]))
    saved = _committed(folder / "checkpoints" / str(steps))
    if state.get("state") == "done" or (evals is not None and saved):
        shown = ", ".join(
            f"{k} {v:.4f}" for k, v in (evals or {}).items() if "_bpb" in k
        )
        return "done", f"{name}: done at step {steps}: {shown}"
    where = state.get("state", "no status.json")
    return (
        None,
        f"{where}, final evals {'in' if evals else 'not in'}, checkpoint {steps} {'saved' if saved else 'not saved'}",
    )


def _eval_names(evals: Any) -> list[str]:
    """The names of a run's evals, from run.json's ``train.eval``."""
    evals = [] if evals is None else evals if isinstance(evals, list) else [evals]
    return [e["name"] for e in evals]


def _final_evals(path: Path, step: int, names: list[str]) -> dict | None:
    """The metrics logged at `step` if every eval of `names` is among them (its
    ``<name>_loss``), else None."""
    if not path.is_file():
        return None
    head, got = f'{{"step": {step},', {}
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if line.startswith(head) and line.endswith("\n"):
                got |= json.loads(line)
    return got if all(f"{n}_loss" in got for n in names) else None


def _committed(checkpoint: Path) -> bool:
    """Whether orbax has committed the checkpoint at `checkpoint`."""
    if (checkpoint / "commit_success.txt").is_file():
        return True
    meta = checkpoint / "_CHECKPOINT_METADATA"
    return meta.is_file() and bool(
        json.loads(meta.read_text()).get("commit_timestamp_nsecs")
    )


# --- report, in either --------------------------------------------------------------------


@dataclass
class Sums:
    """A set's bits and bytes (or nats and targets) by document, over one kind of
    target: of text, or all of them."""

    bits: np.ndarray
    bytes: np.ndarray

    @property
    def value(self) -> float:
        """Their ratio: bits per byte (or nats per target)."""
        return float(self.bits.sum() / self.bytes.sum())


def sums(
    s: EvalSet, lp: np.ndarray, token_bytes: np.ndarray, positions: list, names: list
) -> dict:
    """The set's sums by metric as bux names them (``bpb``, ``loss``, ``bpb_2k``, ...)
    from the log-probabilities `lp` of its tokens: bits per byte over the targets of
    text, the mean loss over every target, and each by bucket of position: from 0
    and from each of `positions` on, by `names`.

    Raises:
        ValueError: A target has no log-probability.
    """
    if np.isnan(lp[s.targets]).any():
        raise ValueError(f"{int(np.isnan(lp[s.targets]).sum())} targets unscored")
    lengths = np.diff(s.offsets)
    doc = np.repeat(np.arange(len(lengths)), lengths)
    within = np.arange(len(s.tokens)) - np.repeat(s.offsets[:-1], lengths)
    nbytes = np.where(s.targets, token_bytes[s.tokens], 0)
    kinds = {
        "bpb": (nbytes > 0, nbytes, 1 / math.log(2)),
        "loss": (s.targets, s.targets, 1.0),
    }
    bucket = np.searchsorted(np.asarray(positions, np.int64), within, "right")
    out = {}
    for kind, (sel, count, unit) in kinds.items():
        nats = np.where(sel, -lp.astype(np.float64), 0) * unit
        out[kind] = _by_doc(doc, nats, count, len(lengths))
        for i, name in enumerate(names):
            inside = bucket == i
            out[f"{kind}_{name}"] = _by_doc(
                doc, nats * inside, count * inside, len(lengths)
            )
    return out


def _by_doc(doc: np.ndarray, bits: np.ndarray, count: np.ndarray, n: int) -> Sums:
    """`bits` and `count` summed by document."""
    return Sums(
        np.bincount(doc, bits, n), np.bincount(doc, count.astype(np.float64), n)
    )


def change(tag: Sums, base: Sums) -> dict[str, float]:
    """How `tag`'s ratio differs from `base`'s, on the same documents: absolute, and
    relative in %, each with its standard error over the documents."""
    d, b = tag.bits - base.bits, base.bytes
    n = int((b > 0).sum())
    delta = float(d.sum() / b.sum())
    se = float(np.sqrt(((d - delta * b) ** 2).sum() * n / max(n - 1, 1)) / b.sum())
    ref = base.value
    return {
        "abs": delta,
        "abs_se": se,
        "pct": 100 * delta / ref,
        "pct_se": 100 * se / ref,
    }


def measured(out: Path) -> tuple[dict, dict[str, dict[str, Sums]], dict]:
    """What ``prepare`` wrote to `out`, each run's sums by set and metric, and each
    run's info."""
    meta = json.loads((out / "prepared.json").read_text())
    token_bytes = np.load(out / "token_bytes.npy")
    sets = {n: EvalSet.load(out, n) for n in meta["sets"]}
    found, infos = {}, {}
    for info_file in sorted((out / "runs").glob("*.json")):
        tag = info_file.stem
        scores = np.load(out / "runs" / f"{tag}.npz")
        found[tag] = {}
        for n, s in sets.items():
            info = meta["sets"][n]
            got = sums(s, scores[n], token_bytes, info["positions"], info["buckets"])
            found[tag] |= {f"{n}_{k}": v for k, v in got.items()}
        infos[tag] = json.loads(info_file.read_text())
    return meta, found, infos


def summary(out: Path, baseline: str) -> dict:
    """The run at `out`: each tag's metrics, their changes from `baseline`'s, and the
    baseline's against the run's logged eval."""
    meta, found, infos = measured(out)
    if baseline not in found:
        raise SystemExit(f"{out}: no {baseline} run to compare with")
    base = found[baseline]
    tags = {}
    for tag, got in found.items():
        tags[tag] = {
            "metrics": {k: v.value for k, v in got.items()},
            "change": {k: change(v, base[k]) for k, v in got.items()}
            if tag != baseline
            else {},
            "engine": infos[tag]["engine"],
            "built": infos[tag]["built"],
            "seconds": infos[tag]["seconds"],
        }
    logs = meta["logged"]["metrics"]
    check = {
        k: {"served": base[k].value, "logged": v, "pct": 100 * (base[k].value / v - 1)}
        for k, v in logs.items()
        if k in base
    }
    keep = ("run", "id", "seed", "folder", "step")
    return (
        {k: meta[k] for k in keep}
        | {"logged_step": meta["logged"]["step"]}
        | {
            "out": str(out),
            "tags": tags,
            "check": check,
        }
    )


def report(outs: list[Path], baseline: str, path: Path | None) -> bool:
    """Print each run's table and write its ``report.json``, and all of them to
    `path`; return whether bfloat16 agrees with every run's logged eval to 0.1%."""
    done = [summary(out, baseline) for out in outs]
    for one in done:
        (Path(one["out"]) / "report.json").write_text(json.dumps(one, indent=1))
        print("\n".join(lines(one, baseline)))
    if path is not None:
        path.write_text(json.dumps(done, indent=1))
    off = [
        f"{one['run']} {k}"
        for one in done
        for k, c in one["check"].items()
        if abs(c["pct"]) > CHECK
    ]
    off += [f"{one['run']} (no logged eval)" for one in done if not one["check"]]
    print(
        f"\n{baseline} against the logged evals: "
        + (f"off at {off}" if off else f"within {CHECK}%")
    )
    return not off


def lines(one: dict, baseline: str) -> list[str]:
    """A run's table: bits per byte by set and bucket for each tag, their change from
    `baseline`'s, and the baseline against the run's logged eval."""
    tags = [t for t in one["tags"] if t != baseline]
    keys = [k for k in one["tags"][baseline]["metrics"] if "_bpb" in k]
    logged = one["check"]
    out = [f"\n{one['run']}  step {one['step']}  ({one['out']})"]
    out.append(
        f"{'':12s} {baseline:>8s} {'logged':>8s} {'diff':>7s}"
        + "".join(f" {t:>22s}" for t in tags)
    )
    for k in keys:
        row = f"{k:12s} {one['tags'][baseline]['metrics'][k]:8.4f}"
        c = logged.get(k)
        row += (
            f" {c['logged']:8.4f} {c['pct']:+6.2f}%" if c else f" {'-':>8s} {'-':>7s}"
        )
        for t in tags:
            d = one["tags"][t]["change"][k]
            row += f" {d['pct']:+7.3f}% ±{2 * d['pct_se']:.3f} {d['abs']:+.4f}"
        out.append(row)
    for t in tags:
        worst = max(abs(one["tags"][t]["change"][f"{n}_bpb"]["pct"]) for n in SETS)
        if worst >= FLAG:
            out.append(f"{t}: a change of {worst:.2f}% needs a quality correction")
    if not one["check"]:
        out.append(f"no logged eval to check {baseline} against")
    elif one["logged_step"] != one["step"]:
        out.append(
            f"the logged eval is step {one['logged_step']}'s, not the checkpoint's"
        )
    return out


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
