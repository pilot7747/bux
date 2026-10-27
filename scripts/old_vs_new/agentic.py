"""Time a model in vLLM under an agentic coding load: long prompts, a few turns each,
many sessions at once.

A session is an agent at work on a repository. Its first prompt is a system prompt
that every session shares, then the session's own context (the repository's files);
each turn the model writes ``--output`` tokens, and the turn after reads them back
with a tool's result of ``--tool`` tokens appended. So each turn's prompt starts with
the last one, and vLLM's prefix cache serves it. ``--concurrency`` sessions run at a
time, ``--sessions`` times as many in all: a closed loop, a new session starting as
one ends. Each concurrency in a list runs in turn, from an empty prefix cache.

Tokens are random ids: the weights are too (``load_format="dummy"``), so only the
time counts. It prints a JSON line of the totals. The tutorial "Old against new at
the same serving cost" (docs/tutorials/old-vs-new.md) runs it on exports of
``bux export FILE.bux OUT``, in vLLM's own environment (no bux, no JAX):

    python scripts/old_vs_new/agentic.py exported/old-4096 --concurrency 8,16,32,64
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field

from vllm import AsyncEngineArgs, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


@dataclass
class Turn:
    """What one turn of a session took."""

    prompt: int  # tokens of its prompt
    cached: int  # of them, those the prefix cache served
    ttft: float  # seconds to its first token
    seconds: float  # seconds to its last token
    output: int  # tokens it wrote


@dataclass
class Totals:
    """Every turn of the run, and its wall-clock seconds."""

    turns: list[Turn] = field(default_factory=list)
    seconds: float = 0.0


def main() -> None:
    a = parser().parse_args()
    asyncio.run(run(a))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model", help="an export's directory")
    p.add_argument(
        "--concurrency",
        type=lambda v: [int(c) for c in v.split(",")],
        default=[16],
        help="sessions at once; a list, 8,16,32, times each in turn",
    )
    p.add_argument(
        "--sessions", type=int, default=2, help="sessions in all, a multiple of each"
    )
    p.add_argument("--system", type=int, default=2048, help="shared prompt tokens")
    p.add_argument("--context", type=int, default=14336, help="a session's own tokens")
    p.add_argument("--turns", type=int, default=4)
    p.add_argument("--output", type=int, default=512, help="tokens written a turn")
    p.add_argument("--tool", type=int, default=2048, help="tokens a tool adds a turn")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="a name for the JSON line")
    p.add_argument(
        "--engine",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="another of vLLM's engine arguments, a JSON value",
    )
    return p


async def run(a: argparse.Namespace) -> None:
    extra = {k: json.loads(v) for k, v in (e.split("=", 1) for e in a.engine)}
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=a.model,
            load_format="dummy",
            dtype="bfloat16",
            skip_tokenizer_init=True,
            max_model_len=a.max_model_len,
            enable_prefix_caching=True,
            gpu_memory_utilization=a.gpu_memory_utilization,
            max_num_seqs=max(a.concurrency),
            max_num_batched_tokens=a.max_num_batched_tokens,
            tensor_parallel_size=a.tp,
            seed=a.seed,
            **extra,
        )
    )
    vocab = engine.model_config.get_vocab_size()
    system = _tokens(random.Random(a.seed), vocab, a.system)
    await _session(engine, a, system, random.Random(-1), vocab, "warmup", Totals(), 1)
    for c in a.concurrency:
        await engine.reset_prefix_cache()
        totals = await _load(engine, a, system, vocab, c)
        print(json.dumps(_summary(a, engine, totals, c)), flush=True)
    engine.shutdown()


async def _load(
    engine: AsyncLLM, a: argparse.Namespace, system: list[int], vocab: int, c: int
) -> Totals:
    """Run `a.sessions` times `c` sessions, `c` at a time, each of its own tokens."""
    totals, todo = Totals(), list(range(a.sessions * c))

    async def worker() -> None:
        while todo:
            i = todo.pop(0)
            rng = random.Random(f"{a.seed}-{c}-{i}")
            await _session(engine, a, system, rng, vocab, f"c{c}-s{i}", totals)

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(c)))
    totals.seconds = time.perf_counter() - start
    return totals


def _tokens(rng: random.Random, vocab: int, n: int) -> list[int]:
    return [rng.randrange(1000, vocab - 1000) for _ in range(n)]


async def _session(
    engine: AsyncLLM,
    a: argparse.Namespace,
    system: list[int],
    rng: random.Random,
    vocab: int,
    name: str,
    totals: Totals,
    turns: int | None = None,
) -> None:
    """Run one session's turns, each prompt the last one, its output and a tool's."""
    prompt = system + _tokens(rng, vocab, a.context)
    params = SamplingParams(
        max_tokens=a.output,
        ignore_eos=True,
        temperature=0.0,
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )
    for t in range(turns or a.turns):
        start, first, out, cached = time.perf_counter(), None, [], 0
        request = TokensPrompt(prompt_token_ids=prompt)
        async for o in engine.generate(request, params, request_id=f"{name}-{t}"):
            first = first or time.perf_counter()
            out += o.outputs[0].token_ids
            cached = o.num_cached_tokens or cached
        now = time.perf_counter()
        ttft = (first or now) - start
        totals.turns.append(Turn(len(prompt), cached, ttft, now - start, len(out)))
        prompt = prompt + out + _tokens(rng, vocab, a.tool)


def _summary(a: argparse.Namespace, engine: AsyncLLM, totals: Totals, c: int) -> dict:
    turns, s = totals.turns, totals.seconds
    prompts = sum(t.prompt for t in turns)
    cached = sum(t.cached for t in turns)
    decode = [(t.seconds - t.ttft) / max(t.output - 1, 1) for t in turns]
    return {
        "tag": a.tag,
        "model": a.model,
        "concurrency": c,
        "sessions": a.sessions * c,
        "seconds": round(s, 2),
        "sessions_per_hour": round(a.sessions * c / s * 3600, 1),
        "output_tok_s": round(sum(t.output for t in turns) / s, 1),
        "prompt_tok_s": round(prompts / s, 1),
        "prefilled_tok_s": round((prompts - cached) / s, 1),
        "cache_hit": round(cached / prompts, 4),
        "ttft_p50": round(statistics.median(t.ttft for t in turns), 3),
        "ttft_p90": round(statistics.quantiles([t.ttft for t in turns], n=10)[-1], 3),
        "tpot_p50_ms": round(statistics.median(decode) * 1e3, 2),
        "kv_blocks": engine.vllm_config.cache_config.num_gpu_blocks,
        "block_size": engine.vllm_config.cache_config.block_size,
    }


if __name__ == "__main__":
    main()
