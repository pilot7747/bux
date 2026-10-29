"""Time the parts of a copy pointer's decoding step in the vLLM plugin
(`bux_vllm.pointer`) on a GPU, beside a plain attention decode of as many bytes.

Run it where the plugin and Triton are installed, with a CUDA GPU::

    python scripts/vllm_pointer_bench.py [--requests 8 64] [--keys 16384]

For each count of requests, each with `--keys` keys before its query, a vocabulary
of 50304 and keys of 128 channels in a paged cache of blocks of 16 (in FP8 and in
bfloat16, the blocks shuffled), it prints the microseconds of:

- **step**: the op's work on the GPU: its rows' write, its buffers' zeroing, the
  kernel;
- **kernel**: the Triton kernel alone, its tokens spread over the vocabulary, or
  512 of them, as a text that repeats itself; and its two passes in one program
  a request, which a query past the shift's range takes (`bux_vllm.copying.shift`),
  as the kernel took every query before it read each key once; and within a
  window of 4096;
- **keys**: the kernel's reads of the keys alone: one pass that scores the queries,
  one program a request as the kernel's;
- **scatter**: its adds into the vocabulary's rows alone, one a position, with
  Triton's default ordering (acq_rel) and relaxed;
- **mixture**: the logits' mixture (`bux_vllm.copying.mixed`), outside graphs as
  vLLM runs it, against the head's log-softmax alone;
- **read**: the bytes of the keys read whole, the bandwidth's floor, and vLLM's
  FlashAttention decoding over as many bytes (a kv head of 64, keys and values).

GPU parts are timed as replays of a CUDA graph, as vLLM decodes, by
`triton.testing.do_bench`, which flushes the L2 cache between calls.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from bux_vllm import copying, picks, pointer

VOCAB, DIM, BLOCK, BN = 50304, 128, 16, 64
REPEATS = 512  # the tokens of a text that repeats itself
WINDOW = 4096  # nl_cplm_w4k's


def timed(fn: Callable[[], object], graph: bool = True) -> float:
    """`fn`'s microseconds a call: of replays of a CUDA graph of it, or eager."""
    fn()
    torch.cuda.synchronize()
    run = fn
    if graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        run = g.replay
    return triton.testing.do_bench(run, warmup=25, rep=200) * 1e3


def rms_normed(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True)).to(x.dtype)


def setup(requests: int, keys: int, fp8: bool, distinct: int) -> SimpleNamespace:
    """A step of `requests` queries over `keys` keys each in a paged cache, of
    `distinct` tokens spread over the vocabulary."""
    dev, n = "cuda", requests * keys
    dtype = torch.uint8 if fp8 else torch.bfloat16
    e4m3 = torch.float8_e4m3fn if fp8 else None
    blocks = torch.randperm(n // BLOCK, device=dev) + 1  # block 0 is vLLM's null
    table = blocks.view(requests, keys // BLOCK).int()
    cache = torch.zeros(n // BLOCK + 1, 1, BLOCK, copying.row_width(DIM), dtype=dtype)
    cache = cache.to(dev)
    at = torch.arange(keys, device=dev)
    slots = (table[:, at // BLOCK].long() * BLOCK + at % BLOCK).flatten()
    k = rms_normed(torch.randn(n, DIM, device=dev)).bfloat16()
    tokens = torch.randint(0, distinct, (n,), device=dev) * (VOCAB // distinct)
    picks.write(cache, slots, copying.encode(k, tokens, dtype, e4m3))
    q = rms_normed(torch.randn(requests, DIM, device=dev)).bfloat16()
    return SimpleNamespace(
        requests=requests,
        keys=keys,
        cache=SimpleNamespace(rows=cache, fp8=e4m3, window=copying.WHOLE),
        table=table,
        tokens=tokens.view(requests, keys).int(),
        q=q,
        k=k[:requests],
        sink=q.float() @ torch.randn(DIM, device=dev) * DIM**-0.5,
        position=torch.full((requests,), keys, device=dev, dtype=torch.int32),
        slots=slots.view(requests, keys)[:, -1],  # each request's last, rewritten
        p_copy=torch.zeros(requests, VOCAB, device=dev),
        a_sink=torch.ones(requests, device=dev),
        total=torch.ones(requests, device=dev),
        logits=torch.randn(requests, VOCAB, device=dev, dtype=torch.bfloat16),
        key_bytes=n * DIM * (1 if fp8 else 2),
    )


def kernel(s: SimpleNamespace, gain: float = 1.0) -> None:
    q = s.q * gain
    pointer._pointed(
        q, s.sink, s.cache, s.table, s.position, s.p_copy, s.a_sink, s.total
    )


def step(s: SimpleNamespace) -> None:
    """The op's GPU work for a step of decoded tokens: write, zero, point."""
    rows = copying.encode(s.k, s.tokens[:, -1], s.cache.rows.dtype, s.cache.fp8)
    picks.write(s.cache.rows, s.slots, rows)
    s.p_copy.zero_()
    s.total.zero_()
    kernel(s)


@triton.jit
def _keys_kernel(
    Q,
    KEYS,
    TABLE,
    OUT,
    keys,
    stride_block,
    stride_slot,
    stride_table,
    scale,
    D: tl.constexpr,
    B: tl.constexpr,
    BN: tl.constexpr,
):
    """The pointer kernel's reads of a request's keys, scored: one pass."""
    r = tl.program_id(0)
    d = tl.arange(0, D)
    q = tl.load(Q + r * D + d).to(tl.float32)
    total = tl.sum(q, 0) * 0.0
    for start in range(0, keys, BN):
        at = start + tl.arange(0, BN)
        block = tl.load(TABLE + r * stride_table + at // B).to(tl.int64)
        base = block * stride_block + (at % B) * stride_slot
        k = tl.load(KEYS + base[:, None] + d[None, :]).to(tl.float32)
        total += tl.sum(tl.exp(tl.sum(k * q[None, :], 1) * scale), 0)
    tl.store(OUT + r, total)


def keys_only(s: SimpleNamespace, out: torch.Tensor) -> None:
    rows = s.cache.rows
    keys = rows.view(s.cache.fp8) if s.cache.fp8 is not None else rows
    _keys_kernel[(s.requests,)](
        s.q,
        keys,
        s.table,
        out,
        s.keys,
        rows.stride(0),
        rows.stride(2),
        s.table.stride(0),
        DIM**-0.5,
        D=DIM,
        B=BLOCK,
        BN=BN,
    )


@triton.jit
def _scatter_kernel(
    TOKENS, P_COPY, keys, vocab, BN: tl.constexpr, RELAXED: tl.constexpr
):
    """The pointer kernel's adds of a request's positions into its row."""
    r = tl.program_id(0)
    for start in range(0, keys, BN):
        token = tl.load(TOKENS + r * keys + start + tl.arange(0, BN))
        share = tl.full([BN], 1e-6, tl.float32)
        if RELAXED:
            tl.atomic_add(P_COPY + r * vocab + token, share, sem="relaxed")
        else:
            tl.atomic_add(P_COPY + r * vocab + token, share)


def scatter(s: SimpleNamespace, relaxed: bool) -> None:
    _scatter_kernel[(s.requests,)](
        s.tokens, s.p_copy, s.keys, VOCAB, BN=BN, RELAXED=relaxed
    )


def attention(s: SimpleNamespace) -> Callable[[], object]:
    """vLLM's FlashAttention decode over a kv head of 64 per token, keys and values
    in bfloat16, about the pointer's bytes."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    blocks = s.table.numel() + 1
    k = torch.randn(blocks, BLOCK, 1, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q = torch.randn(s.requests, 1, 64, device="cuda", dtype=torch.bfloat16)
    starts = torch.arange(s.requests + 1, device="cuda", dtype=torch.int32)
    return lambda: flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q=1,
        cu_seqlens_q=starts,
        max_seqlen_k=s.keys,
        seqused_k=s.position,
        block_table=s.table,
        softmax_scale=64**-0.5,
        causal=True,
    )


def report(name: str, us: float, nbytes: int | None = None) -> None:
    rate = f"  {nbytes / us / 1e3:7.0f} GB/s" if nbytes else ""
    print(f"  {name:44s} {us:9.1f} us{rate}")


def bench(requests: int, keys: int, fp8: bool) -> None:
    """Print the times of the parts of a step of `requests` over `keys` keys."""
    s, r = setup(requests, keys, fp8, VOCAB), setup(requests, keys, fp8, REPEATS)
    print(f"{requests} requests x {keys} keys, {'FP8' if fp8 else 'bf16'}")
    report("step", timed(lambda: step(s)), s.key_bytes)
    report("kernel", timed(lambda: kernel(s)), s.key_bytes)
    report(f"kernel, {REPEATS} tokens", timed(lambda: kernel(r)), r.key_bytes)
    past = 2 * (copying.TOP + copying.FLOOR) / (2 * copying.KEY_NORM * DIM**0.5)
    two = timed(lambda: kernel(s, past))
    report("kernel, two passes in one program a request", two, s.key_bytes)
    w = SimpleNamespace(**vars(s))
    w.cache = SimpleNamespace(**vars(s.cache) | {"window": WINDOW})
    report(f"kernel, a window of {WINDOW}", timed(lambda: kernel(w)))
    out = torch.zeros(requests, device="cuda")
    report("keys: one pass, scored", timed(lambda: keys_only(s, out)), s.key_bytes)
    report(
        "scatter: acq_rel adds, as the kernel's were", timed(lambda: scatter(s, False))
    )
    report("scatter: relaxed adds", timed(lambda: scatter(s, True)))
    report(f"scatter: relaxed adds, {REPEATS} tokens", timed(lambda: scatter(r, True)))
    mix = timed(
        lambda: copying.mixed(s.logits, s.p_copy, s.a_sink, VOCAB - 1), graph=False
    )
    report("mixture, eager", mix)
    soft = timed(lambda: s.logits.float().log_softmax(-1), graph=False)
    report("head's log-softmax alone, eager", soft)
    whole = s.cache.rows.view(torch.uint8)
    report("read: the cache's bytes whole", timed(whole.amax), whole.numel())
    try:
        report("read: FlashAttention decode, kv head of 64", timed(attention(s)))
    except (ImportError, RuntimeError, TypeError) as e:  # the reference alone
        print(f"  attention: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", type=int, nargs="+", default=[8, 64])
    parser.add_argument("--keys", type=int, default=16384)
    args = parser.parse_args()
    print(f"{torch.cuda.get_device_name()}, vocabulary {VOCAB}, keys of {DIM}")
    for requests in args.requests:
        for fp8 in (True, False):
            bench(requests, args.keys, fp8)


if __name__ == "__main__":
    main()
