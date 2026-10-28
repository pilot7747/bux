"""Check and time vLLM's own kernels for DeepSeek-V3.2's indexer on a GPU, on the
layout the plugin's sparse caches keep its keys in: DeepGEMM's FP8 MQA logits of a
prompt's gathered keys (``fp8_fp4_mqa_logits``) and of decoded tokens' paged keys
(``fp8_fp4_paged_mqa_logits``), vLLM's gather of a request's keys
(``cp_gather_indexer_k_quant_cache``), and its top-k of a prompt's and of decoded
tokens' logits. Each is compared with PyTorch on the same FP8 keys and timed beside
the plugin's Triton path (`bux_vllm.kernels`), at the indexer shapes of the zoo and
the timeline.

Run it in vLLM's environment, with a CUDA GPU::

    python scripts/vllm_native.py

It prints a line per check and exits nonzero if any fails or raises.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from typing import Any

import torch

#: The indexers timed: heads, dims, and the keys a query keeps.
SHAPES = {
    "timeline": (4, 64, 2048),
    "msa/qsa": (4, 128, 2048),
    "deepseek": (64, 128, 2048),
}
PAGE = 64  # tokens a kernel block, as DeepGEMM's paged kernel takes them


def timed(run: Callable[[], object], repeats: int = 10) -> float:
    """Milliseconds a call of `run` takes, the median of `repeats` after a warmup."""
    run()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def quantized(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x, (..., D), in FP8 e4m3 with a float32 scale a row, as DeepGEMM's tests
    cast keys: the row's largest magnitude over 448."""
    scale = x.abs().float().amax(-1, keepdim=True).clamp(min=1e-4) / 448.0
    return (x.float() / scale).to(torch.float8_e4m3fn), scale.squeeze(-1)


def region(blocks: int, dims: int) -> torch.Tensor:
    """An index region of `blocks` kernel blocks, inside pages that hold other rows
    too: each block's FP8 values of `PAGE` tokens, then their float32 scales, as
    DeepGEMM's paged layout keeps them, at a block stride of a whole page."""
    page = torch.zeros(blocks, 3, PAGE, 128, dtype=torch.bfloat16, device="cuda")
    raw = page.view(torch.uint8)  # (blocks, 3, PAGE, 256)
    return raw[:, 2].reshape(blocks, -1)[:, : PAGE * (dims + 4)]


def write(
    slots: torch.Tensor, keys: torch.Tensor, scales: torch.Tensor, at: torch.Tensor
) -> None:
    """Write FP8 `keys` (T, D) and their `scales` (T,) at the token `slots` (T,)
    of the index region `at`, (blocks, PAGE (D + 4)) bytes."""
    d = keys.shape[1]
    block, offset = slots // PAGE, slots % PAGE
    values = at[:, : PAGE * d].view(-1, PAGE, d)
    values[block, offset] = keys.view(torch.uint8)
    sf = at[:, PAGE * d : PAGE * (d + 4)].view(-1, PAGE, 4)
    sf[block, offset] = scales.view(torch.uint8).view(-1, 4)


def paged_view(at: torch.Tensor, dims: int) -> torch.Tensor:
    """The region as DeepGEMM's paged kernel takes it, (blocks, PAGE, 1, D + 4)."""
    stride = at.stride(0)
    return at.as_strided(
        (at.shape[0], PAGE, 1, dims + 4), (stride, dims + 4, dims + 4, 1)
    )


def gather_view(at: torch.Tensor, dims: int) -> torch.Tensor:
    """The region as vLLM's gather takes it, (blocks, PAGE, D + 4)."""
    return at.as_strided((at.shape[0], PAGE, dims + 4), (at.stride(0), dims + 4, 1))


def reference(q: torch.Tensor, w: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """``sum_h w_h ReLU(q_h . k_s)`` in float32: q (Q, H, D), w (Q, H), k (S, D)."""
    dots = torch.einsum("qhd,sd->qhs", q.float(), k.float()).relu()
    return torch.einsum("qhs,qh->qs", dots, w.float())


def same_sets(a: torch.Tensor, b: torch.Tensor, logits: torch.Tensor) -> float:
    """The share of rows whose picks score the same in all, as two top-k's may break
    ties apart: each row's sorted picked logits compared."""

    def scored(idx: torch.Tensor) -> torch.Tensor:
        got = logits.gather(1, idx.clamp(min=0).long()).masked_fill(
            idx < 0, float("-inf")
        )
        return got.sort(-1).values

    return (scored(a) == scored(b)).all(-1).float().mean().item()


def check_deep_gemm() -> bool:
    """Whether DeepGEMM imports and its MQA logits run."""
    from vllm.utils import deep_gemm
    from vllm.utils.import_utils import has_deep_gemm

    major, minor = torch.cuda.get_device_capability()
    print(
        f"device {torch.cuda.get_device_name()} sm{major}{minor}, has_deep_gemm {has_deep_gemm()}"
    )
    if not has_deep_gemm():
        return False
    q = torch.randn(8, 4, 64, device="cuda")
    k = torch.randn(32, 64, device="cuda")
    w = torch.rand(8, 4, device="cuda")
    qf, qs = quantized(q)
    kf, ks = quantized(k)
    start = torch.zeros(8, dtype=torch.int32, device="cuda")
    end = torch.arange(1, 9, dtype=torch.int32, device="cuda") * 4
    got = deep_gemm.fp8_fp4_mqa_logits((qf, None), (kf, ks), w * qs, start, end, False)
    want = reference(qf.float() * qs[..., None], w, kf.float() * ks[:, None])
    mask = torch.arange(32, device="cuda") < end[:, None]
    error = (got[:, :32] - want).abs().masked_fill(~mask, 0).max().item()
    print(f"deep_gemm tiny fp8_mqa_logits error {error:.2e}")
    return error < 1e-3 * want.abs().max().item() + 1e-4


def check_prefill(heads: int, dims: int, top_k: int, n: int) -> bool:
    """A prompt of `n` tokens: gather, logits, top-k, against PyTorch; then times."""
    from vllm import _custom_ops as ops
    from vllm.utils import deep_gemm

    blocks = -(-n // PAGE) + 1
    at = region(blocks, dims)
    table = (torch.randperm(blocks - 1, device="cuda") + 1).int()[None]
    slots = (
        table[0].long()[:, None] * PAGE + torch.arange(PAGE, device="cuda")
    ).flatten()[:n]
    keys, scales = quantized(torch.randn(n, dims, device="cuda", dtype=torch.bfloat16))
    write(slots, keys, scales, at)
    gathered = torch.empty(n, dims, dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty(n, 4, dtype=torch.uint8, device="cuda")
    cu = torch.tensor([0, n], dtype=torch.int32, device="cuda")
    ops.cp_gather_indexer_k_quant_cache(gather_view(at, dims), gathered, sf, table, cu)
    sf = sf.view(torch.float32).squeeze(-1)
    ok = torch.equal(
        gathered.view(torch.uint8), keys.view(torch.uint8)
    ) and torch.equal(sf, scales)
    q, qs = quantized(torch.randn(n, heads, dims, device="cuda", dtype=torch.bfloat16))
    w = torch.rand(n, heads, device="cuda") * qs
    start = torch.zeros(n, dtype=torch.int32, device="cuda")
    end = torch.arange(1, n + 1, dtype=torch.int32, device="cuda")
    logits = deep_gemm.fp8_fp4_mqa_logits(
        (q, None), (gathered, sf), w, start, end, False
    )
    rows = min(n, 512)  # the reference's rows, from the last
    want = reference(q[-rows:].float(), w[-rows:], gathered.float() * sf[:, None])
    beyond = torch.arange(n, device="cuda") >= end[-rows:, None]
    error = (logits[-rows:, :n] - want).abs().masked_fill(beyond, 0).max().item()
    error /= want.abs().max().item()
    out = torch.empty(n, top_k, dtype=torch.int32, device="cuda")
    run_topk = lambda: ops.top_k_per_row_prefill(
        logits, start, end, out, n, logits.stride(0), logits.stride(1), top_k
    )
    run_topk()
    masked = logits[-rows:, :n].masked_fill(beyond, float("-inf"))
    kept = min(top_k, n)
    torch_top = masked.topk(kept, -1).indices.int()
    share = same_sets(out[-rows:, :kept], torch_top, masked)
    ok = ok and error < 1e-3 and share > 0.99
    t_gather = timed(
        lambda: ops.cp_gather_indexer_k_quant_cache(
            gather_view(at, dims), gathered, sf.view(torch.uint8).view(n, 4), table, cu
        )
    )
    t_logits = timed(
        lambda: deep_gemm.fp8_fp4_mqa_logits(
            (q, None), (gathered, sf), w, start, end, False
        )
    )
    t_topk = timed(run_topk)
    print(
        f"prefill H{heads} D{dims} k{top_k} n{n}: gather {ok} logits error {error:.1e} "
        f"top-k agree {share:.3f} | gather {t_gather:.2f} ms logits {t_logits:.2f} ms "
        f"top-k {t_topk:.2f} ms"
    )
    return ok


def check_decode(
    heads: int, dims: int, top_k: int, lengths: list[int], longest: int
) -> bool:
    """A decoding step of requests of `lengths`: paged logits and top-k, against
    PyTorch; then times, with `longest` the logits' width (``max_model_len``)."""
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk
    from vllm.utils import deep_gemm

    b = len(lengths)
    per = [-(-n // PAGE) for n in lengths]
    blocks = sum(per) + 1
    at = region(blocks, dims)
    order = torch.randperm(blocks - 1, device="cuda") + 1
    table = torch.zeros(b, -(-longest // PAGE), dtype=torch.int32, device="cuda")
    keys, at_rows = [], 0
    for r, n in enumerate(lengths):
        table[r, : per[r]] = order[at_rows : at_rows + per[r]].int()
        at_rows += per[r]
        slots = (
            table[r, : per[r]].long()[:, None] * PAGE
            + torch.arange(PAGE, device="cuda")
        ).flatten()[:n]
        k, s = quantized(torch.randn(n, dims, device="cuda", dtype=torch.bfloat16))
        write(slots, k, s, at)
        keys.append((k, s))
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")[:, None]
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    schedule = deep_gemm.get_paged_mqa_logits_metadata(lens, PAGE, sms)
    q, qs = quantized(
        torch.randn(b, 1, heads, dims, device="cuda", dtype=torch.bfloat16)
    )
    w = (torch.rand(b, heads, device="cuda") * qs[:, 0]).contiguous()
    view = paged_view(at, dims)
    run_logits = lambda: deep_gemm.fp8_fp4_paged_mqa_logits(
        (q, None), view, w, lens, table, schedule, longest, False
    )
    logits = run_logits()
    error, ok = 0.0, True
    out = torch.full((b, top_k), -1, dtype=torch.int32, device="cuda")
    topk = get_indexer_topk("auto")
    run_topk = lambda: topk(logits, lens, 1, out, top_k, longest)
    run_topk()
    for r, (k, s) in enumerate(keys):
        want = reference(q[r].float(), w[r : r + 1], k.float() * s[:, None])[0]
        n = lengths[r]
        error = max(
            error, (logits[r, :n] - want).abs().max().item() / want.abs().max().item()
        )
        torch_top = want.topk(min(top_k, n)).indices.int()[None]
        ok = (
            ok
            and same_sets(out[r : r + 1, : min(top_k, n)], torch_top, want[None]) == 1.0
        )
    ok = ok and error < 1e-3
    t_logits = timed(run_logits)
    t_topk = timed(run_topk)
    backend = topk.resolve_backend(logits, top_k, b)
    print(
        f"decode H{heads} D{dims} k{top_k} B{b} up to {max(lengths)} of {longest}: "
        f"logits error {error:.1e} top-k {ok} ({backend}) "
        f"| logits {t_logits:.3f} ms top-k {t_topk:.3f} ms"
    )
    return ok


def time_triton(
    heads: int, dims: int, top_k: int, n: int, b: int, ctx: int, longest: int
) -> None:
    """The plugin's Triton path at the same shapes, for scale: a prompt's ranking
    keys and their top-k, and a decoding step's."""
    from bux_vllm import kernels, picks

    g, dk = 11, 64
    share = -(-dims // g)
    blocks = -(-max(n, ctx) // 16) * b + 1
    pages = torch.randn(
        blocks, g, 16, 2 * dk + share, device="cuda", dtype=torch.bfloat16
    )
    cache = picks.Cache(pages, 2 * dk, dims)
    row = (torch.randperm(blocks - 1, device="cuda")[: -(-n // 16)] + 1).int()[None]
    iq = torch.randn(n, heads, dims, device="cuda", dtype=torch.bfloat16)
    iw = torch.randn(n, heads, device="cuda")
    pos = torch.arange(n, device="cuda")
    ranks = lambda: kernels.prompt_scores(
        iq, iw, cache, row, n, True, None, (pos, None)
    )
    keys = ranks()
    t_prompt = timed(ranks, 3)
    t_best = timed(lambda: picks.best_of(keys, top_k), 3)
    shape = (b, -(-longest // 16))
    table = torch.randint(1, blocks, shape, device="cuda", dtype=torch.int32)
    dq = iq[:b].contiguous()
    dpos = torch.full((b,), ctx - 1, device="cuda")
    step = lambda: kernels.scores(
        dq, iw[:b], cache, table, longest, True, None, (dpos, None)
    )
    dkeys = step()
    t_step = timed(step)
    t_dbest = timed(lambda: picks.best_of(dkeys, top_k))
    print(
        f"triton H{heads} D{dims}: prompt of {n} ranks {t_prompt:.2f} ms top-k {t_best:.2f} ms "
        f"| decode B{b} over {longest} ranks {t_step:.3f} ms top-k {t_dbest:.3f} ms"
    )


def plugin_case(lengths: list[int], dims: int, g: int) -> tuple:
    """A paged cache of the plugin's native layout (`bux_vllm.dsa.Region` after `g`
    kv heads of 64 dims a key and a value), random keys, values and indexer keys
    of requests of `lengths`, and the same in its PyTorch layout (spread rows)."""
    from bux_vllm import dsa, picks

    blocks = sum(-(-n // PAGE) for n in lengths) + 1
    cache = torch.zeros(blocks, g + 1, PAGE, 128, dtype=torch.bfloat16, device="cuda")
    spread = torch.zeros(blocks, g, PAGE, 128 + -(-dims // g), device="cuda")
    table = torch.zeros(len(lengths), -(-max(lengths) // PAGE), dtype=torch.int32)
    table = table.cuda()
    order, at = torch.randperm(blocks - 1, device="cuda") + 1, 0
    for r, n in enumerate(lengths):
        count = -(-n // PAGE)
        table[r, :count] = order[at : at + count].int()
        at += count
        pos = torch.arange(n, device="cuda")
        slots = table[r, pos // PAGE].long() * PAGE + pos % PAGE
        kv = torch.randn(n, g, 128, device="cuda", dtype=torch.bfloat16)
        keys = torch.randn(n, dims, device="cuda", dtype=torch.bfloat16)
        picks.write(cache[:, :g], slots, kv)
        dsa.Region(cache, g, dims).write(slots, keys)
        picks.write(spread, slots, picks.merged(kv.float(), keys.float()))
    return cache, spread, table


def fp8_scores(
    region: Any,
    index: tuple[torch.Tensor, torch.Tensor],
    rows: torch.Tensor,
    positions: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """The scores the kernels pick by, by PyTorch over the same FP8 keys and
    queries, (Q, n): -inf for a key a query does not see."""
    q8, w = index
    scores = reference(q8.float(), w, region.dequantized(rows, n))
    see = torch.arange(n, device="cuda") <= positions[:, None]
    return scores.masked_fill(~see, float("-inf"))


def picked_as_best(chosen: torch.Tensor, scores: torch.Tensor) -> bool:
    """Whether the picks `chosen`, (Q, K) with -1 for none, score as the best K of
    `scores` do, within float32's rounding of them (two top-k's may break a near
    tie apart)."""
    k = min(chosen.shape[1], scores.shape[1])
    got = scores.gather(1, chosen.clamp(min=0).long())
    got = got.masked_fill(chosen < 0, float("-inf")).sort(-1, descending=True).values
    want = scores.topk(k, -1).values
    finite = want > float("-inf")
    close = (got[:, :k] - want).abs() <= 1e-4 * scores.abs().nan_to_num(0, 0, 0).max()
    return bool((close | ~finite).all() and (got[:, :k].isinf() == ~finite).all())


def check_plugin(decoding: bool) -> bool:
    """The plugin's native path, `bux_vllm.dsa`'s picks and `bux_vllm.kernels`'
    attention over them, against PyTorch: its picks against the best of the same
    FP8 scores, its output against PyTorch's attention over them, and the share of
    bux's float32 picks (`bux_vllm.picks`) it agrees with; then the attention over
    an FP8 KV cache against the bfloat16 one's."""
    from bux_vllm import dsa, kernels, picks

    torch.manual_seed(0)
    h, g, dk, heads, dims, top_k = 8, 2, 64, 4, 64, 64
    lengths = [300, 97, 1, 513] if decoding else [700]
    cache, spread, table = plugin_case(lengths, dims, g)
    n = max(lengths)
    if decoding:
        positions = torch.tensor([m - 1 for m in lengths], device="cuda")
        rows = table
    else:
        positions = torch.arange(n, device="cuda")
        rows = table[:1].expand(n, -1)
    q = torch.randn(len(positions), h, dk, device="cuda", dtype=torch.bfloat16)
    iq = torch.randn(len(positions), heads, dims, device="cuda", dtype=torch.bfloat16)
    iw = torch.randn(len(positions), heads, device="cuda")
    region = dsa.Region(cache, g, dims)
    index = dsa.queries(iq, iw, dsa.padded_heads(heads))
    if decoding:
        lens = (positions + 1).int()[:, None].contiguous()
        step = dsa.Decoding(lens, dsa.schedule(lens, PAGE), 1024)
        chosen = dsa.decode(region, index, table, step, top_k, "auto")
        best = all(
            picked_as_best(
                chosen[r : r + 1],
                fp8_scores(
                    region,
                    (index[0][r : r + 1], index[1][r : r + 1]),
                    table[r : r + 1],
                    positions[r : r + 1],
                    m,
                ),
            )
            for r, m in enumerate(lengths)
        )
    else:
        chosen = dsa.prompt(region, index, table[:1], n, positions, top_k)
        best = picked_as_best(
            chosen, fp8_scores(region, index, table[:1], positions, n)
        )
    spec = picks.Sparse(top_k, 1, False, False, True, None, dk**-0.5, None, dk, (0, g))
    kept = dsa.as_picks(chosen)
    got = kernels.attend(spec, q, cache[:, :g], rows, kept)
    kv = picks.gather(cache[:, :g], rows, kept[0])
    ref = picks.attend(q, kv, kept[1][:, None], dk, spec.scale)
    error = (got.float() - ref.float()).abs().max().item()
    bux = picks.attend_segment(
        spec,
        q.float(),
        (iq.float(), iw),
        positions,
        picks.Cache(spread, 128, dims),
        table[:1] if not decoding else table,
        n,
    )[1]
    ours = torch.where(kept[1], kept[0], -1).sort(-1).values
    theirs = torch.where(bux[1], bux[0], -1).sort(-1).values
    agree = (ours == theirs).float().mean().item()
    fp8 = cache[:, :g].float().clamp(-448, 448).to(torch.float8_e4m3fn)
    got8 = kernels.attend(spec, q, fp8, rows, kept)
    error8 = (got8.float() - got.float()).abs().max().item()
    layout = "decoding" if decoding else "prefill"
    print(
        f"plugin native {layout}: picks as FP8's best {best}, output error "
        f"{error:.1e}, agreeing with bux's float32 picks {agree:.3f}; "
        f"FP8 KV error {error8:.1e}"
    )
    return best and error < 2e-2 and error8 < 0.25


def check_qsa(decoding: bool) -> bool:
    """QSA on the plugin's native path: blocks of 4 pooled into the FP8 keys by
    `bux_vllm.dsa.pooled` as each completes, picked by the kernels, against the
    best of the same FP8 scores of the pooled keys by PyTorch, and bux's float32
    picks of blocks (`bux_vllm.picks.choose`) they agree with."""
    from bux_vllm import dsa, picks
    from bux_vllm import indexer as plugin_indexer

    torch.manual_seed(0)
    g, heads, dims, size, top_k = 2, 4, 64, 4, 64
    c: dict[str, Any] = {"num_heads": heads, "head_dim": dims, "top_k": top_k}
    c |= {"block_size": size, "pool": "mean", "weighted": False, "norm": None}
    c |= {"rope_layout": "half"}
    rope = {"base": 10000.0, "scaling": None, "fraction": 1.0, "layout": "half"}
    ix = plugin_indexer.Indexer(c, 128, rope, 32).cuda()
    lengths = [300, 97, 3, 513] if decoding else [700]
    blocks = sum(-(-n // PAGE) for n in lengths) + 1
    raw_slots = dsa.raw_slots(dims, 256)
    cache = torch.zeros(blocks, g + raw_slots + 1, PAGE, 128, device="cuda")
    cache = cache.bfloat16()
    raw, region = dsa.Raw(cache, g, dims), dsa.Region(cache, g + raw_slots, dims)
    table = torch.zeros(len(lengths), -(-max(lengths) // PAGE), dtype=torch.int32)
    table, order, at = table.cuda(), torch.randperm(blocks - 1, device="cuda") + 1, 0
    keys = []
    for r, n in enumerate(lengths):
        count = -(-n // PAGE)
        table[r, :count] = order[at : at + count].int()
        at += count
        pos = torch.arange(n, device="cuda")
        rows = table[r : r + 1].expand(n, -1)
        x = torch.randn(n, 128, device="cuda")
        slots = table[r, pos // PAGE].long() * PAGE + pos % PAGE
        raw.write(slots, ix.keys(x, pos))
        dsa.pooled(raw, region, ix, rows, pos, size)
        keys.append(raw.at(rows[:1], pos[None])[0])
    q = torch.randn(
        len(lengths) if decoding else lengths[0], heads, dims, device="cuda"
    )
    w = ix.head_weights(torch.zeros(len(q), 128, device="cuda"))
    index = dsa.queries(q.bfloat16(), w, dsa.padded_heads(heads))
    if decoding:
        positions = torch.tensor([n - 1 for n in lengths], device="cuda")
        lens = ((positions + 1) // size).int()[:, None].contiguous()
        step = dsa.Decoding(lens, dsa.schedule(lens, PAGE), 1024 // size)
        chosen = dsa.decode(region, index, table, step, top_k // size, "auto")
        rows = [r for r in range(len(lengths))]
    else:
        positions = torch.arange(lengths[0], device="cuda")
        ends = (positions + 1) // size
        chosen = dsa.prompt(
            region, index, table[:1], lengths[0] // size, ends, top_k // size
        )
        rows = [0] * len(positions)
    best, agree = True, []
    for t, r in enumerate(rows):
        n = lengths[r] // size
        p = positions[t : t + 1]
        if n == 0:
            best &= bool((chosen[t] < 0).all())
            continue
        q8 = (index[0][t : t + 1], index[1][t : t + 1])
        scores = fp8_scores(region, q8, table[r : r + 1], (p + 1) // size - 1, n)
        best &= picked_as_best(chosen[t : t + 1], scores)
        spec = picks.Sparse(
            top_k, size, True, False, True, None, 1.0, None, 64, (0, 1), indexer=ix
        )
        see = picks.seen(p, lengths[r], None)
        theirs = picks.choose(spec, q[t : t + 1], w[t : t + 1], keys[r], see, p)
        ours = dsa.block_picks(chosen[t : t + 1], p, size)
        a = torch.where(ours[1], ours[0], -1).sort(-1).values
        b = torch.where(theirs[1], theirs[0], -1).sort(-1).values
        agree.append((a == b).float().mean().item())
    layout = "decoding" if decoding else "prefill"
    share = sum(agree) / max(len(agree), 1)
    print(
        f"plugin qsa {layout}: picks as FP8's best {best}, agreeing with bux's {share:.3f}"
    )
    return best


def time_attention(n: int, b: int, ctx: int) -> None:
    """The Triton attention over each query's top 2048 at the timeline's width
    2816 (44 query heads of 64 over 11 kv heads), for a prompt of `n` tokens and a
    decoding step of `b` requests of `ctx`, in bfloat16 and over an FP8 cache."""
    from bux_vllm import kernels, picks

    h, g, dk, top_k = 44, 11, 64, 2048
    blocks = -(-max(n, ctx) // PAGE) * b + 1
    cache = torch.randn(blocks, g, PAGE, 2 * dk, device="cuda", dtype=torch.bfloat16)
    fp8 = cache.clamp(-448, 448).to(torch.float8_e4m3fn)
    spec = picks.Sparse(top_k, 1, False, False, True, None, dk**-0.5, None, dk, (0, g))
    for what, count, length in (("prompt", n, n), ("decode", b, ctx)):
        requests = 1 if what == "prompt" else count  # a prompt's queries share one
        width = -(-length // PAGE)
        table = torch.randint(1, blocks, (requests, width), device="cuda").int()
        table = table.expand(count, -1)
        pos = torch.arange(count, device="cuda") % length if what == "prompt" else None
        pos = torch.full((count,), length - 1, device="cuda") if pos is None else pos
        keys = (torch.rand(count, top_k, device="cuda") * (pos[:, None] + 1)).long()
        kept = (keys, keys <= pos[:, None])
        q = torch.randn(count, h, dk, device="cuda", dtype=torch.bfloat16)
        args = (q, table, kept)
        bf16 = timed(lambda c=cache, a=args: kernels.attend(spec, a[0], c, *a[1:]))
        f8 = timed(lambda c=fp8, a=args: kernels.attend(spec, a[0], c, *a[1:]))
        print(
            f"attention {what} of {count}: bfloat16 {bf16:.2f} ms, FP8 KV {f8:.2f} ms"
        )


def attempt(name: str, check: Callable[[], bool | None]) -> bool:
    """Run `check`, printing what it raises; whether it passed."""
    try:
        return check() is not False
    except Exception:  # noqa: BLE001 - report every failure and go on
        print(f"{name} raised:")
        traceback.print_exc()
        return False


def main() -> int:
    """Run every check, then the timings; 1 if any fails."""
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device("cuda"))  # as vLLM's model runner does
    if not attempt("deep_gemm", check_deep_gemm):
        print("DeepGEMM unusable: native indexer kernels unavailable")
        return 1
    results = []
    for name, (heads, dims, top_k) in SHAPES.items():
        results.append(
            attempt(
                f"{name} prefill", lambda h=heads, d=dims: check_prefill(h, d, 128, 700)
            )
        )
        results.append(
            attempt(
                f"{name} decode",
                lambda h=heads, d=dims: check_decode(
                    h, d, 128, [300, 2500, 1, 4097], 8192
                ),
            )
        )
    for name, (heads, dims, top_k) in SHAPES.items():
        attempt(
            f"{name} prefill 16k",
            lambda h=heads, d=dims, k=top_k: check_prefill(h, d, k, 16384),
        )
        attempt(
            f"{name} decode 64x20k",
            lambda h=heads, d=dims, k=top_k: check_decode(h, d, k, [20000] * 64, 32768),
        )
    attempt("triton", lambda: time_triton(4, 64, 2048, 16384, 64, 20000, 32768))
    for decoding in (False, True):
        results.append(attempt("plugin", lambda d=decoding: check_plugin(d)))
        results.append(attempt("plugin qsa", lambda d=decoding: check_qsa(d)))
    attempt("attention", lambda: time_attention(16384, 64, 20000))
    print("all agree" if all(results) else "MISMATCH")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
