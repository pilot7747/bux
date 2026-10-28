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
    print("all agree" if all(results) else "MISMATCH")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
