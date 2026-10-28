"""Check the vLLM plugin's fused kernels of the residual algebras
(`bux_vllm.streams`) against its PyTorch path (`bux_vllm.residual`) on a GPU, on
random streams, and time both.

Run it where the plugin, vLLM and Triton are installed, with a CUDA GPU::

    python scripts/vllm_residual.py [--tokens 64] [--width 2816]

For each setting (hc with dynamic weights or static ones, single-pass; mhc
likewise; attnres in blocks, the block under way or not, and without blocks), dtype
(float32, bfloat16) and stream (a fresh embedding's copies, or written by a branch),
it runs six branches of the algebra and a merge both ways, on the same weights and
a body of ``tanh(h W)``, and compares every read and the merge: within 1e-4 in
float32 and 3e-2 in bfloat16 of the largest value. It prints each, with the time of
a branch both ways at the width given, and exits nonzero if any differ.
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace
from typing import Any

import torch

from bux_vllm import residual

TOLERANCE = {torch.float32: 1e-4, torch.bfloat16: 3e-2}
SETTINGS = {
    "hc": {"type": "hc", "expansion": 4, "dynamic": True, "single_pass": False},
    "hc of 3": {"type": "hc", "expansion": 3, "dynamic": True, "single_pass": False},
    "hc static": {"type": "hc", "expansion": 4, "dynamic": False, "single_pass": False},
    "hc single pass": {
        "type": "hc",
        "expansion": 4,
        "dynamic": True,
        "single_pass": True,
    },
    "mhc": {"type": "mhc", "expansion": 4, "dynamic": True, "single_pass": False},
    "mhc static": {
        "type": "mhc",
        "expansion": 4,
        "dynamic": False,
        "single_pass": False,
    },
    "mhc single pass": {
        "type": "mhc",
        "expansion": 4,
        "dynamic": True,
        "single_pass": True,
    },
    "attnres blocks of 4": {"type": "attnres", "block_size": 4},
    "attnres blocks of 2": {"type": "attnres", "block_size": 2},
    "attnres": {"type": "attnres", "block_size": None},
}
DEFAULTS = {"norm_eps": 1e-6, "sinkhorn_iters": 20}
BRANCHES = 6


def context() -> Any:
    """A forward pass's context, as a branch reads it: no single-pass read yet."""
    return SimpleNamespace(pending_read=None)


def build(c: dict, d: int, choice: str, body: bool = True) -> residual.Branch:
    """A branch of the algebra `c`, by the path `choice`, on the GPU."""
    residual.CHOICE = choice
    return residual.ALGEBRAS[c["type"]](c, d, body=body).cuda()


def twins(c: dict, d: int, dtype: torch.dtype, body: bool = True) -> tuple:
    """The two paths' branches of the same random weights, in `dtype`."""
    torch.set_default_dtype(dtype)
    torch_path, fused = build(c, d, "torch", body), build(c, d, "triton", body)
    torch.set_default_dtype(torch.float32)
    for (_, a), (_, b) in zip(
        torch_path.named_parameters(), fused.named_parameters(), strict=True
    ):
        a.data = (torch.randn_like(a) / 2).to(dtype)
        b.data.copy_(a.data)
    return torch_path, fused


def through(branches: list, x: Any, body: torch.Tensor) -> tuple[list, Any, Any]:
    """Run the stream x through `branches`, each body ``tanh(h body)``; return what
    each read, the stream after them, and the pass's context."""
    ctx, reads = context(), []
    for branch in branches:
        h, kept = branch.read(x, ctx)
        reads.append(h)
        x = branch.write_back(kept, torch.tanh(h @ body))
    return reads, x, ctx


def merge(c: dict, d: int, dtype: torch.dtype) -> tuple:
    """The two paths' merges, or None for single-pass hc and mhc."""
    if c.get("single_pass"):
        return None, None
    return twins(c, d, dtype, body=False)


def check(name: str, c: dict, d: int, tokens: int, dtype: torch.dtype) -> bool:
    """Compare the paths on `name`'s setting; print a line, return whether they
    agree."""
    c = DEFAULTS | c
    pairs = [twins(c, d, dtype) for _ in range(BRANCHES)]
    x0 = torch.randn(tokens, d, device="cuda", dtype=dtype)
    body = (torch.randn(d, d, device="cuda") / d**0.5).to(dtype)
    want, x, ctx = through([p[0] for p in pairs], residual.widen(c, x0), body)
    got, y, ctx_fused = through([p[1] for p in pairs], residual.widen(c, x0), body)
    readers = merge(c, d, dtype)
    if readers[0] is not None:
        want.append(readers[0].read(x, ctx)[0])
        got.append(readers[1].read(y, ctx_fused)[0])
    worst = max(_error(g, w) for g, w in zip(got, want, strict=True))
    ok = worst <= TOLERANCE[dtype]
    print(f"{name:22s} {dtype!s:15s} worst {worst:.2e}  {'ok' if ok else 'DIFFERS'}")
    return ok


def _error(got: torch.Tensor, want: torch.Tensor) -> float:
    """The largest difference, relative to the largest value of `want`."""
    scale = want.float().abs().max().clamp(min=1e-6)
    return float((got.float() - want.float()).abs().max() / scale)


def timed(c: dict, d: int, tokens: int) -> tuple[float, float]:
    """The milliseconds a branch's read and write take both ways, in bfloat16, at
    `tokens` tokens."""
    c = DEFAULTS | c
    out = []
    for branch in twins(c, d, torch.bfloat16):
        x = residual.widen(
            c, torch.randn(tokens, d, device="cuda", dtype=torch.bfloat16)
        )
        y = torch.randn(tokens, d, device="cuda", dtype=torch.bfloat16)
        out.append(_milliseconds(lambda b=branch, x=x, y=y: _step(b, x, y)))
    return out[0], out[1]


def _step(branch: residual.Branch, x: Any, y: torch.Tensor) -> Any:
    """One branch's read and write, its body's output y."""
    _, kept = branch.read(x, context())
    return branch.write_back(kept, y)


def _milliseconds(fn: Any, repeats: int = 50) -> float:
    """The mean time of `fn` on the GPU, after warming up."""
    for _ in range(3):
        fn()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main() -> None:
    """Check every setting in both dtypes, and time it."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--width", type=int, default=2816)
    args = p.parse_args()
    torch.manual_seed(0)
    bad = [
        name
        for name, c in SETTINGS.items()
        for dtype in (torch.float32, torch.bfloat16)
        for d in (200, args.width)
        if not check(name, c, d, args.tokens, dtype)
    ]
    for name, c in SETTINGS.items():
        slow, fast = timed(c, args.width, args.tokens)
        print(
            f"{name:22s} a branch at d={args.width}: torch {slow:.3f} ms, fused {fast:.3f} ms"
        )
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
