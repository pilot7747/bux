"""Check the vLLM plugin's fused kernels of the residual algebras
(`bux_vllm.streams`) against its PyTorch path (`bux_vllm.residual`) on a GPU, on
random streams, and time both.

Run it where the plugin, vLLM and Triton are installed, with a CUDA GPU::

    python scripts/vllm_residual.py [--tokens 64] [--width 2816]

For each setting (hc with dynamic weights or static ones, single-pass; mhc
likewise; attnres in blocks, the block under way or not, and without blocks), dtype
(float32, bfloat16) and width, it runs six branches of the algebra and a merge,
from a fresh embedding's copies, on the same weights and a body of ``tanh(h W)``,
three ways: the PyTorch path in float32 (the reference), the PyTorch path in the
dtype, and the kernels in it. Every read and the merge are measured against the
reference, relative to its largest value. In float32 the kernels must agree within
1e-4. In bfloat16 they must be off by no more than twice what the PyTorch path is,
which rounds every term of the export where the kernels round their results once:
the rule `scripts/vllm_parity.py` holds vLLM to against bux in bfloat16. It prints
each, with the time of a branch both ways at the width given, and exits nonzero if
any differ.
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace
from typing import Any

import torch

from bux_vllm import residual

TOLERANCE = 1e-4  # the kernels' in float32, of the reference's largest value
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


def paths(c: dict, d: int, dtype: torch.dtype, body: bool = True) -> dict:
    """The reference's, the PyTorch path's and the kernels' branches of the same
    random weights, rounded to `dtype`: the first in float32, the others in it."""
    ways = {"reference": (torch.float32, "torch"), "torch": (dtype, "torch")}
    ways["fused"] = (dtype, "triton")
    out = {}
    for way, (kind, choice) in ways.items():
        torch.set_default_dtype(kind)
        out[way] = build(c, d, choice, body)
    torch.set_default_dtype(torch.float32)
    for params in zip(*(m.parameters() for m in out.values()), strict=True):
        value = (torch.randn_like(params[0]) / 2).to(dtype)
        for param in params:
            param.data.copy_(value)
    return out


def through(branches: list, x: Any, body: torch.Tensor) -> tuple[list, Any, Any]:
    """Run the stream x through `branches`, each body ``tanh(h body)``; return what
    each read, the stream after them, and the pass's context."""
    ctx, got = context(), []
    for branch in branches:
        h, kept = branch.read(x, ctx)
        got.append(h)
        x = branch.write_back(kept, torch.tanh(h @ body.to(h.dtype)))
    return got, x, ctx


def reads(c: dict, d: int, tokens: int, dtype: torch.dtype) -> dict[str, list]:
    """Every read of six branches and the merge's (single-pass hc and mhc have no
    reader of their own), each way of `paths`, of one embedding."""
    branches = [paths(c, d, dtype) for _ in range(BRANCHES)]
    readers = None if c.get("single_pass") else paths(c, d, dtype, body=False)
    x0 = torch.randn(tokens, d, device="cuda").to(dtype)
    body = (torch.randn(d, d, device="cuda") / d**0.5).to(dtype)
    out = {}
    for way in branches[0]:
        start = residual.widen(c, x0.to(torch.float32 if way == "reference" else dtype))
        got, x, ctx = through([b[way] for b in branches], start, body)
        if readers is not None:
            got.append(readers[way].read(x, ctx)[0])
        out[way] = got
    return out


def check(name: str, c: dict, d: int, tokens: int, dtype: torch.dtype) -> bool:
    """Measure the paths against the reference on `name`'s setting; print a line,
    return whether the kernels pass."""
    got = reads(DEFAULTS | c, d, tokens, dtype)
    error = {
        way: max(_error(g, w) for g, w in zip(got[way], got["reference"], strict=True))
        for way in ("torch", "fused")
    }
    bound = TOLERANCE if dtype == torch.float32 else 2 * error["torch"]
    ok = error["fused"] <= bound
    print(
        f"{name:22s} {dtype!s:15s} d={d:5d} kernels {error['fused']:.2e} "
        f"(PyTorch's {error['torch']:.2e})  {'ok' if ok else 'DIFFERS'}"
    )
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
    ways = paths(c, d, torch.bfloat16)
    for branch in (ways["torch"], ways["fused"]):
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
