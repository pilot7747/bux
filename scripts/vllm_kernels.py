"""Check the vLLM plugin's Triton kernels for sparse attention (`bux_vllm.kernels`)
against its PyTorch path (`bux_vllm.picks`) on a GPU, on random paged caches.

Run it where the plugin and Triton are installed, with a CUDA GPU::

    python scripts/vllm_kernels.py

For each setting (DSA's top-k, MSA's blocks and picks per group, QSA's blocks, a
window, sinks, a softcap), dtype
(float32, bfloat16) and layout (one request's prefill, a decoding step of several
requests), it compares the picks, which must be equal in float32, and the outputs,
which must agree within 1e-4 in float32 and 2e-2 in bfloat16 (where a near tie may
pick another key). It exits nonzero if any differ.
"""

from __future__ import annotations

import os
import sys

os.environ["BUX_VLLM_SPARSE_KERNELS"] = "triton"

import torch

from bux_vllm import indexer, kernels, picks

TOLERANCE = {torch.float32: 1e-4, torch.bfloat16: 2e-2}
SETTINGS = {
    "dsa": {},
    "dsa, window, sinks, softcap": {"window": 40, "softcap": 20.0, "sinks": True},
    "dsa, plain dots": {"relu": False},
    "msa blocks": {"block_size": 4, "top_k": 32},
    "msa blocks per group": {"block_size": 4, "top_k": 32, "per_group": True},
    "qsa blocks, window": {"block_size": 4, "top_k": 32, "mean": True, "window": 64},
}


def spec(h: int, g: int, dk: int, heads: int, settings: dict) -> picks.Sparse:
    """A layer's settings: a top 16 of single keys, ReLU'd dots and a scale, unless
    `settings` say otherwise; QSA's with an indexer of `heads` that pools blocks."""
    sinks = torch.randn(h, device="cuda") if settings.get("sinks") else None
    mean, size = settings.get("mean", False), settings.get("block_size", 1)
    pooling = None
    if mean:
        ix = {"num_heads": heads, "head_dim": 32, "block_size": size, "pool": "mean"}
        ix |= {"weighted": True, "norm": None, "rope_layout": "half"}
        pooling = indexer.Indexer(ix, 64, None, 0).cuda()
    return picks.Sparse(
        settings.get("top_k", 16),
        size,
        mean,
        settings.get("per_group", False),
        settings.get("relu", True),
        settings.get("window"),
        dk**-0.5,
        settings.get("softcap"),
        dk,
        (0, g),
        sinks,
        pooling,
    )


def caches(lengths: list[int], g: int, dk: int, i: int, dtype: torch.dtype) -> tuple:
    """Random paged caches of keys and values and of index keys, 16 tokens a block,
    and each request's block table, its blocks shuffled."""
    page, blocks = 16, sum(-(-n // 16) for n in lengths) + 1
    kv = torch.randn(blocks, g, page, 2 * dk, device="cuda", dtype=dtype)
    index = torch.randn(blocks, 1, page, i, device="cuda", dtype=dtype)
    order = torch.randperm(blocks - 1, device="cuda") + 1  # block 0 is vLLM's null
    width = max(-(-n // page) for n in lengths)
    table = torch.zeros(len(lengths), width, dtype=torch.int32, device="cuda")
    at = 0
    for r, n in enumerate(lengths):
        count = -(-n // page)
        table[r, :count] = order[at : at + count]
        at += count
    return (kv, index), table


def compare(name: str, settings: dict, dtype: torch.dtype, decoding: bool) -> bool:
    """Whether the kernels' picks and outputs are the PyTorch path's."""
    torch.manual_seed(0)
    h, g, dk, i = 8, 2, 64, 32
    heads = g if settings.get("per_group") else 4  # a group a head, picking apart
    lengths = [300, 97, 1, 513] if decoding else [300]
    (kv, index), table = caches(lengths, g, dk, i, dtype)
    if decoding:
        positions = torch.tensor([n - 1 for n in lengths], device="cuda")
    else:
        positions = torch.arange(lengths[0], device="cuda")
    rows = (table, table)
    q = torch.randn(len(positions), h, dk, device="cuda", dtype=dtype)
    iq = torch.randn(len(positions), heads, i, device="cuda", dtype=dtype)
    iw = torch.randn(len(positions), heads, device="cuda", dtype=torch.float32)
    layer = spec(h, g, dk, heads, settings)
    assert kernels.takes(layer, (dk, dk))
    n = max(lengths)
    args = (layer, q, (iq, iw), positions, (kv, index), rows, n)
    want, (kept, valid) = picks.attend_segment(*args)
    got, (kept2, valid2) = kernels.attend_segment(*args)
    same = torch.equal(kept[valid], kept2[valid2]) and torch.equal(valid, valid2)
    error = (got.float() - want.float()).abs().max().item()
    # bfloat16's scores may round apart at a near tie, and pick another key.
    ok = (same or dtype != torch.float32) and error <= TOLERANCE[dtype]
    layout = "decoding" if decoding else "prefill"
    print(f"{name:32} {dtype!s:15} {layout:9} picks {same} error {error:.2e}")
    return ok


def main() -> int:
    """Compare every setting, dtype and layout; 1 if any differ."""
    results = [
        compare(name, settings, dtype, decoding)
        for name, settings in SETTINGS.items()
        for dtype in TOLERANCE
        for decoding in (False, True)
    ]
    print("all agree" if all(results) else "MISMATCH")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
