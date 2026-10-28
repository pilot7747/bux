"""Check the vLLM plugin's Triton kernels for sparse attention (`bux_vllm.kernels`)
against its PyTorch paths (`bux_vllm.picks`, and `bux_vllm.compressed` for csa) on a
GPU, on random paged caches.

Run it where the plugin and Triton are installed, with a CUDA GPU::

    python scripts/vllm_kernels.py

For each setting (DSA's top-k, MSA's blocks and picks per group, QSA's blocks, a
window, sinks, a softcap, indexer keys that pad their rows), dtype
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

from bux_vllm import compressed, indexer, kernels, picks

TOLERANCE = {torch.float32: 1e-4, torch.bfloat16: 2e-2}
SETTINGS = {
    "dsa": {},
    "dsa, window, sinks, softcap": {"window": 40, "softcap": 20.0, "sinks": True},
    "dsa, plain dots": {"relu": False},
    "dsa, odd index dims": {"index_dims": 31},
    "msa blocks": {"block_size": 4, "top_k": 32},
    "msa blocks per group": {"block_size": 4, "top_k": 32, "per_group": True},
    "msa blocks of 16": {"block_size": 16, "top_k": 64, "window": 200},
    "msa blocks of 16 per group": {"block_size": 16, "top_k": 64, "per_group": True},
    "qsa blocks, window": {"block_size": 4, "top_k": 32, "mean": True, "window": 64},
    "qsa blocks, interleaved rope, layernorm": {
        "block_size": 4,
        "top_k": 32,
        "mean": True,
        "rope": "interleaved",
        "norm": {"type": "layernorm", "bias": True, "zero_centered": False},
    },
    "qsa blocks, half rope, zero-centered rmsnorm": {
        "block_size": 4,
        "top_k": 32,
        "mean": True,
        "rope": "half",
        "norm": {"type": "rmsnorm", "bias": False, "zero_centered": True},
    },
}
#: A rope of the layer's, which the indexer pairs by its own layout.
ROPE = {"base": 10000.0, "scaling": None, "fraction": 1.0, "layout": "half"}


def spec(h: int, g: int, dk: int, heads: int, settings: dict) -> picks.Sparse:
    """A layer's settings: a top 16 of single keys, ReLU'd dots and a scale, unless
    `settings` say otherwise; QSA's with an indexer of `heads` that pools blocks."""
    sinks = torch.randn(h, device="cuda") if settings.get("sinks") else None
    mean, size = settings.get("mean", False), settings.get("block_size", 1)
    pooling = None
    if mean:
        ix = {"num_heads": heads, "head_dim": 32, "block_size": size, "pool": "mean"}
        norm = settings.get("norm")
        if norm is not None:
            norm = {"eps": 1e-5, "affine": True, "scale_input": False} | norm
        ix |= {"weighted": True, "norm": norm}
        ix |= {"rope_layout": settings.get("rope", "half")}
        rope = ROPE if settings.get("rope") else None
        pooling = indexer.Indexer(ix, 64, rope, 16 if rope else 0).cuda()
        with torch.no_grad():
            for p in pooling.k_norm.parameters():  # gains and biases away from 1, 0
                p.normal_()
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
    """A random paged cache of keys, values and index keys (`bux_vllm.picks.Cache`),
    16 tokens a block, and each request's block table, its blocks shuffled."""
    page, blocks = 16, sum(-(-n // 16) for n in lengths) + 1
    row = 2 * dk + -(-i // g)
    pages = torch.randn(blocks, g, page, row, device="cuda", dtype=dtype)
    order = torch.randperm(blocks - 1, device="cuda") + 1  # block 0 is vLLM's null
    width = max(-(-n // page) for n in lengths)
    table = torch.zeros(len(lengths), width, dtype=torch.int32, device="cuda")
    at = 0
    for r, n in enumerate(lengths):
        count = -(-n // page)
        table[r, :count] = order[at : at + count]
        at += count
    return picks.Cache(pages, 2 * dk, i), table


def picked(keys: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Each query's picks as a set: its valid keys in order, -1 for the rest."""
    return torch.sort(torch.where(valid, keys, -1), dim=-1).values


def compare(name: str, settings: dict, dtype: torch.dtype, decoding: bool) -> bool:
    """Whether the kernels' picks and outputs are the PyTorch path's."""
    torch.manual_seed(0)
    h, g, dk, i = 8, 2, 64, settings.get("index_dims", 32)
    heads = g if settings.get("per_group") else 4  # a group a head, picking apart
    lengths = [300, 97, 1, 513] if decoding else [300]
    cache, table = caches(lengths, g, dk, i, dtype)
    if decoding:
        positions = torch.tensor([n - 1 for n in lengths], device="cuda")
    else:
        positions = torch.arange(lengths[0], device="cuda")
    q = torch.randn(len(positions), h, dk, device="cuda", dtype=dtype)
    iq = torch.randn(len(positions), heads, i, device="cuda", dtype=dtype)
    iw = torch.randn(len(positions), heads, device="cuda", dtype=torch.float32)
    layer = spec(h, g, dk, heads, settings)
    assert kernels.takes(layer, (dk, dk), dtype)
    n = max(lengths)
    args = (layer, q, (iq, iw), positions, cache, table, n)
    want, (kept, valid) = picks.attend_segment(*args)
    got, (kept2, valid2) = kernels.attend_segment(*args)
    same = torch.equal(picked(kept, valid), picked(kept2, valid2))
    error = (got.float() - want.float()).abs().max().item()
    # bfloat16's scores may round apart at a near tie, and pick another key.
    ok = (same or dtype != torch.float32) and error <= TOLERANCE[dtype]
    layout = "decoding" if decoding else "prefill"
    print(f"{name:32} {dtype!s:15} {layout:9} picks {same} error {error:.2e}")
    return ok


#: csa's settings: a window, sinks, a softcap, and compressed latents or none.
CSA = {
    "csa": {"window": 16, "sinks": True},
    "csa, softcap, no window": {"window": None, "softcap": 30.0},
    "csa, window alone": {"window": 16, "sinks": True, "kept": False},
}


def paged(
    lengths: list[int], per: int, width: int, dtype: torch.dtype
) -> compressed.Paged:
    """A random `compressed.Paged` cache of rows of `width`, `per` rows a block,
    for sequences of `lengths` rows, its blocks shuffled."""
    blocks = sum(-(-n // per) for n in lengths) + 1
    cache = torch.randn(blocks, per, width, device="cuda", dtype=dtype)
    order = torch.randperm(blocks - 1, device="cuda") + 1
    table = torch.zeros(len(lengths), max(-(-n // per) for n in lengths) + 1)
    table = table.to(device="cuda", dtype=torch.int32)
    at = 0
    for r, n in enumerate(lengths):
        count = -(-n // per)
        table[r, :count] = order[at : at + count]
        at += count
    return compressed.Paged(cache, table)


def compare_csa(name: str, c: dict, dtype: torch.dtype, decoding: bool) -> bool:
    """Whether the csa kernel's outputs are compressed.attend's."""
    torch.manual_seed(0)
    h, d, ratio, top_k = 8, 64, 4, 24
    lengths = [300, 97, 1, 513] if decoding else [300]
    window = paged(lengths, 16, d, dtype)
    latents = paged([n // ratio + 1 for n in lengths], 4, d, dtype)
    caches = compressed.Caches(window, torch.zeros(1), latents)
    if decoding:
        positions = torch.tensor([n - 1 for n in lengths], device="cuda")
        seq = torch.arange(len(lengths), device="cuda")
    else:
        positions = torch.arange(lengths[0], device="cuda")
        seq = torch.zeros_like(positions)
    steps = compressed.Steps(positions, seq, 0, [], max(lengths))
    seen = (positions + 1) // ratio
    scores = torch.rand(len(positions), max(lengths) // ratio, device="cuda")
    kept = compressed.top(scores, seen, top_k) if c.get("kept", True) else None
    softcap = c.get("softcap")
    settings = compressed.Settings(c["window"], ratio, top_k, True, d**-0.5, softcap)
    sinks = torch.randn(h, device="cuda") if c.get("sinks") else None
    q = torch.randn(len(positions), h, d, device="cuda", dtype=dtype)
    want = compressed.attend(caches, steps, settings, q, kept, sinks)
    got = kernels.csa_attend(caches, steps, settings, q, kept, sinks)
    error = (got.float() - want.float()).abs().max().item()
    layout = "decoding" if decoding else "prefill"
    print(f"{name:32} {dtype!s:15} {layout:9} error {error:.2e}")
    return error <= TOLERANCE[dtype]


def compare_csa_picks(relu: bool, dtype: torch.dtype, decoding: bool) -> bool:
    """Whether the csa kernels' picks are compressed.select's (in float32; a near
    tie in bfloat16 may pick another latent)."""
    torch.manual_seed(0)
    heads, i, ratio, top_k = 4, 64, 4, 24
    lengths = [300, 97, 1, 513] if decoding else [300]
    index = paged([n // ratio + 1 for n in lengths], 4, i, dtype)
    caches = compressed.Caches(index, torch.zeros(1), index, index)
    if decoding:
        positions = torch.tensor([n - 1 for n in lengths], device="cuda")
        seq = torch.arange(len(lengths), device="cuda")
        steps = compressed.Steps(positions, seq, len(lengths), [], max(lengths))
    else:
        positions = torch.arange(lengths[0], device="cuda")
        seq = torch.zeros_like(positions)
        prompts = [(0, lengths[0], 0, lengths[0])]
        steps = compressed.Steps(positions, seq, 0, prompts, lengths[0])
    settings = compressed.Settings(16, ratio, top_k, relu, i**-0.5, None)
    q = torch.randn(len(positions), heads, i, device="cuda", dtype=dtype)
    w = torch.randn(len(positions), heads, device="cuda", dtype=dtype)
    want, _ = compressed.select(caches, steps, settings, q, w)
    got, _ = kernels.csa_select(caches, steps, settings, q, w)
    same = torch.equal(want.sort(-1).values, got.sort(-1).values)  # as sets
    layout = "decoding" if decoding else "prefill"
    name = "csa picks" + ("" if relu else ", plain dots")
    print(f"{name:32} {dtype!s:15} {layout:9} picks {same}")
    return same or dtype != torch.float32


def main() -> int:
    """Compare every setting, dtype and layout; 1 if any differ."""
    results = [
        compare(name, settings, dtype, decoding)
        for name, settings in SETTINGS.items()
        for dtype in TOLERANCE
        for decoding in (False, True)
    ]
    results += [
        compare_csa(name, c, dtype, decoding)
        for name, c in CSA.items()
        for dtype in TOLERANCE
        for decoding in (False, True)
    ]
    results += [
        compare_csa_picks(relu, dtype, decoding)
        for relu in (True, False)
        for dtype in TOLERANCE
        for decoding in (False, True)
    ]
    print("all agree" if all(results) else "MISMATCH")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
