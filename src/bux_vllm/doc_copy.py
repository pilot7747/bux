"""bux's ``doc_copy`` in vLLM: each step's tokens' candidate next tokens, found where
vLLM's model runner prepares the step's inputs, and the layer that adds their copy
vector to the stream.

The candidates of a step depend on its tokens and their sequences' earlier tokens
alone, so the model runner's state (`DocCopyInputs`, vLLM's V2 model runner) finds
them before the model runs, in a buffer of a fixed shape that CUDA graphs replay,
as it gives an engram its ids (`bux_vllm.engram`). A Triton kernel scans each
token's sequence, in vLLM's tokens of its requests, for the earlier windows that
match its own (`bux_vllm.copying`), a program a token: no host sync, no loop over
requests. The layers read them from the forward's context; the one that writes its
``copy`` port pushes the vector, and one that reads it adds it by its own gain.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
from torch import nn

from . import copying
from .common import Ctx, Layer, Site
from .plan import Node, live, plan

BLOCK = 512  # earlier positions a program scans at once


def layers(config: Any) -> list[Node]:
    """The ``doc_copy`` layers that compute candidates, before the first head."""
    calls = live(plan(config.bux, config.bux_reserved).calls)
    nodes = [c.layer for c in calls if c.layer["config"]["type"] == "doc_copy"]
    return [
        n
        for n in nodes
        if not any(p["name"] == "copy" and p["mode"] != "write" for p in n["ports"])
    ]


class DocCopy(Layer):
    """bux's ``doc_copy``: adds ``gain (E[c] s[b] + e[b])`` of each token's candidate
    c and bucket b, which the model runner found (`DocCopyInputs`), E the model's
    first embedding's table, split by vocabulary; to every copy of a widened stream.
    With its ``copy`` port it pushes the vector, or reads it instead and adds it by
    its own gain. Its tables stay in the model's dtype, unquantized."""

    bux_types = ("doc_copy",)

    def __init__(self, c: dict, width: Any, site: Site):
        super().__init__()
        # The embedding is its own layer's: a tuple keeps it out of this module's.
        self.ports, self.embedding = site.ports, (site.embedding,)
        self.gain = nn.Parameter(torch.tensor(float(c["gain"])), requires_grad=False)
        self.reads = any(n == "copy" and m != "write" for n, _, m in site.ports)
        if self.reads:
            return
        if site.embedding is None:
            raise ValueError("doc_copy takes its rows from an embedding before it")
        buckets = 1 + len(c["lengths"]) * c["counts"]
        self.scale = nn.Parameter(torch.zeros(buckets), requires_grad=False)
        d = site.config.bux["d_model"]
        self.bucket_embed = nn.Parameter(torch.zeros(buckets, d), requires_grad=False)

    def forward(self, x: Any, ctx: Ctx) -> Any:
        if self.reads:
            tap, pop = next((t, m == "pop") for n, t, m in self.ports if n == "copy")
            v = ctx.read(tap, pop)
        else:
            assert ctx.doc_copy is not None, "doc_copy reads the runner's candidates"
            found = ctx.doc_copy[:, : x.shape[0]]
            rows = self.embedding[0].tokens(found[0])
            v = copying.vector(rows, found[1].long(), self.scale, self.bucket_embed)
        for name, tap, mode in self.ports:
            if name == "copy" and mode == "write":
                ctx.push(tap, v)
        v = v if x.dim() == 2 else v[:, None]
        return x + (self.gain * v).to(x.dtype)


class DocCopyInputs:
    """What a model runner state adds to the inputs of a model with ``doc_copy``:
    each of the step's tokens' candidate and bucket, (2, T), in a buffer of a fixed
    shape that CUDA graphs replay, 0 for none and for padding."""

    lengths: tuple[int, ...] = ()
    counts: int = 1

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        tokens, device = self.max_num_tokens, self.device  # ty: ignore[unresolved-attribute]
        self.doc_copy = torch.zeros(2, tokens, dtype=torch.int32, device=device)
        self.doc_lengths = torch.tensor(self.lengths, dtype=torch.int32, device=device)

    def prepare_inputs(self, input_batch: Any, req_states: Any) -> dict[str, Any]:
        out = super().prepare_inputs(input_batch, req_states)  # ty: ignore[unresolved-attribute]
        padded = input_batch.num_tokens_after_padding
        found = self.doc_copy[:, :padded]
        found.zero_()
        n, t = input_batch.num_reqs, input_batch.num_tokens
        if n and t:
            reqs = input_batch.idx_mapping[:n]
            starts = input_batch.query_start_loc[: n + 1]
            computed = req_states.num_computed_tokens.gpu[reqs]
            token = torch.arange(t, device=found.device, dtype=starts.dtype)
            seq = torch.searchsorted(starts[1:], token, right=True)
            pos = computed[seq] + token - starts[seq]
            history = req_states.all_token_ids.gpu
            self._scan(history, reqs[seq], pos, found[:, :t])
        return out | {"doc_copy": found}

    def _scan(
        self,
        history: torch.Tensor,
        rows: torch.Tensor,
        pos: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Each token's candidate and bucket into `out`, by `_scan_kernel`."""
        _scan_kernel[(pos.shape[0],)](
            history, history.stride(0), rows.contiguous(), pos.contiguous(),
            self.doc_lengths, out, out.stride(0),
            NL=len(self.lengths), MOST=max(self.lengths), COUNTS=self.counts,
            BLOCK=BLOCK,
        )  # fmt: skip

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        out = super().prepare_dummy_inputs(num_reqs, num_tokens)  # ty: ignore[unresolved-attribute]
        self.doc_copy.zero_()
        return out | {"doc_copy": self.doc_copy[:, :num_tokens]}


def inputs_class(config: Any) -> type | None:
    """`DocCopyInputs` set to the model's ``doc_copy`` layers' settings, which must
    agree; None without one."""
    found = layers(config)
    if not found:
        return None
    settings = {(tuple(n["config"]["lengths"]), n["config"]["counts"]) for n in found}
    if len(settings) > 1:
        raise NotImplementedError("doc_copy layers of different lengths or counts")
    lengths, counts = settings.pop()
    if len(lengths) > 16:
        raise NotImplementedError("doc_copy of more than 16 lengths in vLLM")
    return type(
        "BuxDocCopyInputs", (DocCopyInputs,), {"lengths": lengths, "counts": counts}
    )


@triton.jit
def _scan_kernel(
    hist_ptr, hist_stride, row_ptr, pos_ptr, len_ptr, out_ptr, out_stride,
    NL: tl.constexpr, MOST: tl.constexpr, COUNTS: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """A token's candidate and bucket: its sequence's earlier positions scanned a
    block at a time, each one's common suffix with the token's, up to MOST long,
    then, for each length, the latest that matches and how many do."""
    tok = tl.program_id(0)
    row = tl.load(row_ptr + tok).to(tl.int64)
    p = tl.load(pos_ptr + tok).to(tl.int64)
    h = hist_ptr + row * hist_stride
    which = tl.arange(0, 16)  # a lane per length, NL of them live
    last = tl.full((16,), -1, tl.int64)
    seen = tl.zeros((16,), tl.int32)
    for start in range(0, p, BLOCK):
        j = start + tl.arange(0, BLOCK).to(tl.int64)
        run = j < p
        common = tl.zeros((BLOCK,), tl.int32)
        for k in tl.static_range(MOST):
            mine = tl.load(h + p - k, mask=p - k >= 0, other=-1)
            theirs = tl.load(h + j - k, mask=run & (j - k >= 0), other=-2)
            run = run & (theirs == mine)
            common += run.to(tl.int32)
        for i in tl.static_range(NL):
            n = tl.load(len_ptr + i)
            match = (common >= n) & (j >= n - 1) & (j < p)
            latest = tl.max(tl.where(match, j, -1), axis=0)
            count = tl.sum(match.to(tl.int32), axis=0)
            last = tl.where(which == i, tl.maximum(last, latest), last)
            seen = tl.where(which == i, seen + count, seen)
    lens = tl.load(len_ptr + which, mask=which < NL, other=1 << 30)
    ok = (last >= 0) & (p >= lens - 1) & (which < NL)
    best = tl.max(tl.where(ok, which, -1), axis=0)  # the longest length that matched
    at = tl.max(tl.where(which == best, last, -1), axis=0)
    times = tl.max(tl.where(which == best, seen, 0), axis=0)
    candidate = tl.load(h + at + 1, mask=best >= 0, other=0)
    bucket = tl.where(best >= 0, best * COUNTS + tl.minimum(times, COUNTS), 0)
    tl.store(out_ptr + tok, tl.where(best >= 0, candidate, 0))
    tl.store(out_ptr + out_stride + tok, bucket)
