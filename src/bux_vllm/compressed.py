"""Compressed sparse attention's arithmetic, in plain PyTorch: what bux's ``csa``
computes between its projections and vLLM's paged caches (`bux_vllm.csa`), on a
batch of many sequences' tokens side by side.

A ``csa`` call keeps up to four caches, each a `Paged` table of rows: the window's
token latents, a row per token; the compressed latents and their indexer keys, each
a row per ``compress_ratio`` tokens; and the last ``compress_ratio`` tokens'
projections, which the next compressed latent pools (`cache_rows`). Each is a cache
of its own, so that its rows are a power of two wide where the dims are, as vLLM
needs to lay their blocks side by side. A step writes its tokens' rows
first and reads every key from the caches after, so that a prompt's chunks and a
decoded token read alike. Nothing loops over tokens: a step's queries go in chunks
of at most `CHUNK` elements an intermediate, and the indexer scores a prompt's
queries against its sequence's keys, gathered once.

It imports nothing of vLLM, so that the CPU tests run it against the transformers
export (``bux.export.hf.csa``) on paged caches of their own.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from .common import Norm, Rope

#: The most elements an intermediate of a chunk of queries holds.
CHUNK = 1 << 25

#: The logit of a masked key, as bux's.
MIN = torch.finfo(torch.float32).min


def reads(ports: list | None, name: str) -> bool:
    """Whether a layer of `ports`, ``[name, tap, mode]`` each, reads its port
    `name`."""
    return any(n == name and m != "write" for n, _, m in ports or [])


def writes(ports: list | None, name: str) -> bool:
    """Whether a layer of `ports` writes its port `name`."""
    return any(n == name and m == "write" for n, _, m in ports or [])


def compresses(c: dict, ports: list | None = None) -> bool:
    """Whether a ``csa`` layer makes compressed latents of its own: it has a
    ``compress_ratio``, and reads no others' through its ``kv`` port."""
    return c["compress_ratio"] is not None and not reads(ports, "kv")


def cache_rows(
    c: dict, ports: list | None = None
) -> dict[str, tuple[int, int, int | None]]:
    """The caches a ``csa`` call keeps, by name: the values of a row, the tokens a
    row is for, and the last tokens it keeps (None for all of them). One that reads
    its ``kv`` port keeps its window of token latents alone."""
    d, r, w = c["head_dim"], c["compress_ratio"], c["sliding_window_size"]
    out: dict[str, tuple[int, int, int | None]] = {"window": (d, 1, w)}
    if r is None or not compresses(c, ports):
        return out
    out["compressed"] = (d, r, None)
    if c["indexer"]:
        out["index"] = (c["indexer"]["head_dim"], r, None)
    if r > 1:
        out["state"] = (2 * d, 1, r)
    return out


@dataclass
class Paged:
    """One of a call's paged caches in a batch, as vLLM keeps it.

    vLLM's block 0 is its null block, which no sequence holds: the rows of tokens
    that write nothing (padding) go there.

    Attributes:
        cache: The rows, (blocks, rows a block, width).
        block_table: Each sequence's blocks, (sequences, blocks).
    """

    cache: torch.Tensor
    block_table: torch.Tensor

    def at(
        self, seq: torch.Tensor, index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The block and row of the entries `index` of the sequences `seq` (which
        broadcast); the null block's first row where `index` is below 0."""
        per = self.cache.shape[1]
        i = index.clamp(min=0)
        column = (i // per).clamp(max=self.block_table.shape[1] - 1)
        block = self.block_table[seq, column].long().clamp(min=0)
        none = index < 0
        return block.masked_fill(none, 0), (i % per).masked_fill(none, 0)

    def read(self, seq: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """The rows of entries `index` of sequences `seq`, (*index.shape, width);
        zeros where `index` is below 0."""
        block, row = self.at(seq, index)
        return self.cache[block, row].masked_fill((index < 0)[..., None], 0)

    def put(self, seq: torch.Tensor, index: torch.Tensor, rows: torch.Tensor) -> None:
        """Write `rows` at entries `index` of sequences `seq`; nothing (the null
        block) where `index` is below 0."""
        block, row = self.at(seq, index)
        self.cache[block, row] = rows.to(self.cache.dtype)

    def write(self, slots: torch.Tensor, rows: torch.Tensor) -> None:
        """Write `rows` at vLLM's `slots` (a block times its rows, plus the row);
        nothing (the null block) where a slot is below 0."""
        per = self.cache.shape[1]
        none = slots < 0
        block = (slots // per).masked_fill(none, 0)
        self.cache[block, (slots % per).masked_fill(none, 0)] = rows.to(
            self.cache.dtype
        )


@dataclass
class Steps:
    """A batch's tokens as vLLM schedules them: its decoded tokens first, one a
    sequence, then its prompts' chunks.

    Attributes:
        positions: Each token's position in its sequence, (T,).
        seq: Each token's sequence, its row of the block tables, (T,).
        decodes: How many of the first tokens are decoded ones.
        prompts: Each prompt chunk's first token, tokens, sequence and the sequence's
            length with it.
        longest: The longest sequence's length, or more.
    """

    positions: torch.Tensor
    seq: torch.Tensor
    decodes: int
    prompts: list[tuple[int, int, int, int]]
    longest: int


def prompt_chunks(
    starts: list[int], lengths: list[int], decodes: int
) -> list[tuple[int, int, int, int]]:
    """The prompt chunks of a batch whose sequences' tokens start at `starts` (and
    the last ends at ``starts[-1]``), each sequence `lengths` long with them, after
    its first `decodes` sequences: each chunk's first token, tokens, sequence and
    length (`Steps.prompts`)."""
    return [
        (starts[i], starts[i + 1] - starts[i], i, lengths[i])
        for i in range(decodes, len(starts) - 1)
        if starts[i + 1] > starts[i]
    ]


@dataclass
class Caches:
    """A call's caches in a batch, and the rows its tokens write.

    Attributes:
        window: The token latents.
        slots: Each token's row of `window`, vLLM's slot mapping, (T,).
        compressed: The compressed latents; None for a window alone.
        index: Their indexer keys; None without an indexer.
        state: The projections the next compressed latent pools; None unless
            ``compress_ratio > 1``.
        state_slots: Each token's row of `state`, (T,).
    """

    window: Paged
    slots: torch.Tensor
    compressed: Paged | None = None
    index: Paged | None = None
    state: Paged | None = None
    state_slots: torch.Tensor | None = None


@dataclass(frozen=True)
class Settings:
    """What a call's attention takes of its config.

    Attributes:
        window: The keys each query sees of the token latents, itself among them;
            None for all.
        ratio: The tokens of a compressed latent; None for none.
        top_k: The compressed latents the indexer keeps; None without one.
        relu: Whether the indexer ReLUs its dot products.
        scale: What a logit is multiplied by.
        softcap: The logits' soft cap; None for none.
        pool: The indexer's candidate pools, its blocks' latents and how many
            blocks, which its ``candidates`` port passes on; None without one.
    """

    window: int | None
    ratio: int | None
    top_k: int | None
    relu: bool
    scale: float
    softcap: float | None
    pool: tuple[int, int] | None = None

    @classmethod
    def of(cls, c: dict, scale: float) -> Settings:
        """The settings of the ``csa`` config `c`, its logits multiplied by
        `scale`."""
        ix = c["indexer"]
        pool = (ix["candidate_block_size"], ix["candidate_top_k"]) if ix else None
        return cls(
            c["sliding_window_size"],
            c["compress_ratio"],
            ix and ix["top_k"],
            bool(ix and ix["relu"]),
            scale,
            c["softcap"],
            pool,
        )


@dataclass
class Picks:
    """What a ``csa`` call's ports give it of its picks, or ask of them.

    Attributes:
        kept: The compressed latents each token keeps, (T, top_k) by index (-1 for
            none), which its ``index`` port reads; None to pick its own.
        inside: The candidate pool its picks must lie in, (T, k) by block index,
            which its ``candidates`` port reads; None for none.
        pooled: Whether to make a candidate pool, which its ``candidates`` port
            writes.
    """

    kept: torch.Tensor | None = None
    inside: torch.Tensor | None = None
    pooled: bool = False


def rotated(d: int, rope: dict | None) -> int:
    """How many of d dims `rope` rotates, in whole pairs: 0 for none."""
    return 0 if rope is None else int(d * rope["fraction"]) // 2 * 2


def chunks(n: int, per_row: int) -> list[slice]:
    """Split n rows into chunks of at most `CHUNK` elements, `per_row` a row."""
    size = max(1, CHUNK // max(per_row, 1))
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


# --- compressed latents ------------------------------------------------------------


def pool(caches: Caches, steps: Steps, both: torch.Tensor, ratio: int) -> torch.Tensor:
    """Pool the group of `ratio` tokens each token ends, from the projections and
    their gates, both (T, 2d): per channel, under a softmax of the gates, as
    bux pools them. Each token's both go to the state cache first, and every
    group's come from it: zeros for a token that ends none, (T, d)."""
    assert caches.state is not None and caches.state_slots is not None
    caches.state.write(caches.state_slots, both)
    p = steps.positions
    ends = (p + 1) % ratio == 0
    group = p[:, None] - torch.arange(ratio - 1, -1, -1, device=p.device)
    group = group.masked_fill(~ends[:, None], -1)
    kv, gate = caches.state.read(steps.seq[:, None], group).chunk(2, -1)
    return (kv * gate.softmax(1)).sum(1)


def store(caches: Caches, steps: Steps, rows: torch.Tensor, ratio: int) -> None:
    """Write the compressed latents and their indexer keys, side by side in `rows`
    (T, d + i), of the tokens that end a group of `ratio`; none of padding, which
    writes no token latent either (a slot below 0)."""
    assert caches.compressed is not None
    p, d = steps.positions, caches.compressed.cache.shape[-1]
    ends = ((p + 1) % ratio == 0) & (caches.slots >= 0)
    index = torch.where(ends, p // ratio, -1)
    caches.compressed.put(steps.seq, index, rows[:, :d])
    if caches.index is not None:
        caches.index.put(steps.seq, index, rows[:, d:])


# --- the indexer -------------------------------------------------------------------


def ordered(x: torch.Tensor) -> torch.Tensor:
    """Integers in the order of the float32s `x`, as bux ranks them (-0 below
    +0)."""
    u = x.float().contiguous().view(torch.int32).long() & 0xFFFFFFFF
    return torch.where(u >> 31 == 1, ~u & 0xFFFFFFFF, u | (1 << 31))


def top(
    scores: torch.Tensor,
    seen: torch.Tensor,
    k: int,
    inside: torch.Tensor | None = None,
) -> torch.Tensor:
    """The indices of each row's k highest scores among its first `seen`, (rows,),
    and of those `inside` (rows, s) where given: the first of equal ones, as bux
    keeps them, and -1 past them, (rows, k)."""
    s = scores.shape[-1]
    index = torch.arange(s, device=scores.device)
    valid = index < seen[:, None]
    return top_of(scores, valid if inside is None else valid & inside, k)


def top_of(scores: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    """The indices of each row's k highest `valid` scores, the first of equal ones,
    and -1 past them, (rows, k)."""
    s = scores.shape[-1]
    index = torch.arange(s, device=scores.device)
    key = ordered(scores) * (1 << 31) + (s - 1 - index)  # equal: the first wins
    best, at = key.masked_fill(~valid, -1).topk(min(k, s), -1)
    kept = at.masked_fill(best < 0, -1)
    return torch.nn.functional.pad(kept, (0, k - kept.shape[-1]), value=-1)


def candidates(
    scores: torch.Tensor, seen: torch.Tensor, size: int, k: int
) -> torch.Tensor:
    """The candidate pool of rows of `scores` (rows, s) of the compressed latents,
    each row's first `seen` of them its own: the k best blocks of `size` latents,
    each scored by its best latent, and always the block of the newest a row sees;
    by block index, -1 past them, (rows, k), as bux's ``_candidates``."""
    rows, s = scores.shape
    count = -(-s // size)
    index = torch.arange(s, device=scores.device)
    ranked = scores.float().masked_fill(index >= seen[:, None], -torch.inf)
    ranked = torch.nn.functional.pad(ranked, (0, count * size - s), value=-torch.inf)
    best = ranked.view(rows, count, size).amax(-1)
    newest = ((seen - 1) // size).clamp(min=0)[:, None]
    best = best.scatter(
        -1, newest, torch.where(seen[:, None] > 0, torch.inf, -torch.inf)
    )
    return top_of(best, best > -torch.inf, k)


def within(pool: torch.Tensor, s: int, size: int) -> torch.Tensor:
    """Which of s compressed latents (rows, s) are in the candidate pool's blocks of
    `size`, `pool` (rows, k) by block index, -1 for none."""
    count = -(-s // size)
    blocks = torch.zeros(pool.shape[0], count + 1, dtype=torch.bool, device=pool.device)
    blocks.scatter_(1, pool.clamp(min=-1) + 1, True)  # -1 to a column of its own
    return blocks[:, 1:][:, torch.arange(s, device=pool.device) // size]


def every(seen: torch.Tensor, k: int) -> torch.Tensor:
    """Each row's first `seen` indices of k, -1 past them, (rows, k)."""
    index = torch.arange(k, device=seen.device)
    return index.expand(seen.shape[0], k).masked_fill(index >= seen[:, None], -1)


def scores(
    q: torch.Tensor, w: torch.Tensor, keys: torch.Tensor, relu: bool
) -> torch.Tensor:
    """Each query's score of each key, ``sum_j w_j ReLU(q_j . k)`` (the plain dots
    without `relu`), in float32: q (t, h, i), w (t, h), keys (t or 1, s, i), to
    (t, s)."""
    dots = torch.matmul(q.float(), keys.float().transpose(-1, -2))
    if relu:
        dots = dots.relu()
    return torch.matmul(w.float()[:, None], dots)[:, 0]


def select(
    caches: Caches,
    steps: Steps,
    s: Settings,
    q: torch.Tensor,
    w: torch.Tensor,
    picks: Picks | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The compressed latents each token keeps, by index, (T, K): the indexer's
    top-k of those of groups it ends or follows (and in its candidate pool, where
    `picks` give one), from queries q (T, h, i) weighed by w (T, h); every one of
    them while there are no more than top_k. And the candidate pool it makes,
    (T, k) by block index, where `picks` ask for one; else None."""
    assert caches.index is not None and s.ratio is not None and s.top_k
    picks, top_k = picks or Picks(), s.top_k
    seen = (steps.positions + 1) // s.ratio
    most = steps.longest // s.ratio
    if most <= s.top_k and picks.inside is None and not picks.pooled:
        return every(seen, most), None
    kept = seen.new_full((seen.shape[0], top_k), -1)
    pool = None
    if picks.pooled:
        assert s.pool is not None
        pool = seen.new_full((seen.shape[0], s.pool[1]), -1)

    def pick(c: slice, keys: torch.Tensor) -> None:
        """Pick for the tokens `c` among `keys`, (c or 1, latents, i)."""
        sc = scores(q[c], w[c], keys, s.relu)
        inside = None
        if picks.inside is not None and s.pool is not None:
            inside = within(picks.inside[c], sc.shape[-1], s.pool[0])
        kept[c] = top(sc, seen[c], top_k, inside)
        if pool is not None and s.pool is not None:
            pool[c] = candidates(sc, seen[c], *s.pool)

    _each(caches.index, steps, s.ratio, max(q.shape[1], q.shape[2]), most, pick)
    return kept, pool


def _each(
    keys: Paged,
    steps: Steps,
    ratio: int,
    heads: int,
    most: int,
    pick: Callable[[slice, torch.Tensor], None],
) -> None:
    """Call ``pick(tokens, keys)`` on chunks of the batch: its decoded tokens, each
    with its sequence's keys, then each prompt's, sharing theirs."""
    every_latent = torch.arange(most, device=steps.positions.device)
    for c in chunks(steps.decodes, most * heads):
        pick(c, keys.read(steps.seq[c, None], every_latent[None]))
    for start, n, seq, length in steps.prompts:
        own = torch.arange(length // ratio, device=steps.positions.device)
        k = keys.read(torch.full_like(own, seq), own)[None]
        for c in chunks(n, own.shape[0] * heads):
            pick(slice(start + c.start, start + c.stop), k)


# --- attention ---------------------------------------------------------------------


def softmax_attend(
    q: torch.Tensor,
    keys: torch.Tensor,
    valid: torch.Tensor,
    s: Settings,
    sinks: torch.Tensor | None,
) -> torch.Tensor:
    """Each query's softmax over the keys it may see, its sinks beside them, of the
    keys as values, as bux's softmax attention computes it (the logits and their
    softmax in float32): q (t, h, d), keys (t, n, d), valid (t, n), to (t, h, d)."""
    logits = torch.bmm(q.float(), keys.float().transpose(1, 2)).mul_(s.scale)
    if s.softcap is not None:
        logits.div_(s.softcap).tanh_().mul_(s.softcap)
    keep = valid[:, None]
    logits.masked_fill_(~keep, MIN)
    if sinks is None:
        probs = logits.softmax(-1)
    else:
        sink = sinks.float()[None, :, None].expand(*logits.shape[:-1], 1)
        probs = torch.cat([logits, sink], -1).softmax(-1)[..., :-1]
    probs = probs.masked_fill(~keep, 0)
    return torch.bmm(probs.to(keys.dtype), keys)


def attend(
    caches: Caches,
    steps: Steps,
    s: Settings,
    q: torch.Tensor,
    kept: torch.Tensor | None,
    sinks: torch.Tensor | None,
) -> torch.Tensor:
    """Attend each query, q (T, h, d), to its window of token latents and the
    compressed latents it keeps (`kept`, by index, -1 for none), in one softmax;
    (T, h, d)."""
    p, d = steps.positions, q.shape[-1]
    span = min(s.window or steps.longest, steps.longest)
    behind = torch.arange(span - 1, -1, -1, device=p.device)
    n = span + (0 if kept is None else kept.shape[1])
    out = torch.empty_like(q)
    for c in chunks(q.shape[0], n * max(d, q.shape[1])):
        seq, near = steps.seq[c, None], p[c, None] - behind
        keys, valid = caches.window.read(seq, near), near >= 0
        if kept is not None:
            assert caches.compressed is not None
            far = caches.compressed.read(seq, kept[c])
            keys, valid = torch.cat([keys, far], 1), torch.cat([valid, kept[c] >= 0], 1)
        out[c] = softmax_attend(q[c], keys, valid, s, sinks)
    return out


def mix(
    caches: Caches,
    steps: Steps,
    s: Settings,
    q: torch.Tensor,
    latent: torch.Tensor,
    rows: torch.Tensor | None,
    index: tuple[torch.Tensor, torch.Tensor] | None,
    sinks: torch.Tensor | None,
    picks: Picks | None = None,
    attend_by: Callable[..., torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """A step of a ``csa`` call: write its token latents (T, d) and compressed rows
    (T, width, for the tokens that end a group; None for a call that reads another's
    through its ``kv`` port, whose caches `caches` hold), pick the compressed latents
    each query keeps (by the indexer's queries and weights, `index`, or as its ports'
    `picks` say), and attend the queries q (T, h, d) to them and their windows, by
    `attend` or `attend_by` (the plugin's kernel, `bux_vllm.kernels.csa_attend`).
    Returns the outputs, (T, h, d), the picks and the candidate pool made, if any."""
    caches.window.write(caches.slots, latent)
    picks, kept, pool = picks or Picks(), None, None
    if s.ratio is not None:
        if rows is not None:
            store(caches, steps, rows, s.ratio)
        seen = (steps.positions + 1) // s.ratio
        if picks.kept is not None:
            kept = picks.kept
        elif index is not None:
            kept, pool = select(caches, steps, s, *index, picks)
        else:
            kept = every(seen, steps.longest // s.ratio)
    return (attend_by or attend)(caches, steps, s, q, kept, sinks), kept, pool


# --- the layer's own arithmetic ----------------------------------------------------


class IndexerParts(nn.Module):
    """The indexer's parts that are no projection of the input: its keys'
    projection from the compressed latents, its norms and RoPE, as the export names
    them (``indexer.k``, ``indexer.q_norm``, ``indexer.k_norm``)."""

    def __init__(
        self, ix: dict, d: int, rope: dict | None, norm: dict, keys: bool = True
    ):
        super().__init__()
        i = ix["head_dim"]
        self.heads, self.dim, self.weighted = ix["num_heads"], i, ix["weighted"]
        self.k = nn.Linear(d, i, bias=False) if keys else None
        self.q_norm = Norm(ix["norm"], i) if ix["norm"] else None
        self.k_norm = Norm(ix["norm"] or norm, i) if keys else None
        n = min(rotated(d, rope), i) // 2 * 2
        own = {**rope, "fraction": 1.0} if rope else None
        self.rope = Rope(own, n) if own and n else None

    def keys(self, latent: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The keys of compressed latents (T, d), rotated to `positions`, (T, i)."""
        assert self.k is not None and self.k_norm is not None
        k = self.k_norm(self.k(latent))[:, None]
        return (k if self.rope is None else self.rope(k, positions))[:, 0]

    def queries(
        self, q: torch.Tensor, w: torch.Tensor | None, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The query heads, (T, h, i), from their projection, (T, h i), normalized
        and rotated; and their weights, (T, h), from theirs, with the head dims'
        scale."""
        q = q.view(q.shape[0], self.heads, self.dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.rope is not None:
            q = self.rope(q, positions)
        scale = self.dim**-0.5
        if w is None:
            return q, q.new_full((q.shape[0], self.heads), scale)
        return q, w * (scale * self.heads**-0.5)


class Parts(nn.Module):
    """What bux's ``csa`` computes between its projections and its caches: the norms
    and RoPE of its queries, latents and compressed latents, its indexer's keys, and
    the rotation of the head outputs back; named as the export names them
    (``q_a_norm``, ``q_norm``, ``kv_norm``, ``compress_norm``, ``out_norm``,
    ``indexer.*``)."""

    def __init__(self, c: dict, ports: list | None = None):
        super().__init__()
        d, latent, norm = c["head_dim"], c["q_lora_rank"], c["latent_norm"]
        self.ratio, own = c["compress_ratio"], compresses(c, ports)
        self.q_a_norm = Norm(norm, latent) if latent else None
        self.q_norm = Norm(c["qk_norm"], d) if c["qk_norm"] else None
        self.kv_norm = Norm(norm, d)
        self.compress_norm = Norm(norm, d) if own else None
        self.out_norm = Norm(c["output_norm"], d) if c["output_norm"] else None
        self.rope = Rope(c["rope"], d) if c["rope"] else None
        ix = None if reads(ports, "index") else c["indexer"]
        self.indexer = IndexerParts(ix, d, c["rope"], norm, own) if ix else None

    def rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate the first rope.fraction dims of x, (T, h, d)."""
        return x if self.rope is None else self.rope(x, positions)

    def queries(self, q: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The query heads, (T, h, d), normalized and rotated."""
        return self.rotate(q if self.q_norm is None else self.q_norm(q), positions)

    def latents(self, kv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The token latents, (T, d), from their projection: normalized, rotated."""
        return self.rotate(self.kv_norm(kv)[:, None], positions)[:, 0]

    def compressed(self, pooled: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The compressed latents and their indexer keys side by side, (T, d + i),
        from the pooled projections of the groups the tokens end: each latent
        normalized and rotated to its group's first token, and its key."""
        assert self.compress_norm is not None and self.ratio is not None
        latent = self.compress_norm(pooled)
        first = (positions - (self.ratio - 1)).clamp(min=0)
        rotated = self.rotate(latent[:, None], first)[:, 0]
        if self.indexer is None:
            return rotated
        return torch.cat([rotated, self.indexer.keys(latent, first)], -1)

    def unrotate(self, out: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate the head outputs, (T, h, d), back by their query's position, over
        the gain squared, as bux does."""
        if self.rope is None or self.rope.dims == 0:
            return out
        back, n = self.rope(out, -positions), self.rope.dims
        return torch.cat([back[..., :n] / self.rope.gain**2, back[..., n:]], -1)


def down(
    c: dict, ports: list | None = None
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """The projections of a ``csa``'s input that every rank makes by one GEMM: the
    query latent's, the token latent's, the compressed latent's and its gate's
    (none for a layer that reads its ``kv`` port), and the indexer's head weights
    (none for one that reads its ``index`` port), each by its weight's name and
    width; and the parts of their output, the compressed latent's projection and
    gate together (``compress``)."""
    d, latent, ix = c["head_dim"], c["q_lora_rank"], c["indexer"]
    r = c["compress_ratio"] if compresses(c, ports) else None
    ix = None if reads(ports, "index") else ix
    gate = d if r and r > 1 else 0
    weights = ix["num_heads"] if ix and ix["weighted"] else 0
    parts = [("q_a", latent), ("kv", d), ("compress_kv", d if r else 0)]
    parts += [("compress_gate", gate), ("indexer.weights", weights)]
    split = [("q_a", latent), ("kv", d), ("compress", d + gate if r else 0)]
    split += [("indexer.weights", weights)]
    return [(n, w) for n, w in parts if w], [(n, w) for n, w in split if w]


def check_shared(reader: dict, writer: dict, ports: list) -> None:
    """Raise for the compressed latents of the `writer`'s call that a ``csa`` layer,
    `reader` with the ports `ports`, cannot read through its ``kv`` port.

    Raises:
        NotImplementedError: The writer is no ``csa`` that compresses latents of
            the reader's width and ratio, or keeps no indexer keys of the width of
            the indexer the reader runs.
    """
    if writer["type"] != "csa" or writer["compress_ratio"] is None:
        raise NotImplementedError("csa: a kv port from other than a compressing csa")
    dims = [(c["head_dim"], c["compress_ratio"]) for c in (reader, writer)]
    if dims[0] != dims[1]:
        raise NotImplementedError(f"csa: a kv port of {dims[1]} to {dims[0]}")
    ix, theirs = reader["indexer"], writer["indexer"]
    own = ix and not reads(ports, "index")
    if own and (not theirs or theirs["head_dim"] != ix["head_dim"]):
        raise NotImplementedError("csa: an indexer without its writer's keys")


def block_size(tokens: int, ratio: int) -> int:
    """The tokens of a block of compressed latents: the cache's block size, or the
    least multiple of it that `ratio` divides."""
    return math.lcm(tokens, ratio)
