"""bux's pass over XLA's schedule of a program, after XLA's latency-hiding scheduler:
each reduction of a gradient (a reduce-scatter or an all-reduce) is computed and
started as soon as what it needs is, and the program waits for it before it starts
the `LAG`-th next.

XLA's scheduler places a collective's done just before its result is first read, and
with one collective in flight each start just above the done before it, and the ops
that feed only the start just above that. A gradient's reduction is read where the
step adds it up, after the last layer's backward pass (a microbatch's sum, the
optimizer's update), so the scheduler puts every weight gradient's GEMM there, each
before its reduction, one after another: the GEMMs keep their inputs until then, and
the reductions run beside nothing but each other's GEMMs. The pass moves each GEMM
and its reduction up to where the layer's backward pass makes the GEMM's inputs. On
8 B300s it takes Qwen3-30B-A3B under FSDP at Megatron-Bridge's batch from 40.0k to
42.1k tokens a second a GPU (docs/training.md, "Where the gradients' reductions
start").

It is the pass `NAME` of bux.passes: ``passes.use(reorder.NAME)``.
"""

import bisect
from collections.abc import Sequence
from typing import Any, NamedTuple

from bux import jax_internals, passes

Op = jax_internals.Opcode
NAME = "bux_early_reductions"
_REDUCTIONS = frozenset({Op.kReduceScatter, Op.kAllReduce})
# What never moves up with a reduction: what communicates, copies to or from the host,
# or runs other computations by its own order, and the computation's parameters.
_STAYS = frozenset(
    {
        *(Op.kAsyncStart, Op.kAsyncUpdate, Op.kAsyncDone, Op.kCopyStart, Op.kCopyDone),
        *(Op.kAllGatherStart, Op.kAllGatherDone, Op.kAllReduceStart, Op.kAllReduceDone),
        *(Op.kCollectivePermuteStart, Op.kCollectivePermuteDone, Op.kAllGather),
        *(Op.kAllReduce, Op.kReduceScatter, Op.kAllToAll, Op.kRaggedAllToAll),
        *(Op.kCollectivePermute, Op.kSend, Op.kSendDone, Op.kRecv, Op.kRecvDone),
        *(Op.kInfeed, Op.kOutfeed, Op.kWhile, Op.kConditional, Op.kParameter),
        *(Op.kRngGetAndUpdateState, Op.kAfterAll),
    }
)
# The reductions in flight at once: a done waits for the start of the one this many
# further on.
LAG = 2


@passes.register(NAME, stage=passes.AFTER_SCHEDULING, key=f"lag={LAG}")
def reordered(serialized: bytes) -> bytes | None:
    """Return the scheduled HloModuleProto `serialized` with its reductions started
    early, or None where none moves."""
    return _reordered(jax_internals.hlo_module(serialized))


def _reordered(module: Any) -> bytes | None:
    schedule = module.schedule()
    if schedule is None:
        return None
    moved = False
    for computation in module.make_nonfusion_computations():
        sequence = schedule.sequence(computation)
        order = early(sequence) if sequence else None
        if order is not None:
            schedule.set_sequence(computation, order)
            moved = True
    if not moved:
        return None
    schedule.verify()
    module.set_schedule(schedule)
    return module.as_serialized_hlo_module_proto()


class _Move(NamedTuple):
    """A reduction's start, moved up with the ops that feed only it (`chain`, in
    order) to just after the instruction at position `after`."""

    after: int
    chain: list[Any]
    start: Any


def early(sequence: Sequence[Any]) -> list[Any] | None:
    """Return `sequence`, a computation's instructions in XLA's order, with each
    reduction's start, and what computes only for it, moved up to just after what
    they need, and its done to just before the start of the `LAG`-th reduction moved
    further than it; None where nothing moves."""
    at = {inst.name: i for i, inst in enumerate(sequence)}
    moves = [
        m for s in sequence if _starts_reduction(s) and (m := _move(s, sequence, at))
    ]
    if not moves:
        return None
    key = {inst.name: (i, 0, 0) for i, inst in enumerate(sequence)}
    for move in moves:
        for inst in move.chain:
            key[inst.name] = min(key[inst.name], (move.after, 2, at[inst.name]))
    afters = sorted({m.after for m in moves})
    for move in moves:
        later = bisect.bisect_right(afters, move.after) + LAG - 1
        if later < len(afters):
            done = move.start.users()[0].name
            key[done] = min(key[done], (afters[later], 1, at[done]))
    order = sorted(sequence, key=lambda inst: key[inst.name])
    return order if _in_order(order) else None


def _starts_reduction(inst: Any) -> bool:
    if inst.opcode == Op.kAllReduceStart:
        return True
    return inst.opcode == Op.kAsyncStart and (
        inst.async_wrapped_root().opcode in _REDUCTIONS
    )


def _ends_reduction(inst: Any) -> bool:
    return any(_starts_reduction(x) for x in inst.operands()) and inst.opcode in (
        Op.kAsyncDone,
        Op.kAllReduceDone,
    )


def _move(start: Any, sequence: Sequence[Any], at: dict[str, int]) -> _Move | None:
    """Return where `start` and what computes only for it can go, None where they
    are there already (but for the dones of reductions before them) or the start
    has more than its done to wait for it."""
    if len(start.users()) != 1:
        return None
    chain = _feeding(start)
    inside = {inst.name for inst in chain}
    needed = [at[d] for inst in chain for d in _needs(inst) if d not in inside]
    after = max(needed, default=-1)
    later = [x for x in chain if at[x.name] > after]
    between = sequence[after + 1 : at[start.name]]
    if all(x.name in inside or _ends_reduction(x) for x in between):
        return None
    return _Move(after, sorted(later, key=lambda inst: at[inst.name]), start)


def _feeding(start: Any, most: int = 64) -> list[Any]:
    """Return `start` and the ops before it whose results only it reads, by way of
    others of them: the GEMM of a weight's gradient, its operands' quantizers, the
    bitcasts and converts between."""
    chain, todo = {start.name: start}, [start]
    while todo and len(chain) < most:
        for op in todo.pop().operands():
            if op.name not in chain and _movable(op) and _only_for(op, chain):
                chain[op.name] = op
                todo.append(op)
    return list(chain.values())


def _movable(op: Any) -> bool:
    if op.opcode in _STAYS or op.control_predecessors():
        return False
    effects = op.opcode == Op.kCustomCall and "has_side_effect=true" in op.to_string()
    return not effects


def _only_for(op: Any, chain: dict[str, Any]) -> bool:
    return all(user.name in chain for user in op.users())


def _needs(inst: Any) -> list[str]:
    """The instructions that must run before `inst`: its operands and control
    predecessors."""
    return [x.name for x in (*inst.operands(), *inst.control_predecessors())]


def _in_order(order: Sequence[Any]) -> bool:
    """Whether every instruction of `order` comes after all that it needs."""
    seen: set[str] = set()
    for inst in order:
        if any(name not in seen for name in _needs(inst)):
            return False
        seen.add(inst.name)
    return True
