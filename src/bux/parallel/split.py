"""A backward split in two, as a zero-bubble pipeline runs it (bux.parallel.scheduled):
`B`, the gradient of the input, now, and `W`, that of the weights, later.

    dx, saved = two.inputs(*shared, *b_only)  # B, on the critical path
    dw = two.weights(*shared, saved)          # W, in a tick that would idle

``split`` traces the backward to a jaxpr and walks it back from the outputs, giving
each equation to the part that needs it: to `B` if the input's gradient does, else to
`W`. `B` hands `W` what `W` reads but neither computes nor takes (``Plan.saved``), so
that no equation runs twice. A scan or a cond that both need splits into one of each,
as the backward of the scan over a stage's layers does: the `B` scan stacks what each
of its passes saves, and the `W` scan reads it a pass at a time. Calls of jaxprs (jit,
remat) are inlined first.
"""

import functools
import itertools
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from bux import jax_internals
from bux.jax_internals import core

Key = Hashable  # a variable of a jaxpr, or a value a split scan stacks for W
Env = dict[Any, Any]  # the value of each key
PARTS = ("B", "W")


@dataclass
class Plan:
    """How a jaxpr splits: which part runs each of its equations (`parts`), "B", "W",
    a ScanPlan or a CondPlan for one that splits, or None where neither needs it; the
    inputs each part reads (`reads`); and what `B` hands `W` (`saved`): the values
    `W` reads that `B` computes, or that `W` is not given."""

    jaxpr: core.Jaxpr
    consts: Sequence[Any]
    parts: list["str | ScanPlan | CondPlan | None"]
    reads: dict[str, set[Key]] = field(default_factory=dict)
    saved: list[Key] = field(default_factory=list)


@dataclass
class ScanPlan:
    """A scan split into a scan of each part, its `body` planned for one pass. `B`
    carries the carries `carried` and gives the outputs `outs["B"]`, `W` the other
    carries and the outputs `outs["W"]`; `stacked` names what each pass of `B` saves
    for `W`, as the `B` scan stacks it."""

    body: Plan
    carried: list[int]
    outs: dict[str, list[int]]
    stacked: list[Key]


@dataclass
class CondPlan:
    """A cond split into a cond of each part, each of its `branches` planned. `B`
    gives the outputs `outs["B"]` and `W` those of `outs["W"]`; `saved[k]` names what
    branch k's `B` hands its `W`, which the `B` cond gives for every branch, zeros but
    for the one it ran."""

    branches: list[Plan]
    outs: dict[str, list[int]]
    saved: list[list[Key]]


@dataclass(frozen=True)
class Split:
    """A backward in two parts: ``inputs(*shared, *b_only) -> (the input's gradients,
    saved)`` and ``weights(*shared, saved) -> the weights' gradients``, `saved` the
    arrays of `saved_shapes`."""

    inputs: Callable[..., tuple[list[Any], list[Any]]]
    weights: Callable[..., list[Any]]
    saved_shapes: list[jax.ShapeDtypeStruct]


def split(
    backward: Callable[..., Sequence[Any]],
    shared: Sequence[Any],
    b_only: Sequence[Any],
    weight_grads: int,
) -> Split:
    """Split `backward(*shared, *b_only)`, whose first `weight_grads` outputs are the
    weights' gradients and the others the input's. `shared` and `b_only` are flat
    lists of its arguments' shapes: `B` takes both, and `W` takes `shared` again."""
    traced = jax.make_jaxpr(lambda *a: list(backward(*a)))(*shared, *b_only)
    closed = _inlined(traced)
    jaxpr, n = closed.jaxpr, len(shared)
    outs = range(len(jaxpr.outvars))
    needed = {
        "B": [i >= weight_grads for i in outs],
        "W": [i < weight_grads for i in outs],
    }
    given_to_w = [i < n for i in range(len(jaxpr.invars))]
    plan = _plan(jaxpr, closed.consts, needed, given_to_w)

    def inputs(*args: Any) -> tuple[list[Any], list[Any]]:
        env = _run_part(plan, "B", dict(zip(jaxpr.invars, args, strict=True)))
        grads = [_read(env, v) for v in jaxpr.outvars[weight_grads:]]
        return grads, [env[k] for k in plan.saved]

    def weights(*args: Any) -> list[Any]:
        *again, saved = args
        given = dict(zip(jaxpr.invars[:n], again, strict=True))
        env = _run_part(plan, "W", given | dict(zip(plan.saved, saved, strict=True)))
        return [_read(env, v) for v in jaxpr.outvars[:weight_grads]]

    saved_shapes = jax.eval_shape(lambda *a: inputs(*a)[1], *shared, *b_only)
    return Split(inputs, weights, saved_shapes)


# --- Inlining -------------------------------------------------------------------------


def _inlined(closed: core.ClosedJaxpr) -> core.ClosedJaxpr:
    """Return `closed` with every call of a jaxpr inlined, inside scans too."""

    def run(*args: Any) -> list[Any]:
        return _eval_inlined(closed.jaxpr, closed.consts, args)

    return jax.make_jaxpr(run)(*(v.aval for v in closed.jaxpr.invars))


def _eval_inlined(
    jaxpr: core.Jaxpr, consts: Sequence[Any], args: Sequence[Any]
) -> list[Any]:
    """Return `jaxpr` evaluated on `args`, each equation bound anew but the calls,
    scans and conds, whose jaxprs it evaluates so too: so that its trace inlines
    every call."""
    env = dict(zip(jaxpr.constvars, consts, strict=True))
    env |= dict(zip(jaxpr.invars, args, strict=True))
    for eqn in jaxpr.eqns:
        ins = [_read(env, v) for v in eqn.invars]
        if jax_internals.calls_a_jaxpr(eqn):
            inner = jax_internals.called_jaxpr(eqn)
            outs = _eval_inlined(inner.jaxpr, inner.consts, ins)
        elif jax_internals.is_scan(eqn):
            outs = _rescanned(eqn, ins)
        elif jax_internals.is_cond(eqn):
            outs = _reconded(eqn, ins)
        else:
            outs = jax_internals.bind(eqn, ins)
        env |= dict(zip(eqn.outvars, outs, strict=True))
    return [_read(env, v) for v in jaxpr.outvars]


def _reconded(eqn: core.JaxprEqn, ins: list[Any]) -> list[Any]:
    """Run the cond of `eqn` on `ins`, with its branches inlined."""
    index, *ops = ins

    def branch(b: core.ClosedJaxpr) -> Callable[..., list[Any]]:
        return lambda *ops: _eval_inlined(b.jaxpr, b.consts, ops)

    branches = jax_internals.cond_branches(eqn)
    return jax.lax.switch(index, [branch(b) for b in branches], *ops)


def _rescanned(eqn: core.JaxprEqn, ins: list[Any]) -> list[Any]:
    """Run the scan of `eqn` on `ins`, with its body inlined."""
    body, (c, n) = jax_internals.scan_body(eqn), _counts(eqn)
    consts, init, xs = ins[:c], ins[c : c + n], ins[c + n :]

    def step(carry: list, x: list) -> tuple[list, list]:
        outs = _eval_inlined(body.jaxpr, body.consts, [*consts, *carry, *x])
        return outs[:n], outs[n:]

    carry, ys = _scan(eqn, step, init, xs)
    return [*carry, *ys]


def _counts(eqn: core.JaxprEqn) -> tuple[int, int]:
    """Return how many constants and carries the scan of `eqn` takes."""
    return jax_internals.scan_counts(eqn)


def _scan(
    eqn: core.JaxprEqn, step: Callable, init: list[Any], xs: Any
) -> tuple[Any, Any]:
    return jax.lax.scan(step, init, xs, **jax_internals.scan_options(eqn))


def _read(env: Env, v: Any) -> Any:
    """Return the value of `v`: a literal's own, or a variable's in `env`."""
    return v.val if isinstance(v, core.Literal) else env[v]


# --- Planning -------------------------------------------------------------------------


def _plan(
    jaxpr: core.Jaxpr,
    consts: Sequence[Any],
    needed: dict[str, list[bool]],
    given_to_w: list[bool],
) -> Plan:
    """Plan `jaxpr`, each part giving the outputs `needed[part]` marks, and `W` given
    the inputs `given_to_w` marks. From the outputs back, each equation goes to the
    part that needs its outputs, to `B` if both do, but a scan or a cond both need,
    which splits (``_plan_scan``, ``_plan_cond``)."""
    plan = Plan(jaxpr, consts, [None] * len(jaxpr.eqns))
    need = {p: _vars(jaxpr.outvars, needed[p]) for p in PARTS}
    by_b: set[Key] = set()  # what B defines
    for i in reversed(range(len(jaxpr.eqns))):
        eqn = jaxpr.eqns[i]
        wanted = [any(v in need[p] for v in eqn.outvars) for p in PARTS]
        if all(wanted) and jax_internals.is_scan(eqn):
            scan = plan.parts[i] = _plan_scan(eqn, i, need)
            by_b |= {eqn.outvars[k] for k in [*scan.carried, *scan.outs["B"]]}
            by_b |= set(scan.stacked)
        elif all(wanted) and jax_internals.is_cond(eqn):
            cond = plan.parts[i] = _plan_cond(eqn, i, need)
            by_b |= {eqn.outvars[k] for k in cond.outs["B"]}
            by_b |= set(itertools.chain(*cond.saved))
        elif wanted[0]:  # by B alone, or by both: B hands W what it gives
            plan.parts[i] = "B"
            need["B"] |= _vars(eqn.invars)
            by_b |= set(eqn.outvars)
        elif wanted[1]:
            plan.parts[i] = "W"
            need["W"] |= _vars(eqn.invars)
    ins = {*jaxpr.invars, *jaxpr.constvars}
    plan.reads = {p: need[p] & ins for p in PARTS}
    w_has = {v for v, given in zip(jaxpr.invars, given_to_w, strict=True) if given}
    w_has |= set(jaxpr.constvars)
    w_lacks = (ins - w_has) | by_b  # W computes no input of its own, nor what B does
    plan.saved = _ordered(jaxpr, need["W"] & w_lacks)
    return plan


def _plan_scan(eqn: core.JaxprEqn, index: int, need: dict[str, set[Key]]) -> ScanPlan:
    """Plan a scan whose outputs both parts need, and add to `need` what each of its
    parts reads. `B` carries what it needs of the carries, and any carry its pass
    reads; `W` carries the rest."""
    body, (c, n) = jax_internals.scan_body(eqn), _counts(eqn)
    invars, outs = body.jaxpr.invars, eqn.outvars
    carried = {k for k in range(n) if outs[k] in need["B"]}
    while True:  # until B carries every carry its passes read
        needed = _scan_needed(outs, n, carried, need)
        given_to_w = [
            not (c <= k < c + n and k - c in carried) for k in range(len(invars))
        ]
        sub = _plan(body.jaxpr, body.consts, needed, given_to_w)
        more = {k for k in range(n) if invars[c + k] in sub.reads["B"]} - carried
        if not more:
            break
        carried |= more
    _add_scan_reads(eqn, sub, carried, need)
    stacked: list[Key] = [("stacked", index, i) for i in range(len(sub.saved))]
    need["W"] |= set(stacked)
    ys = range(n, len(outs))
    outs_by = {p: [k for k in ys if needed[p][k]] for p in PARTS}
    return ScanPlan(sub, sorted(carried), outs_by, stacked)


def _scan_needed(
    outs: Sequence[Any], n: int, carried: set[int], need: dict[str, set[Key]]
) -> dict[str, list[bool]]:
    """Return which of a scan's outputs, its `n` carries and then its stacked ones,
    each part's scan gives: `B` the carries in `carried` and the outputs it needs,
    `W` the other carries and the outputs it alone needs."""
    b = [k in carried if k < n else outs[k] in need["B"] for k in range(len(outs))]
    w = [
        k not in carried if k < n else outs[k] in need["W"] and not b[k]
        for k in range(len(outs))
    ]
    return {"B": b, "W": w}


def _add_scan_reads(
    eqn: core.JaxprEqn, body: Plan, carried: set[int], need: dict[str, set[Key]]
) -> None:
    """Add to `need` what each part's scan of `eqn` reads: the first value of each
    carry it carries, and the constants and scanned inputs its `body` reads."""
    c, n = _counts(eqn)
    for k, v in enumerate(eqn.invars):
        if c <= k < c + n:
            need["B" if k - c in carried else "W"] |= _vars([v])
            continue
        for p in PARTS:
            if body.jaxpr.invars[k] in body.reads[p]:
                need[p] |= _vars([v])


def _plan_cond(eqn: core.JaxprEqn, index: int, need: dict[str, set[Key]]) -> CondPlan:
    """Plan a cond whose outputs both parts need, and add to `need` what each part's
    cond reads: the index, the operands its branches read, and for `W` what `B`
    saves."""
    outs = eqn.outvars
    needed = {"B": [v in need["B"] for v in outs]}
    needed["W"] = [v in need["W"] and not b for v, b in zip(outs, needed["B"])]
    branches = [
        _plan(b.jaxpr, b.consts, needed, [True] * len(b.jaxpr.invars))
        for b in jax_internals.cond_branches(eqn)
    ]
    saved: list[list[Key]] = [
        [("cond", index, k, i) for i in range(len(b.saved))]
        for k, b in enumerate(branches)
    ]
    need["B"] |= _vars(eqn.invars[:1])
    need["W"] |= _vars(eqn.invars[:1]) | set(itertools.chain(*saved))
    for k, v in enumerate(eqn.invars[1:]):
        for p in PARTS:
            if _branches_read(branches, k, p):
                need[p] |= _vars([v])
    by = {p: [k for k in range(len(outs)) if needed[p][k]] for p in PARTS}
    return CondPlan(branches, by, saved)


def _branches_read(branches: list[Plan], k: int, part: str) -> bool:
    """Return whether `part` of any of a cond's `branches` reads its operand k."""
    return any(b.jaxpr.invars[k] in b.reads[part] for b in branches)


def _vars(atoms: Sequence[Any], keep: Sequence[bool] | None = None) -> set[Key]:
    """Return the variables of `atoms` that `keep` marks, not the literals."""
    keep = [True] * len(atoms) if keep is None else keep
    return {
        v
        for v, k in zip(atoms, keep, strict=True)
        if k and not isinstance(v, core.Literal)
    }


def _ordered(jaxpr: core.Jaxpr, keys: set[Key]) -> list[Key]:
    """Return `keys` in the order the jaxpr defines them, the stacked ones last."""
    defined = (e.outvars for e in jaxpr.eqns)
    order = [*jaxpr.constvars, *jaxpr.invars, *itertools.chain(*defined)]
    stacked = sorted((k for k in keys if isinstance(k, tuple)), key=str)
    return [v for v in order if v in keys] + stacked


# --- Running --------------------------------------------------------------------------


def _run_part(plan: Plan, part: str, env: Env) -> Env:
    """Return `env`, the values of some of `plan`'s variables, with its constants and
    what `part` computes."""
    env = env | dict(zip(plan.jaxpr.constvars, plan.consts, strict=True))
    _run(plan, env, part)
    return env


def _run(plan: Plan, env: Env, part: str) -> None:
    """Run the equations of `part` in `plan`, reading and writing `env`, and its part
    of each scan and cond that splits."""
    for eqn, p in zip(plan.jaxpr.eqns, plan.parts, strict=True):
        if p == part:
            outs = jax_internals.bind(eqn, [_read(env, v) for v in eqn.invars])
            env |= dict(zip(eqn.outvars, outs, strict=True))
        elif isinstance(p, ScanPlan) and part == "B":
            _run_scan_b(eqn, p, env)
        elif isinstance(p, ScanPlan):
            _run_scan_w(eqn, p, env)
        elif isinstance(p, CondPlan) and part == "B":
            _run_cond_b(eqn, p, env)
        elif isinstance(p, CondPlan):
            _run_cond_w(eqn, p, env)


def _outputs(plan: Plan, env: Env, positions: list[int]) -> list[Any]:
    """Return the values in `env` of `plan`'s outputs at `positions`."""
    return [_read(env, plan.jaxpr.outvars[i]) for i in positions]


def _run_cond_b(eqn: core.JaxprEqn, cond: CondPlan, env: Env) -> None:
    """Run the `B` cond of `cond`: its outputs, and what each branch saves for `W`,
    zeros for the branches it did not run."""
    index, ops = _read(env, eqn.invars[0]), _operands(eqn, cond, env, "B")
    each = range(len(cond.branches))
    shapes = [
        jax.eval_shape(functools.partial(_branch_b, cond, k), ops)[1] for k in each
    ]
    branches = [functools.partial(_branch_b_padded, cond, k, shapes) for k in each]
    outs, saved = jax.lax.switch(index, branches, ops)
    env |= {eqn.outvars[i]: v for i, v in zip(cond.outs["B"], outs, strict=True)}
    for keys, values in zip(cond.saved, saved, strict=True):
        env |= dict(zip(keys, values, strict=True))


def _branch_b(
    cond: CondPlan, k: int, ops: dict[int, Any]
) -> tuple[list[Any], list[Any]]:
    """Run the `B` part of branch k on the cond's operands `ops`: its outputs, and
    what it saves for `W`."""
    plan = cond.branches[k]
    inner = _run_part(plan, "B", _operands_in(plan, ops))
    return _outputs(plan, inner, cond.outs["B"]), [inner[s] for s in plan.saved]


def _branch_b_padded(
    cond: CondPlan,
    k: int,
    shapes: list[list[jax.ShapeDtypeStruct]],
    ops: dict[int, Any],
) -> tuple[list[Any], list[list[Any]]]:
    """Run the `B` part of branch k: its outputs, and what each branch saves, zeros
    of `shapes` for the others."""
    outs, own = _branch_b(cond, k, ops)
    saved = [own if j == k else _zeros(shapes[j]) for j in range(len(shapes))]
    return outs, saved


def _run_cond_w(eqn: core.JaxprEqn, cond: CondPlan, env: Env) -> None:
    """Run the `W` cond of `cond`, each branch from what its `B` part saved."""
    index, ops = _read(env, eqn.invars[0]), _operands(eqn, cond, env, "W")
    saved = [[env[k] for k in keys] for keys in cond.saved]
    each = range(len(cond.branches))
    branches = [functools.partial(_branch_w, cond, k) for k in each]
    outs = jax.lax.switch(index, branches, ops, saved)
    env |= {eqn.outvars[i]: v for i, v in zip(cond.outs["W"], outs, strict=True)}


def _branch_w(
    cond: CondPlan, k: int, ops: dict[int, Any], saved: list[list[Any]]
) -> list[Any]:
    """Run the `W` part of branch k on the cond's operands `ops`, from what its `B`
    part saved, `saved[k]`."""
    plan = cond.branches[k]
    given = _operands_in(plan, ops) | dict(zip(plan.saved, saved[k], strict=True))
    return _outputs(plan, _run_part(plan, "W", given), cond.outs["W"])


def _operands(
    eqn: core.JaxprEqn, cond: CondPlan, env: Env, part: str
) -> dict[int, Any]:
    """Return the operands of the cond that a branch's `part` reads, by position."""
    return {
        k: _read(env, v)
        for k, v in enumerate(eqn.invars[1:])
        if _branches_read(cond.branches, k, part)
    }


def _operands_in(plan: Plan, ops: dict[int, Any]) -> Env:
    """Return a cond's operands `ops` by the variables of the branch `plan`."""
    return {plan.jaxpr.invars[i]: v for i, v in ops.items()}


def _zeros(shapes: list[jax.ShapeDtypeStruct]) -> list[Any]:
    return [jnp.zeros(a.shape, a.dtype) for a in shapes]


def _run_scan_b(eqn: core.JaxprEqn, scan: ScanPlan, env: Env) -> None:
    """Run the `B` scan of `scan`: the carries it carries, its outputs, and what each
    pass saves for `W`, stacked."""
    body, (c, _) = scan.body, _counts(eqn)
    given = _given(eqn, scan, env, "B")
    carry_in = [body.jaxpr.invars[c + k] for k in scan.carried]
    xs_in = _xs_read(eqn, scan, "B")

    def step(carry: list, xs: list) -> tuple[list, tuple[list, list]]:
        inputs = given | dict(zip(carry_in, carry, strict=True))
        inner = _run_part(body, "B", inputs | dict(zip(xs_in, xs, strict=True)))
        ys = _outputs(body, inner, scan.outs["B"])
        saved = [inner[k] for k in body.saved]
        return _outputs(body, inner, scan.carried), (ys, saved)

    init = [_read(env, eqn.invars[c + k]) for k in scan.carried]
    xs = [_read(env, eqn.invars[_index(eqn, v)]) for v in xs_in]
    carry, (ys, stacked) = _scan(eqn, step, init, xs)
    env |= {eqn.outvars[k]: v for k, v in zip(scan.carried, carry, strict=True)}
    env |= {eqn.outvars[k]: v for k, v in zip(scan.outs["B"], ys, strict=True)}
    env |= dict(zip(scan.stacked, stacked, strict=True))


def _run_scan_w(eqn: core.JaxprEqn, scan: ScanPlan, env: Env) -> None:
    """Run the `W` scan of `scan`: the other carries and its outputs, each pass from
    what the `B` scan stacked."""
    body, (c, n) = scan.body, _counts(eqn)
    given = _given(eqn, scan, env, "W")
    own = [k for k in range(n) if k not in scan.carried]
    carry_in = [body.jaxpr.invars[c + k] for k in own]
    xs_in = _xs_read(eqn, scan, "W")

    def step(carry: list, xs: tuple[list, list]) -> tuple[list, list]:
        read, saved = xs
        inputs = given | dict(zip(carry_in, carry, strict=True))
        inputs |= dict(zip(xs_in, read, strict=True))
        inputs |= dict(zip(body.saved, saved, strict=True))
        inner = _run_part(body, "W", inputs)
        return _outputs(body, inner, own), _outputs(body, inner, scan.outs["W"])

    init = [_read(env, eqn.invars[c + k]) for k in own]
    xs = [_read(env, eqn.invars[_index(eqn, v)]) for v in xs_in]
    carry, ys = _scan(eqn, step, init, (xs, [env[k] for k in scan.stacked]))
    env |= {eqn.outvars[k]: v for k, v in zip(own, carry, strict=True)}
    env |= {eqn.outvars[k]: v for k, v in zip(scan.outs["W"], ys, strict=True)}


def _given(eqn: core.JaxprEqn, scan: ScanPlan, env: Env, part: str) -> Env:
    """Return the constants of the scan that `part`'s pass reads, by body variable."""
    invars, (c, _) = scan.body.jaxpr.invars, _counts(eqn)
    return {
        invars[k]: _read(env, eqn.invars[k])
        for k in range(c)
        if invars[k] in scan.body.reads[part]
    }


def _xs_read(eqn: core.JaxprEqn, scan: ScanPlan, part: str) -> list[Any]:
    """Return the body variables of the scanned inputs that `part`'s pass reads."""
    invars, first = scan.body.jaxpr.invars, sum(_counts(eqn))
    return [v for v in invars[first:] if v in scan.body.reads[part]]


def _index(eqn: core.JaxprEqn, body_var: Any) -> int:
    return jax_internals.scan_body(eqn).jaxpr.invars.index(body_var)
