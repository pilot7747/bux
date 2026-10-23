"""A backward split in two: the gradient of the input now, and that of the weights
later, as a zero-bubble pipeline takes them (bux.parallel.scheduled).

Each equation of a backward leads to the input's gradient (`B`), or only to the
weights' (`W`). ``split`` sorts them and gives a function of each part: `B` runs its
equations and hands `W` the values that `W`'s equations read (``Plan.saved``), so
that no equation runs twice. A scan, as the backward of a scan over layers is, splits
the same way into a scan of each part, the `B` scan handing the `W` one its saved
values a pass at a time, stacked. Calls of jaxprs (jit, remat) are inlined first; any
other equation goes whole to the part that needs it, to `B` if both do.

    dx, saved = split.inputs(*free, *rest)    # now, on the critical path
    grads = split.weights(*free, saved)       # later, in a tick that would idle
"""

import itertools
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
from jax.extend import core

# Primitives that call a jaxpr, inlined before the split.
CALLS = frozenset({"pjit", "jit", "closed_call", "core_call", "remat2", "checkpoint"})
Key = Hashable  # a variable of a jaxpr, or a value a split scan stacks for W
Env = dict[Any, Any]  # the value of each key
PARTS = ("B", "W")


@dataclass
class Plan:
    """Which part runs each equation of a jaxpr (None if neither needs it), which of
    its inputs each part reads, and what `B` hands `W` (`saved`): the values `W`
    reads that `B` defines, or that `W` may not read itself."""

    jaxpr: core.Jaxpr
    consts: Sequence[Any]
    parts: list["str | ScanPlan | None"]
    reads: dict[str, set[Key]] = field(default_factory=dict)
    saved: list[Key] = field(default_factory=list)


@dataclass
class ScanPlan:
    """A scan split in two, its `body` planned for one pass. `B` computes the
    `carried` carries and the `ys` outputs `outs["B"]`, `W` the other carries and the
    outputs `outs["W"]`; `stacked` names the saved values the `B` scan stacks."""

    body: Plan
    carried: list[int]
    outs: dict[str, list[int]]
    stacked: list[Key]


@dataclass(frozen=True)
class Split:
    """A backward in two parts: ``inputs(*free, *rest) -> (input grads, saved)`` and
    ``weights(*free, saved) -> weight grads``, `saved` the arrays of `shapes`."""

    inputs: Callable[..., tuple[list[Any], list[Any]]]
    weights: Callable[..., list[Any]]
    shapes: list[jax.ShapeDtypeStruct]


def split(
    backward: Callable[..., Sequence[Any]],
    free: Sequence[Any],
    rest: Sequence[Any],
    weights: int,
) -> Split:
    """Split `backward(*free, *rest)`, whose first `weights` outputs are the weights'
    gradients and the others the input's. `free` and `rest` are flat lists of its
    arguments' shapes; the weights' part may read `free` itself."""
    closed = _inlined(jax.make_jaxpr(lambda *a: list(backward(*a)))(*free, *rest))
    jaxpr, n = closed.jaxpr, len(free)
    outs = range(len(jaxpr.outvars))
    needed = {"B": [i >= weights for i in outs], "W": [i < weights for i in outs]}
    readable = [i < n for i in range(len(jaxpr.invars))]
    plan = _plan(jaxpr, closed.consts, needed, readable)

    def inputs(*args: Any) -> tuple[list[Any], list[Any]]:
        env = _start(plan, dict(zip(jaxpr.invars, args, strict=True)))
        _run(plan, env, "B")
        grads = [_read(env, v) for v in jaxpr.outvars[weights:]]
        return grads, [env[k] for k in plan.saved]

    def weights_of(*args: Any) -> list[Any]:
        *own, saved = args
        env = _start(plan, dict(zip(jaxpr.invars[:n], own, strict=True)))
        env |= dict(zip(plan.saved, saved, strict=True))
        _run(plan, env, "W")
        return [_read(env, v) for v in jaxpr.outvars[:weights]]

    shapes = jax.eval_shape(lambda *a: inputs(*a)[1], *free, *rest)
    return Split(inputs, weights_of, shapes)


# --- Inlining -------------------------------------------------------------------------


def _inlined(closed: core.ClosedJaxpr) -> core.ClosedJaxpr:
    """Return `closed` with every call of a jaxpr inlined, inside scans too."""

    def run(*args: Any) -> list[Any]:
        return _eval_inlined(closed.jaxpr, closed.consts, args)

    return jax.make_jaxpr(run)(*(v.aval for v in closed.jaxpr.invars))


def _eval_inlined(
    jaxpr: core.Jaxpr, consts: Sequence[Any], args: Sequence[Any]
) -> list[Any]:
    env = dict(zip(jaxpr.constvars, consts, strict=True))
    env |= dict(zip(jaxpr.invars, args, strict=True))
    for eqn in jaxpr.eqns:
        ins = [_read(env, v) for v in eqn.invars]
        if eqn.primitive.name in CALLS:
            inner = eqn.params["jaxpr"]
            if not isinstance(inner, core.ClosedJaxpr):
                inner = core.ClosedJaxpr(inner, ())
            outs = _eval_inlined(inner.jaxpr, inner.consts, ins)
        elif eqn.primitive.name == "scan":
            outs = _rescanned(eqn, ins)
        else:
            outs = _bind(eqn, ins)
        env |= dict(zip(eqn.outvars, outs, strict=True))
    return [_read(env, v) for v in jaxpr.outvars]


def _rescanned(eqn: core.JaxprEqn, ins: list[Any]) -> list[Any]:
    """Run the scan of `eqn` on `ins`, with its body inlined."""
    body, (c, n) = eqn.params["jaxpr"], _counts(eqn)
    consts, init, xs = ins[:c], ins[c : c + n], ins[c + n :]

    def step(carry: list, x: list) -> tuple[list, list]:
        outs = _eval_inlined(body.jaxpr, body.consts, [*consts, *carry, *x])
        return outs[:n], outs[n:]

    carry, ys = _scan(eqn, step, init, xs)
    return [*carry, *ys]


def _counts(eqn: core.JaxprEqn) -> tuple[int, int]:
    """Return how many constants and carries the scan of `eqn` takes."""
    consts, carry, _ = eqn.params["ft_in"].unpack()
    return len(consts), len(carry)


def _scan(
    eqn: core.JaxprEqn, step: Callable, init: list[Any], xs: Any
) -> tuple[Any, Any]:
    p = eqn.params
    return jax.lax.scan(
        step, init, xs, length=p["length"], reverse=p["reverse"], unroll=p["unroll"]
    )


def _bind(eqn: core.JaxprEqn, ins: list[Any]) -> list[Any]:
    """Run equation `eqn` on `ins`, as jax.core.eval_jaxpr does."""
    params = eqn.primitive.get_bind_params(eqn.params)
    with eqn.ctx.manager:
        out = eqn.primitive.bind(*ins, **params)
    return out if eqn.primitive.multiple_results else [out]


def _read(env: Env, v: Any) -> Any:
    return v.val if isinstance(v, core.Literal) else env[v]


# --- Planning -------------------------------------------------------------------------


def _plan(
    jaxpr: core.Jaxpr,
    consts: Sequence[Any],
    needed: dict[str, list[bool]],
    readable: list[bool],
) -> Plan:
    """Plan `jaxpr`, each part computing the outputs `needed[part]` marks, `W`
    reading itself the inputs `readable` marks: by a pass from the outputs back,
    each equation goes to the part that needs it, to `B` if both do, but a scan
    both need, which splits (``_plan_scan``)."""
    plan = Plan(jaxpr, consts, [None] * len(jaxpr.eqns))
    need = {p: _vars(jaxpr.outvars, needed[p]) for p in PARTS}
    by_b: set[Key] = set()  # what B defines
    for i in reversed(range(len(jaxpr.eqns))):
        eqn = jaxpr.eqns[i]
        wanted = [any(v in need[p] for v in eqn.outvars) for p in PARTS]
        if all(wanted) and eqn.primitive.name == "scan":
            scan = plan.parts[i] = _plan_scan(eqn, i, need)
            by_b |= {eqn.outvars[k] for k in [*scan.carried, *scan.outs["B"]]}
            by_b |= set(scan.stacked)
        elif wanted[0]:
            plan.parts[i] = "B"
            need["B"] |= _vars(eqn.invars)
            by_b |= set(eqn.outvars)
        elif wanted[1]:
            plan.parts[i] = "W"
            need["W"] |= _vars(eqn.invars)
    ins = {*jaxpr.invars, *jaxpr.constvars}
    plan.reads = {p: need[p] & ins for p in PARTS}
    own = {v for v, r in zip(jaxpr.invars, readable, strict=True) if r}
    own |= set(jaxpr.constvars)
    unreadable = (ins - own) | by_b
    plan.saved = _ordered(jaxpr, need["W"] & unreadable)
    return plan


def _plan_scan(eqn: core.JaxprEqn, index: int, need: dict[str, set[Key]]) -> ScanPlan:
    """Plan a scan whose outputs both parts need, and add to `need` what each of its
    parts reads. `B` carries what it needs of the carries, and any carry its pass
    reads; `W` carries the rest."""
    body, (c, n) = eqn.params["jaxpr"], _counts(eqn)
    invars, outs = body.jaxpr.invars, eqn.outvars
    ys = range(n, len(outs))
    carried = {k for k in range(n) if outs[k] in need["B"]}
    while True:
        needed = {
            "B": [
                k in carried if k < n else outs[k] in need["B"]
                for k in range(len(outs))
            ],
            "W": [k not in carried if k < n else False for k in range(len(outs))],
        }
        for k in ys:
            needed["W"][k] = outs[k] in need["W"] and not needed["B"][k]
        readable = [
            not (c <= k < c + n and k - c in carried) for k in range(len(invars))
        ]
        sub = _plan(body.jaxpr, body.consts, needed, readable)
        more = {k for k in range(n) if invars[c + k] in sub.reads["B"]} - carried
        if not more:
            break
        carried |= more
    for k, v in enumerate(eqn.invars):
        if c <= k < c + n:
            need["B" if k - c in carried else "W"] |= _vars([v])
            continue
        for p in PARTS:
            if invars[k] in sub.reads[p]:
                need[p] |= _vars([v])
    stacked: list[Key] = [("stacked", index, i) for i in range(len(sub.saved))]
    need["W"] |= set(stacked)
    outs_by = {p: [k for k in ys if needed[p][k]] for p in PARTS}
    return ScanPlan(sub, sorted(carried), outs_by, stacked)


def _vars(atoms: Sequence[Any], keep: Sequence[bool] | None = None) -> set[Key]:
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


def _start(plan: Plan, env: Env) -> Env:
    return env | dict(zip(plan.jaxpr.constvars, plan.consts, strict=True))


def _run(plan: Plan, env: Env, part: str) -> None:
    """Run the equations of `part` in `plan`, reading and writing `env`."""
    for eqn, p in zip(plan.jaxpr.eqns, plan.parts, strict=True):
        if p == part:
            outs = _bind(eqn, [_read(env, v) for v in eqn.invars])
            env |= dict(zip(eqn.outvars, outs, strict=True))
        elif isinstance(p, ScanPlan):
            (_run_scan_b if part == "B" else _run_scan_w)(eqn, p, env)


def _run_scan_b(eqn: core.JaxprEqn, scan: ScanPlan, env: Env) -> None:
    """Run the `B` scan of `scan`: its carries and outputs, and each pass's saved
    values, stacked."""
    body, (c, _) = scan.body, _counts(eqn)
    invars, outvars = body.jaxpr.invars, body.jaxpr.outvars
    given = _given(eqn, scan, env, "B")
    carry_in = [invars[c + k] for k in scan.carried]

    def step(carry: list, xs: list) -> tuple[list, tuple[list, list]]:
        inner = _start(body, given | dict(zip(carry_in, carry, strict=True)))
        inner |= dict(zip(_xs_read(eqn, scan, "B"), xs, strict=True))
        _run(body, inner, "B")
        out = [_read(inner, outvars[k]) for k in scan.outs["B"]]
        return [_read(inner, outvars[k]) for k in scan.carried], (
            out,
            [inner[k] for k in body.saved],
        )

    init = [_read(env, eqn.invars[c + k]) for k in scan.carried]
    xs = [_read(env, eqn.invars[_index(eqn, v)]) for v in _xs_read(eqn, scan, "B")]
    carry, (ys, stacked) = _scan(eqn, step, init, xs)
    env |= {eqn.outvars[k]: v for k, v in zip(scan.carried, carry, strict=True)}
    env |= {eqn.outvars[k]: v for k, v in zip(scan.outs["B"], ys, strict=True)}
    env |= dict(zip(scan.stacked, stacked, strict=True))


def _run_scan_w(eqn: core.JaxprEqn, scan: ScanPlan, env: Env) -> None:
    """Run the `W` scan of `scan`, from the values the `B` scan stacked."""
    body, (c, n) = scan.body, _counts(eqn)
    invars, outvars = body.jaxpr.invars, body.jaxpr.outvars
    given = _given(eqn, scan, env, "W")
    own = [k for k in range(n) if k not in scan.carried]
    carry_in = [invars[c + k] for k in own]
    xs_in = _xs_read(eqn, scan, "W")

    def step(carry: list, xs: tuple[list, list]) -> tuple[list, list]:
        read, saved = xs
        inner = _start(body, given | dict(zip(carry_in, carry, strict=True)))
        inner |= dict(zip(xs_in, read, strict=True))
        inner |= dict(zip(body.saved, saved, strict=True))
        _run(body, inner, "W")
        out = [_read(inner, outvars[k]) for k in scan.outs["W"]]
        return [_read(inner, outvars[k]) for k in own], out

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
    return eqn.params["jaxpr"].jaxpr.invars.index(body_var)
