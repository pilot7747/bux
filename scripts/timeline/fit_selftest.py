"""fit.py's self-test: runs drawn from known laws, fit back.

A baseline (an lr scan at d512, a token scan t5-t80 at d512, four rungs, an axis
file's base arm and seeds 1-3 as its replicates; every family has seeds at d512) and three variants:
``const13``, the baseline with 1.3 times its parameters, at two rungs;
``growing``, whose multiplier grows from 1.1 at d512 as N^0.15, at three rungs with
its best lr at the scan's edge; and ``moe``, twice the baseline's parameters a
token goes through, 0.6 of them active and 3.5 times as many in all, with a data
term 0.8 times the baseline's, at four rungs and a token scan. Once without noise,
read from RESULT lines (the fit must give back the laws), once with the noise of
real runs, read from run directories (the truth must sit in the bootstrap's range).
"""

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from fit_law import Law

WIDTHS = (512, 768, 1024, 1280)
TOKENS = {512: 503e6, 768: 1.7e9, 1024: 4.03e9, 1280: 7.86e9}
LAWS = {
    "code_bpb": Law(0.28, 70.0, 0.31, 18.0, 0.29),
    "repo_bpb": Law(0.245, 75.0, 0.32, 22.0, 0.30),
    "web_bpb": Law(0.38, 12.0, 0.17, 6.0, 0.22),
}
WEIGHTS = {"code_bpb": 0.5, "repo_bpb": 0.3, "web_bpb": 0.2}
LR_BEST, PENALTY = 2e-3, 0.02  # relative loss per squared octave off the best lr
GROW = (1.1, 0.15)  # growing's multiplier at d512, and its exponent in N
#: family: (stem, arm, active and total parameters over the baseline's, rungs, lrs at
#: d512, best lr, token scan)
FAMILIES = {
    "baseline": ("baseline", "", 1.0, 1.0, WIDTHS, (1e-3, 2e-3, 4e-3, 8e-3), 2e-3, True),
    "synth/const13": ("synth", "const13", 1.0, 1.0, WIDTHS[:2], (1e-3, 2e-3, 4e-3), 2e-3, False),
    "synth/growing": ("synth", "growing", 1.0, 1.0, WIDTHS[:3], (1e-3, 2e-3, 4e-3), 4e-3, False),
    "synth/moe": ("synth", "moe", 0.6, 3.5, WIDTHS, (1e-3, 2e-3, 4e-3), 2e-3, True),
}  # fmt: skip
TARGETS = {
    "baseline": {"N_active": 6.4e9, "N_total": 6.86e9, "flops_per_token": 4.2e10},
    "synth/const13": {"N_active": 6.4e9, "N_total": 6.86e9, "flops_per_token": 4.2e10},
    "synth/growing": {"N_active": 6.4e9, "N_total": 6.86e9, "flops_per_token": 4.2e10},
    "synth/moe": {"N_active": 4.0e9, "N_total": 23e9, "flops_per_token": 2.6e10},
}


def n_base(width: int) -> float:
    """The baseline's non-embedding parameters: 12 d^2 in each of d / 64 layers."""
    return 12.0 * (width // 64) * width**2


def true_law(family: str, metric: str) -> Law:
    """Each family's law of `metric` in its own active parameters."""
    b = LAWS[metric]
    if family == "synth/const13":
        return Law(b.E, b.A * 1.3**-b.alpha, b.alpha, b.B, b.beta)
    if family == "synth/growing":
        c, g = GROW
        a = b.A * c**-b.alpha * n_base(512) ** (g * b.alpha)
        return Law(b.E, a, b.alpha * (1 + g), b.B, b.beta)
    if family == "synth/moe":
        return Law(b.E, b.A * 2.0**-b.alpha, b.alpha, 0.8 * b.B, b.beta)
    return b


def runs_of(family: str) -> list[tuple[int, int, float]]:
    """Each run of `family`: its rung, its tokens per baseline parameter, its lr."""
    _, _, _, _, widths, lrs, best, scan = FAMILIES[family]
    out = [(512, 20, lr) for lr in lrs] + [(w, 20, best) for w in widths[1:]]
    return out + ([(512, t, best) for t in (5, 10, 40, 80)] if scan else [])


def rows(sigma: float, seed: int) -> list[dict]:
    """Every run's name and final metrics, each loss off by `sigma`: the baseline's
    d512 and d768 again as an axis file's base arm and its d768 and d1024 as seeds
    1-3, and every family's d512 as seeds 1-3 at its best lr, as the campaign ran."""
    rng = np.random.default_rng(seed)
    runs = [(f, *FAMILIES[f][:2], *run, 0) for f in FAMILIES for run in runs_of(f)]
    runs += [("baseline", "synth", "base", w, 20, LR_BEST, 0) for w in (512, 768)]
    runs += [
        ("baseline", "baseline", "", w, 20, LR_BEST, s)
        for w in WIDTHS[1:3]
        for s in (1, 2, 3)
    ]
    runs += [
        (f, *FAMILIES[f][:2], 512, 20, FAMILIES[f][6], s)
        for f in FAMILIES
        for s in (1, 2, 3)
    ]
    out = []
    for family, stem, arm, w, t, lr, s in runs:
        active, best = FAMILIES[family][2], FAMILIES[family][6]
        n, d = active * n_base(w), TOKENS[w] * t / 20
        off = 1 + PENALTY * math.log2(lr / best) ** 2
        noise = np.exp(sigma * rng.standard_normal(len(LAWS)))
        metrics = {
            m: float(true_law(family, m).loss(n, d)) * off * e
            for m, e in zip(LAWS, noise, strict=True)
        }
        labels = [x for x in (arm, f"d{w}", f"t{t}" if t != 20 else "") if x]
        labels += [f"optimizer.lr={lr:g}"] if w == 512 and t == 20 else []
        labels += [f"seed={s}"] if s else []
        out.append(
            {"name": f"{stem}[{', '.join(labels)}]", "lr": lr, "tokens": d, **metrics}
        )
    return out


def lines(rows: list[dict]) -> list[str]:
    """The runs as RESULT lines, named by their directories."""
    out = []
    for i, r in enumerate(rows):
        stem, labels = r["name"].rstrip("]").split("[")
        metrics = {k: v for k, v in r.items() if k in LAWS}
        row = {
            "dir": f"{slug(stem, labels.split(', '))}-{i:010x}",
            "state": "done",
            "lr": r["lr"],
            "budget": r["tokens"],
        }
        out.append("RESULT " + json.dumps(row | metrics))
    return out


def run_dirs(folder: Path, rows: list[dict]) -> None:
    """The runs as bux writes them: status.json, run.json and metrics.jsonl."""
    for i, r in enumerate(rows):
        stem, labels = r["name"].rstrip("]").split("[")
        d = folder / f"{slug(stem, labels.split(', '))}-{i:010x}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "status.json").write_text(json.dumps({"state": "done"}))
        (d / "run.json").write_text(
            json.dumps({"name": r["name"], "train": {"optimizer": {"lr": r["lr"]}}})
        )
        early = {"step": 1, "tokens": r["tokens"] / 2, **{m: 2 * r[m] for m in LAWS}}
        final = {"step": 2, "tokens": r["tokens"], "lr": 0.0, **{m: r[m] for m in LAWS}}
        (d / "metrics.jsonl").write_text(
            json.dumps(early) + "\n" + json.dumps(final) + "\n"
        )


def slug(stem: str, labels: list[str]) -> str:
    """A run's directory name, as ``bux.train.trainer.directory`` makes it."""
    return re.sub(r"[^A-Za-z0-9_.=+-]+", "_", f"{stem}[{', '.join(labels)}]").strip("_")


def counts() -> dict:
    """The counts the compiler would give, for fit_counts' cache: the base arm's
    those of the baseline it repeats."""
    out = {
        f"synth|base|{w}": {"active": n_base(w), "arch": "baseline"} for w in (512, 768)
    }
    for family, (stem, arm, active, total, widths, *_) in FAMILIES.items():
        for w in widths:
            n = n_base(w)
            out[f"{stem}|{arm}|{w}"] = {"d_model": w, "total": total * n + 50304 * w, "nonembed": total * n,
                                        "active": active * n, "moe": arm == "moe", "hybrid": False,
                                        "arch": family}  # fmt: skip
    return out


def write(folder: Path, sigma: float, seed: int, dirs: bool = False) -> list[str]:
    """Write the runs (as RESULT lines, or as run directories if `dirs`), the
    counts and the targets to `folder`; return the arguments of fit.py that read
    them."""
    (folder / "files").mkdir(parents=True, exist_ok=True)
    runs = rows(sigma, seed)
    (folder / "results.txt").write_text("\n".join(lines(runs)) + "\n")
    run_dirs(folder / "runs", runs)
    (folder / "counts.json").write_text(json.dumps(counts()))
    (folder / "targets.json").write_text(json.dumps(TARGETS))
    source = folder / ("runs" if dirs else "results.txt")
    return [str(source), "--out", str(folder / "fit"), "--cache", str(folder / "counts.json"),
            "--targets", str(folder / "targets.json"), "--files", str(folder / "files")]  # fmt: skip


def truth(metric: str) -> dict[str, dict]:
    """What the true laws say of each family, as fit_analysis says it of fits."""
    from fit_analysis import at_target, pct
    from fit_targets import target

    laws = {f: true_law(f, metric) for f in FAMILIES}
    targets = {
        f: target(_raw(v), v["flops_per_token"], FAMILIES[f][1] == "moe", None, "")
        for f, v in TARGETS.items()
    }
    out = {}
    for f, law in laws.items():
        active, widths = FAMILIES[f][2], FAMILIES[f][4]
        parent = "" if f == "baseline" else "baseline"
        q = {"law": law.__dict__, **at_target(f, parent, laws, targets, "baseline")}
        q["rungs"] = {}
        for w in widths:
            n, d = active * n_base(w), TOKENS[w]
            y = float(law.loss(n, d))
            q["rungs"][w] = {"mult": laws["baseline"].n_for(y, d) / n,
                             "delta": pct(y, float(laws["baseline"].loss(n_base(w), d)))}  # fmt: skip
        out[f] = q
    return out


def _raw(v: dict) -> dict:
    return {"n_active": v["N_active"], "n_total": v["N_total"]}


def checks(fit: dict, metric: str, tolerance: dict) -> list[dict]:
    """Each quantity the truth and the fit say, side by side, and whether they agree
    to within `tolerance` (relative, by kind)."""
    true, got = truth(metric), fit["metrics"][metric]
    rows = []
    for f, t in true.items():
        q, boot = got["quantities"][f], got["bootstrap_5_95"].get(f, {})
        wanted = [("law/E", "E")] if f == "baseline" else []
        wanted += [
            ("limited/130B/loss", "loss"),
            ("limited/1T/loss", "loss"),
            ("unlimited/20tpp/speedup", "speedup"),
        ]
        wanted += [("limited/1T/delta", "delta")] if f != "baseline" else []
        wanted += [("law/alpha", "law")] if f == "synth/growing" else []
        wanted += [("law/B", "law"), ("law/beta", "law")] if f == "synth/moe" else []
        if f != "baseline":
            wanted += [(f"rungs/{w}/mult", "mult") for w in t["rungs"]] + [
                ("mult_target", "mult")
            ]
        for path, kind in wanted:
            want, have = _at(t, path), _at(q, path)
            err = have - want if kind == "delta" else 100 * (have / want - 1)
            lo, hi = boot.get(path, (math.nan, math.nan))
            inside = lo <= want <= hi
            rows.append({"metric": metric, "family": f, "quantity": path, "truth": want, "fit": have,
                         "error": err, "boot": (lo, hi), "in range": inside,
                         "ok": abs(err) <= tolerance[kind] or inside})  # fmt: skip
    return rows


def _at(q: Any, path: str) -> float:
    for k in path.split("/"):
        q = q[int(k)] if k.isdigit() and int(k) in q else q[k]
    return float(q)


#: Tolerances by kind, in percent (a delta's in points): without noise, and with
#: the noise of real runs, where the bootstrap's range also passes.
EXACT = {"E": 0.1, "loss": 0.01, "speedup": 0.1, "law": 0.1, "mult": 0.1, "delta": 0.01}
NOISY = {"E": 5, "loss": 1, "speedup": 30, "law": 10, "mult": 5, "delta": 0.5}


def run(out: Path, reps: int, sigma: float = 0.002) -> int:
    """Run both tests and print what each recovered. Return 0 if the fit gives
    back every number without noise, and with noise agrees (within tolerance or
    the bootstrap's range) on `PASS` of them, as a 90% range should."""
    import fit

    rows = []
    for noise, tolerance in ((0.0, EXACT), (sigma, NOISY)):
        args = write(out / f"noise-{noise:g}", noise, seed=1, dirs=bool(noise))
        args += ["--reps", str(reps if noise else 0), "--noise", str(sigma)]
        fit.ARGV = args
        report = fit.run(fit.parser().parse_args(args))
        result = json.loads(report["json"].read_text())
        for metric in LAWS:
            rows += [{"noise": noise, **r} for r in checks(result, metric, tolerance)]
    print(table(rows))
    exact = [r["ok"] for r in rows if not r["noise"]]
    noisy = [r["ok"] for r in rows if r["noise"]]
    covered = [r["in range"] for r in rows if r["noise"]]
    print(f"\nwithout noise: {sum(exact)} of {len(exact)} agree; with noise {sigma:g}: "
          f"{sum(noisy)} of {len(noisy)} agree, the truth in the bootstrap's 5-95% "
          f"range for {sum(covered)} (error: in percent, a delta's in points)")  # fmt: skip
    return 0 if all(exact) and sum(noisy) >= PASS * len(noisy) else 1


PASS = 0.85  # the share of the noisy checks that must agree


def table(rows: list[dict]) -> str:
    head = "noise  metric    family         quantity                 truth        fit    error  bootstrap 5-95%        ok"
    out = [head]
    for r in rows:
        lo, hi = r["boot"]
        rng = f"[{lo:.4g}, {hi:.4g}]" if math.isfinite(lo) else ""
        out.append(f"{r['noise']:<5g}  {r['metric']:<8s}  {r['family']:<13s}  {r['quantity']:<23s}  "
                   f"{r['truth']:9.4g}  {r['fit']:9.4g}  {r['error']:+7.3f}  {rng:<21s}  "
                   f"{'yes' if r['ok'] else 'NO'}")  # fmt: skip
    return "\n".join(out)
