"""Fit the ladders of the tutorial "Old against new at the same serving cost"
(docs/tutorials/old-vs-new.md) and extrapolate each family to its target.

It reads every finished run under RUNS whose name starts with a ladder's, the search
trials too (but a halving rung's, which trains fewer steps), and keeps the best of each
family and width, as each rung's point: the rung at the best learning rate the search
found. A metric of a family is fit as

    L(N) = E + A N^-alpha,

N the parameters a token goes through besides the embeddings (an MoE's routed
experts by top_k / num_experts), E shared by the families, which read the same data.
The fit minimizes the Huber loss of log L, as Hoffmann et al. (2022) do. The range
beside each prediction is the leave-one-rung-out fits'. The learning rates of each
rung's searched trials within 3% of its best fit a parabola in log lr, whose vertex is
the rung's best, and the vertices a power law in N, lr = c N^-beta, which gives the
rungs not searched.

    python scripts/old_vs_new/fit.py runs --old 4096 --new 3072 --plot ladder.png
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flax import nnx
from scipy.optimize import least_squares

from bux.architecture import Model, num_params
from bux.lang import compile, ir

HERE = Path(__file__).resolve().parents[2] / "examples" / "experiments" / "old_vs_new"
FAMILIES = {
    "old": ("old_ladder", HERE / "old_2022.bux"),
    "new": ("new_ladder", HERE / "new_2026.bux"),
}
METRICS = ("objective", "code_bpb", "repo_bpb", "web_bpb")
#: The trials a rung's parabola in log lr fits: those within 3% of its best. The
#: bottom is flat to within the runs' noise, so its five best alone place it anywhere.
NEAR = 1.03


@dataclass
class Point:
    """A finished run: its family, width, active parameters, lr and final metrics."""

    family: str
    width: int
    params: float
    lr: float
    metrics: dict[str, float]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", type=Path)
    p.add_argument("--old", type=int, default=4096, help="the 2022 target's d_model")
    p.add_argument("--new", type=int, default=3072, help="the 2026 target's d_model")
    p.add_argument("--plot", type=Path, help="draw the fits into this image")
    a = p.parse_args()
    points = read(a.runs)
    best = rungs(points)
    targets = {
        "old": active(FAMILIES["old"][1], a.old),
        "new": active(FAMILIES["new"][1], a.new),
    }
    print(table(best))
    print(learning_rates(points))
    for metric in METRICS:
        print(predictions(best, metric, targets))
    if a.plot:
        plot(best, targets, a.plot)


def read(runs: Path) -> list[Point]:
    """Every finished run of a ladder under `runs`, with its last value of each metric."""
    points = []
    for folder in sorted(runs.iterdir()):
        status = folder / "status.json"
        if not status.exists() or json.loads(status.read_text()).get("state") != "done":
            continue
        run = ir.Run.model_validate_json((folder / "run.json").read_text())
        family = next(
            (f for f, (stem, _) in FAMILIES.items() if run.name.startswith(stem + "[")),
            None,
        )
        if family is None or "steps=" in run.name:  # a halving rung, cut short
            continue
        last: dict[str, float] = {}
        for line in (folder / "metrics.jsonl").open():
            last.update(json.loads(line))
        terms = run.train.objective.terms if run.train.objective else ()
        if all(t.metric in last for t in terms):
            last["objective"] = sum(t.weight * last[t.metric] for t in terms)
        lr = run.train.optimizer.lr
        points.append(Point(family, run.arch.d_model, active_of(run.arch), lr, last))
    return points


def active(file: Path, width: int) -> float:
    """The active parameters besides the embeddings of `file`'s model at `width`."""
    return active_of(
        compile(file.read_text(), filename=str(file), overrides=[f"d_model = {width}"])
    )


def active_of(arch: ir.Architecture) -> float:
    """`arch`'s parameters a token goes through, besides the embedding and the head."""
    total = num_params(nnx.eval_shape(lambda: Model(arch, rngs=nnx.Rngs(0))))
    idle = 0
    for layer in arch.layers():
        c = layer.config
        if c.type == "moe":
            mats = 3 if c.expert.type == "gated_mlp" else 2
            idle += (
                (c.num_experts - c.top_k)
                * mats
                * arch.d_model
                * c.expert.intermediate_size
            )
    return float(total - idle - 2 * (arch.vocab_size or 0) * arch.d_model)


def rungs(points: list[Point]) -> dict[str, list[Point]]:
    """The best run of each family and width by its objective, by width."""
    best: dict[tuple[str, int], Point] = {}
    for p in points:
        key = (p.family, p.width)
        if "objective" in p.metrics and (
            key not in best or p.metrics["objective"] < best[key].metrics["objective"]
        ):
            best[key] = p
    out: dict[str, list[Point]] = {f: [] for f in FAMILIES}
    for (family, _), p in sorted(best.items()):
        out[family].append(p)
    return out


def fit(best: dict[str, list[Point]], metric: str) -> dict[str, float]:
    """E, and A and alpha of each family, for `metric`."""
    families = [f for f in FAMILIES if best[f]]
    n = {f: np.array([p.params for p in best[f]]) for f in families}
    y = {f: np.array([p.metrics[metric] for p in best[f]]) for f in families}

    def residuals(x: np.ndarray) -> np.ndarray:
        e = math.exp(x[0])
        out = []
        for i, f in enumerate(families):
            a, alpha = x[1 + 2 * i], x[2 + 2 * i]
            out.append(np.log(e + np.exp(a - alpha * np.log(n[f]))) - np.log(y[f]))
        return np.concatenate(out)

    floor = min(v.min() for v in y.values())
    starts = [
        [math.log(floor * s), *[v for _ in families for v in (a0, al)]]
        for s in (0.3, 0.6, 0.9)
        for a0 in (2.0, 5.0)
        for al in (0.2, 0.4)
    ]
    sol = min(
        (least_squares(residuals, x0, loss="huber", f_scale=1e-3) for x0 in starts),
        key=lambda s: s.cost,
    )
    out = {"E": math.exp(sol.x[0])}
    for i, f in enumerate(families):
        out[f"A_{f}"], out[f"alpha_{f}"] = math.exp(sol.x[1 + 2 * i]), sol.x[2 + 2 * i]
    return out


def predict(params: dict[str, float], family: str, n: float) -> float:
    return params["E"] + params[f"A_{family}"] * n ** -params[f"alpha_{family}"]


def predictions(
    best: dict[str, list[Point]], metric: str, targets: dict[str, float]
) -> str:
    """A line for each family: its fit and its prediction at its target, with the
    range of the leave-one-rung-out fits."""
    if not all(metric in p.metrics for ps in best.values() for p in ps):
        return f"{metric}: not in every run"
    whole = fit(best, metric)
    lines = [f"{metric}: E = {whole['E']:.4f}"]
    for f, n in targets.items():
        if not best[f]:
            continue
        others = []
        for i in range(len(best[f])):
            if len(best[f]) > 2:
                fewer = {**best, f: best[f][:i] + best[f][i + 1 :]}
                others.append(predict(fit(fewer, metric), f, n))
        spread = f" [{min(others):.4f}, {max(others):.4f}]" if others else ""
        lines.append(
            f"  {f}: A = {whole[f'A_{f}']:.3g}, alpha = {whole[f'alpha_{f}']:.3f}; "
            f"at {n / 1e9:.2f}B: {predict(whole, f, n):.4f}{spread}"
        )
    return "\n".join(lines)


def table(best: dict[str, list[Point]]) -> str:
    rows = [
        "family  d_model  active   lr        " + "  ".join(f"{m:>9s}" for m in METRICS)
    ]
    for f, ps in best.items():
        for p in ps:
            vals = "  ".join(f"{p.metrics.get(m, float('nan')):9.4f}" for m in METRICS)
            rows.append(
                f"{f:6s}  {p.width:7d}  {p.params / 1e6:6.1f}M  {p.lr:.2e}  {vals}"
            )
    return "\n".join(rows)


def learning_rates(points: list[Point]) -> str:
    """Each searched rung's best lr, a parabola's vertex in log lr, and the power law
    of each family through them."""
    lines = [
        "learning rates (vertex of a parabola in log lr, over each rung's trials):"
    ]
    for f in FAMILIES:
        vertices = []
        for width in sorted({p.width for p in points if p.family == f}):
            trials = [
                p
                for p in points
                if p.family == f and p.width == width and "objective" in p.metrics
            ]
            if len({p.lr for p in trials}) < 3:
                continue
            x = np.log([p.lr for p in trials])
            y = np.array([p.metrics["objective"] for p in trials])
            near = y <= NEAR * y.min()  # the bottom, where a parabola holds
            c2, c1, _ = np.polyfit(x[near], y[near], 2)
            best = math.exp(-c1 / (2 * c2)) if c2 > 0 else math.exp(x[np.argmin(y)])
            vertices.append((trials[0].params, best))
            lines.append(
                f"  {f} d_model={width}: {len(trials)} trials, best {best:.3g}"
            )
        if len(vertices) >= 2:
            beta, logc = np.polyfit(
                np.log([v[0] for v in vertices]), np.log([v[1] for v in vertices]), 1
            )
            lines.append(f"  {f}: lr = {math.exp(logc):.3g} N^{beta:.3f}")
    return "\n".join(lines)


#: The plot's colors in light and dark mode: the surface, ink, muted ink, gridlines,
#: and each family's (the validated categorical palette's first two slots).
THEMES = {
    "light": (
        "#fcfcfb",
        "#0b0b0b",
        "#898781",
        "#e1e0d9",
        {"old": "#2a78d6", "new": "#eb6834"},
    ),
    "dark": (
        "#1a1a19",
        "#f0efec",
        "#898781",
        "#2c2c2a",
        {"old": "#3987e5", "new": "#d95926"},
    ),
}
LABELS = {"old": "2022 dense", "new": "2026 hybrid MoE"}


def plot(best: dict[str, list[Point]], targets: dict[str, float], out: Path) -> None:
    """Draw each family's rungs and fit of the objective, solid over the rungs and
    dashed beyond them, to a star at its target: in `out`, and in dark colors beside
    it, `<stem>-dark<suffix>`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = fit(best, "objective")
    for theme, (surface, ink, muted, grid, colors) in THEMES.items():
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150, facecolor=surface)
        ax.set_facecolor(surface)
        for f, ps in best.items():
            if not ps:
                continue
            n = np.array([p.params for p in ps])
            seen = np.geomspace(n.min() / 1.3, n.max(), 100)
            ahead = np.geomspace(n.max(), targets[f], 100)
            line = {"color": colors[f], "linewidth": 2, "solid_capstyle": "round"}
            ax.plot(
                seen, [predict(params, f, g) for g in seen], label=LABELS[f], **line
            )
            ax.plot(
                ahead,
                [predict(params, f, g) for g in ahead],
                linestyle=(0, (4, 3)),
                **line,
            )
            dots = {
                "color": colors[f],
                "edgecolors": surface,
                "linewidths": 2,
                "zorder": 3,
            }
            ax.scatter(n, [p.metrics["objective"] for p in ps], s=64, **dots)
            y = predict(params, f, targets[f])
            ax.scatter([targets[f]], [y], marker="*", s=260, **dots)
            higher = y >= max(predict(params, g, targets[g]) for g in best if best[g])
            ax.annotate(
                f"{y:.3f}",
                (targets[f], y),
                xytext=(8, 5 if higher else -5),  # apart, where the two are close
                textcoords="offset points",
                va="center",
                color=ink,
                fontsize=9,
            )
        ax.set_xscale("log")
        ax.set_xlabel("active parameters besides the embeddings", color=muted)
        ax.set_ylabel("objective (bits per byte)", color=muted)
        ax.tick_params(colors=muted, which="both")
        for side in ax.spines.values():
            side.set_color(grid)
        ax.grid(True, which="major", color=grid, linewidth=0.8)
        legend = ax.legend(frameon=False)
        for text in legend.get_texts():
            text.set_color(ink)
        fig.tight_layout()
        dest = (
            out if theme == "light" else out.with_name(f"{out.stem}-dark{out.suffix}")
        )
        fig.savefig(dest, facecolor=surface)
        plt.close(fig)
        print(f"plot: {dest}")


if __name__ == "__main__":
    main()
