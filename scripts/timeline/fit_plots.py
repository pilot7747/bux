"""The fit's plots, in light and dark, in the style of the tutorial "Old against new
at the same serving cost": loss against N per family with its extrapolation to a
star at its target, loss against training GPU-hours (unlimited data), and bars of
each target against the root's on the same tokens (limited data)."""

from pathlib import Path
from typing import Any

import numpy as np
from fit_analysis import LIMITED, Analysis, tokens_label

#: Surface, ink, muted ink, gridlines, and the categorical slots in their order.
THEMES = {
    "light": ("#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9",
              ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]),
    "dark": ("#1a1a19", "#f0efec", "#898781", "#2c2c2a",
             ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]),
}  # fmt: skip
STAR = LIMITED[0]  # the tokens of the star: the limited regime's first budget
LABELS = {"objective": "objective", "code_bpb": "code files", "repo_bpb": "repositories, 16k",
          "web_bpb": "web text"}  # fmt: skip


def chosen(
    a: Analysis, targets: dict, root: str, wanted: list[str] | None
) -> list[str]:
    """The root and up to 7 families: `wanted`, or those with the most rungs and then
    the largest multipliers."""
    if wanted:
        return [root] + [f for f in wanted if f in a.laws and f != root][:7]

    def rank(f: str) -> tuple:
        gain = a.q[f].get("mult_target") or a.q[f].get("mult") or 0
        return -int(a.fams[f].ladder.sum()), -gain

    return [root] + sorted((f for f in a.laws if f != root), key=rank)[:7]


def draw(out: Path, name: str, paint: Any, size: tuple = (7.5, 4.6)) -> list[Path]:
    """Draw `paint(ax, theme)` in each theme to ``<name>.png`` and ``<name>-dark.png``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for theme, (surface, ink, muted, grid, colors) in THEMES.items():
        fig, ax = plt.subplots(figsize=size, dpi=150, facecolor=surface)
        ax.set_facecolor(surface)
        paint(ax, (surface, ink, muted, grid, colors))
        ax.tick_params(colors=muted, which="both")
        for side in ax.spines.values():
            side.set_color(grid)
        ax.grid(True, which="major", color=grid, linewidth=0.8)
        legend = ax.legend(frameon=False, fontsize=8)
        for text in legend.get_texts():
            text.set_color(ink)
        fig.tight_layout()
        dest = out / (f"{name}.png" if theme == "light" else f"{name}-dark.png")
        fig.savefig(dest, facecolor=surface)
        plt.close(fig)
        written.append(dest)
    return written


def ladder(a: Analysis, families: list[str], targets: dict, out: Path) -> list[Path]:
    """Loss against N: each family's rungs, its law (solid over the rungs), and
    dashed to a star at its target on `STAR` tokens, whose loss its legend says."""

    def paint(ax: Any, theme: tuple) -> None:
        surface, _, muted, _, colors = theme
        for f, color in zip(families, colors, strict=False):
            if f in a.laws:
                _family_curve(ax, a, f, targets.get(f), color, surface)
        ax.set_xscale("log")
        ax.set_xlabel(
            "parameters a token goes through, besides the embeddings", color=muted
        )
        ax.set_ylabel(f"{LABELS.get(a.metric, a.metric)} (bits per byte)", color=muted)

    return draw(out, f"ladder-{a.metric}", paint)


def _family_curve(
    ax: Any, a: Analysis, f: str, t: Any, color: str, surface: str
) -> None:
    """Draw family `f`'s rungs, its law over them, and on to its target's star."""
    fam, law = a.fams[f], a.laws[f]
    n, d, y = fam.n[fam.ladder], fam.d[fam.ladder], fam.y[fam.ladder]
    if not len(n):
        return
    star = None if t is None else float(law.loss(t.n_active, STAR))
    label = f if star is None else f"{f}: {star:.3f} at {t.n_active / 1e9:.3g}B"
    slope = np.polyfit(np.log(n), np.log(d), 1)[0] if len(set(n)) > 1 else 1.0
    seen = np.geomspace(n.min() / 1.3, n.max(), 80)
    tokens = d[np.argmax(n)] * (seen / n.max()) ** slope
    ax.plot(
        seen,
        law.loss(seen, tokens),
        color=color,
        linewidth=2,
        label=label,
        solid_capstyle="round",
    )
    ax.scatter(n, y, s=48, color=color, edgecolors=surface, linewidths=2, zorder=3)
    if star is None:
        return
    ahead = np.geomspace(n.max(), t.n_active, 80)
    share = np.log(ahead / n.max()) / max(np.log(t.n_active / n.max()), 1e-9)
    last = np.log(d[np.argmax(n)])
    tokens = np.exp(last + share * (np.log(STAR) - last))
    ax.plot(
        ahead, law.loss(ahead, tokens), color=color, linewidth=2, linestyle=(0, (4, 3))
    )
    ax.scatter([t.n_active], [star], s=200, marker="*", color=color, edgecolors=surface,
               linewidths=1.5, zorder=4)  # fmt: skip


def hours(
    a: Analysis, families: list[str], targets: dict, root: str, out: Path
) -> list[Path]:
    """Unlimited data: each target's loss against its training GPU-hours, the root
    target's at 20 tokens a parameter and at 1T tokens marked."""
    r = targets[root]
    ends = [r.hours(20 * r.n_active) / 10, r.hours(1e12) * 30]
    grid = np.geomspace(*ends, 120)

    def paint(ax: Any, theme: tuple) -> None:
        surface, ink, muted, _, colors = theme
        for f, color in zip(families, colors, strict=False):
            if (t := targets.get(f)) is not None and f in a.laws:
                ax.plot(
                    grid,
                    a.laws[f].loss(t.n_active, t.tokens(grid)),
                    color=color,
                    linewidth=2,
                    label=f,
                )
        base = a.laws[root]
        for d, text in ((20 * r.n_active, "20 tokens/param"), (1e12, "1T tokens")):
            h, loss = r.hours(d), float(base.loss(r.n_active, d))
            ax.scatter(
                [h],
                [loss],
                s=48,
                color=colors[0],
                edgecolors=surface,
                linewidths=2,
                zorder=3,
            )
            ax.annotate(
                text,
                (h, loss),
                xytext=(6, 6),
                textcoords="offset points",
                color=ink,
                fontsize=8,
            )
        ax.set_xscale("log")
        ax.set_xlabel("training GPU-hours on B300s", color=muted)
        ax.set_ylabel(f"{LABELS.get(a.metric, a.metric)} (bits per byte)", color=muted)

    return draw(out, f"unlimited-{a.metric}", paint)


def bars(a: Analysis, root: str, out: Path) -> list[Path]:
    """Limited data: each target's loss against the root target's on the same
    tokens, in percent, for each of `LIMITED`."""
    fams = [f for f in a.q if f != root and "limited" in a.q[f]]
    fams.sort(key=lambda f: a.q[f]["limited"][tokens_label(LIMITED[1])]["delta"])
    labels = [tokens_label(d) for d in LIMITED]

    def paint(ax: Any, theme: tuple) -> None:
        _, ink, muted, _, colors = theme
        y = np.arange(len(fams))
        h = 0.8 / len(labels)
        for k, (label, color) in enumerate(zip(labels, colors, strict=False)):
            v = [a.q[f]["limited"][label]["delta"] for f in fams]
            ax.barh(
                y + (k - 1) * h, v, height=h * 0.9, color=color, label=f"{label} tokens"
            )
        ax.set_yticks(y, fams, color=ink, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color=muted, linewidth=1)
        ax.set_xlabel(
            f"{LABELS.get(a.metric, a.metric)} against {root}'s target, %", color=muted
        )

    size = (7.5, max(3.0, 0.32 * len(fams) + 1.2))
    return draw(out, f"limited-{a.metric}", paint, size) if fams else []
