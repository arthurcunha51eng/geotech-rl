"""Draws the ladder figure from results/, so the figure in the README can always be rebuilt.

    python figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).parents[1]
LEVELS = [f"n{i}" for i in range(9)]
ZONES = (("Warm-up", 0, 2), ("Rule applied", 2, 3), ("Judgement", 5, 4))
ROWS = (("Qwen3-4B", "qwen3-4b-base"), ("+ SFT", "qwen3-4b-sft"), ("+ SFT + GRPO", "qwen3-4b-sft-grpo"),
        ("Claude Opus 5.5, medium effort", "claude-opus-5-5-medium"))
SURFACE, INK, MUTED, NOT_RUN = "#fcfcfb", "#0b0b0b", "#52514e", "#f0efec"
SHADE = LinearSegmentedColormap.from_list("solved", ("#e7f1fb", "#9ec5ef", "#2a78d6", "#123f7a"))


def table() -> list[tuple[str, dict]]:
    ladder = json.loads((ROOT / "results" / "ladder.json").read_text(encoding="utf-8"))
    fixed = json.loads((ROOT / "results" / "fixed_script.json").read_text(encoding="utf-8"))
    script = {k: f"{v['solved_outside_controls']}/{v['episodes_outside_controls']}" for k, v in fixed.items()}
    return [("Fixed script, outside controls", script)] + [(label, ladder["models"][key]) for label, key in ROWS]


def main() -> None:
    rows = table()
    fig, ax = plt.subplots(figsize=(9.2, 3.5), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for y, (_, cells) in enumerate(rows):
        for x, level in enumerate(LEVELS):
            text = cells.get(level)
            if text is None:
                ax.add_patch(Rectangle((x + 0.03, y + 0.05), 0.94, 0.90, facecolor=NOT_RUN, linewidth=0))
                continue
            solved, total = (int(v) for v in text.split("/"))
            share = solved / total
            ax.add_patch(Rectangle((x + 0.03, y + 0.05), 0.94, 0.90, facecolor=SHADE(share), linewidth=0))
            ax.text(x + 0.5, y + 0.5, text, ha="center", va="center", fontsize=8.5, color="white" if share > 0.55 else INK)
    for name, start, width in ZONES:
        ax.text(start + width / 2, -0.62, name, ha="center", va="center", fontsize=8.5, color=MUTED)
        ax.plot([start + 0.05, start + width - 0.05], [-0.38, -0.38], color=MUTED, linewidth=0.8)
    ax.set_xlim(0, 9)
    ax.set_ylim(len(rows), -0.85)
    ax.set_xticks([i + 0.5 for i in range(9)], [f"N{i}" for i in range(9)], fontsize=9, color=INK)
    ax.xaxis.tick_bottom()
    ax.set_yticks([i + 0.5 for i in range(len(rows))], [label for label, _ in rows], fontsize=9, color=INK)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.text(0.012, 0.965, "A small open model clears the warm-up and stops; a frontier model solves what stops it",
             fontsize=11, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.895, "Solved over episodes on held-out instances. Gray: not run. N4 is inherited from a rain filter in the old demonstrations.",
             fontsize=8.5, color=MUTED, va="top")
    fig.subplots_adjust(left=0.235, right=0.995, top=0.84, bottom=0.075)
    out = ROOT / "docs" / "ladder.png"
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
