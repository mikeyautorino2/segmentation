"""Bar chart of one test-set metric across all experiments, from results/comparison.json.

python plot_results.py                          # binary mIoU -> results/mean_iou.png
python plot_results.py --metric traversable_iou # -> results/traversable_iou.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = [  # (key in comparison.json, bar label)
    ("vision", "A: Vision"),
    ("language", "B: Language"),
    ("shuffled_prompts", "C: Shuffled\nprompts"),
    ("shuffled_prior", "Shuffled\nprior"),
    ("prior_only", "D: Prior\nonly"),
    ("prior_threshold", "Raw prior\n(untrained)"),
]
METRIC_NAMES = {"mean_iou": "binary mIoU", "traversable_iou": "traversable IoU",
                "non_traversable_iou": "non-traversable IoU", "pixel_accuracy": "pixel accuracy"}

# Light-mode chart tokens (single series -> one hue, no legend).
SURFACE, INK, INK_2, INK_3, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#2a78d6"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--metric", default="mean_iou", choices=list(METRIC_NAMES))
    args = p.parse_args()

    with open(os.path.join(args.results, "comparison.json")) as f:
        comp = json.load(f)
    values = comp["metrics"][args.metric]
    bars = [(label, values[key] * 100) for key, label in ORDER if key in values]
    best_label, best = max(bars, key=lambda b: b[1])
    name = METRIC_NAMES[args.metric]

    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.17)

    x = range(len(bars))
    ax.bar(x, [v for _, v in bars], width=0.55, color=SERIES, zorder=3)
    for i, (_, v) in enumerate(bars):
        ax.text(i, v + 1.2, f"{v:.2f}%", ha="center", va="bottom", fontsize=10, color=INK, zorder=4)

    ax.set_xticks(list(x), [label for label, _ in bars], fontsize=10.5, color=INK)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], [f"{t}%" for t in (0, 25, 50, 75, 100)], fontsize=10, color=INK_3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    # Headline block, as in a stat tile: label, hero number, context.
    split, n = comp.get("split", "test"), comp.get("num_images", "?")
    fig.text(0.08, 0.95, f"Best {split}-set {name}", fontsize=12, color=INK_2)
    fig.text(0.08, 0.865, f"{best:.2f}%", fontsize=30, fontweight="bold", color=INK)
    fig.text(0.08, 0.815, f"{best_label.replace(chr(10), ' ')} · {n} {split} images", fontsize=12, color=INK_2)


    out = os.path.join(args.results, f"{args.metric}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
