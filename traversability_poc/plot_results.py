"""Bar chart of one test-set metric across all experiments, from results/comparison.json.
Grouped bars: all test images vs. test without the excluded sequence (creek), when evaluate.py reported it.

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

# Light-mode chart tokens; categorical slots 1-2 of the validated default palette.
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--metric", default="mean_iou", choices=list(METRIC_NAMES))
    p.add_argument("--comparison", default="comparison.json", help="file inside --results to plot")
    p.add_argument("--out-name", default=None, help="output file name inside --results (default <metric>.png)")
    p.add_argument("--note", default="", help="extra text after the headline label")
    args = p.parse_args()

    with open(os.path.join(args.results, args.comparison)) as f:
        comp = json.load(f)
    split, n = comp.get("split", "test"), comp.get("num_images", "?")
    series = [(f"All {split} images ({n})", comp["metrics"][args.metric])]
    sub_key = next((k for k in comp if k.startswith("metrics_without_")), None)
    if sub_key:
        seq = sub_key[len("metrics_without_"):]
        series.append((f"Without {seq} ({comp.get('num_images_without_' + seq, '?')})", comp[sub_key][args.metric]))
    keys = [(key, label) for key, label in ORDER if key in series[0][1]]
    name = METRIC_NAMES[args.metric]

    fig, ax = plt.subplots(figsize=(11, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.17)

    width = 0.36 if len(series) > 1 else 0.55
    for s_i, (s_label, vals) in enumerate(series):
        offset = (s_i - (len(series) - 1) / 2) * (width + 0.02)  # small surface gap between adjacent bars
        xs = [i + offset for i in range(len(keys))]
        ys = [vals[k] * 100 for k, _ in keys]
        ax.bar(xs, ys, width=width, color=SERIES[s_i], label=s_label, zorder=3)
        for x_, y_ in zip(xs, ys):
            ax.text(x_, y_ + 1.0, f"{y_:.1f}", ha="center", va="bottom", fontsize=9, color=INK, zorder=4)

    ax.set_xticks(range(len(keys)), [label for _, label in keys], fontsize=10.5, color=INK)
    if len(series) > 1:
        ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=INK_2, ncol=len(series),
                  bbox_to_anchor=(1.0, 1.08))
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], [f"{t}%" for t in (0, 25, 50, 75, 100)], fontsize=10, color=INK_3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    # Headline block, as in a stat tile: best experiment per series.
    fig.text(0.07, 0.95, f"Best {split}-set {name}" + (f" · {args.note}" if args.note else ""), fontsize=12, color=INK_2)
    for s_i, (s_label, vals) in enumerate(series):
        key, label = max(keys, key=lambda kl: vals[kl[0]])
        x0 = 0.07 + 0.30 * s_i
        fig.text(x0, 0.865, f"{vals[key] * 100:.2f}%", fontsize=28, fontweight="bold", color=INK)
        fig.text(x0, 0.815, f"{label.replace(chr(10), ' ')} · {s_label}", fontsize=11, color=INK_2)


    out = os.path.join(args.results, args.out_name or f"{args.metric}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
