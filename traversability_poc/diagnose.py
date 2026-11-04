"""Break the val -> test drop down by sequence and by RUGD semantic class.

python diagnose.py   # -> results/diagnosis.json + printed tables

For every sequence in val and test:
  - binary IoU (traversable, non-traversable, mean) and pixel accuracy for every experiment
  - fraction of traversable pixels in the ground truth (void excluded) and the void fraction
  - distribution of the original RUGD semantic classes
  - per-class error of the vision model: share of each class's pixels predicted traversable,
    next to that class's label in terrain_mapping.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from cache_features import load_cache
from dataset import CLASS_NAMES, IGNORE, NUM_CLASSES, VOID_ID, load_mapping
from evaluate import EXPERIMENTS, load, run


def seq_of(name):
    return name.rsplit("_", 1)[0]


def binary_stats(pred, gt):
    valid = gt != IGNORE
    p, g = pred[valid], gt[valid]
    out = {}
    for cls, key in [(1, "traversable_iou"), (0, "non_traversable_iou")]:
        inter = np.sum((p == cls) & (g == cls))
        union = np.sum((p == cls) | (g == cls))
        out[key] = float(inter / union) if union else float("nan")
    out["mean_iou"] = float(np.nanmean([out["traversable_iou"], out["non_traversable_iou"]]))
    out["pixel_accuracy"] = float(np.mean(p == g))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--out", default="results")
    p.add_argument("--min-share", type=float, default=0.01, help="hide classes below this pixel share in tables")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lut = torch.from_numpy(load_mapping()).to(device)
    lut_np = lut.cpu().numpy()
    ckpts = {exp: load(os.path.join(args.checkpoints, f"{exp}.pt"), device) for exp in EXPERIMENTS}
    backbone = next(iter(ckpts.values()))[1]["backbone"]

    report = {}
    for split in ("val", "test"):
        cache = load_cache(backbone, split, device)
        preds = {exp: run(head, exp, cache, lut, ckpt["args"]["seed"])[1] for exp, (head, ckpt) in ckpts.items()}
        labels = cache["labels"].cpu().numpy()
        by_seq = defaultdict(list)
        for i, name in enumerate(cache["names"]):
            by_seq[seq_of(name)].append(i)

        for seq, idx in sorted(by_seq.items()):
            sem = labels[idx]                       # [n,H,W] class ids
            gt = lut_np[sem]                        # binary / IGNORE
            names = [cache["names"][i] for i in idx]
            entry = {"split": split, "num_images": len(idx)}
            entry["metrics"] = {exp: binary_stats(np.stack([preds[exp][n] for n in names]), gt) for exp in EXPERIMENTS}
            valid = gt != IGNORE
            entry["traversable_fraction"] = float(np.mean(gt[valid] == 1))
            entry["void_fraction"] = float(np.mean(sem == VOID_ID))

            counts = np.bincount(sem.ravel(), minlength=NUM_CLASSES)
            vis = np.stack([preds["vision"][n] for n in names])
            classes = {}
            for c in np.argsort(-counts):
                if counts[c] == 0:
                    continue
                mask = sem == c
                classes[CLASS_NAMES[c]] = {
                    "share": float(counts[c] / counts.sum()),
                    "label": None if lut_np[c] == IGNORE else int(lut_np[c]),
                    "vision_pred_traversable": float(np.mean(vis[mask] == 1)),
                }
            entry["classes"] = classes
            report[seq] = entry

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "diagnosis.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    # ---- printed tables ----
    short = {"vision": "A", "language": "B", "shuffled_prompts": "C", "shuffled_prior": "shufP", "prior_only": "D"}
    print("\nBinary mIoU per sequence (%)  |  GT traversable %, void %")
    print(f"{'split':5} {'sequence':10} {'n':>3}  " + " ".join(f"{short[e]:>6}" for e in EXPERIMENTS)
          + f"  {'trav%':>6} {'void%':>6}  {'travIoU(A)':>10} {'nonIoU(A)':>9}")
    for seq, e in report.items():
        m = e["metrics"]
        print(f"{e['split']:5} {seq:10} {e['num_images']:>3}  "
              + " ".join(f"{100 * m[x]['mean_iou']:6.1f}" for x in EXPERIMENTS)
              + f"  {100 * e['traversable_fraction']:6.1f} {100 * e['void_fraction']:6.1f}"
              + f"  {100 * m['vision']['traversable_iou']:10.1f} {100 * m['vision']['non_traversable_iou']:9.1f}")

    print(f"\nSemantic classes per sequence (share >= {100 * args.min_share:.0f}%): "
          "share % | mapping label | % predicted traversable by A (vision)")
    for seq, e in report.items():
        print(f"\n[{e['split']}] {seq}")
        for cname, c in e["classes"].items():
            if c["share"] < args.min_share:
                continue
            label = "ign" if c["label"] is None else ("trav" if c["label"] else "non")
            wrong = "" if c["label"] is None else (
                100 * (1 - c["vision_pred_traversable"]) if c["label"] else 100 * c["vision_pred_traversable"])
            flag = f"  error {wrong:5.1f}%" if wrong != "" else ""
            print(f"  {cname:26} {100 * c['share']:5.1f}%  {label:4}  pred trav {100 * c['vision_pred_traversable']:5.1f}%{flag}")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
