"""Evaluate both checkpoints on the same validation images, write metrics and visualizations.

python evaluate.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, IGNORE, MEAN, NUM_CLASSES, RUGD, STD, VOID_ID, load_mapping
from model import SegModel

CAVEAT = ("Preliminary proof of concept. The two runs differ in prediction target and loss "
          "(semantic CE + post-hoc mapping vs. direct binary BCE), so this comparison does not "
          "isolate the causal effect of language priors.")


def confusion(pred, gt, n, ignore):
    """Accumulate an n x n confusion matrix (rows = ground truth) over non-ignored pixels."""
    valid = gt != ignore
    return torch.bincount(gt[valid] * n + pred[valid], minlength=n * n).reshape(n, n).cpu()


def semantic_metrics(cm):
    cm = cm.double()
    iou = cm.diag() / (cm.sum(0) + cm.sum(1) - cm.diag())
    present = cm.sum(1) > 0  # mIoU over classes that occur in the val ground truth
    present[VOID_ID] = False
    return {
        "pixel_accuracy": (cm.diag().sum() / cm.sum()).item(),
        "mean_iou": iou[present].mean().item(),
        "per_class_iou": {CLASS_NAMES[i]: round(iou[i].item(), 4) for i in range(NUM_CLASSES) if present[i]},
    }


def binary_metrics(cm):
    cm = cm.double()
    iou = cm.diag() / (cm.sum(0) + cm.sum(1) - cm.diag())
    return {
        "pixel_accuracy": (cm.diag().sum() / cm.sum()).item(),
        "traversable_iou": iou[1].item(),
        "non_traversable_iou": iou[0].item(),
        "mean_iou": iou.mean().item(),
    }


def predict(model, exp, x, lut):
    """Returns (binary traversability prediction [B,H,W], semantic prediction or None)."""
    logits = model(x)
    if exp == "baseline":
        sem = logits[:, 1:].argmax(1) + 1  # never predict void
        return lut[sem].long(), sem
    return (logits[:, 0] > 0).long(), None


def load(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = SegModel(ckpt["num_outputs"], ckpt["backbone"])
    model.head.load_state_dict(ckpt["head"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def run(model, exp, loader, lut, device):
    """Evaluate one model; returns metrics dict and binary predictions keyed by file name."""
    sem_cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    bin_cm = torch.zeros(2, 2, dtype=torch.long)
    preds = {}
    for x, label, names in loader:
        x, label = x.to(device), label.to(device)
        binary, sem = predict(model, exp, x, lut)
        gt_bin = lut[label].long()
        bin_cm += confusion(binary, gt_bin, 2, IGNORE)
        if sem is not None:
            sem_cm += confusion(sem, label, NUM_CLASSES, VOID_ID)
        for name, b in zip(names, binary.cpu().numpy().astype(np.uint8)):
            preds[name] = b
    metrics = {"binary": binary_metrics(bin_cm)}
    if exp == "baseline":
        metrics["semantic"] = semantic_metrics(sem_cm)
    return metrics, preds


def colorize(mask, ignore):
    """Binary mask -> RGB: green traversable, red non-traversable, black ignored."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask == 1] = (40, 180, 70)
    out[mask == 0] = (200, 50, 50)
    out[ignore] = 0
    return out


def visualize(dataset, lut, preds_b, preds_l, out_dir, n=10):
    os.makedirs(out_dir, exist_ok=True)
    lut_np = lut.cpu().numpy()
    for i in np.linspace(0, len(dataset) - 1, n).round().astype(int):
        x, label, name = dataset[i]
        img = np.clip(x.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)
        gt = lut_np[label.numpy()]
        ignore = gt == IGNORE
        panels = [("RGB", img), ("Ground truth", colorize(gt, ignore))]
        for title, pred in [("Baseline (semantic -> map)", preds_b[name]), ("Language-guided (binary)", preds_l[name])]:
            acc = (pred[~ignore] == gt[~ignore]).mean()
            panels.append((f"{title}\nacc {acc:.3f}", colorize(pred, ignore)))
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
        for ax, (title, im) in zip(axes, panels):
            ax.imshow(im)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(name, fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name.replace(".png", ".jpg")), dpi=100)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="checkpoints/baseline.pt")
    p.add_argument("--language", default="checkpoints/language.pt")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lut = torch.from_numpy(load_mapping()).to(device)
    dataset = RUGD("val")
    loader = DataLoader(dataset, batch_size=8, num_workers=4)
    os.makedirs(args.out, exist_ok=True)

    results, preds, backbones = {}, {}, {}
    for exp, path in [("baseline", args.baseline), ("language", args.language)]:
        model, ckpt = load(path, device)
        results[exp], preds[exp] = run(model, exp, loader, lut, device)
        results[exp].update(checkpoint=path, backbone=ckpt["backbone"], best_epoch=ckpt["epoch"], num_val_images=len(dataset))
        backbones[exp] = ckpt["backbone"]
        with open(os.path.join(args.out, f"{exp}_metrics.json"), "w") as f:
            json.dump(results[exp], f, indent=2)
        del model
    assert preds["baseline"].keys() == preds["language"].keys()

    b, l = results["baseline"]["binary"], results["language"]["binary"]
    comparison = {
        "note": CAVEAT,
        "backbones": backbones,
        "num_val_images": len(dataset),
        "metrics": {k: {"baseline": b[k], "language": l[k], "language_minus_baseline": l[k] - b[k]} for k in b},
    }
    with open(os.path.join(args.out, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    visualize(dataset, lut, preds["baseline"], preds["language"], os.path.join(args.out, "visualizations"))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
