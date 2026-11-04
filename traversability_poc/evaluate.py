"""Evaluate all experiments on the held-out test split with the same binary metrics.

python evaluate.py   # reads checkpoints/<exp>.pt for every experiment and the cached test features

Also reports prior_threshold: the raw CLIP prior thresholded at 0.5, with no training at all.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import IGNORE, MEAN, RUGD, STD, load_mapping
from model import TraversabilityHead

EXPERIMENTS = ["vision", "language", "shuffled_prompts", "shuffled_prior", "prior_only"]
NO_DINO = {"prior_only"}  # head sees only the prior channel (1 -> 256 -> 1)
TITLES = {"vision": "A: Vision-only (DINO + 0)", "language": "B: Language (DINO + CLIP prior)",
          "shuffled_prompts": "C: Shuffled prompts (DINO + prior)", "shuffled_prior": "Shuffled prior (other image)",
          "prior_only": "D: Prior-only (no DINO)", "prior_threshold": "Raw prior > 0.5 (untrained)"}
CAVEAT = ("Preliminary proof of concept. All DINO experiments share the backbone, binary target, BCE loss and "
          "385 -> 256 -> 1 head and differ only in the extra input channel: zeros (vision), CLIP prior (language), "
          "CLIP prior with prompt groups randomly reassigned (shuffled_prompts), or the CLIP prior of a different "
          "image (shuffled_prior). prior_only trains a 1 -> 256 -> 1 head on the CLIP prior alone; prior_threshold "
          "is the raw prior > 0.5 without training. Checkpoints are selected on val; metrics are on test sequences "
          "never used for training or selection. Single seed.")


def confusion(pred, gt, n, ignore):
    """Accumulate an n x n confusion matrix (rows = ground truth) over non-ignored pixels."""
    valid = gt != ignore
    return torch.bincount(gt[valid] * n + pred[valid], minlength=n * n).reshape(n, n).cpu()


def binary_metrics(cm):
    cm = cm.double()
    iou = cm.diag() / (cm.sum(0) + cm.sum(1) - cm.diag())
    return {
        "pixel_accuracy": (cm.diag().sum() / cm.sum()).item(),
        "traversable_iou": iou[1].item(),
        "non_traversable_iou": iou[0].item(),
        "mean_iou": iou.mean().item(),
    }


def select_prior(exp, cache, seed):
    """Prior channel for one experiment on one cached split: None (zeros), a prior, or a deranged prior."""
    if exp == "vision":
        return None
    if exp == "shuffled_prompts":
        return cache["prior_shuffled_prompts"]
    prior = cache["prior"]
    if exp in ("language", "prior_only", "prior_threshold"):
        return prior
    assert exp == "shuffled_prior", exp
    # Random order, then a cyclic shift: every image gets the prior of a *different* image.
    order = torch.randperm(len(prior), generator=torch.Generator().manual_seed(seed))
    src = torch.empty_like(order)
    src[order] = order.roll(-1)
    return prior[src.to(prior.device)]


@torch.no_grad()
def run(head, exp, cache, lut, seed, batch_size=16):
    """Evaluate one head on one cached split; returns metrics dict and binary predictions keyed by file name.
    head=None with exp="prior_threshold" evaluates the raw prior > 0.5."""
    prior = select_prior(exp, cache, seed)
    out_size = cache["labels"].shape[-2:]
    cm = torch.zeros(2, 2, dtype=torch.long)
    preds = {}
    for i in range(0, len(cache["names"]), batch_size):
        sl = slice(i, i + batch_size)
        if head is None:
            logits = F.interpolate(prior[sl].float(), size=out_size, mode="bilinear", align_corners=False) - 0.5
        else:
            feats = None if exp in NO_DINO else cache["feats"][sl]
            logits = head(feats, None if prior is None else prior[sl], out_size)
        binary = (logits[:, 0] > 0).long()
        cm += confusion(binary, lut[cache["labels"][sl].long()].long(), 2, IGNORE)
        for name, b in zip(cache["names"][sl], binary.cpu().numpy().astype(np.uint8)):
            preds[name] = b
    return {"binary": binary_metrics(cm)}, preds


def load(path, device):
    ckpt = torch.load(path, map_location="cpu")
    head = TraversabilityHead(ckpt["in_dim"])
    head.load_state_dict(ckpt["head"])
    return head.to(device).eval(), ckpt


def colorize(mask, ignore):
    """Binary mask -> RGB: green traversable, red non-traversable, black ignored."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask == 1] = (40, 180, 70)
    out[mask == 0] = (200, 50, 50)
    out[ignore] = 0
    return out


def visualize(dataset, cache, lut, preds, out_dir, n=10):
    """2 x 5 panels: RGB | GT | prior | shuffled-prompt prior | raw prior > 0.5, then the five trained experiments."""
    os.makedirs(out_dir, exist_ok=True)
    lut_np = lut.cpu().numpy()
    index = {name: i for i, name in enumerate(cache["names"])}

    def upsample(key, i, shape):
        return F.interpolate(cache[key][i][None].float(), size=shape, mode="bilinear",
                             align_corners=False)[0, 0].cpu().numpy()

    for i in np.linspace(0, len(dataset) - 1, n).round().astype(int):
        x, label, name = dataset[i]
        img = np.clip(x.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)
        gt = lut_np[label.numpy()]
        ignore = gt == IGNORE
        panels = [("RGB", img, None), ("Ground truth", colorize(gt, ignore), None),
                  ("CLIP prior P(traversable)", upsample("prior", index[name], gt.shape), "RdYlGn"),
                  ("Shuffled-prompt prior", upsample("prior_shuffled_prompts", index[name], gt.shape), "RdYlGn")]
        for exp in ["prior_threshold"] + EXPERIMENTS:
            pred = preds[exp][name]
            acc = (pred[~ignore] == gt[~ignore]).mean()
            panels.append((f"{TITLES[exp]}\nacc {acc:.3f}", colorize(pred, ignore), None))
        fig, axes = plt.subplots(2, 5, figsize=(20, 9))
        for ax, (title, im, cmap) in zip(axes.flat, panels):
            ax.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(name, fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name.replace(".png", ".jpg")), dpi=100)
        plt.close(fig)


def main():
    from cache_features import load_cache

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lut = torch.from_numpy(load_mapping()).to(device)
    os.makedirs(args.out, exist_ok=True)

    ckpts = {exp: load(os.path.join(args.checkpoints, f"{exp}.pt"), device) for exp in EXPERIMENTS}
    backbones = {ckpt["backbone"] for _, ckpt in ckpts.values()}
    assert len(backbones) == 1, f"experiments use different backbones: {backbones}"
    backbone = backbones.pop()
    cache = load_cache(backbone, args.split, device)

    results, preds = {}, {}
    for exp, (head, ckpt) in ckpts.items():
        assert ckpt["exp"] == exp, (ckpt["exp"], exp)
        results[exp], preds[exp] = run(head, exp, cache, lut, ckpt["args"]["seed"])
        results[exp].update(checkpoint=os.path.join(args.checkpoints, f"{exp}.pt"), backbone=backbone,
                            clip_model=cache["clip_model"], best_epoch=ckpt["epoch"], val_score=ckpt["val_score"],
                            split=args.split, num_images=len(cache["names"]))
        with open(os.path.join(args.out, f"{exp}_metrics.json"), "w") as f:
            json.dump(results[exp], f, indent=2)

    results["prior_threshold"], preds["prior_threshold"] = run(None, "prior_threshold", cache, lut, 0)
    results["prior_threshold"].update(clip_model=cache["clip_model"], split=args.split,
                                      num_images=len(cache["names"]), note="raw CLIP prior > 0.5, no training")
    with open(os.path.join(args.out, "prior_threshold_metrics.json"), "w") as f:
        json.dump(results["prior_threshold"], f, indent=2)

    rows = EXPERIMENTS + ["prior_threshold"]
    m = {exp: results[exp]["binary"] for exp in rows}
    comparison = {
        "note": CAVEAT,
        "backbone": backbone,
        "clip_model": cache["clip_model"],
        "prompts": cache.get("prompts"),
        "split": args.split,
        "num_images": len(cache["names"]),
        "prompt_groups": [{"prompt": p, "real": r, "shuffled": s} for p, r, s in
                          zip(cache["prompt_list"], cache["groups"], cache["shuffled_groups"])],
        "metrics": {k: {**{exp: m[exp][k] for exp in rows},
                        **{f"language_minus_{exp}": m["language"][k] - m[exp][k] for exp in rows if exp != "language"}}
                    for k in m["vision"]},
    }
    with open(os.path.join(args.out, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    visualize(RUGD(args.split), cache, lut, preds, os.path.join(args.out, "visualizations"))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
