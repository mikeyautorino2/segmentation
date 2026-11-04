"""Train the binary traversability head for one experiment on cached features.

All experiments: binary target (LLM mapping), BCEWithLogitsLoss, same per-patch MLP head.
python train.py --exp vision             # A: [DINO features, 0]                                385 -> 256 -> 1
python train.py --exp language           # B: [DINO features, CLIP prior]                       385 -> 256 -> 1
python train.py --exp shuffled_prompts   # C: [DINO features, prior from shuffled prompt groups] 385 -> 256 -> 1
python train.py --exp shuffled_prior     #    [DINO features, CLIP prior of another image]      385 -> 256 -> 1
python train.py --exp prior_only         # D: [CLIP prior] only, no DINO                         1 -> 256 -> 1

Run cache_features.py first.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from cache_features import load_cache
from dataset import IGNORE, load_mapping
from evaluate import EXPERIMENTS, NO_DINO, run, select_prior
from model import DEFAULT_BACKBONE, TraversabilityHead


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=EXPERIMENTS, required=True)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", help="allow replacing an existing checkpoint")
    args = p.parse_args()

    ckpt_path = os.path.join("checkpoints", f"{args.exp}.pt")
    run_dir = os.path.join("results", "runs", args.exp)
    if os.path.exists(ckpt_path) and not args.overwrite:
        raise SystemExit(f"{ckpt_path} exists; pass --overwrite or move it first")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"

    lut = torch.from_numpy(load_mapping()).to(device)
    train, val = load_cache(args.backbone, "train", device), load_cache(args.backbone, "val", device)
    train_prior = select_prior(args.exp, train, args.seed)
    targets = lut[train["labels"].long()]
    out_size = train["labels"].shape[-2:]
    use_dino = args.exp not in NO_DINO
    in_dim = train["feats"].shape[1] if use_dino else 0

    head = TraversabilityHead(in_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(args.seed)
    n = len(train["names"])

    best, log = -1.0, []
    for epoch in range(1, args.epochs + 1):
        head.train()
        t0, total, steps = time.time(), 0.0, 0
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n - args.batch_size + 1, args.batch_size):  # drop_last
            idx = perm[i:i + args.batch_size]
            feats = train["feats"][idx] if use_dino else None
            logits = head(feats, None if train_prior is None else train_prior[idx], out_size)
            target = targets[idx]
            valid = target != IGNORE
            loss = F.binary_cross_entropy_with_logits(logits[:, 0][valid], target[valid].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, steps = total + loss.item(), steps + 1

        head.eval()
        metrics, _ = run(head, args.exp, val, lut, args.seed)
        score = metrics["binary"]["mean_iou"]  # model selection on val binary mIoU; test is held out
        entry = {"epoch": epoch, "train_loss": total / steps, "val_score": score,
                 "val": metrics, "seconds": round(time.time() - t0, 1)}
        log.append(entry)
        print(f"epoch {epoch}: loss {entry['train_loss']:.4f}  val binary mIoU {score:.4f}  ({entry['seconds']}s)")
        if score > best:
            best = score
            torch.save({"head": head.state_dict(), "in_dim": in_dim, "exp": args.exp,
                        "backbone": args.backbone, "clip_model": train["clip_model"], "epoch": epoch,
                        "val_score": score, "args": vars(args)}, ckpt_path)
        with open(os.path.join(run_dir, "train_log.json"), "w") as f:
            json.dump({"args": vars(args), "best_val_score": best, "epochs": log}, f, indent=2)

    print(f"best val score {best:.4f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
