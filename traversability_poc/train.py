"""Train the segmentation head for one experiment.

python train.py --exp baseline   # 25-way semantic segmentation, CrossEntropyLoss
python train.py --exp language   # binary traversability from the LLM mapping, BCEWithLogitsLoss
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import IGNORE, NUM_CLASSES, RUGD, VOID_ID, load_mapping
from evaluate import run
from model import DEFAULT_BACKBONE, SegModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=["baseline", "language"], required=True)
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
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(RUGD("train"), batch_size=args.batch_size, shuffle=True, generator=g,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(RUGD("val"), batch_size=args.batch_size, num_workers=4, pin_memory=True)

    num_outputs = NUM_CLASSES if args.exp == "baseline" else 1
    model = SegModel(num_outputs, args.backbone).to(device)
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr)

    best, log = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, label, _ in train_loader:
            x, label = x.to(device, non_blocking=True), label.to(device, non_blocking=True)
            logits = model(x)
            if args.exp == "baseline":
                loss = F.cross_entropy(logits, label, ignore_index=VOID_ID)
            else:
                target = lut[label]
                valid = target != IGNORE
                loss = F.binary_cross_entropy_with_logits(logits[:, 0][valid], target[valid].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        model.eval()
        metrics, _ = run(model, args.exp, val_loader, lut, device)
        # Model selection: semantic mIoU for the baseline, binary mIoU for the language run.
        score = metrics["semantic"]["mean_iou"] if args.exp == "baseline" else metrics["binary"]["mean_iou"]
        entry = {"epoch": epoch, "train_loss": total / len(train_loader), "val_score": score,
                 "val": metrics, "seconds": round(time.time() - t0, 1)}
        log.append(entry)
        print(f"epoch {epoch}: loss {entry['train_loss']:.4f}  val score {score:.4f}  "
              f"binary mIoU {metrics['binary']['mean_iou']:.4f}  ({entry['seconds']}s)")
        if score > best:
            best = score
            torch.save({"head": model.head.state_dict(), "num_outputs": num_outputs, "backbone": args.backbone,
                        "exp": args.exp, "epoch": epoch, "val_score": score, "args": vars(args)}, ckpt_path)
        with open(os.path.join(run_dir, "train_log.json"), "w") as f:
            json.dump({"args": vars(args), "best_val_score": best, "epochs": log}, f, indent=2)

    print(f"best val score {best:.4f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
