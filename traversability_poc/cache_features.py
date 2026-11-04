"""Run the frozen DINO backbone and CLIP prior once per image and cache the results.

python cache_features.py                                   # DINOv3 + CLIP, all splits
python cache_features.py --backbone facebook/dinov2-small  # -> cache/facebook--dinov2-small/

Writes cache/<backbone-slug>/{train,val,test}.pt with
  names  list[str]
  feats  fp16 [N,C,h,w]  DINO patch features
  prior                   fp32 [N,1,h,w]  CLIP traversability prior, resized to the DINO patch grid
  prior_shuffled_prompts  fp32 [N,1,h,w]  same, with prompt groups randomly reassigned (control)
  labels uint8 [N,H,W]   RUGD class ids
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ROOT, RUGD, SPLITS
from model import DEFAULT_BACKBONE, DinoFeatures
from prior import DEFAULT_CLIP, DEFAULT_PROMPTS, ClipPrior

CACHE_DIR = os.path.join(ROOT, "cache")


def cache_path(backbone, split):
    return os.path.join(CACHE_DIR, backbone.replace("/", "--"), f"{split}.pt")


def load_cache(backbone, split, device):
    path = cache_path(backbone, split)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run: python cache_features.py --backbone {backbone}")
    c = torch.load(path, map_location="cpu")
    assert c["backbone"] == backbone, (c["backbone"], backbone)
    if "prior_shuffled_prompts" not in c:
        raise SystemExit(f"{path} is from an older version; rebuild: python cache_features.py --overwrite")
    for k in ("feats", "prior", "prior_shuffled_prompts", "labels"):
        c[k] = c[k].to(device)
    return c


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--clip", default=DEFAULT_CLIP)
    p.add_argument("--prompts", default=DEFAULT_PROMPTS)
    p.add_argument("--prompt-seed", type=int, default=0, help="seed for the shuffled-prompts control")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    device = "cuda"
    dino = DinoFeatures(args.backbone).to(device).eval()
    prior_model = ClipPrior(args.clip, args.prompts, args.prompt_seed).to(device).eval()
    print("shuffled-prompt groups (1 = treated as traversable):")
    for prompt, real, fake in zip(prior_model.prompts, prior_model.groups, prior_model.shuffled_groups):
        print(f"  real {real} -> shuffled {fake}  {prompt}")

    for split in SPLITS:
        path = cache_path(args.backbone, split)
        if os.path.exists(path) and not args.overwrite:
            print(f"{path} exists, skipping (pass --overwrite to rebuild)")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        names, feats, priors, shuffled, labels = [], [], [], [], []
        for x, label, batch_names in DataLoader(RUGD(split), batch_size=args.batch_size, num_workers=4):
            x = x.to(device)
            f = dino(x)
            pr = F.interpolate(prior_model(x), size=f.shape[-2:], mode="bilinear", align_corners=False)
            feats.append(f.half().cpu())
            priors.append(pr[:, :1].float().cpu())
            shuffled.append(pr[:, 1:].float().cpu())
            labels.append(label.to(torch.uint8))
            names += list(batch_names)
        out = {"names": names, "feats": torch.cat(feats), "prior": torch.cat(priors),
               "prior_shuffled_prompts": torch.cat(shuffled), "labels": torch.cat(labels),
               "backbone": args.backbone, "clip_model": args.clip, "prompts": os.path.relpath(args.prompts, ROOT),
               "prompt_list": prior_model.prompts, "groups": prior_model.groups,
               "shuffled_groups": prior_model.shuffled_groups, "prompt_seed": args.prompt_seed}
        torch.save(out, path)
        print(f"{split}: feats {tuple(out['feats'].shape)} prior {tuple(out['prior'].shape)} "
              f"labels {tuple(out['labels'].shape)} -> {path}")


if __name__ == "__main__":
    main()
