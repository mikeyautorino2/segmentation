"""Frozen DINO feature extractor + per-patch binary traversability head.

Every experiment uses the same head on [DINO features, 1 prior channel] -> 1 logit:
  vision:           prior channel = 0
  language:         prior channel = CLIP prior (prior.py)
  shuffled_prompts: prior channel = CLIP prior with prompt groups randomly reassigned (control)
  shuffled_prior:   prior channel = CLIP prior of a different image (control)
  prior_only:       no DINO features, prior channel only (in_dim = 0: 1 -> 256 -> 1)

python model.py --backbone facebook/dinov2-small   # shape check
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

DEFAULT_BACKBONE = "facebook/dinov3-vits16-pretrain-lvd1689m"


class DinoFeatures(nn.Module):
    """Frozen DINO backbone: [B,3,H,W] -> patch feature map [B,C,h,w]."""

    def __init__(self, backbone=DEFAULT_BACKBONE):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        cfg = self.backbone.config
        self.dim = cfg.hidden_size
        self.patch = cfg.patch_size
        # DINOv3 has register tokens after CLS; DINOv2 (no registers) has none.
        self.skip = 1 + getattr(cfg, "num_register_tokens", 0)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # backbone stays frozen in eval mode
        return self

    @torch.no_grad()
    def forward(self, x):
        H, W = x.shape[-2:]
        h, w = H // self.patch, W // self.patch
        if (H, W) != (h * self.patch, w * self.patch):
            # e.g. 518 with patch 16 -> 512 (518 divides evenly only for patch 14)
            x = F.interpolate(x, size=(h * self.patch, w * self.patch), mode="bilinear", align_corners=False)
        tokens = self.backbone(pixel_values=x).last_hidden_state[:, self.skip:]
        assert tokens.shape[1] == h * w, (tokens.shape, h, w)
        return tokens.transpose(1, 2).reshape(x.shape[0], -1, h, w)


class TraversabilityHead(nn.Module):
    """[B,C,h,w] features (or None when in_dim = 0) + optional [B,1,h',w'] prior -> logits [B,1,*out_size]."""

    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, feats, prior, out_size):
        if feats is None:  # prior-only head
            x = prior.float()
            logits = self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        feats = feats.float()
        if prior is None:
            prior = feats.new_zeros(feats.shape[0], 1, *feats.shape[-2:])
        elif prior.shape[-2:] != feats.shape[-2:]:
            prior = F.interpolate(prior.float(), size=feats.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([feats, prior.float()], 1)
        logits = self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


if __name__ == "__main__":
    from prior import DEFAULT_CLIP, ClipPrior

    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--clip", default=DEFAULT_CLIP)
    args = p.parse_args()
    x = torch.randn(2, 3, 518, 518).cuda()
    dino = DinoFeatures(args.backbone).cuda()
    feats = dino(x)
    prior = F.interpolate(ClipPrior(args.clip).cuda()(x)[:, :1], size=feats.shape[-2:], mode="bilinear", align_corners=False)
    head = TraversabilityHead(dino.dim).cuda()
    print("backbone:", args.backbone, "patch:", dino.patch, "skipped tokens:", dino.skip)
    print("feature map:", tuple(feats.shape), "prior:", tuple(prior.shape),
          f"range [{prior.min().item():.3f}, {prior.max().item():.3f}]")
    print("logits:", tuple(head(feats, prior, x.shape[-2:]).shape), "(zero prior:",
          tuple(head(feats, None, x.shape[-2:]).shape), ")")
    print("trainable head params:", sum(p.numel() for p in head.parameters()))
    prior_head = TraversabilityHead(0).cuda()
    print("prior-only logits:", tuple(prior_head(None, prior, x.shape[-2:]).shape),
          "params:", sum(p.numel() for p in prior_head.parameters()))
