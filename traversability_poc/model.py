"""Frozen DINO backbone + per-patch MLP segmentation head.

python model.py --backbone facebook/dinov2-small   # shape check
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

DEFAULT_BACKBONE = "facebook/dinov3-vits16-pretrain-lvd1689m"


class SegModel(nn.Module):
    def __init__(self, num_outputs, backbone=DEFAULT_BACKBONE, hidden=256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        cfg = self.backbone.config
        self.patch = cfg.patch_size
        # DINOv3 has register tokens after CLS; DINOv2 (no registers) has none.
        self.skip = 1 + getattr(cfg, "num_register_tokens", 0)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, hidden), nn.ReLU(), nn.Linear(hidden, num_outputs)
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # backbone stays frozen in eval mode
        return self

    @torch.no_grad()
    def features(self, x):
        """[B,3,H,W] -> patch feature map [B,C,h,w]."""
        H, W = x.shape[-2:]
        h, w = H // self.patch, W // self.patch
        if (H, W) != (h * self.patch, w * self.patch):
            # e.g. 518 with patch 16 -> 512 (518 divides evenly only for patch 14)
            x = F.interpolate(x, size=(h * self.patch, w * self.patch), mode="bilinear", align_corners=False)
        tokens = self.backbone(pixel_values=x).last_hidden_state[:, self.skip:]
        assert tokens.shape[1] == h * w, (tokens.shape, h, w)
        return tokens.transpose(1, 2).reshape(x.shape[0], -1, h, w)

    def forward(self, x):
        """[B,3,H,W] -> logits [B,num_outputs,H,W]."""
        feats = self.features(x)
        logits = self.head(feats.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    args = p.parse_args()
    model = SegModel(25, args.backbone).cuda()
    x = torch.randn(2, 3, 518, 518).cuda()
    feats = model.features(x)
    print("backbone:", args.backbone, "patch:", model.patch, "skipped tokens:", model.skip)
    print("feature map:", tuple(feats.shape), "logits:", tuple(model(x).shape))
    print("trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
