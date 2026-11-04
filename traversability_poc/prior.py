"""Frozen CLIP spatial traversability prior from language descriptions.

Each CLIP patch token is projected into the joint image/text space and scored against every
description in configs/terrain_prompts.json. The prior is the softmax probability mass on the
"traversable" descriptions: one value in [0, 1] per patch.

ClipPrior returns two channels from one CLIP pass:
  0: prior with the real traversable / non-traversable groups
  1: prior with the groups randomly reassigned across descriptions (same group sizes, fixed seed),
     i.e. the same image and the same text embeddings, but the language no longer means anything
     about traversability. Used by the shuffled_prompts control.
"""
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from dataset import MEAN, ROOT, STD, load_config

DEFAULT_CLIP = "openai/clip-vit-base-patch16"
DEFAULT_PROMPTS = os.path.join(ROOT, "configs", "terrain_prompts.json")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def shuffle_groups(groups, seed):
    """Random reassignment of group labels across prompts, keeping the number of traversable prompts."""
    return random.Random(seed).sample(groups, len(groups))


def load_prompts(path=DEFAULT_PROMPTS):
    """-> (list of prompt strings, list of group labels: 1 traversable, 0 not)."""
    cfg = load_config(path)
    template = cfg.get("template", "{}")
    prompts, groups = [], []
    for group, key in [(1, "traversable"), (0, "non_traversable")]:
        for desc in cfg[key]:
            prompts.append(template.format(desc))
            groups.append(group)
    return prompts, groups


class ClipPrior(nn.Module):
    """Normalized image [B,3,H,W] (dataset normalization) -> [B,2,h,w] on CLIP's patch grid:
    channel 0 = P(traversable) with the real prompt groups, channel 1 = with shuffled prompt groups."""

    def __init__(self, name=DEFAULT_CLIP, prompts_path=DEFAULT_PROMPTS, shuffle_seed=0):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(name)
        self.clip.requires_grad_(False)
        self.clip.eval()
        self.patch = self.clip.config.vision_config.patch_size

        prompts, groups = load_prompts(prompts_path)
        self.prompts, self.groups = prompts, groups
        self.shuffled_groups = shuffle_groups(groups, shuffle_seed)
        tok = CLIPTokenizer.from_pretrained(name)
        with torch.no_grad():
            text = self.clip.get_text_features(**tok(prompts, padding=True, return_tensors="pt"))
        if not torch.is_tensor(text):  # newer transformers return an output object
            text = text.pooler_output
        self.register_buffer("text", F.normalize(text, dim=-1), persistent=False)
        self.register_buffer("groups_t", torch.tensor([groups, self.shuffled_groups], dtype=torch.float32).T,
                             persistent=False)  # [prompts, 2]
        # dataset normalization (ImageNet) -> CLIP normalization, as a per-channel affine map
        scale = torch.tensor(STD) / torch.tensor(CLIP_STD)
        shift = (torch.tensor(MEAN) - torch.tensor(CLIP_MEAN)) / torch.tensor(CLIP_STD)
        self.register_buffer("scale", scale.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("shift", shift.view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        H, W = x.shape[-2:]
        h, w = H // self.patch, W // self.patch
        x = x * self.scale + self.shift
        if (H, W) != (h * self.patch, w * self.patch):
            x = F.interpolate(x, size=(h * self.patch, w * self.patch), mode="bilinear", align_corners=False)
        vm = self.clip.vision_model
        tokens = vm(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[:, 1:]
        emb = F.normalize(self.clip.visual_projection(vm.post_layernorm(tokens)), dim=-1)
        probs = (self.clip.logit_scale.exp() * emb @ self.text.T).softmax(-1)  # [B, h*w, prompts]
        return (probs @ self.groups_t).transpose(1, 2).reshape(x.shape[0], 2, h, w)
