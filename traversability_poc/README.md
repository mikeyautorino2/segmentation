# Language-Guided Traversability Prediction (PoC)

Two runs on RUGD with a frozen DINO backbone and the same per-patch MLP head
(Linear → ReLU → Linear, logits upsampled to 518×518):

| Run | Target | Loss | Binary prediction |
|---|---|---|---|
| `baseline` | 24 RUGD semantic classes (void ignored) | CrossEntropy | predicted class → `configs/terrain_mapping.json` (evaluation only) |
| `language` | traversable / not, from the LLM-derived mapping | BCEWithLogits | `logit > 0` |

Both use the same backbone, split, seed (0) and hyperparameters: AdamW, lr 1e-3, 10 epochs, batch 8.
The checkpoint with the best validation score is kept (semantic mIoU for `baseline`, binary mIoU for `language`).

> **Caveat:** the runs differ in prediction target and loss, so this does not isolate the causal effect
> of language priors. Checkpoints are also selected on the same validation set they are evaluated on.
> Treat results as a preliminary proof of concept.

## Setup

```bash
pip install -r requirements.txt
hf auth login   # DINOv3 is gated: accept the license at
                # https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m first
```

## Run

```bash
cd traversability_poc
python dataset.py                        # download RUGD (~5.7 GB) and write the 500/100 subset
python model.py                          # shape check: (B, 384, 32, 32) feature map for DINOv3 ViT-S/16
python train.py --exp baseline           # -> checkpoints/baseline.pt
python train.py --exp language           # -> checkpoints/language.pt
python evaluate.py                       # -> results/
```

`train.py` refuses to overwrite an existing checkpoint unless you pass `--overwrite`.
Per-epoch logs go to `results/runs/<exp>/train_log.json`.

**Without DINOv3 access**, add `--backbone facebook/dinov2-small` to `model.py` and both `train.py` commands.
`evaluate.py` reads the backbone name from each checkpoint.

## Data

- Source: official RUGD release (`rugd.vision`), 24 classes + void, palette in `dataset.py`.
- Sequence-level split from the official RUGD splits: train images come from the official train sequences,
  val images from `park-8` and `trail-5`. 500 / 100 frames are taken at even spacing (adjacent frames are near-duplicates).
- Images are resized to 518×518 bilinear; masks with nearest-neighbour, then converted from RGB colors to class IDs.
- DINOv3 uses 16-px patches, so `model.py` resizes 518 → 512 before the backbone (DINOv2's 14-px patches fit 518 exactly).

## Outputs

```text
results/
├── baseline_metrics.json   # semantic pixel acc / mIoU / per-class IoU + binary metrics
├── language_metrics.json   # binary metrics
├── comparison.json         # binary pixel acc, traversable IoU, binary mIoU, side by side
├── runs/<exp>/train_log.json
└── visualizations/         # 10 panels: RGB | GT | baseline | language (green = traversable, red = not)
```

Reference binary masks come from the ground-truth semantic masks via the same mapping; void pixels are ignored.
