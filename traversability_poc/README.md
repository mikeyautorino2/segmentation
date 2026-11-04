# Language-Guided Traversability Prediction (PoC)

Experiments on RUGD. All of them predict binary traversability (targets from the LLM-derived
`configs/terrain_mapping.json`) with BCEWithLogits and a per-patch MLP head (Linear → 256 → ReLU → Linear 1,
logits upsampled to 518×518). The DINO experiments share the same 385 → 256 → 1 head and differ only in the 385th input channel:

| | Experiment | DINOv3 | Prior channel | What it tests |
|---|---|---|---|---|
| A | `vision` | yes | zeros | baseline |
| B | `language` | yes | CLIP terrain scores | benefit of the prior |
| C | `shuffled_prompts` | yes | CLIP scores with the prompt groups randomly reassigned | whether meaningful text matters |
| – | `shuffled_prior` | yes | CLIP scores of a *different image* | whether the prior must align spatially with the image |
| D | `prior_only` | no | CLIP terrain scores (1 → 256 → 1 head) | how useful the language prior is by itself |
| – | `prior_threshold` | no | raw CLIP prior > 0.5, no training | zero-shot reference (computed by `evaluate.py`) |

The **CLIP prior** (`prior.py`) comes from frozen `openai/clip-vit-base-patch16`. Each CLIP patch token is projected
into the joint image/text embedding space and scored against the terrain descriptions in `configs/terrain_prompts.json`,
which are grouped into `traversable` and `non_traversable`. The prior is the softmax probability mass on the traversable
descriptions: one value in [0, 1] per patch, resized to the DINO patch grid. For `shuffled_prior`, each split's priors are
permuted with a fixed seeded derangement, so no image keeps its own prior. For `shuffled_prompts`, the same image
and the same text embeddings are used, but the traversable / non-traversable labels are randomly reassigned across the
descriptions with a fixed seed (`cache_features.py --prompt-seed`). The group sizes stay the same, so the text no
longer carries traversability meaning. The assignment is printed by `cache_features.py` and saved in `comparison.json`.

All experiments use the same backbone, splits, seed (0) and hyperparameters: AdamW, lr 1e-3, 10 epochs, batch 8.
The checkpoint with the best **val** binary mIoU is kept. Final metrics are reported on the held-out **test** split.

> **Caveat:** single seed and small subsets (500 / 100 / 100). Treat results as a preliminary proof of concept.

## Setup

```bash
pip install -r requirements.txt
hf auth login   # DINOv3 is gated: accept the license at
                # https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m first
```

## Run

```bash
cd traversability_poc
python dataset.py                        # download RUGD (~5.7 GB), write the 500/100/100 subset + splits/*.txt
python model.py                          # shape check: (B, 384, 32, 32) features + (B, 1, 32, 32) prior
python cache_features.py                 # frozen DINOv3 features + CLIP priors -> cache/<backbone>/{train,val,test}.pt
python train.py --exp vision             # A -> checkpoints/vision.pt
python train.py --exp language           # B -> checkpoints/language.pt
python train.py --exp shuffled_prompts   # C -> checkpoints/shuffled_prompts.pt
python train.py --exp shuffled_prior     #   -> checkpoints/shuffled_prior.pt
python train.py --exp prior_only         # D -> checkpoints/prior_only.pt
python evaluate.py                       # test-split metrics -> results/
```

`train.py` refuses to overwrite an existing checkpoint unless you pass `--overwrite`.
Per-epoch logs go to `results/runs/<exp>/train_log.json`. Training runs only the head on cached features.
If you change `terrain_prompts.json` or the CLIP model, rebuild the cache with `cache_features.py --overwrite`.

**Without DINOv3 access**, add `--backbone facebook/dinov2-small` to `model.py`, `cache_features.py` and every
`train.py` command. `evaluate.py` reads the backbone name from the checkpoints.

## Data

- Source: official RUGD release (`rugd.vision`), 24 classes + void, palette in `dataset.py`.
- Sequence-level splits from the official RUGD splits. No sequence appears in more than one split:
  - train: the official train sequences
  - val: `park-8`, `trail-5`. Used only for checkpoint selection.
  - test: `creek`, `park-1`, `trail-7`, `trail-13`. Used only for final reporting.
- 500 / 100 / 100 frames are taken at even spacing (adjacent frames are near-duplicates). The exact frames are listed in
  `splits/{train,val,test}.txt`, which is tracked in git.
- Images are resized to 518×518 bilinear; masks with nearest-neighbour, then converted from RGB colors to class IDs.
- DINOv3 uses 16-px patches, so `model.py` resizes 518 → 512 before the backbone (DINOv2's 14-px patches fit 518 exactly).

## Outputs

```text
results/
├── <exp>_metrics.json      # binary metrics on test, one per experiment (+ prior_threshold)
├── comparison.json         # binary pixel acc, traversable IoU, binary mIoU per experiment,
│                           # language − every other experiment, and the shuffled prompt groups
├── runs/<exp>/train_log.json
└── visualizations/         # 10 test images, 2×5: RGB | GT | prior | shuffled-prompt prior | prior > 0.5
                            #                  vision | language | shuffled_prompts | shuffled_prior | prior_only
```

Reference binary masks come from the ground-truth semantic masks via the same mapping; void pixels are ignored.
