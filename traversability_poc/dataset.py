"""RUGD download, preprocessing and PyTorch dataset.

python dataset.py   # downloads RUGD (~5.7 GB) and writes data/rugd/{train,val}/{images,labels}
"""
import json
import os
import urllib.request
import zipfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data", "rugd")
RAW_DIR = os.path.join(DATA_DIR, "raw")
URLS = {
    "RUGD_annotations.zip": "http://rugd.vision/data/RUGD_annotations.zip",
    "RUGD_frames-with-annotations.zip": "http://rugd.vision/data/RUGD_frames-with-annotations.zip",
}
IMAGE_SIZE = 518
N_TRAIN, N_VAL = 500, 100
IGNORE = 255

# Official RUGD sequence-level splits (Wigness et al., IROS 2019). We draw train images from
# the official train sequences and val images from the official val sequences, so no
# sequence appears in both.
TRAIN_SEQS = ["park-2", "trail", "trail-3", "trail-4", "trail-6", "trail-9", "trail-10",
              "trail-11", "trail-12", "trail-14", "trail-15", "village"]
VAL_SEQS = ["park-8", "trail-5"]

# RUGD_annotation-colormap.txt: id, name, RGB. Masks are stored as RGB color images.
CLASSES = [
    ("void", (0, 0, 0)), ("dirt", (108, 64, 20)), ("sand", (255, 229, 204)),
    ("grass", (0, 102, 0)), ("tree", (0, 255, 0)), ("pole", (0, 153, 153)),
    ("water", (0, 128, 255)), ("sky", (0, 0, 255)), ("vehicle", (255, 255, 0)),
    ("container/generic-object", (255, 0, 127)), ("asphalt", (64, 64, 64)),
    ("gravel", (255, 128, 0)), ("building", (255, 0, 0)), ("mulch", (153, 76, 0)),
    ("rock-bed", (102, 102, 0)), ("log", (102, 0, 0)), ("bicycle", (0, 255, 128)),
    ("person", (204, 153, 255)), ("fence", (102, 0, 204)), ("bush", (255, 153, 204)),
    ("sign", (0, 102, 102)), ("rock", (153, 204, 255)), ("bridge", (102, 255, 255)),
    ("concrete", (101, 101, 11)), ("picnic-table", (114, 85, 47)),
]
CLASS_NAMES = [name for name, _ in CLASSES]
NUM_CLASSES = len(CLASSES)  # 25, index 0 = void (ignored in losses and metrics)
VOID_ID = 0

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def color_to_ids(rgb):
    """RGB mask (H, W, 3) -> class-id mask (H, W). Unknown colors become void."""
    ids = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for i, (_, color) in enumerate(CLASSES):
        ids[(rgb == color).all(-1)] = i
    return ids


def load_mapping(path=os.path.join(ROOT, "configs", "terrain_mapping.json")):
    """Lookup table: class id -> 1 traversable, 0 not, IGNORE for void/unmapped."""
    with open(path) as f:
        mapping = json.load(f)["mapping"]
    lut = np.full(256, IGNORE, dtype=np.uint8)
    for i, name in enumerate(CLASS_NAMES):
        if i != VOID_ID and name in mapping:
            lut[i] = mapping[name]
    return lut


def download():
    os.makedirs(RAW_DIR, exist_ok=True)
    for name, url in URLS.items():
        path = os.path.join(RAW_DIR, name)
        if not os.path.exists(path):
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, path + ".part")
            os.rename(path + ".part", path)


def pick_frames(names, seqs, n):
    """Evenly spaced frames over the given sequences (adjacent frames are near-duplicates)."""
    pool = sorted(x for x in names if x.split("/")[-2] in seqs)
    idx = np.linspace(0, len(pool) - 1, n).round().astype(int)
    return [pool[i] for i in idx]


def prepare():
    download()
    ann_zip = zipfile.ZipFile(os.path.join(RAW_DIR, "RUGD_annotations.zip"))
    img_zip = zipfile.ZipFile(os.path.join(RAW_DIR, "RUGD_frames-with-annotations.zip"))
    images = {os.path.basename(x): x for x in img_zip.namelist() if x.endswith(".png")}
    labels = [x for x in ann_zip.namelist() if x.endswith(".png")]

    for split, seqs, n in [("train", TRAIN_SEQS, N_TRAIN), ("val", VAL_SEQS, N_VAL)]:
        os.makedirs(os.path.join(DATA_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, split, "labels"), exist_ok=True)
        frames = pick_frames(labels, seqs, n)
        for label_path in frames:
            name = os.path.basename(label_path)
            with img_zip.open(images[name]) as f:
                img = Image.open(f).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            with ann_zip.open(label_path) as f:
                rgb = Image.open(f).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
            img.save(os.path.join(DATA_DIR, split, "images", name))
            Image.fromarray(color_to_ids(np.array(rgb))).save(os.path.join(DATA_DIR, split, "labels", name))
        with open(os.path.join(DATA_DIR, f"{split}.txt"), "w") as f:
            f.write("\n".join(os.path.basename(x) for x in frames) + "\n")
        print(f"{split}: {len(frames)} frames from {seqs}")


class RUGD(Dataset):
    """Returns (normalized image [3,H,W], semantic class-id mask [H,W], file name)."""

    def __init__(self, split):
        self.dir = os.path.join(DATA_DIR, split)
        with open(os.path.join(DATA_DIR, f"{split}.txt")) as f:
            self.names = f.read().split()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = np.array(Image.open(os.path.join(self.dir, "images", name)), dtype=np.float32) / 255
        img = torch.from_numpy((img - MEAN) / STD).permute(2, 0, 1)
        label = torch.from_numpy(np.array(Image.open(os.path.join(self.dir, "labels", name)), dtype=np.int64))
        return img, label, name


if __name__ == "__main__":
    prepare()
