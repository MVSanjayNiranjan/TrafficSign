"""Downloads GTSRB (via torchvision) and caches preprocessed tensors for fast training.

Uses the official annotation CSVs to crop every image to its sign's bounding box
(as a detector would in a car) and to record track ids (each physical sign is
filmed ~30 times; a fair validation split must keep whole tracks together).
"""
import csv

import torch
from PIL import Image
from torchvision.datasets import GTSRB
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm

from common import DATA_DIR, preprocess

GTSRB_DIR = DATA_DIR / "gtsrb"
# torchvision only ships a 26,640-image subset of the training data; this is the
# complete official training set (39,209 images, 1,307 physical signs).
FULL_TRAIN_DIR = DATA_DIR / "gtsrb_full" / "GTSRB" / "Final_Training" / "Images"
FULL_TRAIN_URL = ("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
                  "GTSRB_Final_Training_Images.zip")


def _rows(split):
    """Yields (image_path, (x1, y1, x2, y2), class_id, track_id)."""
    if split == "train":
        for d in sorted(FULL_TRAIN_DIR.iterdir()):
            if not d.is_dir():
                continue
            with open(d / f"GT-{d.name}.csv") as f:
                for r in csv.DictReader(f, delimiter=";"):
                    c = int(r["ClassId"])
                    track = c * 1000 + int(r["Filename"].split("_")[0])
                    yield d / r["Filename"], _box(r), c, track
    else:
        with open(GTSRB_DIR / "GT-final_test.csv") as f:
            for i, r in enumerate(csv.DictReader(f, delimiter=";")):
                yield GTSRB_DIR / "GTSRB" / "Final_Test" / "Images" / r["Filename"], _box(r), int(r["ClassId"]), -1 - i


def _box(r):
    return tuple(int(r[k]) for k in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"))


def _build(split):
    if split == "train":
        if not FULL_TRAIN_DIR.exists():
            download_and_extract_archive(FULL_TRAIN_URL, str(DATA_DIR / "gtsrb_full"))
    else:
        GTSRB(root=str(DATA_DIR), split="test", download=True)  # downloads + extracts if missing
    rows = list(_rows(split))
    xs = torch.empty((len(rows), 3, 48, 48), dtype=torch.uint8)
    ys = torch.empty(len(rows), dtype=torch.long)
    tracks = torch.empty(len(rows), dtype=torch.long)
    for i, (path, box, c, t) in enumerate(tqdm(rows, desc=f"Preprocessing {split}")):
        xs[i] = preprocess(Image.open(path), box)
        ys[i], tracks[i] = c, t
    return xs, ys, tracks


def load_split(split, with_tracks=False):
    """Returns (images uint8 [N,3,48,48], labels [N]) (+ track ids) for 'train' or 'test'."""
    cache = DATA_DIR / f"gtsrb_{split}_full_48_roi_ac.pt"
    if not cache.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        xs, ys, tr = _build(split)
        torch.save({"x": xs, "y": ys, "track": tr}, cache)
    d = torch.load(cache, weights_only=True)
    return (d["x"], d["y"], d["track"]) if with_tracks else (d["x"], d["y"])


if __name__ == "__main__":
    for s in ("train", "test"):
        x, y, t = load_split(s, with_tracks=True)
        print(f"{s}: {tuple(x.shape)} images, {int(y.max()) + 1} classes, {len(t.unique())} tracks")
