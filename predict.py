"""Classify one or more traffic-sign images from the command line.

    python predict.py path/to/sign.png [more images ...] [--threshold 0.9]
"""
import argparse

import torch
from PIL import Image

from common import CLASS_NAMES, get_device, load_models, predict_proba, preprocess


def classify(models, img, device, top_k=3):
    x = preprocess(img).unsqueeze(0).float().div(255).to(device)
    probs = predict_proba(models, x, tta=True)[0]
    conf, idx = probs.topk(top_k)
    return [(CLASS_NAMES[i], c) for i, c in zip(idx.tolist(), conf.tolist())]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="below this confidence the result is reported as UNCERTAIN")
    args = p.parse_args()

    dev = get_device()
    models = load_models(dev)
    for path in args.images:
        top = classify(models, Image.open(path), dev)
        name, conf = top[0]
        status = "OK" if conf >= args.threshold else "UNCERTAIN"
        print(f"\n{path}\n  -> {name}  ({conf * 100:.2f}%)  [{status}]")
        for n, c in top[1:]:
            print(f"     {n}  ({c * 100:.2f}%)")


if __name__ == "__main__":
    main()
