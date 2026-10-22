"""Train one TrafficSignNet. Run several seeds to build an ensemble:

    python train.py --seed 0
    python train.py --seed 1
    python train.py --seed 2
"""
import argparse
import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import MODELS_DIR, RESULTS_DIR, NUM_CLASSES, TrafficSignNet, get_device
from data import load_split


# --------------------------------------------------------------------------- #
# GPU data augmentation (a different random transform for every image)
# --------------------------------------------------------------------------- #
def augment(x):
    """x: float [B,3,H,W] in [0,1] on GPU. Simulates real driving conditions.

    No horizontal flips: mirroring changes meaning (e.g. 'turn left' <-> 'turn right').
    """
    B, dev = x.size(0), x.device
    u = lambda lo, hi: torch.empty(B, device=dev).uniform_(lo, hi)

    # Geometry: rotation, zoom, shear, shift (camera angle, distance, bad crops)
    ang = u(-12, 12) * math.pi / 180
    scale = u(0.9, 1.1)
    shear = u(-0.1, 0.1)
    tx, ty = u(-0.08, 0.08), u(-0.08, 0.08)
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([
        torch.stack([cos, -sin + shear, tx], 1),
        torch.stack([sin, cos, ty], 1),
    ], 1)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)

    # Low resolution (distant signs): 40% of images are shrunk to 10-26 px and scaled back.
    # Without this, signs under 25 px wide were the main source of errors.
    lowres = torch.rand(B, device=dev) < 0.4
    if lowres.any():
        size = int(torch.randint(10, 27, (1,)))
        small = F.interpolate(x[lowres], size=size, mode="area")
        x[lowres] = F.interpolate(small, size=x.shape[-1], mode="bilinear", align_corners=False)

    # Photometric: brightness, contrast, saturation (day/night, glare, fading)
    b = u(0.7, 1.3).view(B, 1, 1, 1)
    c = u(0.7, 1.3).view(B, 1, 1, 1)
    s = u(0.7, 1.3).view(B, 1, 1, 1)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * c + mean
    gray = x.mean(dim=1, keepdim=True)
    x = (x - gray) * s + gray
    x = x * b

    # Motion / focus blur on 15% of images
    blur = torch.rand(B, device=dev) < 0.15
    if blur.any():
        k = torch.tensor([1., 2., 1.], device=dev)
        k = (k[:, None] * k[None, :]) / 16
        k = k.expand(3, 1, 3, 3)
        x[blur] = F.conv2d(F.pad(x[blur], (1, 1, 1, 1), mode="replicate"), k, groups=3)

    # Sensor noise
    x = x + torch.randn_like(x) * u(0, 0.03).view(B, 1, 1, 1)

    # Random occlusion (stickers, snow, branches) on 15% of images
    occ = torch.rand(B, device=dev) < 0.15
    if occ.any():
        H, W = x.shape[2:]
        sz = (torch.rand(B, device=dev) * 0.15 + 0.1) * H
        cy, cx = torch.rand(B, device=dev) * H, torch.rand(B, device=dev) * W
        yy = torch.arange(H, device=dev).view(1, H, 1)
        xx = torch.arange(W, device=dev).view(1, 1, W)
        m = ((yy - cy.view(B, 1, 1)).abs() < sz.view(B, 1, 1) / 2) & \
            ((xx - cx.view(B, 1, 1)).abs() < sz.view(B, 1, 1) / 2) & occ.view(B, 1, 1)
        x = torch.where(m.unsqueeze(1), torch.rand(B, 3, 1, 1, device=dev), x)

    return x.clamp(0, 1)


@torch.no_grad()
def evaluate(model, x, y, bs=1024):
    model.eval()
    correct, loss = 0, 0.0
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].float() / 255
        yb = y[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=xb.is_cuda):
            out = model(xb)
        loss += F.cross_entropy(out.float(), yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    return correct / len(x), loss / len(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--no-stn", action="store_true")
    p.add_argument("--tag", default="", help="suffix for experiment checkpoints")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = get_device()
    torch.backends.cudnn.benchmark = True
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    x, y, tracks = load_split("train", with_tracks=True)
    x_test, y_test = load_split("test")

    # Validation set = whole tracks (each physical sign is filmed ~30 times; splitting
    # frames of one sign between train and val would make validation far too easy).
    # The official test set is NEVER used for choosing the model.
    # --val-frac 0 trains on everything and keeps the final epoch (used for the final ensemble).
    g = torch.Generator().manual_seed(1234)
    val_tracks = []
    for c in range(NUM_CLASSES):
        t = tracks[y == c].unique()
        t = t[torch.randperm(len(t), generator=g)]
        val_tracks.append(t[: round(len(t) * args.val_frac)])
    mask = ~torch.isin(tracks, torch.cat(val_tracks))
    val_idx = (~mask).nonzero().squeeze(1)
    x_tr, y_tr = x[mask].to(dev), y[mask].to(dev)
    x_val, y_val = x[val_idx].to(dev), y[val_idx].to(dev)
    x_test, y_test = x_test.to(dev), y_test.to(dev)
    print(f"train {len(x_tr)} | val {len(x_val)} | test {len(x_test)} | device {dev}")

    # Class-balanced sampling weights: rare signs (e.g. 20 km/h) get seen as often as common ones
    counts = torch.bincount(y_tr, minlength=NUM_CLASSES).float()
    sample_w = (1.0 / counts)[y_tr]

    model = TrafficSignNet(use_stn=not args.no_stn).to(dev)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    steps_per_epoch = len(x_tr) // args.bs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch, pct_start=0.15)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    hist = {"train_loss": [], "train_acc": [], "val_acc": [], "val_loss": []}
    best_val, ckpt_path = -1.0, MODELS_DIR / f"model_seed{args.seed}{args.tag}.pt"
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        # half uniform, half class-balanced sampling
        idx_bal = torch.multinomial(sample_w, len(x_tr) // 2, replacement=True)
        idx_uni = torch.randperm(len(x_tr), device=dev)[: len(x_tr) - len(idx_bal)]
        perm = torch.cat([idx_bal, idx_uni])[torch.randperm(len(x_tr), device=dev)]
        tl, tc, n = 0.0, 0, 0
        for i in range(steps_per_epoch):
            bi = perm[i * args.bs:(i + 1) * args.bs]
            xb = augment(x_tr[bi].float() / 255)
            yb = y_tr[bi]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out = model(xb)
                loss = crit(out.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tl += loss.item() * len(bi)
            tc += (out.argmax(1) == yb).sum().item()
            n += len(bi)

        val_acc, val_loss = evaluate(model, x_val, y_val) if len(x_val) else (float("nan"), float("nan"))
        hist["train_loss"].append(tl / n)
        hist["train_acc"].append(tc / n)
        hist["val_acc"].append(val_acc)
        hist["val_loss"].append(val_loss)
        flag = ""
        if not len(x_val) or val_acc >= best_val:
            best_val = val_acc if len(x_val) else best_val
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, ckpt_path)
            flag = "  *saved*"
        print(f"epoch {epoch:2d}/{args.epochs} | loss {tl / n:.4f} | train {tc / n:.4f} | "
              f"val {val_acc:.4f} | {time.time() - t0:5.0f}s{flag}", flush=True)

    # Final report on the untouched official test set
    model.load_state_dict(torch.load(ckpt_path, weights_only=True)["model"])
    test_acc, _ = evaluate(model, x_test, y_test)
    print(f"\nBest val acc {best_val:.4f} | TEST ACCURACY {test_acc:.4f} ({test_acc * 100:.2f}%)")

    with open(RESULTS_DIR / f"history_seed{args.seed}{args.tag}.json", "w") as f:
        json.dump({**hist, "best_val": best_val, "test_acc": test_acc}, f, indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train (augmented)"); ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].set_ylim(0.9, 1.0); ax[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"training_curves_seed{args.seed}{args.tag}.png", dpi=120)


if __name__ == "__main__":
    main()
