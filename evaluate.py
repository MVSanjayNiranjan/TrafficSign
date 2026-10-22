"""Evaluate the trained model(s) on the official GTSRB test set (12,630 images).

Produces in results/:
  metrics.json, classification_report.txt, confusion_matrix.png,
  misclassified.png, robustness.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from common import CLASS_NAMES, RESULTS_DIR, get_device, load_models, predict_proba
from data import load_split


def batched_proba(models, x, tta, bs=512):
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].float() / 255
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=xb.is_cuda):
            out.append(predict_proba(models, xb, tta=tta))
    return torch.cat(out)


# Corruptions that simulate real driving conditions (applied to [0,1] float images)
def _gauss_blur(x):
    k = torch.tensor([1., 4., 6., 4., 1.], device=x.device)
    k = (k[:, None] * k[None, :]) / 256
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), k.expand(3, 1, 5, 5), groups=3)


def _occlude(x):
    x = x.clone()
    x[:, :, 14:28, 14:28] = 0.5  # grey square covering ~8% of the sign, off-centre
    return x


CORRUPTIONS = {
    "clean": lambda x: x,
    "night (dark)": lambda x: x * 0.3,
    "glare (bright)": lambda x: (x * 1.6 + 0.2).clamp(0, 1),
    "fog": lambda x: x * 0.5 + 0.5 * 0.75,
    "blur": _gauss_blur,
    "sensor noise": lambda x: (x + torch.randn_like(x) * 0.08).clamp(0, 1),
    "low resolution": lambda x: F.interpolate(F.interpolate(x, size=16, mode="bilinear"), size=48, mode="bilinear"),
    "occlusion (sticker)": lambda x: _occlude(x),
}


def main():
    dev = get_device()
    RESULTS_DIR.mkdir(exist_ok=True)
    models = load_models(dev)
    x, y = load_split("test")
    x, y = x.to(dev), y.to(dev)
    y_np = y.cpu().numpy()
    print(f"Loaded {len(models)} model(s); test set: {len(x)} images\n")

    metrics = {}
    for i, m in enumerate(models):
        acc = (batched_proba([m], x, tta=False).argmax(1) == y).float().mean().item()
        metrics[f"model_{i}"] = acc
        print(f"  single model {i}:            {acc * 100:.2f}%")
    p_ens = batched_proba(models, x, tta=False)
    metrics["ensemble"] = (p_ens.argmax(1) == y).float().mean().item()
    print(f"  ensemble ({len(models)} models):        {metrics['ensemble'] * 100:.2f}%")
    p = batched_proba(models, x, tta=True)
    pred = p.argmax(1)
    acc = (pred == y).float().mean().item()
    metrics["ensemble_tta"] = acc
    print(f"  ensemble + test-time aug:   {acc * 100:.2f}%   <-- final system")
    print(f"  errors: {(pred != y).sum().item()} / {len(y)}\n")

    # Confidence threshold: in a car, an uncertain prediction should be flagged, not trusted
    conf = p.max(1).values
    print("Confidence threshold analysis (reject = 'uncertain, rely on other sensors/map'):")
    metrics["threshold"] = {}
    for t in (0.5, 0.6, 0.7, 0.8, 0.85):
        keep = conf >= t
        acc_kept = (pred[keep] == y[keep]).float().mean().item()
        cov = keep.float().mean().item()
        metrics["threshold"][str(t)] = {"coverage": cov, "accuracy": acc_kept}
        print(f"  conf >= {t:.2f}: answers {cov * 100:6.2f}% of signs, accuracy on those {acc_kept * 100:.3f}%")

    # Per-class report + confusion matrix
    pred_np = pred.cpu().numpy()
    report = classification_report(y_np, pred_np, target_names=CLASS_NAMES, digits=4)
    (RESULTS_DIR / "classification_report.txt").write_text(report)
    cm = confusion_matrix(y_np, pred_np, normalize="true")
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(43)); ax.set_yticks(range(43))
    ax.set_xticklabels(range(43), fontsize=7); ax.set_yticklabels([f"{i} {n}" for i, n in enumerate(CLASS_NAMES)], fontsize=7)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix (normalised) - test accuracy {acc * 100:.2f}%")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=130); plt.close(fig)

    # Misclassified examples
    wrong = (pred != y).nonzero().squeeze(1)[:24].cpu()
    if len(wrong):
        cols = 6
        rows = int(np.ceil(len(wrong) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.7))
        for a in np.atleast_1d(axes).ravel():
            a.axis("off")
        for a, i in zip(np.atleast_1d(axes).ravel(), wrong):
            a.imshow(x[i].permute(1, 2, 0).cpu().numpy())
            a.set_title(f"true: {CLASS_NAMES[y_np[i]][:18]}\npred: {CLASS_NAMES[pred_np[i]][:18]}\n"
                        f"conf {conf[i].item():.2f}", fontsize=6.5)
        fig.tight_layout(); fig.savefig(RESULTS_DIR / "misclassified.png", dpi=130); plt.close(fig)

    # Robustness under adverse conditions
    print("\nRobustness under simulated driving conditions:")
    metrics["robustness"] = {}
    torch.manual_seed(0)
    for name, fn in CORRUPTIONS.items():
        correct = 0
        for i in range(0, len(x), 512):
            xb = fn(x[i:i + 512].float() / 255)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=xb.is_cuda):
                correct += (predict_proba(models, xb, tta=False).argmax(1) == y[i:i + 512]).sum().item()
        metrics["robustness"][name] = correct / len(x)
        print(f"  {name:20s} {correct / len(x) * 100:6.2f}%")
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(metrics["robustness"]); vals = [metrics["robustness"][n] * 100 for n in names]
    bars = ax.barh(names, vals, color="#3b7dd8")
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    ax.set_xlim(min(vals) - 5, 101); ax.invert_yaxis(); ax.set_xlabel("Accuracy (%)")
    ax.set_title("Robustness to adverse conditions (GTSRB test set)")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "robustness.png", dpi=130); plt.close(fig)

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved reports and figures to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
