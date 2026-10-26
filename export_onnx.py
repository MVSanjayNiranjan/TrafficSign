"""Export the whole ensemble (3 models + test-time augmentation) to one ONNX file
so the dashboard can run entirely in the browser (GitHub Pages, no server).

    python export_onnx.py      -> docs/model.onnx, then checks it against PyTorch

ONNX model
    input  "image":   float32 [B, 3, 48, 48], RGB in [0, 1], already preprocessed
    output "probs":   float32 [B, 43]  averaged softmax (ensemble x TTA views)
    output "aligned": float32 [B, 3, 48, 48]  what model 0's Spatial Transformer sees
"""
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import IMG_SIZE, ROOT, TTA_VIEWS, load_models
from data import load_split

OUT = ROOT / "docs" / "model.onnx"


def _view_grid(angle_deg, scale):
    a = np.deg2rad(angle_deg)
    theta = torch.tensor([[[np.cos(a) / scale, -np.sin(a) / scale, 0],
                           [np.sin(a) / scale, np.cos(a) / scale, 0]]], dtype=torch.float32)
    return F.affine_grid(theta, (1, 3, IMG_SIZE, IMG_SIZE), align_corners=False)


class EnsembleExport(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        # base sampling grid (align_corners=False) as homogeneous coords [1, H*W, 3]
        lin = (torch.arange(IMG_SIZE, dtype=torch.float32) * 2 + 1) / IMG_SIZE - 1
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        self.register_buffer("base", torch.stack([gx, gy, torch.ones_like(gx)], -1).view(1, -1, 3))
        # fixed TTA view grids (identity view handled without resampling)
        self.register_buffer("views", torch.cat([_view_grid(a, s) for a, s in TTA_VIEWS[1:]]))

    def stn(self, m, x):
        # same as F.affine_grid, written as a matmul so it exports to plain ONNX ops
        # loc[6] is AdaptiveAvgPool2d(4) on a 12x12 map == AvgPool2d(3); fixed pooling exports cleanly
        h = F.avg_pool2d(m.loc[:6](x), 3)
        theta = m.loc[7:](h).view(-1, 2, 3)
        grid = torch.matmul(self.base, theta.transpose(1, 2)).view(-1, IMG_SIZE, IMG_SIZE, 2)
        return F.grid_sample(x, grid, padding_mode="border", align_corners=False)

    def classify(self, m, x):
        x = (x - m.mean) / m.std
        return F.softmax(m.classifier(m.features(self.stn(m, x))), dim=1)

    def forward(self, image):
        b = image.shape[0]
        views = [image] + [F.grid_sample(image, self.views[i:i + 1].expand(b, -1, -1, -1),
                                         padding_mode="border", align_corners=False)
                           for i in range(self.views.shape[0])]
        probs = sum(self.classify(m, v) for m in self.models for v in views)
        probs = probs / (len(self.models) * len(views))
        m0 = self.models[0]
        aligned = self.stn(m0, (image - m0.mean) / m0.std) * m0.std + m0.mean
        return probs, aligned.clamp(0, 1)


def main():
    models = load_models(torch.device("cpu"))
    net = EnsembleExport(models).eval()
    OUT.parent.mkdir(exist_ok=True)
    torch.onnx.export(net, torch.rand(1, 3, IMG_SIZE, IMG_SIZE), str(OUT), dynamo=False,
                      input_names=["image"], output_names=["probs", "aligned"],
                      dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}, "aligned": {0: "batch"}},
                      opset_version=17)
    print(f"Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

    # Check: ONNX output must match PyTorch, and give the same test accuracy
    x, y = load_split("test")
    x = x.float() / 255
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    correct, max_diff = 0, 0.0
    for i in range(0, len(x), 500):
        p_onnx = sess.run(["probs"], {"image": x[i:i + 500].numpy()})[0]
        correct += (p_onnx.argmax(1) == y[i:i + 500].numpy()).sum()
        if i == 0:
            with torch.no_grad():
                p_torch = net(x[:500])[0].numpy()
            max_diff = np.abs(p_onnx - p_torch).max()
    print(f"max |ONNX - PyTorch| = {max_diff:.2e}")
    print(f"ONNX test accuracy: {correct / len(x) * 100:.2f}%")


if __name__ == "__main__":
    main()
