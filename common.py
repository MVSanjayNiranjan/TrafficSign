"""Shared pieces: class names, preprocessing, the neural network, and inference helpers."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
IMG_SIZE = 48
NUM_CLASSES = 43

CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 t",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 t prohibited", "No entry", "General caution",
    "Dangerous curve to the left", "Dangerous curve to the right", "Double curve",
    "Bumpy road", "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
    "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 t",
]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Preprocessing (identical for training and inference)
# --------------------------------------------------------------------------- #
def preprocess(img: Image.Image, box=None) -> torch.Tensor:
    """PIL image -> uint8 tensor [3, 48, 48].

    box: optional (x1, y1, x2, y2) sign bounding box (from a detector or the
    dataset annotations); the crop keeps a 10% margin around it.
    Linearly stretches luminance contrast (autocontrast, 1% clipping) so dark /
    hazy signs (night, fog, shadow) become readable, then resizes. A linear stretch
    is used instead of histogram equalisation, which washed signs out to white.
    """
    img = img.convert("RGB")
    if box is not None:
        x1, y1, x2, y2 = box
        mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        img = img.crop((max(0, round(x1 - mx)), max(0, round(y1 - my)),
                        min(img.width, round(x2 + mx)), min(img.height, round(y2 + my))))
    y, cb, cr = img.convert("YCbCr").split()
    img = Image.merge("YCbCr", (ImageOps.autocontrast(y, cutoff=1), cb, cr)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1)


# --------------------------------------------------------------------------- #
# Model: Spatial Transformer + deep CNN
# --------------------------------------------------------------------------- #
def conv_block(c_in, c_out, drop):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2), nn.Dropout2d(drop),
    )


class TrafficSignNet(nn.Module):
    """CNN with a Spatial Transformer Network (STN) in front.

    The STN learns an affine transform that zooms / rotates / centres the sign
    before classification, which makes the network robust to the imperfect
    crops a real detector produces in a car.
    """

    def __init__(self, num_classes=NUM_CLASSES, use_stn=True):
        super().__init__()
        self.use_stn = use_stn
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        # STN localisation network -> 6 affine parameters
        self.loc = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2), nn.MaxPool2d(2), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, padding=2), nn.MaxPool2d(2), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(32 * 16, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )
        self.loc[-1].weight.data.zero_()
        self.loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.features = nn.Sequential(
            conv_block(3, 64, 0.1),     # 48 -> 24
            conv_block(64, 128, 0.15),  # 24 -> 12
            conv_block(128, 256, 0.2),  # 12 -> 6
            nn.Conv2d(256, 384, 3, padding=1, bias=False), nn.BatchNorm2d(384), nn.ReLU(inplace=True),
        )
        # Flatten (not global-average-pool): keeps WHERE features are inside the sign,
        # which is what separates "80" from "50" or a truck from a car symbol.
        self.classifier = nn.Sequential(
            nn.Conv2d(384, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def stn(self, x):
        # sampling grids need full float32 precision (bf16 would jitter pixels)
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            theta = self.loc(x).view(-1, 2, 3)
            grid = F.affine_grid(theta, x.size(), align_corners=False)
            return F.grid_sample(x, grid, padding_mode="border", align_corners=False)

    def forward(self, x):
        """x: float tensor in [0, 1], shape [B, 3, 48, 48]."""
        x = (x - self.mean) / self.std
        if self.use_stn:
            x = self.stn(x)
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------- #
# Inference helpers: test-time augmentation + ensembles
# --------------------------------------------------------------------------- #
def _affine(x, angle_deg=0.0, scale=1.0):
    a = torch.tensor(np.deg2rad(angle_deg), device=x.device, dtype=x.dtype)
    cos, sin = torch.cos(a) / scale, torch.sin(a) / scale
    theta = torch.stack([torch.stack([cos, -sin, torch.zeros_like(a)]),
                         torch.stack([sin, cos, torch.zeros_like(a)])]).unsqueeze(0)
    grid = F.affine_grid(theta.expand(x.size(0), 2, 3), x.size(), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


TTA_VIEWS = [(0, 1.0), (0, 0.9), (0, 1.1), (-7, 1.0), (7, 1.0)]


@torch.no_grad()
def predict_proba(models, x, tta=True):
    """Average softmax over every model (ensemble) and every TTA view.

    x: float tensor in [0, 1], [B, 3, 48, 48], already on the right device.
    """
    views = TTA_VIEWS if tta else TTA_VIEWS[:1]
    probs = 0
    for m in models:
        for angle, scale in views:
            xv = x if (angle == 0 and scale == 1.0) else _affine(x, angle, scale)
            probs = probs + F.softmax(m(xv).float(), dim=1)
    return probs / (len(models) * len(views))


def load_models(device=None, paths=None):
    device = device or get_device()
    paths = paths or sorted(MODELS_DIR.glob("model_seed*.pt"))
    if not paths:
        raise FileNotFoundError(f"No trained models in {MODELS_DIR}. Run train.py first.")
    models = []
    for p in paths:
        m = TrafficSignNet().to(device)
        ckpt = torch.load(p, map_location=device, weights_only=True)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models.append(m)
    return models
