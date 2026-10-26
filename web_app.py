"""Web dashboard for the traffic sign recogniser.

    python web_app.py            -> http://localhost:7860

Endpoints
    GET  /                 dashboard page
    POST /api/predict      multipart form field "image" -> JSON prediction
    GET  /api/metrics      evaluation metrics (results/metrics.json)
    GET  /api/samples      list of sample images
"""
import base64
import io
import json
import os
import time

import torch
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError

from common import CLASS_NAMES, RESULTS_DIR, ROOT, get_device, load_models, predict_proba, preprocess

THRESHOLD = 0.7
SAMPLES_DIR = ROOT / "samples"
CHARTS = {"robustness.png", "confusion_matrix.png", "misclassified.png", "training_curves_seed0.png"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB uploads

device = get_device()
models = load_models(device)
predict_proba(models, torch.rand(1, 3, 48, 48, device=device))  # warm-up so the first request is fast
print(f"Loaded {len(models)} models on {device}")


def _png_b64(t):
    """uint8/float tensor [3,H,W] -> base64 PNG, upscaled for display."""
    if t.dtype != torch.uint8:
        t = (t.clamp(0, 1) * 255).byte()
    img = Image.fromarray(t.permute(1, 2, 0).cpu().numpy()).resize((144, 144), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return render_template("index.html", n_models=len(models), device=device.type.upper(),
                           threshold=int(THRESHOLD * 100), n_classes=len(CLASS_NAMES))


@app.post("/api/predict")
def api_predict():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="No image uploaded"), 400
    try:
        img = Image.open(f.stream)
        img.load()
    except (UnidentifiedImageError, OSError):
        return jsonify(error="Could not read that file as an image"), 400

    t0 = time.perf_counter()
    x8 = preprocess(img)
    x = x8.unsqueeze(0).float().div(255).to(device)
    probs = predict_proba(models, x, tta=True)[0]
    with torch.no_grad():  # what the Spatial Transformer "looks at" after aligning the sign
        m = models[0]
        aligned = m.stn((x - m.mean) / m.std) * m.std + m.mean
    latency = (time.perf_counter() - t0) * 1000

    conf, idx = probs.topk(5)
    top = [{"id": i, "name": CLASS_NAMES[i], "conf": c} for i, c in zip(idx.tolist(), conf.tolist())]
    return jsonify(
        label=top[0]["name"], class_id=top[0]["id"], confidence=top[0]["conf"],
        status="OK" if top[0]["conf"] >= THRESHOLD else "UNCERTAIN",
        top5=top, latency_ms=round(latency, 1),
        input_png=_png_b64(x8), stn_png=_png_b64(aligned[0]),
    )


@app.get("/api/metrics")
def api_metrics():
    p = RESULTS_DIR / "metrics.json"
    return jsonify(json.loads(p.read_text()) if p.exists() else {})


@app.get("/api/samples")
def api_samples():
    files = sorted(p.name for p in SAMPLES_DIR.glob("*.png")) if SAMPLES_DIR.exists() else []
    return jsonify(files)


@app.get("/samples/<path:name>")
def sample_file(name):
    return send_from_directory(SAMPLES_DIR, name)


@app.get("/charts/<name>")
def chart_file(name):
    if name not in CHARTS:
        abort(404)
    return send_from_directory(RESULTS_DIR, name)


@app.get("/health")
def health():
    return {"status": "ok", "models": len(models), "device": device.type}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=False)
