# Traffic Sign Recognition Using Neural Networks

Deep-learning system that recognises **43 classes of traffic signs** from camera images,
trained and evaluated on the **German Traffic Sign Recognition Benchmark (GTSRB)**:
39,209 training images and 12,630 test images of real road scenes.

**Final accuracy: 99.84% on the official GTSRB test set** (20 errors in 12,630 images).
For reference, measured human performance on this benchmark is 98.84%.

## How it works

```
camera image -> sign crop -> preprocessing -> [ ensemble of CNNs with Spatial Transformer ] -> class + confidence
```

| Component | Why it matters for a self-driving car |
|---|---|
| **Spatial Transformer Network (STN)** | Learns to zoom, rotate and centre the sign before classifying it, so imperfect crops from the detector still work. |
| **Deep CNN** (7 conv layers, BatchNorm, Dropout, flattened FC head, ~4.5 M parameters) | Learns fine details such as the digits on speed-limit signs; the flattened head keeps spatial layout ("80" vs "50"). |
| **Luminance contrast stretch** (autocontrast) | Makes dark (night), shadowed and hazy signs readable without washing out detail. |
| **Heavy GPU augmentation** (rotation, zoom, shear, shift, low resolution, brightness, contrast, saturation, blur, noise, occlusion) | Simulates camera angles, weather, lighting, dirt and stickers. **No horizontal flips**: flipping turns "turn left" into "turn right". |
| **Class-balanced sampling + label smoothing** | Rare signs (e.g. 20 km/h) are learned as well as common ones; the model is less overconfident. |
| **Track-based validation split** | GTSRB films each physical sign about 30 times. Keeping whole tracks together means validation measures performance on signs the model has never seen. |
| **Ensemble + test-time augmentation** | Averages several independently trained models over several views of each image, which cuts errors further. |
| **Confidence threshold** | Below 70% confidence the system reports **UNCERTAIN** instead of guessing, so the car can fall back on maps or other sensors. |

## Project structure

| File | Purpose |
|---|---|
| `common.py` | Class names, preprocessing, the `TrafficSignNet` model, ensemble/TTA inference |
| `data.py` | Downloads GTSRB, crops each image to its sign, caches tensors |
| `train.py` | Trains one model (GPU augmentation, OneCycle LR, AdamW, mixed precision) |
| `evaluate.py` | Test accuracy, per-class report, confusion matrix, robustness tests, threshold analysis |
| `predict.py` | Command-line prediction for image files |
| `app.py` | Desktop GUI: open an image and see the prediction |

## Usage

```bash
pip install -r requirements.txt
python data.py                                   # download + preprocess dataset (once)
python train.py --seed 0 --epochs 30             # model with validation monitoring
python train.py --seed 1 --epochs 30 --val-frac 0  # extra ensemble members trained on all data
python train.py --seed 2 --epochs 30 --val-frac 0
python evaluate.py                               # full report into results/
python predict.py my_sign.jpg                    # classify an image
python app.py                                    # GUI demo
```

Every `models/model_seed*.pt` file is automatically used in the ensemble.

> If the dataset download hangs, your network may have broken IPv6. Download the three
> GTSRB archives manually (e.g. `curl -4 ...`) into `data/gtsrb/` and `data/gtsrb_full/`.

## Results

| System | Test accuracy |
|---|---|
| Single model (seed 0 / 1 / 2) | 99.67% / 99.73% / 99.77% |
| Ensemble of 3 models | 99.83% |
| **Ensemble + test-time augmentation** | **99.84%** |

**Safety threshold (70% confidence):** the system answers 98.3% of signs with **99.97%** accuracy
and flags the rest as UNCERTAIN.

**Robustness** (simulated conditions, test set): night 99.71%, fog 99.53%, blur 99.78%,
sensor noise 99.48%, low resolution 99.08%, strong glare 96.34%,
large grey patch over the sign's centre 78.42% (the hardest case: the patch often hides the digits).

**Key finding during development:** histogram-equalisation preprocessing washed out many signs
(capping accuracy around 97.7%). Replacing it with a linear contrast stretch raised single-model accuracy to 99.7%.

See `results/`:
- `metrics.json`: accuracy of each model, the ensemble, ensemble + TTA, robustness, threshold analysis
- `classification_report.txt`: precision / recall / F1 for each of the 43 classes
- `confusion_matrix.png`, `misclassified.png`, `robustness.png`, `training_curves_seed*.png`

## Limitations

- Heavy occlusion of the sign's centre (stickers, snow over the digits) is the weakest case (78%).
- GTSRB contains German signs only. Signs from other countries need extra training data.
- This is a **classifier**: it expects an image already cropped around a sign. A full car
  pipeline adds a detector (e.g. YOLO) that finds signs in the camera frame first.
- No vision model is 100% reliable. In a real vehicle this must be one input among several
  (HD maps, GPS speed-limit databases, sensor fusion), which is why low-confidence
  predictions are flagged.
