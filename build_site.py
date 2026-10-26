"""Build the static GitHub Pages site in docs/ from the Flask dashboard template.

    python export_onnx.py   # docs/model.onnx (whole ensemble, runs in the browser)
    python build_site.py    # docs/index.html + samples, charts, metrics
"""
import shutil

from common import RESULTS_DIR, ROOT

DOCS = ROOT / "docs"
REPO_URL = "https://github.com/Velerion/TrafficSign"
CHARTS = ["confusion_matrix.png", "misclassified.png", "robustness.png", "training_curves_seed0.png"]
ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/"

# In-browser replacements for the Flask endpoints: preprocessing identical to
# common.preprocess() (luminance autocontrast with 1% cutoff, like PIL), then ONNX inference.
BROWSER_JS = r"""
function loadSamples() {
  $("samples").innerHTML = SAMPLES.map((f) =>
    `<button title="${esc(f.replace(".png", "").replace(/_/g, " "))}" data-f="${esc(f)}"><img src="samples/${f}" alt=""></button>`).join("");
  $("samples").querySelectorAll("button").forEach((b) => b.onclick = async () => {
    const blob = await (await fetch(`samples/${b.dataset.f}`)).blob();
    predict(new File([blob], b.dataset.f, { type: blob.type }));
  });
}

const THRESHOLD = 0.7, S = 48;
let session = null;

async function loadModel() {
  ort.env.wasm.wasmPaths = "__ORT__";
  const res = await fetch("model.onnx");
  const total = +res.headers.get("content-length") || 54.4e6;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  while (true) {
    const step = await reader.read();
    if (step.done) break;
    chunks.push(step.value);
    got += step.value.length;
    $("dl").textContent = Math.min(100, Math.round(got / total * 100)) + "%";
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
  $("empty").textContent = "Upload an image or click a sample to see the result.";
}

function preprocess(img) {
  const scale = Math.min(1, 512 / Math.max(img.width, img.height));
  const w = Math.max(1, Math.round(img.width * scale)), h = Math.max(1, Math.round(img.height * scale));
  const c1 = Object.assign(document.createElement("canvas"), { width: w, height: h });
  const g1 = c1.getContext("2d", { willReadFrequently: true });
  g1.drawImage(img, 0, 0, w, h);
  const d = g1.getImageData(0, 0, w, h), p = d.data, n = w * h;
  const Y = new Float32Array(n), hist = new Array(256).fill(0);
  for (let i = 0; i < n; i++) {
    const y = 0.299 * p[4 * i] + 0.587 * p[4 * i + 1] + 0.114 * p[4 * i + 2];
    Y[i] = y; hist[Math.min(255, Math.round(y))]++;
  }
  let cut = Math.floor(n / 100);
  for (let i = 0; i < 256 && cut > 0; i++) { const t = Math.min(cut, hist[i]); hist[i] -= t; cut -= t; }
  cut = Math.floor(n / 100);
  for (let i = 255; i >= 0 && cut > 0; i--) { const t = Math.min(cut, hist[i]); hist[i] -= t; cut -= t; }
  let lo = 0, hi = 255;
  while (lo < 255 && hist[lo] === 0) lo++;
  while (hi > 0 && hist[hi] === 0) hi--;
  const sc = hi > lo ? 255 / (hi - lo) : 1, off = hi > lo ? -lo * sc : 0;
  for (let i = 0; i < n; i++) {
    const r = p[4 * i], g = p[4 * i + 1], b = p[4 * i + 2];
    const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b, cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    const y = Math.max(0, Math.min(255, Math.floor(Math.round(Y[i]) * sc + off)));
    p[4 * i] = y + 1.402 * (cr - 128);
    p[4 * i + 1] = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128);
    p[4 * i + 2] = y + 1.772 * (cb - 128);
  }
  g1.putImageData(d, 0, 0);
  const c2 = Object.assign(document.createElement("canvas"), { width: S, height: S });
  const g2 = c2.getContext("2d", { willReadFrequently: true });
  g2.imageSmoothingEnabled = true; g2.imageSmoothingQuality = "high";
  g2.drawImage(c1, 0, 0, S, S);
  const q = g2.getImageData(0, 0, S, S).data, x = new Float32Array(3 * S * S);
  for (let i = 0; i < S * S; i++) for (let ch = 0; ch < 3; ch++) x[ch * S * S + i] = q[4 * i + ch] / 255;
  return x;
}

function tensorToPng(arr) {
  const c = Object.assign(document.createElement("canvas"), { width: S, height: S });
  const g = c.getContext("2d"), im = g.createImageData(S, S);
  for (let i = 0; i < S * S; i++) {
    for (let ch = 0; ch < 3; ch++) im.data[4 * i + ch] = Math.round(arr[ch * S * S + i] * 255);
    im.data[4 * i + 3] = 255;
  }
  g.putImageData(im, 0, 0);
  return c.toDataURL();
}

async function runModel(file) {
  const img = await createImageBitmap(file);
  const t0 = performance.now();
  const x = preprocess(img);
  const out = await session.run({ image: new ort.Tensor("float32", x, [1, 3, S, S]) });
  const probs = Array.from(out.probs.data);
  const latency = performance.now() - t0;
  const top5 = probs.map((c, id) => ({ id, name: CLASS_NAMES[id], conf: c }))
    .sort((a, b) => b.conf - a.conf).slice(0, 5);
  return {
    label: top5[0].name, class_id: top5[0].id, confidence: top5[0].conf,
    status: top5[0].conf >= THRESHOLD ? "OK" : "UNCERTAIN", top5, latency_ms: Math.round(latency),
    input_url: tensorToPng(x), stn_url: tensorToPng(out.aligned.data),
  };
}

"""

OLD_FETCH = """  const fd = new FormData(); fd.append("image", file);
  let r;
  try {
    const res = await fetch("/api/predict", { method: "POST", body: fd });
    r = await res.json();
    if (!res.ok) throw new Error(r.error || "Prediction failed");
  } catch (e) {
    $("empty").innerHTML = `<span class="error">${esc(e.message)}</span>`;
    return;
  }"""

NEW_FETCH = """  if (!session) { $("empty").innerHTML = '<span class="spinner"></span> Model still loading, please wait…'; return; }
  let r;
  try {
    r = await runModel(file);
  } catch (e) {
    $("empty").innerHTML = '<span class="error">Could not read that file as an image.</span>';
    return;
  }"""


def sub(html, old, new):
    assert old in html, f"template changed, cannot find: {old[:60]!r}"
    return html.replace(old, new)


def main():
    from common import CLASS_NAMES
    samples = sorted(p.name for p in (ROOT / "samples").glob("*.png"))
    h = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")

    for old, new in [
        ("{{ threshold / 100 }}", "0.7"), ("{{ n_classes }}", "43"), ("{{ n_models }}", "3"), ("{{ threshold }}", "70"),
        ("running on <b>{{ device }}</b>", "runs <b>in your browser</b>"),
        ('<meta name="viewport" content="width=device-width, initial-scale=1">',
         '<meta name="viewport" content="width=device-width, initial-scale=1">\n<meta name="description" content="'
         'Traffic sign recognition with a CNN + Spatial Transformer ensemble (99.84% on GTSRB), running in your browser.">'),
        ("<p class=\"sub\">CNN + Spatial Transformer ensemble, trained on GTSRB (43 sign classes)</p>",
         "<p class=\"sub\">CNN + Spatial Transformer ensemble, trained on GTSRB (43 sign classes). Runs entirely in your "
         f"browser: images never leave your device. <a href=\"{REPO_URL}\" style=\"color:var(--accent)\">Source code</a></p>"),
        ('<div class="empty" id="empty">Upload an image or click a sample to see the result.</div>',
         '<div class="empty" id="empty"><span class="spinner"></span> Loading neural network (54 MB, first visit only)… '
         '<span id="dl"></span></div>'),
        ('src="/charts/confusion_matrix.png"', 'src="charts/confusion_matrix.png"'),
        ('$("chart").src = "/charts/" + b.dataset.img;', '$("chart").src = "charts/" + b.dataset.img;'),
        ('const m = await (await fetch("/api/metrics")).json();', 'const m = await (await fetch("metrics.json")).json();'),
        (OLD_FETCH, NEW_FETCH),
        ('$("v-in").src = "data:image/png;base64," + r.input_png;', '$("v-in").src = r.input_url;'),
        ('$("v-stn").src = "data:image/png;base64," + r.stn_png;', '$("v-stn").src = r.stn_url;'),
        ("loadMetrics(); loadSamples();", "loadMetrics(); loadSamples();\nloadModel().catch((e) => { $(\"empty\").innerHTML = "
         "`<span class=\"error\">Could not load the model: ${esc(e.message)}</span>`; });"),
        ("<script>\nconst $", f'<script src="{ORT}ort.min.js"></script>\n<script>\nconst $'),
    ]:
        h = sub(h, old, new)

    start, end = h.index("async function loadSamples() {"), h.index("async function predict(file) {")
    consts = (f"const SAMPLES = {samples!r};\nconst CLASS_NAMES = {list(CLASS_NAMES)!r};\n").replace("'", '"')
    h = h[:start] + consts + BROWSER_JS.replace("__ORT__", ORT) + h[end:]
    assert "{{" not in h and "/api/" not in h

    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(h, encoding="utf-8")
    (DOCS / ".nojekyll").touch()
    shutil.copytree(ROOT / "samples", DOCS / "samples", dirs_exist_ok=True)
    (DOCS / "charts").mkdir(exist_ok=True)
    for c in CHARTS:
        shutil.copy(RESULTS_DIR / c, DOCS / "charts" / c)
    shutil.copy(RESULTS_DIR / "metrics.json", DOCS / "metrics.json")
    print(f"Built {DOCS / 'index.html'} ({len(h) / 1e3:.0f} kB); model: {(DOCS / 'model.onnx').exists()}")


if __name__ == "__main__":
    main()
