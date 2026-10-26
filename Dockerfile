# Web dashboard image (CPU inference; no GPU needed).
FROM python:3.12-slim

WORKDIR /app
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY common.py web_app.py ./
COPY templates/ templates/
COPY models/ models/
COPY samples/ samples/
COPY results/metrics.json results/confusion_matrix.png results/misclassified.png \
     results/robustness.png results/training_curves_seed0.png results/

ENV PORT=7860
EXPOSE 7860
# One worker keeps a single copy of the models in memory; threads handle concurrent requests.
CMD gunicorn --workers 1 --threads 4 --timeout 60 --bind 0.0.0.0:$PORT web_app:app
