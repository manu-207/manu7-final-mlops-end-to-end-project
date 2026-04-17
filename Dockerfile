# ── Stage 1: Build dependencies ──────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies needed for scikit-learn / pandas builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install all project deps + Flask for the API
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir flask gunicorn


# ── Stage 2: Runtime image ────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY src/         ./src/
COPY params.yaml  .
COPY app.py       .

# The trained model is mounted / copied in at runtime.
# If you want to bake the model into the image, uncomment the line below:
# COPY models/ ./models/

# Create directories expected by the pipeline
RUN mkdir -p data/raw data/processed models

# MLflow tracking URI can be overridden at runtime via environment variable
ENV MLFLOW_TRACKING_URI=""
ENV MODEL_PATH="models/model.pkl"
ENV PORT=5001

EXPOSE 5001

# Use gunicorn for production; falls back to Flask dev server in local dev
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "--timeout", "120", "app:app"]
