import pandas as pd
import pickle
import os
import json
import yaml
import mlflow
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from evidently import Report
from evidently.presets import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesSummary,
    ColumnDriftMetric,
)

# ──────────────────────────────────────────────
# MLflow setup  (same pattern as train.py)
# ──────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://13.233.112.54:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "manu7-mlops"))

params = yaml.safe_load(open("params.yaml"))["train"]

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


def run_monitoring(data_path: str, model_path: str, report_dir: str = "reports"):
    """
    1. Loads the dataset and splits into reference (train) / current (test).
    2. Adds model predictions to both splits.
    3. Runs Evidently DataDrift + DataQuality report.
    4. Saves HTML report  →  reports/data_drift_report.html
    5. Saves JSON summary →  reports/evidently_summary.json
    6. Logs metrics + artifacts to MLflow under run name 'evidently-monitoring'.
    """

    # ── Load data & model ─────────────────────
    data  = pd.read_csv(data_path)
    X     = data[FEATURES]
    y     = data["Outcome"]
    model = pickle.load(open(model_path, "rb"))

    # ── Split: 70 % reference  |  30 % current ─
    X_ref, X_cur, y_ref, y_cur = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # Add predictions so Evidently can track prediction drift too
    X_ref = X_ref.copy()
    X_cur = X_cur.copy()
    X_ref["prediction"] = model.predict(X_ref)
    X_cur["prediction"] = model.predict(X_cur)

    # ── Build Evidently Report ─────────────────
    report = Report(metrics=[
        DatasetDriftMetric(),                          # overall drift share
        DatasetMissingValuesSummary(),                 # data quality
        ColumnDriftMetric(column_name="Glucose"),      # most important feature
        ColumnDriftMetric(column_name="BMI"),
        ColumnDriftMetric(column_name="Age"),
        ColumnDriftMetric(column_name="prediction"),   # prediction drift
        DataDriftPreset(),                             # per-column drift table
        DataQualityPreset(),                           # full quality summary
    ])

    report.run(reference_data=X_ref, current_data=X_cur)

    # ── Save HTML report ───────────────────────
    os.makedirs(report_dir, exist_ok=True)
    html_path = os.path.join(report_dir, "data_drift_report.html")
    report.save_html(html_path)
    print(f"[Evidently] HTML report saved → {html_path}")

    # ── Extract key metrics from JSON ──────────
    result       = report.as_dict()
    metrics_list = result.get("metrics", [])

    drift_share      = None
    missing_share    = None
    glucose_drift    = None
    bmi_drift        = None
    age_drift        = None
    prediction_drift = None

    for m in metrics_list:
        mtype = m.get("metric", "")
        res   = m.get("result", {})

        if mtype == "DatasetDriftMetric":
            drift_share = res.get("share_of_drifted_columns", None)

        elif mtype == "DatasetMissingValuesSummary":
            total_vals   = res.get("current", {}).get("number_of_rows", 1)
            missing_vals = res.get("current", {}).get("number_of_missing_values", 0)
            missing_share = missing_vals / total_vals if total_vals else 0

        elif mtype == "ColumnDriftMetric":
            col = res.get("column_name", "")
            drift_detected = res.get("drift_detected", False)
            if col == "Glucose":
                glucose_drift = int(drift_detected)
            elif col == "BMI":
                bmi_drift = int(drift_detected)
            elif col == "Age":
                age_drift = int(drift_detected)
            elif col == "prediction":
                prediction_drift = int(drift_detected)

    # ── Save JSON summary (for Prometheus scraping) ──
    summary = {
        "drift_share":      round(drift_share, 4)   if drift_share      is not None else 0,
        "missing_share":    round(missing_share, 4) if missing_share    is not None else 0,
        "glucose_drift":    glucose_drift    if glucose_drift    is not None else 0,
        "bmi_drift":        bmi_drift        if bmi_drift        is not None else 0,
        "age_drift":        age_drift        if age_drift        is not None else 0,
        "prediction_drift": prediction_drift if prediction_drift is not None else 0,
    }
    json_path = os.path.join(report_dir, "evidently_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Evidently] JSON summary  saved → {json_path}")
    print(f"[Evidently] Metrics: {summary}")

    # ── Log everything to MLflow ───────────────
    with mlflow.start_run(run_name="evidently-monitoring"):

        # Scalar metrics  → visible in MLflow Metrics tab / charts
        mlflow.log_metric("evidently_drift_share",      summary["drift_share"])
        mlflow.log_metric("evidently_missing_share",    summary["missing_share"])
        mlflow.log_metric("evidently_glucose_drift",    summary["glucose_drift"])
        mlflow.log_metric("evidently_bmi_drift",        summary["bmi_drift"])
        mlflow.log_metric("evidently_age_drift",        summary["age_drift"])
        mlflow.log_metric("evidently_prediction_drift", summary["prediction_drift"])

        # Artifacts  → visible in MLflow Artifacts tab (download & open in browser)
        mlflow.log_artifact(html_path,  artifact_path="evidently")
        mlflow.log_artifact(json_path,  artifact_path="evidently")

        print("[Evidently] Metrics + artifacts logged to MLflow ✅")

    return summary


if __name__ == "__main__":
    summary = run_monitoring(
        data_path  = params["data"],
        model_path = params["model"],
    )
