import pickle
import time
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge

app = Flask(__name__)

metrics = PrometheusMetrics(app)
metrics.info("app_info", "Diabetes Predictor", version="1.0.0")

PREDICTION_COUNTER = Counter(
    "diabetes_predictions_total",
    "Total predictions",
    ["result"]   # 'diabetic' or 'non_diabetic'
)
PREDICTION_LATENCY = Histogram(
    "diabetes_prediction_latency_seconds",
    "Prediction latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)
MODEL_LOADED = Gauge("diabetes_model_loaded", "1 if model is loaded")
GLUCOSE_HISTOGRAM = Histogram(
    "diabetes_input_glucose",
    "Glucose input distribution",
    buckets=[50, 80, 100, 120, 140, 160, 180, 200, 250]
)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

HOME = """..."""  # keep your existing HTML

model = None

def load_model():
    global model
    if model is None:
        model = pickle.load(open("models/model.pkl", "rb"))
        MODEL_LOADED.set(1)
    return model

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HOME, features=FEATURES)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    m = load_model()
    start = time.time()

    if request.is_json:
        data = request.get_json()
        try:
            vals = np.array([float(data[f]) for f in FEATURES]).reshape(1, -1)
        except KeyError as e:
            return jsonify({"error": f"Missing field: {e}"}), 400
        prediction = int(m.predict(vals)[0])
        label = "Diabetic" if prediction == 1 else "Non-Diabetic"
        PREDICTION_LATENCY.observe(time.time() - start)
        PREDICTION_COUNTER.labels(result="diabetic" if prediction == 1 else "non_diabetic").inc()
        GLUCOSE_HISTOGRAM.observe(float(data.get("Glucose", 0)))
        return jsonify({"prediction": prediction, "label": label})
    else:
        try:
            vals = np.array([float(request.form[f]) for f in FEATURES]).reshape(1, -1)
        except (KeyError, ValueError) as e:
            return f"Error: {e}", 400
        prediction = int(m.predict(vals)[0])
        label = "Diabetic" if prediction == 1 else "Non-Diabetic"
        PREDICTION_LATENCY.observe(time.time() - start)
        PREDICTION_COUNTER.labels(result="diabetic" if prediction == 1 else "non_diabetic").inc()
        GLUCOSE_HISTOGRAM.observe(float(request.form.get("Glucose", 0)))
        return render_template_string(HOME, features=FEATURES, result=label)
