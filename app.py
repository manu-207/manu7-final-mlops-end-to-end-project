import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load model at startup
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run the DVC pipeline first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# Feature names based on the diabetes dataset
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Diabetes Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; }
        h1 { color: #333; }
        label { display: block; margin-top: 10px; font-weight: bold; }
        input { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; }
        button { margin-top: 20px; padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; font-size: 16px; }
        button:hover { background: #45a049; }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; font-size: 18px; }
        .positive { background: #ffcccc; color: #c00; }
        .negative { background: #ccffcc; color: #060; }
    </style>
</head>
<body>
    <h1>🩺 Diabetes Prediction</h1>
    <form method="POST" action="/predict_form">
        {% for feature in features %}
        <label>{{ feature }}</label>
        <input type="number" step="any" name="{{ feature }}" required>
        {% endfor %}
        <button type="submit">Predict</button>
    </form>
    {% if result is defined %}
    <div class="result {{ 'positive' if result == 1 else 'negative' }}">
        Prediction: <strong>{{ 'Diabetic' if result == 1 else 'Not Diabetic' }}</strong>
        (Probability: {{ probability }}%)
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, features=FEATURES)


@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        values = [float(request.form[f]) for f in FEATURES]
        input_df = pd.DataFrame([values], columns=FEATURES)
        prediction = int(model.predict(input_df)[0])
        probability = round(float(model.predict_proba(input_df)[0][prediction]) * 100, 2)
        return render_template_string(
            HTML_TEMPLATE, features=FEATURES, result=prediction, probability=probability
        )
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, features=FEATURES, result=-1, probability=0), 400


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON API endpoint.
    Expected payload:
    {
        "Pregnancies": 2, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20,
        "Insulin": 85, "BMI": 28.5, "DiabetesPedigreeFunction": 0.5, "Age": 35
    }
    """
    try:
        data = request.get_json(force=True)
        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        values = [float(data[f]) for f in FEATURES]
        input_df = pd.DataFrame([values], columns=FEATURES)

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][prediction])

        return jsonify({
            "prediction": prediction,
            "label": "Diabetic" if prediction == 1 else "Not Diabetic",
            "probability": round(probability, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
