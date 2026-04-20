import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Global model variable — loaded lazily so import does not crash during tests
model = None

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

HOME = """
<!DOCTYPE html>
<html>
<head><title>Diabetes Predictor</title></head>
<body>
  <h2>Diabetes Prediction</h2>
  <form action="/predict" method="post">
    {% for f in features %}
      <label>{{ f }}: <input type="number" step="any" name="{{ f }}" required></label><br><br>
    {% endfor %}
    <button type="submit">Predict</button>
  </form>
  {% if result is defined %}
    <h3>Result: {{ result }}</h3>
  {% endif %}
</body>
</html>
"""


def load_model():
    global model
    if model is None:
        model = pickle.load(open("models/model.pkl", "rb"))
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

    if request.is_json:
        data = request.get_json()
        try:
            features = np.array([float(data[f]) for f in FEATURES]).reshape(1, -1)
        except KeyError as e:
            return jsonify({"error": f"Missing field: {e}"}), 400

        prediction = int(m.predict(features)[0])
        label = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return jsonify({"prediction": prediction, "label": label})

    else:
        try:
            features = np.array([float(request.form[f]) for f in FEATURES]).reshape(1, -1)
        except (KeyError, ValueError) as e:
            return f"Error: {e}", 400

        prediction = int(m.predict(features)[0])
        label = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return render_template_string(HOME, features=FEATURES, result=label)
