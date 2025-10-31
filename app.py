# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and column names
model = joblib.load("models/estimator.pkl")
columns = joblib.load("models/names.pkl")

# Mapping back encoded labels to names
label_map = {0: "green", 1: "orange", 2: "red", 3: "yellow"}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    if request.method == "POST":
        try:
            # Grab input values from form
            magnitude = float(request.form["magnitude"])
            depth = float(request.form["depth"])
            cdi = float(request.form["cdi"])
            mmi = float(request.form["mmi"])
            sig = float(request.form["sig"])

            # Build feature array
            X = np.array([[magnitude, depth, cdi, mmi, sig]])

            # Predict
            pred_label = model.predict(X)[0]
            proba = None
            # Check if model supports predict_proba
            if hasattr(model.named_steps["model"], "predict_proba"):
                proba_array = model.predict_proba(X)[0]
                confidence = round(100 * proba_array[pred_label], 2)

            result = label_map[pred_label]
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    app.run()
