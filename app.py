# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
from pathlib import Path
from fit import main

app = Flask(__name__)

# paths to the pickle files
MODEL_PATH = Path("models/estimator.pkl")
NAMES_PATH = Path("models/names.pkl")


# inverse encoding
label_map = {0: "green", 1: "orange", 2: "red", 3: "yellow"}

# if pickle files are not found, this function will train the model
def ensure_models():
    print("🔍 Checking for existing model files...")
    if not MODEL_PATH.is_file() or not NAMES_PATH.is_file():
        print("⚠️ Model files not found. Starting training process...")
        main()
        print("✅ Model training complete. Pickle files created.")
    else:
        print("✅ Model files found. Skipping training.")
    
    print("📦 Loading model and feature names...")
    model = joblib.load(MODEL_PATH)
    names = joblib.load(NAMES_PATH)
    print("✅ Model and feature names loaded successfully.")
    return model, names

model, columns = ensure_models()

# main routes
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    if request.method == "POST":
        try:
            magnitude = float(request.form["magnitude"])
            depth = float(request.form["depth"])
            cdi = float(request.form["cdi"])
            mmi = float(request.form["mmi"])
            sig = float(request.form["sig"])

            X = np.array([[magnitude, depth, cdi, mmi, sig]])

            pred_label = model.predict(X)[0]
            if hasattr(model.named_steps["model"], "predict_proba"):
                proba_array = model.predict_proba(X)[0]
                confidence = round(100 * proba_array[pred_label], 2)

            result = label_map[pred_label]
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    app.run()
