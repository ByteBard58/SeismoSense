# app.py
from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
from pathlib import Path
from fit import main
import os

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a_static_secret_key_for_development")

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
    if request.method == "POST":
        try:
            magnitude = float(request.form["magnitude"])
            depth = float(request.form["depth"])
            cdi = float(request.form["cdi"])
            mmi = float(request.form["mmi"])
            sig = float(request.form["sig"])

            X = np.array([[magnitude, depth, cdi, mmi, sig]])

            pred_label = model.predict(X)[0]
            confidence = None
            if hasattr(model.named_steps["model"], "predict_proba"):
                proba_array = model.predict_proba(X)[0]
                confidence = round((float(100 * proba_array[pred_label])),2)

            session["result"] = label_map[pred_label]
            session["confidence"] = confidence
        except Exception as e:
            session["result"] = f"Error: {str(e)}"
            session["confidence"] = None

        # PRG: redirect so a page refresh won't re-submit the form
        return redirect(url_for("index"))

    # GET: consume the result from session (one-time display)
    result = session.pop("result", None)
    confidence = session.pop("confidence", None)
    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    app.run()
