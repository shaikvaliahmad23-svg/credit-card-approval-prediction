"""
app.py
-------
Flask web application that serves the trained credit-card-approval model.

Usage:
    cd app
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained artifacts once, at startup
# ---------------------------------------------------------------------------
with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODELS_DIR, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)
with open(os.path.join(MODELS_DIR, "model_metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

USES_SCALED_INPUT = metadata.get("uses_scaled_input", False)
BEST_MODEL_NAME = metadata.get("best_model_name", "Model")

CATEGORICAL_COLS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
]

FORM_OPTIONS = {
    "CODE_GENDER": ["M", "F"],
    "FLAG_OWN_CAR": ["Y", "N"],
    "FLAG_OWN_REALTY": ["Y", "N"],
    "NAME_INCOME_TYPE": ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
    "NAME_EDUCATION_TYPE": ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"],
    "NAME_FAMILY_STATUS": ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
    "NAME_HOUSING_TYPE": ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"],
    "OCCUPATION_TYPE": ["Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
                         "High skill tech staff", "Accountants", "Medicine staff",
                         "Security staff", "Cooking staff", "Cleaning staff", "Other"],
}


def build_input_row(form):
    """Convert the submitted form into a single-row DataFrame matching the
    schema the model was trained on."""
    data = {
        "CODE_GENDER": form.get("gender"),
        "FLAG_OWN_CAR": form.get("own_car"),
        "FLAG_OWN_REALTY": form.get("own_realty"),
        "CNT_CHILDREN": int(form.get("children", 0)),
        "AMT_INCOME_TOTAL": float(form.get("income", 0)),
        "NAME_INCOME_TYPE": form.get("income_type"),
        "NAME_EDUCATION_TYPE": form.get("education_type"),
        "NAME_FAMILY_STATUS": form.get("family_status"),
        "NAME_HOUSING_TYPE": form.get("housing_type"),
        "FLAG_MOBIL": 1,
        "FLAG_WORK_PHONE": int(form.get("work_phone", 0)),
        "FLAG_PHONE": int(form.get("phone", 0)),
        "FLAG_EMAIL": int(form.get("email", 0)),
        "OCCUPATION_TYPE": form.get("occupation_type"),
        "CNT_FAM_MEMBERS": float(form.get("family_members", 1)),
        "AGE_YEARS": int(form.get("age", 30)),
        "YEARS_EMPLOYED": int(form.get("years_employed", 0)),
        "IS_PENSIONER_NO_JOB": 1 if form.get("income_type") == "Pensioner" and float(form.get("years_employed", 0)) == 0 else 0,
    }
    df = pd.DataFrame([data])
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    # Align columns to what the model expects (add any missing dummy cols as 0)
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)
    return df_aligned


def predict(df_aligned):
    if USES_SCALED_INPUT:
        X = scaler.transform(df_aligned)
    else:
        X = df_aligned.values
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]  # probability of class 1 = high risk / rejected
    return int(pred), float(proba)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", options=FORM_OPTIONS, model_name=BEST_MODEL_NAME)


@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        row = build_input_row(request.form)
        pred, proba = predict(row)

        approved = pred == 0
        confidence = (1 - proba) if approved else proba

        return render_template(
            "result.html",
            approved=approved,
            confidence=round(confidence * 100, 1),
            model_name=BEST_MODEL_NAME,
        )
    except Exception as e:
        return render_template("result.html", error=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)