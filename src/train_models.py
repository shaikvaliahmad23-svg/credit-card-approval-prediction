"""
train_models.py
-----------------
Trains four classification algorithms on the credit card approval dataset:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. XGBoost (Gradient Boosting)

Evaluates each with Accuracy, Precision, Recall, F1, and ROC-AUC, plots a
comparison chart, and saves the best-performing model (by ROC-AUC) plus the
scaler and feature-column list needed by the Flask app for inference.

Run from the project root:
    python src/train_models.py
"""

import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)

from data_preprocessing import (
    load_data, build_feature_target, train_test_split_data, scale_features,
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    # Falls back to sklearn's GradientBoostingClassifier if xgboost isn't
    # installed in the current environment. Install `xgboost` (see
    # requirements.txt) to use the real XGBoost implementation.
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    HAS_XGB = False

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "credit_card_approval.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def get_models():
    xgb_kwargs = dict(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    if not HAS_XGB:
        # GradientBoostingClassifier doesn't accept use_label_encoder/eval_metric
        xgb_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    else:
        xgb_model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42, eval_metric="logloss", use_label_encoder=False,
        )

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": xgb_model,
    }


def evaluate(model, X_test, y_test, needs_scaled=False):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }


def plot_comparison(results_df, out_path):
    ax = results_df.plot(
        kind="bar", figsize=(10, 6), rot=0,
        colormap="viridis", edgecolor="black"
    )
    ax.set_title("Model Comparison — Credit Card Approval Prediction")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    X, y, feature_cols = build_feature_target(df)

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    models = get_models()
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            scores = evaluate(model, X_test_scaled, y_test)
        else:
            model.fit(X_train, y_train)
            scores = evaluate(model, X_test, y_test)
        results[name] = scores
        trained_models[name] = model
        print(f"  -> {scores}")

    results_df = pd.DataFrame(results).T
    print("\n=== Model Comparison ===")
    print(results_df.round(4))

    plot_path = os.path.join(OUTPUTS_DIR, "model_comparison.png")
    plot_comparison(results_df, plot_path)
    print(f"\nSaved comparison chart to {plot_path}")

    best_name = results_df["ROC-AUC"].idxmax()
    best_model = trained_models[best_name]
    print(f"\nBest model by ROC-AUC: {best_name}")

    # Persist artifacts needed by the Flask app
    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    with open(os.path.join(MODELS_DIR, "model_metadata.pkl"), "wb") as f:
        pickle.dump({
            "best_model_name": best_name,
            "uses_scaled_input": best_name == "Logistic Regression",
            "metrics": results_df.to_dict(orient="index"),
        }, f)

    results_df.round(4).to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"))
    print("\nSaved model artifacts to models/ and results to outputs/")


if __name__ == "__main__":
    main()
