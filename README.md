# Credit Card Approval Prediction

Banks receive thousands of credit card applications a day, and manually
reviewing every one is slow and error-prone. This project automates the
approval decision with machine learning: it trains four classification
algorithms on historical-style applicant data and serves the best one
through a Flask web application, so an analyst (or applicant) can get an
instant approve/reject decision.

## What's inside

- **Synthetic dataset generator** (`data/generate_dataset.py`) — builds a
  realistic applicant dataset shaped like the classic
  `application_record.csv` + `credit_record.csv` credit-approval data:
  demographics, income, employment, housing, and a monthly repayment
  `STATUS` history that gets collapsed into a binary risk label
  (`TARGET`: 0 = approve, 1 = reject) — exactly the "multi-class payment
  status → binary label" feature engineering step described in the brief.
- **Preprocessing pipeline** (`src/data_preprocessing.py`) — cleaning,
  one-hot encoding, scaling, and a stratified train/test split.
- **Model training** (`src/train_models.py`) — trains and compares:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost (Gradient Boosting)

  Each model is scored on Accuracy, Precision, Recall, F1, and ROC-AUC.
  The best model (by ROC-AUC) is pickled to `models/best_model.pkl` along
  with its scaler and feature-column list, and a comparison chart is saved
  to `outputs/model_comparison.png`.
- **Flask web app** (`app/`) — an applicant-intake form that calls the
  saved model and returns an instant Approved/Rejected decision with a
  confidence gauge.
- **IBM Watson ML notes** (`notebooks/watson_ml_deployment.md`) — the steps
  to package and deploy `best_model.pkl` on IBM Watson Machine Learning for
  a cloud-hosted, scalable endpoint.

## Project structure

```
credit-card-approval-prediction/
├── app/
│   ├── app.py                  # Flask application
│   ├── templates/
│   │   ├── index.html          # applicant intake form
│   │   └── result.html         # decision page
│   └── static/
│       └── style.css
├── data/
│   ├── generate_dataset.py     # synthetic dataset generator
│   └── credit_card_approval.csv
├── src/
│   ├── data_preprocessing.py
│   └── train_models.py
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── model_metadata.pkl
├── outputs/
│   ├── model_comparison.png
│   └── model_comparison.csv
├── notebooks/
│   └── watson_ml_deployment.md
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Regenerate the dataset (optional — a CSV is already included)

```bash
cd data
python generate_dataset.py
```

### 2. Train the models

```bash
cd src
python train_models.py
```

This prints a metrics table for all four models, saves the comparison
chart to `outputs/model_comparison.png`, and writes the best model's
artifacts to `models/`.

### 3. Run the web app

```bash
cd app
python app.py
```

Open **http://127.0.0.1:5000** and fill in an applicant's details to get
an instant decision.

## Model results

| Model                | Accuracy | Precision | Recall | F1     | ROC-AUC |
|----------------------|:--------:|:---------:|:------:|:------:|:-------:|
| **Logistic Regression** | 0.663 | 0.470 | 0.672 | 0.553 | **0.718** |
| Decision Tree        | 0.630    | 0.435     | 0.634  | 0.516  | 0.674   |
| Random Forest        | 0.672    | 0.479     | 0.617  | 0.539  | 0.712   |
| XGBoost              | 0.721    | 0.604     | 0.296  | 0.398  | 0.710   |

(From the included run &mdash; see `outputs/model_comparison.csv`, or
re-run `src/train_models.py` to regenerate.) **Logistic Regression** came
out on top by ROC-AUC and is the model saved to `models/best_model.pkl`
and served by the Flask app. `class_weight="balanced"` is used for the
non-boosted models since approvals/rejections are imbalanced in the data.

## Note on the dataset

No dataset was provided alongside the project brief, so
`data/generate_dataset.py` synthesizes a dataset with the same shape,
column semantics, and feature-engineering step (binary risk label derived
from multi-class payment `STATUS` codes) as the real-world Kaggle "Credit
Card Approval Prediction" dataset. **To use real applicant data**, replace
`data/credit_card_approval.csv` with your own file that has the same
columns (see the column list in `src/data_preprocessing.py`), then re-run
`src/train_models.py`.

## Skills used

XGBoost · Scikit-learn · Decision Tree Learning · Random Forest · NumPy ·
Matplotlib · Flask · Python
