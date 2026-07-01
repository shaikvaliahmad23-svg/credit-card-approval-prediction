"""
data_preprocessing.py
----------------------
Cleans the raw merged dataset and builds the feature matrix used by all
four models (Logistic Regression, Random Forest, XGBoost, Decision Tree).

Steps:
1. Load CSV
2. Handle missing values
3. Encode categorical variables (one-hot)
4. Scale numeric features (for Logistic Regression)
5. Train/test split (stratified, since classes are imbalanced)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
]

NUMERIC_COLS = [
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "FLAG_MOBIL", "FLAG_WORK_PHONE",
    "FLAG_PHONE", "FLAG_EMAIL", "CNT_FAM_MEMBERS", "AGE_YEARS",
    "YEARS_EMPLOYED", "IS_PENSIONER_NO_JOB",
]

TARGET_COL = "TARGET"
ID_COL = "ID"


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.drop_duplicates()
    # Simple, transparent missing-value handling
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def encode_features(df):
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    return df_encoded


def build_feature_target(df):
    df = clean_data(df)
    df_encoded = encode_features(df)

    feature_cols = [c for c in df_encoded.columns if c not in (TARGET_COL, ID_COL)]
    X = df_encoded[feature_cols]
    y = df_encoded[TARGET_COL]
    return X, y, feature_cols


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    df = load_data("../data/credit_card_approval.csv")
    X, y, cols = build_feature_target(df)
    print("Feature matrix shape:", X.shape)
    print("Target distribution:\n", y.value_counts(normalize=True))
