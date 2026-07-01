"""
generate_dataset.py
--------------------
Generates a synthetic 'Credit Card Approval' dataset that mirrors the
structure of the well-known application_record.csv + credit_record.csv
data used in real-world credit card approval prediction projects.

Two raw tables are simulated and then merged/feature-engineered into a
single modeling-ready CSV, exactly the way a real pipeline would work:

1. application_record  -> demographic / financial info per applicant
2. credit_record        -> monthly repayment STATUS history per applicant
                            (0-5 = days past due buckets, C = paid off,
                             X = no loan that month)

The STATUS history is collapsed into a binary TARGET:
    TARGET = 1  -> "high risk" / REJECTED  (ever seriously past due)
    TARGET = 0  -> "low risk"  / APPROVED

Run:
    python generate_dataset.py
Produces:
    credit_card_approval.csv
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_APPLICANTS = 12000

rng = np.random.default_rng(RANDOM_STATE)


def generate_application_record(n=N_APPLICANTS):
    ids = np.arange(1000000, 1000000 + n)

    gender = rng.choice(["M", "F"], size=n, p=[0.42, 0.58])
    own_car = rng.choice(["Y", "N"], size=n, p=[0.4, 0.6])
    own_realty = rng.choice(["Y", "N"], size=n, p=[0.55, 0.45])
    cnt_children = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.55, 0.22, 0.15, 0.06, 0.02])

    income_type = rng.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
        size=n, p=[0.52, 0.22, 0.14, 0.11, 0.01]
    )
    education_type = rng.choice(
        ["Secondary / secondary special", "Higher education", "Incomplete higher",
         "Lower secondary", "Academic degree"],
        size=n, p=[0.65, 0.24, 0.06, 0.03, 0.02]
    )
    family_status = rng.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        size=n, p=[0.62, 0.16, 0.09, 0.08, 0.05]
    )
    housing_type = rng.choice(
        ["House / apartment", "With parents", "Municipal apartment",
         "Rented apartment", "Office apartment", "Co-op apartment"],
        size=n, p=[0.78, 0.09, 0.06, 0.04, 0.02, 0.01]
    )
    occupation_type = rng.choice(
        ["Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
         "High skill tech staff", "Accountants", "Medicine staff",
         "Security staff", "Cooking staff", "Cleaning staff", "Other"],
        size=n
    )

    # Income correlated loosely with education/income type
    base_income = rng.normal(180000, 70000, size=n)
    base_income += np.where(education_type == "Higher education", 40000, 0)
    base_income += np.where(education_type == "Academic degree", 90000, 0)
    base_income += np.where(income_type == "Pensioner", -50000, 0)
    amt_income_total = np.clip(base_income, 27000, 1500000).round(-2)

    # Age in days-before-today (negative), 21-70 years old
    age_years = rng.integers(21, 70, size=n)
    days_birth = -(age_years * 365 + rng.integers(0, 365, size=n))

    # Employment length in days-before-today (negative); pensioners get +365243 (real-world sentinel)
    employed_years = rng.integers(0, 40, size=n)
    days_employed = -(employed_years * 365 + rng.integers(0, 365, size=n))
    days_employed = np.where(income_type == "Pensioner", 365243, days_employed)

    flag_mobil = np.ones(n, dtype=int)
    flag_work_phone = rng.choice([0, 1], size=n, p=[0.77, 0.23])
    flag_phone = rng.choice([0, 1], size=n, p=[0.71, 0.29])
    flag_email = rng.choice([0, 1], size=n, p=[0.91, 0.09])

    cnt_fam_members = (cnt_children + rng.choice([1, 2], size=n, p=[0.35, 0.65])).astype(float)

    df = pd.DataFrame({
        "ID": ids,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": cnt_children,
        "AMT_INCOME_TOTAL": amt_income_total,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education_type,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing_type,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "FLAG_MOBIL": flag_mobil,
        "FLAG_WORK_PHONE": flag_work_phone,
        "FLAG_PHONE": flag_phone,
        "FLAG_EMAIL": flag_email,
        "OCCUPATION_TYPE": occupation_type,
        "CNT_FAM_MEMBERS": cnt_fam_members,
    })
    return df


def generate_credit_record(app_df):
    """Simulate monthly STATUS history per applicant and derive a TARGET."""
    rows = []
    # Applicants with lower income, shorter employment, and renters skew riskier
    risk_score = (
        -(app_df["AMT_INCOME_TOTAL"] - app_df["AMT_INCOME_TOTAL"].mean()) / app_df["AMT_INCOME_TOTAL"].std()
        + np.where(app_df["NAME_HOUSING_TYPE"].isin(["Rented apartment", "With parents"]), 0.6, 0)
        + np.where(app_df["NAME_INCOME_TYPE"] == "Student", 0.5, 0)
        + rng.normal(0, 1, size=len(app_df))
    )
    prob_bad = 1 / (1 + np.exp(-(risk_score - 1.2)))  # logistic squashing, tuned for ~18% bad rate

    statuses = []
    for i, p_bad in enumerate(prob_bad):
        n_months = rng.integers(6, 30)
        is_bad = rng.random() < p_bad
        if is_bad:
            # at least one seriously-past-due month (status 2-5)
            worst = rng.choice(["2", "3", "4", "5"], p=[0.55, 0.25, 0.12, 0.08])
        else:
            worst = rng.choice(["C", "0", "X"], p=[0.55, 0.35, 0.10])
        statuses.append(worst)

    app_df = app_df.copy()
    app_df["WORST_STATUS"] = statuses
    return app_df


def engineer_features(df):
    """Binary risk label + human-friendly engineered features (matches the
    project's stated feature engineering step: multi-class STATUS -> binary)."""
    bad_statuses = {"2", "3", "4", "5"}
    df["TARGET"] = df["WORST_STATUS"].apply(lambda s: 1 if s in bad_statuses else 0)

    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] // 365).astype(int)
    df["YEARS_EMPLOYED"] = np.where(
        df["DAYS_EMPLOYED"] == 365243, 0, (-df["DAYS_EMPLOYED"] // 365)
    ).astype(int)
    df["IS_PENSIONER_NO_JOB"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)

    df = df.drop(columns=["WORST_STATUS", "DAYS_BIRTH", "DAYS_EMPLOYED"])
    return df


if __name__ == "__main__":
    app_df = generate_application_record()
    merged = generate_credit_record(app_df)
    final_df = engineer_features(merged)

    out_path = "credit_card_approval.csv"
    final_df.to_csv(out_path, index=False)
    print(f"Saved {len(final_df)} rows to {out_path}")
    print(final_df["TARGET"].value_counts(normalize=True).rename("share"))
