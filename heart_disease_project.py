"""
Heart Disease Risk Prediction - PBL project (SPSS + Python)
Python part: EDA, Logistic Regression and Decision Tree, evaluation.

Follows the report:
  * processed Cleveland subset, 303 rows, 14 columns (13 predictors + NUM)
  * binary target: NUM = 0 -> heart_disease = 0 ; NUM 1..4 -> heart_disease = 1
  * missing values ('?') in CA and THAL
  * 227 train / 76 test records (test_size = 0.25 of 303 -> 76)
  * metrics: accuracy, precision, recall, specificity, F1, ROC-AUC, confusion matrix

Educational use only - not a diagnostic tool.
"""
import argparse
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")  # save figures to files; no GUI window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

COLUMNS = ["AGE", "SEX", "CP", "TREST", "CHOL", "FBS", "RESTECG", "THALACH",
           "EXANG", "OLDPEAK", "SLOPE", "CA", "THAL", "NUM"]
UCI_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "heart-disease/processed.cleveland.data")
OUT = "outputs"


def load_data(path):
    """Load processed.cleveland.data (no header, '?' = missing)."""
    if not os.path.exists(path):
        print(f"[info] {path} not found - trying to download from UCI ...")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(UCI_URL, path)
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    # SPSS step: Transform > Recode into Different Variables (0->0, 1-4->1)
    df["heart_disease"] = (df["NUM"] > 0).astype(int)
    return df


def descriptive_and_missing(df):
    print("\n=== Shape ===", df.shape)

    miss = pd.DataFrame({"Valid N": df.notna().sum(),
                         "Missing N": df.isna().sum()})
    miss["Missing %"] = (miss["Missing N"] / len(df) * 100).round(1)
    miss = miss[miss["Missing N"] > 0]
    print("\n=== Missing values ===\n", miss)
    miss.to_csv(f"{OUT}/missing_values.csv")

    num = ["AGE", "TREST", "CHOL", "THALACH", "OLDPEAK", "heart_disease"]
    desc = df[num].agg(["count", "min", "max", "mean", "std"]).T.round(4)
    print("\n=== Descriptive statistics ===\n", desc)
    desc.to_csv(f"{OUT}/descriptive_statistics.csv")

    freq = df["heart_disease"].value_counts().sort_index()
    print("\n=== Target frequency ===\n", freq)
    freq.to_csv(f"{OUT}/target_frequency.csv")


def eda_charts(df):
    # Fig 7 - target distribution
    ax = df["heart_disease"].value_counts().sort_index().plot(
        kind="bar", color=["#4C9F70", "#C8553D"], rot=0)
    ax.set_title("Target class distribution")
    ax.set_xlabel("heart_disease (0 = none, 1 = recorded)")
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2,
                    p.get_height()), ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig7_target_distribution.png", dpi=150); plt.close()

    # Fig 8 / 9 - histograms
    for col, fig_no, name in [("AGE", 8, "Age"), ("CHOL", 9, "Cholesterol")]:
        plt.figure(figsize=(6, 4))
        plt.hist(df[col].dropna(), bins=20, color="#3D6FA3", edgecolor="white")
        plt.title(f"{name} distribution"); plt.xlabel(col); plt.ylabel("Frequency")
        plt.tight_layout(); plt.savefig(f"{OUT}/fig{fig_no}_{col.lower()}_hist.png", dpi=150); plt.close()

    # Fig 10 / 11 - boxplots by class (plus THALACH, OLDPEAK)
    for col, fig_no in [("AGE", 10), ("CHOL", 11), ("THALACH", 12), ("OLDPEAK", 13)]:
        plt.figure(figsize=(5, 4))
        df.boxplot(column=col, by="heart_disease", grid=False)
        plt.suptitle(""); plt.title(f"{col} by heart_disease")
        plt.xlabel("heart_disease"); plt.ylabel(col)
        plt.tight_layout(); plt.savefig(f"{OUT}/fig{fig_no}_{col.lower()}_boxplot.png", dpi=150); plt.close()

    # Crosstabs of selected categorical predictors vs target
    with open(f"{OUT}/crosstabs.txt", "w") as f:
        for col in ["SEX", "CP", "FBS", "RESTECG", "EXANG", "SLOPE", "CA", "THAL"]:
            ct = pd.crosstab(df[col], df["heart_disease"])
            f.write(f"\n--- {col} x heart_disease ---\n{ct}\n")


def evaluate(name, model, X_te, y_te):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    return {
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred),
        "Sensitivity / Recall": recall_score(y_te, pred),
        "Specificity": tn / (tn + fp),
        "F1-score": f1_score(y_te, pred),
        "ROC-AUC": roc_auc_score(y_te, proba),
        "True negatives (TN)": tn, "False positives (FP)": fp,
        "False negatives (FN)": fn, "True positives (TP)": tp,
    }


def modelling(df, seed, test_size, tree_depth):
    X = df[COLUMNS[:-1]]            # 13 predictors (NUM excluded - it IS the target)
    y = df["heart_disease"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    print(f"\nTraining records: {len(X_tr)} | Test records: {len(X_te)}")

    # Missing CA / THAL handled by median imputation inside the pipeline
    models = {
        "Logistic Regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed))]),
        "Decision Tree": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(max_depth=tree_depth, random_state=seed))]),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        results[name] = evaluate(name, model, X_te, y_te)

    res = pd.DataFrame(results)
    res.loc["Training records"] = len(X_tr)
    res.loc["Test records"] = len(X_te)
    res = res.loc[["Training records", "Test records"] + list(res.index[:-2])]
    res.index.name = "Metric"

    print("\n=== Python model results ===")
    print(res.round(3).to_string())
    res.round(3).to_csv(f"{OUT}/test_metrics.csv")

    # Logistic Regression coefficients (in the spirit of SPSS 'Variables in the Equation')
    lr = models["Logistic Regression"].named_steps["clf"]
    coef = pd.DataFrame({"Predictor": X.columns, "Coefficient (scaled)": lr.coef_[0]})
    coef["Odds ratio (per 1 SD)"] = coef["Coefficient (scaled)"].apply(lambda b: pow(2.718281828, b))
    coef.round(3).to_csv(f"{OUT}/logistic_coefficients.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description="Heart disease PBL - Python workflow")
    ap.add_argument("--data", default="data/processed.cleveland.data")
    ap.add_argument("--seed", type=int, default=42, help="random_state for split/models")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--tree-depth", type=int, default=4)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    df = load_data(a.data)
    descriptive_and_missing(df)
    eda_charts(df)
    modelling(df, a.seed, a.test_size, a.tree_depth)
    print(f"\nDone. Files saved in ./{OUT}/")


if __name__ == "__main__":
    main()
