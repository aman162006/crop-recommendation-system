"""
Train, compare and save the crop recommendation model.

Steps (follows the project lifecycle in the presentation):
  1. Load data  2. Profile / validate  3. Stratified split
  4. Train candidate models  5. Cross-validate  6. Evaluate on test set
  7. Save best pipeline + metrics + plots

Run:  python train.py
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config as cfg

warnings.filterwarnings("ignore")


def load_and_profile() -> pd.DataFrame:
    df = pd.read_csv(cfg.DATA_PATH)
    missing = [c for c in cfg.FEATURES + [cfg.TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    print("=== DATA PROFILE ===")
    print(f"Shape            : {df.shape}")
    print(f"Missing values   : {int(df.isna().sum().sum())}")
    print(f"Duplicate rows   : {int(df.duplicated().sum())}")
    print(f"Number of crops  : {df[cfg.TARGET].nunique()}")
    print(f"Samples per crop : min={df[cfg.TARGET].value_counts().min()}, "
          f"max={df[cfg.TARGET].value_counts().max()}\n")
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    return df


def candidate_models() -> dict:
    """Same split / protocol for every algorithm (fair comparison)."""
    rs = cfg.RANDOM_STATE
    return {
        "Decision Tree": Pipeline([("clf", DecisionTreeClassifier(random_state=rs))]),
        "Random Forest": Pipeline([("clf", RandomForestClassifier(
            n_estimators=300, random_state=rs, n_jobs=-1))]),
        "SVM": Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(probability=True, random_state=rs))]),
        "KNN": Pipeline([("scaler", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=2000))]),
    }


def evaluate(pipe, X_test, y_test) -> dict:
    pred = pipe.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_precision": precision_score(y_test, pred, average="macro"),
        "macro_recall": recall_score(y_test, pred, average="macro"),
        "macro_f1": f1_score(y_test, pred, average="macro"),
    }


def main():
    cfg.REPORT_DIR.mkdir(exist_ok=True)
    cfg.MODEL_PATH.parent.mkdir(exist_ok=True)

    df = load_and_profile()
    X, y = df[cfg.FEATURES], df[cfg.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE)
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
    rows, fitted = [], {}
    print("=== MODEL COMPARISON ===")
    for name, pipe in candidate_models().items():
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
        pipe.fit(X_train, y_train)
        m = evaluate(pipe, X_test, y_test)
        m.update(model=name, cv_f1_mean=cv_scores.mean(), cv_f1_std=cv_scores.std())
        rows.append(m)
        fitted[name] = pipe
        print(f"{name:20s} acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}  "
              f"CV-F1={m['cv_f1_mean']:.4f}±{m['cv_f1_std']:.4f}")

    results = pd.DataFrame(rows).set_index("model").sort_values("macro_f1", ascending=False)
    results.to_csv(cfg.REPORT_DIR / "model_comparison.csv")

    
    best_name = "Random Forest"
    best = fitted[best_name]
    print(f"\nSelected model: {best_name}  (best by macro-F1 overall: {results.index[0]})")

    pred = best.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True)
    pd.DataFrame(report).T.to_csv(cfg.REPORT_DIR / "classification_report.csv")
    print("\n=== CLASSIFICATION REPORT (test set) ===")
    print(classification_report(y_test, pred))

    
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    plt.figure(figsize=(13, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout(); plt.savefig(cfg.REPORT_DIR / "confusion_matrix.png", dpi=150); plt.close()

    
    imp = pd.Series(best.named_steps["clf"].feature_importances_, index=cfg.FEATURES).sort_values()
    plt.figure(figsize=(7, 4))
    imp.plot(kind="barh", color="#2e7d32")
    plt.title("Random Forest Feature Importance"); plt.tight_layout()
    plt.savefig(cfg.REPORT_DIR / "feature_importance.png", dpi=150); plt.close()

    
    plt.figure(figsize=(8, 4))
    results["macro_f1"].sort_values().plot(kind="barh", color="#558b2f")
    plt.xlim(results["macro_f1"].min() - 0.02, 1.0)
    plt.title("Macro-F1 by Algorithm (test set)"); plt.tight_layout()
    plt.savefig(cfg.REPORT_DIR / "model_comparison.png", dpi=150); plt.close()

    
    artefact = {
        "pipeline": best,
        "features": cfg.FEATURES,
        "classes": list(best.classes_),
        "version": cfg.MODEL_VERSION,
        "model_name": best_name,
        "metrics": evaluate(best, X_test, y_test),
        "n_test": len(X_test),
    }
    joblib.dump(artefact, cfg.MODEL_PATH)
    with open(cfg.REPORT_DIR / "metrics.json", "w") as f:
        json.dump({k: artefact[k] for k in ("version", "model_name", "metrics", "n_test")}, f, indent=2)
    print(f"\nSaved model  -> {cfg.MODEL_PATH}")
    print(f"Saved report -> {cfg.REPORT_DIR}/")


if __name__ == "__main__":
    main()
