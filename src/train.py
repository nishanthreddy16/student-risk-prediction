"""Repeatable training pipeline.

Run:  python -m src.train
Steps: load data -> data-quality/leakage checks -> chronological split ->
baselines + candidate models -> threshold chosen on VALIDATION only ->
single evaluation on held-out TEST term -> save artifact, metrics, figures.
"""
import json
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             fbeta_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import prepare_X
from .generate_data import generate

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ checks
def data_quality_report(df: pd.DataFrame) -> dict:
    X = df[C.FEATURES]
    return {
        "rows": int(len(df)),
        "positive_rate": float(df[C.TARGET].mean()),
        "missing_pct": {c: round(float(X[c].isna().mean() * 100), 2) for c in C.FEATURES},
        "lms_hours_over_30": int((df["lms_hours_per_week"] > 30).sum()),
        "duplicate_student_ids": int(df["student_id"].duplicated().sum()),
    }


def leakage_checks(df: pd.DataFrame) -> dict:
    model_inputs = set(C.ALL_FEATURES) | set(C.FEATURES)
    assert not model_inputs & set(C.LEAKY), "Target/future column used as feature!"
    assert not model_inputs & set(C.SENSITIVE), "Sensitive attribute used as feature!"
    X = prepare_X(df)
    corr = X.corrwith(df[C.TARGET].astype(float)).abs()
    assert corr.max() < 0.95, f"Suspiciously strong feature: {corr.idxmax()}"
    terms = {t: set(df.loc[df.term == t, "student_id"]) for t in C.TERMS}
    overlap = any(terms[a] & terms[b] for a in C.TERMS for b in C.TERMS if a < b)
    assert not overlap, "Student appears in more than one term"
    return {"max_abs_corr_with_target": round(float(corr.max()), 3),
            "strongest_feature": corr.idxmax(),
            "sensitive_attributes_used": [], "target_columns_used": [],
            "student_overlap_between_terms": False}


# ----------------------------------------------------------------- metrics
def metrics_at(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(thr), 3),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
        "f2": round(float(fbeta_score(y, pred, beta=2, zero_division=0)), 3),
        "roc_auc": round(float(roc_auc_score(y, proba)), 3) if len(set(proba)) > 1 else 0.5,
        "pr_auc": round(float(average_precision_score(y, proba)), 3),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_threshold(y, proba):
    """Missing a struggling student costs more than offering extra help -> F2."""
    grid = np.linspace(0.05, 0.9, 86)
    scores = [fbeta_score(y, (proba >= t).astype(int), beta=2, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def make_pipe(clf):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("clf", clf)])


def candidates():
    return {
        "logistic_regression": make_pipe(LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=C.SEED)),
        "random_forest": make_pipe(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, class_weight="balanced",
            random_state=C.SEED, n_jobs=-1)),
        "hist_gradient_boosting": make_pipe(HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced",
            random_state=C.SEED)),
    }


# -------------------------------------------------------------------- main
def main():
    C.REPORTS_DIR.mkdir(exist_ok=True)
    C.MODEL_PATH.parent.mkdir(exist_ok=True)
    df = pd.read_csv(C.DATA_PATH) if C.DATA_PATH.exists() else generate()

    dq = data_quality_report(df)
    leak = leakage_checks(df)

    tr, va, te = (df[df.term == t] for t in C.TERMS)   # chronological split
    Xtr, Xva, Xte = prepare_X(tr), prepare_X(va), prepare_X(te)
    ytr, yva, yte = tr[C.TARGET].values, va[C.TARGET].values, te[C.TARGET].values

    results = {}
    # --- baseline 1: always predict majority class
    dummy = DummyClassifier(strategy="prior").fit(Xtr.fillna(0), ytr)
    results["baseline_majority_class"] = {
        "validation": metrics_at(yva, dummy.predict_proba(Xva.fillna(0))[:, 1], 0.5),
        "test": metrics_at(yte, dummy.predict_proba(Xte.fillna(0))[:, 1], 0.5)}
    # --- baseline 2: simple rule on average of previous scores
    def rule_proba(X):
        return (1 - X["avg_prev_score"].fillna(X["avg_prev_score"].median()) / 100).values
    results["baseline_rule_avg_prev_score"] = {
        "validation": metrics_at(yva, rule_proba(Xva), best_threshold(yva, rule_proba(Xva))),
        "test": None}
    thr_rule = results["baseline_rule_avg_prev_score"]["validation"]["threshold"]
    results["baseline_rule_avg_prev_score"]["test"] = metrics_at(yte, rule_proba(Xte), thr_rule)

    # --- candidate models: fit on TRAIN, choose threshold on VALIDATION, score TEST once
    fitted, thresholds = {}, {}
    for name, pipe in candidates().items():
        pipe.fit(Xtr, ytr)
        pv = pipe.predict_proba(Xva)[:, 1]
        thr = best_threshold(yva, pv)
        fitted[name], thresholds[name] = pipe, thr
        results[name] = {"validation": metrics_at(yva, pv, thr),
                         "test": metrics_at(yte, pipe.predict_proba(Xte)[:, 1], thr)}

    best = max(fitted, key=lambda n: results[n]["validation"]["f2"])  # selected on validation
    pipe, thr = fitted[best], thresholds[best]
    pte = pipe.predict_proba(Xte)[:, 1]

    # --- error analysis & fairness audit (gender used for AUDIT only)
    pred_te = (pte >= thr).astype(int)
    audit = {}
    for g, idx in te.groupby("gender").groups.items():
        m = te.index.get_indexer(idx)
        audit[g] = {"n": int(len(m)),
                    "flag_rate": round(float(pred_te[m].mean()), 3),
                    "recall": round(float(recall_score(yte[m], pred_te[m], zero_division=0)), 3),
                    "precision": round(float(precision_score(yte[m], pred_te[m], zero_division=0)), 3)}
    fp = te[(pred_te == 1) & (yte == 0)]
    fn = te[(pred_te == 0) & (yte == 1)]
    err = {"false_positive_profile_mean": prepare_X(fp)[["avg_prev_score", "assignment_completion", "attendance_rate"]].mean().round(2).to_dict(),
           "false_negative_profile_mean": prepare_X(fn)[["avg_prev_score", "assignment_completion", "attendance_rate"]].mean().round(2).to_dict()}

    # --- global importance (permutation, on validation)
    pi = permutation_importance(pipe, Xva, yva, scoring="recall", n_repeats=15,
                                random_state=C.SEED)
    importance = dict(sorted(zip(C.ALL_FEATURES, map(float, pi.importances_mean)),
                             key=lambda kv: -kv[1]))

    artifact = {
        "model_name": best, "pipeline": pipe, "threshold": thr,
        "band_cutoffs": {"medium": round(thr * 0.5, 3), "high": round(thr, 3)},
        "features": C.ALL_FEATURES, "raw_features": C.FEATURES,
        "train_medians": Xtr.median().to_dict(),
        "global_importance": importance, "seed": C.SEED,
    }
    joblib.dump(artifact, C.MODEL_PATH)

    out = {"data_quality": dq, "leakage_checks": leak,
           "split": {"train_term": 1, "validation_term": 2, "test_term": 3,
                     "n_train": len(tr), "n_val": len(va), "n_test": len(te)},
           "selected_model": best, "results": results, "fairness_audit_test": audit,
           "error_analysis_test": err, "global_permutation_importance": importance}
    C.METRICS_PATH.write_text(json.dumps(out, indent=2))
    make_figures(df, results, best, results[best]["test"]["confusion_matrix"], importance)

    print(f"Selected model: {best} (threshold {thr:.2f})")
    for n, r in results.items():
        t = r["test"]
        print(f"{n:32s} test P={t['precision']:.2f} R={t['recall']:.2f} F1={t['f1']:.2f}")


def make_figures(df, results, best, cm, importance):
    fig, ax = plt.subplots(figsize=(5, 4))
    m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    ax.imshow(m, cmap="Blues")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, v, ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["No support", "Support"]); ax.set_yticks([0, 1], ["No support", "Support"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Test confusion matrix: {best}")
    fig.tight_layout(); fig.savefig(C.REPORTS_DIR / "confusion_matrix.png", dpi=130); plt.close(fig)

    names = list(results); x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, key in enumerate(["precision", "recall", "f1"]):
        ax.bar(x + (k - 1) * w, [results[n]["test"][key] for n in names], w, label=key)
    ax.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=7); ax.legend()
    ax.set_title("Held-out test term: baselines vs models"); fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "model_comparison.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    items = list(importance.items())[::-1]
    ax.barh([k for k, _ in items], [v for _, v in items])
    ax.set_title("Permutation importance (recall drop, validation)"); fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "feature_importance.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col in zip(axes, ["avg_prev_score" if "avg_prev_score" in df else "prev_score_3",
                              "assignment_completion", "attendance_rate"]):
        if col in df:
            for lab, g in df.groupby(C.TARGET):
                ax.hist(g[col].dropna(), bins=30, alpha=0.6, label=f"needs_support={lab}")
            ax.set_title(col); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(C.REPORTS_DIR / "eda_distributions.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
