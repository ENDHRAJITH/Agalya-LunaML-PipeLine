"""
Model Evaluator
Calculates metrics based on task type
"""
from sklearn.metrics import (
    accuracy_score, classification_report,
    r2_score, mean_squared_error, mean_absolute_error,
    silhouette_score
)
import numpy as np


def evaluate_model(train_result: dict) -> dict:
    task = train_result["task_type"]

    # ── Classification metrics ────────────────────────────────────
    if task == "classification":
        model  = train_result["model"]
        X_test = train_result["X_test"]
        y_test = train_result["y_test"]

        y_pred   = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report   = classification_report(y_test, y_pred, output_dict=True)

        # Per-class precision/recall
        classes = [k for k in report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
        per_class = {
            str(c): {
                "precision": round(report[c]["precision"], 3),
                "recall":    round(report[c]["recall"],    3),
                "f1_score":  round(report[c]["f1-score"],  3),
            }
            for c in classes
        }

        return {
            "accuracy":   round(float(accuracy), 4),
            "per_class":  per_class,
            "macro_f1":   round(report["macro avg"]["f1-score"], 4),
            "task_type":  "classification",
        }

    # ── Regression metrics ────────────────────────────────────────
    elif task == "regression":
        model  = train_result["model"]
        X_test = train_result["X_test"]
        y_test = train_result["y_test"]

        y_pred = model.predict(X_test)
        r2     = r2_score(y_test, y_pred)
        rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae    = float(mean_absolute_error(y_test, y_pred))

        return {
            "r2_score":  round(float(r2),   4),
            "rmse":      round(rmse, 4),
            "mae":       round(mae,  4),
            "task_type": "regression",
        }

    # ── Clustering metrics ────────────────────────────────────────
    elif task == "clustering":
        X      = train_result["X"]
        labels = train_result["labels"]

        unique_labels = set(labels)
        # Remove noise label (-1) for DBSCAN
        valid = unique_labels - {-1}
        n_clusters = len(valid)

        if n_clusters < 2:
            silhouette = 0.0
        else:
            mask = labels != -1
            silhouette = float(silhouette_score(X[mask], labels[mask])) if mask.sum() > 1 else 0.0

        cluster_sizes = {
            str(lbl): int(np.sum(labels == lbl))
            for lbl in sorted(valid)
        }

        return {
            "n_clusters":      n_clusters,
            "silhouette_score": round(silhouette, 4),
            "cluster_sizes":   cluster_sizes,
            "task_type":       "clustering",
        }

    raise ValueError(f"Unknown task type: {task}")
