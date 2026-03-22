"""
Dataset Loader
Loads appropriate built-in dataset based on task type
"""
from sklearn.datasets import (
    load_iris, load_breast_cancer, load_wine,
    load_diabetes, fetch_california_housing,
    make_blobs, make_classification
)
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_dataset(task_type: str, description: str) -> dict:
    text = description.lower()

    # ── Classification datasets ───────────────────────────────────
    if task_type == "classification":

        if any(k in text for k in ["cancer", "tumor", "disease", "medical", "health"]):
            data = load_breast_cancer()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            target_names  = list(data.target_names)
            dataset_name  = "Breast Cancer Wisconsin"

        elif any(k in text for k in ["wine", "alcohol", "beverage"]):
            data = load_wine()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            target_names  = [str(t) for t in data.target_names]
            dataset_name  = "Wine Quality"

        else:
            # Default: Iris (multiclass)
            data = load_iris()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            target_names  = list(data.target_names)
            dataset_name  = "Iris Flowers"

        return {
            "X": X, "y": y,
            "feature_names": feature_names,
            "target_names": target_names,
            "dataset_name": dataset_name,
            "task_type": "classification",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
        }

    # ── Regression datasets ───────────────────────────────────────
    elif task_type == "regression":

        if any(k in text for k in ["house", "home", "property", "real estate", "california"]):
            data = fetch_california_housing()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            dataset_name  = "California Housing"

        else:
            # Default: Diabetes
            data = load_diabetes()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            dataset_name  = "Diabetes Progression"

        return {
            "X": X, "y": y,
            "feature_names": feature_names,
            "target_names": None,
            "dataset_name": dataset_name,
            "task_type": "regression",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
        }

    # ── Clustering datasets ───────────────────────────────────────
    elif task_type == "clustering":
        n_clusters = 4
        if any(k in text for k in ["three", "3 group", "3 cluster"]):
            n_clusters = 3
        elif any(k in text for k in ["five", "5 group", "5 cluster"]):
            n_clusters = 5

        X, y_true = make_blobs(
            n_samples=500,
            n_features=4,
            centers=n_clusters,
            cluster_std=1.2,
            random_state=42
        )

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        return {
            "X": X, "y": y_true,
            "feature_names": feature_names,
            "target_names": [f"Cluster {i}" for i in range(n_clusters)],
            "dataset_name": f"Synthetic Clustering ({n_clusters} groups)",
            "task_type": "clustering",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_clusters": n_clusters,
        }

    raise ValueError(f"Unknown task type: {task_type}")
