"""
Model Trainer
Trains classification, regression, or clustering models
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline        import Pipeline

# Classification
from sklearn.ensemble   import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm        import SVC

# Regression
from sklearn.ensemble   import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Clustering
from sklearn.cluster    import KMeans, DBSCAN

import numpy as np


def train_model(dataset: dict, model_key: str) -> dict:
    X    = dataset["X"]
    y    = dataset.get("y")
    task = dataset["task_type"]

    # ── Classification ────────────────────────────────────────────
    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        MODEL_MAP = {
            "random_forest_classifier":     RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boosting_classifier": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "svm_classifier":               SVC(probability=True, random_state=42),
        }

        clf = MODEL_MAP.get(model_key, RandomForestClassifier(n_estimators=100, random_state=42))

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  clf)
        ])
        pipeline.fit(X_train, y_train)

        return {
            "model":   pipeline,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "task_type": "classification",
        }

    # ── Regression ────────────────────────────────────────────────
    elif task == "regression":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        MODEL_MAP = {
            "random_forest_regressor":     RandomForestRegressor(n_estimators=100, random_state=42),
            "gradient_boosting_regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "linear_regression":           LinearRegression(),
        }

        reg = MODEL_MAP.get(model_key, RandomForestRegressor(n_estimators=100, random_state=42))

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  reg)
        ])
        pipeline.fit(X_train, y_train)

        return {
            "model":   pipeline,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "task_type": "regression",
        }

    # ── Clustering ────────────────────────────────────────────────
    elif task == "clustering":
        n_clusters = dataset.get("n_clusters", 4)

        MODEL_MAP = {
            "kmeans": KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
            "dbscan": DBSCAN(eps=0.5, min_samples=5),
        }

        clusterer = MODEL_MAP.get(model_key, KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
        labels = clusterer.fit_predict(X)

        return {
            "model":     clusterer,
            "X":         X,
            "labels":    labels,
            "task_type": "clustering",
        }

    raise ValueError(f"Unknown task type: {task}")
