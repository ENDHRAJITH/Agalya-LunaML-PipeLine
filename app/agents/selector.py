"""
Model Selector Agent
Picks the best algorithm based on task type
"""

def select_model(task_type: str, description: str) -> dict:
    text = description.lower()

    if task_type == "classification":
        # Neural network hints
        if any(k in text for k in ["deep", "neural", "complex", "image", "text"]):
            return {
                "model_name": "GradientBoosting",
                "model_key": "gradient_boosting_classifier",
                "reason": "GradientBoosting for complex classification patterns"
            }
        # SVM hints
        if any(k in text for k in ["margin", "linear", "boundary"]):
            return {
                "model_name": "SVM",
                "model_key": "svm_classifier",
                "reason": "SVM for linear/margin-based classification"
            }
        # Default: RandomForest
        return {
            "model_name": "RandomForest",
            "model_key": "random_forest_classifier",
            "reason": "RandomForest — robust, handles mixed data well"
        }

    elif task_type == "regression":
        # Gradient boosting hints
        if any(k in text for k in ["complex", "nonlinear", "house", "price", "predict"]):
            return {
                "model_name": "GradientBoosting",
                "model_key": "gradient_boosting_regressor",
                "reason": "GradientBoosting for accurate price/value predictions"
            }
        # Linear hints
        if any(k in text for k in ["linear", "simple", "direct"]):
            return {
                "model_name": "LinearRegression",
                "model_key": "linear_regression",
                "reason": "LinearRegression for direct linear relationships"
            }
        # Default: RandomForest Regressor
        return {
            "model_name": "RandomForest",
            "model_key": "random_forest_regressor",
            "reason": "RandomForest Regressor — handles nonlinear patterns"
        }

    elif task_type == "clustering":
        # DBSCAN hints
        if any(k in text for k in ["density", "noise", "outlier", "anomaly"]):
            return {
                "model_name": "DBSCAN",
                "model_key": "dbscan",
                "reason": "DBSCAN for density-based clustering with noise"
            }
        # Default: KMeans
        return {
            "model_name": "KMeans",
            "model_key": "kmeans",
            "reason": "KMeans — efficient, interpretable clustering"
        }

    return {
        "model_name": "RandomForest",
        "model_key": "random_forest_classifier",
        "reason": "Default: RandomForest classifier"
    }
