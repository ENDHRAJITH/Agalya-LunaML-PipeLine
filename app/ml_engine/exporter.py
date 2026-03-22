"""
Model Exporter
Saves model + generates inference code + zips everything
"""
import os, zipfile, joblib, json
from datetime import datetime


INFERENCE_TEMPLATE = """
# Luna AI Builder - Generated Inference Code
# Task: {task_type}
# Model: {model_name}
# Dataset: {dataset_name}
# Generated: {timestamp}

import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

# ── Predict function ──────────────────────────────────────────────
def predict(features: list):
    \"\"\"
    features: list of {n_features} numeric values
    Feature order: {feature_names}
    \"\"\"
    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)
    return prediction[0]

# ── Example usage ─────────────────────────────────────────────────
if __name__ == "__main__":
    # Replace with your actual feature values
    sample = {sample_input}
    result = predict(sample)
    print(f"Prediction: {{result}}")
"""

README_TEMPLATE = """
# Luna AI Builder - {model_name} Model
## Task: {task_type}
## Dataset: {dataset_name}
## Generated: {timestamp}

## Performance Metrics
{metrics_text}

## Files
- `model.pkl`     - Trained scikit-learn pipeline (scaler + model)
- `predict.py`    - Ready-to-use inference script
- `metadata.json` - Model info and metrics
- `README.md`     - This file

## Quick Start
```bash
pip install scikit-learn joblib numpy
python predict.py
```

## Load in your code
```python
import joblib
import numpy as np

model = joblib.load("model.pkl")
X = np.array([your_features]).reshape(1, -1)
prediction = model.predict(X)
print(prediction)
```

## Feature Names
{feature_names}
"""


def export_model(
    train_result: dict,
    eval_result:  dict,
    dataset:      dict,
    model_info:   dict,
    model_id:     str = "model"
) -> dict:
    os.makedirs("outputs", exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name   = model_info["model_name"].lower().replace(" ", "_")
    folder_name = f"luna_{safe_name}_{timestamp}"
    folder_path = os.path.join("outputs", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # ── 1. Save model ─────────────────────────────────────────────
    model_path = os.path.join(folder_path, "model.pkl")
    joblib.dump(train_result["model"], model_path)

    # ── 2. Metrics text ───────────────────────────────────────────
    task = eval_result["task_type"]

    if task == "classification":
        metrics_text = (
            f"- Accuracy: {eval_result['accuracy'] * 100:.2f}%\n"
            f"- Macro F1: {eval_result['macro_f1']:.4f}"
        )
    elif task == "regression":
        metrics_text = (
            f"- R² Score: {eval_result['r2_score']:.4f}\n"
            f"- RMSE: {eval_result['rmse']:.4f}\n"
            f"- MAE:  {eval_result['mae']:.4f}"
        )
    else:
        metrics_text = (
            f"- Clusters: {eval_result['n_clusters']}\n"
            f"- Silhouette Score: {eval_result['silhouette_score']:.4f}"
        )

    # ── 3. Inference code ─────────────────────────────────────────
    feature_names = dataset.get("feature_names", [])
    n_features    = dataset.get("n_features", 4)
    sample_input  = [0.0] * n_features

    inference_code = INFERENCE_TEMPLATE.format(
        task_type    = task,
        model_name   = model_info["model_name"],
        dataset_name = dataset["dataset_name"],
        timestamp    = timestamp,
        n_features   = n_features,
        feature_names= str(feature_names),
        sample_input = str(sample_input),
    )
    with open(os.path.join(folder_path, "predict.py"), "w", encoding="utf-8") as f:
        f.write(inference_code)

    # ── 4. README ─────────────────────────────────────────────────
    readme = README_TEMPLATE.format(
        model_name   = model_info["model_name"],
        task_type    = task,
        dataset_name = dataset["dataset_name"],
        timestamp    = timestamp,
        metrics_text = metrics_text,
        feature_names= "\n".join(f"  {i+1}. {fn}" for i, fn in enumerate(feature_names)),
    )
    with open(os.path.join(folder_path, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    # ── 5. metadata.json ──────────────────────────────────────────
    metadata = {
        "model_id":    model_id,
        "model_name":  model_info["model_name"],
        "task_type":   task,
        "dataset":     dataset["dataset_name"],
        "n_samples":   dataset["n_samples"],
        "n_features":  dataset["n_features"],
        "metrics":     eval_result,
        "timestamp":   timestamp,
    }
    with open(os.path.join(folder_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # ── 6. ZIP ────────────────────────────────────────────────────
    zip_name = f"{folder_name}.zip"
    zip_path = os.path.join("outputs", zip_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(folder_path):
            zf.write(os.path.join(folder_path, fname), fname)

    # Cleanup folder (keep only zip)
    import shutil
    shutil.rmtree(folder_path, ignore_errors=True)

    return {
        "zip_name":    zip_name,
        "download_url": f"/download/{zip_name}",
        "folder_name": folder_name,
    }
