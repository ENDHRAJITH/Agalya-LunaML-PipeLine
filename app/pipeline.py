"""
Luna ML Pipeline Orchestrator
Runs: classify → select → load_data → train → evaluate → export
"""
from app.agents.classifier         import classify_problem
from app.agents.selector           import select_model
from app.ml_engine.datasets        import load_dataset
from app.ml_engine.trainer         import train_model
from app.ml_engine.evaluator       import evaluate_model
from app.ml_engine.exporter        import export_model


def run_pipeline(problem_description: str, model_id: str = None) -> dict:
    print(f"\n🌙 Luna Pipeline starting...")
    print(f"   Problem: {problem_description}")

    # ── Step 1: Classify problem ──────────────────────────────────
    print("   [1/5] Classifying problem...")
    classification = classify_problem(problem_description)
    task_type  = classification["task_type"]
    confidence = classification["confidence"]
    print(f"   → Task: {task_type} (confidence: {confidence}%)")

    # ── Step 2: Select algorithm ──────────────────────────────────
    print("   [2/5] Selecting algorithm...")
    model_info = select_model(task_type, problem_description)
    print(f"   → Model: {model_info['model_name']}")

    # ── Step 3: Load dataset ──────────────────────────────────────
    print("   [3/5] Loading dataset...")
    dataset = load_dataset(task_type, problem_description)
    print(f"   → Dataset: {dataset['dataset_name']} ({dataset['n_samples']} samples, {dataset['n_features']} features)")

    # ── Step 4: Train model ───────────────────────────────────────
    print("   [4/5] Training model...")
    train_result = train_model(dataset, model_info["model_key"])
    print(f"   → Training complete")

    # ── Step 5: Evaluate ─────────────────────────────────────────
    print("   [5/5] Evaluating...")
    eval_result = evaluate_model(train_result)

    if task_type == "classification":
        print(f"   → Accuracy: {eval_result['accuracy'] * 100:.2f}%")
    elif task_type == "regression":
        print(f"   → R²: {eval_result['r2_score']:.4f}  RMSE: {eval_result['rmse']:.4f}")
    else:
        print(f"   → Clusters: {eval_result['n_clusters']}  Silhouette: {eval_result['silhouette_score']:.4f}")

    # ── Step 6: Export ────────────────────────────────────────────
    print("   [6/6] Exporting ZIP...")
    export_info = export_model(train_result, eval_result, dataset, model_info, model_id or "model")
    print(f"   → ZIP: {export_info['zip_name']}")
    print("   ✅ Pipeline complete!\n")

    # ── Build explanation ─────────────────────────────────────────
    if task_type == "classification":
        explanation = (
            f"Luna detected a {task_type} problem and selected {model_info['model_name']}. "
            f"Trained on {dataset['dataset_name']} with {dataset['n_samples']} samples. "
            f"Achieved {eval_result['accuracy'] * 100:.1f}% accuracy on the test set."
        )
    elif task_type == "regression":
        explanation = (
            f"Luna detected a {task_type} problem and selected {model_info['model_name']}. "
            f"Trained on {dataset['dataset_name']} with {dataset['n_samples']} samples. "
            f"Achieved R² score of {eval_result['r2_score']:.3f} (RMSE: {eval_result['rmse']:.3f})."
        )
    else:
        explanation = (
            f"Luna detected a clustering problem and selected {model_info['model_name']}. "
            f"Found {eval_result['n_clusters']} clusters in {dataset['n_samples']} samples. "
            f"Silhouette score: {eval_result['silhouette_score']:.3f}."
        )

    # ── Final response ────────────────────────────────────────────
    result = {
        "task_type":   task_type,
        "model_type":  model_info["model_name"],
        "model_reason": model_info["reason"],
        "dataset":     dataset["dataset_name"],
        "explanation": explanation,
        "download_url": export_info["download_url"],
        **eval_result,  # accuracy / r2_score / rmse / silhouette_score etc.
    }

    return result
