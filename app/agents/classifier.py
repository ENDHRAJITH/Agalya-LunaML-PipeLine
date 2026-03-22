"""
Problem Classifier Agent
Detects: classification | regression | clustering
"""

def classify_problem(description: str) -> dict:
    text = description.lower()

    # ── Regression keywords ───────────────────────────────────────
    regression_keywords = [
        "predict", "price", "cost", "forecast", "estimate", "sales",
        "revenue", "score", "amount", "value", "rate", "temperature",
        "salary", "income", "continuous", "numeric", "how much", "how many"
    ]

    # ── Clustering keywords ───────────────────────────────────────
    clustering_keywords = [
        "group", "cluster", "segment", "similar", "categorize without",
        "unsupervised", "pattern", "community", "customer segment",
        "divide into groups", "find groups"
    ]

    # ── Classification keywords ───────────────────────────────────
    classification_keywords = [
        "classify", "detect", "spam", "fraud", "disease", "cancer",
        "sentiment", "positive", "negative", "yes or no", "binary",
        "category", "label", "class", "type", "identify", "recognize",
        "is it", "will it", "churn", "default", "approve", "reject"
    ]

    # Score each
    reg_score  = sum(1 for kw in regression_keywords     if kw in text)
    clu_score  = sum(1 for kw in clustering_keywords     if kw in text)
    cls_score  = sum(1 for kw in classification_keywords if kw in text)

    # Determine task
    if clu_score > reg_score and clu_score > cls_score:
        task_type = "clustering"
        confidence = round(clu_score / max(clu_score + reg_score + cls_score, 1) * 100)
    elif reg_score > cls_score:
        task_type = "regression"
        confidence = round(reg_score / max(clu_score + reg_score + cls_score, 1) * 100)
    else:
        task_type = "classification"
        confidence = round(cls_score / max(clu_score + reg_score + cls_score, 1) * 100)

    return {
        "task_type": task_type,
        "confidence": max(confidence, 55),  # minimum 55%
        "scores": {
            "classification": cls_score,
            "regression": reg_score,
            "clustering": clu_score
        }
    }
