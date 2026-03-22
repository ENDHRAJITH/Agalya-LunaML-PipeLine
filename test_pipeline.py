"""
Quick test script - run before starting server
python test_pipeline.py
"""
from app.pipeline import run_pipeline

tests = [
    "Classify emails as spam or not spam",
    "Predict house prices based on location",
    "Segment customers into groups based on behavior",
]

for problem in tests:
    print(f"\n{'='*55}")
    print(f"Testing: {problem}")
    print('='*55)
    result = run_pipeline(problem)
    print(f"Result: {result['task_type']} | {result['model_type']}")
    if 'accuracy' in result:
        print(f"Accuracy: {result['accuracy'] * 100:.1f}%")
    elif 'r2_score' in result:
        print(f"R²: {result['r2_score']}")
    elif 'silhouette_score' in result:
        print(f"Silhouette: {result['silhouette_score']}")
    print(f"Download: {result['download_url']}")

print("\n✅ All tests passed!")
