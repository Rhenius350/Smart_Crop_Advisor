"""
Smart Crop Advisor – Using Genetic Algorithm and Machine Learning
Master Pipeline Execution Script

Executes Modules 1 through 9 sequentially from start to finish.
"""

import sys
import io
import subprocess
from pathlib import Path

# Ensure UTF-8 output encoding for Windows standard output
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent

MODULES = [
    ("Module 1 — Dataset Management", PROJECT_ROOT / "module1" / "dataset_manager.py"),
    ("Module 2 — Data Preprocessing", PROJECT_ROOT / "module2" / "preprocessing.py"),
    ("Module 3 — Feature Analysis", PROJECT_ROOT / "module3" / "feature_analysis.py"),
    ("Module 4 — Train / Validation Split", PROJECT_ROOT / "module4" / "train_validation_split.py"),
    ("Module 5 — Baseline Random Forest", PROJECT_ROOT / "module5" / "baseline_random_forest.py"),
    ("Module 6 — Genetic Algorithm Optimization", PROJECT_ROOT / "module6" / "genetic_algorithm.py"),
    ("Module 7 — GA-Optimized Random Forest", PROJECT_ROOT / "module7" / "optimized_random_forest.py"),
    ("Module 8 — Performance Evaluation & Comparison", PROJECT_ROOT / "module8" / "performance_evaluation.py"),
    ("Module 9 — Explainable AI (LIME + SHAP)", PROJECT_ROOT / "module9" / "explainability.py"),
]


def main():
    print("=" * 60)
    print("SMART CROP ADVISOR")
    print("FULL PIPELINE EXECUTION (MODULES 1 - 9)")
    print("=" * 60)

    for name, script_path in MODULES:
        print(f"\n>>> Running: {name}")
        print("-" * 60)
        res = subprocess.run([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT))
        if res.returncode != 0:
            print(f"\n[ERROR] {name} failed with exit code {res.returncode}.")
            sys.exit(res.returncode)
        print(f"[OK] Completed: {name}")

    print("\n" + "=" * 60)
    print("SMART CROP ADVISOR PIPELINE COMPLETED")
    print("ALL MODULES 1 THROUGH 9 EXECUTED WITH STATUS: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
