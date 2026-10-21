"""
Smart Crop Advisor – Using Genetic Algorithm and Machine Learning
Module 8: Performance Evaluation and Model Comparison Configuration
"""

from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input Split Paths
SPLITS_DIR = PROJECT_ROOT / "dataset" / "splits"
X_TRAIN_PATH = SPLITS_DIR / "X_train.csv"
X_VAL_PATH = SPLITS_DIR / "X_validation.csv"
Y_TRAIN_PATH = SPLITS_DIR / "y_train.csv"
Y_VAL_PATH = SPLITS_DIR / "y_validation.csv"

# Model Paths
MODELS_DIR = PROJECT_ROOT / "models"
LABEL_ENCODER_PATH = MODELS_DIR / "crop_label_encoder.pkl"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_rf_model.pkl"
GA_MODEL_PATH = MODELS_DIR / "ga_optimized_rf_model.pkl"

# Output Results Paths
RESULTS_DIR = PROJECT_ROOT / "results" / "module8"
REPORT_PATH = RESULTS_DIR / "module8_evaluation_report.txt"
FINAL_EVAL_TABLE_PATH = RESULTS_DIR / "final_evaluation_table.csv"
MODEL_COMPARSION_PATH = RESULTS_DIR / "model_comparison.csv"
PRED_ACC_SUMMARY_PATH = RESULTS_DIR / "prediction_accuracy_summary.csv"
GENERALIZATION_GAP_PATH = RESULTS_DIR / "generalization_gap.csv"
FEATURE_IMP_COMP_CSV_PATH = RESULTS_DIR / "feature_importance_comparison.csv"
MODEL_COMPLEXITY_CSV_PATH = RESULTS_DIR / "model_complexity_comparison.csv"
BASELINE_CLASS_REPORT_PATH = RESULTS_DIR / "baseline_classification_report.csv"
GA_CLASS_REPORT_PATH = RESULTS_DIR / "ga_classification_report.csv"
BASELINE_CM_CSV_PATH = RESULTS_DIR / "baseline_confusion_matrix.csv"
GA_CM_CSV_PATH = RESULTS_DIR / "ga_confusion_matrix.csv"
BASELINE_ERROR_PAIRS_PATH = RESULTS_DIR / "baseline_error_pairs.csv"
GA_ERROR_PAIRS_PATH = RESULTS_DIR / "ga_error_pairs.csv"
BASELINE_ERRORS_PATH = RESULTS_DIR / "baseline_errors.csv"
GA_ERRORS_PATH = RESULTS_DIR / "ga_errors.csv"
PRED_ERROR_SUMMARY_PATH = RESULTS_DIR / "prediction_error_summary.csv"
CONFIDENCE_ANALYSIS_PATH = RESULTS_DIR / "confidence_analysis.csv"

# Plot Artifact Paths
PERF_COMP_PNG_PATH = RESULTS_DIR / "performance_comparison.png"
PER_CLASS_F1_PNG_PATH = RESULTS_DIR / "per_class_f1_comparison.png"
BASELINE_CM_PNG_PATH = RESULTS_DIR / "baseline_confusion_matrix.png"
GA_CM_PNG_PATH = RESULTS_DIR / "ga_confusion_matrix.png"
FEATURE_IMP_COMP_PNG_PATH = RESULTS_DIR / "feature_importance_comparison.png"
CONFIDENCE_DIST_PNG_PATH = RESULTS_DIR / "confidence_distribution.png"

# Canonical Features
FEATURE_COLUMNS = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph"
]
TARGET_COLUMN = "label"
