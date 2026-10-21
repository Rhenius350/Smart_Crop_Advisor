"""
Smart Crop Advisor – Using Genetic Algorithm and Machine Learning
Module 7: GA-Optimized Random Forest Configuration
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

# Module 6 Artifact Paths
MODULE6_RESULTS_DIR = PROJECT_ROOT / "results" / "module6"
BEST_PARAMS_PATH = MODULE6_RESULTS_DIR / "best_parameters.json"
GA_CONFIG_PATH = MODULE6_RESULTS_DIR / "ga_configuration.json"
GA_COMPARISON_PATH = MODULE6_RESULTS_DIR / "baseline_vs_ga.csv"

# Model Artifact Output Paths
MODELS_DIR = PROJECT_ROOT / "models"
LABEL_ENCODER_PATH = MODELS_DIR / "crop_label_encoder.pkl"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_rf_model.pkl"
OPTIMIZED_MODEL_PATH = MODELS_DIR / "ga_optimized_rf_model.pkl"
OPTIMIZED_METADATA_PATH = MODELS_DIR / "ga_optimized_rf_metadata.json"

# Module 5 Baseline Results Path
MODULE5_METRICS_PATH = PROJECT_ROOT / "results" / "module5" / "baseline_metrics.csv"

# Module 7 Results Output Paths
RESULTS_DIR = PROJECT_ROOT / "results" / "module7"
METRICS_CSV_PATH = RESULTS_DIR / "ga_rf_metrics.csv"
PREDICTIONS_CSV_PATH = RESULTS_DIR / "ga_rf_predictions.csv"
CLASS_REPORT_CSV_PATH = RESULTS_DIR / "classification_report.csv"
FEATURE_IMP_CSV_PATH = RESULTS_DIR / "ga_rf_feature_importance.csv"
BASELINE_VS_GA_CSV_PATH = RESULTS_DIR / "baseline_vs_ga_rf.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "ga_rf_summary.json"
CONFUSION_MATRIX_PNG_PATH = RESULTS_DIR / "ga_rf_confusion_matrix.png"
FEATURE_IMP_PNG_PATH = RESULTS_DIR / "ga_rf_feature_importance.png"
COMPARISON_PNG_PATH = RESULTS_DIR / "baseline_vs_ga_rf.png"
REPORT_PATH = RESULTS_DIR / "module7_ga_rf_report.txt"

# Canonical Feature & Target Specifications
FEATURE_COLUMNS = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph"
]
TARGET_COLUMN = "label"
EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
