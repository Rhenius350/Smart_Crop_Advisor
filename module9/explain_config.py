"""
Smart Crop Advisor – Using Genetic Algorithm and Machine Learning
Module 9: Explainable AI Using LIME + SHAP Configuration
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
GA_MODEL_PATH = MODELS_DIR / "ga_optimized_rf_model.pkl"

# Module 7 Artifact Paths
MODULE7_RESULTS_DIR = PROJECT_ROOT / "results" / "module7"
M7_PREDICTIONS_PATH = MODULE7_RESULTS_DIR / "ga_rf_predictions.csv"
M7_SUMMARY_PATH = MODULE7_RESULTS_DIR / "ga_rf_summary.json"
M7_METRICS_PATH = MODULE7_RESULTS_DIR / "ga_rf_metrics.csv"

# Output Results Paths
RESULTS_DIR = PROJECT_ROOT / "results" / "module9"
REPORT_PATH = RESULTS_DIR / "module9_explainability_report.txt"

# CSV Files
SHAP_GLOBAL_IMP_CSV = RESULTS_DIR / "shap_global_feature_importance.csv"
SHAP_LOCAL_CSV = RESULTS_DIR / "shap_local_explanations.csv"
LIME_LOCAL_CSV = RESULTS_DIR / "lime_local_explanations.csv"
SHAP_VS_LIME_CSV = RESULTS_DIR / "shap_vs_lime_comparison.csv"
EXPLANATION_CONSISTENCY_CSV = RESULTS_DIR / "explanation_consistency.csv"
RF_VS_SHAP_CSV = RESULTS_DIR / "rf_vs_shap_feature_importance.csv"

# JSON Files
EXPLANATION_DATASET_SUMMARY_JSON = RESULTS_DIR / "explanation_dataset_summary.json"
SHAP_SUMMARY_JSON = RESULTS_DIR / "shap_summary.json"
LIME_SUMMARY_JSON = RESULTS_DIR / "lime_summary.json"
HIGH_CONF_JSON = RESULTS_DIR / "high_confidence_explanation.json"
LOW_CONF_JSON = RESULTS_DIR / "low_confidence_explanation.json"
INCORRECT_PRED_JSON = RESULTS_DIR / "incorrect_prediction_explanation.json"

# Plot Artifact Paths
SHAP_GLOBAL_IMP_PNG = RESULTS_DIR / "shap_global_importance.png"
SHAP_SUMMARY_PNG = RESULTS_DIR / "shap_summary.png"
RF_VS_SHAP_PNG = RESULTS_DIR / "rf_vs_shap_importance.png"

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

# Configurable Settings
RANDOM_STATE = 42
LOCAL_EXPLANATION_COUNT = 5
SHAP_SAMPLE_SIZE = 440
TOP_FEATURES = 10
