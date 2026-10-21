"""
Smart Crop Advisor – Using Genetic Algorithm and Machine Learning
Module 5: Baseline Random Forest Configuration
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

# Model Artifact Output Paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "baseline_rf_model.pkl"
ENCODER_PATH = MODELS_DIR / "crop_label_encoder.pkl"
METADATA_PATH = MODELS_DIR / "baseline_model_metadata.json"

# Results Output Paths
RESULTS_DIR = PROJECT_ROOT / "results" / "module5"
PLOTS_DIR = RESULTS_DIR / "plots"

# Feature & Target Specifications
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

# Baseline Random Forest Hyperparameters (Fixed reference baseline)
BASELINE_PARAMS = {
    "n_estimators": 100,
    "criterion": "gini",
    "random_state": 42,
    "n_jobs": -1
}
