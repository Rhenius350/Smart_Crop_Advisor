"""
Smart Crop Advisor – Using Genetic Algorithm and Machine Learning
Module 4: Train / Validation Split Configuration
"""

from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input & Output Paths
PROCESSED_DATASET_PATH = PROJECT_ROOT / "dataset" / "processed" / "crop_data_clean.csv"
SPLITS_DIR = PROJECT_ROOT / "dataset" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results" / "module4"
PLOTS_DIR = RESULTS_DIR / "plots"

# Split Datasets File Paths
X_TRAIN_PATH = SPLITS_DIR / "X_train.csv"
X_VAL_PATH = SPLITS_DIR / "X_validation.csv"
Y_TRAIN_PATH = SPLITS_DIR / "y_train.csv"
Y_VAL_PATH = SPLITS_DIR / "y_validation.csv"
TRAIN_COMBINED_PATH = SPLITS_DIR / "train.csv"
VAL_COMBINED_PATH = SPLITS_DIR / "validation.csv"

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

# Split Configuration Parameters
VALIDATION_SIZE = 0.20
TRAIN_SIZE = 0.80
RANDOM_STATE = 42
STRATIFY = True
