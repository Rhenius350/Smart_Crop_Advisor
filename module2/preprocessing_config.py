"""
Smart Crop Advisor – Using Genetic Algorithm and Machine Learning
Module 2: Data Preprocessing Configuration
"""

from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset & Report Paths
RAW_DATASET_PATH = PROJECT_ROOT / "dataset" / "Crop_recommendation.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "crop_data_clean.csv"
REPORT_DIR = PROJECT_ROOT / "results"
REPORT_PATH = REPORT_DIR / "module2_preprocessing_report.txt"

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

# Outlier Detection Configuration (IQR Multiplier)
IQR_MULTIPLIER = 1.5

# Invalid Value Validation Rules (Physical bounds)
# Features with non-negative constraints: N, P, K, humidity
# Soil pH must be strictly positive (> 0)
NON_NEGATIVE_FEATURES = ["N", "P", "K", "humidity"]
STRICTLY_POSITIVE_FEATURES = ["ph"]
