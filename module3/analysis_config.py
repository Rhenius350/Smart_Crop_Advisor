"""
Smart Crop Advisor – Using Genetic Algorithm and Machine Learning
Module 3: Feature Analysis Configuration
"""

from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset & Output Directories
PROCESSED_DATASET_PATH = PROJECT_ROOT / "dataset" / "processed" / "crop_data_clean.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "module3"
PLOTS_DIR = RESULTS_DIR / "plots"
CROP_WISE_PLOTS_DIR = PLOTS_DIR / "crop_wise"

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
