"""
Smart Crop Advisor – Using Genetic Algorithm and Machine Learning
Module 6: Genetic Algorithm Hyperparameter Optimization Configuration
"""

from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input Data & Split Paths
RAW_DATASET_PATH = PROJECT_ROOT / "Crop_recommendation.csv"
PROCESSED_DATASET_PATH = PROJECT_ROOT / "dataset" / "processed" / "crop_data_clean.csv"
SPLITS_DIR = PROJECT_ROOT / "dataset" / "splits"
X_TRAIN_PATH = SPLITS_DIR / "X_train.csv"
X_VAL_PATH = SPLITS_DIR / "X_validation.csv"
Y_TRAIN_PATH = SPLITS_DIR / "y_train.csv"
Y_VAL_PATH = SPLITS_DIR / "y_validation.csv"

# Model Artifact & Metadata Paths
MODELS_DIR = PROJECT_ROOT / "models"
LABEL_ENCODER_PATH = MODELS_DIR / "crop_label_encoder.pkl"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_rf_model.pkl"
BASELINE_METADATA_PATH = MODELS_DIR / "baseline_model_metadata.json"

# Baseline Results Paths
MODULE5_RESULTS_DIR = PROJECT_ROOT / "results" / "module5"
MODULE5_METRICS_PATH = MODULE5_RESULTS_DIR / "baseline_metrics.csv"
MODULE5_SUMMARY_PATH = MODULE5_RESULTS_DIR / "baseline_summary.json"

# Module 6 Artifact Output Paths
RESULTS_DIR = PROJECT_ROOT / "results" / "module6"
BEST_PARAMS_PATH = RESULTS_DIR / "best_parameters.json"
GA_CONFIG_PATH = RESULTS_DIR / "ga_configuration.json"
GEN_METRICS_PATH = RESULTS_DIR / "generation_metrics.csv"
POP_HISTORY_PATH = RESULTS_DIR / "population_history.csv"
CACHE_SUMMARY_PATH = RESULTS_DIR / "fitness_cache_summary.json"
BASELINE_VS_GA_PATH = RESULTS_DIR / "baseline_vs_ga.csv"
CONVERGENCE_PLOT_PATH = RESULTS_DIR / "ga_convergence.png"
REPORT_PATH = RESULTS_DIR / "module6_ga_report.txt"

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

# Configurable GA Settings
POPULATION_SIZE = 20
GENERATIONS = 10
CROSSOVER_RATE = 0.80
MUTATION_RATE = 0.20
ELITE_COUNT = 1
TOURNAMENT_SIZE = 3
RANDOM_STATE = 42

# Search Space Ranges
SEARCH_SPACE = {
    "n_estimators": (50, 300),
    "max_depth": (5, 30),
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 10),
    "max_features": ["sqrt", "log2"]
}

# Fixed Random Forest Parameters
FIXED_PARAMS = {
    "criterion": "gini",
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1
}
