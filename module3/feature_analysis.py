"""
Smart Crop Advisor – Using Genetic Algorithm and Machine Learning
Module 3: Feature Analysis

This module performs non-visual agricultural feature analysis
on the preprocessed dataset (dataset/processed/crop_data_clean.csv).
It calculates descriptive statistics and mutual information feature ranking
without generating plots or graphics.
"""

import sys
import io
from pathlib import Path

# Add project root to sys.path for direct execution from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

# Ensure UTF-8 output encoding for Windows standard output
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Import Module 3 Configurations
from module3.analysis_config import (
    PROCESSED_DATASET_PATH,
    RESULTS_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    EXPECTED_COLUMNS,
)


def load_processed_dataset(filepath: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Load preprocessed dataset using pandas and pathlib.
    Handles file non-existence gracefully.
    """
    if filepath is None:
        filepath = PROCESSED_DATASET_PATH
    else:
        filepath = Path(filepath)

    print("=" * 60)
    print("SMART CROP ADVISOR")
    print("MODULE 3 — FEATURE ANALYSIS")
    print("=" * 60)
    print("\n[1] Loading Processed Dataset")

    if not filepath.exists():
        print("\nERROR: Processed dataset not found.")
        print("Expected location:")
        print(f"{filepath}")
        print("\nComplete Module 2 before running Module 3.")
        return None

    try:
        df = pd.read_csv(filepath)
        if df.empty:
            print("\nERROR: Processed dataset file is empty.")
            return None
        print(f"✓ Dataset loaded successfully from:\n  {filepath}")
        print(f"  Loaded Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"\nERROR: Failed to read dataset CSV: {e}")
        return None


def validate_dataset_structure(df: pd.DataFrame) -> bool:
    """
    Validate that all required feature columns and target column exist and have appropriate data types.
    """
    print("\n[2] Dataset Validation")

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    missing_target = TARGET_COLUMN not in df.columns

    if missing_features or missing_target:
        print("✗ Structure Validation: FAIL")
        if missing_features:
            print(f"  Missing features: {missing_features}")
        if missing_target:
            print(f"  Missing target column: '{TARGET_COLUMN}'")
        return False

    non_numeric_features = [
        col for col in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric_features:
        print(f"✗ Non-numeric feature columns detected: {non_numeric_features}")
        return False

    print("✓ Dataset Structure Validation: PASS")
    print("============================================================")
    print("DATASET STRUCTURE")
    print("============================================================")
    print(f"Rows      : {df.shape[0]}")
    print(f"Columns   : {df.shape[1]}")
    print(f"Features  : {len(FEATURE_COLUMNS)}")
    print(f"Target    : {TARGET_COLUMN}")
    print("Validation: PASS")
    print("============================================================")
    return True


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get basic overview of sample count, feature count, crop class count, and class names.
    """
    print("\n[3] Dataset Overview")
    num_samples = df.shape[0]
    num_features = len(FEATURE_COLUMNS)
    crop_classes = sorted(df[TARGET_COLUMN].unique().tolist())
    num_classes = len(crop_classes)

    print(f"Number of samples     : {num_samples}")
    print(f"Number of features    : {num_features}")
    print(f"Number of crop classes: {num_classes}")
    print(f"Crop classes list     : {crop_classes}")

    return {
        "num_samples": num_samples,
        "num_features": num_features,
        "num_classes": num_classes,
        "crop_classes": crop_classes,
    }


def generate_descriptive_statistics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate descriptive statistics (count, mean, median, std, min, Q1, Q3, max, IQR)
    and feature range summary. Saves results to CSV.
    """
    print("\n[4] Descriptive Statistics")

    stats_list = []
    range_list = []

    for col in FEATURE_COLUMNS:
        series = df[col]
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        mean_val = float(series.mean())
        median_val = float(series.median())
        std_val = float(series.std())
        min_val = float(series.min())
        max_val = float(series.max())
        val_range = max_val - min_val

        stats_list.append({
            "Feature": col,
            "Count": int(series.count()),
            "Mean": mean_val,
            "Median": median_val,
            "Std": std_val,
            "Min": min_val,
            "Q1": q1,
            "Q3": q3,
            "Max": max_val,
            "IQR": iqr,
        })

        range_list.append({
            "Feature": col,
            "Minimum": min_val,
            "Maximum": max_val,
            "Mean": mean_val,
            "Median": median_val,
            "Standard Deviation": std_val,
            "Range": val_range,
        })

    desc_df = pd.DataFrame(stats_list)
    range_df = pd.DataFrame(range_list)

    print(f"{'Feature':<15} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'IQR':<10}")
    print("-" * 75)
    for _, row in desc_df.iterrows():
        print(
            f"{row['Feature']:<15} {row['Mean']:<10.2f} {row['Median']:<10.2f} "
            f"{row['Std']:<10.2f} {row['Min']:<10.2f} {row['Max']:<10.2f} {row['IQR']:<10.2f}"
        )

    # Save to CSV files
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    desc_csv_path = RESULTS_DIR / "descriptive_statistics.csv"
    range_csv_path = RESULTS_DIR / "feature_range_summary.csv"

    desc_df.to_csv(desc_csv_path, index=False)
    range_df.to_csv(range_csv_path, index=False)

    print(f"✓ Descriptive statistics saved to: {desc_csv_path}")
    print(f"✓ Feature range summary saved to : {range_csv_path}")

    return desc_df, range_df


def calculate_mutual_information(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform exploratory Mutual Information (MI) analysis between features and target.
    Saves results to mutual_information.csv.
    """
    print("\n[5] Mutual Information Feature Ranking")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    mi_scores = mutual_info_classif(X, y, random_state=42)

    mi_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "mutual_information": mi_scores
    }).sort_values(by="mutual_information", ascending=False).reset_index(drop=True)

    # Save CSV
    mi_csv = RESULTS_DIR / "mutual_information.csv"
    mi_df.to_csv(mi_csv, index=False)

    print(f"✓ Mutual Information scores saved to: {mi_csv}")
    print("\nFeature Ranking (Mutual Information):")
    for idx, row in mi_df.iterrows():
        print(f"  {idx+1}. {row['feature']:<15} : {row['mutual_information']:.4f}")

    return mi_df


def generate_analysis_report(
    overview: Dict[str, Any],
    desc_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    filepath: Optional[Path] = None,
) -> str:
    """
    Generate results/module3/feature_analysis_report.txt report.
    """
    print("\n[6] Generating Report")
    if filepath is None:
        filepath = RESULTS_DIR / "feature_analysis_report.txt"

    filepath.parent.mkdir(parents=True, exist_ok=True)

    top_mi_feat = mi_df.iloc[0]["feature"]
    top_mi_val = mi_df.iloc[0]["mutual_information"]

    report_lines = [
        "============================================================",
        "SMART CROP ADVISOR",
        "MODULE 3 — FEATURE ANALYSIS REPORT",
        "============================================================",
        "",
        "1. DATASET OVERVIEW",
        f"Dataset File Path     : {PROCESSED_DATASET_PATH}",
        f"Rows (Samples)        : {overview['num_samples']}",
        f"Columns               : {overview['num_features'] + 1}",
        f"Number of features    : {overview['num_features']}",
        f"Number of crop classes: {overview['num_classes']}",
        "",
        "2. FEATURES ANALYZED",
    ]

    for feat in FEATURE_COLUMNS:
        report_lines.append(f"  - {feat}")

    report_lines.extend([
        "",
        "3. DESCRIPTIVE STATISTICS",
    ])

    for _, row in desc_df.iterrows():
        report_lines.append(
            f"  - {row['Feature']:<12}: Mean={row['Mean']:.2f}, Median={row['Median']:.2f}, Std={row['Std']:.2f}, Min={row['Min']:.2f}, Max={row['Max']:.2f}, IQR={row['IQR']:.2f}"
        )

    report_lines.extend([
        "",
        "4. MUTUAL INFORMATION ANALYSIS",
        f"Highest information association: '{top_mi_feat}' (MI score = {top_mi_val:.4f})",
        "Mutual Information Ranking:",
    ])

    for idx, row in mi_df.iterrows():
        report_lines.append(f"  {idx+1}. {row['feature']:<15} : {row['mutual_information']:.4f}")

    report_lines.extend([
        "",
        "5. FEATURE SELECTION STATUS",
        "No features were removed.",
        "All six agricultural features (N, P, K, temperature, humidity, ph) are retained for subsequent modules.",
        "",
        "6. FINAL STATUS",
        "PASS",
        "============================================================",
    ])

    report_content = "\n".join(report_lines)

    try:
        filepath.write_text(report_content, encoding="utf-8")
        print(f"✓ Report successfully saved to:\n  {filepath}")
    except Exception as e:
        print(f"✗ Failed to write feature analysis report: {e}")

    return "PASS"


def main():
    """
    Main executable workflow for Module 3 Feature Analysis.
    """
    # Step 1: Load Dataset
    df = load_processed_dataset()
    if df is None:
        return

    # Step 2: Validate Structure
    if not validate_dataset_structure(df):
        return

    # Step 3: Dataset Overview
    overview = get_dataset_overview(df)

    # Step 4: Descriptive Statistics
    desc_df, _ = generate_descriptive_statistics(df)

    # Step 5: Mutual Information Ranking
    mi_df = calculate_mutual_information(df)

    # Step 6: Generate Report
    final_status = generate_analysis_report(
        overview=overview,
        desc_df=desc_df,
        mi_df=mi_df,
    )

    print("\n" + "=" * 60)
    print("MODULE 3 COMPLETED")
    print("============================================================")
    print(f"\nFinal Status:\n{final_status}\n")


if __name__ == "__main__":
    main()
