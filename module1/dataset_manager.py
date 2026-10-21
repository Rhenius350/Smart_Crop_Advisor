"""
Smart Crop Advisor – Using Genetic Algorithm and Machine Learning
Module 1: Dataset Management

This module is responsible for loading, inspecting, validating, and reporting
on the agricultural dataset (Crop_recommendation.csv). It performs read-only checks
and does NOT modify, clean, scale, or preprocess the dataset.
"""

import sys
import io
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

# Ensure UTF-8 output encoding for Windows standard output (handles unicode tick marks cleanly)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# Expected Constants
EXPECTED_FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph"
]
EXPECTED_TARGET = "label"
EXPECTED_COLUMNS = EXPECTED_FEATURES + [EXPECTED_TARGET]


def get_default_dataset_path() -> Path:
    """Returns the default dataset path relative to the project root."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "dataset" / "Crop_recommendation.csv"


def load_dataset(filepath: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Load the CSV dataset using pandas.
    
    Handles file non-existence and corrupted CSV files cleanly.
    Does not modify raw data.
    """
    if filepath is None:
        filepath = get_default_dataset_path()
    else:
        filepath = Path(filepath)

    print("=" * 60)
    print("SMART CROP ADVISOR")
    print("MODULE 1 — DATASET MANAGEMENT")
    print("=" * 60)
    print("\n[1] DATASET LOADING")

    if not filepath.exists():
        print("\nERROR:")
        print("Dataset not found.")
        print("\nExpected location:")
        print(f"{filepath}")
        print("\nPlease download the Kaggle dataset and place the CSV in the dataset folder.")
        return None

    try:
        df = pd.read_csv(filepath)
        if df.empty:
            print("\nERROR:")
            print("Dataset file is empty.")
            return None
        print(f"Dataset loaded successfully from: {filepath}")
        return df

    except pd.errors.EmptyDataError:
        print("\nERROR:")
        print("Dataset file is empty.")
        return None
    except pd.errors.ParserError as pe:
        print("\nERROR:")
        print(f"Failed to parse CSV file: {pe}")
        return None
    except Exception as e:
        print("\nERROR:")
        print(f"An unexpected error occurred while reading the CSV file: {e}")
        return None


def inspect_dataset(df: pd.DataFrame) -> None:
    """
    Display basic shape of the dataset.
    """
    print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")


def validate_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate presence of expected columns and identify missing or extraneous columns.
    Returns a dictionary summarizing column validation results.
    """
    df_columns = list(df.columns)
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df_columns]
    unexpected_cols = [col for col in df_columns if col not in EXPECTED_COLUMNS]

    is_valid = len(missing_cols) == 0
    overall_status = "PASS" if is_valid else "FAIL"

    print(f"Required Validation Status: {overall_status}")

    return {
        "is_valid": is_valid,
        "missing_columns": missing_cols,
        "unexpected_columns": unexpected_cols,
        "overall_status": overall_status
    }


def validate_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that feature columns are numeric and the target column is string/object.
    Does not convert data types automatically.
    """
    non_numeric_features = []
    for col in EXPECTED_FEATURES:
        if col in df.columns:
            is_num = pd.api.types.is_numeric_dtype(df[col])
            if not is_num:
                non_numeric_features.append(col)
        else:
            non_numeric_features.append(col)

    target_valid = False
    if EXPECTED_TARGET in df.columns:
        target_dtype = df[EXPECTED_TARGET].dtype
        is_obj_or_str = pd.api.types.is_object_dtype(df[EXPECTED_TARGET]) or pd.api.types.is_string_dtype(df[EXPECTED_TARGET]) or isinstance(target_dtype, pd.CategoricalDtype)
        target_valid = is_obj_or_str

    is_valid = (len(non_numeric_features) == 0) and target_valid

    return {
        "is_valid": is_valid,
        "non_numeric_features": non_numeric_features,
        "target_valid": target_valid
    }


def check_missing_values(df: pd.DataFrame) -> int:
    """
    Count missing/null values for every column.
    Does not alter missing values.
    """
    missing_counts = df.isnull().sum()
    total_missing = int(missing_counts.sum())
    return total_missing


def check_duplicates(df: pd.DataFrame) -> int:
    """
    Count duplicate rows.
    Does not remove duplicate rows.
    """
    duplicate_count = int(df.duplicated().sum())
    return duplicate_count


def analyze_target(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the target column ('label'): class count, class names, and class distribution.
    """
    if EXPECTED_TARGET not in df.columns:
        return {"unique_classes_count": 0, "classes": [], "distribution": {}}

    unique_classes = df[EXPECTED_TARGET].unique()
    num_classes = len(unique_classes)
    class_counts = df[EXPECTED_TARGET].value_counts()

    return {
        "unique_classes_count": num_classes,
        "classes": list(unique_classes),
        "distribution": class_counts.to_dict()
    }


def generate_summary(
    df: pd.DataFrame,
    column_res: Dict[str, Any],
    dtype_res: Dict[str, Any],
    total_missing: int,
    duplicate_count: int,
    target_res: Dict[str, Any]
) -> str:
    """
    Generate a dataset quality summary report and save it to results/module1_dataset_report.txt.
    """
    pass_columns = column_res["is_valid"]
    pass_dtypes = dtype_res["is_valid"]
    pass_missing = (total_missing == 0)
    pass_duplicates = (duplicate_count == 0)
    pass_target = target_res["unique_classes_count"] > 0

    overall_pass = pass_columns and pass_dtypes and pass_missing and pass_duplicates and pass_target
    status_text = "PASS" if overall_pass else "NEEDS ATTENTION"

    summary_lines = [
        "============================================================",
        "SMART CROP ADVISOR — MODULE 1 DATASET REPORT",
        "============================================================",
        f"Dataset File Path     : dataset/Crop_recommendation.csv",
        f"Dataset Loaded        : YES",
        f"Dataset Shape         : {df.shape[0]} Rows x {df.shape[1]} Columns",
        f"Expected Columns      : {'PASS' if pass_columns else 'FAIL (Missing or extra columns)'}",
        f"Numeric Features Check: {'PASS' if pass_dtypes else 'FAIL (Non-numeric features detected)'}",
        f"Target Column Check   : {'PASS (' + str(target_res['unique_classes_count']) + ' classes)' if pass_target else 'FAIL'}",
        f"Missing Values Count  : {total_missing} ({'PASS' if pass_missing else 'NEEDS ATTENTION'})",
        f"Duplicate Rows Count  : {duplicate_count} ({'PASS' if pass_duplicates else 'NEEDS ATTENTION'})",
        f"Number of Crop Classes: {target_res['unique_classes_count']}",
        "------------------------------------------------------------",
        f"OVERALL DATASET STATUS : {status_text}",
        "============================================================",
        "\nDetailed Crop Class Distribution:",
    ]

    for crop_name, count in target_res["distribution"].items():
        summary_lines.append(f"  - {crop_name:<18} : {count} samples")

    summary_content = "\n".join(summary_lines)

    # Save to results directory
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = results_dir / "module1_dataset_report.txt"

    try:
        report_file.write_text(summary_content, encoding="utf-8")
        print(f"Report successfully saved to: {report_file}")
    except Exception as e:
        print(f"WARNING: Could not write report file: {e}")

    print("=" * 60)
    print("MODULE 1 COMPLETED")
    print("=" * 60)

    return status_text


def main():
    """
    Execute Module 1 workflow in exact required order.
    """
    df = load_dataset()
    if df is None:
        return

    inspect_dataset(df)
    col_results = validate_columns(df)
    dtype_results = validate_data_types(df)
    missing_count = check_missing_values(df)
    duplicate_count = check_duplicates(df)
    target_results = analyze_target(df)

    generate_summary(
        df=df,
        column_res=col_results,
        dtype_res=dtype_results,
        total_missing=missing_count,
        duplicate_count=duplicate_count,
        target_res=target_results
    )


if __name__ == "__main__":
    main()
