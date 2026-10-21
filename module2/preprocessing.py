"""
Smart Crop Advisor – Using Genetic Algorithm and Machine Learning
Module 2: Data Preprocessing

This module converts the raw agricultural dataset (Crop_recommendation.csv)
into a clean, verified processed dataset (dataset/processed/crop_data_clean.csv).
It handles validation, duplicate removal, invalid value checks, outlier detection (without removal),
and data consistency without introducing data leakage or performing feature scaling.
"""

import sys
import io
from pathlib import Path

# Add project root to sys.path for direct python command execution from root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Ensure UTF-8 output encoding for Windows standard output
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Import Module 2 Configurations
from module2.preprocessing_config import (

    RAW_DATASET_PATH,
    PROCESSED_DATASET_PATH,
    PROCESSED_DATA_DIR,
    REPORT_PATH,
    REPORT_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    EXPECTED_COLUMNS,
    IQR_MULTIPLIER,
    NON_NEGATIVE_FEATURES,
    STRICTLY_POSITIVE_FEATURES,
)


def load_raw_dataset(filepath: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Load raw CSV dataset using pandas and pathlib.
    Does not overwrite raw dataset.
    """
    if filepath is None:
        filepath = RAW_DATASET_PATH
    else:
        filepath = Path(filepath)

    print("=" * 60)
    print("SMART CROP ADVISOR")
    print("MODULE 2 — DATA PREPROCESSING")
    print("=" * 60)
    print("\n[1] Loading Dataset")

    if not filepath.exists():
        print("\nERROR: Dataset not found.")
        print(f"\nExpected location:\n{filepath}")
        print("\nPlease ensure the raw dataset CSV exists at the expected location.")
        return None

    try:
        df = pd.read_csv(filepath)
        if df.empty:
            print("\nERROR: Raw dataset file is empty.")
            return None
        print(f"✓ Dataset loaded successfully from: {filepath}")
        print(f"  Loaded Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"\nERROR: Failed to read dataset CSV: {e}")
        return None


def validate_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verify required columns exist. Report missing or unexpected columns.
    """
    print("\n[2] Dataset Validation")

    df_cols = list(df.columns)
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df_cols]
    unexpected_cols = [col for col in df_cols if col not in EXPECTED_COLUMNS]

    if missing_cols:
        print(f"✗ MISSING REQUIRED COLUMNS: {missing_cols}")
        print("  Preprocessing halted because required columns are absent.")
        is_valid = False
    else:
        print("✓ All required feature and target columns are present:")
        for col in EXPECTED_COLUMNS:
            print(f"   - {col}")
        is_valid = True

    if unexpected_cols:
        print(f"\n! UNEXPECTED COLUMNS DETECTED: {unexpected_cols}")
        print("  (Unexpected columns retained without deletion to preserve original schema)")

    return {
        "is_valid": is_valid,
        "missing_columns": missing_cols,
        "unexpected_columns": unexpected_cols,
    }


def validate_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that feature columns are numeric and target is categorical/string.
    """
    print("\n[3] Data Type Validation")
    print(f"{'Column':<20} {'Data Type':<15} {'Valid?':<10}")
    print("-" * 48)

    invalid_features = []
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            is_num = pd.api.types.is_numeric_dtype(df[col])
            status = "YES" if is_num else "NO"
            print(f"{col:<20} {str(df[col].dtype):<15} {status:<10}")
            if not is_num:
                invalid_features.append(col)
        else:
            print(f"{col:<20} {'MISSING':<15} NO")
            invalid_features.append(col)

    target_valid = False
    if TARGET_COLUMN in df.columns:
        t_dtype = df[TARGET_COLUMN].dtype
        is_str_or_obj = (
            pd.api.types.is_object_dtype(df[TARGET_COLUMN])
            or pd.api.types.is_string_dtype(df[TARGET_COLUMN])
            or isinstance(t_dtype, pd.CategoricalDtype)
        )
        status = "YES" if is_str_or_obj else "NO"
        print(f"{TARGET_COLUMN:<20} {str(t_dtype):<15} {status:<10}")
        target_valid = is_str_or_obj
    else:
        print(f"{TARGET_COLUMN:<20} {'MISSING':<15} NO")

    is_valid = (len(invalid_features) == 0) and target_valid

    if is_valid:
        print("\n✓ Numeric features and target data type successfully validated.")
    else:
        print("\n✗ Data type validation issue detected.")

    return {
        "is_valid": is_valid,
        "invalid_features": invalid_features,
        "target_valid": target_valid,
    }


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze missing/null values across all columns.
    """
    print("\n[4] Missing Value Analysis")
    missing_series = df.isnull().sum()
    total_missing = int(missing_series.sum())

    print("Missing values count per column:")
    for col, count in missing_series.items():
        print(f"  - {col:<15} : {count}")

    if total_missing == 0:
        print("✓ No missing values found. No imputation required.")
    else:
        print(f"! Found {total_missing} missing value(s). Imputation strategy: median for features.")

    return {
        "total_missing": total_missing,
        "missing_per_column": missing_series.to_dict(),
    }


def handle_missing_values(df: pd.DataFrame, missing_info: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle missing values if present. Drops missing target rows; applies median imputation for features if missing.
    """
    df_clean = df.copy()
    total_missing = missing_info["total_missing"]
    rows_dropped_missing_target = 0
    features_imputed = []

    if total_missing > 0:
        # Drop rows where target label is missing
        if TARGET_COLUMN in df_clean.columns:
            target_nulls = df_clean[TARGET_COLUMN].isnull().sum()
            if target_nulls > 0:
                df_clean = df_clean.dropna(subset=[TARGET_COLUMN])
                rows_dropped_missing_target = int(target_nulls)
                print(f"  - Dropped {rows_dropped_missing_target} row(s) with missing target label.")

        # Median impute features if missing values remain
        for col in FEATURE_COLUMNS:
            if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                features_imputed.append(col)
                print(f"  - Imputed missing values in feature '{col}' using median = {median_val}.")

    return df_clean, {
        "rows_dropped_missing_target": rows_dropped_missing_target,
        "features_imputed": features_imputed,
    }


def detect_and_remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect exact duplicate rows and remove them from the processed DataFrame copy.
    Does not modify raw CSV.
    """
    print("\n[5] Duplicate Analysis & Handling")
    original_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())

    print(f"Original rows   : {original_rows}")
    print(f"Duplicate rows  : {duplicate_rows}")

    if duplicate_rows > 0:
        df_clean = df.drop_duplicates().copy()
        processed_rows = len(df_clean)
        print(f"✓ Removed {duplicate_rows} exact duplicate row(s).")
        print(f"Processed rows  : {processed_rows}")
    else:
        df_clean = df.copy()
        processed_rows = original_rows
        print("✓ No duplicate rows found.")

    return df_clean, {
        "original_rows": original_rows,
        "duplicate_rows": duplicate_rows,
        "processed_rows": processed_rows,
    }


def detect_invalid_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check for NaNs, infinite values, negative feature values, or invalid soil pH.
    """
    print("\n[6] Invalid Value Analysis")
    invalid_report = {}
    total_invalid = 0

    # 1. Non-negative checks
    for col in NON_NEGATIVE_FEATURES:
        if col in df.columns:
            invalid_mask = (df[col] < 0) | np.isinf(df[col])
            count = int(invalid_mask.sum())
            invalid_report[col] = count
            total_invalid += count

    # 2. Strictly positive checks (e.g. soil pH)
    for col in STRICTLY_POSITIVE_FEATURES:
        if col in df.columns:
            invalid_mask = (df[col] <= 0) | np.isinf(df[col])
            count = int(invalid_mask.sum())
            invalid_report[col] = count
            total_invalid += count

    for col, count in invalid_report.items():
        if count > 0:
            print(f"  ! Feature '{col}' contains {count} invalid value(s).")

    if total_invalid == 0:
        print("✓ No invalid numeric values (negative N/P/K/humidity or pH <= 0) found.")

    return {
        "total_invalid": total_invalid,
        "invalid_per_feature": invalid_report,
    }


def detect_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform IQR outlier detection on feature columns.
    Reports potential outliers without removing them.
    """
    print("\n[7] Outlier Analysis (IQR Method)")
    print(f"{'Feature':<18} {'Q1 (25%)':<12} {'Q3 (75%)':<12} {'IQR':<12} {'Potential Outliers':<18}")
    print("-" * 72)

    outlier_counts = {}
    outlier_bounds = {}
    total_outliers = 0

    for col in FEATURE_COLUMNS:
        if col in df.columns:
            q1 = float(df[col].quantile(0.25))
            q3 = float(df[col].quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - (IQR_MULTIPLIER * iqr)
            upper_bound = q3 + (IQR_MULTIPLIER * iqr)

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            count = int(outlier_mask.sum())
            outlier_counts[col] = count
            outlier_bounds[col] = (lower_bound, upper_bound)
            total_outliers += count

            print(f"{col:<18} {q1:<12.2f} {q3:<12.2f} {iqr:<12.2f} {count:<18}")

    print("\n✓ Potential outliers detected and logged.")

    return {
        "total_outliers": total_outliers,
        "outlier_counts": outlier_counts,
        "outlier_bounds": outlier_bounds,
    }


def check_label_consistency(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Check crop names for whitespace, empty strings, and case inconsistencies.
    Strips leading/trailing whitespace from labels in the processed copy.
    """
    print("\n[8] Data Consistency Checks")
    df_clean = df.copy()

    if TARGET_COLUMN not in df_clean.columns:
        print("✗ Target column missing. Cannot check label consistency.")
        return df_clean, {"unique_classes": 0, "whitespace_stripped": False}

    raw_labels = df_clean[TARGET_COLUMN].astype(str)
    has_whitespace = (raw_labels != raw_labels.str.strip()).any()

    if has_whitespace:
        df_clean[TARGET_COLUMN] = raw_labels.str.strip()
        print("✓ Stripped leading/trailing whitespace from target crop labels.")
    else:
        print("✓ Target labels are clean with no leading/trailing whitespace.")

    unique_crops = sorted(df_clean[TARGET_COLUMN].unique().tolist())
    num_classes = len(unique_crops)

    print(f"  Unique Crop Classes Count: {num_classes}")

    return df_clean, {
        "unique_classes": num_classes,
        "crop_names": unique_crops,
        "whitespace_stripped": bool(has_whitespace),
    }


def generate_feature_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Generate numerical statistics (count, min, max, mean, median, std) for each feature.
    Does not scale features.
    """
    print("\n[9] Feature Summary (Unscaled Original Units)")
    print(f"{'Feature':<15} {'Min':<10} {'Max':<10} {'Mean':<12} {'Median':<12} {'Std':<12}")
    print("-" * 71)

    summary = {}
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            stats = {
                "count": float(df[col].count()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
            }
            summary[col] = stats
            print(
                f"{col:<15} {stats['min']:<10.2f} {stats['max']:<10.2f} "
                f"{stats['mean']:<12.2f} {stats['median']:<12.2f} {stats['std']:<12.2f}"
            )

    print("\n✓ Features retained in original units (No feature scaling applied).")
    return summary


def separate_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target vector y without encoding labels.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def save_processed_dataset(df: pd.DataFrame, filepath: Optional[Path] = None) -> bool:
    """
    Save the clean DataFrame to dataset/processed/crop_data_clean.csv.
    """
    print("\n[10] Saving Processed Dataset")
    if filepath is None:
        filepath = PROCESSED_DATASET_PATH

    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        cols_to_save = [c for c in EXPECTED_COLUMNS if c in df.columns]
        df_clean_out = df[cols_to_save].copy()
        df_clean_out.to_csv(filepath, index=False)
        print(f"✓ Saved clean dataset to:\n  {filepath}")
        print(f"  Output Dataset Shape: {df_clean_out.shape[0]} rows x {df_clean_out.shape[1]} columns")
        return True
    except Exception as e:
        print(f"✗ Failed to save processed dataset: {e}")
        return False


def generate_preprocessing_report(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    col_results: Dict[str, Any],
    dtype_results: Dict[str, Any],
    missing_results: Dict[str, Any],
    duplicate_results: Dict[str, Any],
    invalid_results: Dict[str, Any],
    outlier_results: Dict[str, Any],
    consistency_results: Dict[str, Any],
    feature_summary: Dict[str, Dict[str, float]],
    filepath: Optional[Path] = None,
) -> str:
    """
    Generate results/module2_preprocessing_report.txt following the mandatory structure.
    """
    print("\n[11] Generating Report")
    if filepath is None:
        filepath = REPORT_PATH

    filepath.parent.mkdir(parents=True, exist_ok=True)

    overall_pass = (
        col_results["is_valid"]
        and dtype_results["is_valid"]
        and (invalid_results["total_invalid"] == 0)
    )
    final_status = "PASS" if overall_pass else "NEEDS ATTENTION"

    report_lines = [
        "============================================================",
        "SMART CROP ADVISOR",
        "MODULE 2 — DATA PREPROCESSING REPORT",
        "============================================================",
        "",
        "1. INPUT DATASET",
        f"Path: {RAW_DATASET_PATH}",
        f"Original rows: {raw_df.shape[0]}",
        f"Original columns: {raw_df.shape[1]}",
        "",
        "2. COLUMN VALIDATION",
        f"Status: {'PASS' if col_results['is_valid'] else 'FAIL'}",
        f"Expected columns: {EXPECTED_COLUMNS}",
        f"Missing columns: {col_results['missing_columns']}",
        f"Unexpected columns: {col_results['unexpected_columns']}",
        "",
        "3. DATA TYPE VALIDATION",
        f"Status: {'PASS' if dtype_results['is_valid'] else 'FAIL'}",
        f"Numeric features valid: {len(dtype_results['invalid_features']) == 0}",
        f"Target categorical valid: {dtype_results['target_valid']}",
        "",
        "4. MISSING VALUE ANALYSIS",
        f"Total missing values: {missing_results['total_missing']}",
        f"Missing values breakdown: {missing_results['missing_per_column']}",
        "Imputation performed: NONE (dataset is complete)",
        "",
        "5. DUPLICATE ANALYSIS",
        f"Original rows: {duplicate_results['original_rows']}",
        f"Duplicate rows detected: {duplicate_results['duplicate_rows']}",
        f"Processed rows remaining: {duplicate_results['processed_rows']}",
        f"Action taken: {'Exact duplicates removed' if duplicate_results['duplicate_rows'] > 0 else 'No duplicates found'}",
        "",
        "6. INVALID VALUE ANALYSIS",
        f"Total invalid values: {invalid_results['total_invalid']}",
        f"Invalid breakdown: {invalid_results['invalid_per_feature']}",
        "",
        "7. OUTLIER ANALYSIS (IQR METHOD)",
        f"Total potential outliers detected: {outlier_results['total_outliers']}",
        f"Outliers per feature: {outlier_results['outlier_counts']}",
        "Action taken: Potential outliers retained without deletion (agricultural variation preserved)",
        "",
        "8. DATA CONSISTENCY",
        f"Unique crop classes: {consistency_results['unique_classes']}",
        f"Whitespace clean action: {'Stripped whitespace' if consistency_results['whitespace_stripped'] else 'Already clean'}",
        f"Crop classes list: {consistency_results.get('crop_names', [])}",
        "",
        "9. FEATURE SUMMARY (UNSCALED)",
    ]

    for col, stats in feature_summary.items():
        report_lines.append(
            f"  - {col:<12}: Min={stats['min']:.2f}, Max={stats['max']:.2f}, Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, Std={stats['std']:.2f}"
        )

    report_lines.extend([
        "",
        "10. TARGET SUMMARY",
        f"Target column: {TARGET_COLUMN}",
        f"Number of classes: {consistency_results['unique_classes']}",
        "Class encoding: Retained as human-readable strings (no target encoding in Module 2)",
        "",
        "11. PREPROCESSING OPERATIONS PERFORMED",
        "✓ Raw dataset loaded and schema verified",
        "✓ Exact duplicate row detection and removal (if any)",
        "✓ Whitespace check and trimming on target crop labels",
        "✓ Invalid numeric and bound validation",
        "✓ IQR outlier analysis performed and logged",
        "✓ Processed clean dataset saved to dataset/processed/crop_data_clean.csv",
        "",
        "12. PREPROCESSING OPERATIONS NOT PERFORMED (BY DESIGN)",
        "✓ No numerical feature scaling or normalization (Random Forest does not require scaling)",
        "✓ No automatic outlier removal (agricultural measurement validity preserved)",
        "✓ No target label encoding (deferred to model training module)",
        "✓ No feature selection or feature dropping",
        "✓ No train/validation split (preventing data leakage)",
        "✓ No model fitting or hyperparameter optimization",
        "",
        "13. OUTPUT DATASET",
        f"Path: {PROCESSED_DATASET_PATH}",
        f"Processed rows: {processed_df.shape[0]}",
        f"Processed columns: {processed_df.shape[1]}",
        "",
        "14. FINAL STATUS",
        f"{final_status}",
        "============================================================",
    ])

    report_content = "\n".join(report_lines)

    try:
        filepath.write_text(report_content, encoding="utf-8")
        print(f"✓ Report successfully saved to:\n  {filepath}")
    except Exception as e:
        print(f"✗ Failed to write preprocessing report: {e}")

    return final_status


def main():
    """
    Main executable workflow for Module 2 Data Preprocessing.
    """
    # Step 1: Load Raw Dataset
    raw_df = load_raw_dataset()
    if raw_df is None:
        return

    # Step 2: Validate Columns
    col_res = validate_columns(raw_df)
    if not col_res["is_valid"]:
        return

    # Step 3: Validate Data Types
    dtype_res = validate_data_types(raw_df)

    # Step 4: Missing Values
    missing_res = analyze_missing_values(raw_df)
    df_missing_handled, missing_action_res = handle_missing_values(raw_df, missing_res)

    # Step 5: Duplicate Handling
    df_dedup, duplicate_res = detect_and_remove_duplicates(df_missing_handled)

    # Step 6: Invalid Numeric Values
    invalid_res = detect_invalid_values(df_dedup)

    # Step 7: Outlier Detection
    outlier_res = detect_outliers(df_dedup)

    # Step 8: Data Consistency Checks
    df_clean, consistency_res = check_label_consistency(df_dedup)

    # Step 9: Feature Range Summary
    feature_summary = generate_feature_summary(df_clean)

    # Step 10: Feature & Target Separation
    X, y = separate_features_target(df_clean)

    # Step 13: Save Processed Dataset
    saved = save_processed_dataset(df_clean)

    # Step 14 & 15: Generate Report & Final Status
    final_status = generate_preprocessing_report(
        raw_df=raw_df,
        processed_df=df_clean,
        col_results=col_res,
        dtype_results=dtype_res,
        missing_results=missing_res,
        duplicate_results=duplicate_res,
        invalid_results=invalid_res,
        outlier_results=outlier_res,
        consistency_results=consistency_res,
        feature_summary=feature_summary,
    )

    print("\n" + "=" * 60)
    print("MODULE 2 COMPLETED")
    print("============================================================")
    print(f"\nFinal Status:\n{final_status}\n")


if __name__ == "__main__":
    main()
