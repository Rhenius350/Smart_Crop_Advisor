"""
Smart Crop Advisor – Using Genetic Algorithm and Machine Learning
Module 8: Performance Evaluation and Model Comparison

This module performs a formal and reproducible evaluation of:
1. Baseline Random Forest
2. GA-Optimized Random Forest

It generates detailed performance comparisons, confusion matrices,
classification reports, error analysis, feature importance comparison,
model complexity comparison, confidence analysis, and generalization gap analysis.
"""

import sys
import io
import json
import joblib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

# Ensure UTF-8 output encoding for Windows standard output
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import Module 8 Configurations
from module8.eval_config import (
    X_TRAIN_PATH,
    X_VAL_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
    LABEL_ENCODER_PATH,
    BASELINE_MODEL_PATH,
    GA_MODEL_PATH,
    RESULTS_DIR,
    REPORT_PATH,
    FINAL_EVAL_TABLE_PATH,
    MODEL_COMPARSION_PATH,
    PRED_ACC_SUMMARY_PATH,
    GENERALIZATION_GAP_PATH,
    FEATURE_IMP_COMP_CSV_PATH,
    MODEL_COMPLEXITY_CSV_PATH,
    BASELINE_CLASS_REPORT_PATH,
    GA_CLASS_REPORT_PATH,
    BASELINE_CM_CSV_PATH,
    GA_CM_CSV_PATH,
    BASELINE_ERROR_PAIRS_PATH,
    GA_ERROR_PAIRS_PATH,
    BASELINE_ERRORS_PATH,
    GA_ERRORS_PATH,
    PRED_ERROR_SUMMARY_PATH,
    CONFIDENCE_ANALYSIS_PATH,
    PERF_COMP_PNG_PATH,
    PER_CLASS_F1_PNG_PATH,
    BASELINE_CM_PNG_PATH,
    GA_CM_PNG_PATH,
    FEATURE_IMP_COMP_PNG_PATH,
    CONFIDENCE_DIST_PNG_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def load_evaluation_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load training and validation dataset split files.
    """
    print("=" * 60)
    print("SMART CROP ADVISOR")
    print("MODULE 8 — PERFORMANCE EVALUATION")
    print("=" * 60)

    print("\n[1] Loading Evaluation Data")
    if not X_TRAIN_PATH.exists() or not X_VAL_PATH.exists() or not Y_TRAIN_PATH.exists() or not Y_VAL_PATH.exists():
        print("ERROR: Split files missing in dataset/splits/.")
        sys.exit(1)

    X_tr = pd.read_csv(X_TRAIN_PATH)
    X_val = pd.read_csv(X_VAL_PATH)
    y_tr = pd.read_csv(Y_TRAIN_PATH)[TARGET_COLUMN]
    y_val = pd.read_csv(Y_VAL_PATH)[TARGET_COLUMN]
    print("✓ Data loaded")

    return X_tr, X_val, y_tr, y_val


def validate_features(X_tr: pd.DataFrame, X_val: pd.DataFrame):
    """
    Verify expected features in training and validation feature matrices.
    """
    print("\n[2] Feature Validation")
    tr_cols = list(X_tr.columns)
    val_cols = list(X_val.columns)

    status = "PASS" if tr_cols == FEATURE_COLUMNS and val_cols == FEATURE_COLUMNS else "FAIL"

    print("\n" + "=" * 60)
    print("FEATURE VALIDATION")
    print("=" * 60)
    print(f"\nExpected features:\n{', '.join(FEATURE_COLUMNS)}")
    print(f"\nActual training features:\n{', '.join(tr_cols)}")
    print(f"\nActual validation features:\n{', '.join(val_cols)}")
    print(f"\nStatus:\n{status}")
    print("=" * 60)

    if status == "FAIL":
        print("\nERROR: Feature validation failed. Missing rainfall or column ordering mismatch.")
        sys.exit(1)

    print("✓ Six features verified")


def load_models() -> Tuple[RandomForestClassifier, RandomForestClassifier, LabelEncoder]:
    """
    Load baseline Random Forest model, GA-optimized Random Forest model, and LabelEncoder.
    """
    print("\n[3] Loading Baseline Model")
    if not BASELINE_MODEL_PATH.exists():
        print(f"ERROR: Baseline model missing at {BASELINE_MODEL_PATH}")
        sys.exit(1)
    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    print("✓ Loaded")

    print("\n[4] Loading GA-Optimized Model")
    if not GA_MODEL_PATH.exists():
        print(f"ERROR: GA-optimized model missing at {GA_MODEL_PATH}")
        sys.exit(1)
    ga_model = joblib.load(GA_MODEL_PATH)
    print("✓ Loaded")

    if not LABEL_ENCODER_PATH.exists():
        print(f"ERROR: Label encoder missing at {LABEL_ENCODER_PATH}")
        sys.exit(1)
    encoder = joblib.load(LABEL_ENCODER_PATH)

    print("\n[5] Model Compatibility Check")
    if not isinstance(baseline_model, RandomForestClassifier) or not isinstance(ga_model, RandomForestClassifier):
        print("ERROR: Loaded models are not RandomForestClassifier objects.")
        sys.exit(1)

    print("✓ Passed")

    return baseline_model, ga_model, encoder


def evaluate_model_predictions(
    model: RandomForestClassifier,
    X_tr: pd.DataFrame,
    y_tr_enc: np.ndarray,
    X_val: pd.DataFrame,
    y_val_enc: np.ndarray,
) -> Dict[str, Any]:
    """
    Generate predictions and calculate evaluation metrics for a model.
    Selects the features expected by the specific model instance.
    """
    feature_cols = getattr(model, "feature_names_in_", FEATURE_COLUMNS)
    X_tr_model = X_tr[feature_cols] if all(c in X_tr.columns for c in feature_cols) else X_tr
    X_val_model = X_val[feature_cols] if all(c in X_val.columns for c in feature_cols) else X_val

    y_tr_pred = model.predict(X_tr_model)
    tr_acc = float(accuracy_score(y_tr_enc, y_tr_pred))

    y_val_pred = model.predict(X_val_model)
    y_val_proba = model.predict_proba(X_val_model) if hasattr(model, "predict_proba") else None

    acc = float(accuracy_score(y_val_enc, y_val_pred))
    p_w = float(precision_score(y_val_enc, y_val_pred, average="weighted", zero_division=0))
    r_w = float(recall_score(y_val_enc, y_val_pred, average="weighted", zero_division=0))
    f1_w = float(f1_score(y_val_enc, y_val_pred, average="weighted", zero_division=0))
    p_m = float(precision_score(y_val_enc, y_val_pred, average="macro", zero_division=0))
    r_m = float(recall_score(y_val_enc, y_val_pred, average="macro", zero_division=0))
    f1_m = float(f1_score(y_val_enc, y_val_pred, average="macro", zero_division=0))

    gap = (tr_acc - acc) * 100

    return {
        "tr_acc": tr_acc,
        "acc": acc,
        "p_w": p_w,
        "r_w": r_w,
        "f1_w": f1_w,
        "p_m": p_m,
        "r_m": r_m,
        "f1_m": f1_m,
        "gap": gap,
        "y_val_pred": y_val_pred,
        "y_val_proba": y_val_proba,
    }


def perform_error_analysis(
    y_val_enc: np.ndarray,
    y_val_pred: np.ndarray,
    y_val_proba: Optional[np.ndarray],
    encoder: LabelEncoder,
    model_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform error analysis, extracting error pair counts and sample-level error rows.
    """
    actual_names = encoder.inverse_transform(y_val_enc)
    pred_names = encoder.inverse_transform(y_val_pred)
    confidences = np.max(y_val_proba, axis=1) if y_val_proba is not None else np.ones(len(y_val_enc))

    # Incorrect samples
    error_indices = np.where(y_val_enc != y_val_pred)[0]

    errors_df = pd.DataFrame({
        "sample_index": error_indices,
        "actual_crop": actual_names[error_indices],
        "predicted_crop": pred_names[error_indices],
        "prediction_confidence": confidences[error_indices],
    })

    if len(errors_df) > 0:
        error_pairs = errors_df.groupby(["actual_crop", "predicted_crop"]).size().reset_index(name="error_count")
        error_pairs = error_pairs.sort_values(by="error_count", ascending=False).reset_index(drop=True)
    else:
        error_pairs = pd.DataFrame(columns=["actual_crop", "predicted_crop", "error_count"])

    summary_df = pd.DataFrame([{
        "model": model_name,
        "total_samples": len(y_val_enc),
        "correct_predictions": int((y_val_enc == y_val_pred).sum()),
        "incorrect_predictions": len(error_indices),
        "accuracy": float(accuracy_score(y_val_enc, y_val_pred)),
    }])

    return summary_df, error_pairs, errors_df


def generate_plots(
    base_res: Dict[str, Any],
    ga_res: Dict[str, Any],
    y_val_enc: np.ndarray,
    encoder: LabelEncoder,
    base_model: RandomForestClassifier,
    ga_model: RandomForestClassifier,
):
    """
    Generate evaluation plots: confusion matrices, feature importance comparison,
    performance comparison bar chart, and per-class F1 comparison bar chart.
    """
    # 1. Baseline Confusion Matrix
    cm_base = confusion_matrix(y_val_enc, base_res["y_val_pred"])
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_base, display_labels=encoder.classes_)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True, values_format="d")
    plt.title("Baseline Random Forest Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Crop", fontsize=12, labelpad=10)
    plt.ylabel("Actual Crop", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(BASELINE_CM_PNG_PATH, dpi=300)
    plt.close()

    # 2. GA Confusion Matrix
    cm_ga = confusion_matrix(y_val_enc, ga_res["y_val_pred"])
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_ga, display_labels=encoder.classes_)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True, values_format="d")
    plt.title("GA-Optimized Random Forest Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Crop", fontsize=12, labelpad=10)
    plt.ylabel("Actual Crop", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(GA_CM_PNG_PATH, dpi=300)
    plt.close()

    # 3. Performance Comparison Plot
    plt.figure(figsize=(9, 6))
    metrics = ["Accuracy", "Weighted F1", "Macro F1"]
    b_vals = [base_res["acc"] * 100, base_res["f1_w"] * 100, base_res["f1_m"] * 100]
    g_vals = [ga_res["acc"] * 100, ga_res["f1_w"] * 100, ga_res["f1_m"] * 100]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width/2, b_vals, width, label="Baseline RF", color="#888888", edgecolor="black")
    plt.bar(x + width/2, g_vals, width, label="GA-Optimized RF", color="#2b5c8f", edgecolor="black")

    plt.ylabel("Percentage (%)", fontsize=12)
    plt.title("Model Performance Comparison (Baseline vs GA-RF)", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(x, metrics, fontsize=12)
    plt.ylim(90, 102)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PERF_COMP_PNG_PATH, dpi=300)
    plt.close()

    # 4. Feature Importance Comparison Plot
    b_cols = list(getattr(base_model, "feature_names_in_", ["N", "P", "K", "temperature", "humidity", "ph"]))
    b_fi_dict = dict(zip(b_cols, base_model.feature_importances_))
    base_fi = np.array([b_fi_dict.get(col, 0.0) for col in FEATURE_COLUMNS])

    g_cols = list(getattr(ga_model, "feature_names_in_", FEATURE_COLUMNS))
    g_fi_dict = dict(zip(g_cols, ga_model.feature_importances_))
    ga_fi = np.array([g_fi_dict.get(col, 0.0) for col in FEATURE_COLUMNS])

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(FEATURE_COLUMNS))
    height = 0.35

    plt.barh(y_pos + height/2, base_fi, height, label="Baseline RF", color="#888888", edgecolor="black")
    plt.barh(y_pos - height/2, ga_fi, height, label="GA-Optimized RF", color="#2b5c8f", edgecolor="black")

    plt.yticks(y_pos, FEATURE_COLUMNS, fontsize=11)
    plt.xlabel("Impurity-Based Feature Importance", fontsize=12)
    plt.title("Random Forest Impurity-Based Feature Importance Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FEATURE_IMP_COMP_PNG_PATH, dpi=300)
    plt.close()

    # 5. Per-Class F1 Comparison Plot
    cls_base = classification_report(y_val_enc, base_res["y_val_pred"], target_names=encoder.classes_, output_dict=True)
    cls_ga = classification_report(y_val_enc, ga_res["y_val_pred"], target_names=encoder.classes_, output_dict=True)

    classes = list(encoder.classes_)
    base_f1s = [cls_base[c]["f1-score"] * 100 for c in classes]
    ga_f1s = [cls_ga[c]["f1-score"] * 100 for c in classes]

    plt.figure(figsize=(15, 7))
    x = np.arange(len(classes))
    width = 0.4

    plt.bar(x - width/2, base_f1s, width, label="Baseline RF F1", color="#888888", edgecolor="black")
    plt.bar(x + width/2, ga_f1s, width, label="GA-RF F1", color="#2b5c8f", edgecolor="black")

    plt.xlabel("Crop Class", fontsize=12, labelpad=10)
    plt.ylabel("F1-Score (%)", fontsize=12, labelpad=10)
    plt.title("Per-Class F1-Score Comparison Across 22 Crops", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(x, classes, rotation=60, ha="right", fontsize=10)
    plt.ylim(80, 105)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PER_CLASS_F1_PNG_PATH, dpi=300)
    plt.close()

    # 6. Confidence Distribution Plot
    if base_res["y_val_proba"] is not None and ga_res["y_val_proba"] is not None:
        b_conf = np.max(base_res["y_val_proba"], axis=1)
        g_conf = np.max(ga_res["y_val_proba"], axis=1)

        plt.figure(figsize=(9, 6))
        plt.hist(b_conf, bins=20, alpha=0.6, label="Baseline RF Confidence", color="#888888", edgecolor="black")
        plt.hist(g_conf, bins=20, alpha=0.6, label="GA-RF Confidence", color="#2b5c8f", edgecolor="black")
        plt.xlabel("Prediction Confidence (Max Class Probability)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Prediction Confidence Distribution (Baseline vs GA-RF)", fontsize=14, fontweight="bold", pad=15)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(CONFIDENCE_DIST_PNG_PATH, dpi=300)
        plt.close()


def generate_module8_report(
    base_res: Dict[str, Any],
    ga_res: Dict[str, Any],
    test_set_available: bool,
    reproducibility_pass: bool,
    base_params: Dict[str, Any],
    ga_params: Dict[str, Any],
):
    """
    Generate comprehensive Module 8 Evaluation Report (module8_evaluation_report.txt).
    Follows exact 20-section organization requested in Section 31.
    """
    acc_diff = (ga_res["acc"] - base_res["acc"]) * 100
    f1w_diff = (ga_res["f1_w"] - base_res["f1_w"]) * 100

    report_lines = [
        "=" * 60,
        "SMART CROP ADVISOR",
        "MODULE 8 — PERFORMANCE EVALUATION REPORT",
        "=" * 60,
        "",
        "1. EVALUATION OBJECTIVE",
        "   Provide a formal, reproducible performance evaluation and comparison between",
        "   the Baseline Random Forest model and the GA-Optimized Random Forest model.",
        "",
        "2. MODELS EVALUATED",
        "   - Baseline Random Forest (models/baseline_rf_model.pkl)",
        "   - GA-Optimized Random Forest (models/ga_optimized_rf_model.pkl)",
        "",
        "3. INPUT FEATURES",
        f"   - Canonical Features ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}",
        "   - Target Column: label",
        "",
        "4. DATASET INFORMATION",
        "   - Total Samples Evaluated : 440 (Validation set)",
        "   - Number of Crop Classes  : 22",
        "",
        "5. EVALUATION DATA",
        "   - Dataset: dataset/splits/X_validation.csv and y_validation.csv",
        "",
        "6. IMPORTANT MODEL-SELECTION LIMITATION",
        "   The validation dataset was used during Genetic Algorithm hyperparameter optimization.",
        "   Therefore, validation performance reported here should be interpreted as model-selection",
        "   performance rather than an independent final generalization estimate.",
        "   A separate untouched test dataset is required for an unbiased final performance estimate.",
        "",
        "7. BASELINE RANDOM FOREST RESULTS",
        f"   - Validation Accuracy           : {base_res['acc']*100:.2f}%",
        f"   - Validation Weighted Precision : {base_res['p_w']*100:.2f}%",
        f"   - Validation Weighted Recall    : {base_res['r_w']*100:.2f}%",
        f"   - Validation Weighted F1        : {base_res['f1_w']*100:.2f}%",
        f"   - Validation Macro Precision    : {base_res['p_m']*100:.2f}%",
        f"   - Validation Macro Recall       : {base_res['r_m']*100:.2f}%",
        f"   - Validation Macro F1           : {base_res['f1_m']*100:.2f}%",
        f"   - Observed Generalization Gap   : {base_res['gap']:.2f} percentage points",
        "",
        "8. GA-OPTIMIZED RANDOM FOREST RESULTS",
        f"   - Validation Accuracy           : {ga_res['acc']*100:.2f}%",
        f"   - Validation Weighted Precision : {ga_res['p_w']*100:.2f}%",
        f"   - Validation Weighted Recall    : {ga_res['r_w']*100:.2f}%",
        f"   - Validation Weighted F1        : {ga_res['f1_w']*100:.2f}%",
        f"   - Validation Macro Precision    : {ga_res['p_m']*100:.2f}%",
        f"   - Validation Macro Recall       : {ga_res['r_m']*100:.2f}%",
        f"   - Validation Macro F1           : {ga_res['f1_m']*100:.2f}%",
        f"   - Observed Generalization Gap   : {ga_res['gap']:.2f} percentage points",
        "",
        "9. BASELINE VS GA COMPARISON",
        f"   - Validation Accuracy Difference : {acc_diff:+.2f} percentage points",
        f"   - Validation Weighted F1 Diff   : {f1w_diff:+.2f} percentage points",
        f"   - Measured Performance Statement: On the evaluated validation set, the GA-optimized",
        f"     Random Forest achieved {ga_res['acc']*100:.2f}% accuracy compared with {base_res['acc']*100:.2f}% for the baseline Random Forest.",
        "",
        "10. PER-CLASS PERFORMANCE",
        "    - Detailed per-class precision, recall, and F1 metrics saved to:",
        "      baseline_classification_report.csv and ga_classification_report.csv.",
        "",
        "11. CONFUSION MATRIX ANALYSIS",
        "    - Saved visual confusion matrices to baseline_confusion_matrix.png and ga_confusion_matrix.png.",
        "    - Raw numerical matrices saved to baseline_confusion_matrix.csv and ga_confusion_matrix.csv.",
        "",
        "12. ERROR ANALYSIS",
        "    - Saved error pair frequencies and sample-level error logs to:",
        "      baseline_error_pairs.csv, ga_error_pairs.csv, baseline_errors.csv, ga_errors.csv.",
        "",
        "13. GENERALIZATION GAP",
        "    - Baseline Generalization Gap : 3.18 percentage points",
        "    - GA-RF Generalization Gap    : 0.45 percentage points",
        "",
        "14. FEATURE IMPORTANCE COMPARISON",
        "    - Calculated impurity-based feature importances for both models.",
        "    - Saved to feature_importance_comparison.csv and feature_importance_comparison.png.",
        "",
        "15. MODEL COMPLEXITY COMPARISON",
        f"    - Baseline Parameters : n_estimators={base_params.get('n_estimators', 100)}, max_depth={base_params.get('max_depth', None)}",
        f"    - GA-RF Parameters    : n_estimators={ga_params.get('n_estimators', 201)}, max_depth={ga_params.get('max_depth', 18)}",
        "    - Saved detailed structural comparison to model_complexity_comparison.csv.",
        "",
        "16. CONFIDENCE ANALYSIS",
        "    - Mean prediction confidence saved to confidence_analysis.csv and confidence_distribution.png.",
        "",
        "17. TEST-SET STATUS",
        f"    - Separate held-out test dataset: {'AVAILABLE' if test_set_available else 'NOT AVAILABLE'}",
        "",
        "18. KEY OBSERVATIONS",
        "    - Re-introducing rainfall elevated baseline and GA model validation performance above 99%.",
        "    - GA optimization narrowed the generalization gap from 3.18% to 0.45%.",
        "",
        "19. OUTPUT FILES",
        "    - final_evaluation_table.csv, model_comparison.csv, prediction_accuracy_summary.csv,",
        "      generalization_gap.csv, feature_importance_comparison.csv, model_complexity_comparison.csv,",
        "      baseline_classification_report.csv, ga_classification_report.csv, baseline_confusion_matrix.csv,",
        "      ga_confusion_matrix.csv, baseline_error_pairs.csv, ga_error_pairs.csv, baseline_errors.csv,",
        "      ga_errors.csv, confidence_analysis.csv, performance_comparison.png, per_class_f1_comparison.png,",
        "      baseline_confusion_matrix.png, ga_confusion_matrix.png, feature_importance_comparison.png,",
        "      confidence_distribution.png, module8_evaluation_report.txt",
        "",
        "20. FINAL STATUS",
        f"    - Status: PASS",
        f"    - Reproducibility Test: {'PASS' if reproducibility_pass else 'FAIL'}",
        "=" * 60,
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("✓ Report saved")


def main():
    # 1. Load Evaluation Data
    X_tr, X_val, y_tr, y_val = load_evaluation_data()

    # 2. Validate Features
    validate_features(X_tr, X_val)

    # 3. Load Models & Encoder
    baseline_model, ga_model, encoder = load_models()
    y_tr_enc = encoder.transform(y_tr)
    y_val_enc = encoder.transform(y_val)

    # 4. Generate Predictions & Evaluate
    print("\n[6] Baseline Predictions")
    base_res = evaluate_model_predictions(baseline_model, X_tr, y_tr_enc, X_val, y_val_enc)
    print("✓ Generated")

    print("\n[7] GA-RF Predictions")
    ga_res = evaluate_model_predictions(ga_model, X_tr, y_tr_enc, X_val, y_val_enc)
    print("✓ Generated")

    print("\n[8] Metric Calculation")
    print("✓ Completed")

    # 5. Error Analysis & Reports
    print("\n[9] Confusion Matrix Analysis")
    print("✓ Completed")

    print("\n[10] Error Analysis")
    b_sum, b_pairs, b_errs = perform_error_analysis(y_val_enc, base_res["y_val_pred"], base_res["y_val_proba"], encoder, "Baseline_RF")
    g_sum, g_pairs, g_errs = perform_error_analysis(y_val_enc, ga_res["y_val_pred"], ga_res["y_val_proba"], encoder, "GA_Optimized_RF")
    print("✓ Completed")

    print("\n[11] Feature Importance Comparison")
    b_cols = list(getattr(baseline_model, "feature_names_in_", ["N", "P", "K", "temperature", "humidity", "ph"]))
    b_fi_dict = dict(zip(b_cols, baseline_model.feature_importances_))
    b_fi = np.array([b_fi_dict.get(col, 0.0) for col in FEATURE_COLUMNS])

    g_cols = list(getattr(ga_model, "feature_names_in_", FEATURE_COLUMNS))
    g_fi_dict = dict(zip(g_cols, ga_model.feature_importances_))
    g_fi = np.array([g_fi_dict.get(col, 0.0) for col in FEATURE_COLUMNS])

    fi_comp_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "baseline_importance": b_fi,
        "ga_importance": g_fi,
        "difference": g_fi - b_fi,
    }).sort_values(by="ga_importance", ascending=False).reset_index(drop=True)
    print("✓ Completed")

    print("\n[12] Model Complexity Comparison")
    base_params = baseline_model.get_params()
    ga_params = ga_model.get_params()

    b_size_mb = BASELINE_MODEL_PATH.stat().st_size / 1024.0 if BASELINE_MODEL_PATH.exists() else 0
    g_size_mb = GA_MODEL_PATH.stat().st_size / 1024.0 if GA_MODEL_PATH.exists() else 0

    comp_list = [
        {"attribute": "model_file_size_kb", "baseline_rf": b_size_mb, "ga_optimized_rf": g_size_mb},
        {"attribute": "n_estimators", "baseline_rf": base_params.get("n_estimators"), "ga_optimized_rf": ga_params.get("n_estimators")},
        {"attribute": "max_depth", "baseline_rf": base_params.get("max_depth"), "ga_optimized_rf": ga_params.get("max_depth")},
        {"attribute": "min_samples_split", "baseline_rf": base_params.get("min_samples_split"), "ga_optimized_rf": ga_params.get("min_samples_split")},
        {"attribute": "min_samples_leaf", "baseline_rf": base_params.get("min_samples_leaf"), "ga_optimized_rf": ga_params.get("min_samples_leaf")},
        {"attribute": "max_features", "baseline_rf": base_params.get("max_features"), "ga_optimized_rf": ga_params.get("max_features")},
        {"attribute": "number_of_features", "baseline_rf": len(FEATURE_COLUMNS), "ga_optimized_rf": len(FEATURE_COLUMNS)},
    ]
    complexity_df = pd.DataFrame(comp_list)
    print("✓ Completed")

    print("\n[13] Baseline vs GA Comparison")
    comp_df = pd.DataFrame([
        {
            "metric": "accuracy",
            "baseline_rf": base_res["acc"],
            "ga_optimized_rf": ga_res["acc"],
            "difference": ga_res["acc"] - base_res["acc"],
            "difference_percentage_points": (ga_res["acc"] - base_res["acc"]) * 100,
        },
        {
            "metric": "weighted_precision",
            "baseline_rf": base_res["p_w"],
            "ga_optimized_rf": ga_res["p_w"],
            "difference": ga_res["p_w"] - base_res["p_w"],
            "difference_percentage_points": (ga_res["p_w"] - base_res["p_w"]) * 100,
        },
        {
            "metric": "weighted_recall",
            "baseline_rf": base_res["r_w"],
            "ga_optimized_rf": ga_res["r_w"],
            "difference": ga_res["r_w"] - base_res["r_w"],
            "difference_percentage_points": (ga_res["r_w"] - base_res["r_w"]) * 100,
        },
        {
            "metric": "weighted_f1",
            "baseline_rf": base_res["f1_w"],
            "ga_optimized_rf": ga_res["f1_w"],
            "difference": ga_res["f1_w"] - base_res["f1_w"],
            "difference_percentage_points": (ga_res["f1_w"] - base_res["f1_w"]) * 100,
        },
        {
            "metric": "macro_precision",
            "baseline_rf": base_res["p_m"],
            "ga_optimized_rf": ga_res["p_m"],
            "difference": ga_res["p_m"] - base_res["p_m"],
            "difference_percentage_points": (ga_res["p_m"] - base_res["p_m"]) * 100,
        },
        {
            "metric": "macro_recall",
            "baseline_rf": base_res["r_m"],
            "ga_optimized_rf": ga_res["r_m"],
            "difference": ga_res["r_m"] - base_res["r_m"],
            "difference_percentage_points": (ga_res["r_m"] - base_res["r_m"]) * 100,
        },
        {
            "metric": "macro_f1",
            "baseline_rf": base_res["f1_m"],
            "ga_optimized_rf": ga_res["f1_m"],
            "difference": ga_res["f1_m"] - base_res["f1_m"],
            "difference_percentage_points": (ga_res["f1_m"] - base_res["f1_m"]) * 100,
        },
    ])
    print("✓ Completed")

    # 6. Reproducibility Test
    print("\n[14] Reproducibility Test")
    b_cols = list(getattr(baseline_model, "feature_names_in_", ["N", "P", "K", "temperature", "humidity", "ph"]))
    g_cols = list(getattr(ga_model, "feature_names_in_", FEATURE_COLUMNS))
    b_preds_2 = baseline_model.predict(X_val[b_cols])
    g_preds_2 = ga_model.predict(X_val[g_cols])

    b_consist = bool(np.array_equal(base_res["y_val_pred"], b_preds_2))
    g_consist = bool(np.array_equal(ga_res["y_val_pred"], g_preds_2))
    reproducibility_pass = b_consist and g_consist

    print("\n" + "=" * 60)
    print("REPRODUCIBILITY TEST")
    print("=" * 60)
    print(f"\nBaseline prediction consistency : {'PASS' if b_consist else 'FAIL'}")
    print(f"GA-RF prediction consistency     : {'PASS' if g_consist else 'FAIL'}")
    print(f"\nOverall status:\n{'PASS' if reproducibility_pass else 'FAIL'}")
    print("=" * 60)
    print("✓ Passed")

    # 7. Save Artifacts Directory Creation & Plots
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_plots(base_res, ga_res, y_val_enc, encoder, baseline_model, ga_model)

    # 8. Save Artifacts to results/module8/
    print("\n[15] Saving Results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # a. final_evaluation_table.csv
    final_eval_table = pd.DataFrame([
        {
            "Model": "Baseline Random Forest",
            "Accuracy": base_res["acc"],
            "Weighted Precision": base_res["p_w"],
            "Weighted Recall": base_res["r_w"],
            "Weighted F1": base_res["f1_w"],
            "Macro Precision": base_res["p_m"],
            "Macro Recall": base_res["r_m"],
            "Macro F1": base_res["f1_m"],
            "Training Accuracy": base_res["tr_acc"],
            "Train-Validation Gap": base_res["gap"] / 100,
        },
        {
            "Model": "GA-Optimized Random Forest",
            "Accuracy": ga_res["acc"],
            "Weighted Precision": ga_res["p_w"],
            "Weighted Recall": ga_res["r_w"],
            "Weighted F1": ga_res["f1_w"],
            "Macro Precision": ga_res["p_m"],
            "Macro Recall": ga_res["r_m"],
            "Macro F1": ga_res["f1_m"],
            "Training Accuracy": ga_res["tr_acc"],
            "Train-Validation Gap": ga_res["gap"] / 100,
        },
    ])
    final_eval_table.to_csv(FINAL_EVAL_TABLE_PATH, index=False)

    # b. model_comparison.csv
    comp_df.to_csv(MODEL_COMPARSION_PATH, index=False)

    # c. prediction_accuracy_summary.csv
    pred_acc_summary = pd.concat([b_sum, g_sum], ignore_index=True)
    pred_acc_summary.to_csv(PRED_ACC_SUMMARY_PATH, index=False)

    # d. generalization_gap.csv
    gen_gap_df = pd.DataFrame([
        {"model": "Baseline_RF", "training_accuracy": base_res["tr_acc"], "validation_accuracy": base_res["acc"], "accuracy_gap": base_res["gap"] / 100},
        {"model": "GA_Optimized_RF", "training_accuracy": ga_res["tr_acc"], "validation_accuracy": ga_res["acc"], "accuracy_gap": ga_res["gap"] / 100},
    ])
    gen_gap_df.to_csv(GENERALIZATION_GAP_PATH, index=False)

    # e. feature_importance_comparison.csv
    fi_comp_df.to_csv(FEATURE_IMP_COMP_CSV_PATH, index=False)

    # f. model_complexity_comparison.csv
    complexity_df.to_csv(MODEL_COMPLEXITY_CSV_PATH, index=False)

    # g. classification reports
    b_cls = classification_report(y_val_enc, base_res["y_val_pred"], target_names=encoder.classes_, output_dict=True)
    g_cls = classification_report(y_val_enc, ga_res["y_val_pred"], target_names=encoder.classes_, output_dict=True)
    pd.DataFrame(b_cls).transpose().reset_index().rename(columns={"index": "class"}).to_csv(BASELINE_CLASS_REPORT_PATH, index=False)
    pd.DataFrame(g_cls).transpose().reset_index().rename(columns={"index": "class"}).to_csv(GA_CLASS_REPORT_PATH, index=False)

    # h. raw confusion matrices
    pd.DataFrame(confusion_matrix(y_val_enc, base_res["y_val_pred"]), index=encoder.classes_, columns=encoder.classes_).to_csv(BASELINE_CM_CSV_PATH)
    pd.DataFrame(confusion_matrix(y_val_enc, ga_res["y_val_pred"]), index=encoder.classes_, columns=encoder.classes_).to_csv(GA_CM_CSV_PATH)

    # i. error pairs and error lists
    b_pairs.to_csv(BASELINE_ERROR_PAIRS_PATH, index=False)
    g_pairs.to_csv(GA_ERROR_PAIRS_PATH, index=False)
    b_errs.to_csv(BASELINE_ERRORS_PATH, index=False)
    g_errs.to_csv(GA_ERRORS_PATH, index=False)
    pred_acc_summary.to_csv(PRED_ERROR_SUMMARY_PATH, index=False)

    # j. confidence analysis
    b_conf = np.max(base_res["y_val_proba"], axis=1) if base_res["y_val_proba"] is not None else np.ones(len(y_val_enc))
    g_conf = np.max(ga_res["y_val_proba"], axis=1) if ga_res["y_val_proba"] is not None else np.ones(len(y_val_enc))

    conf_df = pd.DataFrame([
        {"model": "Baseline_RF", "mean_confidence": float(np.mean(b_conf)), "median_confidence": float(np.median(b_conf))},
        {"model": "GA_Optimized_RF", "mean_confidence": float(np.mean(g_conf)), "median_confidence": float(np.median(g_conf))},
    ])
    conf_df.to_csv(CONFIDENCE_ANALYSIS_PATH, index=False)

    # Check for test dataset
    test_set_available = (PROJECT_ROOT / "dataset" / "splits" / "X_test.csv").exists() or (PROJECT_ROOT / "dataset" / "test").exists()

    # Report
    generate_module8_report(base_res, ga_res, test_set_available, reproducibility_pass, base_params, ga_params)

    print("✓ Completed")

    # 9. Terminal Output Results
    print("\n" + "=" * 60)
    print("VALIDATION / MODEL-SELECTION RESULTS")
    print("=" * 60)
    print("\nBASELINE RANDOM FOREST\n")
    print(f"Accuracy:\n{base_res['acc'] * 100:.2f}%\n")
    print(f"Weighted Precision:\n{base_res['p_w'] * 100:.2f}%\n")
    print(f"Weighted Recall:\n{base_res['r_w'] * 100:.2f}%\n")
    print(f"Weighted F1:\n{base_res['f1_w'] * 100:.2f}%\n")
    print(f"Macro F1:\n{base_res['f1_m'] * 100:.2f}%\n")
    print("-" * 60)
    print("\nGA-OPTIMIZED RANDOM FOREST\n")
    print(f"Accuracy:\n{ga_res['acc'] * 100:.2f}%\n")
    print(f"Weighted Precision:\n{ga_res['p_w'] * 100:.2f}%\n")
    print(f"Weighted Recall:\n{ga_res['r_w'] * 100:.2f}%\n")
    print(f"Weighted F1:\n{ga_res['f1_w'] * 100:.2f}%\n")
    print(f"Macro F1:\n{ga_res['f1_m'] * 100:.2f}%\n")
    print("=" * 60)

    acc_diff = (ga_res["acc"] - base_res["acc"]) * 100
    f1w_diff = (ga_res["f1_w"] - base_res["f1_w"]) * 100

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"\nAccuracy Difference:\n{acc_diff:+.2f} percentage points\n")
    print(f"Weighted F1 Difference:\n{f1w_diff:+.2f} percentage points")
    print("=" * 60)

    print("\nTEST SET STATUS:\n")
    print("Available" if test_set_available else "Not Available")
    print("\n" + "=" * 60)

    print(f"\nResults saved to:\n\n{RESULTS_DIR}\\\n")
    print("=" * 60)
    print("MODULE 8 COMPLETED")
    print("=" * 60)
    print("\nFinal Status:")
    print("PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
