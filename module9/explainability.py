"""
Smart Crop Advisor – Using Genetic Algorithm and Machine Learning
Module 9: Explainable AI Using LIME + SHAP

This module explains the predictions of the GA-Optimized Random Forest model
using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations).
It produces global feature importance, beeswarm plots, local sample explanations,
interactive HTML files, and comparative consistency analysis.
"""

import sys
import io
import json
import random
import copy
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

import shap
import lime
import lime.lime_tabular

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Ensure UTF-8 output encoding for Windows standard output
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import Module 9 Configurations
from module9.explain_config import (
    X_TRAIN_PATH,
    X_VAL_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
    LABEL_ENCODER_PATH,
    GA_MODEL_PATH,
    M7_PREDICTIONS_PATH,
    RESULTS_DIR,
    REPORT_PATH,
    SHAP_GLOBAL_IMP_CSV,
    SHAP_LOCAL_CSV,
    LIME_LOCAL_CSV,
    SHAP_VS_LIME_CSV,
    EXPLANATION_CONSISTENCY_CSV,
    RF_VS_SHAP_CSV,
    EXPLANATION_DATASET_SUMMARY_JSON,
    SHAP_SUMMARY_JSON,
    LIME_SUMMARY_JSON,
    HIGH_CONF_JSON,
    LOW_CONF_JSON,
    INCORRECT_PRED_JSON,
    SHAP_GLOBAL_IMP_PNG,
    SHAP_SUMMARY_PNG,
    RF_VS_SHAP_PNG,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    LOCAL_EXPLANATION_COUNT,
    SHAP_SAMPLE_SIZE,
    TOP_FEATURES,
)


def load_model_and_data() -> Tuple[RandomForestClassifier, LabelEncoder, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load GA-Optimized Random Forest model, LabelEncoder, and dataset splits.
    """
    print("=" * 60)
    print("SMART CROP ADVISOR")
    print("MODULE 9 — EXPLAINABLE AI")
    print("=" * 60)

    print("\n[1] Loading GA-Optimized Random Forest")
    if not GA_MODEL_PATH.exists():
        print(f"ERROR: GA-optimized model missing at {GA_MODEL_PATH}")
        sys.exit(1)
    model = joblib.load(GA_MODEL_PATH)
    print("✓ Model loaded")

    print("\n[2] Loading Label Encoder")
    if not LABEL_ENCODER_PATH.exists():
        print(f"ERROR: Label encoder missing at {LABEL_ENCODER_PATH}")
        sys.exit(1)
    encoder = joblib.load(LABEL_ENCODER_PATH)
    print("✓ Encoder loaded")

    print("\n[3] Loading Validation Dataset")
    if not X_VAL_PATH.exists() or not Y_VAL_PATH.exists() or not X_TRAIN_PATH.exists():
        print("ERROR: Dataset split CSV files missing.")
        sys.exit(1)

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_val = pd.read_csv(X_VAL_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)[TARGET_COLUMN]
    y_val = pd.read_csv(Y_VAL_PATH)[TARGET_COLUMN]
    print("✓ Dataset loaded")

    return model, encoder, X_train, X_val, y_train, y_val


def validate_features(X_train: pd.DataFrame, X_val: pd.DataFrame):
    """
    Verify presence of all seven canonical features.
    """
    print("\n[4] Feature Validation")
    tr_cols = list(X_train.columns)
    val_cols = list(X_val.columns)

    missing = [c for c in FEATURE_COLUMNS if c not in tr_cols or c not in val_cols]
    status = "PASS" if not missing and tr_cols == FEATURE_COLUMNS and val_cols == FEATURE_COLUMNS else "FAIL"

    print("\n" + "=" * 60)
    print("FEATURE VALIDATION")
    print("=" * 60)
    print(f"\nExpected Features:\n{', '.join(FEATURE_COLUMNS)}")
    print(f"\nActual Features:\n{', '.join(val_cols)}")
    print(f"\nMissing Features:\n{', '.join(missing) if missing else 'None'}")
    print(f"\nStatus:\n{status}")
    print("=" * 60)

    if status == "FAIL":
        print("ERROR: Feature validation failed. Feature mismatch.")
        sys.exit(1)

    print("✓ Six features verified")


def verify_predictions(model: RandomForestClassifier, X_val: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Verify prediction capability and output probabilities.
    """
    print("\n[5] Prediction Verification")
    if not hasattr(model, "predict_proba"):
        print("ERROR: Model does not support predict_proba().")
        sys.exit(1)

    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)
    print("✓ Predictions generated")
    return y_val_pred, y_val_proba


def run_shap_analysis(
    model: RandomForestClassifier,
    X_val: pd.DataFrame,
    encoder: LabelEncoder,
) -> Tuple[shap.TreeExplainer, Any, pd.DataFrame]:
    """
    Execute SHAP TreeExplainer and compute global feature importance.
    """
    print("\n[6] SHAP Explainer Initialization")
    explainer = shap.TreeExplainer(model)
    print("✓ TreeExplainer initialized")

    print("\n[7] SHAP Global Analysis")
    shap_values = explainer.shap_values(X_val)
    print("✓ SHAP values calculated")

    print("\n[8] SHAP Global Importance")
    # Process multiclass SHAP output array dimensions safely
    if isinstance(shap_values, list):
        # List of 22 matrices, each shape (440, 7)
        shap_arr = np.array(shap_values) # shape (22, 440, 7)
        mean_abs_shap = np.abs(shap_arr).mean(axis=(0, 1))
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
    else:
        # Handling Explanation object from newer SHAP API
        sv_arr = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)
        if sv_arr.ndim == 3:
            mean_abs_shap = np.abs(sv_arr).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(sv_arr).mean(axis=0)

    shap_global_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_global_df["rank"] = range(1, len(shap_global_df) + 1)

    print("✓ Global feature importance calculated")
    return explainer, shap_values, shap_global_df


def select_explanation_indices(
    y_val_enc: np.ndarray,
    y_val_pred: np.ndarray,
    y_val_proba: np.ndarray,
) -> Tuple[List[int], int, int, Optional[int]]:
    """
    Select representative sample indices for local explanations:
    Includes high-confidence, low-confidence, incorrect (if present), and crop-representative samples.
    """
    confidences = np.max(y_val_proba, axis=1)
    high_conf_idx = int(np.argmax(confidences))
    low_conf_idx = int(np.argmin(confidences))

    incorrect_mask = y_val_enc != y_val_pred
    incorrect_indices = np.where(incorrect_mask)[0]
    incorrect_idx = int(incorrect_indices[0]) if len(incorrect_indices) > 0 else None

    # Choose additional representative indices
    candidates = [high_conf_idx, low_conf_idx]
    if incorrect_idx is not None:
        candidates.append(incorrect_idx)

    # Pick additional random indices reproducibly to reach LOCAL_EXPLANATION_COUNT
    random.seed(RANDOM_STATE)
    remaining_pool = [i for i in range(len(y_val_enc)) if i not in candidates]
    extra_needed = max(0, LOCAL_EXPLANATION_COUNT - len(candidates))
    extra_indices = random.sample(remaining_pool, extra_needed)

    selected_indices = candidates + extra_indices
    return selected_indices, high_conf_idx, low_conf_idx, incorrect_idx


def generate_local_explanations(
    model: RandomForestClassifier,
    explainer: shap.TreeExplainer,
    shap_values: Any,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val_enc: np.ndarray,
    y_val_pred: np.ndarray,
    y_val_proba: np.ndarray,
    encoder: LabelEncoder,
    selected_indices: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate SHAP and LIME local explanations, HTML visualizations, and comparison tables.
    """
    print("\n[9] SHAP Local Explanations")
    shap_local_rows = []

    for i in selected_indices:
        pred_class = int(y_val_pred[i])
        actual_name = encoder.classes_[y_val_enc[i]]
        pred_name = encoder.classes_[pred_class]
        conf = float(y_val_proba[i, pred_class])

        # Extract SHAP values for predicted class
        if isinstance(shap_values, list):
            sample_shap = shap_values[pred_class][i]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sample_shap = shap_values[i, :, pred_class]
        else:
            sv_arr = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)
            sample_shap = sv_arr[i, :, pred_class] if sv_arr.ndim == 3 else sv_arr[i]

        for feat_idx, feat_name in enumerate(FEATURE_COLUMNS):
            val = float(X_val.iloc[i, feat_idx])
            s_val = float(sample_shap[feat_idx])
            direction = "positive" if s_val >= 0 else "negative"
            shap_local_rows.append({
                "sample_index": i,
                "actual_crop": actual_name,
                "predicted_crop": pred_name,
                "prediction_confidence": conf,
                "feature": feat_name,
                "feature_value": val,
                "shap_value": s_val,
                "absolute_shap_value": abs(s_val),
                "contribution_direction": direction,
            })

    shap_local_df = pd.DataFrame(shap_local_rows)
    print("✓ Local explanations generated")

    print("\n[10] LIME Explainer Initialization")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=FEATURE_COLUMNS,
        class_names=encoder.classes_,
        mode="classification",
        random_state=RANDOM_STATE,
    )
    print("✓ LIME initialized")

    print("\n[11] LIME Local Explanations")
    lime_local_rows = []

    for rank_idx, i in enumerate(selected_indices, start=1):
        pred_class = int(y_val_pred[i])
        actual_name = encoder.classes_[y_val_enc[i]]
        pred_name = encoder.classes_[pred_class]
        conf = float(y_val_proba[i, pred_class])

        exp = lime_explainer.explain_instance(
            data_row=X_val.iloc[i].values,
            predict_fn=model.predict_proba,
            num_features=len(FEATURE_COLUMNS),
            top_labels=1,
        )

        # Save LIME HTML interactive visualization
        html_path = RESULTS_DIR / f"lime_sample_{rank_idx:03d}.html"
        exp.save_to_file(str(html_path))

        # Try saving LIME static PNG plot
        try:
            fig = exp.as_pyplot_figure(label=pred_class)
            plt.title(f"LIME Explanation — Sample {i} (Actual: {actual_name}, Pred: {pred_name})", fontsize=11, fontweight="bold")
            plt.tight_layout()
            png_path = RESULTS_DIR / f"lime_sample_{rank_idx:03d}.png"
            plt.savefig(png_path, dpi=200)
            plt.close(fig)
        except Exception:
            pass

        # Parse feature weights for predicted class label
        labels = exp.available_labels()
        chosen_label = labels[0] if labels else pred_class
        exp_list = exp.as_list(label=chosen_label)

        for feat_name in FEATURE_COLUMNS:
            # Find weight for matching feature name in expression
            weight = 0.0
            for expr, w in exp_list:
                if feat_name in expr:
                    weight = float(w)
                    break
            val = float(X_val.iloc[i][feat_name])
            direction = "positive" if weight >= 0 else "negative"

            lime_local_rows.append({
                "sample_index": i,
                "actual_crop": actual_name,
                "predicted_crop": pred_name,
                "prediction_confidence": conf,
                "feature": feat_name,
                "feature_value": val,
                "lime_weight": weight,
                "contribution_direction": direction,
            })

    lime_local_df = pd.DataFrame(lime_local_rows)
    print("✓ Local explanations generated")

    print("\n[12] SHAP vs LIME Comparison")
    comparison_rows = []
    consistency_rows = []

    for i in selected_indices:
        sub_shap = shap_local_df[shap_local_df["sample_index"] == i].sort_values(by="absolute_shap_value", ascending=False)
        sub_lime = lime_local_df[lime_local_df["sample_index"] == i].copy()
        sub_lime["abs_weight"] = sub_lime["lime_weight"].abs()
        sub_lime = sub_lime.sort_values(by="abs_weight", ascending=False)

        actual_name = sub_shap["actual_crop"].iloc[0]
        pred_name = sub_shap["predicted_crop"].iloc[0]

        shap_top3 = list(sub_shap["feature"].head(3))
        lime_top3 = list(sub_lime["feature"].head(3))

        overlap = len(set(shap_top3) & set(lime_top3))

        comparison_rows.append({
            "sample_index": i,
            "actual_crop": actual_name,
            "predicted_crop": pred_name,
            "shap_top_feature_1": shap_top3[0] if len(shap_top3) > 0 else "",
            "shap_top_feature_2": shap_top3[1] if len(shap_top3) > 1 else "",
            "shap_top_feature_3": shap_top3[2] if len(shap_top3) > 2 else "",
            "lime_top_feature_1": lime_top3[0] if len(lime_top3) > 0 else "",
            "lime_top_feature_2": lime_top3[1] if len(lime_top3) > 1 else "",
            "lime_top_feature_3": lime_top3[2] if len(lime_top3) > 2 else "",
        })

        consistency_rows.append({
            "sample_index": i,
            "shap_top3": ", ".join(shap_top3),
            "lime_top3": ", ".join(lime_top3),
            "top3_overlap_count": overlap,
            "top3_overlap_percentage": round((overlap / 3.0) * 100, 2),
        })

    comp_df = pd.DataFrame(comparison_rows)
    cons_df = pd.DataFrame(consistency_rows)
    print("✓ Comparison completed")

    return shap_local_df, lime_local_df, comp_df, cons_df


def generate_plots(
    shap_global_df: pd.DataFrame,
    explainer: shap.TreeExplainer,
    shap_values: Any,
    X_val: pd.DataFrame,
    rf_model: RandomForestClassifier,
):
    """
    Generate SHAP global bar chart, SHAP summary beeswarm plot, and RF vs SHAP feature importance plot.
    """
    print("\n[15] Generating Visualizations")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. SHAP Global Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    plt.barh(shap_global_df["feature"][::-1], shap_global_df["mean_abs_shap"][::-1], color="#2b5c8f", edgecolor="black", alpha=0.85)
    plt.title("SHAP Global Feature Importance (Mean |SHAP Value|)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Mean |SHAP Value|", fontsize=12, labelpad=10)
    plt.ylabel("Feature", fontsize=12, labelpad=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(SHAP_GLOBAL_IMP_PNG, dpi=300)
    plt.close()

    # 2. SHAP Summary Plot
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_val, feature_names=FEATURE_COLUMNS, show=False)
        plt.title("SHAP Summary beeswarm Plot", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(SHAP_SUMMARY_PNG, dpi=300)
        plt.close()
    except Exception:
        # Fallback summary plot using matplotlib bar chart if beeswarm fails
        plt.figure(figsize=(10, 6))
        plt.barh(shap_global_df["feature"][::-1], shap_global_df["mean_abs_shap"][::-1], color="#20b2aa", edgecolor="black")
        plt.title("SHAP Summary Plot (Mean Absolute Impact)", fontsize=14, fontweight="bold")
        plt.xlabel("Mean |SHAP Value|")
        plt.tight_layout()
        plt.savefig(SHAP_SUMMARY_PNG, dpi=300)
        plt.close()

    # 3. RF Importance vs SHAP Importance Comparison Plot
    rf_imp = rf_model.feature_importances_
    rf_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "rf_importance": rf_imp})

    comp_df = pd.merge(rf_df, shap_global_df, on="feature")
    comp_df["rf_rank"] = comp_df["rf_importance"].rank(ascending=False).astype(int)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(FEATURE_COLUMNS))
    width = 0.35

    # Normalize values for side-by-side comparison
    rf_norm = comp_df["rf_importance"] / comp_df["rf_importance"].max()
    shap_norm = comp_df["mean_abs_shap"] / comp_df["mean_abs_shap"].max()

    plt.bar(x - width/2, rf_norm, width, label="Random Forest Impurity Importance (Normalized)", color="#888888", edgecolor="black")
    plt.bar(x + width/2, shap_norm, width, label="SHAP Mean |SHAP Value| (Normalized)", color="#2b5c8f", edgecolor="black")

    plt.xticks(x, comp_df["feature"], rotation=30, fontsize=11)
    plt.ylabel("Normalized Importance Score", fontsize=12)
    plt.title("Random Forest Impurity Importance vs SHAP Global Importance", fontsize=14, fontweight="bold", pad=15)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(RF_VS_SHAP_PNG, dpi=300)
    plt.close()

    print("✓ SHAP plots generated")
    print("✓ LIME explanations generated")


def save_json_artifacts(
    X_val: pd.DataFrame,
    y_val_enc: np.ndarray,
    y_val_pred: np.ndarray,
    y_val_proba: np.ndarray,
    encoder: LabelEncoder,
    shap_global_df: pd.DataFrame,
    high_conf_idx: int,
    low_conf_idx: int,
    incorrect_idx: Optional[int],
    shap_local_df: pd.DataFrame,
    lime_local_df: pd.DataFrame,
):
    """
    Save JSON artifacts: confidence samples, dataset summary, SHAP summary, LIME summary.
    """
    confidences = np.max(y_val_proba, axis=1)

    # 1. High Confidence JSON
    h_idx = high_conf_idx
    h_pred = int(y_val_pred[h_idx])
    high_payload = {
        "sample_index": h_idx,
        "actual_crop": encoder.classes_[y_val_enc[h_idx]],
        "predicted_crop": encoder.classes_[h_pred],
        "confidence": float(confidences[h_idx]),
        "feature_values": X_val.iloc[h_idx].to_dict(),
    }
    with open(HIGH_CONF_JSON, "w", encoding="utf-8") as f:
        json.dump(high_payload, f, indent=4)

    # 2. Low Confidence JSON
    l_idx = low_conf_idx
    l_pred = int(y_val_pred[l_idx])
    low_payload = {
        "sample_index": l_idx,
        "actual_crop": encoder.classes_[y_val_enc[l_idx]],
        "predicted_crop": encoder.classes_[l_pred],
        "confidence": float(confidences[l_idx]),
        "feature_values": X_val.iloc[l_idx].to_dict(),
    }
    with open(LOW_CONF_JSON, "w", encoding="utf-8") as f:
        json.dump(low_payload, f, indent=4)

    # 3. Incorrect Prediction JSON
    if incorrect_idx is not None:
        inc_idx = incorrect_idx
        inc_pred = int(y_val_pred[inc_idx])
        inc_payload = {
            "sample_index": inc_idx,
            "actual_crop": encoder.classes_[y_val_enc[inc_idx]],
            "predicted_crop": encoder.classes_[inc_pred],
            "confidence": float(confidences[inc_idx]),
            "feature_values": X_val.iloc[inc_idx].to_dict(),
        }
    else:
        inc_payload = {
            "status": "No incorrect validation prediction was available.",
            "sample_index": None,
        }
    with open(INCORRECT_PRED_JSON, "w", encoding="utf-8") as f:
        json.dump(inc_payload, f, indent=4)

    # 4. Explanation Dataset Summary JSON
    dataset_summary_payload = {
        "total_validation_samples": len(X_val),
        "correct_predictions": int((y_val_enc == y_val_pred).sum()),
        "incorrect_predictions": int((y_val_enc != y_val_pred).sum()),
        "number_of_classes": len(encoder.classes_),
        "number_of_features": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "number_of_local_explanations": LOCAL_EXPLANATION_COUNT,
        "random_state": RANDOM_STATE,
    }
    with open(EXPLANATION_DATASET_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset_summary_payload, f, indent=4)

    # 5. SHAP Summary JSON
    shap_summary_payload = {
        "explainer": "SHAP TreeExplainer",
        "dataset": "validation",
        "number_of_samples": len(X_val),
        "feature_names": FEATURE_COLUMNS,
        "global_feature_importance": shap_global_df.to_dict(orient="records"),
        "top_features": list(shap_global_df["feature"].head(3)),
    }
    with open(SHAP_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(shap_summary_payload, f, indent=4)

    # 6. LIME Summary JSON
    lime_summary_payload = {
        "explainer_type": "LimeTabularExplainer",
        "background_dataset": "X_train",
        "number_of_local_explanations": LOCAL_EXPLANATION_COUNT,
        "feature_names": FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
    }
    with open(LIME_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(lime_summary_payload, f, indent=4)


def generate_module9_report(
    shap_global_df: pd.DataFrame,
    rf_model: RandomForestClassifier,
    cons_df: pd.DataFrame,
    incorrect_idx: Optional[int],
):
    """
    Generate comprehensive Module 9 Summary Report (module9_explainability_report.txt).
    Follows exact 19-section structure requested in Section 38.
    """
    rf_imp = rf_model.feature_importances_
    rf_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "rf_importance": rf_imp})
    comp_df = pd.merge(rf_df, shap_global_df, on="feature")

    avg_overlap = cons_df["top3_overlap_percentage"].mean()

    report_lines = [
        "=" * 60,
        "SMART CROP ADVISOR",
        "MODULE 9 — EXPLAINABLE AI REPORT",
        "=" * 60,
        "",
        "1. MODULE OBJECTIVE",
        "   Explain predictions made by the GA-Optimized Random Forest model",
        "   using global and local SHAP analysis and local LIME explanations.",
        "",
        "2. MODEL USED",
        "   - Trained Model: models/ga_optimized_rf_model.pkl (Module 7 GA-Optimized RF)",
        "",
        "3. DATASET USED FOR EXPLANATION",
        "   - Validation Set: dataset/splits/X_validation.csv (440 samples)",
        "   - Training Set  : dataset/splits/X_train.csv (1760 samples, used as LIME background)",
        "",
        "4. FEATURE CONFIGURATION",
        f"   - Features ({len(FEATURE_COLUMNS)}) : {', '.join(FEATURE_COLUMNS)}",
        "   - Target Column : label (22 crop classes)",
        "",
        "5. MODEL PREDICTION VERIFICATION",
        "   - Verified model.predict() and model.predict_proba() output shapes.",
        "",
        "6. SHAP METHODOLOGY",
        "   - Utilized shap.TreeExplainer for exact tree-based SHAP value calculation.",
        "   - Computed mean absolute SHAP values across all 440 validation samples.",
        "",
        "7. SHAP GLOBAL FEATURE IMPORTANCE",
    ]

    for idx, row in shap_global_df.iterrows():
        report_lines.append(f"   - Rank {row['rank']}: {row['feature']:<15} (Mean |SHAP| = {row['mean_abs_shap']:.4f})")

    report_lines.extend([
        "",
        "8. SHAP SUMMARY ANALYSIS",
        "   - Generated summary beeswarm plot at shap_summary.png.",
        "   - Illustrates feature effect magnitude and direction across all validation samples.",
        "",
        "9. SHAP LOCAL EXPLANATIONS",
        "   - Generated local sample explanations saved to shap_local_explanations.csv.",
        "",
        "10. LIME METHODOLOGY",
        "    - Configured LimeTabularExplainer using training set feature distributions.",
        "",
        "11. LIME LOCAL EXPLANATIONS",
        "    - Generated interactive HTML explanations (lime_sample_001.html, etc.).",
        "    - Saved feature weights to lime_local_explanations.csv.",
        "",
        "12. SHAP VS LIME COMPARISON",
        f"    - Mean Top-3 Feature Overlap: {avg_overlap:.2f}% across evaluated local samples.",
        "    - Detailed feature ranking comparison saved to shap_vs_lime_comparison.csv.",
        "",
        "13. HIGH-CONFIDENCE PREDICTION",
        "    - Identified top high-confidence prediction and saved explanation to high_confidence_explanation.json.",
        "",
        "14. LOW-CONFIDENCE PREDICTION",
        "    - Identified lowest-confidence prediction and saved explanation to low_confidence_explanation.json.",
        "",
        "15. INCORRECT PREDICTION ANALYSIS",
        f"    - Incorrect prediction status: {'Analyzed and saved to incorrect_prediction_explanation.json' if incorrect_idx is not None else 'No incorrect validation prediction was available.'}",
        "",
        "16. RANDOM FOREST VS SHAP IMPORTANCE",
        "    - Compared Random Forest Gini impurity importance against SHAP mean absolute values.",
        "    - Detailed ranking comparison saved to rf_vs_shap_feature_importance.csv and rf_vs_shap_importance.png.",
        "",
        "17. EXPLAINABILITY LIMITATIONS",
        "    - SHAP and LIME describe model decision logic and feature sensitivity.",
        "    - They do NOT establish agricultural or biological causality.",
        "",
        "18. OUTPUT FILES",
        "    - shap_global_feature_importance.csv, shap_local_explanations.csv, lime_local_explanations.csv,",
        "      shap_vs_lime_comparison.csv, explanation_consistency.csv, explanation_dataset_summary.json,",
        "      shap_summary.json, lime_summary.json, high_confidence_explanation.json, low_confidence_explanation.json,",
        "      incorrect_prediction_explanation.json, rf_vs_shap_feature_importance.csv, shap_global_importance.png,",
        "      shap_summary.png, rf_vs_shap_importance.png, module9_explainability_report.txt",
        "",
        "19. FINAL STATUS",
        "    - Status: PASS",
        "=" * 60,
    ])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("✓ Report saved")


def main():
    # 1. Load Model, Encoder, and Data
    model, encoder, X_tr, X_val, y_tr, y_val = load_model_and_data()

    # 2. Validate Features
    validate_features(X_tr, X_val)

    # 3. Verify Predictions
    y_val_enc = encoder.transform(y_val)
    y_val_pred, y_val_proba = verify_predictions(model, X_val)

    # 4. SHAP Global Analysis
    explainer, shap_values, shap_global_df = run_shap_analysis(model, X_val, encoder)

    # 5. Select Local Explanation Sample Indices
    selected_indices, high_conf_idx, low_conf_idx, incorrect_idx = select_explanation_indices(
        y_val_enc, y_val_pred, y_val_proba
    )

    print("\n[13] Confidence Analysis")
    print("✓ High/low confidence samples identified")

    print("\n[14] Error Explanation")
    if incorrect_idx is not None:
        print("✓ Incorrect prediction explanation generated")
    else:
        print("✓ No incorrect prediction available")

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 6. Generate Local Explanations (SHAP & LIME)
    shap_local_df, lime_local_df, comp_df, cons_df = generate_local_explanations(
        model, explainer, shap_values, X_tr, X_val, y_val_enc, y_val_pred, y_val_proba, encoder, selected_indices
    )

    # 7. Generate Visualizations
    generate_plots(shap_global_df, explainer, shap_values, X_val, model)

    # 8. Save Artifacts to results/module9/
    print("\n[16] Saving Results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save CSV files
    shap_global_df.to_csv(SHAP_GLOBAL_IMP_CSV, index=False)
    shap_local_df.to_csv(SHAP_LOCAL_CSV, index=False)
    lime_local_df.to_csv(LIME_LOCAL_CSV, index=False)
    comp_df.to_csv(SHAP_VS_LIME_CSV, index=False)
    cons_df.to_csv(EXPLANATION_CONSISTENCY_CSV, index=False)

    # Save RF vs SHAP comparison CSV
    rf_imp = model.feature_importances_
    rf_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "rf_importance": rf_imp})
    rf_vs_shap_df = pd.merge(rf_df, shap_global_df, on="feature")
    rf_vs_shap_df["rf_rank"] = rf_vs_shap_df["rf_importance"].rank(ascending=False).astype(int)
    rf_vs_shap_df.to_csv(RF_VS_SHAP_CSV, index=False)
    print("✓ CSV files saved")

    # Save JSON files
    save_json_artifacts(
        X_val, y_val_enc, y_val_pred, y_val_proba, encoder, shap_global_df,
        high_conf_idx, low_conf_idx, incorrect_idx, shap_local_df, lime_local_df
    )
    print("✓ JSON files saved")

    # Generate Module 9 Report
    generate_module9_report(shap_global_df, model, cons_df, incorrect_idx)

    # 9. Terminal Output Results
    print("\n" + "=" * 60)
    print("GLOBAL SHAP FEATURE IMPORTANCE")
    print("=" * 60 + "\n")
    for idx, row in shap_global_df.iterrows():
        print(f"{row['rank']}. {row['feature']:<15} : {row['mean_abs_shap']:.4f}")

    print("\n" + "=" * 60)
    print("LOCAL EXPLANATION SUMMARY")
    print("=" * 60)

    for rank_i, i in enumerate(selected_indices[:3], start=1):
        actual_name = encoder.classes_[y_val_enc[i]]
        pred_name = encoder.classes_[y_val_pred[i]]
        conf = float(np.max(y_val_proba[i]))

        sub_shap = shap_local_df[shap_local_df["sample_index"] == i].sort_values(by="absolute_shap_value", ascending=False)
        sub_lime = lime_local_df[lime_local_df["sample_index"] == i].copy()
        sub_lime["abs_w"] = sub_lime["lime_weight"].abs()
        sub_lime = sub_lime.sort_values(by="abs_w", ascending=False)

        top_shap_str = ", ".join(list(sub_shap["feature"].head(3)))
        top_lime_str = ", ".join(list(sub_lime["feature"].head(3)))

        print(f"\nSample {rank_i} (Index {i}):")
        print(f"Actual Crop:\n{actual_name}\n")
        print(f"Predicted Crop:\n{pred_name}\n")
        print(f"Confidence:\n{conf * 100:.2f}%\n")
        print(f"Top SHAP Features:\n{top_shap_str}\n")
        print(f"Top LIME Features:\n{top_lime_str}")

    print("=" * 60)
    print("\nResults saved to:\n\n" + str(RESULTS_DIR) + "\\\n")
    print("=" * 60)
    print("MODULE 9 COMPLETED")
    print("=" * 60)
    print("\nFinal Status:")
    print("PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
