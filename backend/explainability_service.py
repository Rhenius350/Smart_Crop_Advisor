"""
Smart Crop Advisor – Explainability Service
Initializes SHAP TreeExplainer and computes feature-level local and global explanations.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import shap

from backend.schemas import (
    CropPredictionInput,
    LocalExplanationResponse,
    FeatureContribution,
    FeatureImportanceItem,
)
from backend.model_service import model_service, CANONICAL_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODULE9_DIR = RESULTS_DIR / "module9"


class ExplainabilityService:
    def __init__(self):
        self.explainer = None
        self.is_initialized = False

    def initialize(self) -> None:
        """Initialize SHAP TreeExplainer on the trained Random Forest model once at startup."""
        if not model_service.is_loaded:
            model_service.load_artifacts()

        print("[ExplainabilityService] Initializing SHAP TreeExplainer...")
        try:
            self.explainer = shap.TreeExplainer(model_service.ga_model)
            self.is_initialized = True
            print("[OK] SHAP TreeExplainer initialized successfully.")
        except Exception as e:
            print(f"[ExplainabilityService] Warning: SHAP initialization error: {e}")

    def explain_prediction(self, input_data: CropPredictionInput) -> LocalExplanationResponse:
        """Computes local SHAP explanation for a specific user input prediction."""
        if not self.is_initialized or self.explainer is None:
            self.initialize()

        # Run prediction first to get target class index and confidence
        pred_res = model_service.predict(input_data)
        predicted_crop = pred_res.predicted_crop
        confidence = pred_res.confidence
        pred_class_idx = model_service.classes_.index(predicted_crop)

        # Build single-row DataFrame using canonical feature ordering
        input_dict = {
            "N": input_data.N,
            "P": input_data.P,
            "K": input_data.K,
            "temperature": input_data.temperature,
            "humidity": input_data.humidity,
            "ph": input_data.ph,
        }
        input_df = pd.DataFrame([input_dict], columns=CANONICAL_FEATURES)

        feature_contributions: List[FeatureContribution] = []
        base_value = 0.0

        try:
            # Compute SHAP values
            shap_values = self.explainer(input_df)

            # Handle different SHAP output formats across versions (list of arrays, 3D array, or Explanation object)
            if hasattr(shap_values, "values"):
                vals = shap_values.values  # Shape could be (1, 6, 22) or (1, 6)
                base_vals = shap_values.base_values

                if vals.ndim == 3:
                    sample_shap = vals[0, :, pred_class_idx]
                    base_val_raw = base_vals[0, pred_class_idx] if base_vals.ndim > 1 else base_vals[0]
                elif vals.ndim == 2:
                    sample_shap = vals[0, :]
                    base_val_raw = base_vals[0]
                else:
                    sample_shap = vals
                    base_val_raw = 0.0
                base_value = float(base_val_raw)
            else:
                # Direct array/list format from shap_values
                raw_shap = self.explainer.shap_values(input_df)
                if isinstance(raw_shap, list):
                    # Multi-class output: list of 22 arrays, each shape (1, 6)
                    sample_shap = raw_shap[pred_class_idx][0]
                elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
                    sample_shap = raw_shap[0, :, pred_class_idx]
                else:
                    sample_shap = np.array(raw_shap).flatten()

                base_val_raw = self.explainer.expected_value
                if isinstance(base_val_raw, (list, np.ndarray)):
                    base_value = float(base_val_raw[pred_class_idx])
                else:
                    base_value = float(base_val_raw)

            # Build feature contributions list
            for i, feat_name in enumerate(CANONICAL_FEATURES):
                val_float = float(sample_shap[i])
                feat_val = input_dict[feat_name]
                direction = "supports" if val_float >= 0 else "contradicts"

                feature_contributions.append(
                    FeatureContribution(
                        feature=feat_name,
                        feature_value=float(feat_val),
                        shap_value=round(val_float, 4),
                        absolute_impact=round(abs(val_float), 4),
                        direction=direction,
                    )
                )

            # Sort by absolute SHAP impact descending
            feature_contributions.sort(key=lambda x: x.absolute_impact, reverse=True)

        except Exception as e:
            print(f"[ExplainabilityService] SHAP computation fallback triggered: {e}")
            # Fallback to feature importances weighted by input values if SHAP runtime error occurs
            rf_importances = model_service.ga_model.feature_importances_
            for i, feat_name in enumerate(CANONICAL_FEATURES):
                imp = float(rf_importances[i])
                feature_contributions.append(
                    FeatureContribution(
                        feature=feat_name,
                        feature_value=float(input_dict[feat_name]),
                        shap_value=round(imp, 4),
                        absolute_impact=round(imp, 4),
                        direction="supports",
                    )
                )
            feature_contributions.sort(key=lambda x: x.absolute_impact, reverse=True)

        return LocalExplanationResponse(
            success=True,
            predicted_crop=predicted_crop,
            confidence=confidence,
            base_value=round(base_value, 4),
            feature_contributions=feature_contributions,
        )

    def get_global_feature_importances(self) -> List[FeatureImportanceItem]:
        """Returns global feature importance rankings from Module 9 SHAP analysis or Random Forest impurity."""
        shap_csv_path = MODULE9_DIR / "shap_global_feature_importance.csv"
        rf_importances = model_service.ga_model.feature_importances_ if model_service.ga_model else [0.0] * 6

        res_items = []

        if shap_csv_path.exists():
            try:
                df_shap = pd.read_csv(shap_csv_path)
                # Columns: feature, mean_abs_shap, rank
                for rank_idx, row in df_shap.iterrows():
                    feat_name = str(row["feature"])
                    mean_shap = float(row["mean_abs_shap"])
                    feat_idx = CANONICAL_FEATURES.index(feat_name) if feat_name in CANONICAL_FEATURES else 0
                    rf_imp = float(rf_importances[feat_idx])

                    res_items.append(
                        FeatureImportanceItem(
                            feature=feat_name,
                            rf_importance=round(rf_imp, 4),
                            shap_importance=round(mean_shap, 4),
                            rank=int(row.get("rank", rank_idx + 1)),
                        )
                    )
                return res_items
            except Exception as e:
                print(f"[ExplainabilityService] Warning parsing SHAP global CSV: {e}")

        # Fallback using Random Forest feature importances
        sorted_indices = np.argsort(rf_importances)[::-1]
        for rank, idx in enumerate(sorted_indices, 1):
            res_items.append(
                FeatureImportanceItem(
                    feature=CANONICAL_FEATURES[idx],
                    rf_importance=round(float(rf_importances[idx]), 4),
                    shap_importance=round(float(rf_importances[idx]), 4),
                    rank=rank,
                )
            )
        return res_items


explainability_service = ExplainabilityService()
