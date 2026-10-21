"""
Smart Crop Advisor – API Schemas
Defines request and response Pydantic data models for FastAPI endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class CropPredictionInput(BaseModel):
    N: float = Field(..., description="Nitrogen content in soil (mg/kg)", ge=0.0, le=140.0)
    P: float = Field(..., description="Phosphorus content in soil (mg/kg)", ge=0.0, le=150.0)
    K: float = Field(..., description="Potassium content in soil (mg/kg)", ge=0.0, le=210.0)
    temperature: float = Field(..., description="Air temperature (°C)", ge=0.0, le=60.0)
    humidity: float = Field(..., description="Relative humidity (%)", ge=0.0, le=100.0)
    ph: float = Field(..., description="Soil pH value", ge=0.0, le=14.0)

    @field_validator("ph")
    @classmethod
    def validate_ph_strictly_positive(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("Soil pH must be strictly positive (> 0.0)")
        return v


class TopPrediction(BaseModel):
    crop: str
    probability: float


class CropPredictionResponse(BaseModel):
    success: bool = True
    predicted_crop: str
    confidence: float
    model: str
    top_predictions: List[TopPrediction]
    input_summary: Dict[str, float]


class FeatureContribution(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    absolute_impact: float
    direction: str  # "supports" or "contradicts"


class LocalExplanationResponse(BaseModel):
    success: bool = True
    predicted_crop: str
    confidence: float
    base_value: float
    feature_contributions: List[FeatureContribution]


class ModelInfoResponse(BaseModel):
    model: str
    features: List[str]
    classes: int
    training_samples: int
    validation_samples: int
    n_estimators: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    max_features: str


class FeatureImportanceItem(BaseModel):
    feature: str
    rf_importance: float
    shap_importance: Optional[float] = None
    rank: int


class PerformanceMetricRow(BaseModel):
    metric: str
    baseline_rf: str
    ga_optimized_rf: str
    difference: str
