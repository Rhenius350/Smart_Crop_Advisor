import sys
import io

# Ensure UTF-8 output encoding for Windows standard output
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from contextlib import asynccontextmanager
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    CropPredictionInput,
    CropPredictionResponse,
    LocalExplanationResponse,
    ModelInfoResponse,
    FeatureImportanceItem,
)
from backend.model_service import model_service
from backend.explainability_service import explainability_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Event Handler.
    Loads trained Random Forest models and initializes SHAP Explainer ONCE on startup.
    Prevents reloading binaries or retraining models on individual inference calls.
    """
    print("\n" + "=" * 60)
    print("SMART CROP ADVISOR -- FASTAPI BACKEND STARTUP")
    print("=" * 60)
    try:
        model_service.load_artifacts()
        explainability_service.initialize()
        print("[OK] All backend services initialized successfully.\n")
    except Exception as e:
        print(f"[ERROR] Critical Error initializing backend services: {e}")
    yield
    print("[FastAPI Lifespan] Shutting down backend server...")


app = FastAPI(
    title="SMART CROP ADVISOR API",
    description="AI-Powered Precision Agriculture Crop Recommendation System using GA-Optimized Random Forest and SHAP Explainable AI",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for React Frontend (http://localhost:5173)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", summary="API Health Check")
def get_health() -> Dict[str, str]:
    """Returns application health status."""
    return {
        "status": "healthy",
        "application": "SMART CROP ADVISOR"
    }


@app.get("/api/model-info", response_model=ModelInfoResponse, summary="Get Model Details & Hyperparameters")
def get_model_info():
    """Returns information and hyperparameters of the loaded GA-Optimized Random Forest model."""
    try:
        return model_service.get_model_info()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model info: {str(e)}"
        )


@app.post("/api/predict", response_model=CropPredictionResponse, summary="Predict Optimal Crop")
def predict_crop(input_data: CropPredictionInput):
    """
    Accepts 6 soil and environmental parameters (N, P, K, temperature, humidity, ph).
    Returns recommended crop, prediction confidence, model name, and top 3 predictions.
    """
    try:
        return model_service.predict(input_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/api/explain", response_model=LocalExplanationResponse, summary="Local Explainable AI (SHAP)")
def explain_prediction(input_data: CropPredictionInput):
    """
    Computes local SHAP feature contributions for a specific prediction request.
    Explains which features support or contradict the recommended crop.
    """
    try:
        return explainability_service.explain_prediction(input_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explainability error: {str(e)}"
        )


@app.get("/api/feature-importance", response_model=List[FeatureImportanceItem], summary="Global Feature Importance")
def get_feature_importance():
    """Returns global feature importances derived from SHAP analysis and Random Forest impurity."""
    try:
        return explainability_service.get_global_feature_importances()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch feature importances: {str(e)}"
        )


@app.get("/api/performance", summary="Model Performance Metrics")
def get_performance():
    """Returns formal evaluation metrics comparing Baseline Random Forest vs GA-Optimized Random Forest."""
    try:
        return model_service.get_performance_summary()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve performance metrics: {str(e)}"
        )


@app.get("/api/research", summary="Pipeline Research Metadata")
def get_research_data():
    """Returns metadata summary of Modules 1 through 9 pipeline execution."""
    return {
        "project": "SMART CROP ADVISOR",
        "domain": "Machine Learning + Precision Agriculture",
        "core_methodology": "Genetic Algorithm (GA) + Random Forest (RF)",
        "canonical_features": model_service.get_model_info().features,
        "dataset": {
            "total_samples": 2200,
            "training_samples": 1760,
            "validation_samples": 440,
            "split_ratio": "80% Train / 20% Validation",
            "num_classes": 22
        },
        "modules": [
            {"name": "Module 1", "title": "Dataset Management", "status": "PASS"},
            {"name": "Module 2", "title": "Data Preprocessing", "status": "PASS"},
            {"name": "Module 3", "title": "Feature Analysis", "status": "PASS"},
            {"name": "Module 4", "title": "Train/Validation Split", "status": "PASS"},
            {"name": "Module 5", "title": "Baseline Random Forest", "status": "PASS"},
            {"name": "Module 6", "title": "Genetic Algorithm Optimization", "status": "PASS"},
            {"name": "Module 7", "title": "GA-Optimized Random Forest", "status": "PASS"},
            {"name": "Module 8", "title": "Performance Evaluation", "status": "PASS"},
            {"name": "Module 9", "title": "Explainable AI (SHAP + LIME)", "status": "PASS"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
