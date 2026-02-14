"""
FastAPI application for movie success predictions.
Provides REST endpoints for classification and regression using finalized models.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import logging

from src.models.predict_model import PredictionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Movie Success Predictor API",
    description="Predict movie success and revenue using ML models trained on multiple feature sets",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction pipeline
pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global pipeline
    try:
        pipeline = PredictionPipeline(
            model_dir='./models',
            data_dir='./data/processed'
        )
        available = pipeline.get_available_datasets()
        logger.info(f"✓ Models loaded for datasets: {available}")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


# ============================================================================
# Pydantic Models for Input Validation
# ============================================================================

class PredictionRequest(BaseModel):
    """Base model for prediction requests."""
    dataset: str = Field(
        default='all',
        description="Dataset config: 'metadata', 'meta_credits', 'meta_keywords', or 'all'",
        example="all"
    )
    
    @validator('dataset')
    def validate_dataset(cls, v):
        allowed = ['metadata', 'meta_credits', 'meta_keywords', 'all']
        if v not in allowed:
            raise ValueError(f"dataset must be one of {allowed}")
        return v


class MovieRecord(BaseModel):
    """Represents a single movie record for prediction."""
    # Add your actual feature fields here - dynamically validated
    # For now, accepting dynamic input via extra fields
    
    class Config:
        extra = "allow"  # Allow additional fields beyond defined ones


class ClassificationRequest(PredictionRequest):
    """Request for classification predictions."""
    movies: List[Dict[str, Any]] = Field(
        ...,
        description="List of movie records with all required features",
        example=[{
            "budget": 50000000,
            "popularity": 45.5,
            "runtime": 120,
            "release_year": 2020,
            "release_month": 5,
            "vote_average": 7.2,
            "vote_count": 1234,
            "is_collection": 0,
            "is_english": 1,
            "num_genres": 2,
            "num_production_companies": 3,
            "num_production_countries": 1,
            "num_spoken_languages": 1,
            "num_cast": 45,
            "num_crew": 120,
            "num_keywords": 15,
            "has_top_director": 1,
            "has_top_actor": 1,
            "has_top_lead_actor": 1,
            "primary_genre_Adventure": 1,
            "primary_genre_Animation": 0,
            "primary_genre_Comedy": 0,
            "primary_genre_Crime": 0,
            "primary_genre_Documentary": 0,
            "primary_genre_Drama": 0,
            "primary_genre_Family": 0,
            "primary_genre_Fantasy": 0,
            "primary_genre_Foreign": 0,
            "primary_genre_History": 0,
            "primary_genre_Horror": 0,
            "primary_genre_Music": 0,
            "primary_genre_Mystery": 0,
            "primary_genre_Romance": 0,
            "primary_genre_Science Fiction": 0,
            "primary_genre_TV Movie": 0,
            "primary_genre_Thriller": 0,
            "primary_genre_Unknown": 0,
            "primary_genre_War": 0,
            "primary_genre_Western": 0,
        }]
    )
    return_probabilities: bool = Field(
        default=True,
        description="Include probability scores in response"
    )
    
    @validator('movies')
    def validate_movies(cls, v):
        if not v or len(v) == 0:
            raise ValueError("'movies' list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 movies per request")
        return v


class RegressionRequest(PredictionRequest):
    """Request for regression predictions."""
    movies: List[Dict[str, Any]] = Field(
        ...,
        description="List of movie records with all required features",
        example=[{
            "budget": 50000000,
            "popularity": 45.5,
            "runtime": 120,
            "release_year": 2020,
            "release_month": 5,
            "vote_count": 1234,
            "is_collection": 0,
            "is_english": 1,
            "num_genres": 2,
            "num_production_companies": 3,
            "num_production_countries": 1,
            "num_spoken_languages": 1,
            "num_cast": 45,
            "num_crew": 120,
            "num_keywords": 15,
            "has_top_director": 1,
            "has_top_actor": 1,
            "has_top_lead_actor": 1,
            "roi": 2.5,
            "primary_genre_Adventure": 1,
            "primary_genre_Animation": 0,
            "primary_genre_Comedy": 0,
            "primary_genre_Crime": 0,
            "primary_genre_Documentary": 0,
            "primary_genre_Drama": 0,
            "primary_genre_Family": 0,
            "primary_genre_Fantasy": 0,
            "primary_genre_Foreign": 0,
            "primary_genre_History": 0,
            "primary_genre_Horror": 0,
            "primary_genre_Music": 0,
            "primary_genre_Mystery": 0,
            "primary_genre_Romance": 0,
            "primary_genre_Science Fiction": 0,
            "primary_genre_TV Movie": 0,
            "primary_genre_Thriller": 0,
            "primary_genre_Unknown": 0,
            "primary_genre_War": 0,
            "primary_genre_Western": 0,
        }]
    )
    
    @validator('movies')
    def validate_movies(cls, v):
        if not v or len(v) == 0:
            raise ValueError("'movies' list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 movies per request")
        return v


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Movie Success Predictor API",
        "version": "1.0.0",
        "description": "ML-powered predictions using NB05 model improvements (feature selection, XGBoost tuning, threshold optimization)",
        "endpoints": {
            "health": "GET /health",
            "info": "GET /info (feature requirements)",
            "models": "GET /models (NB05 model details & improvements)",
            "predict_classification": "POST /predict/classification",
            "predict_regression": "POST /predict/regression",
            "predict_combined": "POST /predict/combined",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        if pipeline is None:
            raise Exception("Pipeline not initialized")
        
        available_datasets = pipeline.get_available_datasets()
        
        return {
            "status": "healthy",
            "models_loaded": len(available_datasets),
            "available_datasets": available_datasets,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/info")
async def get_info():
    """Get information about available models."""
    try:
        if pipeline is None:
            raise Exception("Pipeline not initialized")
        
        available_datasets = pipeline.get_available_datasets()
        info = {
            "available_datasets": available_datasets,
            "feature_requirements": {}
        }
        
        for dataset in available_datasets:
            try:
                clf_feats = pipeline.get_required_features(dataset, 'classification')
                reg_feats = pipeline.get_required_features(dataset, 'regression')
                info["feature_requirements"][dataset] = {
                    "classification": {
                        "n_features": len(clf_feats),
                        "features": clf_feats[:10],  # Show first 10
                        "total_features": len(clf_feats),
                    },
                    "regression": {
                        "n_features": len(reg_feats),
                        "features": reg_feats[:10],  # Show first 10
                        "total_features": len(reg_feats),
                    }
                }
            except Exception as e:
                logger.warning(f"Could not load feature info for {dataset}: {e}")
        
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def get_models():
    """
    Get detailed information about NB05 model improvements.
    Returns model types, thresholds, feature counts, and improvements applied.
    """
    try:
        if pipeline is None:
            raise Exception("Pipeline not initialized")
        
        available_datasets = pipeline.get_available_datasets()
        models_info = {}
        
        for dataset in available_datasets:
            try:
                models_info[dataset] = pipeline.get_model_info(dataset)
            except Exception as e:
                logger.warning(f"Could not load model info for {dataset}: {e}")
        
        return {
            "service": "Movie Success Predictor API",
            "models_version": "NB05 (Model Improvements)",
            "available_models": len(models_info),
            "models": models_info,
            "nb05_improvements": {
                "feature_selection": "Auto-detected and dropped correlated features (r > 0.85), SelectKBest for significance",
                "model_architecture": "XGBoost classifiers and regressors with GridSearchCV hyperparameter tuning",
                "class_imbalance": "SMOTE for data augmentation + class weights + optimal threshold tuning",
                "threshold_optimization": "Per-dataset F1-score optimization to find best classification threshold",
                "datasets": "Applied to all 4 configurations: metadata, meta_credits, meta_keywords, all"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/classification")
async def predict_classification(request: ClassificationRequest):
    """
    Predict movie success (binary classification).
    
    Returns predictions for whether movies will succeed or fail.
    """
    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # Convert to DataFrame
        X = pd.DataFrame(request.movies)
        
        # Get predictions
        result = pipeline.predict_classification(
            X,
            dataset=request.dataset,
            return_probabilities=request.return_probabilities
        )
        
        return {
            "success": True,
            "dataset": result['dataset'],
            "dataset_label": result['dataset_label'],
            "n_movies": result['n_samples'],
            "predictions": result['predictions'],
            "probabilities": result.get('probabilities'),
            "threshold_used": result.get('threshold_used'),
            "predictions_thresholded": result.get('predictions_thresholded'),
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/regression")
async def predict_regression(request: RegressionRequest):
    """
    Predict movie revenue/rating (regression).
    
    Returns predicted continuous values for financial or rating outcomes.
    """
    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # Convert to DataFrame
        X = pd.DataFrame(request.movies)
        
        # Get predictions
        result = pipeline.predict_regression(X, dataset=request.dataset)
        
        return {
            "success": True,
            "dataset": result['dataset'],
            "dataset_label": result['dataset_label'],
            "n_movies": result['n_samples'],
            "predictions": result['predictions'],
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/combined")
async def predict_combined(request: ClassificationRequest):
    """
    Make both classification and regression predictions.
    
    Returns both success probability and revenue/rating estimates.
    """
    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # Convert to DataFrame
        X = pd.DataFrame(request.movies)
        
        # Get predictions
        result = pipeline.predict_combined(X, dataset=request.dataset)
        
        return {
            "success": True,
            "dataset": result['dataset'],
            "n_movies": X.shape[0],
            "classification": {
                "predictions": result['classification']['predictions'],
                "probabilities": result['classification'].get('probabilities'),
                "threshold_used": result['classification'].get('threshold_used'),
            },
            "regression": {
                "predictions": result['regression']['predictions'],
            }
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
