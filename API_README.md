# Movie Success Predictor API

Complete REST API for predicting movie success using machine learning models trained on 4 different feature configurations.

## Quick Start

### Installation

```bash
# Install dependencies
pip install fastapi uvicorn pandas numpy joblib scikit-learn xgboost lightgbm

# Or use requirements file (if available)
pip install -r requirements.txt
```

### Running the API

```bash
# From project root directory
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Model Improvements (NB05)

This API uses models trained with significant improvements from Notebook 05:

| Improvement | Technique | Impact |
|---|---|---|
| **Feature Selection** | Dropped correlated features (r > 0.85), SelectKBest with p < 0.05 | Reduced noise, faster inference |
| **Advanced Models** | XGBoost & LightGBM with GridSearchCV tuning | Higher accuracy & R² scores |
| **Class Imbalance** | SMOTE + class weights + threshold optimization | Better minority class detection |
| **Threshold Tuning** | Per-dataset F1-score optimization | Optimal classification decisions |

### Getting Model Info

To see detailed information about the NB05 improvements applied to each model:

```http
GET /models
```

This returns the model types, feature counts, optimal thresholds, and all improvements applied per dataset.

## API Endpoints

### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 4,
  "available_datasets": ["metadata", "meta_credits", "meta_keywords", "all"]
}
```

### 2. Get Model Information

```http
GET /info
```

Returns the list of required features for each dataset and task.

### 2.1 Get NB05 Model Details

```http
GET /models
```

**Response:**
```json
{
  "service": "Movie Success Predictor API",
  "models_version": "NB05 (Model Improvements)",
  "available_models": 4,
  "models": {
    "all": {
      "dataset": "all",
      "dataset_label": "All Combined",
      "classification_model": "XGBClassifier",
      "regression_model": "XGBRegressor",
      "classification_threshold": 0.51,
      "n_classification_features": 40,
      "n_regression_features": 40,
      "improvements": [
        "Feature selection (dropped high-correlation features)",
        "XGBoost tuning with GridSearchCV",
        "Class imbalance handling (SMOTE + class weights)",
        "Optimal threshold tuning (per-dataset F1 optimization)"
      ]
    }
  },
  "nb05_improvements": {
    "feature_selection": "Auto-detected and dropped correlated features (r > 0.85), SelectKBest for significance",
    "model_architecture": "XGBoost classifiers and regressors with GridSearchCV hyperparameter tuning",
    "class_imbalance": "SMOTE for data augmentation + class weights + optimal threshold tuning",
    "threshold_optimization": "Per-dataset F1-score optimization to find best classification threshold",
    "datasets": "Applied to all 4 configurations: metadata, meta_credits, meta_keywords, all"
  }
}
```

**Returns:**
- Model types (XGBClassifier, XGBRegressor) for each dataset
- Optimal classification threshold per dataset (tuned for best F1 score)
- Feature counts after selection
- List of NB05 improvements applied

### 3. Classification Prediction

```http
POST /predict/classification
```

**Request Body:**
```json
{
  "dataset": "all",
  "return_probabilities": true,
  "movies": [
    {
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
      "primary_genre_Western": 0
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "dataset": "all",
  "dataset_label": "All Combined",
  "n_samples": 1,
  "model_type": "XGBClassifier",
  "predictions": [1],
  "probabilities": [0.7256],
  "threshold_used": 0.51,
  "predictions_default_threshold": [1]
}
```

**Response Fields:**
- `predictions` (array): **Primary predictions using NB05 optimal threshold** (tuned for best F1-score per dataset)
- `probabilities` (array): Predicted probabilities of success (only if `return_probabilities=true`)
- `threshold_used` (float): The NB05-optimized threshold used for predictions (not always 0.5)
- `predictions_default_threshold` (array): Predictions at default 0.5 threshold (for comparison)
- `model_type` (string): Type of model used (XGBClassifier, LGBMClassifier, etc.)

**Key Notes:**
- The primary `predictions` field uses the **NB05-optimized threshold** per dataset (not 0.5)
- Each dataset has its own optimal threshold tuned to maximize F1-score
- Use `predictions` as the final classification decision
- Use `predictions_default_threshold` only if you need standard 0.5-threshold predictions for comparison

**Parameters:**
- `dataset` (string): One of `"metadata"`, `"meta_credits"`, `"meta_keywords"`, `"all"` (default: `"all"`)
- `return_probabilities` (boolean): Include probability scores (default: `true`)
- `movies` (array): List of movie records with features

### 4. Regression Prediction

```http
POST /predict/regression
```

**Request Body:**
```json
{
  "dataset": "all",
  "movies": [
    {
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
      "primary_genre_Western": 0
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "dataset": "all",
  "dataset_label": "All Combined",
  "n_samples": 1,
  "model_type": "XGBRegressor",
  "predictions": [185420000.53]
}
```

**Response Fields:**
- `predictions` (array): Revenue or rating predictions using NB05-tuned XGBoost regressor
- `model_type` (string): Type of model used (XGBRegressor, LGBMRegressor, etc.)

**Notes:**
- Uses NB05-tuned regression models with feature selection
- For "all" dataset: predicts ROI (return on investment)
- For other datasets: predicts revenue or revenue-related metrics

### 5. Combined Predictions

```http
POST /predict/combined
```

**Request Body:** Same as classification

**Response:**
```json
{
  "success": true,
  "dataset": "all",
  "n_samples": 1,
  "classification": {
    "predictions": [1],
    "probabilities": [0.7256],
    "threshold_used": 0.51
  },
  "regression": {
    "predictions": [185420000.53],
    "model_type": "XGBRegressor"
  }
}
```

## Dataset Configurations

The API supports 4 different feature set configurations:

| Config | Description | Feature Count | Key Features |
|--------|-------------|---|----------|
| `metadata` | Movie metadata only | 34 features | budget, popularity, runtime, ratings, release info, genres |
| `meta_credits` | Metadata + cast/crew | 37 features | + num_cast, num_crew, has_top_director/actor/lead_actor |
| `meta_keywords` | Metadata + keywords | 36 features | + num_keywords, plot tags, themes |
| `all` | Complete feature set | 40 features | All above combined for maximum accuracy |

**Feature Details - "all" Configuration (40 features):**
- **Continuous**: budget, popularity, runtime, vote_average, vote_count, roi (regression only)
- **Binary**: is_collection, is_english, has_top_director, has_top_actor, has_top_lead_actor
- **Counts**: num_genres, num_production_companies, num_production_countries, num_spoken_languages, num_cast, num_crew, num_keywords
- **Time**: release_year, release_month
- **Genres** (20 binary features): Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, Foreign, History, Horror, Music, Mystery, Romance, Science Fiction, TV Movie, Thriller, Unknown, War, Western

Use different configs to balance accuracy vs. data availability.

## Error Handling

### Common Errors

**400 Bad Request** - Missing or invalid features:
```json
{
  "detail": "Validation error: Missing required features: ['budget', 'revenue']"
}
```

**400 Bad Request** - Invalid dataset:
```json
{
  "detail": "Validation error: dataset must be one of ['metadata', 'meta_credits', 'meta_keywords', 'all']"
}
```

**503 Service Unavailable** - Models not loaded:
```json
{
  "detail": "Service unhealthy: Pipeline not initialized"
}
```

### Request Validation

The API validates:
- ✓ All required features present
- ✓ Feature values are numeric
- ✓ Batch size ≤ 1000 records
- ✓ Valid dataset configuration
- ✓ Non-empty input lists

## Python Client Example

```python
import requests
import pandas as pd

# Initialize client
API_URL = "http://localhost:8000"

# Example movie data
movies_data = [
    {
        "budget": 50000000,
        "popularity": 45.5,
        "runtime": 120,
        "release_year": 2020,
        # ... add all required features
    }
]

# Classification prediction
response = requests.post(
    f"{API_URL}/predict/classification",
    json={
        "dataset": "all",
        "return_probabilities": True,
        "movies": movies_data
    }
)

result = response.json()
print(f"Predictions: {result['predictions']}")
print(f"Probabilities: {result['probabilities']}")

# Regression prediction
response = requests.post(
    f"{API_URL}/predict/regression",
    json={
        "dataset": "all",
        "movies": movies_data
    }
)

result = response.json()
print(f"Revenue estimate: {result['predictions']}")
```

## Direct Python Usage (without API)

```python
from src.models.predict_model import PredictionPipeline
import pandas as pd

# Load pipeline
pipeline = PredictionPipeline(
    model_dir='./models',
    data_dir='./data/processed'
)

# Prepare data
X = pd.DataFrame(movies_data)

# Make predictions
clf_result = pipeline.predict_classification(X, dataset='all')
reg_result = pipeline.predict_regression(X, dataset='all')

print(f"Success predictions: {clf_result['predictions']}")
print(f"Revenue estimates: {reg_result['predictions']}")
```

## Architecture

```
src/
├── models/
│   ├── predict_model.py      # PredictionPipeline class (core logic)
│   └── validation.py         # Input validation utilities
├── api.py                    # FastAPI application
└── example_usage.py          # Example script
```

### Key Components

1. **PredictionPipeline** (`predict_model.py`)
   - Loads all trained models
   - Manages feature validation
   - Makes predictions for classification and regression

2. **FastAPI App** (`api.py`)
   - Provides REST endpoints
   - Handles request/response serialization
   - Centralized error handling
   - CORS enabled for cross-origin requests

3. **Validation** (`validation.py`)
   - Input data validation
   - Type coercion
   - Range checking
   - Feature requirement verification

## Model Details

- **Classification**: XGBoost with optimized threshold
- **Regression**: XGBoost
- **Training Data**: 4 different feature configurations
- **Cross-validation**: 5-fold with F1/R² scoring

## Performance Notes

- Classification threshold: Optimized per dataset
- Probability calibration: Model-based probabilities
- Prediction time: ~10-50ms per batch (10-1000 records)

## Troubleshooting

### Models not loading
- Verify `models/` directory exists with `.pkl` files
- Check file permissions
- Ensure correct model naming: `best_clf_model_{dataset}.pkl`

### Feature mismatch errors
- Verify input DataFrame columns match `features_*.csv` files
- Check data types are numeric
- No NaN/null values allowed

### API won't start
- Ensure port 8000 is not in use
- Check Python dependencies installed
- Verify working directory is project root

## Performance Considerations

- **Batch Processing**: Process up to 1000 records per request
- **Caching**: Models are loaded once on startup
- **Parallelization**: Prediction is single-threaded but fast
- **Memory**: ~500MB for all 4 model sets

## Future Enhancements

- [ ] Model versioning and A/B testing
- [ ] Custom threshold configuration per dataset
- [ ] Batch file upload (CSV/Parquet)
- [ ] Explanation/SHAP values for predictions
- [ ] Rate limiting and authentication
- [ ] Caching for duplicate requests
- [ ] Asynchronous batch processing
