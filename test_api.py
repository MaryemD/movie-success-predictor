"""
Comprehensive test and demo script for the Movie Success Predictor API.
Tests both the PredictionPipeline and the FastAPI application.
"""

import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_pipeline_initialization():
    """Test that the prediction pipeline initializes correctly."""
    logger.info("=" * 60)
    logger.info("Testing PredictionPipeline Initialization")
    logger.info("=" * 60)
    
    try:
        from src.models.predict_model import PredictionPipeline
        
        pipeline = PredictionPipeline(
            model_dir='./models',
            data_dir='./data/processed'
        )
        
        available = pipeline.get_available_datasets()
        logger.info(f"✓ Pipeline initialized")
        logger.info(f"✓ Available datasets: {available}")
        
        if not available:
            logger.warning("⚠ No models loaded!")
            return False
        
        return pipeline
    
    except Exception as e:
        logger.error(f"✗ Pipeline initialization failed: {e}")
        return None


def test_feature_requirements(pipeline):
    """Test retrieving feature requirements."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Feature Requirements")
    logger.info("=" * 60)
    
    try:
        for dataset in pipeline.get_available_datasets():
            clf_feats = pipeline.get_required_features(dataset, 'classification')
            reg_feats = pipeline.get_required_features(dataset, 'regression')
            
            logger.info(f"\n{dataset}:")
            logger.info(f"  Classification: {len(clf_feats)} features")
            logger.info(f"  Regression: {len(reg_feats)} features")
            
            if clf_feats:
                logger.info(f"  Sample features: {clf_feats[:3]} ...")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Feature retrieval failed: {e}")
        return False


def create_dummy_data(pipeline, n_samples=3):
    """
    Create realistic dummy data for testing with actual feature structure.
    """
    logger.info(f"\nCreating test data ({n_samples} samples)...")
    
    try:
        # Create realistic dummy data for 'all' dataset
        data_records = [
            {
                'budget': 50000000,
                'popularity': 45.5,
                'release_year': 2020,
                'release_month': 5,
                'runtime': 120,
                'vote_average': 7.2,
                'vote_count': 1234,
                'is_collection': 0,
                'is_english': 1,
                'num_genres': 2,
                'num_production_companies': 3,
                'num_production_countries': 1,
                'num_spoken_languages': 1,
                'num_cast': 45,
                'num_crew': 120,
                'num_keywords': 15,
                'has_top_director': 1,
                'has_top_actor': 1,
                'has_top_lead_actor': 1,
                'primary_genre_Adventure': 1,
                'primary_genre_Animation': 0,
                'primary_genre_Comedy': 0,
                'primary_genre_Crime': 0,
                'primary_genre_Documentary': 0,
                'primary_genre_Drama': 0,
                'primary_genre_Family': 0,
                'primary_genre_Fantasy': 0,
                'primary_genre_Foreign': 0,
                'primary_genre_History': 0,
                'primary_genre_Horror': 0,
                'primary_genre_Music': 0,
                'primary_genre_Mystery': 0,
                'primary_genre_Romance': 0,
                'primary_genre_Science Fiction': 0,
                'primary_genre_TV Movie': 0,
                'primary_genre_Thriller': 0,
                'primary_genre_Unknown': 0,
                'primary_genre_War': 0,
                'primary_genre_Western': 0,
            },
            {
                'budget': 150000000,
                'popularity': 78.2,
                'release_year': 2021,
                'release_month': 7,
                'runtime': 145,
                'vote_average': 8.1,
                'vote_count': 5678,
                'is_collection': 1,
                'is_english': 1,
                'num_genres': 3,
                'num_production_companies': 5,
                'num_production_countries': 2,
                'num_spoken_languages': 2,
                'num_cast': 68,
                'num_crew': 150,
                'num_keywords': 22,
                'has_top_director': 1,
                'has_top_actor': 1,
                'has_top_lead_actor': 1,
                'primary_genre_Adventure': 1,
                'primary_genre_Animation': 0,
                'primary_genre_Comedy': 0,
                'primary_genre_Crime': 0,
                'primary_genre_Documentary': 0,
                'primary_genre_Drama': 1,
                'primary_genre_Family': 0,
                'primary_genre_Fantasy': 0,
                'primary_genre_Foreign': 0,
                'primary_genre_History': 0,
                'primary_genre_Horror': 0,
                'primary_genre_Music': 0,
                'primary_genre_Mystery': 0,
                'primary_genre_Romance': 0,
                'primary_genre_Science Fiction': 0,
                'primary_genre_TV Movie': 0,
                'primary_genre_Thriller': 0,
                'primary_genre_Unknown': 0,
                'primary_genre_War': 0,
                'primary_genre_Western': 0,
            },
            {
                'budget': 30000000,
                'popularity': 32.1,
                'release_year': 2019,
                'release_month': 10,
                'runtime': 100,
                'vote_average': 6.5,
                'vote_count': 789,
                'is_collection': 0,
                'is_english': 0,
                'num_genres': 1,
                'num_production_companies': 2,
                'num_production_countries': 1,
                'num_spoken_languages': 1,
                'num_cast': 25,
                'num_crew': 90,
                'num_keywords': 8,
                'has_top_director': 0,
                'has_top_actor': 0,
                'has_top_lead_actor': 0,
                'primary_genre_Adventure': 0,
                'primary_genre_Animation': 0,
                'primary_genre_Comedy': 1,
                'primary_genre_Crime': 0,
                'primary_genre_Documentary': 0,
                'primary_genre_Drama': 0,
                'primary_genre_Family': 0,
                'primary_genre_Fantasy': 0,
                'primary_genre_Foreign': 0,
                'primary_genre_History': 0,
                'primary_genre_Horror': 0,
                'primary_genre_Music': 0,
                'primary_genre_Mystery': 0,
                'primary_genre_Romance': 0,
                'primary_genre_Science Fiction': 0,
                'primary_genre_TV Movie': 0,
                'primary_genre_Thriller': 0,
                'primary_genre_Unknown': 0,
                'primary_genre_War': 0,
                'primary_genre_Western': 0,
            }
        ]
        
        # Create DataFrame
        X = pd.DataFrame(data_records[:n_samples])
        logger.info(f"✓ Created test DataFrame: {X.shape}")
        logger.info(f"✓ Features: {list(X.columns)}")
        
        return X
    
    except Exception as e:
        logger.error(f"✗ Dummy data creation failed: {e}")
        return None


def test_classification_predictions(pipeline, X):
    """Test classification predictions."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Classification Predictions")
    logger.info("=" * 60)
    
    datasets_to_test = pipeline.get_available_datasets()
    
    for dataset in datasets_to_test:
        try:
            result = pipeline.predict_classification(
                X,
                dataset=dataset,
                return_probabilities=True
            )
            
            logger.info(f"\n{dataset}:")
            logger.info(f"  Predictions: {result['predictions']}")
            if 'probabilities' in result:
                logger.info(f"  Probabilities: {result['probabilities']}")
                logger.info(f"  Threshold: {result.get('threshold_used', 'N/A')}")
            
        except Exception as e:
            logger.error(f"✗ Classification failed for {dataset}: {e}")


def test_regression_predictions(pipeline, X):
    """Test regression predictions."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Regression Predictions")
    logger.info("=" * 60)
    
    datasets_to_test = pipeline.get_available_datasets()
    
    for dataset in datasets_to_test:
        try:
            result = pipeline.predict_regression(X, dataset=dataset)
            
            logger.info(f"\n{dataset}:")
            logger.info(f"  Predictions: {result['predictions']}")
            
        except Exception as e:
            logger.error(f"✗ Regression failed for {dataset}: {e}")


def test_combined_predictions(pipeline, X):
    """Test combined predictions."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Combined Predictions")
    logger.info("=" * 60)
    
    try:
        result = pipeline.predict_combined(X, dataset='all')
        
        logger.info(f"Dataset: {result['dataset']}")
        logger.info(f"Classification: {result['classification']}")
        logger.info(f"Regression: {result['regression']}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Combined prediction failed: {e}")
        return False


def test_validation():
    """Test input validation module."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Input Validation")
    logger.info("=" * 60)
    
    try:
        from src.models.validation import InputValidator, DataTypeValidator
        
        # Test type coercion
        val = DataTypeValidator.coerce_numeric("123.45", "test_field")
        assert val == 123.45
        logger.info("✓ Type coercion works")
        
        # Test range validation
        val = DataTypeValidator.validate_range(50, "test", min_val=0, max_val=100)
        logger.info("✓ Range validation works")
        
        # Test batch validation
        try:
            InputValidator.validate_batch([], ["feature1"], max_records=1000)
        except ValueError as e:
            logger.info("✓ Batch validation correctly rejects empty list")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Validation tests failed: {e}")
        return False


def test_api_models():
    """Test FastAPI model definitions."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing FastAPI Models")
    logger.info("=" * 60)
    
    try:
        from src.api import ClassificationRequest, RegressionRequest
        
        # Test creating request objects
        request_data = {
            "dataset": "all",
            "movies": [{"feature1": 100, "feature2": 50}]
        }
        
        req = ClassificationRequest(**request_data, return_probabilities=True)
        logger.info(f"✓ ClassificationRequest created: {req.dataset}, {len(req.movies)} movies")
        
        req = RegressionRequest(**request_data)
        logger.info(f"✓ RegressionRequest created: {req.dataset}, {len(req.movies)} movies")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ API model tests failed: {e}")
        return False


def print_summary(results):
    """Print test summary."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


def main():
    """Run all tests."""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 10 + "Movie Success Predictor - API Tests" + " " * 13 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    
    results = {}
    
    # Test 1: Pipeline initialization
    pipeline = test_pipeline_initialization()
    results["Pipeline Initialization"] = pipeline is not None
    
    if pipeline is None:
        logger.error("\n✗ Cannot continue without pipeline. Exiting.")
        print_summary(results)
        return False
    
    # Test 2: Feature requirements
    results["Feature Requirements"] = test_feature_requirements(pipeline)
    
    # Test 3: Validation module
    results["Validation Module"] = test_validation()
    
    # Test 4: API models
    results["FastAPI Models"] = test_api_models()
    
    # Create dummy data for remaining tests
    X = create_dummy_data(pipeline, n_samples=2)
    
    if X is not None:
        # Test 5: Classification
        test_classification_predictions(pipeline, X)
        results["Classification Predictions"] = True
        
        # Test 6: Regression
        test_regression_predictions(pipeline, X)
        results["Regression Predictions"] = True
        
        # Test 7: Combined
        results["Combined Predictions"] = test_combined_predictions(pipeline, X)
    else:
        results["Classification Predictions"] = False
        results["Regression Predictions"] = False
        results["Combined Predictions"] = False
    
    # Print summary
    all_passed = print_summary(results)
    
    if all_passed:
        logger.info("\n✓ All tests passed! API is ready to use.")
        logger.info("\nStart the API with:")
        logger.info("  uvicorn src.api:app --reload --host 0.0.0.0 --port 8000")
        logger.info("\nThen visit:")
        logger.info("  http://localhost:8000/docs (Swagger UI)")
        logger.info("  http://localhost:8000/redoc (ReDoc)")
    else:
        logger.warning("\n⚠ Some tests failed. Check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
