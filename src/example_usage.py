"""
Example script demonstrating how to use the PredictionPipeline directly.

Note: This pipeline uses NB05-tuned models with:
- Feature selection (dropped correlated features)
- XGBoost with GridSearchCV optimization
- Class imbalance handling (SMOTE + class weights)
- Per-dataset optimal thresholds (tuned for F1-score)
"""

import pandas as pd
from src.models.predict_model import PredictionPipeline


def main():
    """Example usage of the prediction pipeline (uses NB05 improvements)."""
    
    # Initialize pipeline (loads NB05-tuned models)
    print("Initializing prediction pipeline with NB05-tuned models...")
    pipeline = PredictionPipeline(
        model_dir='./models',
        data_dir='./data/processed'
    )
    
    print(f"Available datasets: {pipeline.get_available_datasets()}\n")
    
    # Example movie data with all required features for 'all' dataset
    example_movies = [
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
        }
    ]
    
    X = pd.DataFrame(example_movies)
    
    # Show NB05 model improvements for 'all' dataset
    print("\n" + "=" * 60)
    print("NB05 Model Improvements (for 'all' dataset)")
    print("=" * 60)
    try:
        model_info = pipeline.get_model_info('all')
        print(f"Classification Model: {model_info['classification_model']}")
        print(f"Regression Model: {model_info['regression_model']}")
        print(f"Optimal Classification Threshold: {model_info['classification_threshold']:.3f}")
        print(f"Classification Features: {model_info['n_classification_features']}")
        print(f"Regression Features: {model_info['n_regression_features']}")
        print(f"\nImprovements Applied:")
        for i, imp in enumerate(model_info['improvements'], 1):
            print(f"  {i}. {imp}")
    except Exception as e:
        print(f"Could not retrieve model info: {e}\n")
    
    # Make classification predictions (success/failure)
    print("\n" + "=" * 60)
    print("Classification Predictions (Success/Failure)")
    print("=" * 60)
    try:
        clf_result = pipeline.predict_classification(
            X, 
            dataset='all',
            return_probabilities=True
        )
        
        print(f"Dataset: {clf_result['dataset_label']}")
        print(f"Model: {clf_result.get('model_type', 'XGBClassifier')}")
        print(f"Predictions (using NB05 optimal threshold={clf_result.get('threshold_used', 0.5):.3f}): {clf_result['predictions']}")
        if 'probabilities' in clf_result:
            print(f"Success probabilities: {clf_result['probabilities']}")
            if 'predictions_default_threshold' in clf_result:
                print(f"Predictions (default threshold=0.5): {clf_result['predictions_default_threshold']}")
        print()
    except Exception as e:
        print(f"Classification error: {e}\n")
    
    # Make regression predictions (revenue/rating estimate)
    print("=" * 60)
    print("Regression Predictions (Revenue/Rating)")
    print("=" * 60)
    try:
        reg_result = pipeline.predict_regression(X, dataset='all')
        
        print(f"Dataset: {reg_result['dataset_label']}")
        print(f"Model: {reg_result.get('model_type', 'XGBRegressor')}")
        print(f"Predictions: {reg_result['predictions']}")
        print()
    except Exception as e:
        print(f"Regression error: {e}\n")
    
    # Combined predictions
    print("=" * 60)
    print("Combined Predictions (using NB05 models)")
    print("=" * 60)
    try:
        combined_result = pipeline.predict_combined(X, dataset='all')
        
        print(f"Dataset: {combined_result['dataset']}")
        print(f"Classification Predictions (optimal threshold): {combined_result['classification']['predictions']}")
        print(f"Classification Probabilities: {combined_result['classification'].get('probabilities', 'N/A')}")
        print(f"Regression Predictions: {combined_result['regression']['predictions']}")
    except Exception as e:
        print(f"Combined prediction error: {e}")


if __name__ == "__main__":
    main()
