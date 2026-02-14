"""
Prediction pipeline for movie success models.
Handles loading trained models and making predictions across 4 dataset configurations.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from pathlib import Path


class PredictionPipeline:
    """
    Loads and manages trained models for all 4 dataset configurations.
    Supports classification (success/failure) and regression (revenue/rating).
    
    Features:
    - Uses NB05 model improvements: feature selection, XGBoost/LightGBM, SMOTE, 
      class imbalance handling, GridSearchCV tuning, and optimal threshold optimization
    - Automatic feature subsetting to match NB05 feature-selected sets
    - Per-dataset optimal classification thresholds (tuned for best F1 score)
    - Graceful handling of extra/missing features
    """

    DATASETS = ['metadata', 'meta_credits', 'meta_keywords', 'all']
    DATASET_LABELS = {
        'metadata': 'Metadata Only',
        'meta_credits': 'Meta + Credits',
        'meta_keywords': 'Meta + Keywords',
        'all': 'All Combined',
    }

    def __init__(self, model_dir: str = '../models', data_dir: str = '../data/processed'):
        """
        Initialize pipeline by loading all trained models and feature lists.
        
        Args:
            model_dir: Directory containing trained model .pkl files
            data_dir: Directory containing feature lists per dataset
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.models = {}
        self.thresholds = {}
        self.features = {}
        
        self._load_all_models()

    def _load_all_models(self):
        """Load all models and metadata for each dataset configuration."""
        for ds_key in self.DATASETS:
            try:
                # Load classification and regression models
                clf_path = self.model_dir / f'best_clf_model_{ds_key}.pkl'
                reg_path = self.model_dir / f'best_reg_model_{ds_key}.pkl'
                threshold_path = self.model_dir / f'clf_threshold_{ds_key}.pkl'
                
                if not clf_path.exists() or not reg_path.exists():
                    print(f"Warning: Models for '{ds_key}' not found.")
                    continue
                
                self.models[ds_key] = {
                    'clf': joblib.load(clf_path),
                    'reg': joblib.load(reg_path),
                }
                
                # Load optimal threshold
                if threshold_path.exists():
                    self.thresholds[ds_key] = joblib.load(threshold_path)['threshold']
                else:
                    self.thresholds[ds_key] = 0.5
                
                # Load feature lists
                clf_features_path = self.data_dir / ds_key / 'features_clf.csv'
                reg_features_path = self.data_dir / ds_key / 'features_reg.csv'
                
                if clf_features_path.exists() and reg_features_path.exists():
                    clf_feats = pd.read_csv(clf_features_path)['feature'].tolist()
                    reg_feats = pd.read_csv(reg_features_path)['feature'].tolist()
                    self.features[ds_key] = {'clf': clf_feats, 'reg': reg_feats}
                else:
                    print(f"Warning: Feature lists for '{ds_key}' not found.")
                    
                print(f"✓ Loaded {self.DATASET_LABELS[ds_key]} models")
                
            except Exception as e:
                print(f"Error loading models for '{ds_key}': {e}")

    def predict_classification(
        self, 
        X: pd.DataFrame, 
        dataset: str = 'all',
        return_probabilities: bool = True
    ) -> Dict:
        """
        Predict movie success (binary classification).
        Uses optimal threshold tuning from NB05 model improvements.
        
        Args:
            X: Input features DataFrame
            dataset: Which dataset config to use ('metadata', 'meta_credits', 'meta_keywords', 'all')
            return_probabilities: Whether to return probability scores
            
        Returns:
            Dictionary with predictions and metadata
        """
        if dataset not in self.DATASETS:
            raise ValueError(f"Invalid dataset. Choose from: {self.DATASETS}")
        
        if dataset not in self.models:
            raise ValueError(f"Models for dataset '{dataset}' not loaded.")
        
        # Validate and subset features (uses NB05 feature-selected features)
        clf_features = self.features[dataset]['clf']
        X_subset = self._validate_and_subset_features(X, clf_features, 'classification')
        
        # Make predictions using NB05-tuned XGBoost model
        model = self.models[dataset]['clf']
        
        result = {
            'dataset': dataset,
            'dataset_label': self.DATASET_LABELS[dataset],
            'n_samples': len(X_subset),
            'model_type': type(model).__name__,
        }
        
        # Get probability predictions (used with NB05 optimal threshold)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_subset)[:, 1]
            
            # Use NB05 optimal threshold for primary predictions
            optimal_threshold = self.thresholds[dataset]
            y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
            
            result['predictions'] = y_pred_optimal.tolist()  # Primary: optimal threshold
            result['threshold_used'] = optimal_threshold
            
            if return_probabilities:
                result['probabilities'] = y_proba.tolist()
                # Also include default threshold (0.5) predictions for comparison
                y_pred_default = model.predict(X_subset)
                result['predictions_default_threshold'] = y_pred_default.tolist()
        else:
            # Fallback for models without predict_proba
            y_pred = model.predict(X_subset)
            result['predictions'] = y_pred.tolist()
            result['threshold_used'] = 'model_default'
        
        return result

    def predict_regression(
        self, 
        X: pd.DataFrame, 
        dataset: str = 'all'
    ) -> Dict:
        """
        Predict movie revenue or rating (regression).
        Uses NB05-tuned XGBoost regression models with feature selection.
        
        Args:
            X: Input features DataFrame
            dataset: Which dataset config to use
            
        Returns:
            Dictionary with predictions and metadata
        """
        if dataset not in self.DATASETS:
            raise ValueError(f"Invalid dataset. Choose from: {self.DATASETS}")
        
        if dataset not in self.models:
            raise ValueError(f"Models for dataset '{dataset}' not loaded.")
        
        # Validate and subset features (uses NB05 feature-selected features)
        reg_features = self.features[dataset]['reg']
        X_subset = self._validate_and_subset_features(X, reg_features, 'regression')
        
        # Make predictions using NB05-tuned XGBoost model
        model = self.models[dataset]['reg']
        y_pred = model.predict(X_subset)
        
        result = {
            'dataset': dataset,
            'dataset_label': self.DATASET_LABELS[dataset],
            'predictions': y_pred.tolist(),
            'n_samples': len(X_subset),
            'model_type': type(model).__name__,
        }
        
        return result

    def predict_combined(
        self, 
        X: pd.DataFrame, 
        dataset: str = 'all'
    ) -> Dict:
        """
        Make both classification and regression predictions.
        
        Args:
            X: Input features DataFrame
            dataset: Which dataset config to use
            
        Returns:
            Combined results from both models
        """
        clf_result = self.predict_classification(X, dataset)
        reg_result = self.predict_regression(X, dataset)
        
        return {
            'dataset': dataset,
            'classification': clf_result,
            'regression': reg_result,
        }

    def _validate_and_subset_features(
        self, 
        X: pd.DataFrame, 
        required_features: List[str],
        task: str
    ) -> pd.DataFrame:
        """
        Validate input features and subset to those required by the model.
        
        Args:
            X: Input DataFrame
            required_features: List of feature names the model expects
            task: 'classification' or 'regression' for error messages
            
        Returns:
            Subset DataFrame with required features in correct order
            
        Raises:
            ValueError: If required features are missing
        """
        missing_features = [f for f in required_features if f not in X.columns]
        if missing_features:
            raise ValueError(
                f"Missing {len(missing_features)} features for {task}: {missing_features}"
            )
        
        # Handle extra features gracefully (just subset to required)
        extra_features = [f for f in X.columns if f not in required_features]
        if extra_features:
            print(f"Note: Dropping {len(extra_features)} extra features not used by model.")
        
        return X[required_features]

    def get_available_datasets(self) -> List[str]:
        """Get list of datasets with loaded models."""
        return list(self.models.keys())

    def get_model_info(self, dataset: str) -> Dict:
        """
        Get detailed information about the NB05-tuned models for a dataset.
        
        Args:
            dataset: Dataset configuration ('metadata', 'meta_credits', 'meta_keywords', 'all')
            
        Returns:
            Dictionary with model types, thresholds, and feature counts
        """
        if dataset not in self.models:
            raise ValueError(f"Dataset '{dataset}' not available.")
        
        n_clf_features = len(self.features[dataset]['clf'])
        n_reg_features = len(self.features[dataset]['reg'])
        
        return {
            'dataset': dataset,
            'dataset_label': self.DATASET_LABELS[dataset],
            'classification_model': type(self.models[dataset]['clf']).__name__,
            'regression_model': type(self.models[dataset]['reg']).__name__,
            'classification_threshold': self.thresholds[dataset],
            'n_classification_features': n_clf_features,
            'n_regression_features': n_reg_features,
            'improvements': [
                'Feature selection (dropped high-correlation features)',
                'XGBoost tuning with GridSearchCV',
                'Class imbalance handling (SMOTE + class weights)',
                'Optimal threshold tuning (per-dataset F1 optimization)',
            ]
        }

    def get_required_features(self, dataset: str, task: str = 'classification') -> List[str]:
        """
        Get the list of required features for a specific model.
        
        Args:
            dataset: Dataset configuration
            task: 'classification' or 'regression'
            
        Returns:
            List of feature names
        """
        if dataset not in self.features:
            raise ValueError(f"Dataset '{dataset}' not available.")
        
        task_key = 'clf' if task == 'classification' else 'reg'
        return self.features[dataset][task_key]
