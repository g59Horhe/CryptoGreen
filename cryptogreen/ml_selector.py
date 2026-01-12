"""
Machine Learning Algorithm Selector Module

This module provides ML-based algorithm selection using a Random Forest
classifier trained on benchmark data.

Example:
    >>> from cryptogreen.ml_selector import MLSelector
    >>> selector = MLSelector('results/models/selector_model.pkl')
    >>> result = selector.select_algorithm('document.pdf')
    >>> print(f"Recommended: {result['algorithm']} ({result['confidence']:.2%})")
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from cryptogreen.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class MLSelector:
    """Select algorithm using trained Random Forest model.
    
    This selector uses a machine learning model trained on benchmark data
    to predict the most energy-efficient algorithm for given file characteristics.
    
    Attributes:
        model: Trained scikit-learn model.
        feature_names: List of feature names expected by the model.
        label_encoder: Mapping from algorithm names to indices.
        
    Example:
        >>> selector = MLSelector()
        >>> result = selector.select_algorithm('data.bin')
        >>> print(f"Algorithm: {result['algorithm']}")
        >>> print(f"Confidence: {result['confidence']:.2%}")
    """
    
    # Default feature names (must match training)
    DEFAULT_FEATURE_NAMES = [
        'file_size_log',
        'file_type_encoded',
        'entropy',
        'entropy_quartile_25',
        'entropy_quartile_75',
        'has_aes_ni',
        'cpu_cores',
    ]
    
    # Algorithm label encoding
    ALGORITHM_LABELS = {
        0: 'AES-128',
        1: 'AES-256',
        2: 'ChaCha20',
        3: 'RSA-2048',
        4: 'RSA-4096',
        5: 'ECC-256',
    }
    
    LABEL_TO_INDEX = {v: k for k, v in ALGORITHM_LABELS.items()}
    
    def __init__(self, model_path: str = 'results/models/selector_model.pkl'):
        """Initialize ML selector.
        
        Args:
            model_path: Path to trained model file (.pkl).
                       If model doesn't exist, selector will operate in
                       "untrained" mode with fallback predictions.
        """
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = self.DEFAULT_FEATURE_NAMES
        self.scaler = None
        self.is_trained = False
        
        # Try to load existing model
        if self.model_path.exists():
            self._load_model(model_path)
        else:
            logger.warning(
                f"Model not found at {model_path}. "
                "ML selector will use fallback predictions. "
                "Train a model with scripts/train_model.py"
            )
    
    def _load_model(self, model_path: str) -> None:
        """Load trained model from pickle file.
        
        Args:
            model_path: Path to model file.
        """
        try:
            import joblib
            
            model_data = joblib.load(model_path)
            
            # Handle different save formats
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.feature_names = model_data.get('feature_names', self.DEFAULT_FEATURE_NAMES)
            else:
                self.model = model_data
            
            self.is_trained = self.model is not None
            logger.info(f"Loaded ML model from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.is_trained = False
    
    def select_algorithm(self, file_path: str) -> dict:
        """Select optimal algorithm using ML model.
        
        Args:
            file_path: Path to file to encrypt.
            
        Returns:
            Dict containing:
                - algorithm: Predicted algorithm name
                - confidence: Prediction confidence (0.0-1.0)
                - probabilities: Dict of {algorithm: probability}
                - features: Features extracted from file
                - feature_importance: Feature importances from model
                - alternatives: Top-3 algorithm recommendations
                
        Example:
            >>> result = selector.select_algorithm('video.mp4')
            >>> print(f"Algorithm: {result['algorithm']}")
            >>> print(f"Confidence: {result['confidence']:.1%}")
        """
        # Extract features
        features = FeatureExtractor.extract_features(file_path)
        
        # Prepare features for model
        X = self._prepare_features(features)
        
        if not self.is_trained:
            return self._fallback_prediction(features)
        
        # Get prediction and probabilities
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Get algorithm name
        algorithm = self.ALGORITHM_LABELS.get(prediction, 'AES-128')
        confidence = float(max(probabilities))
        
        # Build probability dict
        prob_dict = {}
        for i, prob in enumerate(probabilities):
            alg_name = self.ALGORITHM_LABELS.get(i, f'Unknown-{i}')
            prob_dict[alg_name] = float(prob)
        
        # Get top-3 alternatives
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        alternatives = [alg for alg, _ in sorted_probs[:3]]
        
        # Get feature importance
        feature_importance = self._get_feature_importance()
        
        result = {
            'algorithm': algorithm,
            'confidence': confidence,
            'probabilities': prob_dict,
            'features': features,
            'feature_importance': feature_importance,
            'alternatives': alternatives,
        }
        
        logger.debug(f"ML prediction: {algorithm} ({confidence:.1%} confidence)")
        
        return result
    
    def _prepare_features(self, features: dict) -> np.ndarray:
        """Convert feature dict to model input array.
        
        Args:
            features: Feature dict from FeatureExtractor.
            
        Returns:
            NumPy array of features in correct order.
        """
        X = np.array([[
            features['file_size_log'],
            float(features['file_type_encoded']),
            features['entropy'],
            features['entropy_quartile_25'],
            features['entropy_quartile_75'],
            float(features['has_aes_ni']),
            float(features['cpu_cores']),
        ]])
        
        # Apply scaler if available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
    
    def _fallback_prediction(self, features: dict) -> dict:
        """Provide fallback prediction when model is not trained.
        
        Uses simple heuristics based on features.
        
        Args:
            features: Extracted features.
            
        Returns:
            Prediction dict with low confidence.
        """
        has_aes_ni = features.get('has_aes_ni', False)
        entropy = features.get('entropy', 5.0)
        file_size = features.get('file_size_bytes', 1024)
        
        # Simple heuristic
        if has_aes_ni and file_size < 100 * 1024:
            algorithm = 'AES-128'
        elif has_aes_ni:
            algorithm = 'AES-256'
        elif entropy > 7.5:
            algorithm = 'ChaCha20'
        else:
            algorithm = 'ChaCha20'
        
        # Low confidence since no model
        probabilities = {alg: 0.1 for alg in self.ALGORITHM_LABELS.values()}
        probabilities[algorithm] = 0.5
        
        return {
            'algorithm': algorithm,
            'confidence': 0.3,  # Low confidence for fallback
            'probabilities': probabilities,
            'features': features,
            'feature_importance': {},
            'alternatives': [algorithm],
            'is_fallback': True,
        }
    
    def _get_feature_importance(self) -> dict:
        """Get feature importance from trained model.
        
        Returns:
            Dict of {feature_name: importance}.
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        
        return {
            name: float(imp) 
            for name, imp in zip(self.feature_names, importances)
        }
    
    def train_model(
        self,
        features_csv: str = 'data/ml_data/features.csv',
        labels_csv: str = 'data/ml_data/labels.csv',
        output_path: str = 'results/models/selector_model.pkl',
        test_split: float = 0.2
    ) -> dict:
        """Train Random Forest model on benchmark data.
        
        Args:
            features_csv: Path to features CSV file.
            labels_csv: Path to labels CSV file.
            output_path: Where to save trained model.
            test_split: Fraction of data for testing.
            
        Returns:
            Dict containing training metrics:
                - accuracy: Top-1 accuracy
                - top2_accuracy: Top-2 accuracy
                - cv_scores: Cross-validation scores
                - confusion_matrix: Confusion matrix
                - feature_importance: Feature importances
                - classification_report: Per-class metrics
                
        Example:
            >>> selector = MLSelector()
            >>> metrics = selector.train_model()
            >>> print(f"Accuracy: {metrics['accuracy']:.1%}")
        """
        import pandas as pd
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        
        logger.info("Training ML model...")
        
        # Load data
        features_path = Path(features_csv)
        labels_path = Path(labels_csv)
        
        if not features_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Training data not found. Expected:\n"
                f"  - {features_csv}\n"
                f"  - {labels_csv}\n"
                "Run benchmarks and prepare training data first."
            )
        
        X_df = pd.read_csv(features_csv)
        y_df = pd.read_csv(labels_csv)
        
        # Prepare features
        X = X_df[self.feature_names].values
        
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_df['optimal_algorithm'])
        
        logger.info(f"Training data: {len(X)} samples")
        logger.info(f"Features: {self.feature_names}")
        logger.info(f"Classes: {label_encoder.classes_}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        # Top-1 accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Top-2 accuracy
        top2_correct = 0
        for i, proba in enumerate(y_proba):
            top2_indices = np.argsort(proba)[-2:]
            if y_test[i] in top2_indices:
                top2_correct += 1
        top2_accuracy = top2_correct / len(y_test)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=label_encoder.classes_,
            output_dict=True
        )
        
        # Feature importance
        importance = dict(zip(self.feature_names, model.feature_importances_))
        
        logger.info(f"Top-1 Accuracy: {accuracy:.1%}")
        logger.info(f"Top-2 Accuracy: {top2_accuracy:.1%}")
        logger.info(f"CV Scores: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
        
        # Save model
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': self.feature_names,
            'label_encoder': label_encoder,
            'classes': list(label_encoder.classes_),
        }
        
        joblib.dump(model_data, output_path)
        logger.info(f"Model saved to: {output_path}")
        
        # Update instance
        self.model = model
        self.scaler = scaler
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'top2_accuracy': top2_accuracy,
            'cv_scores': list(cv_scores),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': cm.tolist(),
            'feature_importance': importance,
            'classification_report': report,
            'classes': list(label_encoder.classes_),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
        }
    
    def train_from_benchmark_results(
        self,
        benchmark_json: str,
        output_path: str = 'results/models/selector_model.pkl'
    ) -> dict:
        """Train model directly from benchmark results.
        
        Args:
            benchmark_json: Path to benchmark results JSON.
            output_path: Where to save trained model.
            
        Returns:
            Training metrics dict.
        """
        import json
        import pandas as pd
        
        logger.info(f"Loading benchmark results from {benchmark_json}")
        
        with open(benchmark_json, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', data)
        
        # Prepare training data
        features_list = []
        labels_list = []
        
        # Group by file to find optimal algorithm
        file_results = {}
        for result in results:
            file_key = (result['file_name'], result['file_size'])
            if file_key not in file_results:
                file_results[file_key] = []
            file_results[file_key].append(result)
        
        for file_key, file_data in file_results.items():
            # Find algorithm with lowest median energy
            best_result = min(file_data, key=lambda x: x['statistics']['median_energy_j'])
            
            # Only use symmetric encryption algorithms for training
            if best_result['algorithm'] not in ['AES-128', 'AES-256', 'ChaCha20']:
                continue
            
            optimal_algorithm = best_result['algorithm']
            
            # Extract features from any result for this file
            features = {
                'file_size_log': np.log10(best_result['file_size']) if best_result['file_size'] > 0 else 0,
                'file_type_encoded': FeatureExtractor.encode_file_type(best_result['file_type']),
            }
            
            # We need entropy - extract from file if possible, or estimate
            file_path = best_result.get('file_path', '')
            if file_path and Path(file_path).exists():
                try:
                    entropy, q25, q75 = FeatureExtractor.calculate_entropy(file_path)
                except Exception:
                    entropy, q25, q75 = 6.0, 0.001, 0.01  # Default estimates
            else:
                # Estimate based on file type
                file_type = best_result['file_type']
                if file_type in ['txt', 'sql']:
                    entropy, q25, q75 = 4.5, 0.005, 0.02
                elif file_type in ['jpg', 'png', 'mp4', 'zip']:
                    entropy, q25, q75 = 7.8, 0.001, 0.005
                else:
                    entropy, q25, q75 = 6.0, 0.003, 0.01
            
            features['entropy'] = entropy
            features['entropy_quartile_25'] = q25
            features['entropy_quartile_75'] = q75
            
            # Hardware features from benchmark metadata
            hw = best_result.get('hardware', {})
            features['has_aes_ni'] = float(hw.get('has_aes_ni', False))
            features['cpu_cores'] = float(hw.get('cpu_cores', os.cpu_count() or 4))
            
            features_list.append(features)
            labels_list.append(optimal_algorithm)
        
        if len(features_list) < 10:
            raise ValueError(
                f"Insufficient training data: {len(features_list)} samples. "
                "Need at least 10 samples for training."
            )
        
        # Create DataFrames
        features_df = pd.DataFrame(features_list)
        labels_df = pd.DataFrame({'optimal_algorithm': labels_list})
        
        # Save training data
        data_dir = Path('data/ml_data')
        data_dir.mkdir(parents=True, exist_ok=True)
        
        features_df.to_csv(data_dir / 'features.csv', index=False)
        labels_df.to_csv(data_dir / 'labels.csv', index=False)
        
        logger.info(f"Saved training data: {len(features_list)} samples")
        
        # Train model
        return self.train_model(
            features_csv=str(data_dir / 'features.csv'),
            labels_csv=str(data_dir / 'labels.csv'),
            output_path=output_path
        )
    
    def evaluate(self, test_data: list[dict]) -> dict:
        """Evaluate model on test data.
        
        Args:
            test_data: List of dicts with 'file_path' and 'optimal_algorithm'.
            
        Returns:
            Evaluation metrics dict.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        correct = 0
        top2_correct = 0
        predictions = []
        
        for item in test_data:
            result = self.select_algorithm(item['file_path'])
            prediction = result['algorithm']
            actual = item['optimal_algorithm']
            
            if prediction == actual:
                correct += 1
            
            # Check top-2
            if actual in result['alternatives'][:2]:
                top2_correct += 1
            
            predictions.append({
                'file_path': item['file_path'],
                'predicted': prediction,
                'actual': actual,
                'correct': prediction == actual,
                'confidence': result['confidence'],
            })
        
        total = len(test_data)
        
        return {
            'accuracy': correct / total if total > 0 else 0,
            'top2_accuracy': top2_correct / total if total > 0 else 0,
            'total_samples': total,
            'correct_predictions': correct,
            'predictions': predictions,
        }
