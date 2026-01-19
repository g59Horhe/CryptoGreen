#!/usr/bin/env python3
"""
Train ML Model Script

Create ML training dataset from benchmark results and train the selector model.

Features extracted:
    - file_size_log: Log10 of file size in bytes
    - file_type_encoded: Numeric encoding of file type
    - entropy: Shannon entropy of file content
    - entropy_quartile_25: 25th percentile of byte frequency distribution
    - entropy_quartile_75: 75th percentile of byte frequency distribution
    - has_aes_ni: Whether CPU has AES-NI hardware acceleration
    - cpu_cores: Number of CPU cores

Labels:
    - Optimal (lowest energy) algorithm for each configuration

Usage:
    python scripts/train_model.py [OPTIONS]

Options:
    --benchmark-results PATH  Path to benchmark results JSON
    --output-dir PATH         Output directory for dataset (default: data/ml_data)
    --model-output PATH       Where to save model (default: results/models/selector_model.pkl)
    --verbose                 Show detailed output
"""

import argparse
import glob
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Try to import SMOTE for handling class imbalance
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    logger.warning("imbalanced-learn not installed. Install with: pip install imbalanced-learn")
    logger.warning("Training will proceed without SMOTE (class imbalance may affect performance)")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cryptogreen.feature_extractor import FeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_benchmark(benchmark_dir: str = 'results/benchmarks/raw') -> str:
    """Find the latest benchmark results file."""
    pattern = Path(benchmark_dir) / 'benchmark_*.json'
    files = glob.glob(str(pattern))
    
    # Filter out incremental files
    files = [f for f in files if 'incremental' not in f]
    
    if not files:
        raise FileNotFoundError(
            f"No benchmark results found in {benchmark_dir}\n"
            "Run 'python scripts/run_benchmarks.py' first."
        )
    
    files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return files[0]


def load_benchmark_results(filepath: str) -> Tuple[List[Dict], Dict]:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        results = data.get('results', [])
        metadata = data.get('metadata', {})
    else:
        results = data
        metadata = {}
    
    return results, metadata


def find_optimal_algorithms(results: List[Dict]) -> Dict[str, str]:
    """
    Find the optimal (lowest energy) algorithm for each configuration.
    
    Args:
        results: List of benchmark results
        
    Returns:
        Dict mapping config_key -> optimal_algorithm
    """
    # Group results by configuration (file_type + file_size)
    config_energy = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        file_type = r['file_type']
        file_size = r['file_size']
        algorithm = r['algorithm']
        
        config_key = f"{file_type}_{file_size}"
        
        # Get energy measurements
        if 'measurements' in r:
            energies = [m['energy_joules'] for m in r['measurements']]
            median_energy = np.median(energies)
        elif 'statistics' in r:
            median_energy = r['statistics']['median_energy_j']
        else:
            continue
        
        config_energy[config_key][algorithm].append(median_energy)
    
    # Find optimal algorithm for each config
    optimal = {}
    for config_key, alg_energies in config_energy.items():
        # Calculate mean of median energies for each algorithm
        mean_energies = {alg: np.mean(e) for alg, e in alg_energies.items()}
        optimal[config_key] = min(mean_energies, key=mean_energies.get)
    
    return optimal


def extract_features_for_config(
    file_path: str,
    file_type: str,
    file_size: int,
    hardware: Dict
) -> Dict[str, Any]:
    """
    Extract features for a benchmark configuration.
    
    Args:
        file_path: Path to the test file
        file_type: File type extension
        file_size: File size in bytes
        hardware: Hardware info from benchmark
        
    Returns:
        Dict of feature values
    """
    # Try to extract from actual file if it exists
    path = Path(file_path)
    
    if path.exists():
        # Use FeatureExtractor for full feature extraction
        features = FeatureExtractor.extract_features(file_path)
    else:
        # File doesn't exist, compute features from metadata
        file_size_log = np.log10(file_size) if file_size > 0 else 0.0
        file_type_encoded = FeatureExtractor.encode_file_type(file_type)
        
        # Estimate entropy based on file type
        entropy_estimates = {
            'txt': 4.5,      # Plain text - moderate entropy
            'sql': 4.0,      # SQL - structured text
            'pdf': 6.5,      # PDF - mixed binary/text
            'jpg': 7.8,      # JPEG - compressed, high entropy
            'jpeg': 7.8,
            'png': 7.5,      # PNG - compressed
            'mp4': 7.9,      # Video - highly compressed
            'zip': 7.95,     # ZIP - compressed, near-random
        }
        
        entropy = entropy_estimates.get(file_type, 5.0)
        
        # Entropy quartiles estimated from typical file characteristics
        q25_estimates = {
            'txt': 0.001,
            'sql': 0.001,
            'pdf': 0.002,
            'jpg': 0.003,
            'png': 0.003,
            'mp4': 0.003,
            'zip': 0.003,
        }
        
        q75_estimates = {
            'txt': 0.01,
            'sql': 0.008,
            'pdf': 0.015,
            'jpg': 0.02,
            'png': 0.018,
            'mp4': 0.02,
            'zip': 0.02,
        }
        
        features = {
            'file_size_log': file_size_log,
            'file_type_encoded': file_type_encoded,
            'entropy': entropy,
            'entropy_quartile_25': q25_estimates.get(file_type, 0.002),
            'entropy_quartile_75': q75_estimates.get(file_type, 0.015),
        }
    
    # Add hardware features from benchmark metadata
    features['has_aes_ni'] = int(hardware.get('has_aes_ni', False))
    features['cpu_cores'] = hardware.get('cpu_cores', 8)
    
    return features


def create_ml_dataset(results: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create ML training dataset from benchmark results.
    
    Args:
        results: List of benchmark results
        
    Returns:
        Tuple of (features_df, labels_df)
    """
    # Find optimal algorithms
    optimal_algorithms = find_optimal_algorithms(results)
    
    # Group results by config to get unique configurations
    configs = {}
    for r in results:
        file_type = r['file_type']
        file_size = r['file_size']
        config_key = f"{file_type}_{file_size}"
        
        if config_key not in configs:
            configs[config_key] = {
                'file_path': r['file_path'],
                'file_type': file_type,
                'file_size': file_size,
                'hardware': r.get('hardware', {})
            }
    
    # Extract features for each configuration
    feature_records = []
    label_records = []
    
    logger.info(f"Extracting features for {len(configs)} configurations...")
    
    for config_key, config in configs.items():
        if config_key not in optimal_algorithms:
            logger.warning(f"No optimal algorithm found for {config_key}")
            continue
        
        # Extract features
        features = extract_features_for_config(
            config['file_path'],
            config['file_type'],
            config['file_size'],
            config['hardware']
        )
        
        # Select only the required features
        feature_row = {
            'config': config_key,
            'file_size_log': features.get('file_size_log', features.get('file_size_bytes', 0)),
            'file_type_encoded': features['file_type_encoded'],
            'entropy': features['entropy'],
            'entropy_quartile_25': features['entropy_quartile_25'],
            'entropy_quartile_75': features['entropy_quartile_75'],
            'has_aes_ni': features['has_aes_ni'],
            'cpu_cores': features['cpu_cores'],
        }
        
        # Ensure file_size_log is correctly computed
        if 'file_size_log' not in features and 'file_size_bytes' in features:
            feature_row['file_size_log'] = np.log10(features['file_size_bytes']) if features['file_size_bytes'] > 0 else 0
        
        feature_records.append(feature_row)
        
        label_records.append({
            'config': config_key,
            'optimal_algorithm': optimal_algorithms[config_key]
        })
    
    features_df = pd.DataFrame(feature_records)
    labels_df = pd.DataFrame(label_records)
    
    return features_df, labels_df


def print_dataset_summary(features_df: pd.DataFrame, labels_df: pd.DataFrame):
    """Print summary of the created dataset."""
    print()
    print("=" * 70)
    print("ML TRAINING DATASET SUMMARY")
    print("=" * 70)
    print()
    
    print("FEATURE DATASET SHAPE:")
    print(f"  Samples: {len(features_df)}")
    print(f"  Features: {len(features_df.columns) - 1}")  # Exclude 'config' column
    print()
    
    print("FEATURE COLUMNS:")
    feature_cols = [c for c in features_df.columns if c != 'config']
    for col in feature_cols:
        dtype = features_df[col].dtype
        print(f"  - {col}: {dtype}")
    print()
    
    print("FEATURE STATISTICS:")
    print("-" * 70)
    stats_df = features_df[feature_cols].describe()
    print(stats_df.round(4).to_string())
    print()
    
    print("CLASS DISTRIBUTION (Optimal Algorithm):")
    print("-" * 70)
    class_counts = labels_df['optimal_algorithm'].value_counts()
    total = len(labels_df)
    
    for alg, count in class_counts.items():
        pct = (count / total) * 100
        bar = "#" * int(pct / 2)
        print(f"  {alg:12s}: {count:3d} ({pct:5.1f}%) {bar}")
    print()
    
    print(f"  Total configurations: {total}")
    print(f"  Unique algorithms: {labels_df['optimal_algorithm'].nunique()}")
    print()
    
    print("=" * 70)


def save_dataset(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    output_dir: str
):
    """Save the dataset to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save features (excluding config column for ML training)
    features_file = output_path / 'features.csv'
    features_df.to_csv(features_file, index=False)
    logger.info(f"Saved features to: {features_file}")
    
    # Save labels
    labels_file = output_path / 'labels.csv'
    labels_df.to_csv(labels_file, index=False)
    logger.info(f"Saved labels to: {labels_file}")
    
    return features_file, labels_file


def load_dataset(data_dir: str = 'data/ml_data') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load features and labels from CSV files."""
    data_path = Path(data_dir)
    
    features_df = pd.read_csv(data_path / 'features.csv')
    labels_df = pd.read_csv(data_path / 'labels.csv')
    
    return features_df, labels_df


def calculate_top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 2) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        y_true: True labels (encoded)
        y_proba: Predicted probabilities for each class
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy score
    """
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    return correct / len(y_true)


def save_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: List[str],
    output_path: str = 'results/figures/confusion_matrix.png'
):
    """
    Save confusion matrix as a heatmap image.
    
    Args:
        conf_matrix: Confusion matrix array
        class_names: List of class names
        output_path: Path to save the figure
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix - Algorithm Selection', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Algorithm', fontsize=12)
    plt.ylabel('Actual Algorithm', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to: {output_file}")


def train_model(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    model_output: str = 'results/models/selector_model.pkl',
    n_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train RandomForest model with cross-validation.
    
    Args:
        features_df: Features DataFrame
        labels_df: Labels DataFrame
        model_output: Path to save trained model
        n_folds: Number of cross-validation folds
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        Dict containing training metrics and results
    """
    # Prepare features (exclude 'config' column)
    feature_cols = [c for c in features_df.columns if c != 'config']
    X = features_df[feature_cols].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels_df['optimal_algorithm'])
    class_names = label_encoder.classes_
    
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Classes: {list(class_names)}")
    
    # Warn if dataset is small (less than 100 samples)
    if X.shape[0] < 100:
        logger.warning("\n" + "="*70)
        logger.warning("WARNING: Small dataset detected!")
        logger.warning(f"Dataset size: {X.shape[0]} samples")
        logger.warning("Small datasets (<100 samples) produce preliminary results with high variance.")
        logger.warning("Expect CV standard deviation ±10-20%.")
        logger.warning("Recommendations:")
        logger.warning("  1. Collect more benchmark data (target: 500+ samples)")
        logger.warning("  2. Use ensemble methods and regularization")
        logger.warning("  3. Interpret results cautiously - consider rule-based fallback")
        logger.warning("="*70 + "\n")
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    logger.info("\nOriginal class distribution:")
    for cls_idx, count in zip(unique, counts):
        cls_name = class_names[cls_idx]
        pct = (count / len(y)) * 100
        logger.info(f"  {cls_name}: {count} samples ({pct:.1f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"\nTrain set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Check train set class distribution
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    logger.info("\nTrain set class distribution:")
    for cls_idx, count in zip(train_unique, train_counts):
        cls_name = class_names[cls_idx]
        pct = (count / len(y_train)) * 100
        logger.info(f"  {cls_name}: {count} samples ({pct:.1f}%)")
    
    # Apply SMOTE to handle class imbalance
    smote_applied = False
    if SMOTE_AVAILABLE:
        # Check if we have severe imbalance (minority class < 10% of majority)
        class_counts = dict(zip(train_unique, train_counts))
        min_count = min(train_counts)
        max_count = max(train_counts)
        imbalance_ratio = max_count / min_count
        
        logger.info(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 2.0 and min_count >= 2:
            # Determine k_neighbors (must be < minority class size)
            k_neighbors = min(2, min_count - 1) if min_count > 1 else 1
            
            logger.info(f"Applying SMOTE to balance classes (k_neighbors={k_neighbors})...")
            
            try:
                smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                
                # Update training data
                X_train = X_train_resampled
                y_train = y_train_resampled
                smote_applied = True
                
                # Log new distribution
                resampled_unique, resampled_counts = np.unique(y_train, return_counts=True)
                logger.info("\nAfter SMOTE - Train set class distribution:")
                for cls_idx, count in zip(resampled_unique, resampled_counts):
                    cls_name = class_names[cls_idx]
                    pct = (count / len(y_train)) * 100
                    logger.info(f"  {cls_name}: {count} samples ({pct:.1f}%)")
                
                logger.info(f"Train set after SMOTE: {len(X_train)} samples")
                
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}")
                logger.warning("Continuing with original (imbalanced) data")
                logger.warning("Consider collecting more data for minority classes")
        else:
            logger.info("Class imbalance is moderate - proceeding without SMOTE")
    else:
        logger.warning("\nWARNING: SMOTE not available. Class imbalance may affect performance.")
        logger.warning("Install imbalanced-learn: pip install imbalanced-learn")
    
    # Initialize model with specified hyperparameters
    # Hyperparameters optimized for small datasets (n<100):
    # - Reduced n_estimators (20 vs 100) to prevent overfitting
    # - Reduced max_depth (5 vs 10) - shallow trees for small data
    # - Added min_samples_leaf=2 to prevent extreme splits
    # - Added max_features='sqrt' for additional regularization
    # - Added bootstrap=True for ensemble diversity
    # 
    # Note: If SMOTE was applied, don't use class_weight='balanced' (data is already balanced)
    # If SMOTE was NOT applied, use class_weight='balanced' to handle imbalance
    class_weight = None if smote_applied else 'balanced'
    
    # Use smaller hyperparameters for small datasets
    n_samples = X.shape[0]
    if n_samples < 100:
        n_estimators = 20
        max_depth = 5
        logger.info("\nUsing reduced hyperparameters for small dataset:")
        logger.info(f"  n_estimators: {n_estimators} (standard: 100)")
        logger.info(f"  max_depth: {max_depth} (standard: 10)")
    else:
        n_estimators = 100
        max_depth = 10
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,      # Prevent overfitting on small datasets
        max_features='sqrt',     # Additional regularization
        bootstrap=True,          # Ensure bootstrap sampling for diversity
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1
    )
    
    logger.info(f"\nModel configuration:")
    logger.info(f"  class_weight: {class_weight}")
    logger.info(f"  n_estimators: {n_estimators}")
    logger.info(f"  max_depth: {max_depth}")
    logger.info(f"  min_samples_split: 5")
    logger.info(f"  min_samples_leaf: 2")
    logger.info(f"  max_features: sqrt")
    logger.info(f"  bootstrap: True")
    
    # 5-fold stratified cross-validation on training set
    # Note: Using 5-fold (not 10) for small datasets to ensure adequate samples per fold
    logger.info(f"\nPerforming {n_folds}-fold stratified cross-validation...")
    if n_samples < 100:
        logger.info(f"Note: 5-fold CV recommended for small datasets (n={n_samples})")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    logger.info(f"CV Scores: {cv_scores}")
    logger.info(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Warn about high variance in small datasets
    if cv_scores.std() > 0.10:
        logger.warning(f"\nHigh CV variance detected: ±{cv_scores.std() * 2:.1%}")
        logger.warning("This is expected for small datasets (<100 samples).")
        logger.warning("Model predictions may be uncertain. Consider rule-based fallback.")
    
    # Additional CV metrics using different scoring
    cv_precision = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision_weighted')
    cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall_weighted')
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
    
    logger.info(f"CV Precision (weighted): {cv_precision.mean():.4f} (+/- {cv_precision.std() * 2:.4f})")
    logger.info(f"CV Recall (weighted): {cv_recall.mean():.4f} (+/- {cv_recall.std() * 2:.4f})")
    logger.info(f"CV F1 (weighted): {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
    
    # Train final model on full training set
    logger.info("Training final model on full training set...")
    model.fit(X_train, y_train)
    
    # Predictions on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    top1_accuracy = accuracy_score(y_test, y_pred)
    top2_accuracy = calculate_top_k_accuracy(y_test, y_proba, k=2)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Save confusion matrix visualization
    cm_path = Path(model_output).parent.parent / 'figures' / 'confusion_matrix.png'
    save_confusion_matrix(conf_matrix, class_names, str(cm_path))
    
    # Log per-class performance
    logger.info("\nPer-class performance on test set:")
    for i, cls_name in enumerate(class_names):
        logger.info(f"  {cls_name}:")
        logger.info(f"    Precision: {precision[i]:.4f}")
        logger.info(f"    Recall:    {recall[i]:.4f}")
        logger.info(f"    F1-Score:  {f1[i]:.4f}")
        logger.info(f"    Support:   {support[i]}")
    
    # Feature importance
    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Save model, scaler, and label encoder
    output_path = Path(model_output).parent
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_file = Path(model_output)
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to: {model_file}")
    
    # Save scaler
    scaler_file = output_path / 'scaler.pkl'
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler to: {scaler_file}")
    
    # Save label encoder
    encoder_file = output_path / 'label_encoder.pkl'
    with open(encoder_file, 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Saved label encoder to: {encoder_file}")
    
    # Compile results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
            'n_classes': len(class_names),
            'class_names': list(class_names),
            'smote_applied': smote_applied,
            'class_imbalance_ratio': float(imbalance_ratio) if smote_applied or not SMOTE_AVAILABLE else None
        },
        'hyperparameters': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': random_state,
            'class_weight': class_weight,
            'n_jobs': -1
        },
        'cross_validation': {
            'n_folds': n_folds,
            'accuracy_scores': cv_scores.tolist(),
            'accuracy_mean': float(cv_scores.mean()),
            'accuracy_std': float(cv_scores.std()),
            'precision_mean': float(cv_precision.mean()),
            'recall_mean': float(cv_recall.mean()),
            'f1_mean': float(cv_f1.mean())
        },
        'test_metrics': {
            'top1_accuracy': float(top1_accuracy),
            'top2_accuracy': float(top2_accuracy),
            'per_class': {
                class_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(len(class_names))
            }
        },
        'confusion_matrix': conf_matrix.tolist(),
        'feature_importance': {k: float(v) for k, v in sorted_importance},
        'files_saved': {
            'model': str(model_file),
            'scaler': str(scaler_file),
            'label_encoder': str(encoder_file),
            'confusion_matrix': str(cm_path)
        }
    }
    
    # Save training report
    report_file = output_path / 'training_report.json'
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved training report to: {report_file}")
    
    return results


def print_training_report(results: Dict[str, Any]):
    """Print formatted training report."""
    print()
    print("=" * 70)
    print("ML MODEL TRAINING REPORT")
    print("=" * 70)
    print()
    
    # Dataset info
    ds = results['dataset']
    print("DATASET:")
    print(f"  Total samples: {ds['total_samples']}")
    print(f"  Train samples: {ds['train_samples']}")
    print(f"  Test samples: {ds['test_samples']}")
    print(f"  Features: {ds['n_features']}")
    print(f"  Classes: {ds['n_classes']} - {ds['class_names']}")
    if ds.get('smote_applied'):
        print(f"  SMOTE applied: Yes (handled {ds.get('class_imbalance_ratio', 'N/A'):.2f}:1 imbalance)")
    else:
        print(f"  SMOTE applied: No")
    print()
    
    # Hyperparameters
    hp = results['hyperparameters']
    print("HYPERPARAMETERS:")
    print(f"  n_estimators: {hp['n_estimators']}")
    print(f"  max_depth: {hp['max_depth']}")
    print(f"  min_samples_split: {hp['min_samples_split']}")
    print(f"  class_weight: {hp['class_weight']}")
    print(f"  random_state: {hp['random_state']}")
    print()
    
    # Cross-validation
    cv = results['cross_validation']
    print(f"CROSS-VALIDATION ({cv['n_folds']}-FOLD STRATIFIED):")
    print("-" * 70)
    for i, score in enumerate(cv['accuracy_scores'], 1):
        bar = "#" * int(score * 50)
        print(f"  Fold {i}: {score:.4f} {bar}")
    print(f"  Accuracy:  {cv['accuracy_mean']:.4f} (+/- {cv['accuracy_std'] * 2:.4f})")
    print(f"  Precision: {cv['precision_mean']:.4f} (weighted)")
    print(f"  Recall:    {cv['recall_mean']:.4f} (weighted)")
    print(f"  F1-Score:  {cv['f1_mean']:.4f} (weighted)")
    print()
    
    # Test set metrics
    tm = results['test_metrics']
    print("TEST SET METRICS:")
    print("-" * 70)
    print(f"  Top-1 Accuracy: {tm['top1_accuracy']:.4f} ({tm['top1_accuracy']*100:.1f}%)")
    print(f"  Top-2 Accuracy: {tm['top2_accuracy']:.4f} ({tm['top2_accuracy']*100:.1f}%)")
    print()
    
    # Per-class metrics
    print("PER-CLASS METRICS:")
    print("-" * 70)
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("  " + "-" * 52)
    for cls, metrics in tm['per_class'].items():
        print(f"  {cls:<12} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1_score']:>10.4f} {metrics['support']:>10}")
    print()
    
    # Confusion matrix
    print("CONFUSION MATRIX:")
    print("-" * 70)
    class_names = results['dataset']['class_names']
    cm = results['confusion_matrix']
    
    # Header
    print(f"  {'Actual \\ Pred':<12}", end='')
    for cls in class_names:
        print(f"{cls:>10}", end='')
    print()
    print("  " + "-" * (12 + 10 * len(class_names)))
    
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:<12}", end='')
        for val in row:
            print(f"{val:>10}", end='')
        print()
    print()
    
    # Feature importance
    print("FEATURE IMPORTANCE (Top 5):")
    print("-" * 70)
    fi = results['feature_importance']
    for i, (feature, importance) in enumerate(fi.items()):
        if i >= 5:
            break
        bar = "#" * int(importance * 50)
        print(f"  {i+1}. {feature:<25} {importance:.4f} {bar}")
    print()
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Create ML training dataset and train model'
    )
    parser.add_argument(
        '--benchmark-results',
        type=str,
        default=None,
        help='Path to benchmark results JSON'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/ml_data',
        help='Directory for dataset files'
    )
    parser.add_argument(
        '--model-output',
        type=str,
        default='results/models/selector_model.pkl',
        help='Where to save trained model'
    )
    parser.add_argument(
        '--skip-dataset',
        action='store_true',
        help='Skip dataset creation, use existing CSV files'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only train model (implies --skip-dataset)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.train_only:
        args.skip_dataset = True
    
    # Create or load dataset
    if not args.skip_dataset:
        # Find benchmark results
        if args.benchmark_results:
            benchmark_file = args.benchmark_results
        else:
            benchmark_file = find_latest_benchmark()
            logger.info(f"Auto-detected benchmark file: {benchmark_file}")
        
        # Load results
        logger.info(f"Loading benchmark results from: {benchmark_file}")
        results, metadata = load_benchmark_results(benchmark_file)
        logger.info(f"Loaded {len(results)} benchmark results")
        
        # Create dataset
        logger.info("Creating ML training dataset...")
        features_df, labels_df = create_ml_dataset(results)
        
        # Save dataset
        save_dataset(features_df, labels_df, args.data_dir)
        
        # Print dataset summary
        print_dataset_summary(features_df, labels_df)
    else:
        # Load existing dataset
        logger.info(f"Loading existing dataset from: {args.data_dir}")
        features_df, labels_df = load_dataset(args.data_dir)
        logger.info(f"Loaded {len(features_df)} samples")
    
    # Train model
    logger.info("=" * 70)
    logger.info("Starting ML Model Training")
    logger.info("=" * 70)
    
    training_results = train_model(
        features_df,
        labels_df,
        model_output=args.model_output,
        n_folds=5,
        test_size=0.2,
        random_state=42
    )
    
    # Print training report
    print_training_report(training_results)
    
    # Summary
    cv_mean = training_results['cross_validation']['accuracy_mean']
    top1_acc = training_results['test_metrics']['top1_accuracy']
    top2_acc = training_results['test_metrics']['top2_accuracy']
    
    print()
    logger.info("Training complete!")
    logger.info(f"  CV Accuracy: {cv_mean:.1%}")
    logger.info(f"  Test Top-1 Accuracy: {top1_acc:.1%}")
    logger.info(f"  Test Top-2 Accuracy: {top2_acc:.1%}")
    logger.info(f"  Model saved to: {args.model_output}")


if __name__ == '__main__':
    main()
