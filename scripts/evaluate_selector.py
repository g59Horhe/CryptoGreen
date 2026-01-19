#!/usr/bin/env python
"""
Hybrid Selector Evaluation Script

Evaluates the performance of rule-based, ML, and hybrid selectors
by comparing their recommendations against actual optimal algorithms
from benchmark data.

Usage:
    python scripts/evaluate_selector.py
    python scripts/evaluate_selector.py --verbose
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cryptogreen.rule_based_selector import RuleBasedSelector
from cryptogreen.ml_selector import MLSelector
from cryptogreen.hybrid_selector import HybridSelector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Size mapping from config format to file naming
SIZE_CONFIG_TO_FILE = {
    '64': '64B',
    '1024': '1KB',
    '10240': '10KB',
    '102400': '100KB',
    '1048576': '1MB',
    '10485760': '10MB',
    '104857600': '100MB',
}


def load_evaluation_data(data_dir: str = 'data/ml_data') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features and labels from CSV files.
    
    Args:
        data_dir: Directory containing features.csv and labels.csv
        
    Returns:
        Tuple of (features_df, labels_df)
    """
    features_path = Path(data_dir) / 'features.csv'
    labels_path = Path(data_dir) / 'labels.csv'
    
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    
    return features_df, labels_df


def config_to_file_path(config: str, test_files_dir: str = 'data/test_files') -> str:
    """Convert config name to actual file path.
    
    Args:
        config: Config name like 'jpg_64' or 'txt_1048576'
        test_files_dir: Base directory for test files
        
    Returns:
        Path to the test file
    """
    parts = config.rsplit('_', 1)
    file_type = parts[0]
    size_bytes = parts[1]
    
    # Get file size name
    size_name = SIZE_CONFIG_TO_FILE.get(size_bytes, f'{size_bytes}B')
    
    # Build file path
    file_name = f"{file_type}_{size_name}.{file_type}"
    file_path = Path(test_files_dir) / file_type / file_name
    
    return str(file_path)


def evaluate_selectors(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    test_files_dir: str = 'data/test_files',
    model_path: str = 'results/models/selector_model.pkl',
    verbose: bool = False
) -> pd.DataFrame:
    """Evaluate all selectors on the dataset.
    
    Args:
        features_df: Features DataFrame
        labels_df: Labels DataFrame
        test_files_dir: Directory containing test files
        model_path: Path to trained ML model
        verbose: Print detailed output
        
    Returns:
        DataFrame with evaluation results for each config
    """
    # Initialize selectors
    logger.info("Initializing selectors...")
    rule_selector = RuleBasedSelector()
    ml_selector = MLSelector(model_path)
    hybrid_selector = HybridSelector(model_path)
    
    if not ml_selector.is_trained:
        logger.warning("ML model not trained! ML selector will use fallback predictions.")
    
    results = []
    
    logger.info(f"Evaluating {len(features_df)} configurations...")
    
    for idx, (_, feature_row) in enumerate(features_df.iterrows()):
        config = feature_row['config']
        actual_optimal = labels_df[labels_df['config'] == config]['optimal_algorithm'].values[0]
        
        # Get file path
        file_path = config_to_file_path(config, test_files_dir)
        
        if not Path(file_path).exists():
            logger.warning(f"File not found: {file_path}, skipping config {config}")
            continue
        
        if verbose:
            logger.info(f"Evaluating: {config} -> {file_path}")
        
        # Get rule-based recommendation
        rule_result = rule_selector.select_algorithm(file_path)
        rule_algorithm = rule_result['algorithm']
        rule_confidence = rule_result['confidence']
        
        # Get ML recommendation
        ml_result = ml_selector.select_algorithm(file_path)
        ml_algorithm = ml_result['algorithm']
        ml_confidence = ml_result['confidence']
        ml_is_fallback = ml_result.get('is_fallback', False)
        
        # Get hybrid recommendation
        hybrid_result = hybrid_selector.select_algorithm(file_path)
        hybrid_algorithm = hybrid_result['algorithm']
        hybrid_confidence = hybrid_result['confidence']
        hybrid_method = hybrid_result['method']
        
        # Check correctness
        rule_correct = rule_algorithm == actual_optimal
        ml_correct = ml_algorithm == actual_optimal
        hybrid_correct = hybrid_algorithm == actual_optimal
        
        # Record results
        results.append({
            'config': config,
            'file_type': config.rsplit('_', 1)[0],
            'size_bytes': int(config.rsplit('_', 1)[1]),
            'actual_optimal': actual_optimal,
            
            # Rule-based results
            'rule_algorithm': rule_algorithm,
            'rule_confidence': rule_confidence,
            'rule_correct': rule_correct,
            
            # ML results
            'ml_algorithm': ml_algorithm,
            'ml_confidence': ml_confidence,
            'ml_is_fallback': ml_is_fallback,
            'ml_correct': ml_correct,
            
            # Hybrid results
            'hybrid_algorithm': hybrid_algorithm,
            'hybrid_confidence': hybrid_confidence,
            'hybrid_method': hybrid_method,
            'hybrid_correct': hybrid_correct,
            
            # Agreement
            'rule_ml_agree': rule_algorithm == ml_algorithm,
            
            # Analysis
            'hybrid_outperforms_rule': hybrid_correct and not rule_correct,
            'hybrid_outperforms_ml': hybrid_correct and not ml_correct,
            'hybrid_worse_than_rule': not hybrid_correct and rule_correct,
            'hybrid_worse_than_ml': not hybrid_correct and ml_correct,
        })
        
        if verbose and idx % 10 == 0:
            logger.info(f"  Progress: {idx + 1}/{len(features_df)}")
    
    return pd.DataFrame(results)


def calculate_metrics(results_df: pd.DataFrame) -> dict:
    """Calculate evaluation metrics from results.
    
    Args:
        results_df: DataFrame with evaluation results
        
    Returns:
        Dictionary of metrics
    """
    n = len(results_df)
    
    # Overall accuracy
    rule_accuracy = results_df['rule_correct'].sum() / n
    ml_accuracy = results_df['ml_correct'].sum() / n
    hybrid_accuracy = results_df['hybrid_correct'].sum() / n
    
    # Method selection distribution
    method_counts = results_df['hybrid_method'].value_counts()
    method_pcts = method_counts / n
    
    # Confidence distribution
    confidence_counts = results_df['hybrid_confidence'].value_counts()
    confidence_pcts = confidence_counts / n
    
    # Agreement analysis
    agreement_rate = results_df['rule_ml_agree'].sum() / n
    
    # When hybrid outperforms
    hybrid_outperforms_rule = results_df['hybrid_outperforms_rule'].sum()
    hybrid_outperforms_ml = results_df['hybrid_outperforms_ml'].sum()
    hybrid_worse_than_rule = results_df['hybrid_worse_than_rule'].sum()
    hybrid_worse_than_ml = results_df['hybrid_worse_than_ml'].sum()
    
    # Per-method accuracy
    method_accuracy = {}
    for method in results_df['hybrid_method'].unique():
        method_df = results_df[results_df['hybrid_method'] == method]
        if len(method_df) > 0:
            method_accuracy[method] = method_df['hybrid_correct'].sum() / len(method_df)
    
    # Per-size accuracy
    size_accuracy = {
        'rule': {},
        'ml': {},
        'hybrid': {}
    }
    for size in sorted(results_df['size_bytes'].unique()):
        size_df = results_df[results_df['size_bytes'] == size]
        n_size = len(size_df)
        if n_size > 0:
            size_accuracy['rule'][size] = size_df['rule_correct'].sum() / n_size
            size_accuracy['ml'][size] = size_df['ml_correct'].sum() / n_size
            size_accuracy['hybrid'][size] = size_df['hybrid_correct'].sum() / n_size
    
    # Per-file-type accuracy
    type_accuracy = {
        'rule': {},
        'ml': {},
        'hybrid': {}
    }
    for file_type in sorted(results_df['file_type'].unique()):
        type_df = results_df[results_df['file_type'] == file_type]
        n_type = len(type_df)
        if n_type > 0:
            type_accuracy['rule'][file_type] = type_df['rule_correct'].sum() / n_type
            type_accuracy['ml'][file_type] = type_df['ml_correct'].sum() / n_type
            type_accuracy['hybrid'][file_type] = type_df['hybrid_correct'].sum() / n_type
    
    return {
        'total_samples': n,
        'overall_accuracy': {
            'rule_based': rule_accuracy,
            'ml': ml_accuracy,
            'hybrid': hybrid_accuracy,
        },
        'method_selection': {
            'counts': method_counts.to_dict(),
            'percentages': method_pcts.to_dict(),
        },
        'confidence_distribution': {
            'counts': confidence_counts.to_dict(),
            'percentages': confidence_pcts.to_dict(),
        },
        'rule_ml_agreement': agreement_rate,
        'hybrid_comparison': {
            'outperforms_rule': hybrid_outperforms_rule,
            'outperforms_ml': hybrid_outperforms_ml,
            'worse_than_rule': hybrid_worse_than_rule,
            'worse_than_ml': hybrid_worse_than_ml,
        },
        'method_accuracy': method_accuracy,
        'size_accuracy': size_accuracy,
        'type_accuracy': type_accuracy,
    }


def print_evaluation_report(metrics: dict, results_df: pd.DataFrame):
    """Print formatted evaluation report.
    
    Args:
        metrics: Metrics dictionary from calculate_metrics
        results_df: Full results DataFrame
    """
    print()
    print("=" * 70)
    print("HYBRID SELECTOR EVALUATION REPORT")
    print("=" * 70)
    
    # Overall accuracy comparison
    print("\n" + "=" * 70)
    print("ACCURACY COMPARISON")
    print("=" * 70)
    
    acc = metrics['overall_accuracy']
    print(f"\n{'Selector':<20} {'Accuracy':>12} {'Correct':>10} {'Total':>8}")
    print("-" * 50)
    
    n = metrics['total_samples']
    rule_correct = int(acc['rule_based'] * n)
    ml_correct = int(acc['ml'] * n)
    hybrid_correct = int(acc['hybrid'] * n)
    
    print(f"{'Rule-Based':<20} {acc['rule_based']:>11.1%} {rule_correct:>10} {n:>8}")
    print(f"{'ML (RandomForest)':<20} {acc['ml']:>11.1%} {ml_correct:>10} {n:>8}")
    print(f"{'Hybrid':<20} {acc['hybrid']:>11.1%} {hybrid_correct:>10} {n:>8}")
    
    # Best selector
    best = max(acc.items(), key=lambda x: x[1])
    print(f"\n✓ Best: {best[0].replace('_', ' ').title()} ({best[1]:.1%})")
    
    # Improvement
    if acc['hybrid'] >= acc['rule_based'] and acc['hybrid'] >= acc['ml']:
        rule_improvement = acc['hybrid'] - acc['rule_based']
        ml_improvement = acc['hybrid'] - acc['ml']
        print(f"  Hybrid improvement over Rule-Based: +{rule_improvement:.1%}")
        print(f"  Hybrid improvement over ML: +{ml_improvement:.1%}")
    
    # Method selection distribution
    print("\n" + "=" * 70)
    print("METHOD SELECTION DISTRIBUTION")
    print("=" * 70)
    
    method_sel = metrics['method_selection']
    print(f"\n{'Method':<20} {'Count':>10} {'Percentage':>12}")
    print("-" * 42)
    
    for method, count in sorted(method_sel['counts'].items()):
        pct = method_sel['percentages'].get(method, 0)
        bar = "#" * int(pct * 30)
        print(f"{method:<20} {count:>10} {pct:>11.1%} {bar}")
    
    # Per-method accuracy
    print("\n" + "=" * 70)
    print("ACCURACY BY HYBRID METHOD")
    print("=" * 70)
    
    print(f"\n{'Method':<20} {'Accuracy':>12} {'Samples':>10}")
    print("-" * 42)
    
    for method, accuracy in sorted(metrics['method_accuracy'].items()):
        count = method_sel['counts'].get(method, 0)
        print(f"{method:<20} {accuracy:>11.1%} {count:>10}")
    
    # Confidence distribution
    print("\n" + "=" * 70)
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 70)
    
    conf = metrics['confidence_distribution']
    print(f"\n{'Confidence':<12} {'Count':>10} {'Percentage':>12}")
    print("-" * 34)
    
    for level in ['high', 'medium', 'low']:
        count = conf['counts'].get(level, 0)
        pct = conf['percentages'].get(level, 0)
        bar = "#" * int(pct * 30)
        print(f"{level:<12} {count:>10} {pct:>11.1%} {bar}")
    
    # Rule/ML agreement
    print("\n" + "=" * 70)
    print("SELECTOR AGREEMENT ANALYSIS")
    print("=" * 70)
    
    print(f"\nRule-ML Agreement Rate: {metrics['rule_ml_agreement']:.1%}")
    
    comp = metrics['hybrid_comparison']
    print(f"\nHybrid vs Individual Selectors:")
    print(f"  • Hybrid correct when Rule wrong: {comp['outperforms_rule']:>3} cases")
    print(f"  • Hybrid correct when ML wrong:   {comp['outperforms_ml']:>3} cases")
    print(f"  • Hybrid wrong when Rule correct: {comp['worse_than_rule']:>3} cases")
    print(f"  • Hybrid wrong when ML correct:   {comp['worse_than_ml']:>3} cases")
    
    # Accuracy by file size
    print("\n" + "=" * 70)
    print("ACCURACY BY FILE SIZE")
    print("=" * 70)
    
    size_acc = metrics['size_accuracy']
    sizes = sorted(size_acc['rule'].keys())
    
    print(f"\n{'Size':<12} {'Rule':>10} {'ML':>10} {'Hybrid':>10} {'Best':>10}")
    print("-" * 52)
    
    for size in sizes:
        size_name = SIZE_CONFIG_TO_FILE.get(str(size), f"{size}B")
        rule_a = size_acc['rule'].get(size, 0)
        ml_a = size_acc['ml'].get(size, 0)
        hybrid_a = size_acc['hybrid'].get(size, 0)
        
        best_method = 'Rule' if rule_a >= ml_a and rule_a >= hybrid_a else \
                      'ML' if ml_a >= hybrid_a else 'Hybrid'
        
        print(f"{size_name:<12} {rule_a:>9.1%} {ml_a:>9.1%} {hybrid_a:>9.1%} {best_method:>10}")
    
    # Accuracy by file type
    print("\n" + "=" * 70)
    print("ACCURACY BY FILE TYPE")
    print("=" * 70)
    
    type_acc = metrics['type_accuracy']
    types = sorted(type_acc['rule'].keys())
    
    print(f"\n{'Type':<8} {'Rule':>10} {'ML':>10} {'Hybrid':>10} {'Best':>10}")
    print("-" * 48)
    
    for file_type in types:
        rule_a = type_acc['rule'].get(file_type, 0)
        ml_a = type_acc['ml'].get(file_type, 0)
        hybrid_a = type_acc['hybrid'].get(file_type, 0)
        
        best_method = 'Rule' if rule_a >= ml_a and rule_a >= hybrid_a else \
                      'ML' if ml_a >= hybrid_a else 'Hybrid'
        
        print(f"{file_type:<8} {rule_a:>9.1%} {ml_a:>9.1%} {hybrid_a:>9.1%} {best_method:>10}")
    
    # Examples where hybrid outperforms
    print("\n" + "=" * 70)
    print("EXAMPLES: HYBRID OUTPERFORMS INDIVIDUAL SELECTORS")
    print("=" * 70)
    
    # Cases where hybrid is correct but rule is wrong
    hybrid_beats_rule = results_df[results_df['hybrid_outperforms_rule']]
    if len(hybrid_beats_rule) > 0:
        print(f"\n--- Hybrid correct, Rule-Based wrong ({len(hybrid_beats_rule)} cases) ---")
        print(f"{'Config':<20} {'Actual':<12} {'Rule':<12} {'Hybrid':<12}")
        print("-" * 56)
        for _, row in hybrid_beats_rule.head(5).iterrows():
            print(f"{row['config']:<20} {row['actual_optimal']:<12} {row['rule_algorithm']:<12} {row['hybrid_algorithm']:<12}")
    else:
        print("\nNo cases where hybrid outperforms rule-based selector.")
    
    # Cases where hybrid is correct but ML is wrong
    hybrid_beats_ml = results_df[results_df['hybrid_outperforms_ml']]
    if len(hybrid_beats_ml) > 0:
        print(f"\n--- Hybrid correct, ML wrong ({len(hybrid_beats_ml)} cases) ---")
        print(f"{'Config':<20} {'Actual':<12} {'ML':<12} {'Hybrid':<12}")
        print("-" * 56)
        for _, row in hybrid_beats_ml.head(5).iterrows():
            print(f"{row['config']:<20} {row['actual_optimal']:<12} {row['ml_algorithm']:<12} {row['hybrid_algorithm']:<12}")
    else:
        print("\nNo cases where hybrid outperforms ML selector.")
    
    # Error analysis
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    
    # Hybrid errors
    hybrid_errors = results_df[~results_df['hybrid_correct']]
    if len(hybrid_errors) > 0:
        print(f"\n--- Hybrid Errors ({len(hybrid_errors)} cases) ---")
        print(f"{'Config':<20} {'Actual':<12} {'Predicted':<12} {'Method':<15}")
        print("-" * 59)
        for _, row in hybrid_errors.iterrows():
            print(f"{row['config']:<20} {row['actual_optimal']:<12} {row['hybrid_algorithm']:<12} {row['hybrid_method']:<15}")
    else:
        print("\n✓ Hybrid selector had no errors!")
    
    print("\n" + "=" * 70)


def save_results(results_df: pd.DataFrame, metrics: dict, output_dir: str = 'results/models'):
    """Save evaluation results to files.
    
    Args:
        results_df: Full results DataFrame
        metrics: Metrics dictionary
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed CSV
    csv_path = output_path / 'selector_evaluation.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved detailed results to: {csv_path}")
    
    # Save metrics JSON
    metrics_json = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': int(metrics['total_samples']),
        'overall_accuracy': {k: float(v) for k, v in metrics['overall_accuracy'].items()},
        'method_selection': {
            'counts': {str(k): int(v) for k, v in metrics['method_selection']['counts'].items()},
            'percentages': {str(k): float(v) for k, v in metrics['method_selection']['percentages'].items()},
        },
        'confidence_distribution': {
            'counts': {str(k): int(v) for k, v in metrics['confidence_distribution']['counts'].items()},
            'percentages': {str(k): float(v) for k, v in metrics['confidence_distribution']['percentages'].items()},
        },
        'rule_ml_agreement': float(metrics['rule_ml_agreement']),
        'hybrid_comparison': {k: int(v) for k, v in metrics['hybrid_comparison'].items()},
        'method_accuracy': {str(k): float(v) for k, v in metrics['method_accuracy'].items()},
    }
    
    json_path = output_path / 'selector_evaluation_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"Saved metrics to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate hybrid selector performance'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/ml_data',
        help='Directory containing features.csv and labels.csv'
    )
    parser.add_argument(
        '--test-files',
        type=str,
        default='data/test_files',
        help='Directory containing test files'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='results/models/selector_model.pkl',
        help='Path to trained ML model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/models',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("\n" + "=" * 70)
    print("CRYPTOGREEN HYBRID SELECTOR EVALUATION")
    print("=" * 70)
    
    # Load data
    logger.info(f"Loading evaluation data from: {args.data_dir}")
    features_df, labels_df = load_evaluation_data(args.data_dir)
    logger.info(f"Loaded {len(features_df)} configurations")
    
    # Run evaluation
    results_df = evaluate_selectors(
        features_df,
        labels_df,
        args.test_files,
        args.model_path,
        args.verbose
    )
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(results_df)
    
    # Print report
    print_evaluation_report(metrics, results_df)
    
    # Save results
    save_results(results_df, metrics, args.output_dir)
    
    # Summary
    acc = metrics['overall_accuracy']
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Rule-Based Accuracy: {acc['rule_based']:.1%}")
    print(f"  ML Accuracy:         {acc['ml']:.1%}")
    print(f"  Hybrid Accuracy:     {acc['hybrid']:.1%}")
    print(f"\n  Results saved to: {args.output_dir}/selector_evaluation.csv")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
