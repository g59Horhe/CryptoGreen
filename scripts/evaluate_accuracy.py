#!/usr/bin/env python3
"""
Evaluate Accuracy Script

Evaluate the accuracy of the hybrid selector on benchmark data.

Usage:
    python scripts/evaluate_accuracy.py [OPTIONS]

Options:
    --benchmark-results PATH  Path to benchmark results JSON
    --model PATH              Path to trained model
    --verbose                 Show detailed output
"""

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cryptogreen.hybrid_selector import HybridSelector
from cryptogreen.feature_extractor import FeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_benchmark_results(file_path: str) -> list:
    """Load benchmark results from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    return data


def get_optimal_algorithms(results: list) -> dict:
    """Determine optimal algorithm for each file based on benchmark data.
    
    Returns:
        Dict mapping file_name to optimal algorithm.
    """
    # Group by file
    by_file = {}
    for r in results:
        file_key = r['file_name']
        if file_key not in by_file:
            by_file[file_key] = []
        by_file[file_key].append(r)
    
    # Find optimal for each file (lowest median energy)
    optimal = {}
    for file_key, file_results in by_file.items():
        # Only consider symmetric algorithms for fair comparison
        symmetric = [r for r in file_results if r['algorithm'] in ['AES-128', 'AES-256', 'ChaCha20']]
        
        if symmetric:
            best = min(symmetric, key=lambda x: x['statistics']['median_energy_j'])
            optimal[file_key] = {
                'algorithm': best['algorithm'],
                'energy': best['statistics']['median_energy_j'],
                'file_path': best.get('file_path', ''),
                'file_size': best['file_size'],
                'file_type': best['file_type'],
            }
    
    return optimal


def evaluate_selector(
    selector: HybridSelector,
    optimal_algorithms: dict,
    verbose: bool = False
) -> dict:
    """Evaluate selector accuracy against benchmark-determined optimal.
    
    Returns:
        Evaluation metrics dict.
    """
    correct = 0
    top2_correct = 0
    total = 0
    
    results = []
    
    by_method = {
        'both_agree': {'correct': 0, 'total': 0},
        'ml_preferred': {'correct': 0, 'total': 0},
        'rules_preferred': {'correct': 0, 'total': 0},
        'security_override': {'correct': 0, 'total': 0},
    }
    
    for file_name, optimal in optimal_algorithms.items():
        file_path = optimal.get('file_path', '')
        optimal_alg = optimal['algorithm']
        
        # Skip if file doesn't exist
        if not file_path or not Path(file_path).exists():
            # Try to find file
            possible_paths = list(Path('data/test_files').rglob(file_name))
            if possible_paths:
                file_path = str(possible_paths[0])
            else:
                if verbose:
                    logger.debug(f"Skipping {file_name}: file not found")
                continue
        
        try:
            prediction = selector.select_algorithm(file_path)
            predicted_alg = prediction['algorithm']
            method = prediction['method']
            
            is_correct = predicted_alg == optimal_alg
            
            # Check top-2
            ml_rec = prediction.get('ml_recommendation', {})
            alternatives = ml_rec.get('alternatives', [predicted_alg])
            is_top2 = optimal_alg in alternatives[:2]
            
            if is_correct:
                correct += 1
            
            if is_top2:
                top2_correct += 1
            
            total += 1
            
            # Track by method
            if method in by_method:
                by_method[method]['total'] += 1
                if is_correct:
                    by_method[method]['correct'] += 1
            
            results.append({
                'file_name': file_name,
                'optimal': optimal_alg,
                'predicted': predicted_alg,
                'correct': is_correct,
                'top2_correct': is_top2,
                'method': method,
                'confidence': prediction['confidence'],
            })
            
            if verbose:
                status = "✓" if is_correct else "✗"
                print(f"  {status} {file_name}: predicted={predicted_alg}, optimal={optimal_alg}")
                
        except Exception as e:
            logger.warning(f"Error evaluating {file_name}: {e}")
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    top2_accuracy = top2_correct / total if total > 0 else 0
    
    # Method breakdown
    method_accuracy = {}
    for method, counts in by_method.items():
        if counts['total'] > 0:
            method_accuracy[method] = counts['correct'] / counts['total']
    
    return {
        'accuracy': accuracy,
        'top2_accuracy': top2_accuracy,
        'correct': correct,
        'top2_correct': top2_correct,
        'total': total,
        'by_method': by_method,
        'method_accuracy': method_accuracy,
        'predictions': results,
    }


def print_evaluation_report(metrics: dict):
    """Print formatted evaluation report."""
    print()
    print("=" * 60)
    print("CRYPTOGREEN SELECTOR EVALUATION REPORT")
    print("=" * 60)
    print()
    
    print("OVERALL ACCURACY:")
    print(f"  Top-1 Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    print(f"  Top-2 Accuracy: {metrics['top2_accuracy']:.1%} ({metrics['top2_correct']}/{metrics['total']})")
    print()
    
    # Target check
    if metrics['accuracy'] >= 0.85:
        print("  ✓ Meets target accuracy (≥85%)")
    else:
        print(f"  ✗ Below target accuracy (85%). Need {int(0.85 * metrics['total']) - metrics['correct']} more correct predictions")
    
    if metrics['top2_accuracy'] >= 0.95:
        print("  ✓ Meets top-2 target (≥95%)")
    else:
        print(f"  ✗ Below top-2 target (95%)")
    print()
    
    print("ACCURACY BY METHOD:")
    print("-" * 40)
    for method, counts in metrics['by_method'].items():
        if counts['total'] > 0:
            acc = counts['correct'] / counts['total']
            print(f"  {method:20s}: {acc:.1%} ({counts['correct']}/{counts['total']})")
    print()
    
    # Confusion analysis
    predictions = metrics['predictions']
    if predictions:
        print("CONFUSION ANALYSIS:")
        print("-" * 40)
        
        # Count misclassifications
        misclass = {}
        for p in predictions:
            if not p['correct']:
                key = (p['optimal'], p['predicted'])
                misclass[key] = misclass.get(key, 0) + 1
        
        if misclass:
            print("  Most common errors (optimal → predicted):")
            for (opt, pred), count in sorted(misclass.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {opt} → {pred}: {count} times")
        else:
            print("  No misclassifications!")
    
    print()
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate CryptoGreen selector accuracy'
    )
    parser.add_argument(
        '--benchmark-results',
        type=str,
        default=None,
        help='Path to benchmark results JSON'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='results/models/selector_model.pkl',
        help='Path to trained model'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Find benchmark results
    if args.benchmark_results:
        benchmark_file = args.benchmark_results
    else:
        pattern = 'results/benchmarks/raw/benchmark_*.json'
        files = glob.glob(pattern)
        files = [f for f in files if 'incremental' not in f]
        
        if not files:
            logger.error(f"No benchmark results found")
            sys.exit(1)
        
        benchmark_file = max(files, key=lambda x: Path(x).stat().st_mtime)
        logger.info(f"Using benchmark results: {benchmark_file}")
    
    # Load results
    results = load_benchmark_results(benchmark_file)
    logger.info(f"Loaded {len(results)} benchmark results")
    
    # Get optimal algorithms
    optimal = get_optimal_algorithms(results)
    logger.info(f"Determined optimal algorithms for {len(optimal)} files")
    
    # Initialize selector
    logger.info(f"Loading selector with model: {args.model}")
    selector = HybridSelector(model_path=args.model)
    
    if not selector.is_trained():
        logger.warning("ML model not trained. Results may be inaccurate.")
    
    # Evaluate
    logger.info("Evaluating selector accuracy...")
    metrics = evaluate_selector(selector, optimal, verbose=args.verbose)
    
    # Print report
    print_evaluation_report(metrics)


if __name__ == '__main__':
    main()
