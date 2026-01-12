#!/usr/bin/env python3
"""
Train ML Model Script

Train the Random Forest model for algorithm selection using benchmark results.

Usage:
    python scripts/train_model.py [OPTIONS]

Options:
    --benchmark-results PATH  Path to benchmark results JSON
    --output PATH             Where to save model (default: results/models/selector_model.pkl)
    --test-split FLOAT        Test set fraction (default: 0.2)
    --verbose                 Show detailed output

Examples:
    # Train from benchmark results
    python scripts/train_model.py --benchmark-results results/benchmarks/raw/benchmark_*.json

    # Custom output path
    python scripts/train_model.py --output models/my_model.pkl
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

from cryptogreen.ml_selector import MLSelector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_benchmark(benchmark_dir: str = 'results/benchmarks/raw') -> str:
    """Find the latest benchmark results file.
    
    Args:
        benchmark_dir: Directory to search.
        
    Returns:
        Path to latest benchmark file.
        
    Raises:
        FileNotFoundError: If no benchmark files found.
    """
    pattern = Path(benchmark_dir) / 'benchmark_*.json'
    files = glob.glob(str(pattern))
    
    # Filter out incremental files
    files = [f for f in files if 'incremental' not in f]
    
    if not files:
        raise FileNotFoundError(
            f"No benchmark results found in {benchmark_dir}\n"
            "Run 'python scripts/run_benchmarks.py' first."
        )
    
    # Sort by modification time and return latest
    files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    return files[0]


def print_training_report(metrics: dict) -> None:
    """Print formatted training report.
    
    Args:
        metrics: Training metrics dict.
    """
    print()
    print("=" * 60)
    print("MODEL TRAINING REPORT")
    print("=" * 60)
    print()
    
    print("DATASET:")
    print(f"  Training samples: {metrics.get('train_samples', 'N/A')}")
    print(f"  Test samples: {metrics.get('test_samples', 'N/A')}")
    print(f"  Classes: {metrics.get('classes', [])}")
    print()
    
    print("PERFORMANCE:")
    print(f"  Top-1 Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Top-2 Accuracy: {metrics['top2_accuracy']:.1%}")
    print(f"  Cross-Validation: {metrics['cv_mean']:.1%} ± {metrics['cv_std']:.1%}")
    print()
    
    print("FEATURE IMPORTANCE:")
    importance = metrics.get('feature_importance', {})
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feature, imp in sorted_imp:
        bar = '█' * int(imp * 50)
        print(f"  {feature:25s} {imp:.3f} {bar}")
    print()
    
    print("CONFUSION MATRIX:")
    cm = metrics.get('confusion_matrix', [])
    classes = metrics.get('classes', [])
    if cm and classes:
        # Header
        header = "Predicted →"
        print(f"  {'':15s}", end='')
        for cls in classes:
            print(f"{cls:>12s}", end='')
        print()
        print(f"  {'Actual ↓':15s}")
        
        for i, row in enumerate(cm):
            print(f"  {classes[i]:15s}", end='')
            for val in row:
                print(f"{val:12d}", end='')
            print()
    print()
    
    print("PER-CLASS METRICS:")
    report = metrics.get('classification_report', {})
    if report:
        print(f"  {'Class':15s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
        print("  " + "-" * 55)
        for cls in classes:
            if cls in report:
                r = report[cls]
                print(f"  {cls:15s} {r['precision']:>10.2f} {r['recall']:>10.2f} {r['f1-score']:>10.2f} {int(r['support']):>10d}")
    
    print()
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train ML model for CryptoGreen algorithm selection'
    )
    parser.add_argument(
        '--benchmark-results',
        type=str,
        default=None,
        help='Path to benchmark results JSON (auto-detect if not provided)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/models/selector_model.pkl',
        help='Output path for trained model'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
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
        # Handle glob patterns
        if '*' in args.benchmark_results:
            files = glob.glob(args.benchmark_results)
            if not files:
                logger.error(f"No files match pattern: {args.benchmark_results}")
                sys.exit(1)
            benchmark_file = max(files, key=lambda x: Path(x).stat().st_mtime)
        else:
            benchmark_file = args.benchmark_results
    else:
        try:
            benchmark_file = find_latest_benchmark()
            logger.info(f"Auto-detected benchmark file: {benchmark_file}")
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
    
    if not Path(benchmark_file).exists():
        logger.error(f"Benchmark file not found: {benchmark_file}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("CryptoGreen Model Training")
    logger.info("=" * 60)
    logger.info(f"Benchmark results: {benchmark_file}")
    logger.info(f"Output model: {args.output}")
    logger.info(f"Test split: {args.test_split}")
    print()
    
    # Load benchmark data info
    with open(benchmark_file, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', data)
    logger.info(f"Loaded {len(results)} benchmark results")
    
    # Initialize selector and train
    selector = MLSelector()
    
    try:
        metrics = selector.train_from_benchmark_results(
            benchmark_file,
            output_path=args.output
        )
        
        print_training_report(metrics)
        
        # Check if accuracy meets target
        if metrics['accuracy'] >= 0.85:
            logger.info("✓ Model meets target accuracy (≥85%)")
        else:
            logger.warning(f"✗ Model below target accuracy (85%). Got {metrics['accuracy']:.1%}")
            logger.warning("  Consider collecting more benchmark data or tuning features")
        
        if metrics['top2_accuracy'] >= 0.95:
            logger.info("✓ Model meets top-2 accuracy target (≥95%)")
        else:
            logger.warning(f"✗ Model below top-2 accuracy target (95%). Got {metrics['top2_accuracy']:.1%}")
        
        print()
        logger.info(f"Model saved to: {args.output}")
        logger.info("\nNext steps:")
        logger.info("  1. Evaluate accuracy: python scripts/evaluate_accuracy.py")
        logger.info("  2. Test selector: python -c \"from cryptogreen import HybridSelector; ...\"")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
