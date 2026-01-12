#!/usr/bin/env python3
"""
Real World Evaluation Script

Test the selector on real-world files to validate performance outside
the training distribution.

Usage:
    python scripts/real_world_evaluation.py [OPTIONS]

Options:
    --input-dir DIR   Directory containing real files to test
    --verbose         Show detailed output
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cryptogreen.hybrid_selector import HybridSelector
from cryptogreen.benchmark_framework import CryptoBenchmark
from cryptogreen.utils import format_bytes, format_duration, format_energy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_real_world_files(input_dir: str) -> list:
    """Find real-world files for testing.
    
    Returns:
        List of file paths.
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        return []
    
    # Supported extensions
    extensions = {'.txt', '.pdf', '.jpg', '.jpeg', '.png', '.mp4', '.zip', '.sql'}
    
    files = []
    for f in input_path.rglob('*'):
        if f.is_file() and f.suffix.lower() in extensions:
            files.append(str(f))
    
    return files


def evaluate_on_real_files(
    files: list,
    selector: HybridSelector,
    benchmark: CryptoBenchmark,
    verbose: bool = False
) -> dict:
    """Evaluate selector on real-world files.
    
    Returns:
        Evaluation results dict.
    """
    results = []
    
    for file_path in files:
        try:
            file_size = Path(file_path).stat().st_size
            file_name = Path(file_path).name
            
            logger.info(f"Evaluating: {file_name} ({format_bytes(file_size)})")
            
            # Get selector recommendation
            recommendation = selector.select_algorithm(file_path, verbose=verbose)
            selected_alg = recommendation['algorithm']
            
            # Benchmark all algorithms on this file
            energy_by_alg = {}
            
            for alg in ['AES-128', 'AES-256', 'ChaCha20']:
                try:
                    bench_result = benchmark.benchmark_algorithm(
                        alg, file_path, runs=10
                    )
                    energy_by_alg[alg] = bench_result['statistics']['median_energy_j']
                except Exception as e:
                    logger.warning(f"  Benchmark failed for {alg}: {e}")
            
            if not energy_by_alg:
                continue
            
            # Find actual optimal
            actual_optimal = min(energy_by_alg, key=energy_by_alg.get)
            is_correct = selected_alg == actual_optimal
            
            # Calculate energy savings
            if actual_optimal in energy_by_alg and 'AES-256' in energy_by_alg:
                savings_pct = (
                    (energy_by_alg['AES-256'] - energy_by_alg[actual_optimal]) /
                    energy_by_alg['AES-256']
                ) * 100
            else:
                savings_pct = 0
            
            result = {
                'file_path': file_path,
                'file_name': file_name,
                'file_size': file_size,
                'selected': selected_alg,
                'actual_optimal': actual_optimal,
                'is_correct': is_correct,
                'confidence': recommendation['confidence'],
                'method': recommendation['method'],
                'energy_by_alg': energy_by_alg,
                'savings_vs_aes256_pct': savings_pct,
            }
            
            results.append(result)
            
            status = "✓" if is_correct else "✗"
            logger.info(
                f"  {status} Selected: {selected_alg}, Optimal: {actual_optimal}, "
                f"Savings: {savings_pct:.1f}%"
            )
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    # Calculate overall metrics
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    avg_savings = sum(r['savings_vs_aes256_pct'] for r in results) / total if total > 0 else 0
    
    return {
        'total_files': total,
        'correct_predictions': correct,
        'accuracy': correct / total if total > 0 else 0,
        'average_savings_pct': avg_savings,
        'results': results,
    }


def print_evaluation_report(metrics: dict):
    """Print formatted evaluation report."""
    print()
    print("=" * 60)
    print("REAL-WORLD EVALUATION REPORT")
    print("=" * 60)
    print()
    
    print("SUMMARY:")
    print(f"  Files evaluated: {metrics['total_files']}")
    print(f"  Correct predictions: {metrics['correct_predictions']}")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Average energy savings: {metrics['average_savings_pct']:.1f}%")
    print()
    
    # Target check
    if metrics['accuracy'] >= 0.85:
        print("  ✓ Meets accuracy target (≥85%)")
    else:
        print("  ✗ Below accuracy target (85%)")
    
    if metrics['average_savings_pct'] >= 35:
        print("  ✓ Meets energy savings target (≥35%)")
    else:
        print(f"  ✗ Below energy savings target (35%). Got {metrics['average_savings_pct']:.1f}%")
    print()
    
    # Per-file results
    print("DETAILED RESULTS:")
    print("-" * 60)
    
    for r in metrics['results']:
        status = "✓" if r['is_correct'] else "✗"
        print(f"  {status} {r['file_name']}")
        print(f"      Size: {format_bytes(r['file_size'])}")
        print(f"      Selected: {r['selected']} ({r['confidence']} confidence)")
        print(f"      Optimal: {r['actual_optimal']}")
        print(f"      Energy savings: {r['savings_vs_aes256_pct']:.1f}%")
        print()
    
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate CryptoGreen on real-world files'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/real_world_files',
        help='Directory containing real files to test'
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
    
    # Find files
    files = find_real_world_files(args.input_dir)
    
    if not files:
        logger.warning(f"No test files found in {args.input_dir}")
        logger.info("Please add real-world files to data/real_world_files/")
        logger.info("Supported formats: txt, pdf, jpg, png, mp4, zip, sql")
        
        # Create directory
        Path(args.input_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.input_dir) / '.gitkeep').touch()
        
        sys.exit(0)
    
    logger.info(f"Found {len(files)} files to evaluate")
    
    # Initialize selector and benchmark
    selector = HybridSelector(model_path=args.model)
    benchmark = CryptoBenchmark()
    
    if not selector.is_trained():
        logger.warning("ML model not trained. Using rule-based selection.")
    
    # Evaluate
    metrics = evaluate_on_real_files(
        files, selector, benchmark, verbose=args.verbose
    )
    
    # Print report
    print_evaluation_report(metrics)


if __name__ == '__main__':
    main()
