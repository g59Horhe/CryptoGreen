#!/usr/bin/env python3
"""
Run Benchmarks Script

Execute complete benchmark suite for CryptoGreen project.

Usage:
    python scripts/run_benchmarks.py [OPTIONS]

Options:
    --runs N           Number of repetitions per config (default: 100)
    --algorithms LIST  Comma-separated algorithms (default: all)
    --output DIR       Output directory (default: results/benchmarks)
    --resume          Resume from last checkpoint
    --test-mode       Quick test with 10 runs

Examples:
    # Quick test
    python scripts/run_benchmarks.py --test-mode

    # Full benchmark
    python scripts/run_benchmarks.py --runs 100

    # Specific algorithms
    python scripts/run_benchmarks.py --algorithms AES-128,ChaCha20
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cryptogreen.benchmark_framework import CryptoBenchmark
from cryptogreen.energy_meter import RAPLEnergyMeter, RAPLNotAvailableError
from cryptogreen.algorithms import CryptoAlgorithms
from cryptogreen.utils import format_duration, get_system_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def estimate_time(num_files: int, num_algorithms: int, runs: int) -> str:
    """Estimate total benchmark time.
    
    Args:
        num_files: Number of test files.
        num_algorithms: Number of algorithms.
        runs: Runs per configuration.
        
    Returns:
        Human-readable time estimate.
    """
    # Rough estimate: 0.05 seconds per encryption + overhead
    # Actual time depends on file size and algorithm
    operations = num_files * num_algorithms * runs
    seconds = operations * 0.05  # Conservative estimate
    
    return format_duration(seconds)


def check_prerequisites(test_files_dir: str) -> tuple[bool, str]:
    """Check if prerequisites are met.
    
    Returns:
        Tuple of (success, message).
    """
    test_dir = Path(test_files_dir)
    
    if not test_dir.exists():
        return False, (
            f"Test files directory not found: {test_files_dir}\n"
            "Run 'python scripts/generate_test_data.py' first."
        )
    
    # Count test files
    test_files = list(test_dir.rglob('*'))
    test_files = [f for f in test_files if f.is_file() and not f.name.startswith('.')]
    
    if len(test_files) < 10:
        return False, (
            f"Only {len(test_files)} test files found. "
            "Run 'python scripts/generate_test_data.py' to generate more."
        )
    
    return True, f"Found {len(test_files)} test files"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run CryptoGreen benchmarks'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=100,
        help='Number of repetitions per configuration (default: 100)'
    )
    parser.add_argument(
        '--algorithms',
        type=str,
        default='all',
        help='Comma-separated list of algorithms (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/benchmarks',
        help='Output directory (default: results/benchmarks)'
    )
    parser.add_argument(
        '--test-files',
        type=str,
        default='data/test_files',
        help='Test files directory (default: data/test_files)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint (not yet implemented)'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Quick test mode with 10 runs'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Test mode settings
    if args.test_mode:
        args.runs = 10
        logger.info("=" * 60)
        logger.info("TEST MODE: Running with 10 repetitions only")
        logger.info("=" * 60)
    
    # Check RAPL availability
    logger.info("Checking RAPL availability...")
    if RAPLEnergyMeter.is_available():
        logger.info("✓ Hardware RAPL is available")
    else:
        logger.warning("✗ Hardware RAPL not available")
        logger.warning("  Will use software energy estimation")
        logger.warning("  For accurate measurements, ensure RAPL access:")
        logger.warning("    sudo modprobe msr")
        logger.warning("    sudo chmod -R a+r /sys/class/powercap/intel-rapl/")
        print()
        response = input("Continue with software estimation? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Aborted.")
            sys.exit(0)
    
    # Check prerequisites
    ok, message = check_prerequisites(args.test_files)
    if not ok:
        logger.error(message)
        sys.exit(1)
    logger.info(f"✓ {message}")
    
    # Parse algorithms
    if args.algorithms == 'all':
        algorithms = None
        algo_list = CryptoAlgorithms.get_algorithm_names()
    else:
        algorithms = [a.strip() for a in args.algorithms.split(',')]
        algo_list = algorithms
    
    # Count files for estimation
    test_dir = Path(args.test_files)
    test_files = [f for f in test_dir.rglob('*') if f.is_file() and not f.name.startswith('.')]
    
    # Show summary
    print()
    logger.info("=" * 60)
    logger.info("CryptoGreen Benchmark Configuration")
    logger.info("=" * 60)
    logger.info(f"  Test files directory: {args.test_files}")
    logger.info(f"  Number of test files: {len(test_files)}")
    logger.info(f"  Algorithms: {algo_list}")
    logger.info(f"  Runs per config: {args.runs}")
    logger.info(f"  Output directory: {args.output}")
    
    total_ops = len(test_files) * len(algo_list)
    total_measurements = total_ops * args.runs
    logger.info(f"  Total configurations: {total_ops}")
    logger.info(f"  Total measurements: {total_measurements}")
    logger.info(f"  Estimated time: {estimate_time(len(test_files), len(algo_list), args.runs)}")
    logger.info("=" * 60)
    
    # Get system info
    sys_info = get_system_info()
    logger.info("\nSystem Information:")
    logger.info(f"  OS: {sys_info['os']} {sys_info['os_release']}")
    logger.info(f"  CPU: {sys_info['cpu']['model']}")
    logger.info(f"  Cores: {sys_info['cpu']['cores']} ({sys_info['cpu']['threads']} threads)")
    logger.info(f"  AES-NI: {'Yes' if sys_info['cpu']['has_aes_ni'] else 'No'}")
    logger.info(f"  RAM: {sys_info['ram_total_gb']:.1f} GB")
    print()
    
    # Confirm
    if not args.test_mode:
        response = input("Start benchmark? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Aborted.")
            sys.exit(0)
    
    # Run benchmarks
    try:
        benchmark = CryptoBenchmark(output_dir=args.output)
        
        benchmark.run_full_benchmark(
            test_files_dir=args.test_files,
            runs=args.runs,
            algorithms=algorithms
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("Benchmark Complete!")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {args.output}")
        logger.info(f"Total results: {len(benchmark.results)}")
        
        # Summary statistics
        if benchmark.results:
            algorithms_tested = set(r['algorithm'] for r in benchmark.results)
            files_tested = set(r['file_name'] for r in benchmark.results)
            
            logger.info(f"\nSummary:")
            logger.info(f"  Algorithms tested: {len(algorithms_tested)}")
            logger.info(f"  Files tested: {len(files_tested)}")
            logger.info(f"  Total measurements: {len(benchmark.results) * args.runs}")
        
        logger.info("\nNext steps:")
        logger.info("  1. Train ML model: python scripts/train_model.py")
        logger.info("  2. Analyze results: python scripts/analyze_results.py")
        logger.info("  3. Evaluate selector: python scripts/evaluate_accuracy.py")
        
    except KeyboardInterrupt:
        logger.warning("\nBenchmark interrupted by user")
        logger.info("Partial results may have been saved")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
