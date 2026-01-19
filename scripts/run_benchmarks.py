#!/usr/bin/env python3
"""
Run Benchmarks Script - Enhanced Version

Execute complete benchmark suite for CryptoGreen project with:
- Progress tracking with ETA
- Incremental saves (don't lose progress)
- Resume capability from checkpoint
- Comprehensive logging

Usage:
    python scripts/run_benchmarks.py [OPTIONS]

Options:
    --runs N           Number of repetitions per config (default: 100)
    --algorithms LIST  Comma-separated algorithms (default: all)
    --output DIR       Output directory (default: results/benchmarks)
    --resume           Resume from last checkpoint
    --test-mode        Quick test with 10 runs
    --no-confirm       Skip confirmation prompt (for background runs)

Examples:
    # Quick test
    python scripts/run_benchmarks.py --test-mode

    # Full benchmark (background)
    python scripts/run_benchmarks.py --runs 100 --no-confirm

    # Resume interrupted benchmark
    python scripts/run_benchmarks.py --resume
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cryptogreen.benchmark_framework import CryptoBenchmark
from cryptogreen.energy_meter import RAPLEnergyMeter, RAPLNotAvailableError
from cryptogreen.algorithms import CryptoAlgorithms
from cryptogreen.utils import format_duration, get_system_info

# Symmetric encryption algorithms for benchmarking
DEFAULT_ALGORITHMS = ['AES-128', 'AES-256', 'ChaCha20']

# Global for graceful shutdown
_shutdown_requested = False
_benchmark_instance = None


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _shutdown_requested
    if _shutdown_requested:
        print("\nForce quit requested. Exiting immediately.")
        sys.exit(1)
    _shutdown_requested = True
    print("\n\n[!] Shutdown requested. Saving progress and exiting gracefully...")
    print("   (Press Ctrl+C again to force quit)")


def setup_logging(output_dir: str, timestamp: str) -> logging.Logger:
    """Setup logging to both console and file."""
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f'benchmark_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger('cryptogreen.benchmark')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler (INFO level)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console)
    
    # File handler (DEBUG level - captures everything)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)
    
    logger.info(f"[LOG] Logging to: {log_file}")
    
    return logger


class EnhancedBenchmarkRunner:
    """Enhanced benchmark runner with progress tracking and resume capability."""
    
    CHECKPOINT_FILE = 'benchmark_checkpoint.json'
    
    def __init__(
        self,
        output_dir: str = 'results/benchmarks',
        runs: int = 100,
        algorithms: Optional[list[str]] = None,
        test_files_dir: str = 'data/test_files'
    ):
        self.output_dir = Path(output_dir)
        self.runs = runs
        self.test_files_dir = Path(test_files_dir)
        self.algorithms = algorithms or DEFAULT_ALGORITHMS
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = setup_logging(output_dir, self.timestamp)
        
        # Progress tracking
        self.completed = []  # List of (algorithm, file_path) tuples
        self.results = []
        self.start_time = None
        self.times_per_config = []  # Track timing for ETA
        
        # Initialize benchmark framework
        self.benchmark = CryptoBenchmark(output_dir=output_dir)
        
        # Get test files
        self.test_files = self._get_test_files()
        
        # Calculate totals
        self.total_configs = len(self.test_files) * len(self.algorithms)
        self.total_measurements = self.total_configs * self.runs
    
    def _get_test_files(self) -> list[Path]:
        """Get all test files sorted by size."""
        files = list(self.test_files_dir.rglob('*'))
        files = [f for f in files if f.is_file() and not f.name.startswith('.')]
        
        # Sort by file size for better progress estimation
        files.sort(key=lambda f: f.stat().st_size)
        return files
    
    def _get_checkpoint_path(self) -> Path:
        """Get checkpoint file path."""
        return self.output_dir / self.CHECKPOINT_FILE
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        checkpoint = {
            'timestamp': self.timestamp,
            'completed': self.completed,
            'runs': self.runs,
            'algorithms': self.algorithms,
            'test_files_dir': str(self.test_files_dir),
            'results_file': f'benchmark_{self.timestamp}_incremental.json',
            'last_update': datetime.now().isoformat()
        }
        
        checkpoint_path = self._get_checkpoint_path()
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self) -> Optional[dict]:
        """Load checkpoint if exists."""
        checkpoint_path = self._get_checkpoint_path()
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            self.logger.info(f"[CHECKPOINT] Found checkpoint from {checkpoint.get('last_update', 'unknown')}")
            return checkpoint
        except Exception as e:
            self.logger.warning(f"Could not load checkpoint: {e}")
            return None
    
    def resume_from_checkpoint(self, checkpoint: dict) -> bool:
        """Resume benchmark from checkpoint."""
        self.timestamp = checkpoint['timestamp']
        self.completed = [tuple(c) for c in checkpoint['completed']]
        
        # Load existing results
        results_file = self.output_dir / 'raw' / checkpoint['results_file']
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                self.results = data.get('results', [])
        
        completed_count = len(self.completed)
        self.logger.info(f"[RESUME] Resuming: {completed_count}/{self.total_configs} configs already done")
        self.logger.info(f"   {len(self.results)} results loaded")
        
        return True
    
    def _format_eta(self, seconds: float) -> str:
        """Format seconds as human-readable ETA."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def _calculate_eta(self, completed: int) -> str:
        """Calculate estimated time remaining."""
        if not self.times_per_config or completed == 0:
            return "calculating..."
        
        # Use recent average (last 20 configs)
        recent = self.times_per_config[-20:]
        avg_time = sum(recent) / len(recent)
        
        remaining = self.total_configs - completed
        eta_seconds = remaining * avg_time
        
        return self._format_eta(eta_seconds)
    
    def _save_incremental(self):
        """Save results incrementally."""
        output_file = self.output_dir / 'raw' / f'benchmark_{self.timestamp}_incremental.json'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'timestamp': self.timestamp,
            'hardware_info': self.benchmark.hardware_info,
            'total_configs': self.total_configs,
            'completed_configs': len(self.completed),
            'runs_per_config': self.runs,
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _log_progress(self, completed: int, algorithm: str, file_name: str, 
                      energy: float, throughput: float, time_taken: float):
        """Log detailed progress."""
        pct = (completed / self.total_configs) * 100
        eta = self._calculate_eta(completed)
        
        # Progress bar
        bar_width = 30
        filled = int(bar_width * completed / self.total_configs)
        bar = '#' * filled + '-' * (bar_width - filled)
        
        self.logger.info(
            f"[{bar}] {pct:5.1f}% | {completed}/{self.total_configs} | "
            f"ETA: {eta} | {algorithm}: {file_name} | "
            f"{energy:.6f}J | {throughput:.1f}MB/s"
        )
    
    def run(self, resume: bool = False) -> bool:
        """Run the full benchmark suite.
        
        Returns:
            True if completed successfully, False if interrupted.
        """
        global _shutdown_requested, _benchmark_instance
        _benchmark_instance = self
        
        # Check for resume
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                self.resume_from_checkpoint(checkpoint)
        
        self.start_time = time.time()
        completed_count = len(self.completed)
        
        self.logger.info("=" * 70)
        self.logger.info(">>> Starting CryptoGreen Full Benchmark Suite <<<")
        self.logger.info("=" * 70)
        self.logger.info(f"   Test files: {len(self.test_files)}")
        self.logger.info(f"   Algorithms: {self.algorithms}")
        self.logger.info(f"   Runs per config: {self.runs}")
        self.logger.info(f"   Total configurations: {self.total_configs}")
        self.logger.info(f"   Total measurements: {self.total_measurements:,}")
        self.logger.info(f"   Already completed: {completed_count}")
        self.logger.info(f"   Hardware RAPL: {self.benchmark.using_hardware_rapl}")
        self.logger.info("=" * 70)
        
        # Progress report interval (every 10 minutes)
        last_report_time = time.time()
        report_interval = 600  # 10 minutes
        
        try:
            for algorithm in self.algorithms:
                if _shutdown_requested:
                    break
                
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"[ALGO] Algorithm: {algorithm}")
                self.logger.info(f"{'='*50}")
                
                for test_file in self.test_files:
                    if _shutdown_requested:
                        break
                    
                    # Skip if already completed
                    config_key = (algorithm, str(test_file))
                    if config_key in self.completed:
                        continue
                    
                    config_start = time.time()
                    
                    try:
                        # Run benchmark
                        result = self.benchmark.benchmark_algorithm(
                            algorithm,
                            str(test_file),
                            runs=self.runs
                        )
                        
                        self.results.append(result)
                        self.completed.append(config_key)
                        
                        config_time = time.time() - config_start
                        self.times_per_config.append(config_time)
                        
                        # Extract stats for logging
                        stats = result.get('statistics', {})
                        median_energy = stats.get('median_energy_j', 0)
                        throughput = stats.get('throughput_mbps', 0)
                        
                        completed_count = len(self.completed)
                        
                        # Log progress
                        self._log_progress(
                            completed_count,
                            algorithm,
                            test_file.name,
                            median_energy,
                            throughput,
                            config_time
                        )
                        
                        # Save incremental results
                        self._save_incremental()
                        self.save_checkpoint()
                        
                        # Periodic detailed report
                        if time.time() - last_report_time >= report_interval:
                            self._log_detailed_report(completed_count)
                            last_report_time = time.time()
                        
                    except Exception as e:
                        self.logger.error(f"[ERROR] Error: {algorithm} on {test_file.name}: {e}")
                        self.completed.append(config_key)  # Mark as done to skip on resume
                        self.save_checkpoint()
            
            # Benchmark completed
            if not _shutdown_requested:
                self._finalize()
                return True
            else:
                self.logger.info("\n[!] Benchmark interrupted. Progress saved.")
                self.logger.info(f"   Completed: {len(self.completed)}/{self.total_configs}")
                self.logger.info(f"   Resume with: python scripts/run_benchmarks.py --resume")
                return False
                
        except Exception as e:
            self.logger.error(f"[FATAL] Fatal error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self._save_incremental()
            self.save_checkpoint()
            raise
    
    def _log_detailed_report(self, completed: int):
        """Log a detailed progress report."""
        elapsed = time.time() - self.start_time
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("=== PROGRESS REPORT ===")
        self.logger.info("=" * 70)
        self.logger.info(f"   Completed: {completed}/{self.total_configs} configs ({completed/self.total_configs*100:.1f}%)")
        self.logger.info(f"   Measurements: {completed * self.runs:,}/{self.total_measurements:,}")
        self.logger.info(f"   Elapsed time: {self._format_eta(elapsed)}")
        self.logger.info(f"   ETA remaining: {self._calculate_eta(completed)}")
        
        if self.times_per_config:
            avg_time = sum(self.times_per_config) / len(self.times_per_config)
            self.logger.info(f"   Avg time/config: {avg_time:.2f}s")
        
        # Algorithm breakdown
        algo_counts = {}
        for algo, _ in self.completed:
            algo_counts[algo] = algo_counts.get(algo, 0) + 1
        
        self.logger.info("   Per-algorithm progress:")
        for algo in self.algorithms:
            count = algo_counts.get(algo, 0)
            total = len(self.test_files)
            self.logger.info(f"     {algo}: {count}/{total}")
        
        self.logger.info("=" * 70 + "\n")
    
    def _finalize(self):
        """Finalize benchmark and save results."""
        elapsed = time.time() - self.start_time
        
        # Save final results
        final_file = self.output_dir / 'raw' / f'benchmark_{self.timestamp}.json'
        data = {
            'timestamp': self.timestamp,
            'hardware_info': self.benchmark.hardware_info,
            'benchmark_config': {
                'runs_per_config': self.runs,
                'algorithms': self.algorithms,
                'total_configs': self.total_configs,
                'total_measurements': self.total_measurements,
                'elapsed_seconds': elapsed
            },
            'results': self.results
        }
        
        with open(final_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Generate summary CSV
        self._generate_summary_csv()
        
        # Clean up checkpoint
        checkpoint_path = self._get_checkpoint_path()
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("*** BENCHMARK COMPLETE! ***")
        self.logger.info("=" * 70)
        self.logger.info(f"   Total time: {self._format_eta(elapsed)}")
        self.logger.info(f"   Configurations: {len(self.completed)}")
        self.logger.info(f"   Measurements: {len(self.completed) * self.runs:,}")
        self.logger.info(f"   Results: {final_file}")
        self.logger.info("=" * 70)
        self.logger.info("\n[NEXT] Next steps:")
        self.logger.info("   1. Train ML model: python scripts/train_model.py")
        self.logger.info("   2. Analyze results: python scripts/analyze_results.py")
        self.logger.info("   3. Evaluate selector: python scripts/evaluate_accuracy.py")
    
    def _generate_summary_csv(self):
        """Generate summary CSV from results."""
        import csv
        
        csv_file = self.output_dir / 'processed' / f'summary_{self.timestamp}.csv'
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'algorithm', 'file_name', 'file_type', 'file_size_bytes',
                'runs', 'median_energy_j', 'mean_energy_j', 'std_energy_j',
                'median_time_s', 'mean_time_s', 'throughput_mbps',
                'energy_per_byte_uj'
            ])
            
            # Data rows
            for result in self.results:
                stats = result.get('statistics', {})
                writer.writerow([
                    result.get('algorithm', ''),
                    result.get('file_name', ''),
                    result.get('file_type', ''),
                    result.get('file_size', 0),
                    result.get('runs', 0),
                    stats.get('median_energy_j', 0),
                    stats.get('mean_energy_j', 0),
                    stats.get('std_energy_j', 0),
                    stats.get('median_time_s', 0),
                    stats.get('mean_time_s', 0),
                    stats.get('throughput_mbps', 0),
                    stats.get('energy_per_byte_uj', 0)
                ])
        
        self.logger.info(f"   Summary CSV: {csv_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run CryptoGreen benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test:     python scripts/run_benchmarks.py --test-mode
  Full benchmark: python scripts/run_benchmarks.py --runs 100 --no-confirm
  Resume:         python scripts/run_benchmarks.py --resume
        """
    )
    parser.add_argument(
        '--runs', type=int, default=100,
        help='Number of repetitions per configuration (default: 100)'
    )
    parser.add_argument(
        '--algorithms', type=str, default='all',
        help='Comma-separated list of algorithms (default: all)'
    )
    parser.add_argument(
        '--output', type=str, default='results/benchmarks',
        help='Output directory (default: results/benchmarks)'
    )
    parser.add_argument(
        '--test-files', type=str, default='data/test_files',
        help='Test files directory (default: data/test_files)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from last checkpoint'
    )
    parser.add_argument(
        '--test-mode', action='store_true',
        help='Quick test mode with 10 runs and small files only'
    )
    parser.add_argument(
        '--no-confirm', action='store_true',
        help='Skip confirmation prompt (for background/automated runs)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse algorithms
    if args.algorithms == 'all':
        algorithms = None
    else:
        algorithms = [a.strip() for a in args.algorithms.split(',')]
    
    # Test mode overrides
    if args.test_mode:
        args.runs = 10
        print("=" * 60)
        print("TEST MODE: Running quick verification")
        print(f"  - Iterations: {args.runs}")
        print("=" * 60)
    
    # Check RAPL availability
    print("\nChecking system...")
    if RAPLEnergyMeter.is_available():
        print("[OK] Energy measurement available")
    else:
        print("[!] Hardware RAPL not available - using estimation")
    
    # Check test files
    test_dir = Path(args.test_files)
    test_files = list(test_dir.rglob('*'))
    test_files = [f for f in test_files if f.is_file() and not f.name.startswith('.')]
    print(f"[OK] Found {len(test_files)} test files")
    
    # Get algorithms list
    # 3 algorithms × 49 files × 100 runs = 14,700 measurements
    algo_list = algorithms or DEFAULT_ALGORITHMS
    
    # Show configuration
    total_configs = len(test_files) * len(algo_list)
    total_measurements = total_configs * args.runs
    
    print("\n" + "=" * 60)
    print("BENCHMARK CONFIGURATION")
    print("=" * 60)
    print(f"  Test files: {len(test_files)}")
    print(f"  Algorithms: {algo_list}")
    print(f"  Runs per config: {args.runs}")
    print(f"  Total configurations: {total_configs}")
    print(f"  Total measurements: {total_measurements:,}")
    
    # Estimate time (rough: 0.02s per measurement for small files, more for large)
    est_seconds = total_configs * 1.5  # ~1.5s per config on average
    est_hours = est_seconds / 3600
    print(f"  Estimated time: {est_hours:.1f} hours ({est_seconds/60:.0f} minutes)")
    print("=" * 60)
    
    # Get system info
    sys_info = get_system_info()
    print("\nSystem Information:")
    print(f"  OS: {sys_info['os']} {sys_info['os_release']}")
    print(f"  CPU: {sys_info['cpu']['model']}")
    print(f"  Cores: {sys_info['cpu']['cores']} ({sys_info['cpu']['threads']} threads)")
    print(f"  RAM: {sys_info['ram_total_gb']:.1f} GB")
    
    # Confirmation
    if not args.no_confirm and not args.test_mode and not args.resume:
        print()
        response = input("Start benchmark? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Create and run benchmark
    runner = EnhancedBenchmarkRunner(
        output_dir=args.output,
        runs=args.runs,
        algorithms=algorithms,
        test_files_dir=args.test_files
    )
    
    success = runner.run(resume=args.resume)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
