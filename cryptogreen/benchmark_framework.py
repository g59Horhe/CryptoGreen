"""
Benchmark Framework Module

This module provides a comprehensive framework for benchmarking cryptographic
algorithms using hardware energy measurement. It supports systematic testing
across multiple algorithms, file types, and sizes.

Example:
    >>> from cryptogreen.benchmark_framework import CryptoBenchmark
    >>> benchmark = CryptoBenchmark()
    >>> results = benchmark.benchmark_algorithm('AES-128', 'test_file.txt', runs=100)
    >>> print(f"Median energy: {results['statistics']['median_energy_j']:.6f} J")
"""

import json
import logging
import os
import platform
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CryptoBenchmark:
    """Framework for benchmarking cryptographic algorithms.
    
    This class provides methods to systematically benchmark cryptographic
    algorithms measuring energy consumption, execution time, and throughput.
    
    Attributes:
        output_dir: Directory to save benchmark results.
        results: List of benchmark results.
        
    Example:
        >>> benchmark = CryptoBenchmark('results/benchmarks')
        >>> benchmark.run_full_benchmark(runs=100)
        >>> benchmark.save_results()
    """
    
    def __init__(self, output_dir: str = 'results/benchmarks'):
        """Initialize benchmark framework.
        
        Args:
            output_dir: Directory to save results (created if not exists).
        """
        from cryptogreen.energy_meter import RAPLEnergyMeter, RAPLNotAvailableError, SoftwareEnergyEstimator
        from cryptogreen.algorithms import CryptoAlgorithms
        
        self.output_dir = Path(output_dir)
        self.results: list[dict] = []
        self.algorithms = CryptoAlgorithms
        
        # Try to initialize RAPL, fall back to software estimation
        try:
            self.energy_meter = RAPLEnergyMeter()
            self.using_hardware_rapl = True
            logger.info("Using hardware RAPL for energy measurement")
        except RAPLNotAvailableError as e:
            logger.warning(f"RAPL not available: {e}")
            logger.warning("Falling back to software energy estimation")
            self.energy_meter = SoftwareEnergyEstimator()
            self.using_hardware_rapl = False
        
        # Create output directories
        self._create_output_dirs()
        
        # Get hardware info once
        self.hardware_info = self.get_hardware_info()
    
    def _create_output_dirs(self) -> None:
        """Create output directory structure."""
        (self.output_dir / 'raw').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'processed').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    
    def benchmark_algorithm(
        self,
        algorithm_name: str,
        file_path: str,
        runs: int = 100,
        warmup_runs: int = 5
    ) -> dict:
        """Benchmark a single algorithm on a file.
        
        Performs multiple runs of the encryption algorithm on the given file,
        measuring energy consumption, execution time, and calculating statistics.
        Uses median (not mean) for primary energy statistics as specified in paper.
        
        Process:
        1. Read file into memory
        2. Warmup: 5 runs to warm CPU caches (discard results)
        3. For each of 100 runs:
           - Measure energy with RAPL (or SoftwareEnergyEstimator fallback)
           - Measure CPU usage with psutil
           - Measure memory usage
           - Calculate throughput
        4. Calculate statistics with 95% confidence intervals
        
        Args:
            algorithm_name: One of ['AES-128', 'AES-256', 'ChaCha20']
            file_path: Path to file to encrypt.
            runs: Number of repetitions (default 100).
            warmup_runs: Number of warmup runs to discard (default 5).
            
        Returns:
            Dict containing:
                - algorithm: Algorithm name
                - file_path: Path to test file
                - file_size: File size in bytes
                - file_type: File extension
                - runs: Number of runs
                - measurements: List of per-run measurements
                - statistics: Computed statistics (median, mean, std, CI, etc.)
                - timestamp: ISO 8601 timestamp
                - hardware: Hardware information
                
        Example:
            >>> result = benchmark.benchmark_algorithm('AES-128', 'test.txt', runs=100)
            >>> print(f"Median energy: {result['statistics']['median_energy_j']:.6f} J")
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # 1. Read file into memory
        data = path.read_bytes()
        file_size = len(data)
        file_type = path.suffix.lstrip('.').lower()
        
        logger.info(f"Benchmarking {algorithm_name} on {path.name} ({file_size:,} bytes)")
        
        # Get the encryption function
        encrypt_func = self._get_encrypt_function(algorithm_name, data)
        
        # Pre-generate keys for this file (don't measure key generation time)
        # Generate fresh keys per file to avoid reuse across different data
        keys = self._prepare_keys(algorithm_name)
        
        # Initialize psutil for accurate CPU measurement
        try:
            import psutil
            process = psutil.Process()
            # Initialize CPU percent baseline (first call returns 0.0)
            process.cpu_percent(interval=None)
            use_psutil = True
        except ImportError:
            use_psutil = False
            logger.warning("psutil not available, CPU/memory metrics will be limited")
        
        # 2. Warmup runs - warm CPU caches, discard results
        logger.debug(f"Performing {warmup_runs} warmup runs...")
        for _ in range(warmup_runs):
            encrypt_func(**keys)
        
        # 3. Measurement runs
        measurements = []
        
        for run_num in range(runs):
            # Get CPU usage (non-blocking, uses last interval)
            if use_psutil:
                cpu_percent = process.cpu_percent(interval=None)
                # Get memory usage
                memory_mb = process.memory_info().rss / (1024 * 1024)
            else:
                cpu_percent = 0.0
                memory_mb = 0.0
            
            # Measure energy and time
            result = self.energy_meter.measure_function(encrypt_func, **keys)
            
            # Calculate throughput (MB/s)
            file_size_mb = file_size / (1024 * 1024)
            throughput_mbps = file_size_mb / result['duration_seconds'] if result['duration_seconds'] > 0 else 0.0
            
            measurement = {
                'run': run_num + 1,
                'energy_joules': result['energy_joules'],
                'duration_seconds': result['duration_seconds'],
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'throughput_mbps': throughput_mbps,
                'average_power_watts': result['average_power_watts'],
            }
            
            measurements.append(measurement)
        
        # 4. Calculate statistics using median (NOT mean) for energy as primary metric
        energy_values = [m['energy_joules'] for m in measurements]
        duration_values = [m['duration_seconds'] for m in measurements]
        throughput_values = [m['throughput_mbps'] for m in measurements]
        power_values = [m['average_power_watts'] for m in measurements]
        cpu_values = [m['cpu_percent'] for m in measurements] if use_psutil else []
        memory_values = [m['memory_mb'] for m in measurements] if use_psutil else []
        
        # Calculate 95% confidence intervals
        energy_ci_lower, energy_ci_upper = self._calculate_confidence_interval(energy_values)
        duration_ci_lower, duration_ci_upper = self._calculate_confidence_interval(duration_values)
        
        stats = {
            # Energy statistics (median is primary metric per paper)
            'median_energy_j': statistics.median(energy_values),
            'mean_energy_j': statistics.mean(energy_values),
            'std_energy_j': statistics.stdev(energy_values) if len(energy_values) > 1 else 0.0,
            'min_energy_j': min(energy_values),
            'max_energy_j': max(energy_values),
            'energy_ci_95_lower': energy_ci_lower,
            'energy_ci_95_upper': energy_ci_upper,
            
            # Duration statistics
            'median_duration_s': statistics.median(duration_values),
            'mean_duration_s': statistics.mean(duration_values),
            'std_duration_s': statistics.stdev(duration_values) if len(duration_values) > 1 else 0.0,
            'min_duration_s': min(duration_values),
            'max_duration_s': max(duration_values),
            'duration_ci_95_lower': duration_ci_lower,
            'duration_ci_95_upper': duration_ci_upper,
            
            # Throughput statistics
            'mean_throughput_mbps': statistics.mean(throughput_values),
            'median_throughput_mbps': statistics.median(throughput_values),
            'std_throughput_mbps': statistics.stdev(throughput_values) if len(throughput_values) > 1 else 0.0,
            
            # Power statistics
            'median_power_w': statistics.median(power_values),
            'mean_power_w': statistics.mean(power_values),
            'std_power_w': statistics.stdev(power_values) if len(power_values) > 1 else 0.0,
            
            # Efficiency metric
            'energy_per_byte_uj': (statistics.median(energy_values) * 1_000_000) / file_size if file_size > 0 else 0.0,
            'energy_per_mb_j': (statistics.median(energy_values) * 1024 * 1024) / file_size if file_size > 0 else 0.0,
        }
        
        # Add CPU/memory stats if available
        if use_psutil and cpu_values:
            stats['mean_cpu_percent'] = statistics.mean(cpu_values)
            stats['median_cpu_percent'] = statistics.median(cpu_values)
        if use_psutil and memory_values:
            stats['mean_memory_mb'] = statistics.mean(memory_values)
            stats['median_memory_mb'] = statistics.median(memory_values)
        
        result = {
            'algorithm': algorithm_name,
            'file_path': str(path.absolute()),
            'file_name': path.name,
            'file_size': file_size,
            'file_type': file_type,
            'runs': runs,
            'warmup_runs': warmup_runs,
            'measurements': measurements,
            'statistics': stats,
            'timestamp': datetime.now().isoformat(),
            'hardware': self.hardware_info,
            'using_hardware_rapl': self.using_hardware_rapl,
        }
        
        logger.info(
            f"  {algorithm_name}: median={stats['median_energy_j']:.6f}J "
            f"[{stats['energy_ci_95_lower']:.6f}-{stats['energy_ci_95_upper']:.6f}], "
            f"time={stats['median_duration_s']:.4f}s, "
            f"throughput={stats['mean_throughput_mbps']:.2f}MB/s"
        )
        
        return result
    
    def _get_encrypt_function(self, algorithm_name: str, data: bytes) -> Callable:
        """Get encryption function for algorithm.
        
        Returns a callable that performs the encryption operation.
        """
        from cryptogreen.algorithms import CryptoAlgorithms
        
        def aes_128_encrypt(**kwargs):
            return CryptoAlgorithms.aes_128_encrypt(data, **kwargs)
        
        def aes_256_encrypt(**kwargs):
            return CryptoAlgorithms.aes_256_encrypt(data, **kwargs)
        
        def chacha20_encrypt(**kwargs):
            return CryptoAlgorithms.chacha20_encrypt(data, **kwargs)
        
        function_map = {
            'AES-128': aes_128_encrypt,
            'AES-256': aes_256_encrypt,
            'ChaCha20': chacha20_encrypt,
        }
        
        if algorithm_name not in function_map:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        return function_map[algorithm_name]
    
    def _calculate_confidence_interval(
        self,
        values: list[float],
        confidence: float = 0.95
    ) -> tuple[float, float]:
        """Calculate confidence interval for a list of values.
        
        Uses t-distribution for more accurate CI with smaller sample sizes.
        
        Args:
            values: List of measurements.
            confidence: Confidence level (default 0.95 for 95% CI).
            
        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        if len(values) < 2:
            mean_val = values[0] if values else 0.0
            return (mean_val, mean_val)
        
        n = len(values)
        mean = statistics.mean(values)
        std_err = statistics.stdev(values) / (n ** 0.5)
        
        # Use t-distribution for more accurate CI
        # For n=100, t-value ≈ 1.984 (close to z=1.96 for normal)
        try:
            import scipy.stats
            t_value = scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
        except ImportError:
            # Fallback: use z-score approximation
            # For 95% CI: z ≈ 1.96
            t_value = 1.96 if confidence == 0.95 else 2.576  # 99% CI
        
        margin_of_error = t_value * std_err
        
        return (mean - margin_of_error, mean + margin_of_error)
    
    def _prepare_keys(self, algorithm_name: str) -> dict:
        """Pre-generate keys for an algorithm.
        
        Keys are generated once and reused across all runs to avoid
        measuring key generation time.
        """
        if algorithm_name == 'AES-128':
            return {'key': os.urandom(16), 'iv': os.urandom(16)}
        elif algorithm_name == 'AES-256':
            return {'key': os.urandom(32), 'iv': os.urandom(16)}
        elif algorithm_name == 'ChaCha20':
            return {'key': os.urandom(32), 'nonce': os.urandom(12)}
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    def run_full_benchmark(
        self,
        test_files_dir: str = 'data/test_files',
        runs: int = 100,
        algorithms: Optional[list[str]] = None,
        save_incremental: bool = True,
        file_sizes: Optional[list[str]] = None,
        output_prefix: str = 'benchmark'
    ) -> None:
        """Run complete benchmark suite.
        
        Benchmarks all algorithms on all test files with specified
        number of repetitions.
        
        Args:
            test_files_dir: Root directory containing test files.
            runs: Repetitions per configuration.
            algorithms: List of algorithms to test (None = all).
            save_incremental: Save results after each file (default True).
            file_sizes: List of file size suffixes to include (e.g., ['64B', '1KB']).
                       None means include all sizes.
            output_prefix: Prefix for output files (default 'benchmark').
            
        Saves:
            - results/benchmarks/raw/{output_prefix}_TIMESTAMP.json
            - results/benchmarks/processed/summary_TIMESTAMP.csv
            - results/benchmarks/logs/{output_prefix}_TIMESTAMP.log
        """
        from cryptogreen.algorithms import CryptoAlgorithms
        from cryptogreen.utils import ProgressTracker
        
        test_dir = Path(test_files_dir)
        
        if not test_dir.exists():
            raise FileNotFoundError(
                f"Test files directory not found: {test_files_dir}\n"
                "Run 'python scripts/generate_test_data.py' first."
            )
        
        # Get all test files
        test_files = list(test_dir.rglob('*'))
        test_files = [f for f in test_files if f.is_file() and not f.name.startswith('.')]
        
        # Filter by file sizes if specified
        if file_sizes:
            test_files = [
                f for f in test_files
                if any(size in f.name for size in file_sizes)
            ]
            logger.info(f"Filtered to {len(test_files)} files matching sizes: {file_sizes}")
        
        if not test_files:
            raise ValueError(f"No test files found in {test_files_dir}")
        
        # Get algorithms
        if algorithms is None:
            algorithms = CryptoAlgorithms.get_algorithm_names()
        
        total_operations = len(test_files) * len(algorithms)
        
        # Setup logging to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_dir / 'logs' / f'{output_prefix}_{timestamp}.log'
        
        # Store output_prefix for save methods
        self._output_prefix = output_prefix
        self._timestamp = timestamp
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        logger.info("=" * 60)
        logger.info("Starting CryptoGreen Benchmark Suite")
        logger.info("=" * 60)
        logger.info(f"Test files: {len(test_files)}")
        logger.info(f"Algorithms: {algorithms}")
        logger.info(f"Runs per config: {runs}")
        logger.info(f"Total operations: {total_operations}")
        logger.info(f"Estimated time: ~{self._estimate_time(total_operations, runs)}")
        logger.info(f"Hardware RAPL: {self.using_hardware_rapl}")
        logger.info("=" * 60)
        
        # Progress tracking
        tracker = ProgressTracker(total_operations, description="Benchmarking")
        
        self.results = []
        
        try:
            for algorithm in algorithms:
                logger.info(f"\n--- Algorithm: {algorithm} ---")
                
                for test_file in test_files:
                    try:
                        result = self.benchmark_algorithm(
                            algorithm,
                            str(test_file),
                            runs=runs
                        )
                        self.results.append(result)
                        
                        if save_incremental:
                            self._save_incremental(timestamp)
                    except Exception as e:
                        logger.error(f"Error benchmarking {algorithm} on {test_file}: {e}")
                    
                    tracker.update(1)
            
            tracker.close()
            
            # Save final results
            self.save_results(timestamp)
            self._generate_summary_csv(timestamp)
            
            logger.info("\n" + "=" * 60)
            logger.info("Benchmark Complete!")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info(f"Total measurements: {len(self.results) * runs}")
            logger.info("=" * 60)
            
        finally:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
    
    def _estimate_time(self, total_operations: int, runs: int) -> str:
        """Estimate benchmark duration."""
        # Rough estimate: 0.5 seconds per run including overhead
        seconds = total_operations * runs * 0.5 / runs  # Simplify to per-operation
        seconds *= runs
        
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            return f"{seconds / 60:.1f} minutes"
        else:
            return f"{seconds / 3600:.1f} hours"
    
    def _save_incremental(self, timestamp: str) -> None:
        """Save incremental results to prevent data loss."""
        prefix = getattr(self, '_output_prefix', 'benchmark')
        incremental_file = self.output_dir / 'raw' / f'{prefix}_{timestamp}_incremental.json'
        
        with open(incremental_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save benchmark results to JSON file.
        
        Args:
            filename: Optional filename (without extension).
            
        Returns:
            Path to saved file.
        """
        prefix = getattr(self, '_output_prefix', 'benchmark')
        
        if filename is None:
            filename = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        output_file = self.output_dir / 'raw' / f'{prefix}_{filename}.json'
        
        # Add metadata
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_results': len(self.results),
                'hardware': self.hardware_info,
                'using_hardware_rapl': self.using_hardware_rapl,
            },
            'results': self.results,
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_file}")
        return str(output_file)
    
    def _generate_summary_csv(self, timestamp: str) -> str:
        """Generate summary CSV from results.
        
        Returns:
            Path to saved CSV file.
        """
        import csv
        
        output_file = self.output_dir / 'processed' / f'summary_{timestamp}.csv'
        
        fieldnames = [
            'algorithm', 'file_name', 'file_size', 'file_type',
            'runs', 'median_energy_j', 'mean_energy_j', 'std_energy_j',
            'median_duration_s', 'mean_throughput_mbps', 'energy_per_byte_uj',
            'median_power_w', 'timestamp'
        ]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'algorithm': result['algorithm'],
                    'file_name': result['file_name'],
                    'file_size': result['file_size'],
                    'file_type': result['file_type'],
                    'runs': result['runs'],
                    'median_energy_j': result['statistics']['median_energy_j'],
                    'mean_energy_j': result['statistics']['mean_energy_j'],
                    'std_energy_j': result['statistics']['std_energy_j'],
                    'median_duration_s': result['statistics']['median_duration_s'],
                    'mean_throughput_mbps': result['statistics']['mean_throughput_mbps'],
                    'energy_per_byte_uj': result['statistics']['energy_per_byte_uj'],
                    'median_power_w': result['statistics']['median_power_w'],
                    'timestamp': result['timestamp'],
                }
                writer.writerow(row)
        
        logger.info(f"Summary CSV saved to: {output_file}")
        return str(output_file)
    
    def get_hardware_info(self) -> dict:
        """Collect hardware information for reproducibility.
        
        Returns:
            Dict with CPU model, cores, features, kernel, etc.
        """
        info = {
            'cpu_model': 'Unknown',
            'cpu_cores': os.cpu_count() or 1,
            'cpu_threads': os.cpu_count() or 1,
            'has_aes_ni': False,
            'has_arm_crypto': False,
            'ram_total_gb': 0.0,
            'kernel': 'Unknown',
            'os': platform.system(),
            'os_version': platform.version(),
            'python_version': platform.python_version(),
            'cpu_governor': 'unknown',
        }
        
        # Windows-specific detection
        if platform.system() == 'Windows':
            info = self._get_windows_system_info(info)
        
        # Read /proc/cpuinfo on Linux
        cpuinfo_path = Path('/proc/cpuinfo')
        if cpuinfo_path.exists():
            try:
                content = cpuinfo_path.read_text()
                
                # Get model name
                model_match = re.search(r'model name\s*:\s*(.+)', content, re.IGNORECASE)
                if model_match:
                    info['cpu_model'] = model_match.group(1).strip()
                
                # Check for AES-NI
                flags_match = re.search(r'flags\s*:\s*(.+)', content.lower())
                if flags_match and 'aes' in flags_match.group(1):
                    info['has_aes_ni'] = True
                
                # Count cores
                core_ids = re.findall(r'core id\s*:\s*(\d+)', content)
                if core_ids:
                    info['cpu_cores'] = len(set(core_ids))
                    info['cpu_threads'] = len(core_ids)
            except Exception as e:
                logger.debug(f"Error reading /proc/cpuinfo: {e}")
        
        # Get kernel version
        if platform.system() == 'Linux':
            try:
                import subprocess
                result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
                info['kernel'] = result.stdout.strip()
            except Exception:
                pass
        
        # Get RAM info
        try:
            import psutil
            info['ram_total_gb'] = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            pass
        
        # Get CPU governor
        governor_path = Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')
        if governor_path.exists():
            try:
                info['cpu_governor'] = governor_path.read_text().strip()
            except Exception:
                pass
        
        return info
    
    def _get_windows_system_info(self, info: dict) -> dict:
        """Get system information on Windows.
        
        Args:
            info: Base info dict to populate.
            
        Returns:
            Updated info dict with Windows-specific data.
        """
        import subprocess
        
        # Get CPU model using WMIC
        try:
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'name'],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            if len(lines) > 1:
                info['cpu_model'] = lines[1]
        except Exception as e:
            logger.debug(f"WMIC CPU query failed: {e}")
            # Fallback to platform.processor()
            proc = platform.processor()
            if proc:
                info['cpu_model'] = proc
        
        # Get physical/logical cores
        try:
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'NumberOfCores,NumberOfLogicalProcessors'],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 2:
                    info['cpu_cores'] = int(parts[0])
                    info['cpu_threads'] = int(parts[1])
        except Exception as e:
            logger.debug(f"WMIC core count failed: {e}")
        
        # Check for AES-NI by running a quick encryption test
        info['has_aes_ni'] = self._check_aes_ni_windows()
        
        # Get kernel/OS version
        info['kernel'] = platform.version()
        info['cpu_governor'] = 'N/A (Windows)'
        
        return info
    
    def _check_aes_ni_windows(self) -> bool:
        """Check if AES-NI is available on Windows by testing encryption speed.
        
        Returns:
            True if AES-NI appears to be available.
        """
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            import time
            
            # Generate test data
            key = os.urandom(32)
            iv = os.urandom(16)
            data = os.urandom(1024 * 1024)  # 1MB
            
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            # Time the encryption
            start = time.perf_counter()
            _ = encryptor.update(data)
            elapsed = time.perf_counter() - start
            
            # If 1MB encrypts in less than 20ms, likely hardware-accelerated
            # Software AES on modern CPUs is ~50-100ms for 1MB
            has_aesni = elapsed < 0.02
            logger.debug(f"AES-NI check: 1MB encrypted in {elapsed*1000:.2f}ms -> {'Yes' if has_aesni else 'No'}")
            return has_aesni
            
        except Exception as e:
            logger.debug(f"AES-NI check failed: {e}")
            return False

    def load_results(self, file_path: str) -> list[dict]:
        """Load benchmark results from JSON file.
        
        Args:
            file_path: Path to JSON results file.
            
        Returns:
            List of benchmark results.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'results' in data:
            self.results = data['results']
        else:
            self.results = data
        
        logger.info(f"Loaded {len(self.results)} results from {file_path}")
        return self.results
    
    def get_optimal_algorithm(
        self,
        file_size: int,
        file_type: str
    ) -> Optional[str]:
        """Get optimal algorithm for given file characteristics.
        
        Uses benchmark data to determine the most energy-efficient algorithm.
        
        Args:
            file_size: File size in bytes.
            file_type: File extension (lowercase, no dot).
            
        Returns:
            Algorithm name with lowest median energy, or None if no data.
        """
        if not self.results:
            return None
        
        # Find matching results
        matching = [r for r in self.results 
                   if r['file_size'] == file_size and r['file_type'] == file_type]
        
        if not matching:
            # Find closest file size
            size_diffs = [(abs(r['file_size'] - file_size), r) for r in self.results
                         if r['file_type'] == file_type]
            if size_diffs:
                size_diffs.sort(key=lambda x: x[0])
                matching = [r for _, r in size_diffs[:6]]  # Closest 6 results
        
        if not matching:
            return None
        
        # Group by algorithm and find minimum energy
        by_algorithm = {}
        for r in matching:
            alg = r['algorithm']
            energy = r['statistics']['median_energy_j']
            if alg not in by_algorithm or energy < by_algorithm[alg]:
                by_algorithm[alg] = energy
        
        optimal = min(by_algorithm, key=by_algorithm.get)
        return optimal
