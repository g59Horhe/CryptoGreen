#!/usr/bin/env python3
"""
Fixed benchmark that handles small files correctly with batching
"""

import time
import json
import statistics
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def benchmark_file(file_path, algorithm, runs=100):
    """Benchmark with smart batching for small files"""
    
    from cryptogreen.energy_meter import RAPLEnergyMeter
    from cryptogreen.algorithms import CryptoAlgorithms
    
    # Read file
    data = Path(file_path).read_bytes()
    file_size = len(data)
    
    # Initialize
    meter = RAPLEnergyMeter()
    crypto = CryptoAlgorithms()
    
    # Determine batch size based on file size
    # Goal: Each measurement should take at least 10ms for reliable RAPL reading
    if file_size < 1000:  # < 1KB
        batch_size = 1000
    elif file_size < 10000:  # < 10KB
        batch_size = 100
    elif file_size < 100000:  # < 100KB
        batch_size = 10
    else:
        batch_size = 1
    
    print(f"  File size: {file_size:,} bytes, Batch size: {batch_size}")
    
    # Get encryption function
    if algorithm == 'AES-128':
        encrypt_func = lambda: crypto.aes_128_encrypt(data)
    elif algorithm == 'AES-256':
        encrypt_func = lambda: crypto.aes_256_encrypt(data)
    elif algorithm == 'ChaCha20':
        encrypt_func = lambda: crypto.chacha20_encrypt(data)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Wrap in batch if needed
    if batch_size > 1:
        original_func = encrypt_func
        encrypt_func = lambda: [original_func() for _ in range(batch_size)]
    
    # Warmup runs
    logger.info(f"  Warmup: 5 runs")
    for _ in range(5):
        encrypt_func()
    
    # Measure
    logger.info(f"  Measuring: {runs} runs")
    measurements = []
    zero_count = 0
    
    for i in range(runs):
        result = meter.measure_function(encrypt_func)
        
        # Adjust for batching
        energy_j = result['energy_joules'] / batch_size
        duration_s = result['duration_seconds'] / batch_size
        
        if energy_j == 0:
            zero_count += 1
        
        measurements.append({
            'run': i + 1,
            'energy_j': energy_j,
            'duration_s': duration_s,
            'energy_uj': int(energy_j * 1_000_000),
        })
        
        # Progress
        if (i + 1) % 10 == 0:
            logger.info(f"    Progress: {i+1}/{runs} runs, zeros: {zero_count}")
    
    # Calculate statistics
    energies = [m['energy_j'] for m in measurements if m['energy_j'] > 0]
    durations = [m['duration_s'] for m in measurements if m['energy_j'] > 0]
    
    if not energies:
        logger.warning(f"    ❌ ALL MEASUREMENTS ZERO!")
        median_energy = 0
        mean_energy = 0
        std_energy = 0
    else:
        median_energy = statistics.median(energies)
        mean_energy = statistics.mean(energies)
        std_energy = statistics.stdev(energies) if len(energies) > 1 else 0
    
    median_duration = statistics.median(durations) if durations else 0
    throughput = file_size / median_duration / 1024 / 1024 if median_duration > 0 else 0
    
    logger.info(f"    Median energy: {median_energy:.9f} J")
    logger.info(f"    Zero measurements: {zero_count}/{runs} ({zero_count/runs*100:.1f}%)")
    
    return {
        'algorithm': algorithm,
        'file_path': str(file_path),
        'file_size': file_size,
        'file_type': file_path.suffix[1:] if file_path.suffix else 'unknown',
        'batch_size': batch_size,
        'runs': runs,
        'timestamp': datetime.now().isoformat(),
        'measurements': measurements,
        'statistics': {
            'median_energy_j': median_energy,
            'mean_energy_j': mean_energy,
            'std_energy_j': std_energy,
            'median_duration_s': median_duration,
            'throughput_mbps': throughput,
            'zero_count': zero_count,
            'zero_percentage': zero_count / runs * 100,
        }
    }

def main():
    """Run fixed benchmark on all test files"""
    
    test_files_dir = Path("data/test_files")
    output_dir = Path("results/benchmarks_fixed")
    raw_dir = output_dir / "raw"
    
    # Create directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all test files
    test_files = sorted(test_files_dir.rglob("*.*"))
    # Filter out hidden files and directories
    test_files = [f for f in test_files if f.is_file() and not f.name.startswith('.')]
    
    algorithms = ['AES-128', 'AES-256', 'ChaCha20']
    
    print("=" * 70)
    print("CRYPTOGREEN BENCHMARK - FIXED VERSION")
    print("=" * 70)
    print(f"Test files: {len(test_files)}")
    print(f"Algorithms: {len(algorithms)}")
    print(f"Total configurations: {len(test_files) * len(algorithms)}")
    print(f"Runs per config: 100")
    print(f"Total measurements: {len(test_files) * len(algorithms) * 100}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    print()
    
    # Confirm
    try:
        response = input("Start benchmark? This will take 2-4 hours. (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    except KeyboardInterrupt:
        print("\nCancelled.")
        return
    
    results = []
    total = len(test_files) * len(algorithms)
    count = 0
    start_time = time.time()
    
    for file_path in test_files:
        for algorithm in algorithms:
            count += 1
            elapsed = time.time() - start_time
            eta = (elapsed / count * total) - elapsed if count > 0 else 0
            
            print()
            print("=" * 70)
            print(f"[{count}/{total}] {file_path.name} - {algorithm}")
            print(f"Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
            print("=" * 70)
            
            try:
                result = benchmark_file(file_path, algorithm, runs=100)
                results.append(result)
                
                # Save individual result
                safe_filename = f"{file_path.stem}_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                output_file = raw_dir / safe_filename
                
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"  Saved: {output_file.name}")
                
            except Exception as e:
                logger.error(f"  ❌ ERROR: {e}", exc_info=True)
    
    # Final summary
    total_time = time.time() - start_time
    
    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE!")
    print("=" * 70)
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Total configs: {len(results)}")
    
    # Calculate summary statistics
    zero_configs = sum(1 for r in results if r['statistics']['zero_count'] == r['runs'])
    print(f"Configs with all zeros: {zero_configs}/{len(results)} ({zero_configs/len(results)*100:.1f}%)")
    
    energies = [r['statistics']['median_energy_j'] for r in results if r['statistics']['median_energy_j'] > 0]
    if energies:
        print(f"Energy range: {min(energies):.9f} - {max(energies):.6f} J")
        print(f"Median energy: {statistics.median(energies):.6f} J")
        print(f"Mean energy: {statistics.mean(energies):.6f} J")
    
    # Save summary
    summary = {
        'total_configs': len(results),
        'total_time_seconds': total_time,
        'zero_configs': zero_configs,
        'timestamp': datetime.now().isoformat(),
        'energy_statistics': {
            'min': min(energies) if energies else 0,
            'max': max(energies) if energies else 0,
            'median': statistics.median(energies) if energies else 0,
            'mean': statistics.mean(energies) if energies else 0,
        } if energies else {}
    }
    
    summary_file = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"Raw results in: {raw_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
