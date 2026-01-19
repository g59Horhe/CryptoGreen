#!/usr/bin/env python3
"""
Process Benchmark Results

Combines all benchmark JSON files, calculates statistics, identifies optimal
algorithms, and prepares data for ML training.

Usage:
    python scripts/process_results.py [--input-dir DIR] [--output-dir DIR]

Outputs:
    - results/benchmarks/processed/benchmark_summary.csv
    - data/ml_data/labels.csv
    - data/ml_data/features.csv
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import math

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import numpy/scipy for statistics, fall back to stdlib
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available, using basic statistics")

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, confidence intervals will be approximate")


def calculate_statistics(values: list) -> dict:
    """Calculate comprehensive statistics for a list of values."""
    if not values:
        return {
            'median': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0,
            'ci_lower': 0, 'ci_upper': 0, 'count': 0
        }
    
    n = len(values)
    
    if HAS_NUMPY:
        arr = np.array(values)
        median = float(np.median(arr))
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
    else:
        sorted_vals = sorted(values)
        median = sorted_vals[n // 2] if n % 2 else (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0
        std = math.sqrt(variance)
        min_val = min(values)
        max_val = max(values)
    
    # 95% confidence interval
    if n > 1:
        if HAS_SCIPY:
            ci = scipy_stats.t.interval(0.95, n-1, loc=mean, scale=std/math.sqrt(n))
            ci_lower, ci_upper = ci
        else:
            # Approximate using t-distribution critical value for 95% CI
            # For large n, t ~ 1.96
            t_val = 1.96 if n > 30 else 2.0
            margin = t_val * std / math.sqrt(n)
            ci_lower = mean - margin
            ci_upper = mean + margin
    else:
        ci_lower = ci_upper = mean
    
    return {
        'median': median,
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'count': n
    }


def parse_file_size(filename: str) -> tuple[int, str]:
    """Extract file size in bytes and size category from filename.
    
    Examples:
        txt_64B.txt -> (64, '64B')
        jpg_1KB.jpg -> (1024, '1KB')
        pdf_10MB.pdf -> (10485760, '10MB')
    """
    # Size mappings
    size_map = {
        '64B': 64,
        '1KB': 1024,
        '10KB': 10 * 1024,
        '100KB': 100 * 1024,
        '1MB': 1024 * 1024,
        '10MB': 10 * 1024 * 1024,
        '100MB': 100 * 1024 * 1024,
    }
    
    for size_str, size_bytes in size_map.items():
        if size_str in filename:
            return size_bytes, size_str
    
    return 0, 'unknown'


def load_benchmark_files(input_dir: Path) -> list[dict]:
    """Load all benchmark JSON files from directory."""
    all_results = []
    
    # Find all non-incremental, non-test benchmark files
    json_files = list(input_dir.glob('benchmark_*.json'))
    json_files = [f for f in json_files if 'incremental' not in f.name and 'test_' not in f.name]
    
    print(f"\nFound {len(json_files)} benchmark files:")
    
    for json_file in sorted(json_files):
        print(f"  - {json_file.name}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            print(f"    Loaded {len(results)} results")
            
            # Add source file info
            for result in results:
                result['_source_file'] = json_file.name
            
            all_results.extend(results)
            
        except Exception as e:
            print(f"    Error loading: {e}")
    
    return all_results


def process_results(results: list[dict]) -> dict:
    """Process results and calculate statistics by configuration.
    
    Returns:
        Dictionary keyed by (algorithm, file_type, file_size) with statistics.
    """
    # Group results by configuration
    configs = {}
    
    for result in results:
        algorithm = result.get('algorithm', '')
        file_type = result.get('file_type', '')
        file_name = result.get('file_name', '')
        file_size = result.get('file_size', 0)
        
        # Get size category from filename
        _, size_category = parse_file_size(file_name)
        
        key = (algorithm, file_type, size_category)
        
        if key not in configs:
            configs[key] = {
                'algorithm': algorithm,
                'file_type': file_type,
                'file_size_bytes': file_size,
                'file_size_category': size_category,
                'energy_values': [],
                'time_values': [],
                'throughput_values': [],
                'measurements': []
            }
        
        # Add statistics from this result
        stats = result.get('statistics', {})
        
        # Collect individual measurement values if available
        measurements = result.get('measurements', [])
        for m in measurements:
            configs[key]['energy_values'].append(m.get('energy_joules', 0))
            configs[key]['time_values'].append(m.get('duration_seconds', 0))
        
        # Also collect aggregate stats as backup
        configs[key]['measurements'].append({
            'median_energy': stats.get('median_energy_j', 0),
            'mean_energy': stats.get('mean_energy_j', 0),
            'std_energy': stats.get('std_energy_j', 0),
            'median_time': stats.get('median_duration_s', 0),
            'throughput': stats.get('mean_throughput_mbps', 0),
        })
    
    # Calculate final statistics for each configuration
    processed = {}
    
    for key, config in configs.items():
        energy_vals = config['energy_values']
        time_vals = config['time_values']
        
        # If we have individual measurements, use them
        if energy_vals:
            energy_stats = calculate_statistics(energy_vals)
            time_stats = calculate_statistics(time_vals)
        else:
            # Fall back to aggregated stats
            energy_vals = [m['median_energy'] for m in config['measurements']]
            time_vals = [m['median_time'] for m in config['measurements']]
            energy_stats = calculate_statistics(energy_vals)
            time_stats = calculate_statistics(time_vals)
        
        # Calculate throughput
        file_size = config['file_size_bytes']
        median_time = time_stats['median']
        throughput = (file_size / (1024 * 1024)) / median_time if median_time > 0 else 0
        
        # Energy per byte
        energy_per_byte = (energy_stats['median'] * 1_000_000) / file_size if file_size > 0 else 0
        
        processed[key] = {
            'algorithm': config['algorithm'],
            'file_type': config['file_type'],
            'file_size_bytes': file_size,
            'file_size_category': config['file_size_category'],
            'median_energy_j': energy_stats['median'],
            'mean_energy_j': energy_stats['mean'],
            'std_energy_j': energy_stats['std'],
            'ci_lower': energy_stats['ci_lower'],
            'ci_upper': energy_stats['ci_upper'],
            'median_time_s': time_stats['median'],
            'mean_time_s': time_stats['mean'],
            'std_time_s': time_stats['std'],
            'throughput_mbps': throughput,
            'energy_per_byte_uj': energy_per_byte,
            'measurement_count': energy_stats['count'],
        }
    
    return processed


def find_optimal_algorithms(processed: dict) -> dict:
    """Find the lowest-energy algorithm for each file_type/size combination."""
    # Group by file_type and size
    by_config = {}
    
    for key, data in processed.items():
        algorithm, file_type, size_category = key
        config_key = (file_type, size_category)
        
        if config_key not in by_config:
            by_config[config_key] = []
        
        by_config[config_key].append({
            'algorithm': algorithm,
            'median_energy_j': data['median_energy_j'],
            'throughput_mbps': data['throughput_mbps'],
        })
    
    # Find optimal for each configuration
    optimal = {}
    
    for config_key, algorithms in by_config.items():
        file_type, size_category = config_key
        
        # Sort by energy (lowest first)
        sorted_algos = sorted(algorithms, key=lambda x: x['median_energy_j'])
        
        best = sorted_algos[0]
        second_best = sorted_algos[1] if len(sorted_algos) > 1 else None
        
        # Calculate energy savings
        if second_best:
            savings_pct = ((second_best['median_energy_j'] - best['median_energy_j']) 
                          / second_best['median_energy_j'] * 100)
        else:
            savings_pct = 0
        
        optimal[config_key] = {
            'file_type': file_type,
            'file_size': size_category,
            'optimal_algorithm': best['algorithm'],
            'optimal_energy_j': best['median_energy_j'],
            'optimal_throughput_mbps': best['throughput_mbps'],
            'second_best_algorithm': second_best['algorithm'] if second_best else '',
            'energy_savings_pct': savings_pct,
            'all_algorithms': sorted_algos,
        }
    
    return optimal


def save_summary_csv(processed: dict, output_path: Path):
    """Save processed results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by file_type, size, algorithm
    size_order = {'64B': 0, '1KB': 1, '10KB': 2, '100KB': 3, '1MB': 4, '10MB': 5, '100MB': 6}
    
    sorted_data = sorted(
        processed.values(),
        key=lambda x: (x['file_type'], size_order.get(x['file_size_category'], 99), x['algorithm'])
    )
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'algorithm', 'file_type', 'file_size', 'file_size_bytes',
            'median_energy_j', 'mean_energy_j', 'std_energy_j', 
            'ci_lower', 'ci_upper',
            'median_time_s', 'throughput_mbps', 
            'energy_per_byte_uj', 'measurement_count'
        ])
        
        # Data
        for data in sorted_data:
            writer.writerow([
                data['algorithm'],
                data['file_type'],
                data['file_size_category'],
                data['file_size_bytes'],
                f"{data['median_energy_j']:.9f}",
                f"{data['mean_energy_j']:.9f}",
                f"{data['std_energy_j']:.9f}",
                f"{data['ci_lower']:.9f}",
                f"{data['ci_upper']:.9f}",
                f"{data['median_time_s']:.9f}",
                f"{data['throughput_mbps']:.3f}",
                f"{data['energy_per_byte_uj']:.6f}",
                data['measurement_count']
            ])
    
    print(f"\nSaved summary to: {output_path}")
    print(f"  Total configurations: {len(sorted_data)}")


def save_labels_csv(optimal: dict, output_path: Path):
    """Save optimal algorithm labels for ML training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by file_type and size
    size_order = {'64B': 0, '1KB': 1, '10KB': 2, '100KB': 3, '1MB': 4, '10MB': 5, '100MB': 6}
    
    sorted_data = sorted(
        optimal.values(),
        key=lambda x: (x['file_type'], size_order.get(x['file_size'], 99))
    )
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'file_type', 'file_size', 'optimal_algorithm',
            'optimal_energy_j', 'optimal_throughput_mbps',
            'second_best', 'energy_savings_pct'
        ])
        
        # Data
        for data in sorted_data:
            writer.writerow([
                data['file_type'],
                data['file_size'],
                data['optimal_algorithm'],
                f"{data['optimal_energy_j']:.9f}",
                f"{data['optimal_throughput_mbps']:.3f}",
                data['second_best_algorithm'],
                f"{data['energy_savings_pct']:.2f}"
            ])
    
    print(f"Saved labels to: {output_path}")
    print(f"  Total configurations: {len(sorted_data)}")


def save_features_csv(processed: dict, output_path: Path):
    """Save feature data for ML training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create feature rows for each file_type/size combination
    features = {}
    
    for key, data in processed.items():
        algorithm, file_type, size_category = key
        feature_key = (file_type, size_category)
        
        if feature_key not in features:
            features[feature_key] = {
                'file_type': file_type,
                'file_size': size_category,
                'file_size_bytes': data['file_size_bytes'],
            }
        
        # Add algorithm-specific columns
        prefix = algorithm.lower().replace('-', '_')
        features[feature_key][f'{prefix}_energy_j'] = data['median_energy_j']
        features[feature_key][f'{prefix}_time_s'] = data['median_time_s']
        features[feature_key][f'{prefix}_throughput_mbps'] = data['throughput_mbps']
    
    # Sort by file_type and size
    size_order = {'64B': 0, '1KB': 1, '10KB': 2, '100KB': 3, '1MB': 4, '10MB': 5, '100MB': 6}
    sorted_data = sorted(
        features.values(),
        key=lambda x: (x['file_type'], size_order.get(x['file_size'], 99))
    )
    
    if not sorted_data:
        print("No feature data to save")
        return
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted_data[0].keys())
        writer.writeheader()
        writer.writerows(sorted_data)
    
    print(f"Saved features to: {output_path}")
    print(f"  Total rows: {len(sorted_data)}")


def print_summary_statistics(processed: dict, optimal: dict):
    """Print summary statistics to console."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY STATISTICS")
    print("=" * 80)
    
    # Count configurations
    algorithms = set()
    file_types = set()
    file_sizes = set()
    
    for key in processed.keys():
        algorithm, file_type, size = key
        algorithms.add(algorithm)
        file_types.add(file_type)
        file_sizes.add(size)
    
    total_measurements = sum(d['measurement_count'] for d in processed.values())
    
    print(f"\nDataset Overview:")
    print(f"  Algorithms: {len(algorithms)} - {sorted(algorithms)}")
    print(f"  File types: {len(file_types)} - {sorted(file_types)}")
    print(f"  File sizes: {len(file_sizes)} - {sorted(file_sizes, key=lambda x: {'64B':0,'1KB':1,'10KB':2,'100KB':3,'1MB':4,'10MB':5,'100MB':6}.get(x,99))}")
    print(f"  Configurations: {len(processed)}")
    print(f"  Total measurements: {total_measurements:,}")
    
    # Algorithm summary
    print(f"\n{'='*80}")
    print("ALGORITHM PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    for algo in sorted(algorithms):
        algo_data = [d for k, d in processed.items() if k[0] == algo]
        
        if not algo_data:
            continue
        
        avg_energy = sum(d['median_energy_j'] for d in algo_data) / len(algo_data)
        avg_throughput = sum(d['throughput_mbps'] for d in algo_data) / len(algo_data)
        
        # Count how often this algorithm is optimal
        optimal_count = sum(1 for o in optimal.values() if o['optimal_algorithm'] == algo)
        
        print(f"\n{algo}:")
        print(f"  Average energy: {avg_energy:.6f} J")
        print(f"  Average throughput: {avg_throughput:.2f} MB/s")
        print(f"  Optimal for: {optimal_count}/{len(optimal)} configurations ({optimal_count/len(optimal)*100:.1f}%)")
    
    # Optimal algorithm breakdown
    print(f"\n{'='*80}")
    print("OPTIMAL ALGORITHM BY FILE TYPE/SIZE")
    print(f"{'='*80}")
    
    # Group by file type
    by_type = {}
    for config, data in optimal.items():
        ft = data['file_type']
        if ft not in by_type:
            by_type[ft] = []
        by_type[ft].append(data)
    
    for file_type in sorted(by_type.keys()):
        print(f"\n{file_type.upper()}:")
        
        size_order = {'64B': 0, '1KB': 1, '10KB': 2, '100KB': 3, '1MB': 4, '10MB': 5, '100MB': 6}
        sorted_configs = sorted(by_type[file_type], key=lambda x: size_order.get(x['file_size'], 99))
        
        for config in sorted_configs:
            print(f"  {config['file_size']:>6s}: {config['optimal_algorithm']:<12s} "
                  f"({config['optimal_energy_j']:.6f} J, {config['optimal_throughput_mbps']:.1f} MB/s) "
                  f"[saves {config['energy_savings_pct']:.1f}% vs {config['second_best_algorithm']}]")
    
    # Overall optimal algorithm counts
    print(f"\n{'='*80}")
    print("OVERALL OPTIMAL ALGORITHM DISTRIBUTION")
    print(f"{'='*80}")
    
    algo_counts = {}
    for data in optimal.values():
        algo = data['optimal_algorithm']
        algo_counts[algo] = algo_counts.get(algo, 0) + 1
    
    for algo, count in sorted(algo_counts.items(), key=lambda x: -x[1]):
        pct = count / len(optimal) * 100
        bar = '#' * int(pct / 2)
        print(f"  {algo:<12s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Process CryptoGreen benchmark results'
    )
    parser.add_argument(
        '--input-dir', type=str, default='results/benchmarks/raw',
        help='Directory containing benchmark JSON files'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/benchmarks/processed',
        help='Directory for processed output files'
    )
    parser.add_argument(
        '--ml-dir', type=str, default='data/ml_data',
        help='Directory for ML training data'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ml_dir = Path(args.ml_dir)
    
    print("=" * 80)
    print("CryptoGreen Results Processor")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"ML data directory: {ml_dir}")
    
    # Load all benchmark files
    results = load_benchmark_files(input_dir)
    
    if not results:
        print("\nNo benchmark results found!")
        sys.exit(1)
    
    print(f"\nTotal results loaded: {len(results)}")
    
    # Process results
    print("\nProcessing results...")
    processed = process_results(results)
    print(f"Processed {len(processed)} configurations")
    
    # Find optimal algorithms
    print("\nFinding optimal algorithms...")
    optimal = find_optimal_algorithms(processed)
    print(f"Found optimal for {len(optimal)} file type/size combinations")
    
    # Save outputs
    print("\nSaving outputs...")
    
    summary_csv = output_dir / 'benchmark_summary.csv'
    save_summary_csv(processed, summary_csv)
    
    labels_csv = ml_dir / 'labels.csv'
    save_labels_csv(optimal, labels_csv)
    
    features_csv = ml_dir / 'features.csv'
    save_features_csv(processed, features_csv)
    
    # Print summary
    print_summary_statistics(processed, optimal)
    
    print("\n[DONE] Results processing complete!")
    print(f"  - Summary CSV: {summary_csv}")
    print(f"  - Labels CSV: {labels_csv}")
    print(f"  - Features CSV: {features_csv}")


if __name__ == '__main__':
    main()
