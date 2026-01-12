#!/usr/bin/env python3
"""
Analyze Results Script

Analyze benchmark results and generate visualizations.

Usage:
    python scripts/analyze_results.py [OPTIONS]

Options:
    --benchmark-results PATH  Path to benchmark results JSON
    --output-dir DIR          Directory for output figures
    --show                    Display plots interactively
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


def analyze_energy_by_algorithm(results: list) -> dict:
    """Analyze energy consumption by algorithm."""
    by_algorithm = {}
    
    for r in results:
        alg = r['algorithm']
        energy = r['statistics']['median_energy_j']
        
        if alg not in by_algorithm:
            by_algorithm[alg] = []
        by_algorithm[alg].append(energy)
    
    # Calculate statistics
    stats = {}
    for alg, energies in by_algorithm.items():
        import statistics
        stats[alg] = {
            'mean': statistics.mean(energies),
            'median': statistics.median(energies),
            'std': statistics.stdev(energies) if len(energies) > 1 else 0,
            'min': min(energies),
            'max': max(energies),
            'count': len(energies),
        }
    
    return stats


def analyze_energy_by_size(results: list) -> dict:
    """Analyze energy consumption by file size."""
    by_size = {}
    
    for r in results:
        size = r['file_size']
        energy = r['statistics']['median_energy_j']
        alg = r['algorithm']
        
        if size not in by_size:
            by_size[size] = {}
        if alg not in by_size[size]:
            by_size[size][alg] = []
        by_size[size][alg].append(energy)
    
    return by_size


def calculate_savings_vs_baseline(results: list, baseline: str = 'AES-256') -> dict:
    """Calculate energy savings compared to baseline."""
    savings = {}
    
    # Group by file
    by_file = {}
    for r in results:
        file_key = r['file_name']
        if file_key not in by_file:
            by_file[file_key] = {}
        by_file[file_key][r['algorithm']] = r['statistics']['median_energy_j']
    
    # Calculate savings
    for file_key, algorithms in by_file.items():
        if baseline not in algorithms:
            continue
        
        baseline_energy = algorithms[baseline]
        
        # Skip if baseline energy is zero or near-zero
        if baseline_energy < 1e-9:
            continue
        
        for alg, energy in algorithms.items():
            if alg == baseline:
                continue
            
            saving_pct = ((baseline_energy - energy) / baseline_energy) * 100
            
            if alg not in savings:
                savings[alg] = []
            savings[alg].append(saving_pct)
    
    # Calculate average savings
    import statistics
    avg_savings = {}
    for alg, values in savings.items():
        avg_savings[alg] = {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
        }
    
    return avg_savings


def generate_plots(results: list, output_dir: str, show: bool = False):
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
    except ImportError as e:
        logger.error(f"Plotting libraries not available: {e}")
        logger.error("Install with: pip install matplotlib seaborn pandas")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    records = []
    for r in results:
        records.append({
            'algorithm': r['algorithm'],
            'file_size': r['file_size'],
            'file_type': r['file_type'],
            'median_energy_j': r['statistics']['median_energy_j'],
            'median_duration_s': r['statistics']['median_duration_s'],
            'throughput_mbps': r['statistics']['mean_throughput_mbps'],
            'energy_per_byte_uj': r['statistics']['energy_per_byte_uj'],
        })
    
    df = pd.DataFrame(records)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('husl')
    
    # 1. Energy vs File Size (Line plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for alg in df['algorithm'].unique():
        alg_data = df[df['algorithm'] == alg]
        alg_data = alg_data.sort_values('file_size')
        ax.plot(
            alg_data['file_size'],
            alg_data['median_energy_j'],
            marker='o',
            label=alg,
            linewidth=2
        )
    
    ax.set_xlabel('File Size (bytes)', fontsize=12)
    ax.set_ylabel('Median Energy (Joules)', fontsize=12)
    ax.set_title('Energy Consumption vs File Size by Algorithm', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'energy_vs_size.png', dpi=150)
    logger.info(f"Saved: {output_path / 'energy_vs_size.png'}")
    
    if show:
        plt.show()
    plt.close()
    
    # 2. Energy Heatmap (Algorithm x File Type)
    pivot = df.groupby(['algorithm', 'file_type'])['median_energy_j'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.4f',
        cmap='YlOrRd',
        ax=ax
    )
    ax.set_title('Average Energy by Algorithm and File Type (Joules)', fontsize=14)
    ax.set_xlabel('File Type', fontsize=12)
    ax.set_ylabel('Algorithm', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path / 'energy_heatmap.png', dpi=150)
    logger.info(f"Saved: {output_path / 'energy_heatmap.png'}")
    
    if show:
        plt.show()
    plt.close()
    
    # 3. Throughput Comparison (Box plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Only use symmetric algorithms
    symmetric_algs = ['AES-128', 'AES-256', 'ChaCha20']
    symmetric_df = df[df['algorithm'].isin(symmetric_algs)]
    
    sns.boxplot(
        data=symmetric_df,
        x='algorithm',
        y='throughput_mbps',
        ax=ax
    )
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Throughput (MB/s)', fontsize=12)
    ax.set_title('Encryption Throughput by Algorithm', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / 'throughput_comparison.png', dpi=150)
    logger.info(f"Saved: {output_path / 'throughput_comparison.png'}")
    
    if show:
        plt.show()
    plt.close()
    
    # 4. Energy per Byte by File Type
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.barplot(
        data=symmetric_df,
        x='file_type',
        y='energy_per_byte_uj',
        hue='algorithm',
        ax=ax
    )
    ax.set_xlabel('File Type', fontsize=12)
    ax.set_ylabel('Energy per Byte (µJ)', fontsize=12)
    ax.set_title('Energy Efficiency by File Type and Algorithm', fontsize=14)
    ax.legend(title='Algorithm')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / 'energy_per_byte.png', dpi=150)
    logger.info(f"Saved: {output_path / 'energy_per_byte.png'}")
    
    if show:
        plt.show()
    plt.close()
    
    logger.info(f"\nAll plots saved to: {output_path}")


def print_analysis_report(results: list):
    """Print analysis report to console."""
    print()
    print("=" * 70)
    print("CRYPTOGREEN BENCHMARK ANALYSIS REPORT")
    print("=" * 70)
    print()
    
    # Basic statistics
    print("DATASET SUMMARY:")
    print(f"  Total benchmark results: {len(results)}")
    
    algorithms = set(r['algorithm'] for r in results)
    file_types = set(r['file_type'] for r in results)
    file_sizes = set(r['file_size'] for r in results)
    
    print(f"  Algorithms tested: {len(algorithms)}")
    print(f"  File types: {len(file_types)}")
    print(f"  File sizes: {len(file_sizes)}")
    print()
    
    # Energy by algorithm
    print("ENERGY BY ALGORITHM (Median Joules):")
    print("-" * 50)
    
    energy_stats = analyze_energy_by_algorithm(results)
    sorted_stats = sorted(energy_stats.items(), key=lambda x: x[1]['median'])
    
    for alg, stats in sorted_stats:
        print(f"  {alg:15s}: {stats['median']:.6f} J (±{stats['std']:.6f})")
    print()
    
    # Savings vs AES-256
    print("ENERGY SAVINGS vs AES-256 BASELINE:")
    print("-" * 50)
    
    savings = calculate_savings_vs_baseline(results, 'AES-256')
    for alg, stats in sorted(savings.items(), key=lambda x: x[1]['mean'], reverse=True):
        direction = "savings" if stats['mean'] > 0 else "increase"
        print(f"  {alg:15s}: {abs(stats['mean']):.1f}% {direction}")
    print()
    
    # Best algorithm by file type
    print("OPTIMAL ALGORITHM BY FILE TYPE:")
    print("-" * 50)
    
    by_type = {}
    for r in results:
        ft = r['file_type']
        alg = r['algorithm']
        energy = r['statistics']['median_energy_j']
        
        if ft not in by_type:
            by_type[ft] = {}
        if alg not in by_type[ft]:
            by_type[ft][alg] = []
        by_type[ft][alg].append(energy)
    
    for ft in sorted(by_type.keys()):
        avg_by_alg = {alg: sum(e)/len(e) for alg, e in by_type[ft].items()}
        # Only symmetric algorithms
        symmetric_only = {k: v for k, v in avg_by_alg.items() if k in ['AES-128', 'AES-256', 'ChaCha20']}
        if symmetric_only:
            best_alg = min(symmetric_only, key=symmetric_only.get)
            print(f"  {ft:10s}: {best_alg}")
    print()
    
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze CryptoGreen benchmark results'
    )
    parser.add_argument(
        '--benchmark-results',
        type=str,
        default=None,
        help='Path to benchmark results JSON'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/figures',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively'
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
        if '*' in args.benchmark_results:
            files = glob.glob(args.benchmark_results)
            if not files:
                logger.error(f"No files match pattern: {args.benchmark_results}")
                sys.exit(1)
            benchmark_file = max(files, key=lambda x: Path(x).stat().st_mtime)
        else:
            benchmark_file = args.benchmark_results
    else:
        # Auto-detect
        pattern = 'results/benchmarks/raw/benchmark_*.json'
        files = glob.glob(pattern)
        files = [f for f in files if 'incremental' not in f]
        
        if not files:
            logger.error(f"No benchmark results found matching {pattern}")
            sys.exit(1)
        
        benchmark_file = max(files, key=lambda x: Path(x).stat().st_mtime)
        logger.info(f"Auto-detected: {benchmark_file}")
    
    # Load results
    logger.info(f"Loading benchmark results from: {benchmark_file}")
    results = load_benchmark_results(benchmark_file)
    logger.info(f"Loaded {len(results)} results")
    
    # Print analysis
    print_analysis_report(results)
    
    # Generate plots
    logger.info("Generating visualizations...")
    generate_plots(results, args.output_dir, show=args.show)
    
    print()
    logger.info("Analysis complete!")


if __name__ == '__main__':
    main()
