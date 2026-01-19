#!/usr/bin/env python
"""
Energy Savings Calculator

Calculates energy savings achieved by using optimal algorithm selection
compared to a baseline (AES-256).

Usage:
    python scripts/calculate_savings.py
    python scripts/calculate_savings.py --baseline AES-128
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Size categories
SIZE_CATEGORIES = {
    'tiny': (0, 1024),           # 0 - 1KB
    'small': (1024, 10240),      # 1KB - 10KB
    'medium': (10240, 1048576),  # 10KB - 1MB
    'large': (1048576, 10485760),  # 1MB - 10MB
    'huge': (10485760, float('inf')),  # 10MB+
}


def load_benchmark_data(benchmark_dir: str = 'results/benchmarks/raw') -> pd.DataFrame:
    """Load benchmark results from JSON."""
    benchmark_path = Path(benchmark_dir)
    json_files = list(benchmark_path.glob('benchmark_*.json'))
    json_files = [f for f in json_files if 'incremental' not in f.name]
    
    if not json_files:
        raise FileNotFoundError(f"No benchmark files found in {benchmark_dir}")
    
    latest = max(json_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Loading benchmark data from: {latest}")
    
    with open(latest, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    # Convert to DataFrame
    rows = []
    for r in results:
        rows.append({
            'algorithm': r['algorithm'],
            'file_name': r['file_name'],
            'file_type': r['file_type'],
            'file_size': r['file_size'],
            'median_energy_j': r['statistics']['median_energy_j'],
            'mean_energy_j': r['statistics']['mean_energy_j'],
        })
    
    return pd.DataFrame(rows)


def load_labels(labels_path: str = 'data/ml_data/labels.csv') -> pd.DataFrame:
    """Load optimal algorithm labels."""
    return pd.read_csv(labels_path)


def get_size_category(size_bytes: int) -> str:
    """Get size category for a file size."""
    for category, (min_size, max_size) in SIZE_CATEGORIES.items():
        if min_size <= size_bytes < max_size:
            return category
    return 'huge'


def calculate_savings(
    benchmark_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    baseline_algorithm: str = 'AES-256'
) -> pd.DataFrame:
    """Calculate energy savings for each configuration.
    
    Args:
        benchmark_df: DataFrame with benchmark results
        labels_df: DataFrame with optimal algorithm labels
        baseline_algorithm: Algorithm to compare against
        
    Returns:
        DataFrame with savings calculations
    """
    results = []
    
    # Get unique file configurations
    file_configs = benchmark_df.groupby(['file_type', 'file_size']).first().reset_index()
    
    for _, row in file_configs.iterrows():
        file_type = row['file_type']
        file_size = row['file_size']
        
        # Build config name to match labels
        size_map = {
            64: '64',
            1024: '1024',
            10240: '10240',
            102400: '102400',
            1048576: '1048576',
            10485760: '10485760',
            104857600: '104857600',
        }
        size_str = size_map.get(file_size, str(file_size))
        config = f"{file_type}_{size_str}"
        
        # Get optimal algorithm from labels
        label_row = labels_df[labels_df['config'] == config]
        if label_row.empty:
            logger.warning(f"No label found for config: {config}")
            continue
        
        optimal_algorithm = label_row['optimal_algorithm'].values[0]
        
        # Get energy for optimal algorithm
        optimal_data = benchmark_df[
            (benchmark_df['file_type'] == file_type) &
            (benchmark_df['file_size'] == file_size) &
            (benchmark_df['algorithm'] == optimal_algorithm)
        ]
        
        if optimal_data.empty:
            logger.warning(f"No benchmark data for optimal: {optimal_algorithm} on {config}")
            continue
        
        optimal_energy = optimal_data['median_energy_j'].values[0]
        
        # Get energy for baseline algorithm
        baseline_data = benchmark_df[
            (benchmark_df['file_type'] == file_type) &
            (benchmark_df['file_size'] == file_size) &
            (benchmark_df['algorithm'] == baseline_algorithm)
        ]
        
        if baseline_data.empty:
            logger.warning(f"No benchmark data for baseline: {baseline_algorithm} on {config}")
            continue
        
        baseline_energy = baseline_data['median_energy_j'].values[0]
        
        # Calculate savings
        energy_saved_j = baseline_energy - optimal_energy
        savings_pct = (energy_saved_j / baseline_energy) * 100 if baseline_energy > 0 else 0
        
        results.append({
            'config': config,
            'file_type': file_type,
            'file_size': file_size,
            'size_category': get_size_category(file_size),
            'optimal_algorithm': optimal_algorithm,
            'baseline_algorithm': baseline_algorithm,
            'optimal_energy_j': optimal_energy,
            'baseline_energy_j': baseline_energy,
            'energy_saved_j': energy_saved_j,
            'savings_pct': savings_pct,
        })
    
    return pd.DataFrame(results)


def calculate_statistics(savings_df: pd.DataFrame, min_file_size: int = 10240) -> dict:
    """Calculate summary statistics and significance tests.
    
    Args:
        savings_df: DataFrame with savings calculations
        min_file_size: Minimum file size for main results (default 10KB)
        
    Returns:
        Dictionary with statistics
    """
    # Filter for files above minimum size
    main_df = savings_df[savings_df['file_size'] >= min_file_size]
    all_df = savings_df
    
    stats_result = {
        'all_files': {},
        'files_above_10kb': {},
        'by_size_category': {},
        'by_file_type': {},
        'statistical_tests': {},
    }
    
    # Overall statistics - all files
    stats_result['all_files'] = {
        'count': len(all_df),
        'mean_savings_pct': float(all_df['savings_pct'].mean()),
        'median_savings_pct': float(all_df['savings_pct'].median()),
        'std_savings_pct': float(all_df['savings_pct'].std()),
        'min_savings_pct': float(all_df['savings_pct'].min()),
        'max_savings_pct': float(all_df['savings_pct'].max()),
        'total_energy_saved_j': float(all_df['energy_saved_j'].sum()),
        'total_baseline_energy_j': float(all_df['baseline_energy_j'].sum()),
    }
    
    # Overall statistics - files >= 10KB
    if len(main_df) > 0:
        stats_result['files_above_10kb'] = {
            'count': len(main_df),
            'mean_savings_pct': float(main_df['savings_pct'].mean()),
            'median_savings_pct': float(main_df['savings_pct'].median()),
            'std_savings_pct': float(main_df['savings_pct'].std()),
            'min_savings_pct': float(main_df['savings_pct'].min()),
            'max_savings_pct': float(main_df['savings_pct'].max()),
            'total_energy_saved_j': float(main_df['energy_saved_j'].sum()),
            'total_baseline_energy_j': float(main_df['baseline_energy_j'].sum()),
        }
    
    # By size category
    for category in SIZE_CATEGORIES.keys():
        cat_df = all_df[all_df['size_category'] == category]
        if len(cat_df) > 0:
            stats_result['by_size_category'][category] = {
                'count': len(cat_df),
                'mean_savings_pct': float(cat_df['savings_pct'].mean()),
                'median_savings_pct': float(cat_df['savings_pct'].median()),
                'algorithms_used': cat_df['optimal_algorithm'].value_counts().to_dict(),
            }
    
    # By file type
    for file_type in all_df['file_type'].unique():
        type_df = all_df[all_df['file_type'] == file_type]
        if len(type_df) > 0:
            stats_result['by_file_type'][file_type] = {
                'count': len(type_df),
                'mean_savings_pct': float(type_df['savings_pct'].mean()),
                'median_savings_pct': float(type_df['savings_pct'].median()),
                'total_energy_saved_j': float(type_df['energy_saved_j'].sum()),
            }
    
    # Statistical significance tests (on files >= 10KB)
    if len(main_df) >= 2:
        optimal_energies = main_df['optimal_energy_j'].values
        baseline_energies = main_df['baseline_energy_j'].values
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(optimal_energies, baseline_energies)
        
        # Effect size (Cohen's d for paired samples)
        diff = baseline_energies - optimal_energies
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        try:
            w_stat, w_pvalue = stats.wilcoxon(baseline_energies, optimal_energies)
        except ValueError:
            w_stat, w_pvalue = np.nan, np.nan
        
        stats_result['statistical_tests'] = {
            'paired_t_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_at_005': bool(p_value < 0.05),
                'significant_at_001': bool(p_value < 0.01),
            },
            'effect_size': {
                'cohens_d': float(cohens_d),
                'interpretation': interpret_cohens_d(cohens_d),
            },
            'wilcoxon_test': {
                'w_statistic': float(w_stat) if not np.isnan(w_stat) else None,
                'p_value': float(w_pvalue) if not np.isnan(w_pvalue) else None,
            },
            'sample_size': int(len(main_df)),
        }
    
    return stats_result


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return 'negligible'
    elif d < 0.5:
        return 'small'
    elif d < 0.8:
        return 'medium'
    else:
        return 'large'


def print_report(savings_df: pd.DataFrame, statistics: dict):
    """Print formatted report to console."""
    print("\n" + "=" * 70)
    print("CRYPTOGREEN ENERGY SAVINGS REPORT")
    print("=" * 70)
    
    # Overall savings
    print("\n" + "=" * 70)
    print("OVERALL ENERGY SAVINGS (All Files)")
    print("=" * 70)
    
    all_stats = statistics['all_files']
    print(f"\n  Configurations analyzed: {all_stats['count']}")
    print(f"  Mean savings vs AES-256:   {all_stats['mean_savings_pct']:+.1f}%")
    print(f"  Median savings vs AES-256: {all_stats['median_savings_pct']:+.1f}%")
    print(f"  Std deviation:             {all_stats['std_savings_pct']:.1f}%")
    print(f"  Range: {all_stats['min_savings_pct']:+.1f}% to {all_stats['max_savings_pct']:+.1f}%")
    
    # Main results (files >= 10KB)
    if 'files_above_10kb' in statistics and statistics['files_above_10kb']:
        print("\n" + "=" * 70)
        print("MAIN RESULTS (Files >= 10KB)")
        print("=" * 70)
        
        main_stats = statistics['files_above_10kb']
        print(f"\n  Configurations analyzed: {main_stats['count']}")
        print(f"  Mean savings vs AES-256:   {main_stats['mean_savings_pct']:+.1f}%")
        print(f"  Median savings vs AES-256: {main_stats['median_savings_pct']:+.1f}%")
        print(f"  Total energy saved:        {main_stats['total_energy_saved_j']:.2f} J")
    
    # Savings by size category
    print("\n" + "=" * 70)
    print("SAVINGS BY FILE SIZE CATEGORY")
    print("=" * 70)
    
    print(f"\n  {'Category':<12} {'Count':>8} {'Mean':>12} {'Median':>12}  Optimal Algorithms")
    print("  " + "-" * 65)
    
    for category in ['tiny', 'small', 'medium', 'large', 'huge']:
        if category in statistics['by_size_category']:
            cat_stats = statistics['by_size_category'][category]
            algos = ', '.join(f"{k}:{v}" for k, v in cat_stats['algorithms_used'].items())
            print(f"  {category:<12} {cat_stats['count']:>8} {cat_stats['mean_savings_pct']:>+11.1f}% "
                  f"{cat_stats['median_savings_pct']:>+11.1f}%  {algos}")
    
    # Savings by file type
    print("\n" + "=" * 70)
    print("SAVINGS BY FILE TYPE")
    print("=" * 70)
    
    print(f"\n  {'Type':<8} {'Count':>8} {'Mean Savings':>14} {'Total Saved':>14}")
    print("  " + "-" * 45)
    
    for file_type, type_stats in sorted(statistics['by_file_type'].items()):
        print(f"  {file_type:<8} {type_stats['count']:>8} {type_stats['mean_savings_pct']:>+13.1f}% "
              f"{type_stats['total_energy_saved_j']:>13.4f} J")
    
    # Statistical significance
    if 'statistical_tests' in statistics and statistics['statistical_tests']:
        print("\n" + "=" * 70)
        print("STATISTICAL SIGNIFICANCE (Files >= 10KB)")
        print("=" * 70)
        
        tests = statistics['statistical_tests']
        
        print(f"\n  Sample size: {tests['sample_size']} configurations")
        
        t_test = tests['paired_t_test']
        print(f"\n  Paired t-test:")
        print(f"    t-statistic: {t_test['t_statistic']:.4f}")
        print(f"    p-value:     {t_test['p_value']:.2e}")
        print(f"    Significant at α=0.05: {'Yes ✓' if t_test['significant_at_005'] else 'No'}")
        print(f"    Significant at α=0.01: {'Yes ✓' if t_test['significant_at_001'] else 'No'}")
        
        effect = tests['effect_size']
        print(f"\n  Effect size:")
        print(f"    Cohen's d: {effect['cohens_d']:.4f}")
        print(f"    Interpretation: {effect['interpretation']}")
        
        if tests['wilcoxon_test']['p_value'] is not None:
            print(f"\n  Wilcoxon signed-rank test:")
            print(f"    p-value: {tests['wilcoxon_test']['p_value']:.2e}")
    
    # Top savings examples
    print("\n" + "=" * 70)
    print("TOP 5 ENERGY SAVINGS CONFIGURATIONS")
    print("=" * 70)
    
    top_savings = savings_df.nlargest(5, 'savings_pct')
    print(f"\n  {'Config':<20} {'Optimal':<12} {'Baseline':>12} {'Optimal':>12} {'Savings':>10}")
    print(f"  {'':<20} {'Algorithm':<12} {'Energy (J)':>12} {'Energy (J)':>12} {'':>10}")
    print("  " + "-" * 68)
    
    for _, row in top_savings.iterrows():
        print(f"  {row['config']:<20} {row['optimal_algorithm']:<12} "
              f"{row['baseline_energy_j']:>12.6f} {row['optimal_energy_j']:>12.6f} "
              f"{row['savings_pct']:>+9.1f}%")
    
    print("\n" + "=" * 70)


def save_results(savings_df: pd.DataFrame, statistics: dict, output_path: str):
    """Save results to JSON file."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_configurations': len(savings_df),
            'baseline_algorithm': savings_df['baseline_algorithm'].iloc[0] if len(savings_df) > 0 else 'AES-256',
            'mean_savings_pct': statistics['all_files']['mean_savings_pct'],
            'median_savings_pct': statistics['all_files']['median_savings_pct'],
        },
        'statistics': statistics,
        'detailed_results': savings_df.to_dict(orient='records'),
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate energy savings from optimal algorithm selection'
    )
    parser.add_argument(
        '--baseline', type=str, default='AES-256',
        help='Baseline algorithm for comparison (default: AES-256)'
    )
    parser.add_argument(
        '--benchmark-dir', type=str, default='results/benchmarks/raw',
        help='Directory containing benchmark results'
    )
    parser.add_argument(
        '--labels', type=str, default='data/ml_data/labels.csv',
        help='Path to labels CSV file'
    )
    parser.add_argument(
        '--output', type=str, default='results/energy_savings_report.json',
        help='Output file path'
    )
    parser.add_argument(
        '--min-size', type=int, default=10240,
        help='Minimum file size for main results (default: 10240 = 10KB)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CRYPTOGREEN ENERGY SAVINGS CALCULATOR")
    print("=" * 70)
    
    # Load data
    logger.info("Loading benchmark data...")
    benchmark_df = load_benchmark_data(args.benchmark_dir)
    logger.info(f"Loaded {len(benchmark_df)} benchmark results")
    
    logger.info("Loading optimal algorithm labels...")
    labels_df = load_labels(args.labels)
    logger.info(f"Loaded {len(labels_df)} configuration labels")
    
    # Calculate savings
    logger.info(f"Calculating energy savings vs {args.baseline}...")
    savings_df = calculate_savings(benchmark_df, labels_df, args.baseline)
    logger.info(f"Calculated savings for {len(savings_df)} configurations")
    
    # Calculate statistics
    logger.info("Computing statistics...")
    statistics = calculate_statistics(savings_df, args.min_size)
    
    # Print report
    print_report(savings_df, statistics)
    
    # Save results
    save_results(savings_df, statistics, args.output)
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Average energy savings vs {args.baseline}: "
          f"{statistics['all_files']['mean_savings_pct']:+.1f}%")
    
    if statistics['files_above_10kb']:
        print(f"  Average savings (files >= 10KB): "
              f"{statistics['files_above_10kb']['mean_savings_pct']:+.1f}%")
    
    if statistics['statistical_tests']:
        p_val = statistics['statistical_tests']['paired_t_test']['p_value']
        print(f"  Statistical significance: p = {p_val:.2e}")
    
    print(f"\n  Results saved to: {args.output}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
