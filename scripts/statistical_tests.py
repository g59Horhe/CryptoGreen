#!/usr/bin/env python3
"""
Statistical significance testing for CryptoGreen benchmark results.

Performs:
1. Paired t-tests on energy consumption for algorithm pairs
2. Cohen's d effect size calculations
3. Wilcoxon signed-rank tests (non-parametric validation)
4. Significance matrix generation
"""

import json
import sys
import argparse
import logging
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_benchmark_results(filepath: str) -> List[Dict[str, Any]]:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    return data


def extract_energy_data(results: List[Dict]) -> pd.DataFrame:
    """
    Extract energy measurements into a structured DataFrame.
    
    Returns DataFrame with columns: file_type, file_size, algorithm, energy_values
    """
    records = []
    
    for r in results:
        algorithm = r['algorithm']
        file_type = r['file_type']
        file_size = r['file_size']
        
        # Get individual energy measurements
        if 'measurements' in r:
            energy_values = [m['energy_joules'] for m in r['measurements']]
        elif 'statistics' in r:
            # Fallback to statistics if individual measurements not available
            energy_values = [r['statistics']['median_energy_j']]
        else:
            continue
        
        records.append({
            'algorithm': algorithm,
            'file_type': file_type,
            'file_size': file_size,
            'config': f"{file_type}_{file_size}",
            'energy_values': energy_values,
            'mean_energy': np.mean(energy_values),
            'std_energy': np.std(energy_values),
            'n_samples': len(energy_values)
        })
    
    return pd.DataFrame(records)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Cohen's d = (M1 - M2) / pooled_std
    
    Effect size interpretation:
    - Small: |d| < 0.2
    - Medium: 0.2 <= |d| < 0.8
    - Large: |d| >= 0.8
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def paired_statistical_tests(
    df: pd.DataFrame,
    alg1: str,
    alg2: str
) -> Dict[str, Any]:
    """
    Run paired statistical tests between two algorithms.
    
    For each configuration (file_type + file_size), we compare
    energy measurements between the two algorithms.
    """
    # Get data for both algorithms
    df1 = df[df['algorithm'] == alg1].set_index('config')
    df2 = df[df['algorithm'] == alg2].set_index('config')
    
    # Find common configurations
    common_configs = set(df1.index) & set(df2.index)
    
    if len(common_configs) < 2:
        return {
            'alg1': alg1,
            'alg2': alg2,
            'n_configs': len(common_configs),
            'error': 'Insufficient common configurations'
        }
    
    # Collect paired measurements (mean energy per config)
    energies1 = []
    energies2 = []
    all_values1 = []
    all_values2 = []
    
    for config in sorted(common_configs):
        energies1.append(df1.loc[config, 'mean_energy'])
        energies2.append(df2.loc[config, 'mean_energy'])
        all_values1.extend(df1.loc[config, 'energy_values'])
        all_values2.extend(df2.loc[config, 'energy_values'])
    
    energies1 = np.array(energies1)
    energies2 = np.array(energies2)
    all_values1 = np.array(all_values1)
    all_values2 = np.array(all_values2)
    
    # 1. Paired t-test (on mean energies per config)
    t_stat, t_pvalue = stats.ttest_rel(energies1, energies2)
    
    # 2. Wilcoxon signed-rank test (non-parametric)
    # Handle case where differences might be zero
    differences = energies1 - energies2
    non_zero_diff = differences[differences != 0]
    
    if len(non_zero_diff) >= 10:
        w_stat, w_pvalue = stats.wilcoxon(energies1, energies2)
    else:
        # Fall back to Mann-Whitney U for small samples
        w_stat, w_pvalue = stats.mannwhitneyu(energies1, energies2, alternative='two-sided')
    
    # 3. Cohen's d effect size (on all individual measurements)
    d = cohens_d(all_values1, all_values2)
    
    # 4. Summary statistics
    mean1, mean2 = np.mean(energies1), np.mean(energies2)
    pct_diff = ((mean2 - mean1) / mean1) * 100 if mean1 != 0 else 0
    
    return {
        'alg1': alg1,
        'alg2': alg2,
        'n_configs': len(common_configs),
        'n_samples_alg1': len(all_values1),
        'n_samples_alg2': len(all_values2),
        'mean_energy_alg1': mean1,
        'mean_energy_alg2': mean2,
        'pct_difference': pct_diff,
        't_statistic': t_stat,
        't_pvalue': t_pvalue,
        'wilcoxon_statistic': w_stat,
        'wilcoxon_pvalue': w_pvalue,
        'cohens_d': d,
        'effect_size': interpret_cohens_d(d),
        'significant_t': t_pvalue < 0.05,
        'significant_wilcoxon': w_pvalue < 0.05
    }


def run_all_pairwise_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run statistical tests for all algorithm pairs."""
    algorithms = sorted(df['algorithm'].unique())
    results = []
    
    logger.info(f"Running pairwise tests for {len(algorithms)} algorithms...")
    
    for alg1, alg2 in combinations(algorithms, 2):
        logger.info(f"  Testing: {alg1} vs {alg2}")
        result = paired_statistical_tests(df, alg1, alg2)
        results.append(result)
    
    return pd.DataFrame(results)


def create_significance_matrix(
    test_results: pd.DataFrame,
    metric: str = 't_pvalue'
) -> pd.DataFrame:
    """
    Create a matrix showing p-values between all algorithm pairs.
    
    Args:
        test_results: DataFrame with pairwise test results
        metric: Which p-value to use ('t_pvalue' or 'wilcoxon_pvalue')
    """
    # Get all unique algorithms
    algorithms = sorted(set(test_results['alg1'].unique()) | 
                       set(test_results['alg2'].unique()))
    
    # Initialize matrix with NaN (diagonal will be 1.0)
    matrix = pd.DataFrame(
        np.nan,
        index=algorithms,
        columns=algorithms
    )
    
    # Fill diagonal with 1.0 (same algorithm comparison)
    for alg in algorithms:
        matrix.loc[alg, alg] = 1.0
    
    # Fill in p-values
    for _, row in test_results.iterrows():
        if 'error' in row and pd.notna(row.get('error')):
            continue
        
        alg1, alg2 = row['alg1'], row['alg2']
        pvalue = row[metric]
        
        matrix.loc[alg1, alg2] = pvalue
        matrix.loc[alg2, alg1] = pvalue
    
    return matrix


def create_effect_size_matrix(test_results: pd.DataFrame) -> pd.DataFrame:
    """Create a matrix showing Cohen's d effect sizes."""
    algorithms = sorted(set(test_results['alg1'].unique()) | 
                       set(test_results['alg2'].unique()))
    
    matrix = pd.DataFrame(
        0.0,
        index=algorithms,
        columns=algorithms
    )
    
    for _, row in test_results.iterrows():
        if 'error' in row and pd.notna(row.get('error')):
            continue
        
        alg1, alg2 = row['alg1'], row['alg2']
        d = row['cohens_d']
        
        matrix.loc[alg1, alg2] = d
        matrix.loc[alg2, alg1] = -d  # Reverse sign for opposite direction
    
    return matrix


def print_results_summary(test_results: pd.DataFrame):
    """Print a summary of statistical test results."""
    print()
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE TEST RESULTS")
    print("=" * 80)
    print()
    
    # Filter out errors
    valid_results = test_results[~test_results.get('error', pd.Series([None]*len(test_results))).notna()]
    
    # Significant differences (t-test)
    sig_t = valid_results[valid_results['significant_t']]
    print(f"SIGNIFICANT DIFFERENCES (Paired t-test, p < 0.05): {len(sig_t)}/{len(valid_results)}")
    print("-" * 80)
    
    if len(sig_t) > 0:
        sig_t_sorted = sig_t.sort_values('t_pvalue')
        for _, row in sig_t_sorted.iterrows():
            better = row['alg1'] if row['mean_energy_alg1'] < row['mean_energy_alg2'] else row['alg2']
            worse = row['alg2'] if better == row['alg1'] else row['alg1']
            print(f"  {row['alg1']:12s} vs {row['alg2']:12s}: "
                  f"p={row['t_pvalue']:.2e}, d={row['cohens_d']:+.3f} ({row['effect_size']})")
            print(f"    -> {better} uses {abs(row['pct_difference']):.1f}% "
                  f"{'less' if row['pct_difference'] > 0 else 'more'} energy than {worse}")
    print()
    
    # Non-significant differences
    nonsig = valid_results[~valid_results['significant_t']]
    print(f"NON-SIGNIFICANT DIFFERENCES: {len(nonsig)}/{len(valid_results)}")
    print("-" * 80)
    
    if len(nonsig) > 0:
        for _, row in nonsig.iterrows():
            print(f"  {row['alg1']:12s} vs {row['alg2']:12s}: "
                  f"p={row['t_pvalue']:.3f}, d={row['cohens_d']:+.3f} ({row['effect_size']})")
    print()
    
    # Wilcoxon validation
    t_sig = set(valid_results[valid_results['significant_t']].apply(
        lambda r: frozenset([r['alg1'], r['alg2']]), axis=1))
    w_sig = set(valid_results[valid_results['significant_wilcoxon']].apply(
        lambda r: frozenset([r['alg1'], r['alg2']]), axis=1))
    
    agreement = len(t_sig & w_sig)
    total_sig = len(t_sig | w_sig)
    
    print(f"PARAMETRIC vs NON-PARAMETRIC AGREEMENT:")
    print("-" * 80)
    print(f"  Both t-test and Wilcoxon significant: {agreement}")
    print(f"  Only t-test significant: {len(t_sig - w_sig)}")
    print(f"  Only Wilcoxon significant: {len(w_sig - t_sig)}")
    print()
    
    # Effect size summary
    print("EFFECT SIZE SUMMARY:")
    print("-" * 80)
    effect_counts = valid_results['effect_size'].value_counts()
    for effect, count in effect_counts.items():
        print(f"  {effect.capitalize():12s}: {count} comparisons")
    print()
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Run statistical significance tests on benchmark results'
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
        default='results/benchmarks/processed',
        help='Output directory for results'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
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
        import glob
        pattern = 'results/benchmarks/raw/benchmark_*.json'
        files = glob.glob(pattern)
        files = [f for f in files if 'incremental' not in f]
        
        if not files:
            logger.error(f"No benchmark results found matching {pattern}")
            sys.exit(1)
        
        benchmark_file = max(files, key=lambda x: Path(x).stat().st_mtime)
        logger.info(f"Auto-detected: {benchmark_file}")
    
    # Load and process data
    logger.info(f"Loading benchmark results from: {benchmark_file}")
    results = load_benchmark_results(benchmark_file)
    logger.info(f"Loaded {len(results)} benchmark results")
    
    # Extract energy data
    df = extract_energy_data(results)
    logger.info(f"Extracted data for {df['algorithm'].nunique()} algorithms, "
                f"{df['config'].nunique()} configurations")
    
    # Run pairwise tests
    test_results = run_all_pairwise_tests(df)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    output_file = output_path / 'statistical_tests.csv'
    test_results.to_csv(output_file, index=False)
    logger.info(f"Saved detailed results to: {output_file}")
    
    # Create and save significance matrices
    t_matrix = create_significance_matrix(test_results, 't_pvalue')
    t_matrix.to_csv(output_path / 'significance_matrix_ttest.csv')
    logger.info(f"Saved t-test significance matrix")
    
    w_matrix = create_significance_matrix(test_results, 'wilcoxon_pvalue')
    w_matrix.to_csv(output_path / 'significance_matrix_wilcoxon.csv')
    logger.info(f"Saved Wilcoxon significance matrix")
    
    effect_matrix = create_effect_size_matrix(test_results)
    effect_matrix.to_csv(output_path / 'effect_size_matrix.csv')
    logger.info(f"Saved effect size matrix")
    
    # Print summary
    print_results_summary(test_results)
    
    logger.info("Statistical analysis complete!")


if __name__ == '__main__':
    main()
