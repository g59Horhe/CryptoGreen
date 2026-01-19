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
    """Generate publication-quality visualization plots (300 DPI).
    
    Generates 5 main publication-quality figures:
    1. energy_vs_size.png - Log-log plot with optimal algorithm regions annotated
    2. algorithm_selection_pie.png - Pie chart showing selection distribution
    3. accuracy_by_size.png - Bar chart showing selector accuracy by file size
    4. feature_importance.png - Horizontal bar chart of top ML features
    5. confusion_matrix.png - Heatmap of ML prediction errors
    
    Plus additional analysis plots.
    """
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
    
    # Publication-quality settings (300 DPI)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # Color palette for algorithms
    algo_colors = {
        'AES-128': '#3498db',  # Blue
        'AES-256': '#2980b9',  # Dark blue
        'ChaCha20': '#27ae60', # Green
    }
    sns.set_palette([algo_colors['AES-128'], algo_colors['AES-256'], algo_colors['ChaCha20']])
    
    # Only use symmetric algorithms
    symmetric_algs = ['AES-128', 'AES-256', 'ChaCha20']
    symmetric_df = df[df['algorithm'].isin(symmetric_algs)]
    
    # =========================================================================
    # FIGURE 1: Energy vs File Size (Publication-quality log-log plot)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markers = {'AES-128': 'o', 'AES-256': 's', 'ChaCha20': '^'}
    
    for alg in symmetric_algs:
        alg_data = symmetric_df[symmetric_df['algorithm'] == alg].copy()
        alg_data = alg_data.groupby('file_size').agg({
            'median_energy_j': ['mean', 'std']
        }).reset_index()
        alg_data.columns = ['file_size', 'mean_energy', 'std_energy']
        alg_data = alg_data.sort_values('file_size')
        
        ax.errorbar(
            alg_data['file_size'],
            alg_data['mean_energy'],
            yerr=alg_data['std_energy'],
            marker=markers[alg],
            label=alg,
            linewidth=2,
            markersize=8,
            capsize=3,
            color=algo_colors[alg]
        )
    
    ax.set_xlabel('File Size (bytes)', fontsize=12)
    ax.set_ylabel('Energy Consumption (Joules)', fontsize=12)
    ax.set_title('Energy Consumption vs File Size by Algorithm', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add size category annotations
    size_categories = [
        (64, '64B\n(Tiny)'),
        (10240, '10KB\n(Small)'),
        (1048576, '1MB\n(Medium)'),
        (10485760, '10MB\n(Large)')
    ]
    
    for size, label in size_categories:
        ax.axvline(x=size, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add optimal region annotation
    ax.annotate('AES-128 optimal\n(most scenarios)',
                xy=(100000, symmetric_df[symmetric_df['algorithm']=='AES-128']['median_energy_j'].median()),
                xytext=(1000, 0.1),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#3498db'))
    
    plt.tight_layout()
    plt.savefig(output_path / 'energy_vs_size.png', dpi=300, facecolor='white', edgecolor='none')
    logger.info(f"Saved: {output_path / 'energy_vs_size.png'} (300 DPI)")
    
    if show:
        plt.show()
    plt.close()
    
    # =========================================================================
    # FIGURE 2: Algorithm Selection Pie Chart
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Find optimal algorithm for each file_type + file_size combo
    optimal_counts = {}
    grouped = symmetric_df.groupby(['file_type', 'file_size'])
    
    for (ft, fs), group in grouped:
        best_row = group.loc[group['median_energy_j'].idxmin()]
        best_alg = best_row['algorithm']
        optimal_counts[best_alg] = optimal_counts.get(best_alg, 0) + 1
    
    labels = list(optimal_counts.keys())
    sizes = list(optimal_counts.values())
    total = sum(sizes)
    
    colors = [algo_colors.get(l, '#95a5a6') for l in labels]
    
    # Create pie chart with explode for emphasis
    explode = [0.03 if l == 'AES-128' else 0.01 for l in labels]
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct='',
        colors=colors, startangle=90,
        explode=explode,
        wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2)
    )
    
    # Custom labels with percentage and use case
    use_cases = {
        'AES-128': 'General Purpose\n(AES-NI accelerated)',
        'AES-256': 'High Security\n(Sensitive data)',
        'ChaCha20': 'Non-AES Hardware\n(Embedded/Mobile)'
    }
    
    # Add legend with detailed info
    legend_labels = []
    for l, s in zip(labels, sizes):
        pct = 100 * s / total
        use_case = use_cases.get(l, '')
        legend_labels.append(f'{l}: {pct:.1f}% ({s} configs)\n{use_case}')
    
    ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
              fontsize=10, framealpha=0.9, edgecolor='gray')
    
    # Add center text
    centre_circle = plt.Circle((0, 0), 0.4, fc='white', edgecolor='gray', linewidth=1)
    ax.add_patch(centre_circle)
    ax.text(0, 0, f'n={total}\nconfigs', ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_title('Optimal Algorithm Selection Distribution\nby File Configuration', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path / 'algorithm_selection_pie.png', dpi=300, facecolor='white', edgecolor='none')
    logger.info(f"Saved: {output_path / 'algorithm_selection_pie.png'} (300 DPI)")
    
    if show:
        plt.show()
    plt.close()
    
    # =========================================================================
    # FIGURE 3: Accuracy by File Size Category
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define size categories
    def categorize_size(size):
        if size < 1024:
            return 'Tiny (<1KB)'
        elif size < 102400:
            return 'Small (1-100KB)'
        elif size < 1048576:
            return 'Medium (100KB-1MB)'
        elif size < 10485760:
            return 'Large (1-10MB)'
        else:
            return 'Huge (>10MB)'
    
    symmetric_df_copy = symmetric_df.copy()
    symmetric_df_copy['size_category'] = symmetric_df_copy['file_size'].apply(categorize_size)
    
    # Calculate "accuracy" as consistency of optimal algorithm selection
    size_order = ['Tiny (<1KB)', 'Small (1-100KB)', 'Medium (100KB-1MB)', 'Large (1-10MB)', 'Huge (>10MB)']
    
    # For each size category, calculate how consistently one algorithm wins
    accuracy_data = []
    for cat in size_order:
        cat_df = symmetric_df_copy[symmetric_df_copy['size_category'] == cat]
        if len(cat_df) == 0:
            continue
        
        # Group by file type within this category
        optimal_by_type = {}
        for ft in cat_df['file_type'].unique():
            ft_data = cat_df[cat_df['file_type'] == ft]
            min_energy_row = ft_data.loc[ft_data['median_energy_j'].idxmin()]
            optimal_by_type[ft] = min_energy_row['algorithm']
        
        # Consistency = most common algorithm's frequency
        from collections import Counter
        counts = Counter(optimal_by_type.values())
        most_common = counts.most_common(1)[0]
        consistency = most_common[1] / len(optimal_by_type) * 100 if optimal_by_type else 0
        
        accuracy_data.append({
            'Size Category': cat,
            'Consistency (%)': consistency,
            'Dominant Algorithm': most_common[0],
            'Count': len(optimal_by_type)
        })
    
    acc_df = pd.DataFrame(accuracy_data)
    
    # Create bar chart
    bar_colors = [algo_colors.get(row['Dominant Algorithm'], '#95a5a6') for _, row in acc_df.iterrows()]
    bars = ax.bar(range(len(acc_df)), acc_df['Consistency (%)'], color=bar_colors, 
                  edgecolor='black', linewidth=1)
    
    ax.set_xticks(range(len(acc_df)))
    ax.set_xticklabels(acc_df['Size Category'], rotation=15, ha='right')
    ax.set_xlabel('File Size Category', fontsize=12)
    ax.set_ylabel('Selection Consistency (%)', fontsize=12)
    ax.set_title('Algorithm Selection Consistency by File Size\n(Higher = More Consistent Optimal Choice)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect consistency')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels and dominant algorithm
    for i, (bar, row) in enumerate(zip(bars, acc_df.itertuples())):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{row._2:.0f}%\n({row._3})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_by_size.png', dpi=300, facecolor='white', edgecolor='none')
    logger.info(f"Saved: {output_path / 'accuracy_by_size.png'} (300 DPI)")
    
    if show:
        plt.show()
    plt.close()
    
    # =========================================================================
    # FIGURE 4: Feature Importance (from ML model if available)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Try to load feature importance from training report
    training_report_path = Path('results/models/training_report.json')
    
    if training_report_path.exists():
        with open(training_report_path, 'r') as f:
            training_report = json.load(f)
        
        if 'feature_importance' in training_report:
            importance = training_report['feature_importance']
            
            # Sort by importance and get top 10
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features = [f[0] for f in sorted_features]
            values = [f[1] for f in sorted_features]
            
            # Horizontal bar chart
            y_pos = range(len(features))
            bars = ax.barh(y_pos, values, color='#3498db', edgecolor='black', linewidth=0.5)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()  # Top to bottom
            ax.set_xlabel('Feature Importance (Relative)', fontsize=12)
            ax.set_title('Top 10 Features for Algorithm Selection\n(Random Forest Feature Importance)', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontsize=9)
    else:
        # Create placeholder with feature names
        features = ['file_size', 'entropy', 'has_aes_ni', 'file_type_encoded', 'mean_byte_value']
        values = [0.4, 0.25, 0.15, 0.12, 0.08]
        
        y_pos = range(len(features))
        ax.barh(y_pos, values, color='#3498db', edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance (Relative)', fontsize=12)
        ax.set_title('Top Features for Algorithm Selection\n(Estimated - Train model for actual values)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.text(0.5, 0.5, 'Note: Run train_model.py to generate actual values',
               transform=ax.transAxes, ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path / 'feature_importance.png', dpi=300, facecolor='white', edgecolor='none')
    logger.info(f"Saved: {output_path / 'feature_importance.png'} (300 DPI)")
    
    if show:
        plt.show()
    plt.close()
    
    # =========================================================================
    # FIGURE 5: Confusion Matrix
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 7))
    
    if training_report_path.exists():
        with open(training_report_path, 'r') as f:
            training_report = json.load(f)
        
        if 'confusion_matrix' in training_report:
            cm = np.array(training_report['confusion_matrix'])
            classes = training_report.get('classes', symmetric_algs)
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
            
            # Create heatmap
            im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('Normalized Frequency', rotation=-90, va="bottom", fontsize=10)
            
            # Add labels
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, fontsize=11)
            ax.set_yticklabels(classes, fontsize=11)
            
            # Add text annotations
            thresh = cm_normalized.max() / 2.
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})',
                           ha="center", va="center", fontsize=10,
                           color="white" if cm_normalized[i, j] > thresh else "black")
            
            ax.set_xlabel('Predicted Algorithm', fontsize=12)
            ax.set_ylabel('True Optimal Algorithm', fontsize=12)
            ax.set_title('ML Selector Confusion Matrix\n(Count and Normalized %)', 
                        fontsize=14, fontweight='bold')
    else:
        # Create placeholder
        classes = symmetric_algs
        cm = np.array([[8, 0, 2], [0, 1, 0], [1, 0, 2]])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        
        thresh = cm_normalized.max() / 2.
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f'{cm[i, j]}',
                       ha="center", va="center", fontsize=11,
                       color="white" if cm_normalized[i, j] > thresh else "black")
        
        ax.set_xlabel('Predicted Algorithm', fontsize=12)
        ax.set_ylabel('True Optimal Algorithm', fontsize=12)
        ax.set_title('ML Selector Confusion Matrix\n(Placeholder - Train model for actual values)', 
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300, facecolor='white', edgecolor='none')
    logger.info(f"Saved: {output_path / 'confusion_matrix.png'} (300 DPI)")
    
    if show:
        plt.show()
    plt.close()
    
    # =========================================================================
    # Additional plots (legacy support)
    # =========================================================================
    
    # Energy Heatmap (Algorithm x File Type)
    pivot = symmetric_df.groupby(['algorithm', 'file_type'])['median_energy_j'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.4f',
        cmap='YlOrRd',
        ax=ax,
        linewidths=0.5
    )
    ax.set_title('Average Energy by Algorithm and File Type (Joules)', fontsize=14, fontweight='bold')
    ax.set_xlabel('File Type', fontsize=12)
    ax.set_ylabel('Algorithm', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path / 'energy_heatmap.png', dpi=300, facecolor='white', edgecolor='none')
    logger.info(f"Saved: {output_path / 'energy_heatmap.png'} (300 DPI)")
    
    if show:
        plt.show()
    plt.close()
    
    # Throughput Comparison (Box plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(
        data=symmetric_df,
        x='algorithm',
        y='throughput_mbps',
        hue='algorithm',
        ax=ax,
        palette=algo_colors,
        legend=False
    )
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Throughput (MB/s)', fontsize=12)
    ax.set_title('Encryption Throughput by Algorithm', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / 'throughput_comparison.png', dpi=300, facecolor='white', edgecolor='none')
    logger.info(f"Saved: {output_path / 'throughput_comparison.png'} (300 DPI)")
    
    if show:
        plt.show()
    plt.close()
    
    logger.info(f"\nAll publication-quality plots saved to: {output_path}")
    logger.info("Figures generated at 300 DPI for publication submission")


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
