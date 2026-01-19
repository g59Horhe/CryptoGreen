#!/usr/bin/env python3
"""
Generate all figures for CryptoGreen paper from benchmark results
"""

import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("=" * 70)
print("GENERATING ALL FIGURES")
print("=" * 70)

# Load benchmark results
print("\nLoading benchmark results...")
BENCHMARK_DIR = "results/complete_benchmark_20260119_193623/raw"
result_files = glob.glob(f"{BENCHMARK_DIR}/*.json")

if not result_files:
    print("ERROR: No benchmark results found!")
    print(f"Looking in: {BENCHMARK_DIR}")
    exit(1)

results = []
for f in result_files:
    with open(f) as fp:
        results.append(json.load(fp))

print(f"Loaded {len(results)} benchmark results")

# Create output directory
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

################################################################################
# FIGURE 1: Energy vs File Size (scatter plot with regression)
################################################################################
print("\n[1/6] Generating: Energy vs File Size...")

fig, ax = plt.subplots(figsize=(12, 7))

for algo in ['AES-128', 'AES-256', 'ChaCha20']:
    algo_data = [r for r in results if r['algorithm'] == algo]
    sizes = [r['file_size'] for r in algo_data]
    energies = [r['statistics']['median_energy_j'] for r in algo_data]
    
    ax.scatter(sizes, energies, alpha=0.6, s=50, label=algo)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('File Size (bytes)', fontsize=12, fontweight='bold')
ax.set_ylabel('Energy Consumption (Joules)', fontsize=12, fontweight='bold')
ax.set_title('Energy Consumption vs File Size by Algorithm', fontsize=14, fontweight='bold')
ax.legend(title='Algorithm', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'energy_vs_size.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'energy_vs_size.pdf', bbox_inches='tight')
print(f"  ✓ Saved: energy_vs_size.png")

################################################################################
# FIGURE 2: Algorithm Distribution by File Size
################################################################################
print("[2/6] Generating: Algorithm Distribution by Size...")

# Find optimal algorithm for each file
optimal_algos = {}
for file_name in set(r['file_name'] for r in results):
    file_results = [r for r in results if r['file_name'] == file_name]
    min_energy = min(r['statistics']['median_energy_j'] for r in file_results)
    optimal = [r for r in file_results if r['statistics']['median_energy_j'] == min_energy][0]
    optimal_algos[file_name] = {
        'size': optimal['file_size'],
        'algorithm': optimal['algorithm']
    }

# Categorize by size
size_bins = [0, 1000, 10000, 100000, 1000000, 10000000, float('inf')]
size_labels = ['<1KB', '1-10KB', '10-100KB', '100KB-1MB', '1-10MB', '>10MB']

data_by_size = defaultdict(lambda: defaultdict(int))
for file_data in optimal_algos.values():
    size = file_data['size']
    algo = file_data['algorithm']
    
    for i, (low, high) in enumerate(zip(size_bins[:-1], size_bins[1:])):
        if low <= size < high:
            data_by_size[size_labels[i]][algo] += 1
            break

# Create stacked bar chart
fig, ax = plt.subplots(figsize=(12, 7))

algorithms = ['ChaCha20', 'AES-256', 'AES-128']
colors = {'ChaCha20': '#2ecc71', 'AES-256': '#3498db', 'AES-128': '#e74c3c'}

x = np.arange(len(size_labels))
width = 0.6

bottom = np.zeros(len(size_labels))
for algo in algorithms:
    counts = [data_by_size[size][algo] for size in size_labels]
    ax.bar(x, counts, width, label=algo, bottom=bottom, color=colors[algo], alpha=0.8)
    bottom += counts

ax.set_xlabel('File Size Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Files', fontsize=12, fontweight='bold')
ax.set_title('Optimal Algorithm Distribution by File Size', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(size_labels)
ax.legend(title='Optimal Algorithm', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'algorithm_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'algorithm_distribution.pdf', bbox_inches='tight')
print(f"  ✓ Saved: algorithm_distribution.png")

################################################################################
# FIGURE 3: Energy Savings Comparison
################################################################################
print("[3/6] Generating: Energy Savings Comparison...")

# Calculate savings vs AES-256 baseline
savings_data = []
for file_name in optimal_algos.keys():
    file_results = {r['algorithm']: r['statistics']['median_energy_j'] 
                    for r in results if r['file_name'] == file_name}
    
    aes256_energy = file_results.get('AES-256', 0)
    optimal_energy = min(file_results.values())
    
    if aes256_energy > 0:
        savings_pct = (aes256_energy - optimal_energy) / aes256_energy * 100
        savings_data.append({
            'file': file_name,
            'size': optimal_algos[file_name]['size'],
            'savings': savings_pct,
            'optimal': optimal_algos[file_name]['algorithm']
        })

savings_df = pd.DataFrame(savings_data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
ax1.hist(savings_df['savings'], bins=30, alpha=0.7, color='#3498db', edgecolor='black')
ax1.axvline(savings_df['savings'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {savings_df["savings"].mean():.1f}%')
ax1.axvline(savings_df['savings'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {savings_df["savings"].median():.1f}%')
ax1.set_xlabel('Energy Savings (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Energy Savings vs AES-256 Baseline', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Box plot by algorithm
savings_by_algo = []
for algo in ['AES-128', 'AES-256', 'ChaCha20']:
    algo_savings = savings_df[savings_df['optimal'] == algo]['savings'].tolist()
    savings_by_algo.append(algo_savings)

bp = ax2.boxplot(savings_by_algo, labels=['AES-128', 'AES-256', 'ChaCha20'],
                 patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], ['#e74c3c', '#3498db', '#2ecc71']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_xlabel('Optimal Algorithm', fontsize=12, fontweight='bold')
ax2.set_ylabel('Energy Savings (%)', fontsize=12, fontweight='bold')
ax2.set_title('Energy Savings by Optimal Algorithm', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'energy_savings.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'energy_savings.pdf', bbox_inches='tight')
print(f"  ✓ Saved: energy_savings.png")

################################################################################
# FIGURE 4: Throughput Comparison
################################################################################
print("[4/6] Generating: Throughput Comparison...")

fig, ax = plt.subplots(figsize=(12, 7))

for algo in ['AES-128', 'AES-256', 'ChaCha20']:
    algo_data = [r for r in results if r['algorithm'] == algo]
    sizes = [r['file_size'] for r in algo_data]
    throughputs = [r['statistics']['throughput_mbps'] for r in algo_data]
    
    ax.scatter(sizes, throughputs, alpha=0.6, s=50, label=algo)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('File Size (bytes)', fontsize=12, fontweight='bold')
ax.set_ylabel('Throughput (MB/s)', fontsize=12, fontweight='bold')
ax.set_title('Encryption Throughput by Algorithm', fontsize=14, fontweight='bold')
ax.legend(title='Algorithm', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'throughput_comparison.pdf', bbox_inches='tight')
print(f"  ✓ Saved: throughput_comparison.png")

################################################################################
# FIGURE 5: Energy Efficiency (Energy per MB)
################################################################################
print("[5/6] Generating: Energy Efficiency...")

fig, ax = plt.subplots(figsize=(12, 7))

for algo in ['AES-128', 'AES-256', 'ChaCha20']:
    algo_data = [r for r in results if r['algorithm'] == algo]
    sizes = [r['file_size'] for r in algo_data]
    # Energy per MB
    efficiency = [r['statistics']['median_energy_j'] / (r['file_size'] / 1024 / 1024) 
                  for r in algo_data if r['file_size'] > 0]
    sizes_filtered = [s for s, r in zip(sizes, algo_data) if r['file_size'] > 0]
    
    ax.scatter(sizes_filtered, efficiency, alpha=0.6, s=50, label=algo)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('File Size (bytes)', fontsize=12, fontweight='bold')
ax.set_ylabel('Energy per MB (J/MB)', fontsize=12, fontweight='bold')
ax.set_title('Energy Efficiency by Algorithm', fontsize=14, fontweight='bold')
ax.legend(title='Algorithm', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'energy_efficiency.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'energy_efficiency.pdf', bbox_inches='tight')
print(f"  ✓ Saved: energy_efficiency.png")

################################################################################
# FIGURE 6: Dataset Composition
################################################################################
print("[6/6] Generating: Dataset Composition...")

# Count files by source
composition = {
    'Canterbury': len([r for r in results if 'canterbury' in r['file_name'].lower() and r['algorithm'] == 'AES-128']),
    'Calgary': len([r for r in results if 'calgary' in r['file_name'].lower() and r['algorithm'] == 'AES-128']),
    'Silesia': len([r for r in results if 'silesia' in r['file_name'].lower() and r['algorithm'] == 'AES-128']),
    'Gutenberg': len([r for r in results if any(x in r['file_name'].lower() for x in ['alice', 'moby', 'pride', 'frankenstein', 'sherlock']) and r['algorithm'] == 'AES-128']),
    'Real-world': len([r for r in results if 'realworld' in r['file_name'].lower() and r['algorithm'] == 'AES-128']),
    'Synthetic': len([r for r in results if 'synthetic' in r['file_name'].lower() and r['algorithm'] == 'AES-128'])
}

fig, ax = plt.subplots(figsize=(10, 10))

colors_pie = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#95a5a6']
wedges, texts, autotexts = ax.pie(composition.values(), labels=composition.keys(), autopct='%1.1f%%',
                                    colors=colors_pie, startangle=90, textprops={'fontsize': 12})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax.set_title('Dataset Composition (219 files)', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'dataset_composition.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'dataset_composition.pdf', bbox_inches='tight')
print(f"  ✓ Saved: dataset_composition.png")

################################################################################
# Summary Statistics
################################################################################
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(f"\nTotal files analyzed: {len(optimal_algos)}")
print(f"Total measurements: {len(results)}")

print("\nOptimal algorithm distribution:")
algo_counts = defaultdict(int)
for data in optimal_algos.values():
    algo_counts[data['algorithm']] += 1

for algo, count in sorted(algo_counts.items()):
    pct = count / len(optimal_algos) * 100
    print(f"  {algo:10s}: {count:3d} ({pct:5.1f}%)")

print("\nEnergy savings statistics:")
print(f"  Mean:   {savings_df['savings'].mean():6.2f}%")
print(f"  Median: {savings_df['savings'].median():6.2f}%")
print(f"  Max:    {savings_df['savings'].max():6.2f}%")
print(f"  Min:    {savings_df['savings'].min():6.2f}%")

print("\n" + "=" * 70)
print("ALL FIGURES GENERATED SUCCESSFULLY!")
print("=" * 70)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nGenerated files:")
for f in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  - {f.name}")
print("\nPDF versions also available for publication")
print("=" * 70)
