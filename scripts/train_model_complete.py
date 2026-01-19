#!/usr/bin/env python3
"""
Train ML model on complete benchmark dataset (219 files)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from collections import Counter

print("=" * 70)
print("TRAINING ML MODEL - COMPLETE DATASET")
print("=" * 70)

# Load all benchmark results
results_dir = Path("results/complete_benchmark/raw")
result_files = list(results_dir.glob("*.json"))

print(f"Loading {len(result_files)} result files...")

# Parse results
data = []
for file in result_files:
    with open(file) as f:
        result = json.load(f)
    
    data.append({
        'file_name': result['file_name'],
        'file_size': result['file_size'],
        'algorithm': result['algorithm'],
        'median_energy_j': result['statistics']['median_energy_j'],
    })

df = pd.DataFrame(data)

print(f"\nLoaded {len(df)} measurements")
print(f"Unique files: {df['file_name'].nunique()}")
print(f"Algorithms: {df['algorithm'].unique()}")

# Find optimal algorithm for each file
print("\nFinding optimal algorithm for each file...")
optimal_data = []

for file_name in df['file_name'].unique():
    file_df = df[df['file_name'] == file_name]
    
    # Get file size
    file_size = file_df['file_size'].iloc[0]
    
    # Find algorithm with minimum energy
    min_idx = file_df['median_energy_j'].idxmin()
    optimal_algo = file_df.loc[min_idx, 'algorithm']
    min_energy = file_df.loc[min_idx, 'median_energy_j']
    
    # Get all energies
    energies = {row['algorithm']: row['median_energy_j'] for _, row in file_df.iterrows()}
    
    # Determine file type
    if '.txt' in file_name:
        file_type = 'txt'
    elif '.jpg' in file_name or '.png' in file_name:
        file_type = 'image'
    elif '.mp4' in file_name:
        file_type = 'video'
    elif '.pdf' in file_name:
        file_type = 'pdf'
    elif '.sql' in file_name:
        file_type = 'sql'
    elif '.zip' in file_name or '.gz' in file_name:
        file_type = 'compressed'
    else:
        file_type = 'other'
    
    optimal_data.append({
        'file_name': file_name,
        'file_size': file_size,
        'file_type': file_type,
        'optimal_algorithm': optimal_algo,
        'min_energy_j': min_energy,
        'aes128_energy': energies.get('AES-128', 0),
        'aes256_energy': energies.get('AES-256', 0),
        'chacha20_energy': energies.get('ChaCha20', 0),
    })

optimal_df = pd.DataFrame(optimal_data)

print(f"\nOptimal algorithm distribution:")
algo_dist = optimal_df['optimal_algorithm'].value_counts()
for algo, count in algo_dist.items():
    pct = count / len(optimal_df) * 100
    print(f"  {algo:15s}: {count:3d} ({pct:5.1f}%)")

# Show patterns by size
print("\nOptimal algorithm by file size:")
size_bins = [0, 1000, 10000, 100000, 1000000, 10000000, float('inf')]
size_labels = ['<1KB', '1-10KB', '10-100KB', '100KB-1MB', '1-10MB', '>10MB']
optimal_df['size_bin'] = pd.cut(optimal_df['file_size'], bins=size_bins, labels=size_labels)

for size_label in size_labels:
    size_df = optimal_df[optimal_df['size_bin'] == size_label]
    if len(size_df) > 0:
        dist = size_df['optimal_algorithm'].value_counts()
        print(f"\n  {size_label:12s} ({len(size_df):3d} files):")
        for algo, count in dist.items():
            pct = count / len(size_df) * 100
            print(f"    {algo:15s}: {count:3d} ({pct:5.1f}%)")

# Prepare features
print("\n" + "=" * 70)
print("TRAINING MODEL")
print("=" * 70)

X = pd.DataFrame({
    'file_size_log': np.log10(optimal_df['file_size'] + 1),
    'file_type_encoded': LabelEncoder().fit_transform(optimal_df['file_type'])
})

y = optimal_df['optimal_algorithm']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# Cross-validation
print("\nCross-validation (5-fold)...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"CV Scores: {[f'{s:.3f}' for s in cv_scores]}")

# Test accuracy
test_acc = model.score(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.3f}")

# Feature importance
print("\nFeature importance:")
for feat, imp in zip(X.columns, model.feature_importances_):
    print(f"  {feat:20s}: {imp:.3f} ({imp*100:.1f}%)")

# Calculate energy savings vs baseline (AES-256)
baseline_energy = optimal_df['aes256_energy'].sum()
optimal_energy = optimal_df['min_energy_j'].sum()
savings_pct = (baseline_energy - optimal_energy) / baseline_energy * 100

print(f"\n" + "=" * 70)
print("ENERGY SAVINGS")
print("=" * 70)
print(f"Baseline (AES-256) total: {baseline_energy:.3f} J")
print(f"Optimal selection total:  {optimal_energy:.3f} J")
print(f"Energy savings: {savings_pct:.1f}%")

# Per-file savings
optimal_df['savings_pct'] = (optimal_df['aes256_energy'] - optimal_df['min_energy_j']) / optimal_df['aes256_energy'] * 100
print(f"\nPer-file savings statistics:")
print(f"  Mean:   {optimal_df['savings_pct'].mean():.1f}%")
print(f"  Median: {optimal_df['savings_pct'].median():.1f}%")
print(f"  Max:    {optimal_df['savings_pct'].max():.1f}%")
print(f"  Min:    {optimal_df['savings_pct'].min():.1f}%")

# Save model
output_dir = Path("results/models")
output_dir.mkdir(parents=True, exist_ok=True)

model_file = output_dir / "selector_model_complete.pkl"
joblib.dump(model, model_file)
print(f"\n✓ Model saved: {model_file}")

# Save results
results = {
    'train_size': len(X_train),
    'test_size': len(X_test),
    'total_files': len(optimal_df),
    'cv_accuracy_mean': float(cv_scores.mean()),
    'cv_accuracy_std': float(cv_scores.std()),
    'cv_scores': [float(s) for s in cv_scores],
    'test_accuracy': float(test_acc),
    'feature_importance': {
        feat: float(imp) for feat, imp in zip(X.columns, model.feature_importances_)
    },
    'algorithm_distribution': {
        algo: int(count) for algo, count in algo_dist.items()
    },
    'energy_savings_pct': float(savings_pct),
    'savings_stats': {
        'mean': float(optimal_df['savings_pct'].mean()),
        'median': float(optimal_df['savings_pct'].median()),
        'max': float(optimal_df['savings_pct'].max()),
        'min': float(optimal_df['savings_pct'].min()),
    }
}

results_file = output_dir / "training_results_complete.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Results saved: {results_file}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
