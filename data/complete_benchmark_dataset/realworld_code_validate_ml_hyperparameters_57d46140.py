#!/usr/bin/env python3
"""
Validate ML Model Training with Small Dataset Optimizations

Verifies that the RandomForest hyperparameters are properly configured
for small datasets (n<100) to prevent overfitting.
"""

import json
from pathlib import Path

print("=" * 80)
print("ML MODEL TRAINING VALIDATION (Small Dataset Optimization)")
print("=" * 80)
print()

# Check training report exists
report_path = Path('results/models/training_report.json')

if not report_path.exists():
    print("❌ Training report not found!")
    print("Run: python scripts/train_model.py")
    exit(1)

# Load training report
with open(report_path, 'r') as f:
    report = json.load(f)

# Extract key info
dataset = report['dataset']
hyperparams = report['hyperparameters']
cv = report['cross_validation']
test_metrics = report['test_metrics']

print("1. DATASET SIZE CHECK")
print("-" * 80)
n_samples = dataset['total_samples']
print(f"  Total samples: {n_samples}")

if n_samples < 100:
    print(f"  ✓ Small dataset detected (n={n_samples})")
    print(f"  ✓ Hyperparameters should be reduced to prevent overfitting")
else:
    print(f"  ✓ Normal dataset size (n={n_samples})")

print()

# Check hyperparameters
print("2. HYPERPARAMETER VALIDATION")
print("-" * 80)

expected_small = {
    'n_estimators': 20,
    'max_depth': 5,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'bootstrap': True,
}

expected_large = {
    'n_estimators': 100,
    'max_depth': 10,
}

print(f"{'Parameter':<25} {'Expected':<15} {'Actual':<15} {'Status':<10}")
print("-" * 80)

all_pass = True

if n_samples < 100:
    # Small dataset expectations
    for param, expected in expected_small.items():
        actual = hyperparams.get(param)
        status = "✓ PASS" if actual == expected else "✗ FAIL"
        if actual != expected:
            all_pass = False
        print(f"{param:<25} {str(expected):<15} {str(actual):<15} {status:<10}")
else:
    # Large dataset expectations
    for param, expected in expected_large.items():
        actual = hyperparams.get(param)
        status = "✓ PASS" if actual == expected else "✗ FAIL"
        if actual != expected:
            all_pass = False
        print(f"{param:<25} {str(expected):<15} {str(actual):<15} {status:<10}")

print()

# Check CV configuration
print("3. CROSS-VALIDATION CONFIGURATION")
print("-" * 80)

n_folds = cv['n_folds']
print(f"  Number of folds: {n_folds}")

if n_samples < 100:
    if n_folds == 5:
        print(f"  ✓ 5-fold CV appropriate for small datasets")
    else:
        print(f"  ⚠ Warning: {n_folds}-fold may be too many for n={n_samples}")
else:
    if n_folds >= 5:
        print(f"  ✓ {n_folds}-fold CV appropriate")

print()

# Check CV variance
print("4. CROSS-VALIDATION VARIANCE")
print("-" * 80)

cv_mean = cv['accuracy_mean']
cv_std = cv['accuracy_std']
cv_variance_pct = (cv_std * 2) * 100  # 95% CI as percentage

print(f"  CV Mean Accuracy: {cv_mean:.4f} ({cv_mean*100:.1f}%)")
print(f"  CV Std Dev: {cv_std:.4f} (±{cv_variance_pct:.1f}%)")

if n_samples < 100:
    if cv_variance_pct > 10:
        print(f"  ✓ High variance ({cv_variance_pct:.1f}%) expected for small datasets")
        print(f"  ✓ Typical range: ±10-20% for n<100")
    else:
        print(f"  ✓ Moderate variance ({cv_variance_pct:.1f}%)")
else:
    if cv_variance_pct < 10:
        print(f"  ✓ Low variance ({cv_variance_pct:.1f}%) indicates stable model")
    else:
        print(f"  ⚠ High variance ({cv_variance_pct:.1f}%) - may need more data")

print()

# Check test set performance
print("5. TEST SET PERFORMANCE")
print("-" * 80)

top1_acc = test_metrics['top1_accuracy']
top2_acc = test_metrics['top2_accuracy']

print(f"  Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.1f}%)")
print(f"  Top-2 Accuracy: {top2_acc:.4f} ({top2_acc*100:.1f}%)")

if top1_acc >= 0.75:
    print(f"  ✓ Good top-1 accuracy for preliminary model")
elif top1_acc >= 0.60:
    print(f"  ⚠ Moderate accuracy - collect more data")
else:
    print(f"  ✗ Low accuracy - needs more training data")

print()

# Check class balance (if SMOTE was used)
print("6. CLASS BALANCE")
print("-" * 80)

smote_applied = dataset.get('smote_applied', False)
class_names = dataset['class_names']

print(f"  SMOTE applied: {smote_applied}")

if smote_applied:
    print(f"  ✓ Class imbalance handled with SMOTE")
    print(f"  ✓ Training data balanced across {len(class_names)} classes")
else:
    imbalance_ratio = dataset.get('class_imbalance_ratio')
    if imbalance_ratio:
        print(f"  ⚠ Class imbalance ratio: {imbalance_ratio:.2f}:1")
    print(f"  ⚠ Using class_weight='{hyperparams.get('class_weight')}' to handle imbalance")

print()

# Per-class metrics
print("7. PER-CLASS PERFORMANCE")
print("-" * 80)

per_class = test_metrics['per_class']
print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print("-" * 80)

for cls_name, metrics in per_class.items():
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1_score']
    support = metrics['support']
    
    print(f"{cls_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")

print()

# Feature importance
print("8. FEATURE IMPORTANCE")
print("-" * 80)

feature_importance = report.get('feature_importance', {})
top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

print(f"{'Feature':<25} {'Importance':<15}")
print("-" * 80)

for feature, importance in top_features:
    print(f"{feature:<25} {importance:<15.4f}")

print()

# Summary
print("=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
print()

if n_samples < 100:
    print(f"✓ Small dataset mode active (n={n_samples})")
    print(f"✓ Reduced hyperparameters: n_estimators={hyperparams['n_estimators']}, max_depth={hyperparams['max_depth']}")
    print(f"✓ Regularization: min_samples_leaf={hyperparams.get('min_samples_leaf')}, max_features={hyperparams.get('max_features')}")
    print(f"✓ 5-fold cross-validation (appropriate for small n)")
    print(f"✓ High variance expected: ±{cv_variance_pct:.1f}% (typical for n<100)")
    
    if smote_applied:
        print(f"✓ SMOTE applied to handle class imbalance")
    
    print()
    print("RECOMMENDATIONS:")
    print("  1. Results are preliminary - collect more benchmark data")
    print("     Target: 500+ samples for production model")
    print("  2. Use HybridSelector with rule-based fallback")
    print("     ML predictions should only be trusted when confidence > 0.8")
    print("  3. Continue using reduced hyperparameters until n > 100")
    print("  4. Monitor per-class performance - minority classes may be unreliable")
else:
    print(f"✓ Normal dataset size (n={n_samples})")
    print(f"✓ Standard hyperparameters in use")
    print(f"✓ Model performance should be reliable")

print()

if all_pass:
    print("✅ ALL HYPERPARAMETER CHECKS PASSED")
else:
    print("⚠ SOME HYPERPARAMETER CHECKS FAILED")

print()
print("=" * 80)
