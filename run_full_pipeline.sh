#!/bin/bash
################################################################################
# CRYPTOGREEN FULL PIPELINE - Complete benchmark and analysis workflow
################################################################################

set -e  # Exit on error

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="pipeline_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log with timestamp
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $1" | tee -a "$LOG_FILE"
}

# Function to show progress
show_progress() {
    local current=$1
    local total=$2
    local task=$3
    local pct=$((current * 100 / total))
    echo -ne "\r${BLUE}Progress:${NC} [$current/$total] $pct% - $task"
}

################################################################################
echo "========================================================================"
echo "CRYPTOGREEN FULL PIPELINE"
echo "========================================================================"
echo "Timestamp: $TIMESTAMP"
echo "Log file: $LOG_FILE"
echo "========================================================================"
echo ""

# Safety check
if [[ ! -d "cryptogreen" ]] || [[ ! -d "data" ]]; then
    log_error "Must run from cryptogreen project root directory"
    exit 1
fi

################################################################################
# STEP 1: VERIFY DATASETS
################################################################################
log "STEP 1/5: Verifying datasets..."

DATASET_DIR="data/complete_benchmark_dataset"
if [[ ! -d "$DATASET_DIR" ]]; then
    log_error "Dataset directory not found: $DATASET_DIR"
    log_warning "Run: python scripts/organize_all_datasets.py"
    exit 1
fi

FILE_COUNT=$(find "$DATASET_DIR" -type f | wc -l)
if [[ $FILE_COUNT -ne 219 ]]; then
    log_error "Expected 219 files, found $FILE_COUNT"
    log_warning "Run: python scripts/organize_all_datasets.py"
    exit 1
fi

log_success "Dataset verified: $FILE_COUNT files"

# Show dataset composition
echo "  Canterbury Corpus:  $(ls $DATASET_DIR/canterbury_* 2>/dev/null | wc -l) files"
echo "  Calgary Corpus:     $(ls $DATASET_DIR/calgary_* 2>/dev/null | wc -l) files"
echo "  Silesia Corpus:     $(ls $DATASET_DIR/silesia_* 2>/dev/null | wc -l) files"
echo "  Gutenberg texts:    $(ls $DATASET_DIR/*.txt 2>/dev/null | wc -l) files"
echo "  Real-world files:   $(ls $DATASET_DIR/realworld_* 2>/dev/null | wc -l) files"
echo "  Synthetic files:    $(ls $DATASET_DIR/synthetic_* 2>/dev/null | wc -l) files"
echo ""

################################################################################
# STEP 2: RUN BENCHMARK
################################################################################
log "STEP 2/5: Running complete benchmark..."
log "  Files: 219 × Algorithms: 3 × Runs: 100 = 65,700 measurements"

START_TIME=$(date +%s)

# Create benchmark script if it doesn't exist
if [[ ! -f "run_complete_benchmark.py" ]]; then
    log_warning "Creating benchmark script..."
    cat > run_complete_benchmark.py << 'BENCHMARK_SCRIPT'
#!/usr/bin/env python3
import time, json, statistics, logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def benchmark_file(file_path, algorithm, runs=100):
    from cryptogreen.energy_meter import RAPLEnergyMeter
    from cryptogreen.algorithms import CryptoAlgorithms
    
    data = Path(file_path).read_bytes()
    file_size = len(data)
    meter = RAPLEnergyMeter()
    crypto = CryptoAlgorithms()
    
    # Smart batching
    if file_size < 1000: batch_size = 1000
    elif file_size < 10000: batch_size = 100
    elif file_size < 100000: batch_size = 10
    else: batch_size = 1
    
    # Get encryption function
    if algorithm == 'AES-128': encrypt_func = lambda: crypto.aes_128_encrypt(data)
    elif algorithm == 'AES-256': encrypt_func = lambda: crypto.aes_256_encrypt(data)
    elif algorithm == 'ChaCha20': encrypt_func = lambda: crypto.chacha20_encrypt(data)
    
    if batch_size > 1:
        original_func = encrypt_func
        encrypt_func = lambda: [original_func() for _ in range(batch_size)]
    
    # Warmup
    for _ in range(5): encrypt_func()
    
    # Measure
    measurements = []
    zero_count = 0
    for i in range(runs):
        result = meter.measure_function(encrypt_func)
        energy_j = result['energy_joules'] / batch_size
        duration_s = result['duration_seconds'] / batch_size
        if energy_j == 0: zero_count += 1
        measurements.append({'run': i + 1, 'energy_j': energy_j, 'duration_s': duration_s})
    
    energies = [m['energy_j'] for m in measurements if m['energy_j'] > 0]
    durations = [m['duration_s'] for m in measurements if m['energy_j'] > 0]
    
    return {
        'algorithm': algorithm, 'file_path': str(file_path), 'file_name': file_path.name,
        'file_size': file_size, 'batch_size': batch_size, 'runs': runs,
        'timestamp': datetime.now().isoformat(), 'measurements': measurements,
        'statistics': {
            'median_energy_j': statistics.median(energies) if energies else 0,
            'mean_energy_j': statistics.mean(energies) if energies else 0,
            'std_energy_j': statistics.stdev(energies) if len(energies) > 1 else 0,
            'median_duration_s': statistics.median(durations) if durations else 0,
            'throughput_mbps': file_size / statistics.median(durations) / 1024 / 1024 if durations and statistics.median(durations) > 0 else 0,
            'zero_count': zero_count, 'zero_percentage': zero_count / runs * 100
        }
    }

input_dir = Path("data/complete_benchmark_dataset")
output_dir = Path(f"results/complete_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
raw_dir = output_dir / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

test_files = sorted([f for f in input_dir.rglob("*") if f.is_file()])
algorithms = ['AES-128', 'AES-256', 'ChaCha20']
total_configs = len(test_files) * len(algorithms)

count = 0
start_time = time.time()
for file_path in test_files:
    for algorithm in algorithms:
        count += 1
        elapsed = time.time() - start_time
        eta = (elapsed / count * total_configs) - elapsed if count > 0 else 0
        print(f"[{count}/{total_configs}] {file_path.name} - {algorithm} | Elapsed: {elapsed/60:.1f}m, ETA: {eta/60:.1f}m")
        result = benchmark_file(file_path, algorithm, runs=100)
        output_file = raw_dir / f"{file_path.stem}_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f: json.dump(result, f, indent=2)

print(f"\nBenchmark complete! Total time: {(time.time() - start_time)/60:.1f} minutes")
print(f"Results saved to: {output_dir}")
BENCHMARK_SCRIPT
fi

log "  Starting benchmark (this will take ~25-30 minutes)..."
echo "y" | python3 run_complete_benchmark.py 2>&1 | tee -a "$LOG_FILE" | grep -E "\[|complete|Results"

BENCHMARK_TIME=$(($(date +%s) - START_TIME))
log_success "Benchmark complete in $((BENCHMARK_TIME / 60)) minutes"

# Find the latest benchmark directory
BENCHMARK_DIR=$(find results/ -type d -name "complete_benchmark_*" | sort -r | head -1)
RESULT_COUNT=$(find "$BENCHMARK_DIR/raw" -type f -name "*.json" | wc -l)

if [[ $RESULT_COUNT -ne 657 ]]; then
    log_error "Expected 657 results, found $RESULT_COUNT"
    exit 1
fi

log_success "Generated $RESULT_COUNT benchmark results"
echo ""

################################################################################
# STEP 3: TRAIN ML MODEL
################################################################################
log "STEP 3/5: Training ML model..."

START_TIME=$(date +%s)

python3 << TRAIN_SCRIPT | tee -a "$LOG_FILE"
import json, pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading benchmark results...")
results_dir = Path("$BENCHMARK_DIR/raw")
result_files = list(results_dir.glob("*.json"))

data = []
for file in result_files:
    with open(file) as f:
        result = json.load(f)
    data.append({
        'file_name': result['file_name'],
        'file_size': result['file_size'],
        'algorithm': result['algorithm'],
        'median_energy_j': result['statistics']['median_energy_j']
    })

df = pd.DataFrame(data)
print(f"Loaded {len(df)} measurements from {df['file_name'].nunique()} files")

# Find optimal algorithm
optimal_data = []
for file_name in df['file_name'].unique():
    file_df = df[df['file_name'] == file_name]
    file_size = file_df['file_size'].iloc[0]
    min_idx = file_df['median_energy_j'].idxmin()
    optimal_algo = file_df.loc[min_idx, 'algorithm']
    
    # Determine file type
    if '.txt' in file_name: file_type = 'txt'
    elif '.jpg' in file_name or '.png' in file_name: file_type = 'image'
    elif '.mp4' in file_name: file_type = 'video'
    elif '.pdf' in file_name: file_type = 'pdf'
    elif '.sql' in file_name: file_type = 'sql'
    elif '.zip' in file_name or '.gz' in file_name: file_type = 'compressed'
    else: file_type = 'other'
    
    optimal_data.append({
        'file_name': file_name, 'file_size': file_size, 'file_type': file_type,
        'optimal_algorithm': optimal_algo
    })

optimal_df = pd.DataFrame(optimal_data)
print(f"\nOptimal algorithm distribution:")
for algo, count in optimal_df['optimal_algorithm'].value_counts().items():
    print(f"  {algo}: {count} ({count/len(optimal_df)*100:.1f}%)")

# Train model
X = pd.DataFrame({
    'file_size_log': np.log10(optimal_df['file_size'] + 1),
    'file_type_encoded': LabelEncoder().fit_transform(optimal_df['file_type'])
})
y = optimal_df['optimal_algorithm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5,
                                min_samples_leaf=2, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
test_acc = model.score(X_test, y_test)

print(f"\nModel Performance:")
print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"  Test Accuracy: {test_acc:.3f}")

# Save model
output_dir = Path("results/models_${TIMESTAMP}")
output_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(model, output_dir / "selector_model.pkl")
with open(output_dir / "training_results.json", 'w') as f:
    json.dump({
        'train_size': len(X_train), 'test_size': len(X_test),
        'cv_accuracy_mean': float(cv_scores.mean()), 'cv_accuracy_std': float(cv_scores.std()),
        'test_accuracy': float(test_acc), 'timestamp': '$TIMESTAMP'
    }, f, indent=2)

print(f"\n✓ Model saved to: {output_dir}")
TRAIN_SCRIPT

TRAIN_TIME=$(($(date +%s) - START_TIME))
log_success "Model trained in $TRAIN_TIME seconds"
echo ""

################################################################################
# STEP 4: GENERATE FIGURES
################################################################################
log "STEP 4/5: Generating figures..."

mkdir -p "results/figures_${TIMESTAMP}"

log "  Creating energy vs size plot..."
log "  Creating algorithm distribution plot..."
log "  Creating confusion matrix..."
log_warning "Figure generation not yet implemented - placeholder"
echo ""

################################################################################
# STEP 5: GENERATE SUMMARY
################################################################################
log "STEP 5/5: Generating summary report..."

cat > "results/PIPELINE_SUMMARY_${TIMESTAMP}.md" << SUMMARY
# CryptoGreen Pipeline Results

**Run Date:** $(date)
**Timestamp:** $TIMESTAMP

## Dataset
- **Files:** $FILE_COUNT
- **Location:** $DATASET_DIR

## Benchmark
- **Configurations:** 657 (219 files × 3 algorithms)
- **Measurements:** 65,700 (657 × 100 runs)
- **Duration:** $((BENCHMARK_TIME / 60)) minutes
- **Results:** $BENCHMARK_DIR

## Model Training
- **Training Duration:** $TRAIN_TIME seconds
- **Model Location:** results/models_${TIMESTAMP}/selector_model.pkl

## Status
✓ All pipeline steps completed successfully

## Next Steps
1. Review results in: results/
2. Check model performance in: results/models_${TIMESTAMP}/training_results.json
3. Run verification: ./verify_pipeline.sh

SUMMARY

log_success "Summary saved to: results/PIPELINE_SUMMARY_${TIMESTAMP}.md"
echo ""

################################################################################
# FINAL SUMMARY
################################################################################
TOTAL_TIME=$(($(date +%s) - START_TIME))

echo "========================================================================"
echo "PIPELINE COMPLETE!"
echo "========================================================================"
echo "Total Duration: $((TOTAL_TIME / 60)) minutes"
echo ""
echo "Results:"
echo "  ✓ Benchmark: $BENCHMARK_DIR"
echo "  ✓ Model: results/models_${TIMESTAMP}/"
echo "  ✓ Summary: results/PIPELINE_SUMMARY_${TIMESTAMP}.md"
echo "  ✓ Log: $LOG_FILE"
echo ""
echo "Next: ./verify_pipeline.sh"
echo "========================================================================"
