#!/bin/bash
################################################################################
# VERIFICATION SCRIPT - Check pipeline results
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; return 1; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }

ERRORS=0

echo "========================================================================"
echo "CRYPTOGREEN PIPELINE VERIFICATION"
echo "========================================================================"
echo ""

################################################################################
# 1. Check Dataset
################################################################################
echo "1. Dataset Verification"
echo "   ----------------------"

DATASET_DIR="data/complete_benchmark_dataset"
if [[ -d "$DATASET_DIR" ]]; then
    FILE_COUNT=$(find "$DATASET_DIR" -type f | wc -l)
    if [[ $FILE_COUNT -eq 219 ]]; then
        pass "Dataset exists with 219 files"
    else
        fail "Dataset has $FILE_COUNT files (expected 219)" || ERRORS=$((ERRORS + 1))
    fi
else
    fail "Dataset directory not found: $DATASET_DIR" || ERRORS=$((ERRORS + 1))
fi

# Check dataset composition
CANTERBURY=$(ls $DATASET_DIR/canterbury_* 2>/dev/null | wc -l)
CALGARY=$(ls $DATASET_DIR/calgary_* 2>/dev/null | wc -l)
SILESIA=$(ls $DATASET_DIR/silesia_* 2>/dev/null | wc -l)

echo "   Dataset composition:"
echo "     - Canterbury: $CANTERBURY files (expected: 11)"
echo "     - Calgary: $CALGARY files (expected: 18)"
echo "     - Silesia: $SILESIA files (expected: 6)"
echo ""

################################################################################
# 2. Check Benchmark Results
################################################################################
echo "2. Benchmark Results Verification"
echo "   -------------------------------"

BENCHMARK_DIRS=$(find results/ -type d -name "complete_benchmark_*" 2>/dev/null)
if [[ -z "$BENCHMARK_DIRS" ]]; then
    fail "No benchmark results found" || ERRORS=$((ERRORS + 1))
else
    LATEST_BENCHMARK=$(echo "$BENCHMARK_DIRS" | sort -r | head -1)
    RESULT_COUNT=$(find "$LATEST_BENCHMARK/raw" -type f -name "*.json" 2>/dev/null | wc -l)
    
    if [[ $RESULT_COUNT -eq 657 ]]; then
        pass "Benchmark generated 657 result files (219 × 3)"
    else
        fail "Benchmark has $RESULT_COUNT results (expected 657)" || ERRORS=$((ERRORS + 1))
    fi
    
    echo "   Latest benchmark: $LATEST_BENCHMARK"
    
    # Check for zero measurements
    ZERO_COUNT=$(grep -l '"median_energy_j": 0' "$LATEST_BENCHMARK/raw"/*.json 2>/dev/null | wc -l)
    if [[ $ZERO_COUNT -eq 0 ]]; then
        pass "No zero energy measurements (smart batching working)"
    else
        warn "Found $ZERO_COUNT files with zero measurements"
    fi
fi
echo ""

################################################################################
# 3. Check ML Model
################################################################################
echo "3. ML Model Verification"
echo "   ---------------------"

MODEL_DIRS=$(find results/ -type d -name "models_*" 2>/dev/null)
if [[ -z "$MODEL_DIRS" ]]; then
    fail "No trained models found" || ERRORS=$((ERRORS + 1))
else
    LATEST_MODEL=$(echo "$MODEL_DIRS" | sort -r | head -1)
    
    if [[ -f "$LATEST_MODEL/selector_model.pkl" ]]; then
        pass "Model file exists: selector_model.pkl"
    else
        fail "Model file not found" || ERRORS=$((ERRORS + 1))
    fi
    
    if [[ -f "$LATEST_MODEL/training_results.json" ]]; then
        pass "Training results exist"
        
        # Extract metrics
        CV_ACC=$(grep -o '"cv_accuracy_mean": [0-9.]*' "$LATEST_MODEL/training_results.json" | cut -d' ' -f2)
        TEST_ACC=$(grep -o '"test_accuracy": [0-9.]*' "$LATEST_MODEL/training_results.json" | cut -d' ' -f2)
        
        echo "   Model Performance:"
        echo "     - CV Accuracy: $(awk "BEGIN {printf \"%.1f\", $CV_ACC * 100}")%"
        echo "     - Test Accuracy: $(awk "BEGIN {printf \"%.1f\", $TEST_ACC * 100}")%"
        
        # Check if accuracy meets threshold
        if (( $(awk "BEGIN {print ($TEST_ACC >= 0.70)}") )); then
            pass "Model achieves >70% test accuracy"
        else
            warn "Model test accuracy below 70%"
        fi
    else
        fail "Training results not found" || ERRORS=$((ERRORS + 1))
    fi
    
    echo "   Latest model: $LATEST_MODEL"
fi
echo ""

################################################################################
# 4. Check Figures
################################################################################
echo "4. Figures Verification"
echo "   --------------------"

FIGURE_DIRS=$(find results/ -type d -name "figures_*" 2>/dev/null)
if [[ -z "$FIGURE_DIRS" ]]; then
    warn "No figure directories found (figures generation may be pending)"
else
    LATEST_FIGURES=$(echo "$FIGURE_DIRS" | sort -r | head -1)
    FIGURE_COUNT=$(find "$LATEST_FIGURES" -type f \( -name "*.png" -o -name "*.pdf" \) 2>/dev/null | wc -l)
    
    if [[ $FIGURE_COUNT -gt 0 ]]; then
        pass "Found $FIGURE_COUNT figures"
    else
        warn "No figures generated yet"
    fi
    
    echo "   Latest figures: $LATEST_FIGURES"
fi
echo ""

################################################################################
# 5. Check Summary
################################################################################
echo "5. Summary Report Verification"
echo "   ---------------------------"

SUMMARIES=$(find results/ -type f -name "PIPELINE_SUMMARY_*.md" 2>/dev/null)
if [[ -z "$SUMMARIES" ]]; then
    warn "No pipeline summary found"
else
    LATEST_SUMMARY=$(echo "$SUMMARIES" | sort -r | head -1)
    pass "Pipeline summary exists"
    echo "   Latest summary: $LATEST_SUMMARY"
fi
echo ""

################################################################################
# FINAL VERDICT
################################################################################
echo "========================================================================"
if [[ $ERRORS -eq 0 ]]; then
    echo -e "${GREEN}VERIFICATION PASSED${NC} - All checks successful!"
else
    echo -e "${RED}VERIFICATION FAILED${NC} - Found $ERRORS errors"
fi
echo "========================================================================"

exit $ERRORS
