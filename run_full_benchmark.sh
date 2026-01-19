#!/bin/bash

# CryptoGreen Full Benchmark Runner
# This will run for several hours - don't interrupt!

set -e  # Exit on error

# Configuration
RUNS=100
OUTPUT_DIR="results/benchmarks"
LOG_FILE="results/logs/benchmark_$(date +%Y%m%d_%H%M%S).log"

# Create directories
mkdir -p results/logs results/benchmarks/raw results/benchmarks/processed

# Start logging
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo "=========================================="
echo "CryptoGreen Benchmark Suite"
echo "=========================================="
echo "Start time: $(date)"
echo "Runs per config: $RUNS"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# System info
echo "System Information:"
echo "-------------------"
uname -a
cat /proc/cpuinfo | grep "model name" | head -1
lscpu | grep "CPU(s):"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo ""

# Check RAPL
echo "Checking RAPL availability..."
if [ ! -r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj ]; then
    echo "ERROR: RAPL not accessible!"
    echo "Run: sudo chmod -R a+r /sys/class/powercap/intel-rapl/"
    exit 1
fi
echo "✅ RAPL accessible"
echo ""

# Check test files
echo "Checking test files..."
FILE_COUNT=$(find data/test_files -type f -name '*.*' ! -name '.gitkeep' | wc -l)
if [ "$FILE_COUNT" -ne 49 ]; then
    echo "ERROR: Expected 49 test files, found $FILE_COUNT"
    echo "Run: python scripts/generate_test_data.py"
    exit 1
fi
echo "✅ Found $FILE_COUNT test files"
echo ""

# Estimate time
TOTAL_MEASUREMENTS=$((3 * 49 * RUNS))  # 3 algorithms × 49 files × 100 runs
ESTIMATED_HOURS=$((TOTAL_MEASUREMENTS / 3600))  # Rough estimate: 1 measurement/second
echo "Estimated measurements: $TOTAL_MEASUREMENTS"
echo "Estimated time: ~$ESTIMATED_HOURS hours"
echo ""

# Confirm
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Run benchmark
echo "Starting benchmark at $(date)..."
echo ""

python scripts/run_benchmarks.py \
    --runs $RUNS \
    --algorithms AES-128,AES-256,ChaCha20 \
    --output $OUTPUT_DIR \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results saved to: $OUTPUT_DIR"
echo "Log saved to: $LOG_FILE"

# Count results
RESULT_COUNT=$(find $OUTPUT_DIR/raw -name "*.json" | wc -l)
echo "Result files generated: $RESULT_COUNT"

# Re-enable sleep/suspend
sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target 2>/dev/null || true

exit $EXIT_CODE
