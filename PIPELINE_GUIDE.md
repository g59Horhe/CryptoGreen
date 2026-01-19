# CryptoGreen Pipeline - Quick Reference

## Complete Workflow (Fresh Start)

### Step 1: Cleanup Old Results
```bash
./cleanup_and_rebuild.sh
# Type 'y' to confirm
# This deletes old results but keeps all datasets (3.6 GB)
```

### Step 2: Run Full Pipeline
```bash
./run_full_pipeline.sh
# This will:
# - Verify 219 files in dataset
# - Run benchmark (~25-30 minutes)
# - Train ML model (~10 seconds)
# - Generate figures (pending)
# - Create summary report
```

### Step 3: Verify Results
```bash
./verify_pipeline.sh
# Checks:
# ✓ Dataset has 219 files
# ✓ Benchmark generates 657 results
# ✓ Model achieves >70% accuracy
# ✓ All outputs created
```

## Current Status

**What you have now:**
- ✓ 219 files in `data/complete_benchmark_dataset/`
- ✓ Completed benchmark in `results/complete_benchmark/` (657 files)
- ✓ Trained model in `results/models/` (71.3% CV accuracy)
- ✓ Summary in `DATASET_EXPANSION_RESULTS.md`

**What the new pipeline will do:**
- Clean up old scattered results
- Re-run benchmark with timestamp organization
- Retrain model with timestamp
- Generate all figures
- Create comprehensive summary

## Scripts Overview

### cleanup_and_rebuild.sh
- **Purpose**: Delete old results but keep datasets
- **Safe**: Interactive confirmation required
- **Keeps**: All data/ directories (3.6 GB)
- **Deletes**: results/, logs, cache (~33 MB)

### run_full_pipeline.sh
- **Purpose**: Complete end-to-end pipeline
- **Duration**: ~30 minutes total
- **Steps**: 5 stages with progress tracking
- **Outputs**: Timestamped results/
- **Idempotent**: Can run multiple times safely

### verify_pipeline.sh
- **Purpose**: Validate all pipeline outputs
- **Checks**: 
  - Dataset: 219 files
  - Benchmark: 657 results (219 × 3)
  - Model: >70% accuracy
  - Figures: All generated
  - Summary: Complete report
- **Exit code**: 0 = success, >0 = failures

## File Organization

### Before Pipeline Run
```
cryptogreen/
├── data/
│   └── complete_benchmark_dataset/  (219 files, 3.6 GB)
├── cryptogreen/                     (source code)
├── scripts/                         (utility scripts)
└── results/                         (old scattered results)
```

### After Pipeline Run
```
cryptogreen/
├── data/
│   └── complete_benchmark_dataset/  (219 files, unchanged)
├── cryptogreen/                     (source code)
├── scripts/                         (utility scripts)
├── results/
│   ├── complete_benchmark_20260119_193000/
│   │   └── raw/                     (657 JSON files)
│   ├── models_20260119_193000/
│   │   ├── selector_model.pkl
│   │   └── training_results.json
│   ├── figures_20260119_193000/
│   │   └── (plots)
│   └── PIPELINE_SUMMARY_20260119_193000.md
└── pipeline_20260119_193000.log    (full execution log)
```

## Timestamps

All outputs are timestamped (YYYYMMDD_HHMMSS) so you can:
- Run pipeline multiple times without conflicts
- Compare results across runs
- Keep historical data for validation

## Quick Commands

```bash
# Check current status
./verify_pipeline.sh

# Fresh complete run
./cleanup_and_rebuild.sh  # Type 'y'
./run_full_pipeline.sh    # Wait ~30 min
./verify_pipeline.sh      # Check results

# View progress during run
tail -f pipeline_*.log

# Count current benchmark results
ls results/complete_benchmark_*/raw/*.json 2>/dev/null | wc -l

# Check model accuracy
cat results/models_*/training_results.json | grep accuracy
```

## Expected Results

After running the full pipeline, you should have:

| Metric | Expected Value |
|--------|---------------|
| Dataset files | 219 |
| Benchmark results | 657 (219 × 3) |
| Measurements | 65,700 (657 × 100) |
| CV Accuracy | 71-73% ± 9-10% |
| Test Accuracy | 72-75% |
| Energy Savings | 10-11% |
| Zero measurements | 0% |

## Troubleshooting

**Problem**: Benchmark fails with KeyError
- **Solution**: Check that `energy_joules` and `duration_seconds` keys exist

**Problem**: Model accuracy < 70%
- **Solution**: Verify all 657 benchmark results exist, check for zero measurements

**Problem**: Dataset not found
- **Solution**: Run `python scripts/organize_all_datasets.py`

**Problem**: Pipeline takes too long
- **Expected**: ~30 minutes for complete run (benchmark is 25-30 min, training is <1 min)

## Next Steps After Pipeline

1. **Review Results**:
   ```bash
   cat results/PIPELINE_SUMMARY_*.md
   cat results/models_*/training_results.json
   ```

2. **Generate Paper Figures**:
   ```bash
   python scripts/generate_figures.py  # (when implemented)
   ```

3. **Update Documentation**:
   - Copy metrics from training_results.json
   - Update DATASET_EXPANSION_RESULTS.md
   - Prepare paper submission materials

## Paper-Ready Outputs

After verification passes, you have everything needed for publication:
- ✓ 219 files from established benchmarks (Canterbury, Calgary, Silesia)
- ✓ 65,700 energy measurements
- ✓ Trained ML model (>70% accuracy)
- ✓ Proper citations for all public datasets
- ✓ Reproducible pipeline scripts

---

**Last Updated**: January 19, 2026
**Version**: 1.0
