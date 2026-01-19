# CryptoGreen Results

This directory contains the results from the CryptoGreen benchmark and model training.

## Included in Repository (for paper verification)

```
results/
├── models_20260119_194917/
│   ├── selector_model.pkl           (Trained RandomForest model)
│   └── training_results.json        (CV & test accuracy, feature importance)
└── figures/                          (Publication-ready figures)
    ├── energy_vs_size.png/pdf
    ├── algorithm_distribution.png/pdf
    ├── energy_savings.png/pdf
    ├── throughput_comparison.png/pdf
    ├── energy_efficiency.png/pdf
    └── dataset_composition.png/pdf
```

## NOT Included (too large for GitHub)

```
results/
└── complete_benchmark_20260119_193623/
    └── raw/                          (657 JSON files, 9.8 MB)
        ├── aes-128-cbc_*.json
        ├── aes-256-cbc_*.json
        └── chacha20_*.json
```

The raw benchmark results contain:
- **657 configurations** (219 files × 3 algorithms)
- **65,700 measurements** (657 × 100 runs each)
- **0% zero measurements** (validated data quality)
- **Energy range**: 0.000160 - 8.283 J

## Reproducing Results

To reproduce the complete benchmark results locally:

```bash
# Run the full pipeline (~30 minutes)
./run_full_pipeline.sh
```

This will:
1. Verify dataset (219 files)
2. Run benchmark (657 configurations × 100 runs)
3. Train ML model (RandomForest)
4. Generate all figures
5. Create summary report

## Key Results (Paper)

### Model Performance
- **Cross-validation**: 71.3% ± 9.2% accuracy
- **Test accuracy**: 72.7%
- **Feature importance**: file_size_log (89.0%), file_type (11.0%)

### Energy Savings
- **Mean**: 8.97% vs AES-256 baseline
- **Median**: 3.16%
- **Maximum**: 47.61%
- **Total energy**: 80.339 J (baseline) → 71.753 J (optimal)

### Algorithm Selection Patterns
- **<1 KB**: ChaCha20 (100%)
- **1-10 KB**: AES-256 (60%)
- **10-100 KB**: AES-256 (63%)
- **100 KB-1 MB**: Mixed
- **1-10 MB**: AES-128 (79%)
- **>10 MB**: AES-128 (100%)

## For Paper Reviewers

All essential results for paper verification are **included in the repository**:
- ✅ Trained model (selector_model.pkl)
- ✅ Training metrics (training_results.json)
- ✅ All 6 publication figures (PNG + PDF)

Raw benchmark results (9.8 MB) are excluded due to size. To verify:
1. Run `./run_full_pipeline.sh` to regenerate (~30 minutes)
2. Or contact authors for raw data archive

## Verification

Check that your reproduction matches our results:

```bash
# Verify pipeline outputs
./verify_pipeline.sh
```

Expected output:
- ✓ Dataset: 219 files
- ✓ Benchmark: 657 results, 0% zeros
- ✓ Model: >70% accuracy
- ✓ Figures: 6 figures in PNG + PDF
