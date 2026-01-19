#!/bin/bash
################################################################################
# GITHUB CLEANUP - Prepare for publication
################################################################################

set -e

echo "========================================================================"
echo "GITHUB REPOSITORY CLEANUP"
echo "========================================================================"
echo ""
echo "This will REMOVE:"
echo "  ❌ Old benchmark results (benchmarks/, benchmarks_fixed/)"
echo "  ❌ Temporary scripts (test_*.py, debug_*.py, run_benchmark_*.py)"
echo "  ❌ Old model versions (selector_model_fixed.pkl, etc.)"
echo "  ❌ Extracted dataset directories (canterbury_extracted/, etc.)"
echo "  ❌ All log files and temporary files"
echo "  ❌ Python cache (__pycache__/, *.pyc)"
echo "  ❌ Extra documentation files"
echo "  ❌ Old/redundant scripts in scripts/"
echo ""
echo "This will KEEP (for paper publication):"
echo "  ✓ Source code (cryptogreen/)"
echo "  ✓ Essential scripts (organize, train, generate figures)"
echo "  ✓ Latest benchmark results (complete_benchmark_20260119_193623/)"
echo "  ✓ Trained model (models_20260119_194917/)"
echo "  ✓ Generated figures (figures/)"
echo "  ✓ Public datasets (complete_benchmark_dataset/)"
echo "  ✓ Key documentation (DATASET_EXPANSION_RESULTS.md, PIPELINE_GUIDE.md)"
echo "  ✓ requirements.txt, setup.py, README.md"
echo ""

read -p "Continue with cleanup? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Cleaning up..."

################################################################################
# 1. Remove old benchmark results
################################################################################
echo "  [1/10] Removing old benchmark results..."
rm -rf results/benchmarks/ 2>/dev/null || true
rm -rf results/benchmarks_fixed/ 2>/dev/null || true
rm -rf results/benchmarks_test/ 2>/dev/null || true
rm -rf results/diagnostic_report.json 2>/dev/null || true
rm -rf results/energy_savings_report.json 2>/dev/null || true

################################################################################
# 2. Remove extracted dataset directories
################################################################################
echo "  [2/10] Removing extracted dataset directories..."
rm -rf data/public_datasets/canterbury_extracted/ 2>/dev/null || true
rm -rf data/public_datasets/calgary_extracted/ 2>/dev/null || true
rm -rf data/test_files/ 2>/dev/null || true
rm -rf data/test_files_expanded/ 2>/dev/null || true
rm -rf data/real_world_files/ 2>/dev/null || true
rm -rf data/ml_data/ 2>/dev/null || true

################################################################################
# 3. Remove old model files
################################################################################
echo "  [3/10] Removing old model versions..."
rm -rf results/models/ 2>/dev/null || true

################################################################################
# 4. Remove temporary scripts
################################################################################
echo "  [4/10] Removing temporary scripts..."
rm -f test_rapl.py 2>/dev/null || true
rm -f run_benchmark_fixed.py 2>/dev/null || true
rm -f run_complete_benchmark.py 2>/dev/null || true
rm -f debug_benchmark.py 2>/dev/null || true
rm -f run_benchmark_background.sh 2>/dev/null || true
rm -f cleanup_project.sh 2>/dev/null || true
rm -f benchmark_output.txt 2>/dev/null || true

################################################################################
# 5. Remove redundant scripts
################################################################################
echo "  [5/10] Cleaning up scripts directory..."
cd scripts/
rm -f analyze_results.py 2>/dev/null || true
rm -f calculate_savings.py 2>/dev/null || true
rm -f check_system.py 2>/dev/null || true
rm -f diagnose_savings.py 2>/dev/null || true
rm -f evaluate_accuracy.py 2>/dev/null || true
rm -f evaluate_selector.py 2>/dev/null || true
rm -f process_results.py 2>/dev/null || true
rm -f real_world_evaluation.py 2>/dev/null || true
rm -f statistical_tests.py 2>/dev/null || true
rm -f validate_benchmark.py 2>/dev/null || true
rm -f train_model_fixed.py 2>/dev/null || true
cd ..

################################################################################
# 6. Remove all log files
################################################################################
echo "  [6/10] Removing log files..."
rm -f *.log 2>/dev/null || true
rm -f benchmark_progress.log 2>/dev/null || true
rm -f pipeline_*.log 2>/dev/null || true
rm -rf results/logs/ 2>/dev/null || true
rm -rf results/benchmarks/logs/ 2>/dev/null || true

################################################################################
# 7. Remove Python cache
################################################################################
echo "  [7/10] Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

################################################################################
# 8. Remove redundant documentation
################################################################################
echo "  [8/10] Cleaning up documentation..."
rm -f COPILOT_IMPLEMENTATION_SPEC.md 2>/dev/null || true
rm -f FINAL_SUMMARY.md 2>/dev/null || true
rm -f cleanup_and_rebuild.sh 2>/dev/null || true

################################################################################
# 9. Remove processed results
################################################################################
echo "  [9/10] Removing processed results..."
rm -rf results/processed/ 2>/dev/null || true

################################################################################
# 10. Remove venv and build artifacts
################################################################################
echo "  [10/10] Removing build artifacts..."
rm -rf venv/ 2>/dev/null || true
rm -rf build/ 2>/dev/null || true
rm -rf dist/ 2>/dev/null || true
rm -rf .eggs/ 2>/dev/null || true

################################################################################
# Create .gitignore
################################################################################
echo ""
echo "Creating .gitignore..."
cat > .gitignore << 'GITIGNORE'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
*.log

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp

# Jupyter
.ipynb_checkpoints/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Large data files (optional - uncomment if too large for GitHub)
# data/public_datasets/*.tar.gz
# data/complete_benchmark_dataset/silesia_*
GITIGNORE

echo "✓ .gitignore created"

################################################################################
# Summary
################################################################################
echo ""
echo "========================================================================"
echo "CLEANUP COMPLETE - GITHUB READY!"
echo "========================================================================"
echo ""
echo "Repository structure:"
echo ""
echo "cryptogreen/"
echo "├── cryptogreen/              (source code)"
echo "│   ├── __init__.py"
echo "│   ├── algorithms.py"
echo "│   ├── energy_meter.py"
echo "│   ├── feature_extractor.py"
echo "│   ├── ml_selector.py"
echo "│   └── ..."
echo "├── scripts/"
echo "│   ├── organize_all_datasets.py"
echo "│   ├── train_model_complete.py"
echo "│   ├── download_all_datasets.py"
echo "│   └── generate_test_data.py"
echo "├── data/"
echo "│   ├── complete_benchmark_dataset/  (219 files)"
echo "│   └── public_datasets/             (source archives)"
echo "├── results/"
echo "│   ├── complete_benchmark_20260119_193623/"
echo "│   │   └── raw/                     (657 results)"
echo "│   ├── models_20260119_194917/"
echo "│   │   ├── selector_model.pkl"
echo "│   │   └── training_results.json"
echo "│   └── figures/"
echo "│       ├── energy_vs_size.png/pdf"
echo "│       ├── algorithm_distribution.png/pdf"
echo "│       └── ..."
echo "├── generate_all_figures.py"
echo "├── run_full_pipeline.sh"
echo "├── verify_pipeline.sh"
echo "├── DATASET_EXPANSION_RESULTS.md"
echo "├── PIPELINE_GUIDE.md"
echo "├── README.md"
echo "├── requirements.txt"
echo "├── setup.py"
echo "└── .gitignore"
echo ""
echo "Repository size:"
du -sh . 2>/dev/null | head -1
echo ""
echo "Files by directory:"
echo "  cryptogreen/: $(find cryptogreen/ -type f -name "*.py" 2>/dev/null | wc -l) Python files"
echo "  scripts/: $(find scripts/ -type f -name "*.py" 2>/dev/null | wc -l) Python files"
echo "  data/: $(find data/complete_benchmark_dataset/ -type f 2>/dev/null | wc -l) dataset files"
echo "  results/: $(find results/ -type f 2>/dev/null | wc -l) result files"
echo ""
echo "Next steps:"
echo "  1. git init"
echo "  2. git add ."
echo "  3. git commit -m 'Initial commit: CryptoGreen paper submission'"
echo "  4. Create GitHub repository"
echo "  5. git remote add origin <url>"
echo "  6. git push -u origin main"
echo ""
echo "========================================================================"
