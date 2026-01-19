# CryptoGreen Dataset

This directory contains the datasets used in the CryptoGreen paper.

## Dataset Structure

```
data/
├── complete_benchmark_dataset/   (219 files, 1.4 GB) - NOT IN REPO
│   ├── canterbury_*             (11 files from Canterbury Corpus)
│   ├── calgary_*                (18 files from Calgary Corpus)
│   ├── silesia_*                (6 files from Silesia Corpus)
│   ├── gutenberg_*              (5 files from Project Gutenberg)
│   ├── realworld_*              (32 real-world files)
│   └── synthetic_*              (147 synthetic test files)
└── public_datasets/             (43 MB source archives) - NOT IN REPO
    ├── canterbury.tar.gz
    ├── calgary.tar.gz
    ├── silesia_*.bz2
    └── gutenberg_*.txt
```

## Reproducing the Dataset

The complete dataset is **NOT included in this repository** due to size (1.4 GB).

To reproduce it locally:

```bash
# Download public datasets (Canterbury, Calgary, Silesia, Gutenberg)
python3 scripts/download_all_datasets.py

# Generate synthetic test files
python3 scripts/generate_test_data.py

# Organize all files into complete_benchmark_dataset/
python3 scripts/organize_all_datasets.py
```

This will create 219 files across multiple size categories:
- **<1 KB**: 37 files
- **1-10 KB**: 48 files  
- **10-100 KB**: 39 files
- **100 KB-1 MB**: 35 files
- **1-10 MB**: 26 files
- **>10 MB**: 29 files

## Dataset Sources

### Public Benchmark Corpora

1. **Canterbury Corpus** (11 files)
   - Source: http://corpus.canterbury.ac.nz/
   - Citation: Arnold, R., & Bell, T. (1997). A corpus for the evaluation of lossless compression algorithms. In Proceedings DCC'97.

2. **Calgary Corpus** (18 files)
   - Source: http://corpus.canterbury.ac.nz/descriptions/#calgary
   - Citation: Bell, T. C., Witten, I. H., & Cleary, J. G. (1989). Modeling for text compression. ACM Computing Surveys, 21(4), 557-591.

3. **Silesia Corpus** (6 files)
   - Source: http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia
   - Citation: Deorowicz, S. (2003). Universal lossless data compression algorithms. PhD thesis, Silesian University of Technology.

4. **Project Gutenberg** (5 files)
   - alice29.txt (Alice in Wonderland)
   - mobydick.txt (Moby Dick)
   - pride_and_prejudice.txt
   - frankenstein.txt
   - sherlock_holmes.txt

### Real-World Files (32 files)
- Source code, configuration, documentation, data files from real projects

### Synthetic Test Files (147 files)
- Generated with scripts/generate_test_data.py
- 7 file types × 21 size points (64B to 100MB)
- Types: txt, sql, jpg, png, pdf, mp4, zip

## Total Dataset Size

- **Files**: 219
- **Total Size**: 1.4 GB
- **Size Range**: 64 bytes to 100 MB
- **File Types**: 20+ different formats

## Benchmark Results

The complete benchmark results (657 configurations × 100 runs = 65,700 measurements) are stored in:
- `results/complete_benchmark_20260119_193623/raw/` (9.8 MB JSON files)

These raw results are also **NOT included in the repository**. Only the trained model and generated figures are included for paper reproduction.

## For Paper Reviewers

If you need access to the complete dataset and raw benchmark results for verification:
1. Run the reproduction scripts above (~30 minutes)
2. Or contact the authors for a data archive

The trained model (`results/models_20260119_194917/selector_model.pkl`) and all figures are included in the repository for immediate validation of our results.
