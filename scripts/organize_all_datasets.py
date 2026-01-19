#!/usr/bin/env python3
"""Extract and organize all datasets into unified structure"""

import tarfile
import shutil
from pathlib import Path
import os

print("=" * 70)
print("ORGANIZING ALL DATASETS")
print("=" * 70)

# Create organized output directory
output_dir = Path("data/complete_benchmark_dataset")
output_dir.mkdir(parents=True, exist_ok=True)

datasets_collected = {}

# 1. Extract Canterbury Corpus
print("\n[1/6] Extracting Canterbury Corpus...")
canterbury_tar = Path("data/public_datasets/canterbury.tar.gz")
if canterbury_tar.exists():
    with tarfile.open(canterbury_tar, 'r:gz') as tar:
        tar.extractall("data/public_datasets/canterbury_extracted")
    
    # Copy files
    source_dir = Path("data/public_datasets/canterbury_extracted")
    for file in source_dir.rglob("*"):
        if file.is_file():
            dest = output_dir / f"canterbury_{file.name}"
            shutil.copy2(file, dest)
            print(f"  ✓ {dest.name} ({file.stat().st_size:,} bytes)")
    
    datasets_collected['canterbury'] = len(list(source_dir.rglob("*.[!d]*")))

# 2. Extract Calgary Corpus
print("\n[2/6] Extracting Calgary Corpus...")
calgary_tar = Path("data/public_datasets/calgary.tar.gz")
if calgary_tar.exists():
    with tarfile.open(calgary_tar, 'r:gz') as tar:
        tar.extractall("data/public_datasets/calgary_extracted")
    
    # Copy files
    source_dir = Path("data/public_datasets/calgary_extracted")
    for file in source_dir.rglob("*"):
        if file.is_file() and not file.name.startswith('.'):
            dest = output_dir / f"calgary_{file.name}"
            shutil.copy2(file, dest)
            print(f"  ✓ {dest.name} ({file.stat().st_size:,} bytes)")
    
    datasets_collected['calgary'] = len(list((output_dir).glob("calgary_*")))

# 3. Copy Silesia files
print("\n[3/6] Copying Silesia Corpus samples...")
silesia_files = [
    "silesia_dickens", "silesia_mozilla", "silesia_mr",
    "silesia_osdb", "silesia_samba", "silesia_webster"
]

silesia_count = 0
for filename in silesia_files:
    source = Path(f"data/public_datasets/{filename}")
    if source.exists():
        dest = output_dir / filename
        shutil.copy2(source, dest)
        print(f"  ✓ {dest.name} ({source.stat().st_size:,} bytes)")
        silesia_count += 1

datasets_collected['silesia'] = silesia_count

# 4. Copy Gutenberg texts
print("\n[4/6] Copying Project Gutenberg texts...")
gutenberg_dir = Path("data/public_datasets/gutenberg")
gutenberg_count = 0

if gutenberg_dir.exists():
    for file in gutenberg_dir.glob("*.txt"):
        dest = output_dir / f"gutenberg_{file.name}"
        shutil.copy2(file, dest)
        print(f"  ✓ {dest.name} ({file.stat().st_size:,} bytes)")
        gutenberg_count += 1

datasets_collected['gutenberg'] = gutenberg_count

# 5. Copy local real-world files
print("\n[5/6] Copying local real-world files...")
real_world_dir = Path("data/real_world_files")
real_world_count = 0

if real_world_dir.exists():
    for file in real_world_dir.rglob("*"):
        if file.is_file():
            dest = output_dir / f"realworld_{file.name}"
            shutil.copy2(file, dest)
            real_world_count += 1
    print(f"  ✓ {real_world_count} files copied")

datasets_collected['real_world'] = real_world_count

# 6. Copy expanded synthetic files
print("\n[6/6] Copying expanded synthetic test files...")
synthetic_dir = Path("data/test_files_expanded")
synthetic_count = 0

if synthetic_dir.exists():
    for file in synthetic_dir.rglob("*"):
        if file.is_file():
            # Keep original structure in filename
            relative_path = file.relative_to(synthetic_dir)
            dest = output_dir / f"synthetic_{relative_path.parent.name}_{file.name}"
            shutil.copy2(file, dest)
            synthetic_count += 1
    print(f"  ✓ {synthetic_count} files copied")

datasets_collected['synthetic'] = synthetic_count

# Summary
print("\n" + "=" * 70)
print("DATASET ORGANIZATION COMPLETE")
print("=" * 70)

total_files = sum(datasets_collected.values())

print(f"\nFiles by source:")
for source, count in datasets_collected.items():
    print(f"  {source:15s}: {count:3d} files")
print(f"  {'TOTAL':15s}: {total_files:3d} files")

# Calculate total size
total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
print(f"\nTotal dataset size: {total_size / 1024 / 1024:.1f} MB")
print(f"Output directory: {output_dir}")

# Show size distribution
print("\nSize distribution:")
size_ranges = {
    '<1KB': 0, '1KB-10KB': 0, '10KB-100KB': 0,
    '100KB-1MB': 0, '1MB-10MB': 0, '>10MB': 0
}

for file in output_dir.rglob("*"):
    if file.is_file():
        size = file.stat().st_size
        if size < 1024:
            size_ranges['<1KB'] += 1
        elif size < 10240:
            size_ranges['1KB-10KB'] += 1
        elif size < 102400:
            size_ranges['10KB-100KB'] += 1
        elif size < 1048576:
            size_ranges['100KB-1MB'] += 1
        elif size < 10485760:
            size_ranges['1MB-10MB'] += 1
        else:
            size_ranges['>10MB'] += 1

for range_name, count in size_ranges.items():
    print(f"  {range_name:12s}: {count:3d} files")

print("\n" + "=" * 70)
print("Ready for benchmarking!")
print(f"Total measurements: {total_files} files × 3 algorithms × 100 runs = {total_files * 3 * 100:,}")
print("=" * 70)
