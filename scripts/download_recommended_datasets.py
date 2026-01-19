#!/usr/bin/env python3
"""
Download recommended datasets for cryptography benchmarking
All datasets are publicly available and citable
"""

import urllib.request
import tarfile
import gzip
import shutil
from pathlib import Path

output_dir = Path("data/public_datasets")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DOWNLOADING RECOMMENDED PUBLIC DATASETS")
print("=" * 70)

# Dataset 1: Canterbury Corpus (citable benchmark)
print("\n[1/5] Canterbury Corpus")
print("Citation: R. Arnold and T. Bell, 'A corpus for the evaluation")
print("         of lossless compression algorithms', 1997")

try:
    url = "https://corpus.canterbury.ac.nz/resources/cantrbry.tar.gz"
    tar_file = output_dir / "canterbury.tar.gz"
    
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, tar_file)
    
    print("Extracting...")
    with tarfile.open(tar_file, 'r:gz') as tar:
        tar.extractall(output_dir / "canterbury")
    tar_file.unlink()
    
    files = list((output_dir / "canterbury").rglob("*"))
    print(f"✅ {len([f for f in files if f.is_file()])} files")
except Exception as e:
    print(f"❌ Failed: {e}")

# Dataset 2: Calgary Corpus
print("\n[2/5] Calgary Corpus")
print("Citation: T. Bell, I. Witten, J. Cleary, 'Modeling for text")
print("         compression', ACM Computing Surveys, 1989")

try:
    url = "https://corpus.canterbury.ac.nz/resources/calgary.tar.gz"
    tar_file = output_dir / "calgary.tar.gz"
    
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, tar_file)
    
    print("Extracting...")
    with tarfile.open(tar_file, 'r:gz') as tar:
        tar.extractall(output_dir / "calgary")
    tar_file.unlink()
    
    files = list((output_dir / "calgary").rglob("*"))
    print(f"✅ {len([f for f in files if f.is_file()])} files")
except Exception as e:
    print(f"❌ Failed: {e}")

# Dataset 3: Sample from Silesia (large files)
print("\n[3/5] Silesia Corpus Sample")
print("Citation: S. Deorowicz, 'Silesia Compression Corpus', 2003")

try:
    # Download just a few files from Silesia
    silesia_base = "http://sun.aei.polsl.pl/~sdeor/corpus/"
    silesia_dir = output_dir / "silesia"
    silesia_dir.mkdir(exist_ok=True)
    
    files_to_download = [
        "dickens",  # 10MB text
        "mozilla",  # 51MB executable  
        "mr",       # 10MB medical image
        "osdb",     # 10MB database
        "samba",    # 22MB source code
        "webster",  # 41MB dictionary
    ]
    
    for filename in files_to_download:
        try:
            url = f"{silesia_base}{filename}"
            dest = silesia_dir / filename
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, dest)
            size = dest.stat().st_size
            print(f"  ✅ {filename} ({size:,} bytes)")
        except:
            print(f"  ⚠️  {filename} unavailable")
            
except Exception as e:
    print(f"❌ Failed: {e}")

# Dataset 4: Project Gutenberg texts
print("\n[4/5] Project Gutenberg Texts")
print("Citation: Project Gutenberg, www.gutenberg.org")

gutenberg_dir = output_dir / "gutenberg"
gutenberg_dir.mkdir(exist_ok=True)

texts = [
    (1342, "pride_and_prejudice.txt"),
    (84, "frankenstein.txt"),
    (2701, "moby_dick.txt"),
    (1661, "sherlock_holmes.txt"),
    (11, "alice_wonderland.txt"),
]

for book_id, filename in texts:
    try:
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        dest = gutenberg_dir / filename
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, dest)
        size = dest.stat().st_size
        print(f"  ✅ {filename} ({size:,} bytes)")
    except Exception as e:
        print(f"  ⚠️  {filename} failed")

# Dataset 5: Standard test images
print("\n[5/5] Standard Test Images")
print("Citation: USC-SIPI Image Database")

images_dir = output_dir / "images"
images_dir.mkdir(exist_ok=True)

# Using alternative public image sources
image_urls = [
    ("https://homepages.cae.wisc.edu/~ece533/images/airplane.png", "airplane.png"),
    ("https://homepages.cae.wisc.edu/~ece533/images/baboon.png", "baboon.png"),
    ("https://homepages.cae.wisc.edu/~ece533/images/peppers.png", "peppers.png"),
]

for url, filename in image_urls:
    try:
        dest = images_dir / filename
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, dest)
        size = dest.stat().st_size
        print(f"  ✅ {filename} ({size:,} bytes)")
    except:
        print(f"  ⚠️  {filename} failed")

# Summary
print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

total_files = 0
total_size = 0

for subdir in output_dir.rglob("*"):
    if subdir.is_file():
        total_files += 1
        total_size += subdir.stat().st_size

print(f"Total files: {total_files}")
print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
print(f"Location: {output_dir}")
print("\nYou can now cite these established benchmarks in your paper!")
print("=" * 70)
