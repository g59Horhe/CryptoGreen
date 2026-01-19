#!/usr/bin/env python3
"""
Download recommended benchmark datasets for CryptoGreen validation.
Shows progress and speed in terminal.
"""

import subprocess
from pathlib import Path
import sys

# Create output directory
output_dir = Path("data/public_datasets")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Downloading public benchmark datasets with progress")
print(f"Output directory: {output_dir}")
print("=" * 70)
print()

# Dataset URLs and descriptions
datasets = [
    {
        "name": "Canterbury Corpus",
        "url": "https://corpus.canterbury.ac.nz/resources/cantrbry.tar.gz",
        "file": "canterbury.tar.gz",
        "citation": "Arnold, R., & Bell, T. (1997)"
    },
    {
        "name": "Calgary Corpus",
        "url": "https://corpus.canterbury.ac.nz/resources/calgary.tar.gz",
        "file": "calgary.tar.gz",
        "citation": "Bell, T. C., et al. (1989)"
    },
    # Silesia Corpus - individual files
    {
        "name": "Silesia: dickens",
        "url": "http://sun.aei.polsl.pl/~sdeor/corpus/dickens",
        "file": "silesia_dickens",
        "citation": "Deorowicz, S. (2003)"
    },
    {
        "name": "Silesia: mozilla",
        "url": "http://sun.aei.polsl.pl/~sdeor/corpus/mozilla",
        "file": "silesia_mozilla",
        "citation": "Deorowicz, S. (2003)"
    },
    {
        "name": "Silesia: mr",
        "url": "http://sun.aei.polsl.pl/~sdeor/corpus/mr",
        "file": "silesia_mr",
        "citation": "Deorowicz, S. (2003)"
    },
    {
        "name": "Silesia: osdb",
        "url": "http://sun.aei.polsl.pl/~sdeor/corpus/osdb",
        "file": "silesia_osdb",
        "citation": "Deorowicz, S. (2003)"
    },
    {
        "name": "Silesia: samba",
        "url": "http://sun.aei.polsl.pl/~sdeor/corpus/samba",
        "file": "silesia_samba",
        "citation": "Deorowicz, S. (2003)"
    },
    {
        "name": "Silesia: webster",
        "url": "http://sun.aei.polsl.pl/~sdeor/corpus/webster",
        "file": "silesia_webster",
        "citation": "Deorowicz, S. (2003)"
    },
    # Project Gutenberg texts
    {
        "name": "Gutenberg: Pride and Prejudice",
        "url": "https://www.gutenberg.org/files/1342/1342-0.txt",
        "file": "pride_and_prejudice.txt",
        "citation": "Project Gutenberg"
    },
    {
        "name": "Gutenberg: Frankenstein",
        "url": "https://www.gutenberg.org/files/84/84-0.txt",
        "file": "frankenstein.txt",
        "citation": "Project Gutenberg"
    },
    {
        "name": "Gutenberg: Moby Dick",
        "url": "https://www.gutenberg.org/files/2701/2701-0.txt",
        "file": "moby_dick.txt",
        "citation": "Project Gutenberg"
    },
    {
        "name": "Gutenberg: Sherlock Holmes",
        "url": "https://www.gutenberg.org/files/1661/1661-0.txt",
        "file": "sherlock_holmes.txt",
        "citation": "Project Gutenberg"
    },
    {
        "name": "Gutenberg: Alice in Wonderland",
        "url": "https://www.gutenberg.org/files/11/11-0.txt",
        "file": "alice_in_wonderland.txt",
        "citation": "Project Gutenberg"
    },
    # Standard test images
    {
        "name": "Test Image: airplane",
        "url": "https://homepages.cae.wisc.edu/~ece533/images/airplane.png",
        "file": "airplane.png",
        "citation": "USC-SIPI Image Database"
    },
    {
        "name": "Test Image: baboon",
        "url": "https://homepages.cae.wisc.edu/~ece533/images/baboon.png",
        "file": "baboon.png",
        "citation": "USC-SIPI Image Database"
    },
    {
        "name": "Test Image: peppers",
        "url": "https://homepages.cae.wisc.edu/~ece533/images/peppers.png",
        "file": "peppers.png",
        "citation": "USC-SIPI Image Database"
    },
]

# Download each dataset with progress
success_count = 0
failed_count = 0

for i, dataset in enumerate(datasets, 1):
    output_path = output_dir / dataset["file"]
    
    print(f"[{i}/{len(datasets)}] {dataset['name']}")
    
    if output_path.exists():
        size = output_path.stat().st_size / 1024 / 1024  # MB
        print(f"  ✓ Already exists ({size:.2f} MB)")
        success_count += 1
        print()
        continue
    
    print(f"  URL: {dataset['url']}")
    
    try:
        # Use wget with progress bar (preferred)
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(output_path), dataset["url"]],
            check=True,
            timeout=300  # 5 minute timeout per file
        )
        
        if output_path.exists():
            size = output_path.stat().st_size / 1024 / 1024  # MB
            print(f"  ✓ Downloaded successfully ({size:.2f} MB)")
            success_count += 1
        else:
            print(f"  ✗ Download failed - file not created")
            failed_count += 1
            
    except FileNotFoundError:
        # wget not available, try curl
        print(f"  Falling back to curl...")
        try:
            result = subprocess.run(
                ["curl", "-L", "-#", "-o", str(output_path), dataset["url"]],
                check=True,
                timeout=300
            )
            if output_path.exists():
                size = output_path.stat().st_size / 1024 / 1024  # MB
                print(f"  ✓ Downloaded successfully ({size:.2f} MB)")
                success_count += 1
            else:
                print(f"  ✗ Download failed")
                failed_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed_count += 1
            if output_path.exists():
                output_path.unlink()
                
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout (>5 minutes)")
        failed_count += 1
        if output_path.exists():
            output_path.unlink()  # Remove partial download
            
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error (exit code {e.returncode})")
        failed_count += 1
        if output_path.exists():
            output_path.unlink()  # Remove partial download
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        failed_count += 1
        if output_path.exists():
            output_path.unlink()
    
    print()

print("=" * 70)
print("Download Summary:")
print(f"  ✓ Successful: {success_count}/{len(datasets)}")
print(f"  ✗ Failed: {failed_count}/{len(datasets)}")
print(f"  Files saved to: {output_dir}")
print("=" * 70)

# List all downloaded files
print("\nDownloaded files:")
for f in sorted(output_dir.glob("*")):
    size = f.stat().st_size / 1024 / 1024  # MB
    print(f"  {f.name:30s} {size:8.2f} MB")
