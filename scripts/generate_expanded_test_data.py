#!/usr/bin/env python3
"""Generate expanded test dataset with more size points"""

from pathlib import Path
import os

# Expanded size matrix - more granular
SIZES = {
    '32B': 32,
    '64B': 64,
    '128B': 128,
    '256B': 256,
    '512B': 512,
    '1KB': 1_024,
    '2KB': 2_048,
    '5KB': 5_120,
    '10KB': 10_240,
    '25KB': 25_600,
    '50KB': 51_200,
    '100KB': 102_400,
    '250KB': 256_000,
    '500KB': 512_000,
    '1MB': 1_048_576,
    '2MB': 2_097_152,
    '5MB': 5_242_880,
    '10MB': 10_485_760,
    '25MB': 26_214_400,
    '50MB': 52_428_800,
    '100MB': 104_857_600,
}

FILE_TYPES = ['txt', 'jpg', 'png', 'mp4', 'pdf', 'sql', 'zip']

print(f"Expanded dataset: {len(SIZES)} sizes × {len(FILE_TYPES)} types = {len(SIZES) * len(FILE_TYPES)} configs")
print(f"With 3 algorithms × 100 runs = {len(SIZES) * len(FILE_TYPES) * 3 * 100:,} measurements\n")

output_dir = Path("data/test_files_expanded")

# Generate files
for size_name, size_bytes in SIZES.items():
    for file_type in FILE_TYPES:
        # Create directory
        type_dir = output_dir / file_type
        type_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate file
        file_path = type_dir / f"{file_type}_{size_name}.{file_type}"
        
        if file_path.exists():
            continue
        
        # Generate random data of exact size
        data = os.urandom(size_bytes)
        
        with open(file_path, 'wb') as f:
            f.write(data)
        
        print(f"✓ Created: {file_path.name} ({size_bytes:,} bytes)")

print(f"\n✅ Generated {len(SIZES) * len(FILE_TYPES)} test files")
print(f"Location: {output_dir}")
