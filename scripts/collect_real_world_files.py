#!/usr/bin/env python3
"""Collect real-world files for validation"""

import shutil
from pathlib import Path
import hashlib
import os

output_dir = Path("data/real_world_files")
output_dir.mkdir(parents=True, exist_ok=True)

# Define collection strategy
collections = {
    'documents': {
        'extensions': ['.pdf', '.docx', '.txt', '.md', '.doc'],
        'sources': [
            Path.home() / 'Documents',
            Path.home() / 'Downloads',
            Path.home() / 'Desktop',
        ],
        'target': 20
    },
    'images': {
        'extensions': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
        'sources': [
            Path.home() / 'Pictures',
            Path.home() / 'Downloads',
        ],
        'target': 20
    },
    'code': {
        'extensions': ['.py', '.js', '.cpp', '.java', '.html', '.css', '.c', '.h'],
        'sources': [
            Path.home() / 'projects',
            Path.home() / 'Desktop',
            Path('/usr/share/doc'),
        ],
        'target': 20
    },
    'data': {
        'extensions': ['.csv', '.json', '.xml', '.sql', '.db'],
        'sources': [
            Path.home() / 'Documents',
            Path.home() / 'Downloads',
        ],
        'target': 15
    },
    'compressed': {
        'extensions': ['.zip', '.tar', '.gz', '.7z', '.rar'],
        'sources': [
            Path.home() / 'Downloads',
            Path.home() / 'Documents',
        ],
        'target': 10
    },
    'media': {
        'extensions': ['.mp4', '.mp3', '.avi', '.mkv', '.mov', '.wav'],
        'sources': [
            Path.home() / 'Videos',
            Path.home() / 'Music',
            Path.home() / 'Downloads',
        ],
        'target': 10
    }
}

collected = {}
total = 0

for category, config in collections.items():
    collected[category] = []
    
    for source in config['sources']:
        if not source.exists():
            continue
        
        # Find files with matching extensions
        for ext in config['extensions']:
            try:
                # Limit depth to avoid traversing entire filesystem
                files = []
                for root, dirs, filenames in os.walk(source):
                    # Skip hidden directories
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    
                    # Limit depth
                    depth = root.replace(str(source), '').count(os.sep)
                    if depth > 3:
                        continue
                    
                    for filename in filenames:
                        if filename.lower().endswith(ext):
                            files.append(Path(root) / filename)
                
                for file in files:
                    if len(collected[category]) >= config['target']:
                        break
                    
                    # Skip if too large (>100MB) or too small (<100 bytes)
                    if not file.is_file():
                        continue
                    
                    try:
                        size = file.stat().st_size
                    except:
                        continue
                        
                    if size < 100 or size > 100_000_000:
                        continue
                    
                    # Copy to output directory with safe name
                    file_hash = hashlib.md5(str(file).encode()).hexdigest()[:8]
                    new_name = f"{category}_{file.stem}_{file_hash}{file.suffix}"
                    dest = output_dir / new_name
                    
                    try:
                        shutil.copy2(file, dest)
                        collected[category].append({
                            'original': str(file),
                            'copy': str(dest),
                            'size': size
                        })
                        total += 1
                        print(f"✓ {new_name:50s} ({size:>12,} bytes)")
                    except Exception as e:
                        pass
                
                if len(collected[category]) >= config['target']:
                    break
                    
            except Exception as e:
                continue

# Summary
print("\n" + "=" * 70)
print("COLLECTION SUMMARY")
print("=" * 70)
for category, files in collected.items():
    print(f"{category:15s}: {len(files):3d} files")
print(f"{'TOTAL':15s}: {total:3d} files")
print("=" * 70)

if total < 30:
    print("\n⚠️  Warning: Less than 30 files collected")
    print("Consider adjusting source directories or file size limits")
elif total >= 50:
    print("\n✅ Excellent! 50+ real-world files collected")
else:
    print(f"\n✅ Good! {total} real-world files collected")

print(f"\nFiles saved to: {output_dir}")
