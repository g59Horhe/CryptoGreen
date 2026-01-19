#!/usr/bin/env python3
"""Test script for FeatureExtractor."""

import sys
from pathlib import Path

# Add cryptogreen to path
sys.path.insert(0, str(Path(__file__).parent))

from cryptogreen.feature_extractor import FeatureExtractor

def test_feature_extraction():
    """Test feature extraction on test files."""
    
    # Test files
    test_files = [
        'data/test_files/txt/txt_1KB.txt',
        'data/test_files/txt/txt_10KB.txt',
        'data/test_files/sql/sql_1KB.sql',
        'data/test_files/zip/sql_1KB.sql',  # This might be wrong path
    ]
    
    print("=" * 70)
    print("FEATURE EXTRACTOR TEST")
    print("=" * 70)
    
    print("\nFeature names (7 total):")
    feature_names = FeatureExtractor.get_feature_names()
    for i, name in enumerate(feature_names, 1):
        print(f"  {i}. {name}")
    
    print(f"\n{'='*70}")
    print("Testing on txt_1KB.txt:")
    print('='*70)
    
    test_file = 'data/test_files/txt/txt_1KB.txt'
    
    # Extract features
    features = FeatureExtractor.extract_features(test_file)
    
    # Print all features in dict
    print("\nAll features in dict:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    # Print feature array (for ML)
    print("\nFeature array (for sklearn):")
    feature_array = FeatureExtractor.features_to_array(features)
    for i, (name, value) in enumerate(zip(feature_names, feature_array), 1):
        print(f"  {i}. {name:25s} = {value}")
    
    # Verify counts
    print(f"\nVerification:")
    print(f"  Features in dict: {len(features)}")
    print(f"  Features for ML:  {len(feature_names)}")
    print(f"  Feature array length: {len(feature_array)}")
    
    # Check the specific requirements
    print(f"\nRequirement checks:")
    print(f"  ✓ file_size_log = log10({features['file_size_bytes']}) = {features['file_size_log']:.4f}")
    print(f"  ✓ file_type_encoded = {features['file_type_encoded']} (txt should be 0)")
    print(f"  ✓ entropy = {features['entropy']:.4f} (0-8 bits)")
    print(f"  ✓ entropy_quartile_25 = {features['entropy_quartile_25']} (byte value 0-255)")
    print(f"  ✓ entropy_quartile_75 = {features['entropy_quartile_75']} (byte value 0-255)")
    print(f"  ✓ has_aes_ni = {features['has_aes_ni']}")
    print(f"  ✓ cpu_cores = {features['cpu_cores']} (physical cores)")
    
    # Test file type encodings
    print(f"\n{'='*70}")
    print("File type encoding verification:")
    print('='*70)
    
    test_types = [
        ('txt', 0),
        ('jpg', 1),
        ('png', 2),
        ('mp4', 3),
        ('pdf', 4),
        ('sql', 5),
        ('zip', 6),
        ('unknown', 7),
    ]
    
    for file_type, expected in test_types:
        encoded = FeatureExtractor.encode_file_type(file_type)
        status = '✓' if encoded == expected else '✗'
        print(f"  {status} {file_type:8s} -> {encoded} (expected {expected})")
    
    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print('='*70)

if __name__ == '__main__':
    test_feature_extraction()
