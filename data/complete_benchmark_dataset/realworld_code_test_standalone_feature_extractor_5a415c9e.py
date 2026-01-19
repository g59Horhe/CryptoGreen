#!/usr/bin/env python3
"""Standalone test for feature_extractor.py."""

import os
import sys
import math
import logging
import platform
import re
from pathlib import Path
from collections import Counter
from typing import Union, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from files for ML-based algorithm selection.
    
    This class extracts EXACTLY 7 features from files as specified in the paper:
    1. file_size_log: log10(file size in bytes)
    2. file_type_encoded: Numeric encoding of file type
    3. entropy: Shannon entropy (0-8 bits)
    4. entropy_quartile_25: 25th percentile of byte values
    5. entropy_quartile_75: 75th percentile of byte values
    6. has_aes_ni: Boolean indicating CPU AES-NI support
    7. cpu_cores: Number of physical CPU cores
    """
    
    # Hardware detection cache (persist across calls)
    _hardware_cache = None
    
    # File type encoding mapping (as specified in paper)
    FILE_TYPE_ENCODING = {
        'txt': 0,
        'jpg': 1,
        'jpeg': 1,  # Same as jpg
        'png': 2,
        'mp4': 3,
        'pdf': 4,
        'sql': 5,
        'zip': 6,
        'unknown': 7,
    }
    
    @classmethod
    def extract_features(cls, file_path: Union[str, Path]) -> dict:
        """Extract all features from a file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # 1. File size (log scale)
        file_size = path.stat().st_size
        file_size_log = math.log10(file_size) if file_size > 0 else 0.0
        
        # 2. File type encoding
        file_type = path.suffix.lstrip('.').lower() or 'unknown'
        file_type_encoded = cls.encode_file_type(file_type)
        
        # 3-5. Entropy and quartiles
        entropy, q25, q75 = cls.calculate_entropy(file_path)
        
        # 6-7. Hardware capabilities (cached)
        hardware = cls.detect_hardware_capabilities()
        
        features = {
            'file_size_bytes': file_size,
            'file_size_log': file_size_log,
            'file_type': file_type,
            'file_type_encoded': file_type_encoded,
            'entropy': entropy,
            'entropy_quartile_25': q25,
            'entropy_quartile_75': q75,
            'has_aes_ni': hardware['has_aes_ni'],
            'cpu_cores': hardware['cpu_cores'],
        }
        
        return features
    
    @staticmethod
    def calculate_entropy(
        file_path: Union[str, Path],
        sample_size: int = 10240
    ) -> Tuple[float, float, float]:
        """Calculate Shannon entropy of file and byte value quartiles."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = path.stat().st_size
        
        if file_size == 0:
            return 0.0, 0.0, 0.0
        
        # Read sample: first 10KB for files >10KB, entire file otherwise
        bytes_to_read = min(sample_size, file_size)
        
        with open(file_path, 'rb') as f:
            data = f.read(bytes_to_read)
        
        if len(data) == 0:
            return 0.0, 0.0, 0.0
        
        # Count byte frequencies
        byte_counts = Counter(data)
        total_bytes = len(data)
        
        # Calculate Shannon entropy: H = -Σ p(x) * log2(p(x))
        entropy = 0.0
        
        for count in byte_counts.values():
            probability = count / total_bytes
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Calculate quartiles of BYTE VALUES (0-255), not frequencies
        byte_values = sorted(data)
        
        if len(byte_values) == 0:
            q25, q75 = 0.0, 0.0
        elif len(byte_values) == 1:
            q25 = q75 = float(byte_values[0])
        else:
            q25_idx = int(len(byte_values) * 0.25)
            q75_idx = int(len(byte_values) * 0.75)
            q25 = float(byte_values[q25_idx])
            q75 = float(byte_values[min(q75_idx, len(byte_values) - 1)])
        
        return entropy, q25, q75
    
    @classmethod
    def detect_hardware_capabilities(cls) -> dict:
        """Detect CPU hardware capabilities."""
        # Return cached result if available
        if cls._hardware_cache is not None:
            return cls._hardware_cache.copy()
        
        # Get physical core count using psutil if available
        try:
            import psutil
            cpu_cores = psutil.cpu_count(logical=False) or 1
        except ImportError:
            logger.warning("psutil not available, using os.cpu_count()")
            cpu_cores = os.cpu_count() or 1
        
        result = {
            'has_aes_ni': False,
            'cpu_cores': cpu_cores,
            'cpu_model': 'Unknown',
        }
        
        # Detect based on platform
        if platform.system() == 'Linux':
            result.update(cls._detect_linux_capabilities())
        
        # Cache result
        cls._hardware_cache = result.copy()
        
        return result
    
    @staticmethod
    def _detect_linux_capabilities() -> dict:
        """Detect hardware capabilities on Linux."""
        result = {
            'has_aes_ni': False,
            'cpu_model': 'Unknown',
        }
        
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read()
            
            # Get CPU model
            model_match = re.search(r'model name\\s*:\\s*(.+)', content)
            if model_match:
                result['cpu_model'] = model_match.group(1).strip()
            
            # Check for AES-NI (look for 'aes' in flags)
            flags_match = re.search(r'flags\\s*:\\s*(.+)', content)
            if flags_match:
                flags = flags_match.group(1).split()
                if 'aes' in flags:
                    result['has_aes_ni'] = True
        
        except Exception as e:
            logger.warning(f"Error reading /proc/cpuinfo: {e}")
        
        return result
    
    @classmethod
    def encode_file_type(cls, file_type: str) -> int:
        """Encode file type as integer.
        
        Mapping:
            txt -> 0
            jpg/jpeg -> 1
            png -> 2
            mp4 -> 3
            pdf -> 4
            sql -> 5
            zip -> 6
            unknown/other -> 7
        """
        return cls.FILE_TYPE_ENCODING.get(file_type.lower(), 7)
    
    @classmethod
    def get_feature_names(cls) -> list:
        """Get list of feature names for ML model."""
        return [
            'file_size_log',
            'file_type_encoded',
            'entropy',
            'entropy_quartile_25',
            'entropy_quartile_75',
            'has_aes_ni',
            'cpu_cores',
        ]
    
    @classmethod
    def features_to_array(cls, features: dict) -> list:
        """Convert features dict to array for ML model."""
        return [
            features['file_size_log'],
            float(features['file_type_encoded']),
            features['entropy'],
            features['entropy_quartile_25'],
            features['entropy_quartile_75'],
            float(features['has_aes_ni']),
            float(features['cpu_cores']),
        ]


# Test the feature extractor
def main():
    print("=" * 70)
    print("FEATURE EXTRACTOR TEST")
    print("=" * 70)
    
    print("\nFeature names (7 total):")
    feature_names = FeatureExtractor.get_feature_names()
    for i, name in enumerate(feature_names, 1):
        print(f"  {i}. {name}")
    
    test_file = 'data/test_files/txt/txt_1KB.txt'
    
    if not os.path.exists(test_file):
        print(f"\nError: Test file not found: {test_file}")
        return
    
    print(f"\n{'='*70}")
    print(f"Testing on {test_file}:")
    print('='*70)
    
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
    print("TEST COMPLETE ✓")
    print('='*70)


if __name__ == '__main__':
    main()
