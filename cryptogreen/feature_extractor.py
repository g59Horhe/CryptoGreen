"""
Feature Extractor Module

This module provides functionality to extract features from files for use
in machine learning-based algorithm selection. Features include file size,
entropy, file type, and hardware capabilities.

Example:
    >>> from cryptogreen.feature_extractor import FeatureExtractor
    >>> features = FeatureExtractor.extract_features('/path/to/file.txt')
    >>> print(f"Entropy: {features['entropy']:.2f}")
"""

import logging
import math
import os
import platform
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from files for algorithm selection.
    
    This class provides static methods to extract various features from
    files that are useful for predicting optimal encryption algorithms.
    
    Features Extracted:
        - File size (bytes and log-scaled)
        - File type (extension-based)
        - Shannon entropy
        - Entropy quartiles
        - Hardware capabilities (AES-NI, ARM crypto)
        - CPU core count
    
    Example:
        >>> features = FeatureExtractor.extract_features('document.pdf')
        >>> print(f"Size: {features['file_size_bytes']} bytes")
        >>> print(f"Entropy: {features['entropy']:.2f}")
    """
    
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
    
    # Cache for hardware capabilities (detected once per session)
    _hardware_cache: Optional[dict] = None
    
    @classmethod
    def clear_hardware_cache(cls) -> None:
        """Clear the cached hardware capabilities.
        
        Call this method to force re-detection of hardware capabilities.
        Useful after installing new packages like py-cpuinfo.
        """
        cls._hardware_cache = None
        logger.debug("Hardware capabilities cache cleared")
    
    @classmethod
    def extract_features(cls, file_path: Union[str, Path]) -> dict:
        """Extract all features from a file.
        
        Extracts EXACTLY 7 features as specified in paper:
        1. file_size_log: log10(file_size_bytes)
        2. file_type_encoded: txt=0, jpg=1, png=2, mp4=3, pdf=4, sql=5, zip=6
        3. entropy: Shannon entropy (0-8 bits)
        4. entropy_quartile_25: 25th percentile of byte values (0-255)
        5. entropy_quartile_75: 75th percentile of byte values (0-255)
        6. has_aes_ni: Boolean from /proc/cpuinfo flags
        7. cpu_cores: Physical cores only (psutil.cpu_count(logical=False))
        
        Args:
            file_path: Path to the file to analyze.
            
        Returns:
            Dict containing all extracted features.
                
        Raises:
            FileNotFoundError: If file doesn't exist.
            
        Example:
            >>> features = FeatureExtractor.extract_features('test.txt')
            >>> print(features['entropy'])
        """
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
        
        logger.debug(f"Extracted features for {file_path}: {features}")
        return features
    
    @staticmethod
    def calculate_entropy(
        file_path: Union[str, Path],
        sample_size: int = 10240
    ) -> Tuple[float, float, float]:
        """Calculate Shannon entropy of file.
        
        Shannon entropy measures the randomness of data. Higher entropy
        indicates more random/encrypted/compressed data.
        
        For files >10KB: compute on FIRST 10KB only
        For files <=10KB: compute on entire file
        
        Args:
            file_path: Path to file to analyze.
            sample_size: Maximum bytes to sample (default 10KB = 10240 bytes).
            
        Returns:
            Tuple of (entropy, quartile_25, quartile_75):
                - entropy: Shannon entropy in bits (0-8 range)
                - quartile_25: 25th percentile of BYTE VALUES (0-255)
                - quartile_75: 75th percentile of BYTE VALUES (0-255)
                
        Notes:
            - Entropy of 0: All bytes are identical
            - Entropy of 8: Maximum randomness (uniform distribution)
            - Encrypted/compressed data: typically > 7.5
            - Plain text: typically 4-6
            - Binary data: typically 6-8
            
        Example:
            >>> entropy, q25, q75 = FeatureExtractor.calculate_entropy('file.bin')
            >>> if entropy > 7.5:
            ...     print("Data appears to be encrypted or compressed")
        """
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
        # Convert data to sorted list of byte values
        byte_values = sorted(data)
        
        if len(byte_values) == 0:
            q25, q75 = 0.0, 0.0
        elif len(byte_values) == 1:
            q25 = q75 = float(byte_values[0])
        else:
            # Calculate 25th and 75th percentile indices
            q25_idx = int(len(byte_values) * 0.25)
            q75_idx = int(len(byte_values) * 0.75)
            q25 = float(byte_values[q25_idx])
            q75 = float(byte_values[min(q75_idx, len(byte_values) - 1)])
        
        return entropy, q25, q75
    
    @classmethod
    def detect_hardware_capabilities(cls) -> dict:
        """Detect CPU hardware capabilities.
        
        Detects hardware features relevant to cryptographic performance:
        - AES-NI: Intel AES instructions (also supported by AMD)
        - CPU cores: Number of PHYSICAL CPU cores (not threads)
        
        Returns:
            Dict containing:
                - has_aes_ni: bool - AES-NI support from /proc/cpuinfo
                - cpu_cores: int - Physical cores (psutil.cpu_count(logical=False))
                - cpu_model: str - CPU model name
                
        Note:
            Results are cached after first call for performance.
            
        Example:
            >>> hw = FeatureExtractor.detect_hardware_capabilities()
            >>> if hw['has_aes_ni']:
            ...     print("AES-NI available, AES will be fast")
        """
        # Return cached result if available
        if cls._hardware_cache is not None:
            return cls._hardware_cache.copy()
        
        # Get physical core count using psutil
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
        
        system = platform.system()
        
        if system == 'Linux':
            result.update(cls._detect_linux_capabilities())
        elif system == 'Windows':
            result.update(cls._detect_windows_capabilities())
        elif system == 'Darwin':  # macOS
            result.update(cls._detect_macos_capabilities())
        
        # Cache the result
        cls._hardware_cache = result
        logger.info(f"Hardware capabilities: {result}")
        
        return result.copy()
    
    @staticmethod
    def _detect_linux_capabilities() -> dict:
        """Detect hardware capabilities on Linux.
        
        Reads /proc/cpuinfo to detect CPU features.
        
        Returns:
            Dict with detected capabilities.
        """
        result = {
            'has_aes_ni': False,
            'cpu_model': 'Unknown',
        }
        
        cpuinfo_path = Path('/proc/cpuinfo')
        
        if not cpuinfo_path.exists():
            return result
        
        try:
            content = cpuinfo_path.read_text().lower()
            
            # Get CPU model
            model_match = re.search(r'model name\s*:\s*(.+)', content, re.IGNORECASE)
            if model_match:
                result['cpu_model'] = model_match.group(1).strip()
            
            # Check for AES-NI (look for 'aes' in flags)
            # The flags line looks like: flags : fpu vme de pse ... aes ...
            flags_match = re.search(r'flags\s*:\s*(.+)', content)
            if flags_match:
                flags = flags_match.group(1).split()
                if 'aes' in flags:
                    result['has_aes_ni'] = True
            
        except Exception as e:
            logger.warning(f"Error reading /proc/cpuinfo: {e}")
        
        return result
    
    @staticmethod
    def _detect_windows_capabilities() -> dict:
        """Detect hardware capabilities on Windows.
        
        Uses py-cpuinfo library to detect CPU features including AES-NI.
        The 'aes' flag in CPU features indicates AES-NI support on both
        Intel and AMD processors.
        
        Returns:
            Dict with detected capabilities.
        """
        result = {
            'has_aes_ni': False,
            'cpu_model': 'Unknown',
        }
        
        try:
            # Use cpuinfo library (py-cpuinfo package)
            # This properly detects CPU flags including 'aes' for AES-NI
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            
            result['cpu_model'] = info.get('brand_raw', 'Unknown')
            
            # Check for 'aes' flag which indicates AES-NI support
            # Works for both Intel and AMD processors
            flags = info.get('flags', [])
            if flags:
                # Normalize flags to lowercase for comparison
                flags_lower = [f.lower() for f in flags]
                if 'aes' in flags_lower:
                    result['has_aes_ni'] = True
                    logger.debug(f"AES-NI detected via cpuinfo flags")
            
            logger.debug(f"cpuinfo detected: model={result['cpu_model']}, "
                        f"flags_count={len(flags)}, has_aes_ni={result['has_aes_ni']}")
            
        except ImportError:
            logger.warning("py-cpuinfo not installed. Install with: pip install py-cpuinfo")
            # Fallback: assume modern Intel/AMD CPUs have AES-NI
            try:
                import platform
                processor = platform.processor()
                result['cpu_model'] = processor or 'Unknown'
                # Most Intel/AMD CPUs from 2010+ support AES-NI
                if any(brand in processor.upper() for brand in ['INTEL', 'AMD', 'RYZEN', 'CORE']):
                    result['has_aes_ni'] = True
                    logger.info(f"AES-NI assumed for {processor} (cpuinfo not available)")
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Error detecting Windows CPU capabilities: {e}")
            # Fallback for modern CPUs
            result['has_aes_ni'] = True
            logger.info("AES-NI assumed True due to detection error (most modern CPUs support it)")
        
        return result
    
    @staticmethod
    def _detect_macos_capabilities() -> dict:
        """Detect hardware capabilities on macOS.
        
        Uses sysctl to detect CPU features.
        
        Returns:
            Dict with detected capabilities.
        """
        result = {
            'has_aes_ni': False,
            'cpu_model': 'Unknown',
        }
        
        try:
            import subprocess
            
            # Get CPU brand
            brand_result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True
            )
            if brand_result.returncode == 0:
                result['cpu_model'] = brand_result.stdout.strip()
            
            # Check for AES-NI (Intel)
            features_result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.features'],
                capture_output=True,
                text=True
            )
            if features_result.returncode == 0:
                if 'AES' in features_result.stdout.upper():
                    result['has_aes_ni'] = True
            
            # Check for ARM crypto (Apple Silicon)
            if platform.machine() == 'arm64':
                # Apple Silicon has hardware crypto equivalent to AES-NI
                result['has_aes_ni'] = True
        except Exception as e:
            logger.warning(f"Error detecting macOS CPU capabilities: {e}")
        
        return result
    
    @classmethod
    def encode_file_type(cls, file_type: str) -> int:
        """Encode file type as integer for ML model.
        
        Args:
            file_type: File extension (lowercase, no dot).
            
        Returns:
            Integer encoding of file type.
            
        Mapping:
            txt -> 0
            jpg/jpeg -> 1
            png -> 2
            mp4 -> 3
            pdf -> 4
            sql -> 5
            zip -> 6
            unknown/other -> 7
            
        Example:
            >>> FeatureExtractor.encode_file_type('pdf')
            4
            >>> FeatureExtractor.encode_file_type('docx')
            7
        """
        file_type = file_type.lower().strip()
        return cls.FILE_TYPE_ENCODING.get(file_type, cls.FILE_TYPE_ENCODING['unknown'])
    
    @classmethod
    def decode_file_type(cls, encoded: int) -> str:
        """Decode integer back to file type name.
        
        Args:
            encoded: Integer encoding.
            
        Returns:
            File type string.
        """
        reverse_map = {v: k for k, v in cls.FILE_TYPE_ENCODING.items() if k != 'jpeg'}
        return reverse_map.get(encoded, 'unknown')
    
    @classmethod
    def get_feature_names(cls) -> list[str]:
        """Get list of feature names for ML model.
        
        Returns:
            List of feature names in consistent order.
        """
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
    def features_to_array(cls, features: dict) -> list[float]:
        """Convert features dict to array for ML model.
        
        Args:
            features: Features dict from extract_features().
            
        Returns:
            List of feature values in consistent order.
        """
        return [
            features['file_size_log'],
            float(features['file_type_encoded']),
            features['entropy'],
            features['entropy_quartile_25'],
            features['entropy_quartile_75'],
            float(features['has_aes_ni']),
            float(features['cpu_cores']),
        ]
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached hardware capabilities.
        
        Useful for testing or when hardware changes (e.g., VM migration).
        """
        cls._hardware_cache = None
        logger.info("Hardware capability cache cleared")
    
    @staticmethod
    def classify_entropy(entropy: float) -> str:
        """Classify entropy level into human-readable category.
        
        Args:
            entropy: Shannon entropy value (0-8).
            
        Returns:
            Category string: 'low', 'medium', 'high', or 'maximum'.
        """
        if entropy < 3.0:
            return 'low'  # Highly structured data (mostly repeated bytes)
        elif entropy < 5.0:
            return 'medium'  # Text, structured data
        elif entropy < 7.5:
            return 'high'  # Binary data, some compression
        else:
            return 'maximum'  # Encrypted or compressed data
    
    @staticmethod
    def classify_file_size(size_bytes: int) -> str:
        """Classify file size into category.
        
        Args:
            size_bytes: File size in bytes.
            
        Returns:
            Category string: 'tiny', 'small', 'medium', 'large', or 'huge'.
        """
        if size_bytes < 1024:  # < 1KB
            return 'tiny'
        elif size_bytes < 100 * 1024:  # < 100KB
            return 'small'
        elif size_bytes < 10 * 1024 * 1024:  # < 10MB
            return 'medium'
        elif size_bytes < 100 * 1024 * 1024:  # < 100MB
            return 'large'
        else:
            return 'huge'
