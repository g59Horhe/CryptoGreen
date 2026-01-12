"""
CryptoGreen - Intelligent Energy-Efficient Cryptographic Algorithm Selector

This package provides tools for selecting the most energy-efficient cryptographic
algorithm based on file characteristics, hardware capabilities, and security requirements.

Main Components:
    - CryptoAlgorithms: Unified interface for cryptographic operations
    - RAPLEnergyMeter: Hardware energy measurement using RAPL
    - FeatureExtractor: Extract features from files for ML model
    - RuleBasedSelector: Rule-based algorithm selection
    - MLSelector: Machine learning-based algorithm selection
    - HybridSelector: Combined intelligent selector
    - CryptoBenchmark: Benchmarking framework

Example:
    >>> from cryptogreen import HybridSelector
    >>> selector = HybridSelector()
    >>> result = selector.select_algorithm('myfile.pdf')
    >>> print(f"Recommended: {result['algorithm']}")
"""

__version__ = "1.0.0"
__author__ = "CryptoGreen Team"

from cryptogreen.algorithms import CryptoAlgorithms
from cryptogreen.energy_meter import RAPLEnergyMeter, RAPLNotAvailableError
from cryptogreen.feature_extractor import FeatureExtractor
from cryptogreen.rule_based_selector import RuleBasedSelector
from cryptogreen.ml_selector import MLSelector
from cryptogreen.hybrid_selector import HybridSelector
from cryptogreen.benchmark_framework import CryptoBenchmark
from cryptogreen.utils import (
    get_file_size,
    get_file_extension,
    format_bytes,
    format_duration,
    setup_logging,
)

__all__ = [
    # Classes
    "CryptoAlgorithms",
    "RAPLEnergyMeter",
    "RAPLNotAvailableError",
    "FeatureExtractor",
    "RuleBasedSelector",
    "MLSelector",
    "HybridSelector",
    "CryptoBenchmark",
    # Utilities
    "get_file_size",
    "get_file_extension",
    "format_bytes",
    "format_duration",
    "setup_logging",
    # Metadata
    "__version__",
    "__author__",
]
