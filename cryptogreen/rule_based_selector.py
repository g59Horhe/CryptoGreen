"""
Rule-Based Algorithm Selector Module

This module provides a rule-based approach for selecting the optimal
cryptographic algorithm based on file characteristics and system capabilities.

Example:
    >>> from cryptogreen.rule_based_selector import RuleBasedSelector
    >>> selector = RuleBasedSelector()
    >>> result = selector.select_algorithm('document.pdf', security_level='medium')
    >>> print(f"Recommended: {result['algorithm']}")
"""

import logging
from pathlib import Path
from typing import Any, Optional

from cryptogreen.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class RuleBasedSelector:
    """Select algorithm using rule-based decision tree.
    
    This selector uses a hand-crafted decision tree based on:
    - File size and type
    - Hardware capabilities (AES-NI)
    - Security requirements
    - Power mode (battery vs plugged)
    
    The rules are designed to balance security, performance, and energy
    efficiency based on empirical knowledge and benchmarking data.
    
    Attributes:
        hardware: Hardware capabilities dict.
        benchmark_data: Optional benchmark results for performance estimates.
        
    Example:
        >>> selector = RuleBasedSelector()
        >>> result = selector.select_algorithm(
        ...     'large_video.mp4',
        ...     security_level='low',
        ...     power_mode='battery'
        ... )
        >>> print(result['rationale'])
    """
    
    def __init__(self, benchmark_data: Optional[dict] = None):
        """Initialize rule-based selector.
        
        Args:
            benchmark_data: Optional benchmark results for performance estimation.
                           If provided, estimates will be based on actual measurements.
        """
        self.hardware = FeatureExtractor.detect_hardware_capabilities()
        self.benchmark_data = benchmark_data or {}
        
        logger.info(f"RuleBasedSelector initialized: AES-NI={self.hardware['has_aes_ni']}")
    
    def select_algorithm(
        self,
        file_path: str,
        security_level: str = 'medium',
        power_mode: str = 'plugged'
    ) -> dict:
        """Select optimal algorithm using rules.
        
        The decision tree prioritizes:
        1. Security requirements (high security → AES-256)
        2. Hardware capabilities (AES-NI → AES algorithms)
        3. File characteristics (size, entropy, type)
        4. Power mode (battery → lower power algorithms)
        
        Args:
            file_path: Path to file to encrypt.
            security_level: 'low', 'medium', or 'high'.
            power_mode: 'plugged' or 'battery'.
            
        Returns:
            Dict containing:
                - algorithm: Selected algorithm name
                - confidence: 'high', 'medium', or 'low'
                - rationale: Human-readable explanation
                - estimated_energy_j: Estimated energy (if benchmark data available)
                - estimated_duration_s: Estimated time (if benchmark data available)
                - features_used: Features that influenced decision
                - decision_path: List of rules applied
                
        Example:
            >>> result = selector.select_algorithm('data.txt', security_level='high')
            >>> print(f"Algorithm: {result['algorithm']}")
            >>> print(f"Reason: {result['rationale']}")
        """
        # Extract features
        features = FeatureExtractor.extract_features(file_path)
        
        # Initialize decision tracking
        decision_path = []
        
        # Get hardware info
        has_aes_ni = self.hardware.get('has_aes_ni', False)
        
        # Extract key features
        file_size = features['file_size_bytes']
        file_type = features['file_type']
        entropy = features['entropy']
        
        # Decision tree
        algorithm = None
        confidence = 'medium'
        rationale = ""
        
        # Rule 1: High security requirement
        if security_level == 'high':
            decision_path.append("Security level is HIGH")
            
            if has_aes_ni:
                algorithm = 'AES-256'
                confidence = 'high'
                rationale = "AES-256 selected for maximum security with hardware acceleration"
                decision_path.append("AES-NI available → AES-256")
            else:
                algorithm = 'ChaCha20'
                confidence = 'high'
                rationale = "ChaCha20 selected for high security without AES hardware support"
                decision_path.append("No AES-NI → ChaCha20 (constant-time implementation)")
        
        # Rule 2: Small files (< 100KB)
        elif file_size < 100 * 1024:
            decision_path.append(f"Small file ({file_size} bytes < 100KB)")
            
            if has_aes_ni:
                algorithm = 'AES-128'
                confidence = 'high'
                rationale = "AES-128 for small files with hardware acceleration (minimal overhead)"
                decision_path.append("AES-NI available → AES-128 (fastest for small data)")
            else:
                algorithm = 'ChaCha20'
                confidence = 'high'
                rationale = "ChaCha20 for small files without AES hardware (efficient stream cipher)"
                decision_path.append("No AES-NI → ChaCha20 (no padding overhead)")
        
        # Rule 3: High entropy (likely compressed/encrypted)
        elif entropy > 7.5:
            decision_path.append(f"High entropy ({entropy:.2f} > 7.5 bits/byte)")
            
            algorithm = 'ChaCha20'
            confidence = 'medium'
            rationale = "ChaCha20 for high-entropy data (already compressed/encrypted, stream cipher efficient)"
            decision_path.append("High entropy → ChaCha20 (no compression benefit, stream cipher)")
        
        # Rule 4: Already compressed file types
        elif file_type in ['mp4', 'zip', 'jpg', 'jpeg', 'png']:
            decision_path.append(f"Compressed file type: {file_type}")
            
            algorithm = 'ChaCha20'
            confidence = 'medium'
            rationale = f"ChaCha20 for {file_type} files (already compressed, stream cipher efficient)"
            decision_path.append(f"{file_type} is compressed format → ChaCha20")
        
        # Rule 5: Compressible text-like files
        elif file_type in ['txt', 'sql', 'pdf', 'html', 'xml', 'json', 'csv']:
            decision_path.append(f"Text-like file type: {file_type}")
            
            if has_aes_ni:
                algorithm = 'AES-128'
                confidence = 'high'
                rationale = f"AES-128 for {file_type} files with hardware acceleration"
                decision_path.append("AES-NI available → AES-128")
            else:
                algorithm = 'ChaCha20'
                confidence = 'medium'
                rationale = f"ChaCha20 for {file_type} files without AES hardware"
                decision_path.append("No AES-NI → ChaCha20")
        
        # Rule 6: Battery mode optimization
        elif power_mode == 'battery':
            decision_path.append("Battery power mode")
            
            algorithm = 'ChaCha20'
            confidence = 'medium'
            rationale = "ChaCha20 for battery mode (lower power, no specialized hardware needed)"
            decision_path.append("Battery mode → ChaCha20 (power-efficient)")
        
        # Rule 7: Default case
        else:
            decision_path.append("Default case")
            
            if has_aes_ni:
                algorithm = 'AES-128'
                confidence = 'medium'
                rationale = "AES-128 as default with hardware acceleration"
                decision_path.append("Default with AES-NI → AES-128")
            else:
                algorithm = 'ChaCha20'
                confidence = 'medium'
                rationale = "ChaCha20 as default without AES hardware"
                decision_path.append("Default without AES-NI → ChaCha20")
        
        # Get performance estimates
        estimated_energy, estimated_duration = self._estimate_performance(
            algorithm, file_size, file_type
        )
        
        result = {
            'algorithm': algorithm,
            'confidence': confidence,
            'rationale': rationale,
            'estimated_energy_j': estimated_energy,
            'estimated_duration_s': estimated_duration,
            'features_used': {
                'file_size_bytes': file_size,
                'file_type': file_type,
                'entropy': entropy,
                'has_aes_ni': has_aes_ni,
                'security_level': security_level,
                'power_mode': power_mode,
            },
            'decision_path': decision_path,
        }
        
        logger.debug(f"Rule-based selection: {algorithm} ({confidence} confidence)")
        
        return result
    
    def _estimate_performance(
        self,
        algorithm: str,
        file_size: int,
        file_type: str
    ) -> tuple[Optional[float], Optional[float]]:
        """Estimate energy and time from benchmark data.
        
        Args:
            algorithm: Selected algorithm name.
            file_size: File size in bytes.
            file_type: File type extension.
            
        Returns:
            Tuple of (estimated_energy_j, estimated_duration_s).
            Both may be None if no benchmark data available.
        """
        if not self.benchmark_data:
            return None, None
        
        results = self.benchmark_data.get('results', [])
        
        if not results:
            return None, None
        
        # Find matching or closest benchmark result
        matching = [r for r in results 
                   if r['algorithm'] == algorithm and 
                   r.get('file_type') == file_type]
        
        if not matching:
            # Find any result for this algorithm
            matching = [r for r in results if r['algorithm'] == algorithm]
        
        if not matching:
            return None, None
        
        # Find closest file size
        matching.sort(key=lambda r: abs(r['file_size'] - file_size))
        best_match = matching[0]
        
        # Scale estimates by file size ratio
        size_ratio = file_size / best_match['file_size'] if best_match['file_size'] > 0 else 1.0
        
        estimated_energy = best_match['statistics']['median_energy_j'] * size_ratio
        estimated_duration = best_match['statistics']['median_duration_s'] * size_ratio
        
        return estimated_energy, estimated_duration
    
    def get_all_recommendations(
        self,
        file_path: str,
        security_level: str = 'medium',
        power_mode: str = 'plugged'
    ) -> list[dict]:
        """Get recommendations for all algorithms.
        
        Returns a ranked list of all algorithms with explanations.
        
        Args:
            file_path: Path to file to encrypt.
            security_level: 'low', 'medium', or 'high'.
            power_mode: 'plugged' or 'battery'.
            
        Returns:
            List of dicts, each containing algorithm info, sorted by suitability.
        """
        features = FeatureExtractor.extract_features(file_path)
        has_aes_ni = self.hardware.get('has_aes_ni', False)
        
        file_size = features['file_size_bytes']
        entropy = features['entropy']
        
        recommendations = []
        
        # Score each algorithm
        algorithms = ['AES-128', 'AES-256', 'ChaCha20']
        
        for alg in algorithms:
            score = 0
            reasons = []
            
            # Security scoring
            if security_level == 'high':
                if alg == 'AES-256':
                    score += 10
                    reasons.append("Best for high security")
                elif alg == 'ChaCha20':
                    score += 8
                    reasons.append("Good security with constant-time")
                else:
                    score += 5
                    reasons.append("Lower key size for high security")
            
            # Hardware scoring
            if has_aes_ni and alg in ['AES-128', 'AES-256']:
                score += 5
                reasons.append("Hardware acceleration available")
            elif not has_aes_ni and alg == 'ChaCha20':
                score += 5
                reasons.append("No specialized hardware needed")
            
            # Size scoring
            if file_size < 100 * 1024:  # Small file
                if alg in ['AES-128', 'ChaCha20']:
                    score += 3
                    reasons.append("Efficient for small files")
            
            # Entropy scoring
            if entropy > 7.5:
                if alg == 'ChaCha20':
                    score += 3
                    reasons.append("Stream cipher for high-entropy data")
            
            # Battery scoring
            if power_mode == 'battery':
                if alg == 'ChaCha20':
                    score += 2
                    reasons.append("Power efficient")
            
            estimated_energy, estimated_duration = self._estimate_performance(
                alg, file_size, features['file_type']
            )
            
            recommendations.append({
                'algorithm': alg,
                'score': score,
                'reasons': reasons,
                'estimated_energy_j': estimated_energy,
                'estimated_duration_s': estimated_duration,
            })
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations
    
    def explain_rules(self) -> str:
        """Get human-readable explanation of selection rules.
        
        Returns:
            Multi-line string explaining the decision tree.
        """
        return """
CryptoGreen Rule-Based Selection Decision Tree
==============================================

1. IF security_level == 'high':
   - WITH AES-NI hardware → AES-256 (high confidence)
   - WITHOUT AES-NI → ChaCha20 (high confidence)

2. ELIF file_size < 100KB:
   - WITH AES-NI hardware → AES-128 (high confidence)
   - WITHOUT AES-NI → ChaCha20 (high confidence)

3. ELIF entropy > 7.5 (compressed/encrypted):
   → ChaCha20 (medium confidence)

4. ELIF file_type in [mp4, zip, jpg, png]:
   → ChaCha20 (medium confidence, already compressed)

5. ELIF file_type in [txt, sql, pdf, html, xml]:
   - WITH AES-NI hardware → AES-128 (high confidence)
   - WITHOUT AES-NI → ChaCha20 (medium confidence)

6. ELIF power_mode == 'battery':
   → ChaCha20 (medium confidence, power efficient)

7. ELSE (default):
   - WITH AES-NI hardware → AES-128 (medium confidence)
   - WITHOUT AES-NI → ChaCha20 (medium confidence)

Hardware Detection
------------------
- AES-NI: {has_aes_ni}
- CPU: {cpu_model}
- Cores: {cpu_cores}
""".format(
            has_aes_ni=self.hardware.get('has_aes_ni', False),
            cpu_model=self.hardware.get('cpu_model', 'Unknown'),
            cpu_cores=self.hardware.get('cpu_cores', 'Unknown'),
        )
