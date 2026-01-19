"""
Hybrid Algorithm Selector Module

This module combines rule-based and ML-based selectors to provide
intelligent algorithm selection with high accuracy and explainability.

Example:
    >>> from cryptogreen.hybrid_selector import HybridSelector
    >>> selector = HybridSelector()
    >>> result = selector.select_algorithm('document.pdf', verbose=True)
    >>> print(selector.explain_decision(result))
"""

import logging
from pathlib import Path
from typing import Optional

from cryptogreen.rule_based_selector import RuleBasedSelector
from cryptogreen.ml_selector import MLSelector
from cryptogreen.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class HybridSelector:
    """Intelligent selector combining rules and ML.
    
    This selector uses both rule-based heuristics and machine learning
    to provide optimal algorithm selection. It combines the interpretability
    of rules with the data-driven accuracy of ML.
    
    Decision Logic:
        1. Get recommendations from both selectors
        2. If both agree AND ML confidence > 0.8: high confidence
        3. If high security requested: prefer rules (proven security)
        4. If ML confidence > 0.8: prefer ML
        5. If ML confidence > 0.6: moderate preference for ML
        6. Default: prefer rules (interpretable)
    
    Attributes:
        rule_selector: Rule-based selector instance.
        ml_selector: ML-based selector instance.
        
    Example:
        >>> selector = HybridSelector()
        >>> result = selector.select_algorithm('large_file.zip')
        >>> print(f"Algorithm: {result['algorithm']}")
        >>> print(f"Method: {result['method']}")
    """
    
    def __init__(
        self,
        model_path: str = 'results/models/selector_model.pkl',
        benchmark_data: Optional[dict] = None
    ):
        """Initialize hybrid selector.
        
        Args:
            model_path: Path to trained ML model.
            benchmark_data: Optional benchmark results for rule tuning.
        """
        self.rule_selector = RuleBasedSelector(benchmark_data)
        self.ml_selector = MLSelector(model_path)
        
        # Track selector usage statistics
        self._selection_stats = {
            'both_agree': 0,
            'ml_preferred': 0,
            'rules_preferred': 0,
            'rules_fallback': 0,
            'security_override': 0,
            'total_selections': 0,
        }
        
        logger.info("HybridSelector initialized")
        logger.info(f"  ML model trained: {self.ml_selector.is_trained}")
        logger.info(f"  AES-NI available: {self.rule_selector.hardware.get('has_aes_ni', False)}")
    
    def select_algorithm(
        self,
        file_path: str,
        security_level: str = 'medium',
        power_mode: str = 'plugged',
        verbose: bool = False
    ) -> dict:
        """Select optimal algorithm using hybrid approach.
        
        Combines rule-based and ML recommendations to select the best
        algorithm based on multiple factors.
        
        Args:
            file_path: Path to file to encrypt.
            security_level: 'low', 'medium', or 'high'.
            power_mode: 'plugged' or 'battery'.
            verbose: Print decision process to console.
            
        Returns:
            Dict containing:
                - algorithm: Selected algorithm name
                - confidence: 'high', 'medium', or 'low'
                - method: How decision was made:
                    - 'both_agree': Both selectors agreed
                    - 'ml_preferred': ML had high confidence
                    - 'rules_preferred': Rules were preferred
                    - 'security_override': High security forced rules
                - rationale: Human-readable explanation
                - estimated_energy_j: Estimated energy consumption
                - estimated_duration_s: Estimated execution time
                - rule_recommendation: Full rule-based result
                - ml_recommendation: Full ML result
                
        Example:
            >>> result = selector.select_algorithm('video.mp4', verbose=True)
            >>> print(f"Selected: {result['algorithm']}")
            >>> print(f"Confidence: {result['confidence']}")
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"CryptoGreen Hybrid Selection")
            print(f"{'=' * 60}")
            print(f"File: {file_path}")
            print(f"Security Level: {security_level}")
            print(f"Power Mode: {power_mode}")
            print()
        
        # Get rule-based recommendation
        rule_result = self.rule_selector.select_algorithm(
            file_path, security_level, power_mode
        )
        rule_algorithm = rule_result['algorithm']
        
        if verbose:
            print(f"Rule-Based Recommendation:")
            print(f"  Algorithm: {rule_algorithm}")
            print(f"  Confidence: {rule_result['confidence']}")
            print(f"  Rationale: {rule_result['rationale']}")
            print()
        
        # Get ML recommendation
        ml_result = self.ml_selector.select_algorithm(file_path)
        ml_algorithm = ml_result['algorithm']
        ml_confidence = ml_result['confidence']
        
        if verbose:
            print(f"ML Recommendation:")
            print(f"  Algorithm: {ml_algorithm}")
            print(f"  Confidence: {ml_confidence:.1%}")
            if ml_result.get('is_fallback'):
                print(f"  (Using fallback - model not trained)")
            print()
        
        # Hybrid decision logic (from paper Section III.D.3)
        algorithm = None
        confidence = 'medium'
        method = 'rules_preferred'
        rationale = ""
        
        # Track this selection
        self._selection_stats['total_selections'] += 1
        
        # Case 1: Both agree with high ML confidence (>0.8)
        if rule_algorithm == ml_algorithm and ml_confidence > 0.8 and not ml_result.get('is_fallback'):
            algorithm = rule_algorithm
            confidence = 'high'
            method = 'both_agree'
            rationale = (
                f"Both rule-based and ML selectors agree on {algorithm} "
                f"with high ML confidence ({ml_confidence:.1%})"
            )
            self._selection_stats['both_agree'] += 1
        
        # Case 2: High security - always prefer rules (proven security guarantees)
        elif security_level == 'high':
            algorithm = rule_result['algorithm']
            confidence = 'high'
            method = 'rules_preferred'
            rationale = (
                f"High security level requires rule-based {algorithm} "
                f"for proven security guarantees. ML suggested {ml_algorithm} ({ml_confidence:.1%})"
            )
            self._selection_stats['security_override'] += 1
        
        # Case 3: ML has high confidence (>0.8) and not fallback
        elif ml_confidence > 0.8 and not ml_result.get('is_fallback'):
            algorithm = ml_algorithm
            confidence = 'high'
            method = 'ml_preferred'
            rationale = (
                f"ML has high confidence ({ml_confidence:.1%}) for {algorithm}. "
                f"Rule-based suggested {rule_algorithm}"
            )
            self._selection_stats['ml_preferred'] += 1
        
        # Case 4: ML confidence is low (<0.8) or is fallback - use rules
        else:
            algorithm = rule_algorithm
            confidence = rule_result['confidence']
            method = 'rules_fallback'
            rationale = (
                f"Using rule-based {algorithm} due to low ML confidence ({ml_confidence:.1%}). "
                f"ML suggested {ml_algorithm}"
            )
            self._selection_stats['rules_fallback'] += 1
        
        # Log disagreements for analysis
        if rule_algorithm != ml_algorithm:
            logger.info(
                f"Selector disagreement: rules={rule_algorithm}, "
                f"ml={ml_algorithm} ({ml_confidence:.0%}). "
                f"Selected: {algorithm} ({method})"
            )
        
        if verbose:
            print(f"{'=' * 60}")
            print(f"Final Selection: {algorithm}")
            print(f"Confidence: {confidence}")
            print(f"Method: {method}")
            print(f"Rationale: {rationale}")
            print()
            print(f"Selector Usage Stats (session):")
            print(f"  Both Agree (ML>0.8):  {self._selection_stats['both_agree']}")
            print(f"  ML Preferred (ML>0.8): {self._selection_stats['ml_preferred']}")
            print(f"  Rules Fallback (<0.8): {self._selection_stats['rules_fallback']}")
            print(f"  Security Override:     {self._selection_stats['security_override']}")
            print(f"  Total Selections:      {self._selection_stats['total_selections']}")
            print(f"{'=' * 60}")
        
        # Get performance estimates from rule selector
        estimated_energy = rule_result.get('estimated_energy_j')
        estimated_duration = rule_result.get('estimated_duration_s')
        
        return {
            'algorithm': algorithm,
            'confidence': confidence,
            'method': method,
            'rationale': rationale,
            'estimated_energy_j': estimated_energy,
            'estimated_duration_s': estimated_duration,
            'rule_recommendation': rule_result,
            'ml_recommendation': ml_result,
        }
    
    def explain_decision(self, result: dict) -> str:
        """Generate human-readable explanation of decision.
        
        Args:
            result: Result dict from select_algorithm().
            
        Returns:
            Multi-line explanation string with:
                - Selected algorithm
                - Why it was chosen
                - Expected energy/time
                - What each selector recommended
                - Confidence level and reasoning
        """
        lines = [
            "",
            "=" * 60,
            "CRYPTOGREEN ALGORITHM SELECTION EXPLANATION",
            "=" * 60,
            "",
            f"SELECTED ALGORITHM: {result['algorithm']}",
            f"CONFIDENCE: {result['confidence'].upper()}",
            "",
            "RATIONALE:",
            f"  {result['rationale']}",
            "",
        ]
        
        # Performance estimates
        if result.get('estimated_energy_j') is not None:
            lines.extend([
                "ESTIMATED PERFORMANCE:",
                f"  Energy:   {result['estimated_energy_j']:.6f} J",
            ])
            if result.get('estimated_duration_s') is not None:
                lines.append(f"  Duration: {result['estimated_duration_s']:.4f} s")
            lines.append("")
        
        # Decision method
        method_explanations = {
            'both_agree': "Both rule-based and ML selectors agreed on this choice (ML confidence >0.8)",
            'ml_preferred': "ML model had high confidence (>0.8), overriding rules",
            'rules_preferred': "Rule-based selection preferred (high security requirement)",
            'rules_fallback': "Rule-based selection used due to low ML confidence (<0.8)",
            'security_override': "High security requirement - using proven rules",
        }
        
        lines.extend([
            "DECISION METHOD:",
            f"  {method_explanations.get(result['method'], result['method'])}",
            "",
        ])
        
        # Rule-based details
        rule_rec = result.get('rule_recommendation', {})
        lines.extend([
            "RULE-BASED ANALYSIS:",
            f"  Recommendation: {rule_rec.get('algorithm', 'N/A')}",
            f"  Confidence: {rule_rec.get('confidence', 'N/A')}",
        ])
        
        if rule_rec.get('decision_path'):
            lines.append("  Decision Path:")
            for step in rule_rec['decision_path']:
                lines.append(f"    - {step}")
        
        lines.append("")
        
        # ML details
        ml_rec = result.get('ml_recommendation', {})
        lines.extend([
            "ML ANALYSIS:",
            f"  Recommendation: {ml_rec.get('algorithm', 'N/A')}",
            f"  Confidence: {ml_rec.get('confidence', 0):.1%}",
        ])
        
        if ml_rec.get('is_fallback'):
            lines.append("  (Model not trained - using fallback)")
        elif ml_rec.get('alternatives'):
            lines.append(f"  Top Alternatives: {', '.join(ml_rec['alternatives'])}")
        
        if ml_rec.get('probabilities'):
            lines.append("  Probability Distribution:")
            for alg, prob in sorted(ml_rec['probabilities'].items(), key=lambda x: x[1], reverse=True):
                lines.append(f"    {alg}: {prob:.1%}")
        
        lines.append("")
        
        # Features used
        features = rule_rec.get('features_used', {})
        if features:
            lines.extend([
                "FILE CHARACTERISTICS:",
                f"  Size: {features.get('file_size_bytes', 'N/A')} bytes",
                f"  Type: {features.get('file_type', 'N/A')}",
                f"  Entropy: {features.get('entropy', 0):.2f} bits/byte",
                "",
                "SYSTEM CONFIGURATION:",
                f"  AES-NI: {'Yes' if features.get('has_aes_ni') else 'No'}",
                f"  Security Level: {features.get('security_level', 'N/A')}",
                f"  Power Mode: {features.get('power_mode', 'N/A')}",
            ])
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    def batch_select(
        self,
        file_paths: list[str],
        security_level: str = 'medium',
        power_mode: str = 'plugged'
    ) -> list[dict]:
        """Select algorithms for multiple files.
        
        Args:
            file_paths: List of file paths.
            security_level: Security level for all files.
            power_mode: Power mode for all files.
            
        Returns:
            List of selection results.
        """
        results = []
        
        for file_path in file_paths:
            try:
                result = self.select_algorithm(
                    file_path, security_level, power_mode
                )
                result['file_path'] = file_path
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append({
                    'file_path': file_path,
                    'error': str(e),
                })
        
        return results
    
    def get_selection_stats(self, results: list[dict]) -> dict:
        """Get statistics on selection results.
        
        Args:
            results: List of selection results from batch_select.
            
        Returns:
            Statistics dict.
        """
        if not results:
            return {}
        
        valid_results = [r for r in results if 'algorithm' in r]
        
        # Count algorithms
        algorithm_counts = {}
        for r in valid_results:
            alg = r['algorithm']
            algorithm_counts[alg] = algorithm_counts.get(alg, 0) + 1
        
        # Count methods
        method_counts = {}
        for r in valid_results:
            method = r.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Count confidence levels
        confidence_counts = {}
        for r in valid_results:
            conf = r.get('confidence', 'unknown')
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        return {
            'total_files': len(results),
            'successful': len(valid_results),
            'errors': len(results) - len(valid_results),
            'algorithm_distribution': algorithm_counts,
            'method_distribution': method_counts,
            'confidence_distribution': confidence_counts,
        }
    
    def is_trained(self) -> bool:
        """Check if ML model is trained.
        
        Returns:
            True if ML model is trained and ready.
        """
        return self.ml_selector.is_trained
    
    def get_selection_statistics(self) -> dict:
        """Get statistics on selector usage.
        
        Returns:
            Dict with counts and percentages of how each selector was used.
            
        Example:
            >>> stats = selector.get_selection_statistics()
            >>> print(f"ML used: {stats['ml_percentage']:.1f}%")
        """
        total = self._selection_stats['total_selections']
        
        if total == 0:
            return {
                'total_selections': 0,
                'both_agree': 0,
                'ml_preferred': 0,
                'rules_preferred': 0,
                'rules_fallback': 0,
                'security_override': 0,
            }
        
        return {
            'total_selections': total,
            'both_agree': self._selection_stats['both_agree'],
            'both_agree_percentage': (self._selection_stats['both_agree'] / total) * 100,
            'ml_preferred': self._selection_stats['ml_preferred'],
            'ml_preferred_percentage': (self._selection_stats['ml_preferred'] / total) * 100,
            'rules_preferred': self._selection_stats['rules_preferred'],
            'rules_preferred_percentage': (self._selection_stats['rules_preferred'] / total) * 100,
            'rules_fallback': self._selection_stats['rules_fallback'],
            'rules_fallback_percentage': (self._selection_stats['rules_fallback'] / total) * 100,
            'security_override': self._selection_stats['security_override'],
            'security_override_percentage': (self._selection_stats['security_override'] / total) * 100,
        }
    
    def reset_statistics(self):
        """Reset selector usage statistics."""
        self._selection_stats = {
            'both_agree': 0,
            'ml_preferred': 0,
            'rules_preferred': 0,
            'rules_fallback': 0,
            'security_override': 0,
            'total_selections': 0,
        }
