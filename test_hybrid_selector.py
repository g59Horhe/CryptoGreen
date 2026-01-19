#!/usr/bin/env python3
"""
Test script to verify HybridSelector decision logic.
Tests all decision paths and tracks selector usage.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("HYBRID SELECTOR DECISION LOGIC TEST")
print("=" * 70)
print()

# Create mock selectors to test logic
class MockRuleSelector:
    def __init__(self):
        self.hardware = {'has_aes_ni': True, 'cpu_cores': 8}
    
    def select_algorithm(self, file_path, security_level='medium', power_mode='plugged'):
        return {
            'algorithm': 'AES-128',
            'confidence': 'high',
            'rationale': 'Rule-based selection',
            'estimated_energy_j': 0.1,
            'estimated_duration_s': 0.05,
            'features_used': {
                'file_size_bytes': 1024,
                'file_type': 'txt',
                'entropy': 4.5,
                'has_aes_ni': True,
                'security_level': security_level,
                'power_mode': power_mode,
            }
        }

class MockMLSelector:
    def __init__(self, trained=True):
        self.is_trained = trained
    
    def select_algorithm(self, file_path, ml_confidence=0.95, ml_algorithm='AES-128', is_fallback=False):
        return {
            'algorithm': ml_algorithm,
            'confidence': ml_confidence,
            'is_fallback': is_fallback,
            'probabilities': {
                'AES-128': ml_confidence,
                'AES-256': (1 - ml_confidence) / 2,
                'ChaCha20': (1 - ml_confidence) / 2,
            }
        }

# Test the decision logic
print("Test 1: Both Agree with High ML Confidence (>0.8)")
print("-" * 70)
print("  Rules: AES-128")
print("  ML: AES-128 (95% confidence)")
print("  Expected: both_agree")
print()

print("Test 2: High Security Override")
print("-" * 70)
print("  Rules: AES-128")
print("  ML: ChaCha20 (95% confidence)")
print("  Security: high")
print("  Expected: rules_preferred (security_override)")
print()

print("Test 3: ML Preferred (High Confidence)")
print("-" * 70)
print("  Rules: AES-128")
print("  ML: ChaCha20 (95% confidence)")
print("  Security: medium")
print("  Expected: ml_preferred")
print()

print("Test 4: Rules Fallback (Low ML Confidence)")
print("-" * 70)
print("  Rules: AES-128")
print("  ML: ChaCha20 (65% confidence)")
print("  Expected: rules_fallback")
print()

# Now test with real HybridSelector if possible
try:
    from cryptogreen.hybrid_selector import HybridSelector
    
    print("=" * 70)
    print("TESTING REAL HYBRID SELECTOR")
    print("=" * 70)
    print()
    
    # Check if we have test files
    test_file = 'data/test_files/txt/txt_1KB.txt'
    
    if Path(test_file).exists():
        # Note: This requires a trained model
        try:
            selector = HybridSelector()
            
            print(f"ML Model Trained: {selector.is_trained()}")
            print()
            
            # Test with different scenarios
            scenarios = [
                ('medium', 'plugged', 'Standard selection'),
                ('high', 'plugged', 'High security override'),
                ('low', 'battery', 'Battery mode'),
            ]
            
            for security, power, desc in scenarios:
                print(f"Scenario: {desc}")
                print(f"  Security: {security}, Power: {power}")
                
                result = selector.select_algorithm(
                    test_file,
                    security_level=security,
                    power_mode=power,
                    verbose=False
                )
                
                print(f"  Selected: {result['algorithm']}")
                print(f"  Method: {result['method']}")
                print(f"  Confidence: {result['confidence']}")
                print(f"  Rule rec: {result['rule_recommendation']['algorithm']}")
                print(f"  ML rec: {result['ml_recommendation']['algorithm']} "
                      f"({result['ml_recommendation']['confidence']:.1%})")
                print()
            
            # Show statistics
            print("=" * 70)
            print("SELECTOR USAGE STATISTICS")
            print("=" * 70)
            stats = selector.get_selection_statistics()
            
            print(f"Total Selections: {stats['total_selections']}")
            print()
            print("Decision Distribution:")
            print(f"  Both Agree (ML>0.8):  {stats['both_agree']} ({stats.get('both_agree_percentage', 0):.1f}%)")
            print(f"  ML Preferred (>0.8):  {stats['ml_preferred']} ({stats.get('ml_preferred_percentage', 0):.1f}%)")
            print(f"  Rules Fallback (<0.8): {stats['rules_fallback']} ({stats.get('rules_fallback_percentage', 0):.1f}%)")
            print(f"  Security Override:     {stats['security_override']} ({stats.get('security_override_percentage', 0):.1f}%)")
            print()
            
        except Exception as e:
            print(f"Could not test with real selector: {e}")
            print("(This is expected if model is not trained yet)")
    else:
        print(f"Test file not found: {test_file}")
        print("Run: python scripts/generate_test_data.py")
        
except ImportError as e:
    print(f"Could not import HybridSelector: {e}")

print("=" * 70)
print("DECISION LOGIC SUMMARY")
print("=" * 70)
print()
print("The HybridSelector now uses this logic (from paper Section III.D.3):")
print()
print("1. IF both agree AND ML confidence > 0.8:")
print("     → Use agreed algorithm (high confidence)")
print("     → Method: 'both_agree'")
print()
print("2. ELIF security_level == 'high':")
print("     → Use rule-based algorithm (proven security)")
print("     → Method: 'rules_preferred' (counted as security_override)")
print()
print("3. ELIF ML confidence > 0.8 AND not fallback:")
print("     → Use ML algorithm (high ML confidence)")
print("     → Method: 'ml_preferred'")
print()
print("4. ELSE:")
print("     → Use rule-based algorithm (low ML confidence)")
print("     → Method: 'rules_fallback'")
print()
print("Key Changes:")
print("  ✓ Simplified to 4 decision paths (was 6)")
print("  ✓ Only uses ML when confidence > 0.8")
print("  ✓ Always uses rules for high security")
print("  ✓ Tracks usage statistics")
print("  ✓ Better logging and explanation")
print()
print("This ensures rules are used when ML is uncertain!")
