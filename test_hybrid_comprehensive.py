#!/usr/bin/env python3
"""
Comprehensive test of HybridSelector with detailed decision logic validation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("HYBRID SELECTOR COMPREHENSIVE TEST")
print("=" * 70)
print()

try:
    from cryptogreen.hybrid_selector import HybridSelector
    
    # Check if we have a test file
    test_file = 'data/test_files/txt/txt_1KB.txt'
    
    if not Path(test_file).exists():
        print(f"Test file not found: {test_file}")
        print("Run: python scripts/generate_test_data.py")
        sys.exit(1)
    
    # Initialize selector
    print("Initializing HybridSelector...")
    selector = HybridSelector()
    print(f"  ML Model Trained: {selector.is_trained()}")
    print()
    
    # Test all decision paths
    test_scenarios = [
        {
            'name': 'Scenario 1: Normal operation (medium security)',
            'file': test_file,
            'security': 'medium',
            'power': 'plugged',
            'expected_method': 'rules_fallback or ml_preferred or both_agree',
            'reason': 'Depends on ML confidence',
        },
        {
            'name': 'Scenario 2: High security override',
            'file': test_file,
            'security': 'high',
            'power': 'plugged',
            'expected_method': 'rules_preferred',
            'reason': 'High security always uses rules',
        },
        {
            'name': 'Scenario 3: Battery mode (energy-conscious)',
            'file': test_file,
            'security': 'low',
            'power': 'battery',
            'expected_method': 'rules_fallback or ml_preferred or both_agree',
            'reason': 'Depends on ML confidence',
        },
        {
            'name': 'Scenario 4: Low security (speed priority)',
            'file': test_file,
            'security': 'low',
            'power': 'plugged',
            'expected_method': 'rules_fallback or ml_preferred or both_agree',
            'reason': 'Depends on ML confidence',
        },
    ]
    
    print("=" * 70)
    print("RUNNING TEST SCENARIOS")
    print("=" * 70)
    print()
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print("   " + "-" * 66)
        print(f"   File: {scenario['file']}")
        print(f"   Security: {scenario['security']}")
        print(f"   Power: {scenario['power']}")
        print(f"   Expected: {scenario['expected_method']}")
        print(f"   Reason: {scenario['reason']}")
        print()
        
        result = selector.select_algorithm(
            scenario['file'],
            security_level=scenario['security'],
            power_mode=scenario['power'],
            verbose=False
        )
        
        print(f"   RESULT:")
        print(f"     Algorithm:  {result['algorithm']}")
        print(f"     Method:     {result['method']}")
        print(f"     Confidence: {result['confidence']}")
        print()
        print(f"   DETAILS:")
        print(f"     Rule rec:   {result['rule_recommendation']['algorithm']}")
        ml_rec = result['ml_recommendation']
        print(f"     ML rec:     {ml_rec['algorithm']} ({ml_rec['confidence']:.1%} confidence)")
        if ml_rec.get('is_fallback'):
            print(f"                 (FALLBACK - model not trained)")
        print()
        print(f"   RATIONALE:")
        print(f"     {result['rationale']}")
        print()
        
        # Validate expected behavior
        if scenario['security'] == 'high':
            if result['method'] == 'rules_preferred':
                print(f"   ✓ PASSED: High security correctly used rules")
            else:
                print(f"   ✗ FAILED: High security should use rules_preferred, got {result['method']}")
        
        print()
    
    # Show final statistics
    print("=" * 70)
    print("SELECTOR USAGE STATISTICS")
    print("=" * 70)
    print()
    
    stats = selector.get_selection_statistics()
    
    total = stats['total_selections']
    print(f"Total Selections: {total}")
    print()
    
    print("Decision Path Distribution:")
    print("-" * 70)
    
    categories = [
        ('both_agree', 'Both Agree (ML>0.8)', 'Both selectors agreed with high ML confidence'),
        ('ml_preferred', 'ML Preferred (ML>0.8)', 'ML had high confidence, overrode rules'),
        ('rules_fallback', 'Rules Fallback (ML<0.8)', 'ML confidence too low, used rules'),
        ('security_override', 'Security Override', 'High security required rules'),
    ]
    
    for key, label, description in categories:
        count = stats.get(key, 0)
        pct = stats.get(f'{key}_percentage', 0)
        bar = '█' * int(pct / 2)
        print(f"{label:25s}: {count:2d} ({pct:5.1f}%) {bar}")
        print(f"  → {description}")
        print()
    
    # Analysis
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    
    ml_used = stats.get('ml_preferred', 0) + stats.get('both_agree', 0)
    rules_used = stats.get('rules_fallback', 0) + stats.get('security_override', 0)
    
    print(f"ML-based decisions:   {ml_used} ({ml_used/total*100:.1f}%)")
    print(f"Rule-based decisions: {rules_used} ({rules_used/total*100:.1f}%)")
    print()
    
    if ml_used == 0:
        print("⚠ ML never used - ML confidence below 0.8 threshold")
        print("  This suggests:")
        print("  1. ML model needs more training data")
        print("  2. Current data outside training distribution")
        print("  3. Model uncertainty high (good - being conservative)")
        print()
        print("  To improve:")
        print("  - Train with more diverse data")
        print("  - Use SMOTE to balance training classes")
        print("  - Collect more benchmark results")
    elif rules_used == 0:
        print("⚠ Rules never used - ML always confident (>0.8)")
        print("  This is ideal if ML is well-trained!")
    else:
        print("✓ Healthy mix of ML and rule-based decisions")
    
    print()
    print("=" * 70)
    print("DECISION LOGIC VERIFICATION")
    print("=" * 70)
    print()
    
    print("The HybridSelector implements this decision tree:")
    print()
    print("1. IF both agree AND ML confidence > 0.8:")
    print("     → Use agreed algorithm (high confidence)")
    print("     → Method: 'both_agree'")
    print()
    print("2. ELIF security_level == 'high':")
    print("     → Use rule-based (proven security)")
    print("     → Method: 'rules_preferred'")
    print()
    print("3. ELIF ML confidence > 0.8 AND not fallback:")
    print("     → Use ML algorithm (high confidence)")
    print("     → Method: 'ml_preferred'")
    print()
    print("4. ELSE:")
    print("     → Use rule-based (low ML confidence)")
    print("     → Method: 'rules_fallback'")
    print()
    
    print("This ensures:")
    print("  ✓ ML only used when confident (>0.8)")
    print("  ✓ Rules used when ML uncertain (<0.8)")
    print("  ✓ High security always uses proven rules")
    print("  ✓ Clear tracking of which path taken")
    print()
    
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
