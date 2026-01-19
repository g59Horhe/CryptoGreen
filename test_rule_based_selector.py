#!/usr/bin/env python3
"""
Test RuleBasedSelector decision tree matching paper Section III.D.1.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("RULE-BASED SELECTOR TEST (Paper Section III.D.1)")
print("=" * 70)
print()

try:
    from cryptogreen.rule_based_selector import RuleBasedSelector
    from cryptogreen.feature_extractor import FeatureExtractor
    
    # Check test files exist
    test_files = {
        'txt_1KB': 'data/test_files/txt/txt_1KB.txt',
        'txt_1MB': 'data/test_files/txt/txt_1MB.txt',
        'zip_1MB': 'data/test_files/zip/zip_1MB.zip',
        'mp4_1MB': 'data/test_files/mp4/mp4_1MB.mp4',
    }
    
    missing = [name for name, path in test_files.items() if not Path(path).exists()]
    if missing:
        print(f"Missing test files: {missing}")
        print("Run: python scripts/generate_test_data.py")
        sys.exit(1)
    
    # Initialize selector
    print("Initializing RuleBasedSelector...")
    selector = RuleBasedSelector()
    print(f"  Hardware: AES-NI={selector.hardware['has_aes_ni']}")
    print()
    
    # Test decision tree rules
    test_scenarios = [
        {
            'name': 'Rule 1: No AES-NI → ChaCha20',
            'description': 'Systems without hardware acceleration',
            'test': 'Not testable (requires no AES-NI hardware)',
            'expected': 'ChaCha20',
            'skip': True,
        },
        {
            'name': 'Rule 2: Small files (<100KB) → AES-128',
            'description': 'Small files benefit from minimal overhead',
            'file': test_files['txt_1KB'],
            'security': 'medium',
            'expected': 'AES-128',
            'skip': False,
        },
        {
            'name': 'Rule 3a: High entropy (>7.5) → ChaCha20',
            'description': 'Already compressed data',
            'test': 'Check if any test files have entropy > 7.5',
            'skip': True,  # Need to check entropy first
        },
        {
            'name': 'Rule 3b: Compressed types (zip, mp4) → ChaCha20',
            'description': 'Compressed file types benefit from stream cipher',
            'file': test_files['zip_1MB'],
            'security': 'medium',
            'expected': 'ChaCha20',
            'skip': False,
        },
        {
            'name': 'Rule 4: High security → AES-256',
            'description': 'Explicit high security requirement',
            'file': test_files['txt_1MB'],
            'security': 'high',
            'expected': 'AES-256',
            'skip': False,
        },
        {
            'name': 'Rule 5: Default → AES-128',
            'description': 'General files with AES-NI hardware',
            'file': test_files['txt_1MB'],
            'security': 'medium',
            'expected': 'AES-128',
            'skip': False,
        },
    ]
    
    print("=" * 70)
    print("TESTING DECISION TREE RULES")
    print("=" * 70)
    print()
    
    passed = 0
    failed = 0
    skipped = 0
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print("   " + "-" * 66)
        print(f"   Description: {scenario['description']}")
        
        if scenario.get('skip'):
            print(f"   Status: SKIPPED - {scenario.get('test', 'Not applicable')}")
            print()
            skipped += 1
            continue
        
        # Run test
        result = selector.select_algorithm(
            scenario['file'],
            security_level=scenario['security'],
            power_mode='plugged'
        )
        
        # Check result
        actual = result['algorithm']
        expected = scenario['expected']
        
        print(f"   Expected: {expected}")
        print(f"   Actual:   {actual}")
        print(f"   Decision: {result['decision_path'][-1]}")
        print(f"   Rationale: {result['rationale']}")
        
        if actual == expected:
            print(f"   ✓ PASSED")
            passed += 1
        else:
            print(f"   ✗ FAILED")
            failed += 1
        
        print()
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total:   {len(test_scenarios)}")
    print()
    
    if failed > 0:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("✅ ALL TESTS PASSED")
    
    # Additional validation: Check features extracted correctly
    print()
    print("=" * 70)
    print("FEATURE EXTRACTION VALIDATION")
    print("=" * 70)
    print()
    
    for name, path in test_files.items():
        print(f"{name}:")
        features = FeatureExtractor.extract_features(path)
        file_size = 10 ** features['file_size_log']
        file_type = FeatureExtractor.decode_file_type(int(features['file_type_encoded']))
        
        print(f"  Size: {file_size:.0f} bytes (log={features['file_size_log']:.2f})")
        print(f"  Type: {file_type} (encoded={features['file_type_encoded']:.0f})")
        print(f"  Entropy: {features['entropy']:.2f}")
        print(f"  AES-NI: {bool(features['has_aes_ni'])}")
        print()
    
    # Decision distribution analysis
    print("=" * 70)
    print("DECISION DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print()
    
    print("Testing all 49 files to predict algorithm distribution...")
    
    all_test_files = []
    for ext_dir in Path('data/test_files').iterdir():
        if ext_dir.is_dir():
            all_test_files.extend(list(ext_dir.glob('*')))
    
    if not all_test_files:
        print("No test files found!")
    else:
        distribution = {'AES-128': 0, 'ChaCha20': 0, 'AES-256': 0}
        
        for file_path in all_test_files:
            result = selector.select_algorithm(str(file_path), security_level='medium')
            distribution[result['algorithm']] += 1
        
        total = len(all_test_files)
        print(f"Total files tested: {total}")
        print()
        print("Algorithm Distribution:")
        print("-" * 70)
        
        for algo in ['AES-128', 'ChaCha20', 'AES-256']:
            count = distribution[algo]
            pct = (count / total * 100) if total > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"  {algo:10s}: {count:3d} ({pct:5.1f}%) {bar}")
        
        print()
        print("Expected distribution from paper:")
        print("  AES-128:   ~71% (optimal for most files with AES-NI)")
        print("  ChaCha20:  ~21% (compressed types, high entropy)")
        print("  AES-256:   ~8%  (high security only)")
        print()
        
        # Validate distribution is reasonable
        aes128_pct = (distribution['AES-128'] / total * 100)
        chacha20_pct = (distribution['ChaCha20'] / total * 100)
        
        if aes128_pct > 50:  # Should be majority
            print("✓ AES-128 is majority (expected for optimal performance)")
        else:
            print("⚠ AES-128 not majority - may not match optimal distribution")
        
        if chacha20_pct > 0:
            print("✓ ChaCha20 used for some files (compressed types)")
        
    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
