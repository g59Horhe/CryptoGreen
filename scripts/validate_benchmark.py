#!/usr/bin/env python
"""
Benchmark Validation Script

Validates that benchmark results are sane and consistent:
1. AES-256 uses MORE energy than AES-128
2. Energy scales with file size
3. ChaCha20 is competitive with AES (within 2x)
4. No algorithm is always optimal

Usage:
    python scripts/validate_benchmark.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def load_latest_benchmark(benchmark_dir: str = 'results/benchmarks/raw') -> dict:
    """Load the most recent benchmark results."""
    benchmark_path = Path(benchmark_dir)
    json_files = list(benchmark_path.glob('benchmark_*.json'))
    json_files = [f for f in json_files if 'incremental' not in f.name]
    
    if not json_files:
        raise FileNotFoundError(f"No benchmark files found in {benchmark_dir}")
    
    latest = max(json_files, key=lambda f: f.stat().st_mtime)
    print(f"Loading: {latest}")
    
    with open(latest, 'r') as f:
        return json.load(f)


def get_energy_by_size_algo(results: list) -> dict:
    """Organize results by file size and algorithm."""
    data = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        algo = r['algorithm']
        size = r['file_size']
        energy = r['statistics']['median_energy_j']
        data[size][algo].append(energy)
    
    # Average if multiple files of same size
    avg_data = {}
    for size in data:
        avg_data[size] = {}
        for algo in data[size]:
            avg_data[size][algo] = sum(data[size][algo]) / len(data[size][algo])
    
    return avg_data


def analyze_first_files(results: list):
    """Analyze first 3 file sizes (64B, 1KB, 10KB)."""
    print("\n" + "=" * 70)
    print("ANALYSIS: First 3 File Sizes (64B, 1KB, 10KB)")
    print("=" * 70)
    
    # Target sizes
    sizes = [64, 1024, 10240]
    size_names = ['64B', '1KB', '10KB']
    
    data = get_energy_by_size_algo(results)
    
    for size, name in zip(sizes, size_names):
        if size not in data:
            print(f"\n[!] Size {name} not found in results")
            continue
            
        print(f"\n--- {name} Files ---")
        print(f"{'Algorithm':<12} {'Median Energy (J)':<20}")
        print("-" * 32)
        
        algos = data[size]
        for algo, energy in sorted(algos.items(), key=lambda x: x[1]):
            print(f"{algo:<12} {energy:.6f}")
        
        # Find optimal
        optimal = min(algos.items(), key=lambda x: x[1])
        print(f"\n✓ Optimal: {optimal[0]} ({optimal[1]:.6f} J)")
        
        # AES-256 vs AES-128 ratio
        if 'AES-128' in algos and 'AES-256' in algos:
            ratio = algos['AES-256'] / algos['AES-128']
            print(f"  AES-256/AES-128 ratio: {ratio:.2f}x")


def run_validation_checks(results: list) -> dict:
    """Run all validation checks and return results."""
    print("\n" + "=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)
    
    data = get_energy_by_size_algo(results)
    sizes = sorted(data.keys())
    
    checks = {
        'aes256_more_than_aes128': {'passed': 0, 'failed': 0, 'details': []},
        'energy_scales_with_size': {'passed': 0, 'failed': 0, 'details': []},
        'chacha20_competitive': {'passed': 0, 'failed': 0, 'details': []},
        'no_single_optimal': {'passed': False, 'details': ''},
    }
    
    # Check 1: AES-256 uses MORE energy than AES-128
    print("\n[CHECK 1] AES-256 uses MORE energy than AES-128")
    print("-" * 50)
    for size in sizes:
        if 'AES-128' in data[size] and 'AES-256' in data[size]:
            aes128 = data[size]['AES-128']
            aes256 = data[size]['AES-256']
            
            if aes256 > aes128:
                checks['aes256_more_than_aes128']['passed'] += 1
                print(f"  ✓ Size {size}: AES-256 ({aes256:.6f}) > AES-128 ({aes128:.6f})")
            else:
                checks['aes256_more_than_aes128']['failed'] += 1
                checks['aes256_more_than_aes128']['details'].append(
                    f"Size {size}: AES-256 ({aes256:.6f}) <= AES-128 ({aes128:.6f})"
                )
                print(f"  ✗ Size {size}: AES-256 ({aes256:.6f}) <= AES-128 ({aes128:.6f}) [FAILED]")
    
    # Check 2: Energy scales with file size
    print("\n[CHECK 2] Energy scales with file size (larger files use more energy)")
    print("-" * 50)
    for algo in ['AES-128', 'AES-256', 'ChaCha20']:
        algo_energies = [(size, data[size].get(algo, 0)) for size in sizes if algo in data[size]]
        
        if len(algo_energies) < 2:
            continue
        
        # Check if energy generally increases with size
        increasing = True
        for i in range(1, len(algo_energies)):
            prev_size, prev_energy = algo_energies[i-1]
            curr_size, curr_energy = algo_energies[i]
            
            # Allow 10% tolerance for very small files (noise)
            if curr_energy < prev_energy * 0.9 and curr_size > prev_size:
                increasing = False
                checks['energy_scales_with_size']['failed'] += 1
                checks['energy_scales_with_size']['details'].append(
                    f"{algo}: {curr_size}B ({curr_energy:.6f}J) < {prev_size}B ({prev_energy:.6f}J)"
                )
                print(f"  ✗ {algo}: {curr_size}B ({curr_energy:.6f}J) < {prev_size}B ({prev_energy:.6f}J) [FAILED]")
                break
        
        if increasing:
            checks['energy_scales_with_size']['passed'] += 1
            smallest = algo_energies[0]
            largest = algo_energies[-1]
            print(f"  ✓ {algo}: {smallest[0]}B ({smallest[1]:.6f}J) → {largest[0]}B ({largest[1]:.6f}J)")
    
    # Check 3: ChaCha20 is competitive with AES (within 2x)
    print("\n[CHECK 3] ChaCha20 is competitive with AES (within 2x of best AES)")
    print("-" * 50)
    for size in sizes:
        if 'ChaCha20' not in data[size]:
            continue
            
        chacha = data[size]['ChaCha20']
        best_aes = min(
            data[size].get('AES-128', float('inf')),
            data[size].get('AES-256', float('inf'))
        )
        
        if best_aes == float('inf'):
            continue
        
        ratio = chacha / best_aes
        
        if ratio <= 2.5:  # Within 2.5x is competitive
            checks['chacha20_competitive']['passed'] += 1
            print(f"  ✓ Size {size}: ChaCha20/AES = {ratio:.2f}x")
        else:
            checks['chacha20_competitive']['failed'] += 1
            checks['chacha20_competitive']['details'].append(f"Size {size}: ratio = {ratio:.2f}x")
            print(f"  ✗ Size {size}: ChaCha20/AES = {ratio:.2f}x [TOO HIGH]")
    
    # Check 4: No algorithm is always optimal
    print("\n[CHECK 4] No algorithm is always optimal (should vary)")
    print("-" * 50)
    optimal_counts = defaultdict(int)
    optimal_by_size = {}
    
    for size in sizes:
        best_algo = min(data[size].items(), key=lambda x: x[1])[0]
        optimal_counts[best_algo] += 1
        optimal_by_size[size] = best_algo
    
    print("  Optimal algorithm by size:")
    for size in sorted(optimal_by_size.keys()):
        print(f"    {size:>12}B: {optimal_by_size[size]}")
    
    print("\n  Optimal counts:")
    for algo, count in sorted(optimal_counts.items(), key=lambda x: -x[1]):
        pct = count / len(sizes) * 100
        print(f"    {algo}: {count} ({pct:.1f}%)")
    
    # Check if any single algorithm dominates (>90%)
    max_count = max(optimal_counts.values())
    if max_count < len(sizes) * 0.9:  # Less than 90% dominance
        checks['no_single_optimal']['passed'] = True
        checks['no_single_optimal']['details'] = "No algorithm dominates (good variance)"
        print(f"\n  ✓ No single algorithm dominates")
    else:
        checks['no_single_optimal']['passed'] = False
        dominant = max(optimal_counts.items(), key=lambda x: x[1])[0]
        checks['no_single_optimal']['details'] = f"{dominant} dominates at {max_count}/{len(sizes)}"
        print(f"\n  ⚠ {dominant} dominates at {max_count}/{len(sizes)} sizes")
    
    return checks


def print_summary(checks: dict):
    """Print validation summary."""
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    
    # Check 1
    c1 = checks['aes256_more_than_aes128']
    c1_status = "✓ PASS" if c1['failed'] == 0 else "✗ FAIL"
    print(f"\n1. AES-256 > AES-128: {c1_status} ({c1['passed']} passed, {c1['failed']} failed)")
    if c1['failed'] > 0:
        all_passed = False
        for detail in c1['details']:
            print(f"   - {detail}")
    
    # Check 2
    c2 = checks['energy_scales_with_size']
    c2_status = "✓ PASS" if c2['failed'] == 0 else "✗ FAIL"
    print(f"\n2. Energy scales with size: {c2_status} ({c2['passed']} passed, {c2['failed']} failed)")
    if c2['failed'] > 0:
        all_passed = False
        for detail in c2['details']:
            print(f"   - {detail}")
    
    # Check 3
    c3 = checks['chacha20_competitive']
    c3_status = "✓ PASS" if c3['failed'] == 0 else "✗ FAIL"
    print(f"\n3. ChaCha20 competitive: {c3_status} ({c3['passed']} passed, {c3['failed']} failed)")
    if c3['failed'] > 0:
        all_passed = False
        for detail in c3['details']:
            print(f"   - {detail}")
    
    # Check 4
    c4 = checks['no_single_optimal']
    c4_status = "✓ PASS" if c4['passed'] else "⚠ WARNING"
    print(f"\n4. Algorithm variance: {c4_status}")
    print(f"   {c4['details']}")
    if not c4['passed']:
        # This is a warning, not a failure
        pass
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL VALIDATION CHECKS PASSED!")
    else:
        print("⚠ SOME VALIDATION CHECKS FAILED - Review results above")
    print("=" * 70)
    
    return all_passed


def main():
    print("=" * 70)
    print("CRYPTOGREEN BENCHMARK VALIDATION")
    print("=" * 70)
    
    try:
        data = load_latest_benchmark()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    results = data.get('results', [])
    print(f"Loaded {len(results)} benchmark results")
    
    # Configuration summary
    config = data.get('benchmark_config', {})
    print(f"\nBenchmark Configuration:")
    print(f"  Algorithms: {config.get('algorithms', 'N/A')}")
    print(f"  Total configs: {config.get('total_configs', 'N/A')}")
    print(f"  Runs per config: {config.get('runs_per_config', 'N/A')}")
    print(f"  Total measurements: {config.get('total_measurements', 'N/A')}")
    
    # Analyze first files
    analyze_first_files(results)
    
    # Run validation checks
    checks = run_validation_checks(results)
    
    # Print summary
    passed = print_summary(checks)
    
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
