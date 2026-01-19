#!/usr/bin/env python3
"""Test RAPL energy measurement"""

import time
from pathlib import Path

def read_rapl(path="/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"):
    """Read RAPL energy counter"""
    with open(path, 'r') as f:
        return int(f.read().strip())

# Test 1: Read counter twice
print("Test 1: Reading RAPL counter twice...")
e1 = read_rapl()
time.sleep(0.1)
e2 = read_rapl()
print(f"  First read:  {e1:,} µJ")
print(f"  Second read: {e2:,} µJ")
print(f"  Difference:  {e2-e1:,} µJ ({(e2-e1)/1e6:.6f} J)")

if e2 == e1:
    print("  ❌ Counter NOT incrementing!")
else:
    print("  ✅ Counter is incrementing")

# Test 2: Measure computation
print("\nTest 2: Measuring actual work...")
e_start = read_rapl()
t_start = time.perf_counter()

# Do heavy computation
result = sum(i**2 for i in range(10_000_000))

t_end = time.perf_counter()
e_end = read_rapl()

duration = t_end - t_start
energy_uj = e_end - e_start
energy_j = energy_uj / 1_000_000

print(f"  Start energy: {e_start:,} µJ")
print(f"  End energy:   {e_end:,} µJ")
print(f"  Duration:     {duration:.6f} s")
print(f"  Energy used:  {energy_uj:,} µJ = {energy_j:.6f} J")
print(f"  Power:        {energy_j/duration:.2f} W")

if energy_j < 0.01:
    print(f"  ❌ Energy too low! RAPL might be broken")
elif energy_j > 100:
    print(f"  ❌ Energy too high! Something wrong")
else:
    print(f"  ✅ Energy measurement looks good!")
