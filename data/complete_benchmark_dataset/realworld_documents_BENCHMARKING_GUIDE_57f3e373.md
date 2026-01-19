# System Optimization Guide for CryptoGreen Benchmarking

## Quick Start

### 1. Optimize System for Benchmarking
```bash
cd ~/Desktop/cryptogreen  # or wherever your project is
sudo ./scripts/optimize_system_for_benchmarking.sh
```

This configures:
- ✓ CPU governor → performance (no frequency scaling)
- ✓ RAPL permissions (energy measurement access)
- ✓ MSR module loaded
- ✓ Turbo boost enabled (default, faster but variable)

### 2. Run Benchmarks
```bash
# Quick test (10 runs per config, ~2 minutes)
python scripts/run_benchmarks.py --test-mode

# Full benchmark (100 runs per config, ~20 minutes)
python scripts/run_benchmarks.py
```

### 3. Restore Normal Settings
```bash
sudo ./scripts/restore_system_defaults.sh
```

## Advanced Options

### For Maximum Consistency (Slower but More Reproducible)
```bash
# Disable turbo boost for consistent frequencies
sudo ./scripts/optimize_system_for_benchmarking.sh --disable-turbo
```

**Trade-off:**
- ✓ More consistent energy measurements (±2% variance)
- ✗ Benchmarks take 20-40% longer
- ✓ Better for comparing algorithms

### Prevent System Sleep During Long Benchmarks
```bash
# Temporarily disable sleep (recommended for overnight runs)
sudo systemctl mask sleep.target suspend.target hibernate.target

# Or use caffeine (GUI tool)
sudo apt install caffeine
caffeine &
```

### Check Current System State
```bash
# CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Turbo boost (AMD)
cat /sys/devices/system/cpu/cpufreq/boost

# Turbo boost (Intel)
cat /sys/devices/system/cpu/intel_pstate/no_turbo

# RAPL accessibility
cat /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
```

## Best Practices for Accurate Benchmarks

### Before Starting
1. ✓ Close unnecessary applications (browsers, IDEs, Slack, Discord)
2. ✓ Disconnect external displays (if possible)
3. ✓ Ensure system is plugged in (not on battery)
4. ✓ Let system idle for 30 seconds after optimization
5. ✓ Ensure adequate cooling (laptops: use cooling pad)

### During Benchmarking
1. ✗ Don't use the system (no keyboard/mouse input)
2. ✗ Don't start other applications
3. ✗ Don't let the system sleep
4. ✓ Monitor progress with `tail -f results/benchmarks/logs/*.log`

### After Benchmarking
1. ✓ Run `sudo ./scripts/restore_system_defaults.sh`
2. ✓ Review results: `python scripts/analyze_results.py`
3. ✓ Train model: `python scripts/train_model.py`
4. ✓ Evaluate accuracy: `python scripts/evaluate_accuracy.py`

## System Requirements

### Minimum
- Linux kernel with RAPL support (3.13+)
- x86-64 CPU with RAPL (Intel Sandy Bridge+ or AMD Zen+)
- Python 3.8+
- Root access (for system configuration)

### Verified Hardware
- ✓ AMD Ryzen 7 7700X (tested)
- ✓ Intel Core i7/i9 (should work)
- ✓ AMD Ryzen 5000/7000 series (should work)

### Not Supported
- ARM processors (no RAPL)
- Virtual machines (no RAPL access)
- Windows/macOS (RAPL access differs)

## Troubleshooting

### RAPL Not Available
```bash
# Check if RAPL files exist
ls /sys/class/powercap/intel-rapl/

# If not, check kernel support
dmesg | grep -i rapl

# Load intel_rapl module
sudo modprobe intel_rapl_msr
```

### Permission Denied on RAPL
```bash
# Fix permissions
sudo chmod -R a+r /sys/class/powercap/intel-rapl/
sudo chmod -R a+r /dev/cpu/*/msr
```

### CPU Governor Won't Change
```bash
# Check available governors
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors

# Some systems require installing cpufrequtils
sudo apt install cpufrequtils
```

### Inconsistent Energy Measurements
Possible causes:
1. Other applications running → close them
2. Turbo boost variability → use `--disable-turbo`
3. Thermal throttling → ensure adequate cooling
4. Background services → check `htop` or `top`

## Configuration Files

### Created by Optimization Script
- CPU governor: `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
- Turbo boost: `/sys/devices/system/cpu/cpufreq/boost` (AMD)
- Turbo boost: `/sys/devices/system/cpu/intel_pstate/no_turbo` (Intel)
- RAPL: `/sys/class/powercap/intel-rapl/`

### Restoration
All settings are restored to defaults by `restore_system_defaults.sh`:
- Governor: powersave
- Turbo: enabled
- Sleep targets: unmasked

## Performance Impact

### powersave → performance
- CPU runs at max frequency constantly
- Energy consumption increases by 5-15W idle
- No performance loss during benchmarks
- Reduces frequency scaling artifacts

### Turbo Enabled vs Disabled
- Enabled: 20-40% faster, more variance (±5%)
- Disabled: Slower, less variance (±2%)
- Recommendation: Keep enabled unless comparing algorithms

## Example Benchmark Session

```bash
# 1. Optimize system
sudo ./scripts/optimize_system_for_benchmarking.sh

# 2. Wait 30 seconds for system to stabilize
sleep 30

# 3. Run full benchmarks (overnight if needed)
python scripts/run_benchmarks.py > benchmark.log 2>&1 &

# 4. Monitor progress
tail -f benchmark.log

# 5. When complete, restore defaults
sudo ./scripts/restore_system_defaults.sh

# 6. Analyze results
python scripts/analyze_results.py
python scripts/train_model.py
python scripts/evaluate_accuracy.py
```

## Additional Resources

- Paper: Section IV.B (Experimental Setup)
- RAPL documentation: Intel 64 and IA-32 Architectures SDM
- Linux power management: `/Documentation/power/` in kernel tree
- CryptoGreen README: `README.md`
