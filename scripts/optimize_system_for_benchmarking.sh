#!/bin/bash
# System Optimization Script for CryptoGreen Benchmarking
# 
# This script configures the system for consistent, reproducible benchmarks:
# - Sets CPU governor to performance (no frequency scaling)
# - Optionally disables CPU turbo boost (more consistent but slower)
# - Ensures RAPL access
# - Prevents system sleep/suspend
#
# Usage:
#   sudo ./scripts/optimize_system_for_benchmarking.sh [--disable-turbo]

set -e

echo "=========================================="
echo "CryptoGreen Benchmark System Optimization"
echo "=========================================="
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ ERROR: Please run as root (use sudo)"
    exit 1
fi

# 1. Load MSR module
echo "1. Loading MSR module..."
modprobe msr 2>/dev/null || echo "   Already loaded"
echo "   ✓ MSR module ready"
echo

# 2. Ensure RAPL permissions
echo "2. Setting RAPL permissions..."
chmod -R a+r /sys/class/powercap/intel-rapl/ 2>/dev/null || echo "   Already set"
chmod -R a+r /dev/cpu/*/msr 2>/dev/null || echo "   Already set"
echo "   ✓ RAPL accessible"
echo

# 3. Set CPU governor to performance
echo "3. Setting CPU governor to 'performance'..."
GOVERNOR_BEFORE=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
echo "   Current governor: $GOVERNOR_BEFORE"

if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
    GOVERNOR_AFTER=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    echo "   ✓ Governor set to: $GOVERNOR_AFTER"
else
    echo "   ⚠ CPU frequency scaling not available"
fi
echo

# 4. CPU Turbo Boost (optional)
if [ "$1" == "--disable-turbo" ]; then
    echo "4. Disabling CPU turbo boost (--disable-turbo flag)..."
    
    # AMD CPUs
    if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
        TURBO_BEFORE=$(cat /sys/devices/system/cpu/cpufreq/boost)
        echo "   Current boost: $TURBO_BEFORE (AMD)"
        echo 0 | tee /sys/devices/system/cpu/cpufreq/boost > /dev/null
        TURBO_AFTER=$(cat /sys/devices/system/cpu/cpufreq/boost)
        echo "   ✓ Turbo boost disabled: $TURBO_AFTER"
    
    # Intel CPUs
    elif [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        TURBO_BEFORE=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
        echo "   Current no_turbo: $TURBO_BEFORE (Intel)"
        echo 1 | tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
        TURBO_AFTER=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
        echo "   ✓ Turbo boost disabled: $TURBO_AFTER"
    
    else
        echo "   ⚠ Turbo boost control not available"
    fi
    echo
    echo "   ⚠ WARNING: Turbo boost disabled for consistency"
    echo "   ⚠ Benchmarks will be slower but more reproducible"
else
    echo "4. CPU turbo boost: ENABLED (default)"
    
    if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
        TURBO=$(cat /sys/devices/system/cpu/cpufreq/boost)
        echo "   Current boost: $TURBO (AMD)"
    elif [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
        echo "   Current no_turbo: $TURBO (Intel - 0=enabled, 1=disabled)"
    fi
    
    echo "   💡 TIP: Use --disable-turbo flag for more consistent results"
    echo "           (trades speed for reproducibility)"
fi
echo

# 5. Display CPU info
echo "5. CPU Information:"
CPU_MODEL=$(lscpu | grep "Model name" | cut -d ':' -f 2 | xargs)
CPU_CORES=$(nproc)
CPU_FREQ_MIN=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq 2>/dev/null || echo "N/A")
CPU_FREQ_MAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo "N/A")
CPU_FREQ_CUR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo "N/A")

echo "   Model: $CPU_MODEL"
echo "   Cores: $CPU_CORES"
if [ "$CPU_FREQ_MIN" != "N/A" ]; then
    echo "   Frequency range: $(($CPU_FREQ_MIN / 1000)) MHz - $(($CPU_FREQ_MAX / 1000)) MHz"
    echo "   Current frequency: $(($CPU_FREQ_CUR / 1000)) MHz"
fi
echo

echo "=========================================="
echo "✓ System optimized for benchmarking!"
echo "=========================================="
echo
echo "RECOMMENDATIONS FOR ACCURATE BENCHMARKS:"
echo "  1. Close unnecessary applications (browsers, IDEs, etc.)"
echo "  2. Disconnect from external displays (if possible)"
echo "  3. Ensure system is plugged in (not on battery)"
echo "  4. Let system idle for 30 seconds before starting"
echo "  5. Don't interact with system during benchmark"
echo
echo "To prevent sleep during long benchmarks:"
echo "  sudo systemctl mask sleep.target suspend.target"
echo
echo "To restore defaults after benchmarking:"
echo "  sudo ./scripts/restore_system_defaults.sh"
echo
