#!/bin/bash
# Restore System Defaults After Benchmarking
#
# This script restores normal system settings after benchmarking.
# Run this after completing benchmarks to return to power-saving mode.
#
# Usage:
#   sudo ./scripts/restore_system_defaults.sh

set -e

echo "=========================================="
echo "Restoring System Defaults"
echo "=========================================="
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ ERROR: Please run as root (use sudo)"
    exit 1
fi

# 1. Set CPU governor back to powersave
echo "1. Restoring CPU governor to 'powersave'..."
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo powersave | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
    GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    echo "   ✓ Governor set to: $GOVERNOR"
else
    echo "   ⚠ CPU frequency scaling not available"
fi
echo

# 2. Re-enable CPU turbo boost
echo "2. Re-enabling CPU turbo boost..."

# AMD CPUs
if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    echo 1 | tee /sys/devices/system/cpu/cpufreq/boost > /dev/null
    TURBO=$(cat /sys/devices/system/cpu/cpufreq/boost)
    echo "   ✓ Turbo boost enabled: $TURBO (AMD)"

# Intel CPUs
elif [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo 0 | tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
    TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
    echo "   ✓ Turbo boost enabled: $TURBO (Intel - 0=enabled)"

else
    echo "   ⚠ Turbo boost control not available"
fi
echo

# 3. Unmask sleep targets (if they were masked)
echo "3. Re-enabling sleep/suspend targets..."
systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target 2>/dev/null || echo "   Already unmasked"
echo "   ✓ Sleep targets restored"
echo

echo "=========================================="
echo "✓ System defaults restored!"
echo "=========================================="
echo
echo "CURRENT SETTINGS:"
echo "  Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'N/A')"

if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    echo "  Turbo boost: $(cat /sys/devices/system/cpu/cpufreq/boost) (AMD)"
elif [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo "  Turbo boost: $([ $(cat /sys/devices/system/cpu/intel_pstate/no_turbo) -eq 0 ] && echo 'enabled' || echo 'disabled') (Intel)"
fi
echo
