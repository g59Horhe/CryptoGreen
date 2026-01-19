#!/usr/bin/env python3
"""
System Compatibility Check for CryptoGreen Energy Benchmarking

This script checks if your system has the necessary capabilities for
accurate energy measurement and cryptographic benchmarking.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_status(item: str, status: bool, details: str = ""):
    """Print a status line with checkmark or X."""
    symbol = "✓" if status else "✗"
    color_start = ""
    color_end = ""
    
    # Try to use colors on supported terminals
    if sys.platform != "win32" or os.environ.get("TERM"):
        color_start = "\033[92m" if status else "\033[91m"
        color_end = "\033[0m"
    
    status_text = f"[{symbol}]" if sys.platform != "win32" else ("[OK]" if status else "[FAIL]")
    print(f"  {status_text} {item}")
    if details:
        print(f"      → {details}")


def check_rapl_support():
    """Check if Intel RAPL (Running Average Power Limit) is available."""
    print_header("RAPL Energy Measurement Support")
    
    rapl_path = Path("/sys/class/powercap/intel-rapl/")
    rapl_available = False
    can_read_energy = False
    
    # Check if running on Linux
    if platform.system() != "Linux":
        print_status("Linux OS", False, f"Current OS: {platform.system()}")
        print("\n  ⚠ RAPL is only available on Linux systems.")
        print("    On Windows/macOS, energy measurement will use estimation methods.")
        return False, False
    
    print_status("Linux OS", True)
    
    # Check if RAPL directory exists
    if rapl_path.exists():
        print_status("RAPL interface exists", True, str(rapl_path))
        rapl_available = True
        
        # Try to find and read energy values
        energy_files = list(rapl_path.glob("*/energy_uj"))
        if energy_files:
            print_status("Energy measurement files found", True, f"Found {len(energy_files)} domain(s)")
            
            # Try to read energy value
            try:
                with open(energy_files[0], 'r') as f:
                    energy_value = int(f.read().strip())
                print_status("Can read energy values", True, f"Current: {energy_value} µJ")
                can_read_energy = True
            except PermissionError:
                print_status("Can read energy values", False, "Permission denied")
                print("\n  ⚠ Try running with sudo or add user to appropriate group:")
                print("    sudo chmod o+r /sys/class/powercap/intel-rapl/*/energy_uj")
            except Exception as e:
                print_status("Can read energy values", False, str(e))
        else:
            print_status("Energy measurement files found", False, "No energy_uj files")
    else:
        print_status("RAPL interface exists", False, "Directory not found")
        
        # Check for AMD RAPL (available in newer kernels)
        amd_rapl_path = Path("/sys/class/powercap/amd-rapl/")
        if amd_rapl_path.exists():
            print_status("AMD RAPL interface", True, str(amd_rapl_path))
            rapl_available = True
        else:
            print_status("AMD RAPL interface", False, "Not found")
    
    return rapl_available, can_read_energy


def check_cpu_info():
    """Check CPU information and capabilities."""
    print_header("CPU Information")
    
    cpu_model = "Unknown"
    cpu_cores = os.cpu_count() or 0
    has_aesni = False
    cpu_governor = "Unknown"
    
    system = platform.system()
    
    if system == "Linux":
        # Read /proc/cpuinfo
        try:
            with open("/proc/cpuinfo", 'r') as f:
                cpuinfo = f.read()
            
            # Extract CPU model
            for line in cpuinfo.split('\n'):
                if line.startswith('model name'):
                    cpu_model = line.split(':')[1].strip()
                    break
            
            # Check for AES-NI
            has_aesni = 'aes' in cpuinfo.lower()
            
        except Exception as e:
            print(f"  Warning: Could not read /proc/cpuinfo: {e}")
        
        # Check CPU governor
        try:
            governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            if governor_path.exists():
                cpu_governor = governor_path.read_text().strip()
        except Exception:
            cpu_governor = "Not available"
            
    elif system == "Windows":
        # Use wmic or platform info
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                cpu_model = lines[1].strip()
        except Exception:
            cpu_model = platform.processor() or "Unknown"
        
        # Check for AES-NI on Windows (check if CPU supports it)
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "caption,description"],
                capture_output=True, text=True, timeout=10
            )
            # Most modern CPUs support AES-NI
            # We'll verify with a Python test
            has_aesni = check_aesni_python()
        except Exception:
            has_aesni = check_aesni_python()
        
        cpu_governor = "N/A (Windows)"
        
    elif system == "Darwin":  # macOS
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=10
            )
            cpu_model = result.stdout.strip()
            
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features"],
                capture_output=True, text=True, timeout=10
            )
            has_aesni = 'AES' in result.stdout.upper()
        except Exception:
            cpu_model = platform.processor() or "Unknown"
            has_aesni = check_aesni_python()
        
        cpu_governor = "N/A (macOS)"
    
    print(f"  CPU Model:    {cpu_model}")
    print(f"  CPU Cores:    {cpu_cores} (logical)")
    print(f"  CPU Governor: {cpu_governor}")
    print()
    print_status("AES-NI Hardware Acceleration", has_aesni,
                 "Hardware-accelerated AES encryption" if has_aesni else "Software AES only")
    
    # Governor warning for benchmarking
    if cpu_governor not in ["performance", "N/A (Windows)", "N/A (macOS)", "Unknown", "Not available"]:
        print(f"\n  ⚠ For accurate benchmarks, consider setting governor to 'performance':")
        print(f"    sudo cpupower frequency-set -g performance")
    
    return {
        "model": cpu_model,
        "cores": cpu_cores,
        "aesni": has_aesni,
        "governor": cpu_governor
    }


def check_aesni_python():
    """Check if AES-NI is available by testing cryptographic operations."""
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        import time
        
        # Quick benchmark to see if AES is hardware-accelerated
        key = os.urandom(32)
        iv = os.urandom(16)
        data = os.urandom(1024 * 1024)  # 1MB
        
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        start = time.perf_counter()
        _ = encryptor.update(data)
        elapsed = time.perf_counter() - start
        
        # If 1MB encrypts in less than 10ms, likely hardware-accelerated
        # This is a heuristic, not definitive
        return elapsed < 0.05  # 50ms threshold
        
    except ImportError:
        return None  # Can't determine
    except Exception:
        return None


def check_python_dependencies():
    """Check if required Python packages are installed."""
    print_header("Python Dependencies")
    
    required_packages = [
        ("cryptography", "Cryptographic algorithms"),
        ("psutil", "System monitoring"),
        ("numpy", "Numerical computations"),
        ("pandas", "Data analysis"),
        ("sklearn", "Machine learning (scikit-learn)"),
        ("matplotlib", "Plotting"),
    ]
    
    all_installed = True
    
    for package, description in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print_status(f"{package}", True, description)
        except ImportError:
            print_status(f"{package}", False, f"Not installed - {description}")
            all_installed = False
    
    if not all_installed:
        print("\n  To install missing packages:")
        print("    pip install -r requirements.txt")
    
    return all_installed


def check_alternative_energy_methods():
    """Check for alternative energy measurement methods."""
    print_header("Alternative Energy Measurement Methods")
    
    methods_available = []
    
    # Check psutil for CPU power estimation
    try:
        import psutil
        print_status("psutil (CPU utilization-based estimation)", True,
                    "Can estimate energy from CPU usage")
        methods_available.append("psutil")
    except ImportError:
        print_status("psutil", False, "Not installed")
    
    # Check for Intel Power Gadget (Windows/macOS)
    if platform.system() == "Windows":
        power_gadget_paths = [
            r"C:\Program Files\Intel\Power Gadget 3.6\PowerLog3.0.exe",
            r"C:\Program Files\Intel\Power Gadget\PowerLog.exe",
        ]
        found = any(Path(p).exists() for p in power_gadget_paths)
        print_status("Intel Power Gadget", found,
                    "Hardware power measurement" if found else "Not installed")
        if found:
            methods_available.append("power_gadget")
    
    # Check for AMD uProf
    if platform.system() in ["Linux", "Windows"]:
        uprof_available = False
        uprof_path = None
        if platform.system() == "Windows":
            uprof_paths = [
                r"C:\Program Files\AMD\AMDuProf\bin\AMDuProfCLI.exe",
                r"C:\Program Files\AMD\AMDuProf\bin\AMDuProfPcm.exe",
            ]
            for p in uprof_paths:
                if Path(p).exists():
                    uprof_available = True
                    uprof_path = p
                    break
        else:
            try:
                result = subprocess.run(["which", "AMDuProfCLI"], 
                                       capture_output=True, timeout=5)
                uprof_available = result.returncode == 0
            except Exception:
                pass
        
        print_status("AMD uProf", uprof_available,
                    f"AMD power profiling ({uprof_path})" if uprof_available else "Not installed - Download: https://www.amd.com/en/developer/uprof.html")
        if uprof_available:
            methods_available.append("amd_uprof")
    
    # Check for HWiNFO (Windows)
    if platform.system() == "Windows":
        hwinfo_paths = [
            r"C:\Program Files\HWiNFO64\HWiNFO64.exe",
            r"C:\Program Files (x86)\HWiNFO32\HWiNFO32.exe",
        ]
        found = any(Path(p).exists() for p in hwinfo_paths)
        print_status("HWiNFO", found,
                    "Hardware monitoring" if found else "Not installed (optional)")
        if found:
            methods_available.append("hwinfo")
    
    return methods_available


def print_recommendations():
    """Print setup recommendations based on system configuration."""
    print_header("Recommendations")
    
    system = platform.system()
    
    if system == "Linux":
        print("""
  For Linux systems with RAPL support:
  
  1. Enable RAPL access for non-root users:
     sudo chmod o+r /sys/class/powercap/intel-rapl/*/energy_uj
     
  2. For AMD CPUs (Zen architecture), ensure kernel 5.8+ for RAPL support
  
  3. Set CPU governor to 'performance' for consistent benchmarks:
     sudo cpupower frequency-set -g performance
     
  4. Disable CPU frequency scaling during benchmarks:
     echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
        """)
    
    elif system == "Windows":
        print("""
  For Windows systems:
  
  1. RAPL is not directly accessible on Windows
     The benchmark will use CPU utilization-based energy estimation
  
  2. For more accurate measurements, consider:
     - Intel Power Gadget: https://www.intel.com/content/www/us/en/developer/articles/tool/power-gadget.html
     - AMD uProf: https://developer.amd.com/amd-uprof/
     - HWiNFO: https://www.hwinfo.com/
  
  3. For best benchmark results:
     - Close background applications
     - Disable Windows Defender real-time scanning temporarily
     - Set power plan to 'High Performance'
     - Run benchmarks with administrator privileges
        """)
    
    elif system == "Darwin":
        print("""
  For macOS systems:
  
  1. RAPL is not accessible on macOS
     The benchmark will use CPU utilization-based energy estimation
  
  2. For Intel Macs, Intel Power Gadget may work
  
  3. For Apple Silicon, native energy APIs are limited
     
  4. For best results, close background applications
        """)


def main():
    """Run all system checks."""
    print("\n" + "="*60)
    print(" CryptoGreen System Compatibility Check")
    print(" Checking energy measurement and benchmarking capabilities")
    print("="*60)
    
    print(f"\n  System: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  Architecture: {platform.machine()}")
    
    # Run all checks
    rapl_available, can_read_energy = check_rapl_support()
    cpu_info = check_cpu_info()
    deps_ok = check_python_dependencies()
    alt_methods = check_alternative_energy_methods()
    
    # Summary
    print_header("Summary")
    
    ready_for_benchmarking = deps_ok and (cpu_info.get("cores", 0) > 0)
    accurate_energy = rapl_available and can_read_energy
    
    print(f"\n  Ready for benchmarking: {'Yes' if ready_for_benchmarking else 'No'}")
    print(f"  Accurate energy measurement: {'Yes (RAPL)' if accurate_energy else 'No (will use estimation)'}")
    print(f"  AES-NI acceleration: {'Yes' if cpu_info.get('aesni') else 'No/Unknown'}")
    print(f"  Alternative energy methods: {', '.join(alt_methods) if alt_methods else 'None detected'}")
    
    if not accurate_energy:
        print("\n  Note: Without RAPL, energy measurements will be estimated based on")
        print("        CPU utilization. Results will be relative, not absolute.")
    
    print_recommendations()
    
    print("\n" + "="*60)
    if ready_for_benchmarking:
        print(" ✓ System is ready for CryptoGreen benchmarking!")
    else:
        print(" ✗ Please install missing dependencies before benchmarking")
    print("="*60 + "\n")
    
    return 0 if ready_for_benchmarking else 1


if __name__ == "__main__":
    sys.exit(main())
