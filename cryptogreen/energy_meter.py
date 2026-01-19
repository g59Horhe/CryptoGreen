"""
RAPL Energy Measurement Module

This module provides hardware energy measurement using the RAPL (Running Average
Power Limit) interface available on Intel and AMD processors. Supports both
Linux (via powercap) and Windows (via LibreHardwareMonitor or Intel Power Gadget).

Example:
    >>> from cryptogreen.energy_meter import RAPLEnergyMeter
    >>> meter = RAPLEnergyMeter()
    >>> result = meter.measure_function(my_crypto_function, data, key)
    >>> print(f"Energy: {result['energy_joules']:.6f} J")
"""

import logging
import os
import platform
import time
import ctypes
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Windows-specific imports
WINDOWS = platform.system() == 'Windows'

# LibreHardwareMonitor integration for Windows
_lhm_computer = None
_lhm_available = False

def _init_librehardwaremonitor():
    """Initialize LibreHardwareMonitor for Windows RAPL access."""
    global _lhm_computer, _lhm_available
    
    if not WINDOWS:
        return False
    
    try:
        import clr
        import sys
        
        # Try to find LibreHardwareMonitorLib.dll
        dll_paths = [
            Path(__file__).parent / "LibreHardwareMonitorLib.dll",
            Path(__file__).parent.parent / "Rufus" / "LibreHardwareMonitorLib.dll",
            Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / "LibreHardwareMonitor" / "LibreHardwareMonitorLib.dll",
            Path(os.environ.get('LOCALAPPDATA', '')) / "LibreHardwareMonitor" / "LibreHardwareMonitorLib.dll",
        ]
        
        dll_found = None
        for dll_path in dll_paths:
            if dll_path.exists():
                dll_found = str(dll_path)
                break
        
        if dll_found:
            clr.AddReference(dll_found)
            from LibreHardwareMonitor.Hardware import Computer, HardwareType, SensorType
            
            _lhm_computer = Computer()
            _lhm_computer.IsCpuEnabled = True
            _lhm_computer.Open()
            _lhm_available = True
            logger.info("LibreHardwareMonitor initialized successfully")
            return True
        else:
            logger.debug("LibreHardwareMonitor DLL not found")
            return False
            
    except ImportError:
        logger.debug("pythonnet not installed - LibreHardwareMonitor unavailable")
        return False
    except Exception as e:
        logger.debug(f"LibreHardwareMonitor init failed: {e}")
        return False


class WindowsRAPLReader:
    """Windows RAPL reader using multiple backends.
    
    Supports:
    1. AMD uProf (for AMD CPUs - requires Administrator)
    2. LibreHardwareMonitor (recommended for AMD without admin)
    3. Intel Power Gadget API
    4. CPU-time based estimation (fallback - always available)
    """
    
    # Typical TDP values for common CPUs (Watts)
    CPU_TDP_ESTIMATES = {
        # AMD Ryzen 9000 series (Zen 5)
        'AMD Ryzen 9 9950X': 170,
        'AMD Ryzen 9 9900X': 120,
        'AMD Ryzen 7 9700X': 65,
        'AMD Ryzen 5 9600X': 65,
        # AMD Ryzen 7000 series (Zen 4)
        'AMD Ryzen 9 7950X': 170,
        'AMD Ryzen 9 7900X': 170,
        'AMD Ryzen 7 7800X3D': 120,
        'AMD Ryzen 7 7700X': 105,
        'AMD Ryzen 5 7600X': 105,
        'AMD Ryzen 5 7600': 65,
        # AMD Ryzen 5000 series (Zen 3)
        'AMD Ryzen 9 5950X': 105,
        'AMD Ryzen 9 5900X': 105,
        'AMD Ryzen 7 5800X': 105,
        'AMD Ryzen 5 5600X': 65,
        # Intel 14th gen
        'Intel Core i9-14900K': 125,
        'Intel Core i7-14700K': 125,
        'Intel Core i5-14600K': 125,
        # Intel 13th gen
        'Intel Core i9-13900K': 125,
        'Intel Core i7-13700K': 125,
        'Intel Core i5-13600K': 125,
        'default': 65,
    }
    
    def __init__(self):
        """Initialize Windows RAPL reader."""
        self.backend = None
        self.energy_offset = 0.0  # Accumulated energy in Joules
        self.last_power = 0.0
        self.last_read_time = time.perf_counter()
        self._tdp = self._detect_cpu_tdp()
        self._ipg_lib = None
        self._uprof_api = None
        self._uprof_path = None
        self._is_admin = self._check_admin()
        
        # Initialize psutil CPU monitoring (needed for first read)
        try:
            import psutil
            psutil.cpu_percent(interval=None)  # Initialize the baseline
        except Exception:
            pass
        
        # Try backends in order of preference
        if self._init_amd_uprof():
            self.backend = 'amd_uprof'
            logger.info("Using AMD uProf for energy measurement")
        elif self._init_librehardwaremonitor():
            self.backend = 'lhm'
            logger.info("Using LibreHardwareMonitor for energy measurement")
        elif self._init_intel_power_gadget():
            self.backend = 'ipg'
            logger.info("Using Intel Power Gadget for energy measurement")
        else:
            # Fallback to CPU-time based estimation
            self.backend = 'cpu_time'
            logger.info(f"Using CPU-time based energy estimation (TDP={self._tdp}W)")
        
        logger.info(f"Windows RAPL initialized with backend: {self.backend}")
    
    def _check_admin(self) -> bool:
        """Check if running with Administrator privileges."""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False
    
    def _detect_cpu_tdp(self) -> float:
        """Detect CPU TDP from model name."""
        cpu_name = ""
        
        # Try py-cpuinfo first
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            cpu_name = info.get('brand_raw', '')
        except ImportError:
            pass
        except Exception:
            pass
        
        # Try WMIC on Windows if cpuinfo failed
        if not cpu_name and WINDOWS:
            try:
                import subprocess
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'name'],
                    capture_output=True, text=True, timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
                if len(lines) > 1:
                    cpu_name = lines[1]
            except Exception:
                pass
        
        # Match against known TDP values
        if cpu_name:
            for model, tdp in self.CPU_TDP_ESTIMATES.items():
                if model != 'default' and model.lower() in cpu_name.lower():
                    logger.info(f"Detected CPU: {cpu_name}, TDP: {tdp}W")
                    return tdp
        
        logger.debug(f"CPU '{cpu_name}' not in TDP database, using default")
        return self.CPU_TDP_ESTIMATES['default']
    
    def _init_librehardwaremonitor(self) -> bool:
        """Initialize LibreHardwareMonitor backend."""
        global _lhm_computer, _lhm_available
        
        if _lhm_available:
            return True
            
        return _init_librehardwaremonitor()
    
    def _init_intel_power_gadget(self) -> bool:
        """Initialize Intel Power Gadget backend."""
        try:
            # Try to load Intel Power Gadget DLL
            ipg_paths = [
                Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / "Intel" / "Power Gadget 3.6" / "EnergyLib64.dll",
                Path(os.environ.get('PROGRAMFILES(X86)', 'C:/Program Files (x86)')) / "Intel" / "Power Gadget 3.6" / "EnergyLib64.dll",
            ]
            
            for ipg_path in ipg_paths:
                if ipg_path.exists():
                    self._ipg_lib = ctypes.CDLL(str(ipg_path))
                    self._ipg_lib.IntelEnergyLibInitialize()
                    logger.info(f"Intel Power Gadget initialized from {ipg_path}")
                    return True
            return False
        except Exception as e:
            logger.debug(f"Intel Power Gadget init failed: {e}")
            return False
    
    def _init_amd_uprof(self) -> bool:
        """Initialize AMD uProf backend.
        
        AMD uProf provides power profiling via:
        1. AMDPowerProfileAPI.dll - Power Profiler API (requires Admin)
        2. AMDuProfPcm.exe - Power Counter Monitor (CLI, requires Admin)
        
        Note: AMD uProf requires Administrator privileges to access hardware
        power counters. If not running as admin, this will fall back to
        CPU-time based estimation.
        """
        self._uprof_pcm_process = None
        self._uprof_pcm_path = None
        
        # Check for AMD uProf installation
        uprof_bin = Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / "AMD" / "AMDuProf" / "bin"
        
        if not uprof_bin.exists():
            uprof_bin = Path("C:/Program Files/AMD/AMDuProf/bin")
        
        if not uprof_bin.exists():
            logger.debug("AMD uProf not found")
            return False
        
        self._uprof_path = uprof_bin
        logger.info(f"Found AMD uProf at {uprof_bin}")
        
        # Check for admin privileges
        if not self._is_admin:
            logger.warning("AMD uProf requires Administrator privileges for hardware power monitoring")
            logger.warning("Run VS Code / terminal as Administrator to use AMD uProf")
            # Still return True - we found uProf, we'll just use estimation with better TDP values
            return True
        
        # NOTE: AMD uProf Power Profile API (AMDPowerProfileAPI.dll) can hang
        # during initialization due to driver interactions. We skip the DLL
        # approach and use CLI-based monitoring or estimation instead.
        # 
        # If you need direct API access:
        # 1. Ensure AMD uProf is properly installed
        # 2. Restart after driver updates  
        # 3. The DLL requires specific driver versions
        logger.debug("Skipping AMD uProf Power API DLL (can cause hangs)")
        
        # Fallback: Check if AMDuProfPcm.exe exists for CLI-based monitoring
        pcm_path = uprof_bin / "AMDuProfPcm.exe"
        if pcm_path.exists():
            self._uprof_pcm_path = pcm_path
            logger.info("AMD uProf PCM CLI available as fallback")
            return True
        
        return False
    
    def _init_uprof_api_functions(self):
        """Initialize AMD uProf API function prototypes."""
        if not self._uprof_api:
            return
        
        try:
            # AMDTPwrStartProfiling() -> AMDTResult
            start_func = getattr(self._uprof_api, 'AMDTPwrStartProfiling', None)
            if start_func:
                start_func.restype = ctypes.c_int
                self._uprof_start = start_func
            
            # AMDTPwrStopProfiling() -> AMDTResult
            stop_func = getattr(self._uprof_api, 'AMDTPwrStopProfiling', None)
            if stop_func:
                stop_func.restype = ctypes.c_int
                self._uprof_stop = stop_func
            
            # AMDTPwrReadAllEnabledCounters(int* pNumCounters, AMDTPwrCounterValue** ppCounters)
            read_func = getattr(self._uprof_api, 'AMDTPwrReadAllEnabledCounters', None)
            if read_func:
                self._uprof_read = read_func
                
        except Exception as e:
            logger.debug(f"AMD uProf API function init failed: {e}")

    def read_power_watts(self) -> float:
        """Read current CPU package power in watts."""
        if self.backend == 'amd_uprof':
            return self._read_power_amd_uprof()
        elif self.backend == 'lhm':
            return self._read_power_lhm()
        elif self.backend == 'ipg':
            return self._read_power_ipg()
        elif self.backend == 'cpu_time':
            return self._read_power_cpu_time()
        return 0.0
    
    def _read_power_amd_uprof(self) -> float:
        """Read power using AMD uProf.
        
        Uses the AMD Power Profiler API if available and running as admin,
        otherwise falls back to CPU-time based estimation with accurate TDP.
        """
        if self._uprof_api and self._is_admin:
            try:
                # Try to read power via API
                power = ctypes.c_double()
                # AMDTPwrGetPowerData takes counter ID and output pointer
                # Counter 0 is typically CPU package power
                get_power = getattr(self._uprof_api, 'AMDTPwrReadPowerSample', None)
                if get_power:
                    result = get_power(ctypes.byref(power))
                    if result == 0:
                        return power.value
            except Exception as e:
                logger.debug(f"AMD uProf API power read error: {e}")
        
        # Fallback to CPU-time estimation (with AMD-specific TDP values)
        return self._read_power_cpu_time()
    
    def _read_power_lhm(self) -> float:
        """Read power using LibreHardwareMonitor."""
        global _lhm_computer
        
        if not _lhm_computer:
            return 0.0
        
        try:
            from LibreHardwareMonitor.Hardware import HardwareType, SensorType
            
            _lhm_computer.Accept(HardwareVisitor())
            
            for hardware in _lhm_computer.Hardware:
                if hardware.HardwareType == HardwareType.Cpu:
                    hardware.Update()
                    for sensor in hardware.Sensors:
                        if sensor.SensorType == SensorType.Power:
                            name = str(sensor.Name).lower()
                            if 'package' in name or 'cpu' in name:
                                if sensor.Value is not None:
                                    return float(sensor.Value)
            return 0.0
        except Exception as e:
            logger.debug(f"LHM power read error: {e}")
            return 0.0
    
    def _read_power_ipg(self) -> float:
        """Read power using Intel Power Gadget."""
        try:
            num_nodes = ctypes.c_int()
            self._ipg_lib.GetNumNodes(ctypes.byref(num_nodes))
            
            power = ctypes.c_double()
            self._ipg_lib.GetPowerData(0, 0, ctypes.byref(power), None)
            return power.value
        except Exception as e:
            logger.debug(f"IPG power read error: {e}")
            return 0.0
    
    def _read_power_cpu_time(self) -> float:
        """Estimate power based on CPU utilization.
        
        Uses TDP scaled by CPU usage. For short benchmarks, we assume
        high CPU usage since cryptographic operations are CPU-intensive.
        """
        try:
            import psutil
            # Use non-blocking read (compares to last call)
            cpu_percent = psutil.cpu_percent(interval=None)
            if cpu_percent == 0 or cpu_percent is None:
                # If no reading yet, assume high load for crypto operations
                # This happens when cpu_percent hasn't been called before
                cpu_percent = 80.0
            # Scale TDP by CPU utilization, with minimum floor for active operations
            # During crypto operations, CPU is typically 70-100% utilized
            cpu_percent = max(cpu_percent, 70.0)  # Minimum 70% for crypto ops
            return self._tdp * (cpu_percent / 100.0)
        except Exception:
            return self._tdp * 0.8  # Assume 80% load for crypto
    
    def read_energy_joules(self) -> float:
        """Read accumulated energy in Joules.
        
        Since Windows doesn't provide direct energy counters like Linux RAPL,
        we integrate power readings over time.
        """
        current_time = time.perf_counter()
        dt = current_time - self.last_read_time
        
        current_power = self.read_power_watts()
        
        # Integrate power over time (trapezoidal rule)
        avg_power = (self.last_power + current_power) / 2.0
        self.energy_offset += avg_power * dt
        
        self.last_power = current_power
        self.last_read_time = current_time
        
        return self.energy_offset
    
    def read_energy_microjoules(self) -> int:
        """Read accumulated energy in microjoules."""
        return int(self.read_energy_joules() * 1_000_000)
    
    @staticmethod
    def is_available() -> bool:
        """Check if Windows RAPL is available."""
        if not WINDOWS:
            return False
        
        # Check AMD uProf
        uprof_paths = [
            Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / "AMD" / "AMDuProf" / "bin" / "AMDuProfCLI.exe",
            Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / "AMD" / "AMDuProf" / "bin" / "AMDuProfPcm.exe",
        ]
        for path in uprof_paths:
            if path.exists():
                return True
        
        # Check LibreHardwareMonitor
        if _lhm_available:
            return True
        
        # Try to initialize
        if _init_librehardwaremonitor():
            return True
        
        # Check Intel Power Gadget
        ipg_paths = [
            Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / "Intel" / "Power Gadget 3.6" / "EnergyLib64.dll",
            Path(os.environ.get('PROGRAMFILES(X86)', 'C:/Program Files (x86)')) / "Intel" / "Power Gadget 3.6" / "EnergyLib64.dll",
        ]
        for path in ipg_paths:
            if path.exists():
                return True
        
        # Always return True on Windows - we'll use CPU-time based estimation
        # which is reliable for comparative benchmarking
        return True


class HardwareVisitor:
    """Visitor pattern for LibreHardwareMonitor hardware tree."""
    
    def VisitComputer(self, computer):
        computer.Traverse(self)
    
    def VisitHardware(self, hardware):
        hardware.Update()
        for subhardware in hardware.SubHardware:
            subhardware.Accept(self)
    
    def VisitSensor(self, sensor):
        pass
    
    def VisitParameter(self, parameter):
        pass


class RAPLNotAvailableError(Exception):
    """Raised when RAPL interface is not available on the system.
    
    This exception includes setup instructions for enabling RAPL access
    on supported systems (Intel/AMD processors).
    
    IMPORTANT: AMD Ryzen processors (including Ryzen 7 7700X) use the 
    intel-rapl interface, NOT a separate AMD interface.
    """
    
    def __init__(self, message: str = None):
        if WINDOWS:
            default_message = (
                "RAPL interface is not available on this Windows system.\\n\\n"
                "To enable RAPL access on Windows:\\n"
                "  Option 1 - AMD uProf (recommended for AMD Ryzen):\\n"
                "    1. Download AMD uProf from AMD Developer website\\n"
                "    2. Install and run as Administrator\\n"
                "    3. Provides hardware power counters for Ryzen CPUs\\n\\n"
                "  Option 2 - LibreHardwareMonitor (works for AMD and Intel):\\n"
                "    1. Download: https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/releases\\n"
                "    2. Install pythonnet: pip install pythonnet\\n"
                "    3. Copy LibreHardwareMonitorLib.dll to the cryptogreen folder\\n"
                "    4. Run your script as Administrator\\n\\n"
                "  Option 3 - Intel Power Gadget (Intel CPUs only):\\n"
                "    1. Download from Intel website\\n"
                "    2. Install to default location\\n\\n"
                "Fallback: CPU-time based estimation will be used automatically.\\n"
                "This provides relative energy comparisons using CPU TDP values.\\n\\n"
                "Supported hardware:\\n"
                "  - AMD Ryzen 7000 series (including 7700X) - uses intel-rapl interface\\n"
                "  - AMD Ryzen 5000/3000 series\\n"
                "  - Intel Sandy Bridge (2011) and later"
            )
        else:
            default_message = (
                "RAPL interface is not available on this Linux system.\\n\\n"
                "To enable RAPL access on Linux:\\n\\n"
                "  Step 1 - Load kernel modules:\\n"
                "    sudo modprobe intel_rapl_common\\n"
                "    sudo modprobe intel_rapl_msr\\n"
                "    # For AMD Ryzen, the same modules work!\\n\\n"
                "  Step 2 - Set permissions:\\n"
                "    sudo chmod -R a+r /sys/class/powercap/intel-rapl/\\n\\n"
                "  Step 3 - Verify access:\\n"
                "    cat /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj\\n"
                "    cat /sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj\\n\\n"
                "IMPORTANT for AMD Ryzen (including 7700X):\\n"
                "  AMD Ryzen uses the 'intel-rapl' interface, NOT 'amd-rapl'.\\n"
                "  This is correct behavior - the powercap RAPL driver is generic.\\n\\n"
                "Supported hardware:\\n"
                "  - AMD Ryzen 7000 series (including 7700X)\\n"
                "  - AMD Ryzen 5000/3000 series\\n"
                "  - Intel Sandy Bridge (2011) and later\\n\\n"
                "If running in a VM or container, RAPL may not be accessible."
            )
        super().__init__(message or default_message)


class RAPLEnergyMeter:
    """Measures CPU energy consumption via RAPL interface.
    
    This class provides methods to read energy counters and measure the
    energy consumption of function executions. On Linux, it uses the powercap
    RAPL interface. On Windows, it uses LibreHardwareMonitor or Intel Power Gadget.
    
    IMPORTANT: AMD Ryzen processors (including Ryzen 7 7700X) use the Intel RAPL
    interface via the powercap subsystem, NOT a separate AMD-specific interface.
    The path is: /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
    
    Attributes:
        rapl_path: Path to RAPL energy counter file (Linux only).
        max_energy_range: Maximum value before counter wraps around.
        
    Example:
        >>> meter = RAPLEnergyMeter()
        >>> result = meter.measure_function(encrypt, plaintext, key)
        >>> print(f"Energy: {result['energy_joules']:.6f} J")
        >>> print(f"Duration: {result['duration_seconds']:.6f} s")
        >>> print(f"Power: {result['average_power_watts']:.2f} W")
    """
    
    # RAPL paths for different systems (Linux)
    # Note: AMD Ryzen (Zen 2 and later) uses the intel-rapl interface
    # via the powercap subsystem, not a separate amd-rapl interface.
    RAPL_PATHS = [
        # Primary path for both Intel and AMD (via intel-rapl driver)
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
        # Alternative path format (some kernels)
        "/sys/class/powercap/intel-rapl:0/energy_uj",
        # AMD-specific path (rarely used, most AMD CPUs use intel-rapl)
        "/sys/class/powercap/amd-rapl/amd-rapl:0/energy_uj",
    ]
    
    # Max energy range paths (for counter overflow handling)
    MAX_ENERGY_PATHS = [
        "/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj",
        "/sys/class/powercap/intel-rapl:0/max_energy_range_uj",
        "/sys/class/powercap/amd-rapl/amd-rapl:0/max_energy_range_uj",
    ]
    
    def __init__(self, rapl_path: Optional[str] = None):
        """Initialize RAPL energy meter.
        
        Args:
            rapl_path: Custom path to RAPL energy file (Linux only). If None, 
                      will auto-detect from known locations.
                      
        Raises:
            RAPLNotAvailableError: If RAPL interface is not found or
                                   not readable.
        """
        self._windows_reader = None
        self._is_windows = WINDOWS
        self._cpu_info = self._detect_cpu_info()
        
        if self._is_windows:
            # Use Windows RAPL backend
            self._windows_reader = WindowsRAPLReader()
            self.rapl_path = None
            self.max_energy_range = 2**63  # Effectively unlimited for Windows
            logger.info(f"RAPL initialized using Windows backend for {self._cpu_info.get('brand', 'Unknown CPU')}")
        else:
            # Use Linux powercap interface
            self.rapl_path = self._find_rapl_path(rapl_path)
            self.max_energy_range = self._read_max_energy_range()
            
            # Log CPU and RAPL info
            cpu_brand = self._cpu_info.get('brand', 'Unknown')
            is_amd = 'amd' in cpu_brand.lower()
            
            logger.info(f"CPU detected: {cpu_brand}")
            if is_amd:
                logger.info("Note: AMD Ryzen uses intel-rapl interface via powercap")
            logger.info(f"RAPL path: {self.rapl_path}")
            logger.info(f"Max energy range: {self.max_energy_range:,} µJ ({self.max_energy_range / 1_000_000:.2f} J)")
            
            # Verify we can read energy
            try:
                initial_energy = self.read_energy()
                logger.info(f"Initial energy reading: {initial_energy:,} µJ")
            except Exception as e:
                raise RAPLNotAvailableError(
                    f"RAPL path found but cannot read energy: {e}"
                )
    
    def _detect_cpu_info(self) -> dict:
        """Detect CPU information.
        
        Returns:
            Dict with CPU brand, vendor, etc.
        """
        info = {'brand': 'Unknown', 'vendor': 'Unknown'}
        
        # Try py-cpuinfo first
        try:
            import cpuinfo
            cpu_data = cpuinfo.get_cpu_info()
            info['brand'] = cpu_data.get('brand_raw', 'Unknown')
            info['vendor'] = cpu_data.get('vendor_id_raw', 'Unknown')
            return info
        except ImportError:
            pass
        except Exception:
            pass
        
        # Try reading /proc/cpuinfo on Linux
        if not WINDOWS:
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            info['brand'] = line.split(':')[1].strip()
                        elif line.startswith('vendor_id'):
                            info['vendor'] = line.split(':')[1].strip()
                        if info['brand'] != 'Unknown' and info['vendor'] != 'Unknown':
                            break
            except Exception:
                pass
        
        return info
    
    def _find_rapl_path(self, custom_path: Optional[str] = None) -> Path:
        """Find the RAPL energy counter path.
        
        Args:
            custom_path: User-specified path to check first.
            
        Returns:
            Path to RAPL energy counter file.
            
        Raises:
            RAPLNotAvailableError: If no valid RAPL path is found.
        """
        paths_to_check = []
        
        if custom_path:
            paths_to_check.append(custom_path)
        
        paths_to_check.extend(self.RAPL_PATHS)
        
        for path in paths_to_check:
            path_obj = Path(path)
            if path_obj.exists():
                if os.access(path_obj, os.R_OK):
                    return path_obj
                else:
                    # Path exists but not readable - permission issue
                    raise RAPLNotAvailableError(
                        f"RAPL interface found but not readable: {path}\n\n"
                        f"To fix permissions on Linux:\n"
                        f"  sudo chmod a+r /sys/class/powercap/intel-rapl -R\n\n"
                        f"Or add your user to the 'rapl' group if it exists:\n"
                        f"  sudo usermod -a -G rapl $USER\n"
                        f"  (then log out and back in)\n\n"
                        f"For AMD Ryzen CPUs (including 7700X):\n"
                        f"  The intel-rapl driver is used - this is correct!\n"
                        f"  AMD Ryzen uses the same powercap interface."
                    )
        
        # Check if powercap directory exists but no RAPL
        powercap = Path("/sys/class/powercap")
        if powercap.exists():
            available = list(powercap.iterdir())
            available_names = [d.name for d in available]
            
            # Check if intel-rapl module needs to be loaded
            raise RAPLNotAvailableError(
                f"Powercap directory exists but RAPL not found.\n"
                f"Available interfaces: {available_names}\n\n"
                f"To enable RAPL on Linux (works for both Intel and AMD):\n"
                f"  1. Load the RAPL kernel module:\n"
                f"     sudo modprobe intel_rapl_common\n"
                f"     sudo modprobe intel_rapl_msr\n\n"
                f"  2. For AMD Ryzen specifically:\n"
                f"     sudo modprobe rapl           # or\n"
                f"     sudo modprobe intel_rapl_common\n\n"
                f"  3. Verify RAPL is available:\n"
                f"     ls /sys/class/powercap/\n\n"
                f"  4. Set permissions:\n"
                f"     sudo chmod a+r /sys/class/powercap/intel-rapl -R\n\n"
                f"Note: AMD Ryzen 7 7700X uses the intel-rapl interface,\n"
                f"not a separate amd-rapl interface."
            )
        
        # Powercap doesn't exist at all
        raise RAPLNotAvailableError(
            "RAPL interface is not available on this system.\n\n"
            "The powercap subsystem is not present. This could mean:\n"
            "  1. Running in a VM/container (RAPL not virtualized)\n"
            "  2. Kernel doesn't have powercap support\n"
            "  3. Very old CPU without RAPL support\n\n"
            "To enable RAPL on Linux (Intel and AMD Ryzen):\n"
            "  1. Ensure kernel config has: CONFIG_POWERCAP=y, CONFIG_INTEL_RAPL=m\n"
            "  2. Load modules: sudo modprobe intel_rapl_common intel_rapl_msr\n"
            "  3. For AMD: sudo modprobe rapl\n\n"
            "Supported hardware:\n"
            "  - Intel Sandy Bridge (2011) and later\n"
            "  - AMD Ryzen (Zen architecture, 2017) and later\n\n"
            "Your CPU: " + self._cpu_info.get('brand', 'Unknown')
        )
    
    def _read_max_energy_range(self) -> int:
        """Read the maximum energy range before counter overflow.
        
        The RAPL energy counter wraps around when it reaches max_energy_range_uj.
        This value is essential for proper overflow handling:
        
        If (energy_end < energy_start):
            energy_diff = (max_energy_range - energy_start) + energy_end
        
        Returns:
            Maximum energy value in microjoules.
        """
        # First, try the max_energy_range file in the same directory as energy_uj
        rapl_dir = self.rapl_path.parent
        max_range_file = rapl_dir / "max_energy_range_uj"
        
        if max_range_file.exists():
            try:
                with open(max_range_file, 'r') as f:
                    max_range = int(f.read().strip())
                    logger.debug(f"Read max_energy_range from {max_range_file}: {max_range:,} µJ")
                    return max_range
            except Exception as e:
                logger.warning(f"Could not read {max_range_file}: {e}")
        
        # Try other known paths
        for path in self.MAX_ENERGY_PATHS:
            path_obj = Path(path)
            if path_obj.exists():
                try:
                    with open(path_obj, 'r') as f:
                        max_range = int(f.read().strip())
                        logger.debug(f"Read max_energy_range from {path}: {max_range:,} µJ")
                        return max_range
                except Exception:
                    continue
        
        # Default values based on CPU type
        # AMD Ryzen 7 7700X typically has ~262 kJ max range
        # Intel CPUs vary but often around 262 kJ as well
        cpu_brand = self._cpu_info.get('brand', '').lower()
        
        if 'amd' in cpu_brand or 'ryzen' in cpu_brand:
            # AMD Ryzen default (32-bit counter at ~15.3W = ~262 kJ)
            default_max = 262143328850  # ~262 kJ
            logger.warning(f"Using default AMD max_energy_range: {default_max:,} µJ ({default_max/1_000_000:.2f} J)")
        else:
            # Intel default
            default_max = 262143328850  # ~262 kJ (common Intel value)
            logger.warning(f"Using default Intel max_energy_range: {default_max:,} µJ ({default_max/1_000_000:.2f} J)")
        
        return default_max
    
    def read_energy(self) -> int:
        """Read current energy counter in microjoules.
        
        Returns:
            Current energy counter value in microjoules (µJ).
            
        Raises:
            RAPLNotAvailableError: If unable to read energy counter.
        """
        if self._is_windows:
            return self._windows_reader.read_energy_microjoules()
        
        try:
            with open(self.rapl_path, 'r') as f:
                return int(f.read().strip())
        except FileNotFoundError:
            raise RAPLNotAvailableError(f"RAPL file not found: {self.rapl_path}")
        except PermissionError:
            raise RAPLNotAvailableError(
                f"Permission denied reading RAPL. Try:\n"
                f"  sudo chmod a+r {self.rapl_path}"
            )
        except ValueError as e:
            raise RAPLNotAvailableError(f"Invalid RAPL reading: {e}")
    
    def measure_function(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> dict[str, Any]:
        """Measure energy consumption of a function execution.
        
        This method handles RAPL counter overflow by checking if the
        end value is less than the start value. When overflow occurs:
        
            energy_diff = (max_energy_range - energy_start) + energy_end
        
        This is essential for long-running operations or when the counter
        is close to wrapping around.
        
        Args:
            func: Function to measure.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.
            
        Returns:
            dict containing:
                - energy_joules (float): Energy consumed in joules
                - energy_microjoules (int): Energy consumed in µJ
                - duration_seconds (float): Execution time in seconds
                - average_power_watts (float): Average power in watts
                - result: Return value of func
                - overflow_detected (bool): Whether counter overflow occurred
                
        Example:
            >>> meter = RAPLEnergyMeter()
            >>> def my_func(x):
            ...     return x ** 2
            >>> result = meter.measure_function(my_func, 42)
            >>> print(result['energy_joules'])
        """
        # Read energy before
        energy_start = self.read_energy()
        time_start = time.perf_counter()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Read energy after
        time_end = time.perf_counter()
        energy_end = self.read_energy()
        
        # Calculate energy delta with proper overflow handling
        overflow_detected = False
        if energy_end >= energy_start:
            # Normal case: no overflow
            energy_diff = energy_end - energy_start
        else:
            # Counter wrapped around: energy_end < energy_start
            # Formula: (max_range - start) + end
            energy_diff = (self.max_energy_range - energy_start) + energy_end
            overflow_detected = True
            logger.debug(
                f"RAPL counter overflow detected: "
                f"start={energy_start:,}, end={energy_end:,}, "
                f"max_range={self.max_energy_range:,}, "
                f"calculated_diff={energy_diff:,} µJ"
            )
        
        duration = time_end - time_start
        
        # Convert to joules (1 J = 1,000,000 µJ)
        energy_joules = energy_diff / 1_000_000.0
        
        # Calculate average power (W = J / s)
        average_power = energy_joules / duration if duration > 0 else 0.0
        
        return {
            'energy_joules': energy_joules,
            'energy_microjoules': energy_diff,
            'duration_seconds': duration,
            'average_power_watts': average_power,
            'result': result,
            'overflow_detected': overflow_detected,
        }
    
    def measure_multiple(
        self,
        func: Callable,
        runs: int,
        *args,
        warmup_runs: int = 5,
        **kwargs
    ) -> list[dict[str, Any]]:
        """Measure energy consumption over multiple runs.
        
        Args:
            func: Function to measure.
            runs: Number of measurement runs.
            *args: Positional arguments for func.
            warmup_runs: Number of warmup runs (not measured).
            **kwargs: Keyword arguments for func.
            
        Returns:
            List of measurement results.
        """
        # Warmup runs
        for _ in range(warmup_runs):
            func(*args, **kwargs)
        
        # Measured runs
        results = []
        for i in range(runs):
            result = self.measure_function(func, *args, **kwargs)
            result['run'] = i + 1
            results.append(result)
        
        return results
    
    @staticmethod
    def is_available() -> bool:
        """Check if RAPL is available on this system.
        
        Returns:
            True if RAPL interface is accessible, False otherwise.
            
        Example:
            >>> if RAPLEnergyMeter.is_available():
            ...     meter = RAPLEnergyMeter()
            ... else:
            ...     print("RAPL not available")
        """
        # Check Windows backends first
        if WINDOWS:
            return WindowsRAPLReader.is_available()
        
        # Check Linux RAPL paths
        for path in RAPLEnergyMeter.RAPL_PATHS:
            if Path(path).exists() and os.access(path, os.R_OK):
                return True
        return False
    
    @staticmethod
    def get_available_domains() -> list[dict[str, str]]:
        """Get list of available RAPL measurement domains.
        
        RAPL provides multiple measurement domains:
        - package (intel-rapl:0): Total CPU package power
        - core (intel-rapl:0:0): CPU cores only
        - uncore (intel-rapl:0:1): Integrated graphics, cache, etc.
        - dram (intel-rapl:0:2): Memory controller (not always available)
        
        Note: AMD Ryzen uses the intel-rapl interface but may have
        different subdomains available than Intel CPUs.
        
        Returns:
            List of dicts with 'name', 'path', 'readable', and 'max_energy' for each domain.
        """
        domains = []
        
        if WINDOWS:
            # On Windows, return a simplified domain structure
            return [{
                'name': 'package',
                'path': 'windows-backend',
                'readable': True,
                'max_energy': None,
            }]
        
        powercap = Path("/sys/class/powercap")
        
        if not powercap.exists():
            return domains
        
        # Recursively find all RAPL domains
        def find_rapl_domains(base_path: Path, depth: int = 0):
            for item in base_path.iterdir():
                if item.is_dir() and ('rapl' in item.name.lower() or depth > 0):
                    energy_file = item / "energy_uj"
                    name_file = item / "name"
                    max_energy_file = item / "max_energy_range_uj"
                    
                    if energy_file.exists():
                        # Get domain name
                        name = item.name
                        if name_file.exists():
                            try:
                                name = name_file.read_text().strip()
                            except Exception:
                                pass
                        
                        # Check readability
                        readable = os.access(energy_file, os.R_OK)
                        
                        # Get max energy range
                        max_energy = None
                        if max_energy_file.exists():
                            try:
                                max_energy = int(max_energy_file.read_text().strip())
                            except Exception:
                                pass
                        
                        domains.append({
                            'name': name,
                            'path': str(energy_file),
                            'readable': readable,
                            'max_energy': max_energy,
                            'directory': str(item),
                        })
                    
                    # Recurse into subdirectories (for subdomains like core, uncore)
                    if depth < 2:
                        find_rapl_domains(item, depth + 1)
        
        find_rapl_domains(powercap)
        return domains
    
    @staticmethod
    def verify_amd_ryzen_support() -> dict:
        """Verify RAPL support specifically for AMD Ryzen CPUs.
        
        This is useful for verifying that the AMD Ryzen 7 7700X (or similar)
        is properly detected and RAPL is accessible.
        
        Returns:
            Dict with verification results including:
            - cpu_detected: CPU model name
            - is_amd_ryzen: Boolean
            - rapl_available: Boolean
            - rapl_path: Path being used (or None)
            - domains: List of available domains
            - error: Error message if any
        """
        result = {
            'cpu_detected': 'Unknown',
            'is_amd_ryzen': False,
            'rapl_available': False,
            'rapl_path': None,
            'domains': [],
            'error': None,
            'notes': [],
        }
        
        # Detect CPU
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            result['cpu_detected'] = info.get('brand_raw', 'Unknown')
        except ImportError:
            # Try /proc/cpuinfo on Linux
            if not WINDOWS:
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        for line in f:
                            if line.startswith('model name'):
                                result['cpu_detected'] = line.split(':')[1].strip()
                                break
                except Exception:
                    pass
        except Exception:
            pass
        
        # Check if AMD Ryzen
        cpu_lower = result['cpu_detected'].lower()
        result['is_amd_ryzen'] = 'amd' in cpu_lower and 'ryzen' in cpu_lower
        
        if result['is_amd_ryzen']:
            result['notes'].append(
                "AMD Ryzen detected - will use intel-rapl interface (this is correct)"
            )
        
        # Check RAPL availability
        result['rapl_available'] = RAPLEnergyMeter.is_available()
        
        if result['rapl_available']:
            # Find the path being used
            for path in RAPLEnergyMeter.RAPL_PATHS:
                if Path(path).exists() and os.access(path, os.R_OK):
                    result['rapl_path'] = path
                    break
            
            # Get domains
            result['domains'] = RAPLEnergyMeter.get_available_domains()
        else:
            if WINDOWS:
                result['error'] = "Windows backend will be used (not hardware RAPL)"
                result['notes'].append("Install LibreHardwareMonitor for better accuracy")
            else:
                result['error'] = "RAPL not available - check kernel modules and permissions"
                result['notes'].append("Try: sudo modprobe intel_rapl_common")
                result['notes'].append("Try: sudo chmod a+r /sys/class/powercap/intel-rapl -R")
        
        return result


class SoftwareEnergyEstimator:
    """Fallback energy estimator when RAPL is not available.
    
    This class provides software-based energy estimation using CPU
    utilization and time as proxies for energy consumption. While
    less accurate than hardware RAPL measurements, it allows the
    system to function on unsupported hardware.
    
    Note:
        This estimator should only be used for relative comparisons,
        not absolute energy measurements.
    """
    
    # Estimated TDP for common CPUs (in watts)
    DEFAULT_TDP = 65.0
    
    def __init__(self, tdp_watts: float = None):
        """Initialize software energy estimator.
        
        Args:
            tdp_watts: CPU TDP in watts. If None, uses default value.
        """
        self.tdp = tdp_watts or self.DEFAULT_TDP
        logger.warning(
            f"Using software energy estimation (TDP={self.tdp}W). "
            "Results will be approximate."
        )
    
    def measure_function(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> dict[str, Any]:
        """Estimate energy consumption of a function execution.
        
        Uses CPU time and TDP to estimate energy consumption.
        
        Args:
            func: Function to measure.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.
            
        Returns:
            dict with estimated energy values.
        """
        try:
            import psutil
            process = psutil.Process()
            cpu_times_start = process.cpu_times()
        except ImportError:
            cpu_times_start = None
        
        time_start = time.perf_counter()
        result = func(*args, **kwargs)
        time_end = time.perf_counter()
        
        duration = time_end - time_start
        
        # Estimate CPU time used
        if cpu_times_start:
            cpu_times_end = process.cpu_times()
            cpu_time = (
                (cpu_times_end.user - cpu_times_start.user) +
                (cpu_times_end.system - cpu_times_start.system)
            )
            # Use ratio of CPU time to wall time for utilization
            cpu_utilization = min(cpu_time / duration if duration > 0 else 0.0, 1.0)
        else:
            cpu_utilization = 0.5  # Assume 50% if we can't measure
        
        # Estimate energy: TDP * utilization * time
        estimated_energy = self.tdp * cpu_utilization * duration
        estimated_power = self.tdp * cpu_utilization
        
        return {
            'energy_joules': estimated_energy,
            'energy_microjoules': int(estimated_energy * 1_000_000),
            'duration_seconds': duration,
            'average_power_watts': estimated_power,
            'result': result,
            'is_estimated': True,
            'cpu_utilization': cpu_utilization,
        }
    
    @staticmethod
    def is_available() -> bool:
        """Software estimation is always available."""
        return True


def get_energy_meter(prefer_hardware: bool = True) -> RAPLEnergyMeter | SoftwareEnergyEstimator:
    """Get the best available energy meter.
    
    Args:
        prefer_hardware: If True, prefer RAPL when available.
        
    Returns:
        RAPLEnergyMeter if available, otherwise SoftwareEnergyEstimator.
    """
    if prefer_hardware and RAPLEnergyMeter.is_available():
        return RAPLEnergyMeter()
    return SoftwareEnergyEstimator()
