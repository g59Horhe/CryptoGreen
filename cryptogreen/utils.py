"""
Utility Functions Module

This module provides common utility functions used throughout the CryptoGreen
package, including file operations, formatting, and logging setup.

Example:
    >>> from cryptogreen.utils import format_bytes, setup_logging
    >>> setup_logging('DEBUG')
    >>> print(format_bytes(1024 * 1024))
    '1.00 MB'
"""

import logging
import os
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Configure logging for CryptoGreen.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        format_string: Optional custom format string.
        
    Returns:
        Configured root logger.
        
    Example:
        >>> logger = setup_logging('DEBUG', 'app.log')
        >>> logger.info("Application started")
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[]
    )
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes.
    
    Args:
        file_path: Path to file.
        
    Returns:
        File size in bytes.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    return Path(file_path).stat().st_size


def get_file_extension(file_path: Union[str, Path]) -> str:
    """Get file extension without dot, lowercase.
    
    Args:
        file_path: Path to file.
        
    Returns:
        Lowercase extension without dot.
        
    Example:
        >>> get_file_extension('/path/to/file.PDF')
        'pdf'
    """
    return Path(file_path).suffix.lstrip('.').lower()


def format_bytes(size_bytes: int, precision: int = 2) -> str:
    """Format byte size to human-readable string.
    
    Args:
        size_bytes: Size in bytes.
        precision: Decimal places.
        
    Returns:
        Human-readable size string.
        
    Example:
        >>> format_bytes(1536)
        '1.50 KB'
        >>> format_bytes(1048576)
        '1.00 MB'
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.{precision}f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.{precision}f} PB"


def format_duration(seconds: float, precision: int = 3) -> str:
    """Format duration to human-readable string.
    
    Args:
        seconds: Duration in seconds.
        precision: Decimal places for sub-second values.
        
    Returns:
        Human-readable duration string.
        
    Example:
        >>> format_duration(0.001234)
        '1.234 ms'
        >>> format_duration(65.5)
        '1m 5.500s'
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.{precision}f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.{precision}f} ms"
    elif seconds < 60:
        return f"{seconds:.{precision}f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.{precision}f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.{precision}f}s"


def format_energy(joules: float, precision: int = 6) -> str:
    """Format energy to human-readable string.
    
    Args:
        joules: Energy in joules.
        precision: Decimal places.
        
    Returns:
        Human-readable energy string.
        
    Example:
        >>> format_energy(0.000123)
        '123.000 µJ'
        >>> format_energy(1.5)
        '1.500000 J'
    """
    if joules < 0.001:
        return f"{joules * 1_000_000:.{precision}f} µJ"
    elif joules < 1.0:
        return f"{joules * 1000:.{precision}f} mJ"
    else:
        return f"{joules:.{precision}f} J"


def parse_size_string(size_str: str) -> int:
    """Parse size string to bytes.
    
    Args:
        size_str: Size string like '1KB', '10MB', '64B'.
        
    Returns:
        Size in bytes.
        
    Example:
        >>> parse_size_string('1KB')
        1024
        >>> parse_size_string('10MB')
        10485760
    """
    size_str = size_str.upper().strip()
    
    # Match number and unit
    match = re.match(r'^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)?$', size_str)
    if not match:
        raise ValueError(f"Invalid size string: {size_str}")
    
    value = float(match.group(1))
    unit = match.group(2) or 'B'
    
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
        'TB': 1024 ** 4,
    }
    
    return int(value * multipliers[unit])


def get_timestamp() -> str:
    """Get current timestamp in ISO 8601 format.
    
    Returns:
        Timestamp string.
        
    Example:
        >>> get_timestamp()
        '2025-01-11T14:30:00'
    """
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


def get_timestamp_filename() -> str:
    """Get current timestamp suitable for filenames.
    
    Returns:
        Timestamp string without special characters.
        
    Example:
        >>> get_timestamp_filename()
        '20250111_143000'
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating if necessary.
    
    Args:
        path: Path to directory.
        
    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cpu_info() -> dict:
    """Get CPU information.
    
    Returns:
        Dict with CPU model, cores, and features.
    """
    info = {
        'model': 'Unknown',
        'cores': os.cpu_count() or 1,
        'threads': os.cpu_count() or 1,
        'has_aes_ni': False,
        'has_arm_crypto': False,
        'architecture': platform.machine(),
    }
    
    # Try to read from /proc/cpuinfo on Linux
    cpuinfo_path = Path('/proc/cpuinfo')
    if cpuinfo_path.exists():
        try:
            content = cpuinfo_path.read_text()
            
            # Get model name
            model_match = re.search(r'model name\s*:\s*(.+)', content)
            if model_match:
                info['model'] = model_match.group(1).strip()
            
            # Check for AES-NI
            if 'aes' in content.lower():
                info['has_aes_ni'] = True
            
            # Check for ARM crypto
            if 'aes' in content.lower() and 'arm' in platform.machine().lower():
                info['has_arm_crypto'] = True
            
            # Count physical cores
            core_ids = re.findall(r'core id\s*:\s*(\d+)', content)
            if core_ids:
                info['cores'] = len(set(core_ids))
                info['threads'] = len(core_ids)
        except Exception:
            pass
    
    # Windows alternative
    if platform.system() == 'Windows':
        try:
            import wmi
            w = wmi.WMI()
            for processor in w.Win32_Processor():
                info['model'] = processor.Name
                info['cores'] = processor.NumberOfCores
                info['threads'] = processor.NumberOfLogicalProcessors
                break
        except ImportError:
            pass
    
    return info


def get_system_info() -> dict:
    """Get system information for benchmark reproducibility.
    
    Returns:
        Dict with system details.
    """
    info = {
        'os': platform.system(),
        'os_version': platform.version(),
        'os_release': platform.release(),
        'kernel': 'Unknown',
        'python_version': platform.python_version(),
        'cpu': get_cpu_info(),
        'ram_total_gb': 0.0,
    }
    
    # Get kernel version on Linux
    if platform.system() == 'Linux':
        try:
            result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
            info['kernel'] = result.stdout.strip()
        except Exception:
            pass
    
    # Get RAM info
    try:
        import psutil
        info['ram_total_gb'] = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    
    return info


def get_cpu_governor() -> str:
    """Get CPU frequency governor setting.
    
    Returns:
        Governor name or 'unknown'.
    """
    governor_path = Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')
    
    if governor_path.exists():
        try:
            return governor_path.read_text().strip()
        except Exception:
            pass
    
    return 'unknown'


def check_cpu_boost() -> Optional[bool]:
    """Check if CPU boost is enabled.
    
    Returns:
        True if enabled, False if disabled, None if unknown.
    """
    boost_path = Path('/sys/devices/system/cpu/cpufreq/boost')
    
    if boost_path.exists():
        try:
            value = boost_path.read_text().strip()
            return value == '1'
        except Exception:
            pass
    
    return None


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """Validate file path.
    
    Args:
        file_path: Path to validate.
        must_exist: Whether file must exist.
        
    Returns:
        Validated Path object.
        
    Raises:
        FileNotFoundError: If file doesn't exist and must_exist is True.
        ValueError: If path is invalid.
    """
    path = Path(file_path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if must_exist and not path.is_file():
        raise ValueError(f"Not a file: {path}")
    
    return path


def calculate_statistics(values: list[float]) -> dict:
    """Calculate basic statistics for a list of values.
    
    Args:
        values: List of numeric values.
        
    Returns:
        Dict with min, max, mean, median, std.
    """
    import statistics
    
    if not values:
        return {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'count': 0,
        }
    
    return {
        'min': min(values),
        'max': max(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
        'count': len(values),
    }


def estimate_benchmark_time(
    num_files: int,
    num_algorithms: int,
    runs_per_config: int,
    time_per_run_seconds: float = 0.5
) -> str:
    """Estimate total benchmark time.
    
    Args:
        num_files: Number of test files.
        num_algorithms: Number of algorithms to benchmark.
        runs_per_config: Repetitions per configuration.
        time_per_run_seconds: Estimated time per run.
        
    Returns:
        Human-readable time estimate.
    """
    total_operations = num_files * num_algorithms * runs_per_config
    total_seconds = total_operations * time_per_run_seconds
    
    return format_duration(total_seconds)


class ProgressTracker:
    """Track progress of long-running operations.
    
    Example:
        >>> tracker = ProgressTracker(total=100, description="Processing")
        >>> for i in range(100):
        ...     # Do work
        ...     tracker.update(1)
        >>> tracker.close()
    """
    
    def __init__(
        self,
        total: int,
        description: str = "",
        use_tqdm: bool = True
    ):
        """Initialize progress tracker.
        
        Args:
            total: Total number of items.
            description: Progress bar description.
            use_tqdm: Whether to use tqdm (falls back to simple logging).
        """
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = datetime.now()
        self._pbar = None
        
        if use_tqdm:
            try:
                from tqdm import tqdm
                self._pbar = tqdm(total=total, desc=description)
            except ImportError:
                pass
    
    def update(self, n: int = 1) -> None:
        """Update progress.
        
        Args:
            n: Number of items completed.
        """
        self.current += n
        
        if self._pbar:
            self._pbar.update(n)
        else:
            # Simple progress logging
            if self.current % max(1, self.total // 10) == 0:
                percent = (self.current / self.total) * 100
                elapsed = (datetime.now() - self.start_time).total_seconds()
                eta = (elapsed / self.current) * (self.total - self.current) if self.current > 0 else 0
                logging.info(f"{self.description}: {percent:.0f}% ({self.current}/{self.total}) ETA: {format_duration(eta)}")
    
    def close(self) -> None:
        """Close progress tracker."""
        if self._pbar:
            self._pbar.close()
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logging.info(f"{self.description}: Completed in {format_duration(elapsed)}")
