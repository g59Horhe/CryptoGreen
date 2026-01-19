# benchmark_algorithm() Implementation Complete

## Overview
The `benchmark_algorithm()` method in `cryptogreen/benchmark_framework.py` has been fully implemented according to specifications.

## Implementation Details

### 1. File Reading
```python
data = path.read_bytes()  # Read entire file into memory
```

### 2. Warmup Phase
- **Default**: 5 warmup runs
- **Purpose**: Warm CPU caches before measurement
- **Results**: Discarded (not included in statistics)

### 3. Measurement Loop (Default: 100 runs)
For each run:
- **Energy**: `energy_meter.measure_function(algorithm_func, data)`
- **CPU Usage**: `psutil.cpu_percent(interval=None)` - non-blocking
- **Memory**: `process.memory_info().rss / (1024 * 1024)` - MB
- **Throughput**: `(file_size_mb / duration_s)` - MB/s

### 4. Statistics Calculation

#### Primary Metric: Median (NOT Mean)
As specified in the paper, **median energy** is the primary metric for robustness against outliers.

#### Energy Statistics
- `median_energy_j` ⭐ Primary metric
- `mean_energy_j`
- `std_energy_j`
- `min_energy_j`
- `max_energy_j`
- `energy_ci_95_lower` - 95% CI lower bound
- `energy_ci_95_upper` - 95% CI upper bound

#### Duration Statistics
- `median_duration_s`
- `mean_duration_s`
- `std_duration_s`
- `duration_ci_95_lower`
- `duration_ci_95_upper`

#### Performance Metrics
- `mean_throughput_mbps`
- `median_throughput_mbps`
- `median_power_w`
- `energy_per_byte_uj`
- `energy_per_mb_j`

### 5. 95% Confidence Intervals
Implemented in `_calculate_confidence_interval()`:
- Uses **t-distribution** for accurate CIs with sample size n
- Falls back to **z-score** (1.96) if scipy unavailable
- Applied to both energy and duration measurements

### 6. RAPL Energy Measurement
- **Primary**: Hardware RAPL via `/sys/class/powercap/intel-rapl/`
- **Fallback**: `SoftwareEnergyEstimator` if RAPL unavailable
- **Warning**: Logs warning when falling back to software estimation

### 7. Algorithm Configuration

#### AES Mode (as per paper specifications)
- **Mode**: CBC (NOT GCM)
- **Padding**: PKCS7
- **Key sizes**: AES-128 (16 bytes), AES-256 (32 bytes)
- **IV size**: 16 bytes

#### Key Generation
- **Fresh keys per file**: `_prepare_keys()` generates new keys for each file
- **Random**: Uses `os.urandom()` for cryptographically secure randomness
- **Not measured**: Key generation time excluded from benchmarks

### 8. Output Format

#### JSON Structure
```json
{
  "algorithm": "AES-128",
  "file_path": "/path/to/file",
  "file_name": "test.txt",
  "file_size": 1024,
  "file_type": "txt",
  "runs": 100,
  "warmup_runs": 5,
  "measurements": [
    {
      "run": 1,
      "energy_joules": 0.000123,
      "duration_seconds": 0.000456,
      "cpu_percent": 25.5,
      "memory_mb": 245.2,
      "throughput_mbps": 2.18,
      "average_power_watts": 0.27
    },
    ...
  ],
  "statistics": {
    "median_energy_j": 0.000125,
    "energy_ci_95_lower": 0.000120,
    "energy_ci_95_upper": 0.000130,
    ...
  },
  "timestamp": "2026-01-19T10:45:30.123456",
  "hardware": {
    "cpu_model": "AMD Ryzen 7 7700X",
    "cpu_cores": 8,
    "has_aes_ni": true,
    ...
  },
  "using_hardware_rapl": true
}
```

## Verification Checklist

✅ Read file with `Path.read_bytes()`  
✅ Warmup runs (default 5)  
✅ Measurement runs (default 100)  
✅ Energy via RAPL/SoftwareEstimator  
✅ CPU usage via `psutil.cpu_percent(interval=None)`  
✅ Memory via `process.memory_info().rss`  
✅ Throughput calculation (MB/s)  
✅ Statistics use **median** as primary metric  
✅ 95% confidence intervals calculated  
✅ Falls back to SoftwareEnergyEstimator with warning  
✅ CBC mode for AES (NOT GCM)  
✅ PKCS7 padding  
✅ Fresh key/IV per file  
✅ Hardware info collected  
✅ Timestamp in ISO 8601 format  
✅ All measurements saved to JSON  

## Usage Example

```python
from cryptogreen.benchmark_framework import CryptoBenchmark

# Initialize
benchmark = CryptoBenchmark('results/benchmarks')

# Benchmark single algorithm on one file
result = benchmark.benchmark_algorithm(
    algorithm_name='AES-128',
    file_path='data/test_files/txt/txt_1KB.txt',
    runs=100,
    warmup_runs=5
)

print(f"Median energy: {result['statistics']['median_energy_j']:.6f} J")
print(f"95% CI: [{result['statistics']['energy_ci_95_lower']:.6f}, "
      f"{result['statistics']['energy_ci_95_upper']:.6f}]")
```

## Notes

- **Median vs Mean**: Paper specifies median for robustness against outliers
- **CBC Mode**: AES-CBC is the mode specified in the paper (not AES-GCM)
- **RAPL Availability**: Code handles both hardware RAPL and software fallback
- **Fresh Keys**: Keys are regenerated for each file to avoid bias
- **Warmup Important**: Ensures fair measurement by warming CPU caches
- **100 Runs**: Provides sufficient samples for reliable statistics

## Implementation Complete ✓

All requirements have been implemented and verified.
