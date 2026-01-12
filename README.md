# CryptoGreen 🌿

**Intelligent Energy-Efficient Cryptographic Algorithm Selector**

CryptoGreen uses a hybrid ML + rule-based approach to automatically select the most energy-efficient cryptographic algorithm based on file characteristics and hardware capabilities.

## 🎯 Project Goals

- **85%+ selection accuracy** for choosing optimal encryption algorithm
- **35%+ energy savings** compared to AES-256 baseline
- Support for AMD Ryzen 7 7700X with RAPL energy measurement

## 🔧 Supported Algorithms

| Algorithm | Type | Key Size | Best For |
|-----------|------|----------|----------|
| AES-128 | Symmetric | 128-bit | Small files with AES-NI |
| AES-256 | Symmetric | 256-bit | High security requirements |
| ChaCha20 | Stream | 256-bit | Systems without AES-NI |
| RSA-2048 | Asymmetric | 2048-bit | Key exchange |
| RSA-4096 | Asymmetric | 4096-bit | High-security key exchange |
| ECC-256 | Asymmetric | 256-bit | Signing/verification |

## 📁 Supported File Types

- **Text:** `.txt`, `.sql`
- **Images:** `.jpg`, `.png`
- **Video:** `.mp4`
- **Documents:** `.pdf`
- **Archives:** `.zip`

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Linux with RAPL support (AMD Ryzen 7 7700X recommended)
- Root access for RAPL energy measurement

### Installation

```bash
# Clone the repository
git clone https://github.com/cryptogreen/cryptogreen.git
cd cryptogreen

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### System Setup (Linux)

```bash
# Enable RAPL access
sudo modprobe msr
sudo chmod -R a+r /dev/cpu/*/msr
sudo chmod -R a+r /sys/class/powercap/intel-rapl/

# Set CPU governor to performance (recommended for benchmarking)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Optional: Disable CPU boost for consistent measurements
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
```

### Usage

#### Generate Test Data

```bash
python scripts/generate_test_data.py
```

#### Run Benchmarks

```bash
# Quick test (10 runs per configuration)
python scripts/run_benchmarks.py --test-mode

# Full benchmark (100 runs per configuration)
python scripts/run_benchmarks.py --runs 100
```

#### Train ML Model

```bash
python scripts/train_model.py --benchmark-results results/benchmarks/raw/benchmark_*.json
```

#### Use the Selector

```python
from cryptogreen.hybrid_selector import HybridSelector

selector = HybridSelector()
result = selector.select_algorithm(
    file_path='myfile.pdf',
    security_level='medium',
    power_mode='plugged'
)

print(f"Recommended: {result['algorithm']}")
print(f"Confidence: {result['confidence']}")
print(f"Rationale: {result['rationale']}")
```

#### Command Line Interface

```bash
# Select algorithm for a file
cryptogreen select myfile.pdf --security medium

# Run benchmark
cryptogreen benchmark --runs 100

# Analyze results
cryptogreen analyze results/benchmarks/raw/benchmark_*.json
```

## 📊 Project Structure

```
cryptogreen/
├── cryptogreen/              # Main package
│   ├── algorithms.py         # Crypto algorithm wrappers
│   ├── energy_meter.py       # RAPL energy measurement
│   ├── benchmark_framework.py # Benchmarking engine
│   ├── feature_extractor.py  # File feature extraction
│   ├── rule_based_selector.py # Rule-based selector
│   ├── ml_selector.py        # ML-based selector
│   └── hybrid_selector.py    # Combined selector
│
├── scripts/                  # Utility scripts
│   ├── generate_test_data.py
│   ├── run_benchmarks.py
│   ├── train_model.py
│   └── analyze_results.py
│
├── data/                     # Data directory
│   ├── test_files/          # Generated test files
│   └── ml_data/             # Training data
│
├── results/                  # Results directory
│   ├── benchmarks/          # Benchmark results
│   ├── models/              # Trained ML models
│   └── figures/             # Generated plots
│
└── tests/                    # Unit tests
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cryptogreen --cov-report=html

# Run specific test
pytest tests/test_algorithms.py -v
```

## 📈 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Selection Accuracy | >85% | ⏳ |
| Top-2 Accuracy | >95% | ⏳ |
| Energy Savings | >35% | ⏳ |
| Measurement Count | 15,000+ | ⏳ |

## 🔬 Methodology

### Energy Measurement

CryptoGreen uses hardware RAPL (Running Average Power Limit) to measure CPU energy consumption:

1. Read RAPL energy counter before operation
2. Execute cryptographic operation
3. Read RAPL energy counter after operation
4. Calculate energy delta in microjoules

### Algorithm Selection

The hybrid selector combines:

1. **Rule-Based Selection:** Decision tree based on:
   - File size and type
   - Hardware capabilities (AES-NI)
   - Security requirements
   - Power mode (battery vs plugged)

2. **ML-Based Selection:** Random Forest classifier trained on:
   - File entropy
   - File size (log-scaled)
   - File type encoding
   - Hardware features

### Decision Logic

```
IF both selectors agree AND ML confidence > 0.8:
    RETURN agreed algorithm (high confidence)
ELIF security_level == 'high':
    RETURN rule-based selection
ELIF ML confidence > 0.8:
    RETURN ML selection
ELSE:
    RETURN rule-based selection (default)
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting a pull request.

## 📚 References

- [RAPL Documentation](https://www.kernel.org/doc/html/latest/power/powercap/powercap.html)
- [Intel RAPL Power Capping](https://01.org/blogs/2014/running-average-power-limit-%E2%80%93-rapl)
- [ChaCha20 RFC 7539](https://tools.ietf.org/html/rfc7539)
