# CRYPTOGREEN PROJECT IMPLEMENTATION SPECIFICATION
# Read this file completely before generating any code
# This is the authoritative reference for the entire implementation

## PROJECT OVERVIEW
**Project Name:** CryptoGreen  
**Goal:** Intelligent energy-efficient cryptographic algorithm selector using ML + rule-based hybrid approach  
**Hardware:** AMD Ryzen 7 7700X with RAPL energy measurement support  
**Target:** 85%+ selection accuracy, 35%+ energy savings vs AES-256 baseline  
**Timeline:** 7 weeks remaining (Weeks 8-14)  

## CRITICAL REQUIREMENTS
1. **Energy Measurement:** Use hardware RAPL (AMD Ryzen support via /sys/class/powercap/intel-rapl/)
2. **Algorithms:** AES-128, AES-256, ChaCha20, RSA-2048, RSA-4096, ECC-256
3. **File Types:** txt, jpg, png, mp4, pdf, zip, sql (6 types)
4. **File Sizes:** 64B, 1KB, 10KB, 100KB, 1MB, 10MB, 100MB (7 sizes)
5. **Runs per config:** 100 repetitions minimum
6. **Target measurements:** 15,000+ total

---

## PROJECT STRUCTURE

```
cryptogreen/
├── .copilot-instructions.md          # THIS FILE - read by Copilot
├── README.md                          # User documentation
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installer
├── .gitignore                        # Git ignore file
│
├── cryptogreen/                      # Main package
│   ├── __init__.py                   # Package exports
│   ├── algorithms.py                 # Crypto algorithm wrappers
│   ├── energy_meter.py              # RAPL energy measurement
│   ├── benchmark_framework.py       # Benchmarking engine
│   ├── feature_extractor.py         # File feature extraction
│   ├── rule_based_selector.py       # Rule-based selector
│   ├── ml_selector.py               # ML-based selector  
│   ├── hybrid_selector.py           # Combined selector
│   └── utils.py                     # Utility functions
│
├── scripts/                          # Utility scripts
│   ├── generate_test_data.py        # Create test files
│   ├── run_benchmarks.py            # Execute benchmark suite
│   ├── analyze_results.py           # Statistical analysis
│   ├── train_model.py               # Train ML model
│   ├── evaluate_accuracy.py         # Test selector accuracy
│   └── real_world_evaluation.py     # Real-world file testing
│
├── data/                            # Data directory
│   ├── test_files/                  # Generated test files
│   │   ├── txt/                     # Text files by size
│   │   ├── jpg/                     # JPEG images by size
│   │   ├── png/                     # PNG images by size
│   │   ├── mp4/                     # Video files by size
│   │   ├── pdf/                     # PDF files by size
│   │   ├── zip/                     # Compressed files by size
│   │   └── sql/                     # Database dumps by size
│   ├── real_world_files/            # Real files for validation
│   └── ml_data/                     # Training data
│       ├── features.csv             # Extracted features
│       └── labels.csv               # Optimal algorithms
│
├── results/                         # Results directory
│   ├── benchmarks/                  # Benchmark results
│   │   ├── raw/                     # Raw measurement data (JSON)
│   │   └── processed/               # Analyzed results (CSV)
│   ├── models/                      # Trained ML models
│   │   └── selector_model.pkl       # Random Forest model
│   ├── figures/                     # Generated plots
│   └── logs/                        # Execution logs
│
├── tests/                           # Unit tests
│   ├── test_algorithms.py
│   ├── test_energy_meter.py
│   ├── test_selectors.py
│   └── test_integration.py
│
└── cryptogreen_cli.py              # Command-line interface

```

---

## DETAILED MODULE SPECIFICATIONS

### 1. cryptogreen/energy_meter.py
**Purpose:** Hardware energy measurement using AMD RAPL

**Class:** `RAPLEnergyMeter`

**Methods:**
```python
class RAPLEnergyMeter:
    """Measures CPU energy consumption via RAPL interface"""
    
    def __init__(self):
        """Initialize RAPL energy meter
        - Detect RAPL path: /sys/class/powercap/intel-rapl/intel-rapl:0
        - Verify read permissions
        - Read max_energy_range_uj for overflow handling
        """
        pass
    
    def read_energy(self) -> int:
        """Read current energy counter in microjoules
        Returns:
            int: Current energy counter value (microjoules)
        """
        pass
    
    def measure_function(self, func, *args, **kwargs) -> dict:
        """Measure energy consumption of a function execution
        
        Args:
            func: Function to measure
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            dict: {
                'energy_joules': float,
                'energy_microjoules': int,
                'duration_seconds': float,
                'average_power_watts': float,
                'result': Any  # Function return value
            }
        
        Handles:
            - Counter overflow (check if end < start)
            - Sub-millisecond timing precision
        """
        pass
    
    @staticmethod
    def is_available() -> bool:
        """Check if RAPL is available on this system"""
        pass
```

**Implementation Notes:**
- RAPL path: `/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj`
- Max range path: `/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj`
- Handle counter wraparound: if energy_diff < 0, add max_energy_range_uj
- Use `time.perf_counter()` for high-resolution timing
- Raise clear exception if RAPL unavailable with setup instructions

**Error Handling:**
```python
class RAPLNotAvailableError(Exception):
    """Raised when RAPL interface is not available"""
    pass
```

---

### 2. cryptogreen/algorithms.py
**Purpose:** Wrappers for all cryptographic algorithms

**Class:** `CryptoAlgorithms`

**Methods:**
```python
class CryptoAlgorithms:
    """Unified interface for all cryptographic algorithms"""
    
    @staticmethod
    def aes_128_encrypt(data: bytes, key: bytes = None, iv: bytes = None) -> bytes:
        """Encrypt data with AES-128 CBC mode
        
        Args:
            data: Plaintext bytes
            key: 16-byte key (generated if None)
            iv: 16-byte IV (generated if None)
            
        Returns:
            bytes: Ciphertext (PKCS7 padded)
            
        Uses:
            - cryptography.hazmat.primitives.ciphers
            - AES algorithm with 128-bit key
            - CBC mode with PKCS7 padding
        """
        pass
    
    @staticmethod
    def aes_256_encrypt(data: bytes, key: bytes = None, iv: bytes = None) -> bytes:
        """Encrypt data with AES-256 CBC mode
        
        Args:
            data: Plaintext bytes
            key: 32-byte key (generated if None)
            iv: 16-byte IV (generated if None)
            
        Returns:
            bytes: Ciphertext (PKCS7 padded)
        """
        pass
    
    @staticmethod
    def chacha20_encrypt(data: bytes, key: bytes = None, nonce: bytes = None) -> bytes:
        """Encrypt data with ChaCha20 stream cipher
        
        Args:
            data: Plaintext bytes
            key: 32-byte key (generated if None)
            nonce: 12-byte nonce (generated if None)
            
        Returns:
            bytes: Ciphertext
            
        Uses:
            - Crypto.Cipher.ChaCha20 from pycryptodome
        """
        pass
    
    @staticmethod
    def rsa_2048_encrypt_key(key_data: bytes) -> bytes:
        """Simulate RSA-2048 key exchange encryption
        
        Args:
            key_data: Symmetric key to encrypt (max 190 bytes for OAEP)
            
        Returns:
            bytes: Encrypted key
            
        Uses:
            - cryptography.hazmat.primitives.asymmetric.rsa
            - OAEP padding with SHA256
            - Generate new 2048-bit key pair
            
        Note: Only encrypts small key data, not full files
        """
        pass
    
    @staticmethod
    def rsa_4096_encrypt_key(key_data: bytes) -> bytes:
        """Simulate RSA-4096 key exchange encryption"""
        pass
    
    @staticmethod
    def ecc_256_sign(data: bytes) -> bytes:
        """Sign data with ECC NIST P-256 (secp256r1)
        
        Args:
            data: Data to sign
            
        Returns:
            bytes: Signature
            
        Uses:
            - cryptography.hazmat.primitives.asymmetric.ec
            - NIST P-256 curve (SECP256R1)
            - ECDSA with SHA256
        """
        pass
    
    @staticmethod
    def get_algorithm_names() -> list[str]:
        """Return list of all supported algorithm names"""
        return ['AES-128', 'AES-256', 'ChaCha20', 'RSA-2048', 'RSA-4096', 'ECC-256']
```

**Security Requirements:**
- Use cryptographically secure random key generation (os.urandom)
- Never reuse keys or IVs in examples
- Use standard padding schemes (PKCS7 for block ciphers, OAEP for RSA)
- Document which library provides each algorithm

---

### 3. cryptogreen/benchmark_framework.py
**Purpose:** Execute comprehensive benchmarking

**Class:** `CryptoBenchmark`

**Methods:**
```python
class CryptoBenchmark:
    """Framework for benchmarking cryptographic algorithms"""
    
    def __init__(self, output_dir: str = 'results/benchmarks'):
        """Initialize benchmark framework
        
        Args:
            output_dir: Directory to save results
        """
        self.energy_meter = RAPLEnergyMeter()
        self.output_dir = Path(output_dir)
        self.results = []
        
    def benchmark_algorithm(
        self,
        algorithm_name: str,
        file_path: str,
        runs: int = 100
    ) -> dict:
        """Benchmark a single algorithm on a file
        
        Args:
            algorithm_name: One of ['AES-128', 'AES-256', 'ChaCha20', ...]
            file_path: Path to file to encrypt
            runs: Number of repetitions
            
        Returns:
            dict: {
                'algorithm': str,
                'file_path': str,
                'file_size': int,
                'file_type': str,
                'runs': int,
                'measurements': [
                    {
                        'run': int,
                        'energy_joules': float,
                        'duration_seconds': float,
                        'cpu_percent': float,
                        'memory_mb': float,
                        'throughput_mbps': float
                    },
                    ...
                ],
                'statistics': {
                    'median_energy_j': float,
                    'mean_energy_j': float,
                    'std_energy_j': float,
                    'min_energy_j': float,
                    'max_energy_j': float,
                    'median_duration_s': float,
                    'mean_throughput_mbps': float
                },
                'timestamp': str (ISO 8601),
                'hardware': {
                    'cpu_model': str,
                    'cpu_cores': int,
                    'has_aes_ni': bool,
                    'kernel': str,
                    'cpu_governor': str
                }
            }
        
        Process:
            1. Read file data
            2. Warm-up: Run algorithm 5 times (discard results)
            3. For each run:
               - Get CPU usage (psutil)
               - Measure energy with energy_meter
               - Get memory usage
               - Calculate throughput
            4. Calculate statistics (median preferred over mean)
            5. Return results dict
        """
        pass
    
    def run_full_benchmark(
        self,
        test_files_dir: str = 'data/test_files',
        runs: int = 100,
        algorithms: list = None
    ) -> None:
        """Run complete benchmark suite
        
        Args:
            test_files_dir: Root directory containing test files
            runs: Repetitions per configuration
            algorithms: List of algorithms (None = all)
            
        Process:
            1. Discover all test files in test_files_dir
            2. For each algorithm:
               For each file:
                   benchmark_algorithm(algorithm, file, runs)
                   Save incremental results (JSON)
            3. Generate summary CSV
            4. Log progress and estimated time remaining
        
        Saves:
            - results/benchmarks/raw/benchmark_TIMESTAMP.json
            - results/benchmarks/processed/summary_TIMESTAMP.csv
            - results/benchmarks/logs/benchmark_TIMESTAMP.log
        """
        pass
    
    def save_results(self, filename: str = None) -> None:
        """Save benchmark results to JSON file"""
        pass
    
    @staticmethod
    def get_hardware_info() -> dict:
        """Collect hardware information for reproducibility
        
        Returns:
            dict: {
                'cpu_model': str,  # From /proc/cpuinfo
                'cpu_cores': int,
                'cpu_threads': int,
                'has_aes_ni': bool,  # Check CPU flags
                'has_arm_crypto': bool,
                'ram_total_gb': float,
                'kernel': str,  # uname -r
                'os': str,
                'cpu_governor': str  # Check /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
            }
        """
        pass
```

**Implementation Notes:**
- Use `psutil` for CPU and memory monitoring
- Check CPU flags for AES-NI: `grep -m 1 aes /proc/cpuinfo`
- Calculate throughput: `(file_size_bytes / (1024*1024)) / duration_seconds`
- Use median for energy (robust to outliers)
- Save results incrementally (don't lose hours of benchmarking!)
- Add progress bar with `tqdm`

---

### 4. cryptogreen/feature_extractor.py
**Purpose:** Extract features from files for ML model

**Class:** `FeatureExtractor`

**Methods:**
```python
class FeatureExtractor:
    """Extract features from files for algorithm selection"""
    
    @staticmethod
    def extract_features(file_path: str) -> dict:
        """Extract all features from a file
        
        Args:
            file_path: Path to file
            
        Returns:
            dict: {
                'file_size_bytes': int,
                'file_size_log': float,  # log10(file_size)
                'file_type': str,  # Extension without dot
                'file_type_encoded': int,  # Numeric encoding
                'entropy': float,  # Shannon entropy (0-8)
                'entropy_quartile_25': float,
                'entropy_quartile_75': float,
                'has_aes_ni': bool,
                'has_arm_crypto': bool,
                'cpu_cores': int
            }
        """
        pass
    
    @staticmethod
    def calculate_entropy(file_path: str, sample_size: int = 10240) -> tuple[float, float, float]:
        """Calculate Shannon entropy of file
        
        Args:
            file_path: Path to file
            sample_size: Bytes to sample (10KB default)
            
        Returns:
            tuple: (entropy, quartile_25, quartile_75)
            
        Algorithm:
            1. Read first sample_size bytes (or entire file if smaller)
            2. Calculate byte frequency
            3. Apply Shannon entropy formula: H = -Σ p(x) * log2(p(x))
            4. Calculate quartiles from byte distribution
            
        Notes:
            - Entropy range: 0 (all same byte) to 8 (random data)
            - Encrypted/compressed data: entropy > 7.2
            - Text files: entropy ~ 4-6
            - Images (uncompressed): entropy ~ 6-8
        """
        pass
    
    @staticmethod
    def detect_hardware_capabilities() -> dict:
        """Detect CPU hardware capabilities
        
        Returns:
            dict: {
                'has_aes_ni': bool,
                'has_arm_crypto': bool,
                'cpu_cores': int,
                'cpu_model': str
            }
            
        Detection:
            - AES-NI: Check 'aes' in /proc/cpuinfo flags
            - ARM Crypto: Check 'aes' in /proc/cpuinfo features (ARM)
            - CPU cores: psutil.cpu_count(logical=False)
        """
        pass
    
    @staticmethod
    def encode_file_type(file_type: str) -> int:
        """Encode file type as integer
        
        Mapping:
            txt -> 0
            jpg -> 1
            png -> 2
            mp4 -> 3
            pdf -> 4
            zip -> 5
            sql -> 6
        """
        pass
```

**Implementation Notes:**
- For large files (>10KB), sample first 10KB for entropy
- Handle missing/empty files gracefully
- Normalize file type (lowercase, remove dot)
- Cache hardware capabilities (detect once per session)

---

### 5. cryptogreen/rule_based_selector.py
**Purpose:** Rule-based algorithm selection

**Class:** `RuleBasedSelector`

**Methods:**
```python
class RuleBasedSelector:
    """Select algorithm using rule-based decision tree"""
    
    def __init__(self, benchmark_data: dict = None):
        """Initialize rule-based selector
        
        Args:
            benchmark_data: Optional benchmark results for rule tuning
        """
        self.hardware = FeatureExtractor.detect_hardware_capabilities()
        self.benchmark_data = benchmark_data
        
    def select_algorithm(
        self,
        file_path: str,
        security_level: str = 'medium',
        power_mode: str = 'plugged'
    ) -> dict:
        """Select optimal algorithm using rules
        
        Args:
            file_path: Path to file to encrypt
            security_level: 'low', 'medium', or 'high'
            power_mode: 'plugged' or 'battery'
            
        Returns:
            dict: {
                'algorithm': str,
                'confidence': str,  # 'high', 'medium', 'low'
                'rationale': str,  # Human-readable explanation
                'estimated_energy_j': float,  # From benchmark data
                'estimated_duration_s': float,
                'features_used': dict,
                'decision_path': list[str]  # Rules applied
            }
        
        Decision Tree:
            IF security_level == 'high':
                IF has_aes_ni:
                    RETURN 'AES-256' (confidence: high)
                ELSE:
                    RETURN 'ChaCha20' (confidence: high)
            
            ELIF file_size < 100KB:
                IF has_aes_ni:
                    RETURN 'AES-128' (confidence: high)
                ELSE:
                    RETURN 'ChaCha20' (confidence: high)
            
            ELIF entropy > 7.5:  # Already compressed/encrypted
                RETURN 'ChaCha20' (confidence: medium)
            
            ELIF file_type in ['mp4', 'zip', 'jpg']:  # Already compressed
                RETURN 'ChaCha20' (confidence: medium)
            
            ELIF file_type in ['txt', 'sql', 'pdf']:  # Compressible text
                IF has_aes_ni:
                    RETURN 'AES-128' (confidence: high)
                ELSE:
                    RETURN 'ChaCha20' (confidence: medium)
            
            ELSE:  # Default case
                IF has_aes_ni:
                    RETURN 'AES-128' (confidence: medium)
                ELSE:
                    RETURN 'ChaCha20' (confidence: medium)
        """
        pass
    
    def _estimate_performance(
        self,
        algorithm: str,
        file_size: int,
        file_type: str
    ) -> tuple[float, float]:
        """Estimate energy and time from benchmark data
        
        Returns:
            tuple: (estimated_energy_j, estimated_duration_s)
        """
        pass
```

**Implementation Notes:**
- Track decision path for explainability
- Use benchmark data for performance estimates if available
- Rationale should be clear and actionable
- Consider power_mode: prefer lower-power algorithms on battery

---

### 6. cryptogreen/ml_selector.py
**Purpose:** Machine learning-based algorithm selection

**Class:** `MLSelector`

**Methods:**
```python
class MLSelector:
    """Select algorithm using trained Random Forest model"""
    
    def __init__(self, model_path: str = 'results/models/selector_model.pkl'):
        """Initialize ML selector
        
        Args:
            model_path: Path to trained model file
        """
        self.model = self._load_model(model_path)
        self.feature_names = [
            'file_size_log',
            'file_type_encoded',
            'entropy',
            'entropy_quartile_25',
            'entropy_quartile_75',
            'has_aes_ni',
            'cpu_cores'
        ]
        
    def select_algorithm(self, file_path: str) -> dict:
        """Select optimal algorithm using ML model
        
        Args:
            file_path: Path to file to encrypt
            
        Returns:
            dict: {
                'algorithm': str,
                'confidence': float,  # 0.0-1.0
                'probabilities': dict,  # {algorithm: probability}
                'features': dict,  # Features used
                'feature_importance': dict,  # {feature: importance}
                'alternatives': list[str]  # Top-3 algorithms
            }
        """
        pass
    
    def train_model(
        self,
        features_csv: str = 'data/ml_data/features.csv',
        labels_csv: str = 'data/ml_data/labels.csv',
        output_path: str = 'results/models/selector_model.pkl'
    ) -> dict:
        """Train Random Forest model
        
        Args:
            features_csv: Path to features CSV
            labels_csv: Path to labels CSV
            output_path: Where to save trained model
            
        Returns:
            dict: {
                'accuracy': float,
                'top2_accuracy': float,
                'cv_scores': list[float],  # 5-fold CV
                'confusion_matrix': array,
                'feature_importance': dict
            }
        
        Model Configuration:
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        
        Evaluation:
            - 5-fold stratified cross-validation
            - Calculate top-1 and top-2 accuracy
            - Generate confusion matrix
            - Extract feature importances
        """
        pass
    
    def _load_model(self, model_path: str):
        """Load trained model from pickle file"""
        pass
    
    def _prepare_features(self, features: dict) -> np.ndarray:
        """Convert feature dict to model input array"""
        pass
```

**Implementation Notes:**
- Use scikit-learn's RandomForestClassifier
- Save model with joblib (better for large numpy arrays)
- Handle missing model file gracefully
- Normalize features if needed (StandardScaler)
- Return top-3 predictions for alternatives

---

### 7. cryptogreen/hybrid_selector.py
**Purpose:** Combine rule-based and ML selectors

**Class:** `HybridSelector`

**Methods:**
```python
class HybridSelector:
    """Intelligent selector combining rules and ML"""
    
    def __init__(
        self,
        model_path: str = 'results/models/selector_model.pkl',
        benchmark_data: dict = None
    ):
        """Initialize hybrid selector"""
        self.rule_selector = RuleBasedSelector(benchmark_data)
        self.ml_selector = MLSelector(model_path)
        
    def select_algorithm(
        self,
        file_path: str,
        security_level: str = 'medium',
        power_mode: str = 'plugged',
        verbose: bool = False
    ) -> dict:
        """Select optimal algorithm using hybrid approach
        
        Args:
            file_path: Path to file to encrypt
            security_level: 'low', 'medium', or 'high'
            power_mode: 'plugged' or 'battery'
            verbose: Print decision process
            
        Returns:
            dict: {
                'algorithm': str,
                'confidence': str,  # 'high', 'medium', 'low'
                'method': str,  # 'both_agree', 'ml_preferred', 'rules_preferred'
                'rationale': str,
                'estimated_energy_j': float,
                'estimated_duration_s': float,
                'rule_recommendation': dict,
                'ml_recommendation': dict
            }
        
        Decision Logic:
            1. Get rule-based recommendation
            2. Get ML recommendation
            
            3. IF both agree AND ml_confidence > 0.8:
                   RETURN agreed algorithm (confidence: high, method: both_agree)
            
            4. ELIF security_level == 'high':
                   RETURN rule-based (confidence: high, method: rules_preferred)
                   # Prioritize proven security
            
            5. ELIF ml_confidence > 0.8:
                   RETURN ML (confidence: high, method: ml_preferred)
                   # Trust data-driven decision
            
            6. ELIF ml_confidence > 0.6:
                   RETURN ML (confidence: medium, method: ml_preferred)
            
            7. ELSE:
                   RETURN rule-based (confidence: medium, method: rules_preferred)
                   # Default to interpretable rules
        """
        pass
    
    def explain_decision(self, result: dict) -> str:
        """Generate human-readable explanation of decision
        
        Returns:
            str: Multi-line explanation with:
                - Selected algorithm
                - Why it was chosen
                - Expected energy/time
                - What each selector recommended
                - Confidence level and reasoning
        """
        pass
```

**Implementation Notes:**
- Log disagreements between selectors for analysis
- Provide clear explanation of hybrid decision
- Allow overriding security level
- Consider power_mode in decision weighting

---

### 8. scripts/generate_test_data.py
**Purpose:** Generate synthetic test files

**Main Function:**
```python
def generate_test_data(output_dir: str = 'data/test_files'):
    """Generate all test files for benchmarking
    
    Generates:
        - 6 file types (txt, jpg, png, mp4, pdf, zip, sql)
        - 7 sizes each (64B, 1KB, 10KB, 100KB, 1MB, 10MB, 100MB)
        - Total: 42 files
    
    File Generation:
        txt: Random ASCII text (lorem ipsum style)
        jpg: Solid color JPEG (use PIL)
        png: Solid color PNG (use PIL)
        mp4: Black frame video (use opencv or ffmpeg)
        pdf: Text PDF (use reportlab)
        zip: Compressed random data (use zipfile)
        sql: SQL INSERT statements (random data)
    
    Naming:
        {type}_{size}.{ext}
        Examples: txt_1KB.txt, jpg_10MB.jpg
    """
    pass
```

**Implementation Notes:**
- Use appropriate library for each file type
- Ensure exact file sizes (pad/trim as needed)
- Create directory structure automatically
- Log creation progress
- Verify files after creation

---

### 9. scripts/run_benchmarks.py
**Purpose:** Execute full benchmark suite

**Main Script:**
```python
#!/usr/bin/env python3
"""
Run complete benchmark suite

Usage:
    python scripts/run_benchmarks.py [OPTIONS]

Options:
    --runs N           Number of repetitions (default: 100)
    --algorithms LIST  Comma-separated algorithms (default: all)
    --output DIR       Output directory (default: results/benchmarks)
    --resume          Resume from last checkpoint
    --test-mode       Quick test with 10 runs
"""

import argparse
from cryptogreen.benchmark_framework import CryptoBenchmark

def main():
    parser = argparse.ArgumentParser(description='Run CryptoGreen benchmarks')
    parser.add_argument('--runs', type=int, default=100, help='Repetitions per config')
    parser.add_argument('--algorithms', type=str, default='all', help='Algorithms to test')
    parser.add_argument('--output', type=str, default='results/benchmarks', help='Output directory')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--test-mode', action='store_true', help='Quick test (10 runs)')
    
    args = parser.parse_args()
    
    # Setup
    if args.test_mode:
        args.runs = 10
        print("TEST MODE: Running with 10 repetitions only")
    
    # Parse algorithms
    if args.algorithms == 'all':
        algorithms = None  # Use all
    else:
        algorithms = [a.strip() for a in args.algorithms.split(',')]
    
    # Check RAPL availability
    from cryptogreen.energy_meter import RAPLEnergyMeter
    if not RAPLEnergyMeter.is_available():
        print("ERROR: RAPL not available. Run setup instructions.")
        sys.exit(1)
    
    # Run benchmarks
    benchmark = CryptoBenchmark(output_dir=args.output)
    
    print(f"Starting benchmark suite:")
    print(f"  Runs per config: {args.runs}")
    print(f"  Algorithms: {algorithms or 'all'}")
    print(f"  Output: {args.output}")
    print(f"  Estimated time: {estimate_time(args.runs)} hours")
    print()
    
    benchmark.run_full_benchmark(
        test_files_dir='data/test_files',
        runs=args.runs,
        algorithms=algorithms
    )
    
    print("Benchmark complete!")
    print(f"Results saved to: {args.output}")

if __name__ == '__main__':
    main()
```

---

### 10. scripts/train_model.py
**Purpose:** Train ML model on benchmark results

**Main Script:**
```python
#!/usr/bin/env python3
"""
Train ML model for algorithm selection

Usage:
    python scripts/train_model.py [OPTIONS]

Options:
    --benchmark-results PATH  Path to benchmark results JSON
    --output PATH             Where to save model (default: results/models/selector_model.pkl)
    --test-split FLOAT        Test set fraction (default: 0.2)
"""

def prepare_training_data(benchmark_results: dict) -> tuple:
    """Convert benchmark results to training data
    
    Args:
        benchmark_results: Loaded benchmark JSON
        
    Returns:
        tuple: (features_df, labels_series)
        
    Process:
        1. For each benchmark result:
           - Extract features (file_size_log, entropy, etc.)
           - Determine optimal algorithm (lowest median energy)
        2. Create pandas DataFrame
        3. Handle class imbalance (if needed)
    """
    pass

def main():
    # Parse arguments
    # Load benchmark results
    # Prepare training data
    # Initialize MLSelector
    # Train model
    # Evaluate model
    # Save model
    # Print results
    pass
```

---

## DEPENDENCIES (requirements.txt)

```
# Core cryptographic libraries
cryptography==42.0.0
pycryptodome==3.19.0

# Machine learning
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
scipy==1.11.3
joblib==1.3.2

# System monitoring
psutil==5.9.6

# Plotting
matplotlib==3.8.2
seaborn==0.13.0

# Image generation (for test data)
Pillow==10.1.0
reportlab==4.0.7

# Progress bars
tqdm==4.66.1

# Testing
pytest==7.4.3
pytest-cov==4.1.0

# CLI
click==8.1.7
```

---

## SETUP INSTRUCTIONS

### System Setup
```bash
# 1. Enable RAPL access
sudo modprobe msr
sudo chmod -R a+r /dev/cpu/*/msr
sudo chmod -R a+r /sys/class/powercap/intel-rapl/

# 2. Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 3. Disable CPU boost for consistency (optional)
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost

# 4. Install dependencies
pip install -r requirements.txt
```

### Quick Start
```bash
# 1. Generate test data
python scripts/generate_test_data.py

# 2. Run quick test (10 runs)
python scripts/run_benchmarks.py --test-mode

# 3. Verify results
ls -lh results/benchmarks/

# 4. Train model (after full benchmarks)
python scripts/train_model.py --benchmark-results results/benchmarks/raw/benchmark_*.json

# 5. Test hybrid selector
python -c "
from cryptogreen.hybrid_selector import HybridSelector
selector = HybridSelector()
result = selector.select_algorithm('data/test_files/txt/txt_1MB.txt', verbose=True)
print(result)
"
```

---

## TESTING REQUIREMENTS

### Unit Tests
```python
# tests/test_energy_meter.py
def test_rapl_available():
    """Test RAPL availability detection"""
    pass

def test_energy_measurement():
    """Test energy measurement on dummy function"""
    pass

# tests/test_algorithms.py
def test_aes_128_encrypt():
    """Test AES-128 encryption produces valid output"""
    pass

def test_all_algorithms_work():
    """Test all algorithms can encrypt/decrypt"""
    pass

# tests/test_selectors.py
def test_rule_selector_high_security():
    """Test rule selector chooses AES-256 for high security"""
    pass

def test_ml_selector_predicts():
    """Test ML selector returns prediction"""
    pass

def test_hybrid_selector_agreement():
    """Test hybrid selector when both agree"""
    pass
```

### Integration Test
```python
def test_full_pipeline():
    """Test complete pipeline: generate data -> benchmark -> train -> select"""
    pass
```

---

## CODING STANDARDS

### Style
- Follow PEP 8
- Use type hints for all function signatures
- Docstrings in Google style
- Maximum line length: 100 characters

### Error Handling
- Raise specific exceptions with clear messages
- Log errors with context
- Never silently fail

### Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

### Performance
- Use generators for large datasets
- Cache expensive computations
- Profile critical paths
- Minimize file I/O

---

## EXPECTED OUTPUT FILES

After running complete pipeline:

```
results/
├── benchmarks/
│   ├── raw/
│   │   └── benchmark_20250112_143022.json  # Raw measurements
│   ├── processed/
│   │   └── summary_20250112_143022.csv     # Statistical summary
│   └── logs/
│       └── benchmark_20250112_143022.log   # Execution log
│
├── models/
│   └── selector_model.pkl                  # Trained Random Forest
│
└── figures/
    ├── energy_vs_size.png                  # Line plot
    ├── energy_heatmap.png                  # Algorithm × file type
    ├── throughput_comparison.png           # Box plots
    ├── feature_importance.png              # Bar chart
    └── confusion_matrix.png                # ML accuracy
```

---

## BENCHMARKING TIPS

1. **Time Estimation:** 
   - ~0.5 seconds per run (with warmup)
   - 42 files × 6 algorithms × 100 runs = 25,200 operations
   - Estimated total: 3-4 hours

2. **System Stability:**
   - Close all other applications
   - Disable background updates
   - Monitor CPU temperature
   - Run overnight if possible

3. **Incremental Progress:**
   - Save results after each file
   - Log progress frequently
   - Support resume from checkpoint

4. **Validation:**
   - Sanity check: AES with AES-NI should be faster than without
   - ChaCha20 should be competitive on non-AES-NI systems
   - Energy should scale linearly with file size

---

## SUCCESS CRITERIA

✅ **Benchmark Phase:**
- 15,000+ measurements collected
- <5% measurement error (validate with repeated runs)
- Clear energy differences between algorithms (statistical significance p < 0.05)
- Results reproducible within 10% variance

✅ **ML Model:**
- >85% accuracy on test set
- >95% top-2 accuracy
- Feature importance makes intuitive sense
- Model generalizes to unseen file types

✅ **Hybrid Selector:**
- Rule and ML selectors agree >70% of time
- Hybrid selector achieves >85% accuracy
- Decisions explainable in plain language

✅ **Energy Savings:**
- >35% average savings vs AES-256 baseline
- Savings consistent across file types
- Overhead <1% of encryption time

---

## COMMON PITFALLS TO AVOID

❌ **Don't:**
- Forget to handle RAPL counter overflow
- Use mean instead of median for timing data
- Skip warm-up runs before measurement
- Generate keys inside measurement loop
- Ignore CPU frequency scaling
- Run benchmarks with other programs active

✅ **Do:**
- Check RAPL availability before starting
- Save results incrementally (don't lose progress!)
- Log everything for debugging
- Validate measurements make sense
- Document hardware configuration
- Use proper statistical tests

---

## TROUBLESHOOTING

### RAPL Not Available
```bash
# Check if RAPL exists
ls /sys/class/powercap/intel-rapl/

# Load MSR module
sudo modprobe msr

# Check kernel support
dmesg | grep -i rapl
```

### Inconsistent Results
```bash
# Fix CPU frequency
sudo cpupower frequency-set -g performance

# Check background processes
top

# Monitor CPU temperature
watch -n 1 sensors
```

### Model Accuracy Too Low
- Check class balance in training data
- Try different features (add file header analysis)
- Tune hyperparameters (GridSearchCV)
- Collect more diverse training data

---

## FINAL NOTES FOR COPILOT

When generating code:
1. **Always** include proper error handling
2. **Always** add logging statements
3. **Always** include docstrings
4. **Always** use type hints
5. **Always** validate inputs
6. **Never** hardcode paths (use arguments/config)
7. **Never** ignore errors silently
8. **Prefer** explicit over implicit
9. **Prefer** readability over cleverness
10. **Test** incrementally as you build

This specification is comprehensive. Follow it closely and the implementation will succeed.

Good luck! 🚀
