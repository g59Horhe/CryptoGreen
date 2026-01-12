#!/usr/bin/env python3
"""
Integration tests for CryptoGreen.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cryptogreen import (
    CryptoAlgorithms,
    CryptoBenchmark,
    FeatureExtractor,
    HybridSelector,
    RuleBasedSelector,
    MLSelector,
)
from cryptogreen.energy_meter import RAPLEnergyMeter, SoftwareEnergyEstimator


class TestEndToEndEncryption(unittest.TestCase):
    """End-to-end encryption workflow tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.selector = HybridSelector()
        self.crypto = CryptoAlgorithms()
        
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test file
        self.test_file = Path(self.temp_dir) / "test_document.txt"
        with open(self.test_file, 'w') as f:
            f.write("This is a test document for end-to-end encryption testing.\n" * 100)
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_select_and_encrypt_workflow(self):
        """Test selecting an algorithm and encrypting a file."""
        # Step 1: Get algorithm recommendation
        recommendation = self.selector.select_algorithm(str(self.test_file))
        algorithm = recommendation['algorithm']
        
        # Step 2: Read file
        with open(self.test_file, 'rb') as f:
            data = f.read()
        
        # Step 3: Encrypt based on recommendation
        if algorithm == 'AES-128':
            ciphertext, metadata = self.crypto.aes_128_encrypt(data)
            decrypted = self.crypto.aes_128_decrypt(
                ciphertext, metadata['key'], metadata['iv']
            )
        elif algorithm == 'AES-256':
            ciphertext, metadata = self.crypto.aes_256_encrypt(data)
            decrypted = self.crypto.aes_256_decrypt(
                ciphertext, metadata['key'], metadata['iv']
            )
        elif algorithm == 'ChaCha20':
            ciphertext, metadata = self.crypto.chacha20_encrypt(data)
            decrypted = self.crypto.chacha20_decrypt(
                ciphertext, metadata['key'], metadata['nonce']
            )
        else:
            self.skipTest(f"Asymmetric algorithm {algorithm} not suitable for file encryption")
        
        # Verify roundtrip
        self.assertEqual(decrypted, data)
    
    def test_batch_file_processing(self):
        """Test processing multiple files."""
        # Create multiple files
        files = []
        for i, (name, size) in enumerate([
            ('small.txt', 100),
            ('medium.txt', 10000),
            ('large.bin', 100000),
        ]):
            path = Path(self.temp_dir) / name
            if name.endswith('.bin'):
                with open(path, 'wb') as f:
                    f.write(os.urandom(size))
            else:
                with open(path, 'w') as f:
                    f.write("x" * size)
            files.append(str(path))
        
        # Batch select
        results = self.selector.batch_select(files)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('algorithm', result)
            self.assertIn('confidence', result)


class TestBenchmarkIntegration(unittest.TestCase):
    """Integration tests for benchmarking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmark = CryptoBenchmark()
        
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "benchmark_test.txt"
        with open(self.test_file, 'w') as f:
            f.write("Benchmark test data " * 500)
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_benchmark_single_algorithm(self):
        """Test benchmarking a single algorithm."""
        result = self.benchmark.benchmark_algorithm(
            'AES-128',
            str(self.test_file),
            runs=3
        )
        
        self.assertIn('algorithm', result)
        self.assertIn('statistics', result)
        self.assertIn('median_time_s', result['statistics'])
        self.assertIn('median_energy_j', result['statistics'])
    
    def test_benchmark_all_symmetric(self):
        """Test benchmarking all symmetric algorithms."""
        algorithms = ['AES-128', 'AES-256', 'ChaCha20']
        results = []
        
        for alg in algorithms:
            result = self.benchmark.benchmark_algorithm(
                alg,
                str(self.test_file),
                runs=3
            )
            results.append(result)
        
        self.assertEqual(len(results), 3)
        
        # Should be able to compare results
        throughputs = [r['statistics']['throughput_mbps'] for r in results]
        self.assertEqual(len(throughputs), 3)
    
    def test_benchmark_provides_efficiency_metric(self):
        """Test that benchmark provides efficiency (MB/J)."""
        result = self.benchmark.benchmark_algorithm(
            'AES-256',
            str(self.test_file),
            runs=3
        )
        
        self.assertIn('efficiency_mb_per_j', result['statistics'])
        self.assertGreater(result['statistics']['efficiency_mb_per_j'], 0)


class TestFeatureExtractionIntegration(unittest.TestCase):
    """Integration tests for feature extraction pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()
        self.selector = RuleBasedSelector()
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_features_used_in_selection(self):
        """Test that extracted features are used in selection."""
        # Create files with different characteristics
        low_entropy_file = Path(self.temp_dir) / "low_entropy.txt"
        with open(low_entropy_file, 'w') as f:
            f.write("a" * 10000)  # Very low entropy
        
        high_entropy_file = Path(self.temp_dir) / "high_entropy.bin"
        with open(high_entropy_file, 'wb') as f:
            f.write(os.urandom(10000))  # High entropy
        
        # Extract features
        low_features = self.extractor.extract_features(str(low_entropy_file))
        high_features = self.extractor.extract_features(str(high_entropy_file))
        
        # Verify entropy difference
        self.assertLess(low_features['entropy'], high_features['entropy'])
        
        # Both should produce valid selections
        low_result = self.selector.select_algorithm(str(low_entropy_file))
        high_result = self.selector.select_algorithm(str(high_entropy_file))
        
        self.assertIsNotNone(low_result['algorithm'])
        self.assertIsNotNone(high_result['algorithm'])
    
    def test_file_type_detection(self):
        """Test that file type is detected correctly."""
        # Create different file types
        txt_file = Path(self.temp_dir) / "document.txt"
        jpg_file = Path(self.temp_dir) / "image.jpg"
        
        with open(txt_file, 'w') as f:
            f.write("Text document content")
        
        # Create minimal valid JPEG header
        with open(jpg_file, 'wb') as f:
            f.write(b'\xFF\xD8\xFF\xE0' + os.urandom(1000))
        
        txt_features = self.extractor.extract_features(str(txt_file))
        jpg_features = self.extractor.extract_features(str(jpg_file))
        
        # Should have different file type encodings
        self.assertIn('file_type_encoded', txt_features)
        self.assertIn('file_type_encoded', jpg_features)


class TestEnergyMeasurementIntegration(unittest.TestCase):
    """Integration tests for energy measurement."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rapl = RAPLEnergyMeter()
        self.software = SoftwareEnergyEstimator()
        self.crypto = CryptoAlgorithms()
    
    def test_measure_encryption_energy(self):
        """Test measuring energy of actual encryption."""
        data = os.urandom(100000)  # 100 KB
        
        def encrypt():
            return self.crypto.aes_256_encrypt(data)
        
        # Use software estimator (works on all platforms)
        result = self.software.measure_function(encrypt)
        
        self.assertIn('energy_j', result)
        self.assertIn('time_s', result)
        self.assertGreater(result['energy_j'], 0)
        self.assertGreater(result['time_s'], 0)
    
    def test_compare_algorithm_energy(self):
        """Test comparing energy of different algorithms."""
        data = os.urandom(50000)  # 50 KB
        
        results = {}
        
        for name, func in [
            ('AES-128', lambda: self.crypto.aes_128_encrypt(data)),
            ('AES-256', lambda: self.crypto.aes_256_encrypt(data)),
            ('ChaCha20', lambda: self.crypto.chacha20_encrypt(data)),
        ]:
            measurement = self.software.measure_function(func)
            results[name] = measurement['energy_j']
        
        # All should have positive energy
        for alg, energy in results.items():
            self.assertGreater(energy, 0, f"{alg} should have positive energy")


class TestSelectorPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete selector pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rule_selector = RuleBasedSelector()
        self.ml_selector = MLSelector()
        self.hybrid_selector = HybridSelector()
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_all_selectors_handle_same_file(self):
        """Test that all selectors can handle the same input."""
        test_file = Path(self.temp_dir) / "unified_test.txt"
        with open(test_file, 'w') as f:
            f.write("Unified test content " * 500)
        
        rule_result = self.rule_selector.select_algorithm(str(test_file))
        ml_result = self.ml_selector.select_algorithm(str(test_file))
        hybrid_result = self.hybrid_selector.select_algorithm(str(test_file))
        
        # All should return valid results
        valid_algs = ['AES-128', 'AES-256', 'ChaCha20', 'RSA-2048', 'RSA-4096', 'ECC-256']
        
        self.assertIn(rule_result['algorithm'], valid_algs)
        self.assertIn(ml_result['algorithm'], valid_algs)
        self.assertIn(hybrid_result['algorithm'], valid_algs)
    
    def test_security_level_propagates(self):
        """Test that security level is respected by all selectors."""
        test_file = Path(self.temp_dir) / "security_test.txt"
        with open(test_file, 'w') as f:
            f.write("Security test content " * 500)
        
        # High security should select AES-256
        rule_result = self.rule_selector.select_algorithm(
            str(test_file), security_level='high'
        )
        hybrid_result = self.hybrid_selector.select_algorithm(
            str(test_file), security_level='high'
        )
        
        self.assertEqual(rule_result['algorithm'], 'AES-256')
        self.assertEqual(hybrid_result['algorithm'], 'AES-256')


class TestConfigurationIntegration(unittest.TestCase):
    """Integration tests for configuration handling."""
    
    def test_custom_model_path(self):
        """Test using custom model path."""
        custom_path = "custom/model/path.pkl"
        selector = HybridSelector(model_path=custom_path)
        
        # Should initialize without error
        self.assertIsInstance(selector, HybridSelector)
    
    def test_hardware_detection_consistent(self):
        """Test that hardware detection is consistent."""
        extractor1 = FeatureExtractor()
        extractor2 = FeatureExtractor()
        
        caps1 = extractor1.detect_hardware_capabilities()
        caps2 = extractor2.detect_hardware_capabilities()
        
        self.assertEqual(caps1['cpu_cores'], caps2['cpu_cores'])
        self.assertEqual(caps1['has_aes_ni'], caps2['has_aes_ni'])


if __name__ == '__main__':
    unittest.main()
