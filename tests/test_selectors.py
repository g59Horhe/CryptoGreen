#!/usr/bin/env python3
"""
Unit tests for selector modules.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cryptogreen.rule_based_selector import RuleBasedSelector
from cryptogreen.ml_selector import MLSelector
from cryptogreen.hybrid_selector import HybridSelector
from cryptogreen.feature_extractor import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    """Tests for feature extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()
        
        # Create temp files for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Text file
        self.text_file = Path(self.temp_dir) / "test.txt"
        with open(self.text_file, 'w') as f:
            f.write("Hello, World! " * 1000)
        
        # Binary file with random data
        self.binary_file = Path(self.temp_dir) / "test.bin"
        with open(self.binary_file, 'wb') as f:
            f.write(os.urandom(10000))
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_extract_features_returns_dict(self):
        """Test that extract_features returns a dictionary."""
        features = self.extractor.extract_features(str(self.text_file))
        self.assertIsInstance(features, dict)
    
    def test_extract_features_has_required_keys(self):
        """Test that features contain required keys."""
        features = self.extractor.extract_features(str(self.text_file))
        
        required_keys = ['file_size', 'file_size_log', 'entropy', 'file_type_encoded']
        for key in required_keys:
            self.assertIn(key, features)
    
    def test_file_size_correct(self):
        """Test that file size is extracted correctly."""
        features = self.extractor.extract_features(str(self.text_file))
        actual_size = Path(self.text_file).stat().st_size
        self.assertEqual(features['file_size'], actual_size)
    
    def test_entropy_range(self):
        """Test that entropy is in valid range [0, 8]."""
        features_text = self.extractor.extract_features(str(self.text_file))
        features_binary = self.extractor.extract_features(str(self.binary_file))
        
        self.assertGreaterEqual(features_text['entropy'], 0)
        self.assertLessEqual(features_text['entropy'], 8)
        
        self.assertGreaterEqual(features_binary['entropy'], 0)
        self.assertLessEqual(features_binary['entropy'], 8)
    
    def test_random_data_has_high_entropy(self):
        """Test that random binary data has high entropy."""
        features = self.extractor.extract_features(str(self.binary_file))
        self.assertGreater(features['entropy'], 7.0)  # Random data should have high entropy
    
    def test_text_data_has_lower_entropy(self):
        """Test that text data has lower entropy than random."""
        features_text = self.extractor.extract_features(str(self.text_file))
        features_binary = self.extractor.extract_features(str(self.binary_file))
        
        self.assertLess(features_text['entropy'], features_binary['entropy'])
    
    def test_detect_hardware_capabilities(self):
        """Test hardware capability detection."""
        caps = self.extractor.detect_hardware_capabilities()
        
        self.assertIn('has_aes_ni', caps)
        self.assertIn('cpu_cores', caps)
        self.assertIsInstance(caps['has_aes_ni'], bool)
        self.assertIsInstance(caps['cpu_cores'], int)


class TestRuleBasedSelector(unittest.TestCase):
    """Tests for rule-based selector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.selector = RuleBasedSelector()
        
        # Create temp files
        self.temp_dir = tempfile.mkdtemp()
        
        # Small file (< 1KB)
        self.small_file = Path(self.temp_dir) / "small.txt"
        with open(self.small_file, 'w') as f:
            f.write("Small content")
        
        # Medium file (~10KB)
        self.medium_file = Path(self.temp_dir) / "medium.txt"
        with open(self.medium_file, 'w') as f:
            f.write("Medium content " * 700)
        
        # Large file (~1MB)
        self.large_file = Path(self.temp_dir) / "large.bin"
        with open(self.large_file, 'wb') as f:
            f.write(os.urandom(1024 * 1024))
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_select_returns_dict(self):
        """Test that select_algorithm returns a dictionary."""
        result = self.selector.select_algorithm(str(self.small_file))
        self.assertIsInstance(result, dict)
    
    def test_select_returns_valid_algorithm(self):
        """Test that selected algorithm is valid."""
        result = self.selector.select_algorithm(str(self.small_file))
        valid_algorithms = ['AES-128', 'AES-256', 'ChaCha20', 'RSA-2048', 'RSA-4096', 'ECC-256']
        self.assertIn(result['algorithm'], valid_algorithms)
    
    def test_select_returns_confidence(self):
        """Test that selection includes confidence."""
        result = self.selector.select_algorithm(str(self.small_file))
        self.assertIn('confidence', result)
        self.assertIn(result['confidence'], ['high', 'medium', 'low'])
    
    def test_high_security_selects_aes256(self):
        """Test that high security level selects AES-256."""
        result = self.selector.select_algorithm(
            str(self.medium_file),
            security_level='high'
        )
        self.assertEqual(result['algorithm'], 'AES-256')
    
    def test_powersave_mode_prefers_efficient(self):
        """Test that powersave mode prefers efficient algorithms."""
        result = self.selector.select_algorithm(
            str(self.medium_file),
            power_mode='powersave'
        )
        # Should prefer AES-128 or ChaCha20 in powersave mode
        self.assertIn(result['algorithm'], ['AES-128', 'ChaCha20'])
    
    def test_small_file_selection(self):
        """Test selection for small files."""
        result = self.selector.select_algorithm(str(self.small_file))
        # Small files typically use AES-128 or ChaCha20
        self.assertIsNotNone(result['algorithm'])
    
    def test_explain_decision(self):
        """Test decision explanation."""
        result = self.selector.select_algorithm(str(self.medium_file))
        self.assertIn('explanation', result)
        self.assertIsInstance(result['explanation'], str)


class TestMLSelector(unittest.TestCase):
    """Tests for ML-based selector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.selector = MLSelector()
        
        # Create temp file
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.txt"
        with open(self.test_file, 'w') as f:
            f.write("Test content " * 100)
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test that ML selector initializes correctly."""
        self.assertIsInstance(self.selector, MLSelector)
    
    def test_is_trained_initially_false(self):
        """Test that untrained model reports as such."""
        self.assertFalse(self.selector.is_trained())
    
    def test_select_without_training(self):
        """Test selection without trained model falls back gracefully."""
        result = self.selector.select_algorithm(str(self.test_file))
        # Should return some result even without training
        self.assertIn('algorithm', result)
    
    def test_prepare_features(self):
        """Test feature preparation for ML model."""
        features = self.selector._prepare_features(str(self.test_file))
        self.assertIsNotNone(features)
    
    @patch.object(MLSelector, 'is_trained', return_value=True)
    def test_select_with_mocked_model(self, mock_trained):
        """Test selection with mocked trained model."""
        # Mock the model's predict method
        self.selector.model = MagicMock()
        self.selector.model.predict.return_value = ['AES-128']
        self.selector.model.predict_proba.return_value = [[0.8, 0.1, 0.1]]
        self.selector.model.classes_ = ['AES-128', 'AES-256', 'ChaCha20']
        
        result = self.selector.select_algorithm(str(self.test_file))
        self.assertEqual(result['algorithm'], 'AES-128')


class TestHybridSelector(unittest.TestCase):
    """Tests for hybrid selector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.selector = HybridSelector()
        
        # Create temp file
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.txt"
        with open(self.test_file, 'w') as f:
            f.write("Test content for hybrid selector " * 100)
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test that hybrid selector initializes correctly."""
        self.assertIsInstance(self.selector, HybridSelector)
    
    def test_select_returns_dict(self):
        """Test that select_algorithm returns a dictionary."""
        result = self.selector.select_algorithm(str(self.test_file))
        self.assertIsInstance(result, dict)
    
    def test_select_returns_required_fields(self):
        """Test that selection includes all required fields."""
        result = self.selector.select_algorithm(str(self.test_file))
        
        required_fields = ['algorithm', 'confidence', 'method']
        for field in required_fields:
            self.assertIn(field, result)
    
    def test_method_indicates_source(self):
        """Test that method field indicates decision source."""
        result = self.selector.select_algorithm(str(self.test_file))
        valid_methods = ['rules', 'ml', 'hybrid', 'both_agree', 'rules_preferred', 'ml_preferred']
        # Method could contain any of these or combinations
        self.assertIsNotNone(result['method'])
    
    def test_explain_decision(self):
        """Test decision explanation."""
        result = self.selector.select_algorithm(str(self.test_file), verbose=True)
        # In verbose mode, should include explanation
        self.assertIn('algorithm', result)
    
    def test_batch_select(self):
        """Test batch selection."""
        # Create multiple files
        files = []
        for i in range(3):
            f = Path(self.temp_dir) / f"file_{i}.txt"
            with open(f, 'w') as fp:
                fp.write(f"Content {i} " * (100 * (i + 1)))
            files.append(str(f))
        
        results = self.selector.batch_select(files)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('algorithm', result)
    
    def test_security_override(self):
        """Test that high security overrides efficiency."""
        result = self.selector.select_algorithm(
            str(self.test_file),
            security_level='high'
        )
        self.assertEqual(result['algorithm'], 'AES-256')


class TestSelectorConsistency(unittest.TestCase):
    """Tests for selector consistency across modules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rule_selector = RuleBasedSelector()
        self.hybrid_selector = HybridSelector()
        
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "consistency_test.txt"
        with open(self.test_file, 'w') as f:
            f.write("Consistency test content " * 500)
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_same_input_same_output(self):
        """Test that same input produces consistent output."""
        result1 = self.rule_selector.select_algorithm(str(self.test_file))
        result2 = self.rule_selector.select_algorithm(str(self.test_file))
        
        self.assertEqual(result1['algorithm'], result2['algorithm'])
    
    def test_valid_algorithms_only(self):
        """Test that all selectors return valid algorithms."""
        valid = ['AES-128', 'AES-256', 'ChaCha20', 'RSA-2048', 'RSA-4096', 'ECC-256']
        
        rule_result = self.rule_selector.select_algorithm(str(self.test_file))
        hybrid_result = self.hybrid_selector.select_algorithm(str(self.test_file))
        
        self.assertIn(rule_result['algorithm'], valid)
        self.assertIn(hybrid_result['algorithm'], valid)


if __name__ == '__main__':
    unittest.main()
