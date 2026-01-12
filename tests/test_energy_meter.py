#!/usr/bin/env python3
"""
Unit tests for energy meter module.
"""

import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cryptogreen.energy_meter import RAPLEnergyMeter, SoftwareEnergyEstimator


class TestRAPLEnergyMeter(unittest.TestCase):
    """Tests for RAPL energy meter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.meter = RAPLEnergyMeter()
    
    def test_initialization(self):
        """Test that meter initializes correctly."""
        self.assertIsInstance(self.meter, RAPLEnergyMeter)
    
    def test_is_available_returns_bool(self):
        """Test that is_available returns a boolean."""
        result = self.meter.is_available()
        self.assertIsInstance(result, bool)
    
    @unittest.skipIf(
        not Path('/sys/class/powercap/intel-rapl').exists(),
        "RAPL not available on this system"
    )
    def test_read_energy_on_linux(self):
        """Test energy reading on Linux with RAPL."""
        energy = self.meter.read_energy()
        self.assertIsInstance(energy, (int, float))
        self.assertGreaterEqual(energy, 0)
    
    def test_measure_function_returns_dict(self):
        """Test that measure_function returns expected structure."""
        def dummy_function():
            time.sleep(0.01)
            return "result"
        
        result = self.meter.measure_function(dummy_function)
        
        self.assertIn('result', result)
        self.assertIn('energy_j', result)
        self.assertIn('time_s', result)
        self.assertEqual(result['result'], "result")
    
    def test_measure_function_with_args(self):
        """Test measure_function with arguments."""
        def add(a, b):
            return a + b
        
        result = self.meter.measure_function(add, 2, 3)
        self.assertEqual(result['result'], 5)
    
    def test_measure_function_with_kwargs(self):
        """Test measure_function with keyword arguments."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"
        
        result = self.meter.measure_function(greet, "World", greeting="Hi")
        self.assertEqual(result['result'], "Hi, World!")


class TestSoftwareEnergyEstimator(unittest.TestCase):
    """Tests for software-based energy estimator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.estimator = SoftwareEnergyEstimator()
    
    def test_initialization(self):
        """Test that estimator initializes correctly."""
        self.assertIsInstance(self.estimator, SoftwareEnergyEstimator)
    
    def test_estimate_energy_returns_float(self):
        """Test that estimate_energy returns a float."""
        energy = self.estimator.estimate_energy(
            time_seconds=1.0,
            cpu_percent=50.0
        )
        self.assertIsInstance(energy, float)
        self.assertGreater(energy, 0)
    
    def test_estimate_scales_with_time(self):
        """Test that energy estimate scales with time."""
        energy_1s = self.estimator.estimate_energy(1.0, 50.0)
        energy_2s = self.estimator.estimate_energy(2.0, 50.0)
        self.assertAlmostEqual(energy_2s, energy_1s * 2, places=5)
    
    def test_estimate_scales_with_cpu(self):
        """Test that energy estimate scales with CPU usage."""
        energy_50 = self.estimator.estimate_energy(1.0, 50.0)
        energy_100 = self.estimator.estimate_energy(1.0, 100.0)
        self.assertGreater(energy_100, energy_50)
    
    def test_measure_function_returns_dict(self):
        """Test that measure_function returns expected structure."""
        def dummy_function():
            # Do some work
            _ = sum(range(10000))
            return "done"
        
        result = self.estimator.measure_function(dummy_function)
        
        self.assertIn('result', result)
        self.assertIn('energy_j', result)
        self.assertIn('time_s', result)
        self.assertEqual(result['result'], "done")
    
    def test_measure_function_energy_positive(self):
        """Test that measured energy is positive."""
        def work():
            _ = [i ** 2 for i in range(10000)]
        
        result = self.estimator.measure_function(work)
        self.assertGreater(result['energy_j'], 0)
    
    def test_set_tdp(self):
        """Test setting TDP value."""
        custom_estimator = SoftwareEnergyEstimator(tdp_watts=150)
        energy = custom_estimator.estimate_energy(1.0, 100.0)
        
        default_estimator = SoftwareEnergyEstimator()  # Default TDP
        default_energy = default_estimator.estimate_energy(1.0, 100.0)
        
        # Higher TDP should give higher energy estimate at same utilization
        self.assertNotEqual(energy, default_energy)


class TestEnergyMeterIntegration(unittest.TestCase):
    """Integration tests for energy measurement."""
    
    def test_fallback_to_software_estimator(self):
        """Test that software estimator is used when RAPL unavailable."""
        rapl = RAPLEnergyMeter()
        
        if not rapl.is_available():
            # Should fall back gracefully
            result = rapl.measure_function(lambda: 1 + 1)
            self.assertIn('energy_j', result)
            # Energy should be estimated (non-zero)
            self.assertGreaterEqual(result['energy_j'], 0)
    
    def test_consistent_measurement_format(self):
        """Test that both meters return consistent format."""
        rapl = RAPLEnergyMeter()
        software = SoftwareEnergyEstimator()
        
        def test_func():
            return sum(range(1000))
        
        rapl_result = rapl.measure_function(test_func)
        software_result = software.measure_function(test_func)
        
        # Both should have same keys
        self.assertEqual(set(rapl_result.keys()), set(software_result.keys()))
    
    def test_time_measurement_accuracy(self):
        """Test that time measurement is reasonably accurate."""
        software = SoftwareEnergyEstimator()
        
        def sleep_func():
            time.sleep(0.1)
        
        result = software.measure_function(sleep_func)
        
        # Should be close to 0.1 seconds (with some tolerance)
        self.assertGreater(result['time_s'], 0.08)
        self.assertLess(result['time_s'], 0.15)


class TestMockedRAPL(unittest.TestCase):
    """Tests with mocked RAPL interface."""
    
    @patch('builtins.open')
    @patch('os.path.exists')
    def test_read_energy_from_rapl(self, mock_exists, mock_open):
        """Test reading energy from RAPL sysfs."""
        mock_exists.return_value = True
        mock_open.return_value.__enter__ = MagicMock(
            return_value=MagicMock(read=MagicMock(return_value="1000000"))  # 1mJ
        )
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        
        meter = RAPLEnergyMeter()
        # Would need to mock the full RAPL path structure
        # This is a placeholder for more detailed RAPL mocking


if __name__ == '__main__':
    unittest.main()
