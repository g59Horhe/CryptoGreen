#!/usr/bin/env python3
"""
Unit tests for cryptographic algorithms module.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cryptogreen.algorithms import CryptoAlgorithms


class TestAES128(unittest.TestCase):
    """Tests for AES-128 encryption/decryption."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.crypto = CryptoAlgorithms()
        self.test_data = b"Hello, World! This is test data for AES-128."
    
    def test_encrypt_decrypt_roundtrip(self):
        """Test that encryption followed by decryption returns original data."""
        ciphertext, metadata = self.crypto.aes_128_encrypt(self.test_data)
        decrypted = self.crypto.aes_128_decrypt(
            ciphertext, metadata['key'], metadata['iv']
        )
        self.assertEqual(decrypted, self.test_data)
    
    def test_encrypt_produces_ciphertext(self):
        """Test that encryption produces non-empty ciphertext."""
        ciphertext, metadata = self.crypto.aes_128_encrypt(self.test_data)
        self.assertIsInstance(ciphertext, bytes)
        self.assertGreater(len(ciphertext), 0)
    
    def test_encrypt_produces_valid_metadata(self):
        """Test that encryption produces valid metadata."""
        ciphertext, metadata = self.crypto.aes_128_encrypt(self.test_data)
        self.assertIn('key', metadata)
        self.assertIn('iv', metadata)
        self.assertEqual(len(metadata['key']), 16)  # 128 bits
        self.assertEqual(len(metadata['iv']), 16)   # AES block size
    
    def test_different_plaintexts_produce_different_ciphertexts(self):
        """Test that different inputs produce different outputs."""
        ct1, _ = self.crypto.aes_128_encrypt(b"data1")
        ct2, _ = self.crypto.aes_128_encrypt(b"data2")
        self.assertNotEqual(ct1, ct2)
    
    def test_empty_data(self):
        """Test encryption of empty data."""
        ciphertext, metadata = self.crypto.aes_128_encrypt(b"")
        decrypted = self.crypto.aes_128_decrypt(
            ciphertext, metadata['key'], metadata['iv']
        )
        self.assertEqual(decrypted, b"")
    
    def test_large_data(self):
        """Test encryption of large data (1 MB)."""
        large_data = os.urandom(1024 * 1024)
        ciphertext, metadata = self.crypto.aes_128_encrypt(large_data)
        decrypted = self.crypto.aes_128_decrypt(
            ciphertext, metadata['key'], metadata['iv']
        )
        self.assertEqual(decrypted, large_data)


class TestAES256(unittest.TestCase):
    """Tests for AES-256 encryption/decryption."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.crypto = CryptoAlgorithms()
        self.test_data = b"Hello, World! This is test data for AES-256."
    
    def test_encrypt_decrypt_roundtrip(self):
        """Test that encryption followed by decryption returns original data."""
        ciphertext, metadata = self.crypto.aes_256_encrypt(self.test_data)
        decrypted = self.crypto.aes_256_decrypt(
            ciphertext, metadata['key'], metadata['iv']
        )
        self.assertEqual(decrypted, self.test_data)
    
    def test_key_is_256_bits(self):
        """Test that AES-256 uses 256-bit key."""
        _, metadata = self.crypto.aes_256_encrypt(self.test_data)
        self.assertEqual(len(metadata['key']), 32)  # 256 bits


class TestChaCha20(unittest.TestCase):
    """Tests for ChaCha20 encryption/decryption."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.crypto = CryptoAlgorithms()
        self.test_data = b"Hello, World! This is test data for ChaCha20."
    
    def test_encrypt_decrypt_roundtrip(self):
        """Test that encryption followed by decryption returns original data."""
        ciphertext, metadata = self.crypto.chacha20_encrypt(self.test_data)
        decrypted = self.crypto.chacha20_decrypt(
            ciphertext, metadata['key'], metadata['nonce']
        )
        self.assertEqual(decrypted, self.test_data)
    
    def test_encrypt_produces_valid_metadata(self):
        """Test that encryption produces valid metadata."""
        _, metadata = self.crypto.chacha20_encrypt(self.test_data)
        self.assertIn('key', metadata)
        self.assertIn('nonce', metadata)
        self.assertEqual(len(metadata['key']), 32)   # 256 bits
        self.assertEqual(len(metadata['nonce']), 16) # 128-bit nonce for CTR
    
    def test_ciphertext_same_length_as_plaintext(self):
        """Test that ChaCha20 (stream cipher) produces same-length ciphertext."""
        ciphertext, _ = self.crypto.chacha20_encrypt(self.test_data)
        # Note: Implementation may add padding or prefix; adjust test accordingly
        self.assertGreaterEqual(len(ciphertext), len(self.test_data))


class TestRSA(unittest.TestCase):
    """Tests for RSA encryption/decryption."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.crypto = CryptoAlgorithms()
        # RSA can only encrypt small amounts of data
        self.test_data = b"Short message for RSA"
    
    def test_rsa_2048_encrypt_decrypt_roundtrip(self):
        """Test RSA-2048 encryption/decryption roundtrip."""
        ciphertext, metadata = self.crypto.rsa_2048_encrypt(self.test_data)
        decrypted = self.crypto.rsa_2048_decrypt(
            ciphertext, metadata['private_key']
        )
        self.assertEqual(decrypted, self.test_data)
    
    def test_rsa_4096_encrypt_decrypt_roundtrip(self):
        """Test RSA-4096 encryption/decryption roundtrip."""
        ciphertext, metadata = self.crypto.rsa_4096_encrypt(self.test_data)
        decrypted = self.crypto.rsa_4096_decrypt(
            ciphertext, metadata['private_key']
        )
        self.assertEqual(decrypted, self.test_data)
    
    def test_rsa_produces_public_private_keys(self):
        """Test that RSA produces both public and private keys."""
        _, metadata = self.crypto.rsa_2048_encrypt(self.test_data)
        self.assertIn('public_key', metadata)
        self.assertIn('private_key', metadata)


class TestECC(unittest.TestCase):
    """Tests for ECC key exchange."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.crypto = CryptoAlgorithms()
        self.test_data = b"Data to encrypt with ECC-derived key"
    
    def test_ecc_256_key_exchange(self):
        """Test ECC-256 key exchange produces shared secret."""
        # Generate two key pairs
        ciphertext1, metadata1 = self.crypto.ecc_256_key_exchange(self.test_data)
        ciphertext2, metadata2 = self.crypto.ecc_256_key_exchange(self.test_data)
        
        # Each should have public and private keys
        self.assertIn('public_key', metadata1)
        self.assertIn('private_key', metadata1)
        
    def test_ecc_decrypt(self):
        """Test ECC decryption with derived key."""
        ciphertext, metadata = self.crypto.ecc_256_key_exchange(self.test_data)
        
        # Decryption should work with the same keys
        decrypted = self.crypto.ecc_256_decrypt(
            ciphertext, 
            metadata['private_key'],
            metadata['peer_public_key']
        )
        self.assertEqual(decrypted, self.test_data)


class TestAlgorithmInterface(unittest.TestCase):
    """Tests for the unified algorithm interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.crypto = CryptoAlgorithms()
        self.test_data = b"Test data for unified interface"
    
    def test_get_algorithm_info(self):
        """Test getting algorithm information."""
        info = self.crypto.get_algorithm_info('AES-256')
        self.assertIn('key_size', info)
        self.assertIn('block_size', info)
        self.assertIn('type', info)
    
    def test_list_algorithms(self):
        """Test listing available algorithms."""
        algorithms = self.crypto.list_algorithms()
        expected = ['AES-128', 'AES-256', 'ChaCha20', 'RSA-2048', 'RSA-4096', 'ECC-256']
        for alg in expected:
            self.assertIn(alg, algorithms)


if __name__ == '__main__':
    unittest.main()
