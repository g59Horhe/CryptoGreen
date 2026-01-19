"""
Cryptographic Algorithms Module

This module provides a unified interface for all supported symmetric
encryption algorithms (AES-128, AES-256, ChaCha20).

Example:
    >>> from cryptogreen.algorithms import CryptoAlgorithms
    >>> data = b"Hello, World!"
    >>> ciphertext = CryptoAlgorithms.aes_128_encrypt(data)
    >>> print(f"Encrypted: {len(ciphertext)} bytes")
"""

import logging
import os
from typing import Optional, Tuple

# Cryptography library imports
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

# PyCryptodome for ChaCha20
from Crypto.Cipher import ChaCha20

logger = logging.getLogger(__name__)


class CryptoAlgorithms:
    """Unified interface for symmetric encryption algorithms.
    
    This class provides static methods for each supported algorithm.
    All methods use cryptographically secure random key generation
    when keys are not provided.
    
    Supported Algorithms:
        - AES-128: 128-bit AES in CBC mode with PKCS7 padding
        - AES-256: 256-bit AES in CBC mode with PKCS7 padding
        - ChaCha20: 256-bit stream cipher
    
    Example:
        >>> data = b"Secret message"
        >>> ciphertext = CryptoAlgorithms.aes_128_encrypt(data)
        >>> print(f"Ciphertext length: {len(ciphertext)}")
    """
    
    # Algorithm metadata
    ALGORITHMS = {
        'AES-128': {
            'type': 'symmetric',
            'key_size': 16,  # bytes
            'iv_size': 16,
            'block_size': 16,
            'description': 'AES with 128-bit key, CBC mode',
        },
        'AES-256': {
            'type': 'symmetric',
            'key_size': 32,
            'iv_size': 16,
            'block_size': 16,
            'description': 'AES with 256-bit key, CBC mode',
        },
        'ChaCha20': {
            'type': 'stream',
            'key_size': 32,
            'nonce_size': 12,
            'description': 'ChaCha20 stream cipher',
        },
    }
    
    @staticmethod
    def aes_128_encrypt(
        data: bytes,
        key: Optional[bytes] = None,
        iv: Optional[bytes] = None
    ) -> bytes:
        """Encrypt data with AES-128 CBC mode.
        
        Args:
            data: Plaintext bytes to encrypt.
            key: 16-byte encryption key. Generated if None.
            iv: 16-byte initialization vector. Generated if None.
            
        Returns:
            Ciphertext bytes with PKCS7 padding.
            
        Example:
            >>> data = b"Hello, World!"
            >>> key = os.urandom(16)
            >>> iv = os.urandom(16)
            >>> ciphertext = CryptoAlgorithms.aes_128_encrypt(data, key, iv)
        """
        if key is None:
            key = os.urandom(16)
        if iv is None:
            iv = os.urandom(16)
        
        if len(key) != 16:
            raise ValueError(f"AES-128 key must be 16 bytes, got {len(key)}")
        if len(iv) != 16:
            raise ValueError(f"AES IV must be 16 bytes, got {len(iv)}")
        
        # Apply PKCS7 padding
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        # Encrypt
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        return ciphertext
    
    @staticmethod
    def aes_128_decrypt(
        ciphertext: bytes,
        key: bytes,
        iv: bytes
    ) -> bytes:
        """Decrypt AES-128 CBC encrypted data.
        
        Args:
            ciphertext: Encrypted bytes.
            key: 16-byte decryption key.
            iv: 16-byte initialization vector.
            
        Returns:
            Decrypted plaintext bytes.
        """
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove PKCS7 padding
        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()
        
        return data
    
    @staticmethod
    def aes_256_encrypt(
        data: bytes,
        key: Optional[bytes] = None,
        iv: Optional[bytes] = None
    ) -> bytes:
        """Encrypt data with AES-256 CBC mode.
        
        Args:
            data: Plaintext bytes to encrypt.
            key: 32-byte encryption key. Generated if None.
            iv: 16-byte initialization vector. Generated if None.
            
        Returns:
            Ciphertext bytes with PKCS7 padding.
            
        Example:
            >>> data = b"High security message"
            >>> ciphertext = CryptoAlgorithms.aes_256_encrypt(data)
        """
        if key is None:
            key = os.urandom(32)
        if iv is None:
            iv = os.urandom(16)
        
        if len(key) != 32:
            raise ValueError(f"AES-256 key must be 32 bytes, got {len(key)}")
        if len(iv) != 16:
            raise ValueError(f"AES IV must be 16 bytes, got {len(iv)}")
        
        # Apply PKCS7 padding
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        # Encrypt
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        return ciphertext
    
    @staticmethod
    def aes_256_decrypt(
        ciphertext: bytes,
        key: bytes,
        iv: bytes
    ) -> bytes:
        """Decrypt AES-256 CBC encrypted data.
        
        Args:
            ciphertext: Encrypted bytes.
            key: 32-byte decryption key.
            iv: 16-byte initialization vector.
            
        Returns:
            Decrypted plaintext bytes.
        """
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove PKCS7 padding
        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()
        
        return data
    
    @staticmethod
    def chacha20_encrypt(
        data: bytes,
        key: Optional[bytes] = None,
        nonce: Optional[bytes] = None
    ) -> bytes:
        """Encrypt data with ChaCha20 stream cipher.
        
        Args:
            data: Plaintext bytes to encrypt.
            key: 32-byte encryption key. Generated if None.
            nonce: 12-byte nonce. Generated if None.
            
        Returns:
            Ciphertext bytes.
            
        Note:
            ChaCha20 is a stream cipher, so ciphertext length equals
            plaintext length (no padding needed).
            
        Example:
            >>> data = b"Fast encryption"
            >>> ciphertext = CryptoAlgorithms.chacha20_encrypt(data)
        """
        if key is None:
            key = os.urandom(32)
        if nonce is None:
            nonce = os.urandom(12)
        
        if len(key) != 32:
            raise ValueError(f"ChaCha20 key must be 32 bytes, got {len(key)}")
        if len(nonce) != 12:
            raise ValueError(f"ChaCha20 nonce must be 12 bytes, got {len(nonce)}")
        
        # PyCryptodome ChaCha20 uses 8-byte nonce by default, we use RFC 7539 style
        cipher = ChaCha20.new(key=key, nonce=nonce)
        ciphertext = cipher.encrypt(data)
        
        return ciphertext
    
    @staticmethod
    def chacha20_decrypt(
        ciphertext: bytes,
        key: bytes,
        nonce: bytes
    ) -> bytes:
        """Decrypt ChaCha20 encrypted data.
        
        Args:
            ciphertext: Encrypted bytes.
            key: 32-byte decryption key.
            nonce: 12-byte nonce used for encryption.
            
        Returns:
            Decrypted plaintext bytes.
        """
        cipher = ChaCha20.new(key=key, nonce=nonce)
        return cipher.decrypt(ciphertext)
    
    @classmethod
    def encrypt(
        cls,
        algorithm_name: str,
        data: bytes,
        **kwargs
    ) -> bytes:
        """Encrypt data using specified algorithm.
        
        This is a convenience method that dispatches to the appropriate
        algorithm-specific method.
        
        Args:
            algorithm_name: One of the supported algorithm names.
            data: Data to encrypt.
            **kwargs: Algorithm-specific keyword arguments.
            
        Returns:
            Encrypted data or signature.
            
        Raises:
            ValueError: If algorithm name is not recognized.
            
        Example:
            >>> data = b"Message"
            >>> ciphertext = CryptoAlgorithms.encrypt('AES-128', data)
        """
        algorithm_map = {
            'AES-128': cls.aes_128_encrypt,
            'AES-256': cls.aes_256_encrypt,
            'ChaCha20': cls.chacha20_encrypt,
        }
        
        if algorithm_name not in algorithm_map:
            raise ValueError(
                f"Unknown algorithm: {algorithm_name}. "
                f"Supported: {list(algorithm_map.keys())}"
            )
        
        return algorithm_map[algorithm_name](data, **kwargs)
    
    @staticmethod
    def get_algorithm_names() -> list[str]:
        """Return list of all supported algorithm names.
        
        Returns:
            List of algorithm names.
            
        Example:
            >>> names = CryptoAlgorithms.get_algorithm_names()
            >>> print(names)
            ['AES-128', 'AES-256', 'ChaCha20']
        """
        return ['AES-128', 'AES-256', 'ChaCha20']
    
    @classmethod
    def get_algorithm_info(cls, algorithm_name: str) -> dict:
        """Get metadata about an algorithm.
        
        Args:
            algorithm_name: Name of the algorithm.
            
        Returns:
            Dict with algorithm metadata.
        """
        if algorithm_name not in cls.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        return cls.ALGORITHMS[algorithm_name].copy()
    
    @classmethod
    def generate_keys(cls, algorithm_name: str) -> Tuple:
        """Pre-generate keys for an algorithm.
        
        Args:
            algorithm_name: Name of the algorithm.
            
        Returns:
            Tuple of (key, iv/nonce) for the algorithm.
        """
        if algorithm_name == 'AES-128':
            return os.urandom(16), os.urandom(16)
        elif algorithm_name == 'AES-256':
            return os.urandom(32), os.urandom(16)
        elif algorithm_name == 'ChaCha20':
            return os.urandom(32), os.urandom(12)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
