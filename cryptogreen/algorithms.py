"""
Cryptographic Algorithms Module

This module provides a unified interface for all supported cryptographic
algorithms including symmetric encryption (AES-128, AES-256, ChaCha20),
asymmetric encryption (RSA-2048, RSA-4096), and digital signatures (ECC-256).

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
from cryptography.hazmat.primitives import padding, hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding as asym_padding
from cryptography.hazmat.backends import default_backend

# PyCryptodome for ChaCha20
from Crypto.Cipher import ChaCha20

logger = logging.getLogger(__name__)


class CryptoAlgorithms:
    """Unified interface for all cryptographic algorithms.
    
    This class provides static methods for each supported algorithm.
    All methods use cryptographically secure random key generation
    when keys are not provided.
    
    Supported Algorithms:
        - AES-128: 128-bit AES in CBC mode with PKCS7 padding
        - AES-256: 256-bit AES in CBC mode with PKCS7 padding
        - ChaCha20: 256-bit stream cipher
        - RSA-2048: 2048-bit RSA with OAEP padding
        - RSA-4096: 4096-bit RSA with OAEP padding
        - ECC-256: ECDSA with NIST P-256 curve
    
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
        'RSA-2048': {
            'type': 'asymmetric',
            'key_size': 2048,  # bits
            'max_encrypt_size': 190,  # bytes with OAEP SHA256
            'description': 'RSA 2048-bit with OAEP padding',
        },
        'RSA-4096': {
            'type': 'asymmetric',
            'key_size': 4096,
            'max_encrypt_size': 446,
            'description': 'RSA 4096-bit with OAEP padding',
        },
        'ECC-256': {
            'type': 'signature',
            'curve': 'SECP256R1',
            'description': 'ECDSA with NIST P-256 curve',
        },
    }
    
    # Cache for RSA keys (key generation is expensive)
    _rsa_2048_key = None
    _rsa_4096_key = None
    _ecc_256_key = None
    
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
    def rsa_2048_encrypt_key(
        cls,
        key_data: bytes,
        public_key=None
    ) -> bytes:
        """Encrypt symmetric key with RSA-2048.
        
        This simulates RSA key exchange by encrypting a symmetric key
        using RSA with OAEP padding.
        
        Args:
            key_data: Symmetric key to encrypt (max 190 bytes for OAEP with SHA256).
            public_key: RSA public key. If None, generates new key pair.
            
        Returns:
            Encrypted key bytes.
            
        Note:
            RSA is typically used only for key exchange, not for encrypting
            large amounts of data. For large data, use hybrid encryption.
            
        Example:
            >>> symmetric_key = os.urandom(32)  # AES-256 key
            >>> encrypted_key = CryptoAlgorithms.rsa_2048_encrypt_key(symmetric_key)
        """
        max_size = cls.ALGORITHMS['RSA-2048']['max_encrypt_size']
        if len(key_data) > max_size:
            raise ValueError(
                f"RSA-2048 can only encrypt up to {max_size} bytes with OAEP, "
                f"got {len(key_data)} bytes"
            )
        
        if public_key is None:
            # Generate new key pair (cached for performance)
            if cls._rsa_2048_key is None:
                logger.debug("Generating RSA-2048 key pair...")
                cls._rsa_2048_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                    backend=default_backend()
                )
            public_key = cls._rsa_2048_key.public_key()
        
        ciphertext = public_key.encrypt(
            key_data,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return ciphertext
    
    @classmethod
    def rsa_2048_decrypt_key(
        cls,
        ciphertext: bytes,
        private_key=None
    ) -> bytes:
        """Decrypt RSA-2048 encrypted key.
        
        Args:
            ciphertext: Encrypted key bytes.
            private_key: RSA private key. If None, uses cached key.
            
        Returns:
            Decrypted key bytes.
        """
        if private_key is None:
            if cls._rsa_2048_key is None:
                raise ValueError("No private key available for decryption")
            private_key = cls._rsa_2048_key
        
        plaintext = private_key.decrypt(
            ciphertext,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return plaintext
    
    @classmethod
    def rsa_4096_encrypt_key(
        cls,
        key_data: bytes,
        public_key=None
    ) -> bytes:
        """Encrypt symmetric key with RSA-4096.
        
        Args:
            key_data: Symmetric key to encrypt (max 446 bytes for OAEP with SHA256).
            public_key: RSA public key. If None, generates new key pair.
            
        Returns:
            Encrypted key bytes.
            
        Example:
            >>> symmetric_key = os.urandom(32)
            >>> encrypted_key = CryptoAlgorithms.rsa_4096_encrypt_key(symmetric_key)
        """
        max_size = cls.ALGORITHMS['RSA-4096']['max_encrypt_size']
        if len(key_data) > max_size:
            raise ValueError(
                f"RSA-4096 can only encrypt up to {max_size} bytes with OAEP, "
                f"got {len(key_data)} bytes"
            )
        
        if public_key is None:
            if cls._rsa_4096_key is None:
                logger.debug("Generating RSA-4096 key pair...")
                cls._rsa_4096_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=4096,
                    backend=default_backend()
                )
            public_key = cls._rsa_4096_key.public_key()
        
        ciphertext = public_key.encrypt(
            key_data,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return ciphertext
    
    @classmethod
    def rsa_4096_decrypt_key(
        cls,
        ciphertext: bytes,
        private_key=None
    ) -> bytes:
        """Decrypt RSA-4096 encrypted key.
        
        Args:
            ciphertext: Encrypted key bytes.
            private_key: RSA private key. If None, uses cached key.
            
        Returns:
            Decrypted key bytes.
        """
        if private_key is None:
            if cls._rsa_4096_key is None:
                raise ValueError("No private key available for decryption")
            private_key = cls._rsa_4096_key
        
        plaintext = private_key.decrypt(
            ciphertext,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return plaintext
    
    @classmethod
    def ecc_256_sign(
        cls,
        data: bytes,
        private_key=None
    ) -> bytes:
        """Sign data with ECC NIST P-256 (ECDSA).
        
        Args:
            data: Data to sign.
            private_key: EC private key. If None, generates new key pair.
            
        Returns:
            Digital signature bytes.
            
        Example:
            >>> data = b"Important document"
            >>> signature = CryptoAlgorithms.ecc_256_sign(data)
        """
        if private_key is None:
            if cls._ecc_256_key is None:
                logger.debug("Generating ECC P-256 key pair...")
                cls._ecc_256_key = ec.generate_private_key(
                    ec.SECP256R1(),
                    backend=default_backend()
                )
            private_key = cls._ecc_256_key
        
        signature = private_key.sign(
            data,
            ec.ECDSA(hashes.SHA256())
        )
        
        return signature
    
    @classmethod
    def ecc_256_verify(
        cls,
        data: bytes,
        signature: bytes,
        public_key=None
    ) -> bool:
        """Verify ECC NIST P-256 signature.
        
        Args:
            data: Original data that was signed.
            signature: Signature to verify.
            public_key: EC public key. If None, uses cached key's public key.
            
        Returns:
            True if signature is valid, False otherwise.
        """
        if public_key is None:
            if cls._ecc_256_key is None:
                raise ValueError("No public key available for verification")
            public_key = cls._ecc_256_key.public_key()
        
        try:
            public_key.verify(
                signature,
                data,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except Exception:
            return False
    
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
            'RSA-2048': cls.rsa_2048_encrypt_key,
            'RSA-4096': cls.rsa_4096_encrypt_key,
            'ECC-256': cls.ecc_256_sign,
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
            ['AES-128', 'AES-256', 'ChaCha20', 'RSA-2048', 'RSA-4096', 'ECC-256']
        """
        return ['AES-128', 'AES-256', 'ChaCha20', 'RSA-2048', 'RSA-4096', 'ECC-256']
    
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
    def reset_cached_keys(cls) -> None:
        """Clear all cached asymmetric keys.
        
        This forces new key generation on next use of RSA/ECC algorithms.
        Useful for benchmarking key generation separately.
        """
        cls._rsa_2048_key = None
        cls._rsa_4096_key = None
        cls._ecc_256_key = None
        logger.info("Cleared cached asymmetric keys")
    
    @classmethod
    def generate_keys(cls, algorithm_name: str) -> Tuple:
        """Pre-generate keys for an algorithm.
        
        Args:
            algorithm_name: Name of the algorithm.
            
        Returns:
            Tuple of (key, iv/nonce) for symmetric algorithms,
            or (private_key, public_key) for asymmetric algorithms.
        """
        if algorithm_name == 'AES-128':
            return os.urandom(16), os.urandom(16)
        elif algorithm_name == 'AES-256':
            return os.urandom(32), os.urandom(16)
        elif algorithm_name == 'ChaCha20':
            return os.urandom(32), os.urandom(12)
        elif algorithm_name == 'RSA-2048':
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            return private_key, private_key.public_key()
        elif algorithm_name == 'RSA-4096':
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
                backend=default_backend()
            )
            return private_key, private_key.public_key()
        elif algorithm_name == 'ECC-256':
            private_key = ec.generate_private_key(
                ec.SECP256R1(),
                backend=default_backend()
            )
            return private_key, private_key.public_key()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
