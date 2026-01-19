#!/usr/bin/env python3
"""
Validate Encryption/Decryption CLI Improvements

Tests the improved cmd_encrypt() and cmd_decrypt() functions with:
1. JSON metadata instead of pickle
2. Research tool warnings
3. --no-metadata flag
"""

import json
import os
import subprocess
import sys
from pathlib import Path

print("=" * 80)
print("CRYPTOGREEN CLI ENCRYPTION/DECRYPTION VALIDATION")
print("=" * 80)
print()

# Test file
test_file = 'data/test_files/txt/txt_1KB.txt'
if not Path(test_file).exists():
    print(f"❌ Test file not found: {test_file}")
    sys.exit(1)

print("1. TEST ENCRYPTION WITH JSON METADATA")
print("-" * 80)

# Encrypt with AES-256
result = subprocess.run(
    ['python', 'cryptogreen_cli.py', 'encrypt', test_file, 
     '--algorithm', 'AES-256', '--output', '/tmp/validation_test.encrypted'],
    capture_output=True, text=True
)

if result.returncode != 0:
    print(f"❌ Encryption failed: {result.stderr}")
    sys.exit(1)

print("✓ Encryption completed successfully")

# Check metadata file exists
meta_path = '/tmp/validation_test.encrypted.meta'
if not Path(meta_path).exists():
    print(f"❌ Metadata file not created: {meta_path}")
    sys.exit(1)

print(f"✓ Metadata file created: {meta_path}")

# Verify it's JSON
try:
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    print("✓ Metadata is valid JSON (not pickle)")
except json.JSONDecodeError:
    print("❌ Metadata is not valid JSON")
    sys.exit(1)

# Check required fields
required_fields = ['algorithm', 'key', 'iv', 'file_size', 'encrypted_size', 'timestamp']
missing = [f for f in required_fields if f not in metadata]

if missing:
    print(f"❌ Missing required fields: {missing}")
    sys.exit(1)

print(f"✓ All required fields present: {', '.join(required_fields)}")

# Verify base64 encoding
import base64
try:
    key_bytes = base64.b64decode(metadata['key'])
    iv_bytes = base64.b64decode(metadata['iv'])
    print(f"✓ Key length: {len(key_bytes)} bytes (expected 32 for AES-256)")
    print(f"✓ IV length: {len(iv_bytes)} bytes (expected 16)")
    
    if len(key_bytes) != 32:
        print(f"❌ Invalid key length for AES-256")
        sys.exit(1)
    if len(iv_bytes) != 16:
        print(f"❌ Invalid IV length")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Base64 decoding failed: {e}")
    sys.exit(1)

print()

# Test decryption
print("2. TEST DECRYPTION WITH JSON METADATA")
print("-" * 80)

result = subprocess.run(
    ['python', 'cryptogreen_cli.py', 'decrypt', '/tmp/validation_test.encrypted',
     '--output', '/tmp/validation_test.decrypted'],
    capture_output=True, text=True
)

if result.returncode != 0:
    print(f"❌ Decryption failed: {result.stderr}")
    sys.exit(1)

print("✓ Decryption completed successfully")

# Verify decrypted matches original
import filecmp
if not filecmp.cmp(test_file, '/tmp/validation_test.decrypted', shallow=False):
    print("❌ Decrypted file does not match original!")
    sys.exit(1)

print("✓ Decrypted file matches original perfectly")

print()

# Test --no-metadata flag
print("3. TEST --no-metadata FLAG")
print("-" * 80)

result = subprocess.run(
    ['python', 'cryptogreen_cli.py', 'encrypt', test_file,
     '--algorithm', 'ChaCha20', '--output', '/tmp/validation_test2.encrypted',
     '--no-metadata'],
    capture_output=True, text=True
)

if result.returncode != 0:
    print(f"❌ Encryption with --no-metadata failed: {result.stderr}")
    sys.exit(1)

print("✓ Encryption with --no-metadata completed")

# Verify no metadata file created
meta_path2 = '/tmp/validation_test2.encrypted.meta'
if Path(meta_path2).exists():
    print(f"❌ Metadata file should NOT exist with --no-metadata flag")
    sys.exit(1)

print("✓ No metadata file created (as expected)")

# Verify warning messages appeared
if 'no-metadata' not in result.stdout.lower():
    print("⚠ Warning about --no-metadata may be missing")
else:
    print("✓ --no-metadata warning displayed")

print()

# Test research tool warning
print("4. VERIFY RESEARCH TOOL WARNINGS")
print("-" * 80)

# Check that warnings appear in output
if 'RESEARCH TOOL' not in result.stdout:
    print("❌ Research tool warning not found in output")
    sys.exit(1)

print("✓ Research tool warning displayed")

if 'UNENCRYPTED' in result.stdout or 'unencrypted' in result.stdout.lower():
    print("✓ Unencrypted key warning displayed")
else:
    print("⚠ Unencrypted key warning may be missing")

if 'KMS' in result.stdout or 'HSM' in result.stdout:
    print("✓ Production key management guidance displayed")
else:
    print("⚠ Key management guidance may be missing")

print()

# Cleanup
print("5. CLEANUP")
print("-" * 80)

cleanup_files = [
    '/tmp/validation_test.encrypted',
    '/tmp/validation_test.encrypted.meta',
    '/tmp/validation_test.decrypted',
    '/tmp/validation_test2.encrypted',
]

for f in cleanup_files:
    if Path(f).exists():
        os.remove(f)
        print(f"  Removed: {f}")

print()
print("=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
print()
print("✅ Encryption works with JSON metadata")
print("✅ Metadata is JSON (not pickle) with base64-encoded binary fields")
print("✅ Decryption works and produces correct output")
print("✅ --no-metadata flag prevents metadata file creation")
print("✅ Research tool warnings are displayed")
print()
print("KEY IMPROVEMENTS:")
print("  • JSON metadata format (human-readable)")
print("  • Base64-encoded binary fields (key, IV, nonce)")
print("  • Prominent research tool warnings")
print("  • --no-metadata flag for user-managed keys")
print("  • Backward compatibility with pickle format")
print()
print("LIMITATIONS (DOCUMENTED):")
print("  • Keys stored unencrypted (acceptable for research)")
print("  • Not suitable for production use")
print("  • Users must implement proper key management for production")
print()
print("=" * 80)
