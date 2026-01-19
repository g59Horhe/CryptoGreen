#!/usr/bin/env python3
"""
CryptoGreen Command-Line Interface

Provides easy access to CryptoGreen's energy-efficient cryptographic
algorithm selection.

Usage:
    cryptogreen select <file> [--security-level LEVEL]
    cryptogreen benchmark <file> [--algorithm ALG]
    cryptogreen train [--benchmark-file FILE]
    cryptogreen info
"""

import argparse
import base64
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from cryptogreen import (
    CryptoAlgorithms,
    CryptoBenchmark,
    FeatureExtractor,
    HybridSelector,
    RuleBasedSelector,
    MLSelector,
)
from cryptogreen.utils import (
    format_bytes,
    format_duration,
    format_energy,
    get_system_info,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_select(args):
    """Handle the 'select' command."""
    file_path = args.file
    
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1
    
    # Initialize selector
    selector = HybridSelector(model_path=args.model)
    
    # Get recommendation
    result = selector.select_algorithm(
        file_path=file_path,
        security_level=args.security_level,
        power_mode=args.power_mode,
        verbose=args.verbose
    )
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print()
        print("=" * 50)
        print("ALGORITHM RECOMMENDATION")
        print("=" * 50)
        print(f"File: {Path(file_path).name}")
        print(f"Size: {format_bytes(Path(file_path).stat().st_size)}")
        print()
        print(f"Recommended: {result['algorithm']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Method: {result['method']}")
        print()
        
        if 'explanation' in result:
            print("Explanation:")
            print(f"  {result['explanation']}")
        print("=" * 50)
    
    return 0


def cmd_benchmark(args):
    """Handle the 'benchmark' command."""
    file_path = args.file
    
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1
    
    benchmark = CryptoBenchmark()
    
    if args.algorithm:
        # Benchmark single algorithm
        algorithms = [args.algorithm]
    else:
        # Benchmark all symmetric algorithms
        algorithms = ['AES-128', 'AES-256', 'ChaCha20']
    
    results = []
    
    for alg in algorithms:
        logger.info(f"Benchmarking {alg}...")
        
        try:
            result = benchmark.benchmark_algorithm(
                alg,
                file_path,
                runs=args.runs
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Benchmark failed for {alg}: {e}")
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print()
        print("=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"File: {Path(file_path).name}")
        print(f"Size: {format_bytes(Path(file_path).stat().st_size)}")
        print(f"Runs: {args.runs}")
        print()
        
        for result in results:
            stats = result['statistics']
            print(f"{result['algorithm']}:")
            print(f"  Median time: {format_duration(stats['median_duration_s'])}")
            print(f"  Median energy: {format_energy(stats['median_energy_j'])}")
            print(f"  Throughput: {stats['mean_throughput_mbps']:.2f} MB/s")
            if stats['median_energy_j'] > 0:
                efficiency = (Path(file_path).stat().st_size / 1_000_000) / stats['median_energy_j']
            else:
                efficiency = 0
            print(f"  Efficiency: {efficiency:.2f} MB/J")
            print()
        
        # Find most efficient
        if len(results) > 1:
            most_efficient = min(results, key=lambda r: r['statistics']['median_duration_s'])
            print(f"Fastest: {most_efficient['algorithm']}")
        
        print("=" * 60)
    
    return 0


def cmd_train(args):
    """Handle the 'train' command."""
    from pathlib import Path
    import pandas as pd
    
    benchmark_file = args.benchmark_file
    
    if not Path(benchmark_file).exists():
        print(f"Error: Benchmark file not found: {benchmark_file}", file=sys.stderr)
        print("Run 'python scripts/run_benchmarks.py' first to generate benchmark data.")
        return 1
    
    # Load benchmark data
    logger.info(f"Loading benchmark data from {benchmark_file}")
    df = pd.read_csv(benchmark_file)
    
    # Train model
    logger.info("Training ML model...")
    selector = MLSelector()
    selector.train_from_benchmark_results(df)
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selector.save_model(str(output_path))
    
    print()
    print("=" * 50)
    print("MODEL TRAINING COMPLETE")
    print("=" * 50)
    print(f"Model saved to: {output_path}")
    print()
    
    return 0


def cmd_info(args):
    """Handle the 'info' command."""
    system_info = get_system_info()
    
    print()
    print("=" * 50)
    print("CRYPTOGREEN SYSTEM INFO")
    print("=" * 50)
    print()
    
    print("Hardware:")
    print(f"  Platform: {system_info.get('os', 'Unknown')} {system_info.get('os_release', '')}")
    print(f"  CPU: {system_info.get('cpu', 'Unknown')}")
    
    import os
    cpu_count = os.cpu_count() or 1
    print(f"  Cores: {cpu_count}")
    print(f"  Memory: {system_info.get('ram_total_gb', 0):.1f} GB")
    print()
    
    print("Features:")
    extractor = FeatureExtractor()
    hw_caps = extractor.detect_hardware_capabilities()
    print(f"  AES-NI: {'✓' if hw_caps.get('has_aes_ni', False) else '✗'}")
    print(f"  AVX: {'✓' if hw_caps.get('has_avx', False) else '✗'}")
    print(f"  AVX2: {'✓' if hw_caps.get('has_avx2', False) else '✗'}")
    print()
    
    print("Energy Measurement:")
    try:
        from cryptogreen.energy_meter import RAPLEnergyMeter
        rapl = RAPLEnergyMeter()
        if rapl.is_available():
            print("  RAPL: ✓ Available")
        else:
            print("  RAPL: ✗ Not available (using software estimation)")
    except Exception:
        print("  RAPL: ✗ Not available (using software estimation)")
    print()
    
    print("Available Algorithms:")
    algorithms = ['AES-128', 'AES-256', 'ChaCha20']
    for alg in algorithms:
        print(f"  • {alg}")
    print()
    
    print("=" * 50)
    
    return 0


def cmd_encrypt(args):
    """Handle the 'encrypt' command."""
    file_path = args.file
    
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1
    
    # Display research tool warning
    print()
    print("=" * 70)
    print("WARNING: This is a RESEARCH TOOL, not for production use!")
    print("Keys are saved UNENCRYPTED in .meta file (or not saved with --no-metadata)")
    print("For production, use proper key management (KMS, HSM, vault, etc.)")
    print("This tool is intended for benchmarking and research purposes only.")
    print("=" * 70)
    print()
    
    # Read input file
    with open(file_path, 'rb') as f:
        data = f.read()
    
    algorithm = args.algorithm
    
    if algorithm == 'auto':
        # Use selector to choose algorithm
        selector = HybridSelector()
        result = selector.select_algorithm(file_path)
        algorithm = result['algorithm']
        print(f"Auto-selected algorithm: {algorithm}")
    
    # Get encryption function and generate keys
    crypto = CryptoAlgorithms()
    
    try:
        # Generate keys/IVs before encryption
        import os
        
        if algorithm == 'AES-128':
            key = os.urandom(16)  # 128 bits
            iv = os.urandom(16)
            encrypted = crypto.aes_128_encrypt(data, key, iv)
            metadata = {'key': key, 'iv': iv}
        elif algorithm == 'AES-256':
            key = os.urandom(32)  # 256 bits
            iv = os.urandom(16)
            encrypted = crypto.aes_256_encrypt(data, key, iv)
            metadata = {'key': key, 'iv': iv}
        elif algorithm == 'ChaCha20':
            key = os.urandom(32)  # 256 bits
            nonce = os.urandom(12)  # 96 bits
            encrypted = crypto.chacha20_encrypt(data, key, nonce)
            metadata = {'key': key, 'nonce': nonce}
        else:
            print(f"Error: Unsupported algorithm for file encryption: {algorithm}")
            return 1
        
        # Save output
        output_path = args.output or f"{file_path}.encrypted"
        with open(output_path, 'wb') as f:
            f.write(encrypted)
        
        # Save metadata to JSON (more readable than pickle)
        metadata_saved = False
        if not args.no_metadata:
            metadata_path = f"{output_path}.meta"
            
            # Prepare metadata with base64-encoded binary data
            meta_json = {
                'algorithm': algorithm,
                'file_size': len(data),
                'encrypted_size': len(encrypted),
                'timestamp': datetime.now().isoformat(),
            }
            
            # Add algorithm-specific fields
            if 'key' in metadata:
                meta_json['key'] = base64.b64encode(metadata['key']).decode('utf-8')
            if 'iv' in metadata:
                meta_json['iv'] = base64.b64encode(metadata['iv']).decode('utf-8')
            if 'nonce' in metadata:
                meta_json['nonce'] = base64.b64encode(metadata['nonce']).decode('utf-8')
            
            with open(metadata_path, 'w') as f:
                json.dump(meta_json, f, indent=2)
            
            metadata_saved = True
        
        print()
        print("=" * 70)
        print("ENCRYPTION COMPLETE")
        print("=" * 70)
        print(f"Algorithm: {algorithm}")
        print(f"Input: {file_path} ({format_bytes(len(data))})")
        print(f"Output: {output_path} ({format_bytes(len(encrypted))})")
        
        if metadata_saved:
            print(f"Metadata: {metadata_path}")
            print()
            print("⚠ WARNING: Metadata file contains UNENCRYPTED key material!")
            print("  - For research/testing only")
            print("  - Do NOT use in production")
            print("  - Delete metadata after use or store securely")
            print("  - Use --no-metadata flag to skip saving keys")
        else:
            print()
            print("⚠ No metadata saved (--no-metadata flag)")
            print("  - You must manage encryption keys yourself")
            print("  - Decryption will not be possible without the key")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"Error during encryption: {e}", file=sys.stderr)
        return 1
    
    return 0


def cmd_decrypt(args):
    """Handle the 'decrypt' command."""
    file_path = args.file
    
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1
    
    # Load metadata from JSON
    metadata_path = args.metadata or f"{file_path}.meta"
    if not Path(metadata_path).exists():
        print(f"Error: Metadata file not found: {metadata_path}", file=sys.stderr)
        print("\nHint: Metadata file should be at {file_path}.meta", file=sys.stderr)
        print("      Or specify with --metadata flag", file=sys.stderr)
        print("      If encrypted with --no-metadata, you must provide keys manually.", file=sys.stderr)
        return 1
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        # Try pickle for backward compatibility
        try:
            import pickle
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print("⚠ Warning: Using legacy pickle metadata format")
            print("  Consider re-encrypting with current version for JSON format")
        except Exception as e:
            print(f"Error: Could not load metadata file: {e}", file=sys.stderr)
            return 1
    
    # Read encrypted data
    with open(file_path, 'rb') as f:
        encrypted = f.read()
    
    algorithm = metadata['algorithm']
    crypto = CryptoAlgorithms()
    
    try:
        # Decode base64-encoded binary fields if present
        if isinstance(metadata.get('key'), str):
            key = base64.b64decode(metadata['key'])
        else:
            key = metadata['key']
        
        if algorithm == 'AES-128':
            if isinstance(metadata.get('iv'), str):
                iv = base64.b64decode(metadata['iv'])
            else:
                iv = metadata['iv']
            decrypted = crypto.aes_128_decrypt(encrypted, key, iv)
        elif algorithm == 'AES-256':
            if isinstance(metadata.get('iv'), str):
                iv = base64.b64decode(metadata['iv'])
            else:
                iv = metadata['iv']
            decrypted = crypto.aes_256_decrypt(encrypted, key, iv)
        elif algorithm == 'ChaCha20':
            if isinstance(metadata.get('nonce'), str):
                nonce = base64.b64decode(metadata['nonce'])
            else:
                nonce = metadata['nonce']
            decrypted = crypto.chacha20_decrypt(encrypted, key, nonce)
        else:
            print(f"Error: Unsupported algorithm: {algorithm}")
            return 1
        
        # Save output
        output_path = args.output or file_path.replace('.encrypted', '.decrypted')
        with open(output_path, 'wb') as f:
            f.write(decrypted)
        
        print()
        print("=" * 50)
        print("DECRYPTION COMPLETE")
        print("=" * 50)
        print(f"Algorithm: {algorithm}")
        print(f"Input: {file_path}")
        print(f"Output: {output_path}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during decryption: {e}", file=sys.stderr)
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='cryptogreen',
        description='CryptoGreen: Energy-efficient cryptographic algorithm selector'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Select command
    select_parser = subparsers.add_parser(
        'select',
        help='Get algorithm recommendation for a file'
    )
    select_parser.add_argument('file', help='File to analyze')
    select_parser.add_argument(
        '--security-level',
        choices=['standard', 'high'],
        default='standard',
        help='Required security level'
    )
    select_parser.add_argument(
        '--power-mode',
        choices=['performance', 'balanced', 'powersave'],
        default='balanced',
        help='Power mode'
    )
    select_parser.add_argument(
        '--model',
        default='results/models/selector_model.pkl',
        help='Path to trained model'
    )
    select_parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    select_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    select_parser.set_defaults(func=cmd_select)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        'benchmark',
        help='Benchmark encryption algorithms on a file'
    )
    benchmark_parser.add_argument('file', help='File to benchmark')
    benchmark_parser.add_argument(
        '--algorithm', '-a',
        choices=['AES-128', 'AES-256', 'ChaCha20'],
        help='Specific algorithm (default: all)'
    )
    benchmark_parser.add_argument(
        '--runs', '-r',
        type=int,
        default=10,
        help='Number of runs'
    )
    benchmark_parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    benchmark_parser.set_defaults(func=cmd_benchmark)
    
    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train the ML model from benchmark data'
    )
    train_parser.add_argument(
        '--benchmark-file',
        default='results/benchmark_results.csv',
        help='Path to benchmark CSV file'
    )
    train_parser.add_argument(
        '--output', '-o',
        default='results/models/selector_model.pkl',
        help='Output path for trained model'
    )
    train_parser.set_defaults(func=cmd_train)
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show system information and capabilities'
    )
    info_parser.set_defaults(func=cmd_info)
    
    # Encrypt command
    encrypt_parser = subparsers.add_parser(
        'encrypt',
        help='Encrypt a file using selected algorithm'
    )
    encrypt_parser.add_argument('file', help='File to encrypt')
    encrypt_parser.add_argument(
        '--algorithm', '-a',
        choices=['auto', 'AES-128', 'AES-256', 'ChaCha20'],
        default='auto',
        help='Encryption algorithm (default: auto-select)'
    )
    encrypt_parser.add_argument(
        '--output', '-o',
        help='Output file path'
    )
    encrypt_parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Do not save encryption keys to metadata file (you must manage keys yourself)'
    )
    encrypt_parser.set_defaults(func=cmd_encrypt)
    
    # Decrypt command
    decrypt_parser = subparsers.add_parser(
        'decrypt',
        help='Decrypt a file'
    )
    decrypt_parser.add_argument('file', help='File to decrypt')
    decrypt_parser.add_argument(
        '--metadata', '-m',
        help='Path to metadata file'
    )
    decrypt_parser.add_argument(
        '--output', '-o',
        help='Output file path'
    )
    decrypt_parser.set_defaults(func=cmd_decrypt)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
