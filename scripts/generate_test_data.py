#!/usr/bin/env python3
"""
Generate Test Data Script

Creates synthetic test files for benchmarking cryptographic algorithms.
Generates files of various types and sizes according to the CryptoGreen specification.

Usage:
    python scripts/generate_test_data.py [OPTIONS]

Options:
    --output-dir DIR   Output directory (default: data/test_files)
    --verify          Verify files after creation
    --verbose         Show detailed progress

File Types:
    - txt: Random ASCII text (lorem ipsum style)
    - jpg: Solid color JPEG images
    - png: Solid color PNG images
    - mp4: Black frame video (requires ffmpeg or falls back to binary)
    - pdf: Text PDF documents
    - zip: Compressed random data
    - sql: SQL INSERT statements

File Sizes:
    64B, 1KB, 10KB, 100KB, 1MB, 10MB, 100MB
"""

import argparse
import io
import logging
import os
import random
import string
import sys
import zipfile
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File sizes to generate
SIZES = {
    '64B': 64,
    '1KB': 1024,
    '10KB': 10 * 1024,
    '100KB': 100 * 1024,
    '1MB': 1024 * 1024,
    '10MB': 10 * 1024 * 1024,
    '100MB': 100 * 1024 * 1024,
}

# File types to generate
FILE_TYPES = ['txt', 'jpg', 'png', 'mp4', 'pdf', 'zip', 'sql']


def generate_random_text(size: int) -> bytes:
    """Generate random ASCII text data.
    
    Args:
        size: Target size in bytes.
        
    Returns:
        Random text as bytes.
    """
    # Lorem ipsum base text
    lorem = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
        "Nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
        "reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
        "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in "
        "culpa qui officia deserunt mollit anim id est laborum.\n"
    )
    
    result = []
    current_size = 0
    
    while current_size < size:
        result.append(lorem)
        current_size += len(lorem)
    
    text = ''.join(result)[:size]
    return text.encode('utf-8')


def generate_sql_data(size: int) -> bytes:
    """Generate SQL INSERT statements.
    
    Args:
        size: Target size in bytes.
        
    Returns:
        SQL statements as bytes.
    """
    tables = ['users', 'orders', 'products', 'transactions', 'logs']
    
    lines = [
        "-- CryptoGreen Test SQL Data\n",
        "-- Generated for benchmarking purposes\n\n",
    ]
    current_size = sum(len(line) for line in lines)
    
    record_id = 1
    while current_size < size:
        table = random.choice(tables)
        
        if table == 'users':
            name = ''.join(random.choices(string.ascii_letters, k=10))
            email = f"{name.lower()}@example.com"
            line = f"INSERT INTO users (id, name, email, created_at) VALUES ({record_id}, '{name}', '{email}', NOW());\n"
        elif table == 'orders':
            amount = random.randint(100, 10000) / 100
            line = f"INSERT INTO orders (id, user_id, amount, status) VALUES ({record_id}, {random.randint(1, 1000)}, {amount}, 'completed');\n"
        elif table == 'products':
            product_name = ''.join(random.choices(string.ascii_letters, k=15))
            price = random.randint(100, 100000) / 100
            line = f"INSERT INTO products (id, name, price, stock) VALUES ({record_id}, '{product_name}', {price}, {random.randint(0, 1000)});\n"
        elif table == 'transactions':
            line = f"INSERT INTO transactions (id, type, amount, timestamp) VALUES ({record_id}, 'transfer', {random.randint(1, 10000)}, NOW());\n"
        else:
            message = ''.join(random.choices(string.ascii_letters + ' ', k=50))
            line = f"INSERT INTO logs (id, level, message) VALUES ({record_id}, 'INFO', '{message}');\n"
        
        lines.append(line)
        current_size += len(line)
        record_id += 1
    
    sql = ''.join(lines)[:size]
    return sql.encode('utf-8')


def generate_jpeg_image(size: int) -> bytes:
    """Generate a JPEG image of approximately the target size.
    
    Args:
        size: Target size in bytes.
        
    Returns:
        JPEG image bytes.
    """
    try:
        from PIL import Image
        
        # Start with a base size and adjust
        # JPEG compression is variable, so we need to iterate
        width = height = 100
        
        # Estimate dimensions based on target size
        # JPEG is roughly 1-3 bytes per pixel after compression
        pixels_needed = size // 2
        dim = int(pixels_needed ** 0.5)
        width = height = max(dim, 10)
        
        # Create image with random colors for more realistic compression
        image = Image.new('RGB', (width, height))
        pixels = image.load()
        
        for i in range(width):
            for j in range(height):
                pixels[i, j] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
        
        # Adjust quality to get closer to target size
        buffer = io.BytesIO()
        quality = 85
        
        for _ in range(5):  # Try a few iterations
            buffer.seek(0)
            buffer.truncate()
            image.save(buffer, format='JPEG', quality=quality)
            current_size = buffer.tell()
            
            if abs(current_size - size) < size * 0.1:  # Within 10%
                break
            
            if current_size < size:
                quality = min(quality + 5, 98)
                # Make image larger
                width = int(width * 1.2)
                height = int(height * 1.2)
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            else:
                quality = max(quality - 5, 10)
        
        data = buffer.getvalue()
        
        # Pad or trim to exact size if needed
        if len(data) < size:
            # Add JPEG comment to pad (0xFF 0xFE length data)
            padding_needed = size - len(data)
            # Insert before end marker (0xFF 0xD9)
            data = data[:-2] + b'\xFF\xFE' + padding_needed.to_bytes(2, 'big') + b'\x00' * (padding_needed - 4) + data[-2:]
        elif len(data) > size:
            data = data[:size-2] + b'\xFF\xD9'  # Ensure valid JPEG ending
        
        return data
        
    except ImportError:
        logger.warning("PIL not available, generating random bytes for JPEG")
        # Generate bytes with JPEG-like header
        data = b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        data += os.urandom(size - len(data) - 2)
        data += b'\xFF\xD9'  # JPEG end marker
        return data[:size]


def generate_png_image(size: int) -> bytes:
    """Generate a PNG image of approximately the target size.
    
    Args:
        size: Target size in bytes.
        
    Returns:
        PNG image bytes.
    """
    try:
        from PIL import Image
        
        # PNG compression is more predictable
        # Estimate: random data compresses to about 1 byte per pixel
        pixels_needed = size
        dim = int(pixels_needed ** 0.5)
        width = height = max(dim, 10)
        
        # Create image with random pixels (hard to compress)
        image = Image.new('RGB', (width, height))
        pixels = image.load()
        
        for i in range(width):
            for j in range(height):
                pixels[i, j] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
        
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', compress_level=6)
        data = buffer.getvalue()
        
        # Adjust size by resizing
        while len(data) < size * 0.8:
            width = int(width * 1.2)
            height = int(height * 1.2)
            image = image.resize((width, height), Image.Resampling.NEAREST)
            for i in range(width):
                for j in range(height):
                    pixels = image.load()
                    pixels[i, j] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', compress_level=6)
            data = buffer.getvalue()
        
        # Pad to exact size by adding PNG chunks
        if len(data) < size:
            # Add padding using tEXt chunk
            padding = size - len(data)
            chunk_data = b'Comment\x00' + b'X' * (padding - 12)
            chunk_len = len(chunk_data)
            
            import zlib
            crc = zlib.crc32(b'tEXt' + chunk_data) & 0xffffffff
            
            chunk = (
                chunk_len.to_bytes(4, 'big') +
                b'tEXt' +
                chunk_data +
                crc.to_bytes(4, 'big')
            )
            
            # Insert before IEND
            data = data[:-12] + chunk + data[-12:]
        
        return data[:size]
        
    except ImportError:
        logger.warning("PIL not available, generating random bytes for PNG")
        # PNG header
        header = b'\x89PNG\r\n\x1a\n'
        data = header + os.urandom(size - len(header))
        return data[:size]


def generate_mp4_video(size: int) -> bytes:
    """Generate an MP4 video file of approximately the target size.
    
    Args:
        size: Target size in bytes.
        
    Returns:
        MP4 video bytes (or MP4-like binary data).
    """
    # Try using ffmpeg if available
    try:
        import subprocess
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Calculate duration for target size (rough estimate: 100KB/s at low bitrate)
        duration = max(1, size // (100 * 1024))
        bitrate = (size * 8) // duration if duration > 0 else 1000
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', f'color=c=black:s=320x240:d={duration}',
            '-c:v', 'libx264',
            '-b:v', f'{bitrate}',
            '-an',  # No audio
            tmp_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60
        )
        
        if result.returncode == 0 and os.path.exists(tmp_path):
            with open(tmp_path, 'rb') as f:
                data = f.read()
            os.unlink(tmp_path)
            
            # Pad or trim to target size
            if len(data) < size:
                data += b'\x00' * (size - len(data))
            return data[:size]
    except Exception as e:
        logger.debug(f"ffmpeg not available: {e}")
    
    # Fallback: generate valid MP4-like structure
    # Minimal ftyp box + random mdat
    ftyp = b'\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom'
    mdat_size = size - len(ftyp) - 8
    mdat = mdat_size.to_bytes(4, 'big') + b'mdat' + os.urandom(mdat_size)
    
    return (ftyp + mdat)[:size]


def generate_pdf_document(size: int) -> bytes:
    """Generate a PDF document of approximately the target size.
    
    Args:
        size: Target size in bytes.
        
    Returns:
        PDF document bytes.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        
        # Generate text content
        y_position = 750
        chars_written = 0
        target_chars = size // 2  # Rough estimate
        
        while chars_written < target_chars:
            text = ''.join(random.choices(string.ascii_letters + ' ', k=80))
            c.drawString(50, y_position, text)
            chars_written += 80
            y_position -= 12
            
            if y_position < 50:
                c.showPage()
                y_position = 750
        
        c.save()
        data = buffer.getvalue()
        
        # Pad to target size
        if len(data) < size:
            # Add PDF comment to pad
            padding = b'%' + b'X' * (size - len(data) - 1)
            # Insert before %%EOF
            eof_pos = data.rfind(b'%%EOF')
            if eof_pos > 0:
                data = data[:eof_pos] + padding + data[eof_pos:]
        
        return data[:size]
        
    except ImportError:
        logger.warning("reportlab not available, generating minimal PDF")
        # Minimal valid PDF
        pdf = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer << /Size 4 /Root 1 0 R >>
startxref
190
%%EOF
"""
        # Pad with comments
        if len(pdf) < size:
            padding = b'\n%' + b'X' * (size - len(pdf) - 2) + b'\n'
            pdf = pdf[:-6] + padding + b'%%EOF\n'
        
        return pdf[:size]


def generate_zip_file(size: int) -> bytes:
    """Generate a ZIP file of approximately the target size.
    
    Args:
        size: Target size in bytes.
        
    Returns:
        ZIP file bytes.
    """
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # ZIP overhead is roughly 100-200 bytes per file
        # For very small files, just add minimal content
        content_size = max(1, size - 200)  # Ensure at least 1 byte
        content = os.urandom(content_size)
        zf.writestr('data.bin', content)
    
    data = buffer.getvalue()
    
    # Adjust if needed
    if len(data) < size:
        # Add more files
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            remaining = max(1, size - 100)
            file_num = 1
            while remaining > 0:
                chunk_size = min(remaining, 1024 * 1024)
                zf.writestr(f'data_{file_num}.bin', os.urandom(chunk_size))
                remaining -= chunk_size + 50  # Approximate overhead
                file_num += 1
        data = buffer.getvalue()
    
    return data[:size] if len(data) > size else data


def generate_file(file_type: str, size: int, output_path: Path) -> bool:
    """Generate a single test file.
    
    Args:
        file_type: Type of file to generate.
        size: Target size in bytes.
        output_path: Path to write file.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        generators = {
            'txt': generate_random_text,
            'sql': generate_sql_data,
            'jpg': generate_jpeg_image,
            'png': generate_png_image,
            'mp4': generate_mp4_video,
            'pdf': generate_pdf_document,
            'zip': generate_zip_file,
        }
        
        generator = generators.get(file_type)
        if not generator:
            logger.error(f"Unknown file type: {file_type}")
            return False
        
        data = generator(size)
        
        # Ensure exact size
        if len(data) < size:
            data += b'\x00' * (size - len(data))
        elif len(data) > size:
            data = data[:size]
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(data)
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating {file_type} file: {e}")
        return False


def verify_file(file_path: Path, expected_size: int) -> bool:
    """Verify a generated file.
    
    Args:
        file_path: Path to file.
        expected_size: Expected size in bytes.
        
    Returns:
        True if file is valid, False otherwise.
    """
    if not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        return False
    
    actual_size = file_path.stat().st_size
    
    # Allow 10% tolerance for compressed formats
    tolerance = expected_size * 0.1
    
    if abs(actual_size - expected_size) > tolerance:
        logger.warning(
            f"Size mismatch for {file_path.name}: "
            f"expected {expected_size}, got {actual_size}"
        )
        return False
    
    return True


def generate_test_data(
    output_dir: str = 'data/test_files',
    verify: bool = False,
    verbose: bool = False
) -> dict:
    """Generate all test files for benchmarking.
    
    Args:
        output_dir: Output directory for test files.
        verify: Whether to verify files after creation.
        verbose: Show detailed progress.
        
    Returns:
        Dict with generation statistics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_files = len(FILE_TYPES) * len(SIZES)
    created = 0
    failed = 0
    skipped = 0
    
    logger.info(f"Generating {total_files} test files in {output_dir}")
    logger.info(f"File types: {FILE_TYPES}")
    logger.info(f"Sizes: {list(SIZES.keys())}")
    
    for file_type in FILE_TYPES:
        type_dir = output_path / file_type
        type_dir.mkdir(exist_ok=True)
        
        for size_name, size_bytes in SIZES.items():
            filename = f"{file_type}_{size_name}.{file_type}"
            file_path = type_dir / filename
            
            if file_path.exists():
                if verbose:
                    logger.info(f"Skipping existing: {filename}")
                skipped += 1
                continue
            
            if verbose:
                logger.info(f"Generating: {filename} ({size_bytes} bytes)")
            
            success = generate_file(file_type, size_bytes, file_path)
            
            if success:
                created += 1
                
                if verify:
                    if not verify_file(file_path, size_bytes):
                        logger.warning(f"Verification failed: {filename}")
            else:
                failed += 1
    
    # Create .gitkeep files
    for file_type in FILE_TYPES:
        gitkeep = output_path / file_type / '.gitkeep'
        gitkeep.touch()
    
    logger.info(f"\nGeneration complete:")
    logger.info(f"  Created: {created}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Failed: {failed}")
    
    return {
        'output_dir': str(output_path),
        'total_expected': total_files,
        'created': created,
        'skipped': skipped,
        'failed': failed,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate test data for CryptoGreen benchmarks'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/test_files',
        help='Output directory (default: data/test_files)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify files after creation'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed progress'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    stats = generate_test_data(
        output_dir=args.output_dir,
        verify=args.verify,
        verbose=args.verbose
    )
    
    if stats['failed'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
