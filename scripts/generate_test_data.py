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
        Random text as bytes of EXACTLY the target size.
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
    
    # Repeat lorem text to fill the size
    repeats = (size // len(lorem)) + 1
    text = (lorem * repeats)[:size]
    data = text.encode('utf-8')
    
    # Ensure EXACTLY the target size
    if len(data) < size:
        data += b' ' * (size - len(data))
    
    return data[:size]


def generate_sql_data(size: int) -> bytes:
    """Generate SQL INSERT statements.
    
    Args:
        size: Target size in bytes.
        
    Returns:
        SQL statements as bytes of EXACTLY the target size.
    """
    tables = ['users', 'orders', 'products', 'transactions', 'logs']
    
    lines = [
        "-- CryptoGreen Test SQL Data\n",
        "-- Generated for benchmarking purposes\n\n",
    ]
    current_size = sum(len(line.encode('utf-8')) for line in lines)
    
    record_id = 1
    while current_size < size - 100:  # Leave room for final padding
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
        current_size += len(line.encode('utf-8'))
        record_id += 1
    
    sql = ''.join(lines)
    data = sql.encode('utf-8')
    
    # Pad to exact size with SQL comments
    if len(data) < size:
        padding = b'-- ' + b'X' * (size - len(data) - 3)
        data = data + padding
    
    return data[:size]


def generate_jpeg_image(size: int) -> bytes:
    """Generate a JPEG image of EXACTLY the target size.
    
    Args:
        size: Target size in bytes.
        
    Returns:
        JPEG image bytes of EXACTLY the target size.
    """
    try:
        from PIL import Image
        
        # Start with a base size and adjust
        # For small sizes, use small image; for large, use larger image
        if size < 10000:
            width = height = 50
        elif size < 100000:
            width = height = 200
        else:
            # Estimate dimensions based on target size
            pixels_needed = size // 2
            dim = int(pixels_needed ** 0.5)
            width = height = max(dim, 100)
        
        # Create solid color image (compresses well, predictable size)
        color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        image = Image.new('RGB', (width, height), color)
        
        # Adjust quality to get closer to target size
        buffer = io.BytesIO()
        quality = 85
        
        # Binary search for right quality/size combination
        for attempt in range(10):
            buffer.seek(0)
            buffer.truncate()
            image.save(buffer, format='JPEG', quality=quality)
            current_size = buffer.tell()
            
            if current_size == size:
                break
            elif abs(current_size - size) < 100:  # Close enough
                break
            
            if current_size < size:
                # Need larger file
                if quality < 95:
                    quality = min(quality + 5, 95)
                else:
                    # Increase image size
                    width = int(width * 1.1)
                    height = int(height * 1.1)
                    image = Image.new('RGB', (width, height), color)
            else:
                # File too large
                quality = max(quality - 5, 10)
        
        data = buffer.getvalue()
        
        # Pad or trim to EXACT size
        if len(data) < size:
            # JPEG padding: insert before end marker (0xFF 0xD9)
            padding_size = size - len(data)
            if data.endswith(b'\xFF\xD9'):
                # Add comment segment 0xFFFE (max 65535 bytes per segment)
                if padding_size >= 4:
                    # JPEG comment segment length is 2 bytes (includes length bytes themselves)
                    # Max segment payload is 65535 - 2 = 65533 bytes
                    remaining_padding = padding_size
                    padded_data = data[:-2]  # Remove end marker temporarily
                    
                    while remaining_padding > 2:
                        # Each comment can hold at most 65533 bytes of actual padding
                        chunk_size = min(remaining_padding - 2, 65533)
                        segment_len = chunk_size + 2  # +2 for the length bytes themselves
                        
                        # Ensure segment_len fits in 2 bytes (max 65535)
                        if segment_len > 65535:
                            segment_len = 65535
                            chunk_size = 65533
                        
                        comment_chunk = b'\xFF\xFE' + segment_len.to_bytes(2, 'big') + b'\x00' * chunk_size
                        padded_data += comment_chunk
                        remaining_padding -= len(comment_chunk)
                    
                    # Add any remaining bytes as simple padding
                    if remaining_padding > 0:
                        padded_data += b'\x00' * remaining_padding
                    
                    data = padded_data + b'\xFF\xD9'  # Add back end marker
                else:
                    data = data + b'\x00' * padding_size
            else:
                data = data + b'\x00' * padding_size
        elif len(data) > size:
            # Trim and ensure valid JPEG ending
            data = data[:size-2] + b'\xFF\xD9'
        
        return data
        
    except ImportError:
        logger.warning("PIL not available, generating JPEG-like bytes")
        # JPEG header and end marker with random data
        header = b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        end = b'\xFF\xD9'
        middle_size = size - len(header) - len(end)
        data = header + os.urandom(max(0, middle_size)) + end
        return data[:size]


def generate_png_image(size: int) -> bytes:
    """Generate a PNG image of EXACTLY the target size.
    
    Args:
        size: Target size in bytes.
        
    Returns:
        PNG image bytes of EXACTLY the target size.
    """
    try:
        from PIL import Image
        import zlib
        
        # For small sizes, use tiny image; for large, use bigger
        if size < 1000:
            width = height = 10
        elif size < 10000:
            width = height = 50
        elif size < 100000:
            width = height = 150
        else:
            # Estimate dimensions
            pixels_needed = size
            dim = int(pixels_needed ** 0.5)
            width = height = max(dim // 2, 100)
        
        # Create gradient image (less compressible than solid color)
        image = Image.new('RGB', (width, height))
        pixels = image.load()
        
        for i in range(width):
            for j in range(height):
                # Gradient pattern - harder to compress
                pixels[i, j] = (
                    (i * 255) // width,
                    (j * 255) // height,
                    ((i + j) * 255) // (width + height)
                )
        
        # Try different compression levels
        buffer = io.BytesIO()
        compress_level = 6
        
        for attempt in range(10):
            buffer.seek(0)
            buffer.truncate()
            image.save(buffer, format='PNG', compress_level=compress_level)
            current_size = buffer.tell()
            
            if current_size == size or abs(current_size - size) < 100:
                break
            
            if current_size < size * 0.9:
                # Need more data - increase image size
                width = int(width * 1.15)
                height = int(height * 1.15)
                image = Image.new('RGB', (width, height))
                pixels = image.load()
                for i in range(width):
                    for j in range(height):
                        pixels[i, j] = (
                            (i * 255) // width,
                            (j * 255) // height,
                            ((i + j) * 255) // (width + height)
                        )
            elif current_size > size:
                # Try more compression
                compress_level = min(compress_level + 1, 9)
        
        data = buffer.getvalue()
        
        # Pad to exact size with tEXt chunk
        if len(data) < size and data.endswith(b'IEND\xaeB`\x82'):
            padding_needed = size - len(data)
            if padding_needed >= 12:  # Minimum chunk size
                chunk_data = b'CryptoGreen benchmark padding: ' + b'X' * (padding_needed - 50)
                chunk_len = len(chunk_data)
                crc = zlib.crc32(b'tEXt' + chunk_data) & 0xffffffff
                
                chunk = (
                    chunk_len.to_bytes(4, 'big') +
                    b'tEXt' +
                    chunk_data +
                    crc.to_bytes(4, 'big')
                )
                
                # Insert before IEND
                data = data[:-12] + chunk + data[-12:]
        
        # Final adjustment - pad with null bytes if still short
        if len(data) < size:
            data = data[:-4] + b'\x00' * (size - len(data)) + data[-4:]
        
        return data[:size]
        
    except ImportError:
        logger.warning("PIL not available, generating PNG-like bytes")
        # PNG signature
        header = b'\x89PNG\r\n\x1a\n'
        # Minimal IHDR chunk
        ihdr = b'\x00\x00\x00\x0dIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
        # IEND chunk
        iend = b'\x00\x00\x00\x00IEND\xaeB`\x82'
        
        middle_size = size - len(header) - len(ihdr) - len(iend)
        middle = os.urandom(max(0, middle_size))
        
        data = header + ihdr + middle + iend
        return data[:size]


def generate_mp4_video(size: int) -> bytes:
    """Generate an MP4 video file of EXACTLY the target size.
    
    Args:
        size: Target size in bytes.
        
    Returns:
        MP4 video bytes of EXACTLY the target size.
    """
    # Try using ffmpeg if available
    try:
        import subprocess
        import tempfile
        
        # Only try ffmpeg for larger files (>10KB) as it has overhead
        if size > 10000:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp_path = tmp.name
            
            # Calculate duration and bitrate for target size
            # Base overhead ~5KB for MP4 container
            target_video_size = max(size - 5000, 1000)
            duration = max(1, target_video_size // (50 * 1024))  # ~50KB/s base
            bitrate = (target_video_size * 8) // max(duration, 1)
            
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-f', 'lavfi',
                '-i', f'color=c=black:s=160x120:d={duration}',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-b:v', f'{bitrate}',
                '-maxrate', f'{bitrate}',
                '-bufsize', f'{bitrate * 2}',
                '-an',  # No audio
                '-movflags', '+faststart',
                tmp_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0 and os.path.exists(tmp_path):
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                os.unlink(tmp_path)
                
                # Pad or trim to exact size
                if len(data) < size:
                    # Add null padding
                    data += b'\x00' * (size - len(data))
                
                return data[:size]
    except (ImportError, FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"ffmpeg not available or failed: {e}")
    
    # Fallback: generate minimal valid MP4 structure
    # ftyp box (file type)
    ftyp = b'\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41'
    
    # free box for padding (can be any size)
    mdat_size = size - len(ftyp) - 8
    if mdat_size < 0:
        mdat_size = 0
    
    # mdat box (media data) - filled with zeros
    mdat_header = (mdat_size + 8).to_bytes(4, 'big') + b'mdat'
    mdat_content = b'\x00' * mdat_size
    
    data = ftyp + mdat_header + mdat_content
    return data[:size]


def generate_pdf_document(size: int) -> bytes:
    """Generate a PDF document of EXACTLY the target size.
    
    Args:
        size: Target size in bytes.
        
    Returns:
        PDF document bytes of EXACTLY the target size.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        
        # Generate text content to fill pages
        y_position = 750
        chars_written = 0
        target_chars = size  # Overestimate to ensure we have enough
        
        text_line = "CryptoGreen benchmark test data. " * 3
        
        while chars_written < target_chars:
            if y_position < 50:
                c.showPage()
                y_position = 750
            
            c.drawString(50, y_position, text_line[:80])
            chars_written += 80
            y_position -= 15
        
        c.save()
        data = buffer.getvalue()
        
        # Pad to target size with PDF comments
        if len(data) < size:
            # PDF comments can be added before %%EOF
            padding_size = size - len(data)
            padding = b'\n%' + b'X' * (padding_size - 2) + b'\n'
            
            # Find %%EOF and insert padding before it
            eof_pos = data.rfind(b'%%EOF')
            if eof_pos > 0:
                data = data[:eof_pos] + padding + data[eof_pos:]
            else:
                data = data + padding
        
        return data[:size]
        
    except ImportError:
        logger.warning("reportlab not available, generating minimal PDF")
        # Minimal valid PDF structure
        pdf_header = b"%PDF-1.4\n"
        pdf_catalog = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        pdf_pages = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        pdf_page = b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        pdf_xref = b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
        pdf_trailer = b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n200\n%%EOF\n"
        
        pdf_base = pdf_header + pdf_catalog + pdf_pages + pdf_page + pdf_xref + pdf_trailer
        
        # Pad with comments if needed
        if len(pdf_base) < size:
            padding_size = size - len(pdf_base)
            # Insert padding before trailer
            padding = b'%' + b'CryptoGreen-' * (padding_size // 12) + b'X' * (padding_size % 12) + b'\n'
            
            trailer_pos = pdf_base.rfind(b'trailer')
            if trailer_pos > 0:
                data = pdf_base[:trailer_pos] + padding + pdf_base[trailer_pos:]
            else:
                data = pdf_base + padding
        else:
            data = pdf_base
        
        return data[:size]


def generate_zip_file(size: int) -> bytes:
    """Generate a ZIP file of EXACTLY the target size.
    
    Args:
        size: Target size in bytes.
        
    Returns:
        ZIP file bytes of EXACTLY the target size.
    """
    buffer = io.BytesIO()
    
    # ZIP has overhead (headers, central directory), estimate ~100-200 bytes per file
    # For very small sizes, use uncompressed random data
    if size < 200:
        # Minimal ZIP with tiny file
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_STORED) as zf:
            content = b'X' * max(1, size - 150)
            zf.writestr('data.txt', content)
    else:
        # Larger files - use compression
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Estimate content size needed
            # Compressed random data typically reduces to ~60-80% of original
            content_size = int(size * 0.9)  # Slightly less to account for headers
            
            # Random data compresses well, text compresses better
            # Mix of both for predictable size
            content = os.urandom(content_size // 2) + b'CryptoGreen' * (content_size // 22)
            zf.writestr('data.bin', content)
    
    data = buffer.getvalue()
    
    # Adjust to exact size
    attempt = 0
    while len(data) != size and attempt < 5:
        buffer = io.BytesIO()
        
        if len(data) < size:
            # Need more data
            shortage = size - len(data)
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Increase content size
                content_size = int((size + shortage) * 0.9)
                content = os.urandom(content_size // 2) + b'X' * (content_size // 2)
                zf.writestr('data.bin', content)
        else:
            # Too much data
            excess = len(data) - size
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                content_size = int((size - excess) * 0.9)
                content = os.urandom(max(1, content_size // 2)) + b'Y' * max(1, content_size // 2)
                zf.writestr('data.bin', content)
        
        data = buffer.getvalue()
        attempt += 1
    
    # Final padding/trimming
    if len(data) < size:
        # ZIP files can have trailing data (not part of archive) - safe to pad
        data += b'\x00' * (size - len(data))
    
    return data[:size]


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
    """Verify a generated file has EXACTLY the expected size.
    
    Args:
        file_path: Path to file.
        expected_size: Expected size in bytes.
        
    Returns:
        True if file is EXACTLY the expected size, False otherwise.
    """
    if not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        return False
    
    actual_size = file_path.stat().st_size
    
    if actual_size != expected_size:
        logger.error(
            f"Size mismatch for {file_path.name}: "
            f"expected {expected_size} bytes, got {actual_size} bytes "
            f"(diff: {actual_size - expected_size:+d})"
        )
        return False
    
    return True


def generate_test_data(
    output_dir: str = 'data/test_files',
    verify: bool = True,
    verbose: bool = False,
    force: bool = False
) -> dict:
    """Generate all test files for benchmarking.
    
    Generates EXACTLY 49 test files (7 types × 7 sizes).
    Each file will be EXACTLY the target size in bytes.
    
    Args:
        output_dir: Output directory for test files.
        verify: Whether to verify files have exact sizes after creation.
        verbose: Show detailed progress.
        force: Regenerate files even if they exist.
        
    Returns:
        Dict with generation statistics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_files = len(FILE_TYPES) * len(SIZES)
    created = 0
    failed = 0
    skipped = 0
    verified = 0
    verification_failed = 0
    
    logger.info("=" * 70)
    logger.info(f"CryptoGreen Test Data Generator")
    logger.info(f"Generating {total_files} test files (7 types × 7 sizes)")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)
    logger.info(f"File types: {', '.join(FILE_TYPES)}")
    logger.info(f"File sizes: {', '.join(SIZES.keys())}")
    logger.info("=" * 70)
    
    for file_type in FILE_TYPES:
        type_dir = output_path / file_type
        type_dir.mkdir(exist_ok=True)
        
        logger.info(f"\nGenerating {file_type.upper()} files...")
        
        for size_name, size_bytes in SIZES.items():
            filename = f"{file_type}_{size_name}.{file_type}"
            file_path = type_dir / filename
            
            if file_path.exists() and not force:
                if verbose:
                    logger.info(f"  ✓ Skipping existing: {filename}")
                skipped += 1
                
                # Still verify if requested
                if verify:
                    if verify_file(file_path, size_bytes):
                        verified += 1
                    else:
                        verification_failed += 1
                continue
            
            if verbose:
                logger.info(f"  → Generating: {filename} ({size_bytes:,} bytes)")
            else:
                # Show progress without verbose
                print(f"  Generating {filename}...", end=' ', flush=True)
            
            success = generate_file(file_type, size_bytes, file_path)
            
            if success:
                created += 1
                
                # Verify exact size
                if verify:
                    if verify_file(file_path, size_bytes):
                        verified += 1
                        actual_size = file_path.stat().st_size
                        if verbose:
                            logger.info(f"    ✓ Verified: {actual_size:,} bytes (exact match)")
                        else:
                            print(f"✓ ({file_path.stat().st_size:,} bytes)")
                    else:
                        verification_failed += 1
                        actual_size = file_path.stat().st_size if file_path.exists() else 0
                        logger.error(f"    ✗ Verification FAILED: {actual_size:,} bytes (expected {size_bytes:,})")
                        if not verbose:
                            print(f"✗ FAILED")
                else:
                    if not verbose:
                        print("✓")
            else:
                failed += 1
                logger.error(f"    ✗ Generation FAILED: {filename}")
                if not verbose:
                    print("✗ FAILED")
    
    # Create .gitkeep files to preserve directory structure
    for file_type in FILE_TYPES:
        gitkeep = output_path / file_type / '.gitkeep'
        gitkeep.touch()
    
    logger.info("\n" + "=" * 70)
    logger.info("Generation Summary:")
    logger.info("=" * 70)
    logger.info(f"  Total expected:         {total_files}")
    logger.info(f"  Created:                {created}")
    logger.info(f"  Skipped (existing):     {skipped}")
    logger.info(f"  Failed:                 {failed}")
    if verify:
        logger.info(f"  Verified (exact size):  {verified}")
        logger.info(f"  Verification failed:    {verification_failed}")
    logger.info("=" * 70)
    
    if verification_failed > 0:
        logger.warning(f"\n⚠ {verification_failed} file(s) did not match expected size!")
    
    if failed == 0 and verification_failed == 0:
        logger.info("\n✓ All test files generated successfully with exact sizes!")
    
    return {
        'output_dir': str(output_path),
        'total_expected': total_files,
        'created': created,
        'skipped': skipped,
        'failed': failed,
        'verified': verified,
        'verification_failed': verification_failed,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate test data for CryptoGreen benchmarks - EXACTLY 49 files (7 types × 7 sizes)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
File Types: txt, jpg, png, mp4, pdf, sql, zip (7 types)
File Sizes: 64B, 1KB, 10KB, 100KB, 1MB, 10MB, 100MB (7 sizes)
Total:      7 × 7 = 49 files

Each file will be EXACTLY the specified size in bytes.

Examples:
  # Generate all files with verification
  python scripts/generate_test_data.py --verify

  # Regenerate all files (overwrite existing)
  python scripts/generate_test_data.py --force --verify

  # Generate with verbose output
  python scripts/generate_test_data.py --verbose --verify
        """
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
        default=True,
        help='Verify files have exact sizes after creation (default: enabled)'
    )
    parser.add_argument(
        '--no-verify',
        dest='verify',
        action='store_false',
        help='Skip verification of file sizes'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Regenerate files even if they already exist'
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
        verbose=args.verbose,
        force=args.force
    )
    
    # Exit with error code if any failures
    if stats['failed'] > 0 or stats.get('verification_failed', 0) > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
