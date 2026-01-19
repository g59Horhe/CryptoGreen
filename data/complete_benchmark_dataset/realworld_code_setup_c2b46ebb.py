"""
CryptoGreen - Intelligent Energy-Efficient Cryptographic Algorithm Selector

A hybrid ML + rule-based approach for selecting the most energy-efficient
cryptographic algorithm based on file characteristics and hardware capabilities.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cryptogreen",
    version="1.0.0",
    author="CryptoGreen Team",
    author_email="cryptogreen@example.com",
    description="Intelligent Energy-Efficient Cryptographic Algorithm Selector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cryptogreen/cryptogreen",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security :: Cryptography",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cryptogreen=cryptogreen_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cryptogreen": ["py.typed"],
    },
)
