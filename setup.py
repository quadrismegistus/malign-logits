"""
Setup script for malign-logits package.
"""

from setuptools import setup, find_packages
import os

# Read the README for long description
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Experiments in traumatizing AI"

# Read requirements
try:
    with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    requirements = [
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'accelerate>=0.20.0',
        'bitsandbytes>=0.41.0',
        'pandas>=1.5.0',
        'tqdm>=4.65.0',
    ]

setup(
    name="malign-logits",
    version="0.1.0",
    author="rj416",
    description="A toolkit for psychoanalytic analysis of LLM probability distributions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rj416/malign-logits",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
            "mypy",
        ],
        "notebooks": [
            "jupyter",
            "matplotlib",
            "seaborn",
        ],
    },
    include_package_data=True,
    package_data={
        "malign_logits": ["*.py"],
    },
    entry_points={
        "console_scripts": [
            # Add any command-line scripts here if needed
        ],
    },
    keywords="llm ai psychoanalysis transformers huggingface",
    project_urls={
        "Bug Reports": "https://github.com/rj416/malign-logits/issues",
        "Source": "https://github.com/rj416/malign-logits",
    },
)