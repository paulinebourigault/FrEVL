"""
Setup script for FrEVL package
"""

import os
from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
README_PATH = Path(__file__).parent / "README.md"
with open(README_PATH, encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
REQUIREMENTS_PATH = Path(__file__).parent / "requirements.txt"
with open(REQUIREMENTS_PATH, encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read version
VERSION_PATH = Path(__file__).parent / "frevl" / "__init__.py"
version = None
with open(VERSION_PATH, encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

if version is None:
    version = "1.0.0"

setup(
    name="frevl",
    version=version,
    author="Emmanuelle Bourigault",
    author_email="emmanuelle.bourigault@research.org",
    description="FrEVL: Frozen Embeddings for Efficient Vision-Language Understanding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EmmanuelleB985/FrEVL",
    project_urls={
        "Bug Tracker": "https://github.com/EmmanuelleB985/FrEVL/issues",
        "Documentation": "https://emmanelleb985.github.io/FrEVL/",
        "Source Code": "https://github.com/EmmanuelleB985/FrEVL",
        "Paper": "https://arxiv.org/pdf/2508.04469",
    },
    packages=find_packages(exclude=["tests*", "benchmarks*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
        "demo": [
            "gradio>=4.0.0",
            "streamlit>=1.28.0",
        ],
        "serve": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "redis>=5.0.0",
        ],
        "all": [
            "gradio>=4.0.0",
            "streamlit>=1.28.0",
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "redis>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "frevl-train=frevl.cli:train",
            "frevl-evaluate=frevl.cli:evaluate",
            "frevl-demo=frevl.cli:demo",
            "frevl-serve=frevl.cli:serve",
            "frevl-download=frevl.cli:download",
        ],
    },
    include_package_data=True,
    package_data={
        "frevl": [
            "configs/*.yaml",
            "configs/*.json",
            "assets/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "vision-language",
        "multimodal",
        "clip",
        "vqa",
        "visual-question-answering",
        "efficient-ml",
        "frozen-embeddings",
        "pytorch",
        "deep-learning",
        "computer-vision",
        "nlp",
    ],
)
