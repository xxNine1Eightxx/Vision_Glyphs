#!/usr/bin/env python3
from setuptools import setup, find_packages
from pathlib import Path

README = Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else ""

setup(
    name="vision-glyphs",
    version="0.1.0",
    author="Nine 1 Eight",
    author_email="founder918tech@gmail.com",
    description="Symbolic Object Detection Suite with R-CNN & Faster R-CNN",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/xxNine1Eightxx/Vision-Glyphs",
    project_urls={
        "Source": "https://github.com/xxNine1Eightxx/Vision-Glyphs",
        "Issues": "https://github.com/xxNine1Eightxx/Vision-Glyphs/issues",
    },
    license="MIT",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
        "Pillow>=10.0.0",
        "scikit-image>=0.21.0",
        "packaging>=23.0",
        # torch/torchvision intentionally omitted here; glyph_loader resolves platform-appropriate wheels
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "flake8>=6.0.0"],
    },
    entry_points={
        "console_scripts": [
            "vision-glyphs=vision_glyphs.rcnn_suite:main",
            "glyph-loader=vision_glyphs.glyph_loader:main",
        ],
    },
)
