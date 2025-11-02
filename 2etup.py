from setuptools import setup, find_packages
from pathlib import Path

readme = ""
p = Path("README.md")
if p.exists():
    readme = p.read_text(encoding="utf-8")

setup(
    name="vision-glyphs",
    version="0.1.0",
    author="Nine 1 Eight",
    author_email="founder918tech@gmail.com",
    description="Symbolic Object Detection Suite with R-CNN & Faster R-CNN",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/xxNine1Eightxx/Vision_Glyphs",
    packages=find_packages(exclude=("tests","examples","docs")),
    license="MIT",
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
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.18.0",
        "scikit-image>=0.21.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "Pillow>=10.0.0",
        "packaging>=23.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0","black>=23.0.0","flake8>=6.0.0"],
    },
    entry_points={
        "console_scripts": [
            "vision-glyphs=vision_glyphs.rcnn_suite:main",
            "glyph-loader=vision_glyphs.glyph_loader:main",
        ],
    },
    include_package_data=True,
)
