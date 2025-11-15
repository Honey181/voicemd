"""
VoiceMD - Voice Analysis Application
Setup configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

setup(
    name="voicemd-gui",
    version="1.0.1",
    author="Honey181 (based on work by Jeremy Pinto)",
    author_email="",
    description="Modern offline voice analysis application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Honey181/voicemd",
    packages=find_packages(),
    py_modules=['app_gui', 'app_predictor', 'download_models'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8,<3.14",
    install_requires=[
        "torch>=1.13.0",
        "torchaudio>=0.13.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "scipy>=1.9.0",
        "PyYAML>=6.0",
        "requests>=2.28.0",
    ],
    entry_points={
        'console_scripts': [
            'voicemd-gui=app_gui:main',
            'voicemd-download=download_models:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.yaml', 'README.md', 'LICENSE', '*.pt'],
        'voicemd': ['*.yaml'],
    },
)
