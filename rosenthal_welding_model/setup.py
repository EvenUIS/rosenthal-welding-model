from setuptools import setup, find_packages

setup(
    name             = "rosenthal-welding-model",
    version          = "1.0.0",
    author           = "Even Englund",
    description      = "PhD research tool for thermal analysis of welding processes",
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    packages         = find_packages(),
    python_requires  = ">=3.8",
    install_requires = [
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
)
