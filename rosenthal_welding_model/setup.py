from setuptools import setup, find_packages

# Read long description safely
with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rosenthal-welding-model",
    version="1.0.0",
    author="Even Englund",
    description="PhD research tool for thermal analysis of welding processes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "streamlit>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pylint>=2.15.0",
            "pycodestyle>=2.9.0",
            "pydocstyle>=6.1.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
    project_urls={
        "Source": "https://github.com/EvenUIS/rosenthal-welding-model",
    },
)
