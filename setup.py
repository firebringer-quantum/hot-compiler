from setuptools import setup, find_packages

setup(
    name="hot-framework",
    version="2.0.0",
    description="Hardware-Optimized Techniques for Quantum Circuit Execution",
    author="Justin Grammens / Firebringer AI",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "qiskit>=0.45.0",
        "qiskit-ibm-runtime>=0.21.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "networkx>=2.6",
        "dataclasses-json>=0.5.7",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
