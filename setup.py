from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tpusight",
    version="0.1.0",
    author="TPUsight Team",
    description="A comprehensive TPU profiler inspired by NVIDIA Nsight",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "jax>=0.4.20",
        "jaxlib>=0.4.20",
        "numpy>=1.24.0",
        "jupyterlab>=4.0.0",
        "ipywidgets>=8.1.0",
        "plotly>=5.18.0",
        "rich>=13.7.0",
        "humanize>=4.9.0",
        "pandas>=2.1.0",
    ],
    extras_require={
        "tpu": ["jax[tpu]>=0.4.20"],
        "dev": ["pytest", "black", "ruff"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

