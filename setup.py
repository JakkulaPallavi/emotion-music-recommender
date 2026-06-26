from setuptools import setup, find_packages

setup(
    name="emotion-music-recommender",
    version="1.0.0",
    description="An AI/ML system that detects emotions from text and recommends matching music.",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
)
