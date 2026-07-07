from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="datawash-inspector",
    version="0.4.1",
    author="kishown",
    author_email="kishown@example.com",
    description="An automated end-to-end data cleaning, preprocessing, and EDA pipeline.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kishown-e22194/datawash",
    packages=find_packages(include=["datawash*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.11.0",
        "plotly>=5.14.0"
    ],
)
