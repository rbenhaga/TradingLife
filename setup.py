from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="crypto-trading-bot",
    version="0.1.0",
    author="Rayane Ben Haga",
    author_email="rayanebenhaga@gmail.com",
    description="Un algorithme de trading automatisé pour les cryptomonnaies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheRedFiire/crypto-trading-bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "python-dotenv",
        "ccxt",
        "pandas",
        "numpy",
        "sqlalchemy>=2.0.0",
        "fastapi>=0.104.0",
        "pandas-ta>=0.3.14b0",
    ],
    entry_points={
        "console_scripts": [
            "cryptobot=src.main:main",
        ],
    },
)