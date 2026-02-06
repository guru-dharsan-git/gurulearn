#!/usr/bin/env python
"""
Backward-compatible setup.py that delegates to pyproject.toml.

For modern installations, use: pip install .
This file is kept for legacy compatibility.
"""
from setuptools import setup

if __name__ == "__main__":
    setup()
