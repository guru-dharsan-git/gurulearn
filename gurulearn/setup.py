from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='gurulearn',
    version='1.0.20',
    description='library for MLModelAnalysis and multi image model(bug fix 3.0)',
    author='Guru Dharsan T',
    author_email='gurudharsan123@gmail.com',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
    'opencv-python',
    'scipy',
    'matplotlib',
    'tensorflow==2.16.1',
    'opencv-python-headless',
    'keras',
    'pandas',
    'numpy',
    'plotly',
    'scikit-learn',
    'librosa',
    'tqdm',
    'resampy',
    'pillow',
    'xgboost'
    ],
)
