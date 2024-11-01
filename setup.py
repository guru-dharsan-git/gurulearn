from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='gurulearn',
    version='1.1.1',  
    description='Library for ML model analysis, multi-image model support,CT scan processing, and audio recognition(added feature confidance to audio recognition)',
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
        'xgboost',
        'seaborn'  
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
