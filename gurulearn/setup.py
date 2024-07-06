from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='gurulearn',
    version='1.0.9',
    description='library for linear_regression and gvgg16 model generation(fixed bugs),audio_classify and audio_classify_predict',
    author='Guru Dharsan T',
    author_email='gurudharsan123@gmail.com',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'scipy',
        'matplotlib',
        'tensorflow==2.16.1',
        'Keras',
        'pandas',
        'numpy',
        'plotly',
        'scikit-learn',
        'librosa' ,
        'tqdm',
        'resampy'


    ],
)
