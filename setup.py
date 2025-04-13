from setuptools import setup, find_packages
from pathlib import Path

# Read README.md with UTF-8 encoding
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='gurulearn',
    version='3.0',
    description='Comprehensive ML library for model analysis, computer vision, medical imaging, and audio processing with enhanced features including confidence metrics and flowbot integration (modularity introduced) used lazy loader to fix slow loading',
    author='Guru Dharsan T',
    author_email='gurudharsan123@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        # Core Data Science
        'numpy>=1.22,<2',
        'pandas>=1.3',
        'scipy>=1.9',

        # Machine Learning
        'scikit-learn>=1.0',
        'xgboost>=1.7',

        # Deep Learning (TensorFlow includes Keras)
        'tensorflow==2.16.1',

        # Image Processing (headless for server compatibility)
        'opencv-python-headless>=4.5',
        'pillow>=9.0',

        # Audio Processing
        'librosa>=0.9',
        'resampy>=0.4',

        # Visualization
        'matplotlib>=3.5',
        'seaborn>=0.12',
        'plotly>=5.10',

        # Agent langchain and dependencies
        'langchain==0.3.23',
        'langchain-ollama==0.3.1',
        'onnxruntime==1.21.0',  # Choose either this or GPU version, not both
        # 'onnxruntime-gpu',    # Commented out to avoid conflicts
        'tokenizers',
        'langchain-community==0.3.21',  # Added explicit dependency for tokenizers
        'faiss-cpu==1.10.0',
        # Utilities
        'tqdm>=4.64',
    ],
    package_data={
        '': ['*.json', '*.txt', '*.onnx'],  # Include model files if needed
    },
    include_package_data=True,  # Make sure package data is included
    extras_require={
        'dev': [
            'pytest>=7.0',
            'black>=22.0',
            'flake8>=5.0'
        ],
        'gpu': [
            'onnxruntime-gpu',  # Moved to extras_require
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
    python_requires='>=3.9',
    keywords='machine learning, deep learning, computer vision, medical imaging, audio processing, AI',
)