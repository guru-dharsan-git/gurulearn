# Gurulearn - Comprehensive AI/ML Library
# Modern lazy loading implementation for fast imports

"""
GuruLearn: A comprehensive Python library integrating machine learning,
computer vision, audio processing, and conversational AI capabilities.

Modules:
    - FlowBot: Conversational flow management for chatbots
    - CTScanProcessor: Medical image processing and denoising
    - AudioRecognition: Audio classification with deep learning
    - ImageClassifier: Image classification with multiple architectures
    - MLModelAnalysis: Automated ML model training and evaluation
    - QAAgent: RAG-based question answering with vector stores
    - ocr: Character-level OCR with CTC decoding (training & inference)
"""

from __future__ import annotations

__version__ = "5.1.0"
__author__ = "Guru Dharsan T"
__email__ = "gurudharsan123@gmail.com"

# Type hints for IDE support (actual imports are lazy)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ChatFlow import FlowBot as _FlowBot
    from .CtScan import CTScanProcessor as _CTScanProcessor
    from .Audio import AudioRecognition as _AudioRecognition
    from .Image_Classification import ImageClassifier as _ImageClassifier
    from .Machine_Learning import MLModelAnalysis as _MLModelAnalysis
    from .AgentQA import QAAgent as _QAAgent

# Module-level cache for lazy loaded classes
_loaded_modules: dict[str, type] = {}

# Mapping of public names to their module paths
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlowBot": (".ChatFlow", "FlowBot"),
    "CTScanProcessor": (".CtScan", "CTScanProcessor"),
    "AudioRecognition": (".Audio", "AudioRecognition"),
    "ImageClassifier": (".Image_Classification", "ImageClassifier"),
    "MLModelAnalysis": (".Machine_Learning", "MLModelAnalysis"),
    "QAAgent": (".AgentQA", "QAAgent"),
}

# Submodule lazy imports
_LAZY_SUBMODULES: dict[str, str] = {
    "ocr": "gurulearn.ocr",
}

__all__ = [
    "FlowBot",
    "CTScanProcessor", 
    "AudioRecognition",
    "ImageClassifier",
    "MLModelAnalysis",
    "QAAgent",
    "__version__",
]


def __getattr__(name: str):
    """
    Lazy loading implementation using module-level __getattr__.
    
    This ensures heavy dependencies (torch, tensorflow, langchain, etc.)
    are only imported when the specific class is actually accessed.
    """
    if name in _loaded_modules:
        return _loaded_modules[name]
    
    if name in _LAZY_IMPORTS:
        module_path, class_name = _LAZY_IMPORTS[name]
        
        # Import the module and get the class
        import importlib
        module = importlib.import_module(module_path, package=__name__)
        cls = getattr(module, class_name)
        
        # Cache for future access
        _loaded_modules[name] = cls
        return cls
    
    # Submodule lazy loading (e.g. gurulearn.ocr)
    if name in _LAZY_SUBMODULES:
        import importlib
        mod = importlib.import_module(_LAZY_SUBMODULES[name])
        _loaded_modules[name] = mod
        return mod
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return list of public names for tab completion."""
    return __all__ + list(_LAZY_IMPORTS.keys())


def check_dependencies(verbose: bool = True) -> dict[str, bool]:
    """
    Check if all optional dependencies are available.
    
    Args:
        verbose: If True, print status of each dependency
        
    Returns:
        Dictionary mapping dependency names to availability status
    """
    dependencies = {
        # Core ML
        "numpy": "numpy",
        "pandas": "pandas", 
        "scikit-learn": "sklearn",
        
        # Deep Learning
        "torch": "torch",
        "torchvision": "torchvision",
        "tensorflow": "tensorflow",
        
        # Computer Vision  
        "opencv": "cv2",
        "pillow": "PIL",
        
        # Audio
        "librosa": "librosa",
        
        # LLM/RAG
        "langchain": "langchain",
        "langchain-ollama": "langchain_ollama",
        "faiss": "faiss",
        
        # Visualization
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "plotly": "plotly",
    }
    
    status = {}
    for name, import_name in dependencies.items():
        try:
            __import__(import_name)
            status[name] = True
            if verbose:
                print(f"✓ {name}")
        except ImportError:
            status[name] = False
            if verbose:
                print(f"✗ {name} (pip install {name})")
    
    return status