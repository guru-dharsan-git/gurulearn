"""
Dependency checking utilities for gurulearn.

Provides functions to check if required and optional dependencies are installed.
"""

from __future__ import annotations

import sys
from typing import NamedTuple


class DependencyStatus(NamedTuple):
    """Status of a dependency check."""
    name: str
    installed: bool
    version: str | None
    required: bool
    install_cmd: str


# Core dependencies required for basic functionality
CORE_DEPENDENCIES = {
    "numpy": {"import": "numpy", "min_version": "1.22.0"},
    "pandas": {"import": "pandas", "min_version": "1.3.0"},
}

# Optional dependencies grouped by feature
OPTIONAL_DEPENDENCIES = {
    "ml": {
        "scikit-learn": {"import": "sklearn", "min_version": "1.0.0"},
        "xgboost": {"import": "xgboost", "min_version": "1.7.0"},
    },
    "vision": {
        "torch": {"import": "torch", "min_version": "2.0.0"},
        "torchvision": {"import": "torchvision", "min_version": "0.15.0"},
        "opencv": {"import": "cv2", "min_version": "4.5.0"},
        "pillow": {"import": "PIL", "min_version": "9.0.0"},
    },
    "audio": {
        "tensorflow": {"import": "tensorflow", "min_version": "2.16.0"},
        "librosa": {"import": "librosa", "min_version": "0.9.0"},
    },
    "agent": {
        "langchain": {"import": "langchain", "min_version": "0.3.0"},
        "langchain-ollama": {"import": "langchain_ollama", "min_version": "0.3.0"},
        "faiss-cpu": {"import": "faiss", "min_version": "1.7.0"},
    },
}


def _get_version(module) -> str | None:
    """Get version string from a module."""
    for attr in ("__version__", "VERSION", "version"):
        version = getattr(module, attr, None)
        if version is not None:
            if callable(version):
                version = version()
            return str(version)
    return None


def check_dependency(name: str, import_name: str, required: bool = False) -> DependencyStatus:
    """
    Check if a single dependency is installed.
    
    Args:
        name: Display name of the dependency
        import_name: Python import name
        required: Whether this is a required dependency
        
    Returns:
        DependencyStatus with installation information
    """
    try:
        module = __import__(import_name)
        version = _get_version(module)
        return DependencyStatus(
            name=name,
            installed=True,
            version=version,
            required=required,
            install_cmd=f"pip install {name}"
        )
    except ImportError:
        return DependencyStatus(
            name=name,
            installed=False,
            version=None,
            required=required,
            install_cmd=f"pip install {name}"
        )


def ensure_dependencies(feature: str | None = None, verbose: bool = False) -> dict[str, bool]:
    """
    Check and report on dependency availability.
    
    Args:
        feature: Optional feature group to check ('ml', 'vision', 'audio', 'agent')
                If None, checks all dependencies
        verbose: If True, print detailed status
        
    Returns:
        Dictionary mapping dependency names to availability status
        
    Raises:
        ImportError: If required dependencies are missing (with helpful message)
    """
    status = {}
    missing_required = []
    
    # Always check core dependencies
    for name, info in CORE_DEPENDENCIES.items():
        result = check_dependency(name, info["import"], required=True)
        status[name] = result.installed
        if not result.installed:
            missing_required.append(result)
        elif verbose:
            print(f"✓ {name} ({result.version or 'unknown version'})")
    
    # Check optional dependencies
    if feature is None:
        # Check all optional dependencies
        groups = OPTIONAL_DEPENDENCIES.values()
    elif feature in OPTIONAL_DEPENDENCIES:
        groups = [OPTIONAL_DEPENDENCIES[feature]]
    else:
        groups = []
    
    for group in groups:
        for name, info in group.items():
            result = check_dependency(name, info["import"], required=False)
            status[name] = result.installed
            if verbose:
                if result.installed:
                    print(f"✓ {name} ({result.version or 'unknown version'})")
                else:
                    print(f"✗ {name} - install with: {result.install_cmd}")
    
    # Raise error if required dependencies are missing
    if missing_required:
        missing_list = "\n".join(f"  - {d.name}: {d.install_cmd}" for d in missing_required)
        raise ImportError(
            f"Missing required dependencies:\n{missing_list}\n\n"
            f"Install all with: pip install {' '.join(d.name for d in missing_required)}"
        )
    
    return status