# In gurulearn/__init__.py or a separate module
def ensure_dependencies():
    """Ensure all required dependencies are loaded."""
    try:
        import onnxruntime
        import tokenizers
        import chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2
        # Import other modules that might be dynamically loaded
    except ImportError as e:
        print(f"Warning: Some dependencies might be missing: {e}")