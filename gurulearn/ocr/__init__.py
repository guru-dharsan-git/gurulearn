"""
GuruLearn OCR — character-level OCR with CTC decoding.

Provides tools for YOLO dataset management, synthetic data augmentation,
VGG-BiLSTM model training, and self-contained inference.

Quick start::

    # Inference (no dataset needed)
    from gurulearn.ocr import OCRPredictor
    p = OCRPredictor("best_model.guruocr")
    print(p.predict("image.jpg").text)

    # Training
    from gurulearn.ocr import OCRTrainer
    trainer = OCRTrainer("path/to/yolo_dataset", "output/")
    trainer.train(epochs=100)

    # Full pipeline
    from gurulearn.ocr import OCRPipeline
    pipeline = OCRPipeline(dataset_name="aircraft", ...)
    pipeline.run_all()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .augment import AugmentResult as _AR
    from .augment import shuffle_augment as _sa
    from .data import MergeResult as _MR
    from .data import RebalanceResult as _RR
    from .data import SplitResult as _SR
    from .data import load_class_names as _lcn
    from .data import merge_datasets as _md
    from .data import rebalance_splits as _rs
    from .data import split_datasets as _sd
    from .inference import OCRPredictor as _OP
    from .inference import PredictionResult as _PR
    from .model import EvalResult as _ER
    from .model import OCRTrainer as _OT
    from .model import TrainHistory as _TH
    from .model import VGG_OCR as _VO
    from .pipeline import OCRPipeline as _OPL
    from .pipeline import PipelineResult as _PLR

__all__ = [
    # Data
    "split_datasets",
    "merge_datasets",
    "rebalance_splits",
    "load_class_names",
    "SplitResult",
    "MergeResult",
    "RebalanceResult",
    # Augmentation
    "shuffle_augment",
    "AugmentResult",
    # Model & Training
    "VGG_OCR",
    "OCRTrainer",
    "TrainHistory",
    "EvalResult",
    # Inference
    "OCRPredictor",
    "PredictionResult",
    # Pipeline
    "OCRPipeline",
    "PipelineResult",
]

# Lazy import mapping
_LAZY: dict[str, tuple[str, str]] = {
    "split_datasets": (".data", "split_datasets"),
    "merge_datasets": (".data", "merge_datasets"),
    "rebalance_splits": (".data", "rebalance_splits"),
    "load_class_names": (".data", "load_class_names"),
    "SplitResult": (".data", "SplitResult"),
    "MergeResult": (".data", "MergeResult"),
    "RebalanceResult": (".data", "RebalanceResult"),
    "shuffle_augment": (".augment", "shuffle_augment"),
    "AugmentResult": (".augment", "AugmentResult"),
    "VGG_OCR": (".model", "VGG_OCR"),
    "OCRTrainer": (".model", "OCRTrainer"),
    "TrainHistory": (".model", "TrainHistory"),
    "EvalResult": (".model", "EvalResult"),
    "OCRPredictor": (".inference", "OCRPredictor"),
    "PredictionResult": (".inference", "PredictionResult"),
    "OCRPipeline": (".pipeline", "OCRPipeline"),
    "PipelineResult": (".pipeline", "PipelineResult"),
}

_cache: dict[str, object] = {}


def __getattr__(name: str):
    if name in _cache:
        return _cache[name]
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        import importlib
        mod = importlib.import_module(mod_path, package=__name__)
        obj = getattr(mod, attr)
        _cache[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
