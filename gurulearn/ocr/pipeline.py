"""
End-to-end OCR pipeline.

Orchestrates: split → augment → merge → rebalance → train → evaluate,
either all at once or step-by-step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .augment import AugmentResult, shuffle_augment
from .data import (
    MergeResult,
    RebalanceResult,
    SplitResult,
    load_class_names,
    merge_datasets,
    rebalance_splits,
    split_datasets,
)
from .inference import OCRPredictor
from .model import EvalResult, OCRTrainer, TrainHistory


@dataclass
class PipelineResult:
    """Aggregated result of a full pipeline run."""
    split: Optional[SplitResult] = None
    augment: Optional[AugmentResult] = None
    merge: Optional[MergeResult] = None
    rebalance: Optional[RebalanceResult] = None
    train: Optional[TrainHistory] = None
    evaluate: Optional[EvalResult] = None
    model_path: str = ""


class OCRPipeline:
    """
    Automated end-to-end OCR pipeline.

    Example — run everything::

        pipeline = OCRPipeline(
            source_dirs=["dataset_v1", "dataset_v2"],
            output_root="segregated_datasets",
            augment_count=30000,
            doubles_count=5200,
            train_epochs=150,
        )
        result = pipeline.run_all()

    Example — run step-by-step::

        pipeline = OCRPipeline(...)
        pipeline.step_split()
        pipeline.step_augment()
        pipeline.step_merge()
        pipeline.step_rebalance()
        pipeline.step_train()
        pipeline.step_evaluate()

    Args:
        source_dirs: YOLO dataset directories to start from.
        output_root: Root for segregated/augmented datasets.
        dataset_name: Name of the target dataset to augment and train on
            (default ``"aircraft"``).
        augment_count: Total synthetic images to generate.
        doubles_count: Synthetic images with doubled letters.
        train_epochs: Max training epochs.
        train_batch_size: Batch size for training.
        train_lr: Learning rate.
        patience: Early-stopping patience.
        img_h: Image height for the OCR model.
        img_w: Image width for the OCR model.
        split_keywords: Keyword→category mapping for dataset splitting.
            If *None*, no split step is performed.
        seed: Random seed.
        verbose: Print progress.
    """

    def __init__(
        self,
        source_dirs: Optional[Sequence[str | Path]] = None,
        output_root: str | Path = "segregated_datasets",
        dataset_name: str = "aircraft",
        augment_count: int = 30000,
        doubles_count: int = 5200,
        train_epochs: int = 150,
        train_batch_size: int = 64,
        train_lr: float = 1e-4,
        patience: int = 5,
        img_h: int = 48,
        img_w: int = 128,
        split_keywords: Optional[Dict[str, str]] = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.source_dirs = [Path(d) for d in source_dirs] if source_dirs else []
        self.output_root = Path(output_root).resolve()
        self.dataset_name = dataset_name
        self.augment_count = augment_count
        self.doubles_count = doubles_count
        self.train_epochs = train_epochs
        self.train_batch_size = train_batch_size
        self.train_lr = train_lr
        self.patience = patience
        self.img_h = img_h
        self.img_w = img_w
        self.split_keywords = split_keywords
        self.seed = seed
        self.verbose = verbose

        # Derived paths
        self.dataset_dir = self.output_root / self.dataset_name
        self.augmented_dir = self.output_root / f"{self.dataset_name}_augmented"
        self.model_output_dir = self.output_root / f"{self.dataset_name}_model"

        self.result = PipelineResult()

    # -- Individual steps ---------------------------------------------------

    def step_split(self) -> SplitResult:
        """Step 1: Split source datasets by filename keywords."""
        if not self.source_dirs:
            raise ValueError("source_dirs required for split step.")
        if not self.split_keywords:
            raise ValueError("split_keywords required for split step.")

        if self.verbose:
            print("=" * 60)
            print("STEP 1: Splitting datasets")
            print("=" * 60)

        self.result.split = split_datasets(
            source_dirs=self.source_dirs,
            output_root=str(self.output_root),
            keywords=self.split_keywords,
        )

        if self.verbose:
            print(f"  Scanned: {self.result.split.seen} images")
            for cat, cnt in self.result.split.categories.items():
                print(f"  {cat}: {cnt}")

        return self.result.split

    def step_augment(self, source_dir: Optional[str | Path] = None) -> AugmentResult:
        """
        Step 2: Generate synthetic augmented images.

        Args:
            source_dir: Override the dataset to augment.  Defaults to
                ``output_root/dataset_name``.
        """
        src = Path(source_dir) if source_dir else self.dataset_dir

        if self.verbose:
            print("\n" + "=" * 60)
            print("STEP 2: Augmenting dataset")
            print("=" * 60)

        self.result.augment = shuffle_augment(
            source_dir=str(src),
            output_dir=str(self.augmented_dir),
            num_output=self.augment_count,
            doubles=self.doubles_count,
            seed=self.seed,
            verbose=self.verbose,
        )
        return self.result.augment

    def step_merge(
        self,
        sources: Optional[Sequence[str | Path]] = None,
        output_name: str = "overall_merged",
    ) -> MergeResult:
        """
        Step 3: Merge multiple datasets.

        Args:
            sources: Explicit source directories.  If *None*, merges all
                subdirectories of ``output_root``.
            output_name: Name for the merged output.
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("STEP 3: Merging datasets")
            print("=" * 60)

        self.result.merge = merge_datasets(
            source_root=str(self.output_root),
            sources=[str(s) for s in sources] if sources else None,
            output_name=output_name,
            overwrite=True,
        )

        if self.verbose:
            print(f"  Merged {self.result.merge.sources_merged} sources")
            print(f"  Total images: {self.result.merge.total_images}")

        return self.result.merge

    def step_rebalance(
        self,
        dataset_dir: Optional[str | Path] = None,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> RebalanceResult:
        """
        Step 4: Rebalance train/valid/test splits.

        Args:
            dataset_dir: Dataset to rebalance.  Defaults to the augmented dir.
        """
        target = Path(dataset_dir) if dataset_dir else self.augmented_dir

        if self.verbose:
            print("\n" + "=" * 60)
            print("STEP 4: Rebalancing splits")
            print("=" * 60)

        self.result.rebalance = rebalance_splits(
            dataset_dir=str(target),
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            seed=self.seed,
        )

        if self.verbose:
            for split, (imgs, lbls) in self.result.rebalance.split_counts.items():
                print(f"  {split}: {imgs} images, {lbls} labels")

        return self.result.rebalance

    def step_train(
        self,
        data_dir: Optional[str | Path] = None,
        **kwargs,
    ) -> TrainHistory:
        """
        Step 5: Train the OCR model.

        Args:
            data_dir: Training dataset.  Defaults to the augmented dir.
            **kwargs: Extra args forwarded to :meth:`OCRTrainer.train`.
        """
        target = Path(data_dir) if data_dir else self.augmented_dir

        if self.verbose:
            print("\n" + "=" * 60)
            print("STEP 5: Training model")
            print("=" * 60)

        self._trainer = OCRTrainer(
            data_dir=str(target),
            output_dir=str(self.model_output_dir),
            img_h=self.img_h,
            img_w=self.img_w,
            seed=self.seed,
        )

        train_kwargs = {
            "epochs": self.train_epochs,
            "batch_size": self.train_batch_size,
            "lr": self.train_lr,
            "patience": self.patience,
            "verbose": self.verbose,
        }
        train_kwargs.update(kwargs)

        self.result.train = self._trainer.train(**train_kwargs)
        self.result.model_path = str(self.model_output_dir / "best_model.guruocr")
        return self.result.train

    def step_evaluate(self, split: str = "test") -> EvalResult:
        """
        Step 6: Evaluate the trained model.

        Args:
            split: Dataset split to evaluate (``"test"``, ``"valid"``).
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("STEP 6: Evaluating model")
            print("=" * 60)

        if not hasattr(self, "_trainer") or self._trainer is None:
            # Create a trainer that loads the saved model
            target = self.augmented_dir
            self._trainer = OCRTrainer(
                data_dir=str(target),
                output_dir=str(self.model_output_dir),
                img_h=self.img_h,
                img_w=self.img_w,
            )

        self.result.evaluate = self._trainer.evaluate(split=split, verbose=self.verbose)

        # Also save training plots if available
        if self.result.train:
            self._trainer.plot_results()

        return self.result.evaluate

    # -- All-in-one ---------------------------------------------------------

    def run_all(
        self,
        skip_split: bool = False,
        skip_merge: bool = True,
        skip_rebalance: bool = True,
    ) -> PipelineResult:
        """
        Run the entire pipeline from start to finish.

        Args:
            skip_split: Skip the dataset splitting step.
            skip_merge: Skip the dataset merging step.
            skip_rebalance: Skip rebalancing after augmentation.

        Returns:
            :class:`PipelineResult` with all step results.
        """
        if self.verbose:
            print("╔" + "═" * 58 + "╗")
            print("║  GuruLearn OCR Pipeline                                  ║")
            print("╚" + "═" * 58 + "╝")

        # 1. Split
        if not skip_split and self.source_dirs and self.split_keywords:
            self.step_split()

        # 2. Augment
        self.step_augment()

        # 3. Merge
        if not skip_merge:
            self.step_merge()

        # 4. Rebalance
        if not skip_rebalance:
            self.step_rebalance()

        # 5. Train
        self.step_train()

        # 6. Evaluate
        self.step_evaluate()

        if self.verbose:
            print("\n" + "╔" + "═" * 58 + "╗")
            print("║  Pipeline Complete!                                      ║")
            print("╚" + "═" * 58 + "╝")
            if self.result.evaluate:
                print(f"  Final accuracy: {self.result.evaluate.accuracy * 100:.2f}%")
            print(f"  Model saved:    {self.result.model_path}")
            print(f"\n  Quick inference:")
            print(f'    from gurulearn.ocr import OCRPredictor')
            print(f'    p = OCRPredictor("{self.result.model_path}")')
            print(f'    print(p.predict("image.jpg").text)')

        return self.result

    def get_predictor(self) -> OCRPredictor:
        """
        Return an :class:`OCRPredictor` loaded with the best trained model.

        Convenience method to go directly from training to inference.
        """
        model_path = self.model_output_dir / "best_model.guruocr"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run training first."
            )
        return OCRPredictor(str(model_path))
