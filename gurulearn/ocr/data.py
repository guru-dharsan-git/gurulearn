"""
Dataset utilities for YOLO-format OCR datasets.

Provides functions to split, merge, and rebalance YOLO datasets.
Refactored from split_datasets.py, merge_segregated_datasets.py,
and rebalance_yolo_split.py.
"""

from __future__ import annotations

import ast
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------

def load_class_names(data_yaml: Path | str) -> List[str]:
    """
    Parse a YOLO ``data.yaml`` and return the list of class names.

    Args:
        data_yaml: Path to the ``data.yaml`` file.

    Returns:
        Ordered list of class-name strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If class names cannot be parsed.
    """
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    for raw in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("names:"):
            parsed = ast.literal_eval(line.split(":", 1)[1].strip())
            if isinstance(parsed, list):
                return [str(x) for x in parsed]

    raise ValueError(f"Could not parse class names from: {data_yaml}")


def _write_data_yaml(root: Path, names: List[str]) -> None:
    """Write a YOLO ``data.yaml`` with train/val/test paths and class info."""
    content = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names!r}\n"
    )
    (root / "data.yaml").write_text(content, encoding="utf-8")


def _ensure_split_dirs(root: Path) -> None:
    for split in SPLITS:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)


def _unique_dest(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        candidate = directory / f"{stem}__dup{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
#  split_datasets
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """Result of a ``split_datasets`` operation."""
    seen: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    missing_labels: int = 0
    removed_from_sources: int = 0
    output_dirs: List[str] = field(default_factory=list)


def split_datasets(
    source_dirs: Sequence[str | Path],
    output_root: str | Path = "segregated_datasets",
    keywords: Optional[Dict[str, str]] = None,
    remove_from_sources: bool = False,
) -> SplitResult:
    """
    Split YOLO datasets into category-based sub-datasets by filename keyword.

    For every image file, the stem is checked for keyword matches.  Matched
    files go to ``output_root/<category>/``; unmatched files go to
    ``output_root/remaining_merged/``.

    Args:
        source_dirs: One or more YOLO dataset root directories.
        output_root: Where the segregated datasets will be created.
        keywords: Mapping of ``{keyword: category_name}``.  Defaults to
            ``{"aircraft": "aircraft", "supplier": "suppliers"}``.
        remove_from_sources: Delete matched files from the source directory
            after copying.

    Returns:
        A :class:`SplitResult` with counts per category.
    """
    source_dirs = [Path(d).resolve() for d in source_dirs]
    output_root = Path(output_root).resolve()

    if keywords is None:
        keywords = {"aircraft": "aircraft", "supplier": "suppliers"}

    # Collect category names + "remaining_merged"
    category_names = list(dict.fromkeys(keywords.values()))
    all_buckets = category_names + ["remaining_merged"]

    # Create output dirs
    bucket_roots: Dict[str, Path] = {}
    for bucket in all_buckets:
        bucket_roots[bucket] = output_root / bucket
        _ensure_split_dirs(bucket_roots[bucket])

    # Attempt to copy class names from first usable source
    nc, names = 0, []
    for sd in source_dirs:
        try:
            names = load_class_names(sd / "data.yaml")
            nc = len(names)
            break
        except (FileNotFoundError, ValueError):
            pass

    if names:
        for bucket in all_buckets:
            _write_data_yaml(bucket_roots[bucket], names)

    result = SplitResult(output_dirs=[str(bucket_roots[b]) for b in all_buckets])
    cat_counts: Dict[str, int] = {b: 0 for b in all_buckets}

    def _classify(stem_lower: str) -> str:
        for kw, cat in keywords.items():
            if kw in stem_lower:
                return cat
        return "remaining_merged"

    for sd in source_dirs:
        tag = sd.name.replace(" ", "_")
        for split in SPLITS:
            images_dir = sd / split / "images"
            labels_dir = sd / split / "labels"
            if not images_dir.exists():
                continue

            for img in sorted(images_dir.iterdir()):
                if not img.is_file() or img.suffix.lower() not in IMAGE_SUFFIXES:
                    continue

                result.seen += 1
                cat = _classify(img.stem.lower())
                cat_counts[cat] += 1

                safe_stem = f"{tag}__{split}__{img.stem}"
                dst_img_dir = bucket_roots[cat] / split / "images"
                dst_lbl_dir = bucket_roots[cat] / split / "labels"

                image_dst = _unique_dest(dst_img_dir, safe_stem, img.suffix.lower())
                _copy_file(img, image_dst)

                label_src = labels_dir / f"{img.stem}.txt"
                label_dst = dst_lbl_dir / f"{image_dst.stem}.txt"
                if label_src.exists():
                    _copy_file(label_src, label_dst)
                else:
                    result.missing_labels += 1
                    label_dst.parent.mkdir(parents=True, exist_ok=True)
                    label_dst.write_text("", encoding="utf-8")

                if remove_from_sources and cat != "remaining_merged":
                    img.unlink(missing_ok=False)
                    if label_src.exists():
                        label_src.unlink(missing_ok=True)
                    result.removed_from_sources += 1

    result.categories = cat_counts
    return result


# ---------------------------------------------------------------------------
#  merge_datasets
# ---------------------------------------------------------------------------

@dataclass
class MergeResult:
    """Result of a ``merge_datasets`` operation."""
    sources_merged: int = 0
    total_images: int = 0
    total_labels: int = 0
    split_counts: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    output_dir: str = ""


def merge_datasets(
    source_root: str | Path = "segregated_datasets",
    sources: Optional[Sequence[str | Path]] = None,
    output_name: str = "overall_merged",
    overwrite: bool = False,
) -> MergeResult:
    """
    Merge multiple YOLO datasets (same class set) into a single dataset.

    Args:
        source_root: Root that contains the datasets to merge.
        sources: Explicit list of dataset directories.  If *None*, every
            subdirectory of *source_root* (except *output_name*) is used.
        output_name: Name of the merged output folder under *source_root*.
        overwrite: Delete existing output before merging.

    Returns:
        A :class:`MergeResult` with image/label counts per split.
    """
    source_root = Path(source_root).resolve()
    out_root = source_root / output_name

    if sources:
        source_list = sorted(Path(s).resolve() for s in sources)
    else:
        source_list = sorted(
            p.resolve() for p in source_root.iterdir()
            if p.is_dir() and p.name != output_name
        )

    if not source_list:
        raise ValueError("No source datasets found.")

    # Verify class names match
    names = load_class_names(source_list[0] / "data.yaml")
    for src in source_list[1:]:
        other = load_class_names(src / "data.yaml")
        if other != names:
            raise ValueError(f"Class names mismatch in {src}")

    if out_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {out_root}. Pass overwrite=True to rebuild."
            )
        shutil.rmtree(out_root)

    _ensure_split_dirs(out_root)
    _write_data_yaml(out_root, names)

    total_images = 0
    total_labels = 0

    for src in source_list:
        tag = src.name
        for split in SPLITS:
            src_imgs = src / split / "images"
            src_lbls = src / split / "labels"
            dst_imgs = out_root / split / "images"
            dst_lbls = out_root / split / "labels"
            if not src_imgs.exists():
                continue

            for img in sorted(src_imgs.iterdir()):
                if not img.is_file() or img.suffix.lower() not in IMAGE_SUFFIXES:
                    continue

                dst_name = f"{tag}__{img.name}"
                dst_img = _unique_dest(dst_imgs, Path(dst_name).stem, img.suffix.lower())
                shutil.copy2(img, dst_img)
                total_images += 1

                src_lbl = src_lbls / f"{img.stem}.txt"
                dst_lbl = dst_lbls / f"{dst_img.stem}.txt"
                if src_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)
                else:
                    dst_lbl.write_text("", encoding="utf-8")
                total_labels += 1

    # Count per split
    split_counts = {}
    for split in SPLITS:
        imgs = out_root / split / "images"
        lbls = out_root / split / "labels"
        ic = sum(1 for p in imgs.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        lc = sum(1 for p in lbls.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
        split_counts[split] = (ic, lc)

    return MergeResult(
        sources_merged=len(source_list),
        total_images=total_images,
        total_labels=total_labels,
        split_counts=split_counts,
        output_dir=str(out_root),
    )


# ---------------------------------------------------------------------------
#  rebalance_splits
# ---------------------------------------------------------------------------

@dataclass
class _Record:
    image: Path
    label: Optional[Path]


@dataclass
class RebalanceResult:
    """Result of a ``rebalance_splits`` operation."""
    total_images: int = 0
    split_counts: Dict[str, Tuple[int, int]] = field(default_factory=dict)


def rebalance_splits(
    dataset_dir: str | Path,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> RebalanceResult:
    """
    Rebalance a YOLO dataset's train/valid/test split **in-place**.

    All images across splits are pooled, shuffled, and re-distributed
    according to the given ratios.

    Args:
        dataset_dir: Root of the YOLO dataset (must contain train/valid/test).
        train_ratio: Fraction for training.
        valid_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`RebalanceResult` with final counts per split.
    """
    dataset_dir = Path(dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    _ensure_split_dirs(dataset_dir)

    # Collect all records
    records: List[_Record] = []
    for split in SPLITS:
        imgs = dataset_dir / split / "images"
        lbls = dataset_dir / split / "labels"
        if not imgs.exists():
            continue
        for img in sorted(imgs.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            lbl = lbls / f"{img.stem}.txt"
            records.append(_Record(image=img, label=lbl if lbl.exists() else None))

    if not records:
        raise ValueError("No images found in dataset.")

    rnd = random.Random(seed)
    rnd.shuffle(records)
    total = len(records)

    # Compute counts
    ratios = (train_ratio, valid_ratio, test_ratio)
    ratio_sum = sum(ratios)
    normalized = tuple(r / ratio_sum for r in ratios)
    raw = [total * r for r in normalized]
    base = [int(x) for x in raw]
    remainder = total - sum(base)
    frac_order = sorted(range(3), key=lambda i: (raw[i] - base[i]), reverse=True)
    for idx in frac_order[:remainder]:
        base[idx] += 1
    train_n, valid_n, test_n = base

    # Move to staging
    staging = dataset_dir / ".resplit_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging_imgs = staging / "images"
    staging_lbls = staging / "labels"
    staging_imgs.mkdir(parents=True)
    staging_lbls.mkdir(parents=True)

    staged: List[_Record] = []
    for rec in records:
        dst_img = staging_imgs / rec.image.name
        if dst_img.exists():
            dst_img = _unique_dest(staging_imgs, rec.image.stem, rec.image.suffix)
        shutil.move(str(rec.image), str(dst_img))
        dst_lbl = None
        if rec.label and rec.label.exists():
            dst_lbl = staging_lbls / f"{dst_img.stem}.txt"
            shutil.move(str(rec.label), str(dst_lbl))
        staged.append(_Record(image=dst_img, label=dst_lbl))

    # Clear original split dirs
    for split in SPLITS:
        for sub in ("images", "labels"):
            d = dataset_dir / split / sub
            if d.exists():
                for p in d.iterdir():
                    if p.is_file():
                        p.unlink()

    # Redistribute
    assignments = (
        ("train", 0, train_n),
        ("valid", train_n, train_n + valid_n),
        ("test", train_n + valid_n, train_n + valid_n + test_n),
    )
    for split, start, end in assignments:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        for rec in staged[start:end]:
            dst_img = img_dir / rec.image.name
            if dst_img.exists():
                dst_img = _unique_dest(img_dir, rec.image.stem, rec.image.suffix)
            shutil.move(str(rec.image), str(dst_img))
            if rec.label and rec.label.exists():
                shutil.move(str(rec.label), str(lbl_dir / f"{dst_img.stem}.txt"))

    # Cleanup
    if staging.exists():
        shutil.rmtree(staging)

    # Report
    split_counts = {}
    for split in SPLITS:
        imgs = dataset_dir / split / "images"
        lbls = dataset_dir / split / "labels"
        ic = sum(1 for p in imgs.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        lc = sum(1 for p in lbls.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
        split_counts[split] = (ic, lc)

    return RebalanceResult(total_images=total, split_counts=split_counts)
