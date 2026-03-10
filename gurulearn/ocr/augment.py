"""
Shuffle-sequence augmentation for YOLO OCR datasets.

Generates synthetic training images by extracting character crops from
existing YOLO-labelled images and composing new random sequences.
Supports double-letter injection for CTC training.

Refactored from shuffle_sequencev3.py.
"""

from __future__ import annotations

import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .data import load_class_names, IMAGE_SUFFIXES, SPLITS

# -- Defaults ---------------------------------------------------------------
_OUTPUT_W, _OUTPUT_H = 512, 64
_MIN_CHARS, _MAX_CHARS = 5, 7
_CHAR_PAD = 3
_BG_RANGE = (220, 255)

_SPACING_TOUCHING = 0.15
_SPACING_TIGHT = 0.40
_SPACING_NORMAL = 0.30
_SPACING_WIDE = 0.10

_SCALE_JITTER_LO = 0.80
_SCALE_JITTER_HI = 1.10


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _load_crops(
    img_dir: Path,
    lbl_dir: Path,
    num_classes: int,
    max_source: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Dict[int, list], Dict[int, int]]:
    """Load character crops from YOLO-labelled images."""
    crops_by_class: Dict[int, list] = defaultdict(list)
    source_counts: Dict[int, int] = defaultdict(int)

    img_files = sorted(
        [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    )
    if max_source:
        img_files = img_files[:max_source]

    total = len(img_files)
    if verbose:
        print(f"  Loading crops from {total} source images ...")

    for idx, img_path in enumerate(img_files):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        with open(lbl_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = max(0, int((cx - bw / 2) * iw))
            y1 = max(0, int((cy - bh / 2) * ih))
            x2 = min(iw, int((cx + bw / 2) * iw))
            y2 = min(ih, int((cy + bh / 2) * ih))
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            if cls_id < 0 or cls_id >= num_classes:
                continue
            crops_by_class[cls_id].append(img[y1:y2, x1:x2].copy())
            source_counts[cls_id] += 1

        if verbose and ((idx + 1) % 500 == 0 or idx + 1 == total):
            print(f"    ... {idx + 1}/{total}")

    return dict(crops_by_class), dict(source_counts)


def _build_double_schedule(num_doubles: int, available: List[int]) -> list:
    per_class = num_doubles // len(available)
    remainder = num_doubles % len(available)
    assignments = []
    shuffled = list(available)
    random.shuffle(shuffled)
    for i, cid in enumerate(shuffled):
        count = per_class + (1 if i < remainder else 0)
        assignments.extend([cid] * count)
    random.shuffle(assignments)

    schedule = []
    for dbl_class in assignments:
        seq_len = random.randint(max(_MIN_CHARS, 3), _MAX_CHARS)
        other_slots = seq_len - 2
        other = [random.choice(available) for _ in range(other_slots)]
        dbl_pos = random.randint(0, seq_len - 2)
        class_ids = list(other)
        class_ids.insert(dbl_pos, dbl_class)
        class_ids.insert(dbl_pos + 1, dbl_class)
        schedule.append((seq_len, class_ids, dbl_pos))
    return schedule


def _build_normal_schedule(
    num_normal: int,
    crops_by_class: Dict[int, list],
    source_counts: Dict[int, int],
) -> list:
    available = sorted(crops_by_class.keys())
    if not available:
        raise RuntimeError("No crops found!")

    lengths = []
    num_per_len = num_normal // (_MAX_CHARS - _MIN_CHARS + 1)
    rem = num_normal % (_MAX_CHARS - _MIN_CHARS + 1)
    for length in range(_MIN_CHARS, _MAX_CHARS + 1):
        count = num_per_len + (1 if length - _MIN_CHARS < rem else 0)
        lengths.extend([length] * count)
    random.shuffle(lengths)

    total_slots = sum(lengths)
    max_count = max(source_counts.values()) if source_counts else 0

    # Deficit pool
    deficit_pool: list = []
    for cid in available:
        cnt = source_counts.get(cid, 0)
        if cnt < max_count:
            deficit_pool.extend([cid] * (max_count - cnt))
    random.shuffle(deficit_pool)
    deficit_pool = deficit_pool[:total_slots]

    # Uniform pool
    phase2 = total_slots - len(deficit_pool)
    if phase2 > 0:
        reps = math.ceil(phase2 / len(available))
        uniform = []
        for cid in available:
            uniform.extend([cid] * reps)
        random.shuffle(uniform)
        uniform = uniform[:phase2]
        while len(uniform) < phase2:
            uniform.append(random.choice(available))
        random.shuffle(uniform)
    else:
        uniform = []

    pool = deficit_pool + uniform
    random.shuffle(pool)

    schedule = []
    offset = 0
    for length in lengths:
        schedule.append((length, pool[offset:offset + length]))
        offset += length
    return schedule


def _compose_image(
    class_ids: List[int],
    crops_by_class: Dict[int, list],
    double_position: Optional[int] = None,
    output_w: int = _OUTPUT_W,
    output_h: int = _OUTPUT_H,
) -> Tuple[np.ndarray, list]:
    """Compose a synthetic image from character crops."""
    n = len(class_ids)
    bg_val = random.randint(*_BG_RANGE)
    canvas = np.full((output_h, output_w, 3), bg_val, dtype=np.uint8)

    target_h = output_h - 2 * _CHAR_PAD
    resized = []
    for cid in class_ids:
        crop = random.choice(crops_by_class[cid])
        ch, cw = crop.shape[:2]
        jitter = random.uniform(_SCALE_JITTER_LO, _SCALE_JITTER_HI)
        new_h = max(4, min(output_h - 2, int(target_h * jitter)))
        scale = new_h / ch
        new_w = max(4, int(cw * scale))
        resized.append((cid, cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)))

    # Spacing
    r = random.random()
    if r < _SPACING_TOUCHING:
        gap_lo, gap_hi = 0, 1
    elif r < _SPACING_TOUCHING + _SPACING_TIGHT:
        gap_lo, gap_hi = 1, 2
    elif r < _SPACING_TOUCHING + _SPACING_TIGHT + _SPACING_NORMAL:
        gap_lo, gap_hi = _CHAR_PAD, _CHAR_PAD * 2
    else:
        gap_lo, gap_hi = _CHAR_PAD * 2, _CHAR_PAD * 4

    gaps = [random.randint(gap_lo, gap_hi) for _ in range(n + 1)]
    if double_position is not None and double_position + 1 < n:
        gaps[double_position + 1] = random.randint(0, 1)

    total_char_w = sum(img.shape[1] for _, img in resized)
    total_w = total_char_w + sum(gaps)
    if total_w > output_w:
        min_gap = 0 if double_position is not None else gap_lo
        avail = output_w - (n + 1) * min_gap
        if avail < n * 4:
            avail = n * 4
        sd = avail / max(total_char_w, 1)
        new_resized = []
        for cid, img in resized:
            h, w = img.shape[:2]
            nw = max(4, int(w * sd))
            nh = max(4, min(int(h * sd), output_h - 2))
            new_resized.append((cid, cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)))
        resized = new_resized
        gaps = [min_gap] * (n + 1)
        if double_position is not None and double_position + 1 < n:
            gaps[double_position + 1] = 0

    labels = []
    x_cursor = gaps[0]
    for i, (cid, res_img) in enumerate(resized):
        rh, rw = res_img.shape[:2]
        slack = output_h - rh
        y_off = random.randint(0, slack) if slack > 0 else 0
        y_off = max(0, min(y_off, output_h - rh))
        x_off = max(0, min(int(x_cursor), output_w - rw))
        pw = min(rw, output_w - x_off)
        ph = min(rh, output_h - y_off)
        if pw > 0 and ph > 0:
            canvas[y_off:y_off + ph, x_off:x_off + pw] = res_img[:ph, :pw]
        cx = (x_off + pw / 2) / output_w
        cy = (y_off + ph / 2) / output_h
        bw = pw / output_w
        bh = ph / output_h
        labels.append((cid, cx, cy, bw, bh))
        x_cursor += rw + gaps[i + 1]

    return canvas, labels


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

@dataclass
class AugmentResult:
    """Result of a ``shuffle_augment`` operation."""
    total_synthetic: int = 0
    doubles_count: int = 0
    normal_count: int = 0
    original_train_copied: int = 0
    output_dir: str = ""
    class_distribution: Dict[str, int] = field(default_factory=dict)


def shuffle_augment(
    source_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    num_output: int = 30000,
    doubles: int = 5200,
    seed: int = 42,
    max_source: Optional[int] = None,
    copy_original: bool = True,
    verbose: bool = True,
) -> AugmentResult:
    """
    Generate synthetic shuffled-sequence images with double-letter support.

    Reads character crops from the *source_dir* training split and composes
    new random-sequence images.  Auto-discovers classes from ``data.yaml``.

    Args:
        source_dir: YOLO dataset root containing ``train/images``, ``train/labels``,
            and ``data.yaml``.
        output_dir: Where to write the augmented dataset.  Defaults to
            ``<source_dir>_augmented``.
        num_output: Total number of synthetic images to create.
        doubles: Number of images that must contain a doubled character.
        seed: Random seed.
        max_source: Limit number of source images (for testing).
        copy_original: If *True*, copy original train/valid/test into output.
        verbose: Print progress.

    Returns:
        An :class:`AugmentResult` summary.
    """
    source_dir = Path(source_dir).resolve()
    if output_dir is None:
        output_dir = source_dir.parent / f"{source_dir.name}_augmented"
    else:
        output_dir = Path(output_dir).resolve()

    random.seed(seed)
    np.random.seed(seed)

    if doubles > num_output:
        raise ValueError(f"doubles ({doubles}) cannot exceed num_output ({num_output})")

    # Discover classes
    class_names = load_class_names(source_dir / "data.yaml")
    num_classes = len(class_names)
    if verbose:
        print(f"  Classes ({num_classes}): {class_names}")

    # Create output dirs
    out_train_img = output_dir / "train" / "images"
    out_train_lbl = output_dir / "train" / "labels"
    out_train_img.mkdir(parents=True, exist_ok=True)
    out_train_lbl.mkdir(parents=True, exist_ok=True)

    train_img = source_dir / "train" / "images"
    train_lbl = source_dir / "train" / "labels"

    # 1. Load crops
    crops_by_class, source_counts = _load_crops(
        train_img, train_lbl, num_classes, max_source, verbose
    )
    if not crops_by_class:
        raise RuntimeError("No crops extracted — check paths and label files.")

    available = sorted(crops_by_class.keys())
    num_doubles = doubles
    num_normal = num_output - num_doubles

    if verbose:
        print(f"  Generating {num_output} synthetic ({num_doubles} double + {num_normal} normal)")

    # 2. Schedules
    double_sched = _build_double_schedule(num_doubles, available)
    normal_sched = _build_normal_schedule(num_normal, crops_by_class, source_counts)

    # 3. Compose
    idx = 0
    for _, class_ids, dbl_pos in double_sched:
        canvas, labels = _compose_image(class_ids, crops_by_class, double_position=dbl_pos)
        fname = f"synth_dbl_{idx:06d}"
        cv2.imwrite(str(out_train_img / f"{fname}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(out_train_lbl / f"{fname}.txt", "w") as f:
            for cid, cx, cy, bw, bh in labels:
                f.write(f"{cid} {cx:.10f} {cy:.10f} {bw:.10f} {bh:.10f}\n")
        idx += 1
        if verbose and idx % 2000 == 0:
            print(f"    ... {idx}/{num_output}")

    for _, class_ids in normal_sched:
        canvas, labels = _compose_image(class_ids, crops_by_class)
        fname = f"synth_nrm_{idx:06d}"
        cv2.imwrite(str(out_train_img / f"{fname}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(out_train_lbl / f"{fname}.txt", "w") as f:
            for cid, cx, cy, bw, bh in labels:
                f.write(f"{cid} {cx:.10f} {cy:.10f} {bw:.10f} {bh:.10f}\n")
        idx += 1
        if verbose and idx % 2000 == 0:
            print(f"    ... {idx}/{num_output}")

    # 4. Copy original data
    orig_count = 0
    if copy_original:
        for f in train_img.iterdir():
            if f.is_file():
                shutil.copy2(str(f), str(out_train_img / f.name))
                orig_count += 1
        for f in train_lbl.iterdir():
            if f.is_file():
                shutil.copy2(str(f), str(out_train_lbl / f.name))

        # Copy valid / test
        for split in ("valid", "test"):
            src_img = source_dir / split / "images"
            src_lbl = source_dir / split / "labels"
            dst_img = output_dir / split / "images"
            dst_lbl = output_dir / split / "labels"
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)
            if src_img.exists():
                for f in src_img.iterdir():
                    if f.is_file():
                        shutil.copy2(str(f), str(dst_img / f.name))
            if src_lbl.exists():
                for f in src_lbl.iterdir():
                    if f.is_file():
                        shutil.copy2(str(f), str(dst_lbl / f.name))

    # 5. Write data.yaml
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n")
        f.write(f"\nnc: {num_classes}\n")
        f.write(f"names: {class_names}\n")

    # Collect distribution
    synth_counts: Dict[str, int] = defaultdict(int)
    for _, class_ids, _ in double_sched:
        for cid in class_ids:
            synth_counts[class_names[cid]] += 1
    for _, class_ids in normal_sched:
        for cid in class_ids:
            synth_counts[class_names[cid]] += 1

    if verbose:
        print(f"  Done! Output: {output_dir}")

    return AugmentResult(
        total_synthetic=num_output,
        doubles_count=num_doubles,
        normal_count=num_normal,
        original_train_copied=orig_count,
        output_dir=str(output_dir),
        class_distribution=dict(synth_counts),
    )
