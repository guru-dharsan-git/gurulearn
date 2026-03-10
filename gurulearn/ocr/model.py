"""
VGG-BiLSTM OCR model, dataset, loss, and trainer.

Provides a complete training pipeline for CTC-based OCR on YOLO-format
datasets.  Auto-discovers classes from ``data.yaml`` so it works with
any character set, not just A–Z.

Refactored from ss-supplier.ipynb.
"""

from __future__ import annotations

import ast
import json
import os
import random
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .data import load_class_names, IMAGE_SUFFIXES

# ---------------------------------------------------------------------------
#  .guruocr format helpers
# ---------------------------------------------------------------------------

_GURUOCR_VERSION = 1


def save_guruocr(
    path: str | Path,
    state_dict: dict,
    class_names: List[str],
    img_h: int,
    img_w: int,
    hidden: int,
    num_layers: int,
) -> Path:
    """
    Save a trained model as a ``.guruocr`` file.

    The file is a zip archive containing:
    - ``weights.pth`` — serialised state dict
    - ``metadata.json`` — class names, image dims, model config
    """
    import torch  # lazy

    path = Path(path)
    if path.suffix != ".guruocr":
        path = path.with_suffix(".guruocr")

    with tempfile.TemporaryDirectory() as tmp:
        weights_path = Path(tmp) / "weights.pth"
        meta_path = Path(tmp) / "metadata.json"

        torch.save(state_dict, weights_path)

        meta = {
            "version": _GURUOCR_VERSION,
            "class_names": class_names,
            "num_classes": len(class_names),
            "img_h": img_h,
            "img_w": img_w,
            "hidden": hidden,
            "num_layers": num_layers,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(weights_path, "weights.pth")
            zf.write(meta_path, "metadata.json")

    return path


def load_guruocr(path: str | Path) -> Tuple[dict, dict]:
    """
    Load a ``.guruocr`` file.

    Returns:
        (state_dict, metadata_dict)
    """
    import torch  # lazy

    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        with tempfile.TemporaryDirectory() as tmp:
            zf.extractall(tmp)
            state_dict = torch.load(
                Path(tmp) / "weights.pth",
                map_location="cpu",
                weights_only=True,
            )
            meta = json.loads((Path(tmp) / "metadata.json").read_text(encoding="utf-8"))

    return state_dict, meta


# ---------------------------------------------------------------------------
#  VGG_OCR Model
# ---------------------------------------------------------------------------

def _build_vgg_ocr(num_classes: int, hidden: int = 256, num_layers: int = 3):
    """Build a VGG-BiLSTM OCR model.  Returns an ``nn.Module``."""
    import torch.nn as nn
    import torch.nn.functional as F

    class VGG_OCR(nn.Module):
        """VGG-style CNN + BiLSTM for CTC-based OCR."""

        def __init__(self, nc: int = num_classes, hid: int = hidden, nlayers: int = num_layers):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),

                nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),

                nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=(3, 1)),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
            )
            self.rnn = nn.LSTM(
                input_size=512,
                hidden_size=hid,
                num_layers=nlayers,
                batch_first=True,
                bidirectional=True,
                dropout=0.3 if nlayers > 1 else 0.0,
            )
            self.fc = nn.Linear(hid * 2, nc)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

        def forward(self, x):
            conv = self.cnn(x)
            conv = conv.squeeze(2).permute(0, 2, 1)
            rnn_out, _ = self.rnn(conv)
            out = self.fc(rnn_out)
            return F.log_softmax(out, dim=2)

    return VGG_OCR(num_classes, hidden, num_layers)


# Make VGG_OCR importable at module level
class VGG_OCR:
    """
    Factory for VGG-BiLSTM OCR models.

    Use ``VGG_OCR(num_classes, hidden, num_layers)`` to create a model.
    The actual ``nn.Module`` is returned.
    """

    def __new__(cls, num_classes: int, hidden: int = 256, num_layers: int = 3):
        return _build_vgg_ocr(num_classes, hidden, num_layers)


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------

def _parse_yolo_to_ids(label_path: str, num_tokens: int) -> List[int]:
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                cls = int(parts[0])
                x = float(parts[1])
                if 0 <= cls < num_tokens:
                    boxes.append((cls, x))
    boxes.sort(key=lambda b: b[1])
    return [cls for cls, _ in boxes]


def _load_split_samples(
    data_dir: Path,
    split: str,
    num_tokens: int,
) -> List[Dict[str, Any]]:
    """Load samples for a single split."""
    img_dir = data_dir / split / "images"
    lbl_dir = data_dir / split / "labels"
    if not img_dir.exists():
        return []

    samples = []
    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        ids = _parse_yolo_to_ids(str(lbl_path), num_tokens)
        if ids:
            samples.append({"img": str(img_path), "ids": ids})
    return samples


# ---------------------------------------------------------------------------
#  Weighted CTC Loss
# ---------------------------------------------------------------------------

class WeightedCTCLoss:
    """
    CTC Loss with optional per-sample weight boosting for focus tokens.

    This is a callable, not an ``nn.Module``, to avoid importing torch at
    module level.
    """

    def __init__(
        self,
        blank: int,
        confused_token_ids: Optional[Set[int]] = None,
        weight_boost: float = 1.5,
    ):
        import torch.nn as nn
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="none")
        self.confused_ids = confused_token_ids or set()
        self.boost = weight_boost

    def __call__(self, log_probs, targets, input_lengths, target_lengths):
        import torch
        losses = self.ctc(log_probs, targets, input_lengths, target_lengths)
        weights = torch.ones_like(losses)
        if self.confused_ids:
            for i in range(targets.size(0)):
                tl = target_lengths[i].item()
                tgt = targets[i, :tl]
                if any(int(d.item()) in self.confused_ids for d in tgt):
                    weights[i] = self.boost
        return (losses * weights).mean()


# ---------------------------------------------------------------------------
#  Decoding & Metrics
# ---------------------------------------------------------------------------

def greedy_decode(output, blank: int) -> List[List[int]]:
    """Greedy CTC decode → list of token-id sequences."""
    preds = []
    for seq in output:
        indices = seq.argmax(dim=1).cpu().numpy().tolist()
        result = []
        prev = -1
        for idx in indices:
            if idx != prev and idx != blank:
                result.append(int(idx))
            prev = idx
        preds.append(result)
    return preds


def _edit_distance(a: List[int], b: List[int]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (
                dp[i - 1][j - 1]
                if a[i - 1] == b[j - 1]
                else 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
            )
    return dp[m][n]


def _compute_metrics(preds, labels, lengths) -> Tuple[float, float]:
    correct = 0
    total_cer = 0.0
    for pred, label, ln in zip(preds, labels, lengths):
        target = [int(l.item()) for l in label[:ln]]
        if pred == target:
            correct += 1
        total_cer += _edit_distance(pred, target) / max(len(target), 1)
    acc = correct / max(len(preds), 1)
    avg_cer = total_cer / max(len(preds), 1)
    return acc, avg_cer


# ---------------------------------------------------------------------------
#  OCRTrainer
# ---------------------------------------------------------------------------

@dataclass
class TrainHistory:
    """Stores per-epoch metrics."""
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)


@dataclass
class EvalResult:
    """Final evaluation results."""
    accuracy: float = 0.0
    cer: float = 0.0
    loss: float = 0.0


class OCRTrainer:
    """
    High-level trainer for VGG-BiLSTM CTC OCR models.

    Reads ``data.yaml`` to auto-discover classes so it works with **any**
    YOLO-format character dataset.

    Example::

        trainer = OCRTrainer("path/to/dataset", "output/")
        trainer.train(epochs=100)
        result = trainer.evaluate()
        trainer.plot_results()
    """

    def __init__(
        self,
        data_dir: str | Path,
        output_dir: str | Path = "ocr_output",
        img_h: int = 48,
        img_w: int = 128,
        hidden: int = 256,
        num_layers: int = 3,
        focus_tokens: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.img_h = img_h
        self.img_w = img_w
        self.hidden = hidden
        self.num_layers = num_layers
        self.seed = seed

        # Discover classes
        self.class_names = load_class_names(self.data_dir / "data.yaml")
        self.num_tokens = len(self.class_names)
        self.id2char = {i: n for i, n in enumerate(self.class_names)}
        self.char2id = {n: i for i, n in self.id2char.items()}
        self.blank = self.num_tokens
        self.num_classes = self.num_tokens + 1  # +1 for CTC blank

        # Focus tokens
        self.focus_token_ids: Set[int] = set()
        if focus_tokens:
            self.focus_token_ids = {self.char2id[t] for t in focus_tokens if t in self.char2id}

        # Will be set during train()
        self.model = None
        self.history = TrainHistory()
        self.device = None

    def _ids_to_text(self, ids: List[int]) -> str:
        parts = []
        for tid in ids:
            tok = self.id2char.get(int(tid), "?")
            parts.append(tok if len(tok) == 1 else f"<{tok}>")
        return "".join(parts)

    def train(
        self,
        epochs: int = 150,
        batch_size: int = 64,
        lr: float = 1e-4,
        patience: int = 5,
        num_workers: int = 2,
        oversample_factor: int = 4,
        weight_boost: float = 1.3,
        verbose: bool = True,
    ) -> TrainHistory:
        """
        Train the OCR model.

        Args:
            epochs: Maximum training epochs.
            batch_size: Batch size.
            lr: Learning rate.
            patience: Early-stopping patience.
            num_workers: DataLoader workers.
            oversample_factor: Oversample factor for focus-token samples.
            weight_boost: CTC loss boost for focus-token samples.
            verbose: Print progress.

        Returns:
            :class:`TrainHistory` with per-epoch metrics.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        try:
            from torch.amp import GradScaler, autocast
            scaler = GradScaler("cuda")
        except ImportError:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        if verbose:
            print(f"Device: {self.device}")
            print(f"Classes ({self.num_tokens}): {self.class_names}")

        # Load data
        train_samples = _load_split_samples(self.data_dir, "train", self.num_tokens)
        valid_samples = _load_split_samples(self.data_dir, "valid", self.num_tokens)

        if not train_samples:
            raise RuntimeError("No training samples found!")
        if verbose:
            print(f"Train: {len(train_samples)}, Valid: {len(valid_samples)}")

        # Build augmentations
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2

            train_aug = A.Compose([
                A.Resize(self.img_h, self.img_w),
                A.ShiftScaleRotate(
                    shift_limit=0.03, scale_limit=0.05, rotate_limit=3,
                    border_mode=cv2.BORDER_REPLICATE, p=0.5,
                ),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
            strong_aug = A.Compose([
                A.Resize(self.img_h, self.img_w),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.08, rotate_limit=5,
                    border_mode=cv2.BORDER_REPLICATE, p=0.6,
                ),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
            val_aug = A.Compose([
                A.Resize(self.img_h, self.img_w),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
        except ImportError:
            raise ImportError(
                "albumentations is required for OCR training. "
                "Install with: pip install albumentations"
            )

        blank = self.blank
        focus_ids = self.focus_token_ids

        # Dataset class (inner to avoid top-level torch import)
        class _OCRDataset(Dataset):
            def __init__(self, samples, transform, strong_transform=None,
                         confused_ids=None, os_factor=1):
                self.samples = samples
                self.transform = transform
                self.strong_transform = strong_transform or transform
                self.confused_ids = confused_ids or set()
                self.indices = []
                self.is_focus = []
                for i, s in enumerate(samples):
                    has_focus = any(t in self.confused_ids for t in s["ids"])
                    if has_focus and os_factor > 1:
                        self.indices.extend([i] * os_factor)
                        self.is_focus.extend([True] * os_factor)
                    else:
                        self.indices.append(i)
                        self.is_focus.append(has_focus)

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                s = self.samples[self.indices[idx]]
                img = cv2.imread(s["img"], cv2.IMREAD_GRAYSCALE)
                img = np.stack([img] * 3, axis=-1)
                tfm = self.strong_transform if self.is_focus[idx] else self.transform
                img = tfm(image=img)["image"][0:1]
                return img, torch.tensor(s["ids"]), len(s["ids"])

        def _collate(batch):
            imgs, labels, lengths = zip(*batch)
            imgs = torch.stack(imgs)
            max_len = max(lengths)
            padded = torch.full((len(labels), max_len), blank, dtype=torch.long)
            for i, (lbl, ln) in enumerate(zip(labels, lengths)):
                padded[i, :ln] = lbl
            return imgs, padded, torch.tensor(lengths)

        train_ds = _OCRDataset(
            train_samples, train_aug, strong_aug, focus_ids, oversample_factor
        )
        val_ds = _OCRDataset(valid_samples, val_aug)

        train_loader = DataLoader(
            train_ds, batch_size, shuffle=True, collate_fn=_collate,
            pin_memory=True, num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_ds, batch_size, shuffle=False, collate_fn=_collate,
            pin_memory=True, num_workers=num_workers,
        )

        # Model
        self.model = _build_vgg_ocr(self.num_classes, self.hidden, self.num_layers)
        self.model.to(self.device)

        if verbose:
            params = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {params:,}")

        criterion = WeightedCTCLoss(
            blank=self.blank, confused_token_ids=focus_ids, weight_boost=weight_boost
        )
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        best_acc = 0.0
        no_improve = 0
        self.history = TrainHistory()

        for epoch in range(epochs):
            # -- Train --
            self.model.train()
            total_loss = 0
            all_preds, all_labels, all_lens = [], [], []

            for imgs, labels, lens in train_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()

                with torch.amp.autocast(self.device.type):
                    out = self.model(imgs)
                    out_t = out.permute(1, 0, 2)
                    input_lens = torch.full(
                        (out_t.size(1),), out_t.size(0), dtype=torch.long
                    )
                    loss = criterion(out_t, labels, input_lens, lens)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

                with torch.no_grad():
                    all_preds.extend(greedy_decode(out, self.blank))
                    all_labels.extend(labels.cpu())
                    all_lens.extend(lens.cpu())

            train_loss = total_loss / max(len(train_loader), 1)
            train_acc, train_cer = _compute_metrics(all_preds, all_labels, all_lens)

            # -- Validate --
            self.model.eval()
            total_loss = 0
            all_preds, all_labels, all_lens = [], [], []

            with torch.no_grad():
                for imgs, labels, lens in val_loader:
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)
                    out = self.model(imgs)
                    out_t = out.permute(1, 0, 2)
                    input_lens = torch.full(
                        (out_t.size(1),), out_t.size(0), dtype=torch.long
                    )
                    loss = criterion(out_t, labels, input_lens, lens)
                    total_loss += loss.item()
                    all_preds.extend(greedy_decode(out, self.blank))
                    all_labels.extend(labels.cpu())
                    all_lens.extend(lens.cpu())

            val_loss = total_loss / max(len(val_loader), 1)
            val_acc, val_cer = _compute_metrics(all_preds, all_labels, all_lens)
            scheduler.step(val_acc)

            self.history.train_loss.append(train_loss)
            self.history.train_acc.append(train_acc)
            self.history.val_loss.append(val_loss)
            self.history.val_acc.append(val_acc)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs}  "
                    f"Train: loss={train_loss:.4f} acc={train_acc * 100:.1f}%  "
                    f"Val: loss={val_loss:.4f} acc={val_acc * 100:.1f}%"
                )

            if val_acc > best_acc:
                best_acc = val_acc
                no_improve = 0
                # Save as .guruocr
                save_guruocr(
                    self.output_dir / "best_model.guruocr",
                    self.model.state_dict(),
                    self.class_names,
                    self.img_h,
                    self.img_w,
                    self.hidden,
                    self.num_layers,
                )
                if verbose:
                    print(f"  ↳ Best model saved ({best_acc * 100:.2f}%)")
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print("Early stopping.")
                    break

        if verbose:
            print(f"\nTraining complete! Best accuracy: {best_acc * 100:.2f}%")

        return self.history

    def evaluate(self, split: str = "test", verbose: bool = True) -> EvalResult:
        """
        Evaluate the model on a dataset split.

        Args:
            split: ``"test"``, ``"valid"``, or ``"train"``.
            verbose: Print results.

        Returns:
            :class:`EvalResult`.
        """
        import torch

        if self.model is None:
            # Try to load best model
            best = self.output_dir / "best_model.guruocr"
            if best.exists():
                state_dict, meta = load_guruocr(best)
                self.model = _build_vgg_ocr(
                    meta["num_classes"] + 1, meta["hidden"], meta["num_layers"]
                )
                self.model.load_state_dict(state_dict)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
            else:
                raise RuntimeError("No model loaded. Train first or provide a model.")

        from torch.utils.data import Dataset, DataLoader

        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
        except ImportError:
            raise ImportError("albumentations required for evaluation.")

        val_aug = A.Compose([
            A.Resize(self.img_h, self.img_w),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])

        samples = _load_split_samples(self.data_dir, split, self.num_tokens)
        if not samples:
            raise ValueError(f"No samples found in {split} split.")

        blank = self.blank

        class _DS(Dataset):
            def __init__(self, samps, tfm):
                self.s = samps
                self.t = tfm

            def __len__(self):
                return len(self.s)

            def __getitem__(self, i):
                s = self.s[i]
                img = cv2.imread(s["img"], cv2.IMREAD_GRAYSCALE)
                img = np.stack([img] * 3, axis=-1)
                img = self.t(image=img)["image"][0:1]
                return img, torch.tensor(s["ids"]), len(s["ids"])

        def _coll(batch):
            imgs, labels, lengths = zip(*batch)
            imgs = torch.stack(imgs)
            ml = max(lengths)
            padded = torch.full((len(labels), ml), blank, dtype=torch.long)
            for i, (lbl, ln) in enumerate(zip(labels, lengths)):
                padded[i, :ln] = lbl
            return imgs, padded, torch.tensor(lengths)

        loader = DataLoader(
            _DS(samples, val_aug), 64, shuffle=False, collate_fn=_coll, pin_memory=True
        )

        criterion = WeightedCTCLoss(blank=self.blank)
        self.model.eval()
        total_loss = 0
        all_preds, all_labels, all_lens = [], [], []

        with torch.no_grad():
            for imgs, labels, lens in loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                out = self.model(imgs)
                out_t = out.permute(1, 0, 2)
                il = torch.full((out_t.size(1),), out_t.size(0), dtype=torch.long)
                total_loss += criterion(out_t, labels, il, lens).item()
                all_preds.extend(greedy_decode(out, self.blank))
                all_labels.extend(labels.cpu())
                all_lens.extend(lens.cpu())

        acc, cer = _compute_metrics(all_preds, all_labels, all_lens)
        loss = total_loss / max(len(loader), 1)

        if verbose:
            print(f"\n{split.upper()} Results:")
            print(f"  Accuracy: {acc * 100:.2f}%")
            print(f"  TER:      {cer * 100:.2f}%")
            print(f"  Loss:     {loss:.4f}")

        return EvalResult(accuracy=acc, cer=cer, loss=loss)

    def plot_results(self, save: bool = True) -> None:
        """Plot training curves and save to output_dir."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(self.history.train_loss, label="Train")
        ax1.plot(self.history.val_loss, label="Valid")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot([a * 100 for a in self.history.train_acc], label="Train")
        ax2.plot([a * 100 for a in self.history.val_acc], label="Valid")
        ax2.set_title("Accuracy (%)")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            path = self.output_dir / "training_curves.png"
            plt.savefig(path, dpi=150)
            print(f"Saved: {path}")
        plt.close()
