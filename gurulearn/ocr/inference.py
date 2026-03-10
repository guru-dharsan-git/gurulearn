"""
OCR inference — load a ``.guruocr`` model and predict text from images.

No dataset directory required; all metadata (class names, image dimensions,
model configuration) is bundled inside the ``.guruocr`` file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np

from .model import load_guruocr, _build_vgg_ocr, greedy_decode


@dataclass
class PredictionResult:
    """Result of a single image prediction."""
    text: str
    token_ids: List[int] = field(default_factory=list)
    confidence: float = 0.0


class OCRPredictor:
    """
    Load a trained ``.guruocr`` model and predict text from images.

    The ``.guruocr`` file is fully self-contained — it carries the model
    weights **and** all metadata (class names, image size, architecture
    config), so no dataset directory is needed.

    Example::

        predictor = OCRPredictor("best_model.guruocr")
        result = predictor.predict("image.jpg")
        print(result.text)         # e.g. "VTTGH"
        print(result.confidence)   # e.g. 0.97

        results = predictor.predict_batch(["a.jpg", "b.jpg"])
    """

    def __init__(self, model_path: str | Path, device: Optional[str] = None):
        """
        Args:
            model_path: Path to a ``.guruocr`` file.
            device: ``"cuda"`` or ``"cpu"``.  Auto-detected if *None*.
        """
        import torch

        self.model_path = Path(model_path)
        state_dict, self.meta = load_guruocr(self.model_path)

        self.class_names: List[str] = self.meta["class_names"]
        self.num_tokens = self.meta["num_classes"]
        self.blank = self.num_tokens
        self.img_h = self.meta["img_h"]
        self.img_w = self.meta["img_w"]
        self.id2char = {i: n for i, n in enumerate(self.class_names)}

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Rebuild model from metadata
        num_classes_ctc = self.num_tokens + 1  # +1 for CTC blank
        self.model = _build_vgg_ocr(
            num_classes_ctc,
            self.meta["hidden"],
            self.meta["num_layers"],
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _ids_to_text(self, ids: List[int]) -> str:
        parts = []
        for tid in ids:
            tok = self.id2char.get(int(tid), "?")
            parts.append(tok if len(tok) == 1 else f"<{tok}>")
        return "".join(parts)

    def _preprocess(self, image: np.ndarray) -> "torch.Tensor":
        """Resize, normalise, and convert a grayscale image to a tensor."""
        import torch

        img = cv2.resize(image, (self.img_w, self.img_h))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # same normalisation as training
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return tensor.to(self.device)

    def predict(self, image: Union[str, Path, np.ndarray]) -> PredictionResult:
        """
        Predict text from a single image.

        Args:
            image: File path or a numpy array (BGR or grayscale).

        Returns:
            :class:`PredictionResult` with text, token IDs, and confidence.
        """
        import torch

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
        elif isinstance(image, np.ndarray):
            img = image
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise TypeError(f"Expected str, Path, or ndarray, got {type(image)}")

        tensor = self._preprocess(img)

        with torch.no_grad():
            output = self.model(tensor)  # (1, T, num_classes)
            preds = greedy_decode(output, self.blank)
            token_ids = preds[0]

            # Compute mean confidence over predicted frames
            probs = torch.exp(output[0])  # log_softmax → softmax
            max_probs = probs.max(dim=1).values
            # Only consider non-blank frames
            frame_preds = output[0].argmax(dim=1).cpu().numpy()
            non_blank = [i for i, p in enumerate(frame_preds) if p != self.blank]
            if non_blank:
                conf = float(max_probs[non_blank].mean().cpu())
            else:
                conf = 0.0

        return PredictionResult(
            text=self._ids_to_text(token_ids),
            token_ids=token_ids,
            confidence=conf,
        )

    def predict_batch(
        self,
        images: Sequence[Union[str, Path, np.ndarray]],
    ) -> List[PredictionResult]:
        """
        Predict text from multiple images.

        Args:
            images: List of file paths or numpy arrays.

        Returns:
            List of :class:`PredictionResult`.
        """
        return [self.predict(img) for img in images]

    def visualize(
        self,
        image: Union[str, Path, np.ndarray],
        save_path: Optional[str | Path] = None,
    ) -> None:
        """
        Show the image with the prediction overlaid.

        Args:
            image: File path or numpy array.
            save_path: If given, save the visualisation instead of showing it.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for visualisation.")
            return

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        else:
            img = image
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result = self.predict(image)

        plt.figure(figsize=(10, 3))
        plt.imshow(img, cmap="gray")
        plt.title(
            f"Prediction: {result.text}  (conf: {result.confidence:.2f})",
            fontsize=14,
        )
        plt.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(str(save_path), dpi=150)
            print(f"Saved: {save_path}")
        else:
            plt.show()
        plt.close()

    def __repr__(self) -> str:
        return (
            f"OCRPredictor(model='{self.model_path.name}', "
            f"classes={self.num_tokens}, "
            f"img={self.img_h}×{self.img_w})"
        )
