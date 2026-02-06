"""
Image Classification - Deep learning image classification with multiple architectures.

Provides training and inference for image classification using various backbone models.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, transforms
from torchvision.models import (
    ResNet50_Weights, VGG16_Weights, MobileNet_V2_Weights,
    Inception_V3_Weights, DenseNet121_Weights, EfficientNet_B0_Weights,
    ConvNeXt_Tiny_Weights, ViT_B_16_Weights
)


ModelName = Literal[
    "simple_cnn", "vgg16", "resnet50", "mobilenet", "inceptionv3", 
    "densenet", "efficientnet", "convnext", "vit"
]


@dataclass
class TrainingHistory:
    """Training history and metrics."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    best_val_acc: float = 0.0


@dataclass
class PredictionResult:
    """Result from image prediction."""
    class_name: str
    probability: float
    class_index: int
    top_k: list[tuple[str, float]] = field(default_factory=list)


class ImageDataset(Dataset):
    """Dataset for image classification."""
    
    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        img_dir: str | Path | None = None,
        transform: transforms.Compose | None = None,
        img_column: str | None = None,
        label_column: str | None = None
    ):
        self.img_paths: list[str] = []
        self.labels: list[int] = []
        self.transform = transform
        self.classes: list[str] = []

        if dataframe is not None and img_column and label_column:
            self._load_from_dataframe(dataframe, img_column, label_column)
        elif img_dir is not None:
            self._load_from_directory(Path(img_dir))

    def _load_from_dataframe(self, df: pd.DataFrame, img_column: str, label_column: str) -> None:
        """Load data from a DataFrame."""
        self.img_paths = df[img_column].tolist()
        self.classes = sorted(df[label_column].unique().tolist())
        
        label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        self.labels = [label_to_idx[label] for label in df[label_column]]

    def _load_from_directory(self, img_dir: Path) -> None:
        """Load data from a directory structure."""
        self.classes = sorted([
            d.name for d in img_dir.iterdir() 
            if d.is_dir() and not d.name.startswith(".")
        ])
        
        label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        
        for label in self.classes:
            label_dir = img_dir / label
            for img_file in label_dir.iterdir():
                if img_file.suffix.lower() in self.SUPPORTED_FORMATS:
                    self.img_paths.append(str(img_file))
                    self.labels.append(label_to_idx[label])

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_paths[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a placeholder
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, torch.tensor(self.labels[idx], dtype=torch.long)


class ImageClassifier:
    """
    Image classifier with multiple backbone architectures.
    
    Example:
        >>> classifier = ImageClassifier()
        >>> model, history = classifier.train(
        ...     train_dir="data/train",
        ...     epochs=10,
        ...     model_name="resnet50"
        ... )
        >>> predictions = classifier.predict("test_image.jpg", top_k=3)
    """

    def __init__(self, device: str | None = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.classes: list[str] = []
        self.model: nn.Module | None = None
        self.input_size = (224, 224)
        self.transform: transforms.Compose | None = None

    def _get_optimal_workers(self) -> int:
        """Get optimal number of dataloader workers for the platform."""
        if os.name == "nt":  # Windows
            return 0  # Windows has issues with multiprocessing in DataLoader
        return min(4, os.cpu_count() or 1)

    def _build_simple_cnn(self, num_classes: int) -> nn.Module:
        """Build a simple CNN with adaptive pooling."""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),  # Adaptive pooling for any input size
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _build_vgg16(self, num_classes: int, finetune: bool) -> nn.Module:
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        if not finetune:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier[-1] = nn.Linear(4096, num_classes)
        return model

    def _build_resnet50(self, num_classes: int, finetune: bool) -> nn.Module:
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        if not finetune:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        # Unfreeze fc layers
        for param in model.fc.parameters():
            param.requires_grad = True
        return model

    def _build_mobilenet(self, num_classes: int, finetune: bool) -> nn.Module:
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        if not finetune:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.last_channel, num_classes)
        return model

    def _build_inceptionv3(self, num_classes: int, finetune: bool) -> nn.Module:
        model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
        if not finetune:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        for param in model.fc.parameters():
            param.requires_grad = True
        self.input_size = (299, 299)  # InceptionV3 requires larger input
        return model

    def _build_densenet(self, num_classes: int, finetune: bool) -> nn.Module:
        model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        if not finetune:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    def _build_efficientnet(self, num_classes: int, finetune: bool) -> nn.Module:
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        if not finetune:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    def _build_convnext(self, num_classes: int, finetune: bool) -> nn.Module:
        model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        if not finetune:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    def _build_vit(self, num_classes: int, finetune: bool) -> nn.Module:
        model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        if not finetune:
            for param in model.parameters():
                param.requires_grad = False
        model.heads[-1] = nn.Linear(model.heads[-1].in_features, num_classes)
        for param in model.heads.parameters():
            param.requires_grad = True
        return model

    def _select_model(
        self, 
        num_classes: int, 
        dataset_size: int, 
        model_name: ModelName | None = None,
        finetune: bool = False
    ) -> nn.Module:
        """Select and build the appropriate model."""
        # Auto-select based on dataset size if not specified
        if model_name is None:
            if dataset_size < 1000:
                model_name = "simple_cnn"
            elif dataset_size < 5000:
                model_name = "mobilenet"
            else:
                model_name = "resnet50"

        builders = {
            "simple_cnn": lambda: self._build_simple_cnn(num_classes),
            "vgg16": lambda: self._build_vgg16(num_classes, finetune),
            "resnet50": lambda: self._build_resnet50(num_classes, finetune),
            "mobilenet": lambda: self._build_mobilenet(num_classes, finetune),
            "inceptionv3": lambda: self._build_inceptionv3(num_classes, finetune),
            "densenet": lambda: self._build_densenet(num_classes, finetune),
            "efficientnet": lambda: self._build_efficientnet(num_classes, finetune),
            "convnext": lambda: self._build_convnext(num_classes, finetune),
            "vit": lambda: self._build_vit(num_classes, finetune),
        }
        
        if model_name not in builders:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(builders.keys())}")
        
        return builders[model_name]().to(self.device)

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scaler: torch.amp.GradScaler | None = None
    ) -> tuple[float, float]:
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            # Mixed precision training
            if scaler and self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return running_loss / len(loader), correct / total

    def _evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module
    ) -> tuple[float, float]:
        """Evaluate model on a dataset."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return running_loss / len(loader), correct / total

    def train(
        self,
        train_dir: str | Path | None = None,
        test_dir: str | Path | None = None,
        csv_file: str | Path | None = None,
        img_column: str | None = None,
        label_column: str | None = None,
        epochs: int = 10,
        model_name: ModelName | None = None,
        finetune: bool = False,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        save_path: str | Path = "trained_model.pth",
        use_amp: bool = True,
        save_plots: bool = True
    ) -> tuple[nn.Module, TrainingHistory]:
        """
        Train an image classification model.
        
        Args:
            train_dir: Directory with training images in class folders
            test_dir: Directory with test images (optional)
            csv_file: CSV file with image paths and labels
            img_column: Column name for image paths in CSV
            label_column: Column name for labels in CSV
            epochs: Number of training epochs
            model_name: Model architecture to use
            finetune: Whether to finetune all layers
            batch_size: Training batch size
            learning_rate: Learning rate
            save_path: Path to save the trained model
            use_amp: Use automatic mixed precision (CUDA only)
            save_plots: Whether to save training plots
            
        Returns:
            Tuple of (trained model, training history)
        """
        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load data
        if csv_file:
            df = pd.read_csv(csv_file)
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
            train_dataset = ImageDataset(dataframe=train_df, transform=self.transform,
                                         img_column=img_column, label_column=label_column)
            val_dataset = ImageDataset(dataframe=val_df, transform=val_transform,
                                       img_column=img_column, label_column=label_column)
            self.classes = train_dataset.classes
        else:
            if not train_dir:
                raise ValueError("Either train_dir or csv_file must be provided")
            
            train_dataset = ImageDataset(img_dir=train_dir, transform=self.transform)
            self.classes = train_dataset.classes
            
            if test_dir and Path(test_dir).exists():
                val_dataset = ImageDataset(img_dir=test_dir, transform=val_transform)
            else:
                train_size = int(0.8 * len(train_dataset))
                val_size = len(train_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(
                    train_dataset, [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )

        if len(train_dataset) == 0:
            raise ValueError("No training images found!")

        num_workers = self._get_optimal_workers()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Classes ({len(self.classes)}): {self.classes}")

        # Build model
        self.model = self._select_model(len(self.classes), len(train_dataset), model_name, finetune)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.01
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        
        # Mixed precision scaler
        scaler = torch.amp.GradScaler() if use_amp and self.device.type == "cuda" else None

        # Training loop
        history = TrainingHistory()
        best_model_state = None

        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(self.model, train_loader, criterion, optimizer, scaler)
            val_loss, val_acc = self._evaluate(self.model, val_loader, criterion)
            
            scheduler.step(val_loss)
            
            history.train_loss.append(train_loss)
            history.train_acc.append(train_acc)
            history.val_loss.append(val_loss)
            history.val_acc.append(val_acc)
            
            if val_acc > history.best_val_acc:
                history.best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Load best weights
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"\nBest validation accuracy: {history.best_val_acc:.4f}")

        # Save model
        self.save_model(save_path)

        # Save plots
        if save_plots:
            self._save_plots(history, val_loader, Path(save_path).parent or Path("."))

        return self.model, history

    def _save_plots(self, history: TrainingHistory, val_loader: DataLoader, output_dir: Path) -> None:
        """Save training plots and confusion matrix."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(history.train_acc, label="Train")
        axes[0].plot(history.val_acc, label="Validation")
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        
        axes[1].plot(history.train_loss, label="Train")
        axes[1].plot(history.val_loss, label="Validation")
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "training_history.png", dpi=150)
        plt.close()

        # Confusion matrix
        self.model.eval()
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.cpu().numpy())
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close()

    def save_model(self, save_path: str | Path) -> None:
        """Save the model and metadata."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), save_path)
        
        # Save metadata
        info = {
            "classes": self.classes,
            "input_size": list(self.input_size)
        }
        info_path = save_path.with_suffix(".json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"Model saved to {save_path}")

    def load(
        self, 
        model_path: str | Path, 
        model_name: ModelName | None = None,
        num_classes: int | None = None
    ) -> nn.Module:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            model_name: Model architecture (inferred from metadata if available)
            num_classes: Number of classes (required if metadata not available)
            
        Returns:
            Loaded model
        """
        model_path = Path(model_path)
        info_path = model_path.with_suffix(".json")
        
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            self.classes = info["classes"]
            self.input_size = tuple(info["input_size"])
            num_classes = len(self.classes)
        elif num_classes is None:
            raise ValueError("num_classes required when metadata file is not available")
        
        self.model = self._select_model(num_classes, 1000, model_name)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded from {model_path}")
        return self.model

    def predict(
        self, 
        image_path: str | Path | None = None,
        image: Image.Image | None = None,
        top_k: int = 1
    ) -> PredictionResult:
        """
        Predict class for an image.
        
        Args:
            image_path: Path to the image file
            image: PIL Image (alternative to image_path)
            top_k: Number of top predictions to return
            
        Returns:
            PredictionResult with predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load() first.")
        
        if image_path is None and image is None:
            raise ValueError("Either image_path or image must be provided")
        
        if image_path:
            image = Image.open(image_path).convert("RGB")
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.classes)))
        
        top_predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            class_name = self.classes[idx] if self.classes else f"Class {idx}"
            top_predictions.append((class_name, float(prob)))
        
        best_class, best_prob = top_predictions[0]
        best_idx = int(top_indices[0].cpu().numpy())
        
        return PredictionResult(
            class_name=best_class,
            probability=best_prob,
            class_index=best_idx,
            top_k=top_predictions
        )

    def export_onnx(self, save_path: str | Path, opset_version: int = 14) -> None:
        """
        Export model to ONNX format.
        
        Args:
            save_path: Path to save the ONNX model
            opset_version: ONNX opset version
        """
        if self.model is None:
            raise ValueError("No model to export")
        
        self.model.eval()
        dummy_input = torch.randn(1, 3, *self.input_size).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            str(save_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        print(f"ONNX model exported to {save_path}")
