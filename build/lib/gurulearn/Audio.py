"""
Audio - Audio classification with deep learning.

Provides training and inference for audio classification using CNN-LSTM models.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Handle Keras import for different TensorFlow versions
try:
    # TensorFlow 2.16+ with standalone Keras 3
    import keras
    from keras import layers, Model, optimizers, callbacks
except ImportError:
    # TensorFlow 2.x with bundled Keras
    from tensorflow import keras
    from tensorflow.keras import layers, Model, optimizers, callbacks


@dataclass
class PredictionResult:
    """Result from audio prediction."""
    label: str
    class_index: int
    confidence: float
    all_probabilities: np.ndarray


@dataclass
class TrainingHistory:
    """Training history and metrics."""
    train_accuracy: list[float]
    val_accuracy: list[float]
    train_loss: list[float]
    val_loss: list[float]
    test_accuracy: float
    test_loss: float


class AudioRecognition:
    """
    Audio classification using CNN-LSTM architecture.
    
    Args:
        sample_rate: Audio sample rate for processing (default: 16000)
        n_mfcc: Number of MFCC coefficients to extract (default: 20)
        
    Example:
        >>> model = AudioRecognition()
        >>> model.audiotrain("data/audio", epochs=50, model_dir="models")
        >>> result = model.predict("test.wav", model_dir="models")
        >>> print(f"Predicted: {result.label} ({result.confidence:.2%})")
    """

    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 20):
        self.model: Model | None = None
        self.le = LabelEncoder()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self._label_mapping: dict[str, str] = {}

    def extract_features(self, audio_signal: np.ndarray, sr: int | None = None) -> np.ndarray:
        """
        Extract audio features (MFCCs, chroma, spectral contrast).
        
        Args:
            audio_signal: Audio waveform
            sr: Sample rate (uses self.sample_rate if None)
            
        Returns:
            Combined feature array of shape (time_steps, n_features)
        """
        sr = sr or self.sample_rate
        
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=self.n_mfcc)
        chroma = librosa.feature.chroma_stft(y=audio_signal, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_signal, sr=sr)
        
        combined_features = np.concatenate([mfccs, chroma, spectral_contrast], axis=0)
        return combined_features.T

    def augment_audio(self, signal: np.ndarray, sr: int) -> list[np.ndarray]:
        """
        Apply audio augmentation techniques.
        
        Args:
            signal: Original audio signal
            sr: Sample rate
            
        Returns:
            List of augmented audio signals
        """
        augmented = []
        
        # Time stretching
        rate = np.random.uniform(0.8, 1.2)
        stretched = librosa.effects.time_stretch(signal, rate=rate)
        augmented.append(stretched)
        
        # Pitch shifting
        n_steps = np.random.randint(-2, 3)
        pitched = librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)
        augmented.append(pitched)
        
        # Adding noise
        noise = np.random.normal(0, 0.005, len(signal))
        noisy = signal + noise
        augmented.append(noisy.astype(np.float32))
        
        return augmented

    def load_data_with_augmentation(
        self, 
        data_dir: str | Path,
        augment: bool = True
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Load audio data from directory with optional augmentation.
        
        Args:
            data_dir: Directory with subdirectories for each class
            augment: Whether to apply augmentation
            
        Returns:
            Tuple of (features list, labels array)
        """
        data_dir = Path(data_dir)
        X: list[np.ndarray] = []
        y: list[str] = []
        
        # Supported audio formats
        audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        
        for label_dir in data_dir.iterdir():
            if not label_dir.is_dir():
                continue
                
            label = label_dir.name
            
            for audio_file in label_dir.iterdir():
                if audio_file.suffix.lower() not in audio_extensions:
                    continue
                
                try:
                    signal, sr = librosa.load(str(audio_file), sr=self.sample_rate)
                    
                    # Extract features from original
                    features = self.extract_features(signal, sr)
                    X.append(features)
                    y.append(label)
                    
                    # Add augmented versions
                    if augment:
                        for aug_signal in self.augment_audio(signal, sr):
                            aug_features = self.extract_features(aug_signal, sr)
                            X.append(aug_features)
                            y.append(label)
                            
                except Exception as e:
                    print(f"Warning: Could not process {audio_file}: {e}")
        
        return X, np.array(y)

    def _build_model(self, input_shape: tuple[int, int], num_classes: int) -> Model:
        """Build the CNN-LSTM model architecture."""
        inputs = layers.Input(shape=input_shape)
        
        # CNN layers
        x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        x = layers.Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(x)
        x = layers.Bidirectional(layers.LSTM(64, dropout=0.3))(x)
        
        # Dense layers
        x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.002))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.6)(x)
        
        x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.002))(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        
        return Model(inputs=inputs, outputs=outputs)

    def audiotrain(
        self,
        data_path: str | Path,
        epochs: int = 50,
        batch_size: int = 32,
        test_size: float = 0.2,
        learning_rate: float = 0.001,
        model_dir: str | Path = "model_folder",
        augment: bool = True,
        save_plots: bool = True
    ) -> TrainingHistory:
        """
        Train the audio classification model.
        
        Args:
            data_path: Directory containing audio data organized by class
            epochs: Number of training epochs
            batch_size: Training batch size
            test_size: Fraction of data for testing
            learning_rate: Initial learning rate
            model_dir: Directory to save the trained model
            augment: Whether to apply data augmentation
            save_plots: Whether to save training plots
            
        Returns:
            TrainingHistory with training metrics
        """
        np.random.seed(42)
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load and prepare data
        print("Loading and processing audio data...")
        X, y = self.load_data_with_augmentation(data_path, augment=augment)
        
        if len(X) == 0:
            raise ValueError(f"No audio files found in {data_path}")
        
        y_encoded = self.le.fit_transform(y)
        num_classes = len(np.unique(y_encoded))
        
        print(f"Loaded {len(X)} samples across {num_classes} classes")
        print(f"Classes: {list(self.le.classes_)}")

        # Pad sequences to same length
        max_length = max(len(feat) for feat in X)
        X_padded = keras.utils.pad_sequences(
            X, maxlen=max_length, padding="post", dtype="float32"
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_padded, y_encoded, 
            test_size=test_size, 
            random_state=42, 
            stratify=y_encoded
        )

        # Build model
        input_shape = (max_length, X_padded.shape[2])
        self.model = self._build_model(input_shape, num_classes)

        # Learning rate schedule
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=100,
            decay_rate=0.9
        )
        optimizer = optimizers.Adam(learning_rate=lr_schedule)

        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Callbacks
        training_callbacks = [
            callbacks.EarlyStopping(
                monitor="val_accuracy", 
                patience=10, 
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.5, 
                patience=5, 
                min_lr=1e-5
            )
        ]

        # Train
        print("\nStarting training...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=training_callbacks,
            verbose=1
        )

        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Classification report
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=self.le.classes_))

        # Save plots
        if save_plots:
            self._save_training_plots(history, y_test, y_pred_classes, model_dir)

        # Save model
        self._save_model(model_dir)

        return TrainingHistory(
            train_accuracy=history.history["accuracy"],
            val_accuracy=history.history["val_accuracy"],
            train_loss=history.history["loss"],
            val_loss=history.history["val_loss"],
            test_accuracy=test_accuracy,
            test_loss=test_loss
        )

    def _save_training_plots(
        self, 
        history, 
        y_test: np.ndarray, 
        y_pred: np.ndarray,
        output_dir: Path
    ) -> None:
        """Save training history and confusion matrix plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=self.le.classes_, yticklabels=self.le.classes_)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        # Training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "training_history.png", dpi=150)
        plt.close()

    def _save_model(self, model_dir: Path) -> None:
        """Save model and label mapping."""
        # Save model in native Keras format
        model_path = model_dir / "audio_recognition_model.keras"
        self.model.save(model_path)

        # Save label mapping
        label_mapping = {str(i): label for i, label in enumerate(self.le.classes_)}
        mapping_path = model_dir / "label_mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(label_mapping, f, indent=2)

        print(f"Model saved to {model_path}")

    def _load_model(self, model_dir: str | Path) -> None:
        """Load model and label mapping."""
        model_dir = Path(model_dir)
        
        # Try new format first, fall back to legacy .h5
        model_path = model_dir / "audio_recognition_model.keras"
        if not model_path.exists():
            model_path = model_dir / "audio_recognition_model.h5"
        
        self.model = keras.models.load_model(model_path)
        
        mapping_path = model_dir / "label_mapping.json"
        with open(mapping_path, "r", encoding="utf-8") as f:
            self._label_mapping = json.load(f)

    def predict(self, input_wav: str | Path, model_dir: str | Path = "model_folder") -> PredictionResult:
        """
        Predict the class of an audio file.
        
        Args:
            input_wav: Path to the audio file
            model_dir: Directory containing the trained model
            
        Returns:
            PredictionResult with label, class index, confidence, and probabilities
        """
        if self.model is None or not self._label_mapping:
            self._load_model(model_dir)

        # Load and extract features
        signal, sr = librosa.load(str(input_wav), sr=self.sample_rate)
        features = self.extract_features(signal, sr)
        
        # Pad to model's expected input length
        max_length = self.model.input_shape[1]
        features_padded = keras.utils.pad_sequences(
            [features], maxlen=max_length, padding="post", dtype="float32"
        )

        # Predict
        predictions = self.model.predict(features_padded, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        predicted_label = self._label_mapping[str(predicted_class)]

        return PredictionResult(
            label=predicted_label,
            class_index=predicted_class,
            confidence=confidence,
            all_probabilities=predictions[0]
        )

    def predict_batch(
        self, 
        audio_paths: list[str | Path], 
        model_dir: str | Path = "model_folder"
    ) -> list[PredictionResult]:
        """
        Predict classes for multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            model_dir: Directory containing the trained model
            
        Returns:
            List of PredictionResult for each file
        """
        if self.model is None or not self._label_mapping:
            self._load_model(model_dir)

        results = []
        for path in audio_paths:
            try:
                result = self.predict(path, model_dir)
                results.append(result)
            except Exception as e:
                print(f"Warning: Could not process {path}: {e}")
        
        return results
