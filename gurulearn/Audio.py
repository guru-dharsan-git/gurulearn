





import json
import os
import librosa
from matplotlib import pyplot as plt
import numpy as np
from sklearn.calibration import LabelEncoder
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


class AudioRecognition:
    def __init__(self):
        self.model = None
        self.le = LabelEncoder()

    def augment_audio(self, signal, sr):
        stretched = librosa.effects.time_stretch(signal, rate=np.random.uniform(0.8, 1.2))
        pitched = librosa.effects.pitch_shift(signal, sr=sr, n_steps=np.random.randint(-2, 3))
        noise = np.random.normal(0, 0.005, len(signal))
        noisy = signal + noise
        return [stretched, pitched, noisy]

    def load_data_with_augmentation(self, data_dir):
        X, y = [], []
        for label in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, label)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(folder_path, file_name)
                        signal, sr = librosa.load(file_path, sr=16000)

                        def extract_features(audio_signal):
                            mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=20)
                            chroma = librosa.feature.chroma_stft(y=audio_signal, sr=sr)
                            spectral_contrast = librosa.feature.spectral_contrast(y=audio_signal, sr=sr)
                            combined_features = np.concatenate([mfccs, chroma, spectral_contrast], axis=0)
                            return combined_features.T

                        original_features = extract_features(signal)
                        X.append(original_features)
                        y.append(label)

                        augmented_signals = self.augment_audio(signal, sr)
                        for aug_signal in augmented_signals:
                            aug_features = extract_features(aug_signal)
                            X.append(aug_features)
                            y.append(label)

        return X, np.array(y)

    def audiotrain(self, data_path, epochs=50, batch_size=32, test_size=0.2, learning_rate=0.001, model_dir='model_folder'):
        np.random.seed(42)
        tf.random.set_seed(42)

        X, y = self.load_data_with_augmentation(data_path)
        y_encoded = self.le.fit_transform(y)

        max_length = max(len(mfcc) for mfcc in X)
        X_padded = pad_sequences(X, maxlen=max_length, padding='post', dtype='float32')
        X_padded = X_padded.reshape(X_padded.shape[0], X_padded.shape[1], X_padded.shape[2])

        X_train, X_test, y_train, y_test = train_test_split(
            X_padded, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=100,
            decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(max_length, X_padded.shape[2])),
            tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.3)),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
        ])

        self.model.compile(
            optimizer=optimizer, 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
        ]

        history = self.model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks
        )

        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=self.le.classes_, 
                    yticklabels=self.le.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')

        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred_classes, 
            target_names=self.le.classes_
        ))

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_history.png')

        # Ensure the model directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save the model
        model_path = os.path.join(model_dir, 'audio_recognition_model.h5')
        self.model.save(model_path)

        # Save label mapping as a separate JSON file
        label_mapping = {str(index): label for index, label in enumerate(self.le.classes_)}
        label_mapping_path = os.path.join(model_dir, 'label_mapping.json')
        with open(label_mapping_path, 'w') as f:
            json.dump(label_mapping, f)

    def predict(self, input_wav, model_dir='model_folder'):
        # Load the trained model
        model_path = os.path.join(model_dir, 'audio_recognition_model.h5')
        self.model = tf.keras.models.load_model(model_path)
        
        # Load the label mapping from the JSON file
        label_mapping_path = os.path.join(model_dir, 'label_mapping.json')
        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)

        # Load audio and extract features
        signal, sr = librosa.load(input_wav, sr=16000)
        features = self.extract_features(signal)
        features_padded = pad_sequences([features], maxlen=self.model.input_shape[1], padding='post', dtype='float32')

        # Predict
        predictions = self.model.predict(features_padded)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = label_mapping[str(predicted_class)]

        return predicted_label
    
    def predict_class(self, input_wav, model_dir='model_folder'):
        # Load the trained model
        model_path = os.path.join(model_dir, 'audio_recognition_model.h5')
        self.model = tf.keras.models.load_model(model_path)
        
        # Load the label mapping from the JSON file
        label_mapping_path = os.path.join(model_dir, 'label_mapping.json')
        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)

        # Load audio and extract features
        signal, sr = librosa.load(input_wav, sr=16000)
        features = self.extract_features(signal)
        features_padded = pad_sequences([features], maxlen=self.model.input_shape[1], padding='post', dtype='float32')

        # Predict
        predictions = self.model.predict(features_padded)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return predicted_class
    
    def predict_class_conf(self, input_wav, model_dir='model_folder'):
        # Load the trained model
        model_path = os.path.join(model_dir, 'audio_recognition_model.h5')
        self.model = tf.keras.models.load_model(model_path)
        
        # Load the label mapping from the JSON file
        label_mapping_path = os.path.join(model_dir, 'label_mapping.json')
        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)

        # Load audio and extract features
        signal, sr = librosa.load(input_wav, sr=16000)
        features = self.extract_features(signal)
        features_padded = pad_sequences([features], maxlen=self.model.input_shape[1], padding='post', dtype='float32')

        # Predict
        predictions = self.model.predict(features_padded)
        predicted_class = max(predictions)[0]
        return predicted_class     

    def extract_features(self, audio_signal):
        sr = 16000
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=audio_signal, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_signal, sr=sr)
        combined_features = np.concatenate([mfccs, chroma, spectral_contrast], axis=0)
        return combined_features.T
        

