# Standard library imports
import os
import glob
import json
import warnings
from datetime import datetime

# Data manipulation and analysis
import numpy as np
import pandas as pd
from scipy import ndimage

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score, 
    mean_squared_error, 
    mean_absolute_error,
    confusion_matrix, 
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import (
    VGG16,
    ResNet50,
    MobileNet,
    InceptionV3,
    DenseNet121,
    EfficientNetB0,
    Xception,
    NASNetMobile,
    InceptionResNetV2
)
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

# Image and Audio Processing
import cv2
import librosa

# XGBoost (with error handling)
try:
    from xgboost import XGBRegressor
except ImportError:
    print("XGBoost is not installed. Install it with `pip install xgboost` if you plan to use XGBRegressor.")

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)



class CTScanProcessor:
    def __init__(self, kernel_size=5, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.kernel_size = kernel_size
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def sharpen(self, image):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def median_denoise(self, image):
        return ndimage.median_filter(image, size=self.kernel_size)

    def enhance_contrast(self, image):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return clahe.apply(image)

    def enhanced_denoise(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Could not read the image")

        denoised = self.median_denoise(image)
        denoised = self.enhance_contrast(denoised)
        denoised = self.sharpen(denoised)
        return denoised

    def evaluate_quality(self, original, denoised):
        if original is None or denoised is None:
            raise ValueError("Original or denoised image is None.")

        original = original.astype(float)
        denoised = denoised.astype(float)

        mse = np.mean((original - denoised) ** 2) + 1e-10
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        signal_power = np.mean(denoised ** 2)
        noise_power = np.mean((original - denoised) ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100
        
        detail_orig = np.std(original)
        detail_denoise = np.std(denoised)
        detail_ratio = detail_denoise / detail_orig if detail_orig > 0 else 1
        
        return {
            'MSE': mse,
            'PSNR': psnr,
            'SNR': snr,
            'Detail_Preservation': detail_ratio * 100  
        }

    def compare_images(self, original, processed, output_path):
        """Save a side-by-side comparison of the original and processed images."""
        if original is None or processed is None:
            raise ValueError("Original or processed image is None.")
        
        comparison = np.hstack((original, processed))
        cv2.imwrite(output_path, comparison)
        return comparison

    def print_best_metrics(self, metrics):
        if metrics is None:
            print("No metrics to display.")
            return
        
        print("\nFinal metrics for best result:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

    def process_ct_scan(self, input_path, output_folder, comparison_folder="comparison", compare=False):
        try:
            os.makedirs(output_folder, exist_ok=True)
            if compare and comparison_folder:
                os.makedirs(comparison_folder, exist_ok=True)

            
            original = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if original is None:
                raise ValueError("Could not read the original image")
            
            
            denoised = self.enhanced_denoise(input_path)
            metrics = self.evaluate_quality(original, denoised)

            print(f"\nDenoising metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.2f}")
            
            
            output_path = os.path.join(output_folder, os.path.basename(input_path).replace('.jpg', '_denoised.jpg'))
            cv2.imwrite(output_path, denoised)

            
            if compare and comparison_folder:
                comparison_path = os.path.join(comparison_folder, os.path.basename(input_path).replace('.jpg', '_comparison.jpg'))
                self.compare_images(original, denoised, comparison_path)

            self.print_best_metrics(metrics)

            return denoised, metrics
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None, None



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
        








###easy chatbot creator
import pandas as pd
import json
import os


class FlowBot:


    def __init__(self, data):
        self.df = data.copy()
        self.df_display = data.copy()
        self.df_clean = data.copy()
        for col in self.df_clean.select_dtypes(include='object'):
            self.df_clean[col] = self.df_clean[col].astype(str).str.strip().str.lower()
        self.flow = []
        self.prompts = {}
        self.result_columns = []
        self.sessions = {}
        self.personal_info_fields = {}
        self.chat_history = {}

    def add_personal_info(self, field, prompt, required=True):
        """Add a personal information field to collect from the user"""
        self.personal_info_fields[field] = {
            'prompt': prompt,
            'required': required
        }

    def add(self, field, prompt, required=True):
        """Add a step to the booking flow"""
        if field not in self.df.columns:
            raise ValueError(f"Column '{field}' not found in dataset")
        self.flow.append({
            'field': field,
            'required': required
        })
        self.prompts[field] = prompt

    def finish(self, *result_columns):
        """Set result columns to display in final output"""
        if not result_columns:
            raise ValueError("At least one result column must be specified")
            
        for column in result_columns:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in dataset")
                
        self.result_columns = list(result_columns)

    def get_suggestions(self, user_id):
        """Get available options based on current state"""
        session = self.sessions[user_id]
        current_step = session['step']
        filtered = self.df_clean.copy()
        for step in self.flow[:current_step]:
            field = step['field']
            if field in session['selections']:
                val = session['selections'][field]
                if val:
                    filtered = filtered[filtered[field] == val.lower()]
        current_field = self.flow[current_step]['field']
        options = filtered[current_field].unique().tolist()
        display_options = []
        for opt in options:
            if pd.notna(opt):
                mask = self.df_clean[current_field] == opt
                display_val = self.df_display.loc[mask, current_field].iloc[0]
                display_options.append(display_val)
        return [str(opt) for opt in display_options if opt and pd.notna(opt)]

    def _log_interaction(self, user_id, user_input, bot_response):
        """Helper method to log interactions to chat history"""
        if user_id not in self.chat_history:
            self.chat_history[user_id] = []
        if bot_response is not None or user_input:
            self.chat_history[user_id].append({
                'user_input': user_input,
                'bot_response': bot_response
            })

    def process(self, user_id, text):
        """Process user input and return response"""
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                'step': 0,
                'selections': {},
                'completed': False,
                'personal_info': {}
            }
            self.chat_history[user_id] = []
            
        session = self.sessions[user_id]
        if session['completed']:
            self.reset_session(user_id)

        if len(session['personal_info']) < len(self.personal_info_fields):
            response = self._collect_personal_info(user_id, text)
            return response

        current_step = session['step']
        if current_step >= len(self.flow):
            return self._finalize_response(user_id)

        current_field = self.flow[current_step]['field']
        required = self.flow[current_step]['required']

        if not text.strip():
            if required:
                message = f"This field is required. Please choose from: {', '.join(self.get_suggestions(user_id))}"
                self._log_interaction(user_id, "", message)
                response = {
                    'message': message,
                    'suggestions': self.get_suggestions(user_id)
                }
                return response
            else:
                session['selections'][current_field] = None
                session['step'] += 1

        else:
            cleaned_input = str(text).strip().lower()
            available = [str(x).lower() for x in self.get_suggestions(user_id)]
            
            if cleaned_input not in available and text not in self.get_suggestions(user_id):
                if required:
                    message = f"Invalid option. Please choose from: {', '.join(self.get_suggestions(user_id))}"
                    self._log_interaction(user_id, text, message)
                    response = {
                        'message': message,
                        'suggestions': self.get_suggestions(user_id)
                    }
                    return response
                else:
                    session['selections'][current_field] = None
                    session['step'] += 1
            else:
                mask = self.df_display[current_field].astype(str).str.lower() == cleaned_input
                if any(mask):
                    clean_value = self.df_clean.loc[mask, current_field].iloc[0]
                else:
                    clean_value = cleaned_input
                session['selections'][current_field] = clean_value
                session['step'] += 1

        if session['step'] >= len(self.flow):
            self._log_interaction(user_id, text, self._generate_final_message(user_id))
            return self._finalize_response(user_id)

        next_field = self.flow[session['step']]['field']
        next_prompt = self.prompts[next_field]
        response = {
            'message': next_prompt,
            'suggestions': self.get_suggestions(user_id)
        }
        
        self._log_interaction(user_id, text, next_prompt)
        return response

    def _collect_personal_info(self, user_id, text):
        """Collect personal information from the user"""
        session = self.sessions[user_id]
        personal_info = session['personal_info']
        fields = list(self.personal_info_fields.keys())
        
        for i, field in enumerate(fields):
            if field not in personal_info:
                info = self.personal_info_fields[field]
                if not text.strip():
                    response = {
                        'message': info['prompt'],
                        'suggestions': []
                    }
                    if i == 0:
                        self._log_interaction(user_id, "", info['prompt'])
                    return response
                else:
                    personal_info[field] = text.strip()
                    
                    if i + 1 < len(fields):
                        next_field = fields[i + 1]
                        next_prompt = self.personal_info_fields[next_field]['prompt']
                    else:
                        next_prompt = self.prompts[self.flow[0]['field']] if self.flow else None
                    
                    self._log_interaction(user_id, text, next_prompt)
                    
                    if i + 1 < len(fields):
                        return {
                            'message': next_prompt,
                            'suggestions': []
                        }
                    else:
                        session['step'] = 0
                        if self.flow:
                            return {
                                'message': next_prompt,
                                'suggestions': self.get_suggestions(user_id)
                            }
                        else:
                            return self._finalize_response(user_id)
        return self.process(user_id, "")

    def _generate_final_message(self, user_id):
        """Generate the final results message"""
        session = self.sessions[user_id]
        filtered = self.df_clean.copy()
        for field, value in session['selections'].items():
            if value:
                filtered = filtered[filtered[field] == value]
        results = self.df_display.loc[filtered.index]
        
        if len(results) == 0:
            return "No results found matching your criteria"
        
        final_message = f"Found {len(results)} matching options:\n"
        for _, row in results.iterrows():
            result_items = [f"{col}: {row[col]}" for col in self.result_columns]
            final_message += f"- {' | '.join(result_items)}\n"
        return final_message

    def _finalize_response(self, user_id):
        """Generate final results"""
        session = self.sessions[user_id]
        filtered = self.df_clean.copy()
        for field, value in session['selections'].items():
            if value:
                filtered = filtered[filtered[field] == value]
        results = self.df_display.loc[filtered.index]
        
        final_message = self._generate_final_message(user_id)
        response = {
            'completed': True,
            'results': results[self.result_columns].to_dict('records'),
            'message': final_message
        }
        
        session['completed'] = True
        self._save_to_json(user_id)
        return response

    def _save_to_json(self, user_id):
        """Save chat history and personal info to a JSON file"""
        session = self.sessions[user_id]
        data_to_save = {
            'personal_info': session['personal_info'],
            'chat_history': self.chat_history[user_id]
        }
        if not os.path.exists('user_data'):
            os.makedirs('user_data')
        with open(f'user_data/{user_id}.json', 'w') as f:
            json.dump(data_to_save, f, indent=4)

    def reset_session(self, user_id):
        """Reset user's session"""
        self.sessions[user_id] = {
            'step': 0,
            'selections': {},
            'completed': False,
            'personal_info': {}
        }
        self.chat_history[user_id] = []

        