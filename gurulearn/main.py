import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNet, InceptionV3, DenseNet121,
    EfficientNetB0, Xception, NASNetMobile, InceptionResNetV2
)
from tensorflow.keras.callbacks import ReduceLROnPlateau
import plotly.graph_objs as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from scipy import ndimage
import cv2
import librosa
from tensorflow.keras.preprocessing.sequence import pad_sequences
import seaborn as sns
import json

try:
    from xgboost import XGBRegressor
except ImportError:
    print("XGBoost is not installed. Install it with `pip install xgboost` if you plan to use XGBoostRegressor.")


class MLModelAnalysis:
    def __init__(self, model_type='linear_regression'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}

        # Initialize the appropriate model based on model_type
        if model_type == 'linear_regression':
            self.model = LinearRegression()
        elif model_type == 'decision_tree':
            self.model = DecisionTreeRegressor()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor()
        elif model_type == 'svm':
            self.model = SVR()
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor()
        elif model_type == 'knn':
            self.model = KNeighborsRegressor()
        elif model_type == 'ada_boost':
            self.model = AdaBoostRegressor()
        elif model_type == 'mlp':
            self.model = MLPRegressor()
        elif model_type == 'xgboost':
            self.model = XGBRegressor() if 'XGBRegressor' in globals() else None
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        if self.model is None:
            raise ValueError(f"Model type '{model_type}' requires additional dependencies. Ensure all packages are installed.")

    def preprocess_data(self, csv_file, x_elements, y_element):
        data = pd.read_csv(csv_file)
        
        # Remove rows with null values in any of the relevant columns
        data = data.dropna(subset=x_elements + [y_element])
    
        # Label encode if there are categorical columns
        for col in x_elements + [y_element]:
            if data[col].dtype == 'object':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                self.encoders[col] = le
    
        # Handle date as a potential x_element
        for element in x_elements:
            if element == 'Date':
                data[element] = pd.to_datetime(data[element]).map(pd.Timestamp.toordinal)
    
        X = np.array(data[x_elements])
        Y = data[y_element]
        return X, Y

    def plot_model(self, X_train, Y_train, X_train_scaled, layout):
        trace_actual = go.Scatter(x=X_train[:, 0].flatten(), y=Y_train, mode='markers', name='Actual')
        trace_predicted = go.Scatter(x=X_train[:, 0].flatten(), y=self.model.predict(X_train_scaled), mode='lines', name='Predicted')
        fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)
        fig.show()

    def train_and_evaluate(self, csv_file, x_elements, y_element, model_save_path=None):
        X, Y = self.preprocess_data(csv_file, x_elements, y_element)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
        
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, Y_train)
        
        # Calculate predictions and metrics
        Y_train_pred = self.model.predict(X_train_scaled)
        Y_test_pred = self.model.predict(X_test_scaled)

        train_r2 = r2_score(Y_train, Y_train_pred)
        test_r2 = r2_score(Y_test, Y_test_pred)
        train_mse = mean_squared_error(Y_train, Y_train_pred)
        test_mse = mean_squared_error(Y_test, Y_test_pred)

        print(f'{self.model_type.capitalize()} - Train R-squared score: {train_r2}')
        print(f'{self.model_type.capitalize()} - Test R-squared score: {test_r2}')
        print(f'{self.model_type.capitalize()} - Train Mean Squared Error: {train_mse}')
        print(f'{self.model_type.capitalize()} - Test Mean Squared Error: {test_mse}')

        # Plot if the model is linear or SVM and x_elements has one feature (for simple visualization)
        if self.model_type in ['linear_regression', 'svm'] and len(x_elements) == 1:
            layout = go.Layout(
                title=f'{y_element} vs. {x_elements[0]}',
                xaxis=dict(title=x_elements[0]),
                yaxis=dict(title=y_element)
            )
            self.plot_model(X_train, Y_train, X_train_scaled, layout)

        if model_save_path:
            with open(model_save_path, 'wb') as f:
                pickle.dump({'model': self.model, 'scaler': self.scaler, 'encoders': self.encoders}, f)
            print(f'Model and encoders saved to {model_save_path}')

    def load_model_and_predict(self, model_path, input_data):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.encoders = model_data.get('encoders', {})

        for col, encoder in self.encoders.items():
            if col in input_data:
                input_data[col] = encoder.transform([input_data[col]])[0]

        for col in input_data:
            if col == 'Date':
                input_data[col] = pd.to_datetime(input_data[col]).toordinal()

        X_new = np.array([input_data[col] for col in sorted(input_data.keys())]).reshape(1, -1)
        X_new_scaled = self.scaler.transform(X_new)

        prediction = self.model.predict(X_new_scaled)
        return prediction[0]

class ImageClassifier:
    def __init__(self):
        pass

    def _get_files(self, directory):
        """
        Count the total number of files in a given directory, including subdirectories.
        """
        if not os.path.exists(directory):
            return 0
        count = 0
        for current_path, dirs, files in os.walk(directory):
            for dr in dirs:
                count += len(glob.glob(os.path.join(current_path, dr + "/*")))
        return count
    def _load_csv_data(self, csv_file, img_column, label_column, img_size=(224, 224)):
        data = pd.read_csv(csv_file)
        images = []
        labels = []

        for _, row in data.iterrows():
            img = load_img(row[img_column], target_size=img_size)
            img = img_to_array(img) / 255.0  # Normalize image
            images.append(img)
            labels.append(row[label_column])

        images = np.array(images)
        labels = pd.get_dummies(labels).values  # One-hot encode labels
        return images, labels

    def _select_model(self, num_classes, dataset_size, force=None, finetune=False):
        """
        Selects the appropriate model based on dataset size or a forced model choice.

        Args:
        - num_classes: Number of output classes.
        - dataset_size: Number of samples in the dataset.
        - force: (Optional) Force selection of a specific model.
        - finetune: (Optional) If True, allows fine-tuning of certain layers in pretrained models.
        """
        if force and force.startswith("cnn"):
            return getattr(self, f"_build_{force}_model")(num_classes)
        elif force == "simple_cnn" or (force is None and dataset_size < 1000):
            return self._build_simple_cnn(num_classes)
        elif force == "vgg16" or (force is None and dataset_size < 5000):
            return self._build_vgg16_model(num_classes, finetune)
        elif force == "resnet50" or (force is None and dataset_size >= 5000):
            return self._build_resnet50_model(num_classes, finetune)
        elif force == "mobilenet":
            return self._build_mobilenet_model(num_classes, finetune)
        elif force == "inceptionv3":
            return self._build_inceptionv3_model(num_classes, finetune)
        elif force == "densenet":
            return self._build_densenet_model(num_classes, finetune)
        elif force == "efficientnet":
            return self._build_efficientnet_model(num_classes, finetune)
        elif force == "xception":
            return self._build_xception_model(num_classes, finetune)
        elif force == "nasnetmobile":
            return self._build_nasnet_model(num_classes, finetune)
        elif force == "inceptionresnetv2":
            return self._build_inception_resnetv2_model(num_classes, finetune)
        else:
            raise ValueError("Invalid model choice. Please specify 'simple_cnn', 'vgg16', 'resnet50', 'mobilenet', 'inceptionv3', 'densenet', 'efficientnet', 'xception', 'nasnetmobile', 'inceptionresnetv2', or 'cnn1' to 'cnn10'.")

    # Predefined simple CNN model
    def _build_simple_cnn(self, num_classes):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # Builds base model with optional fine-tuning for transfer learning models
    def _build_model_with_base(self, base_model, num_classes, finetune, dense_units=256):
        if not finetune:
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for layer in base_model.layers[:-4]:
                layer.trainable = False
        x = Flatten()(base_model.output)
        x = Dense(dense_units, activation="relu")(x)
        predictions = Dense(num_classes, activation="softmax")(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # Pretrained Models
    def _build_vgg16_model(self, num_classes, finetune):
        return self._build_model_with_base(VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3)), num_classes, finetune)

    def _build_resnet50_model(self, num_classes, finetune):
        return self._build_model_with_base(ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3)), num_classes, finetune, dense_units=512)

    def _build_mobilenet_model(self, num_classes, finetune):
        return self._build_model_with_base(MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3)), num_classes, finetune)

    def _build_inceptionv3_model(self, num_classes, finetune):
        return self._build_model_with_base(InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3)), num_classes, finetune, dense_units=512)

    def _build_densenet_model(self, num_classes, finetune):
        return self._build_model_with_base(DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3)), num_classes, finetune, dense_units=512)

    def _build_efficientnet_model(self, num_classes, finetune):
        return self._build_model_with_base(EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3)), num_classes, finetune)

    def _build_xception_model(self, num_classes, finetune):
        return self._build_model_with_base(Xception(weights="imagenet", include_top=False, input_shape=(224, 224, 3)), num_classes, finetune, dense_units=512)

    def _build_nasnet_model(self, num_classes, finetune):
        return self._build_model_with_base(NASNetMobile(weights="imagenet", include_top=False, input_shape=(224, 224, 3)), num_classes, finetune)

    def _build_inception_resnetv2_model(self, num_classes, finetune):
        return self._build_model_with_base(InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3)), num_classes, finetune, dense_units=512)

    # Custom CNNs (cnn1 to cnn10)
    def _build_cnn1_model(self, num_classes):
        # Inspired by LeNet-5 (simple and effective for smaller datasets)
        model = Sequential([
            Conv2D(32, (5, 5), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (5, 5), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _build_cnn2_model(self, num_classes):
        # A simplified version of AlexNet with fewer parameters
        model = Sequential([
            Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((3, 3), strides=2),
            Conv2D(256, (5, 5), activation='relu', padding='same'),
            MaxPooling2D((3, 3), strides=2),
            Conv2D(384, (3, 3), activation='relu', padding='same'),
            Conv2D(384, (3, 3), activation='relu', padding='same'),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((3, 3), strides=2),
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _build_cnn3_model(self, num_classes):
        # Based on VGG-16 but smaller
        model = Sequential([
            Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _build_cnn4_model(self, num_classes):
        # Inception-inspired model with parallel convolutions
        model = Sequential([
            Conv2D(64, (7, 7), strides=2, activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((3, 3), strides=2),
            Conv2D(192, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((3, 3), strides=2),
            Conv2D(128, (1, 1), activation='relu', padding='same'),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((3, 3), strides=2),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _build_cnn5_model(self, num_classes):
        # Inspired by ResNet, with skip connections
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        skip = x
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.add([x, skip])  # Skip connection
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _build_cnn6_model(self, num_classes):
        # Modified DenseNet-like model with dense connections
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
        x3 = tf.keras.layers.concatenate([x, x1, x2])  # Dense connection
        x4 = Conv2D(64, (3, 3), activation='relu', padding='same')(x3)
        x = MaxPooling2D((2, 2))(x4)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _build_cnn7_model(self, num_classes):
        # Deep, multi-layered architecture similar to more recent CNNs
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # Continue to define cnn3 to cnn10 models similarly with progressive depth and complexity...
    def plot_accuracy(self, history):
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.show()

    def plot_confusion_matrix(self, model, generator):
        y_true = generator.classes
        y_pred = np.argmax(model.predict(generator), axis=1)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=generator.class_indices.keys())
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()

    def img_train(self, train_dir=None, test_dir=None, csv_file=None, img_column=None, label_column=None, 
                  epochs=10, device="cpu", force=None, finetune=False):
        if device.lower() == "cuda":
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) == 0:
                raise RuntimeError("No CUDA devices found. Make sure CUDA is properly configured.")
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        elif device.lower() != "cpu":
            raise ValueError("Invalid device specified. Please specify either 'cpu' or 'cuda'.")

        if csv_file:  # Load data from CSV if csv_file is provided
            images, labels = self._load_csv_data(csv_file, img_column, label_column)
            train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
            train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
            val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32)
            num_classes = labels.shape[1]
        else:  # Load data from directories if directory paths are provided
            train_samples = self._get_files(train_dir)
            num_classes = len(glob.glob(train_dir + "/*"))

            if test_dir:
                train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
                test_datagen = ImageDataGenerator(rescale=1./255)
                train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32)
                validation_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32)
            else:
                train_datagen = ImageDataGenerator(
                    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)
                train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, subset='training')
                validation_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, subset='validation')

        model = self._select_model(num_classes, len(train_images) if csv_file else train_samples, force, finetune)

        if csv_file:
            history = model.fit(
                train_dataset, epochs=epochs, validation_data=val_dataset,
                callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.000001)],
                shuffle=True
            )
        else:
            history = model.fit(
                train_generator, epochs=epochs, validation_data=validation_generator,
                callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.000001)],
                shuffle=True
            )

        model.save('selected_model.h5')
        print(f"Model training completed and saved as 'selected_model.h5'")

        self.plot_accuracy(history)

        if not csv_file:
            self.plot_confusion_matrix(model, validation_generator)


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
        
