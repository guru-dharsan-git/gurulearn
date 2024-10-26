import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
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
from sklearn.metrics import mean_absolute_error

# Linear regression-related methods encapsulated in the LinearRegressionAnalysis class
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objs as go

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


