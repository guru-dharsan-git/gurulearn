import tensorflow as tf
import pandas as pd
import numpy as np
import glob
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.applications import ( # type: ignore
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
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay
)

import os
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


def super_image_model(train_dir, test_dir, target_size=(224, 224), batch_size=32, 
                      learning_rate=0.001, epochs=5):
    """
    Trains a VGG16-based feature extraction model on the given dataset.

    Parameters:
        train_dir (str): Path to the training dataset directory.
        test_dir (str): Path to the testing dataset directory.
        target_size (tuple): Target size for image resizing (default: (224, 224)).
        batch_size (int): Batch size for training and testing (default: 32).
        learning_rate (float): Learning rate for the optimizer (default: 0.001).
        epochs (int): Number of training epochs (default: 5).

    Returns:
        Model: The trained classification model.
        Model: The feature extractor model.
    """
    # Step 1: Load and preprocess the dataset
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=target_size, batch_size=batch_size, 
        class_mode='categorical', shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=target_size, batch_size=batch_size, 
        class_mode='categorical', shuffle=False
    )

    num_classes = len(train_generator.class_indices)

    # Step 2: Load the pre-trained VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*target_size, 3))

    # Step 3: Preprocessing and modification of backbone network
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='fc128')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Classification model
    classification_layer = Dense(num_classes, activation='softmax', name='classification')(x)
    full_model = Model(inputs=base_model.input, outputs=classification_layer)

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    full_model.compile(
        optimizer=Adam(learning_rate=learning_rate), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    # Train the model
    full_model.fit(train_generator, validation_data=test_generator, epochs=epochs)

    # Save the model to a file
    full_model.save('model.h5')



