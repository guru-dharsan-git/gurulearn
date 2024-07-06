import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os 
import cv2
import scipy
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from keras.layers import Convolution2D,Dense,MaxPool2D,Activation,Dropout,Flatten
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
import os
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import sys

def plot_linear_regression(csv_file, x_name, y_name, x_element, y_element):
    # Load data
    t = pd.read_csv(csv_file)


    if x_element == 'Date':
        t['Date'] = pd.to_datetime(t['Date'])

    # Plotly layout
    layout = go.Layout(
        title=f'{y_name} vs. {x_name}',
        xaxis=dict(title=x_name, titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
        yaxis=dict(title=y_name, titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f'))
    )


    X = np.array(t[x_element]).reshape(-1, 1)
    Y = t[y_element]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)


    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    lm = LinearRegression()
    lm.fit(X_train_scaled, Y_train)


    trace0 = go.Scatter(x=X_train.flatten(), y=Y_train, mode='markers', name='Actual')
    trace1 = go.Scatter(x=X_train.flatten(), y=lm.predict(X_train_scaled), mode='lines', name='Predicted')
    t_data = [trace0, trace1]
    layout.xaxis.title.text = x_name
    layout.yaxis.title.text = y_name
    fig = go.Figure(data=t_data, layout=layout)

    
    fig.show()



from sklearn.metrics import r2_score, mean_squared_error

def linear_regression_accuracy(csv_file, x_name, y_name, x_element, y_element):
    # Load data
    t = pd.read_csv(csv_file)

    # Convert 'Date' column to pandas datetime object if used as X-axis element
    if x_element == 'Date':
        t['Date'] = pd.to_datetime(t['Date'])

    # Prepare data for regression
    X = np.array(t[x_element]).reshape(-1, 1)
    Y = t[y_element]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

    # Standardize features
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Linear regression
    lm = LinearRegression()
    lm.fit(X_train_scaled, Y_train)

    # Predictions
    Y_train_pred = lm.predict(X_train_scaled)
    Y_test_pred = lm.predict(X_test_scaled)

    # Calculate accuracy metrics
    train_r2 = r2_score(Y_train, Y_train_pred)
    test_r2 = r2_score(Y_test, Y_test_pred)
    train_mse = mean_squared_error(Y_train, Y_train_pred)
    test_mse = mean_squared_error(Y_test, Y_test_pred)

    # Print accuracy metrics
    print(f'Train R-squared score: {train_r2}')
    print(f'Test R-squared score: {test_r2}')
    print(f'Train Mean Squared Error: {train_mse}')
    print(f'Test Mean Squared Error: {test_mse}')






















#gvgg:
def get_files(directory):
    if not os.path.exists(directory):
        return 0
    count=0
    # crawls inside folders
    for current_path,dirs,files in os.walk(directory):
        for dr in dirs:
            count+= len(glob.glob(os.path.join(current_path,dr+"/*")))
    return count

def vgg_train(train_dir, test_dir, epochs, device="cpu"):
    # Set device to CPU or CUDA (GPU)
    if device.lower() == "cuda":
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) == 0:
            raise RuntimeError("No CUDA devices found. Make sure CUDA is properly configured.")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    elif device.lower() != "cpu":
        raise ValueError("Invalid device specified. Please specify either 'cpu' or 'cuda'.")

    train_samples = get_files(train_dir)
    num_classes = len(glob.glob(train_dir + "/*")) 
    test_samples = get_files(test_dir)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    input_shape = (224,224,3)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224,224), batch_size=32)
    test_generator = test_datagen.flow_from_directory(test_dir, shuffle=True, target_size=(224,224), batch_size=32)

    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='relu', name="conv2d_1"))
    model.add(MaxPooling2D(pool_size=(3, 3), name="max_pooling2d_1"))
    model.add(Conv2D(32, (3, 3), activation='relu', name="conv2d_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_2"))
    model.add(Conv2D(64, (3, 3), activation='relu', name="conv2d_3"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_3"))   
    model.add(Flatten(name="flatten_1"))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))          
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    validation_generator = train_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32
    )

    def create_Base_model_from_VGG16():  
        model = VGG16(
            weights="imagenet",
            include_top=False, 
            input_shape=(224,224, 3) # goruntu boyutu
        ) 
        for layer in model.layers:
            layer.trainable = False
        return model 

    create_Base_model_from_VGG16().summary()

    def add_custom_layers():
        model = create_Base_model_from_VGG16()
        x = model.output
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        predictions = tf.keras.layers.Dense(num_classes, activation="softmax")(x)   
        final_model = tf.keras.models.Model(inputs=model.input, outputs=predictions) 
        final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
        return final_model

    validation_generator = train_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32
    )

    model_from_vgg16 = add_custom_layers()

    history2 = model_from_vgg16.fit(
        train_generator,
        steps_per_epoch=None,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=None,
        verbose=1,
        callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.000001)],
        use_multiprocessing=False,
        shuffle=True
    )

    model_from_vgg16.save('model_GVGG16.h5')



sys.stdout.reconfigure(encoding='utf-8')

def audio_classify(audio_dataset_path, metadata_path, num_epochs):
    try:
        import os
        import numpy as np
        import pandas as pd
        import librosa
        from sklearn.preprocessing import LabelEncoder, to_categorical
        from sklearn.model_selection import train_test_split
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation
        from keras.callbacks import ModelCheckpoint
        from datetime import datetime
        from tqdm import tqdm

        # Load metadata
        metadata = pd.read_csv(metadata_path)

        # Feature extractor function
        def features_extractor(file):
            audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
            mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
            return mfccs_scaled_features

        # Extract features from each audio file
        extracted_features = []
        for index_num, row in tqdm(metadata.iterrows()):
            file_name = os.path.join(os.path.abspath(audio_dataset_path), 'fold'+str(row["fold"]), str(row["slice_file_name"]))
            final_class_labels = row["class"]
            data = features_extractor(file_name)
            extracted_features.append([data, final_class_labels])

        # Convert extracted features to DataFrame
        extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])

        # Split dataset into independent and dependent datasets
        X = np.array(extracted_features_df['feature'].tolist())
        y = np.array(extracted_features_df['class'].tolist())

        # Label encoding
        labelencoder = LabelEncoder()
        y = to_categorical(labelencoder.fit_transform(y))

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Define model
        num_labels = y.shape[1]
        model = Sequential()

        # First layer
        model.add(Dense(100, input_shape=(40,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Second layer
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Third layer
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Final layer
        model.add(Dense(num_labels))
        model.add(Activation('softmax'))

        # Model summary
        model.summary()

        # Compile model
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        # Create a ModelCheckpoint callback
        checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.keras', verbose=1, save_best_only=True)

        # Record start time
        start = datetime.now()

        # Train model
        model.fit(X_train, y_train, batch_size=32, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)

        # Calculate training duration
        duration = datetime.now() - start
        print("Training completed in time: ", duration)

        # Evaluate model
        test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print("Test Accuracy: ", test_accuracy[1])

        # Return test accuracy, model, and label encoder
        return test_accuracy[1], model, labelencoder

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None


def audio_classify_predict(model, audio_file, labelencoder):
    # Load the audio file
    audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')

    # Extract MFCC features
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    # Reshape the MFCC features
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

    # Predict the class using the model
    predicted_label = np.argmax(model.predict(mfccs_scaled_features), axis=-1)

    # Get the class name from the label encoder
    prediction_class = labelencoder.inverse_transform(predicted_label)

    return prediction_class[0]


# First, train the audio classification model and get the model, test accuracy, and label encoder
# test_accuracy, model, labelencoder = audio_classify("path_to_audio_dataset", "path_to_metadata", num_epochs=10)

# if model is not None and labelencoder is not None:
#     # Now that we have the trained model and label encoder, we can use them for predictions
    
#     # Example usage of audio_classify_predict function
#     prediction = audio_classify_predict(model, "path_to_audio_file_for_prediction.wav", labelencoder)
#     print("Predicted class:", prediction)
# else:
#     print("Error occurred during model training.")

