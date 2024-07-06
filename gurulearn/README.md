python setup.py sdist bdist_wheel


linear regression:
# plot_linear_regression(csv_file, x_name, y_name, x_element, y_element)
# linear_regression_accuracy(csv_file, x_name, y_name, x_element, y_element)

gvgg:
# vgg_train("train", "test", 1)           # Train for 5 epochs using CPU (default)
# vgg_train("train_dataset", "test_dataset", 5, "cuda")   # Train for 5 epochs using CUDA
# vgg_train("train_dataset", "test_dataset", 5, "cpu")    # Train for 5 epochs using CPU

audio_classify:
# First, train the audio classification model and get the model, test accuracy, and label encoder
# test_accuracy, model, labelencoder = audio_classify("path_to_audio_dataset", "path_to_metadata", num_epochs=10)

# if model is not None and labelencoder is not None:
#     # Now that we have the trained model and label encoder, we can use them for predictions
    
#     # Example usage of audio_classify_predict function
#     prediction = audio_classify_predict(model, "path_to_audio_file_for_prediction.wav", labelencoder)
#     print("Predicted class:", prediction)
# else:
#     print("Error occurred during model training.")