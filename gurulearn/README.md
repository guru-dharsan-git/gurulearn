# ######                                                                      MLModelAnalysis:                                                                  #######

**MLModelAnalysis** is a versatile and reusable Python class designed to streamline training, evaluation, and prediction processes for various machine learning regression models. This tool allows users to switch seamlessly between models, perform consistent data preprocessing, evaluate models, and make predictions, making it highly adaptable for different machine learning tasks.

## Supported Models

- Linear Regression (`linear_regression`)
- Decision Tree Regressor (`decision_tree`)
- Random Forest Regressor (`random_forest`)
- Support Vector Machine (`svm`)
- Gradient Boosting Regressor (`gradient_boosting`)
- K-Nearest Neighbors (`knn`)
- AdaBoost Regressor (`ada_boost`)
- Neural Network (MLP Regressor) (`mlp`)
- XGBoost Regressor (`xgboost`)

## Installation

To use **MLModelAnalysis**, install the following dependencies:
```bash
pip install scikit-learn pandas numpy plotly xgboost
```

## Usage

### 1. Initializing the Model

Initialize the **MLModelAnalysis** class by specifying the `model_type` parameter, which sets the machine learning model you wish to use.

```python
from ml_model_analysis import MLModelAnalysis

# Initialize with Linear Regression
analysis = MLModelAnalysis(model_type='linear_regression')

# Initialize with Random Forest
analysis = MLModelAnalysis(model_type='random_forest')

# Initialize with XGBoost
analysis = MLModelAnalysis(model_type='xgboost')
```

### 2. Training and Evaluating the Model

The `train_and_evaluate` method handles data preprocessing, model training, and metric evaluation. Optionally, it can save the trained model, scaler, and encoders for later use.

#### Parameters
- `csv_file`: Path to the CSV file containing the dataset.
- `x_elements`: List of feature columns.
- `y_element`: Name of the target column.
- `model_save_path` (Optional): Path to save the trained model, scaler, and encoders.

#### Example
```python
# Set the parameters
csv_file = 'data.csv'                     # Path to the data file
x_elements = ['feature1', 'feature2']      # Feature columns
y_element = 'target'                       # Target column

# Initialize the model
analysis = MLModelAnalysis(model_type='random_forest')

# Train and evaluate the model
analysis.train_and_evaluate(csv_file=csv_file, x_elements=x_elements, y_element=y_element, model_save_path='random_forest_model.pkl')
```
After running this code, the model displays R-squared and Mean Squared Error (MSE) metrics for both the training and test sets. If `model_save_path` is specified, the model will be saved for future predictions.

### 3. Loading the Model and Making Predictions

The `load_model_and_predict` method allows you to load a saved model and make predictions on new input data.

#### Parameters
- `model_path`: Path to the saved model file.
- `input_data`: Dictionary containing feature names and values for prediction.

#### Example
```python
# Define input data for prediction
input_data = {
    'feature1': 5.1,
    'feature2': 2.3
}

# Load the model and make a prediction
prediction = analysis.load_model_and_predict(model_path='random_forest_model.pkl', input_data=input_data)
print(f'Prediction: {prediction}')
```

### 4. Visualization

For `linear_regression` or `svm` models with only one feature, the `train_and_evaluate` method will automatically generate a Plotly plot of actual vs. predicted values for quick visualization.

#### Example Use Cases

- **Regression Analysis with Random Forest**
    ```python
    analysis = MLModelAnalysis(model_type='random_forest')
    analysis.train_and_evaluate(csv_file='data.csv', x_elements=['feature1', 'feature2'], y_element='target', model_save_path='random_forest_model.pkl')
    ```

- **Quick Prediction with a Pre-trained Model**
    ```python
    prediction = analysis.load_model_and_predict(model_path='random_forest_model.pkl', input_data={'feature1': 5.1, 'feature2': 2.3})
    print(f'Prediction: {prediction}')
    ```

- **Effortless Model Switching**
    ```python
    # Specify a new model type to use a different algorithm
    analysis = MLModelAnalysis(model_type='xgboost')
    ```

## Additional Notes

- **Plotting**: Visualizations are supported for linear models and SVM with single-feature datasets.
- **Model Saving**: The `model_save_path` parameter in `train_and_evaluate` stores the model, scaler, and encoders, allowing consistent predictions when reloading the model later.
- **Dependencies**: Ensure required libraries are installed (`scikit-learn`, `pandas`, `numpy`, `plotly`, and `xgboost`).

## License

This project is licensed under the MIT License.




# ######                                                                      image_classify:                                                                  ###### #

# example usages:

image_classifier = ImageClassifier()
image_classifier.img_train(
    train_dir='train',
    test_dir='test',  # Optional, can be None to split training data
    epochs=1,
    device='cpu',
    force='vgg16',
    finetune=True
)


image_classifier = ImageClassifier()
image_classifier.img_train(
    train_dir='train',
    test_dir='test',  # Optional, can be None to split training data
    epochs=1,
    device='cpu',
    force='vgg16'
)


# Using directory data
image_classifier = ImageClassifier()
image_classifier.img_train(
    train_dir='path/to/train',
    test_dir='path/to/test',
    epochs=10,
    device='cuda',
    force='resnet50',
    finetune=True
)

# Using CSV data
image_classifier.img_train(
    csv_file='path/to/data.csv',
    img_column='image_path',  # Column name in CSV containing image paths
    label_column='label',      # Column name in CSV containing labels
    epochs=10,
    device='cuda',
    force='resnet50',
    finetune=True
)


consist of 20 cnn architecture excluding finetuning
# Instantiate and train the classifier with a specific model and fine-tuning enabled
image_classifier = ImageClassifier()
image_classifier.img_train("path_to_train_dir", "path_to_test_dir", epochs=10, device="cuda", force="efficientnet", finetune=True)


# ####                                                                     **CTScanProcessor**                                                                  #### # 

---

# CTScanProcessor

**CTScanProcessor** is a Python class designed for advanced processing and quality evaluation of CT scan images. This tool is highly beneficial for applications in medical imaging, data science, and deep learning, providing noise reduction, contrast enhancement, detail preservation, and quality evaluation.

## Features

- **Sharpening**: Enhances image details by applying a sharpening filter.
- **Median Denoising**: Reduces noise while preserving edges using a median filter.
- **Contrast Enhancement**: Enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
- **Quality Metrics**: Calculates image quality metrics such as MSE, PSNR, SNR, and Detail Preservation Ratio to evaluate the effectiveness of processing.
- **Image Comparison**: Creates side-by-side comparisons of original and processed images.

## Installation

This class requires the following libraries:
- OpenCV
- NumPy
- SciPy

To install the required dependencies, use:
```bash
pip install opencv-python-headless numpy scipy
```

## Usage

1. **Initialize the Processor**
   ```python
   from ct_scan_processor import CTScanProcessor
   processor = CTScanProcessor(kernel_size=5, clip_limit=2.0, tile_grid_size=(8, 8))
   ```

2. **Process a CT Scan**
   Use the `process_ct_scan` method to process a CT scan image and get quality metrics.
   ```python
   denoised, metrics = processor.process_ct_scan("path_to_ct_scan.jpg", "output_folder", compare=True)
   ```

3. **Quality Metrics**
   After processing, the class returns metrics such as Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), Signal-to-Noise Ratio (SNR), and Detail Preservation Ratio.

4. **Compare Images**
   If `compare=True`, a side-by-side comparison image is saved in the specified comparison folder.

### Example

```python
if __name__ == "__main__":
    processor = CTScanProcessor()
    denoised, metrics = processor.process_ct_scan("path_to_ct_scan.jpg", "output_folder", compare=True)
```

## Quality Metrics

The following metrics are calculated to evaluate the quality of the denoised image:

- **MSE**: Mean Squared Error between the original and processed images.
- **PSNR**: Peak Signal-to-Noise Ratio to measure image quality.
- **SNR**: Signal-to-Noise Ratio to measure signal strength relative to noise.
- **Detail Preservation**: Percentage of preserved details after processing.

## Methods

- `sharpen(image)`: Sharpens the input image.
- `median_denoise(image)`: Denoises the input image using a median filter.
- `enhance_contrast(image)`: Enhances contrast using CLAHE.
- `enhanced_denoise(image_path)`: Processes a CT scan image with denoising, contrast enhancement, and sharpening.
- `evaluate_quality(original, denoised)`: Computes MSE, PSNR, SNR, and Detail Preservation.
- `compare_images(original, processed, output_path)`: Saves a side-by-side comparison of the original and processed images.
- `process_ct_scan(input_path, output_folder, comparison_folder="comparison", compare=False)`: Runs the full CT scan processing pipeline and saves the results.

## License

This project is licensed under the MIT License.

## Contributions

Contributions are welcome! Feel free to submit pull requests or open issues.
