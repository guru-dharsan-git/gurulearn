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





### ImageClassifier ###

The `ImageClassifier` class in `gurulearn` provides an extensive set of tools for image classification, supporting both custom CNNs and pre-trained models for transfer learning. It includes utilities for data loading, model selection based on dataset size, and model training and evaluation.

#### Key Features

- **Flexible Model Selection**: Automatically selects a model based on dataset size, or allows users to force a specific model (custom CNNs or pre-trained models like VGG16, ResNet50, MobileNet, etc.).
- **Data Loading Options**: Supports loading images from directories or CSV files.
- **Transfer Learning with Fine-Tuning**: Offers optional fine-tuning of pre-trained models for enhanced accuracy.
- **Custom CNN Architectures**: Provides a variety of custom CNN models (`cnn1` to `cnn10`) for different levels of complexity.
- **Evaluation Tools**: Built-in functions for visualizing training accuracy and displaying confusion matrices.

#### Usage

To use the `ImageClassifier` class, follow these steps:

#### 1. Importing and Initializing

```python
from gurulearn import ImageClassifier

# Initialize the image classifier
image_classifier = ImageClassifier()
```

#### 2. Training the Model

The `img_train` method trains an image classification model using either a directory of images or a CSV file with image paths and labels.

```python
image_classifier.img_train(
    train_dir="path/to/train/data",  # or specify csv_file, img_column, label_column for CSV data
    test_dir="path/to/test/data",    # Optional, only if test data is in a separate directory
    epochs=10,
    device="cpu",                    # Set to "cuda" if using a GPU
    force="vgg16",                   # Force specific model choice (optional)
    finetune=True                    # Fine-tune pre-trained models (optional)
)
```

Parameters:
- **train_dir**: Directory containing training images (organized in subdirectories by class) or `csv_file` for CSV data.
- **test_dir**: Directory containing test images (optional, use if separate from `train_dir`).
- **csv_file**: Path to CSV file if loading data from CSV.
- **img_column**: Column name in the CSV containing image paths.
- **label_column**: Column name in the CSV containing labels.
- **epochs**: Number of training epochs (default: 10).
- **device**: Device to use for training, either `"cpu"` or `"cuda"` (default: `"cpu"`).
- **force**: Specify a particular model (options include `"simple_cnn"`, `"vgg16"`, `"resnet50"`, etc.).
- **finetune**: Whether to fine-tune pre-trained models (default: `False`).

#### Supported Models

The `ImageClassifier` class can select models based on dataset size or through forced selection. Models include:

- **Custom CNNs**: `cnn1` to `cnn10` (e.g., simple CNN, ResNet-inspired, Inception-inspired).
- **Pre-trained Models**: `"vgg16"`, `"resnet50"`, `"mobilenet"`, `"inceptionv3"`, `"densenet"`, `"efficientnet"`, `"xception"`, `"nasnetmobile"`, `"inceptionresnetv2"`.
- **Model Auto-Selection**: Based on dataset size, the class can automatically select the appropriate model.

#### Example Workflow

```python
# Initialize the classifier
image_classifier = ImageClassifier()

# Train the model using images organized in directories
image_classifier.img_train(
    train_dir="data/train_images",
    test_dir="data/test_images",
    epochs=20,
    device="cuda",      # Use GPU if available
    force="resnet50",   # Force ResNet50 model
    finetune=True       # Enable fine-tuning
)
```

#### 3. Plotting Training Accuracy

The `plot_accuracy` method displays the training and validation accuracy across epochs.

```python
history = image_classifier.img_train(train_dir="data/train_images", epochs=10)
image_classifier.plot_accuracy(history)
```

#### 4. Displaying Confusion Matrix

After training, you can plot a confusion matrix to evaluate the model's predictions on validation data.

```python
image_classifier.plot_confusion_matrix(model, validation_generator)
```

#### Files Created

- **Model File**: `selected_model.h5` - The trained model is saved for future use.

### Model Selection Guidelines

The `_select_model` method automatically chooses a model based on dataset size if no specific model is forced. For smaller datasets, simpler models (like `simple_cnn` or `vgg16`) are preferred, while for larger datasets, deeper models (like `resnet50`) are selected for improved accuracy.

#### Model Architectures

Each custom CNN model (from `cnn1` to `cnn10`) and pre-trained model architecture (VGG16, ResNet50, etc.) provides a unique structure optimized for specific types of datasets and computational capacities.



# ####                                                                     **CTScanProcessor**                                                                  #### # 


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


### AudioRecognition ###

The `AudioRecognition` class in `gurulearn` provides tools for audio data augmentation, feature extraction, model training, and prediction, making it suitable for tasks like audio classification and speech recognition.

#### Key Features

- **Data Augmentation**: Supports time-stretching, pitch-shifting, and noise addition for audio data augmentation.
- **Feature Extraction**: Extracts MFCCs, FAISS, and spectral contrast features from audio signals.
- **Model Training**: Trains a deep learning model for audio classification using a Conv1D and BiLSTM-based architecture.
- **Prediction**: Predicts the class of a given audio file based on a trained model.

#### Usage

To use the `AudioRecognition` class, follow these steps:

#### 1. Importing and Initializing

```python
from gurulearn import AudioRecognition

# Initialize the audio recognition class
audio_recognition = AudioRecognition()
```

#### 2. Loading Data with Augmentation

The `load_data_with_augmentation` method loads audio data from a specified directory and performs augmentation to improve model generalization.

```python
data_dir = "path/to/audio/data"
X, y = audio_recognition.load_data_with_augmentation(data_dir)
```

This method returns feature vectors (`X`) and labels (`y`) for training.

#### 3. Training the Model

The `audiotrain` method trains an audio classification model. This method also generates a confusion matrix and training history plot, which are saved in the specified model directory.

```python
audio_recognition.audiotrain(
    data_path="path/to/audio/data",
    epochs=50,
    batch_size=32,
    test_size=0.2,
    learning_rate=0.001,
    model_dir='model_folder'
)
```

Parameters:
- **data_path**: Directory path where audio data is stored (organized by class label).
- **epochs**: Number of training epochs (default: 50).
- **batch_size**: Training batch size (default: 32).
- **test_size**: Proportion of data to use for testing (default: 0.2).
- **learning_rate**: Initial learning rate for model training (default: 0.001).
- **model_dir**: Directory where the model and label mappings will be saved.

#### 4. Predicting the Class of an Audio File

After training, you can predict the class of a new audio file using the `predict` or `predict_class` methods.

```python
# Path to the input audio file
input_wav = "path/to/audio/file.wav"

# Predict the label of the audio file
predicted_label = audio_recognition.predict(input_wav)
print(f"Predicted Label: {predicted_label}")
```

The `predict` method returns the predicted label (text), while `predict_class` returns the numeric class index.

#### Example Workflow

```python
# Initialize the audio recognition instance
audio_recognition = AudioRecognition()

# Load data and perform augmentation
X, y = audio_recognition.load_data_with_augmentation('data/audio_files')

# Train the model on the audio dataset
audio_recognition.audiotrain(
    data_path='data/audio_files',
    epochs=30,
    batch_size=32,
    learning_rate=0.001
)

# Predict the class of a new audio sample
predicted_label = audio_recognition.predict('data/test_audio.wav')
print("Predicted Label:", predicted_label)
```

#### Files Created

- **Confusion Matrix**: `confusion_matrix.png` - Saved in the current directory after training.
- **Training History**: `training_history.png` - Contains plots for model accuracy and loss.
- **Model**: `audio_recognition_model.h5` - Saved in the specified model directory.
- **Label Mapping**: `label_mapping.json` - Contains mappings of class indices to labels.

##  Introducing FlowBot  
`FlowBot` is a flexible framework for creating **dynamic, guided interactions** (chatbots, booking systems, surveys) that adapt to user input and filter datasets in real time. Perfect for travel booking, customer support, or personalized recommendations!  


---

## Installation  
```bash
pip install gurulearn
```

---

##  Quick Start  
**Build a Travel Booking Bot in 5 Steps**:  

```python
import pandas as pd
from gurulearn import FlowBot

# Sample dataset
hotels = pd.DataFrame({
    'destination': ['Paris', 'Tokyo', 'New York'],
    'price_range': ['$$$', '$$', '$'],
    'hotel_name': ['Luxury Palace', 'Mountain View', 'Downtown Inn']
})

# Initialize FlowBot
bot = FlowBot(hotels)

# Collect user email first
bot.add_personal_info("email", "Please enter your email:")

# Define workflow
bot.add("destination", "Where would you like to go?", required=True)
bot.add("price_range", "Choose your budget:", required=False)
bot.finish("hotel_name", "price_range")  # Final output columns

# Simulate user interaction
response = bot.process("user123", "")  # Start flow!
print(response['message'])  # "Where would you like to go?"
print(response['suggestions'])  # ["Paris", "Tokyo", "New York"]

# User selects 'Paris'
response = bot.process("user123", "Paris")
print(response['message'])  # "Choose your budget:"
print(response['suggestions'])  # ["$$$", "$$"]
```

---

##  Key Features  

### 1. **Dynamic Suggestions**  
Auto-filter valid options based on prior choices:  
```python
bot.add("activity", "Choose an activity:", required=True)
# Suggests only activities available in the selected destination
```

### 2. **Personalized Data Collection**  
```python
bot.add_personal_info("phone", "Your phone number:", required=True)
```

### 3. **Session Management**  
Resume progress or reset conversations:  
```python
bot.reset_session("user123")  # Restart workflow
```

### 4. **Save Results**  
User data and chat history auto-saved to JSON:  
```json
user_data/user123.json
{
  "personal_info": {"email": "user@example.com"},
  "chat_history": [...]
}
```

---

## Detailed Usage  

### Initialize FlowBot  
```python
bot = FlowBot(
    data=df,  # Your pandas DataFrame
)
```

### Add Workflow Steps  
```python
bot.add(
    field="room_type",  # DataFrame column to filter
    prompt="Select room type:",  # User prompt
    required=True  # Force valid input
)
```

### Get Final Results  
```python
results = response['results']  # Filtered DataFrame rows as dicts
# Example: [{'hotel_name': 'Luxury Palace', 'price_range': '$$$'}]
```

---

## 🔧 Dependencies  
- Python 3.7+  
- `pandas`  

---


## 📜 License  
[MIT License](LICENSE)  

---

## Get Help  
Found a bug? Open an [issue](https://github.com/guru-dharsan-git/gurulearn/issues).  

--- 
 **Happy Building!**  
*Tag your projects with #gurulearn to share them with the community!*  

---



# GuruLearn

## QAAgent: Create intelligent QA systems with RAG

GuruLearn's QAAgent provides a simple yet powerful way to create domain-specific question-answering agents using Retrieval Augmented Generation (RAG).

## Installation

```bash
pip install gurulearn
```

## Dependencies

GuruLearn requires the following dependencies:

- langchain-ollama
- langchain-core
- FAISS
- pandas
- langchain-community

Install them using:



## Quick Start

```python
from gurulearn import QAAgent
import pandas as pd

# Load your data
df = pd.read_csv("customer_support_tickets.csv")

# Create an agent
support_agent = QAAgent(
    data=df,
    page_content_fields=["Title", "Description"],
    metadata_fields=["Category", "Priority"],
    system_prompt="You are a helpful customer support agent."
)

# Query the agent
answer = support_agent.query("How do I reset my password?")
print(answer)

# Or use interactive mode
support_agent.interactive_mode()
```

## Features

- **Simple Setup**: Create powerful RAG-based QA systems with minimal code
- **Flexible Data Support**: Works with pandas DataFrames or lists of dictionaries
- **Custom Prompting**: Define system prompts and instructions to shape agent responses
- **Vector Database Integration**: Automatically creates and manages embeddings for efficient retrieval
- **Interactive Mode**: Built-in console interface for quick experimentation

## API Reference

### QAAgent

```python
QAAgent(
    data,                    # DataFrame or list of dictionaries containing the source data
    page_content_fields,     # Field(s) to use as document content
    metadata_fields=None,    # Fields to include as metadata
    llm_model="llama3.2",    # Ollama model to use for generation
    k=5,                     # Number of documents to retrieve
    embedding_model="mxbai-embed-large",  # Ollama model for embeddings
    db_location="./langchain_db",  # Directory to store vector database
    collection_name="documents",          # Name of the collection in the vector store
    prompt_template=None,    # Custom prompt template (if None, a default will be used)
    system_prompt="You are an expert in answering questions about the provided information."
)
```

#### Methods

- **query(question)**: Query the agent with a question and get a response
- **interactive_mode()**: Start an interactive console session for querying the agent

## Examples

### Restaurant Review QA System

```python
from gurulearn import QAAgent
import pandas as pd

# Load restaurant review data
df = pd.read_csv("restaurant_reviews.csv")

# Create a restaurant review agent
restaurant_agent = QAAgent(
    data=df,
    page_content_fields=["Title", "Review"],
    metadata_fields=["Rating", "Date"],
    llm_model="llama3.2",
    k=5,
    db_location="./restaurant_db",
    collection_name="restaurant_reviews",
    system_prompt="You are an expert in answering questions about a pizza restaurant."
)

# Ask questions about the restaurant
result = restaurant_agent.query("What do customers say about the pepperoni pizza?")
print(result)
```

### HR Policy Assistant

```python
from gurulearn import QAAgent

# Create an HR policy assistant
hr_documents = [
    {"Policy": "Parental Leave", "Description": "Employees are entitled to 12 weeks of paid parental leave...", "Department": "HR", "LastUpdated": "2023-09-01"},
    {"Policy": "Remote Work", "Description": "Employees may work remotely up to 3 days per week...", "Department": "HR", "LastUpdated": "2023-10-15"},
    # More policy documents...
]

hr_agent = QAAgent(
    data=hr_documents,
    page_content_fields=["Policy", "Description"],
    metadata_fields=["Department", "LastUpdated"],
    db_location="./_db",
    collection_name="hr_policies",
    system_prompt="You are an HR assistant providing information about company policies."
)

# Query the HR assistant
hr_agent.interactive_mode()
```

## Advanced Usage

### Custom Prompt Templates

You can define custom prompt templates to control how the agent processes and responds to queries:

```python
custom_template = """
You are a technical support specialist for computer hardware.

CONTEXT:
{reviews}

USER QUESTION:
{question}

Please provide a concise answer focusing only on the information found in the context.
If the information isn't in the context, admit you don't know.
"""

support_agent = QAAgent(
    data=support_df,
    page_content_fields=["Issue", "Resolution"],
    prompt_template=custom_template
)
```

### Using with Different Models

QAAgent works with any Ollama model:

```python
# Using with different Ollama models
medical_agent = QAAgent(
    data=medical_data,
    page_content_fields="text",
    llm_model="nous-hermes2:Q5_K_M", 
    embedding_model="nomic-embed-text"
)
```

## License

MIT
