<p align="center">
  <img src="https://img.shields.io/badge/version-5.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.9+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License">
  <img src="https://img.shields.io/badge/typed-yes-brightgreen.svg" alt="Typed">
</p>

# Gurulearn

> **A unified AI/ML toolkit for deep learning, computer vision, audio processing, and conversational AI.**

Built with lazy loading for minimal import overhead (~0.001s). Production-ready with type hints.

---

## 📦 Installation

```bash
pip install gurulearn              # Core only
pip install gurulearn[vision]      # + PyTorch image classification
pip install gurulearn[audio]       # + TensorFlow audio recognition  
pip install gurulearn[agent]       # + LangChain RAG agent
pip install gurulearn[ocr]         # + CTC-based character OCR
pip install gurulearn[full]        # All features
```

---

## 🖼️ ImageClassifier

PyTorch-based image classification with 9 model architectures.

### Data Loading Options

```python
from gurulearn import ImageClassifier

clf = ImageClassifier()

# Option 1: From directory structure (data/train/{class_name}/*.jpg)
model, history = clf.train(train_dir="data/train", test_dir="data/test")

# Option 2: From CSV file
model, history = clf.train(
    csv_file="data.csv",
    img_column="image_path",
    label_column="class"
)
```

### Training Parameters

```python
model, history = clf.train(
    train_dir="data/train",
    epochs=20,
    batch_size=32,
    model_name="resnet50",     # See models below
    finetune=True,             # Finetune all layers
    learning_rate=0.001,
    use_amp=True,              # Mixed precision (GPU)
    save_path="model.pth"
)
```

### Available Models

| Model | Best For | Parameters |
|-------|----------|------------|
| `simple_cnn` | Small datasets (<1K) | 3M |
| `vgg16` | General purpose | 138M |
| `resnet50` | Large datasets | 25M |
| `mobilenet` | Mobile deployment | 3.5M |
| `inceptionv3` | Fine-grained | 23M |
| `densenet` | Feature reuse | 8M |
| `efficientnet` | Accuracy/size balance | 5M |
| `convnext` | Modern CNN | 28M |
| `vit` | Vision Transformer | 86M |

### Prediction

```python
# Load saved model
clf.load("model.pth", model_name="resnet50")

# Single image prediction
result = clf.predict("image.jpg", top_k=3)
print(result.class_name)       # "cat"
print(result.probability)      # 0.95
print(result.top_k)            # [("cat", 0.95), ("dog", 0.03), ...]

# From PIL Image
from PIL import Image
result = clf.predict(image=Image.open("image.jpg"))

# Export for production
clf.export_onnx("model.onnx")
```

---

## 🎵 AudioRecognition

TensorFlow/Keras CNN-LSTM for audio classification.

### Data Loading

```python
from gurulearn import AudioRecognition

audio = AudioRecognition(sample_rate=16000, n_mfcc=20)

# From directory structure (data/{class_name}/*.wav)
# Supports: .wav, .mp3, .flac, .ogg, .m4a
history = audio.audiotrain(
    data_path="data/audio",
    epochs=50,
    batch_size=32,
    augment=True,              # Time stretch, pitch shift, noise
    model_dir="models"
)
```

### Training Output

- `models/audio_recognition_model.keras` - Trained model
- `models/label_mapping.json` - Class labels
- `models/confusion_matrix.png` - Evaluation plot
- `models/training_history.png` - Loss/accuracy curves

### Prediction

```python
# Single file
result = audio.predict("sample.wav", model_dir="models")
print(result.label)            # "speech"
print(result.confidence)       # 0.92
print(result.all_probabilities)  # [0.92, 0.05, 0.03]

# Batch prediction
results = audio.predict_batch(
    ["file1.wav", "file2.wav"], 
    model_dir="models"
)
```

---

## 📊 MLModelAnalysis

AutoML for regression and classification with 10+ algorithms.

### Data Loading

```python
from gurulearn import MLModelAnalysis

ml = MLModelAnalysis(
    task_type="auto",              # "auto", "regression", "classification"
    auto_feature_engineering=True  # Extract date features
)

# From CSV
result = ml.train_and_evaluate(
    csv_file="data.csv",
    target_column="price",
    test_size=0.2,
    model_name=None,               # Auto-select best model
    save_path="model.joblib"
)
```

### Available Models

**Regression**: `linear_regression`, `decision_tree`, `random_forest`, `gradient_boosting`, `svm`, `knn`, `ada_boost`, `mlp`, `xgboost`*, `lightgbm`*

**Classification**: `logistic_regression`, `decision_tree`, `random_forest`, `gradient_boosting`, `svm`, `knn`, `ada_boost`, `mlp`, `xgboost`*, `lightgbm`*

*Optional dependencies

### Prediction

```python
# Load and predict
ml.load_model("model.joblib")

# From dictionary
prediction = ml.predict({"feature1": 42, "category": "A"})

# From DataFrame
predictions = ml.predict(test_df)

# Compare all models
comparison = ml.compare_models("data.csv", "target", cv=5)
```

---

## 💬 FlowBot

Guided conversation flows with real-time data filtering.

### Data Loading

```python
from gurulearn import FlowBot
import pandas as pd

# From DataFrame
bot = FlowBot(pd.read_csv("hotels.csv"), data_dir="user_sessions")

# From list of dicts
bot = FlowBot([
    {"city": "Paris", "price": "$$$", "name": "Le Grand"},
    {"city": "Tokyo", "price": "$$", "name": "Sakura Inn"}
])
```

### Building Flows

```python
# Add filter steps
bot.add("city", "Select destination:", required=True)
bot.add("price", "Choose budget:")

# Define output columns
bot.finish("name", "price")

# Validate flow
errors = bot.validate()
```

### Processing & Prediction

```python
# Process user input (maintains session state)
response = bot.process("user123", "Paris")

# Response structure
{
    "message": "Choose budget:",
    "suggestions": ["$$$", "$$"],
    "completed": False
}

# Final response
{
    "completed": True,
    "results": [{"name": "Le Grand", "price": "$$$"}],
    "message": "Found 1 matching options"
}

# Async support
response = await bot.aprocess("user123", "Paris")

# Export history
history_df = bot.export_history("user123", format="dataframe")
```

---

## 🤖 QAAgent

RAG-based question answering with LangChain + Ollama.

### Data Loading

```python
from gurulearn import QAAgent
import pandas as pd

# From DataFrame
agent = QAAgent(
    data=pd.read_csv("docs.csv"),
    page_content_fields=["title", "content"],
    metadata_fields=["category", "date"],
    llm_model="llama3.2",
    embedding_model="mxbai-embed-large",
    db_location="./vector_db"
)

# From list of dicts
agent = QAAgent(
    data=[{"title": "Policy", "content": "..."}],
    page_content_fields="content"
)

# Load existing index (no data needed)
agent = QAAgent(db_location="./existing_db")
```

### Querying

```python
# Simple query
answer = agent.query("What is the refund policy?")

# With source documents
result = agent.query("What is the refund policy?", return_sources=True)
print(result["answer"])
print(result["sources"])

# Direct similarity search (no LLM)
docs = agent.similarity_search("refund", k=5)

# Interactive mode
agent.interactive_mode()

# Add more documents
agent.add_documents(new_df, "content", ["category"])
```

---

## 🏥 CTScanProcessor

Medical image enhancement with quality metrics.

### Processing

```python
from gurulearn import CTScanProcessor

processor = CTScanProcessor(
    kernel_size=5,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
)

# Single image - supports .jpg, .png, .dcm, .nii
result = processor.process_ct_scan(
    "scan.jpg",
    output_folder="output/",
    compare=True               # Save side-by-side comparison
)

# Batch processing
results = processor.process_batch(
    input_folder="scans/",
    output_folder="processed/"
)
```

### Quality Metrics

```python
# result.metrics contains:
print(result.metrics.mse)      # Mean Squared Error
print(result.metrics.psnr)     # Peak Signal-to-Noise Ratio (dB)
print(result.metrics.snr)      # Signal-to-Noise Ratio (dB)
print(result.metrics.detail_preservation)  # Percentage
```

### Individual Operations

```python
import numpy as np

# Apply individual filters
sharpened = processor.sharpen(image)
denoised = processor.median_denoise(image)
enhanced = processor.enhance_contrast(image)
bilateral = processor.bilateral_denoise(image)

# Compare quality
metrics = processor.evaluate_quality(original, processed)
```

---

## 🔤 OCR Module

Character-level OCR with VGG-BiLSTM + CTC decoding. Auto-discovers classes from `data.yaml` — works with any character set.(YOLO DATASET FORMAT)

### Installation

```bash
pip install gurulearn[ocr]
```

### Quick Inference

```python
from gurulearn.ocr import OCRPredictor

# Load model — NO dataset needed, everything is in the .guruocr file
predictor = OCRPredictor("best_model.guruocr")

result = predictor.predict("image.jpg")
print(result.text)          # "ASDF"
print(result.confidence)    # 0.97

# Batch prediction
results = predictor.predict_batch(["img1.jpg", "img2.jpg"])

# Visualize with overlay
predictor.visualize("image.jpg", save_path="output.png")
```

### Training

```python
from gurulearn.ocr import OCRTrainer

trainer = OCRTrainer(
    data_dir="path/to/yolo_dataset",  # Must have data.yaml + train/valid/test splits
    output_dir="output/",
    img_h=48, img_w=128,               # Model input size
    hidden=256, num_layers=3,           # Architecture config
    focus_tokens=["I", "O"],            # Optional: boost learning for confusable chars
)

history = trainer.train(
    epochs=150,
    batch_size=64,
    lr=1e-4,
    patience=5,
)

# Evaluate on test set
result = trainer.evaluate()  # accuracy, CER, loss

# Save training curves
trainer.plot_results()
```

The trainer saves a `.guruocr` file — a self-contained archive with model weights + metadata (class names, image dimensions, architecture config). This means inference never needs the original dataset.

### Dataset Utilities

```python
from gurulearn.ocr import split_datasets, merge_datasets, rebalance_splits, shuffle_augment

# Split datasets by filename keywords
result = split_datasets(
    source_dirs=["dataset_v1", "dataset_v2"],
    output_root="segregated_datasets",
    keywords={"aircraft": "aircraft", "supplier": "suppliers"},
)

# Generate synthetic augmented images with double-letter support
result = shuffle_augment(
    source_dir="segregated_datasets/aircraft",
    num_output=30000,
    doubles=5200,   # ~200 per letter for CTC double-character learning
)

# Merge multiple YOLO datasets
result = merge_datasets(source_root="segregated_datasets", output_name="merged")

# Rebalance train/valid/test split ratios
result = rebalance_splits("path/to/dataset", train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1)
```

### Automated Pipeline

Run the full workflow — split → augment → train → evaluate — in one command:

```python
from gurulearn.ocr import OCRPipeline

pipeline = OCRPipeline(
    source_dirs=["dataset_v1", "dataset_v2"],
    output_root="segregated_datasets",
    dataset_name="aircraft",
    split_keywords={"aircraft": "aircraft", "supplier": "suppliers"},
    augment_count=30000,
    doubles_count=5200,
    train_epochs=150,
)

# Everything at once
result = pipeline.run_all()

# Or step by step
pipeline.step_split()
pipeline.step_augment()
pipeline.step_merge()
pipeline.step_rebalance()
pipeline.step_train()
pipeline.step_evaluate()

# Get a predictor from the trained model
predictor = pipeline.get_predictor()
print(predictor.predict("test.jpg").text)
```

---

## ⚡ Performance

- **Lazy Loading**: ~0.001s import time
- **GPU Auto-Detection**: CUDA for PyTorch/TensorFlow
- **Mixed Precision**: Automatic FP16 on compatible GPUs
- **Batch Processing**: All modules support batch inference

---

## 📄 License

MIT License - [Guru Dharsan T](https://github.com/guru-dharsan-git)
