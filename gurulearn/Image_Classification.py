import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from PIL import Image
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json

class ImageDataset(Dataset):
    def __init__(self, dataframe=None, img_dir=None, transform=None, img_column=None, label_column=None):
        self.img_paths = []
        self.labels = []
        self.transform = transform
        self.classes = []

        if dataframe is not None and img_column is not None and label_column is not None:
            self.dataframe = dataframe
            self.img_paths = self.dataframe[img_column].values
            self.classes = sorted(self.dataframe[label_column].unique())
            
            # One-hot encode labels
            label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
            self.labels = [label_to_idx[label] for label in self.dataframe[label_column].values]
        elif img_dir is not None:
            # Get classes from directory names
            self.classes = sorted([d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))])
            
            # Create label to index mapping
            label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
            
            # Load images and labels
            for label in self.classes:
                label_dir = os.path.join(img_dir, label)
                if os.path.isdir(label_dir):
                    for img_file in os.listdir(label_dir):
                        img_path = os.path.join(label_dir, img_file)
                        if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            self.img_paths.append(img_path)
                            self.labels.append(label_to_idx[label])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and the label
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, torch.tensor(self.labels[idx], dtype=torch.long)

class ImageClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = None
        self.model = None
        self.input_size = (224, 224)
        self.transform = None

    def _get_files(self, directory):
        if not os.path.exists(directory):
            return 0
        return sum([len(files) for r, d, files in os.walk(directory)])

    def _select_model(self, num_classes, dataset_size, model_name=None, finetune=False):
        if model_name == "simple_cnn" or (model_name is None and dataset_size < 1000):
            model = self._build_simple_cnn(num_classes)
        elif model_name == "vgg16" or (model_name is None and dataset_size < 5000):
            model = self._build_vgg16(num_classes, finetune)
        elif model_name == "resnet50" or (model_name is None and dataset_size >= 5000):
            model = self._build_resnet50(num_classes, finetune)
        elif model_name == "mobilenet":
            model = self._build_mobilenet(num_classes, finetune)
        elif model_name == "inceptionv3":
            model = self._build_inceptionv3(num_classes, finetune)
        elif model_name == "densenet":
            model = self._build_densenet(num_classes, finetune)
        elif model_name == "efficientnet":
            model = self._build_efficientnet(num_classes, finetune)
        else:
            raise ValueError(f"Invalid model choice: {model_name}")
        
        return model.to(self.device)

    # Simple CNN Models
    def _build_simple_cnn(self, num_classes):
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    # Pretrained Models
    def _build_vgg16(self, num_classes, finetune):
        model = models.vgg16(pretrained=True)
        if not finetune:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        return model

    def _build_resnet50(self, num_classes, finetune):
        model = models.resnet50(pretrained=True)
        if not finetune:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return model

    def _build_mobilenet(self, num_classes, finetune):
        model = models.mobilenet_v2(pretrained=True)
        if not finetune:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, num_classes)
        )
        return model

    def _build_inceptionv3(self, num_classes, finetune):
        model = models.inception_v3(pretrained=True)
        if not finetune:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        model.aux_logits = False  # Disable auxiliary output
        return model

    def _build_densenet(self, num_classes, finetune):
        model = models.densenet121(pretrained=True)
        if not finetune:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    def _build_efficientnet(self, num_classes, finetune):
        model = models.efficientnet_b0(pretrained=True)
        if not finetune:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        return model

    def train_model(self, model, train_loader, val_loader, epochs, criterion, optimizer, scheduler=None):
        best_val_acc = 0.0
        best_model_state = None
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        print(f"Training on {self.device}")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Print progress
                if (i + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                          f'Loss: {loss.item():.4f}')
            
            train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            
            # Validation phase
            val_loss, val_acc = self.evaluate_model(model, val_loader, criterion)
            
            # Learning rate scheduler
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')
        
        # Load best model weights
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            
        return model, history

    def evaluate_model(self, model, loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return running_loss / len(loader), correct / total

    def plot_accuracy(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.grid(True)
        plt.savefig('accuracy_plot.png')
        plt.show()

    def plot_loss(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.grid(True)
        plt.savefig('loss_plot.png')
        plt.show()

    def plot_confusion_matrix(self, model, loader):
        model.eval()
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.show()

    def train(self, train_dir=None, test_dir=None, csv_file=None, img_column=None, label_column=None,
              epochs=10, model_name=None, finetune=False, batch_size=32, learning_rate=0.001,
              save_path='trained_model.pth'):
        """
        Train an image classification model.
        
        Args:
            train_dir: Directory containing training images organized in class folders
            test_dir: Directory containing test images organized in class folders
            csv_file: CSV file with image paths and labels
            img_column: Column name in CSV containing image paths
            label_column: Column name in CSV containing labels
            epochs: Number of training epochs
            model_name: Model architecture to use ('simple_cnn', 'vgg16', 'resnet50', etc.)
            finetune: Whether to finetune all layers of the model
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            save_path: Path to save the trained model
            
        Returns:
            Trained model and training history
        """
        
        # Data transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load data
        if csv_file:
            df = pd.read_csv(csv_file)
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
            
            train_dataset = ImageDataset(dataframe=train_df, transform=self.transform, 
                                         img_column=img_column, label_column=label_column)
            val_dataset = ImageDataset(dataframe=val_df, transform=self.transform,
                                       img_column=img_column, label_column=label_column)
            self.classes = train_dataset.classes
        else:
            if not os.path.exists(train_dir):
                raise ValueError(f"Training directory not found: {train_dir}")
                
            train_dataset = ImageDataset(img_dir=train_dir, transform=self.transform)
            self.classes = train_dataset.classes
            
            if test_dir and os.path.exists(test_dir):
                val_dataset = ImageDataset(img_dir=test_dir, transform=self.transform)
            else:
                print("No test directory provided. Using 20% of training data for validation.")
                train_size = int(0.8 * len(train_dataset))
                val_size = len(train_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(
                    train_dataset, [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
        
        if len(train_dataset) == 0:
            raise ValueError("No training images found!")
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Classes: {self.classes}")
        
        # Initialize model
        dataset_size = len(train_dataset)
        num_classes = len(self.classes)
        self.model = self._select_model(num_classes, dataset_size, model_name, finetune)
        
        # Training setup
        if finetune:
            params_to_update = self.model.parameters()
        else:
            params_to_update = [p for p in self.model.parameters() if p.requires_grad]
            
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params_to_update, lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=1e-6)
        
        # Train model
        self.model, history = self.train_model(
            self.model, train_loader, val_loader, epochs, criterion, optimizer, scheduler
        )
        
        # Save model and class information
        self.save_model(save_path)
        
        # Plots
        self.plot_accuracy(history)
        self.plot_loss(history)
        self.plot_confusion_matrix(self.model, val_loader)
        
        return self.model, history
    
    def save_model(self, save_path='trained_model.pth'):
        """
        Save the model and class information.
        
        Args:
            save_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
            
        # Save model state dict
        model_dir = os.path.dirname(save_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        torch.save(self.model.state_dict(), save_path)
        
        # Save class information
        model_info = {
            'classes': self.classes,
            'input_size': self.input_size
        }
        
        info_path = os.path.splitext(save_path)[0] + '_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f)
            
        print(f"Model saved to {save_path}")
        print(f"Model info saved to {info_path}")
    
    def load(self, model_path, model_name=None, num_classes=None):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            model_name: Name of the model architecture
            num_classes: Number of classes (required if model_info file is not available)
            
        Returns:
            Loaded model
        """
        # Try to load model info
        info_path = os.path.splitext(model_path)[0] + '_info.json'
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            self.classes = model_info['classes']
            self.input_size = tuple(model_info['input_size'])
            num_classes = len(self.classes)
        elif num_classes is None:
            raise ValueError("Number of classes must be provided if model_info file is not available")
        
        # Initialize model architecture
        if model_name:
            # Use a fixed dataset size (doesn't matter for loading)
            self.model = self._select_model(num_classes, 1000, model_name)
        else:
            # Try to infer model architecture from filename
            filename = os.path.basename(model_path).lower()
            if 'vgg' in filename:
                self.model = self._select_model(num_classes, 1000, 'vgg16')
            elif 'resnet' in filename:
                self.model = self._select_model(num_classes, 1000, 'resnet50')
            elif 'mobile' in filename:
                self.model = self._select_model(num_classes, 1000, 'mobilenet')
            elif 'inception' in filename:
                self.model = self._select_model(num_classes, 1000, 'inceptionv3')
            elif 'dense' in filename:
                self.model = self._select_model(num_classes, 1000, 'densenet')
            elif 'efficient' in filename:
                self.model = self._select_model(num_classes, 1000, 'efficientnet')
            else:
                self.model = self._select_model(num_classes, 1000, 'simple_cnn')
        
        # Load model state dict
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Set up transform
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded from {model_path}")
        if self.classes:
            print(f"Classes: {self.classes}")
            
        return self.model
    
    def predict(self, image_path=None, image=None, top_k=1):
        """
        Predict class for an image.
        
        Args:
            image_path: Path to the image file
            image: PIL Image object (alternative to image_path)
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_name, probability) tuples for top k predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
            
        if image_path is None and image is None:
            raise ValueError("Either image_path or image must be provided")
            
        if image_path:
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                raise ValueError(f"Error loading image {image_path}: {e}")
                
        if self.transform:
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        else:
            # Default transform if none is defined
            transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            
        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.classes)))
        
        # Map indices to class names
        predictions = []
        for i, (prob, idx) in enumerate(zip(top_probs.cpu().numpy(), top_indices.cpu().numpy())):
            if self.classes:
                class_name = self.classes[idx]
            else:
                class_name = f"Class {idx}"
            predictions.append((class_name, float(prob)))
            
        return predictions


def main():
    """Example usage of the ImageClassifier class"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or test an image classification model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'], 
                        help='Mode: train or predict')
    parser.add_argument('--train_dir', type=str, help='Directory with training images organized in class folders')
    parser.add_argument('--test_dir', type=str, help='Directory with test images organized in class folders')
    parser.add_argument('--csv_file', type=str, help='CSV file with image paths and labels')
    parser.add_argument('--img_column', type=str, help='Column name in CSV for image paths')
    parser.add_argument('--label_column', type=str, help='Column name in CSV for labels')
    parser.add_argument('--model_name', type=str, default='resnet50', 
                        choices=['simple_cnn', 'vgg16', 'resnet50', 'mobilenet', 'inceptionv3', 'densenet', 'efficientnet'],
                        help='Model architecture to use')
    parser.add_argument('--finetune', action='store_true', help='Finetune all layers of the model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--model_path', type=str, default='trained_model.pth', 
                        help='Path to save or load the model')
    parser.add_argument('--image_path', type=str, help='Path to image for prediction')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top predictions to return')
    
    args = parser.parse_args()
    classifier = ImageClassifier()
    
    if args.mode == 'train':
        if args.train_dir or args.csv_file:
            classifier.train(
                train_dir=args.train_dir,
                test_dir=args.test_dir,
                csv_file=args.csv_file,
                img_column=args.img_column,
                label_column=args.label_column,
                epochs=args.epochs,
                model_name=args.model_name,
                finetune=args.finetune,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                save_path=args.model_path
            )
        else:
            print("Error: Either train_dir or csv_file must be provided for training")
            
    elif args.mode == 'predict':
        if args.model_path and args.image_path:
            classifier.load(args.model_path, args.model_name)
            predictions = classifier.predict(args.image_path, top_k=args.top_k)
            
            print(f"\nPredictions for {args.image_path}:")
            for i, (class_name, prob) in enumerate(predictions):
                print(f"{i+1}. {class_name}: {prob:.4f}")
        else:
            print("Error: model_path and image_path must be provided for prediction")
            
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
