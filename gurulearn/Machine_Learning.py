"""
Machine Learning - Automated ML model training and evaluation.

Provides automated model selection, preprocessing, and evaluation for regression
and classification tasks.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Regression models
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor,
    AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# Optional imports
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


TaskType = Literal["regression", "classification", "auto"]


@dataclass
class ModelMetrics:
    """Metrics from model evaluation."""
    # Regression metrics
    train_r2: float | None = None
    test_r2: float | None = None
    train_mse: float | None = None
    test_mse: float | None = None
    train_mae: float | None = None
    test_mae: float | None = None
    
    # Classification metrics
    train_accuracy: float | None = None
    test_accuracy: float | None = None
    train_f1: float | None = None
    test_f1: float | None = None
    classification_report: str | None = None
    
    def to_dict(self) -> dict[str, float | str | None]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ModelResult:
    """Result from model training."""
    pipeline: Pipeline
    metrics: ModelMetrics
    model_name: str
    feature_names: list[str]
    target_name: str
    task_type: str


class MLModelAnalysis:
    """
    Automated Machine Learning model analysis and training.
    
    Supports both regression and classification tasks with automatic
    model selection, preprocessing, and evaluation.
    
    Args:
        task_type: Type of task ('regression', 'classification', or 'auto')
        auto_feature_engineering: Whether to automatically engineer date features
        
    Example:
        >>> analyzer = MLModelAnalysis(task_type="auto")
        >>> result = analyzer.train_and_evaluate(
        ...     csv_file="data.csv",
        ...     target_column="price",
        ...     save_path="model.joblib"
        ... )
        >>> prediction = analyzer.predict({"feature1": 25, "feature2": "category_a"})
    """

    def __init__(
        self, 
        task_type: TaskType = "auto",
        auto_feature_engineering: bool = True
    ):
        self.task_type = task_type
        self.auto_feature_engineering = auto_feature_engineering
        self.model: Pipeline | None = None
        self.preprocessor: ColumnTransformer | None = None
        self.feature_names: list[str] = []
        self.target_name: str = ""
        self._label_encoder: LabelEncoder | None = None
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize available models for both tasks."""
        self.regression_models = {
            "linear_regression": LinearRegression(),
            "decision_tree": DecisionTreeRegressor(random_state=42),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(random_state=42),
            "svm": SVR(),
            "knn": KNeighborsRegressor(),
            "ada_boost": AdaBoostRegressor(random_state=42),
            "mlp": MLPRegressor(max_iter=500, random_state=42),
        }
        
        self.classification_models = {
            "logistic_regression": LogisticRegression(max_iter=500, random_state=42),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "svm": SVC(probability=True, random_state=42),
            "knn": KNeighborsClassifier(),
            "ada_boost": AdaBoostClassifier(random_state=42),
            "mlp": MLPClassifier(max_iter=500, random_state=42),
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            self.regression_models["xgboost"] = XGBRegressor(random_state=42, n_jobs=-1)
            self.classification_models["xgboost"] = XGBClassifier(random_state=42, n_jobs=-1)
        
        # Add LightGBM if available
        if HAS_LIGHTGBM:
            self.regression_models["lightgbm"] = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
            self.classification_models["lightgbm"] = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)

    def _detect_task_type(self, y: pd.Series) -> str:
        """Auto-detect whether task is regression or classification."""
        if y.dtype == "object" or y.dtype.name == "category":
            return "classification"
        
        unique_ratio = len(y.unique()) / len(y)
        if unique_ratio < 0.05 or len(y.unique()) <= 10:
            return "classification"
        
        return "regression"

    def _engineer_date_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Extract features from date columns."""
        df = df.copy()
        date_cols = []
        
        for col in df.columns:
            if col == target_column:
                continue
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > len(df) * 0.5:  # At least 50% valid dates
                    date_cols.append(col)
            except (TypeError, ValueError):
                continue
        
        for col in date_cols:
            dt_col = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"] = dt_col.dt.year
            df[f"{col}_month"] = dt_col.dt.month
            df[f"{col}_day"] = dt_col.dt.day
            df[f"{col}_dayofweek"] = dt_col.dt.dayofweek
            df[f"{col}_quarter"] = dt_col.dt.quarter
        
        df.drop(columns=date_cols, inplace=True)
        return df

    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessing pipeline for features."""
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
        
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ],
            remainder="passthrough"
        )

    def _get_feature_names(self, preprocessor: ColumnTransformer, X: pd.DataFrame) -> list[str]:
        """Extract feature names after preprocessing."""
        feature_names = []
        
        for name, transformer, features in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                names = transformer.get_feature_names_out(features)
                feature_names.extend(names)
            elif isinstance(transformer, Pipeline):
                last_step = transformer.steps[-1][1]
                if hasattr(last_step, "get_feature_names_out"):
                    names = last_step.get_feature_names_out(features)
                    feature_names.extend(names)
                else:
                    feature_names.extend(features)
            else:
                feature_names.extend(features)
        
        return feature_names

    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray, task_type: str, cv: int = 5) -> float:
        """Evaluate model using cross-validation."""
        scoring = "r2" if task_type == "regression" else "accuracy"
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return float(np.mean(scores))
        except Exception:
            return -np.inf

    def _auto_select_model(self, X: np.ndarray, y: np.ndarray, task_type: str) -> str:
        """Automatically select the best model."""
        models = self.regression_models if task_type == "regression" else self.classification_models
        best_score = -np.inf
        best_model_name = list(models.keys())[0]
        
        print("Evaluating models...")
        for name, model in models.items():
            try:
                score = self._evaluate_model(clone(model), X, y, task_type)
                print(f"  {name}: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_model_name = name
            except Exception as e:
                print(f"  {name}: Error - {e}")
        
        print(f"\nBest model: {best_model_name} (score: {best_score:.4f})")
        return best_model_name

    def train_and_evaluate(
        self,
        csv_file: str | Path,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        save_path: str | Path | None = None,
        model_name: str | None = None
    ) -> ModelResult:
        """
        Train and evaluate a model on the provided data.
        
        Args:
            csv_file: Path to the CSV file
            target_column: Name of the target column
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            save_path: Path to save the trained model (optional)
            model_name: Specific model to use (auto-selects if None)
            
        Returns:
            ModelResult with the trained pipeline and metrics
        """
        # Load and preprocess data
        df = pd.read_csv(csv_file)
        self.target_name = target_column
        
        if self.auto_feature_engineering:
            df = self._engineer_date_features(df, target_column)
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Detect task type
        if self.task_type == "auto":
            detected_type = self._detect_task_type(y)
            print(f"Auto-detected task type: {detected_type}")
        else:
            detected_type = self.task_type
        
        # Encode target for classification
        if detected_type == "classification":
            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y)
        
        # Create preprocessor
        self.preprocessor = self._create_preprocessor(X)
        X_processed = self.preprocessor.fit_transform(X)
        self.feature_names = self._get_feature_names(self.preprocessor, X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state
        )
        
        # Select model
        models = self.regression_models if detected_type == "regression" else self.classification_models
        
        if model_name is None:
            model_name = self._auto_select_model(X_train, y_train, detected_type)
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
        
        selected_model = clone(models[model_name])
        
        # Create full pipeline
        self.model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("model", selected_model)
        ])
        
        # Fit on original data (pipeline handles preprocessing)
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.model.fit(X_train_orig, y_train_orig)
        
        # Evaluate
        train_pred = self.model.predict(X_train_orig)
        test_pred = self.model.predict(X_test_orig)
        
        if detected_type == "regression":
            metrics = ModelMetrics(
                train_r2=r2_score(y_train_orig, train_pred),
                test_r2=r2_score(y_test_orig, test_pred),
                train_mse=mean_squared_error(y_train_orig, train_pred),
                test_mse=mean_squared_error(y_test_orig, test_pred),
                train_mae=mean_absolute_error(y_train_orig, train_pred),
                test_mae=mean_absolute_error(y_test_orig, test_pred)
            )
        else:
            metrics = ModelMetrics(
                train_accuracy=accuracy_score(y_train_orig, train_pred),
                test_accuracy=accuracy_score(y_test_orig, test_pred),
                train_f1=f1_score(y_train_orig, train_pred, average="weighted"),
                test_f1=f1_score(y_test_orig, test_pred, average="weighted"),
                classification_report=classification_report(y_test_orig, test_pred)
            )
        
        # Print metrics
        print(f"\n{'=' * 40}")
        print("Model Evaluation Metrics")
        print("=" * 40)
        for name, value in metrics.to_dict().items():
            if isinstance(value, float):
                print(f"{name}: {value:.4f}")
            elif isinstance(value, str):
                print(f"\n{name}:\n{value}")
        
        # Save model
        if save_path:
            self._save_model(save_path, detected_type)
        
        return ModelResult(
            pipeline=self.model,
            metrics=metrics,
            model_name=model_name,
            feature_names=self.feature_names,
            target_name=self.target_name,
            task_type=detected_type
        )

    def _save_model(self, path: str | Path, task_type: str) -> None:
        """Save the trained model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "pipeline": self.model,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "task_type": task_type,
            "label_encoder": self._label_encoder
        }
        
        joblib.dump(data, path)
        print(f"\nModel saved to {path}")

    def load_model(self, path: str | Path) -> "MLModelAnalysis":
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Self for method chaining
        """
        data = joblib.load(path)
        self.model = data["pipeline"]
        self.feature_names = data["feature_names"]
        self.target_name = data["target_name"]
        self.task_type = data["task_type"]
        self._label_encoder = data.get("label_encoder")
        self.preprocessor = self.model.named_steps["preprocessor"]
        
        print(f"Model loaded from {path}")
        return self

    def predict(self, input_data: dict[str, Any] | pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            input_data: Dictionary or DataFrame of features
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("No model loaded. Call train_and_evaluate() or load_model() first.")
        
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        predictions = self.model.predict(df)
        
        # Decode labels for classification
        if self._label_encoder is not None:
            predictions = self._label_encoder.inverse_transform(predictions)
        
        return predictions

    def plot_feature_importance(self, top_n: int = 20) -> None:
        """
        Plot feature importance if available.
        
        Args:
            top_n: Number of top features to display
        """
        if self.model is None:
            raise ValueError("No model loaded.")
        
        model = self.model.named_steps["model"]
        
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
            if len(importances) != len(self.feature_names):
                print("Feature importance not available for this model.")
                return
        else:
            print("Feature importance not available for this model type.")
            return
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.barh(
            [self.feature_names[i] if i < len(self.feature_names) else f"Feature {i}" for i in indices[::-1]],
            importances[indices[::-1]]
        )
        plt.xlabel("Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        plt.show()

    def compare_models(
        self, 
        csv_file: str | Path, 
        target_column: str, 
        cv: int = 5
    ) -> pd.DataFrame:
        """
        Compare all available models using cross-validation.
        
        Args:
            csv_file: Path to the CSV file
            target_column: Name of the target column
            cv: Number of cross-validation folds
            
        Returns:
            DataFrame with model comparison results
        """
        df = pd.read_csv(csv_file)
        
        if self.auto_feature_engineering:
            df = self._engineer_date_features(df, target_column)
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Detect task type
        if self.task_type == "auto":
            detected_type = self._detect_task_type(y)
        else:
            detected_type = self.task_type
        
        if detected_type == "classification":
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        preprocessor = self._create_preprocessor(X)
        X_processed = preprocessor.fit_transform(X)
        
        models = self.regression_models if detected_type == "regression" else self.classification_models
        
        results = []
        for name, model in models.items():
            try:
                score = self._evaluate_model(clone(model), X_processed, y, detected_type, cv)
                results.append({"Model": name, "Score": score})
            except Exception as e:
                results.append({"Model": name, "Score": None, "Error": str(e)})
        
        comparison_df = pd.DataFrame(results).sort_values("Score", ascending=False)
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df