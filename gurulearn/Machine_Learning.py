from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score, 
    mean_squared_error, 
    mean_absolute_error
)
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
import numpy as np
import pandas as pd
import pickle

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
    print("XGBoost is not installed. Install with `pip install xgboost` to use XGBRegressor.")

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


class MLModelAnalysis:
    def __init__(self, model_type='auto', auto_feature_engineering=True):
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.feature_names = []
        self.target_name = ''
        self.auto_feature_engineering = auto_feature_engineering
        self.model_comparison = pd.DataFrame()
        self._initialize_models()

    def _initialize_models(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'decision_tree': DecisionTreeRegressor(),
            'random_forest': RandomForestRegressor(),
            'svm': SVR(),
            'gradient_boosting': GradientBoostingRegressor(),
            'knn': KNeighborsRegressor(),
            'ada_boost': AdaBoostRegressor(),
            'mlp': MLPRegressor(max_iter=1000)
        }
        if XGBRegressor is not None:
            self.models['xgboost'] = XGBRegressor()

    def _auto_detect_types(self, df):
        date_cols = []
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                date_cols.append(col)
            except (TypeError, ValueError):
                continue
        return date_cols

    def _create_preprocessor(self, X):
        numeric_features = X.select_dtypes(include=np.number).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)],
            remainder='passthrough')

        return preprocessor

    def preprocess_data(self, csv_file, target_column):
        df = pd.read_csv(csv_file)
        self.target_name = target_column
        
        if self.auto_feature_engineering:
            date_cols = self._auto_detect_types(df.drop(columns=[target_column]))
            for col in date_cols:
                dt_col = pd.to_datetime(df[col])
                df[f'{col}_year'] = dt_col.dt.year
                df[f'{col}_month'] = dt_col.dt.month
                df[f'{col}_day'] = dt_col.dt.day
                df[f'{col}_dayofweek'] = dt_col.dt.dayofweek
            df.drop(columns=date_cols, inplace=True)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        self.preprocessor = self._create_preprocessor(X)
        X_processed = self.preprocessor.fit_transform(X)
        self.feature_names = self._get_feature_names()
        
        return X_processed, y

    def _get_feature_names(self):
        """Get feature names after preprocessing"""
        col_names = []
        for name, trans, features in self.preprocessor.transformers_:
            if trans == 'passthrough':
                continue
            if hasattr(trans, 'get_feature_names_out'):
                names = trans.get_feature_names_out(features)
                col_names.extend(names)
            else:
                col_names.extend(features)
        return col_names

    def _evaluate_model(self, model, X, y, cv=5):
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        return np.mean(scores)

    def auto_select_model(self, X, y):
        best_score = -np.inf
        best_model = None
        
        for name, model in self.models.items():
            try:
                pipeline = Pipeline(steps=[
                    ('preprocessor', clone(self.preprocessor)),
                    ('model', clone(model))])
                
                score = self._evaluate_model(pipeline, X, y)
                if score > best_score:
                    best_score = score
                    best_model = name
            except Exception as e:
                print(f"Error with {name}: {str(e)}")
                continue
                
        print(f"Best model: {best_model} with R²: {best_score:.3f}")
        self.model_type = best_model
        return best_model

    def train_and_evaluate(self, csv_file, target_column, test_size=0.2, 
                         random_state=42, save_path=None):
        X, y = self.preprocess_data(csv_file, target_column)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        
        if self.model_type == 'auto':
            best_model_name = self.auto_select_model(X_train, y_train)
            self.model = self.models[best_model_name]
        else:
            self.model = self.models[self.model_type]
        
        full_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', self.model)])
        
        full_pipeline.fit(X_train, y_train)
        
        # Evaluation
        train_pred = full_pipeline.predict(X_train)
        test_pred = full_pipeline.predict(X_test)
        
        metrics = {
            'Train R²': r2_score(y_train, train_pred),
            'Test R²': r2_score(y_test, test_pred),
            'Train MSE': mean_squared_error(y_train, train_pred),
            'Test MSE': mean_squared_error(y_test, test_pred),
            'Train MAE': mean_absolute_error(y_train, train_pred),
            'Test MAE': mean_absolute_error(y_test, test_pred)
        }
        
        print(f"\n{' Metric ':~^40}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        
        if save_path:
            self._save_model(full_pipeline, save_path)
            
        return full_pipeline, metrics

    def _save_model(self, pipeline, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'pipeline': pipeline,
                'feature_names': self.feature_names,
                'target_name': self.target_name,
            }, f)
        print(f"Model saved to {path}")

    def load_model(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['pipeline']
        self.feature_names = data['feature_names']
        self.target_name = data['target_name']
        self.preprocessor = self.model.named_steps['preprocessor']
        return self

    def predict(self, input_data):
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
            
        return self.model.predict(df)

    def plot_feature_importance(self):
        if not hasattr(self.model.named_steps['model'], 'feature_importances_'):
            print("Feature importance not available for this model type")
            return

        importances = self.model.named_steps['model'].feature_importances_
        features = self.feature_names
        
        if go is not None:
            fig = go.Figure(go.Bar(
                x=importances,
                y=features,
                orientation='h'))
            
            fig.update_layout(
                title='Feature Importance',
                xaxis_title='Importance',
                yaxis_title='Features')
            fig.show()
        else:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.barh(features, importances)
            plt.title("Feature Importance")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MLModelAnalysis(auto_feature_engineering=True)
    
    # Train and evaluate model
    pipeline, metrics = analyzer.train_and_evaluate(
        csv_file='your_data.csv',
        target_column='target',
        save_path='best_model.pkl'
    )
    
    # Make prediction
    sample_input = {
        'feature1': 25,
        'feature2': 'category_value',
        'date_feature': '2023-01-01'
    }
    prediction = analyzer.predict(sample_input)
    print(f"\nPrediction: {prediction[0]:.2f}")
    
    # Plot feature importance
    analyzer.plot_feature_importance()