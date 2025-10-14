# from __future__ import annotations

# from pathlib import Path
# from typing import Dict, Any, List

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score


# class DiabetesModel:
#     """Diabetes prediction model that loads a CSV from data/.

#     Expected CSV columns:
#     Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
#     """

#     def __init__(self, data_dir: Path):
#         self.data_dir = data_dir
#         self.feature_names: List[str] = [
#             "Pregnancies",
#             "Glucose",
#             "BloodPressure",
#             "SkinThickness",
#             "Insulin",
#             "BMI",
#             "DiabetesPedigreeFunction",
#             "Age",
#         ]
#         self.scaler = StandardScaler()
#         self.model = RandomForestClassifier(n_estimators=200, random_state=42)
#         self._train()

#     def _load_csv(self) -> pd.DataFrame:
       
#         csv_path = self.data_dir / "pima_diabetes.csv"
#         if not csv_path.exists():
#             raise FileNotFoundError(
#                 f"Missing {csv_path}. Please place your diabetes CSV as 'pima_diabetes.csv' in {self.data_dir}"
#             )
#         df = pd.read_csv(csv_path)
#         return df

#     def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
#         df = df.copy()
       
#         zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
#         for c in zero_cols:
#             if c in df.columns:
#                 df[c] = df[c].replace(0, np.nan)
#                 df[c] = df[c].fillna(df[c].median())
#         return df

#     def _train(self) -> None:
#         df = self._load_csv()
#         df = self._preprocess(df)
#         X = df[self.feature_names]
#         y = df["Outcome"].astype(int)
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=y
#         )

#         self.model.fit(X_train, y_train)
#         acc = accuracy_score(y_test, self.model.predict(X_test))
#         self.accuracy = float(acc)

#     def default_input_example(self) -> Dict[str, float]:
#         return {
#             "Pregnancies": 0,
#             "Glucose": 120,
#             "BloodPressure": 70,
#             "SkinThickness": 20,
#             "Insulin": 85,
#             "BMI": 25.0,
#             "DiabetesPedigreeFunction": 0.5,
#             "Age": 21,
#         }

#     def predict(self, user_values: Dict[str, Any]) -> Dict[str, Any]:
#         row = {k: float(user_values.get(k, 0.0)) for k in self.feature_names}
#         X = pd.DataFrame([row])[self.feature_names]
#         pred = int(self.model.predict(X)[0])
#         proba = self.model.predict_proba(X)[0].tolist()
#         return {
#             "prediction": pred,
#             "probabilities": {"no_diabetes": proba[0], "diabetes": proba[1]},
#              "model_accuracy": self.accuracy,
#         }

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class DiabetesModel:
    """
    Diabetes Detection Model for Flask Application
    Compatible with various dataset formats and handles missing columns gracefully
    """
    
    def __init__(self, data_dir=None):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.data_dir = data_dir
        
        # Expected feature names in order of importance
        self.expected_features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
            'BMI', 'DiabetesPedigreeFunction', 'Age', 'PhysicalActivity', 
            'Smoking', 'Alcohol', 'MotherDiabetes', 'FatherDiabetes', 
            'SiblingsDiabetes', 'GrandparentsDiabetes'
        ]
        
        # Core features that are most important (from original Pima dataset)
        self.core_features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        self._load_or_create_data()
        self._train()
        
    def _load_or_create_data(self):
        """Load data from file or create sample data"""
        csv_path = None
        
        # Try to find CSV file in data directory
        if self.data_dir and os.path.exists(self.data_dir):
            for file in os.listdir(self.data_dir):
                if file.endswith('.csv'):
                    csv_path = os.path.join(self.data_dir, file)
                    break
        
        if csv_path and os.path.exists(csv_path):
            try:
                print(f"Loading data from: {csv_path}")
                self.data = pd.read_csv(csv_path)
                print(f"Data loaded successfully. Shape: {self.data.shape}")
                print(f"Columns found: {list(self.data.columns)}")
                
                # Check if this looks like a diabetes dataset
                if 'Outcome' not in self.data.columns and 'Diabetes' not in self.data.columns:
                    # Try to find a target column
                    possible_targets = ['target', 'label', 'class', 'result', 'prediction']
                    target_col = None
                    for col in possible_targets:
                        if col in self.data.columns:
                            target_col = col
                            break
                    
                    if target_col:
                        self.data['Outcome'] = self.data[target_col]
                    else:
                        print("Warning: No target column found. Creating sample data.")
                        self._create_sample_data()
                        return
                        
            except Exception as e:
                print(f"Error loading CSV file: {e}")
                print("Creating sample data instead...")
                self._create_sample_data()
        else:
            print("No CSV file found in data directory. Creating sample data...")
            self._create_sample_data()
        
        # Add missing columns with appropriate defaults
        self._add_missing_features()
        
    def _create_sample_data(self):
        """Create sample Pima Indian Diabetes Dataset"""
        pima_data = """Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
5,116,74,0,0,25.6,0.201,30,0
3,78,50,32,88,31.0,0.248,26,1
10,115,0,0,0,35.3,0.134,29,0
2,197,70,45,543,30.5,0.158,53,1
8,125,96,0,0,0.0,0.232,54,1
4,110,92,0,0,37.6,0.191,30,0
10,168,74,0,0,38.0,0.537,34,1
10,139,80,0,0,27.1,1.441,57,0
1,189,60,23,846,30.1,0.398,59,1
5,166,72,19,175,25.8,0.587,51,1
7,100,0,0,0,30.0,0.484,32,1
0,118,84,47,230,45.8,0.551,31,1
7,107,74,0,0,29.6,0.254,31,1
1,103,30,38,83,43.3,0.183,33,0
1,115,70,30,96,34.6,0.529,32,1
3,126,88,41,235,39.3,0.704,27,0
8,99,84,0,0,35.4,0.388,50,0
7,196,90,0,0,39.8,0.451,41,1
9,119,80,35,0,29.0,0.263,29,1
11,143,94,33,146,36.6,0.254,51,1
10,125,70,26,115,31.1,0.205,41,1
7,147,76,0,0,39.4,0.257,43,1
1,97,66,15,140,23.2,0.487,22,0
13,145,82,19,110,22.2,0.245,57,0
5,117,92,0,0,34.1,0.337,38,0
2,108,64,0,0,30.8,0.158,21,0
1,181,78,42,293,40.0,1.258,22,1
0,165,76,43,255,47.9,0.259,26,0
8,197,74,0,0,25.9,1.191,39,1
9,184,85,15,0,30.0,1.213,49,1
11,155,76,28,150,33.3,1.353,51,1
3,113,44,13,0,22.4,0.14,22,0
2,74,0,0,0,0.0,0.102,22,0
7,83,78,26,71,29.3,0.767,36,0
0,101,65,28,0,24.6,0.237,22,0
5,137,108,0,0,48.8,0.227,37,1
2,110,74,29,125,32.4,0.698,27,0
13,106,72,54,0,36.6,0.178,45,0
2,100,68,25,71,38.5,0.324,26,0
15,136,70,32,110,37.1,0.153,43,1
1,107,68,19,0,26.5,0.165,24,0
1,80,55,0,0,19.1,0.258,21,0
4,123,80,15,176,32.0,0.443,34,0
7,81,78,40,48,46.7,0.261,42,0
4,134,72,0,0,23.8,0.277,60,1"""
        
        from io import StringIO
        self.data = pd.read_csv(StringIO(pima_data))
        print("Sample Pima Indian Diabetes Dataset created")
        print(f"Dataset shape: {self.data.shape}")
        
    def _add_missing_features(self):
        """Add missing features with appropriate default values"""
        np.random.seed(42)  # For reproducible results
        
        # Add lifestyle and family history features if they don't exist
        if 'PhysicalActivity' not in self.data.columns:
            # Physical activity inversely correlated with BMI
            self.data['PhysicalActivity'] = np.where(
                self.data.get('BMI', 25) > 30, 
                np.random.choice([0, 1], len(self.data), p=[0.7, 0.3]),
                np.random.choice([0, 1], len(self.data), p=[0.3, 0.7])
            )
        
        if 'Smoking' not in self.data.columns:
            # Smoking more common in older individuals
            self.data['Smoking'] = np.where(
                self.data.get('Age', 30) > 40,
                np.random.choice([0, 1], len(self.data), p=[0.7, 0.3]),
                np.random.choice([0, 1], len(self.data), p=[0.8, 0.2])
            )
        
        if 'Alcohol' not in self.data.columns:
            self.data['Alcohol'] = np.random.choice([0, 1], len(self.data), p=[0.7, 0.3])
        
        # Add family history features based on DiabetesPedigreeFunction if available
        pedigree = self.data.get('DiabetesPedigreeFunction', 0.5)
        
        if 'MotherDiabetes' not in self.data.columns:
            self.data['MotherDiabetes'] = np.where(pedigree > 0.5, 1, 0)
        
        if 'FatherDiabetes' not in self.data.columns:
            self.data['FatherDiabetes'] = np.where(pedigree > 0.7, 1, 0)
            
        if 'SiblingsDiabetes' not in self.data.columns:
            self.data['SiblingsDiabetes'] = np.where(pedigree > 0.3, 1, 0)
            
        if 'GrandparentsDiabetes' not in self.data.columns:
            self.data['GrandparentsDiabetes'] = np.where(pedigree > 0.4, 1, 0)
        
        # Ensure all core features exist with default values
        defaults = {
            'Pregnancies': 0,
            'Glucose': 100,
            'BloodPressure': 80,
            'SkinThickness': 20,
            'Insulin': 100,
            'BMI': 25.0,
            'DiabetesPedigreeFunction': 0.5,
            'Age': 30
        }
        
        for feature, default_value in defaults.items():
            if feature not in self.data.columns:
                self.data[feature] = default_value
                
    def _preprocess_data(self):
        """Preprocess the data for training"""
        # Handle zero values in medical measurements
        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        for col in cols_with_zeros:
            if col in self.data.columns:
                self.data[col] = self.data[col].replace(0, np.nan)
                self.data[col].fillna(self.data[col].median(), inplace=True)
        
        # Determine which features are actually available
        available_features = []
        for feature in self.expected_features:
            if feature in self.data.columns:
                available_features.append(feature)
            elif feature in self.core_features:
                print(f"Warning: Core feature '{feature}' not found in dataset")
        
        self.feature_names = available_features
        print(f"Using {len(self.feature_names)} features for training: {self.feature_names}")
        
        # Prepare training data
        X = self.data[self.feature_names]
        
        # Handle target variable
        if 'Outcome' in self.data.columns:
            y = self.data['Outcome']
        elif 'Diabetes' in self.data.columns:
            y = self.data['Diabetes']
        else:
            raise ValueError("No target variable found. Please ensure your dataset has an 'Outcome' or 'Diabetes' column.")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the data
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
    def _train(self):
        """Train multiple machine learning models"""
        print("Preprocessing data...")
        self._preprocess_data()
        
        print("Training multiple ML models...")
        
        models_to_train = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            
            try:
                if name == 'Random Forest':
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                else:
                    model.fit(self.X_train_scaled, self.y_train)
                    y_pred = model.predict(self.X_test_scaled)
                    y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
                accuracy = accuracy_score(self.y_test, y_pred)
                auc_score = roc_auc_score(self.y_test, y_pred_proba)
                
                self.models[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  {name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if self.models:
            # Select best model
            self.best_model_name = max(self.models.keys(), 
                                     key=lambda x: self.models[x]['accuracy'])
            self.best_model = self.models[self.best_model_name]['model']
            print(f"Best model: {self.best_model_name}")
        else:
            raise ValueError("No models were successfully trained")
            
    def predict(self, input_data):
        """Predict diabetes risk for input data"""
        if self.best_model is None:
            raise ValueError("Model not trained yet")
        
        # Convert input to DataFrame if it's a dict
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in input_df.columns:
                # Set default values for missing features
                defaults = {
                    'Pregnancies': 0, 'Glucose': 100, 'BloodPressure': 80,
                    'SkinThickness': 20, 'Insulin': 100, 'BMI': 25.0,
                    'DiabetesPedigreeFunction': 0.5, 'Age': 30,
                    'PhysicalActivity': 1, 'Smoking': 0, 'Alcohol': 0,
                    'MotherDiabetes': 0, 'FatherDiabetes': 0,
                    'SiblingsDiabetes': 0, 'GrandparentsDiabetes': 0
                }
                input_df[feature] = defaults.get(feature, 0)
        
        # Select only the features used in training
        input_df = input_df[self.feature_names]
        
        # Make prediction
        if self.best_model_name == 'Random Forest':
            prediction = self.best_model.predict(input_df)[0]
            probability = self.best_model.predict_proba(input_df)[0]
        else:
            input_scaled = self.scaler.transform(input_df)
            prediction = self.best_model.predict(input_scaled)[0]
            probability = self.best_model.predict_proba(input_scaled)[0]
        
        # Get model performance metrics
        model_info = self.models[self.best_model_name]
        
        # Create probabilities object for template compatibility
        probabilities = {
            'no_diabetes': float(probability[0]),
            'diabetes': float(probability[1])
        }
        
        return {
            'prediction': int(prediction),
            'probability_no_diabetes': float(probability[0]),
            'probability_diabetes': float(probability[1]),
            'probabilities': probabilities,  # Nested object for templates
            'model_used': self.best_model_name,
            'confidence': float(probability[prediction]),
            'model_accuracy': float(model_info['accuracy']),
            'model_auc_score': float(model_info['auc_score']),
            'features_used': self.feature_names,
            'feature_count': len(self.feature_names)
        }
    
    def get_feature_importance(self):
        """Get feature importance from the best model"""
        if self.best_model_name == 'Random Forest':
            importance = dict(zip(self.feature_names, 
                                self.best_model.feature_importances_))
            return sorted(importance.items(), key=lambda x: x[1], reverse=True)
        else:
            return None
    
    def save_model(self, filepath):
        """Save the trained model to a file"""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'models': self.models
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from a file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.models = model_data['models']
        print(f"Model loaded from {filepath}")
    
    def default_input_example(self):
        """Provide default input example for Flask app"""
        return {
            'Gender': 'Male',
            'Pregnancies': 0,
            'Glucose': 120,
            'BloodPressure': 80,
            'SkinThickness': 25,
            'Insulin': 100,
            'BMI': 28.5,
            'DiabetesPedigreeFunction': 0.5,
            'Age': 35,
            'PhysicalActivity': 1,
            'Smoking': 0,
            'Alcohol': 0,
            'MotherDiabetes': 0,
            'FatherDiabetes': 0,
            'SiblingsDiabetes': 0,
            'GrandparentsDiabetes': 0
        }
    
    def get_model_info(self):
        """Get information about the trained models"""
        if not self.models:
            return {"error": "No models trained"}
        
        info = {
            'best_model': self.best_model_name,
            'available_models': list(self.models.keys()),
            'features_used': self.feature_names,
            'feature_count': len(self.feature_names),
            'training_data_shape': f"{len(self.X_train)} training samples"
        }
        
        # Add model performance metrics
        for name, model_data in self.models.items():
            info[f'{name}_accuracy'] = round(model_data['accuracy'], 4)
            info[f'{name}_auc_score'] = round(model_data['auc_score'], 4)
        
        return info
    
    def validate_input(self, input_data):
        """Validate input data and provide helpful error messages"""
        errors = []
        warnings = []
        
        # Check for required core features
        core_features_present = [f for f in self.core_features if f in input_data and input_data[f] is not None]
        
        if len(core_features_present) < 4:  # At least half of core features
            warnings.append("Some important medical measurements are missing. Using default values.")
        
        # Validate ranges for key features
        validations = {
            'Age': (0, 120, "Age should be between 0 and 120 years"),
            'BMI': (10, 70, "BMI should be between 10 and 70"),
            'Glucose': (50, 400, "Glucose level should be between 50 and 400 mg/dL"),
            'BloodPressure': (40, 200, "Blood pressure should be between 40 and 200 mmHg"),
            'Pregnancies': (0, 20, "Number of pregnancies should be between 0 and 20")
        }
        
        for field, (min_val, max_val, message) in validations.items():
            if field in input_data and input_data[field] is not None:
                try:
                    value = float(input_data[field])
                    if not (min_val <= value <= max_val):
                        errors.append(message)
                except (ValueError, TypeError):
                    errors.append(f"{field} must be a valid number")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def calculate_risk_score(self, input_data):
        """Calculate comprehensive risk score with detailed breakdown"""
        risk_factors = []
        total_score = 0
        
        # Age scoring
        age = input_data.get('Age', 30)
        if age >= 45:
            if age >= 65:
                risk_factors.append(("Age ≥ 65", 3))
                total_score += 3
            elif age >= 55:
                risk_factors.append(("Age 55-64", 2))
                total_score += 2
            else:
                risk_factors.append(("Age 45-54", 1))
                total_score += 1
        
        # BMI scoring
        bmi = input_data.get('BMI', 25)
        if bmi >= 40:
            risk_factors.append(("Severe obesity (BMI ≥ 40)", 4))
            total_score += 4
        elif bmi >= 35:
            risk_factors.append(("Obesity Class II (BMI 35-39.9)", 3))
            total_score += 3
        elif bmi >= 30:
            risk_factors.append(("Obesity Class I (BMI 30-34.9)", 2))
            total_score += 2
        elif bmi >= 25:
            risk_factors.append(("Overweight (BMI 25-29.9)", 1))
            total_score += 1
        
        # Glucose scoring
        glucose = input_data.get('Glucose', 100)
        if glucose >= 200:
            risk_factors.append(("Very high glucose (≥ 200)", 4))
            total_score += 4
        elif glucose >= 140:
            risk_factors.append(("High glucose (140-199)", 3))
            total_score += 3
        elif glucose >= 126:
            risk_factors.append(("Diabetic range glucose (126-139)", 2))
            total_score += 2
        elif glucose >= 100:
            risk_factors.append(("Pre-diabetic glucose (100-125)", 1))
            total_score += 1
        
        # Blood pressure scoring
        bp = input_data.get('BloodPressure', 80)
        if bp >= 180:
            risk_factors.append(("Hypertensive crisis (≥ 180)", 4))
            total_score += 4
        elif bp >= 160:
            risk_factors.append(("Stage 2 hypertension (160-179)", 3))
            total_score += 3
        elif bp >= 140:
            risk_factors.append(("Stage 1 hypertension (140-159)", 2))
            total_score += 2
        elif bp >= 130:
            risk_factors.append(("Elevated blood pressure (130-139)", 1))
            total_score += 1
        
        # Family history scoring
        family_score = 0
        if input_data.get('MotherDiabetes', 0) == 1:
            risk_factors.append(("Mother has diabetes", 2))
            family_score += 2
        if input_data.get('FatherDiabetes', 0) == 1:
            risk_factors.append(("Father has diabetes", 2))
            family_score += 2
        if input_data.get('SiblingsDiabetes', 0) == 1:
            risk_factors.append(("Sibling has diabetes", 1))
            family_score += 1
        if input_data.get('GrandparentsDiabetes', 0) == 1:
            risk_factors.append(("Grandparent had diabetes", 1))
            family_score += 1
        
        total_score += family_score
        
        # Lifestyle factors
        if input_data.get('Smoking', 0) == 1:
            risk_factors.append(("Smoking", 2))
            total_score += 2
        if input_data.get('PhysicalActivity', 1) == 0:
            risk_factors.append(("Sedentary lifestyle", 1))
            total_score += 1
        if input_data.get('Alcohol', 0) == 1:
            risk_factors.append(("Regular alcohol consumption", 1))
            total_score += 1
        
        # Pregnancy history (for women)
        pregnancies = input_data.get('Pregnancies', 0)
        if pregnancies >= 4:
            risk_factors.append((f"Multiple pregnancies ({pregnancies})", 2))
            total_score += 2
        elif pregnancies > 0:
            risk_factors.append((f"Previous pregnancies ({pregnancies})", 1))
            total_score += 1
        
        # Determine overall risk level
        if total_score <= 2:
            risk_level = "LOW"
            risk_color = "🟢"
            risk_description = "Minimal risk factors present"
        elif total_score <= 5:
            risk_level = "MODERATE"
            risk_color = "🟡"
            risk_description = "Some risk factors present - monitor closely"
        elif total_score <= 8:
            risk_level = "HIGH"
            risk_color = "🟠"
            risk_description = "Multiple risk factors - lifestyle changes recommended"
        else:
            risk_level = "VERY HIGH"
            risk_color = "🔴"
            risk_description = "Significant risk factors - medical evaluation needed"
        
        return {
            'risk_factors': risk_factors,
            'total_score': total_score,
            'family_score': family_score,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_description': risk_description,
            'max_possible_score': 25  # Approximate maximum
        }
    
    def get_recommendations(self, input_data, prediction_result, risk_assessment):
        """Get personalized recommendations based on assessment"""
        recommendations = []
        
        risk_level = risk_assessment['risk_level']
        
        # General recommendations based on risk level
        if risk_level == "LOW":
            recommendations.extend([
                "Maintain current healthy lifestyle habits",
                "Continue regular physical activity",
                "Annual health check-ups are sufficient"
            ])
        elif risk_level == "MODERATE":
            recommendations.extend([
                "Increase physical activity to 150+ minutes per week",
                "Monitor blood sugar every 6 months",
                "Consider dietary modifications to reduce refined sugars"
            ])
        elif risk_level == "HIGH":
            recommendations.extend([
                "Consult healthcare provider within 1-2 months",
                "Implement structured diet and exercise plan",
                "Monitor blood glucose levels regularly",
                "Consider joining a diabetes prevention program"
            ])
        else:  # VERY HIGH
            recommendations.extend([
                "URGENT: Schedule appointment with healthcare provider",
                "Comprehensive diabetes screening recommended",
                "Immediate lifestyle intervention needed",
                "Consider medication evaluation with doctor"
            ])
        
        # Specific recommendations based on risk factors
        bmi = input_data.get('BMI', 25)
        if bmi >= 30:
            recommendations.append("Weight loss of 5-10% can significantly reduce diabetes risk")
        
        if input_data.get('BloodPressure', 80) >= 140:
            recommendations.append("Blood pressure management is crucial - consult cardiologist")
        
        if input_data.get('Smoking', 0) == 1:
            recommendations.append("Smoking cessation will significantly improve your health outcomes")
        
        if input_data.get('PhysicalActivity', 1) == 0:
            recommendations.append("Start with 30 minutes of moderate exercise 3-4 times per week")
        
        return recommendations
    
    def get_comprehensive_result(self, input_data):
        """Get comprehensive prediction result with all template variables"""
        # Get basic prediction
        prediction_result = self.predict(input_data)
        
        # Get risk assessment
        risk_assessment = self.calculate_risk_score(input_data)
        
        # Get recommendations
        recommendations = self.get_recommendations(input_data, prediction_result, risk_assessment)
        
        # Get input validation
        validation = self.validate_input(input_data)
        
        # Combine all results for template
        comprehensive_result = {
            # Basic prediction results
            'prediction': prediction_result['prediction'],
            'probability_no_diabetes': prediction_result['probability_no_diabetes'],
            'probability_diabetes': prediction_result['probability_diabetes'],
            'probabilities': prediction_result['probabilities'],  # Nested object
            'confidence': prediction_result['confidence'],
            
            # Model information
            'model_used': prediction_result['model_used'],
            'model_accuracy': prediction_result['model_accuracy'],
            'model_auc_score': prediction_result['model_auc_score'],
            'features_used': prediction_result['features_used'],
            'feature_count': prediction_result['feature_count'],
            
            # Risk assessment
            'risk_factors': risk_assessment['risk_factors'],
            'total_risk_score': risk_assessment['total_score'],
            'family_risk_score': risk_assessment['family_score'],
            'risk_level': risk_assessment['risk_level'],
            'risk_color': risk_assessment['risk_color'],
            'risk_description': risk_assessment['risk_description'],
            'max_risk_score': risk_assessment['max_possible_score'],
            
            # Recommendations and validation
            'recommendations': recommendations,
            'validation_errors': validation['errors'],
            'validation_warnings': validation['warnings'],
            'is_valid_input': validation['is_valid'],
            
            # Additional computed fields for templates
            'risk_percentage': round((risk_assessment['total_score'] / risk_assessment['max_possible_score']) * 100, 1),
            'diabetes_risk_text': 'High Risk' if prediction_result['prediction'] == 1 else 'Low Risk',
            'confidence_percentage': round(prediction_result['confidence'] * 100, 1),
            'model_accuracy_percentage': round(prediction_result['model_accuracy'] * 100, 1),
            'auc_score_percentage': round(prediction_result['model_auc_score'] * 100, 1),
            
            # Input data (for display purposes)
            'input_data': input_data,
            
            # Status indicators
            'has_risk_factors': len(risk_assessment['risk_factors']) > 0,
            'has_family_history': risk_assessment['family_score'] > 0,
            'needs_medical_attention': risk_assessment['risk_level'] in ['HIGH', 'VERY HIGH'],
            
            # Color coding for UI
            'prediction_color': 'danger' if prediction_result['prediction'] == 1 else 'success',
            'risk_level_color': {
                'LOW': 'success',
                'MODERATE': 'warning', 
                'HIGH': 'orange',
                'VERY HIGH': 'danger'
            }.get(risk_assessment['risk_level'], 'info')
        }
        
        return comprehensive_result

# For backwards compatibility
def load_model(data_dir):
    """Load the diabetes model - for backwards compatibility"""
    return DiabetesModel(data_dir)

if __name__ == "__main__":
    # Test the model
    print("Testing Diabetes Model...")
    model = DiabetesModel()
    
    # Test prediction
    test_data = {
        'Pregnancies': 2,
        'Glucose': 120,
        'BloodPressure': 80,
        'SkinThickness': 25,
        'Insulin': 100,
        'BMI': 28.5,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 35
    }
    
    result = model.predict(test_data)
    print(f"\nTest Prediction Results:")
    print(f"Prediction: {'Diabetes Risk' if result['prediction'] == 1 else 'No Diabetes Risk'}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Model Used: {result['model_used']}")
