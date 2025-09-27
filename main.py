# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
# from sklearn.model_selection import cross_val_score
# import warnings
# warnings.filterwarnings('ignore')

# class DiabetesDetector:
#     """
#     Complete Diabetes Detection System using Machine Learning
#     Based on Pima Indian Diabetes Dataset with family history integration
#     """
    
#     def __init__(self):
#         self.models = {}
#         self.scaler = StandardScaler()
#         self.feature_names = [
#             'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#             'BMI', 'DiabetesPedigreeFunction', 'Age', 'PhysicalActivity', 
#             'Smoking', 'Alcohol', 'MotherDiabetes', 'FatherDiabetes', 
#             'SiblingsDiabetes', 'GrandparentsDiabetes'
#         ]
#         self.load_pima_dataset()
        
#     def load_pima_dataset(self):
#         """Load and prepare the Pima Indian Diabetes Dataset"""
       
#         pima_data = """Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
# 6,148,72,35,0,33.6,0.627,50,1
# 1,85,66,29,0,26.6,0.351,31,0
# 8,183,64,0,0,23.3,0.672,32,1
# 1,89,66,23,94,28.1,0.167,21,0
# 0,137,40,35,168,43.1,2.288,33,1
# 5,116,74,0,0,25.6,0.201,30,0
# 3,78,50,32,88,31.0,0.248,26,1
# 10,115,0,0,0,35.3,0.134,29,0
# 2,197,70,45,543,30.5,0.158,53,1
# 8,125,96,0,0,0.0,0.232,54,1
# 4,110,92,0,0,37.6,0.191,30,0
# 10,168,74,0,0,38.0,0.537,34,1
# 10,139,80,0,0,27.1,1.441,57,0
# 1,189,60,23,846,30.1,0.398,59,1
# 5,166,72,19,175,25.8,0.587,51,1
# 7,100,0,0,0,30.0,0.484,32,1
# 0,118,84,47,230,45.8,0.551,31,1
# 7,107,74,0,0,29.6,0.254,31,1
# 1,103,30,38,83,43.3,0.183,33,0
# 1,115,70,30,96,34.6,0.529,32,1
# 3,126,88,41,235,39.3,0.704,27,0
# 8,99,84,0,0,35.4,0.388,50,0
# 7,196,90,0,0,39.8,0.451,41,1
# 9,119,80,35,0,29.0,0.263,29,1
# 11,143,94,33,146,36.6,0.254,51,1
# 10,125,70,26,115,31.1,0.205,41,1
# 7,147,76,0,0,39.4,0.257,43,1
# 1,97,66,15,140,23.2,0.487,22,0
# 13,145,82,19,110,22.2,0.245,57,0
# 5,117,92,0,0,34.1,0.337,38,0"""
        
#         from io import StringIO
#         df = pd.read_csv(StringIO(pima_data))
        
#         np.random.seed(42) 
#         df['PhysicalActivity'] = np.where(df['BMI'] > 30, 
#                                         np.random.choice([0, 1], len(df), p=[0.7, 0.3]),
#                                         np.random.choice([0, 1], len(df), p=[0.3, 0.7]))
        
#         df['Smoking'] = np.where(df['Age'] > 40, 
#                                np.random.choice([0, 1], len(df), p=[0.7, 0.3]),
#                                np.random.choice([0, 1], len(df), p=[0.8, 0.2]))
        
#         df['Alcohol'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
        
#         df['MotherDiabetes'] = np.where(df['DiabetesPedigreeFunction'] > 0.5, 1, 0)
#         df['FatherDiabetes'] = np.where(df['DiabetesPedigreeFunction'] > 0.7, 1, 0)
#         df['SiblingsDiabetes'] = np.where(df['DiabetesPedigreeFunction'] > 0.3, 1, 0)
#         df['GrandparentsDiabetes'] = np.where(df['DiabetesPedigreeFunction'] > 0.4, 1, 0)
        
#         self.data = df
#         print("✅ Pima Indian Diabetes Dataset loaded successfully!")
#         print(f"📊 Dataset shape: {df.shape}")
        
#     def preprocess_data(self):
#         """Preprocess the data for training"""
       
#         cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
#         for col in cols_with_zeros:
#             self.data[col] = self.data[col].replace(0, np.nan)
#             self.data[col].fillna(self.data[col].median(), inplace=True)
        
#         feature_cols = [col for col in self.data.columns if col != 'Outcome']
#         X = self.data[feature_cols]
#         y = self.data['Outcome']
        
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=y
#         )
        
#         self.X_train_scaled = self.scaler.fit_transform(self.X_train)
#         self.X_test_scaled = self.scaler.transform(self.X_test)
        
#         print("✅ Data preprocessing completed!")
        
#     def train_models(self):
#         """Train multiple machine learning models"""
#         print("🧠 Training multiple ML models...")
        
#         models_to_train = {
#             'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
#             'Logistic Regression': LogisticRegression(random_state=42),
#             'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
#         }
        
#         for name, model in models_to_train.items():
#             print(f"\n📈 Training {name}...")
            
#             if name == 'Random Forest':
#                 model.fit(self.X_train, self.y_train)
#                 y_pred = model.predict(self.X_test)
#             else:
#                 model.fit(self.X_train_scaled, self.y_train)
#                 y_pred = model.predict(self.X_test_scaled)
            
#             accuracy = accuracy_score(self.y_test, y_pred)
            
#             if name == 'Random Forest':
#                 y_pred_proba = model.predict_proba(self.X_test)[:, 1]
#             else:
#                 y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
#             auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
#             # Store model and results
#             self.models[name] = {
#                 'model': model,
#                 'accuracy': accuracy,
#                 'auc_score': auc_score,
#                 'predictions': y_pred,
#                 'probabilities': y_pred_proba
#             }
            
#             print(f"   Accuracy: {accuracy:.4f}")
#             print(f"   AUC Score: {auc_score:.4f}")
        
#         self.best_model_name = max(self.models.keys(), 
#                                  key=lambda x: self.models[x]['accuracy'])
#         print(f"\n🏆 Best model: {self.best_model_name}")
        
#     def evaluate_models(self):
#         """Generate comprehensive model evaluation"""
#         print("\n📊 COMPREHENSIVE MODEL EVALUATION")
#         print("="*50)
        
#         for name, results in self.models.items():
#             print(f"\n🔍 {name} Results:")
#             print(f"Accuracy: {results['accuracy']:.4f}")
#             print(f"AUC Score: {results['auc_score']:.4f}")
            
#             print(f"\nClassification Report:")
#             print(classification_report(self.y_test, results['predictions']))
            
#     def visualize_results(self):
#         """Create visualizations for model performance"""
#         fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
#         model_names = list(self.models.keys())
#         accuracies = [self.models[name]['accuracy'] for name in model_names]
#         auc_scores = [self.models[name]['auc_score'] for name in model_names]
        
#         ax1 = axes[0, 0]
#         x = np.arange(len(model_names))
#         width = 0.35
        
#         ax1.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
#         ax1.bar(x + width/2, auc_scores, width, label='AUC Score', alpha=0.8)
#         ax1.set_xlabel('Models')
#         ax1.set_ylabel('Score')
#         ax1.set_title('Model Performance Comparison')
#         ax1.set_xticks(x)
#         ax1.set_xticklabels(model_names, rotation=45)
#         ax1.legend()
        
#         best_results = self.models[self.best_model_name]
#         cm = confusion_matrix(self.y_test, best_results['predictions'])
        
#         ax2 = axes[0, 1]
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
#         ax2.set_title(f'Confusion Matrix - {self.best_model_name}')
#         ax2.set_xlabel('Predicted')
#         ax2.set_ylabel('Actual')
        
#         if 'Random Forest' in self.models:
#             rf_model = self.models['Random Forest']['model']
#             feature_importance = pd.DataFrame({
#                 'feature': self.X_train.columns,
#                 'importance': rf_model.feature_importances_
#             }).sort_values('importance', ascending=True)
            
#             ax3 = axes[1, 0]
#             ax3.barh(feature_importance['feature'], feature_importance['importance'])
#             ax3.set_title('Feature Importance (Random Forest)')
#             ax3.set_xlabel('Importance')
        
#         ax4 = axes[1, 1]
#         self.data['Outcome'].value_counts().plot(kind='pie', ax=ax4, autopct='%1.1f%%')
#         ax4.set_title('Diabetes Distribution in Dataset')
#         ax4.set_ylabel('')
        
#         plt.tight_layout()
#         plt.show()
        
#     def predict_diabetes(self, user_data):
#         """Predict diabetes for new user data"""
#         best_model = self.models[self.best_model_name]['model']
        
#         user_df = pd.DataFrame([user_data])
        
#         for col in self.X_train.columns:
#             if col not in user_df.columns:
#                 user_df[col] = 0
        
#         user_df = user_df[self.X_train.columns]
        
#         if self.best_model_name != 'Random Forest':
#             user_scaled = self.scaler.transform(user_df)
#             prediction = best_model.predict(user_scaled)[0]
#             probability = best_model.predict_proba(user_scaled)[0]
#         else:
#             prediction = best_model.predict(user_df)[0]
#             probability = best_model.predict_proba(user_df)[0]
        
#         return prediction, probability
    
#     def calculate_risk_factors(self, user_data):
#         """Calculate individual risk factors"""
#         risk_factors = []
        
#         if user_data.get('Age', 0) > 45:
#             risk_factors.append("Age over 45")
#         if user_data.get('BMI', 0) > 25:
#             risk_factors.append("Overweight (BMI > 25)")
#         if user_data.get('BMI', 0) > 30:
#             risk_factors.append("Obesity (BMI > 30)")
#         if user_data.get('Glucose', 0) > 140:
#             risk_factors.append("High glucose levels")
#         if user_data.get('BloodPressure', 0) > 140:
#             risk_factors.append("High blood pressure")
#         if user_data.get('MotherDiabetes', 0) == 1:
#             risk_factors.append("Mother has diabetes")
#         if user_data.get('FatherDiabetes', 0) == 1:
#             risk_factors.append("Father has diabetes")
#         if user_data.get('SiblingsDiabetes', 0) == 1:
#             risk_factors.append("Sibling has diabetes")
#         if user_data.get('Smoking', 0) == 1:
#             risk_factors.append("Smoking")
#         if user_data.get('PhysicalActivity', 1) == 0:
#             risk_factors.append("Low physical activity")
        
#         return risk_factors

# def get_user_input():
#     """Get user input for diabetes prediction"""
#     print("\n" + "="*60)
#     print("🩺 DIABETES RISK ASSESSMENT")
#     print("="*60)
#     print("Please provide the following information:")
    
#     user_data = {}
    
#     print("\n👤 Personal Information:")
#     user_data['Age'] = float(input("Age: "))
#     user_data['Pregnancies'] = float(input("Number of pregnancies (0 if male): "))
    
#     print("\n🏥 Medical Measurements:")
#     user_data['Glucose'] = float(input("Glucose level (mg/dL, normal: 70-100): "))
#     user_data['BloodPressure'] = float(input("Blood pressure (systolic, normal: 90-120): "))
#     user_data['BMI'] = float(input("BMI (normal: 18.5-24.9): "))
#     user_data['SkinThickness'] = float(input("Skin thickness (mm, optional, press 0 if unknown): ") or "20")
#     user_data['Insulin'] = float(input("Insulin level (mu U/ml, optional, press 0 if unknown): ") or "100")
    
#     print("\n👨‍👩‍👧‍👦 Family History:")
#     user_data['MotherDiabetes'] = int(input("Mother has diabetes? (1=Yes, 0=No): "))
#     user_data['FatherDiabetes'] = int(input("Father has diabetes? (1=Yes, 0=No): "))
#     user_data['SiblingsDiabetes'] = int(input("Any siblings have diabetes? (1=Yes, 0=No): "))
#     user_data['GrandparentsDiabetes'] = int(input("Any grandparents had diabetes? (1=Yes, 0=No): "))
    
#     print("\n🏃‍♀️ Lifestyle Factors:")
#     user_data['PhysicalActivity'] = int(input("Regular physical activity? (1=Yes, 0=No): "))
#     user_data['Smoking'] = int(input("Do you smoke? (1=Yes, 0=No): "))
#     user_data['Alcohol'] = int(input("Regular alcohol consumption? (1=Yes, 0=No): "))
    
#     pedigree = 0
#     if user_data['MotherDiabetes']: pedigree += 0.5
#     if user_data['FatherDiabetes']: pedigree += 0.5
#     if user_data['SiblingsDiabetes']: pedigree += 0.3
#     if user_data['GrandparentsDiabetes']: pedigree += 0.2
#     user_data['DiabetesPedigreeFunction'] = min(pedigree, 2.0)
    
#     return user_data

# def main():
#     """Main function to run the diabetes detection system"""
#     print("🚀 Initializing Diabetes Detection System...")
    
#     detector = DiabetesDetector()
    
#     detector.preprocess_data()
#     detector.train_models()
#     detector.evaluate_models()
    
#     print("\n📊 Generating visualizations...")
#     detector.visualize_results()
    
#     while True:
#         print("\n" + "="*60)
#         choice = input("Would you like to:\n1. Make a prediction\n2. Exit\nChoice (1/2): ")
        
#         if choice == '1':
           
#             user_data = get_user_input()
            
#             prediction, probability = detector.predict_diabetes(user_data)
#             risk_factors = detector.calculate_risk_factors(user_data)
            
#             print("\n" + "="*60)
#             print("🔍 DIABETES RISK ASSESSMENT RESULTS")
#             print("="*60)
            
#             if prediction == 1:
#                 print("⚠️  HIGH RISK: Based on the provided information,")
#                 print("   you may be at risk for Type 2 diabetes.")
#                 risk_level = "HIGH"
#                 color = "🔴"
#             else:
#                 print("✅ LOW RISK: Based on the provided information,")
#                 print("   you appear to have a low risk for diabetes.")
#                 risk_level = "LOW"
#                 color = "🟢"
            
#             print(f"\n{color} Risk Level: {risk_level}")
#             print(f"📊 Confidence: {probability[prediction]*100:.1f}%")
#             print(f"🧠 Model Used: {detector.best_model_name}")
            
#             print(f"\n📋 Detailed Probabilities:")
#             print(f"   No Diabetes: {probability[0]*100:.1f}%")
#             print(f"   Diabetes: {probability[1]*100:.1f}%")
            
#             if risk_factors:
#                 print(f"\n⚡ Risk Factors Identified:")
#                 for factor in risk_factors:
#                     print(f"   • {factor}")
#             else:
#                 print(f"\n✅ No major risk factors identified")
            
#             print("\n" + "⚠️"*20)
#             print("MEDICAL DISCLAIMER:")
#             print("This is an AI-based screening tool for educational purposes only.")
#             print("It should NOT replace professional medical advice, diagnosis, or treatment.")
#             print("Please consult with a healthcare provider for accurate medical assessment.")
#             print("⚠️"*20)
            
#         elif choice == '2':
#             print("👋 Thank you for using the Diabetes Detection System!")
#             break
#         else:
#             print("❌ Invalid choice. Please try again.")

# if __name__ == "__main__":
#     main()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class DiabetesDetector:
    """
    Complete Diabetes Detection System using Machine Learning
    Based on Pima Indian Diabetes Dataset with family history integration
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
            'BMI', 'DiabetesPedigreeFunction', 'Age', 'PhysicalActivity', 
            'Smoking', 'Alcohol', 'MotherDiabetes', 'FatherDiabetes', 
            'SiblingsDiabetes', 'GrandparentsDiabetes'
        ]
        self.load_pima_dataset()
        
    def load_pima_dataset(self):
        """Load and prepare the Pima Indian Diabetes Dataset"""
       
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
5,117,92,0,0,34.1,0.337,38,0"""
        
        from io import StringIO
        df = pd.read_csv(StringIO(pima_data))
        
        np.random.seed(42) 
        df['PhysicalActivity'] = np.where(df['BMI'] > 30, 
                                        np.random.choice([0, 1], len(df), p=[0.7, 0.3]),
                                        np.random.choice([0, 1], len(df), p=[0.3, 0.7]))
        
        df['Smoking'] = np.where(df['Age'] > 40, 
                               np.random.choice([0, 1], len(df), p=[0.7, 0.3]),
                               np.random.choice([0, 1], len(df), p=[0.8, 0.2]))
        
        df['Alcohol'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
        
        df['MotherDiabetes'] = np.where(df['DiabetesPedigreeFunction'] > 0.5, 1, 0)
        df['FatherDiabetes'] = np.where(df['DiabetesPedigreeFunction'] > 0.7, 1, 0)
        df['SiblingsDiabetes'] = np.where(df['DiabetesPedigreeFunction'] > 0.3, 1, 0)
        df['GrandparentsDiabetes'] = np.where(df['DiabetesPedigreeFunction'] > 0.4, 1, 0)
        
        self.data = df
        print("✅ Pima Indian Diabetes Dataset loaded successfully!")
        print(f"📊 Dataset shape: {df.shape}")
        
    def preprocess_data(self):
        """Preprocess the data for training"""
       
        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        for col in cols_with_zeros:
            self.data[col] = self.data[col].replace(0, np.nan)
            self.data[col].fillna(self.data[col].median(), inplace=True)
        
        feature_cols = [col for col in self.data.columns if col != 'Outcome']
        X = self.data[feature_cols]
        y = self.data['Outcome']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✅ Data preprocessing completed!")
        
    def train_models(self):
        """Train multiple machine learning models"""
        print("🧠 Training multiple ML models...")
        
        models_to_train = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        for name, model in models_to_train.items():
            print(f"\n📈 Training {name}...")
            
            if name == 'Random Forest':
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            else:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            
            if name == 'Random Forest':
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            else:
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store model and results
            self.models[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   AUC Score: {auc_score:.4f}")
        
        self.best_model_name = max(self.models.keys(), 
                                 key=lambda x: self.models[x]['accuracy'])
        print(f"\n🏆 Best model: {self.best_model_name}")
        
    def evaluate_models(self):
        """Generate comprehensive model evaluation"""
        print("\n📊 COMPREHENSIVE MODEL EVALUATION")
        print("="*50)
        
        for name, results in self.models.items():
            print(f"\n🔍 {name} Results:")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"AUC Score: {results['auc_score']:.4f}")
            
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, results['predictions']))
            
    def visualize_results(self):
        """Create visualizations for model performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = list(self.models.keys())
        accuracies = [self.models[name]['accuracy'] for name in model_names]
        auc_scores = [self.models[name]['auc_score'] for name in model_names]
        
        ax1 = axes[0, 0]
        x = np.arange(len(model_names))
        width = 0.35
        
        ax1.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        ax1.bar(x + width/2, auc_scores, width, label='AUC Score', alpha=0.8)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.legend()
        
        best_results = self.models[self.best_model_name]
        cm = confusion_matrix(self.y_test, best_results['predictions'])
        
        ax2 = axes[0, 1]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title(f'Confusion Matrix - {self.best_model_name}')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            ax3 = axes[1, 0]
            ax3.barh(feature_importance['feature'], feature_importance['importance'])
            ax3.set_title('Feature Importance (Random Forest)')
            ax3.set_xlabel('Importance')
        
        ax4 = axes[1, 1]
        self.data['Outcome'].value_counts().plot(kind='pie', ax=ax4, autopct='%1.1f%%')
        ax4.set_title('Diabetes Distribution in Dataset')
        ax4.set_ylabel('')
        
        plt.tight_layout()
        plt.show()
        
    def predict_diabetes(self, user_data):
        """Predict diabetes for new user data"""
        best_model = self.models[self.best_model_name]['model']
        
        user_df = pd.DataFrame([user_data])
        
        for col in self.X_train.columns:
            if col not in user_df.columns:
                user_df[col] = 0
        
        user_df = user_df[self.X_train.columns]
        
        if self.best_model_name != 'Random Forest':
            user_scaled = self.scaler.transform(user_df)
            prediction = best_model.predict(user_scaled)[0]
            probability = best_model.predict_proba(user_scaled)[0]
        else:
            prediction = best_model.predict(user_df)[0]
            probability = best_model.predict_proba(user_df)[0]
        
        return prediction, probability
    
    def calculate_blood_pressure_score(self, bp):
        """Calculate blood pressure risk score"""
        if bp < 120:
            return 0, "Normal (< 120)"
        elif 120 <= bp < 135:
            return 1, "Elevated (120-134)"
        elif 135 <= bp < 150:
            return 2, "High Stage 1 (135-149)"
        elif 150 <= bp < 165:
            return 3, "High Stage 2 (150-164)"
        else:
            return 4, "Very High (≥ 165)"
    
    def calculate_glucose_score(self, glucose):
        """Calculate glucose risk score"""
        if glucose < 100:
            return 0, "Normal (< 100)"
        elif 100 <= glucose < 126:
            return 1, "Pre-diabetic (100-125)"
        elif 126 <= glucose < 140:
            return 2, "Diabetic Stage 1 (126-139)"
        elif 140 <= glucose < 180:
            return 3, "Diabetic Stage 2 (140-179)"
        else:
            return 4, "Severe Diabetic (≥ 180)"
    
    def calculate_bmi_score(self, bmi):
        """Calculate BMI risk score"""
        if bmi < 18.5:
            return 0, "Underweight (< 18.5)"
        elif 18.5 <= bmi < 25:
            return 0, "Normal (18.5-24.9)"
        elif 25 <= bmi < 30:
            return 1, "Overweight (25-29.9)"
        elif 30 <= bmi < 35:
            return 2, "Obese Class I (30-34.9)"
        elif 35 <= bmi < 40:
            return 3, "Obese Class II (35-39.9)"
        else:
            return 4, "Obese Class III (≥ 40)"
    
    def calculate_age_score(self, age):
        """Calculate age risk score"""
        if age < 30:
            return 0, "Low Risk (< 30)"
        elif 30 <= age < 40:
            return 1, "Mild Risk (30-39)"
        elif 40 <= age < 50:
            return 2, "Moderate Risk (40-49)"
        elif 50 <= age < 60:
            return 3, "High Risk (50-59)"
        else:
            return 4, "Very High Risk (≥ 60)"
    
    def calculate_insulin_score(self, insulin):
        """Calculate insulin risk score"""
        if insulin < 16:
            return 1, "Low (< 16) - Possible insulin deficiency"
        elif 16 <= insulin < 166:
            return 0, "Normal (16-166)"
        elif 166 <= insulin < 250:
            return 2, "Elevated (166-249) - Insulin resistance"
        elif 250 <= insulin < 400:
            return 3, "High (250-399) - Significant resistance"
        else:
            return 4, "Very High (≥ 400) - Severe resistance"
    
    def calculate_risk_factors(self, user_data):
        """Calculate individual risk factors with graduated scoring"""
        risk_factors = []
        total_risk_score = 0
        
        # Age scoring
        age_score, age_desc = self.calculate_age_score(user_data.get('Age', 0))
        if age_score > 0:
            risk_factors.append(f"Age: {age_desc} (Score: {age_score})")
            total_risk_score += age_score
        
        # BMI scoring
        bmi_score, bmi_desc = self.calculate_bmi_score(user_data.get('BMI', 0))
        if bmi_score > 0:
            risk_factors.append(f"BMI: {bmi_desc} (Score: {bmi_score})")
            total_risk_score += bmi_score
        
        # Glucose scoring
        glucose_score, glucose_desc = self.calculate_glucose_score(user_data.get('Glucose', 0))
        if glucose_score > 0:
            risk_factors.append(f"Glucose: {glucose_desc} (Score: {glucose_score})")
            total_risk_score += glucose_score
        
        # Blood pressure scoring
        bp_score, bp_desc = self.calculate_blood_pressure_score(user_data.get('BloodPressure', 0))
        if bp_score > 0:
            risk_factors.append(f"Blood Pressure: {bp_desc} (Score: {bp_score})")
            total_risk_score += bp_score
        
        # Insulin scoring
        insulin_score, insulin_desc = self.calculate_insulin_score(user_data.get('Insulin', 100))
        if insulin_score > 0:
            risk_factors.append(f"Insulin: {insulin_desc} (Score: {insulin_score})")
            total_risk_score += insulin_score
        
        # Family history scoring
        family_score = 0
        if user_data.get('MotherDiabetes', 0) == 1:
            risk_factors.append("Mother has diabetes (Score: 2)")
            total_risk_score += 2
            family_score += 2
        
        if user_data.get('FatherDiabetes', 0) == 1:
            risk_factors.append("Father has diabetes (Score: 2)")
            total_risk_score += 2
            family_score += 2
        
        if user_data.get('SiblingsDiabetes', 0) == 1:
            risk_factors.append("Sibling has diabetes (Score: 1)")
            total_risk_score += 1
            family_score += 1
        
        if user_data.get('GrandparentsDiabetes', 0) == 1:
            risk_factors.append("Grandparent had diabetes (Score: 1)")
            total_risk_score += 1
            family_score += 1
        
        # Lifestyle factors
        if user_data.get('Smoking', 0) == 1:
            risk_factors.append("Smoking (Score: 2)")
            total_risk_score += 2
        
        if user_data.get('PhysicalActivity', 1) == 0:
            risk_factors.append("Low physical activity (Score: 1)")
            total_risk_score += 1
        
        if user_data.get('Alcohol', 0) == 1:
            risk_factors.append("Regular alcohol consumption (Score: 1)")
            total_risk_score += 1
        
        # Pregnancy risk (for women)
        pregnancies = user_data.get('Pregnancies', 0)
        if pregnancies >= 4:
            risk_factors.append(f"Multiple pregnancies ({pregnancies}) (Score: 2)")
            total_risk_score += 2
        elif pregnancies > 0:
            risk_factors.append(f"Previous pregnancies ({pregnancies}) (Score: 1)")
            total_risk_score += 1
        
        return risk_factors, total_risk_score, family_score
    
    def get_risk_level_description(self, total_score):
        """Get overall risk level based on total score"""
        if total_score <= 2:
            return "LOW", "🟢", "Minimal risk factors present"
        elif total_score <= 5:
            return "MODERATE", "🟡", "Some risk factors present - monitor closely"
        elif total_score <= 8:
            return "HIGH", "🟠", "Multiple risk factors - consider lifestyle changes"
        else:
            return "VERY HIGH", "🔴", "Significant risk factors - seek medical evaluation"

def get_user_input():
    """Get user input for diabetes prediction"""
    print("\n" + "="*60)
    print("🩺 DIABETES RISK ASSESSMENT")
    print("="*60)
    print("Please provide the following information:")
    
    user_data = {}
    
    print("\n👤 Personal Information:")
    user_data['Age'] = float(input("Age: "))
    user_data['Pregnancies'] = float(input("Number of pregnancies (0 if male): "))
    
    print("\n🏥 Medical Measurements:")
    user_data['Glucose'] = float(input("Glucose level (mg/dL, normal: 70-100): "))
    user_data['BloodPressure'] = float(input("Blood pressure (systolic, normal: 90-120): "))
    user_data['BMI'] = float(input("BMI (normal: 18.5-24.9): "))
    user_data['SkinThickness'] = float(input("Skin thickness (mm, optional, press 0 if unknown): ") or "20")
    user_data['Insulin'] = float(input("Insulin level (mu U/ml, optional, press 0 if unknown): ") or "100")
    
    print("\n👨‍👩‍👧‍👦 Family History:")
    user_data['MotherDiabetes'] = int(input("Mother has diabetes? (1=Yes, 0=No): "))
    user_data['FatherDiabetes'] = int(input("Father has diabetes? (1=Yes, 0=No): "))
    user_data['SiblingsDiabetes'] = int(input("Any siblings have diabetes? (1=Yes, 0=No): "))
    user_data['GrandparentsDiabetes'] = int(input("Any grandparents had diabetes? (1=Yes, 0=No): "))
    
    print("\n🏃‍♀️ Lifestyle Factors:")
    user_data['PhysicalActivity'] = int(input("Regular physical activity? (1=Yes, 0=No): "))
    user_data['Smoking'] = int(input("Do you smoke? (1=Yes, 0=No): "))
    user_data['Alcohol'] = int(input("Regular alcohol consumption? (1=Yes, 0=No): "))
    
    # Calculate pedigree function based on family history
    pedigree = 0
    if user_data['MotherDiabetes']: pedigree += 0.5
    if user_data['FatherDiabetes']: pedigree += 0.5
    if user_data['SiblingsDiabetes']: pedigree += 0.3
    if user_data['GrandparentsDiabetes']: pedigree += 0.2
    user_data['DiabetesPedigreeFunction'] = min(pedigree, 2.0)
    
    return user_data

def main():
    """Main function to run the diabetes detection system"""
    print("🚀 Initializing Enhanced Diabetes Detection System...")
    
    detector = DiabetesDetector()
    
    detector.preprocess_data()
    detector.train_models()
    detector.evaluate_models()
    
    print("\n📊 Generating visualizations...")
    detector.visualize_results()
    
    while True:
        print("\n" + "="*60)
        choice = input("Would you like to:\n1. Make a prediction\n2. Exit\nChoice (1/2): ")
        
        if choice == '1':
           
            user_data = get_user_input()
            
            prediction, probability = detector.predict_diabetes(user_data)
            risk_factors, total_risk_score, family_score = detector.calculate_risk_factors(user_data)
            risk_level, color, description = detector.get_risk_level_description(total_risk_score)
            
            print("\n" + "="*60)
            print("🔍 ENHANCED DIABETES RISK ASSESSMENT RESULTS")
            print("="*60)
            
            if prediction == 1:
                print("⚠️  HIGH RISK: Based on the provided information,")
                print("   you may be at risk for Type 2 diabetes.")
                ai_risk_level = "HIGH"
                ai_color = "🔴"
            else:
                print("✅ LOW RISK: Based on the provided information,")
                print("   you appear to have a low risk for diabetes.")
                ai_risk_level = "LOW"
                ai_color = "🟢"
            
            print(f"\n{ai_color} AI Model Risk Level: {ai_risk_level}")
            print(f"{color} Comprehensive Risk Level: {risk_level}")
            print(f"📊 AI Confidence: {probability[prediction]*100:.1f}%")
            print(f"📈 Total Risk Score: {total_risk_score}/20+")
            print(f"🧬 Family History Score: {family_score}")
            print(f"🧠 Model Used: {detector.best_model_name}")
            
            print(f"\n📋 Detailed AI Probabilities:")
            print(f"   No Diabetes: {probability[0]*100:.1f}%")
            print(f"   Diabetes: {probability[1]*100:.1f}%")
            
            print(f"\n💡 Risk Assessment Summary:")
            print(f"   {description}")
            
            if risk_factors:
                print(f"\n⚡ Risk Factors Identified (with scores):")
                for factor in risk_factors:
                    print(f"   • {factor}")
            else:
                print(f"\n✅ No major risk factors identified")
            
            # Recommendations based on risk level
            print(f"\n🎯 Personalized Recommendations:")
            if risk_level == "LOW":
                print("   • Continue healthy lifestyle habits")
                print("   • Annual health check-ups")
                print("   • Maintain current weight and activity level")
            elif risk_level == "MODERATE":
                print("   • Increase physical activity to 150+ minutes/week")
                print("   • Monitor blood sugar every 6 months")
                print("   • Consider dietary modifications")
            elif risk_level == "HIGH":
                print("   • Consult healthcare provider within 1-2 months")
                print("   • Implement structured diet and exercise plan")
                print("   • Monitor blood glucose regularly")
                print("   • Consider diabetes prevention program")
            else:  # VERY HIGH
                print("   • URGENT: Schedule appointment with healthcare provider")
                print("   • Comprehensive diabetes screening recommended")
                print("   • Immediate lifestyle intervention needed")
                print("   • Consider medication evaluation")
            
            print("\n" + "⚠️"*20)
            print("MEDICAL DISCLAIMER:")
            print("This is an AI-based screening tool for educational purposes only.")
            print("It should NOT replace professional medical advice, diagnosis, or treatment.")
            print("Please consult with a healthcare provider for accurate medical assessment.")
            print("The scoring system is based on general risk factors and may not")
            print("account for individual medical conditions or circumstances.")
            print("⚠️"*20)
            
        elif choice == '2':
            print("👋 Thank you for using the Enhanced Diabetes Detection System!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()