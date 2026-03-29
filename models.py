# models.py - Model Training and Prediction

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Support Vector Machine": SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
            "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        self.results = None
        self.trained_models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def train_all(self, X, y, test_size=0.2):
        """Train all models"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        results = []
        
        for name, model in self.models.items():
            try:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', model)
                ])
                
                start_time = time.time()
                pipeline.fit(self.X_train, self.y_train)
                train_time = time.time() - start_time
                
                y_pred = pipeline.predict(self.X_test)
                
                acc = accuracy_score(self.y_test, y_pred)
                prec = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                
                try:
                    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
                    cv_mean = cv_scores.mean()
                except:
                    cv_mean = 0
                
                results.append({
                    "Model": name,
                    "Accuracy": f"{acc:.4f}",
                    "Accuracy_Score": acc,
                    "Precision": f"{prec:.4f}",
                    "Recall": f"{rec:.4f}",
                    "F1-Score": f"{f1:.4f}",
                    "CV Score": f"{cv_mean:.4f}",
                    "Training Time": f"{train_time:.2f}s",
                    "Model Object": pipeline
                })
                
                self.trained_models[name] = pipeline
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                results.append({
                    "Model": name,
                    "Accuracy": "Error",
                    "Accuracy_Score": 0,
                    "Precision": "Error",
                    "Recall": "Error",
                    "F1-Score": "Error",
                    "CV Score": "Error",
                    "Training Time": "Error",
                    "Model Object": None
                })
        
        self.results = pd.DataFrame(results)
        self.results = self.results.sort_values('Accuracy_Score', ascending=False)
        
        return self.results
    
    def get_best_model(self):
        """Get best model"""
        if len(self.results) > 0:
            best_row = self.results.iloc[0]
            return best_row["Model"], self.trained_models.get(best_row["Model"])
        return None, None
    
    def predict(self, model_name, X):
        """Make prediction"""
        if model_name in self.trained_models:
            model = self.trained_models[model_name]
            pred = model.predict(X)
            proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            return pred, proba
        return None, None