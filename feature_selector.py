# feature_selector.py - Auto Feature Selection

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.preprocessing import LabelEncoder

class AutoFeatureSelector:
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col
        self.X = None
        self.y = None
        self.feature_importances = None
        self.selected_features = None
        
    def prepare_data(self):
        """Prepare data for feature selection"""
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.median())
        
        self.X = X
        self.y = y
        return X, y
    
    def get_correlation_features(self, threshold=0.1):
        """Get features based on correlation with target"""
        if self.X is None:
            self.prepare_data()
        
        correlations = []
        for col in self.X.columns:
            corr = abs(self.X[col].corr(self.y))
            correlations.append((col, corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected = [col for col, corr in correlations if corr >= threshold]
        
        return selected
    
    def get_mutual_info_features(self, k=10):
        """Get features based on mutual information"""
        if self.X is None:
            self.prepare_data()
        
        mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
        mi_df = pd.DataFrame({
            'Feature': self.X.columns,
            'MI_Score': mi_scores
        }).sort_values('MI_Score', ascending=False)
        
        selected = mi_df.head(k)['Feature'].tolist()
        self.feature_importances = mi_df
        
        return selected
    
    def get_random_forest_features(self, k=10):
        """Get features using Random Forest importance"""
        if self.X is None:
            self.prepare_data()
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X, self.y)
        
        importance_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        selected = importance_df.head(k)['Feature'].tolist()
        self.feature_importances = importance_df
        
        return selected
    
    def get_rfe_features(self, k=10):
        """Get features using RFE"""
        if self.X is None:
            self.prepare_data()
        
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        rfe = RFE(estimator, n_features_to_select=min(k, len(self.X.columns)))
        rfe.fit(self.X, self.y)
        
        selected = self.X.columns[rfe.support_].tolist()
        
        return selected
    
    def auto_select_features(self, method='auto', k=10):
        """Auto select features based on best method"""
        if method == 'correlation':
            selected = self.get_correlation_features()
        elif method == 'mutual_info':
            selected = self.get_mutual_info_features(k)
        elif method == 'random_forest':
            selected = self.get_random_forest_features(k)
        elif method == 'rfe':
            selected = self.get_rfe_features(k)
        else:  # auto - use all methods and combine
            corr_features = set(self.get_correlation_features(threshold=0.1))
            mi_features = set(self.get_mutual_info_features(k))
            rf_features = set(self.get_random_forest_features(k))
            rfe_features = set(self.get_rfe_features(k))
            
            all_features = corr_features | mi_features | rf_features | rfe_features
            
            feature_freq = {}
            for f in all_features:
                freq = 0
                if f in corr_features: freq += 1
                if f in mi_features: freq += 1
                if f in rf_features: freq += 1
                if f in rfe_features: freq += 1
                feature_freq[f] = freq
            
            selected = sorted(feature_freq.keys(), 
                            key=lambda x: feature_freq[x], 
                            reverse=True)[:k]
        
        self.selected_features = selected
        return selected