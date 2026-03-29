# app_complete.py - Complete Automatic Feature Selection + 5 Models + Reports

import streamlit as st
import sqlite3
import hashlib
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
import time
import os
from datetime import datetime
import numpy as np
import json
from io import BytesIO, StringIO
import base64
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="CardioAI Pro - Auto Feature Selection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# CUSTOM CSS
# ---------------------------
def get_css():
    return """
    <style>
        * { font-family: 'Segoe UI', system-ui, sans-serif; }
        
        .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
            margin-bottom: 1rem;
        }
        
        .gradient-text {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2rem;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.6rem 1rem;
            font-weight: 600;
            border-radius: 12px;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 1rem;
            color: white;
            text-align: center;
        }
        
        .success-alert {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            padding: 1rem;
            border-radius: 12px;
            color: white;
            text-align: center;
        }
        
        .error-alert {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            padding: 1rem;
            border-radius: 12px;
            color: white;
            text-align: center;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 1.5rem;
        }
        
        .best-model {
            border: 2px solid #10b981;
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border-radius: 15px;
            padding: 1rem;
        }
        
        @media (max-width: 768px) {
            .gradient-text { font-size: 1.5rem !important; }
            .metric-card { padding: 0.5rem !important; }
        }
        
     
        
        .feature-card {
            background: rgba(102, 126, 234, 0.1);
            border-radius: 10px;
            padding: 0.5rem;
            margin: 0.25rem;
            display: inline-block;
        }
    </style>
    """

# ---------------------------
# DATABASE
# ---------------------------
class Database:
    def __init__(self):
        self.db_path = "users.db"
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            email TEXT,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            login_count INTEGER DEFAULT 0
        )
        """)
        
        c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            risk_level TEXT,
            risk_percentage REAL,
            model_used TEXT,
            data TEXT
        )
        """)
        
        conn.commit()
        conn.close()
    
    def hash_pwd(self, pwd):
        return hashlib.sha256((pwd + "salt2024").encode()).hexdigest()
    
    def add_user(self, username, password, email, full_name):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password, email, full_name) VALUES (?,?,?,?)",
                     (username, self.hash_pwd(password), email, full_name))
            conn.commit()
            return True
        except:
            return False
        finally:
            conn.close()
    
    def verify(self, username, password):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        data = c.fetchone()
        conn.close()
        if data and data[0] == self.hash_pwd(password):
            self.update_login(username)
            return True
        return False
    
    def update_login(self, username):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("UPDATE users SET login_count = login_count + 1 WHERE username=?", (username,))
        conn.commit()
        conn.close()
    
    def get_user(self, username):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT full_name, email, login_count FROM users WHERE username=?", (username,))
        data = c.fetchone()
        conn.close()
        return data
    
    def save_pred(self, username, risk_level, risk_percent, model_used, data):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO predictions (username, risk_level, risk_percentage, model_used, data) VALUES (?,?,?,?,?)",
                 (username, risk_level, risk_percent, model_used, json.dumps(data)))
        conn.commit()
        conn.close()

# ---------------------------
# AUTO FEATURE SELECTOR
# ---------------------------
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
        # Separate features and target
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
    
    def get_correlation_features(self, threshold=0.2):
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
        from sklearn.feature_selection import mutual_info_classif
        
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
        rfe = RFE(estimator, n_features_to_select=k)
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
            corr_features = set(self.get_correlation_features(threshold=0.15))
            mi_features = set(self.get_mutual_info_features(k))
            rf_features = set(self.get_random_forest_features(k))
            rfe_features = set(self.get_rfe_features(k))
            
            # Combine all features
            all_features = corr_features | mi_features | rf_features | rfe_features
            
            # Rank features based on frequency
            feature_freq = {}
            for f in all_features:
                freq = 0
                if f in corr_features: freq += 1
                if f in mi_features: freq += 1
                if f in rf_features: freq += 1
                if f in rfe_features: freq += 1
                feature_freq[f] = freq
            
            # Sort by frequency
            selected = sorted(feature_freq.keys(), key=lambda x: feature_freq[x], reverse=True)[:k]
        
        self.selected_features = selected
        return selected

# ---------------------------
# MODEL TRAINER
# ---------------------------
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
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        results = []
        
        for name, model in self.models.items():
            try:
                # Create pipeline
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', model)
                ])
                
                # Train
                start_time = time.time()
                pipeline.fit(self.X_train, self.y_train)
                train_time = time.time() - start_time
                
                # Predict
                y_pred = pipeline.predict(self.X_test)
                
                # Calculate metrics
                acc = accuracy_score(self.y_test, y_pred)
                prec = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                
                # Cross-validation score
                cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
                
                results.append({
                    "Model": name,
                    "Accuracy": f"{acc:.4f}",
                    "Accuracy_Score": acc,
                    "Precision": f"{prec:.4f}",
                    "Recall": f"{rec:.4f}",
                    "F1-Score": f"{f1:.4f}",
                    "CV Score": f"{cv_scores.mean():.4f}",
                    "Training Time": f"{train_time:.2f}s",
                    "Model Object": pipeline
                })
                
                self.trained_models[name] = pipeline
                
            except Exception as e:
                st.error(f"Error training {name}: {str(e)}")
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

# ---------------------------
# REPORT GENERATOR
# ---------------------------
class ReportGenerator:
    def __init__(self):
        pass
    
    def generate_html_report(self, results_df, dataset_info, feature_importance, best_model, predictions):
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CardioAI Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
                .container {{ max-width: 1200px; margin: auto; background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); }}
                h1 {{ color: #667eea; text-align: center; }}
                h2 {{ color: #764ba2; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
                .metric {{ display: inline-block; padding: 10px; margin: 10px; background: #f0f0f0; border-radius: 10px; }}
                .best {{ background: #10b981; color: white; padding: 5px 10px; border-radius: 5px; }}
                .footer {{ text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>❤️ CardioAI - Heart Disease Prediction Report</h1>
                <p style="text-align: center;">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>📊 Dataset Information</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Samples</td><td>{dataset_info.get('samples', 'N/A')}</td></tr>
                    <tr><td>Total Features</td><td>{dataset_info.get('features', 'N/A')}</td></tr>
                    <tr><td>Target Column</td><td>{dataset_info.get('target', 'N/A')}</td></tr>
                    <tr><td>Positive Cases</td><td>{dataset_info.get('positive', 'N/A')}</td></tr>
                    <tr><td>Negative Cases</td><td>{dataset_info.get('negative', 'N/A')}</td></tr>
                </table>
                
                <h2>🤖 Model Performance Comparison</h2>
                {results_df.to_html(index=False)}
                
                <h2>🏆 Best Model</h2>
                <div class="best">
                    <strong>{best_model}</strong> - Highest Accuracy
                </div>
                
                <h2>📈 Feature Importance (Top 10)</h2>
                {feature_importance.head(10).to_html(index=False)}
                
                <h2>💊 Recommendations</h2>
                <ul>
                    <li>Regular exercise (30 minutes daily)</li>
                    <li>Healthy diet with low saturated fats</li>
                    <li>Regular medical check-ups</li>
                    <li>Stress management techniques</li>
                    <li>Adequate sleep (7-8 hours)</li>
                </ul>
                
                <div class="footer">
                    <p>This report is generated by CardioAI Pro - Advanced Heart Disease Prediction System</p>
                    <p>For medical advice, please consult a healthcare professional.</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def generate_pdf_report(self, results_df, dataset_info, feature_importance, best_model, predictions):
        """Generate PDF report using reportlab"""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#667eea'))
            story.append(Paragraph("CardioAI - Heart Disease Prediction Report", title_style))
            story.append(Spacer(1, 20))
            
            # Date
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Dataset Info
            story.append(Paragraph("Dataset Information", styles['Heading2']))
            data = [
                ["Total Samples", str(dataset_info.get('samples', 'N/A'))],
                ["Total Features", str(dataset_info.get('features', 'N/A'))],
                ["Target Column", dataset_info.get('target', 'N/A')],
                ["Positive Cases", str(dataset_info.get('positive', 'N/A'))],
                ["Negative Cases", str(dataset_info.get('negative', 'N/A'))]
            ]
            table = Table(data, colWidths=[2*inch, 3*inch])
            table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.lightgrey)]))
            story.append(table)
            story.append(Spacer(1, 20))
            
            # Best Model
            story.append(Paragraph(f"Best Model: {best_model}", styles['Heading2']))
            
            doc.build(story)
            buffer.seek(0)
            return buffer
        except:
            return None

# ---------------------------
# LOGIN PAGE
# ---------------------------
def login_page():
    st.markdown(get_css(), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 3rem;">❤️</div>
            <h1 class="gradient-text">CardioAI Pro</h1>
            <p>Auto Feature Selection | 5 Models | Smart Predictions</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        db = Database()
        
        with tab1:
            with st.form("login"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if username and password:
                        if db.verify(username, password):
                            st.session_state.auth = True
                            st.session_state.user = username
                            st.success("✅ Login successful!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                    else:
                        st.warning("⚠️ Please fill all fields")
        
        with tab2:
            with st.form("signup"):
                full = st.text_input("Full Name")
                email = st.text_input("Email")
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")
                submit = st.form_submit_button("Sign Up", use_container_width=True)
                
                if submit:
                    if all([full, email, user, pwd, confirm]):
                        if len(pwd) >= 6:
                            if pwd == confirm:
                                if db.add_user(user, pwd, email, full):
                                    st.success("✅ Account created! Please login.")
                                    st.balloons()
                                else:
                                    st.error("❌ Username already exists")
                            else:
                                st.error("❌ Passwords don't match")
                        else:
                            st.warning("⚠️ Password must be at least 6 characters")
                    else:
                        st.warning("⚠️ Please fill all fields")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# MAIN APP
# ---------------------------
def main_app():
    st.markdown(get_css(), unsafe_allow_html=True)
    
    # Session state
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    if "df" not in st.session_state:
        st.session_state.df = None
    if "trainer" not in st.session_state:
        st.session_state.trainer = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = None
    if "feature_importance" not in st.session_state:
        st.session_state.feature_importance = None
    if "target_col" not in st.session_state:
        st.session_state.target_col = None
    
    db = Database()
    user_data = db.get_user(st.session_state.user)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 2rem;">👤</div>
            <h3>{st.session_state.user}</h3>
            <p>{user_data[0] if user_data else 'User'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        pages = {
            "🏠 Dashboard": "Dashboard",
            "📊 Auto Feature Selection": "Feature",
            "🤖 Train Models": "Train",
            "❤️ Predict": "Predict",
            "📈 Model Comparison": "Comparison",
            "📊 Analytics": "Analytics",
            "📜 History": "History",
            "📄 Reports": "Reports",
            "⚙️ Settings": "Settings"
        }
        
        for display, page in pages.items():
            if st.button(display, use_container_width=True):
                st.session_state.page = page
                st.rerun()
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.auth = False
            st.rerun()
    
    # Page routing
    page = st.session_state.page
    
    if page == "Dashboard":
        st.markdown('<h1 class="gradient-text">Dashboard</h1>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h2>5</h2>
                <p>ML Models</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status = "✅" if st.session_state.trainer else "❌"
            st.markdown(f"""
            <div class="metric-card">
                <h2>{status}</h2>
                <p>Models Trained</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{len(st.session_state.get('history', []))}</h2>
                <p>Predictions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            features_count = len(st.session_state.selected_features) if st.session_state.selected_features else 0
            st.markdown(f"""
            <div class="metric-card">
                <h2>{features_count}</h2>
                <p>Features Selected</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Auto Select Features", use_container_width=True):
                st.session_state.page = "Feature"
                st.rerun()
        with col2:
            if st.button("🚀 Train Models", use_container_width=True):
                st.session_state.page = "Train"
                st.rerun()
        
        st.markdown("---")
        
        st.info("""
        **CardioAI Pro - Advanced Heart Disease Prediction System**
        
        **Workflow:**
        1. 📁 **Upload Dataset** - Upload CSV file with heart disease data
        2. 🔍 **Auto Feature Selection** - Automatically select best features
        3. 🤖 **Train 5 Models** - Train all models simultaneously
        4. ❤️ **Make Predictions** - Get instant risk assessment
        5. 📄 **Generate Reports** - Download professional reports
        
        **Features:**
        - Automatic target column detection
        - Multiple feature selection methods
        - 5 ML models comparison
        - Real-time predictions
        - PDF & HTML reports
        - Mobile responsive design
        """)
    
    elif page == "Feature":
        st.markdown('<h1 class="gradient-text">Auto Feature Selection</h1>', unsafe_allow_html=True)
        
        # Dataset upload
        with st.expander("📁 Upload Dataset", expanded=st.session_state.df is None):
            uploaded = st.file_uploader("Choose CSV Dataset", type=["csv"])
            
            if uploaded:
                df = pd.read_csv(uploaded)
                st.session_state.df = df
                st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                st.dataframe(df.head(), use_container_width=True)
                
                # Automatic target column detection
                st.subheader("🎯 Target Column Detection")
                
                possible_targets = []
                for col in df.columns:
                    col_lower = col.lower()
                    if any(word in col_lower for word in ['heart', 'disease', 'target', 'class', 'label', 'hd']):
                        possible_targets.append(col)
                
                if possible_targets:
                    st.info(f"🔍 Detected possible target columns: {', '.join(possible_targets)}")
                    
                    # Manual or automatic selection
                    selection_mode = st.radio("Select Target Column Mode:", ["Auto Detect", "Manual Select"])
                    
                    if selection_mode == "Auto Detect":
                        target_col = possible_targets[0]
                        st.success(f"✅ Auto selected: **{target_col}**")
                    else:
                        target_col = st.selectbox("Select Target Column", df.columns.tolist())
                    
                    st.session_state.target_col = target_col
                    
                    # Show target distribution
                    st.subheader("Target Distribution")
                    target_counts = df[target_col].value_counts()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(target_counts)
                    with col2:
                        fig = px.pie(values=target_counts.values, names=target_counts.index, title="Class Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature selection method
                    st.subheader("🔧 Feature Selection Method")
                    
                    method = st.selectbox(
                        "Select Method",
                        ["Auto (Combine All)", "Correlation", "Mutual Information", "Random Forest Importance", "RFE"]
                    )
                    
                    k_features = st.slider("Number of Features to Select", 3, min(20, len(df.columns)-1), 10)
                    
                    if st.button("🚀 Run Feature Selection", use_container_width=True):
                        with st.spinner("Analyzing features..."):
                            selector = AutoFeatureSelector(df, target_col)
                            selector.prepare_data()
                            
                            method_map = {
                                "Auto (Combine All)": "auto",
                                "Correlation": "correlation",
                                "Mutual Information": "mutual_info",
                                "Random Forest Importance": "random_forest",
                                "RFE": "rfe"
                            }
                            
                            selected = selector.auto_select_features(method_map[method], k_features)
                            st.session_state.selected_features = selected
                            st.session_state.feature_importance = selector.feature_importances
                            
                            st.success(f"✅ Selected {len(selected)} features")
                            
                            # Display selected features
                            st.subheader("📋 Selected Features")
                            cols = st.columns(4)
                            for i, feat in enumerate(selected):
                                with cols[i % 4]:
                                    st.markdown(f"✅ {feat}")
                            
                            # Feature importance chart
                            if selector.feature_importances is not None:
                                st.subheader("📊 Feature Importance")
                                fig = px.bar(selector.feature_importances.head(15), 
                                           x="Importance" if "Importance" in selector.feature_importances.columns else "MI_Score",
                                           y="Feature", orientation='h',
                                           title="Top 15 Features by Importance")
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ No target column detected! Please manually select one.")
                    target_col = st.selectbox("Select Target Column", df.columns.tolist())
                    st.session_state.target_col = target_col
    
    elif page == "Train":
        st.markdown('<h1 class="gradient-text">Train 5 Models</h1>', unsafe_allow_html=True)
        
        if st.session_state.df is None:
            st.warning("⚠️ Please upload a dataset first!")
            if st.button("📁 Go to Upload", use_container_width=True):
                st.session_state.page = "Feature"
                st.rerun()
        elif st.session_state.selected_features is None:
            st.warning("⚠️ Please run feature selection first!")
            if st.button("🔍 Go to Feature Selection", use_container_width=True):
                st.session_state.page = "Feature"
                st.rerun()
        else:
            st.success(f"✅ Dataset: {st.session_state.df.shape[0]} rows")
            st.success(f"✅ Selected Features: {len(st.session_state.selected_features)}")
            st.success(f"✅ Target Column: {st.session_state.target_col}")
            
            # Display selected features
            st.subheader("Features to Use")
            cols = st.columns(5)
            for i, feat in enumerate(st.session_state.selected_features):
                with cols[i % 5]:
                    st.markdown(f"✅ {feat}")
            
            # Training options
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            
            if st.button("🚀 Train All 5 Models", use_container_width=True):
                with st.spinner("Training models... This may take a moment..."):
                    # Prepare data
                    df = st.session_state.df
                    target = st.session_state.target_col
                    features = st.session_state.selected_features
                    
                    X = df[features].copy()
                    y = df[target].copy()
                    
                    # Handle categorical features
                    for col in X.columns:
                        if X[col].dtype == 'object':
                            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                    
                    # Handle missing values
                    X = X.fillna(X.median())
                    
                    # Train models
                    trainer = ModelTrainer()
                    results = trainer.train_all(X, y, test_size)
                    
                    st.session_state.trainer = trainer
                    st.session_state.results = results
                    
                    st.success("✅ All 5 models trained successfully!")
                    st.balloons()
                    
                    # Display results
                    st.subheader("📊 Training Results")
                    display_df = results.drop(columns=['Model Object', 'Accuracy_Score'])
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Best model
                    best_model = results.iloc[0]["Model"]
                    best_acc = results.iloc[0]["Accuracy"]
                    st.markdown(f"""
                    <div class="best-model">
                        <h3>🏆 Best Model: {best_model}</h3>
                        <p>Accuracy: {best_acc}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    elif page == "Predict":
        st.markdown('<h1 class="gradient-text">Heart Disease Prediction</h1>', unsafe_allow_html=True)
        
        if st.session_state.trainer is None:
            st.warning("⚠️ No models trained! Please train models first.")
            if st.button("🚀 Go to Training", use_container_width=True):
                st.session_state.page = "Train"
                st.rerun()
        elif st.session_state.selected_features is None:
            st.warning("⚠️ No features selected! Please run feature selection first.")
            if st.button("🔍 Go to Feature Selection", use_container_width=True):
                st.session_state.page = "Feature"
                st.rerun()
        else:
            # Get best model
            best_name, best_model = st.session_state.trainer.get_best_model()
            st.success(f"🎯 Using Best Model: **{best_name}**")
            
            # Input form
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Patient Information")
            
            # Create input fields for selected features
            cols = st.columns(2)
            inputs = {}
            
            # Get sample data for defaults
            df = st.session_state.df
            features = st.session_state.selected_features
            
            for i, feat in enumerate(features[:12]):  # Limit to 12 features for better UI
                with cols[i % 2]:
                    default_val = float(df[feat].median()) if df[feat].dtype in ['int64', 'float64'] else 0
                    inputs[feat] = st.number_input(
                        feat.replace("_", " ").title(),
                        value=default_val,
                        format="%.2f"
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Model selection for prediction
            model_options = list(st.session_state.trainer.models.keys())
            selected_model = st.selectbox("Select Model for Prediction", model_options, index=model_options.index(best_name))
            
            if st.button("🔍 Analyze Risk", use_container_width=True):
                with st.spinner(f"Analyzing with {selected_model}..."):
                    # Prepare input
                    df_input = pd.DataFrame([inputs])
                    
                    # Ensure all features exist
                    for feat in features:
                        if feat not in df_input.columns:
                            df_input[feat] = 0
                    
                    df_input = df_input[features]
                    
                    # Handle categorical
                    for col in df_input.columns:
                        if df_input[col].dtype == 'object':
                            df_input[col] = LabelEncoder().fit_transform(df_input[col].astype(str))
                    
                    df_input = df_input.fillna(0)
                    
                    # Predict
                    pred, proba = st.session_state.trainer.predict(selected_model, df_input)
                    
                    if pred is not None:
                        risk_percent = proba[0][1] * 100 if proba is not None else 0
                        risk_level = "High Risk" if pred[0] == 1 else "Low Risk"
                        
                        # Save to database
                        db.save_pred(st.session_state.user, risk_level, risk_percent, selected_model, inputs)
                        
                        # Save to session
                        if "history" not in st.session_state:
                            st.session_state.history = []
                        
                        st.session_state.history.insert(0, {
                            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "Model": selected_model,
                            "Risk": risk_level,
                            "Probability": f"{risk_percent:.1f}%",
                            **inputs
                        })
                        
                        # Display result
                        if pred[0] == 1:
                            st.markdown(f"""
                            <div class="error-alert">
                                <h2>⚠️ {risk_level}</h2>
                                <p style="font-size: 20px;">Risk Probability: {risk_percent:.1f}%</p>
                                <p>Model: {selected_model}</p>
                                <p>Please consult a healthcare professional immediately!</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="success-alert">
                                <h2>✅ {risk_level}</h2>
                                <p style="font-size: 20px;">Health Probability: {100-risk_percent:.1f}%</p>
                                <p>Model: {selected_model}</p>
                                <p>Keep up healthy habits!</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Recommendations
                        with st.expander("📋 Detailed Recommendations"):
                            if pred[0] == 1:
                                st.markdown("""
                                ### 🏥 Medical Recommendations
                                - **Immediate Actions:** Schedule a cardiologist appointment
                                - **Lifestyle Changes:** 
                                    - 30 minutes of moderate exercise daily
                                    - Reduce salt and saturated fat intake
                                    - Quit smoking if applicable
                                - **Monitoring:** Check blood pressure weekly
                                - **Medication:** Take prescribed medicines regularly
                                """)
                            else:
                                st.markdown("""
                                ### 💪 Preventive Recommendations
                                - **Exercise:** 150 minutes of moderate activity per week
                                - **Diet:** Mediterranean diet rich in fruits, vegetables, whole grains
                                - **Sleep:** 7-8 hours of quality sleep
                                - **Stress Management:** Practice meditation or yoga
                                - **Regular Checkups:** Annual health screening
                                """)
    
    elif page == "Comparison":
        st.markdown('<h1 class="gradient-text">Model Comparison</h1>', unsafe_allow_html=True)
        
        if st.session_state.results is not None:
            results_df = st.session_state.results.copy()
            display_df = results_df.drop(columns=['Model Object', 'Accuracy_Score'])
            st.dataframe(display_df, use_container_width=True)
            
            # Best model highlight
            best_model = results_df.iloc[0]["Model"]
            best_acc = results_df.iloc[0]["Accuracy"]
            st.markdown(f"""
            <div class="best-model">
                <h3>🏆 Best Performing Model: {best_model}</h3>
                <p>Accuracy: {best_acc} | F1-Score: {results_df.iloc[0]['F1-Score']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(results_df, x="Model", y="Accuracy_Score", 
                           title="Model Accuracy Comparison",
                           color="Model", text="Accuracy")
                fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                metrics = results_df.melt(id_vars=["Model"], 
                                         value_vars=["Accuracy_Score", "Precision", "Recall", "F1-Score"],
                                         var_name="Metric", value_name="Score")
                metrics['Score'] = pd.to_numeric(metrics['Score'], errors='coerce')
                metrics = metrics.dropna()
                
                fig = px.line_polar(metrics, r="Score", theta="Metric", 
                                   color="Model", line_close=True,
                                   title="Model Performance Radar")
                st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrices
            if st.session_state.trainer and hasattr(st.session_state.trainer, 'X_test'):
                st.subheader("Confusion Matrices")
                cols = st.columns(3)
                model_list = list(st.session_state.trainer.models.keys())
                
                for idx, model_name in enumerate(model_list[:3]):
                    if model_name in st.session_state.trainer.trained_models:
                        with cols[idx]:
                            model = st.session_state.trainer.trained_models[model_name]
                            y_pred = model.predict(st.session_state.trainer.X_test)
                            cm = confusion_matrix(st.session_state.trainer.y_test, y_pred)
                            fig_cm = px.imshow(cm, text_auto=True, title=model_name)
                            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning("⚠️ No models trained yet! Please train models first.")
            if st.button("🚀 Go to Training", use_container_width=True):
                st.session_state.page = "Train"
                st.rerun()
    
    elif page == "Analytics":
        st.markdown('<h1 class="gradient-text">Analytics Dashboard</h1>', unsafe_allow_html=True)
        
        if st.session_state.df is not None:
            df = st.session_state.df
            target = st.session_state.target_col
            
            if target and target in df.columns:
                # Distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(df, names=target, title="Target Distribution", hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Statistics")
                    st.write(f"**Total:** {len(df)}")
                    st.write(f"**Positive:** {df[target].sum()}")
                    st.write(f"**Negative:** {len(df) - df[target].sum()}")
                    st.write(f"**Balance:** {(df[target].sum() / len(df) * 100):.1f}% Positive")
                
                # Correlation
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    st.subheader("Correlation Matrix")
                    corr = numeric_df.corr()
                    fig = px.imshow(corr, text_auto=True, aspect="auto",
                                   color_continuous_scale="RdBu")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Summary
                st.subheader("Summary Statistics")
                st.dataframe(df.describe(), use_container_width=True)
        else:
            st.info("📊 No dataset loaded. Please upload a dataset first.")
    
    elif page == "History":
        st.markdown('<h1 class="gradient-text">Prediction History</h1>', unsafe_allow_html=True)
        
        if "history" in st.session_state and st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Export to CSV", use_container_width=True):
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="history.csv">Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            with col2:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.history = []
                    st.rerun()
        else:
            st.info("📝 No predictions yet. Make your first prediction!")
    
    elif page == "Reports":
        st.markdown('<h1 class="gradient-text">Generate Reports</h1>', unsafe_allow_html=True)
        
        if st.session_state.results is not None:
            # Report options
            report_type = st.selectbox("Report Type", ["HTML Report", "CSV Export", "JSON Export"])
            
            # Dataset info
            dataset_info = {
                'samples': len(st.session_state.df) if st.session_state.df is not None else 0,
                'features': len(st.session_state.selected_features) if st.session_state.selected_features else 0,
                'target': st.session_state.target_col if st.session_state.target_col else 'N/A',
                'positive': st.session_state.df[st.session_state.target_col].sum() if st.session_state.df is not None and st.session_state.target_col else 0,
                'negative': len(st.session_state.df) - st.session_state.df[st.session_state.target_col].sum() if st.session_state.df is not None and st.session_state.target_col else 0
            }
            
            # Feature importance
            if st.session_state.feature_importance is not None:
                feature_imp = st.session_state.feature_importance
            else:
                feature_imp = pd.DataFrame({'Feature': st.session_state.selected_features or [], 'Importance': [1/len(st.session_state.selected_features)] * len(st.session_state.selected_features or [])})
            
            # Best model
            best_model = st.session_state.results.iloc[0]["Model"] if len(st.session_state.results) > 0 else "N/A"
            
            if st.button("📄 Generate Report", use_container_width=True):
                with st.spinner("Generating report..."):
                    report_gen = ReportGenerator()
                    
                    if report_type == "HTML Report":
                        html = report_gen.generate_html_report(
                            st.session_state.results.drop(columns=['Model Object', 'Accuracy_Score'], errors='ignore'),
                            dataset_info, feature_imp, best_model, []
                        )
                        b64 = base64.b64encode(html.encode()).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="cardioai_report.html">Download HTML Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        st.success("Report generated!")
                    
                    elif report_type == "CSV Export":
                        csv = st.session_state.results.drop(columns=['Model Object', 'Accuracy_Score'], errors='ignore').to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="model_comparison.csv">Download CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                    elif report_type == "JSON Export":
                        json_data = st.session_state.results.drop(columns=['Model Object', 'Accuracy_Score'], errors='ignore').to_json(orient='records', indent=2)
                        b64 = base64.b64encode(json_data.encode()).decode()
                        href = f'<a href="data:file/json;base64,{b64}" download="model_comparison.json">Download JSON</a>'
                        st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No models trained yet! Please train models first.")
            if st.button("🚀 Go to Training", use_container_width=True):
                st.session_state.page = "Train"
                st.rerun()
    
    elif page == "Settings":
        st.markdown('<h1 class="gradient-text">Settings</h1>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        user_data = db.get_user(st.session_state.user)
        if user_data:
            st.write(f"**👤 Name:** {user_data[0]}")
            st.write(f"**📧 Email:** {user_data[1]}")
            st.write(f"**🔑 Total Logins:** {user_data[2]}")
        
        st.markdown("---")
        
        st.subheader("System Information")
        st.write("**Version:** 4.0 Pro")
        st.write("**Models:** 5 Advanced ML Models")
        st.write("**Feature Selection:** Auto + 4 Methods")
        st.write("**Reports:** HTML, CSV, JSON")
        
        st.markdown("---")
        
        st.subheader("Feature Selection Methods")
        st.markdown("""
        - **Correlation:** Selects features highly correlated with target
        - **Mutual Information:** Measures mutual dependence between features and target
        - **Random Forest:** Uses feature importance from Random Forest
        - **RFE:** Recursive Feature Elimination
        - **Auto:** Combines all methods for best results
        """)
        
        st.markdown("---")
        
        if st.button("🔄 Reset All Data", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# RUN APP
# ---------------------------
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    login_page()
else:
    main_app()