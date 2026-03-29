# app.py
# Complete Streamlit app with Signup/Login/CAPTCHA/Roles + ML automatic feature selection

import streamlit as st
import sqlite3
import hashlib
import random
import string
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

# ----------------------------
# Config / Paths
# ----------------------------
DB_PATH = "users.db"
MODEL_PATH = "heart_model.joblib"

# ----------------------------
# Database helpers
# ----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username: str, password: str, role: str = "user") -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                  (username, hash_password(password), role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(username: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password, role FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    return row

# ----------------------------
# CAPTCHA helper
# ----------------------------
def generate_captcha(length: int = 5) -> str:
    choices = string.ascii_uppercase + string.digits
    return "".join(random.choice(choices) for _ in range(length))

# ----------------------------
# Initialize
# ----------------------------
init_db()

if "auth" not in st.session_state:
    st.session_state["auth"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None
if "show_signup" not in st.session_state:
    st.session_state["show_signup"] = False
if "captcha" not in st.session_state:
    st.session_state["captcha"] = generate_captcha()

st.set_page_config(page_title="Heart Predictor", layout="wide")

# ----------------------------
# AUTH UI
# ----------------------------
if not st.session_state["auth"]:
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    captcha_val = st.session_state["captcha"]
    st.markdown(f"### CAPTCHA: {captcha_val}")
    captcha_input = st.text_input("Enter CAPTCHA")

    if st.button("Login"):
        record = get_user(username)
        if record and record[0] == hash_password(password) and captcha_input == captcha_val:
            st.session_state["auth"] = True
            st.session_state["user"] = username
            st.session_state["role"] = record[1]
            st.rerun()
        else:
            st.error("Login failed")

    if st.button("Signup"):
        add_user(username, password)
        st.success("User created")

    st.stop()

# ----------------------------
# MAIN APP
# ----------------------------
st.title("❤️ Heart Disease Predictor")

# ✅ REQUIRED COLS FIX
REQUIRED_COLS = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak","slope","target"
]

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ✅ DATASET FIX (correct place)
    df.columns = [c.strip() for c in df.columns]

    df = df.rename(columns={
        "Age": "age",
        "Sex": "sex",
        "ChestPainType": "cp",
        "RestingBP": "trestbps",
        "Cholesterol": "chol",
        "FastingBS": "fbs",
        "RestingECG": "restecg",
        "MaxHR": "thalach",
        "ExerciseAngina": "exang",
        "Oldpeak": "oldpeak",
        "ST_Slope": "slope",
        "HeartDisease": "target"
    })

    # encode
    df["sex"] = df["sex"].map({'M':1, 'F':0})
    df["exang"] = df["exang"].map({'Y':1, 'N':0})
    df["cp"] = df["cp"].astype("category").cat.codes
    df["restecg"] = df["restecg"].astype("category").cat.codes
    df["slope"] = df["slope"].astype("category").cat.codes

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    st.write(df.head())

    X = df.drop("target", axis=1)
    y = df["target"]

    if st.button("Train Model"):

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        model = RandomForestClassifier()
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        st.write("Accuracy:", accuracy_score(y_test, y_pred))

        joblib.dump(pipeline, MODEL_PATH)

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Predict"):
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([np.zeros(len(REQUIRED_COLS)-1)], columns=REQUIRED_COLS[:-1])

    pred = model.predict(sample)[0]

    st.write("Prediction:", pred)