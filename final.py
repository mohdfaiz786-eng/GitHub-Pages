import streamlit as st
import sqlite3
import hashlib
import pandas as pd
import joblib
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# DATABASE
# ---------------------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT UNIQUE,
        password TEXT
    )
    """)
    conn.commit()
    conn.close()

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

def add_user(u, p):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?,?)", (u, hash_password(p)))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def get_user(u):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (u,))
    data = c.fetchone()
    conn.close()
    return data

# ---------------------------
# INIT
# ---------------------------
init_db()

if "auth" not in st.session_state:
    st.session_state.auth = False
if "user" not in st.session_state:
    st.session_state.user = None

st.set_page_config(page_title="Heart Predictor", layout="wide")

# ---------------------------
# LOGIN
# ---------------------------
def login():
    st.title("🔐 Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        user = get_user(u)
        if user and user[0] == hash_password(p):
            st.session_state.auth = True
            st.session_state.user = u
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid Credentials")

    st.markdown("---")

    st.subheader("Signup")
    new_u = st.text_input("New Username")
    new_p = st.text_input("New Password", type="password")

    if st.button("Create Account"):
        if add_user(new_u, new_p):
            st.success("Account Created")
        else:
            st.error("User already exists")

# ---------------------------
# AUTH CHECK
# ---------------------------
if not st.session_state.auth:
    login()
    st.stop()

# ---------------------------
# MAIN APP
# ---------------------------
st.title("❤️ Heart Disease Prediction System")
st.write(f"Welcome **{st.session_state.user}**")

if st.button("Logout"):
    st.session_state.auth = False
    st.rerun()

st.markdown("---")

# ---------------------------
# DATA UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    if "HeartDisease" not in df.columns:
        st.error("❌ 'HeartDisease' column missing")
        st.stop()

    # Encoding categorical
    X = pd.get_dummies(df.drop("HeartDisease", axis=1))
    y = df["HeartDisease"]

    # ---------------------------
    # FEATURE IMPORTANCE
    # ---------------------------
    rf = RandomForestClassifier()
    rf.fit(X, y)

    imp_df = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    st.subheader("Feature Importance")
    st.dataframe(imp_df)

    top_k = st.slider("Select Top Features", 3, len(X.columns), 8)
    selected_features = imp_df["feature"].head(top_k).tolist()

    st.write("Selected Features:", selected_features)

    # ---------------------------
    # TRAIN MODEL
    # ---------------------------
    if st.button("Train Model"):

        X_sel = X[selected_features]

        X_train, X_test, y_train, y_test = train_test_split(
            X_sel, y, test_size=0.25, random_state=42
        )

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier())
        ])

        model.fit(X_train, y_train)

        defaults = X_sel.median().to_dict()

        joblib.dump((model, selected_features, defaults), "model.joblib")

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"✅ Model Trained | Accuracy: {acc:.2f}")

        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True)
        st.plotly_chart(fig)

# ---------------------------
# PREDICTION
# ---------------------------
st.markdown("---")
st.header("Prediction")

try:
    model, features, defaults = joblib.load("model.joblib")
except:
    st.warning("⚠️ Train model first")
    st.stop()

mode = st.radio("Input Mode", ["Manual", "Auto Fill"])

inputs = {}
cols = st.columns(2)

for i, col in enumerate(features):
    with cols[i % 2]:
        if mode == "Auto Fill":
            inputs[col] = st.number_input(col, value=float(defaults[col]))
        else:
            inputs[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    df_input = pd.DataFrame([inputs])

    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1] * 100

    if pred == 1:
        st.error(f"❤️ High Risk ({prob:.2f}%)")
    else:
        st.success(f"💚 Low Risk ({100-prob:.2f}%)")