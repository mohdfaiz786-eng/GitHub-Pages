# api.py - FastAPI Server for Model Predictions
# Run separately: uvicorn api:app --reload --port 8000

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------
# FASTAPI APP INIT
# ---------------------------
app = FastAPI(
    title="CardioAI Pro API",
    description="Heart Disease Prediction REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - Allow all origins (for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# REQUEST MODELS (Pydantic)
# ---------------------------

class PredictionRequest(BaseModel):
    """Single prediction request"""
    age: float = Field(..., description="Age in years", ge=0, le=120)
    sex: float = Field(..., description="Sex (0=female, 1=male)", ge=0, le=1)
    cp: Optional[float] = Field(None, description="Chest pain type (0-3)")
    trestbps: Optional[float] = Field(None, description="Resting blood pressure")
    chol: Optional[float] = Field(None, description="Serum cholesterol")
    fbs: Optional[float] = Field(None, description="Fasting blood sugar > 120 mg/dl")
    restecg: Optional[float] = Field(None, description="Resting ECG results")
    thalach: Optional[float] = Field(None, description="Maximum heart rate achieved")
    exang: Optional[float] = Field(None, description="Exercise induced angina")
    oldpeak: Optional[float] = Field(None, description="ST depression induced by exercise")
    slope: Optional[float] = Field(None, description="Slope of peak exercise ST segment")
    ca: Optional[float] = Field(None, description="Number of major vessels")
    thal: Optional[float] = Field(None, description="Thalassemia")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 55,
                "sex": 1,
                "trestbps": 130,
                "chol": 220,
                "thalach": 138,
                "oldpeak": 0.6
            }
        }

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    patients: List[PredictionRequest] = Field(..., description="List of patients")

class PredictionResponse(BaseModel):
    """Prediction response"""
    status: str
    risk_level: str
    risk_percentage: float
    prediction: int
    model_used: str
    timestamp: str
    recommendations: List[str]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    status: str
    total_patients: int
    high_risk_count: int
    low_risk_count: int
    predictions: List[dict]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str
    timestamp: str

# ---------------------------
# GLOBAL VARIABLES
# ---------------------------
model = None
features = None
scaler = None
model_name = None

# ---------------------------
# LOAD MODEL FUNCTION (FIXED)
# ---------------------------
def load_model():
    """Load trained model from file"""
    global model, features, scaler, model_name
    
    try:
        # Check current directory for model file
        model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
        
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            model_data = joblib.load(model_path)
            
            # Handle different save formats
            if isinstance(model_data, dict):
                model = model_data.get("model")
                features = model_data.get("features")
                scaler = model_data.get("scaler")
                model_name = model_data.get("model_name", "CardioAI Model")
                print(f"Model loaded successfully: {model_name}")
                print(f"Features: {features if features else 'Not specified'}")
            else:
                model = model_data
                model_name = "CardioAI Model"
                print("Model loaded successfully")
            
            return True
        else:
            print(f"Model file not found at: {model_path}")
            print("Please train model first in Streamlit app")
            return False
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def make_prediction(data: Dict) -> Dict:
    """Make single prediction"""
    global model, features
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train model first.")
    
    try:
        # Convert to DataFrame
        df_input = pd.DataFrame([data])
        
        # Use features if available
        if features and len(features) > 0:
            # Ensure all features exist
            for feat in features:
                if feat not in df_input.columns:
                    df_input[feat] = 0
            df_input = df_input[features]
        else:
            # If no features defined, use only available columns
            pass
        
        # Make prediction
        prediction = model.predict(df_input)[0]
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df_input)[0]
            risk_percent = proba[1] * 100 if len(proba) > 1 else proba[0] * 100
        else:
            risk_percent = 50.0
        
        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        
        # Recommendations based on risk
        if prediction == 1:
            recommendations = [
                "Consult a cardiologist immediately",
                "Schedule ECG and stress test",
                "Start medication as prescribed",
                "Monitor blood pressure daily",
                "Reduce salt and saturated fat intake",
                "Begin light exercise under supervision"
            ]
        else:
            recommendations = [
                "Maintain regular health checkups",
                "Exercise 30 minutes daily",
                "Eat balanced diet with fruits and vegetables",
                "Manage stress through meditation",
                "Get 7-8 hours of quality sleep",
                "Annual health screening recommended"
            ]
        
        return {
            "status": "success",
            "risk_level": risk_level,
            "risk_percentage": round(risk_percent, 2),
            "prediction": int(prediction),
            "model_used": model_name,
            "timestamp": datetime.now().isoformat(),
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ---------------------------
# API ENDPOINTS
# ---------------------------

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - Health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint"""
    return make_prediction(request.dict())

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    results = []
    high_risk = 0
    low_risk = 0
    
    for patient in request.patients:
        try:
            result = make_prediction(patient.dict())
            results.append(result)
            if result["risk_level"] == "High Risk":
                high_risk += 1
            else:
                low_risk += 1
        except Exception as e:
            results.append({
                "status": "error",
                "error": str(e)
            })
    
    return {
        "status": "success",
        "total_patients": len(request.patients),
        "high_risk_count": high_risk,
        "low_risk_count": low_risk,
        "predictions": results
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_name": model_name,
        "features": features if features else [],
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

# ---------------------------
# LOAD MODEL ON STARTUP
# ---------------------------
@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    print("=" * 50)
    print("Starting CardioAI API Server...")
    load_model()
    print(f"Model loaded: {model_name if model_name else 'Unknown'}")
    print("API available at: http://localhost:8000")
    print("Docs available at: http://localhost:8000/docs")
    print("=" * 50)

# ---------------------------
# RUN SERVER
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)