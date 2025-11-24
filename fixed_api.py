#!/usr/bin/env python3
"""
Fixed FastAPI server that uses the same feature engineering as simple_test.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json
from loguru import logger
from pathlib import Path
import joblib
import sys

# Add src to path
sys.path.append('src')

# Initialize FastAPI app
app = FastAPI(
    title="PharmaTrail-X Analytics API - FIXED",
    description="Clinical Trial Delay Prediction with Proper Feature Engineering",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
delay_model = None
anomaly_model = None

# Pydantic models
class PredictionRequest(BaseModel):
    trial_data: Dict[str, Any]
    include_anomaly_detection: bool = True

class PredictionResponse(BaseModel):
    delay_probability: float
    delay_prediction: str
    confidence: float
    anomaly_score: Optional[float] = None
    anomaly_detected: Optional[bool] = None
    risk_factors: List[str]

def create_proper_features(trial_data: Dict[str, Any]) -> np.ndarray:
    """
    Create features using the same logic as simple_test.py
    """
    # Extract values with defaults
    age = trial_data.get('Age', 65)
    bmi = trial_data.get('BMI', 27)
    adherence = trial_data.get('Medication_Adherence', 80)
    adr_rate = trial_data.get('ADR_Rate', 0.1)
    efficacy = trial_data.get('Efficacy_Score', 50)
    bp_sys = trial_data.get('BP_Systolic', 130)
    bp_dia = trial_data.get('BP_Diastolic', 80)
    alt = trial_data.get('ALT_Level', 25)
    ast = trial_data.get('AST_Level', 25)
    creatinine = trial_data.get('Creatinine', 1.0)
    satisfaction = trial_data.get('Satisfaction_Score', 8)
    week = trial_data.get('Week', 4)
    
    # Basic time features
    visit_month = 2
    visit_quarter = 1
    visit_day_of_week = 1
    
    # Safety features (same logic as simple_test.py)
    alt_elevated = 1 if alt > 40 else 0
    ast_elevated = 1 if ast > 40 else 0
    creatinine_elevated = 1 if creatinine > 1.2 else 0
    hypertension_risk = 1 if (bp_sys > 140 or bp_dia > 90) else 0
    
    # Safety risk score (same calculation as simple_test.py)
    adr_severity = 2 if adr_rate > 0.3 else 1 if adr_rate > 0.1 else 0
    safety_risk_score = (
        adr_severity * adr_rate * 0.4 +
        (alt_elevated + ast_elevated + creatinine_elevated) * 0.3 +
        hypertension_risk * 0.3
    )
    
    # Patient aggregations (simulate patient-level stats)
    patient_avg_adherence = adherence
    patient_avg_adr = adr_rate
    patient_avg_efficacy = efficacy
    patient_avg_safety = safety_risk_score
    
    # Create feature vector matching simple_test.py exactly
    features = [
        age,                           # Age
        bmi,                          # BMI
        0.5,                          # Risk_Score (default)
        bp_sys,                       # BP_Systolic
        bp_dia,                       # BP_Diastolic
        alt,                          # ALT_Level
        ast,                          # AST_Level
        creatinine,                   # Creatinine
        efficacy,                     # Efficacy_Score
        adherence,                    # Medication_Adherence(%)
        satisfaction,                 # Satisfaction_Score
        week,                         # Week
        visit_month,                  # visit_month
        visit_quarter,                # visit_quarter
        visit_day_of_week,           # visit_day_of_week
        alt_elevated,                # alt_elevated
        ast_elevated,                # ast_elevated
        creatinine_elevated,         # creatinine_elevated
        hypertension_risk,           # hypertension_risk
        safety_risk_score,           # safety_risk_score
        patient_avg_adherence,       # patient_avg_Medication_Adherence(%)
        patient_avg_adr,             # patient_avg_ADR_Reported
        patient_avg_efficacy,        # patient_avg_Efficacy_Score
        patient_avg_safety           # patient_avg_safety_risk_score
    ]
    
    return np.array(features).reshape(1, -1)

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global delay_model, anomaly_model
    
    logger.info("Starting PharmaTrail-X Analytics API - FIXED VERSION")
    
    # Try to load existing models
    try:
        model_path = Path("models")
        if model_path.exists():
            delay_model_file = model_path / "delay_model.pkl"
            anomaly_model_file = model_path / "anomaly_model.pkl"
            
            if delay_model_file.exists():
                delay_model = joblib.load(delay_model_file)
                logger.info("✅ Loaded delay prediction model")
            
            if anomaly_model_file.exists():
                anomaly_model = joblib.load(anomaly_model_file)
                logger.info("✅ Loaded anomaly detection model")
                
    except Exception as e:
        logger.warning(f"Could not load existing models: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PharmaTrail-X Analytics API - FIXED VERSION",
        "version": "1.1.0",
        "status": "running",
        "fix": "Now uses proper feature engineering pipeline",
        "models_loaded": {
            "delay_prediction": delay_model is not None,
            "anomaly_detection": anomaly_model is not None
        }
    }

@app.post("/analytics/predict", response_model=PredictionResponse)
async def predict_delay(request: PredictionRequest):
    """
    Predict delay probability using proper feature engineering
    """
    try:
        if delay_model is None:
            raise HTTPException(
                status_code=400, 
                detail="Model not trained. Please train the model first using: python simple_test.py"
            )
        
        # Create proper features using same logic as simple_test.py
        X = create_proper_features(request.trial_data)
        
        # Make prediction
        delay_prob = float(delay_model.predict_proba(X)[0, 1])
        delay_prediction = "delayed" if delay_prob > 0.5 else "on_time"
        confidence = abs(delay_prob - 0.5) * 2
        
        # Anomaly detection
        anomaly_score = None
        anomaly_detected = None
        if request.include_anomaly_detection and anomaly_model is not None:
            anomaly_score = float(anomaly_model.decision_function(X)[0])
            anomaly_detected = anomaly_model.predict(X)[0] == -1
        
        # Risk factors based on actual input values
        risk_factors = []
        if delay_prob > 0.7:
            risk_factors.append("High delay probability")
        if anomaly_detected:
            risk_factors.append("Anomalous pattern detected")
        
        adherence = request.trial_data.get('Medication_Adherence', 80)
        if adherence < 70:
            risk_factors.append("Low medication adherence")
            
        adr_rate = request.trial_data.get('ADR_Rate', 0)
        if adr_rate > 0.3:
            risk_factors.append("High adverse event rate")
            
        bp_sys = request.trial_data.get('BP_Systolic', 130)
        if bp_sys > 140:
            risk_factors.append("Hypertension risk")
            
        alt = request.trial_data.get('ALT_Level', 25)
        ast = request.trial_data.get('AST_Level', 25)
        if alt > 40 or ast > 40:
            risk_factors.append("Elevated liver enzymes")
            
        creatinine = request.trial_data.get('Creatinine', 1.0)
        if creatinine > 1.2:
            risk_factors.append("Elevated creatinine")
        
        return PredictionResponse(
            delay_probability=delay_prob,
            delay_prediction=delay_prediction,
            confidence=confidence,
            anomaly_score=anomaly_score,
            anomaly_detected=anomaly_detected,
            risk_factors=risk_factors
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/analytics/demo_predict")
async def demo_predict():
    """
    Demo prediction with sample high-risk data
    """
    sample_data = {
        "Age": 72,
        "BMI": 35.5,
        "Medication_Adherence": 45,  # Low adherence
        "ADR_Rate": 0.6,            # High ADR rate
        "Efficacy_Score": 25,       # Low efficacy
        "BP_Systolic": 160,         # High BP
        "BP_Diastolic": 105,
        "ALT_Level": 65,            # Elevated
        "AST_Level": 65,            # Elevated
        "Creatinine": 1.8,          # Elevated
        "Satisfaction_Score": 3,     # Low satisfaction
        "Week": 12
    }
    
    request = PredictionRequest(
        trial_data=sample_data,
        include_anomaly_detection=True
    )
    
    return await predict_delay(request)

@app.get("/analytics/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.1.0 - FIXED",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": {
            "delay_prediction": delay_model is not None,
            "anomaly_detection": anomaly_model is not None
        },
        "fix_applied": "Proper feature engineering pipeline"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fixed_api:app", host="0.0.0.0", port=8001, reload=True)
