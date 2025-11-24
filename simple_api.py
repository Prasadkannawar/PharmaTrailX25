#!/usr/bin/env python3
"""
Simple FastAPI server for PharmaTrail-X without database dependencies
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
    title="PharmaTrail-X Analytics API",
    description="Clinical Trial Delay Prediction and Analytics Platform - Phase 1",
    version="1.0.0"
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
model_info = {}

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

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global delay_model, anomaly_model, model_info
    
    logger.info("Starting PharmaTrail-X Analytics API")
    
    # Try to load existing models
    try:
        model_path = Path("models")
        if model_path.exists():
            delay_model_file = model_path / "delay_model.pkl"
            anomaly_model_file = model_path / "anomaly_model.pkl"
            
            if delay_model_file.exists():
                delay_model = joblib.load(delay_model_file)
                logger.info("Loaded delay prediction model")
            
            if anomaly_model_file.exists():
                anomaly_model = joblib.load(anomaly_model_file)
                logger.info("Loaded anomaly detection model")
            
            model_info = {
                "delay_model_loaded": delay_model is not None,
                "anomaly_model_loaded": anomaly_model is not None,
                "last_updated": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.warning(f"Could not load existing models: {e}")
        model_info = {
            "delay_model_loaded": False,
            "anomaly_model_loaded": False,
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PharmaTrail-X Analytics API - Phase 1",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": {
            "delay_prediction": delay_model is not None,
            "anomaly_detection": anomaly_model is not None
        },
        "endpoints": {
            "train": "/analytics/train",
            "predict": "/analytics/predict",
            "model_info": "/analytics/model_info",
            "health": "/analytics/health"
        }
    }

@app.post("/analytics/train")
async def train_models():
    """
    Train models using the simple pipeline
    """
    try:
        logger.info("Starting model training via API")
        
        # Import and run the simple test pipeline
        import subprocess
        import sys
        
        # Run the simple test script
        result = subprocess.run([
            sys.executable, "simple_test.py"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Reload models
            await startup_event()
            
            return {
                "status": "success",
                "message": "Models trained successfully",
                "output": result.stdout[-500:],  # Last 500 chars
                "models_loaded": {
                    "delay_prediction": delay_model is not None,
                    "anomaly_detection": anomaly_model is not None
                }
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Training failed: {result.stderr}"
            )
            
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/analytics/predict", response_model=PredictionResponse)
async def predict_delay(request: PredictionRequest):
    """
    Predict delay probability for a given trial configuration
    """
    try:
        if delay_model is None:
            raise HTTPException(
                status_code=400, 
                detail="Model not trained. Please train the model first using /analytics/train"
            )
        
        # Create a simple feature vector from the trial data
        # This is a simplified version - in production you'd use the full feature engineering
        features = [
            request.trial_data.get('Age', 65),
            request.trial_data.get('BMI', 27),
            request.trial_data.get('Risk_Score', 0.5),
            request.trial_data.get('BP_Systolic', 130),
            request.trial_data.get('BP_Diastolic', 80),
            request.trial_data.get('ALT_Level', 25),
            request.trial_data.get('AST_Level', 25),
            request.trial_data.get('Creatinine', 1.0),
            request.trial_data.get('Efficacy_Score', 50),
            request.trial_data.get('Medication_Adherence', 80),
            request.trial_data.get('Satisfaction_Score', 8),
            request.trial_data.get('Week', 4),
            2,  # visit_month
            1,  # visit_quarter  
            1,  # visit_day_of_week
            0,  # alt_elevated
            0,  # ast_elevated
            0,  # creatinine_elevated
            0,  # hypertension_risk
            0.2,  # safety_risk_score
            80,  # patient_avg_Medication_Adherence
            0.1,  # patient_avg_ADR_Reported
            50,  # patient_avg_Efficacy_Score
            0.2   # patient_avg_safety_risk_score
        ]
        
        # Ensure we have the right number of features
        while len(features) < 24:
            features.append(0)
        features = features[:24]
        
        # Make prediction
        X = np.array(features).reshape(1, -1)
        delay_prob = float(delay_model.predict_proba(X)[0, 1])
        delay_prediction = "delayed" if delay_prob > 0.5 else "on_time"
        confidence = abs(delay_prob - 0.5) * 2
        
        # Anomaly detection
        anomaly_score = None
        anomaly_detected = None
        if request.include_anomaly_detection and anomaly_model is not None:
            anomaly_score = float(anomaly_model.decision_function(X)[0])
            anomaly_detected = anomaly_model.predict(X)[0] == -1
        
        # Risk factors
        risk_factors = []
        if delay_prob > 0.7:
            risk_factors.append("High delay probability")
        if anomaly_detected:
            risk_factors.append("Anomalous pattern detected")
        if request.trial_data.get('Medication_Adherence', 80) < 70:
            risk_factors.append("Low medication adherence")
        if request.trial_data.get('ADR_Rate', 0) > 0.3:
            risk_factors.append("High adverse event rate")
        
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

@app.get("/analytics/model_info")
async def get_model_info():
    """
    Get information about the currently active models
    """
    return {
        "model_status": model_info,
        "models_available": {
            "delay_prediction": delay_model is not None,
            "anomaly_detection": anomaly_model is not None
        },
        "api_version": "1.0.0",
        "phase": "Phase 1 - Core AI Engine"
    }

@app.get("/analytics/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": {
            "delay_prediction": delay_model is not None,
            "anomaly_detection": anomaly_model is not None
        },
        "phase": "Phase 1"
    }

@app.post("/analytics/demo_predict")
async def demo_predict():
    """
    Demo prediction with sample data
    """
    sample_data = {
        "study_id": "PHX-2025-01",
        "trial_id": "TRIAL-NEUROLOGY",
        "phase": "Phase III",
        "Age": 65,
        "BMI": 28.5,
        "Medication_Adherence": 75,
        "ADR_Rate": 0.2,
        "Efficacy_Score": 45
    }
    
    request = PredictionRequest(
        trial_data=sample_data,
        include_anomaly_detection=True
    )
    
    return await predict_delay(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("simple_api:app", host="0.0.0.0", port=8000, reload=True)
