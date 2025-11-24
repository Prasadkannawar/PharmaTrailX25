from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json
from loguru import logger
from pathlib import Path

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.ingestion.data_ingester import DataIngester
from src.preprocessing.feature_engineer import FeatureEngineer
from src.models.ml_pipeline import MLPipeline
from src.models.database import get_db, create_tables
from config.settings import settings

# Phase 2 imports
from src.nlp.nlp_engine import ClinicalNLPEngine, AEExtractionResult, SummaryResult
from src.blockchain.ledger import BlockchainLedger, log_prediction_event, log_nlp_event

# Initialize FastAPI app
app = FastAPI(
    title="PharmaTrail-X Analytics API",
    description="Clinical Trial Delay Prediction and Analytics Platform",
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

# Global instances
ml_pipeline = MLPipeline()
feature_engineer = FeatureEngineer()
data_ingester = DataIngester()

# Phase 2 instances
nlp_engine: Optional[ClinicalNLPEngine] = None
blockchain_ledger: Optional[BlockchainLedger] = None

# Pydantic models for API requests/responses
class TrainingRequest(BaseModel):
    data_source: str = "clinical_trials_csv"
    hyperparameter_tuning: bool = True
    retrain_models: bool = True

class PredictionRequest(BaseModel):
    trial_data: Dict[str, Any]
    include_anomaly_detection: bool = True

class TrialConfiguration(BaseModel):
    study_id: str
    trial_id: str
    phase: str
    therapeutic_area: str
    site_id: str
    principal_investigator: str
    region: str
    target_enrollment: int
    current_enrollment: int
    planned_duration_weeks: int

class PredictionResponse(BaseModel):
    delay_probability: float
    delay_prediction: str
    confidence: float
    anomaly_score: Optional[float] = None
    anomaly_detected: Optional[bool] = None
    feature_importance: Dict[str, float]
    risk_factors: List[str]

class ModelInfoResponse(BaseModel):
    model_status: Dict[str, Any]
    training_metrics: Dict[str, Any]
    feature_importance: Dict[str, Any]
    last_training_date: Optional[str] = None

# Phase 2 Pydantic models
class NLPRequest(BaseModel):
    text: str
    trial_id: str
    patient_id: Optional[str] = None

class NLPResponse(BaseModel):
    entities: List[Dict[str, Any]]
    ae_events: List[Dict[str, Any]]
    severity_classification: str
    confidence_score: float
    processing_timestamp: str

class SummaryRequest(BaseModel):
    text: str
    max_length: int = 150

class SummaryResponse(BaseModel):
    summary: str
    key_points: List[str]
    word_count_reduction: float
    confidence_score: float
    processing_timestamp: str

class BlockchainLogRequest(BaseModel):
    event_type: str
    event_data: Dict[str, Any]

class BlockchainLogResponse(BaseModel):
    success: bool
    block_hash: str
    block_index: int
    timestamp: str

class IntegratedPredictionRequest(BaseModel):
    trial_data: Dict[str, Any]
    clinical_text: Optional[str] = None
    include_nlp: bool = True
    include_blockchain_audit: bool = True

class IntegratedPredictionResponse(BaseModel):
    # Phase 1 results
    delay_probability: float
    delay_prediction: str
    confidence: float
    anomaly_score: Optional[float] = None
    anomaly_detected: Optional[bool] = None
    risk_factors: List[str]
    
    # Phase 2 results
    nlp_results: Optional[Dict[str, Any]] = None
    blockchain_hash: Optional[str] = None
    
    # Combined insights
    combined_risk_score: float
    regulatory_flags: List[str]

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global nlp_engine, blockchain_ledger
    
    logger.info("Starting PharmaTrail-X Analytics API")
    
    # Create database tables
    create_tables()
    
    # Try to load existing models
    try:
        model_path = Path(settings.MODEL_REGISTRY_PATH)
        if model_path.exists():
            ml_pipeline.load_models(str(model_path))
            feature_engineer.load_preprocessors(str(model_path))
            logger.info("Loaded existing models")
    except Exception as e:
        logger.warning(f"Could not load existing models: {e}")
    
    # Initialize Phase 2 components
    try:
        logger.info("Initializing Phase 2 components...")
        nlp_engine = ClinicalNLPEngine()
        blockchain_ledger = BlockchainLedger("pharmatrail-x-main")
        logger.info("✅ Phase 2 components initialized")
    except Exception as e:
        logger.error(f"❌ Error initializing Phase 2 components: {e}")
        nlp_engine = None
        blockchain_ledger = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PharmaTrail-X Analytics API - Phase 2",
        "version": "2.0.0",
        "status": "running",
        "phase1_endpoints": {
            "train": "/analytics/train",
            "predict": "/analytics/predict", 
            "model_info": "/analytics/model_info",
            "health": "/analytics/health",
            "batch_predict": "/analytics/batch_predict"
        },
        "phase2_endpoints": {
            "nlp_adverse_events": "/nlp/ae",
            "nlp_summarization": "/nlp/summary",
            "blockchain_log": "/blockchain/log_event",
            "blockchain_chain": "/blockchain/get_chain",
            "integrated_prediction": "/predict/integrated"
        },
        "components": {
            "phase1_models": ml_pipeline.delay_model is not None,
            "nlp_engine": nlp_engine is not None,
            "blockchain_ledger": blockchain_ledger is not None
        }
    }

@app.post("/analytics/train")
async def train_models(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """
    Train or retrain the delay prediction and anomaly detection models
    """
    try:
        logger.info("Starting model training pipeline")
        
        # Load and process data
        if request.data_source == "clinical_trials_csv":
            # Use the existing CSV file
            csv_path = "PharmaTrailX_ClinicalTrialMaster2.csv"
            df = data_ingester.ingest_clinical_trials_csv(csv_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid data source")
        
        # Feature engineering
        df_features = feature_engineer.engineer_features(df)
        X, feature_names = feature_engineer.prepare_ml_features(df_features)
        
        # Prepare target variable (delay prediction)
        y = (df_features['delay_probability'] > settings.DELAY_THRESHOLD).astype(int)
        
        # Train models
        training_results = {}
        
        if request.retrain_models or ml_pipeline.delay_model is None:
            # Train delay prediction model
            delay_metrics = ml_pipeline.train_delay_predictor(
                X, y, hyperparameter_tuning=request.hyperparameter_tuning
            )
            training_results['delay_prediction'] = delay_metrics
            
            # Train anomaly detection model
            anomaly_metrics = ml_pipeline.train_anomaly_detector(X)
            training_results['anomaly_detection'] = anomaly_metrics
        
        # Save models and preprocessors
        model_path = Path(settings.MODEL_REGISTRY_PATH)
        model_path.mkdir(exist_ok=True)
        
        ml_pipeline.save_models(str(model_path))
        feature_engineer.save_preprocessors(str(model_path))
        
        # Load data to database in background
        background_tasks.add_task(data_ingester.load_to_database, df_features, db)
        
        logger.info("Model training completed successfully")
        
        return {
            "status": "success",
            "message": "Models trained successfully",
            "training_results": training_results,
            "model_path": str(model_path),
            "features_count": len(feature_names),
            "training_samples": len(X)
        }
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/analytics/predict", response_model=PredictionResponse)
async def predict_delay(request: PredictionRequest):
    """
    Predict delay probability for a given trial configuration
    """
    try:
        if ml_pipeline.delay_model is None:
            raise HTTPException(
                status_code=400, 
                detail="Model not trained. Please train the model first using /analytics/train"
            )
        
        # Convert trial data to DataFrame
        trial_df = pd.DataFrame([request.trial_data])
        
        # Feature engineering (using existing preprocessors)
        trial_features = feature_engineer.engineer_features(trial_df)
        X, _ = feature_engineer.prepare_ml_features(trial_features)
        
        # Make predictions
        delay_pred, delay_prob = ml_pipeline.predict_delay(X)
        delay_probability = float(delay_prob[0])
        delay_prediction = "delayed" if delay_pred[0] == 1 else "on_time"
        
        # Anomaly detection
        anomaly_score = None
        anomaly_detected = None
        if request.include_anomaly_detection and ml_pipeline.anomaly_model is not None:
            anomaly_pred, anomaly_scores = ml_pipeline.detect_anomalies(X)
            anomaly_score = float(anomaly_scores[0])
            anomaly_detected = anomaly_pred[0] == -1
        
        # Feature importance
        feature_importance = ml_pipeline.get_feature_importance('delay_prediction')
        
        # Identify risk factors
        risk_factors = []
        if delay_probability > 0.7:
            risk_factors.append("High delay probability")
        if anomaly_detected:
            risk_factors.append("Anomalous trial pattern detected")
        
        # Add specific risk factors based on feature values
        if 'safety_risk_score' in request.trial_data:
            if request.trial_data['safety_risk_score'] > 0.5:
                risk_factors.append("High safety risk score")
        
        return PredictionResponse(
            delay_probability=delay_probability,
            delay_prediction=delay_prediction,
            confidence=abs(delay_probability - 0.5) * 2,  # Convert to 0-1 confidence
            anomaly_score=anomaly_score,
            anomaly_detected=anomaly_detected,
            feature_importance=dict(list(feature_importance.items())[:10]),  # Top 10 features
            risk_factors=risk_factors
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/analytics/model_info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about the currently active models
    """
    try:
        model_info = ml_pipeline.get_model_info()
        
        # Get feature importance
        feature_importance = {}
        if ml_pipeline.feature_importance:
            feature_importance = ml_pipeline.feature_importance
        
        # Check for last training date (from model files)
        last_training_date = None
        model_path = Path(settings.MODEL_REGISTRY_PATH)
        if model_path.exists():
            model_files = list(model_path.glob("*.pkl"))
            if model_files:
                latest_file = max(model_files, key=lambda x: x.stat().st_mtime)
                last_training_date = datetime.fromtimestamp(
                    latest_file.stat().st_mtime
                ).isoformat()
        
        return ModelInfoResponse(
            model_status=model_info,
            training_metrics=ml_pipeline.model_metrics,
            feature_importance=feature_importance,
            last_training_date=last_training_date
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/analytics/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": {
            "delay_prediction": ml_pipeline.delay_model is not None,
            "anomaly_detection": ml_pipeline.anomaly_model is not None
        }
    }

@app.post("/analytics/batch_predict")
async def batch_predict(trials: List[Dict[str, Any]]):
    """
    Batch prediction for multiple trials
    """
    try:
        if ml_pipeline.delay_model is None:
            raise HTTPException(
                status_code=400, 
                detail="Model not trained. Please train the model first."
            )
        
        results = []
        
        for trial_data in trials:
            # Convert to DataFrame and predict
            trial_df = pd.DataFrame([trial_data])
            trial_features = feature_engineer.engineer_features(trial_df)
            X, _ = feature_engineer.prepare_ml_features(trial_features)
            
            delay_pred, delay_prob = ml_pipeline.predict_delay(X)
            
            results.append({
                "trial_id": trial_data.get("trial_id", "unknown"),
                "delay_probability": float(delay_prob[0]),
                "delay_prediction": "delayed" if delay_pred[0] == 1 else "on_time"
            })
        
        return {
            "status": "success",
            "predictions": results,
            "total_trials": len(trials)
        }
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# ============================================================================
# PHASE 2 ENDPOINTS - NLP & BLOCKCHAIN
# ============================================================================

@app.post("/nlp/ae", response_model=NLPResponse)
async def extract_adverse_events(request: NLPRequest, background_tasks: BackgroundTasks):
    """
    Extract adverse events from clinical text using NLP
    """
    if not nlp_engine:
        raise HTTPException(status_code=503, detail="NLP engine not initialized")
    
    try:
        logger.info(f"Processing AE extraction for trial {request.trial_id}")
        
        # Extract adverse events
        ae_result = nlp_engine.extract_adverse_events(request.text, request.trial_id)
        
        # Prepare response
        response = NLPResponse(
            entities=[
                {
                    "text": entity.text,
                    "label": entity.label,
                    "start": entity.start,
                    "end": entity.end,
                    "confidence": entity.confidence,
                    "severity": entity.severity
                }
                for entity in ae_result.entities
            ],
            ae_events=ae_result.ae_events,
            severity_classification=ae_result.severity_classification,
            confidence_score=ae_result.confidence_score,
            processing_timestamp=ae_result.processing_timestamp
        )
        
        # Log to blockchain in background
        if blockchain_ledger:
            background_tasks.add_task(
                log_nlp_to_blockchain,
                request.trial_id,
                "adverse_event_extraction",
                {
                    "entity_count": len(ae_result.entities),
                    "ae_events": ae_result.ae_events,
                    "severity": ae_result.severity_classification,
                    "confidence": ae_result.confidence_score,
                    "patient_id": request.patient_id
                }
            )
        
        logger.info(f"✅ AE extraction completed: {len(ae_result.entities)} entities found")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in AE extraction: {e}")
        raise HTTPException(status_code=500, detail=f"AE extraction failed: {str(e)}")

@app.post("/nlp/summary", response_model=SummaryResponse)
async def summarize_clinical_text(request: SummaryRequest):
    """
    Summarize clinical text and extract key points
    """
    if not nlp_engine:
        raise HTTPException(status_code=503, detail="NLP engine not initialized")
    
    try:
        logger.info("Processing text summarization")
        
        # Summarize text
        summary_result = nlp_engine.summarize_clinical_text(request.text, request.max_length)
        
        response = SummaryResponse(
            summary=summary_result.summary,
            key_points=summary_result.key_points,
            word_count_reduction=summary_result.word_count_reduction,
            confidence_score=summary_result.confidence_score,
            processing_timestamp=summary_result.processing_timestamp
        )
        
        logger.info(f"✅ Text summarization completed: {summary_result.word_count_reduction:.1%} reduction")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in text summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/blockchain/log_event", response_model=BlockchainLogResponse)
async def log_blockchain_event(request: BlockchainLogRequest):
    """
    Log an event to the blockchain audit ledger
    """
    if not blockchain_ledger:
        raise HTTPException(status_code=503, detail="Blockchain ledger not initialized")
    
    try:
        logger.info(f"Logging blockchain event: {request.event_type}")
        
        # Log event to blockchain
        block_hash = blockchain_ledger.log_event(request.event_type, request.event_data)
        
        # Get the latest block info
        latest_block = blockchain_ledger.chain[-1]
        
        response = BlockchainLogResponse(
            success=True,
            block_hash=block_hash,
            block_index=latest_block.index,
            timestamp=latest_block.timestamp
        )
        
        logger.info(f"✅ Event logged to blockchain: Block #{latest_block.index}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error logging to blockchain: {e}")
        raise HTTPException(status_code=500, detail=f"Blockchain logging failed: {str(e)}")

@app.get("/blockchain/get_chain")
async def get_blockchain_chain(limit: Optional[int] = None):
    """
    Retrieve the blockchain audit chain
    """
    if not blockchain_ledger:
        raise HTTPException(status_code=503, detail="Blockchain ledger not initialized")
    
    try:
        chain_data = blockchain_ledger.get_chain(limit=limit)
        stats = blockchain_ledger.get_chain_stats()
        
        return {
            "chain_id": blockchain_ledger.chain_id,
            "total_blocks": len(chain_data),
            "blocks": chain_data,
            "stats": stats,
            "integrity_verified": blockchain_ledger.verify_chain_integrity()["is_valid"]
        }
        
    except Exception as e:
        logger.error(f"❌ Error retrieving blockchain: {e}")
        raise HTTPException(status_code=500, detail=f"Blockchain retrieval failed: {str(e)}")

@app.post("/predict/integrated", response_model=IntegratedPredictionResponse)
async def integrated_prediction(request: IntegratedPredictionRequest, background_tasks: BackgroundTasks):
    """
    Integrated prediction combining Phase 1 AI models with Phase 2 NLP and blockchain
    """
    try:
        logger.info("Processing integrated prediction")
        
        # Phase 1: Standard delay prediction
        if ml_pipeline.delay_model is None:
            raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")
        
        # Convert to DataFrame and predict
        trial_df = pd.DataFrame([request.trial_data])
        trial_features = feature_engineer.engineer_features(trial_df)
        X, _ = feature_engineer.prepare_ml_features(trial_features)
        
        delay_pred, delay_prob = ml_pipeline.predict_delay(X)
        delay_probability = float(delay_prob[0])
        delay_prediction = "delayed" if delay_pred[0] == 1 else "on_time"
        
        # Anomaly detection
        anomaly_score = None
        anomaly_detected = None
        if ml_pipeline.anomaly_model is not None:
            anomaly_pred = ml_pipeline.predict_anomaly(X)
            anomaly_score = float(ml_pipeline.anomaly_model.decision_function(X)[0])
            anomaly_detected = anomaly_pred[0] == -1
        
        # Risk factors
        risk_factors = []
        if delay_probability > 0.7:
            risk_factors.append("High delay probability")
        if anomaly_detected:
            risk_factors.append("Anomalous trial pattern detected")
        
        # Phase 2: NLP processing (if text provided)
        nlp_results = None
        if request.clinical_text and request.include_nlp and nlp_engine:
            try:
                ae_result = nlp_engine.extract_adverse_events(
                    request.clinical_text, 
                    request.trial_data.get("trial_id", "unknown")
                )
                
                nlp_results = {
                    "entities_found": len(ae_result.entities),
                    "ae_events": ae_result.ae_events,
                    "severity_classification": ae_result.severity_classification,
                    "confidence_score": ae_result.confidence_score
                }
                
                # Add NLP-based risk factors
                if ae_result.severity_classification in ["severe", "life-threatening"]:
                    risk_factors.append("Severe adverse events detected in text")
                
            except Exception as e:
                logger.warning(f"NLP processing failed: {e}")
                nlp_results = {"error": str(e)}
        
        # Combined risk score
        base_risk = delay_probability
        nlp_risk = 0.0
        if nlp_results and "ae_events" in nlp_results:
            severe_aes = len([ae for ae in nlp_results["ae_events"] if ae.get("severity") == "severe"])
            nlp_risk = min(severe_aes * 0.1, 0.3)  # Max 0.3 additional risk
        
        combined_risk_score = (base_risk * 0.7) + (nlp_risk * 0.3)
        
        # Regulatory flags
        regulatory_flags = []
        if delay_probability > 0.8:
            regulatory_flags.append("HIGH_DELAY_RISK")
        if anomaly_detected:
            regulatory_flags.append("ANOMALOUS_PATTERN")
        if nlp_results and nlp_results.get("severity_classification") == "severe":
            regulatory_flags.append("SEVERE_AE_DETECTED")
        
        # Prepare response
        response = IntegratedPredictionResponse(
            delay_probability=delay_probability,
            delay_prediction=delay_prediction,
            confidence=abs(delay_probability - 0.5) * 2,
            anomaly_score=anomaly_score,
            anomaly_detected=anomaly_detected,
            risk_factors=risk_factors,
            nlp_results=nlp_results,
            blockchain_hash=None,  # Will be set by background task
            combined_risk_score=combined_risk_score,
            regulatory_flags=regulatory_flags
        )
        
        # Log to blockchain in background
        if request.include_blockchain_audit and blockchain_ledger:
            background_tasks.add_task(
                log_prediction_to_blockchain,
                request.trial_data.get("trial_id", "unknown"),
                {
                    "delay_probability": delay_probability,
                    "delay_prediction": delay_prediction,
                    "combined_risk_score": combined_risk_score,
                    "regulatory_flags": regulatory_flags,
                    "nlp_processed": request.include_nlp and request.clinical_text is not None,
                    "anomaly_detected": anomaly_detected
                }
            )
        
        logger.info(f"✅ Integrated prediction completed: {delay_prediction} ({delay_probability:.3f})")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in integrated prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Integrated prediction failed: {str(e)}")

# Background task functions
async def log_nlp_to_blockchain(trial_id: str, event_type: str, nlp_data: Dict[str, Any]):
    """Background task to log NLP events to blockchain"""
    try:
        if blockchain_ledger:
            log_nlp_event(blockchain_ledger, {
                "trial_id": trial_id,
                "event_type": event_type,
                **nlp_data
            })
    except Exception as e:
        logger.error(f"Error logging NLP event to blockchain: {e}")

async def log_prediction_to_blockchain(trial_id: str, prediction_data: Dict[str, Any]):
    """Background task to log predictions to blockchain"""
    try:
        if blockchain_ledger:
            log_prediction_event(blockchain_ledger, {
                "trial_id": trial_id,
                **prediction_data
            })
    except Exception as e:
        logger.error(f"Error logging prediction to blockchain: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD
    )
