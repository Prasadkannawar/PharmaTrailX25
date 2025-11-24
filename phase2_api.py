#!/usr/bin/env python3
"""
PharmaTrail-X Phase 2 Integrated API
Combines Phase 1 delay prediction with Phase 2 NLP and blockchain capabilities
"""

import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
import joblib
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.nlp.clinical_nlp_engine import ClinicalNLPEngine
from src.blockchain.audit_ledger import PharmaBlockchain, AuditLogger

# Initialize FastAPI app
app = FastAPI(
    title="PharmaTrail-X Phase 2 Integrated Platform",
    description="Complete clinical trial intelligence platform with AI delay prediction, NLP, and blockchain audit",
    version="2.0.0"
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
nlp_engine: Optional[ClinicalNLPEngine] = None
blockchain: Optional[PharmaBlockchain] = None
audit_logger: Optional[AuditLogger] = None
delay_model = None
anomaly_model = None

# Pydantic models
class IntegratedPredictionRequest(BaseModel):
    trial_id: str = Field(..., description="Clinical trial identifier")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    
    # Phase 1 data (structured)
    clinical_data: Dict[str, Any] = Field(..., description="Structured clinical data")
    
    # Phase 2 data (unstructured)
    clinical_text: Optional[str] = Field(None, description="Clinical narrative text")
    investigator_notes: Optional[str] = Field(None, description="Investigator notes")
    ae_reports: Optional[List[str]] = Field(None, description="Adverse event reports")
    
    # Processing options
    include_nlp: bool = Field(True, description="Include NLP processing")
    include_blockchain_audit: bool = Field(True, description="Log to blockchain")

class IntegratedPredictionResponse(BaseModel):
    trial_id: str
    patient_id: Optional[str]
    
    # Phase 1 results
    delay_probability: float
    delay_prediction: str
    confidence: float
    anomaly_score: Optional[float]
    anomaly_detected: Optional[bool]
    risk_factors: List[str]
    
    # Phase 2 results
    nlp_results: Optional[Dict[str, Any]] = None
    extracted_aes: Optional[List[Dict[str, Any]]] = None
    text_summaries: Optional[List[Dict[str, Any]]] = None
    
    # Audit trail
    blockchain_hashes: List[str] = Field(default_factory=list)
    processing_timestamp: str
    
    # Enhanced insights
    combined_risk_score: float
    regulatory_flags: List[str]

@app.on_event("startup")
async def startup_event():
    """Initialize all Phase 2 components"""
    global nlp_engine, blockchain, audit_logger, delay_model, anomaly_model
    
    logger.info("🚀 Starting PharmaTrail-X Phase 2 Integrated Platform...")
    
    try:
        # Initialize NLP engine
        logger.info("Loading clinical NLP engine...")
        nlp_engine = ClinicalNLPEngine()
        
        # Initialize blockchain
        logger.info("Initializing blockchain audit ledger...")
        blockchain = PharmaBlockchain("pharmatrail-x-integrated")
        audit_logger = AuditLogger(blockchain)
        
        # Load Phase 1 models
        logger.info("Loading Phase 1 ML models...")
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
        
        # Log startup event
        startup_data = {
            "platform": "pharmatrail_x_phase2",
            "version": "2.0.0",
            "components_loaded": {
                "nlp_engine": nlp_engine is not None,
                "blockchain": blockchain is not None,
                "delay_model": delay_model is not None,
                "anomaly_model": anomaly_model is not None
            },
            "startup_time": datetime.utcnow().isoformat()
        }
        
        if audit_logger:
            audit_logger.log_system_event(
                trial_id="SYSTEM",
                system_data={
                    "event": "phase2_platform_startup",
                    "component": "integrated_platform",
                    "status": "success",
                    "details": startup_data
                }
            )
        
        logger.info("✅ Phase 2 Integrated Platform startup complete")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "platform": "PharmaTrail-X Phase 2 Integrated Platform",
        "version": "2.0.0",
        "status": "running",
        "capabilities": [
            "ai_delay_prediction",
            "clinical_nlp_processing",
            "adverse_event_extraction",
            "text_summarization",
            "blockchain_audit_logging",
            "anomaly_detection",
            "integrated_risk_assessment"
        ],
        "components": {
            "phase1_models_loaded": delay_model is not None and anomaly_model is not None,
            "nlp_engine_active": nlp_engine is not None,
            "blockchain_active": blockchain is not None
        },
        "endpoints": {
            "integrated_prediction": "/predict/integrated",
            "nlp_only": "/nlp/ae",
            "blockchain_audit": "/blockchain/log_event",
            "health": "/health"
        }
    }

@app.post("/predict/integrated", response_model=IntegratedPredictionResponse)
async def integrated_prediction(
    request: IntegratedPredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Integrated prediction combining Phase 1 AI models with Phase 2 NLP and blockchain
    """
    logger.info(f"Processing integrated prediction for trial {request.trial_id}")
    
    try:
        # Phase 1: Delay Prediction
        delay_prob, delay_pred, confidence, anomaly_score, anomaly_detected, risk_factors = await _phase1_prediction(
            request.clinical_data
        )
        
        # Phase 2: NLP Processing
        nlp_results = None
        extracted_aes = []
        text_summaries = []
        
        if request.include_nlp and nlp_engine:
            nlp_results, extracted_aes, text_summaries = await _phase2_nlp_processing(
                request.clinical_text,
                request.investigator_notes,
                request.ae_reports,
                request.trial_id
            )
        
        # Combined Risk Assessment
        combined_risk_score = _calculate_combined_risk(
            delay_prob, nlp_results, extracted_aes
        )
        
        # Regulatory Flags
        regulatory_flags = _identify_regulatory_flags(
            delay_prob, extracted_aes, anomaly_detected
        )
        
        # Prepare response
        response = IntegratedPredictionResponse(
            trial_id=request.trial_id,
            patient_id=request.patient_id,
            delay_probability=delay_prob,
            delay_prediction=delay_pred,
            confidence=confidence,
            anomaly_score=anomaly_score,
            anomaly_detected=anomaly_detected,
            risk_factors=risk_factors,
            nlp_results=nlp_results,
            extracted_aes=extracted_aes,
            text_summaries=text_summaries,
            combined_risk_score=combined_risk_score,
            regulatory_flags=regulatory_flags,
            processing_timestamp=datetime.utcnow().isoformat()
        )
        
        # Blockchain Audit Logging
        if request.include_blockchain_audit and audit_logger:
            background_tasks.add_task(
                _log_integrated_prediction,
                request.trial_id,
                request,
                response
            )
        
        logger.info(f"✅ Integrated prediction completed for trial {request.trial_id}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in integrated prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Integrated prediction failed: {str(e)}")

async def _phase1_prediction(clinical_data: Dict[str, Any]) -> tuple:
    """Execute Phase 1 delay prediction"""
    if not delay_model:
        return 0.5, "unknown", 0.0, None, None, ["Model not loaded"]
    
    try:
        # Create feature vector (same logic as fixed_api.py)
        features = _create_feature_vector(clinical_data)
        X = np.array(features).reshape(1, -1)
        
        # Make predictions
        delay_prob = float(delay_model.predict_proba(X)[0, 1])
        delay_pred = "delayed" if delay_prob > 0.5 else "on_time"
        confidence = abs(delay_prob - 0.5) * 2
        
        # Anomaly detection
        anomaly_score = None
        anomaly_detected = None
        if anomaly_model:
            anomaly_score = float(anomaly_model.decision_function(X)[0])
            anomaly_detected = anomaly_model.predict(X)[0] == -1
        
        # Risk factors
        risk_factors = _identify_risk_factors(clinical_data, delay_prob, anomaly_detected)
        
        return delay_prob, delay_pred, confidence, anomaly_score, anomaly_detected, risk_factors
        
    except Exception as e:
        logger.error(f"Error in Phase 1 prediction: {e}")
        return 0.5, "error", 0.0, None, None, [f"Prediction error: {str(e)}"]

async def _phase2_nlp_processing(
    clinical_text: Optional[str],
    investigator_notes: Optional[str], 
    ae_reports: Optional[List[str]],
    trial_id: str
) -> tuple:
    """Execute Phase 2 NLP processing"""
    nlp_results = {}
    extracted_aes = []
    text_summaries = []
    
    try:
        # Process clinical text
        if clinical_text and nlp_engine:
            ae_extraction = nlp_engine.extract_adverse_events(clinical_text, trial_id)
            summary = nlp_engine.summarize_clinical_text(clinical_text)
            
            nlp_results["clinical_text"] = {
                "entities": len(ae_extraction.entities),
                "ae_events": len(ae_extraction.ae_events),
                "severity": ae_extraction.severity_classification,
                "confidence": ae_extraction.confidence_score
            }
            
            extracted_aes.extend(ae_extraction.ae_events)
            text_summaries.append({
                "source": "clinical_text",
                "summary": summary.summary,
                "key_points": summary.key_points,
                "reduction": summary.word_count_reduction
            })
        
        # Process investigator notes
        if investigator_notes and nlp_engine:
            ae_extraction = nlp_engine.extract_adverse_events(investigator_notes, trial_id)
            summary = nlp_engine.summarize_clinical_text(investigator_notes)
            
            nlp_results["investigator_notes"] = {
                "entities": len(ae_extraction.entities),
                "ae_events": len(ae_extraction.ae_events),
                "severity": ae_extraction.severity_classification,
                "confidence": ae_extraction.confidence_score
            }
            
            extracted_aes.extend(ae_extraction.ae_events)
            text_summaries.append({
                "source": "investigator_notes",
                "summary": summary.summary,
                "key_points": summary.key_points,
                "reduction": summary.word_count_reduction
            })
        
        # Process AE reports
        if ae_reports and nlp_engine:
            for i, ae_report in enumerate(ae_reports):
                ae_extraction = nlp_engine.extract_adverse_events(ae_report, trial_id)
                extracted_aes.extend(ae_extraction.ae_events)
                
                nlp_results[f"ae_report_{i}"] = {
                    "entities": len(ae_extraction.entities),
                    "ae_events": len(ae_extraction.ae_events),
                    "severity": ae_extraction.severity_classification,
                    "confidence": ae_extraction.confidence_score
                }
        
        return nlp_results, extracted_aes, text_summaries
        
    except Exception as e:
        logger.error(f"Error in Phase 2 NLP processing: {e}")
        return {"error": str(e)}, [], []

def _create_feature_vector(clinical_data: Dict[str, Any]) -> List[float]:
    """Create feature vector for Phase 1 models"""
    # Same logic as fixed_api.py
    age = clinical_data.get('Age', 65)
    bmi = clinical_data.get('BMI', 27)
    adherence = clinical_data.get('Medication_Adherence', 80)
    adr_rate = clinical_data.get('ADR_Rate', 0.1)
    efficacy = clinical_data.get('Efficacy_Score', 50)
    bp_sys = clinical_data.get('BP_Systolic', 130)
    bp_dia = clinical_data.get('BP_Diastolic', 80)
    alt = clinical_data.get('ALT_Level', 25)
    ast = clinical_data.get('AST_Level', 25)
    creatinine = clinical_data.get('Creatinine', 1.0)
    satisfaction = clinical_data.get('Satisfaction_Score', 8)
    week = clinical_data.get('Week', 4)
    
    # Basic time features
    visit_month = 2
    visit_quarter = 1
    visit_day_of_week = 1
    
    # Safety features
    alt_elevated = 1 if alt > 40 else 0
    ast_elevated = 1 if ast > 40 else 0
    creatinine_elevated = 1 if creatinine > 1.2 else 0
    hypertension_risk = 1 if (bp_sys > 140 or bp_dia > 90) else 0
    
    # Safety risk score
    adr_severity = 2 if adr_rate > 0.3 else 1 if adr_rate > 0.1 else 0
    safety_risk_score = (
        adr_severity * adr_rate * 0.4 +
        (alt_elevated + ast_elevated + creatinine_elevated) * 0.3 +
        hypertension_risk * 0.3
    )
    
    # Patient aggregations
    patient_avg_adherence = adherence
    patient_avg_adr = adr_rate
    patient_avg_efficacy = efficacy
    patient_avg_safety = safety_risk_score
    
    return [
        age, bmi, 0.5, bp_sys, bp_dia, alt, ast, creatinine, efficacy, adherence,
        satisfaction, week, visit_month, visit_quarter, visit_day_of_week,
        alt_elevated, ast_elevated, creatinine_elevated, hypertension_risk,
        safety_risk_score, patient_avg_adherence, patient_avg_adr,
        patient_avg_efficacy, patient_avg_safety
    ]

def _identify_risk_factors(clinical_data: Dict[str, Any], delay_prob: float, anomaly_detected: bool) -> List[str]:
    """Identify risk factors from clinical data"""
    risk_factors = []
    
    if delay_prob > 0.7:
        risk_factors.append("High delay probability")
    if anomaly_detected:
        risk_factors.append("Anomalous pattern detected")
    
    adherence = clinical_data.get('Medication_Adherence', 80)
    if adherence < 70:
        risk_factors.append("Low medication adherence")
    
    adr_rate = clinical_data.get('ADR_Rate', 0)
    if adr_rate > 0.3:
        risk_factors.append("High adverse event rate")
    
    bp_sys = clinical_data.get('BP_Systolic', 130)
    if bp_sys > 140:
        risk_factors.append("Hypertension risk")
    
    alt = clinical_data.get('ALT_Level', 25)
    ast = clinical_data.get('AST_Level', 25)
    if alt > 40 or ast > 40:
        risk_factors.append("Elevated liver enzymes")
    
    creatinine = clinical_data.get('Creatinine', 1.0)
    if creatinine > 1.2:
        risk_factors.append("Elevated creatinine")
    
    return risk_factors

def _calculate_combined_risk(delay_prob: float, nlp_results: Dict, extracted_aes: List) -> float:
    """Calculate combined risk score from Phase 1 and Phase 2 results"""
    # Base risk from delay probability
    base_risk = delay_prob
    
    # NLP risk factors
    nlp_risk = 0.0
    if nlp_results:
        # Count severe AEs
        severe_aes = len([ae for ae in extracted_aes if ae.get('severity') == 'severe'])
        nlp_risk = min(severe_aes * 0.1, 0.3)  # Max 0.3 additional risk
    
    # Combined risk (weighted average)
    combined_risk = (base_risk * 0.7) + (nlp_risk * 0.3)
    return min(combined_risk, 1.0)

def _identify_regulatory_flags(delay_prob: float, extracted_aes: List, anomaly_detected: bool) -> List[str]:
    """Identify regulatory compliance flags"""
    flags = []
    
    if delay_prob > 0.8:
        flags.append("HIGH_DELAY_RISK")
    
    if anomaly_detected:
        flags.append("ANOMALOUS_PATTERN")
    
    # Check for serious AEs
    serious_aes = [ae for ae in extracted_aes if ae.get('severity') in ['severe', 'life-threatening']]
    if len(serious_aes) > 2:
        flags.append("MULTIPLE_SERIOUS_AES")
    
    # Check for protocol deviations
    protocol_deviations = [ae for ae in extracted_aes if 'deviation' in ae.get('event_text', '').lower()]
    if protocol_deviations:
        flags.append("PROTOCOL_DEVIATION")
    
    return flags

async def _log_integrated_prediction(
    trial_id: str,
    request: IntegratedPredictionRequest,
    response: IntegratedPredictionResponse
):
    """Log integrated prediction to blockchain"""
    try:
        if audit_logger:
            # Log the integrated prediction
            prediction_data = {
                "model_type": "integrated_prediction",
                "prediction": {
                    "delay_probability": response.delay_probability,
                    "delay_prediction": response.delay_prediction,
                    "combined_risk_score": response.combined_risk_score,
                    "regulatory_flags": response.regulatory_flags
                },
                "features": {
                    "clinical_data_hash": hashlib.sha256(str(request.clinical_data).encode()).hexdigest(),
                    "nlp_processed": request.include_nlp,
                    "text_sources": {
                        "clinical_text": request.clinical_text is not None,
                        "investigator_notes": request.investigator_notes is not None,
                        "ae_reports": len(request.ae_reports) if request.ae_reports else 0
                    }
                },
                "confidence": response.confidence,
                "model_version": "phase2_integrated_v2.0"
            }
            
            audit_logger.log_model_prediction(trial_id, prediction_data)
            
    except Exception as e:
        logger.error(f"Error logging to blockchain: {e}")

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "platform": "PharmaTrail-X Phase 2",
        "version": "2.0.0",
        "components": {
            "phase1_delay_model": delay_model is not None,
            "phase1_anomaly_model": anomaly_model is not None,
            "phase2_nlp_engine": nlp_engine is not None,
            "phase2_blockchain": blockchain is not None
        },
        "capabilities": {
            "integrated_prediction": True,
            "clinical_nlp": nlp_engine is not None,
            "blockchain_audit": blockchain is not None,
            "regulatory_compliance": True
        },
        "blockchain_status": {
            "total_blocks": len(blockchain.chain) if blockchain else 0,
            "integrity_verified": blockchain.verify_chain_integrity()["is_valid"] if blockchain else False
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("phase2_api:app", host="0.0.0.0", port=8004, reload=True)
