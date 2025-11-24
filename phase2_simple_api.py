#!/usr/bin/env python3
"""
Simplified Phase 2 API for PharmaTrail-X
No database dependency - focuses on Phase 2 NLP and blockchain functionality
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.nlp.nlp_engine import ClinicalNLPEngine
from src.blockchain.ledger import BlockchainLedger, log_prediction_event, log_nlp_event

# Initialize FastAPI app
app = FastAPI(
    title="PharmaTrail-X Phase 2 API",
    description="Clinical NLP and Blockchain Audit Platform",
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
blockchain_ledger: Optional[BlockchainLedger] = None

# Pydantic models
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
    # Phase 1 results (simplified)
    delay_probability: float
    delay_prediction: str
    confidence: float
    risk_factors: List[str]
    
    # Phase 2 results
    nlp_results: Optional[Dict[str, Any]] = None
    blockchain_hash: Optional[str] = None
    
    # Combined insights
    combined_risk_score: float
    regulatory_flags: List[str]

@app.on_event("startup")
async def startup_event():
    """Initialize Phase 2 components"""
    global nlp_engine, blockchain_ledger
    
    logger.info("🚀 Starting PharmaTrail-X Phase 2 API...")
    
    try:
        # Initialize Phase 2 components
        logger.info("Initializing NLP engine...")
        nlp_engine = ClinicalNLPEngine()
        
        logger.info("Initializing blockchain ledger...")
        blockchain_ledger = BlockchainLedger("pharmatrail-x-phase2")
        
        logger.info("✅ Phase 2 components initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error initializing Phase 2 components: {e}")
        nlp_engine = None
        blockchain_ledger = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PharmaTrail-X Phase 2 API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "nlp_adverse_events": "/nlp/ae",
            "nlp_summarization": "/nlp/summary",
            "blockchain_log": "/blockchain/log_event",
            "blockchain_chain": "/blockchain/get_chain",
            "integrated_prediction": "/predict/integrated",
            "health": "/health"
        },
        "components": {
            "nlp_engine": nlp_engine is not None,
            "blockchain_ledger": blockchain_ledger is not None
        }
    }

@app.post("/nlp/ae", response_model=NLPResponse)
async def extract_adverse_events(request: NLPRequest, background_tasks: BackgroundTasks):
    """Extract adverse events from clinical text using NLP"""
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
    """Summarize clinical text and extract key points"""
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
    """Log an event to the blockchain audit ledger"""
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
    """Retrieve the blockchain audit chain"""
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
    """Integrated prediction combining simplified Phase 1 with Phase 2 NLP and blockchain"""
    try:
        logger.info("Processing integrated prediction")
        
        # Phase 1: Simplified delay prediction (rule-based)
        clinical_data = request.trial_data
        
        # Simple risk calculation based on key factors
        age = clinical_data.get('Age', 65)
        adherence = clinical_data.get('Medication_Adherence', 80)
        adr_rate = clinical_data.get('ADR_Rate', 0.1)
        bp_sys = clinical_data.get('BP_Systolic', 130)
        alt = clinical_data.get('ALT_Level', 25)
        
        # Calculate delay probability
        delay_probability = 0.3  # Base risk
        
        if age > 70:
            delay_probability += 0.1
        if adherence < 70:
            delay_probability += 0.2
        if adr_rate > 0.3:
            delay_probability += 0.2
        if bp_sys > 140:
            delay_probability += 0.1
        if alt > 40:
            delay_probability += 0.15
        
        delay_probability = min(delay_probability, 1.0)
        delay_prediction = "delayed" if delay_probability > 0.5 else "on_time"
        
        # Risk factors
        risk_factors = []
        if delay_probability > 0.7:
            risk_factors.append("High delay probability")
        if adherence < 70:
            risk_factors.append("Low medication adherence")
        if adr_rate > 0.3:
            risk_factors.append("High adverse event rate")
        if bp_sys > 140:
            risk_factors.append("Hypertension risk")
        if alt > 40:
            risk_factors.append("Elevated liver enzymes")
        
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
        if nlp_results and nlp_results.get("severity_classification") == "severe":
            regulatory_flags.append("SEVERE_AE_DETECTED")
        if adherence < 50:
            regulatory_flags.append("POOR_COMPLIANCE")
        
        # Prepare response
        response = IntegratedPredictionResponse(
            delay_probability=delay_probability,
            delay_prediction=delay_prediction,
            confidence=abs(delay_probability - 0.5) * 2,
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
                    "nlp_processed": request.include_nlp and request.clinical_text is not None
                }
            )
        
        logger.info(f"✅ Integrated prediction completed: {delay_prediction} ({delay_probability:.3f})")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in integrated prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Integrated prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "nlp_engine": nlp_engine is not None,
            "blockchain_ledger": blockchain_ledger is not None
        },
        "blockchain_stats": {
            "total_blocks": len(blockchain_ledger.chain) if blockchain_ledger else 0,
            "integrity_verified": blockchain_ledger.verify_chain_integrity()["is_valid"] if blockchain_ledger else False
        }
    }

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
    uvicorn.run("phase2_simple_api:app", host="0.0.0.0", port=8005, reload=True)
