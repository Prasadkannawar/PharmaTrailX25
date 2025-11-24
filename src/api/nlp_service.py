#!/usr/bin/env python3
"""
FastAPI NLP Service for PharmaTrail-X Phase 2
Clinical NLP endpoints with blockchain audit logging
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

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.nlp.clinical_nlp_engine import ClinicalNLPEngine, AdverseEventExtraction, ClinicalSummary
from src.blockchain.audit_ledger import PharmaBlockchain, AuditLogger

# Initialize FastAPI app
app = FastAPI(
    title="PharmaTrail-X NLP Service",
    description="Clinical-grade NLP engine with BioBERT/ClinicalBERT for adverse event extraction and text summarization",
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

# Pydantic models
class NLPRequest(BaseModel):
    text: str = Field(..., description="Clinical text to process")
    trial_id: str = Field(..., description="Clinical trial identifier")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    document_type: Optional[str] = Field("clinical_note", description="Type of clinical document")
    include_audit: bool = Field(True, description="Whether to log to blockchain audit trail")

class AdverseEventRequest(NLPRequest):
    extract_severity: bool = Field(True, description="Extract severity classifications")
    extract_lab_values: bool = Field(True, description="Extract lab value abnormalities")

class SummarizationRequest(NLPRequest):
    max_length: int = Field(150, description="Maximum length of summary")
    extract_key_points: bool = Field(True, description="Extract key clinical points")

class AdverseEventResponse(BaseModel):
    trial_id: str
    patient_id: Optional[str]
    entities: List[Dict[str, Any]]
    ae_events: List[Dict[str, Any]]
    severity_classification: str
    confidence_score: float
    processing_timestamp: str
    model_version: str
    audit_hash: Optional[str] = None

class SummarizationResponse(BaseModel):
    trial_id: str
    patient_id: Optional[str]
    original_text_length: int
    summary: str
    key_points: List[str]
    word_count_reduction: float
    confidence_score: float
    processing_timestamp: str
    audit_hash: Optional[str] = None

class NLPHealthResponse(BaseModel):
    status: str
    nlp_engine_loaded: bool
    blockchain_active: bool
    model_info: Dict[str, Any]
    total_processed: int
    uptime: str

# Global counters
processing_stats = {
    "ae_extractions": 0,
    "summarizations": 0,
    "total_processed": 0,
    "start_time": datetime.utcnow()
}

@app.on_event("startup")
async def startup_event():
    """Initialize NLP engine and blockchain"""
    global nlp_engine, blockchain, audit_logger
    
    logger.info("🚀 Starting PharmaTrail-X NLP Service...")
    
    try:
        # Initialize NLP engine
        logger.info("Loading clinical NLP engine...")
        nlp_engine = ClinicalNLPEngine()
        
        # Initialize blockchain
        logger.info("Initializing blockchain audit ledger...")
        blockchain = PharmaBlockchain("pharmatrail-x-nlp-audit")
        audit_logger = AuditLogger(blockchain)
        
        # Log startup event
        startup_data = {
            "service": "nlp_service",
            "version": "2.0.0",
            "models_loaded": nlp_engine.get_model_info()["models_loaded"],
            "startup_time": datetime.utcnow().isoformat()
        }
        
        audit_logger.log_system_event(
            trial_id="SYSTEM",
            system_data={
                "event": "service_startup",
                "component": "nlp_service",
                "status": "success",
                "details": startup_data
            }
        )
        
        logger.info("✅ NLP Service startup complete")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "PharmaTrail-X NLP Service",
        "version": "2.0.0",
        "status": "running",
        "capabilities": [
            "adverse_event_extraction",
            "clinical_text_summarization",
            "entity_recognition",
            "severity_classification",
            "blockchain_audit_logging"
        ],
        "endpoints": {
            "adverse_events": "/nlp/ae",
            "summarization": "/nlp/summary",
            "health": "/nlp/health",
            "model_info": "/nlp/model_info"
        }
    }

@app.post("/nlp/ae", response_model=AdverseEventResponse)
async def extract_adverse_events(
    request: AdverseEventRequest,
    background_tasks: BackgroundTasks
):
    """
    Extract adverse events from clinical text using BioBERT/ClinicalBERT
    """
    if not nlp_engine:
        raise HTTPException(status_code=503, detail="NLP engine not initialized")
    
    try:
        logger.info(f"Processing AE extraction for trial {request.trial_id}")
        
        # Extract adverse events
        extraction_result = nlp_engine.extract_adverse_events(
            text=request.text,
            trial_id=request.trial_id
        )
        
        # Prepare response
        response = AdverseEventResponse(
            trial_id=request.trial_id,
            patient_id=request.patient_id,
            entities=[
                {
                    "text": entity.text,
                    "label": entity.label,
                    "start": entity.start,
                    "end": entity.end,
                    "confidence": entity.confidence,
                    "severity": entity.severity,
                    "category": entity.category
                }
                for entity in extraction_result.entities
            ],
            ae_events=extraction_result.ae_events,
            severity_classification=extraction_result.severity_classification,
            confidence_score=extraction_result.confidence_score,
            processing_timestamp=extraction_result.processing_timestamp,
            model_version=extraction_result.model_version
        )
        
        # Log to blockchain if requested
        if request.include_audit and audit_logger:
            text_hash = hashlib.sha256(request.text.encode()).hexdigest()
            
            audit_data = {
                "type": "adverse_event_extraction",
                "entity_count": len(extraction_result.entities),
                "ae_events": extraction_result.ae_events,
                "confidence": extraction_result.confidence_score,
                "model_version": extraction_result.model_version,
                "text_hash": text_hash,
                "patient_id": request.patient_id,
                "document_type": request.document_type
            }
            
            # Log in background to avoid blocking response
            background_tasks.add_task(
                log_nlp_event,
                request.trial_id,
                audit_data
            )
        
        # Update stats
        processing_stats["ae_extractions"] += 1
        processing_stats["total_processed"] += 1
        
        logger.info(f"✅ AE extraction completed for trial {request.trial_id}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in AE extraction: {e}")
        raise HTTPException(status_code=500, detail=f"AE extraction failed: {str(e)}")

@app.post("/nlp/summary", response_model=SummarizationResponse)
async def summarize_clinical_text(
    request: SummarizationRequest,
    background_tasks: BackgroundTasks
):
    """
    Summarize clinical text and extract key points
    """
    if not nlp_engine:
        raise HTTPException(status_code=503, detail="NLP engine not initialized")
    
    try:
        logger.info(f"Processing text summarization for trial {request.trial_id}")
        
        # Summarize text
        summary_result = nlp_engine.summarize_clinical_text(
            text=request.text,
            max_length=request.max_length
        )
        
        # Prepare response
        response = SummarizationResponse(
            trial_id=request.trial_id,
            patient_id=request.patient_id,
            original_text_length=len(request.text),
            summary=summary_result.summary,
            key_points=summary_result.key_points,
            word_count_reduction=summary_result.word_count_reduction,
            confidence_score=summary_result.confidence_score,
            processing_timestamp=summary_result.processing_timestamp
        )
        
        # Log to blockchain if requested
        if request.include_audit and audit_logger:
            text_hash = hashlib.sha256(request.text.encode()).hexdigest()
            
            audit_data = {
                "type": "text_summarization",
                "original_length": len(request.text),
                "summary_length": len(summary_result.summary),
                "reduction_ratio": summary_result.word_count_reduction,
                "confidence": summary_result.confidence_score,
                "text_hash": text_hash,
                "patient_id": request.patient_id,
                "document_type": request.document_type
            }
            
            # Log in background
            background_tasks.add_task(
                log_nlp_event,
                request.trial_id,
                audit_data
            )
        
        # Update stats
        processing_stats["summarizations"] += 1
        processing_stats["total_processed"] += 1
        
        logger.info(f"✅ Text summarization completed for trial {request.trial_id}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in text summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.get("/nlp/health", response_model=NLPHealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = datetime.utcnow() - processing_stats["start_time"]
    
    return NLPHealthResponse(
        status="healthy" if nlp_engine and blockchain else "degraded",
        nlp_engine_loaded=nlp_engine is not None,
        blockchain_active=blockchain is not None,
        model_info=nlp_engine.get_model_info() if nlp_engine else {},
        total_processed=processing_stats["total_processed"],
        uptime=str(uptime)
    )

@app.get("/nlp/model_info")
async def get_model_info():
    """Get information about loaded NLP models"""
    if not nlp_engine:
        raise HTTPException(status_code=503, detail="NLP engine not initialized")
    
    model_info = nlp_engine.get_model_info()
    
    return {
        "service_version": "2.0.0",
        "nlp_engine": model_info,
        "processing_stats": processing_stats,
        "blockchain_active": blockchain is not None,
        "audit_logging": audit_logger is not None
    }

@app.post("/nlp/batch_ae")
async def batch_adverse_event_extraction(
    requests: List[AdverseEventRequest],
    background_tasks: BackgroundTasks
):
    """
    Process multiple AE extraction requests in batch
    """
    if not nlp_engine:
        raise HTTPException(status_code=503, detail="NLP engine not initialized")
    
    if len(requests) > 50:
        raise HTTPException(status_code=400, detail="Batch size limited to 50 requests")
    
    results = []
    
    for req in requests:
        try:
            # Process each request
            extraction_result = nlp_engine.extract_adverse_events(
                text=req.text,
                trial_id=req.trial_id
            )
            
            response = {
                "trial_id": req.trial_id,
                "patient_id": req.patient_id,
                "status": "success",
                "entities": len(extraction_result.entities),
                "ae_events": len(extraction_result.ae_events),
                "confidence": extraction_result.confidence_score
            }
            
            # Log to blockchain in background
            if req.include_audit and audit_logger:
                text_hash = hashlib.sha256(req.text.encode()).hexdigest()
                audit_data = {
                    "type": "batch_ae_extraction",
                    "entity_count": len(extraction_result.entities),
                    "ae_events": extraction_result.ae_events,
                    "text_hash": text_hash,
                    "batch_processing": True
                }
                
                background_tasks.add_task(
                    log_nlp_event,
                    req.trial_id,
                    audit_data
                )
            
            results.append(response)
            processing_stats["ae_extractions"] += 1
            
        except Exception as e:
            results.append({
                "trial_id": req.trial_id,
                "patient_id": req.patient_id,
                "status": "error",
                "error": str(e)
            })
    
    processing_stats["total_processed"] += len(requests)
    
    return {
        "batch_size": len(requests),
        "successful": len([r for r in results if r.get("status") == "success"]),
        "failed": len([r for r in results if r.get("status") == "error"]),
        "results": results,
        "processing_timestamp": datetime.utcnow().isoformat()
    }

@app.get("/nlp/stats")
async def get_processing_stats():
    """Get processing statistics"""
    uptime = datetime.utcnow() - processing_stats["start_time"]
    
    return {
        "service_uptime": str(uptime),
        "total_processed": processing_stats["total_processed"],
        "ae_extractions": processing_stats["ae_extractions"],
        "summarizations": processing_stats["summarizations"],
        "blockchain_blocks": len(blockchain.chain) if blockchain else 0,
        "average_per_hour": processing_stats["total_processed"] / max(uptime.total_seconds() / 3600, 1),
        "service_health": "healthy" if nlp_engine and blockchain else "degraded"
    }

# Background task functions
async def log_nlp_event(trial_id: str, audit_data: Dict[str, Any]):
    """Background task to log NLP events to blockchain"""
    try:
        if audit_logger:
            audit_logger.log_nlp_extraction(trial_id, audit_data)
    except Exception as e:
        logger.error(f"Error logging NLP event to blockchain: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("nlp_service:app", host="0.0.0.0", port=8002, reload=True)
