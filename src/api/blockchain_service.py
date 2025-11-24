#!/usr/bin/env python3
"""
FastAPI Blockchain Service for PharmaTrail-X Phase 2
Blockchain audit ledger endpoints for regulatory compliance
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.blockchain.audit_ledger import PharmaBlockchain, AuditLogger, ChainMetadata

# Initialize FastAPI app
app = FastAPI(
    title="PharmaTrail-X Blockchain Service",
    description="Immutable audit ledger for regulatory compliance and data integrity",
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
blockchain: Optional[PharmaBlockchain] = None
audit_logger: Optional[AuditLogger] = None

# Pydantic models
class LogEventRequest(BaseModel):
    trial_id: str = Field(..., description="Clinical trial identifier")
    event_type: str = Field(..., description="Type of event being logged")
    event_payload: Dict[str, Any] = Field(..., description="Event data payload")

class LogEventResponse(BaseModel):
    success: bool
    block_hash: str
    block_index: int
    timestamp: str
    trial_id: str
    event_type: str

class ChainResponse(BaseModel):
    chain_id: str
    total_blocks: int
    blocks: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class IntegrityCheckResponse(BaseModel):
    is_valid: bool
    total_blocks: int
    errors: List[str]
    verification_timestamp: str

class AuditTrailResponse(BaseModel):
    trial_id: str
    total_events: int
    events_by_type: Dict[str, List[Dict[str, Any]]]
    first_event: Optional[str]
    last_event: Optional[str]

@app.on_event("startup")
async def startup_event():
    """Initialize blockchain service"""
    global blockchain, audit_logger
    
    logger.info("🚀 Starting PharmaTrail-X Blockchain Service...")
    
    try:
        # Initialize blockchain
        blockchain = PharmaBlockchain("pharmatrail-x-main-audit")
        audit_logger = AuditLogger(blockchain)
        
        # Log service startup
        startup_data = {
            "service": "blockchain_service",
            "version": "2.0.0",
            "startup_time": datetime.utcnow().isoformat(),
            "chain_id": blockchain.chain_id
        }
        
        audit_logger.log_system_event(
            trial_id="SYSTEM",
            system_data={
                "event": "blockchain_service_startup",
                "component": "blockchain_service",
                "status": "success",
                "details": startup_data
            }
        )
        
        logger.info("✅ Blockchain Service startup complete")
        
    except Exception as e:
        logger.error(f"❌ Error during blockchain service startup: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    metadata = blockchain.get_chain_metadata() if blockchain else None
    
    return {
        "service": "PharmaTrail-X Blockchain Service",
        "version": "2.0.0",
        "status": "running",
        "purpose": "Immutable audit logging for regulatory compliance",
        "chain_metadata": {
            "chain_id": metadata.chain_id if metadata else "not_initialized",
            "total_blocks": metadata.total_blocks if metadata else 0,
            "integrity_verified": metadata.integrity_verified if metadata else False
        },
        "endpoints": {
            "log_event": "/blockchain/log_event",
            "get_chain": "/blockchain/get_chain",
            "verify_integrity": "/blockchain/verify_integrity",
            "audit_trail": "/blockchain/audit_trail/{trial_id}"
        }
    }

@app.post("/blockchain/log_event", response_model=LogEventResponse)
async def log_event(request: LogEventRequest):
    """
    Log a new event to the blockchain audit ledger
    """
    if not blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    try:
        logger.info(f"Logging event: {request.event_type} for trial {request.trial_id}")
        
        # Log event to blockchain
        block_hash = blockchain.log_event(
            trial_id=request.trial_id,
            event_type=request.event_type,
            event_payload=request.event_payload
        )
        
        # Get the latest block for response
        latest_block = blockchain.chain[-1]
        
        response = LogEventResponse(
            success=True,
            block_hash=block_hash,
            block_index=latest_block.index,
            timestamp=latest_block.timestamp,
            trial_id=request.trial_id,
            event_type=request.event_type
        )
        
        logger.info(f"✅ Event logged successfully: Block #{latest_block.index}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error logging event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log event: {str(e)}")

@app.get("/blockchain/get_chain", response_model=ChainResponse)
async def get_chain(
    trial_id: Optional[str] = Query(None, description="Filter by trial ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    limit: Optional[int] = Query(None, description="Limit number of blocks returned")
):
    """
    Retrieve the blockchain or filtered subset
    """
    if not blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    try:
        # Get filtered chain
        chain_data = blockchain.get_chain(trial_id=trial_id, event_type=event_type)
        
        # Apply limit if specified
        if limit and limit > 0:
            chain_data = chain_data[-limit:]  # Get most recent blocks
        
        # Get metadata
        metadata = blockchain.get_chain_metadata()
        
        response = ChainResponse(
            chain_id=blockchain.chain_id,
            total_blocks=len(chain_data),
            blocks=chain_data,
            metadata={
                "full_chain_blocks": metadata.total_blocks,
                "genesis_timestamp": metadata.genesis_timestamp,
                "last_block_hash": metadata.last_block_hash,
                "integrity_verified": metadata.integrity_verified,
                "filters_applied": {
                    "trial_id": trial_id,
                    "event_type": event_type,
                    "limit": limit
                }
            }
        )
        
        logger.info(f"✅ Retrieved {len(chain_data)} blocks from blockchain")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error retrieving chain: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chain: {str(e)}")

@app.get("/blockchain/verify_integrity", response_model=IntegrityCheckResponse)
async def verify_chain_integrity():
    """
    Verify the integrity of the entire blockchain
    """
    if not blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    try:
        logger.info("🔍 Performing blockchain integrity verification...")
        
        verification_result = blockchain.verify_chain_integrity()
        
        response = IntegrityCheckResponse(
            is_valid=verification_result["is_valid"],
            total_blocks=verification_result["total_blocks"],
            errors=verification_result["errors"],
            verification_timestamp=verification_result["verification_timestamp"]
        )
        
        status = "✅ VALID" if verification_result["is_valid"] else "❌ INVALID"
        logger.info(f"Integrity verification complete: {status}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error during integrity verification: {e}")
        raise HTTPException(status_code=500, detail=f"Integrity verification failed: {str(e)}")

@app.get("/blockchain/audit_trail/{trial_id}", response_model=AuditTrailResponse)
async def get_trial_audit_trail(trial_id: str):
    """
    Get complete audit trail for a specific trial
    """
    if not blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    try:
        logger.info(f"Retrieving audit trail for trial: {trial_id}")
        
        audit_trail = blockchain.get_trial_audit_trail(trial_id)
        
        response = AuditTrailResponse(
            trial_id=audit_trail["trial_id"],
            total_events=audit_trail["total_events"],
            events_by_type=audit_trail["events_by_type"],
            first_event=audit_trail["first_event"],
            last_event=audit_trail["last_event"]
        )
        
        logger.info(f"✅ Retrieved audit trail: {audit_trail['total_events']} events")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error retrieving audit trail: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve audit trail: {str(e)}")

@app.get("/blockchain/metadata")
async def get_chain_metadata():
    """Get blockchain metadata"""
    if not blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    metadata = blockchain.get_chain_metadata()
    
    return {
        "chain_id": metadata.chain_id,
        "genesis_timestamp": metadata.genesis_timestamp,
        "total_blocks": metadata.total_blocks,
        "last_block_hash": metadata.last_block_hash,
        "integrity_verified": metadata.integrity_verified,
        "last_verification": metadata.last_verification,
        "storage_location": str(blockchain.storage_path),
        "difficulty": blockchain.difficulty
    }

@app.get("/blockchain/stats")
async def get_blockchain_stats():
    """Get blockchain statistics"""
    if not blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    # Count events by type
    event_types = {}
    trial_counts = {}
    
    for block in blockchain.chain:
        event_type = block.event_type
        trial_id = block.trial_id
        
        event_types[event_type] = event_types.get(event_type, 0) + 1
        trial_counts[trial_id] = trial_counts.get(trial_id, 0) + 1
    
    return {
        "total_blocks": len(blockchain.chain),
        "unique_trials": len(trial_counts),
        "events_by_type": event_types,
        "top_trials": dict(sorted(trial_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        "chain_size_mb": blockchain.storage_path.stat().st_size / (1024 * 1024) if blockchain.storage_path.exists() else 0,
        "average_block_time": "~1 second",  # Based on our mining difficulty
        "integrity_status": blockchain.verify_chain_integrity()["is_valid"]
    }

@app.post("/blockchain/bulk_log")
async def bulk_log_events(events: List[LogEventRequest]):
    """
    Log multiple events in bulk
    """
    if not blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    if len(events) > 100:
        raise HTTPException(status_code=400, detail="Bulk logging limited to 100 events")
    
    results = []
    
    for event in events:
        try:
            block_hash = blockchain.log_event(
                trial_id=event.trial_id,
                event_type=event.event_type,
                event_payload=event.event_payload
            )
            
            latest_block = blockchain.chain[-1]
            
            results.append({
                "success": True,
                "trial_id": event.trial_id,
                "event_type": event.event_type,
                "block_hash": block_hash,
                "block_index": latest_block.index
            })
            
        except Exception as e:
            results.append({
                "success": False,
                "trial_id": event.trial_id,
                "event_type": event.event_type,
                "error": str(e)
            })
    
    successful = len([r for r in results if r["success"]])
    failed = len([r for r in results if not r["success"]])
    
    return {
        "total_events": len(events),
        "successful": successful,
        "failed": failed,
        "results": results,
        "processing_timestamp": datetime.utcnow().isoformat()
    }

@app.get("/blockchain/health")
async def health_check():
    """Health check endpoint"""
    if not blockchain:
        return {"status": "unhealthy", "error": "Blockchain not initialized"}
    
    try:
        # Quick integrity check on recent blocks
        recent_blocks = blockchain.chain[-10:] if len(blockchain.chain) > 10 else blockchain.chain
        integrity_sample = True
        
        for i, block in enumerate(recent_blocks[1:], 1):
            if block.previous_hash != recent_blocks[i-1].block_hash:
                integrity_sample = False
                break
        
        return {
            "status": "healthy",
            "blockchain_active": True,
            "total_blocks": len(blockchain.chain),
            "integrity_sample_check": integrity_sample,
            "chain_id": blockchain.chain_id,
            "last_block_time": blockchain.chain[-1].timestamp if blockchain.chain else None
        }
        
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "blockchain_active": False
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("blockchain_service:app", host="0.0.0.0", port=8003, reload=True)
