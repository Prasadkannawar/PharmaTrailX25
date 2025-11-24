#!/usr/bin/env python3
"""
Blockchain Audit Ledger for PharmaTrail-X Phase 2
Lightweight permissioned blockchain for regulatory compliance
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import threading
from pathlib import Path
from loguru import logger

@dataclass
class AuditBlock:
    """Individual block in the audit chain"""
    index: int
    timestamp: str
    trial_id: str
    event_type: str
    event_payload: Dict[str, Any]
    event_payload_hash: str
    previous_hash: str
    block_hash: str
    nonce: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary"""
        return asdict(self)

@dataclass
class ChainMetadata:
    """Metadata about the blockchain"""
    chain_id: str
    genesis_timestamp: str
    total_blocks: int
    last_block_hash: str
    integrity_verified: bool
    last_verification: str

class PharmaBlockchain:
    """
    Lightweight permissioned blockchain for clinical trial audit logging
    Designed for data integrity, not cryptocurrency
    """
    
    def __init__(self, chain_id: str = "pharmatrail-x-audit"):
        self.chain_id = chain_id
        self.chain: List[AuditBlock] = []
        self.pending_transactions: List[Dict[str, Any]] = []
        self.difficulty = 2  # Low difficulty for fast writes
        self.lock = threading.Lock()
        
        # Storage
        self.storage_path = Path("blockchain_data")
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize or load existing chain
        self._initialize_chain()
        
        logger.info(f"✅ PharmaBlockchain initialized: {chain_id}")
    
    def _initialize_chain(self):
        """Initialize blockchain or load existing chain"""
        chain_file = self.storage_path / f"{self.chain_id}.json"
        
        if chain_file.exists():
            self._load_chain_from_file()
        else:
            self._create_genesis_block()
            self._save_chain_to_file()
    
    def _create_genesis_block(self):
        """Create the genesis block"""
        genesis_payload = {
            "message": "PharmaTrail-X Audit Ledger Genesis Block",
            "version": "2.0.0",
            "created_by": "PharmaTrail-X System",
            "purpose": "Regulatory compliance and data integrity"
        }
        
        genesis_block = AuditBlock(
            index=0,
            timestamp=datetime.utcnow().isoformat(),
            trial_id="GENESIS",
            event_type="genesis",
            event_payload=genesis_payload,
            event_payload_hash=self._calculate_hash(json.dumps(genesis_payload, sort_keys=True)),
            previous_hash="0",
            block_hash="",
            nonce=0
        )
        
        # Mine the genesis block
        genesis_block.block_hash = self._mine_block(genesis_block)
        self.chain.append(genesis_block)
        
        logger.info("🎯 Genesis block created")
    
    def log_event(self, trial_id: str, event_type: str, event_payload: Dict[str, Any]) -> str:
        """
        Log a new event to the blockchain
        Returns the block hash of the created block
        """
        with self.lock:
            try:
                # Validate input
                if not trial_id or not event_type:
                    raise ValueError("trial_id and event_type are required")
                
                # Add metadata to payload
                enhanced_payload = {
                    **event_payload,
                    "logged_at": datetime.utcnow().isoformat(),
                    "system_version": "PharmaTrail-X v2.0",
                    "chain_id": self.chain_id
                }
                
                # Calculate payload hash
                payload_json = json.dumps(enhanced_payload, sort_keys=True)
                payload_hash = self._calculate_hash(payload_json)
                
                # Get previous block hash
                previous_hash = self.chain[-1].block_hash if self.chain else "0"
                
                # Create new block
                new_block = AuditBlock(
                    index=len(self.chain),
                    timestamp=datetime.utcnow().isoformat(),
                    trial_id=trial_id,
                    event_type=event_type,
                    event_payload=enhanced_payload,
                    event_payload_hash=payload_hash,
                    previous_hash=previous_hash,
                    block_hash="",
                    nonce=0
                )
                
                # Mine the block (find valid hash)
                new_block.block_hash = self._mine_block(new_block)
                
                # Add to chain
                self.chain.append(new_block)
                
                # Save to persistent storage
                self._save_chain_to_file()
                
                logger.info(f"✅ Event logged: {event_type} for trial {trial_id} (Block #{new_block.index})")
                return new_block.block_hash
                
            except Exception as e:
                logger.error(f"❌ Error logging event: {e}")
                raise
    
    def get_chain(self, trial_id: Optional[str] = None, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve the blockchain or filtered subset
        """
        chain_data = [block.to_dict() for block in self.chain]
        
        # Apply filters
        if trial_id:
            chain_data = [block for block in chain_data if block['trial_id'] == trial_id]
        
        if event_type:
            chain_data = [block for block in chain_data if block['event_type'] == event_type]
        
        return chain_data
    
    def verify_chain_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the entire blockchain
        """
        logger.info("🔍 Verifying blockchain integrity...")
        
        verification_result = {
            "is_valid": True,
            "total_blocks": len(self.chain),
            "errors": [],
            "verification_timestamp": datetime.utcnow().isoformat()
        }
        
        for i, block in enumerate(self.chain):
            # Verify block hash
            calculated_hash = self._calculate_block_hash(block)
            if calculated_hash != block.block_hash:
                verification_result["is_valid"] = False
                verification_result["errors"].append(f"Block {i}: Invalid block hash")
            
            # Verify previous hash linkage (except genesis)
            if i > 0:
                if block.previous_hash != self.chain[i-1].block_hash:
                    verification_result["is_valid"] = False
                    verification_result["errors"].append(f"Block {i}: Invalid previous hash linkage")
            
            # Verify payload hash
            payload_json = json.dumps(block.event_payload, sort_keys=True)
            calculated_payload_hash = self._calculate_hash(payload_json)
            if calculated_payload_hash != block.event_payload_hash:
                verification_result["is_valid"] = False
                verification_result["errors"].append(f"Block {i}: Invalid payload hash")
        
        status = "✅ VALID" if verification_result["is_valid"] else "❌ INVALID"
        logger.info(f"Blockchain verification complete: {status}")
        
        return verification_result
    
    def get_chain_metadata(self) -> ChainMetadata:
        """Get metadata about the blockchain"""
        integrity_check = self.verify_chain_integrity()
        
        return ChainMetadata(
            chain_id=self.chain_id,
            genesis_timestamp=self.chain[0].timestamp if self.chain else "",
            total_blocks=len(self.chain),
            last_block_hash=self.chain[-1].block_hash if self.chain else "",
            integrity_verified=integrity_check["is_valid"],
            last_verification=integrity_check["verification_timestamp"]
        )
    
    def get_trial_audit_trail(self, trial_id: str) -> Dict[str, Any]:
        """Get complete audit trail for a specific trial"""
        trial_blocks = [block for block in self.chain if block.trial_id == trial_id]
        
        # Organize by event type
        events_by_type = {}
        for block in trial_blocks:
            event_type = block.event_type
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            
            events_by_type[event_type].append({
                "timestamp": block.timestamp,
                "payload": block.event_payload,
                "block_hash": block.block_hash,
                "block_index": block.index
            })
        
        return {
            "trial_id": trial_id,
            "total_events": len(trial_blocks),
            "events_by_type": events_by_type,
            "first_event": trial_blocks[0].timestamp if trial_blocks else None,
            "last_event": trial_blocks[-1].timestamp if trial_blocks else None
        }
    
    def _mine_block(self, block: AuditBlock) -> str:
        """
        Mine a block by finding a hash that meets the difficulty requirement
        """
        target = "0" * self.difficulty
        
        while True:
            block_hash = self._calculate_block_hash(block)
            
            if block_hash.startswith(target):
                return block_hash
            
            block.nonce += 1
            
            # Prevent infinite loops in case of issues
            if block.nonce > 1000000:
                logger.warning(f"Mining took too long, reducing difficulty")
                self.difficulty = max(1, self.difficulty - 1)
                target = "0" * self.difficulty
                block.nonce = 0
    
    def _calculate_block_hash(self, block: AuditBlock) -> str:
        """Calculate hash for a block"""
        block_string = (
            f"{block.index}{block.timestamp}{block.trial_id}{block.event_type}"
            f"{block.event_payload_hash}{block.previous_hash}{block.nonce}"
        )
        return self._calculate_hash(block_string)
    
    def _calculate_hash(self, data: str) -> str:
        """Calculate SHA-256 hash of data"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _save_chain_to_file(self):
        """Save blockchain to persistent storage"""
        try:
            chain_file = self.storage_path / f"{self.chain_id}.json"
            chain_data = {
                "chain_id": self.chain_id,
                "blocks": [block.to_dict() for block in self.chain],
                "metadata": {
                    "total_blocks": len(self.chain),
                    "last_updated": datetime.utcnow().isoformat()
                }
            }
            
            with open(chain_file, 'w') as f:
                json.dump(chain_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving chain to file: {e}")
    
    def _load_chain_from_file(self):
        """Load blockchain from persistent storage"""
        try:
            chain_file = self.storage_path / f"{self.chain_id}.json"
            
            with open(chain_file, 'r') as f:
                chain_data = json.load(f)
            
            # Reconstruct blocks
            self.chain = []
            for block_data in chain_data["blocks"]:
                block = AuditBlock(**block_data)
                self.chain.append(block)
            
            logger.info(f"✅ Loaded existing blockchain with {len(self.chain)} blocks")
            
            # Verify integrity after loading
            integrity = self.verify_chain_integrity()
            if not integrity["is_valid"]:
                logger.error("❌ Loaded blockchain failed integrity check!")
                raise ValueError("Blockchain integrity compromised")
                
        except Exception as e:
            logger.error(f"Error loading chain from file: {e}")
            # Create new chain if loading fails
            self._create_genesis_block()

class AuditLogger:
    """
    High-level interface for logging various types of audit events
    """
    
    def __init__(self, blockchain: PharmaBlockchain):
        self.blockchain = blockchain
    
    def log_model_prediction(self, trial_id: str, prediction_data: Dict[str, Any]) -> str:
        """Log a model prediction event"""
        return self.blockchain.log_event(
            trial_id=trial_id,
            event_type="model_prediction",
            event_payload={
                "model_type": prediction_data.get("model_type", "unknown"),
                "prediction_result": prediction_data.get("prediction", {}),
                "input_features": prediction_data.get("features", {}),
                "confidence_score": prediction_data.get("confidence", 0.0),
                "model_version": prediction_data.get("model_version", "unknown")
            }
        )
    
    def log_data_ingestion(self, trial_id: str, ingestion_data: Dict[str, Any]) -> str:
        """Log a data ingestion event"""
        return self.blockchain.log_event(
            trial_id=trial_id,
            event_type="data_ingestion",
            event_payload={
                "data_source": ingestion_data.get("source", "unknown"),
                "records_processed": ingestion_data.get("record_count", 0),
                "data_hash": ingestion_data.get("data_hash", ""),
                "processing_status": ingestion_data.get("status", "unknown"),
                "file_metadata": ingestion_data.get("metadata", {})
            }
        )
    
    def log_nlp_extraction(self, trial_id: str, nlp_data: Dict[str, Any]) -> str:
        """Log an NLP extraction event"""
        return self.blockchain.log_event(
            trial_id=trial_id,
            event_type="nlp_extraction",
            event_payload={
                "extraction_type": nlp_data.get("type", "unknown"),
                "entities_extracted": nlp_data.get("entity_count", 0),
                "adverse_events": nlp_data.get("ae_events", []),
                "confidence_score": nlp_data.get("confidence", 0.0),
                "model_version": nlp_data.get("model_version", "unknown"),
                "text_hash": nlp_data.get("text_hash", "")
            }
        )
    
    def log_protocol_deviation(self, trial_id: str, deviation_data: Dict[str, Any]) -> str:
        """Log a protocol deviation event"""
        return self.blockchain.log_event(
            trial_id=trial_id,
            event_type="protocol_deviation",
            event_payload={
                "deviation_type": deviation_data.get("type", "unknown"),
                "severity": deviation_data.get("severity", "unknown"),
                "description": deviation_data.get("description", ""),
                "patient_id": deviation_data.get("patient_id", ""),
                "site_id": deviation_data.get("site_id", ""),
                "corrective_action": deviation_data.get("corrective_action", "")
            }
        )
    
    def log_adverse_event(self, trial_id: str, ae_data: Dict[str, Any]) -> str:
        """Log an adverse event"""
        return self.blockchain.log_event(
            trial_id=trial_id,
            event_type="adverse_event",
            event_payload={
                "ae_term": ae_data.get("term", ""),
                "severity": ae_data.get("severity", "unknown"),
                "seriousness": ae_data.get("seriousness", "non-serious"),
                "patient_id": ae_data.get("patient_id", ""),
                "onset_date": ae_data.get("onset_date", ""),
                "outcome": ae_data.get("outcome", "unknown"),
                "causality": ae_data.get("causality", "unknown")
            }
        )
    
    def log_system_event(self, trial_id: str, system_data: Dict[str, Any]) -> str:
        """Log a system event"""
        return self.blockchain.log_event(
            trial_id=trial_id,
            event_type="system_event",
            event_payload={
                "event_name": system_data.get("event", "unknown"),
                "component": system_data.get("component", "system"),
                "status": system_data.get("status", "unknown"),
                "details": system_data.get("details", {}),
                "user_id": system_data.get("user_id", "system")
            }
        )
