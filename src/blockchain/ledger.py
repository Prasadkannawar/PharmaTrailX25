#!/usr/bin/env python3
"""
Blockchain Ledger for PharmaTrail-X Phase 2
Simple but functional blockchain for audit logging
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from loguru import logger

@dataclass
class Block:
    """Individual block in the blockchain"""
    index: int
    timestamp: str
    data: Dict[str, Any]
    previous_hash: str
    hash: str
    nonce: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary"""
        return asdict(self)

class BlockchainLedger:
    """
    Simple blockchain ledger for audit logging
    """
    
    def __init__(self, chain_id: str = "pharmatrail-x"):
        self.chain_id = chain_id
        self.chain: List[Block] = []
        self.storage_path = Path("blockchain_data")
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize or load existing chain
        self._initialize_chain()
        
        logger.info(f"✅ Blockchain ledger initialized: {chain_id}")
    
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
        genesis_data = {
            "message": "PharmaTrail-X Genesis Block",
            "version": "2.0.0",
            "created_at": datetime.utcnow().isoformat(),
            "purpose": "Clinical trial audit logging"
        }
        
        genesis_block = Block(
            index=0,
            timestamp=datetime.utcnow().isoformat(),
            data=genesis_data,
            previous_hash="0",
            hash="",
            nonce=0
        )
        
        # Calculate hash for genesis block
        genesis_block.hash = self._calculate_hash(genesis_block)
        self.chain.append(genesis_block)
        
        logger.info("🎯 Genesis block created")
    
    def append_block(self, data: Dict[str, Any]) -> str:
        """
        Add a new block to the chain
        Returns the hash of the created block
        """
        try:
            # Get previous block hash
            previous_hash = self.chain[-1].hash if self.chain else "0"
            
            # Create new block
            new_block = Block(
                index=len(self.chain),
                timestamp=datetime.utcnow().isoformat(),
                data=data,
                previous_hash=previous_hash,
                hash="",
                nonce=0
            )
            
            # Calculate hash
            new_block.hash = self._calculate_hash(new_block)
            
            # Add to chain
            self.chain.append(new_block)
            
            # Save to file
            self._save_chain_to_file()
            
            logger.info(f"✅ Block #{new_block.index} added to chain")
            return new_block.hash
            
        except Exception as e:
            logger.error(f"❌ Error adding block: {e}")
            raise
    
    def get_chain(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the blockchain as a list of dictionaries
        """
        chain_data = [block.to_dict() for block in self.chain]
        
        if limit:
            return chain_data[-limit:]  # Return most recent blocks
        
        return chain_data
    
    def get_block_by_hash(self, block_hash: str) -> Optional[Dict[str, Any]]:
        """Get a specific block by its hash"""
        for block in self.chain:
            if block.hash == block_hash:
                return block.to_dict()
        return None
    
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
            calculated_hash = self._calculate_hash(block)
            if calculated_hash != block.hash:
                verification_result["is_valid"] = False
                verification_result["errors"].append(f"Block {i}: Invalid hash")
            
            # Verify previous hash linkage (except genesis)
            if i > 0:
                if block.previous_hash != self.chain[i-1].hash:
                    verification_result["is_valid"] = False
                    verification_result["errors"].append(f"Block {i}: Invalid previous hash")
        
        status = "✅ VALID" if verification_result["is_valid"] else "❌ INVALID"
        logger.info(f"Blockchain verification: {status}")
        
        return verification_result
    
    def log_event(self, event_type: str, event_data: Dict[str, Any]) -> str:
        """
        Log an event to the blockchain
        """
        try:
            # Prepare event data
            block_data = {
                "event_type": event_type,
                "event_data": event_data,
                "logged_at": datetime.utcnow().isoformat(),
                "chain_id": self.chain_id
            }
            
            # Add block to chain
            block_hash = self.append_block(block_data)
            
            logger.info(f"✅ Event logged: {event_type}")
            return block_hash
            
        except Exception as e:
            logger.error(f"❌ Error logging event: {e}")
            raise
    
    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get all events of a specific type"""
        events = []
        
        for block in self.chain:
            if block.data.get("event_type") == event_type:
                events.append({
                    "block_index": block.index,
                    "timestamp": block.timestamp,
                    "event_data": block.data.get("event_data", {}),
                    "block_hash": block.hash
                })
        
        return events
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        event_types = {}
        
        for block in self.chain:
            event_type = block.data.get("event_type", "unknown")
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            "chain_id": self.chain_id,
            "total_blocks": len(self.chain),
            "genesis_timestamp": self.chain[0].timestamp if self.chain else None,
            "latest_timestamp": self.chain[-1].timestamp if self.chain else None,
            "event_types": event_types,
            "integrity_verified": self.verify_chain_integrity()["is_valid"]
        }
    
    def _calculate_hash(self, block: Block) -> str:
        """Calculate SHA-256 hash for a block"""
        # Create string representation of block data
        block_string = f"{block.index}{block.timestamp}{json.dumps(block.data, sort_keys=True)}{block.previous_hash}{block.nonce}"
        
        # Calculate SHA-256 hash
        return hashlib.sha256(block_string.encode()).hexdigest()
    
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
                block = Block(**block_data)
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

# Convenience functions for common event types
def log_prediction_event(ledger: BlockchainLedger, prediction_data: Dict[str, Any]) -> str:
    """Log a model prediction event"""
    return ledger.log_event("model_prediction", prediction_data)

def log_nlp_event(ledger: BlockchainLedger, nlp_data: Dict[str, Any]) -> str:
    """Log an NLP extraction event"""
    return ledger.log_event("nlp_extraction", nlp_data)

def log_data_event(ledger: BlockchainLedger, data_info: Dict[str, Any]) -> str:
    """Log a data ingestion event"""
    return ledger.log_event("data_ingestion", data_info)

def log_system_event(ledger: BlockchainLedger, system_info: Dict[str, Any]) -> str:
    """Log a system event"""
    return ledger.log_event("system_event", system_info)
