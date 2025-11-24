#!/usr/bin/env python3
"""
Phase 2 Testing Suite for PharmaTrail-X
Tests NLP engine, blockchain audit ledger, and integrated platform
"""

import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.nlp.clinical_nlp_engine import ClinicalNLPEngine
from src.blockchain.audit_ledger import PharmaBlockchain, AuditLogger

def test_clinical_nlp_engine():
    """Test the clinical NLP engine"""
    logger.info("🧪 Testing Clinical NLP Engine...")
    
    try:
        # Initialize NLP engine
        nlp_engine = ClinicalNLPEngine()
        
        # Test data
        clinical_text = """
        Patient experienced severe headache and nausea on day 3 of treatment.
        ALT levels elevated to 65 U/L (normal <40). Patient reported moderate fatigue
        and decreased appetite. Adverse event classified as Grade 2 severity.
        Protocol deviation noted: medication taken 2 hours late on day 5.
        Creatinine increased to 1.8 mg/dL from baseline 1.0 mg/dL.
        """
        
        investigator_notes = """
        72-year-old male patient with BMI 35.5. Medication adherence poor at 45%.
        Multiple adverse events reported including hypertension (BP 160/105),
        liver enzyme elevation, and kidney function decline. Recommend dose reduction
        and increased monitoring. Patient satisfaction score: 3/10.
        """
        
        # Test adverse event extraction
        logger.info("Testing adverse event extraction...")
        ae_result = nlp_engine.extract_adverse_events(clinical_text, "TEST-001")
        
        print(f"✅ AE Extraction Results:")
        print(f"   - Entities found: {len(ae_result.entities)}")
        print(f"   - AE events: {len(ae_result.ae_events)}")
        print(f"   - Severity: {ae_result.severity_classification}")
        print(f"   - Confidence: {ae_result.confidence_score:.3f}")
        
        # Test text summarization
        logger.info("Testing text summarization...")
        summary_result = nlp_engine.summarize_clinical_text(investigator_notes)
        
        print(f"✅ Summarization Results:")
        print(f"   - Original length: {len(investigator_notes)} chars")
        print(f"   - Summary length: {len(summary_result.summary)} chars")
        print(f"   - Reduction: {summary_result.word_count_reduction:.1%}")
        print(f"   - Key points: {len(summary_result.key_points)}")
        print(f"   - Summary: {summary_result.summary[:100]}...")
        
        # Test model info
        model_info = nlp_engine.get_model_info()
        print(f"✅ Model Info:")
        print(f"   - Primary model: {model_info['primary_model']}")
        print(f"   - Device: {model_info['device']}")
        print(f"   - Models loaded: {model_info['models_loaded']}")
        
        logger.info("✅ Clinical NLP Engine tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ NLP Engine test failed: {e}")
        return False

def test_blockchain_audit_ledger():
    """Test the blockchain audit ledger"""
    logger.info("🧪 Testing Blockchain Audit Ledger...")
    
    try:
        # Initialize blockchain
        blockchain = PharmaBlockchain("test-chain")
        audit_logger = AuditLogger(blockchain)
        
        # Test basic event logging
        logger.info("Testing event logging...")
        
        # Log various types of events
        events_to_log = [
            {
                "trial_id": "TEST-001",
                "event_type": "data_ingestion",
                "payload": {
                    "source": "clinical_csv",
                    "records": 1000,
                    "status": "success"
                }
            },
            {
                "trial_id": "TEST-001", 
                "event_type": "model_prediction",
                "payload": {
                    "model": "delay_predictor",
                    "probability": 0.75,
                    "confidence": 0.85
                }
            },
            {
                "trial_id": "TEST-002",
                "event_type": "adverse_event",
                "payload": {
                    "patient_id": "P001",
                    "ae_term": "headache",
                    "severity": "moderate"
                }
            }
        ]
        
        block_hashes = []
        for event in events_to_log:
            block_hash = blockchain.log_event(
                trial_id=event["trial_id"],
                event_type=event["event_type"],
                event_payload=event["payload"]
            )
            block_hashes.append(block_hash)
        
        print(f"✅ Event Logging Results:")
        print(f"   - Events logged: {len(events_to_log)}")
        print(f"   - Total blocks: {len(blockchain.chain)}")
        print(f"   - Block hashes: {[h[:8] + '...' for h in block_hashes]}")
        
        # Test chain integrity
        logger.info("Testing chain integrity...")
        integrity_result = blockchain.verify_chain_integrity()
        
        print(f"✅ Integrity Check:")
        print(f"   - Chain valid: {integrity_result['is_valid']}")
        print(f"   - Total blocks verified: {integrity_result['total_blocks']}")
        print(f"   - Errors: {len(integrity_result['errors'])}")
        
        # Test audit trail retrieval
        logger.info("Testing audit trail retrieval...")
        audit_trail = blockchain.get_trial_audit_trail("TEST-001")
        
        print(f"✅ Audit Trail (TEST-001):")
        print(f"   - Total events: {audit_trail['total_events']}")
        print(f"   - Event types: {list(audit_trail['events_by_type'].keys())}")
        
        # Test chain metadata
        metadata = blockchain.get_chain_metadata()
        print(f"✅ Chain Metadata:")
        print(f"   - Chain ID: {metadata.chain_id}")
        print(f"   - Total blocks: {metadata.total_blocks}")
        print(f"   - Integrity verified: {metadata.integrity_verified}")
        
        logger.info("✅ Blockchain Audit Ledger tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Blockchain test failed: {e}")
        return False

def test_integrated_processing():
    """Test integrated Phase 1 + Phase 2 processing"""
    logger.info("🧪 Testing Integrated Processing...")
    
    try:
        # Initialize components
        nlp_engine = ClinicalNLPEngine()
        blockchain = PharmaBlockchain("integrated-test")
        audit_logger = AuditLogger(blockchain)
        
        # Test data combining Phase 1 and Phase 2 inputs
        test_case = {
            "trial_id": "INTEGRATED-001",
            "patient_id": "P001",
            "clinical_data": {
                "Age": 72,
                "BMI": 35.5,
                "Medication_Adherence": 45,
                "ADR_Rate": 0.6,
                "Efficacy_Score": 25,
                "BP_Systolic": 160,
                "BP_Diastolic": 105,
                "ALT_Level": 65,
                "AST_Level": 65,
                "Creatinine": 1.8,
                "Satisfaction_Score": 3,
                "Week": 12
            },
            "clinical_text": """
            Patient reports severe fatigue and decreased appetite since starting treatment.
            Laboratory results show ALT 65 U/L, AST 65 U/L, and creatinine 1.8 mg/dL.
            Blood pressure elevated at 160/105 mmHg. Patient adherence poor at 45%.
            Grade 3 adverse event: severe headache requiring intervention.
            Protocol deviation: missed dose on day 10 due to nausea.
            """,
            "investigator_notes": """
            72-year-old male with multiple comorbidities. Poor medication compliance
            noted throughout study. Multiple adverse events including hypertension,
            liver enzyme elevation, and renal impairment. Recommend dose reduction
            and increased safety monitoring. Patient satisfaction very low.
            """
        }
        
        # Process with NLP
        logger.info("Processing clinical text with NLP...")
        ae_extraction = nlp_engine.extract_adverse_events(
            test_case["clinical_text"], 
            test_case["trial_id"]
        )
        
        text_summary = nlp_engine.summarize_clinical_text(
            test_case["investigator_notes"]
        )
        
        print(f"✅ NLP Processing Results:")
        print(f"   - AE entities: {len(ae_extraction.entities)}")
        print(f"   - AE events: {len(ae_extraction.ae_events)}")
        print(f"   - Severity: {ae_extraction.severity_classification}")
        print(f"   - Summary reduction: {text_summary.word_count_reduction:.1%}")
        
        # Log to blockchain
        logger.info("Logging integrated results to blockchain...")
        
        # Log NLP extraction
        nlp_data = {
            "type": "integrated_nlp_extraction",
            "entity_count": len(ae_extraction.entities),
            "ae_events": ae_extraction.ae_events,
            "confidence": ae_extraction.confidence_score,
            "text_hash": hashlib.sha256(test_case["clinical_text"].encode()).hexdigest()
        }
        
        nlp_block_hash = audit_logger.log_nlp_extraction(
            test_case["trial_id"], 
            nlp_data
        )
        
        # Log integrated prediction (simulated)
        prediction_data = {
            "model_type": "integrated_prediction",
            "prediction": {
                "delay_probability": 0.85,  # High risk based on data
                "combined_risk_score": 0.90,
                "regulatory_flags": ["HIGH_DELAY_RISK", "MULTIPLE_SERIOUS_AES"]
            },
            "features": {
                "clinical_data_hash": hashlib.sha256(str(test_case["clinical_data"]).encode()).hexdigest(),
                "nlp_processed": True
            },
            "confidence": 0.88,
            "model_version": "integrated_v2.0"
        }
        
        prediction_block_hash = audit_logger.log_model_prediction(
            test_case["trial_id"],
            prediction_data
        )
        
        print(f"✅ Blockchain Logging:")
        print(f"   - NLP block hash: {nlp_block_hash[:16]}...")
        print(f"   - Prediction block hash: {prediction_block_hash[:16]}...")
        
        # Verify audit trail
        audit_trail = blockchain.get_trial_audit_trail(test_case["trial_id"])
        print(f"✅ Audit Trail:")
        print(f"   - Total events: {audit_trail['total_events']}")
        print(f"   - Event types: {list(audit_trail['events_by_type'].keys())}")
        
        # Verify chain integrity
        integrity = blockchain.verify_chain_integrity()
        print(f"✅ Chain Integrity: {integrity['is_valid']}")
        
        logger.info("✅ Integrated processing tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integrated processing test failed: {e}")
        return False

def test_regulatory_compliance_features():
    """Test regulatory compliance and audit features"""
    logger.info("🧪 Testing Regulatory Compliance Features...")
    
    try:
        blockchain = PharmaBlockchain("compliance-test")
        audit_logger = AuditLogger(blockchain)
        
        # Simulate regulatory events
        regulatory_events = [
            {
                "type": "serious_adverse_event",
                "trial_id": "REG-001",
                "data": {
                    "patient_id": "P001",
                    "ae_term": "myocardial_infarction",
                    "severity": "life-threatening",
                    "seriousness": "serious",
                    "causality": "possibly_related",
                    "outcome": "recovered",
                    "reported_to_fda": True
                }
            },
            {
                "type": "protocol_deviation",
                "trial_id": "REG-001", 
                "data": {
                    "deviation_type": "inclusion_criteria_violation",
                    "severity": "major",
                    "patient_id": "P002",
                    "description": "Patient enrolled despite exclusion criteria",
                    "corrective_action": "Patient discontinued from study"
                }
            },
            {
                "type": "data_integrity_check",
                "trial_id": "REG-001",
                "data": {
                    "check_type": "source_data_verification",
                    "records_verified": 500,
                    "discrepancies_found": 3,
                    "resolution_status": "resolved"
                }
            }
        ]
        
        # Log regulatory events
        for event in regulatory_events:
            if event["type"] == "serious_adverse_event":
                audit_logger.log_adverse_event(event["trial_id"], event["data"])
            elif event["type"] == "protocol_deviation":
                audit_logger.log_protocol_deviation(event["trial_id"], event["data"])
            else:
                audit_logger.log_system_event(event["trial_id"], {
                    "event": event["type"],
                    "details": event["data"]
                })
        
        print(f"✅ Regulatory Events Logged:")
        print(f"   - Total events: {len(regulatory_events)}")
        print(f"   - Blockchain blocks: {len(blockchain.chain)}")
        
        # Generate compliance report
        audit_trail = blockchain.get_trial_audit_trail("REG-001")
        
        compliance_report = {
            "trial_id": "REG-001",
            "audit_period": {
                "start": audit_trail["first_event"],
                "end": audit_trail["last_event"]
            },
            "total_events": audit_trail["total_events"],
            "event_breakdown": audit_trail["events_by_type"],
            "integrity_verified": blockchain.verify_chain_integrity()["is_valid"],
            "regulatory_readiness": True
        }
        
        print(f"✅ Compliance Report:")
        print(f"   - Trial ID: {compliance_report['trial_id']}")
        print(f"   - Total events: {compliance_report['total_events']}")
        print(f"   - Event types: {list(compliance_report['event_breakdown'].keys())}")
        print(f"   - Integrity verified: {compliance_report['integrity_verified']}")
        print(f"   - Regulatory ready: {compliance_report['regulatory_readiness']}")
        
        logger.info("✅ Regulatory compliance tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Regulatory compliance test failed: {e}")
        return False

def main():
    """Run all Phase 2 tests"""
    logger.info("🚀 Starting PharmaTrail-X Phase 2 Test Suite...")
    
    test_results = {
        "nlp_engine": False,
        "blockchain_ledger": False,
        "integrated_processing": False,
        "regulatory_compliance": False
    }
    
    # Run tests
    test_results["nlp_engine"] = test_clinical_nlp_engine()
    test_results["blockchain_ledger"] = test_blockchain_audit_ledger()
    test_results["integrated_processing"] = test_integrated_processing()
    test_results["regulatory_compliance"] = test_regulatory_compliance_features()
    
    # Summary
    passed = sum(test_results.values())
    total = len(test_results)
    
    print(f"\n{'='*60}")
    print(f"📊 PHASE 2 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title():<30} {status}")
    
    print(f"{'='*60}")
    print(f"Overall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 ALL PHASE 2 TESTS PASSED!")
        print("✅ PharmaTrail-X Phase 2 is ready for deployment!")
    else:
        print("⚠️  Some tests failed. Please review and fix issues.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
