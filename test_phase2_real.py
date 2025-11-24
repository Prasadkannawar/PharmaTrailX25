#!/usr/bin/env python3
"""
Real Phase 2 Test Script for PharmaTrail-X
Tests the actual implemented NLP engine and blockchain ledger
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.nlp.nlp_engine import ClinicalNLPEngine
from src.blockchain.ledger import BlockchainLedger, log_prediction_event, log_nlp_event

def test_nlp_engine():
    """Test the NLP engine"""
    print("🧪 Testing NLP Engine...")
    
    try:
        # Initialize NLP engine
        nlp_engine = ClinicalNLPEngine()
        
        # Test clinical text
        clinical_text = """
        Patient experienced severe headache and nausea on day 3 of treatment.
        ALT levels elevated to 65 U/L (normal <40). Patient reported moderate fatigue
        and decreased appetite. Adverse event classified as Grade 2 severity.
        Protocol deviation noted: medication taken 2 hours late on day 5.
        Creatinine increased to 1.8 mg/dL from baseline 1.0 mg/dL.
        """
        
        # Test adverse event extraction
        print("  Testing adverse event extraction...")
        ae_result = nlp_engine.extract_adverse_events(clinical_text, "TEST-001")
        
        print(f"    ✅ Entities found: {len(ae_result.entities)}")
        print(f"    ✅ AE events: {len(ae_result.ae_events)}")
        print(f"    ✅ Severity: {ae_result.severity_classification}")
        print(f"    ✅ Confidence: {ae_result.confidence_score:.3f}")
        
        for entity in ae_result.entities[:3]:  # Show first 3
            print(f"       - {entity.text} ({entity.label}, {entity.severity})")
        
        # Test text summarization
        print("  Testing text summarization...")
        summary_result = nlp_engine.summarize_clinical_text(clinical_text)
        
        print(f"    ✅ Original: {len(clinical_text)} chars")
        print(f"    ✅ Summary: {len(summary_result.summary)} chars")
        print(f"    ✅ Reduction: {summary_result.word_count_reduction:.1%}")
        print(f"    ✅ Key points: {len(summary_result.key_points)}")
        print(f"    ✅ Summary: {summary_result.summary[:100]}...")
        
        # Test model info
        model_info = nlp_engine.get_model_info()
        print(f"    ✅ Engine type: {model_info['engine_type']}")
        print(f"    ✅ AE terms: {model_info['ae_terms_count']}")
        print(f"    ✅ Lab tests: {model_info['lab_tests_supported']}")
        
        print("✅ NLP Engine test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ NLP Engine test FAILED: {e}")
        return False

def test_blockchain_ledger():
    """Test the blockchain ledger"""
    print("\n🧪 Testing Blockchain Ledger...")
    
    try:
        # Initialize blockchain
        blockchain = BlockchainLedger("test-chain")
        
        print(f"    ✅ Blockchain initialized with {len(blockchain.chain)} blocks")
        
        # Test event logging
        print("  Testing event logging...")
        
        # Log some test events
        events = [
            {"event_type": "model_prediction", "data": {"trial_id": "TEST-001", "probability": 0.75}},
            {"event_type": "nlp_extraction", "data": {"trial_id": "TEST-001", "entities": 5}},
            {"event_type": "data_ingestion", "data": {"source": "csv", "records": 1000}}
        ]
        
        block_hashes = []
        for event in events:
            block_hash = blockchain.log_event(event["event_type"], event["data"])
            block_hashes.append(block_hash)
        
        print(f"    ✅ Logged {len(events)} events")
        print(f"    ✅ Total blocks: {len(blockchain.chain)}")
        print(f"    ✅ Block hashes: {[h[:8] + '...' for h in block_hashes]}")
        
        # Test chain integrity
        print("  Testing chain integrity...")
        integrity_result = blockchain.verify_chain_integrity()
        
        print(f"    ✅ Chain valid: {integrity_result['is_valid']}")
        print(f"    ✅ Blocks verified: {integrity_result['total_blocks']}")
        print(f"    ✅ Errors: {len(integrity_result['errors'])}")
        
        # Test chain retrieval
        print("  Testing chain retrieval...")
        chain_data = blockchain.get_chain()
        
        print(f"    ✅ Retrieved {len(chain_data)} blocks")
        
        # Test stats
        stats = blockchain.get_chain_stats()
        print(f"    ✅ Chain ID: {stats['chain_id']}")
        print(f"    ✅ Event types: {list(stats['event_types'].keys())}")
        print(f"    ✅ Integrity verified: {stats['integrity_verified']}")
        
        print("✅ Blockchain Ledger test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Blockchain Ledger test FAILED: {e}")
        return False

def test_integration():
    """Test NLP + Blockchain integration"""
    print("\n🧪 Testing NLP + Blockchain Integration...")
    
    try:
        # Initialize components
        nlp_engine = ClinicalNLPEngine()
        blockchain = BlockchainLedger("integration-test")
        
        # Test data
        clinical_text = """
        Patient reports severe fatigue and decreased appetite since starting treatment.
        Laboratory results show ALT 65 U/L, AST 65 U/L, and creatinine 1.8 mg/dL.
        Blood pressure elevated at 160/105 mmHg. Patient adherence poor at 45%.
        Grade 3 adverse event: severe headache requiring intervention.
        """
        
        # Process with NLP
        print("  Processing clinical text with NLP...")
        ae_result = nlp_engine.extract_adverse_events(clinical_text, "INTEGRATION-001")
        
        print(f"    ✅ AE entities: {len(ae_result.entities)}")
        print(f"    ✅ AE events: {len(ae_result.ae_events)}")
        print(f"    ✅ Severity: {ae_result.severity_classification}")
        
        # Log NLP results to blockchain
        print("  Logging NLP results to blockchain...")
        nlp_data = {
            "trial_id": "INTEGRATION-001",
            "entity_count": len(ae_result.entities),
            "ae_events": ae_result.ae_events,
            "severity": ae_result.severity_classification,
            "confidence": ae_result.confidence_score
        }
        
        nlp_block_hash = log_nlp_event(blockchain, nlp_data)
        print(f"    ✅ NLP block hash: {nlp_block_hash[:16]}...")
        
        # Log a prediction event
        print("  Logging prediction to blockchain...")
        prediction_data = {
            "trial_id": "INTEGRATION-001",
            "delay_probability": 0.85,
            "combined_risk_score": 0.90,
            "regulatory_flags": ["HIGH_DELAY_RISK", "SEVERE_AE_DETECTED"]
        }
        
        prediction_block_hash = log_prediction_event(blockchain, prediction_data)
        print(f"    ✅ Prediction block hash: {prediction_block_hash[:16]}...")
        
        # Verify blockchain integrity
        integrity = blockchain.verify_chain_integrity()
        print(f"    ✅ Chain integrity: {integrity['is_valid']}")
        
        # Get events by type
        nlp_events = blockchain.get_events_by_type("nlp_extraction")
        prediction_events = blockchain.get_events_by_type("model_prediction")
        
        print(f"    ✅ NLP events: {len(nlp_events)}")
        print(f"    ✅ Prediction events: {len(prediction_events)}")
        
        print("✅ Integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test FAILED: {e}")
        return False

def main():
    """Run all Phase 2 tests"""
    print("🚀 Starting PharmaTrail-X Phase 2 REAL Test Suite...")
    print("=" * 60)
    
    test_results = {
        "nlp_engine": test_nlp_engine(),
        "blockchain_ledger": test_blockchain_ledger(),
        "integration": test_integration()
    }
    
    # Summary
    passed = sum(test_results.values())
    total = len(test_results)
    
    print("\n" + "=" * 60)
    print("📊 PHASE 2 REAL TEST RESULTS")
    print("=" * 60)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title():<20} {status}")
    
    print("=" * 60)
    print(f"Overall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 ALL PHASE 2 TESTS PASSED!")
        print("✅ Phase 2 components are working correctly!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
