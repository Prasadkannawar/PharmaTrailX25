#!/usr/bin/env python3
"""
Test Phase 2 API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8005"

def test_root_endpoint():
    """Test root endpoint"""
    print("🧪 Testing root endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Version: {data['version']}")
            print(f"✅ Status: {data['status']}")
            print(f"✅ Components: {data['components']}")
            print(f"✅ Endpoints: {list(data['endpoints'].keys())}")
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_nlp_ae_endpoint():
    """Test NLP adverse event extraction endpoint"""
    print("\n🧪 Testing NLP AE extraction endpoint...")
    
    try:
        payload = {
            "text": "Patient experienced severe headache and nausea. ALT elevated to 65 U/L. Protocol deviation: missed dose on day 5.",
            "trial_id": "TEST-001",
            "patient_id": "P001"
        }
        
        response = requests.post(f"{BASE_URL}/nlp/ae", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Entities found: {len(data['entities'])}")
            print(f"✅ AE events: {len(data['ae_events'])}")
            print(f"✅ Severity: {data['severity_classification']}")
            print(f"✅ Confidence: {data['confidence_score']:.3f}")
            
            # Show first few entities
            for i, entity in enumerate(data['entities'][:3]):
                print(f"   Entity {i+1}: {entity['text']} ({entity['label']}, {entity['severity']})")
            
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_nlp_summary_endpoint():
    """Test NLP text summarization endpoint"""
    print("\n🧪 Testing NLP summarization endpoint...")
    
    try:
        payload = {
            "text": "Patient is a 72-year-old male with multiple comorbidities including hypertension and diabetes. He has been experiencing severe fatigue and decreased appetite since starting the study medication. Laboratory results show elevated liver enzymes with ALT 65 U/L and AST 60 U/L. Blood pressure remains elevated at 160/105 mmHg despite antihypertensive therapy. Patient adherence has been poor at approximately 45% based on pill counts. Multiple adverse events have been reported including headache, nausea, and dizziness. The investigator recommends dose reduction and increased safety monitoring.",
            "max_length": 100
        }
        
        response = requests.post(f"{BASE_URL}/nlp/summary", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Summary: {data['summary']}")
            print(f"✅ Word reduction: {data['word_count_reduction']:.1%}")
            print(f"✅ Key points: {len(data['key_points'])}")
            print(f"✅ Confidence: {data['confidence_score']:.3f}")
            
            for i, point in enumerate(data['key_points'][:3]):
                print(f"   Key point {i+1}: {point}")
            
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_blockchain_log_endpoint():
    """Test blockchain logging endpoint"""
    print("\n🧪 Testing blockchain logging endpoint...")
    
    try:
        payload = {
            "event_type": "test_event",
            "event_data": {
                "trial_id": "TEST-001",
                "message": "Testing blockchain logging",
                "timestamp": "2025-11-20T14:30:00Z"
            }
        }
        
        response = requests.post(f"{BASE_URL}/blockchain/log_event", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: {data['success']}")
            print(f"✅ Block hash: {data['block_hash'][:16]}...")
            print(f"✅ Block index: {data['block_index']}")
            print(f"✅ Timestamp: {data['timestamp']}")
            
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_blockchain_get_chain_endpoint():
    """Test blockchain chain retrieval endpoint"""
    print("\n🧪 Testing blockchain chain retrieval endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/blockchain/get_chain?limit=5")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chain ID: {data['chain_id']}")
            print(f"✅ Total blocks: {data['total_blocks']}")
            print(f"✅ Integrity verified: {data['integrity_verified']}")
            print(f"✅ Event types: {list(data['stats']['event_types'].keys())}")
            
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_integrated_prediction_endpoint():
    """Test integrated prediction endpoint"""
    print("\n🧪 Testing integrated prediction endpoint...")
    
    try:
        payload = {
            "trial_data": {
                "trial_id": "TEST-001",
                "Age": 72,
                "Medication_Adherence": 45,
                "ADR_Rate": 0.6,
                "BP_Systolic": 160,
                "ALT_Level": 65
            },
            "clinical_text": "Patient reports severe fatigue and elevated liver enzymes. Multiple adverse events including headache and nausea. Poor medication compliance noted.",
            "include_nlp": True,
            "include_blockchain_audit": True
        }
        
        response = requests.post(f"{BASE_URL}/predict/integrated", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Delay probability: {data['delay_probability']:.3f}")
            print(f"✅ Delay prediction: {data['delay_prediction']}")
            print(f"✅ Confidence: {data['confidence']:.3f}")
            print(f"✅ Combined risk score: {data['combined_risk_score']:.3f}")
            print(f"✅ Risk factors: {data['risk_factors']}")
            print(f"✅ Regulatory flags: {data['regulatory_flags']}")
            
            if data['nlp_results']:
                print(f"✅ NLP entities: {data['nlp_results']['entities_found']}")
                print(f"✅ NLP severity: {data['nlp_results']['severity_classification']}")
            
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health_endpoint():
    """Test health endpoint"""
    print("\n🧪 Testing health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data['status']}")
            print(f"✅ Components: {data['components']}")
            print(f"✅ Blockchain blocks: {data['blockchain_stats']['total_blocks']}")
            print(f"✅ Blockchain integrity: {data['blockchain_stats']['integrity_verified']}")
            
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all endpoint tests"""
    print("🚀 Testing PharmaTrail-X Phase 2 API Endpoints...")
    print("=" * 60)
    
    tests = [
        ("Root Endpoint", test_root_endpoint),
        ("NLP AE Extraction", test_nlp_ae_endpoint),
        ("NLP Summarization", test_nlp_summary_endpoint),
        ("Blockchain Logging", test_blockchain_log_endpoint),
        ("Blockchain Chain Retrieval", test_blockchain_get_chain_endpoint),
        ("Integrated Prediction", test_integrated_prediction_endpoint),
        ("Health Check", test_health_endpoint)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print("\n" + "=" * 60)
    print("📊 API ENDPOINT TEST RESULTS")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print("=" * 60)
    print(f"Overall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 ALL API ENDPOINT TESTS PASSED!")
        print("✅ Phase 2 API is fully functional!")
    else:
        print("⚠️  Some endpoint tests failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
