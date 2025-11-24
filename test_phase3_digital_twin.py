#!/usr/bin/env python3
"""
Comprehensive Test Suite for PharmaTrail-X Phase 3 Digital Twin
Tests simulation engine, scenario modeling, and API endpoints
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import requests
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.simulation.digital_twin_engine import (
    DigitalTwinEngine, TrialParameters, Site, Patient
)
from src.simulation.scenario_engine import (
    ScenarioEngine, ScenarioParameters
)
from src.api.digital_twin_service import DigitalTwinService

def test_digital_twin_engine():
    """Test the core Digital Twin simulation engine"""
    print("🧪 Testing Digital Twin Engine...")
    
    try:
        # Create test trial parameters
        sites = [
            Site(
                site_id="SITE-001",
                name="Test Site 1",
                region="North America",
                capacity=50,
                staff_count=4,
                staff_skill_level=0.8,
                enrollment_rate=2.5,
                query_resolution_time=4.0,
                data_entry_delay=1.5
            ),
            Site(
                site_id="SITE-002", 
                name="Test Site 2",
                region="Europe",
                capacity=40,
                staff_count=3,
                staff_skill_level=0.7,
                enrollment_rate=2.0,
                query_resolution_time=5.0,
                data_entry_delay=2.0
            )
        ]
        
        trial_params = TrialParameters(
            trial_id="TEST-TRIAL-001",
            target_enrollment=100,
            planned_duration_weeks=52,
            sites=sites,
            visit_schedule=[0, 14, 28, 56, 84, 112, 140, 168],
            dropout_base_rate=0.15,
            query_rate_per_visit=0.3,
            data_entry_sla=7.0
        )
        
        # Initialize and run simulation
        twin_engine = DigitalTwinEngine()
        twin_engine.initialize_simulation(trial_params, simulation_days=90)
        
        print("  Running 90-day simulation...")
        results = twin_engine.run_simulation(simulation_days=90)
        
        # Validate results
        print(f"    ✅ Trial ID: {results.trial_id}")
        print(f"    ✅ Simulation days: {results.simulation_days}")
        print(f"    ✅ Total enrolled: {results.total_enrolled}")
        print(f"    ✅ Total completed: {results.total_completed}")
        print(f"    ✅ Total dropouts: {results.total_dropouts}")
        print(f"    ✅ Active patients: {results.active_patients}")
        print(f"    ✅ Total queries: {results.total_queries}")
        print(f"    ✅ Resolved queries: {results.resolved_queries}")
        print(f"    ✅ Delay risk score: {results.delay_risk_score:.3f}")
        print(f"    ✅ Timeline data points: {len(results.timeline_data)}")
        print(f"    ✅ Site metrics: {len(results.site_metrics)}")
        
        # Validate data integrity
        assert results.total_enrolled >= 0, "Enrolled patients should be non-negative"
        assert results.total_completed >= 0, "Completed patients should be non-negative"
        assert results.total_dropouts >= 0, "Dropout patients should be non-negative"
        assert results.total_enrolled == results.total_completed + results.total_dropouts + results.active_patients, "Patient counts should balance"
        assert 0 <= results.delay_risk_score <= 1, "Delay risk score should be between 0 and 1"
        assert len(results.timeline_data) > 0, "Timeline data should not be empty"
        assert len(results.site_metrics) == len(sites), "Should have metrics for all sites"
        
        print("✅ Digital Twin Engine test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Digital Twin Engine test FAILED: {e}")
        return False

def test_scenario_engine():
    """Test the scenario modeling engine"""
    print("\n🧪 Testing Scenario Engine...")
    
    try:
        # Create baseline trial parameters
        sites = [
            Site(
                site_id="SITE-001",
                name="Baseline Site 1",
                region="North America",
                capacity=50,
                staff_count=3,
                staff_skill_level=0.7,
                enrollment_rate=2.0,
                query_resolution_time=6.0,
                data_entry_delay=2.5
            )
        ]
        
        baseline_params = TrialParameters(
            trial_id="SCENARIO-TEST-001",
            target_enrollment=50,
            planned_duration_weeks=26,
            sites=sites,
            dropout_base_rate=0.2,
            query_rate_per_visit=0.4
        )
        
        # Create scenario parameters (staff augmentation)
        scenario_params = ScenarioParameters(
            scenario_id="staff_augmentation_test",
            scenario_name="Staff Augmentation Test",
            description="Double staff and improve skills",
            staff_multiplier=2.0,
            skill_improvement=0.2,
            query_time_reduction=0.3,
            training_cost=15000
        )
        
        # Initialize scenario engine
        scenario_engine = ScenarioEngine()
        
        print("  Running baseline simulation...")
        baseline_twin = DigitalTwinEngine()
        baseline_twin.initialize_simulation(baseline_params, 60)
        baseline_results = baseline_twin.run_simulation(60)
        
        print("  Running scenario simulation...")
        scenario_results = scenario_engine.run_scenario_simulation(
            baseline_params, scenario_params, 60
        )
        
        print("  Comparing scenarios...")
        comparison = scenario_engine.compare_scenarios(
            baseline_results, scenario_results, scenario_params
        )
        
        # Validate comparison results
        print(f"    ✅ Baseline enrolled: {comparison.baseline_results.total_enrolled}")
        print(f"    ✅ Scenario enrolled: {comparison.scenario_results.total_enrolled}")
        print(f"    ✅ Enrollment improvement: {comparison.enrollment_improvement:.1%}")
        print(f"    ✅ Completion improvement: {comparison.completion_improvement:.1%}")
        print(f"    ✅ Risk score improvement: {comparison.risk_score_improvement:.3f}")
        print(f"    ✅ Time savings: {comparison.time_savings_days:.1f} days")
        print(f"    ✅ Cost savings: ${comparison.cost_savings:,.0f}")
        print(f"    ✅ ROI: {comparison.roi_percentage:.1f}%")
        print(f"    ✅ Comparison timeline points: {len(comparison.comparison_timeline)}")
        
        # Generate recommendations
        print("  Testing scenario recommendations...")
        recommendations = scenario_engine.generate_scenario_recommendations(baseline_params, 60)
        
        print(f"    ✅ Generated recommendations: {len(recommendations)}")
        for i, rec in enumerate(recommendations[:3]):
            print(f"       {i+1}. {rec.scenario_name}: {rec.description}")
        
        assert len(recommendations) > 0, "Should generate at least one recommendation"
        assert comparison.enrollment_improvement is not None, "Should calculate enrollment improvement"
        assert len(comparison.comparison_timeline) > 0, "Should generate comparison timeline"
        
        print("✅ Scenario Engine test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Scenario Engine test FAILED: {e}")
        return False

def test_digital_twin_service():
    """Test the Digital Twin API service"""
    print("\n🧪 Testing Digital Twin Service...")
    
    try:
        # Initialize service
        service = DigitalTwinService()
        
        # Create test trial config
        from src.api.digital_twin_service import TrialConfig, SiteConfig, SimulationRequest
        
        site_configs = [
            SiteConfig(
                site_id="SERVICE-SITE-001",
                name="Service Test Site",
                region="Test Region",
                capacity=30,
                staff_count=2,
                staff_skill_level=0.8,
                enrollment_rate=1.5,
                query_resolution_time=4.0,
                data_entry_delay=1.8
            )
        ]
        
        trial_config = TrialConfig(
            trial_id="SERVICE-TEST-001",
            target_enrollment=30,
            planned_duration_weeks=20,
            sites=site_configs
        )
        
        simulation_request = SimulationRequest(
            trial_config=trial_config,
            simulation_days=45,
            random_seed=42
        )
        
        print("  Running service simulation...")
        import asyncio
        
        async def run_service_test():
            # Test baseline simulation
            results = await service.run_baseline_simulation(simulation_request)
            
            print(f"    ✅ Service trial ID: {results.trial_id}")
            print(f"    ✅ Service enrolled: {results.total_enrolled}")
            print(f"    ✅ Service timeline points: {len(results.timeline_data)}")
            print(f"    ✅ Service site metrics: {len(results.site_metrics)}")
            
            # Test recommendations
            recommendations = await service.get_scenario_recommendations(trial_config, 45)
            print(f"    ✅ Service recommendations: {len(recommendations)}")
            
            return results, recommendations
        
        # Run async test
        results, recommendations = asyncio.run(run_service_test())
        
        # Validate service results
        assert results.trial_id == "SERVICE-TEST-001", "Trial ID should match"
        assert results.simulation_days == 45, "Simulation days should match"
        assert len(results.timeline_data) > 0, "Should have timeline data"
        assert len(results.site_metrics) == 1, "Should have metrics for one site"
        assert len(recommendations) > 0, "Should generate recommendations"
        
        # Test service info
        service_info = service.get_service_info()
        print(f"    ✅ Service name: {service_info['service_name']}")
        print(f"    ✅ Service version: {service_info['version']}")
        print(f"    ✅ Capabilities: {len(service_info['capabilities'])}")
        
        print("✅ Digital Twin Service test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Digital Twin Service test FAILED: {e}")
        return False

def test_api_endpoints():
    """Test the Phase 3 API endpoints"""
    print("\n🧪 Testing Phase 3 API Endpoints...")
    
    # Note: This test assumes the API is running on port 8006
    BASE_URL = "http://localhost:8006"
    
    try:
        # Test root endpoint
        print("  Testing root endpoint...")
        response = requests.get(f"{BASE_URL}/", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"    ✅ API Version: {data['version']}")
            print(f"    ✅ Phase 3 endpoints: {list(data['phase3_endpoints'].keys())}")
            print(f"    ✅ Components: {data['components']}")
            
            # Test twin info endpoint
            print("  Testing /twin/info endpoint...")
            info_response = requests.get(f"{BASE_URL}/twin/info", timeout=5)
            
            if info_response.status_code == 200:
                info_data = info_response.json()
                print(f"    ✅ Twin service: {info_data['service_name']}")
                print(f"    ✅ Capabilities: {len(info_data['capabilities'])}")
                
                # Test simulation endpoint with sample data
                print("  Testing /twin/simulate endpoint...")
                simulation_payload = {
                    "trial_config": {
                        "trial_id": "API-TEST-001",
                        "target_enrollment": 20,
                        "planned_duration_weeks": 12,
                        "sites": [
                            {
                                "site_id": "API-SITE-001",
                                "name": "API Test Site",
                                "region": "Test",
                                "capacity": 25,
                                "staff_count": 2,
                                "staff_skill_level": 0.8,
                                "enrollment_rate": 1.8,
                                "query_resolution_time": 4.5,
                                "data_entry_delay": 2.0
                            }
                        ]
                    },
                    "simulation_days": 30,
                    "random_seed": 123
                }
                
                sim_response = requests.post(
                    f"{BASE_URL}/twin/simulate", 
                    json=simulation_payload,
                    timeout=30
                )
                
                if sim_response.status_code == 200:
                    sim_data = sim_response.json()
                    print(f"    ✅ Simulation enrolled: {sim_data['total_enrolled']}")
                    print(f"    ✅ Simulation risk score: {sim_data['delay_risk_score']:.3f}")
                    print(f"    ✅ Timeline data points: {len(sim_data['timeline_data'])}")
                    
                    print("✅ Phase 3 API Endpoints test PASSED!")
                    return True
                else:
                    print(f"    ❌ Simulation endpoint failed: {sim_response.status_code}")
                    print(f"    Response: {sim_response.text}")
                    return False
            else:
                print(f"    ❌ Twin info endpoint failed: {info_response.status_code}")
                return False
        else:
            print(f"    ❌ Root endpoint failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"    ⚠️  API not running or not accessible: {e}")
        print("    💡 Start the API with: python phase3_integrated_api.py")
        return False
    except Exception as e:
        print(f"❌ Phase 3 API Endpoints test FAILED: {e}")
        return False

def test_simulation_accuracy():
    """Test simulation accuracy and consistency"""
    print("\n🧪 Testing Simulation Accuracy...")
    
    try:
        # Create deterministic test scenario
        np.random.seed(42)  # Set seed for reproducibility
        
        sites = [
            Site(
                site_id="ACCURACY-001",
                name="Accuracy Test Site",
                region="Test",
                capacity=100,
                staff_count=5,
                staff_skill_level=0.9,
                enrollment_rate=3.0,
                query_resolution_time=3.0,
                data_entry_delay=1.0
            )
        ]
        
        trial_params = TrialParameters(
            trial_id="ACCURACY-TEST",
            target_enrollment=50,
            planned_duration_weeks=20,
            sites=sites,
            dropout_base_rate=0.1,
            query_rate_per_visit=0.2
        )
        
        # Run multiple simulations to test consistency
        results_list = []
        
        for i in range(3):
            np.random.seed(42)  # Same seed for consistency
            twin_engine = DigitalTwinEngine()
            twin_engine.initialize_simulation(trial_params, 60)
            results = twin_engine.run_simulation(60)
            results_list.append(results)
        
        # Check consistency across runs
        enrollments = [r.total_enrolled for r in results_list]
        completions = [r.total_completed for r in results_list]
        risk_scores = [r.delay_risk_score for r in results_list]
        
        print(f"    ✅ Enrollments across runs: {enrollments}")
        print(f"    ✅ Completions across runs: {completions}")
        print(f"    ✅ Risk scores across runs: {[f'{r:.3f}' for r in risk_scores]}")
        
        # Test logical constraints
        for i, results in enumerate(results_list):
            print(f"    Run {i+1} validation:")
            print(f"      Total = Active + Completed + Dropouts: {results.total_enrolled} = {results.active_patients} + {results.total_completed} + {results.total_dropouts}")
            
            # Validate patient balance
            total_check = results.active_patients + results.total_completed + results.total_dropouts
            assert results.total_enrolled == total_check, f"Patient counts don't balance in run {i+1}"
            
            # Validate query logic
            assert results.resolved_queries <= results.total_queries, f"Can't resolve more queries than generated in run {i+1}"
            
            # Validate risk score bounds
            assert 0 <= results.delay_risk_score <= 1, f"Risk score out of bounds in run {i+1}"
        
        # Test with different parameters
        print("  Testing parameter sensitivity...")
        
        # High-risk scenario
        high_risk_sites = [
            Site(
                site_id="HIGH-RISK-001",
                name="High Risk Site",
                region="Test",
                capacity=20,
                staff_count=1,
                staff_skill_level=0.3,
                enrollment_rate=0.5,
                query_resolution_time=15.0,
                data_entry_delay=8.0
            )
        ]
        
        high_risk_params = TrialParameters(
            trial_id="HIGH-RISK-TEST",
            target_enrollment=30,
            planned_duration_weeks=20,
            sites=high_risk_sites,
            dropout_base_rate=0.4,
            query_rate_per_visit=0.8
        )
        
        np.random.seed(42)
        high_risk_twin = DigitalTwinEngine()
        high_risk_twin.initialize_simulation(high_risk_params, 60)
        high_risk_results = high_risk_twin.run_simulation(60)
        
        print(f"    ✅ High-risk scenario risk score: {high_risk_results.delay_risk_score:.3f}")
        print(f"    ✅ High-risk scenario enrollment: {high_risk_results.total_enrolled}")
        
        # High-risk should have higher risk score than baseline
        baseline_risk = results_list[0].delay_risk_score
        assert high_risk_results.delay_risk_score > baseline_risk, "High-risk scenario should have higher risk score"
        
        print("✅ Simulation Accuracy test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Simulation Accuracy test FAILED: {e}")
        return False

def main():
    """Run all Phase 3 Digital Twin tests"""
    print("🚀 Starting PharmaTrail-X Phase 3 Digital Twin Test Suite...")
    print("=" * 70)
    
    tests = [
        ("Digital Twin Engine", test_digital_twin_engine),
        ("Scenario Engine", test_scenario_engine),
        ("Digital Twin Service", test_digital_twin_service),
        ("Simulation Accuracy", test_simulation_accuracy),
        ("API Endpoints", test_api_endpoints)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print("\n" + "=" * 70)
    print("📊 PHASE 3 DIGITAL TWIN TEST RESULTS")
    print("=" * 70)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print("=" * 70)
    print(f"Overall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 ALL PHASE 3 DIGITAL TWIN TESTS PASSED!")
        print("✅ Phase 3 Digital Twin is fully functional!")
        print("\n💡 To test API endpoints, start the server:")
        print("   python phase3_integrated_api.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
