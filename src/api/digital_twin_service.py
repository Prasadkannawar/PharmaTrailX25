#!/usr/bin/env python3
"""
Digital Twin FastAPI Service for PharmaTrail-X Phase 3
API endpoints for simulation and scenario modeling
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.simulation.digital_twin_engine import (
    DigitalTwinEngine, TrialParameters, Site, SimulationResults
)
from src.simulation.scenario_engine import (
    ScenarioEngine, ScenarioParameters, ScenarioComparison
)

# Pydantic models for API requests/responses
class SiteConfig(BaseModel):
    site_id: str
    name: str
    region: str
    capacity: int = Field(default=50, ge=1, le=200)
    staff_count: int = Field(default=3, ge=1, le=20)
    staff_skill_level: float = Field(default=0.8, ge=0.1, le=1.0)
    enrollment_rate: float = Field(default=2.0, ge=0.1, le=10.0)
    query_resolution_time: float = Field(default=5.0, ge=0.5, le=30.0)
    data_entry_delay: float = Field(default=2.0, ge=0.1, le=14.0)

class TrialConfig(BaseModel):
    trial_id: str
    target_enrollment: int = Field(ge=10, le=10000)
    planned_duration_weeks: int = Field(ge=4, le=260)
    sites: List[SiteConfig]
    visit_schedule: List[int] = Field(default=[0, 14, 28, 56, 84, 112, 140, 168])
    dropout_base_rate: float = Field(default=0.15, ge=0.0, le=0.8)
    query_rate_per_visit: float = Field(default=0.3, ge=0.0, le=1.0)
    data_entry_sla: float = Field(default=7.0, ge=1.0, le=30.0)

class SimulationRequest(BaseModel):
    trial_config: TrialConfig
    simulation_days: int = Field(default=365, ge=30, le=1095)
    random_seed: Optional[int] = None

class ScenarioRequest(BaseModel):
    scenario_id: str
    scenario_name: str
    description: str
    baseline_trial_config: TrialConfig
    simulation_days: int = Field(default=365, ge=30, le=1095)
    
    # Site modifications
    add_sites: List[SiteConfig] = Field(default_factory=list)
    remove_sites: List[str] = Field(default_factory=list)
    modify_sites: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Global improvements
    staff_multiplier: float = Field(default=1.0, ge=0.5, le=3.0)
    skill_improvement: float = Field(default=0.0, ge=0.0, le=0.5)
    query_time_reduction: float = Field(default=0.0, ge=0.0, le=0.8)
    data_entry_improvement: float = Field(default=0.0, ge=0.0, le=0.8)
    enrollment_boost: float = Field(default=0.0, ge=0.0, le=2.0)
    dropout_reduction: float = Field(default=0.0, ge=0.0, le=0.8)
    query_rate_reduction: float = Field(default=0.0, ge=0.0, le=0.8)
    
    # Cost factors
    additional_cost_per_site: float = Field(default=0.0, ge=0.0)
    training_cost: float = Field(default=0.0, ge=0.0)
    tooling_cost: float = Field(default=0.0, ge=0.0)

class TimelineDataPoint(BaseModel):
    day: int
    timestamp: float
    enrolled_patients: int
    active_patients: int
    completed_patients: int
    dropout_patients: int
    open_queries: int
    resolved_queries: int
    avg_data_delay: float

class SiteMetrics(BaseModel):
    enrolled_patients: int
    completed_patients: int
    dropout_patients: int
    active_queries: int
    resolved_queries: int
    enrollment_rate: float
    completion_rate: float
    avg_data_delay: float

class SimulationResponse(BaseModel):
    trial_id: str
    simulation_days: int
    completion_date: Optional[str]
    total_enrolled: int
    total_completed: int
    total_dropouts: int
    active_patients: int
    total_queries: int
    resolved_queries: int
    pending_queries: int
    avg_data_entry_delay: float
    delay_risk_score: float
    cost_estimate: float
    timeline_data: List[TimelineDataPoint]
    site_metrics: Dict[str, SiteMetrics]
    scenario_id: Optional[str] = None

class ScenarioComparisonResponse(BaseModel):
    scenario_id: str
    scenario_name: str
    description: str
    baseline_results: SimulationResponse
    scenario_results: SimulationResponse
    
    # Improvement metrics
    enrollment_improvement: float
    completion_improvement: float
    dropout_reduction: float
    query_resolution_improvement: float
    data_delay_improvement: float
    risk_score_improvement: float
    
    # Time and cost analysis
    time_savings_days: float
    cost_savings: float
    roi_percentage: float
    
    # Dashboard-ready comparison data
    comparison_timeline: List[Dict[str, Any]]

class DigitalTwinService:
    """Digital Twin API service for clinical trial simulation"""
    
    def __init__(self):
        self.twin_engine = DigitalTwinEngine()
        self.scenario_engine = ScenarioEngine()
        self.simulation_cache: Dict[str, SimulationResults] = {}
        
    def convert_trial_config_to_params(self, config: TrialConfig) -> TrialParameters:
        """Convert API TrialConfig to internal TrialParameters"""
        sites = []
        for site_config in config.sites:
            site = Site(
                site_id=site_config.site_id,
                name=site_config.name,
                region=site_config.region,
                capacity=site_config.capacity,
                staff_count=site_config.staff_count,
                staff_skill_level=site_config.staff_skill_level,
                enrollment_rate=site_config.enrollment_rate,
                query_resolution_time=site_config.query_resolution_time,
                data_entry_delay=site_config.data_entry_delay
            )
            sites.append(site)
        
        return TrialParameters(
            trial_id=config.trial_id,
            target_enrollment=config.target_enrollment,
            planned_duration_weeks=config.planned_duration_weeks,
            sites=sites,
            visit_schedule=config.visit_schedule,
            dropout_base_rate=config.dropout_base_rate,
            query_rate_per_visit=config.query_rate_per_visit,
            data_entry_sla=config.data_entry_sla
        )
    
    def convert_results_to_response(self, results: SimulationResults) -> SimulationResponse:
        """Convert internal SimulationResults to API response"""
        timeline_data = [
            TimelineDataPoint(**point) for point in results.timeline_data
        ]
        
        site_metrics = {
            site_id: SiteMetrics(**metrics) 
            for site_id, metrics in results.site_metrics.items()
        }
        
        return SimulationResponse(
            trial_id=results.trial_id,
            simulation_days=results.simulation_days,
            completion_date=results.completion_date,
            total_enrolled=results.total_enrolled,
            total_completed=results.total_completed,
            total_dropouts=results.total_dropouts,
            active_patients=results.active_patients,
            total_queries=results.total_queries,
            resolved_queries=results.resolved_queries,
            pending_queries=results.pending_queries,
            avg_data_entry_delay=results.avg_data_entry_delay,
            delay_risk_score=results.delay_risk_score,
            cost_estimate=results.cost_estimate,
            timeline_data=timeline_data,
            site_metrics=site_metrics,
            scenario_id=results.scenario_id
        )
    
    async def run_baseline_simulation(self, request: SimulationRequest) -> SimulationResponse:
        """Run baseline digital twin simulation"""
        try:
            logger.info(f"🎯 Running baseline simulation for trial {request.trial_config.trial_id}")
            
            # Set random seed if provided
            if request.random_seed:
                np.random.seed(request.random_seed)
            
            # Convert config to parameters
            trial_params = self.convert_trial_config_to_params(request.trial_config)
            
            # Initialize and run simulation
            self.twin_engine.initialize_simulation(trial_params, request.simulation_days)
            results = self.twin_engine.run_simulation(request.simulation_days)
            
            # Cache results for scenario comparisons
            cache_key = f"{request.trial_config.trial_id}_{request.simulation_days}"
            self.simulation_cache[cache_key] = results
            
            # Convert to API response
            response = self.convert_results_to_response(results)
            
            logger.info(f"✅ Baseline simulation completed for {request.trial_config.trial_id}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in baseline simulation: {e}")
            raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
    
    async def run_scenario_simulation(self, request: ScenarioRequest) -> ScenarioComparisonResponse:
        """Run scenario simulation and comparison"""
        try:
            logger.info(f"🎭 Running scenario simulation: {request.scenario_name}")
            
            # Convert configs to parameters
            baseline_params = self.convert_trial_config_to_params(request.baseline_trial_config)
            
            # Create scenario parameters
            scenario_params = ScenarioParameters(
                scenario_id=request.scenario_id,
                scenario_name=request.scenario_name,
                description=request.description,
                add_sites=[
                    {
                        'site_id': site.site_id,
                        'name': site.name,
                        'region': site.region,
                        'capacity': site.capacity,
                        'staff_count': site.staff_count,
                        'staff_skill_level': site.staff_skill_level,
                        'enrollment_rate': site.enrollment_rate,
                        'query_resolution_time': site.query_resolution_time,
                        'data_entry_delay': site.data_entry_delay
                    }
                    for site in request.add_sites
                ],
                remove_sites=request.remove_sites,
                modify_sites=request.modify_sites,
                staff_multiplier=request.staff_multiplier,
                skill_improvement=request.skill_improvement,
                query_time_reduction=request.query_time_reduction,
                data_entry_improvement=request.data_entry_improvement,
                enrollment_boost=request.enrollment_boost,
                dropout_reduction=request.dropout_reduction,
                query_rate_reduction=request.query_rate_reduction,
                additional_cost_per_site=request.additional_cost_per_site,
                training_cost=request.training_cost,
                tooling_cost=request.tooling_cost
            )
            
            # Run baseline simulation if not cached
            cache_key = f"{request.baseline_trial_config.trial_id}_{request.simulation_days}"
            if cache_key not in self.simulation_cache:
                baseline_twin = DigitalTwinEngine()
                baseline_twin.initialize_simulation(baseline_params, request.simulation_days)
                baseline_results = baseline_twin.run_simulation(request.simulation_days)
                self.simulation_cache[cache_key] = baseline_results
            else:
                baseline_results = self.simulation_cache[cache_key]
            
            # Run scenario simulation
            scenario_results = self.scenario_engine.run_scenario_simulation(
                baseline_params, scenario_params, request.simulation_days
            )
            
            # Compare scenarios
            comparison = self.scenario_engine.compare_scenarios(
                baseline_results, scenario_results, scenario_params
            )
            
            # Convert to API response
            response = ScenarioComparisonResponse(
                scenario_id=request.scenario_id,
                scenario_name=request.scenario_name,
                description=request.description,
                baseline_results=self.convert_results_to_response(comparison.baseline_results),
                scenario_results=self.convert_results_to_response(comparison.scenario_results),
                enrollment_improvement=comparison.enrollment_improvement,
                completion_improvement=comparison.completion_improvement,
                dropout_reduction=comparison.dropout_reduction,
                query_resolution_improvement=comparison.query_resolution_improvement,
                data_delay_improvement=comparison.data_delay_improvement,
                risk_score_improvement=comparison.risk_score_improvement,
                time_savings_days=comparison.time_savings_days,
                cost_savings=comparison.cost_savings,
                roi_percentage=comparison.roi_percentage,
                comparison_timeline=comparison.comparison_timeline
            )
            
            logger.info(f"✅ Scenario simulation completed: {request.scenario_name}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in scenario simulation: {e}")
            raise HTTPException(status_code=500, detail=f"Scenario simulation failed: {str(e)}")
    
    async def get_scenario_recommendations(
        self, 
        trial_config: TrialConfig,
        simulation_days: int = 365
    ) -> List[Dict[str, Any]]:
        """Get recommended scenarios for what-if analysis"""
        try:
            logger.info(f"💡 Generating scenario recommendations for {trial_config.trial_id}")
            
            # Convert to parameters
            baseline_params = self.convert_trial_config_to_params(trial_config)
            
            # Generate recommendations
            recommendations = self.scenario_engine.generate_scenario_recommendations(
                baseline_params, simulation_days
            )
            
            # Convert to API format
            api_recommendations = []
            for rec in recommendations:
                api_recommendations.append({
                    'scenario_id': rec.scenario_id,
                    'scenario_name': rec.scenario_name,
                    'description': rec.description,
                    'estimated_cost': rec.training_cost + rec.tooling_cost,
                    'complexity': 'Low' if rec.training_cost + rec.tooling_cost < 20000 else 'Medium' if rec.training_cost + rec.tooling_cost < 50000 else 'High',
                    'parameters': {
                        'staff_multiplier': rec.staff_multiplier,
                        'skill_improvement': rec.skill_improvement,
                        'query_time_reduction': rec.query_time_reduction,
                        'data_entry_improvement': rec.data_entry_improvement,
                        'enrollment_boost': rec.enrollment_boost,
                        'dropout_reduction': rec.dropout_reduction,
                        'add_sites_count': len(rec.add_sites),
                        'training_cost': rec.training_cost,
                        'tooling_cost': rec.tooling_cost
                    }
                })
            
            logger.info(f"✅ Generated {len(api_recommendations)} scenario recommendations")
            return api_recommendations
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get digital twin service information"""
        return {
            "service_name": "PharmaTrail-X Digital Twin",
            "version": "3.0.0",
            "capabilities": [
                "clinical_trial_simulation",
                "patient_enrollment_modeling",
                "site_performance_simulation",
                "query_resolution_modeling",
                "data_entry_delay_simulation",
                "dropout_behavior_modeling",
                "staffing_impact_analysis",
                "scenario_what_if_analysis",
                "cost_benefit_analysis",
                "roi_calculations"
            ],
            "simulation_engine": self.twin_engine.get_model_info(),
            "supported_scenarios": [
                "staff_augmentation",
                "process_optimization", 
                "training_programs",
                "site_expansion",
                "technology_upgrades",
                "custom_modifications"
            ],
            "output_formats": [
                "timeline_trajectories",
                "site_level_metrics",
                "scenario_comparisons",
                "dashboard_ready_json",
                "gantt_chart_data",
                "heatmap_data"
            ]
        }
