#!/usr/bin/env python3
"""
Scenario Modeling Engine for PharmaTrail-X Phase 3
What-if analysis and scenario comparison capabilities
"""

import copy
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from loguru import logger
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.simulation.digital_twin_engine import (
    DigitalTwinEngine, TrialParameters, Site, SimulationResults
)

@dataclass
class ScenarioParameters:
    """Parameters for what-if scenario modeling"""
    scenario_id: str
    scenario_name: str
    description: str
    
    # Site modifications
    add_sites: List[Dict[str, Any]] = field(default_factory=list)
    remove_sites: List[str] = field(default_factory=list)
    modify_sites: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Staffing changes
    staff_multiplier: float = 1.0  # Global staff increase/decrease
    skill_improvement: float = 0.0  # Skill level increase (0.0 to 1.0)
    
    # Process improvements
    query_time_reduction: float = 0.0  # Percentage reduction (0.0 to 1.0)
    data_entry_improvement: float = 0.0  # Percentage improvement (0.0 to 1.0)
    enrollment_boost: float = 0.0  # Percentage increase (0.0 to 1.0)
    
    # Training and tooling effects
    dropout_reduction: float = 0.0  # Percentage reduction in dropout rate
    query_rate_reduction: float = 0.0  # Reduction in query generation rate
    
    # Cost factors
    additional_cost_per_site: float = 0.0  # Additional daily cost per site
    training_cost: float = 0.0  # One-time training cost
    tooling_cost: float = 0.0  # One-time tooling cost

@dataclass
class ScenarioComparison:
    """Comparison results between baseline and scenario"""
    baseline_results: SimulationResults
    scenario_results: SimulationResults
    scenario_params: ScenarioParameters
    
    # Improvement metrics
    enrollment_improvement: float
    completion_improvement: float
    dropout_reduction: float
    query_resolution_improvement: float
    data_delay_improvement: float
    risk_score_improvement: float
    
    # Time and cost savings
    time_savings_days: float
    cost_savings: float
    roi_percentage: float
    
    # Comparative timeline data
    comparison_timeline: List[Dict[str, Any]]

class ScenarioEngine:
    """
    Scenario modeling engine for what-if analysis
    
    Capabilities:
    - Site addition/removal/modification
    - Staffing level adjustments
    - Process improvement modeling
    - Training and tooling impact simulation
    - Cost-benefit analysis
    - ROI calculations
    """
    
    def __init__(self):
        self.baseline_results: Optional[SimulationResults] = None
        self.scenario_results: Dict[str, SimulationResults] = {}
        
    def create_scenario_trial_params(
        self, 
        baseline_params: TrialParameters, 
        scenario_params: ScenarioParameters
    ) -> TrialParameters:
        """
        Create modified trial parameters for scenario simulation
        
        Args:
            baseline_params: Original trial parameters
            scenario_params: Scenario modifications
            
        Returns:
            Modified TrialParameters for scenario simulation
        """
        # Deep copy baseline parameters
        scenario_trial_params = copy.deepcopy(baseline_params)
        
        logger.info(f"🎭 Creating scenario: {scenario_params.scenario_name}")
        
        # Apply site modifications
        sites_dict = {site.site_id: site for site in scenario_trial_params.sites}
        
        # Remove sites
        for site_id in scenario_params.remove_sites:
            if site_id in sites_dict:
                del sites_dict[site_id]
                logger.info(f"🏥❌ Removed site: {site_id}")
        
        # Add new sites
        for site_config in scenario_params.add_sites:
            new_site = Site(
                site_id=site_config['site_id'],
                name=site_config.get('name', f"Site {site_config['site_id']}"),
                region=site_config.get('region', 'Unknown'),
                capacity=site_config.get('capacity', 50),
                staff_count=site_config.get('staff_count', 3),
                staff_skill_level=site_config.get('staff_skill_level', 0.8),
                enrollment_rate=site_config.get('enrollment_rate', 2.0),
                query_resolution_time=site_config.get('query_resolution_time', 5.0),
                data_entry_delay=site_config.get('data_entry_delay', 2.0)
            )
            sites_dict[new_site.site_id] = new_site
            logger.info(f"🏥✅ Added site: {new_site.site_id} ({new_site.name})")
        
        # Modify existing sites
        for site_id, modifications in scenario_params.modify_sites.items():
            if site_id in sites_dict:
                site = sites_dict[site_id]
                for attr, value in modifications.items():
                    if hasattr(site, attr):
                        setattr(site, attr, value)
                        logger.info(f"🏥🔧 Modified {site_id}.{attr} = {value}")
        
        # Apply global modifications to all sites
        for site in sites_dict.values():
            # Staff multiplier
            if scenario_params.staff_multiplier != 1.0:
                original_staff = site.staff_count
                site.staff_count = max(1, int(site.staff_count * scenario_params.staff_multiplier))
                logger.info(f"👥 {site.site_id} staff: {original_staff} → {site.staff_count}")
            
            # Skill improvement
            if scenario_params.skill_improvement > 0:
                original_skill = site.staff_skill_level
                site.staff_skill_level = min(1.0, site.staff_skill_level + scenario_params.skill_improvement)
                logger.info(f"📈 {site.site_id} skill: {original_skill:.2f} → {site.staff_skill_level:.2f}")
            
            # Query time reduction
            if scenario_params.query_time_reduction > 0:
                original_time = site.query_resolution_time
                site.query_resolution_time *= (1 - scenario_params.query_time_reduction)
                logger.info(f"❓⚡ {site.site_id} query time: {original_time:.1f} → {site.query_resolution_time:.1f} days")
            
            # Data entry improvement
            if scenario_params.data_entry_improvement > 0:
                original_delay = site.data_entry_delay
                site.data_entry_delay *= (1 - scenario_params.data_entry_improvement)
                logger.info(f"📝⚡ {site.site_id} data delay: {original_delay:.1f} → {site.data_entry_delay:.1f} days")
            
            # Enrollment boost
            if scenario_params.enrollment_boost > 0:
                original_rate = site.enrollment_rate
                site.enrollment_rate *= (1 + scenario_params.enrollment_boost)
                logger.info(f"👤📈 {site.site_id} enrollment: {original_rate:.1f} → {site.enrollment_rate:.1f} patients/week")
        
        # Apply trial-level modifications
        if scenario_params.dropout_reduction > 0:
            original_dropout = scenario_trial_params.dropout_base_rate
            scenario_trial_params.dropout_base_rate *= (1 - scenario_params.dropout_reduction)
            logger.info(f"🚪📉 Dropout rate: {original_dropout:.3f} → {scenario_trial_params.dropout_base_rate:.3f}")
        
        if scenario_params.query_rate_reduction > 0:
            original_query_rate = scenario_trial_params.query_rate_per_visit
            scenario_trial_params.query_rate_per_visit *= (1 - scenario_params.query_rate_reduction)
            logger.info(f"❓📉 Query rate: {original_query_rate:.3f} → {scenario_trial_params.query_rate_per_visit:.3f}")
        
        # Update sites list
        scenario_trial_params.sites = list(sites_dict.values())
        
        return scenario_trial_params
    
    def run_scenario_simulation(
        self, 
        baseline_params: TrialParameters,
        scenario_params: ScenarioParameters,
        simulation_days: int = 365
    ) -> SimulationResults:
        """
        Run simulation for a specific scenario
        
        Args:
            baseline_params: Original trial parameters
            scenario_params: Scenario modifications
            simulation_days: Simulation duration
            
        Returns:
            SimulationResults for the scenario
        """
        logger.info(f"🎬 Running scenario simulation: {scenario_params.scenario_name}")
        
        # Create scenario trial parameters
        scenario_trial_params = self.create_scenario_trial_params(baseline_params, scenario_params)
        
        # Initialize and run digital twin
        twin_engine = DigitalTwinEngine()
        twin_engine.initialize_simulation(scenario_trial_params, simulation_days)
        
        # Run simulation
        results = twin_engine.run_simulation(simulation_days)
        results.scenario_id = scenario_params.scenario_id
        
        # Store results
        self.scenario_results[scenario_params.scenario_id] = results
        
        logger.info(f"✅ Scenario simulation completed: {scenario_params.scenario_name}")
        
        return results
    
    def compare_scenarios(
        self,
        baseline_results: SimulationResults,
        scenario_results: SimulationResults,
        scenario_params: ScenarioParameters
    ) -> ScenarioComparison:
        """
        Compare baseline and scenario results
        
        Args:
            baseline_results: Baseline simulation results
            scenario_results: Scenario simulation results
            scenario_params: Scenario parameters for cost calculation
            
        Returns:
            ScenarioComparison with detailed analysis
        """
        logger.info(f"📊 Comparing scenario: {scenario_params.scenario_name}")
        
        # Calculate improvements
        enrollment_improvement = (
            (scenario_results.total_enrolled - baseline_results.total_enrolled) / 
            max(baseline_results.total_enrolled, 1)
        )
        
        completion_improvement = (
            (scenario_results.total_completed - baseline_results.total_completed) / 
            max(baseline_results.total_completed, 1)
        )
        
        dropout_reduction = (
            (baseline_results.total_dropouts - scenario_results.total_dropouts) / 
            max(baseline_results.total_dropouts, 1)
        )
        
        query_resolution_improvement = (
            (scenario_results.resolved_queries - baseline_results.resolved_queries) / 
            max(baseline_results.resolved_queries, 1)
        )
        
        data_delay_improvement = (
            (baseline_results.avg_data_entry_delay - scenario_results.avg_data_entry_delay) / 
            max(baseline_results.avg_data_entry_delay, 1)
        )
        
        risk_score_improvement = (
            baseline_results.delay_risk_score - scenario_results.delay_risk_score
        )
        
        # Calculate time savings
        baseline_completion_rate = baseline_results.total_completed / max(baseline_results.total_enrolled, 1)
        scenario_completion_rate = scenario_results.total_completed / max(scenario_results.total_enrolled, 1)
        
        if scenario_completion_rate > baseline_completion_rate:
            time_savings_days = baseline_results.simulation_days * (
                (scenario_completion_rate - baseline_completion_rate) / scenario_completion_rate
            )
        else:
            time_savings_days = 0.0
        
        # Calculate cost savings
        direct_cost_savings = baseline_results.cost_estimate - scenario_results.cost_estimate
        
        # Add scenario implementation costs
        implementation_costs = (
            scenario_params.training_cost + 
            scenario_params.tooling_cost +
            (scenario_params.additional_cost_per_site * len(scenario_results.site_metrics) * scenario_results.simulation_days)
        )
        
        net_cost_savings = direct_cost_savings - implementation_costs
        
        # Calculate ROI
        roi_percentage = (net_cost_savings / max(implementation_costs, 1)) * 100 if implementation_costs > 0 else 0
        
        # Create comparative timeline
        comparison_timeline = []
        baseline_timeline = {item['day']: item for item in baseline_results.timeline_data}
        scenario_timeline = {item['day']: item for item in scenario_results.timeline_data}
        
        all_days = set(baseline_timeline.keys()) | set(scenario_timeline.keys())
        
        for day in sorted(all_days):
            baseline_data = baseline_timeline.get(day, {})
            scenario_data = scenario_timeline.get(day, {})
            
            comparison_timeline.append({
                'day': day,
                'baseline_enrolled': baseline_data.get('enrolled_patients', 0),
                'scenario_enrolled': scenario_data.get('enrolled_patients', 0),
                'baseline_completed': baseline_data.get('completed_patients', 0),
                'scenario_completed': scenario_data.get('completed_patients', 0),
                'baseline_queries': baseline_data.get('open_queries', 0),
                'scenario_queries': scenario_data.get('open_queries', 0),
                'baseline_delay': baseline_data.get('avg_data_delay', 0),
                'scenario_delay': scenario_data.get('avg_data_delay', 0)
            })
        
        comparison = ScenarioComparison(
            baseline_results=baseline_results,
            scenario_results=scenario_results,
            scenario_params=scenario_params,
            enrollment_improvement=enrollment_improvement,
            completion_improvement=completion_improvement,
            dropout_reduction=dropout_reduction,
            query_resolution_improvement=query_resolution_improvement,
            data_delay_improvement=data_delay_improvement,
            risk_score_improvement=risk_score_improvement,
            time_savings_days=time_savings_days,
            cost_savings=net_cost_savings,
            roi_percentage=roi_percentage,
            comparison_timeline=comparison_timeline
        )
        
        logger.info(f"📈 Scenario analysis complete:")
        logger.info(f"   Enrollment improvement: {enrollment_improvement:.1%}")
        logger.info(f"   Completion improvement: {completion_improvement:.1%}")
        logger.info(f"   Risk score improvement: {risk_score_improvement:.3f}")
        logger.info(f"   Time savings: {time_savings_days:.1f} days")
        logger.info(f"   Cost savings: ${net_cost_savings:,.0f}")
        logger.info(f"   ROI: {roi_percentage:.1f}%")
        
        return comparison
    
    def generate_scenario_recommendations(
        self, 
        baseline_params: TrialParameters,
        simulation_days: int = 365
    ) -> List[ScenarioParameters]:
        """
        Generate recommended scenarios for what-if analysis
        
        Args:
            baseline_params: Original trial parameters
            simulation_days: Simulation duration
            
        Returns:
            List of recommended ScenarioParameters
        """
        recommendations = []
        
        # Scenario 1: Staff augmentation
        recommendations.append(ScenarioParameters(
            scenario_id="staff_augmentation",
            scenario_name="Staff Augmentation",
            description="Increase staff by 50% at all sites",
            staff_multiplier=1.5,
            additional_cost_per_site=200,  # Additional daily cost
            training_cost=10000  # One-time training cost
        ))
        
        # Scenario 2: Process optimization
        recommendations.append(ScenarioParameters(
            scenario_id="process_optimization",
            scenario_name="Process Optimization",
            description="Reduce query resolution time by 30% and data entry delays by 25%",
            query_time_reduction=0.3,
            data_entry_improvement=0.25,
            tooling_cost=25000  # Process improvement tools
        ))
        
        # Scenario 3: Training program
        recommendations.append(ScenarioParameters(
            scenario_id="training_program",
            scenario_name="Comprehensive Training Program",
            description="Improve staff skill levels and reduce dropout rates",
            skill_improvement=0.2,
            dropout_reduction=0.2,
            query_rate_reduction=0.15,
            training_cost=15000
        ))
        
        # Scenario 4: Site expansion
        new_sites = []
        for i in range(2):  # Add 2 new sites
            new_sites.append({
                'site_id': f'NEW-{i+1:02d}',
                'name': f'New Site {i+1}',
                'region': 'Expansion',
                'capacity': 40,
                'staff_count': 4,
                'staff_skill_level': 0.85,
                'enrollment_rate': 2.5,
                'query_resolution_time': 4.0,
                'data_entry_delay': 1.5
            })
        
        recommendations.append(ScenarioParameters(
            scenario_id="site_expansion",
            scenario_name="Site Expansion",
            description="Add 2 high-performing sites",
            add_sites=new_sites,
            additional_cost_per_site=500,  # New site setup cost
            tooling_cost=50000  # Site setup costs
        ))
        
        # Scenario 5: Technology upgrade
        recommendations.append(ScenarioParameters(
            scenario_id="technology_upgrade",
            scenario_name="Technology Upgrade",
            description="Comprehensive technology and process improvements",
            query_time_reduction=0.4,
            data_entry_improvement=0.35,
            enrollment_boost=0.2,
            skill_improvement=0.15,
            query_rate_reduction=0.25,
            tooling_cost=75000,
            training_cost=20000
        ))
        
        logger.info(f"💡 Generated {len(recommendations)} scenario recommendations")
        
        return recommendations
