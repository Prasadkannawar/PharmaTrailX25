#!/usr/bin/env python3
"""
Digital Twin Simulation Engine for PharmaTrail-X Phase 3
SimPy-based discrete event simulation for clinical trial modeling
"""

import simpy
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import random
from loguru import logger
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

@dataclass
class Patient:
    """Individual patient in the simulation"""
    patient_id: str
    site_id: str
    enrollment_date: float
    age: int
    gender: str
    risk_score: float = 0.5
    dropout_probability: float = 0.1
    visits_completed: int = 0
    total_visits: int = 8
    status: str = "active"  # active, completed, dropout
    queries_generated: int = 0
    data_entry_delays: List[float] = field(default_factory=list)

@dataclass
class Site:
    """Clinical trial site"""
    site_id: str
    name: str
    region: str
    capacity: int
    staff_count: int
    staff_skill_level: float = 0.8  # 0.0 to 1.0
    enrollment_rate: float = 2.0  # patients per week
    query_resolution_time: float = 5.0  # days average
    data_entry_delay: float = 2.0  # days average
    patients: List[Patient] = field(default_factory=list)
    active_queries: int = 0
    total_queries_resolved: int = 0

@dataclass
class TrialParameters:
    """Trial configuration parameters"""
    trial_id: str
    target_enrollment: int
    planned_duration_weeks: int
    sites: List[Site]
    visit_schedule: List[int] = field(default_factory=lambda: [0, 14, 28, 56, 84, 112, 140, 168])  # days
    dropout_base_rate: float = 0.15
    query_rate_per_visit: float = 0.3  # queries per visit
    data_entry_sla: float = 7.0  # days
    
@dataclass
class SimulationResults:
    """Results from digital twin simulation"""
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
    timeline_data: List[Dict[str, Any]]
    site_metrics: Dict[str, Dict[str, Any]]
    scenario_id: Optional[str] = None

class DigitalTwinEngine:
    """
    SimPy-based Digital Twin Engine for Clinical Trial Simulation
    
    Models:
    - Patient enrollment dynamics
    - Site-level query resolution cycles  
    - Data entry delays (DED)
    - Dropout behavior with risk profiles
    - Staffing effects on throughput
    """
    
    def __init__(self):
        self.env: Optional[simpy.Environment] = None
        self.trial_params: Optional[TrialParameters] = None
        self.patients: List[Patient] = []
        self.timeline_data: List[Dict[str, Any]] = []
        self.daily_metrics: Dict[str, List[float]] = {
            'enrolled_patients': [],
            'active_patients': [],
            'completed_patients': [],
            'dropout_patients': [],
            'open_queries': [],
            'resolved_queries': [],
            'avg_data_delay': []
        }
        
    def initialize_simulation(self, trial_params: TrialParameters, simulation_days: int = 365):
        """Initialize the simulation environment"""
        self.env = simpy.Environment()
        self.trial_params = trial_params
        self.patients = []
        self.timeline_data = []
        self.daily_metrics = {key: [] for key in self.daily_metrics.keys()}
        
        # Reset site states
        for site in self.trial_params.sites:
            site.patients = []
            site.active_queries = 0
            site.total_queries_resolved = 0
        
        logger.info(f"🎯 Initializing Digital Twin for trial {trial_params.trial_id}")
        logger.info(f"📊 Target enrollment: {trial_params.target_enrollment}")
        logger.info(f"🏥 Sites: {len(trial_params.sites)}")
        logger.info(f"⏱️  Simulation duration: {simulation_days} days")
        
    def patient_enrollment_process(self, site: Site):
        """
        SimPy process for patient enrollment at a site
        Models realistic enrollment patterns with variability
        """
        enrolled_count = 0
        
        while True:
            # Calculate inter-arrival time based on enrollment rate and variability
            base_interval = 7.0 / site.enrollment_rate  # days between patients
            
            # Add variability (exponential distribution for realistic modeling)
            interval = np.random.exponential(base_interval)
            
            # Staff skill affects enrollment efficiency
            efficiency_factor = 0.5 + (site.staff_skill_level * 0.5)
            interval = interval / efficiency_factor
            
            yield self.env.timeout(interval)
            
            # Check if we've reached target enrollment
            total_enrolled = sum(len(s.patients) for s in self.trial_params.sites)
            if total_enrolled >= self.trial_params.target_enrollment:
                break
                
            # Create new patient
            patient = Patient(
                patient_id=f"{site.site_id}-P{enrolled_count+1:03d}",
                site_id=site.site_id,
                enrollment_date=self.env.now,
                age=np.random.randint(18, 80),
                gender=np.random.choice(['M', 'F']),
                risk_score=np.random.beta(2, 5),  # Skewed toward lower risk
                dropout_probability=self.trial_params.dropout_base_rate * (1 + np.random.normal(0, 0.3))
            )
            
            site.patients.append(patient)
            self.patients.append(patient)
            enrolled_count += 1
            
            logger.debug(f"👤 Patient {patient.patient_id} enrolled at {site.site_id} (day {self.env.now:.1f})")
            
            # Start patient journey process
            self.env.process(self.patient_journey_process(patient, site))
    
    def patient_journey_process(self, patient: Patient, site: Site):
        """
        SimPy process modeling individual patient journey through trial
        Handles visits, dropouts, and data generation
        """
        visit_number = 0
        
        for visit_day in self.trial_params.visit_schedule:
            if patient.status != "active":
                break
                
            # Wait until visit day
            visit_time = patient.enrollment_date + visit_day
            if visit_time > self.env.now:
                yield self.env.timeout(visit_time - self.env.now)
            
            # Check for dropout before visit
            dropout_check = np.random.random()
            adjusted_dropout_prob = patient.dropout_probability * (1 + patient.risk_score * 0.5)
            
            if dropout_check < adjusted_dropout_prob:
                patient.status = "dropout"
                logger.debug(f"🚪 Patient {patient.patient_id} dropped out before visit {visit_number}")
                break
            
            # Conduct visit
            visit_number += 1
            patient.visits_completed = visit_number
            
            # Generate queries based on visit complexity and site efficiency
            query_probability = self.trial_params.query_rate_per_visit * (1 - site.staff_skill_level * 0.3)
            if np.random.random() < query_probability:
                patient.queries_generated += 1
                site.active_queries += 1
                
                # Start query resolution process
                self.env.process(self.query_resolution_process(site))
            
            # Generate data entry delay
            base_delay = site.data_entry_delay
            skill_factor = 1 - (site.staff_skill_level - 0.5) * 0.4
            actual_delay = np.random.gamma(2, base_delay * skill_factor / 2)
            patient.data_entry_delays.append(actual_delay)
            
            logger.debug(f"🏥 Patient {patient.patient_id} completed visit {visit_number} (day {self.env.now:.1f})")
        
        # Mark patient as completed if they finished all visits
        if patient.status == "active" and patient.visits_completed >= patient.total_visits:
            patient.status = "completed"
            logger.debug(f"✅ Patient {patient.patient_id} completed trial")
    
    def query_resolution_process(self, site: Site):
        """
        SimPy process for query resolution cycle
        Models realistic query handling with staff capacity constraints
        """
        # Base resolution time affected by staff workload and skill
        workload_factor = 1 + (site.active_queries / (site.staff_count * 10))  # Capacity constraint
        skill_factor = 1 / (0.5 + site.staff_skill_level * 0.5)
        
        resolution_time = site.query_resolution_time * workload_factor * skill_factor
        resolution_time = max(resolution_time, 0.5)  # Minimum 0.5 days
        
        yield self.env.timeout(resolution_time)
        
        # Resolve query
        site.active_queries = max(0, site.active_queries - 1)
        site.total_queries_resolved += 1
        
        logger.debug(f"❓ Query resolved at {site.site_id} (day {self.env.now:.1f})")
    
    def data_collection_process(self):
        """
        SimPy process for collecting daily metrics
        Runs every day to capture simulation state
        """
        day = 0
        while True:
            # Collect daily metrics
            enrolled = len([p for p in self.patients if p.enrollment_date <= self.env.now])
            active = len([p for p in self.patients if p.status == "active"])
            completed = len([p for p in self.patients if p.status == "completed"])
            dropouts = len([p for p in self.patients if p.status == "dropout"])
            
            open_queries = sum(site.active_queries for site in self.trial_params.sites)
            resolved_queries = sum(site.total_queries_resolved for site in self.trial_params.sites)
            
            # Calculate average data entry delay
            all_delays = []
            for patient in self.patients:
                all_delays.extend(patient.data_entry_delays)
            avg_delay = np.mean(all_delays) if all_delays else 0.0
            
            # Store daily metrics
            self.daily_metrics['enrolled_patients'].append(enrolled)
            self.daily_metrics['active_patients'].append(active)
            self.daily_metrics['completed_patients'].append(completed)
            self.daily_metrics['dropout_patients'].append(dropouts)
            self.daily_metrics['open_queries'].append(open_queries)
            self.daily_metrics['resolved_queries'].append(resolved_queries)
            self.daily_metrics['avg_data_delay'].append(avg_delay)
            
            # Store timeline data point
            self.timeline_data.append({
                'day': day,
                'timestamp': self.env.now,
                'enrolled_patients': enrolled,
                'active_patients': active,
                'completed_patients': completed,
                'dropout_patients': dropouts,
                'open_queries': open_queries,
                'resolved_queries': resolved_queries,
                'avg_data_delay': avg_delay
            })
            
            day += 1
            yield self.env.timeout(1.0)  # Wait 1 day
    
    def run_simulation(self, simulation_days: int = 365) -> SimulationResults:
        """
        Execute the digital twin simulation
        
        Args:
            simulation_days: Number of days to simulate
            
        Returns:
            SimulationResults with comprehensive metrics
        """
        logger.info(f"🚀 Starting Digital Twin simulation for {simulation_days} days...")
        
        # Start enrollment processes for each site
        for site in self.trial_params.sites:
            self.env.process(self.patient_enrollment_process(site))
        
        # Start data collection process
        self.env.process(self.data_collection_process())
        
        # Run simulation
        self.env.run(until=simulation_days)
        
        # Calculate final metrics
        total_enrolled = len(self.patients)
        total_completed = len([p for p in self.patients if p.status == "completed"])
        total_dropouts = len([p for p in self.patients if p.status == "dropout"])
        active_patients = len([p for p in self.patients if p.status == "active"])
        
        total_queries = sum(p.queries_generated for p in self.patients)
        resolved_queries = sum(site.total_queries_resolved for site in self.trial_params.sites)
        pending_queries = max(0, total_queries - resolved_queries)  # Ensure non-negative
        
        # Calculate average data entry delay
        all_delays = []
        for patient in self.patients:
            all_delays.extend(patient.data_entry_delays)
        avg_data_entry_delay = np.mean(all_delays) if all_delays else 0.0
        
        # Calculate delay risk score
        completion_rate = total_completed / max(total_enrolled, 1)
        dropout_rate = total_dropouts / max(total_enrolled, 1)
        query_backlog_rate = pending_queries / max(total_queries, 1) if total_queries > 0 else 0
        delay_over_sla = np.mean([d > self.trial_params.data_entry_sla for d in all_delays]) if all_delays else 0
        
        delay_risk_score = (
            (1 - completion_rate) * 0.3 +
            dropout_rate * 0.3 +
            query_backlog_rate * 0.2 +
            delay_over_sla * 0.2
        )
        
        # Ensure risk score is within bounds
        delay_risk_score = max(0.0, min(1.0, delay_risk_score))
        
        # Estimate completion date
        completion_date = None
        if completion_rate > 0:
            days_per_completion = simulation_days / max(total_completed, 1)
            remaining_completions = self.trial_params.target_enrollment - total_completed
            estimated_days_remaining = remaining_completions * days_per_completion
            completion_date = (datetime.now() + timedelta(days=estimated_days_remaining)).isoformat()
        
        # Calculate cost estimate (simplified)
        cost_per_patient_day = 150  # USD
        cost_per_site_day = 500  # USD
        total_patient_days = sum(simulation_days - p.enrollment_date for p in self.patients)
        total_site_days = len(self.trial_params.sites) * simulation_days
        cost_estimate = (total_patient_days * cost_per_patient_day) + (total_site_days * cost_per_site_day)
        
        # Generate site-level metrics
        site_metrics = {}
        for site in self.trial_params.sites:
            site_patients = [p for p in self.patients if p.site_id == site.site_id]
            site_metrics[site.site_id] = {
                'enrolled_patients': len(site_patients),
                'completed_patients': len([p for p in site_patients if p.status == "completed"]),
                'dropout_patients': len([p for p in site_patients if p.status == "dropout"]),
                'active_queries': site.active_queries,
                'resolved_queries': site.total_queries_resolved,
                'enrollment_rate': len(site_patients) / (simulation_days / 7) if simulation_days > 0 else 0,
                'completion_rate': len([p for p in site_patients if p.status == "completed"]) / max(len(site_patients), 1),
                'avg_data_delay': np.mean([d for p in site_patients for d in p.data_entry_delays]) if site_patients else 0
            }
        
        results = SimulationResults(
            trial_id=self.trial_params.trial_id,
            simulation_days=simulation_days,
            completion_date=completion_date,
            total_enrolled=total_enrolled,
            total_completed=total_completed,
            total_dropouts=total_dropouts,
            active_patients=active_patients,
            total_queries=total_queries,
            resolved_queries=resolved_queries,
            pending_queries=pending_queries,
            avg_data_entry_delay=avg_data_entry_delay,
            delay_risk_score=delay_risk_score,
            cost_estimate=cost_estimate,
            timeline_data=self.timeline_data,
            site_metrics=site_metrics
        )
        
        logger.info(f"✅ Digital Twin simulation completed!")
        logger.info(f"📊 Enrolled: {total_enrolled}, Completed: {total_completed}, Dropouts: {total_dropouts}")
        logger.info(f"❓ Queries: {total_queries} total, {resolved_queries} resolved, {pending_queries} pending")
        logger.info(f"⚠️  Delay risk score: {delay_risk_score:.3f}")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the digital twin engine"""
        return {
            "engine_type": "simpy_digital_twin",
            "version": "1.0.0",
            "capabilities": [
                "patient_enrollment_simulation",
                "site_query_resolution_modeling", 
                "data_entry_delay_simulation",
                "dropout_behavior_modeling",
                "staffing_throughput_effects",
                "scenario_what_if_analysis"
            ],
            "simulation_components": [
                "enrollment_dynamics",
                "patient_journey",
                "query_resolution_cycle",
                "data_collection_timeline"
            ],
            "output_formats": [
                "timeline_data",
                "site_metrics", 
                "summary_statistics",
                "dashboard_ready_json"
            ]
        }
