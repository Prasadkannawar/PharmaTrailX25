import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
from sqlalchemy.orm import Session
from src.models.database import get_db, ClinicalTrial, Patient, PatientVisit, TrialMetrics
from config.settings import settings
import random

class DataIngester:
    def __init__(self):
        self.raw_data_path = Path(settings.RAW_DATA_PATH)
        self.processed_data_path = Path(settings.PROCESSED_DATA_PATH)
        
    def ingest_clinical_trials_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Ingest ClinicalTrials.gov CSV data and process into structured format
        """
        logger.info(f"Ingesting clinical trials data from {csv_path}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} records from CSV")
            
            # Convert visit dates to datetime
            df['Visit_Date'] = pd.to_datetime(df['Visit_Date'])
            
            # Add derived delay indicators (mock for now)
            df = self._add_delay_indicators(df)
            
            # Save to processed data lake
            processed_file = self.processed_data_path / "clinical_trials_processed.parquet"
            df.to_parquet(processed_file, index=False)
            logger.info(f"Saved processed data to {processed_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error ingesting CSV data: {e}")
            raise
    
    def _add_delay_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add delay indicators based on patient-level patterns since we have only one trial
        """
        # Since we have only one trial, create variation at patient level
        patient_delay_data = []
        
        # Group by patient to calculate individual delay probabilities
        patient_groups = df.groupby(['Patient_ID'])
        
        for patient_id, patient_data in patient_groups:
            # Calculate patient-specific metrics
            avg_adherence = patient_data['Medication_Adherence(%)'].mean()
            adr_count = patient_data['ADR_Reported'].sum()
            adr_rate = patient_data['ADR_Reported'].mean()
            efficacy_trend = patient_data['Efficacy_Score'].diff().mean()  # Trend over time
            visit_count = len(patient_data)
            
            # Create patient-specific delay probability
            # Use a simpler seed approach
            seed_val = sum(ord(c) for c in str(patient_id)) % 1000
            np.random.seed(seed_val)
            
            # Base probability varies by patient characteristics
            base_prob = 0.2 + np.random.uniform(0, 0.3)
            
            # Adherence factor (lower adherence = higher delay risk)
            adherence_factor = max(0, (75 - avg_adherence) / 100) * 0.3
            
            # ADR factor
            adr_factor = min(adr_rate * 0.4, 0.3)
            
            # Efficacy trend factor (declining efficacy = higher delay risk)
            efficacy_factor = max(0, -efficacy_trend / 20) * 0.2 if not np.isnan(efficacy_trend) else 0
            
            # Visit compliance factor
            expected_visits = patient_data['Week'].max()
            visit_compliance = visit_count / max(expected_visits, 1)
            compliance_factor = max(0, (1 - visit_compliance)) * 0.2
            
            # Calculate final delay probability
            delay_prob = base_prob + adherence_factor + adr_factor + efficacy_factor + compliance_factor
            delay_prob = min(max(delay_prob, 0.05), 0.95)
            
            # Determine if delayed (use varying thresholds for balance)
            threshold = 0.45 + np.random.uniform(-0.1, 0.1)  # Varying threshold
            is_delayed = delay_prob > threshold
            delay_days = int(delay_prob * 45) if is_delayed else 0
            
            # Debug: ensure values are not NaN
            if np.isnan(delay_prob):
                logger.warning(f"NaN delay_prob for patient {patient_id}")
                delay_prob = 0.5  # Default value
                is_delayed = True
            
            patient_delay_data.append({
                'Patient_ID': patient_id,
                'delay_probability': delay_prob,
                'is_delayed': is_delayed,
                'delay_days': delay_days,
                'patient_adherence': avg_adherence,
                'patient_adr_rate': adr_rate,
                'patient_visit_compliance': visit_compliance
            })
        
        # Merge patient delay data back to main dataframe
        patient_delay_df = pd.DataFrame(patient_delay_data)
        df = df.merge(patient_delay_df, on=['Patient_ID'], how='left')
        
        # Also add trial-level aggregations
        trial_groups = df.groupby(['Study_ID', 'Trial_ID'])
        trial_data = []
        
        for (study_id, trial_id), group in trial_groups:
            unique_patients = group['Patient_ID'].nunique()
            date_range = (group['Visit_Date'].max() - group['Visit_Date'].min()).days
            enrollment_velocity = unique_patients / max(date_range / 7, 1)
            
            trial_data.append({
                'Study_ID': study_id,
                'Trial_ID': trial_id,
                'enrollment_velocity': enrollment_velocity,
                'total_patients': unique_patients,
                'avg_adherence': group['Medication_Adherence(%)'].mean(),
                'adr_rate': group['ADR_Reported'].mean()
            })
        
        trial_df = pd.DataFrame(trial_data)
        df = df.merge(trial_df, on=['Study_ID', 'Trial_ID'], how='left')
        
        return df
    
    def generate_mock_fhir_data(self, num_trials: int = 5) -> List[Dict]:
        """
        Generate mock FHIR-like clinical trial data
        """
        logger.info(f"Generating {num_trials} mock FHIR trial records")
        
        therapeutic_areas = ['Oncology', 'Cardiology', 'Neurology', 'Immunology', 'Endocrinology']
        phases = ['Phase I', 'Phase II', 'Phase III']
        regions = ['North America', 'Europe', 'Asia-Pacific']
        
        fhir_data = []
        
        for i in range(num_trials):
            trial_data = {
                "resourceType": "ResearchStudy",
                "id": f"fhir-trial-{i+1:03d}",
                "identifier": [
                    {
                        "system": "http://clinicaltrials.gov",
                        "value": f"NCT{random.randint(10000000, 99999999)}"
                    }
                ],
                "title": f"Mock Clinical Trial {i+1}",
                "status": "active",
                "phase": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/research-study-phase",
                            "code": random.choice(phases).lower().replace(" ", "-"),
                            "display": random.choice(phases)
                        }
                    ]
                },
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/research-study-category",
                                "code": "treatment",
                                "display": "Treatment"
                            }
                        ]
                    }
                ],
                "condition": [
                    {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": str(random.randint(100000, 999999)),
                                "display": random.choice(therapeutic_areas)
                            }
                        ]
                    }
                ],
                "period": {
                    "start": (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
                    "end": (datetime.now() + timedelta(days=random.randint(180, 730))).isoformat()
                },
                "enrollment": {
                    "target": random.randint(50, 500),
                    "actual": random.randint(10, 400)
                },
                "site": {
                    "reference": f"Organization/site-{random.randint(1, 20):02d}",
                    "display": f"Clinical Site {random.randint(1, 20)}"
                },
                "principalInvestigator": {
                    "reference": f"Practitioner/pi-{random.randint(1, 50):02d}",
                    "display": f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'])}"
                },
                "location": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/iso3166-1-3",
                            "code": random.choice(['USA', 'GBR', 'DEU', 'FRA', 'JPN']),
                            "display": random.choice(regions)
                        }
                    ]
                }
            }
            
            fhir_data.append(trial_data)
        
        # Save FHIR data to raw data lake
        fhir_file = self.raw_data_path / "mock_fhir_trials.json"
        with open(fhir_file, 'w') as f:
            json.dump(fhir_data, f, indent=2)
        
        logger.info(f"Saved mock FHIR data to {fhir_file}")
        return fhir_data
    
    def load_to_database(self, df: pd.DataFrame, db: Session):
        """
        Load processed data into PostgreSQL database
        """
        logger.info("Loading data into database")
        
        try:
            # Process trials
            trials_data = df.groupby(['Study_ID', 'Trial_ID']).first().reset_index()
            
            for _, row in trials_data.iterrows():
                # Check if trial already exists
                existing_trial = db.query(ClinicalTrial).filter(
                    ClinicalTrial.study_id == row['Study_ID'],
                    ClinicalTrial.trial_id == row['Trial_ID']
                ).first()
                
                if not existing_trial:
                    trial = ClinicalTrial(
                        study_id=row['Study_ID'],
                        trial_id=row['Trial_ID'],
                        phase=row['Phase'],
                        therapeutic_area=row['Therapeutic_Area'],
                        principal_investigator=row['Principal_Investigator'],
                        site_id=row['Site_ID'],
                        region=row['Region'],
                        is_delayed=row['is_delayed'],
                        delay_days=row['delay_days'],
                        delay_probability=row['delay_probability']
                    )
                    db.add(trial)
                    db.flush()  # Get the trial ID
                    
                    # Add trial metrics
                    metrics = TrialMetrics(
                        trial_id=trial.id,
                        current_enrollment=row['total_patients'],
                        enrollment_velocity=row['enrollment_velocity'],
                        data_entry_lag_days=random.uniform(1, 10)
                    )
                    db.add(metrics)
            
            # Process patients
            patients_data = df.groupby(['Patient_ID', 'Study_ID', 'Trial_ID']).first().reset_index()
            
            for _, row in patients_data.iterrows():
                # Get trial ID
                trial = db.query(ClinicalTrial).filter(
                    ClinicalTrial.study_id == row['Study_ID'],
                    ClinicalTrial.trial_id == row['Trial_ID']
                ).first()
                
                if trial:
                    existing_patient = db.query(Patient).filter(
                        Patient.patient_id == row['Patient_ID'],
                        Patient.trial_id == trial.id
                    ).first()
                    
                    if not existing_patient:
                        patient = Patient(
                            patient_id=row['Patient_ID'],
                            trial_id=trial.id,
                            age=row['Age'],
                            sex=row['Sex'],
                            bmi=row['BMI'],
                            treatment_arm=row['Treatment_Arm'],
                            risk_score=row['Risk_Score'],
                            enrollment_date=row['Visit_Date']
                        )
                        db.add(patient)
            
            # Process visits
            for _, row in df.iterrows():
                # Get patient
                trial = db.query(ClinicalTrial).filter(
                    ClinicalTrial.study_id == row['Study_ID'],
                    ClinicalTrial.trial_id == row['Trial_ID']
                ).first()
                
                if trial:
                    patient = db.query(Patient).filter(
                        Patient.patient_id == row['Patient_ID'],
                        Patient.trial_id == trial.id
                    ).first()
                    
                    if patient:
                        visit = PatientVisit(
                            patient_id=patient.id,
                            week=row['Week'],
                            visit_date=row['Visit_Date'],
                            bp_systolic=row['BP_Systolic'],
                            bp_diastolic=row['BP_Diastolic'],
                            alt_level=row['ALT_Level'],
                            ast_level=row['AST_Level'],
                            creatinine=row['Creatinine'],
                            adr_reported=bool(row['ADR_Reported']),
                            adr_type=row['ADR_Type'] if row['ADR_Type'] != 'None' else None,
                            adr_severity=row['ADR_Severity'] if row['ADR_Severity'] > 0 else None,
                            efficacy_score=row['Efficacy_Score'],
                            medication_adherence=row['Medication_Adherence(%)'],
                            satisfaction_score=row['Satisfaction_Score']
                        )
                        db.add(visit)
            
            db.commit()
            logger.info("Successfully loaded data into database")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error loading data to database: {e}")
            raise
