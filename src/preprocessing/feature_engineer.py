import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from loguru import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
from config.settings import settings

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        self.feature_names = []
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline
        """
        logger.info("Starting feature engineering pipeline")
        
        # Create a copy to avoid modifying original data
        features_df = df.copy()
        
        # 1. Time-based features
        features_df = self._create_time_features(features_df)
        
        # 2. Enrollment and velocity features
        features_df = self._create_enrollment_features(features_df)
        
        # 3. Patient-level aggregations
        features_df = self._create_patient_aggregations(features_df)
        
        # 4. Site-level features
        features_df = self._create_site_features(features_df)
        
        # 5. Risk and safety features
        features_df = self._create_safety_features(features_df)
        
        # 6. Protocol and compliance features
        features_df = self._create_compliance_features(features_df)
        
        # 7. Derive final delay score
        features_df = self._create_delay_score(features_df)
        
        logger.info(f"Feature engineering complete. Created {len(features_df.columns)} features")
        return features_df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features
        """
        logger.info("Creating time-based features")
        
        # Ensure Visit_Date is datetime
        df['Visit_Date'] = pd.to_datetime(df['Visit_Date'])
        
        # Extract date components
        df['visit_year'] = df['Visit_Date'].dt.year
        df['visit_month'] = df['Visit_Date'].dt.month
        df['visit_quarter'] = df['Visit_Date'].dt.quarter
        df['visit_day_of_week'] = df['Visit_Date'].dt.dayofweek
        df['visit_week_of_year'] = df['Visit_Date'].dt.isocalendar().week
        
        # Calculate time since trial start for each trial
        trial_start_dates = df.groupby(['Study_ID', 'Trial_ID'])['Visit_Date'].min()
        df = df.merge(
            trial_start_dates.rename('trial_start_date'), 
            left_on=['Study_ID', 'Trial_ID'], 
            right_index=True
        )
        
        df['days_since_trial_start'] = (df['Visit_Date'] - df['trial_start_date']).dt.days
        df['weeks_since_trial_start'] = df['days_since_trial_start'] / 7
        
        # Timeline gaps (difference between expected and actual visit dates)
        df['expected_visit_date'] = df['trial_start_date'] + pd.to_timedelta(df['Week'] * 7, unit='days')
        df['timeline_gap_days'] = (df['Visit_Date'] - df['expected_visit_date']).dt.days
        df['timeline_gap_abs'] = df['timeline_gap_days'].abs()
        
        return df
    
    def _create_enrollment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enrollment and velocity features
        """
        logger.info("Creating enrollment features")
        
        # Calculate enrollment metrics per trial
        enrollment_stats = df.groupby(['Study_ID', 'Trial_ID']).agg({
            'Patient_ID': 'nunique',
            'Visit_Date': ['min', 'max'],
            'Week': 'max'
        }).round(2)
        
        enrollment_stats.columns = ['total_patients', 'first_visit', 'last_visit', 'max_week']
        enrollment_stats['trial_duration_days'] = (
            enrollment_stats['last_visit'] - enrollment_stats['first_visit']
        ).dt.days
        enrollment_stats['enrollment_velocity'] = (
            enrollment_stats['total_patients'] / 
            np.maximum(enrollment_stats['trial_duration_days'] / 7, 1)
        )
        
        # Merge back to main dataframe
        df = df.merge(enrollment_stats, left_on=['Study_ID', 'Trial_ID'], right_index=True)
        
        # Patient enrollment order within trial
        patient_enrollment = df.groupby(['Study_ID', 'Trial_ID', 'Patient_ID'])['Visit_Date'].min().reset_index()
        patient_enrollment['enrollment_order'] = patient_enrollment.groupby(['Study_ID', 'Trial_ID'])['Visit_Date'].rank()
        
        df = df.merge(
            patient_enrollment[['Study_ID', 'Trial_ID', 'Patient_ID', 'enrollment_order']], 
            on=['Study_ID', 'Trial_ID', 'Patient_ID']
        )
        
        return df
    
    def _create_patient_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create patient-level aggregated features
        """
        logger.info("Creating patient aggregation features")
        
        # Patient-level statistics
        patient_stats = df.groupby(['Study_ID', 'Trial_ID', 'Patient_ID']).agg({
            'BP_Systolic': ['mean', 'std', 'min', 'max'],
            'BP_Diastolic': ['mean', 'std', 'min', 'max'],
            'ALT_Level': ['mean', 'std', 'max'],
            'AST_Level': ['mean', 'std', 'max'],
            'Creatinine': ['mean', 'std', 'max'],
            'Efficacy_Score': ['mean', 'std', 'min', 'max', 'last'],
            'Medication_Adherence(%)': ['mean', 'std', 'min'],
            'Satisfaction_Score': ['mean', 'std', 'min'],
            'ADR_Reported': ['sum', 'mean'],
            'ADR_Severity': ['mean', 'max'],
            'Week': 'max'
        }).round(2)
        
        # Flatten column names
        patient_stats.columns = ['_'.join(col).strip() for col in patient_stats.columns]
        patient_stats = patient_stats.add_prefix('patient_')
        
        # Merge back to main dataframe
        df = df.merge(patient_stats, left_on=['Study_ID', 'Trial_ID', 'Patient_ID'], right_index=True)
        
        return df
    
    def _create_site_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create site-level features
        """
        logger.info("Creating site-level features")
        
        # Site-level aggregations
        site_stats = df.groupby(['Site_ID']).agg({
            'Patient_ID': 'nunique',
            'Study_ID': 'nunique',
            'ADR_Reported': 'mean',
            'ADR_Severity': 'mean',
            'Medication_Adherence(%)': 'mean',
            'Efficacy_Score': 'mean',
            'timeline_gap_abs': 'mean',
            'Age': 'mean',
            'BMI': 'mean'
        }).round(2)
        
        site_stats.columns = [f'site_{col}' for col in site_stats.columns]
        
        # Site deviation index (higher = more problematic)
        site_stats['site_deviation_index'] = (
            site_stats['site_ADR_Reported'] * 0.3 +
            (1 - site_stats['site_Medication_Adherence(%)'] / 100) * 0.3 +
            site_stats['site_timeline_gap_abs'] / 7 * 0.2 +  # weeks
            (1 - site_stats['site_Efficacy_Score'] / 100) * 0.2
        )
        
        # Merge back to main dataframe
        df = df.merge(site_stats, left_on='Site_ID', right_index=True)
        
        return df
    
    def _create_safety_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create safety and risk features
        """
        logger.info("Creating safety features")
        
        # ADR patterns
        df['adr_severity_weighted'] = df['ADR_Severity'] * df['ADR_Reported']
        
        # Lab value risk flags
        df['alt_elevated'] = (df['ALT_Level'] > 40).astype(int)  # Normal < 40 U/L
        df['ast_elevated'] = (df['AST_Level'] > 40).astype(int)
        df['creatinine_elevated'] = (df['Creatinine'] > 1.2).astype(int)  # Normal < 1.2 mg/dL
        
        # Vital signs risk
        df['hypertension_risk'] = ((df['BP_Systolic'] > 140) | (df['BP_Diastolic'] > 90)).astype(int)
        
        # Composite safety score
        df['safety_risk_score'] = (
            df['adr_severity_weighted'] * 0.4 +
            (df['alt_elevated'] + df['ast_elevated'] + df['creatinine_elevated']) * 0.3 +
            df['hypertension_risk'] * 0.3
        )
        
        return df
    
    def _create_compliance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create compliance and protocol adherence features
        """
        logger.info("Creating compliance features")
        
        # Adherence categories
        df['adherence_category'] = pd.cut(
            df['Medication_Adherence(%)'], 
            bins=[0, 70, 85, 95, 100], 
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        
        # Visit compliance (based on timeline gaps)
        df['visit_compliant'] = (df['timeline_gap_abs'] <= 3).astype(int)  # Within 3 days
        
        # Query density (mock - would be actual in real implementation)
        df['query_density'] = np.random.poisson(2, len(df))  # Average 2 queries per visit
        
        # Data entry lag (mock)
        df['data_entry_lag'] = np.random.exponential(3, len(df))  # Average 3 days lag
        
        return df
    
    def _create_delay_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite delay risk score
        """
        logger.info("Creating delay score")
        
        # Normalize components to 0-1 scale
        timeline_score = np.clip(df['timeline_gap_abs'] / 14, 0, 1)  # 2 weeks max
        safety_score = np.clip(df['safety_risk_score'] / 5, 0, 1)
        adherence_score = 1 - (df['Medication_Adherence(%)'] / 100)
        site_score = np.clip(df['site_deviation_index'], 0, 1)
        
        # Weighted composite score
        df['delay_risk_score'] = (
            timeline_score * 0.35 +
            safety_score * 0.25 +
            adherence_score * 0.25 +
            site_score * 0.15
        )
        
        return df
    
    def prepare_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for machine learning
        """
        logger.info("Preparing features for ML")
        
        # Select numeric features for ML
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target variables and IDs
        exclude_features = [
            'delay_probability', 'is_delayed', 'delay_days',
            'Study_ID', 'Trial_ID', 'Patient_ID', 'Week'
        ]
        
        ml_features = [f for f in numeric_features if f not in exclude_features]
        
        # Create feature matrix
        X = df[ml_features].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy='median')
                    X[col] = self.imputers[col].fit_transform(X[[col]]).flatten()
                else:
                    X[col] = self.imputers[col].transform(X[[col]]).flatten()
        
        # Store feature names
        self.feature_names = ml_features
        
        logger.info(f"Prepared {len(ml_features)} features for ML")
        return X, ml_features
    
    def save_preprocessors(self, path: str):
        """
        Save preprocessing objects
        """
        save_path = Path(path)
        save_path.mkdir(exist_ok=True)
        
        joblib.dump(self.scaler, save_path / "scaler.pkl")
        joblib.dump(self.label_encoders, save_path / "label_encoders.pkl")
        joblib.dump(self.imputers, save_path / "imputers.pkl")
        joblib.dump(self.feature_names, save_path / "feature_names.pkl")
        
        logger.info(f"Saved preprocessors to {save_path}")
    
    def load_preprocessors(self, path: str):
        """
        Load preprocessing objects
        """
        load_path = Path(path)
        
        self.scaler = joblib.load(load_path / "scaler.pkl")
        self.label_encoders = joblib.load(load_path / "label_encoders.pkl")
        self.imputers = joblib.load(load_path / "imputers.pkl")
        self.feature_names = joblib.load(load_path / "feature_names.pkl")
        
        logger.info(f"Loaded preprocessors from {load_path}")
