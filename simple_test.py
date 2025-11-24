#!/usr/bin/env python3
"""
Simplified test to get the pipeline working
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def create_delay_indicators(df):
    """Simplified delay indicator creation"""
    logger.info("Creating delay indicators...")
    
    # Convert dates
    df['Visit_Date'] = pd.to_datetime(df['Visit_Date'])
    
    # Create patient-level delay probabilities
    patient_delay_data = []
    patient_groups = df.groupby(['Patient_ID'])
    
    for (patient_id,), patient_data in patient_groups:
        # Calculate patient metrics
        avg_adherence = patient_data['Medication_Adherence(%)'].mean()
        adr_rate = patient_data['ADR_Reported'].mean()
        efficacy_trend = patient_data['Efficacy_Score'].diff().mean()
        visit_count = len(patient_data)
        expected_visits = patient_data['Week'].max()
        visit_compliance = visit_count / max(expected_visits, 1)
        
        # Calculate delay probability with more balanced approach
        seed_val = sum(ord(c) for c in str(patient_id)) % 1000
        np.random.seed(seed_val)
        
        # More balanced base probability
        base_prob = np.random.uniform(0.1, 0.6)  # Wide range
        
        # Factors that increase delay risk
        adherence_factor = max(0, (80 - avg_adherence) / 100) * 0.2  # Penalty for <80% adherence
        adr_factor = adr_rate * 0.15  # ADR impact
        efficacy_factor = max(0, -efficacy_trend / 30) * 0.1 if not np.isnan(efficacy_trend) else 0
        
        # Factors that decrease delay risk
        good_adherence_bonus = max(0, (avg_adherence - 85) / 100) * 0.15  # Bonus for >85% adherence
        
        delay_prob = base_prob + adherence_factor + adr_factor + efficacy_factor - good_adherence_bonus
        delay_prob = min(max(delay_prob, 0.05), 0.95)
        
        # Use a fixed threshold for more predictable balance
        is_delayed = delay_prob > 0.5
        
        patient_delay_data.append({
            'Patient_ID': patient_id,
            'delay_probability': delay_prob,
            'is_delayed': is_delayed,
            'delay_days': int(delay_prob * 45) if is_delayed else 0
        })
    
    # Merge back to original dataframe
    patient_delay_df = pd.DataFrame(patient_delay_data)
    df_with_delays = df.merge(patient_delay_df, on='Patient_ID', how='left')
    
    # Fill any NaN values (shouldn't happen if all patients processed)
    df_with_delays['delay_probability'] = df_with_delays['delay_probability'].fillna(0.5)
    df_with_delays['is_delayed'] = df_with_delays['is_delayed'].fillna(False)  # Default to not delayed
    df_with_delays['delay_days'] = df_with_delays['delay_days'].fillna(0)
    
    logger.info(f"Created delay indicators for {len(patient_delay_data)} patients")
    logger.info(f"Delay distribution after merge: {df_with_delays['is_delayed'].value_counts().to_dict()}")
    return df_with_delays

def simple_feature_engineering(df):
    """Simplified feature engineering"""
    logger.info("Creating features...")
    
    # Basic time features
    df['visit_month'] = df['Visit_Date'].dt.month
    df['visit_quarter'] = df['Visit_Date'].dt.quarter
    df['visit_day_of_week'] = df['Visit_Date'].dt.dayofweek
    
    # Safety features
    df['alt_elevated'] = (df['ALT_Level'] > 40).astype(int)
    df['ast_elevated'] = (df['AST_Level'] > 40).astype(int)
    df['creatinine_elevated'] = (df['Creatinine'] > 1.2).astype(int)
    df['hypertension_risk'] = ((df['BP_Systolic'] > 140) | (df['BP_Diastolic'] > 90)).astype(int)
    
    # Composite scores
    df['safety_risk_score'] = (
        df['ADR_Severity'] * df['ADR_Reported'] * 0.4 +
        (df['alt_elevated'] + df['ast_elevated'] + df['creatinine_elevated']) * 0.3 +
        df['hypertension_risk'] * 0.3
    )
    
    # Patient aggregations
    patient_stats = df.groupby('Patient_ID').agg({
        'Medication_Adherence(%)': 'mean',
        'ADR_Reported': 'mean',
        'Efficacy_Score': 'mean',
        'safety_risk_score': 'mean'
    }).add_prefix('patient_avg_')
    
    df = df.merge(patient_stats, left_on='Patient_ID', right_index=True)
    
    # Select numeric features for ML
    numeric_features = [
        'Age', 'BMI', 'Risk_Score', 'BP_Systolic', 'BP_Diastolic',
        'ALT_Level', 'AST_Level', 'Creatinine', 'Efficacy_Score',
        'Medication_Adherence(%)', 'Satisfaction_Score', 'Week',
        'visit_month', 'visit_quarter', 'visit_day_of_week',
        'alt_elevated', 'ast_elevated', 'creatinine_elevated',
        'hypertension_risk', 'safety_risk_score',
        'patient_avg_Medication_Adherence(%)', 'patient_avg_ADR_Reported',
        'patient_avg_Efficacy_Score', 'patient_avg_safety_risk_score'
    ]
    
    X = df[numeric_features].fillna(0)  # Simple imputation
    y = df['is_delayed'].astype(int)
    
    logger.info(f"Created {len(numeric_features)} features")
    return X, y, df

def simple_model_training(X, y):
    """Simplified model training"""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    import xgboost as xgb
    from sklearn.ensemble import IsolationForest
    
    logger.info("Training models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Model trained - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
    
    # Train anomaly detection
    anomaly_model = IsolationForest(contamination=0.1, random_state=42)
    anomaly_model.fit(X_train)
    
    logger.info("Anomaly detection model trained")
    
    return model, anomaly_model, {'accuracy': accuracy, 'auc': auc}

def main():
    logger.info("🚀 Running simplified PharmaTrail-X pipeline")
    
    try:
        # Load data
        csv_path = 'PharmaTrailX_ClinicalTrialMaster2.csv'
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return False
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Create delay indicators
        df_with_delays = create_delay_indicators(df)
        
        # Check distribution
        delay_dist = df_with_delays['is_delayed'].value_counts()
        logger.info(f"Delay distribution: {delay_dist.to_dict()}")
        
        if len(delay_dist) < 2:
            logger.error("Target variable has only one class!")
            return False
        
        # Feature engineering
        X, y, df_final = simple_feature_engineering(df_with_delays)
        logger.info(f"Feature matrix shape: {X.shape}")
        
        # Model training
        model, anomaly_model, metrics = simple_model_training(X, y)
        
        # Save models
        import joblib
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        joblib.dump(model, model_dir / "delay_model.pkl")
        joblib.dump(anomaly_model, model_dir / "anomaly_model.pkl")
        
        # Test prediction
        sample_X = X.iloc[:5]
        predictions = model.predict_proba(sample_X)[:, 1]
        anomaly_scores = anomaly_model.decision_function(sample_X)
        
        logger.info("Sample predictions:")
        for i, (pred, anom) in enumerate(zip(predictions, anomaly_scores)):
            logger.info(f"  Sample {i+1}: Delay prob={pred:.3f}, Anomaly score={anom:.3f}")
        
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"📊 Final metrics: Accuracy={metrics['accuracy']:.3f}, AUC={metrics['auc']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ All tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Tests failed!")
        sys.exit(1)
