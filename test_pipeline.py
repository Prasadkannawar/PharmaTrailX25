#!/usr/bin/env python3
"""
Simple test script for PharmaTrail-X Phase 1 pipeline
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.ingestion.data_ingester import DataIngester
from src.preprocessing.feature_engineer import FeatureEngineer
from src.models.ml_pipeline import MLPipeline

def test_pipeline():
    """Test the basic pipeline functionality"""
    logger.info("🧪 Testing PharmaTrail-X Phase 1 Pipeline")
    
    try:
        # Step 1: Data Ingestion
        logger.info("📊 Testing data ingestion...")
        data_ingester = DataIngester()
        
        csv_path = 'PharmaTrailX_ClinicalTrialMaster2.csv'
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return False
        
        df = data_ingester.ingest_clinical_trials_csv(csv_path)
        logger.info(f"✅ Ingested {len(df)} records")
        
        # Check target variable distribution
        delay_distribution = df['is_delayed'].value_counts()
        logger.info(f"📊 Delay distribution: {delay_distribution.to_dict()}")
        
        if len(delay_distribution) < 2:
            logger.error("❌ Target variable has only one class!")
            return False
        
        # Step 2: Feature Engineering
        logger.info("🔧 Testing feature engineering...")
        feature_engineer = FeatureEngineer()
        
        df_features = feature_engineer.engineer_features(df)
        X, feature_names = feature_engineer.prepare_ml_features(df_features)
        
        logger.info(f"✅ Created {len(feature_names)} features")
        logger.info(f"📈 Feature matrix shape: {X.shape}")
        
        # Step 3: Simple Model Training (without hyperparameter tuning)
        logger.info("🤖 Testing model training...")
        ml_pipeline = MLPipeline()
        
        # Prepare target variable
        y = df_features['is_delayed'].astype(int)
        
        # Check class distribution
        class_counts = np.bincount(y)
        logger.info(f"📊 Class distribution: {dict(enumerate(class_counts))}")
        
        if len(class_counts) < 2 or min(class_counts) == 0:
            logger.error("❌ Insufficient class distribution for training!")
            return False
        
        # Train with simple parameters (no hyperparameter tuning)
        delay_metrics = ml_pipeline.train_delay_predictor(
            X, y, hyperparameter_tuning=False
        )
        
        logger.info("✅ Delay prediction model trained")
        logger.info(f"   📊 Accuracy: {delay_metrics['accuracy']:.3f}")
        logger.info(f"   📊 AUC: {delay_metrics['roc_auc']:.3f}")
        
        # Train anomaly detection
        anomaly_metrics = ml_pipeline.train_anomaly_detector(X)
        logger.info("✅ Anomaly detection model trained")
        
        # Test prediction
        logger.info("🧪 Testing predictions...")
        sample_data = X.iloc[:5]
        delay_pred, delay_prob = ml_pipeline.predict_delay(sample_data)
        anomaly_pred, anomaly_scores = ml_pipeline.detect_anomalies(sample_data)
        
        logger.info("✅ Predictions successful")
        
        # Save models
        model_path = Path("models")
        model_path.mkdir(exist_ok=True)
        ml_pipeline.save_models(str(model_path))
        feature_engineer.save_preprocessors(str(model_path))
        
        logger.info(f"✅ Models saved to {model_path}")
        
        logger.info("🎉 Pipeline test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline()
    if success:
        logger.info("✅ All tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Tests failed!")
        sys.exit(1)
